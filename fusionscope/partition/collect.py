"""Collect one resource configuration at a time, in new pinned CPU processes."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import threading
from time import perf_counter_ns
from zipfile import ZipFile

from fusionscope.common import ROOT, dump, read, sha
from fusionscope.axis_probe.collect import identities
from fusionscope.axis_probe.design import now
from fusionscope.phase2.experiment import csv_text, zip_write
from run import worker_env

PROTOCOL = ROOT/"docs/plans/partition_boundary.json"
PLAN = read(PROTOCOL)
PUBLISHED = ROOT/"results/partition_boundary"
SEAL = ROOT/"docs/plans/partition_holdout.json"


def environment(resource):
    import psutil
    proc = psutil.Process()
    original = proc.cpu_affinity()
    if resource != "default":
        count = {"one":1, "four":4}[resource]
        if len(original) < count:
            raise ValueError((resource,original))
        proc.cpu_affinity(original[:count])
    from fusionscope.bench import environment as jax_environment
    import jaxlib
    env = jax_environment("cpu")
    env["original_affinity"] = original
    env["resource"] = resource
    binary, = [p for p in Path(jaxlib.__file__).parent.iterdir() if "jax_common" in p.name and p.is_file()]
    env["compiler_sha256"] = sha(binary)
    return env


def lock(phase):
    paths = [PROTOCOL, Path(__file__), Path(__file__).with_name("workloads.py")]
    if phase == "holdout":
        paths += [SEAL, Path(__file__).with_name("policies.py")]
    return {p.name:sha(p) for p in paths}


def kernel_structure(hlo):
    from fusionscope.hlo import Module, REF
    mod = Module(hlo)
    ops = {op.name:op for op in mod.comps[mod.entry]}
    root = next(op for op in ops.values() if op.root)
    outputs = REF.findall(root.operands) if root.kind == "tuple" else [root.name]

    def dependencies(roots):
        seen, stack = set(), list(roots)
        while stack:
            name = stack.pop()
            if name not in seen:
                seen.add(name)
                stack.extend(REF.findall(ops[name].operands))
        return seen

    main, obs = dependencies(outputs[:1]), dependencies(outputs[1:])
    result = []
    for name, op in ops.items():
        if op.kind in ("fusion", "custom-call", "reduce", "reduce-window", "while"):
            role = "shared" if name in main and name in obs else "main" if name in main else "observer"
            annotation = re.search(r'"outer_dimension_partitions"\s*:\s*\[([^\]]+)\]', op.attrs)
            partitions = list(map(int,re.findall(r"\d+",annotation[1]))) if annotation else []
            result.append(dict(name=name, kind=op.kind, role=role, partitions=partitions))
    return result


def capture(folder, phase, env):
    import jax
    from fusionscope.hlo import Module
    from fusionscope.evidence import scheduled_text_digest
    from .workloads import cases, compile_one, inputs, KEYS, SEEDS
    from zipfile import ZIP_DEFLATED
    target = folder/(env["resource"]+"_ir.zip")
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    partial = target.with_suffix(".partial")
    with ZipFile(partial, "w", ZIP_DEFLATED, compresslevel=5) as archive:
        archive.writestr("environment.json",json.dumps(env))
        for c in cases(phase):
            args = inputs(c)[SEEDS[0]]
            for key in KEYS:
                before = set(Path("dump").glob("*.cpu_after_optimizations.txt"))
                fn = compile_one(c,key,args)
                after = set(Path("dump").glob("*.cpu_after_optimizations.txt")) - before
                if len(after) != 1:
                    raise ValueError((c,key,after))
                optimized, = after
                prefix = optimized.name.removesuffix("cpu_after_optimizations.txt")
                hlo = optimized.read_text(encoding="utf8")
                meta = identities(fn)
                if (Module(hlo).fingerprint() != meta["fingerprint"] or
                        scheduled_text_digest(hlo) != meta["named_schedule"]):
                    raise ValueError((c,key,"dump"))
                meta["kernels"] = kernel_structure(hlo)
                objects = {}
                for p in sorted(Path("dump").glob(prefix+"*")):
                    if p.is_file() and p.name.endswith((".txt",".o","ir-with-opt.ll")):
                        leaf = p.name.removeprefix(prefix)
                        payload = p.read_bytes()
                        archive.writestr(f"{c.name}/{key}/{leaf}",payload)
                        if leaf.endswith(".o"):
                            objects[leaf] = hashlib.sha256(payload).hexdigest()
                meta["objects"] = objects
                archive.writestr(f"{c.name}/{key}/index.json",json.dumps(meta))
                jax.clear_caches()
    partial.replace(target)


def validate(case, data, executables):
    import jax
    import numpy as np
    from .workloads import program
    from fusionscope.cpu_cost.workloads import oracle
    fn = jax.jit(program(case,"diagnostic"))
    checks = []
    for seed,args in data.items():
        _, sites = fn(*args)
        arrays = [np.asarray(z) for z in sites[:2]]
        outs = {k:tuple(map(np.asarray,f(*args))) for k,f in executables.items()}
        main = outs["V0"][0]
        for key,(y,*qs) in outs.items():
            request = "count" if key.startswith("global") else "row"
            refs = [] if key == "V0" else [v for v,_ in oracle(request,arrays)]
            errors = [int(np.max(np.abs(v.astype(np.int64)-ref))) for v,ref in zip(qs,refs,strict=True)]
            types = all(v.dtype == np.int32 and v.shape == ref.shape for v,ref in zip(qs,refs,strict=True))
            pair = outs["global" if request == "count" else "S"][1:] if qs else ()
            identical = all(np.array_equal(v,s) for v,s in zip(qs,pair,strict=True))
            main_ok = (y.dtype == np.float32 and y.shape == main.shape and np.isfinite(y).all()
                       and np.allclose(y,main,rtol=3e-4,atol=3e-4))
            checks.append(dict(graph=case.name,seed=seed,key=key,main_ok=bool(main_ok),
                    counts_typed=types,direct_match=identical,max_count_error=max(errors,default=0),
                    valid=bool(main_ok and types and identical and not any(errors))))
    return checks


def collect(folder, phase, env, index):
    import jax
    from fusionscope.bench import timed
    from .workloads import cases, inputs, compile_one, KEYS, SEEDS
    timing = PLAN["timing"]
    rng = random.Random(f"partition-20261120/{phase}/{env['resource']}/{index}")
    graph_order = list(cases(phase))
    rng.shuffle(graph_order)
    times, builds, checks, decisions = [], [], [], []
    started = now()
    for c in graph_order:
        data = inputs(c)
        if phase == "holdout":
            from .policies import measure_choices
            decisions.extend(measure_choices(c))
        fns = {}
        keys = list(KEYS)
        rng.shuffle(keys)
        for order,key in enumerate(keys):
            start = perf_counter_ns()
            fn = compile_one(c,key,data[SEEDS[0]])
            elapsed = perf_counter_ns()-start
            builds.append(dict(graph=c.name,key=key,order=order,build_ns=elapsed,**identities(fn)))
            fns[key] = fn
        checks.extend(validate(c,data,fns))
        lanes = [(k,a) for k in KEYS for a in timing["aliases"]]
        seeds = list(SEEDS)
        for rnd in range(-timing["warmups"],timing["rounds"]):
            rng.shuffle(seeds)
            for block,seed in enumerate(seeds):
                rng.shuffle(lanes)
                for pos,(key,alias) in enumerate(lanes):
                    ns = timed(fns[key],data[seed])
                    times.append((c.name,seed,rnd,block,pos,key,alias,ns))
        del fns,fn,data
        jax.clear_caches()
        gc.collect()
    import csv,io
    raw = io.StringIO(newline="")
    wr = csv.writer(raw,lineterminator="\n")
    wr.writerow(("graph","seed","round","block","pos","key","alias","ns"))
    wr.writerows(times)
    meta = dict(process=index,pid=os.getpid(),start=started.isoformat(),end=now().isoformat(),
                environment=env,phase=phase,order=[c.name for c in graph_order],lock=lock(phase))
    files = {"meta.json":json.dumps(meta),"timings.csv":raw.getvalue(),
             "builds.csv":csv_text(builds),"checks.csv":csv_text(checks)}
    if decisions:
        files["decisions.csv"] = csv_text(decisions)
    zip_write(folder/env["resource"]/f"p{index}.zip",files)
    print(phase,env["resource"],index,len(times),flush=True)


def profile(folder, env):
    import jax
    import psutil
    from fusionscope.bench import timed
    from .workloads import cases, inputs, compile_one, SEEDS
    proc = psutil.Process()
    output = []
    for c in cases("discovery"):
        if c.name not in ("t_E1_chain","two_sites_large","two_sites_small","large_work"):
            continue
        data = inputs(c)[SEEDS[0]]
        for key in ("J","off","global","global_off","P"):
            fn = compile_one(c,key,data)
            for _ in range(12):
                timed(fn,data)
            def snapshot():
                return {t.id:t.user_time+t.system_time for t in proc.threads()}
            before = snapshot()
            start = perf_counter_ns()
            n = 0
            while perf_counter_ns()-start < 500_000_000:
                timed(fn,data)
                n += 1
            elapsed = (perf_counter_ns()-start)/1e9
            after = snapshot()
            deltas = {str(k):v-before[k] for k,v in after.items() if k in before and v>before[k]}
            output.append(dict(graph=c.name,key=key,wall_s=elapsed,calls=n,caller_tid=threading.get_native_id(),
                threads_before=len(before),threads_after=len(after),cpu_seconds_by_tid=deltas,
                average_busy_cpus=sum(deltas.values())/elapsed,threads_over_31ms=sum(d>=0.03125 for d in deltas.values()),
                **identities(fn)))
        jax.clear_caches()
    dump(folder/(env["resource"]+"_profile.json"),dict(environment=env,rows=output))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("capture","measure","profile","worker"))
    ap.add_argument("--phase",choices=("discovery","holdout"),default="discovery")
    ap.add_argument("--resource",choices=("one","four","default"),default="default")
    ap.add_argument("--out",type=Path,default=ROOT/"reproduction/partition_boundary")
    ap.add_argument("--job",choices=("capture","measure","profile"))
    ap.add_argument("--index",type=int,default=0)
    args = ap.parse_args()
    base = args.out.resolve()
    folder = base/args.phase
    if args.phase == "holdout" and not SEAL.exists():
        raise RuntimeError("The independent study has no frozen decision rule")
    if args.action == "worker":
        env = environment(args.resource)
        if args.job == "capture":
            capture(folder,args.phase,env)
        elif args.job == "measure":
            collect(folder,args.phase,env,args.index)
        else:
            profile(folder,env)
        return
    flags = worker_env("cpu")
    flags["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable,"-B","-m",__package__+".collect","worker","--job",args.action,
           "--out",str(base),"--phase",args.phase,"--resource",args.resource]
    if args.action == "capture":
        with tempfile.TemporaryDirectory(prefix="fs-grain-") as scratch:
            (Path(scratch)/"dump").mkdir()
            flags["XLA_FLAGS"] = "--xla_dump_to=dump --xla_dump_hlo_as_text=true --xla_dump_hlo_pass_re=parallel.task"
            subprocess.run(cmd,env=flags,cwd=scratch,check=True)
    elif args.action == "measure":
        dest = folder/args.resource
        dest.mkdir(parents=True,exist_ok=False)
        for i in range(PLAN["timing"]["processes"]):
            subprocess.run(cmd+["--index",str(i)],env=flags,cwd=ROOT,check=True)
        dump(dest/"manifest.json",dict(phase=args.phase,resource=args.resource,lock=lock(args.phase),
             archives={p.name:sha(p) for p in sorted(dest.glob("p*.zip"))}))
    else:
        subprocess.run(cmd,env=flags,cwd=ROOT,check=True)


if __name__ == "__main__":
    main()
