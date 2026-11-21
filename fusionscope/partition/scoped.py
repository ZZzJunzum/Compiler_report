"""Run the observer closure with its own task-assignment setting in stock XLA."""
import argparse
import csv
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from time import perf_counter_ns
from zipfile import ZIP_DEFLATED, ZipFile

from fusionscope.common import ROOT, dump, read, sha
from fusionscope.axis_probe.collect import identities
from fusionscope.axis_probe.design import now
from fusionscope.phase2.experiment import csv_text, zip_write
from run import worker_env
from .collect import PLAN, PROTOCOL, environment, kernel_structure, validate

DESIGN = ROOT/"docs/plans/scoped_assignment.json"
SEAL = ROOT/"docs/plans/scoped_holdout.json"
PUBLISHED = ROOT/"results/scoped_assignment"
KEYS = ("J","S","split_on","split_off","off")


def lock(phase):
    here = Path(__file__).parent
    paths = [DESIGN,PROTOCOL,Path(__file__),here/"collect.py",here/"workloads.py"]
    if phase=="holdout":
        paths.extend([SEAL,here/"scoped_policy.py"])
    return {p.name:sha(p) for p in paths}


def candidate(c,key,args,capture=None):
    import jax
    from fusionscope.cpu_cost.workloads import observe
    from .workloads import activations, program
    parts,charges = {},{}

    def build(role,fn,disable):
        opts = {"xla_disable_hlo_passes":"cpu-parallel-task-assigner"} if disable else {}
        if capture is not None:
            before = set(Path("dump").glob("*.cpu_after_optimizations.txt"))
        start = perf_counter_ns()
        compiled = jax.jit(fn).lower(*args).compile(compiler_options=opts)
        charges[role] = perf_counter_ns()-start
        parts[role] = identities(compiled)
        if capture is not None:
            recent = set(Path("dump").glob("*.cpu_after_optimizations.txt"))-before
            if len(recent)!=1:
                raise ValueError((c,key,role,recent))
            dumpfile, = recent
            prefix = dumpfile.name.removesuffix("cpu_after_optimizations.txt")
            stem = f"{c.name}/{key}/{role}/"
            parts[role]["kernels"] = kernel_structure(dumpfile.read_text(encoding="utf8"))
            objects = {}
            for f in sorted(Path("dump").glob(prefix+"*")):
                if f.name.endswith((".o",".txt","ir-with-opt.ll")):
                    leaf = f.name.removeprefix(prefix)
                    content = f.read_bytes()
                    capture.writestr(stem+leaf,content)
                    if leaf.endswith(".o"):
                        objects[leaf] = hashlib.sha256(content).hexdigest()
            parts[role]["objects"] = objects
        return compiled

    if key in ("split_off","split_on"):
        def observer(x,p):
            _,sites = activations(c,x,p)
            return observe(sites,"row",True)
        main = None if c.kind=="minimal" else build("main",program(c,"V0"),False)
        counts = build("observer",observer,key=="split_off")
        def invoke(x,p):
            y = x if main is None else main(x,p)[0]
            return (y,*counts(x,p))
    else:
        invoke = build("whole",program(c,"J" if key=="off" else key),key=="off")
    return invoke,parts,charges


def collect(base,phase,env,index,archive=None):
    import jax
    from .workloads import cases,inputs,SEEDS,compile_one
    from fusionscope.bench import timed
    started = now()
    rng = random.Random(f"scoped/{phase}/{env['resource']}/{index}")
    graphs = list(cases(phase))
    rng.shuffle(graphs)
    times,checks,builds,decisions = [],[],[],[]
    if archive is not None:
        archive.writestr("environment.json",json.dumps(env))
    for c in graphs:
        data = inputs(c)
        args = data[SEEDS[0]]
        order = list(KEYS)
        rng.shuffle(order)
        fns = {}
        if phase=="holdout" and archive is None:
            from .scoped_policy import measure_choices
            decisions.extend(measure_choices(c))
        for i,k in enumerate(order):
            fn,parts,charges = candidate(c,k,args,archive)
            if archive is not None:
                archive.writestr(f"{c.name}/{k}/index.json",json.dumps(parts))
                jax.clear_caches()
            else:
                fns[k] = fn
                builds.append(dict(graph=c.name,key=k,order=i,build_ns=sum(charges.values()),
                                   component_ns=json.dumps(charges),components=json.dumps(parts)))
        if archive is not None:
            continue
        base_fn = compile_one(c,"V0",args)
        verified = validate(c,data,{**fns,"V0":base_fn})
        checks.extend(r for r in verified if r["key"]!="V0")
        seeds, lanes = list(SEEDS),[(k,a) for k in KEYS for a in PLAN["timing"]["aliases"]]
        for rnd in range(-PLAN["timing"]["warmups"],PLAN["timing"]["rounds"]):
            rng.shuffle(seeds)
            for block,seed in enumerate(seeds):
                rng.shuffle(lanes)
                for pos,(key,alias) in enumerate(lanes):
                    ns = timed(fns[key],data[seed])
                    times.append((c.name,seed,rnd,block,pos,key,alias,ns))
        del fns,fn,base_fn,data,args
        jax.clear_caches()
        gc.collect()
    if archive is not None:
        return
    raw = io.StringIO(newline="")
    writer = csv.writer(raw,lineterminator="\n")
    writer.writerow(("graph","seed","round","block","pos","key","alias","ns"))
    writer.writerows(times)
    meta = dict(process=index,pid=os.getpid(),start=started.isoformat(),end=now().isoformat(),
                environment=env,phase=phase,order=[c.name for c in graphs],lock=lock(phase))
    files = {"meta.json":json.dumps(meta),"timings.csv":raw.getvalue(),"checks.csv":csv_text(checks),
             "builds.csv":csv_text(builds)}
    if decisions:
        files["decisions.csv"] = csv_text(decisions)
    zip_write(base/phase/env["resource"]/f"p{index}.zip",files)
    print(phase,env["resource"],index,len(times),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("measure","capture","worker"))
    ap.add_argument("--phase",choices=("discovery","holdout"),default="discovery")
    ap.add_argument("--resource",choices=("one","four","default"),default="default")
    ap.add_argument("--out",type=Path,default=ROOT/"reproduction/scoped_assignment")
    ap.add_argument("--index",type=int,default=0)
    ap.add_argument("--dump",action="store_true")
    a = ap.parse_args()
    base = a.out.resolve()
    if a.phase=="holdout" and not SEAL.exists():
        raise RuntimeError("Observer-scoped rule must be frozen first")
    if a.action=="worker":
        env = environment(a.resource)
        if a.dump:
            target = base/a.phase/(a.resource+"_ir.zip")
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            partial = target.with_suffix(".partial")
            with ZipFile(partial,"w",ZIP_DEFLATED,compresslevel=5) as z:
                collect(base,a.phase,env,a.index,z)
            partial.replace(target)
        else:
            collect(base,a.phase,env,a.index)
        return
    flags = worker_env("cpu")
    flags["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable,"-B","-m",__package__+".scoped","worker","--out",str(base),
           "--phase",a.phase,"--resource",a.resource]
    if a.action=="capture":
        with tempfile.TemporaryDirectory(prefix="fs-scope-") as tmp:
            (Path(tmp)/"dump").mkdir()
            flags["XLA_FLAGS"] = "--xla_dump_to=dump --xla_dump_hlo_as_text=true --xla_dump_hlo_pass_re=parallel.task"
            subprocess.run(cmd+["--dump"],env=flags,cwd=tmp,check=True)
    else:
        dest = base/a.phase/a.resource
        dest.mkdir(parents=True,exist_ok=False)
        for i in range(PLAN["timing"]["processes"]):
            subprocess.run(cmd+["--index",str(i)],env=flags,cwd=ROOT,check=True)
        dump(dest/"manifest.json",dict(phase=a.phase,resource=a.resource,lock=lock(a.phase),
             archives={p.name:sha(p) for p in sorted(dest.glob("p*.zip"))}))


if __name__=="__main__":
    main()
