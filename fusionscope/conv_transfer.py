"""Prospective test of a one-image CPU convolution exception on matched readouts."""
import argparse
import gc
import json
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
from time import perf_counter_ns
from zipfile import ZipFile

from fusionscope.common import ROOT, SPEC, dump, read, sha
from fusionscope.conv_selection import MODELS, decide, fit, ir_facts, skip_operand, training_data
from fusionscope.phase3.design import Case, development, heldout
from run import pin, worker_env


SEEDS = (701, 1907, 3011)
ROUNDS = 30
PROCESSES = 5
VARIANTS = ("V0", "B1", "V2", "J1")
POLICIES = ("cap16", "exception", *MODELS)


def cases():
    return [Case(f"C{b}_{side}x{channel}_{top}", "CC", (b,side,side,channel), top, "review", b)
            for side,channel in ((28,16),(36,24)) for b in (4,8,16) for top in ("input","shared")]


def compile_case(c, v, args):
    import jax
    import jax.numpy as jnp
    from fusionscope.graphs import first_stage, layer
    from fusionscope.phase2.variants import compile_part

    def body(x,p):
        z1=first_stage("C",x,p)
        with jax.named_scope("u_relu"):
            a=jnp.maximum(z1,jnp.float32(0))
        with jax.named_scope("v_z"):
            z2=layer("C",a,p[2],p[3])+x
        with jax.named_scope("v_relu"):
            b=jnp.maximum(z2,jnp.float32(0))
        with jax.named_scope("audit_skip"):
            skip=(a if c.top=="shared" else x)*jnp.float32(.25)
        y=b*jnp.float32(.75)+jnp.float32(.125)+skip
        if v=="diagnostic":
            return y,z1,z2
        if v in ("V0","B1"):
            return y
        with jax.named_scope("joint_counts"):
            masks=((a>0).astype(jnp.int32),(b>0).astype(jnp.int32))
            q1,q2=jax.lax.reduce(masks,(jnp.int32(0),jnp.int32(0)),
                lambda u,w: (u[0]+w[0],u[1]+w[1]),tuple(range(a.ndim)))
        return y,q1,q2

    def per_image(x,p):
        out=jax.lax.map(lambda sample:body(sample,p), x.reshape((x.shape[0],1,*x.shape[1:])))
        if v=="B1":
            return out.reshape(x.shape)
        ys,us,vs=out
        return ys.reshape(x.shape),jnp.sum(us,dtype=jnp.int32),jnp.sum(vs,dtype=jnp.int32)
    return compile_part("main",per_image if v in ("B1","J1") else body,args)


def frozen_design(out):
    from fusionscope.phase3.experiment import signature
    review=read(out.parent/"analysis/summary.json")
    if not review["exception_advances"]:
        raise ValueError("The predeclared convolution exception did not pass the repeat.")
    old_shapes={c.shape for c in development()+heldout()}
    if any(c.shape in old_shapes for c in cases()):
        raise ValueError("A transfer shape was used in development or Phase III confirmation.")
    train=training_data()
    models={kind:fit(train,kind) for kind in MODELS}
    source={**signature(), **{name:sha(ROOT/name) for name in (
        "fusionscope/conv_transfer.py", "fusionscope/conv_selection.py", "fusionscope/hlo.py",
        "fusionscope/phase3/data.py", "docs/plans/CPU_Headroom_Audit.md")}}
    return dict(families=["CC"], seeds=list(SEEDS), rounds=ROUNDS, processes=PROCESSES,
        source=source, screen=sha(out.parent/"analysis/summary.json"), training=train, models=models,
        grids={c.name:dict(shape=c.shape,top=c.top,choices={
            k:decide(k,models,c) for k in POLICIES if k!="candidate_ir"}) for c in cases()})


def checked(out):
    design=read(out/"design.json")
    if any(sha(ROOT/p)!=digest for p,digest in design["source"].items()):
        raise ValueError("The transfer design or source changed.")
    return design


def score_clock(fn):
    # Charge source feature construction and policy evaluation once per workload.
    fn()
    samples=[]
    for _ in range(7):
        start=perf_counter_ns()
        for _ in range(30):
            fn()
        samples.append((perf_counter_ns()-start)/30000)
    return median(samples)


def inspect(out):
    pin()
    import jax
    from fusionscope.bench import environment
    from fusionscope.phase2.experiment import zip_write
    from fusionscope.phase3.workloads import arguments
    design=checked(out)
    if (out/"predictions.json").exists():
        existing=read(out/"predictions.json")
        if existing["design_sha256"]!=sha(out/"design.json") or existing["ir_sha256"]!=sha(out/"ir.zip"):
            raise ValueError("The existing compiler evidence or design differs.")
        return
    rng=random.Random("conv-review-design-20261118")
    grid=cases()
    rng.shuffle(grid)
    rows,ir={},{}
    for c in grid:
        args=jax.device_put(arguments(c,SEEDS[0]))
        jax.block_until_ready(args)
        order=["V2","J1"]
        rng.shuffle(order)
        parts={v:compile_case(c,v,args) for v in order}
        facts,memory,texts={},{},{}
        start=perf_counter_ns()
        for v,part in parts.items():
            texts[v]=part.compiled.as_text()
            memory[v]=part.compiled.memory_analysis().temp_size_in_bytes
            facts[v]=ir_facts(texts[v],memory[v],c.shape)
        differences=[x-y for x,y in zip(facts["J1"],facts["V2"])]
        extraction=(perf_counter_ns()-start)/1000
        policies={}
        scores={}
        for k in POLICIES:
            def choose():
                return decide(k,design["models"],c,differences)
            policies[k]=choose()
            scores[k]=score_clock(choose)
            if k!="candidate_ir" and policies[k]!=design["grids"][c.name]["choices"][k]:
                raise ValueError(f"Frozen source choice changed for {c.name}")
        rows[c.name]=dict(choices=policies,features=differences,temp_bytes=memory,
            extraction_us=extraction,score_us=scores,direct_skip=skip_operand(texts["V2"]))
        ir.update({f"{c.name}/{v}.hlo":text for v,text in texts.items()})
        jax.clear_caches()
    zip_write(out/"ir.zip",ir)
    dump(out/"predictions.json",dict(design_sha256=sha(out/"design.json"),ir_sha256=sha(out/"ir.zip"),
        environment=environment("cpu"),cases=rows))
    print(f"Locked decisions for {len(rows)} graphs without timing a candidate.",flush=True)


def validate(c,inputs,fns):
    import numpy as np
    from jax import block_until_ready
    diag=compile_case(c,"diagnostic",inputs[SEEDS[0]]).compiled
    rtol,atol=SPEC["tolerances"]["C"]
    rows=[]
    for seed,args in inputs.items():
        reference,z1,z2=block_until_ready(diag(*args))
        ref=np.asarray(reference)
        counts=[int(np.count_nonzero(np.asarray(z)>0)) for z in (z1,z2)]
        for v,fn in fns.items():
            r=block_until_ready(fn(*args))
            y,*qs=r if isinstance(r,tuple) else (r,)
            y=np.asarray(y)
            expected=[] if v in ("V0","B1") else counts
            observed=[int(q) for q in qs]
            ok=(y.shape==c.shape and y.dtype==np.float32 and np.isfinite(y).all() and
                np.allclose(y,ref,rtol=rtol,atol=atol) and observed==expected and
                all(q.shape==() and q.dtype==np.int32 for q in qs))
            rows.append(dict(case=c.name,seed=seed,variant=v,expected=expected,actual=observed,
                             max_abs=float(np.max(np.abs(y-ref))),valid=bool(ok)))
    return rows


def measure(out,pid):
    pin()
    import jax
    from fusionscope.bench import environment
    from fusionscope.hlo import Module
    from fusionscope.phase2.experiment import csv_text, zip_write
    from fusionscope.phase3.experiment import block
    from fusionscope.phase3.workloads import arguments
    checked(out)
    pred=read(out/"predictions.json")
    env=environment("cpu")
    if (pred["design_sha256"]!=sha(out/"design.json") or pred["ir_sha256"]!=sha(out/"ir.zip")
            or pred["environment"]!=env):
        raise ValueError("Prediction evidence or environment differs before confirmation.")
    digest=sha(out/"predictions.json")
    target=out/"repeat"/f"p{pid}.zip"
    if target.exists():
        with ZipFile(target) as z:
            meta=json.loads(z.read("environment.json"))
            if z.testzip() or (meta["process"],meta["lock"],meta["environment"])!=(pid,digest,env):
                raise ValueError(f"Incompatible existing confirmation: {target}")
        return
    rng=random.Random(f"conv-review-confirmation:{pid}")
    grid=cases()
    rng.shuffle(grid)
    samples,checks,builds=[],[],[]
    for c in grid:
        inputs={s:jax.device_put(arguments(c,s)) for s in SEEDS}
        jax.block_until_ready(inputs)
        args=inputs[SEEDS[0]]
        order=list(VARIANTS)
        rng.shuffle(order)
        fns={}
        for v in order:
            start=perf_counter_ns()
            part=compile_case(c,v,args)
            elapsed=perf_counter_ns()-start
            fns[v]=part.compiled
            builds.append(dict(case=c.name,process=pid,variant=v,component="main",build_wall_ns=elapsed,
                               fingerprint=Module(part.compiled.as_text()).fingerprint()))
        checks.extend(validate(c,inputs,fns))
        gc.collect()
        samples.extend(block(c,pid,"candidate",fns,args,rng,ROUNDS))
        for v in order:
            samples.extend(block(c,pid,"AA_"+v,dict.fromkeys("ABCD",fns[v]),args,rng,ROUNDS))
        jax.clear_caches()
    info=dict(lock=digest,process=pid,environment=env,case_order=[c.name for c in grid],
              rounds=ROUNDS,warmups=6,seeds=SEEDS)
    zip_write(target,{"environment.json":json.dumps(info,indent=2),"timings.csv":csv_text(samples),
                      "builds.csv":csv_text(builds),"correctness.csv":csv_text(checks)})
    print(f"Transfer {pid+1}/{PROCESSES}: {len(samples)} calls, {len(checks)} checks",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("freeze","inspect","measure"))
    parser.add_argument("--out",type=Path,default=ROOT/"reproduction/headroom/conv_transfer")
    parser.add_argument("--process",type=int,choices=range(PROCESSES))
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    out=args.out.resolve()
    if args.stage=="freeze":
        design=frozen_design(out)
        path=out/"design.json"
        if not path.exists():
            dump(path,design)
        elif read(path)!=json.loads(json.dumps(design)):
            raise ValueError("Another design already exists.")
        print(f"Source policies frozen: {sha(path)}",flush=True)
    elif args.worker and args.stage=="inspect":
        inspect(out)
    elif args.worker and args.process is not None:
        measure(out,args.process)
    else:
        prefix=[sys.executable,"-B","-m","fusionscope.conv_transfer",args.stage,"--out",str(out),"--worker"]
        for pid in ([None] if args.stage=="inspect" else
                    [args.process] if args.process is not None else range(PROCESSES)):
            cmd=prefix if pid is None else [*prefix,"--process",str(pid)]
            subprocess.run(cmd,env=worker_env("cpu"),cwd=ROOT,check=True)


if __name__=="__main__":
    main()
