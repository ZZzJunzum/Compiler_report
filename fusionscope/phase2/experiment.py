"""Run one CPU cohort at a time. Published data and new runs use separate roots."""
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
from time import perf_counter_ns
from zipfile import ZIP_DEFLATED, ZipFile

from fusionscope.common import ROOT, SPEC, conditions, describe, dump, read, sha
from .audit_ir import ARCHIVE, PLAN

PHASES={"screen":"screening","train":"matrix","confirm":"confirmation","external":"external"}


def cases_for(phase):
    if phase=="screen":
        return list(PLAN["screen_cases"])
    return list(PLAN["external"]) if phase=="external" else conditions()


def csv_text(rows):
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def zip_write(path,items):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".partial")
    with ZipFile(tmp,"w",ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in items.items():
            z.writestr(name,data)
    tmp.replace(path)


def zip_rows(path,name):
    with ZipFile(path) as archive:
        return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf8"))))


def measure_block(case,process,block,fns,args,rng):
    from fusionscope.bench import timed
    keys=list(fns)
    rows=[]
    for round_id in range(-PLAN["warmups"],PLAN["rounds"]):
        rng.shuffle(keys)
        for order,key in enumerate(keys):
            elapsed=timed(fns[key],args)
            rows.append(dict(case=case,process=process,block=block,round=round_id,order=order,
                             variant=key,elapsed_ns=elapsed))
    return rows


def validate(case,all_args,fns,out):
    import jax
    import numpy as np
    from fusionscope.graphs import program
    diag=jax.jit(program(case,"diagnostic"))
    rtol,atol=SPEC["tolerances"][describe(case)[0]]
    checks=[]
    for seed,args in all_args.items():
        ref=np.asarray(fns["V0"](*args))
        _,zu,zv=diag(*args)
        qs=[int(np.count_nonzero(np.asarray(z)>0)) for z in (zu,zv)]
        for name,fn in fns.items():
            result=fn(*args)
            y,*q=result if isinstance(result,tuple) else (result,)
            y=np.asarray(y)
            expected=[] if name=="V0" else qs
            actual=[int(v) for v in q]
            valid_y=bool(y.dtype==ref.dtype==np.float32 and y.shape==ref.shape
                         and np.isfinite(y).all() and np.allclose(y,ref,rtol=rtol,atol=atol))
            valid_q=actual==expected and all(v.shape==() and v.dtype==np.int32 for v in q)
            checks.append(dict(case=case,variant=name,seed=seed,y_max_abs=float(np.max(np.abs(y-ref))),
                               actual_counts=str(actual),reference_counts=str(expected),
                               y_valid=valid_y,counts_valid=valid_q,valid=valid_y and valid_q))
            if not (valid_y and valid_q):
                path=out/"failures"/f"{case}-{seed}.npz"
                if not path.exists():
                    path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,**{f"leaf{i}":np.asarray(v) for i,v in enumerate(jax.tree.leaves(args))})
    return checks


def worker(out,phase,process):
    from run import pin
    pin()
    import jax
    import psutil
    from fusionscope.bench import environment
    from fusionscope.evidence import scheduled_text_digest
    from fusionscope.graphs import inputs
    from .variants import build,options
    folder=out/PHASES[phase]/"raw"
    target=folder/f"p{process}.zip"
    env=environment("cpu")
    signature={name:sha(path) for name,path in {
        "protocol":ARCHIVE/"protocol.json","graphs":ROOT/"fusionscope/graphs.py",
        "variants":Path(__file__).with_name("variants.py"),"worker":Path(__file__)}.items()}
    if target.exists():
        with ZipFile(target) as z:
            meta=json.loads(z.read("environment.json"))
            if z.testzip() or meta["measurement_signature"]!=signature or meta["environment"]!=env:
                raise ValueError(f"Existing process archive differs: {target}. Use a fresh output directory.")
        return
    timings,checks,builds=[],[],[]
    cases=cases_for(phase)
    rng=random.Random(f"{PLAN['random_seed']}:{phase}:{process}")
    rng.shuffle(cases)
    for case in cases:
        args_by_seed={s:jax.device_put(inputs(case,s),jax.devices()[0]) for s in PLAN["input_seeds"]}
        jax.block_until_ready(args_by_seed)
        args=args_by_seed[PLAN["timed_seed"]]
        labels=options(case)
        rng.shuffle(labels)
        fns,ir,manifest={}, {}, []
        for name in labels:
            fns[name],parts=build(case,name,args)
            for part in parts:
                prefix=f"{name}/{part.name}"
                hlo=part.compiled.as_text()
                ir[prefix+"/optimized_hlo.txt"]=hlo
                ir[prefix+"/stablehlo.mlir"]=part.lowered.as_text()
                mem=part.compiled.memory_analysis()
                manifest.append(dict(case=case,variant=name,component=part.name,lower_ns=part.lower_ns,
                                     compile_ns=part.compile_ns,hlo_digest=scheduled_text_digest(hlo),
                                     argument_bytes=mem.argument_size_in_bytes,output_bytes=mem.output_size_in_bytes,
                                     temp_bytes=mem.temp_size_in_bytes,alias_bytes=mem.alias_size_in_bytes))
        identity=sorted((p["variant"],p["component"],p["hlo_digest"]) for p in manifest)
        stamp=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
        archive=f"ir/{case}-{stamp}.zip"
        if not (out/archive).exists():
            zip_write(out/archive,ir)
        for row in manifest:
            builds.append({**row,"ir_archive":archive})
        checks.extend(validate(case,args_by_seed,fns,out))
        gc.collect()
        timings.extend(measure_block(case,process,"candidate",fns,args,rng))
        for name in ("V0","V1"):
            lanes=dict.fromkeys(("A","B","C","D"),fns[name])
            timings.extend(measure_block(case,process,"AA_"+name,lanes,args,rng))
        jax.clear_caches()
    peak=getattr(psutil.Process().memory_info(),"peak_wset",None)
    meta={"phase":phase,"process":process,"pid":os.getpid(),"case_order":cases,"environment":env,
          "measurement_signature":signature,"process_lifetime_peak_working_set_bytes":peak,
          "endpoint":"device-resident inputs to readiness of all outputs, no host copy; negative rounds are warmup"}
    zip_write(target,{"environment.json":json.dumps(meta,indent=2),"timings.csv":csv_text(timings),
                      "correctness.csv":csv_text(checks),"builds.csv":csv_text(builds)})
    print(f"{phase} p{process} | {len(cases)} conditions, {len(timings)} warmup/timed calls",flush=True)


def run_phase(out,phase):
    from run import worker_env
    for process in range(PLAN["processes"][phase]):
        subprocess.run([sys.executable,"-m","fusionscope.phase2.experiment","--worker",phase,
                        "--process",str(process),"--out",str(out)],cwd=ROOT,env=worker_env("cpu"),check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",choices=("audit","screen","matrix","model","confirm","external","report","verify","all"),default="all")
    ap.add_argument("--out",type=Path)
    ap.add_argument("--worker",choices=PHASES,help=argparse.SUPPRESS)
    ap.add_argument("--process",type=int,default=0,help=argparse.SUPPRESS)
    args=ap.parse_args()
    out=(args.out or (ARCHIVE if args.stage in ("report","verify") else ROOT/"reproduction/phase2")).resolve()
    if args.worker:
        worker(out,args.worker,args.process)
        return
    stages=("audit","screen","matrix","model","confirm","external","report","verify") if args.stage=="all" else (args.stage,)
    for stage in stages:
        if stage=="audit":
            from .audit_ir import run
            run(out)
        elif stage=="screen":
            if not (out/"analysis/duplicated_ops.csv").exists():
                raise ValueError("Run the IR audit first")
            run_phase(out,"screen")
            from .structure import screen_summary
            screen_summary(out)
        elif stage=="matrix":
            if not read(out/"screening/gate.json")["structural_change"]:
                raise ValueError("Screen did not demonstrate a structural change")
            run_phase(out,"train")
        elif stage=="model":
            from .cost_model import lock
            lock(out)
        elif stage in ("confirm","external"):
            from .cost_model import verify_lock
            verify_lock(out)
            run_phase(out,stage)
        elif stage=="report":
            from .report import report
            report(out)
        else:
            from .verify import verify
            verify(out)


if __name__=="__main__":
    main()
