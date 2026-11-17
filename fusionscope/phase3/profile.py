"""Separate CPU profiling and LLVM checks for apparent wins on equivalent HLO."""
import argparse
import ctypes
import hashlib
from pathlib import Path
import random
import re
from statistics import median
import subprocess
import sys
import tempfile

from fusionscope.common import ROOT, dump, records, table
from fusionscope.phase2.experiment import zip_write
from .design import MECHANISM, development, heldout, mechanism_options


def cycles():
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    query=kernel.QueryProcessCycleTime
    query.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulonglong)]
    query.restype=ctypes.c_int
    def read():
        value=ctypes.c_ulonglong()
        if not query(ctypes.c_void_p(-1),ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.value
    return read


def normalized_ll(data):
    text=data.decode()
    lines=[]
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith((";","source_filename","!")):
            continue
        # Compiler-selected identities/debug attachments can change between modules.
        line=re.sub(r', ![\w.]+ !\d+','',line)
        lines.append(line)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def worker(out):
    from run import pin
    pin()
    import jax
    from fusionscope.bench import timed
    from fusionscope.hlo import Module
    from .workloads import arguments, build
    folder=out/"profile"
    cases={c.name:c for c in development()+heldout()}
    counter_rows=[]
    counter_state={"event":"Windows QueryProcessCycleTime","scope":"all threads of this pinned process",
                   "multiplexing":"no PMU multiplexing was configured",
                   "other_events":"retired instructions, cache misses and DRAM bytes unavailable in this interface"}
    try:
        clock=cycles()
        overhead=median((lambda first:clock()-first)(clock()) for _ in range(51))
        for name in MECHANISM:
            c=cases[name]
            args=arguments(c,42)
            fns={v:build(c,v,args)[0] for v in mechanism_options(c)}
            rng=random.Random("profile:"+name)
            for pid in range(-6,15):
                keys=list(fns)
                rng.shuffle(keys)
                for v in keys:
                    begin=clock()
                    ns=timed(fns[v],args)
                    total=clock()-begin
                    if pid>=0:
                        counter_rows.append(dict(case=name,variant=v,round=pid,wall_ns=ns,process_cycles=total,
                                                 empty_read_cycles=overhead))
            jax.clear_caches()
        table(folder/"process_cycles.csv",counter_rows)
        counter_state["available"]=True
    except (OSError,AttributeError) as exc:
        counter_state.update(available=False,error=str(exc))
    dump(folder/"counters.json",counter_state)
    pairs=[r for r in records(out/"analysis/equivalence.csv") if r["equal_hlo"]=="True" and r["resolved"]=="True"]
    audit, measures, llvm = [],[],{}
    for row in pairs:
        c=cases[row["case"]]
        args=arguments(c,101)
        jax.block_until_ready(args)
        fns,hashes,fps={}, {}, {}
        for v in (row["direct"],row["tile"]):
            before=set(Path("dump").glob("*"))
            fn,(part,)=build(c,v,args)
            fns[v]=fn
            fps[v]=Module(part.compiled.as_text()).fingerprint()
            files=sorted(p for p in set(Path("dump").glob("*"))-before if p.name.endswith("ir-with-opt.ll"))
            hashes[v]=sorted(normalized_ll(p.read_bytes()) for p in files)
            for p in files:
                llvm[f"{c.name}/{v}/{p.name}"]=p.read_bytes()
        rng=random.Random("llvm:"+c.name+row["tile"])
        times={v:[] for v in fns}
        for i in range(-6,30):
            keys=list(fns)
            rng.shuffle(keys)
            for v in keys:
                ns=timed(fns[v],args)
                measures.append(dict(case=c.name,direct=row["direct"],tile=row["tile"],variant=v,round=i,elapsed_ns=ns))
                if i>=0:
                    times[v].append(ns/1000)
        direct,tile=row["direct"],row["tile"]
        audit.append(dict(case=c.name,direct=direct,tile=tile,profile_hlo_equal=fps[direct]==fps[tile],
                     direct_kernels=len(hashes[direct]),tile_kernels=len(hashes[tile]),
                     equal_normalized_llvm=bool(hashes[tile]) and hashes[direct]==hashes[tile],
                     direct_fingerprints=str(hashes[direct]),tile_fingerprints=str(hashes[tile]),
                     confirmation_gap_us=row["paired_direct_minus_tile_us"],confirmation_floor_us=row["floor_us"],
                     separate_recheck_gap_us=median(times[direct])-median(times[tile])))
        jax.clear_caches()
    if pairs:
        table(folder/"llvm_checks.csv",audit)
        table(folder/"llvm_rechecks.csv",measures)
        if llvm:
            zip_write(folder/"equivalent_programs_llvm.zip",llvm)
    dump(folder/"coverage.json",dict(mechanism_counter_configurations=len({(r["case"],r["variant"]) for r in counter_rows}),
                                    equal_hlo_pairs_rechecked=len(audit),llvm_members=len(llvm)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--worker",action="store_true")
    a=ap.parse_args()
    out=a.out.resolve()
    if a.worker:
        worker(out)
        return
    from run import worker_env
    with tempfile.TemporaryDirectory(prefix="fusion-profile-") as tmp:
        env=worker_env("cpu")
        env.update(PYTHONPATH=str(ROOT),XLA_FLAGS="--xla_dump_to=dump --xla_dump_hlo_as_text=true")
        subprocess.run([sys.executable,"-m","fusionscope.phase3.profile","--out",str(out),"--worker"],
                       cwd=tmp,env=env,check=True)


if __name__=="__main__":
    main()
