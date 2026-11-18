"""Post-timing check of whether the joint program still exports a mapped full tensor."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

from fusionscope.common import ROOT, read, records, table
from fusionscope.phase2.experiment import zip_rows, zip_write
from run import pin, worker_env
from .experiment import PROCESSES, SEEDS
from .features import trigger
from .workloads import arguments, cases, parameters, program


def collect(out):
    import jax
    if (out/"test/joint_hlo.zip").exists() or not (out/"test/p4.zip").is_file():
        raise ValueError("Collect joint listings once, after the last blind process")
    files = {}
    chosen = read(out/"gate.json")["choices"]
    for c in cases("test"):
        args = arguments(c,parameters(c),SEEDS["test"][0])
        for key in chosen:
            graph, contract = key.split("/")
            if c.name == graph:
                files[key+".hlo"] = jax.jit(program(c,contract+"_J")).lower(*args).compile().as_text()
    zip_write(out/"test/joint_hlo.zip",files)


def derive(out):
    graphs = {c.name:c for c in cases("test")}
    gate = read(out/"gate.json")["choices"]
    fps = {}
    for i in range(PROCESSES["test"]):
        fps[i] = {(r["case"],r["variant"]):r["fingerprint"] for r in zip_rows(out/f"test/p{i}.zip","builds.csv")}
    rows = []
    with ZipFile(out/"test/joint_hlo.zip") as z:
        assert z.testzip() is None and set(z.namelist()) == {k+".hlo" for k in gate}
        for r in records(out/"analysis/rewrite_pairs.csv"):
            c, q = r["case"], r["contract"]
            key = c+"/"+q
            anchors, fingerprint = trigger(z.read(key+".hlo").decode(),graphs[c],q)
            assert all(fps[i][c,q+"_J"]==fingerprint for i in fps)
            before = gate[key]["anchors"]
            rows.append(dict(case=c,contract=q,outcome=r["outcome"],ratio=r["ratio"],
                             separate_exports=len(before),joint_exports=len(anchors),
                             separate_anchors=" | ".join(before),joint_anchors=" | ".join(anchors),
                             fingerprint=fingerprint))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["collect","analyze","verify","worker"])
    parser.add_argument("--out",type=Path,default=ROOT/"reproduction/cpu_cost")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == "collect":
        subprocess.run([sys.executable,"-B","-m","fusionscope.cpu_cost.mechanisms","worker","--out",str(out)],
                       env=worker_env("cpu"), check=True)
    elif args.stage == "worker":
        pin()
        collect(out)
    else:
        dest = out/"mechanisms.csv"
        rows = derive(out)
        if args.stage == "analyze":
            table(dest,rows)
        else:
            assert records(dest) == [{k:str(v) for k,v in r.items()} for r in rows]
        print(json.dumps(dict(Counter(f"{r['separate_exports']}->{r['joint_exports']}" for r in rows))))


if __name__ == "__main__":
    main()
