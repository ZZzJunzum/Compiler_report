"""Check scoped binaries, fit the registered rules, and replay held-out costs."""
import argparse
from itertools import product
import hashlib
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from fusionscope.axis_probe.analysis import require
from fusionscope.common import dump,read,sha,table
from .analysis import evaluate,load_resource,paired
from .fit import fit
from .scoped import KEYS,PUBLISHED,SEAL,lock
from .workloads import RESOURCES,SEEDS,cases


def load(folder,phase,resource,ir):
    policy = None
    if phase=="holdout":
        from . import scoped_policy as policy
    return load_resource(folder,phase,resource,{},keys=KEYS,signature=lock,policy_set=policy)


def fitted(base):
    spec = fit(base,("J","S","split_off"),load)
    spec.update(scope="Compile the unchanged main with native XLA. Disable task assignment only for the separately compiled observer and its required producers. For a declared identity main, forward input directly. Split_on uses identical graph boundaries and native assignment for both.",
        comparators=["native_J","separate","scoped_off","resource","grain"],
        selection="Fixed mappings are free. Each adaptive choice includes querying current OS affinity and traversing its frozen rule. Cost includes every selected component build and one end-to-end ready latency times N.",
        train="As registered in scoped_assignment.json, minimize the discovery mean of log paired total-cost ratios at N=1000 using the same tree fitter as the packed panel. Tie preference J, S, split_off.")
    return spec


def derive(base,phase,dest):
    archive = base/phase/"default_ir.zip"
    ids,objects = {},[]
    with ZipFile(archive) as z:
        require(z.testzip() is None,archive)
        env = json.loads(z.read("environment.json"))
        original = base.parent/"partition_boundary"/phase/"default_ir.zip"
        # The neighboring independent panel exposes the original main-only object.
        with ZipFile(original) as baseline:
            for c in cases(phase):
                for k in KEYS:
                    stem = f"{c.name}/{k}/"
                    parts = json.loads(z.read(stem+"index.json"))
                    ids[c.name,k] = parts
                    for role,entry in parts.items():
                        for leaf,digest in entry["objects"].items():
                            require(hashlib.sha256(z.read(stem+role+"/"+leaf)).hexdigest()==digest,(c,k,leaf))
                on,off = (ids[c.name,k] for k in ("split_on","split_off"))
                require(set(on)==set(off)==({"observer"} if c.kind=="minimal" else {"main","observer"}),c)
                main_equal = None
                if "main" in on:
                    source = json.loads(baseline.read(f"{c.name}/V0/index.json"))
                    main_equal = on["main"]["objects"]==off["main"]["objects"]==source["objects"]
                    require(main_equal and on["main"]["named_schedule"]==source["named_schedule"],c.name)
                a,b = on["observer"],off["observer"]
                objects.append(dict(graph=c.name,main_build="identity" if main_equal is None else "native",
                    main_matches_native=main_equal,observer_identical=a["objects"]==b["objects"],
                    observer_on_parts=json.dumps([v["partitions"] for v in a["kernels"] if v["partitions"]]),
                    observer_off_parts=json.dumps([v["partitions"] for v in b["kernels"] if v["partitions"]])))
    procs, effects,checks,charges = {},[],[],[]
    for res in RESOURCES:
        ps = load(base/phase,phase,res,{})
        procs[res] = ps
        for p in ps:
            require(p["meta"]["environment"]["compiler_sha256"]==env["compiler_sha256"],res)
            for (graph,k),r in p["build"].items():
                parts,costs = json.loads(r["components"]),json.loads(r["component_ns"])
                require(set(parts)==set(costs)==set(ids[graph,k]) and sum(costs.values())==int(r["build_ns"]),r)
                for role,part in parts.items():
                    require(all(v==ids[graph,k][role][x] for x,v in part.items()),(res,graph,k,role))
        for c in cases(phase):
            for l,r in (("split_off","J"),("split_on","J"),("split_off","split_on"),("J","off"),("J","S")):
                eff,_,_ = paired(ps,c.name,l,r)
                effects.append(dict(resource=res,graph=c.name,kind=c.kind,left=l,right=r,**eff))
            for k in KEYS:
                ok = sum(p["checks"][c.name,k,s] for p in ps for s in SEEDS)
                checks.append(dict(resource=res,graph=c.name,key=k,passed=ok,total=len(ps)*len(SEEDS)))
                charges.append(dict(resource=res,graph=c.name,key=k,
                    latency_us=median(p["means"][c.name,k,"A"] for p in ps),
                    build_ms=median(int(p["build"][c.name,k]["build_ns"]) for p in ps)/1e6))
    for filename,entries in (("scoping",objects),("pairs",effects),("validity",checks),("latencies",charges)):
        table(dest/(filename+".csv"),entries)
    if phase=="holdout":
        from . import scoped_policy
        evaluate(procs,dest,scoped_policy,SEAL)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("freeze","check-freeze","report","verify"))
    ap.add_argument("--phase",choices=("discovery","holdout"),default="discovery")
    ap.add_argument("--out",type=Path,default=PUBLISHED)
    a = ap.parse_args()
    dest = a.out/a.phase/"analysis"
    if a.action=="freeze":
        if SEAL.exists():
            raise FileExistsError(SEAL)
        dump(SEAL,fitted(a.out))
    elif a.action=="check-freeze":
        require(read(SEAL)==fitted(a.out),"scoped freeze differs")
    elif a.action=="report":
        derive(a.out,a.phase,dest)
    else:
        with TemporaryDirectory() as root:
            temp = Path(root)
            derive(a.out,a.phase,temp)
            require({f.name:sha(f) for f in temp.iterdir()}=={f.name:sha(f) for f in dest.iterdir()},
                    "scoped replay differs")
    print(a.action,a.phase,"OK")


if __name__=="__main__":
    main()
