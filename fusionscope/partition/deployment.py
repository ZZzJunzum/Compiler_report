"""Price reusing an existing main executable and locate per-process break-even points."""
import argparse
from collections import defaultdict
from itertools import product
import json
from math import ceil,floor
from pathlib import Path
from statistics import geometric_mean,median
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from fusionscope.axis_probe.analysis import require,rows
from fusionscope.common import ROOT,dump,records,sha,table


def interval(fixed,slope):
    # Return the inclusive interval of positive integer N for which the choice is cheaper.
    if slope<0:
        return max(1,floor(fixed/-slope)+1),None
    if slope==0:
        return (1,None) if fixed<0 else (None,None)
    last = ceil(-fixed/slope)-1
    return (1,last) if last>=1 else (None,None)


def derive(root,dest):
    budgets = (1,100,1000,10000)
    intervals,aggregates,cell_rows = [],[],[]
    for panel in ("partition_boundary","scoped_assignment"):
        src = root/panel/"holdout"
        charges = records(src/"analysis/charges.csv")
        charges = [r for r in charges if r["policy"] in ("shape_guard","native_J")]
        groups = {(r["resource"],r["graph"],int(r["n"]),int(r["process"]),r["policy"]):r for r in charges}
        cells = sorted({(r["resource"],r["graph"]) for r in charges})
        cached = {}
        for res,i in product(("one","four","default"),range(5)):
            with ZipFile(src/res/f"p{i}.zip") as z:
                for b in rows(z,"builds.csv"):
                    components = json.loads(b.get("component_ns","{}"))
                    cached[res,b["graph"],i,b["key"]] = components.get("main",0)/1000
        for scenario in ("cold","main_available"):
            ratios = defaultdict(list)
            for res,g in cells:
                per_budget = defaultdict(list)
                for i in range(5):
                    chosen,base = (groups[res,g,1,i,k] for k in ("shape_guard","native_J"))
                    reuse = cached[res,g,i,chosen["chosen"]] if scenario=="main_available" else 0
                    fixed_policy = float(chosen["selection_us"])+float(chosen["build_us"])-reuse
                    fixed_base = float(base["build_us"])
                    t,u = float(chosen["call_us"]),float(base["call_us"])
                    lo,hi = interval(fixed_policy-fixed_base,t-u)
                    intervals.append(dict(panel=panel,scenario=scenario,resource=res,graph=g,process=i,
                        decision=chosen["chosen"],valid=chosen["valid"],fixed_delta_us=fixed_policy-fixed_base,
                        call_delta_us=t-u,first_winning_n=lo,last_winning_n=hi,
                        outcome="never" if lo is None else "unbounded" if hi is None else "bounded"))
                    for n in budgets:
                        total,ref = fixed_policy+n*t,fixed_base+n*u
                        per_budget[n].append((total/ref,(total-ref)/1000))
                for n,costs in per_budget.items():
                    ratio = median(x[0] for x in costs)
                    ratios[n].append(ratio)
                    cell_rows.append(dict(panel=panel,scenario=scenario,resource=res,graph=g,n=n,
                         ratio_to_native=ratio,paired_delta_ms=median(x[1] for x in costs)))
            for n,values in ratios.items():
                aggregates.append(dict(panel=panel,scenario=scenario,n=n,ratio_to_native=geometric_mean(values),
                                       cheaper_cells=sum(v<1 for v in values),total_cells=len(values)))
    table(dest/"break_even.csv",intervals)
    table(dest/"cells.csv",cell_rows)
    table(dest/"budgets.csv",aggregates)


if __name__=="__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify",action="store_true")
    args = ap.parse_args()
    base = ROOT/"results"
    dest = base/"partition_boundary/deployment"
    if args.verify:
        with TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            derive(base,scratch)
            require({p.name:sha(p) for p in dest.iterdir()}=={p.name:sha(p) for p in scratch.iterdir()},
                    "deployment sensitivity differs")
    else:
        derive(base,dest)
