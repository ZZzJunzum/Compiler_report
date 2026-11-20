"""Freeze source-only policies using the discovery cohorts before compiling held-out graphs."""
import argparse
from itertools import product
from math import exp, log
from statistics import median

from fusionscope.common import dump, sha
from .analysis import load_resource
from .collect import PUBLISHED, SEAL
from .workloads import RESOURCES, SEEDS, cases

MENU = ("J","S","P")
PRIMARY_N = 1000
GRID = [2**i for i in range(22)]


def fit(base):
    inputs,losses,valid = [],[],[]
    for resource in RESOURCES:
        processes = load_resource(base/"discovery","discovery",resource,{})
        ncpu = len(processes[0]["meta"]["environment"]["affinity"])
        for c in cases("discovery"):
            costs = {k:[int(p["build"][c.name,k]["build_ns"])/1000 + PRIMARY_N*p["means"][c.name,k,"A"]
                        for p in processes] for k in MENU}
            inputs.append((ncpu,c.rows*c.width/ncpu,c.rows/ncpu))
            losses.append({k:log(median(x/y for x,y in zip(costs[k],costs["J"],strict=True))) for k in MENU})
            valid.append({k:all(p["checks"][c.name,k,s] for p in processes for s in SEEDS) for k in MENU})

    def leaf(ids):
        options = [k for k in MENU if all(valid[i][k] for i in ids)]
        return min(options,key=lambda k:sum(losses[i][k] for i in ids))

    def stump(feature,ids):
        options = []
        for threshold in GRID:
            lo = [i for i in ids if inputs[i][feature] <= threshold]
            hi = [i for i in ids if inputs[i][feature] > threshold]
            if min(len(lo),len(hi))<2:
                continue
            a,b = leaf(lo),leaf(hi)
            err = sum(losses[i][a] for i in lo)+sum(losses[i][b] for i in hi)
            options.append((err,dict(feature=feature,at=threshold,low=a,high=b)))
        baseline = leaf(ids)
        options.insert(0,(sum(losses[i][baseline] for i in ids),baseline))
        return min(options,key=lambda p:round(p[0],12))

    ids = list(range(len(inputs)))
    by_cpu = {str(n):leaf([i for i in ids if inputs[i][0]==n]) for n in sorted({x[0] for x in inputs})}
    grain_loss,grain = stump(1,ids)
    choices = [(grain_loss,grain)]
    for split in GRID:
        lo = [i for i in ids if inputs[i][2]<=split]
        hi = [i for i in ids if inputs[i][2]>split]
        if min(len(lo),len(hi))<2:
            continue
        ll,l = stump(1,lo)
        rr,r = stump(1,hi)
        choices.append((ll+rr,dict(feature=2,at=split,low=l,high=r)))
    best_loss,guard = min(choices,key=lambda p:round(p[0],12))
    return dict(scope="Two rowwise int32 predicate counts on the last axis with source arrays of the same shape. The packed implementation is eligible only under its uint32 bound.",
        inputs=["allowed logical CPUs","input elements per allowed CPU","output rows per allowed CPU"],
        cpu_map=by_cpu,unknown_affinity_default="J",grain=grain,shape_guard=guard,
        selection="With no prior affinity lookup, the three adaptive policies query the OS once per graph. Static native policies cost zero to select. Packed policies check the width. Compile only the chosen implementation and reuse it for N calls. Evaluator times every alternative for matched comparisons.",
        train="Equal weight per discovery graph/resource cell. Minimize mean log paired total-cost ratio at N=1000 against native J, with lowering+build charged. Grid of powers of two, minimum two cells per leaf. Joint, separate, packed eligible at every leaf whose observed semantic checks all pass. Ties favor a simpler tree, smaller threshold and J then S then P.",
        training_cells=len(inputs),training_ratios=dict(grain=exp(grain_loss/len(ids)),shape_guard=exp(best_loss/len(ids))),
        discovery_manifests={r:sha(base/"discovery"/r/"manifest.json") for r in RESOURCES},
        comparators=["native_J","separate","packed","resource","grain"],calls=[1,100,1000,10000],
        primary_N=PRIMARY_N,max_ratio=0.95,
        endpoint="All 4 reserved structures x 2 sizes x 3 affinities = 24 cells, equal weight. Per cell median paired total-cost ratio over 5 processes, then geometric mean across cells. Strongest simple comparator has lowest aggregate cost ratio against native J at N=1000. A method claim requires ratio <= 0.95 against that comparator, all chosen output checks valid, and zero resolved native-J latency harms at default affinity. Raw and invalid cells remain visible. Publish each policy and group and every budget irrespective of outcome.")


if __name__=="__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify",action="store_true")
    args = ap.parse_args()
    expected = fit(PUBLISHED)
    if args.verify:
        from fusionscope.common import read
        if read(SEAL)!=expected:
            raise ValueError("Frozen rule differs from discovery replay")
    elif SEAL.exists():
        raise FileExistsError(SEAL)
    else:
        dump(SEAL,expected)
