"""Last independent test of an absolute-time interpolation and a direct paired cost model."""
import argparse
from collections import Counter, defaultdict
import gc
import json
from math import exp, log
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np

from fusionscope.common import ROOT, dump, read, records, sha, table
from fusionscope.paired_repeat import contrast, load_runs
from fusionscope.phase2.experiment import csv_text, zip_rows, zip_write
from run import pin, worker_env
from .experiment import validation
from .features import source, stable_features
from .model import archive_features, development_targets
from .workloads import Case, arguments, cases, parameters, program, variants

SEEDS = (941,2803,7919)
QUERIES = ("count","row","energy","shared","sample")


def fresh():
    graphs = []
    for fam in "EMNC":
        shapes = [(3,19,19,14),(5,25,25,18)] if fam == "C" else [(96,112),(384,176)]
        for i,(shape,top) in enumerate(zip(shapes, ["chain","branch"], strict=True)):
            graphs.append(Case(f"p_{fam}{i}",fam,shape,top,i,"new_shape"))
    for fam,shape in [("G",(112,104)),("R",(2,21,21,16)),("B",(2,20,20,20))]:
        graphs.append(Case("p_"+fam,fam,shape,"native",0,"new_architecture"))
    return graphs


def menu(c):
    return [n for n in variants(c) if n=="V0" or n.split("_")[0] in QUERIES]


def coordinate(row, mode):
    if mode=="source":
        src = row["source"]
        group = next((k for k in ("conv","dot","norm") if src.get(k)), "elementwise")
        return group, log(src["main"]+src["conv"]+src["dot"]+src["norm"])
    s = row["stable"]
    group = next((g for k,g in [("conv_flops","conv"),("dot_flops","dot"),("op_rsqrt","norm")] if s.get(k)), "elementwise")
    work = sum(s.get(k,0) for k in ("conv_flops","dot_flops","unary_domain","variadic_domain"))
    work += sum(v for k,v in s.items() if k.startswith("volume_") and
                k[7:] not in ("constant","broadcast_in_dim","reshape","return","convolution","dot_general","reduce"))
    return group, log(max(work,1))


def ratio_targets(parent):
    process = []
    for i in range(3):
        by = defaultdict(list)
        for r in zip_rows(parent/f"development/p{i}.zip","timings.csv"):
            if int(r["round"]) >= 0:
                by[r["case"]+"/"+r["variant"]].append(int(r["elapsed_ns"]))
        process.append({k:median(v) for k,v in by.items()})
    return {k:median(log(p[k]/p[k[:-1]+"S"]) for p in process) for k in process[0] if k.endswith("_J")}


def curve(points, x, is_ratio):
    points = sorted(points)
    xs, ys = map(np.asarray, zip(*points))
    if not np.all(np.diff(xs)>0):
        raise ValueError("Development work must distinguish the four sizes")
    if is_ratio or xs[0]<=x<=xs[-1]:
        return float(np.interp(x,xs,ys))
    i = 0 if x<xs[0] else len(xs)-2
    slope = np.clip((ys[i+1]-ys[i])/(xs[i+1]-xs[i]),0,1.5)
    return float(ys[i]+slope*(x-xs[i]))


def predict(parent, test):
    train = archive_features(parent/"development/lowered.zip")
    times, bad = development_targets(parent)
    ratios = ratio_targets(parent)
    assert not bad
    results = defaultdict(dict)
    for mode in ("source","stable"):
        tiers = defaultdict(list)
        for c in cases("development"):
            for n in menu(c):
                key = c.name+"/"+n
                base = key[:-1]+"S" if n.endswith("_J") else key
                group,x = coordinate(train[base],mode)
                y = ratios[key] if n.endswith("_J") else log(times[key][0])
                tiers[group,n,c.tier].append((x,y))
        curves = defaultdict(list)
        for (g,n,_), rows in tiers.items():
            curves[g,n].append(tuple(np.median(rows,axis=0)))
        for c in fresh():
            for n in menu(c):
                key = c.name+"/"+n
                base = key[:-1]+"S" if n.endswith("_J") else key
                g,x = coordinate(test[base],mode)
                y = curve(curves[g,n],x,n.endswith("_J"))
                results[key][mode] = exp(y) * (results[base][mode] if n.endswith("_J") else 1)
    for row in results.values():
        row["combined"] = (row["source"]*row["stable"])**.5
    return dict(results)


def lock(parent):
    import jax
    folder = parent/"paired"
    if (folder/"lock.json").exists():
        raise FileExistsError(folder/"lock.json")
    files, features = {}, {}
    for c in fresh():
        args = arguments(c,parameters(c),SEEDS[0])
        for n in menu(c):
            key = c.name+"/"+n
            module = jax.jit(program(c,n)).lower(*args).compiler_ir()
            features[key] = dict(source=source(c,n),stable=stable_features(module))
            files[key+".mlir"] = str(module)
    files["features.json"] = json.dumps(features,sort_keys=True)
    zip_write(folder/"lowered.zip",files)
    dump(folder/"lock.json", dict(training=sha(parent/"predictions.json"),protocol=sha(folder/"protocol.json"),
                                  ir=sha(folder/"lowered.zip"), predictions=predict(parent,features)))


def measure(folder, idx, *, graphs_for_run=fresh, menu_for_case=menu, seeds=SEEDS,
            rounds=12, order_seed=88093):
    import jax
    from fusionscope.bench import environment
    from fusionscope.hlo import Module
    from fusionscope.phase3.experiment import block
    target = folder/f"runs/p{idx}.zip"
    digest = sha(folder/"lock.json")
    if target.exists():
        with ZipFile(target) as z:
            assert z.testzip() is None and json.loads(z.read("environment.json"))["lock"]==digest
        return
    rng = random.Random(f"{order_seed}:{idx}")
    graphs = graphs_for_run()
    rng.shuffle(graphs)
    timings, checks, builds = [], [], []
    for c in graphs:
        p = parameters(c)
        batches = {s:arguments(c,p,s) for s in seeds}
        jax.block_until_ready(batches)
        args = batches[seeds[0]]
        fns, labels = {}, menu_for_case(c)
        rng.shuffle(labels)
        for n in labels:
            start = perf_counter_ns()
            fns[n] = jax.jit(program(c,n)).lower(*args).compile()
            elapsed = perf_counter_ns()-start
            builds.append(dict(case=c.name,variant=n,component="main",process=idx,build_wall_ns=elapsed,
                               fingerprint=Module(fns[n].as_text()).fingerprint()))
        checks.extend(validation(c,fns,batches))
        timings.extend(block(c,idx,"candidate",fns,args,rng,rounds))
        for n in labels:
            timings.extend(block(c,idx,"AA_"+n,dict.fromkeys("ABCD",fns[n]),args,rng,rounds))
        del fns,batches,args,p
        jax.clear_caches()
        gc.collect()
    meta = dict(process=idx,environment=environment("cpu"),lock=digest,case_order=[c.name for c in graphs],
                rounds=rounds,warmups=6,seeds=seeds)
    zip_write(target,{"environment.json":json.dumps(meta),"timings.csv":csv_text(timings),
                      "correctness.csv":csv_text(checks),"builds.csv":csv_text(builds)})
    print(f"Process {idx}: {len(builds)} configurations, {sum(r['valid'] for r in checks)}/{len(checks)} checks",flush=True)


def derive(parent):
    folder = parent/"paired"
    sealed = read(folder/"lock.json")
    assert sealed["training"]==sha(parent/"predictions.json") and sealed["ir"]==sha(folder/"lowered.zip")
    assert sealed["protocol"]==sha(folder/"protocol.json")
    regenerated = predict(parent,archive_features(folder/"lowered.zip"))
    pred = sealed["predictions"]
    for k,row in regenerated.items():
        for m,value in row.items():
            np.testing.assert_allclose(value,pred[k][m],rtol=1e-10)
    graphs = fresh()
    data, checks,_ = load_runs(folder/"runs",{c.name:menu(c) for c in graphs},sha(folder/"lock.json"),5,12,SEEDS)
    act, pairs = [], []
    for c in graphs:
        p = data[c.name]
        for n in menu(c):
            k = c.name+"/"+n
            act.append(dict(case=c.name,group=c.group,variant=n,runtime_us=median(r["times"][n] for r in p),**pred[k]))
            if n.endswith("_J"):
                s = n[:-1]+"S"
                choices = {m:"J" if x<pred[c.name+"/"+s][m] else "S" for m,x in pred[k].items()}
                pairs.append(dict(case=c.name,group=c.group,contract=n[:-2],**contrast(p,n,s),**choices))
    metrics, policies = [], []
    for group in ("new_shape","new_architecture"):
        q = [r for r in act if r["group"]==group and r["variant"]!="V0"]
        for m in ("source","stable","combined"):
            error = np.exp(np.abs(np.log([r[m]/r["runtime_us"] for r in q])))
            metrics.append(dict(group=group,model=m,observed=len(q),median_q=float(np.median(error)),
                                p90_q=float(np.percentile(error,90))))
    for m in ("source","stable","combined"):
        groups = defaultdict(list)
        count = Counter()
        for r in pairs:
            separate = r[m]=="S"
            groups[r["case"]].append(-log(r["ratio"]) if separate else 0.)
            if separate:
                count[r["outcome"]]+=1
        x = np.array([np.mean(v) for v in groups.values()])
        rng = np.random.default_rng(1871)
        ci = np.exp(np.percentile(x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1),[2.5,97.5]))
        policies.append(dict(model=m,pairs=len(pairs),chooses_separate=sum(count.values()),helped=count["loss"],
                             hurt=count["gain"],unresolved=count["unresolved"],graph_balanced_to_joint=exp(x.mean()),
                             ci_low=ci[0],ci_high=ci[1],advantage=bool(ci[1]<1)))
    return {"measurements":act,"comparisons":pairs,"accuracy":metrics,"policies":policies}, checks


def verify_inputs(folder):
    from jax._src.interpreters.mlir import make_ir_context
    from jax.extend.mlir import ir
    path = folder/"lowered.zip"
    features = archive_features(path)
    with ZipFile(path) as z, make_ir_context():
        for c in fresh():
            for n in menu(c):
                key = c.name+"/"+n
                module = ir.Module.parse(z.read(key+".mlir").decode())
                assert features[key] == dict(source=source(c,n),stable=stable_features(module))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["lock","test","report","verify","worker_lock","worker_time"])
    parser.add_argument("--out",type=Path,default=ROOT/"reproduction/cpu_cost")
    parser.add_argument("--index",type=int,default=0)
    args = parser.parse_args()
    parent = args.out.resolve()
    folder = parent/"paired"
    if args.stage in ("lock","test"):
        import shutil
        folder.mkdir(parents=True,exist_ok=True)
        if not (folder/"protocol.json").exists():
            shutil.copyfile(ROOT/"results/cpu_cost/paired/protocol.json",folder/"protocol.json")
        job = "worker_lock" if args.stage=="lock" else "worker_time"
        for i in range(1 if args.stage=="lock" else 5):
            subprocess.run([sys.executable,"-B","-m","fusionscope.cpu_cost.paired_model",job,"--out",str(parent),
                            "--index",str(i)],env=worker_env("cpu"),check=True)
    elif args.stage.startswith("worker"):
        pin()
        if args.stage=="worker_lock":
            lock(parent)
        else:
            measure(folder,args.index)
    else:
        if args.stage=="verify":
            verify_inputs(folder)
        tables,checks = derive(parent)
        for k,rows in tables.items():
            path = folder/(k+".csv")
            if args.stage=="report":
                table(path,rows)
            else:
                assert records(path)==[{k:str(v) for k,v in r.items()} for r in rows]
        print(f"{checks} checks; {len(tables['comparisons'])} paired comparisons")


if __name__=="__main__":
    main()
