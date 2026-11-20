"""Replay balanced timing blocks, paired estimators and charged policy costs."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations, product
import json
from pathlib import Path
from statistics import fmean, geometric_mean, median
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from fusionscope.axis_probe.analysis import require, rows
from fusionscope.common import dump, read, sha, table
from .collect import PLAN, PUBLISHED, SEAL, lock
from .workloads import cases, KEYS, RESOURCES, SEEDS

TIMING = PLAN["timing"]
ALIASES = TIMING["aliases"]
PAIRS = (("J","S"),("global","global_S"),("J","off"),("global","global_off"),
         ("P","J"),("P","S"),("P","off"),("off","S"))
IDENTITY = ("fingerprint","named_schedule","temp_bytes")


def process_means(raw, names):
    rounds = range(-TIMING["warmups"],TIMING["rounds"])
    lanes = set(product(KEYS,ALIASES))
    blocks, samples = defaultdict(list), defaultdict(list)
    for r in raw:
        g,rnd,seed = r["graph"],int(r["round"]),int(r["seed"])
        lane = r["key"],r["alias"]
        blocks[g,rnd,seed].append((int(r["block"]),int(r["pos"]),lane))
        ns = int(r["ns"])
        require(ns > 0,r)
        if rnd >= 0:
            samples[g,seed,*lane].append(ns)
    require(set(blocks)==set(product(names,rounds,SEEDS)),"missing graph/round/input")
    for g,rnd in product(names,rounds):
        ordering = []
        for seed in SEEDS:
            chunk = blocks[g,rnd,seed]
            require(sorted(x[1] for x in chunk)==list(range(len(lanes))) and
                    {x[2] for x in chunk}==lanes and len({x[0] for x in chunk})==1,(g,rnd,seed))
            ordering.append(chunk[0][0])
        require(sorted(ordering)==list(range(len(SEEDS))),(g,rnd,"input order"))
    return {(g,k,a):fmean(median(samples[g,s,k,a]) for s in SEEDS)/1000
            for g,k,a in product(names,KEYS,ALIASES)}


def paired(processes, graph, left, right):
    def contrasts(key, a, b):
        return [p["means"][graph,key,a]-p["means"][graph,key,b] for p in processes]
    shams = [contrasts(k,a,b) for k in (left,right) for a,b in combinations(ALIASES,2)]
    effect = [p["means"][graph,left,"A"]-p["means"][graph,right,"A"] for p in processes]
    delta = median(effect)
    floor = float(np.percentile([abs(median(c)) for c in shams],95))
    verdict = "slower" if delta > floor else "faster" if delta < -floor else "unresolved"
    return dict(delta_us=delta,floor_us=floor,verdict=verdict),effect,shams


def interactions(processes, graph, row_pair, global_pair):
    _,r,rs = paired(processes,graph,*row_pair)
    _,g,gs = paired(processes,graph,*global_pair)
    delta = median(x-y for x,y in zip(r,g,strict=True))
    shams = [abs(median(x-sign*y for x,y in zip(a,b,strict=True)))
             for a,b,sign in product(rs,gs,(1,-1))]
    floor = float(np.percentile(shams,95))
    return dict(delta_us=delta,floor_us=floor,clears=abs(delta)>floor)


def load_resource(folder, phase, resource, ir):
    names = [c.name for c in cases(phase)]
    ref_lock = lock(phase)
    dest = folder/resource
    manifest = read(dest/"manifest.json")
    files = {f"p{i}.zip" for i in range(TIMING["processes"])}
    require(manifest["resource"]==resource and manifest["phase"]==phase and manifest["lock"]==ref_lock
            and set(manifest["archives"])==files=={f.name for f in dest.glob("p*.zip")},dest)
    processes, pids, intervals = [],set(),[]
    for i in range(TIMING["processes"]):
        p = dest/f"p{i}.zip"
        require(sha(p)==manifest["archives"][p.name],p)
        with ZipFile(p) as z:
            meta = json.loads(z.read("meta.json"))
            env = meta["environment"]
            expected_affinity = env["original_affinity"][:{"one":1,"four":4,"default":None}[resource]]
            require(meta["process"]==i and meta["pid"] not in pids and meta["lock"]==ref_lock
                    and sorted(meta["order"])==sorted(names) and meta["phase"]==phase
                    and env["affinity"]==expected_affinity and env["resource"]==resource,meta)
            if ir:
                require(env["compiler_sha256"]==ir["env"]["compiler_sha256"],"compiler changed")
            pids.add(meta["pid"])
            start,end = (datetime.fromisoformat(meta[k]) for k in ("start","end"))
            require(end>start and (not intervals or start>=intervals[-1][1]),"overlapping workers")
            intervals.append((start,end))
            raw = rows(z,"timings.csv")
            m = process_means(raw,names)
            build_rows,checks = rows(z,"builds.csv"),rows(z,"checks.csv")
            b = {(r["graph"],r["key"]):r for r in build_rows}
            require(set(b)==set(product(names,KEYS)) and len(b)==len(build_rows),"missing builds")
            for graph in names:
                require(sorted(int(b[graph,k]["order"]) for k in KEYS)==list(range(len(KEYS))),graph)
            for (graph,key),rec in b.items():
                require(int(rec["build_ns"])>0,rec)
                if ir:
                    idx = ir[graph,key]
                    require(all(str(idx[k])==rec[k] for k in IDENTITY),(resource,graph,key,"IR mismatch"))
            require(len(checks)==len(names)*len(KEYS)*len(SEEDS) and
                    {(r["graph"],r["key"],int(r["seed"])) for r in checks}==set(product(names,KEYS,SEEDS)),
                    "missing numerical checks")
            ok = {(r["graph"],r["key"],int(r["seed"])):r["valid"]=="True" for r in checks}
            for r in checks:
                require((r["valid"]=="True")== (r["main_ok"]==r["counts_typed"]==r["direct_match"]=="True"
                        and int(r["max_count_error"])==0),r)
            dec = {}
            if phase=="holdout":
                from .policies import POLICIES, choice
                dec_rows = rows(z,"decisions.csv")
                dec = {(r["graph"],r["policy"]):r for r in dec_rows}
                require(len(dec)==len(dec_rows) and set(dec)==set(product(names,POLICIES)),"missing decisions")
                for c,policy in product(cases(phase),POLICIES):
                    decision = dec[c.name,policy]
                    require(decision["chosen"]==choice(policy,c.rows,c.width,len(expected_affinity))
                            and int(decision["selection_ns"])>=0,decision)
            processes.append(dict(means=m,build=b,checks=ok,dec=dec,meta=meta,raw_rows=len(raw)))
    return processes


def read_ir(path, phase):
    if not path.exists():
        return {}
    with ZipFile(path) as z:
        require(z.testzip() is None,path)
        index = {"env":json.loads(z.read("environment.json"))}
        for c,key in product(cases(phase),KEYS):
            stem = f"{c.name}/{key}/"
            item = json.loads(z.read(stem+"index.json"))
            import hashlib
            for n,digest in item["objects"].items():
                require(hashlib.sha256(z.read(stem+n)).hexdigest()==digest,(c.name,key,n))
            index[c.name,key] = item
        return index


def compile_tables(index,phase,resource):
    summary = []
    for c,k in product(cases(phase),KEYS):
        item = index[c.name,k]
        partitions = {role:[x["partitions"] for x in item["kernels"] if x["role"]==role and x["partitions"]]
                      for role in ("observer","main","shared")}
        other = {"J":"off","global":"global_off","P":"J"}.get(k)
        lhs = Counter(item["objects"].values())
        rhs = Counter(index[c.name,other]["objects"].values()) if other else Counter()
        summary.append(dict(resource=resource,graph=c.name,key=k,temp_bytes=item["temp_bytes"],
          observer_partitions=json.dumps(partitions["observer"]),main_partitions=json.dumps(partitions["main"]),
          shared_partitions=json.dumps(partitions["shared"]),objects=sum(lhs.values()),compare_to=other or "",
          identical_objects=sum((lhs & rhs).values()) if other else "",
          only_this=sum((lhs-rhs).values()) if other else "",only_other=sum((rhs-lhs).values()) if other else ""))
    return summary


def derive(base,phase,dest,require_ir=True):
    folder = base/phase
    ref = read_ir(folder/"default_ir.zip",phase)
    if require_ir:
        require(ref,"default compiler archive missing")
    comparisons,interaction,latencies,compilers,profiles,sessions,validity = [],[],[],[],[],[],[]
    data = {}
    for resource in RESOURCES:
        own = ref if resource=="default" else read_ir(folder/(resource+"_ir.zip"),phase)
        ir = own or ref
        procs = load_resource(folder,phase,resource,ir)
        data[resource] = procs
        if own:
            compilers.extend(compile_tables(own,phase,resource))
        for p in procs:
            meta = p["meta"]
            sessions.append(dict(resource=resource,pid=meta["pid"],start=meta["start"],end=meta["end"],
                                 affinity=json.dumps(meta["environment"]["affinity"]),rows=p["raw_rows"],
                                 checks=len(p["checks"]),failed=sum(not ok for ok in p["checks"].values())))
        for c in cases(phase):
            common = dict(resource=resource,cpus=len(procs[0]["meta"]["environment"]["affinity"]),
                          graph=c.name,kind=c.kind,rows=c.rows,width=c.width)
            for left,right in PAIRS:
                eff,_,_ = paired(procs,c.name,left,right)
                comparisons.append(dict(**common,left=left,right=right,**eff))
            for name,r,g in (("row_minus_global_joint_effect",("J","S"),("global","global_S")),
                ("row_minus_global_disabling_effect",("off","J"),("global_off","global"))):
                interaction.append(dict(**common,contrast=name,**interactions(procs,c.name,r,g)))
            for k in KEYS:
                valid = sum(p["checks"][c.name,k,s] for p in procs for s in SEEDS)
                validity.append(dict(**common,key=k,passed=valid,total=len(procs)*len(SEEDS)))
                latencies.append(dict(**common,key=k,
                    latency_us=median(p["means"][c.name,k,"A"] for p in procs),
                    build_ms=median(int(p["build"][c.name,k]["build_ns"]) for p in procs)/1e6))
        prof = folder/(resource+"_profile.json")
        if prof.exists():
            doc = read(prof)
            require(doc["environment"]["affinity"]==procs[0]["meta"]["environment"]["affinity"],prof)
            expected_graphs = ("t_E1_chain","two_sites_large","two_sites_small","large_work")
            require({(r["graph"],r["key"]) for r in doc["rows"]}==set(product(expected_graphs,
                    ("J","off","global","global_off","P"))) and len(doc["rows"])==20,prof)
            for row in doc["rows"]:
                b = procs[0]["build"][row["graph"],row["key"]]
                require(all(str(row[k])==b[k] for k in IDENTITY) and row["wall_s"]>=.5,row)
                profiles.append(dict(resource=resource,**{k:row[k] for k in
                    ("graph","key","calls","wall_s","average_busy_cpus","threads_over_31ms","threads_before","threads_after")}))
    for filename,entries in (("pairs",comparisons),("interactions",interaction),("latencies",latencies),
                  ("kernels",compilers),("profiles",profiles),("sessions",sessions),("validity",validity)):
        if entries:
            table(dest/(filename+".csv"),entries)
    if phase=="holdout":
        evaluate(data,dest)
    return data


def evaluate(data,dest):
    from .policies import POLICIES
    seal = read(SEAL)
    costs,outcomes,summary = [],[],[]
    for resource,c in product(RESOURCES,cases("holdout")):
        procs = data[resource]
        native = paired(procs,c.name,"J","S")[0]["verdict"]
        assignment = paired(procs,c.name,"J","off")[0]["verdict"]
        for pol in POLICIES:
            selected = [p["dec"][c.name,pol] for p in procs]
            key = selected[0]["chosen"]
            require(all(d["chosen"]==key for d in selected),pol)
            vs = paired(procs,c.name,key,"J")[0]
            valid = all(p["checks"][c.name,key,s] for p in procs for s in SEEDS)
            tag = dict(resource=resource,graph=c.name,kind=c.kind,policy=pol,chosen=key)
            outcomes.append(dict(**tag,valid=valid,native_joint_vs_separate=native,native_on_vs_off=assignment,**vs,
                selection_us=median(int(d["selection_ns"]) for d in selected)/1000))
            for n in seal["calls"]:
                for p,d in zip(procs,selected,strict=True):
                    select_us = int(d["selection_ns"])/1000
                    build_us = int(p["build"][c.name,key]["build_ns"])/1000
                    call_us = p["means"][c.name,key,"A"]
                    costs.append(dict(**tag,n=n,process=p["meta"]["process"],selection_us=select_us,
                        build_us=build_us,call_us=call_us,total_us=select_us+build_us+n*call_us,valid=valid))
    by = {(r["resource"],r["graph"],r["policy"],r["n"],r["process"]):r["total_us"] for r in costs}
    comparisons = {}
    grouped = defaultdict(list)
    for n,pol,ref in product(seal["calls"],POLICIES,POLICIES):
        ratios = []
        for resource,c in product(RESOURCES,cases("holdout")):
            ratio = median(by[resource,c.name,pol,n,i]/by[resource,c.name,ref,n,i]
                           for i in range(TIMING["processes"]))
            ratios.append(ratio)
            grouped[pol,ref,n,c.kind].append(ratio)
        comparisons[pol,ref,n] = geometric_mean(ratios)
        summary.append(dict(n=n,policy=pol,baseline=ref,ratio=geometric_mean(ratios),
                            below_one=sum(r<1 for r in ratios),above_one=sum(r>1 for r in ratios),cells=len(ratios)))
    rule = "shape_guard"
    ref = min(seal["comparators"], key=lambda s: comparisons[s,"native_J",seal["primary_N"]])
    target = [r for r in outcomes if r["policy"]==rule]
    ratio = comparisons[rule,ref,seal["primary_N"]]
    verdict = dict(primary_N=seal["primary_N"],strongest_simple=ref,ratio=ratio,
        per_structure={k:geometric_mean(v) for (p,b,n,k),v in grouped.items()
                       if (p,b,n)==(rule,ref,seal["primary_N"])},
        exact_cells=sum(r["valid"] for r in target),cells=len(target),
        unmodified=sum(r["chosen"]=="J" for r in target),
        faster=sum(r["verdict"]=="faster" for r in target),slower=sum(r["verdict"]=="slower" for r in target),
        unresolved=sum(r["verdict"]=="unresolved" for r in target),
        native_regressions=sum(r["native_joint_vs_separate"]=="slower" for r in target),
        native_regressions_helped=sum(r["native_joint_vs_separate"]=="slower" and r["verdict"]=="faster" for r in target),
        native_wins=sum(r["native_joint_vs_separate"]=="faster" for r in target),
        native_wins_harmed=sum(r["native_joint_vs_separate"]=="faster" and r["verdict"]=="slower" for r in target),
        native_assignment_wins=sum(r["native_on_vs_off"]=="faster" for r in target),
        native_assignment_wins_harmed=sum(r["native_on_vs_off"]=="faster" and r["verdict"]=="slower" for r in target),
        default_harms=sum(r["resource"]=="default" and r["verdict"]=="slower" for r in target))
    verdict["method_gate_passed"] = (ratio <= seal["max_ratio"] and verdict["exact_cells"]==len(target)
                                     and verdict["default_harms"]==0)
    table(dest/"policy_effects.csv",outcomes)
    table(dest/"charges.csv",costs)
    table(dest/"cost_ratios.csv",summary)
    dump(dest/"verdict.json",verdict)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("report","verify"))
    ap.add_argument("--out",type=Path,default=PUBLISHED)
    ap.add_argument("--phase",choices=("discovery","holdout"),required=True)
    a = ap.parse_args()
    dest = a.out/a.phase/"analysis"
    if a.action=="report":
        derive(a.out,a.phase,dest)
    else:
        with TemporaryDirectory() as root:
            temp = Path(root)
            derive(a.out,a.phase,temp)
            require({f.name:sha(f) for f in temp.iterdir()}=={f.name:sha(f) for f in dest.iterdir()},
                    (a.phase,"derived files differ"))
    print(a.action,a.phase,"OK")


if __name__=="__main__":
    main()
