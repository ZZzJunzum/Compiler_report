from collections import defaultdict
from itertools import combinations
from statistics import median

import numpy as np

from .audit_ir import PLAN
from .experiment import PHASES, zip_rows


def cohort(out,phase,raw_dir=None):
    all_cases=defaultdict(lambda:{"p":[],"aa0":[],"aa1":[],"invalid":set(),"builds":[],"checks":[]})
    folder=raw_dir if raw_dir is not None else out/PHASES[phase]/"raw"
    for pid in range(PLAN["processes"][phase]):
        archive=folder/f"p{pid}.zip"
        groups=defaultdict(lambda:defaultdict(list))
        for row in zip_rows(archive,"timings.csv"):
            if int(row["round"])>=0:
                groups[row["case"],row["block"]][row["variant"]].append(int(row["elapsed_ns"])/1e3)
        for (case,block),variants in groups.items():
            vals={k:median(v) for k,v in variants.items()}
            if block=="candidate":
                all_cases[case]["p"].append(vals)
            else:
                key="aa0" if block=="AA_V0" else "aa1"
                all_cases[case][key].extend(abs(a-b) for a,b in combinations(vals.values(),2))
        for row in zip_rows(archive,"correctness.csv"):
            case=all_cases[row["case"]]
            case["checks"].append({"process":pid,**row})
            if row["valid"]!="True":
                case["invalid"].add(row["variant"])
        for row in zip_rows(archive,"builds.csv"):
            all_cases[row["case"]]["builds"].append({"process":pid,**row})
    for c in all_cases.values():
        p=c["p"]
        c["noise"]=max(float(np.quantile(c["aa0"],.95)),float(np.quantile(c["aa1"],.95)))
        c["t"]={m:median(v[m] for v in p) for m in p[0]}
        c["overhead"]=median(v["V1"]-v["V0"] for v in p)
        c["recovery_ok"]=c["overhead"]>c["noise"] and sum(v["V1"]>v["V0"] for v in p)>len(p)/2
    return dict(all_cases)


def metrics(c,name):
    times=c["p"]
    saved=[p["V1"]-p[name] for p in times]
    delta=median(saved)
    return dict(latency_us=c["t"][name],min_process_us=min(p[name] for p in times),
                max_process_us=max(p[name] for p in times),speedup=median(p["V1"]/p[name] for p in times),
                overhead_us=median(p[name]-p["V0"] for p in times),saved_vs_v1_us=delta,
                aa_floor_us=c["noise"],recovery=delta/c["overhead"] if c["recovery_ok"] and name!="V0" else None,
                resolved_gain=delta>c["noise"] and sum(d>0 for d in saved)>len(times)/2,
                resolved_loss=-delta>c["noise"] and sum(d<0 for d in saved)>len(times)/2,valid=name not in c["invalid"])
