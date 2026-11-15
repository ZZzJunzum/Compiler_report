from collections import defaultdict
from itertools import combinations
from statistics import median

import numpy as np

from .common import DOUBLES, read, records


def summarize(root, phase, case):
    processes, checks = [], []
    dirs = sorted((root / phase / "raw").glob(f"p*/{case}"))
    for folder in dirs:
        if not (folder / "done.json").exists():
            raise ValueError(f"incomplete {folder}")
        samples = defaultdict(list)
        for row in records(folder / "times.csv"):
            samples[row["mode"]].append(int(row["elapsed_ns"])/1000)
        processes.append({k:median(v) for k,v in samples.items()})
        checks.extend(read(folder / "checks.json"))
    if not processes:
        raise ValueError(f"No data for {phase}/{case}")
    keys = [k for k in processes[0] if not k.startswith("A")]
    base = "only_y" if "only_y" in keys else "base"
    times = {k:median(p[k] for p in processes) for k in keys}
    delta = {k:median(p[k]-p[base] for p in processes) for k in keys}
    ratios = {k:median(p[k]/p[base] for p in processes) for k in keys}
    valid = {k:all(row["valid"] for row in checks if row["mode"]==k) for k in keys}
    labels = ["A0","A1","A2","A3"]
    aa_pair, aa_i = [], []
    for p in processes:
        for a,b in combinations(labels,2):
            aa_pair.append(abs(p[a]-p[b]))
            c,d = [k for k in labels if k not in (a,b)]
            aa_i.append(abs(p[a]+p[b]-p[c]-p[d]))
    return dict(processes=processes, times=times, delta=delta, ratios=ratios, valid=valid,
                checks=checks, aa_us=float(np.quantile(aa_pair,.95)), aa_i_us=float(np.quantile(aa_i,.95)))


def interactions(s):
    rows = []
    for mode in DOUBLES:
        u,v = mode.split("_")
        vals = [p[mode]-p[f"u_{u}"]-p[f"v_{v}"]+p["base"] for p in s["processes"]]
        center = median(vals)
        rows.append({"mode":mode,"interaction_us":center,"min_us":min(vals),"max_us":max(vals),
                     "aa_pseudo_p95_us":s["aa_i_us"],"outside_floor":abs(center)>s["aa_i_us"],
                     "same_sign_every_process":all(v>0 for v in vals) or all(v<0 for v in vals),
                     "valid":all(s["valid"][k] for k in (mode,f"u_{u}",f"v_{v}","base"))})
    return rows


def ranking(s):
    order = sorted((k for k in DOUBLES if s["valid"][k]), key=lambda k:s["times"][k])
    if len(order)<2:
        return order, True
    winner, second = order[:2]
    gap = s["times"][second] - s["times"][winner]
    votes = sum(min(order, key=p.get)==winner for p in s["processes"])
    return order, gap <= s["aa_us"] or votes <= len(s["processes"])/2
