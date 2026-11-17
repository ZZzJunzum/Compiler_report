"""Process-paired timing summaries in microseconds and same-executable noise controls."""
from collections import defaultdict
from itertools import combinations
from statistics import median
from zipfile import ZipFile
import json

import numpy as np

from fusionscope.phase2.experiment import zip_rows
from .design import REPEATS


def cohort(out, stage, policy=""):
    folders = sorted((out/stage/policy).glob("p*.zip"))
    if len(folders) != REPEATS[stage]:
        raise ValueError(f"Incomplete {stage}/{policy} cohort: {len(folders)}")
    result = defaultdict(lambda: dict(processes=[], aa=[], interactions=[], invalid=set(), builds=[], blocks=[]))
    for pid, path in enumerate(folders):
        with ZipFile(path) as z:
            meta = json.loads(z.read("environment.json"))
            if meta["process"] != pid or meta["stage"] != stage or meta["policy"] != policy:
                raise ValueError(f"Unexpected process identity in {path}")
        samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in zip_rows(path, "timings.csv"):
            if int(r["round"]) >= 0:
                samples[r["case"]][r["block"]][r["variant"]].append(int(r["elapsed_ns"])/1000)
        for c, blocks in samples.items():
            dev = result[c]
            lanes = {k: {v: median(obs) for v,obs in b.items()} for k,b in blocks.items()}
            dev["processes"].append(lanes.pop("candidate"))
            aa, interactions = defaultdict(list), defaultdict(list)
            for name, batch in lanes.items():
                a,b,c1,d = [batch[k] for k in "ABCD"]
                aa[name] = [abs(x-y) for x,y in combinations(batch.values(), 2)]
                interactions[name] = [abs(a+b-c1-d),abs(a+c1-b-d),abs(a+d-b-c1)]
            dev["aa"].append(dict(aa))
            dev["interactions"].append(dict(interactions))
        for r in zip_rows(path, "builds.csv"):
            result[r["case"]]["builds"].append(dict(r, process=pid))
        for r in zip_rows(path, "blocks.csv"):
            result[r["case"]]["blocks"].append(int(r["block_wall_ns"])/1000)
        if stage != "discovery":
            for r in zip_rows(path, "correctness.csv"):
                if r["valid"] != "True":
                    result[r["case"]]["invalid"].add(r["variant"])
    for data in result.values():
        ps = data["processes"]
        data["us"] = {v: median(p[v] for p in ps) for v in ps[0]}
        for key, dest in (("aa","floor"), ("interactions","interaction_floor")):
            data[dest] = max(float(np.percentile([v for p in data[key] for v in p[name]],95))
                             for name in data[key][0])
    return dict(result)


def direction(data, a, b):
    """Positive if a is faster than b."""
    d = [p[b]-p[a] for p in data["processes"]]
    med = median(d)
    sign = 1 if med > 0 else -1
    resolved = abs(med)>data["floor"] and sum(sign*x>0 for x in d)>len(d)/2
    return med, bool(resolved)
