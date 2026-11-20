"""Read fresh paired timings and their same-executable controls from process archives."""
from collections import defaultdict
from itertools import combinations
import json
from statistics import median
from zipfile import ZipFile

import numpy as np


def load_runs(folder, lanes, lock_hash, processes, rounds, seeds):
    from fusionscope.phase2.experiment import zip_rows

    archives = sorted(folder.glob("p*.zip"))
    if [p.name for p in archives] != [f"p{i}.zip" for i in range(processes)]:
        raise ValueError(f"Incomplete process set in {folder}")
    result = defaultdict(list)
    checks = 0
    environment = None
    for i, path in enumerate(archives):
        with ZipFile(path) as z:
            env = json.loads(z.read("environment.json"))
            if z.testzip() or env["process"] != i or env["lock"] != lock_hash:
                raise ValueError(f"Wrong process or prediction lock: {path}")
        if environment is None:
            environment = env["environment"]
        if (env["environment"] != environment or tuple(env["seeds"]) != tuple(seeds)
                or env["rounds"] != rounds or set(env["case_order"]) != set(lanes)):
            raise ValueError(f"Environment or measurement design changed: {path}")
        samples = defaultdict(dict)
        for r in zip_rows(path, "timings.csv"):
            key = (r["case"], r["block"], r["variant"])
            index = int(r["round"])
            if index in samples[key] or int(r["process"]) != i or int(r["elapsed_ns"]) <= 0:
                raise ValueError(f"Bad sample in {path}: {key}, round {index}")
            samples[key][index] = int(r["elapsed_ns"]) / 1000
        builds = defaultdict(dict)
        for r in zip_rows(path, "builds.csv"):
            parts = builds[r["case"], r["variant"]]
            if r["component"] in parts or int(r["process"]) != i:
                raise ValueError(f"Duplicate or mislabeled build: {r}")
            parts[r["component"]] = r
        validations = zip_rows(path, "correctness.csv")
        good = {(r["case"], r["variant"], int(r["seed"])): r["valid"] == "True"
                for r in validations}
        required_checks = {(c, v, s) for c, vs in lanes.items() for v in {*vs, "V0"} for s in seeds}
        if len(validations) != len(good) or good.keys() != required_checks or not all(good.values()):
            raise ValueError(f"Missing or failed output checks in {path}")
        checks += len(good)
        for c, variants in lanes.items():
            def process_median(block, label):
                timings = samples.pop((c, block, label))
                if timings.keys() != set(range(-env["warmups"], rounds)):
                    raise ValueError(f"Incomplete rounds for {c}/{block}/{label}")
                return median(timings[j] for j in range(rounds))

            row = dict(times={}, aa={}, builds={}, fingerprints={})
            for v in variants:
                row["times"][v] = process_median("candidate", v)
                aliases = [process_median("AA_"+v, label) for label in "ABCD"]
                row["aa"][v] = [abs(a-b) for a,b in combinations(aliases, 2)]
                parts = builds[c, v]
                # Staged schedules have two components but one inclusive build clock.
                clocks = {int(r["build_wall_ns"]) for r in parts.values()}
                if len(clocks) != 1:
                    raise ValueError(f"Missing or inconsistent build time: {c}/{v}")
                row["builds"][v] = clocks.pop() / 1000
                row["fingerprints"][v] = tuple(sorted((k,r["fingerprint"]) for k,r in parts.items()))
            result[c].append(row)
        if samples:
            raise ValueError(f"Unexpected timing lanes in {path}")
    return dict(result), checks, environment


def contrast(rows, alternative, fixed):
    """Describe a paired effect against an empirical A/A repeatability floor.

    The 95th percentile is not a calibrated significance threshold: alias
    contrasts share medians and are correlated. Sign counts add no separate
    evidence in odd panels; even panels retain the historical majority screen.
    """
    if not rows or any(not np.isfinite(r["times"][v]) or r["times"][v] <= 0
                       for r in rows for v in (fixed, alternative)):
        raise ValueError("Paired timings must be finite, positive, and nonempty")
    gains = [r["times"][fixed] - r["times"][alternative] for r in rows]
    ratios = [r["times"][alternative] / r["times"][fixed] for r in rows]
    controls = [[d for r in rows for d in r["aa"][v]] for v in (fixed, alternative)]
    if (any(not values for values in controls)
            or any(not np.isfinite(value) or value < 0 for values in controls for value in values)):
        raise ValueError("A/A differences must be finite, nonnegative, and nonempty")
    floor = max(float(np.percentile(values, 95)) for values in controls)
    gain = median(gains)
    # A nonzero median in an odd panel already implies a strict sign majority.
    # Preserve the historical majority screen for even panels.
    resolved = abs(gain) > floor
    if len(rows) % 2 == 0:
        resolved = resolved and sum(g * gain > 0 for g in gains) > len(rows)/2
    return dict(gain_us=gain, ratio=median(ratios), floor_us=floor, positive=sum(g>0 for g in gains),
                outcome="gain" if resolved and gain>0 else "loss" if resolved else "unresolved",
                same_hlo=all(r["fingerprints"][alternative] == r["fingerprints"][fixed] for r in rows))
