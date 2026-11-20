"""Replay the matched interaction estimator and verify complete input blocks."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
from itertools import combinations, product
import json
from pathlib import Path
from statistics import fmean, median
from zipfile import ZipFile

import numpy as np
from fusionscope.common import read, sha, dump, table
from .design import (ALIASES, AXIS_GRAPHS, DESIGN, MINI, PRIMARY, PUBLISHED, ROUNDS, SEEDS,
                     VARIANTS, cells, design_lock, graphs, lanes, settings)


def rows(z, name):
    return list(csv.DictReader(z.read(name).decode().splitlines()))


def require(condition, context):
    if not condition:
        raise ValueError(str(context))


def process_medians(raw, panel):
    blocks = defaultdict(list)
    data = defaultdict(list)
    for r in raw:
        g, seed, rnd = r["graph"], int(r["seed"]), int(r["round"])
        ns = int(r["elapsed_ns"])
        key = (r["variant"], r["setting"], r["alias"])
        require(ns > 0, r)
        blocks[g, rnd, seed].append((int(r["position"]), int(r["block"]), key))
        if rnd >= 0:
            data[g, seed, *key].append(ns/1000)
    required_lanes = set(lanes(panel))
    require(set(blocks) == set(product(graphs(panel), ROUNDS, SEEDS)), "missing or extra input block")
    for g, rnd in product(graphs(panel), ROUNDS):
        order = []
        for seed in SEEDS:
            block = blocks[g, rnd, seed]
            require(len(block) == len(required_lanes) and {k for _, _, k in block} == required_lanes
                    and sorted(p for p, _, _ in block) == list(range(len(block))), (g, rnd, seed))
            positions = {b for _, b, _ in block}
            require(len(positions) == 1, "inconsistent seed block")
            order += list(positions)
        require(sorted(order) == list(range(len(SEEDS))), "invalid input order")
    return {key: median(xs) for key, xs in data.items()}


def effects(panel, graph, processes):
    c0, c1, r0, r1 = cells(panel)
    real_by_seed, real_by_process, means = [], [], []
    for p, m in enumerate(processes):
        for seed in SEEDS:
            a, b, c, d = [m[graph, seed, *cell, "A"] for cell in (c0, c1, r0, r1)]
            real_by_seed.append(dict(process=p, seed=seed, global_diff_us=b-a, row_diff_us=d-c,
                                     interaction_us=(d-c)-(b-a)))
        recent = real_by_seed[-len(SEEDS):]
        real_by_process.append({k: fmean(row[k] for row in recent)
                                for k in ("global_diff_us", "row_diff_us", "interaction_us")})
        # Seed averaging is linear, so each sham can use these means before the process median.
        means.append({lane: fmean(m[graph, s, *lane] for s in SEEDS) for lane in lanes(panel)})

    def sham(candidates):
        return [[m[(*cell, a)] - m[(*cell, b)] for m in means]
                for cell in candidates for a, b in combinations(ALIASES, 2)]

    left, right = sham((c0, c1)), sham((r0, r1))
    null = [median(y-sign*x for x, y in zip(l, r, strict=True))
            for l, r, sign in product(left, right, (1, -1))]
    summary = {}
    for key, controls in (("global_diff", list(map(median, left))), ("row_diff", list(map(median, right))),
                          ("interaction", null)):
        value = median(row[key+"_us"] for row in real_by_process)
        floor = float(np.percentile(np.abs(controls), 95, method="linear"))
        summary.update({key+"_us": value, key+"_floor_us": floor, key+"_clears": abs(value) > floor})
    latencies = [dict(variant=v, setting=opt, median_us=median(m[v, opt, "A"] for m in means))
                 for opt in settings(panel) for v in VARIANTS]
    return summary, real_by_seed, real_by_process, latencies


def verify_session(out, manifest, ir):
    folder = out/"sessions"/manifest["name"]
    nproc = DESIGN["processes_per_session"]
    panel, width = manifest["panel"], manifest["cpus"]
    require(panel in ("axis", "task") and width in (1, 4), manifest)
    require(manifest["lock"] == design_lock(), "protocol hash mismatch")
    require(set(manifest["archives"]) == {f"p{i}.zip" for i in range(nproc)}
            == {p.name for p in folder.glob("p*.zip")}, folder)
    checks, fixed, medians = [], [], []
    provenance, pids, intervals = [], [], []
    nrows, nbuilds = 0, 0
    for p in range(nproc):
        file = folder/f"p{p}.zip"
        require(sha(file) == manifest["archives"][file.name], file)
        with ZipFile(file) as z:
            require(z.testzip() is None, file)
            meta = json.loads(z.read("environment.json"))
            env = meta["environment"]
            require(meta["lock"] == design_lock() and meta["input_seeds"] == SEEDS
                    and meta["weight_seed"] == 3701 and meta["process"] == p
                    and meta["session"] == manifest["name"] and meta["panel"] == panel
                    and len(env["affinity"]) == width and set(meta["graph_order"]) == set(graphs(panel))
                    and env["backend"] == "cpu" and not env["flags"]["XLA_FLAGS"]
                    and env["flags"]["JAX_ENABLE_COMPILATION_CACHE"].lower() == "false", file)
            start, end = [datetime.fromisoformat(meta[k]) for k in ("start", "end")]
            require(meta["local_date"] == start.date().isoformat() == end.date().isoformat()
                    == manifest["date"] and meta["timezone"] == manifest["timezone"] == "America/Chicago"
                    and end > start, file)
            intervals.append((start, end))
            for k in ("jax", "jaxlib", "compiler_sha256", "affinity"):
                require(env[k] == ir[width]["environment"][k], (file, k))
            provenance.append((env, meta["sources"]))
            pids.append(meta["pid"])
            raw = rows(z, "timings.csv")
            nrows += len(raw)
            medians.append(process_medians(raw, panel))
            builds = rows(z, "builds.csv")
            nbuilds += len(builds)
            required = set(product(graphs(panel), VARIANTS, settings(panel)))
            require(len(builds) == len(required) and
                    {(r["graph"], r["variant"], r["setting"]) for r in builds} == required, file)
            for r in builds:
                ref = ir[width][r["graph"], r["variant"], r["setting"]]
                require(all(str(ref[k]) == r[k] for k in ("fingerprint", "named_schedule", "temp_bytes"))
                        and int(r["build_ns"]) > 0, (file, r))
            validation = rows(z, "correctness.csv")
            required_inputs = {(g, v, opt, s) for g, v, opt in required
                               for s in (list(map(str, SEEDS)) +
                                         (["signed_zero_and_normal", "producer_cancellation"]
                                          if g in MINI or g.startswith(("t_E", "t_M")) else []))}
            require(len(validation) == len(required_inputs) and
                    {(r["graph"], r["variant"], r["setting"], r["input"]) for r in validation}
                    == required_inputs, file)
            for r in validation:
                ok = (r["main_ok"] == "True" and r["counts_typed"] == "True" and
                      int(r["diagnostic_error"]) == int(r["separate_error"]) == 0)
                require(ok == (r["valid"] == "True"), r)
                checks.append(dict(process=p, **r))
            local = json.loads(z.read("fixed_sites.json"))
            require(len(local) == 12 and {(r["input"], r["variant"]) for r in local} == set(product(
                ("normal_and_signed_zero", "zero_nextafter", "nonfinite"), VARIANTS[1:])), file)
            for r in local:
                require(r["exact"] == (r["actual"] == r["ieee_reference"]), r)
                fixed.append(dict(process=p, **r))
    require(all(v == provenance[0] for v in provenance) and len(set(pids)) == nproc, folder)
    require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), folder)
    return medians, checks, fixed, nrows, nbuilds


def derive(out):
    from .ir import inspect
    ir, structures, kernels = inspect(out)
    summaries, seeds, processes, latencies, checks, boundary = [], [], [], [], [], []
    sessions = []
    for folder in sorted((out/"sessions").iterdir()):
        if not folder.is_dir():
            continue
        manifest = read(folder/"session.json")
        require(manifest["name"] == folder.name, folder)
        meta = dict(session=folder.name, local_date=manifest["date"],
                    panel=manifest["panel"], cpus=manifest["cpus"])
        ms, validation, fixed, ns, nb = verify_session(out, manifest, ir)
        checks.extend(dict(**meta, **r) for r in validation)
        boundary.extend(dict(**meta, **r) for r in fixed)
        for g in graphs(meta["panel"]):
            common = dict(**meta, graph=g)
            summary, by_seed, by_process, times = effects(meta["panel"], g, ms)
            summaries.append(dict(**common, **summary))
            seeds.extend(dict(**common, **r) for r in by_seed)
            processes.extend(dict(**common, process=p, **r) for p, r in enumerate(by_process))
            latencies.extend(dict(**common, **r) for r in times)
        sessions.append(dict(**meta, raw_calls=ns, matched_builds=nb, whole_graph_checks=len(validation),
                             whole_graph_failures=sum(r["valid"] != "True" for r in validation),
                             fixed_site_checks=len(fixed), fixed_site_reference_failures=sum(not r["exact"] for r in fixed)))
    axis_dates = sorted({s["local_date"] for s in sessions if s["panel"] == "axis" and s["cpus"] == 1})
    status = dict(primary=PRIMARY, discovery_dates=axis_dates, distinct_dates=len(axis_dates),
                  date_requirement_met=len(axis_dates) >= DESIGN["discovery_distinct_dates_minimum"],
                  sessions=sessions)
    return ({"contrasts": summaries, "seed_effects": seeds, "process_effects": processes,
             "latencies": latencies, "structure": structures, "llvm_kernels": kernels,
             "whole_graph_failures": [r for r in checks if r["valid"] != "True"]}, status, boundary)


def main():
    from fusionscope.phase2.experiment import csv_text
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("report", "verify"))
    ap.add_argument("--out", type=Path, default=PUBLISHED)
    args = ap.parse_args()
    tables, status, fixed = derive(args.out)
    destination = args.out/"analysis"
    for name, data in tables.items():
        path = destination/(name+".csv")
        if not data and name == "whole_graph_failures":
            require(not path.exists(), f"Stale failure table {path}")
            continue
        if args.action == "report":
            table(path, data)
        else:
            require(path.read_text(encoding="utf8") == csv_text(data), path)
    for name, value in (("status", status), ("fixed_sites", fixed)):
        path = destination/(name+".json")
        if args.action == "report":
            dump(path, value)
        else:
            require(read(path) == value, path)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
