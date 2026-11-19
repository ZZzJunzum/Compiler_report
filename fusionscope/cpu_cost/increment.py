"""Forecast the time added by an observation using only the ordinary program's HLO."""
import argparse
from collections import Counter, defaultdict
import gc
import json
from math import prod
from pathlib import Path
import re
from statistics import median
import subprocess
import sys
from time import perf_counter_ns
from zipfile import ZipFile

import numpy as np

from fusionscope.common import ROOT, dump, read, records, sha, table
from fusionscope.hlo import Module
from fusionscope.phase2.experiment import zip_rows, zip_write
from run import pin, worker_env
from .experiment import PUBLISHED
from .features import source
from .model import ALPHAS, development_targets, fit, score
from .paired_model import fresh, measure
from .workloads import Case, CONTRACTS, arguments, axes, cases, parameters, program, site_shapes, variants

PANEL = PUBLISHED / "increment"
MODES = ("source", "ir", "combined")


def new_graphs(spec):
    graphs = []
    for family in "EMNCGRB":
        known = family in "EMNC"
        shapes = spec["shapes"][family if family in "CGRB" else "matrix"]
        for tier, shape in enumerate(shapes):
            for top in (("chain", "branch") if known else ("native",)):
                graphs.append(Case(f"i_{family}{tier}_{top}", family, tuple(shape), top, tier,
                                   "new_shape" if known else "new_architecture"))
    prior = cases("development") + cases("test") + fresh()
    keys = {(c.family, c.shape) for c in prior}
    if any((c.family, c.shape) in keys for c in graphs):
        raise ValueError("Confirmatory shape also occurs in the earlier cost panels")
    return graphs


def tensor_shape(value):
    match = re.match(r"\w+\[([\d,]*)\]", value.op.shape)
    return tuple(int(d) for d in match[1].split(",") if d) if match else None


def ir_counts(module, domains):
    counts = Counter(fusion_groups=len({v.group for v in module.main if v.fused}))
    for v in module.main:
        shape = tensor_shape(v)
        if shape is None:
            continue
        n = prod(shape)
        kind = v.op.kind
        counts["ops_" + kind] += 1
        counts["elements_" + kind] += n
        consumers = set(module.children[v])
        if len(consumers) > 1:
            counts["fanout_elements"] += n * (len(consumers) - 1)
        if v in module.exports:
            counts["export_elements"] += n
        if shape in domains and kind not in ("parameter", "constant", "broadcast"):
            counts["domain_fused" if v.fused else "domain_outside_fusion"] += n
            if v in module.exports:
                counts["domain_export"] += n
            if len(consumers) > 1:
                counts["domain_shared"] += n
        if kind in ("dot", "convolution", "custom-call", "reduce"):
            counts[kind + "_operand_elements"] += sum(prod(s) for p in v.parents
                                                       if (s := tensor_shape(p)) is not None)
    return dict(counts)


def features(case, module):
    shapes = site_shapes(case)
    src = source(case, "V0")
    result = {}
    for request, targets in CONTRACTS.items():
        domains = {shapes[site] for site, _, _ in targets}
        query = Counter()
        for site, kind, reduction in targets:
            shape = shapes[site]
            elements = prod(shape)
            query["requested_elements"] += elements
            query["largest_domain"] = max(query["largest_domain"], elements)
            query["outputs"] += (elements if kind == "full" else 1) if reduction == "none" else (
                elements // prod(shape[i] for i in axes(reduction, shape)))
        result[request] = dict(query=dict(query), source=src, ir=ir_counts(module, domains))
    return result


def development_labels():
    # First validate every archived round, build and correctness row using the original protocol.
    _, rejected = development_targets(PUBLISHED)
    if rejected:
        raise ValueError(f"Development output failures: {rejected}")
    deltas = defaultdict(list)
    baseline_fingerprints = {}
    for i in range(3):
        path = PUBLISHED / f"development/p{i}.zip"
        rows = defaultdict(list)
        for r in zip_rows(path, "timings.csv"):
            if int(r["round"]) >= 0:
                rows[r["case"], r["variant"]].append(int(r["elapsed_ns"]))
        for (name, variant), samples in rows.items():
            if variant != "V0":
                deltas[name, variant].append((median(samples) - median(rows[name, "V0"])) / 1000)
        for r in zip_rows(path, "builds.csv"):
            if r["variant"] == "V0" and baseline_fingerprints.setdefault(r["case"], r["fingerprint"]) != r["fingerprint"]:
                raise ValueError(f"Development V0 fingerprint changed: {r['case']}")
    return {key: median(values) for key, values in deltas.items()}, baseline_fingerprints


def covariates(row, mode):
    groups = ("query", "source", "ir") if mode == "combined" else ("query", mode)
    return {g + "." + key: n for g in groups for key, n in row[g].items()}


def design(rows, columns):
    raw = np.asarray([[r.get(col, 0) for col in columns] for r in rows], dtype=float)
    return np.hstack([raw, np.log1p(raw)])


def estimate(panel_features, heldout, labels):
    training = cases("development")
    fits, predictions = {}, {}
    for mode in MODES:
        designs = {}
        errors = defaultdict(list)
        for variant in sorted({v for _, v in labels}):
            request = variant.split("_")[0]
            train = [c for c in training if (c.name, variant) in labels]
            rows = [covariates(panel_features[c.name][request], mode) for c in train]
            columns = sorted({k for row in rows for k in row})
            x = design(rows, columns)
            y = np.array([labels[c.name, variant] for c in train])
            tiers = np.array([c.tier for c in train])
            for tier in np.unique(tiers):
                keep = tiers != tier
                for alpha in ALPHAS:
                    pred = score(x[~keep], fit(x[keep], y[keep], alpha))
                    errors[alpha].extend(abs(y[~keep] - pred))
            designs[variant] = (request, columns, x, y)
        cv = {a: float(np.mean(e)) for a, e in errors.items()}
        alpha = min(cv, key=lambda a: (cv[a], -a))
        models = {}
        for variant, (request, columns, x, y) in designs.items():
            model = fit(x, y, alpha)
            models[variant] = dict(columns=columns, fit=model)
            for c in heldout:
                if variant in variants(c):
                    row = covariates(panel_features[c.name][request], mode)
                    pred = score(design([row], columns), model)[0]
                    predictions.setdefault(c.name + "/" + variant, {"zero": 0.0})[mode] = float(pred)
        fits[mode] = dict(penalty=alpha, cv_mae_us={str(a): e for a, e in cv.items()}, models=models)
    return fits, predictions


def baseline_text(compiled):
    lines = compiled.as_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(("%", "ENTRY %")))
    text = "\n".join([lines[0], "", *lines[start:]]) + "\n"
    return re.sub(r'source_file="[^"]*"', 'source_file=""', text)


def inputs():
    paths = ["protocol.json", *[f"development/p{i}.zip" for i in range(3)]]
    return {p: sha(PUBLISHED / p) for p in paths}


def lock(folder, spec):
    import jax
    from fusionscope.bench import environment
    if (folder / "lock.json").exists() or list((folder / "runs").glob("p*.zip")):
        raise FileExistsError("Use a fresh output directory for a new prediction lock")
    heldout = new_graphs(spec)
    labels, dev_fp = development_labels()
    files, panel_features, builds = {}, {}, {}
    for c in cases("development") + heldout:
        args = arguments(c, parameters(c), spec["measurement"]["seeds"][0])
        jax.block_until_ready(args)
        start = perf_counter_ns()
        compiled = jax.jit(program(c, "V0")).lower(*args).compile()
        built = perf_counter_ns()
        text = baseline_text(compiled)
        module = Module(text)
        panel_features[c.name] = features(c, module)
        elapsed = perf_counter_ns()
        fp = module.fingerprint()
        if c.name in dev_fp and fp != dev_fp[c.name]:
            raise ValueError(f"Baseline no longer matches the development executable: {c.name}")
        builds[c.name] = dict(fingerprint=fp, build_us=(built-start)/1000, extract_us=(elapsed-built)/1000)
        files[c.name + ".hlo"] = text
        del compiled, args
        jax.clear_caches()
        gc.collect()
    fits, predictions = estimate(panel_features, heldout, labels)
    files["features.json"] = json.dumps(panel_features, sort_keys=True)
    zip_write(folder / "baseline_hlo.zip", files)
    dump(folder / "models.json", fits)
    dump(folder / "lock.json", dict(protocol=sha(folder/"protocol.json"), training=inputs(),
                                   baselines=builds, hlo=sha(folder/"baseline_hlo.zip"),
                                   models=sha(folder/"models.json"), environment=environment("cpu"),
                                   predictions=predictions))
    print(f"Sealed {len(predictions)} increments for {len(heldout)} graphs.", flush=True)


def verify_lock(folder, spec, replay):
    sealed = read(folder / "lock.json")
    if (sealed["protocol"] != sha(folder/"protocol.json") or sealed["training"] != inputs()
            or sealed["hlo"] != sha(folder/"baseline_hlo.zip") or sealed["models"] != sha(folder/"models.json")):
        raise ValueError("Prediction lock inputs changed")
    if replay:
        heldout = new_graphs(spec)
        with ZipFile(folder/"baseline_hlo.zip") as archive:
            panels = json.loads(archive.read("features.json"))
            for c in cases("development") + heldout:
                module = Module(archive.read(c.name+".hlo").decode())
                if features(c, module) != panels[c.name] or module.fingerprint() != sealed["baselines"][c.name]["fingerprint"]:
                    raise ValueError(f"Baseline IR audit failed: {c.name}")
        labels, _ = development_labels()
        fits, predictions = estimate(panels, heldout, labels)
        if fits != read(folder/"models.json") or predictions != sealed["predictions"]:
            raise ValueError("Archived inputs did not reproduce the frozen predictions")
    return sealed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["lock", "test", "report", "verify", "worker_lock", "worker_time", "worker_verify"])
    parser.add_argument("--out", type=Path, default=ROOT/"reproduction/cpu_increment")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    folder = args.out.resolve()
    spec = read(PANEL/"protocol.json")
    if args.stage == "lock" and not (folder/"protocol.json").exists():
        dump(folder/"protocol.json", spec)
    if read(folder/"protocol.json") != spec:
        raise ValueError("Protocol mismatch")
    m = spec["measurement"]
    if args.stage in ("lock", "test", "verify"):
        stage = {"lock": "worker_lock", "test": "worker_time", "verify": "worker_verify"}[args.stage]
        for i in range(m["processes"] if args.stage == "test" else 1):
            subprocess.run([sys.executable, "-B", "-m", "fusionscope.cpu_cost.increment", stage, "--out",
                            str(folder), "--index", str(i)], env=worker_env("cpu"), cwd=ROOT, check=True)
        return
    pin()
    if args.stage == "worker_lock":
        lock(folder, spec)
        return
    sealed = verify_lock(folder, spec, args.stage == "worker_verify")
    if args.stage == "worker_time":
        measure(folder, args.index, graphs_for_run=lambda: new_graphs(spec), menu_for_case=variants,
                seeds=m["seeds"], rounds=m["rounds"], order_seed=m["order_seed"])
        return
    from .increment_results import analyze
    tables, summary = analyze(folder, spec, new_graphs(spec), sealed)
    for key, rows in tables.items():
        path = folder/(key + ".csv")
        if args.stage == "report":
            table(path, rows)
        elif records(path) != [{k: str(v) for k, v in r.items()} for r in rows]:
            raise ValueError(f"Result table differs: {path}")
    if args.stage == "report":
        dump(folder/"summary.json", summary)
    elif summary != read(folder/"summary.json"):
        raise ValueError("Summary differs")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
