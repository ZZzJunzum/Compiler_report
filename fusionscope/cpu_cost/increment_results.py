"""Score frozen increment forecasts and the separate/joint choices they imply."""
from collections import defaultdict
from statistics import median

import numpy as np

from fusionscope.common import sha
from fusionscope.paired_repeat import contrast, load_runs
from .workloads import variants

METHODS = ("zero", "source", "ir", "combined")


def draws(graphs):
    # One index matrix for all paired comparisons. Keep both readouts of a shape together.
    rng = np.random.default_rng(47291)
    result = []
    for family in sorted({c.family for c in graphs}):
        clusters = [[i for i, c in enumerate(graphs) if c.family == family and c.tier == t]
                    for t in sorted({c.tier for c in graphs if c.family == family})]
        choices = rng.integers(len(clusters), size=(10000, len(clusters)))
        result.append(np.array(clusters)[choices].reshape(10000, -1))
    return np.hstack(result)


def interval(x, indices):
    lo, hi = np.percentile(np.asarray(x)[indices].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def analyze(folder, spec, graphs, sealed):
    m = spec["measurement"]
    lanes = {c.name: variants(c) for c in graphs}
    runs, checks, env = load_runs(folder/"runs", lanes, sha(folder/"lock.json"),
                                  m["processes"], m["rounds"], m["seeds"])
    if env != sealed["environment"]:
        raise ValueError("Baseline and confirmation environments differ")
    expected = {c.name + "/" + v for c in graphs for v in lanes[c.name] if v != "V0"}
    if sealed["predictions"].keys() != expected:
        raise ValueError("Incorrect prediction coverage")
    observations, pairs, accuracy, policies = [], [], [], []
    by_graph = defaultdict(list)
    choose = defaultdict(list)
    for c in graphs:
        processes = runs[c.name]
        if any(p["fingerprints"]["V0"] != (("main", sealed["baselines"][c.name]["fingerprint"]),)
               for p in processes):
            raise ValueError(f"Confirmation baseline changed: {c.name}")
        for v in lanes[c.name][1:]:
            pred = sealed["predictions"][c.name + "/" + v]
            deltas = [p["times"][v] - p["times"]["V0"] for p in processes]
            actual = median(deltas)
            diff = contrast(processes, v, "V0")
            row = dict(case=c.name, cohort=c.group, family=c.family, shape="x".join(map(str,c.shape)),
                       variant=v, delta_us=actual, floor_us=diff["floor_us"],
                       resolved=diff["outcome"] != "unresolved")
            row.update({f"p{i}_delta_us": d for i,d in enumerate(deltas)})
            row.update({k + "_us": pred[k] for k in METHODS})
            by_graph[c.name].append(row)
            observations.append(row)
            if not v.endswith("_J"):
                continue
            sep = v[:-1] + "S"
            sep_pred = sealed["predictions"][c.name + "/" + sep]
            comp = contrast(processes, sep, v)
            row = dict(case=c.name, cohort=c.group, request=v[:-2], separate_to_joint=comp["ratio"],
                       gain_us=comp["gain_us"], floor_us=comp["floor_us"], outcome=comp["outcome"])
            row.update({k: "S" if sep_pred[k] < pred[k] else "J" for k in METHODS})
            choose[c.name].append(row)
            pairs.append(row)
    passed = {}
    for cohort in ("new_shape", "new_architecture"):
        group = [c for c in graphs if c.group == cohort]
        boot = draws(group)
        signed = {k: [np.array([r[k + "_us"] - r["delta_us"] for r in by_graph[c.name]]) for c in group]
                  for k in METHODS}
        mae = {k: np.array([np.mean(abs(e)) for e in errors]) for k, errors in signed.items()}
        for k in METHODS:
            errors = np.concatenate(signed[k])
            delta = mae[k] - mae["source"]
            lo, hi = interval(delta, boot)
            row = dict(cohort=cohort, model=k, graphs=len(group), programs=len(errors),
                       graph_mae_us=float(mae[k].mean()), bias_us=float(np.mean([np.mean(e) for e in signed[k]])),
                       median_ae_us=float(np.median(abs(errors))), p90_ae_us=float(np.percentile(abs(errors),90)),
                       mae_minus_source_us=float(delta.mean()), difference_lo_us=lo, difference_hi_us=hi,
                       mae_minus_zero_us=float(np.mean(mae[k] - mae["zero"])))
            accuracy.append(row)
            if k == "combined":
                passed[cohort] = bool(mae[k].mean() <= 0.9*mae["source"].mean() and hi < 0
                                      and mae[k].mean() < mae["zero"].mean())
            eligible = [r for c in group for r in choose[c.name]]
            changed = [r for r in eligible if r[k] == "S"]
            # Every graph in these cohorts has at least one eligible pair (including bottleneck/shared).
            log_ratios = [np.mean([np.log(r["separate_to_joint"]) if r[k] == "S" else 0.
                                   for r in choose[c.name]]) for c in group]
            lower, upper = interval(log_ratios, boot)
            policies.append(dict(cohort=cohort, model=k, eligible=len(eligible), separate=len(changed),
                                 resolved_gains=sum(r["outcome"] == "gain" for r in changed),
                                 resolved_losses=sum(r["outcome"] == "loss" for r in changed),
                                 unresolved=sum(r["outcome"] == "unresolved" for r in changed),
                                 time_to_joint=float(np.exp(np.mean(log_ratios))),
                                 ci_low=float(np.exp(lower)), ci_high=float(np.exp(upper))))
    summary = dict(graphs=len(graphs), observed_programs=len(observations), eligible_choices=len(pairs),
                   output_checks=checks, measured_processes=m["processes"], ir_criterion=passed,
                   negative_increments=sum(r["delta_us"] < 0 for r in observations),
                   resolved_increments=sum(r["resolved"] for r in observations),
                   archived_runs={f"p{i}": sha(folder/f"runs/p{i}.zip") for i in range(m["processes"])})
    return dict(measurements=observations, accuracy=accuracy, choices=pairs, policies=policies), summary
