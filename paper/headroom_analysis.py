"""Recalculate CPU opportunity, repeatability and net savings against joint cap-16."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import geometric_mean, median
from tempfile import TemporaryDirectory

from fusionscope.common import ROOT, dump, read, records, sha, table
from fusionscope.headroom import ARCHIVE, PROCESSES, ROUNDS, SEEDS, lock_for_run, nominations
from fusionscope.paired_repeat import contrast, load_runs
from fusionscope.phase3.analysis import build_cost, crossover
from fusionscope.phase3.data import cohort
from fusionscope.phase3.design import development, heldout
from fusionscope.phase3.models import ordered
from paper.check_diagnostics import fixed_candidate


HORIZONS = (1, 10, 100, 1000, 10000)


def opportunity_budget(case, d, fixed, base, calibration):
    legal = [v for v in ordered(case) if v not in d["invalid"]]
    builds = {v: build_cost(d, v) for v in legal}
    rows = []
    for h in HORIZONS:
        # The runtime winner need not minimize compile cost plus H invocations.
        net = {v: h*(d["us"][fixed]-d["us"][v]) - (builds[v]-builds[fixed]) for v in legal}
        v = max(legal, key=net.get)
        for mode, info in (("free_identity", 0.), ("baseline_reused", base),
                           ("baseline_cold", base+calibration)):
            rows.append(dict(case=case.name, horizon=h, mode=mode, chosen=v,
                time_saving_us=d["us"][fixed]-d["us"][v], extra_compile_us=builds[v]-builds[fixed],
                information_us=info, max_net_us=net[v]-info))
    return rows


def historical_costs():
    held = cohort(ARCHIVE, "confirmation")
    setup = {(r["case"],r["policy"]): r for r in records(ARCHIVE/"analysis/setup.csv")}
    result = []
    for c in heldout():
        d = held[c.name]
        f = fixed_candidate(c, "J16")
        for policy in ("static", "timed_rank_alt"):
            row = setup[c.name,policy]
            if row["valid"] != "True" or f in d["invalid"]:
                raise ValueError(f"Invalid comparator or candidate on {c.name}")
            v = row["chosen"]
            # Match the historical deployment endpoint: difference of marginal medians.
            saving = d["us"][f] - d["us"][v]
            for mode in ("reused","cold"):
                delta = float(row[mode+"_setup_us"]) - build_cost(d,f)
                kind, threshold = crossover(delta, saving)
                result.append(dict(case=c.name, group=c.group, policy=policy, mode=mode,
                    fixed=f, chosen=v, per_call_saving_us=saving, extra_setup_us=delta,
                    nonworse=kind, horizon=threshold,
                    **{f"net_{h}_us": h*saving-delta for h in HORIZONS}))
    return result


def analyze(data_root, output):
    lock, digest = lock_for_run(data_root)
    import json
    if lock != json.loads(json.dumps(nominations())):
        raise ValueError("Development nominations do not reproduce the lock.")
    lanes = {n: item["test"] for n,item in lock["cases"].items()}
    repeats, checks, env = load_runs(data_root/"repeat", lanes, digest, PROCESSES, ROUNDS, SEEDS)
    old = cohort(ARCHIVE, "development")
    extraction = read(ARCHIVE/"static/baseline.json")
    cal = read(ARCHIVE/"static/calibration.json")
    if cal["environment"] != env:
        raise ValueError("The archive and independent repeats use different CPU environments.")
    rows, conv, budget = [], [], []
    for c in development():
        d, ps = old[c.name], repeats[c.name]
        item = lock["cases"][c.name]
        f,v = item["fixed"],item["nominee"]
        measured = contrast(ps,v,f)
        base = extraction[c.name]["extraction_ns"]/1000
        delta_build = median(p["builds"][v]-p["builds"][f] for p in ps)
        info = base + delta_build
        saving = measured["gain_us"]
        kind, horizon = crossover(info, saving)
        rows.append(dict(case=c.name, family=c.ops[0], fixed=f, nominee=v,
            development_headroom_us=item["headroom_us"], development_paired_us=item["paired_us"],
            development_ratio=d["us"][v]/d["us"][f], development_floor_us=item["floor_us"],
            development_resolved=item["resolved_gain"], **measured, extra_compile_us=delta_build,
            baseline_extraction_us=base, incremental_setup_us=info, nonworse=kind, horizon=horizon,
            **{f"repeat_net_{h}_us":h*saving-info for h in HORIZONS}))
        budget.extend(opportunity_budget(c, d, f, base, cal["calibration_ns"]/1000))
        if c.ops=="CC" and c.shape[0] in (4,8):
            conv.append(dict(case=c.name, fixed=f, exception="J1", **contrast(ps,"J1",f)))
    historic = historical_costs()
    table(output/"headroom.csv", rows)
    table(output/"conv_repeat.csv", conv)
    table(output/"oracle_budget.csv", budget)
    table(output/"historical_deployment.csv", historic)
    resolved = [r for r in rows if r["outcome"]=="gain"]
    totals = {}
    for fam in "EMNC":
        fam_rows = [r for r in rows if r["family"]==fam]
        totals[fam] = dict(outcomes=dict(Counter(r["outcome"] for r in fam_rows)),
            signed_total_us=sum(r["gain_us"] for r in fam_rows),
            positive_total_us=sum(max(0,r["gain_us"]) for r in fam_rows))
    four = {r["case"] for r in conv}
    summary = dict(lock_sha256=digest, repeat_archives={p.name:sha(p) for p in sorted((data_root/"repeat").glob("p*.zip"))},
        checks=checks, candidate_count=sum(c["legal"] for c in lock["cases"].values()),
        median_headroom_us=median(r["development_headroom_us"] for r in rows),
        development_ratio=geometric_mean(r["development_ratio"] for r in rows),
        development_resolved=sum(r["development_resolved"] for r in rows),
        repeat_median_gain_us=median(r["gain_us"] for r in rows),
        repeat_ratio=geometric_mean(r["ratio"] for r in rows),
        repeat_outcomes=dict(Counter(r["outcome"] for r in rows)),
        old_gains_repeated=sum(r["development_resolved"] for r in resolved),
        same_choice=sum(r["fixed"]==r["nominee"] for r in rows), family=totals,
        four_conv_share_of_positive_development=sum(r["development_headroom_us"] for r in rows if r["case"] in four)/
            sum(r["development_headroom_us"] for r in rows),
        four_conv_share_of_positive_repeat=sum(max(0,r["gain_us"]) for r in rows if r["case"] in four)/
            sum(max(0,r["gain_us"]) for r in rows),
        max_net_positive={mode:{str(h):sum(r["max_net_us"]>0 for r in budget if r["mode"]==mode and r["horizon"]==h)
            for h in HORIZONS} for mode in ("free_identity","baseline_reused","baseline_cold")},
        historical_net_positive={f"{group}/{policy}/{mode}":{str(h):sum(r[f"net_{h}_us"]>0 for r in historic
            if (r["group"],r["policy"],r["mode"])==(group,policy,mode)) for h in HORIZONS}
            for group in ("primary","transfer") for mode in ("cold","reused") for policy in ("static","timed_rank_alt")},
        exception_advances=len(conv)==4 and all(r["outcome"]=="gain" and r["ratio"]<=.95 for r in conv))
    dump(output/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"results/headroom")
    parser.add_argument("--write", action="store_true", help="Publish derived tables for this data directory.")
    args=parser.parse_args()
    published=args.data/"analysis"
    if args.write:
        summary=analyze(args.data, published)
    else:
        with TemporaryDirectory() as tmp:
            out=Path(tmp)
            summary=analyze(args.data,out)
            for p in out.iterdir():
                if p.read_bytes()!=(published/p.name).read_bytes():
                    raise ValueError(f"Published evidence differs from the archives: {p.name}")
    print(f"Verified {summary['checks']} output checks; repeat outcomes: {summary['repeat_outcomes']}; "
          f"convolution exception advances: {summary['exception_advances']}")


if __name__=="__main__":
    main()
