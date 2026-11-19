"""Reproduce the paper's cost-model diagnostics from the archived CPU data.

Run from the repository root with python -m paper.check_diagnostics or
python paper/check_diagnostics.py.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from math import log, prod
from pathlib import Path
from statistics import geometric_mean, median

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusionscope.common import ROOT, read
from fusionscope.phase3.analysis import bootstrap
from fusionscope.phase3.data import cohort
from fusionscope.phase3.design import development, heldout
from fusionscope.phase3.models import FAMILIES, GRID, encode, learn, ordered, score


ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "phase2"
COST_INPUTS = ("host_calls", "total_mops", "transfer_mb", "reduce_levels", "loop_steps")


def read_rows(file):
    with (ARCHIVE / file).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def index(rows, *columns):
    return {tuple(row[key] for key in columns): row for row in rows}


def check_original_selector():
    lock_file = ARCHIVE / "model/prediction_lock.json"
    lock = json.loads(lock_file.read_text(encoding="utf-8"))
    digest = hashlib.sha256(lock_file.read_bytes()).hexdigest()
    if digest != (lock_file.parent / "prediction_lock.sha256").read_text().strip():
        raise ValueError("The prediction lock has changed.")
    for file, expected in lock["inputs"].items():
        if hashlib.sha256((ARCHIVE / file).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed input to the frozen prediction: {file}")

    features = index(read_rows("model/features.csv"), "case", "variant")
    predictions = index(read_rows("model/predictions.csv"), "case", "variant")
    models = json.loads((ARCHIVE / "model/model.json").read_text(encoding="utf-8"))
    decisions = index(read_rows("model/heldout_results.csv"), "case", "policy")
    structure = read_rows("analysis/candidate_structure.csv")
    times = index([r for r in read_rows("analysis/timings.csv") if r["phase"] == "confirm"],
                  "case", "variant")
    cases = sorted({case for case, _ in times})

    deltas = []
    for case, variant in predictions:
        if case not in cases or variant == "V2":
            continue
        joint = features[case, "V2"]
        other = features[case, variant]
        deltas.append([float(other[k]) - float(joint[k]) for k in COST_INPUTS])
    print("Synthetic alternatives compared to joint:", len(deltas))
    print("Coordinatewise dominated by joint:", sum(min(d) >= 0 for d in deltas))
    print("Minimum alternative-minus-joint difference by coordinate:",
          dict(zip(COST_INPUTS, map(min, zip(*deltas)))))
    print("Strictly positive coefficients in every synthetic fold:",
          all(min(models[f]["parameters"]["analytic_coefficients"]) > 0 for f in "EMNC"))

    changes = Counter()
    beyond_floor = Counter()
    for case in cases:
        ridge = decisions[case, "ridge"]
        if ridge["chosen"] == "V2":
            changes["same"] += 1
            continue
        diff = float(ridge["chosen_us"]) - float(decisions[case, "fixed_V2"]["chosen_us"])
        outcome = "improved" if diff < 0 else "slowed" if diff > 0 else "tied"
        changes[outcome] += 1
        if abs(diff) > float(ridge["aa_floor_us"]):
            beyond_floor[outcome] += 1
    print("Ridge decisions relative to fixed joint:", dict(changes))
    print("Changed choices beyond the A/A floor (difference of medians):", dict(beyond_floor))

    matched = {(r["case"], r["variant"]) for r in structure if r["case"] in cases
               and r["variant"].startswith("V4_") and r["same_single_executable_as_V1"] == "True"}
    gaps = [abs(float(predictions[c, v]["ridge_ratio"]) / float(predictions[c, "V1"]["ridge_ratio"]) - 1)
            for c, v in matched]
    print("One-tile controls with matching separate-count HLO:", len(matched))
    print("Absolute predicted relative gap: median {:.2%}, above 5% {:d}".format(
        median(gaps), sum(gap > .05 for gap in gaps)))
    for family, column in (("M", "dot_sites"), ("N", "base_reduce_sites"), ("C", "conv_sites")):
        fold = models[family]
        levels = {part: sorted({int(features[c, "V0"][column]) for c in fold[part]})
                  for part in ("train", "test")}
        print(f"Fold {family}, baseline feature {column}:", levels)

    print("Tile time divided by joint time (1, 4, 16, 64, 256 rows)")
    for case in ("E3_branch", "N2_branch"):
        for data, column in ((times, "latency_us"), (predictions, "analytic_us"),
                             (predictions, "ridge_ratio")):
            joint = float(data[case, "V2"][column])
            print(case, column, [round(float(data[case, f"V4_t{b}"][column]) / joint, 3)
                                 for b in (1, 4, 16, 64, 256)])


def fixed_candidate(case, rule):
    """Use a direct schedule when a capped tile covers the entire batch."""
    if rule.startswith("V"):
        return rule
    extent = max(tile for tile in case.tiles if tile <= int(rule[1:]))
    if extent == case.shape[0]:
        return "V2" if rule.startswith("J") else "V1"
    return f"{rule[0]}{extent}"


def source_facts(case):
    width = case.shape[-1]
    weight_elements = {"M": width**2, "C": 9 * width**2, "N": width, "E": width}
    parameters = sum(weight_elements[op] + width for op in case.ops)
    return dict(base_work=0, fanout=0, shared_prefix_ops=0,
                base_bytes=4 * (2 * prod(case.shape) + parameters))


def paired_ratio(data, variant, reference="V2"):
    return median(row[variant] / row[reference] for row in data["processes"])


def best(features, model):
    return min(features, key=lambda name: score(model, features[name]))


def refit_without_hlo(cases, features, timings):
    groups = {}
    for case in cases:
        data = timings[case.name]
        times = data["us"]
        examples = [(*item, times[v], times["V0"]) for v, item in features[case.name].items()
                    if v not in data["invalid"]]
        groups[case.name] = (examples, data["floor"])

    def validation_loss(alpha):
        regrets = []
        for family in FAMILIES:
            train = [groups[c.name] for c in cases if c.ops[0] != family]
            model = learn("log_cross", alpha, train)
            for c in cases:
                if c.ops[0] == family:
                    chosen = best(features[c.name], model)
                    data = timings[c.name]
                    valid = [v for v in features[c.name] if v not in data["invalid"]]
                    regrets.append(log(data["us"][chosen] / min(data["us"][v] for v in valid))
                                   if chosen in valid else 10.)
        return sum(regrets) / len(regrets)

    alpha = min(GRID, key=validation_loss)
    return learn("log_cross", alpha, list(groups.values()))


def check_revised_selector():
    out = ROOT / "results/phase3/cpu"
    dev, fresh = development(), heldout()
    primary = [case for case in fresh if case.group == "primary"]
    old = cohort(out, "development")
    new = cohort(out, "confirmation")
    if any(d["invalid"] for d in (*old.values(), *new.values())):
        raise ValueError("The published audit assumes the verified, fully valid CPU cohorts.")
    static = out / "static"
    lock = read(static / "lock.json")
    choices = {c.name: lock["choices"][c.name]["static"] for c in fresh}
    model = read(static / "models.json")["final"][lock["primary"]]
    calibration = read(static / "calibration.json")
    bases = read(static / "baseline.json")

    rules = ["V2", "V1", "V3"] + [f"{kind}{cap}" for cap in (1, 4, 16, 64, 256)
                                  for kind in ("S", "J")]

    def development_time(rule):
        return geometric_mean(paired_ratio(old[c.name], fixed_candidate(c, rule)) for c in dev)

    rule = min(rules, key=development_time)
    fixed = {c.name: fixed_candidate(c, rule) for c in primary}
    print("Development-selected fixed rule:", rule)
    print("Fixed rule / whole joint:",
          geometric_mean(paired_ratio(new[c.name], fixed[c.name]) for c in primary))

    # The registered bootstrap compares its 'static' choice with its 'V2' denominator.
    # Here these two lanes carry the actual model and the development-selected fixed rule.
    comparison, labels, regrets = {}, {}, {}
    for c in primary:
        data = new[c.name]
        selected, reference = choices[c.name], fixed[c.name]
        comparison[c.name] = dict(invalid=set(), processes=[
            {"model": row[selected], "V2": row[reference]} for row in data["processes"]])
        labels[c.name] = {"static": "model"}
        oracle = min(data["us"][v] for v in ordered(c))
        regrets[c.name] = {p: {"regret": data["us"][v] / oracle - 1}
                           for p, v in (("static", selected), ("V2", reference))}
    paired = bootstrap(primary, comparison, labels, regrets)
    print("Model / development-selected rule, paired ratio and interval:",
          paired["primary_ratio"], paired["interval"])

    features = {}
    centered_scores = {}
    largest_gap = 0.
    unchanged = comparisons = 0
    for c in dev + fresh:
        source = source_facts(c)
        baseline = bases[c.name]["facts"]
        if source["base_bytes"] != baseline["base_bytes"]:
            raise ValueError(f"Input and output size mismatch on {c.name}")
        stripped = {v: encode(c, v, source, calibration) for v in ordered(c)}
        features[c.name] = stripped
        if c.group != "development":
            original = {v: score(model, encode(c, v, baseline, calibration)) for v in ordered(c)}
            if min(original, key=original.get) != choices[c.name]:
                raise ValueError(f"Original model does not reproduce the lock on {c.name}")
            alternative = {v: score(model, item) for v, item in stripped.items()}
            centered_scores[c.name] = {v: value - original["V2"] for v, value in original.items()}
            gaps = [abs(centered_scores[c.name][v] - (alternative[v] - alternative["V2"]))
                    for v in alternative]
            largest_gap = max(largest_gap, max(gaps))
            comparisons += len(gaps)
            unchanged += min(alternative, key=alternative.get) == choices[c.name]
    print("Frozen choices retained without HLO:", unchanged, "/", len(fresh))
    print("Centered scores compared and largest difference:", comparisons, largest_gap)
    refit = refit_without_hlo(dev, features, old)
    chosen = {c.name: best(features[c.name], refit) for c in fresh}
    print("Source-only refit / whole joint:",
          geometric_mean(paired_ratio(new[c.name], chosen[c.name]) for c in primary))
    print("Changed choices after refitting:", [c.name for c in fresh if chosen[c.name] != choices[c.name]])
    pairs = [(c.name, c.name.replace("_chain", "_branch")) for c in primary if c.top == "chain"]
    print("Chain/branch pairs with the same frozen choice:",
          sum(choices[chain] == choices[branch] for chain, branch in pairs), "/", len(pairs))
    print("Largest chain/branch difference in centered scores:",
          max(abs(value - centered_scores[branch][v]) for chain, branch in pairs
              for v, value in centered_scores[chain].items()))
    for name in ("EH1_chain", "EH1_branch"):
        data = new[name]
        print(name, choices[name], "whole and selected medians, paired ratio:",
              data["us"]["V2"], data["us"][choices[name]], paired_ratio(data, choices[name]))
    mixes = []
    for process in range(len(new[primary[0].name]["processes"])):
        rows = {c.name: new[c.name]["processes"][process] for c in primary}
        selected = sum(row[choices[name]] for name, row in rows.items())
        whole_joint = sum(row["V2"] for row in rows.values())
        mixes.append(selected / whole_joint)
    print("One invocation per primary graph, median ratio of summed times:", median(mixes))


def main():
    check_original_selector()
    check_revised_selector()


if __name__ == "__main__":
    main()
