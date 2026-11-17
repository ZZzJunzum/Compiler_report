"""Reproduce the manuscript's retrospective diagnostics from the frozen CSVs."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median


ARCHIVE = Path(__file__).resolve().parents[1] / "results" / "phase2"
COST_INPUTS = ("host_calls", "total_mops", "transfer_mb", "reduce_levels", "loop_steps")


def read_rows(file):
    with (ARCHIVE / file).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def index(rows, *columns):
    return {tuple(row[key] for key in columns): row for row in rows}


def main():
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

    print("Figure 1: tile time divided by joint time (1, 4, 16, 64, 256 rows)")
    for case in ("E3_branch", "N2_branch"):
        for data, column in ((times, "latency_us"), (predictions, "analytic_us"),
                             (predictions, "ridge_ratio")):
            joint = float(data[case, "V2"][column])
            print(case, column, [round(float(data[case, f"V4_t{b}"][column]) / joint, 3)
                                 for b in (1, 4, 16, 64, 256)])


if __name__ == "__main__":
    main()
