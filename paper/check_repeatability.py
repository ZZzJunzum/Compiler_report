"""Check the repeatability screen against boundaries and archived raw timings.

This imports the repository's actual contrast implementation without JAX.
It reaggregates timing samples and A/A aliases, checks ZIP CRCs, and compares
the resulting statistics with the increment and choice CSVs. It does not run
new performance measurements, validate output correctness or HLO, or verify
the original prediction lock and archived cryptographic hashes.

Usage: python paper/check_repeatability.py [--data PATH]
The default data directory is results/cpu_cost/increment. Results go to stdout.
"""
import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from io import StringIO
from itertools import combinations, product
import json
from pathlib import Path
from statistics import median
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fusionscope.paired_repeat import contrast


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def synthetic_rows(differences, floor):
    return [dict(times={"fixed": 100.0, "alternative": 100.0 - difference},
                 aa={"fixed": [floor] * 6, "alternative": [floor] * 6},
                 fingerprints={"fixed": ("same",), "alternative": ("same",)})
            for difference in differences]


def check_boundaries():
    """Compare the historical rule across odd/even, zero, tie, and floor cases."""
    checked = 0
    expected_fields = {"gain_us", "ratio", "floor_us", "positive", "outcome", "same_hlo"}
    for count in range(1, 6):
        for values in product((-3.0, -1.0, 0.0, 1.0, 3.0), repeat=count):
            center = median(values)
            majority = sum(value * center > 0 for value in values) > count / 2
            for floor in (0.0, 0.5, 1.0, 4.0):
                resolved = abs(center) > floor and majority
                expected = ("gain" if resolved and center > 0 else
                            "loss" if resolved else "unresolved")
                result = contrast(synthetic_rows(values, floor), "alternative", "fixed")
                label = f"values={values}, floor={floor}"
                require(set(result) == expected_fields, f"Result fields changed: {label}")
                require(result["outcome"] == expected, f"Historical rule changed: {label}")
                require(result["gain_us"] == center, f"Median changed: {label}")
                require(result["positive"] == sum(value > 0 for value in values),
                        f"Descriptive sign count changed: {label}")
                require(result["same_hlo"] is True, f"Fingerprint comparison changed: {label}")
                checked += 1

    invalid = [[]]
    for lane in ("fixed", "alternative"):
        for bad in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            rows = synthetic_rows([1, 2, 3], 0.0)
            rows[0]["times"][lane] = bad
            invalid.append(rows)
        for bad in (-1.0, float("nan"), float("inf"), float("-inf")):
            rows = synthetic_rows([1, 2, 3], 0.0)
            rows[0]["aa"][lane] = [bad] * 6
            invalid.append(rows)
        rows = synthetic_rows([1, 2, 3], 0.0)
        for row in rows:
            row["aa"][lane] = []
        invalid.append(rows)
    for index, rows in enumerate(invalid):
        try:
            contrast(rows, "alternative", "fixed")
        except ValueError:
            continue
        raise AssertionError(f"Invalid input accepted: case {index}")
    return dict(finite_boundary_checks=checked, invalid_input_checks=len(invalid))


def read_processes(data):
    """Rebuild only the timing/A/A inputs; dummy fingerprints assert no HLO facts."""
    protocol = json.loads((data / "protocol.json").read_text())
    measurement = protocol["measurement"]
    process_count = measurement["processes"]
    wanted = {f"p{index}.zip" for index in range(process_count)}
    require({path.name for path in (data / "runs").glob("p*.zip")} == wanted,
            "The process archive set differs from the protocol")
    runs = defaultdict(list)
    timing_rows = 0
    for index in range(process_count):
        path = data / f"runs/p{index}.zip"
        with ZipFile(path) as archive:
            require(archive.testzip() is None, f"ZIP CRC failure: {path}")
            metadata = json.loads(archive.read("environment.json"))
            require(metadata["process"] == index, f"Wrong process identity: {path}")
            require(metadata["rounds"] == measurement["rounds"]
                    and metadata["warmups"] == measurement["warmups"],
                    f"Timing design differs from the protocol: {path}")
            raw = list(csv.DictReader(StringIO(archive.read("timings.csv").decode())))
        timing_rows += len(raw)
        samples = defaultdict(dict)
        for row in raw:
            key = row["case"], row["block"], row["variant"]
            iteration = int(row["round"])
            elapsed = int(row["elapsed_ns"])
            require(iteration not in samples[key] and int(row["process"]) == index and elapsed > 0,
                    f"Duplicate or invalid timing: {path}, {key}, round {iteration}")
            samples[key][iteration] = elapsed / 1000
        expected_rounds = set(range(-metadata["warmups"], metadata["rounds"]))
        medians = {}
        for key, observations in samples.items():
            require(observations.keys() == expected_rounds, f"Incomplete rounds: {path}, {key}")
            medians[key] = median(observations[iteration] for iteration in range(metadata["rounds"]))
        for case in sorted({key[0] for key in medians}):
            variants = sorted(key[2] for key in medians if key[0] == case and key[1] == "candidate")
            row = dict(times={}, aa={}, fingerprints={})
            for variant in variants:
                row["times"][variant] = medians[case, "candidate", variant]
                aliases = [medians[case, "AA_" + variant, label] for label in "ABCD"]
                row["aa"][variant] = [abs(left - right) for left, right in combinations(aliases, 2)]
                row["fingerprints"][variant] = ("not_checked",)
            runs[case].append(row)
    return runs, process_count, timing_rows


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def check_archives(data):
    runs, process_count, timing_rows = read_processes(data)
    archived = read_csv(data / "measurements.csv")
    changed = 0
    cohorts = {}
    for row in archived:
        processes = runs[row["case"]]
        variant = row["variant"]
        result = contrast(processes, variant, "V0")
        label = f"{row['case']}/{variant}"
        require(-result["gain_us"] == float(row["delta_us"]), f"Delta differs: {label}")
        require(result["floor_us"] == float(row["floor_us"]), f"A/A floor differs: {label}")
        resolved = result["outcome"] != "unresolved"
        changed += resolved != (row["resolved"] == "True")
        for index, process in enumerate(processes):
            delta = process["times"][variant] - process["times"]["V0"]
            require(delta == float(row[f"p{index}_delta_us"]), f"Process delta differs: {label}/p{index}")
        counts = cohorts.setdefault(row["cohort"], dict(programs=0, resolved=0, negative=0,
                                                       resolved_negative=0))
        negative = float(row["delta_us"]) < 0
        counts["programs"] += 1
        counts["resolved"] += resolved
        counts["negative"] += negative
        counts["resolved_negative"] += resolved and negative
    require(changed == 0, f"Increment resolution labels changed: {changed}")

    choices = read_csv(data / "choices.csv")
    choice_changes = 0
    for row in choices:
        result = contrast(runs[row["case"]], row["request"] + "_S", row["request"] + "_J")
        label = f"{row['case']}/{row['request']}"
        for key, column in (("gain_us", "gain_us"), ("floor_us", "floor_us"),
                            ("ratio", "separate_to_joint")):
            require(result[key] == float(row[column]), f"Choice {key} differs: {label}")
        choice_changes += result["outcome"] != row["outcome"]
    require(choice_changes == 0, f"Choice resolution labels changed: {choice_changes}")
    return dict(raw_archives=process_count, raw_timing_rows=timing_rows,
                increment_rows=len(archived), changed_increment_labels=changed,
                choice_rows=len(choices), changed_choice_labels=choice_changes, cohorts=cohorts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/cpu_cost/increment")
    args = parser.parse_args()
    data = args.data.resolve()
    report = dict(data=str(data),
                  contrast_sha256=sha256((ROOT / "fusionscope/paired_repeat.py").read_bytes()).hexdigest(),
                  **check_boundaries(), **check_archives(data),
                  scope=("Reaggregated archived timings and A/A medians; checked ZIP CRCs and timing "
                         "round coverage. No new performance measurements, output-correctness or "
                         "HLO validation, prediction-lock replay, or archive-hash verification."))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
