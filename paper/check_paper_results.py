"""Replay the frozen selector diagnostics and the subsequent whole-call subtraction check."""
from statistics import median

from fusionscope.common import ROOT, read, records
from paper.check_diagnostics import main as check_selectors


def check_observation_increments():
    """Contrast frozen total-time forecasts with the measured, process-paired CPU increments."""
    root = ROOT / "results" / "cpu_cost"
    predictions = read(root / "predictions.json")["predictions"]
    measurements = records(root / "analysis" / "measurements.csv")
    for group in ("new_shape", "new_architecture"):
        observed = [r for r in measurements if r["group"] == group and r["variant"] != "V0"]
        print(f"Median absolute increment error in microseconds, {group} ({len(observed)} programs)")
        for model in ("constant", "source", "stable", "combined"):
            errors = []
            for row in observed:
                case, variant = row["case"], row["variant"]
                predicted_observed = predictions[f"{case}/{variant}"][model]["runtime_us"]
                predicted_main = predictions[f"{case}/V0"][model]["runtime_us"]
                predicted_increment = predicted_observed - predicted_main
                errors.append(abs(predicted_increment - float(row["delta_to_main_us"])))
            print(f"  {model}: {median(errors):.2f}")


if __name__ == "__main__":
    check_selectors()
    check_observation_increments()
