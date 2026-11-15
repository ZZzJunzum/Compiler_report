"""Command line entry for the registered FusionScope experiment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from fusionscope.common import ROOT, SPEC, conditions, read


def worker_env(backend):
    env = os.environ.copy()
    env.update(JAX_PLATFORMS="cpu" if backend == "cpu" else "cuda", JAX_ENABLE_X64="false",
               JAX_ENABLE_COMPILATION_CACHE="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", XLA_FLAGS="", PYTHONUTF8="1")
    return env


def pin():
    import psutil
    proc = psutil.Process()
    proc.cpu_affinity(proc.cpu_affinity()[:1])


def launch(args, phase, cases, repeats):
    for idx in range(repeats):
        call = [sys.executable, str(ROOT / "run.py"), "--backend", args.backend, "--out", str(args.out),
                "--worker", phase, "--process", str(idx), "--cases", *cases]
        subprocess.run(call, env=worker_env(args.backend), cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["first","pilot","matrix","model","confirm","external",
                                           "controls","report","all","recommend","verify"], default="pilot")
    parser.add_argument("--backend", choices=["cpu","gpu"], default="cpu")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--case", default="R2")
    parser.add_argument("--query", default="positive_count")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--process", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--cases", nargs="+", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out = (args.out or ROOT / "results" / args.backend).resolve()
    if args.worker:
        pin()
        if args.worker == "baselines":
            from fusionscope.features import export_baselines
            export_baselines(args.out, args.cases)
        elif args.worker == "validation":
            from fusionscope.reference import check
            check(args.out)
        elif args.worker == "external_singles":
            from fusionscope.external import calibrate
            calibrate(args.out, args.cases, args.process)
        else:
            from fusionscope.bench import worker
            worker(args.out, args.worker, args.cases, args.process, args.backend)
        return
    stages = ["pilot","matrix","model","confirm","external","controls","report","verify"] if args.stage == "all" else [args.stage]
    count = SPEC["timing"]
    for stage in stages:
        if stage in ("first","pilot"):
            launch(args, "validation", ["E0_chain"], 1)
            launch(args, "probe4", ["E0_chain"], count["pilot_processes"])
            launch(args, "pilot", ["E0_chain"], count["pilot_processes"])
            from fusionscope.pilot import first_report
            first_report(args.out)
            if stage == "pilot":
                launch(args, "pilot", ["E3_chain","M0_chain","M3_chain"], count["pilot_processes"])
        elif stage == "matrix":
            if not (args.out / "pilot/first_report.md").exists():
                raise SystemExit("Run --stage pilot first to validate the instruments.")
            launch(args, "matrix", conditions(), count["matrix_processes"])
        elif stage == "model":
            launch(args, "baselines", conditions() + ["R1","R2"], 1)
            from fusionscope.model import lock_predictions
            lock_predictions(args.out)
        elif stage in ("confirm","external","controls"):
            from fusionscope.model import verify_lock
            lock = verify_lock(args.out)
            if stage == "confirm":
                launch(args, "confirm", conditions(), count["confirmation_processes"])
            elif stage == "external":
                from fusionscope.external import seal
                launch(args, "external_singles", ["R1","R2"], count["confirmation_processes"])
                seal(args.out)
                launch(args, "external", ["R1","R2"], count["external_processes"])
            else:
                launch(args, "boundary", lock["boundary_cases"], count["boundary_processes"])
        elif stage == "report":
            from fusionscope.report import report
            report(args.out)
        elif stage == "verify":
            from fusionscope.audit import verify
            verify(args.out)
        elif stage == "recommend":
            from fusionscope.model import recommend
            print(recommend(args.out, args.case, args.query))


if __name__ == "__main__":
    main()
