"""Freeze the development winners and remeasure them against the joint cap-16 rule."""
import argparse
import gc
import json
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns
from zipfile import ZipFile

from fusionscope.common import ROOT, dump, read, sha
from fusionscope.phase3.design import development
from run import pin, worker_env


ARCHIVE = ROOT / "results/phase3/cpu"
PROCESSES = 5
ROUNDS = 30
SEEDS = (42, 1729, 2026)


def source_hashes():
    from fusionscope.phase3.experiment import signature
    paths = ["fusionscope/headroom.py", "fusionscope/hlo.py", "fusionscope/phase3/data.py",
             "paper/check_diagnostics.py", "fusionscope/phase3/models.py"]
    return {**signature(), **{path: sha(ROOT / path) for path in paths}}


def nominations():
    from fusionscope.phase3.data import cohort, direction
    from fusionscope.phase3.models import ordered
    from paper.check_diagnostics import fixed_candidate
    data = cohort(ARCHIVE, "development")
    cases = {}
    for case in development():
        d = data[case.name]
        legal = [v for v in ordered(case) if v not in d["invalid"]]
        fixed = fixed_candidate(case, "J16")
        if fixed not in legal:
            raise ValueError(f"Invalid fixed rule on {case.name}")
        nominee = min(legal, key=d["us"].get)
        gain, resolved = direction(d, nominee, fixed)
        test = list(dict.fromkeys([fixed, nominee]))
        # Also test a single, simple exception on the four biggest development opportunities.
        if case.ops == "CC" and case.shape[0] in (4, 8) and "J1" not in test:
            test.append("J1")
        cases[case.name] = dict(fixed=fixed, nominee=nominee, test=test, legal=len(legal),
            headroom_us=d["us"][fixed] - d["us"][nominee], paired_us=gain,
            floor_us=d["floor"], resolved_gain=bool(resolved and gain > 0))
    return dict(rule="J16", processes=PROCESSES, rounds=ROUNDS, seeds=SEEDS, cases=cases,
                development_files={p.name: sha(p) for p in sorted((ARCHIVE/"development").glob("p*.zip"))},
                source=source_hashes())


def freeze(out):
    path = out/"lock.json"
    lock = nominations()
    if path.exists():
        if json.loads(json.dumps(lock)) != read(path):
            raise ValueError("Existing nomination lock differs. Use a new output directory.")
    else:
        dump(path, lock)
    print(f"{len(lock['cases'])} nominations frozen: {sha(path)}", flush=True)


def lock_for_run(out):
    path = out/"lock.json"
    lock = read(path)
    if lock["source"] != source_hashes() or any(
        sha(ARCHIVE/"development"/name) != digest for name, digest in lock["development_files"].items()):
        raise ValueError("The nomination code or development evidence changed after the freeze.")
    return lock, sha(path)


def collect(out, pid):
    pin()
    import jax
    from fusionscope.bench import environment
    from fusionscope.hlo import Module
    from fusionscope.phase2.experiment import csv_text, zip_write
    from fusionscope.phase3.experiment import block, correctness
    from fusionscope.phase3.workloads import arguments, build

    lock, digest = lock_for_run(out)
    env = environment("cpu")
    target = out/"repeat"/f"p{pid}.zip"
    if target.exists():
        with ZipFile(target) as z:
            info = json.loads(z.read("environment.json"))
            if z.testzip() or (info["lock"], info["process"], info["environment"]) != (digest, pid, env):
                raise ValueError(f"Incompatible or damaged result: {target}")
        return
    rng = random.Random(f"joint16-repeat-20261118:{pid}")
    cases = development()
    rng.shuffle(cases)
    times, checks, builds = [], [], []
    for c in cases:
        lanes = lock["cases"][c.name]["test"]
        inputs = {s: jax.device_put(arguments(c, s)) for s in SEEDS}
        jax.block_until_ready(inputs)
        args = inputs[SEEDS[0]]
        order = ["V0", *lanes]
        rng.shuffle(order)
        fns = {}
        for v in order:
            start = perf_counter_ns()
            fns[v], parts = build(c, v, args)
            elapsed = perf_counter_ns() - start
            for part in parts:
                builds.append(dict(case=c.name, process=pid, variant=v, component=part.name,
                    build_wall_ns=elapsed, fingerprint=Module(part.compiled.as_text()).fingerprint()))
        checks.extend(correctness(c, inputs, fns, out))
        gc.collect()
        times.extend(block(c, pid, "candidate", {v:fns[v] for v in lanes}, args, rng, ROUNDS))
        for v in lanes:
            times.extend(block(c, pid, "AA_"+v, dict.fromkeys("ABCD", fns[v]), args, rng, ROUNDS))
        jax.clear_caches()
    meta = dict(lock=digest, process=pid, environment=env, case_order=[c.name for c in cases],
                seeds=SEEDS, rounds=ROUNDS, warmups=6)
    zip_write(target, {"environment.json": json.dumps(meta, indent=2),
                       "timings.csv": csv_text(times), "correctness.csv": csv_text(checks),
                       "builds.csv": csv_text(builds)})
    print(f"Repeat {pid+1}/{PROCESSES}: {len(times)} calls, {len(checks)} checks", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "repeat"])
    parser.add_argument("--out", type=Path, default=ROOT/"reproduction/headroom")
    parser.add_argument("--process", type=int, choices=range(PROCESSES))
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == "freeze":
        freeze(out)
    elif args.process is not None:
        collect(out, args.process)
    else:
        for i in range(PROCESSES):
            subprocess.run([sys.executable, "-B", "-m", "fusionscope.headroom", "repeat",
                            "--out", str(out), "--process", str(i)], env=worker_env("cpu"), cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
