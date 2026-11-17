"""Run the prospective CPU study. See README for freeze order and output paths."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter_ns
from zipfile import ZipFile

from fusionscope.common import ROOT, SPEC, dump, read, sha
from fusionscope.phase2.experiment import csv_text, zip_write
from .design import MECHANISM, REPEATS, ROUNDS, development, heldout, mechanism_options


PUBLISHED = ROOT / "results/phase3/cpu"
STAGES = ("mechanism", "development", "calibrate", "baselines", "lock", "discovery",
          "choose", "confirmation", "analyze", "verify")
FROZEN = ("design.py", "experiment.py", "workloads.py")


def signature():
    paths = [Path(__file__).with_name(name) for name in FROZEN]
    paths += [ROOT / name for name in ("protocol.json", "results/phase2/protocol.json", "run.py",
              "fusionscope/graphs.py", "fusionscope/phase2/variants.py", "fusionscope/bench.py")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def block(case, process, name, fns, args, rng, rounds):
    from fusionscope.bench import timed
    rows = []
    keys = list(fns)
    for i in range(-6, rounds):
        rng.shuffle(keys)
        for order, key in enumerate(keys):
            rows.append(dict(case=case.name, process=process, block=name, round=i, order=order,
                             variant=key, elapsed_ns=timed(fns[key], args)))
    return rows


def correctness(case, args_by_seed, fns, out):
    import jax
    import numpy as np
    from .workloads import primal
    diag = jax.jit(primal(case, "diagnostic"))
    rtol, atol = SPEC["tolerances"]["N" if case.group == "transfer" else case.ops[0]]
    rows = []
    for seed, args in args_by_seed.items():
        ref = np.asarray(fns["V0"](*args))
        _, u, v = diag(*args)
        counts = [int(np.count_nonzero(np.asarray(z) > 0)) for z in (u, v)]
        for name, fn in fns.items():
            r = fn(*args)
            y, *qs = r if isinstance(r, tuple) else (r,)
            y = np.asarray(y)
            expected = [] if name == "V0" or name.startswith("B") else counts
            actual = [int(q) for q in qs]
            ok_y = bool(y.shape == ref.shape and y.dtype == ref.dtype == np.float32 and
                        np.isfinite(y).all() and np.allclose(y, ref, rtol=rtol, atol=atol))
            ok_q = actual == expected and all(q.shape == () and q.dtype == np.int32 for q in qs)
            rows.append(dict(case=case.name, variant=name, seed=seed, max_abs=float(np.max(np.abs(y-ref))),
                             actual=str(actual), expected=str(expected), y_valid=ok_y, counts_valid=ok_q,
                             valid=ok_y and ok_q))
            if not (ok_y and ok_q):
                path = out / "failures" / f"{case.name}-{seed}.npz"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, **{f"leaf{i}": np.asarray(v) for i, v in enumerate(jax.tree.leaves(args))})
    return rows


def collect(out, stage, process, policy):
    import jax
    from fusionscope.bench import environment
    from fusionscope.evidence import scheduled_text_digest
    from .workloads import arguments, build
    folder = out / stage
    if policy:
        folder /= policy
    target = folder / f"p{process}.zip"
    env, sig = environment("cpu"), signature()
    if target.exists():
        with ZipFile(target) as z:
            meta = json.loads(z.read("environment.json"))
            if z.testzip() or meta["signature"] != sig or meta["environment"] != env:
                raise ValueError(f"Existing run differs: {target}")
        return
    fresh = stage in ("confirmation", "discovery")
    seeds = (101, 4471, 8231) if fresh else (42, 1729, 2026)
    cases = heldout() if fresh else development()
    if stage == "mechanism":
        cases = [c for c in cases if c.name in MECHANISM]
    rng = random.Random(f"32041:{stage}:{process}:{policy}")
    rng.shuffle(cases)
    times, checks, builds, overheads = [], [], [], []
    choices = read(out/"static/lock.json")["choices"] if stage == "discovery" else {}
    for case in cases:
        by_seed = {s: jax.device_put(arguments(case, s)) for s in (seeds[:1] if policy else seeds)}
        jax.block_until_ready(by_seed)
        args = by_seed[seeds[0]]
        labels = mechanism_options(case) if stage == "mechanism" else case.options
        if policy:
            labels = ["V2", choices[case.name][policy]]
        rng.shuffle(labels)
        ir, local, fns = {}, [], {}
        for name in labels:
            start = perf_counter_ns()
            fns[name], parts = build(case, name, args)
            wall = perf_counter_ns() - start
            for part in parts:
                hlo = part.compiled.as_text()
                mem = part.compiled.memory_analysis()
                stem = f"{name}/{part.name}/"
                ir[stem+"optimized_hlo.txt"] = hlo
                ir[stem+"stablehlo.mlir"] = part.lowered.as_text()
                local.append(dict(case=case.name, variant=name, component=part.name, lower_ns=part.lower_ns,
                                  compile_ns=part.compile_ns, build_wall_ns=wall, digest=scheduled_text_digest(hlo),
                                  argument_bytes=mem.argument_size_in_bytes, output_bytes=mem.output_size_in_bytes,
                                  temp_bytes=mem.temp_size_in_bytes, alias_bytes=mem.alias_size_in_bytes))
        # Preserve distinct scheduled modules, even across processes and clean checkouts.
        identity = sorted((p["variant"], p["component"], p["digest"]) for p in local)
        tag = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
        archive = f"ir/{case.name}-{tag}.zip"
        if not (out/archive).exists():
            zip_write(out/archive, ir)
        builds.extend(dict(p, archive=archive) for p in local)
        if not policy:
            checks.extend(correctness(case, by_seed, fns, out))
        gc.collect()
        start = perf_counter_ns()
        times.extend(block(case, process, "candidate", fns, args, rng, ROUNDS[stage]))
        for name in (("V2",) if policy else ("V0", "V1", "V2")):
            times.extend(block(case, process, "AA_"+name, dict.fromkeys("ABCD", fns[name]), args, rng, ROUNDS[stage]))
        overheads.append(dict(case=case.name, block_wall_ns=perf_counter_ns()-start))
        jax.clear_caches()
    meta = dict(stage=stage, policy=policy, process=process, pid=os.getpid(), environment=env, signature=sig,
                case_order=[c.name for c in cases], seeds=seeds, warmups=6, rounds=ROUNDS[stage])
    content = {"environment.json": json.dumps(meta, indent=2), "timings.csv": csv_text(times),
               "builds.csv": csv_text(builds), "blocks.csv": csv_text(overheads)}
    if checks:
        content["correctness.csv"] = csv_text(checks)
    zip_write(target, content)
    print(f"{stage} {policy} p{process}: {len(times)} calls, {len(checks)} validations", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stages", nargs="+", choices=STAGES)
    ap.add_argument("--out", type=Path, default=ROOT/"reproduction/phase3")
    ap.add_argument("--worker", type=int)
    ap.add_argument("--policy", default="")
    a = ap.parse_args()
    out = a.out.resolve()
    from run import pin, worker_env
    for stage in a.stages:
        if a.worker is not None:
            pin()
            if stage in ("calibrate", "baselines"):
                from .calibration import collect_static
                collect_static(out, stage)
            else:
                collect(out, stage, a.worker, a.policy)
            continue
        if stage in ("discovery", "confirmation"):
            from .models import verify_lock
            verify_lock(out, discovery=stage == "confirmation")
        if stage in (*REPEATS, "calibrate", "baselines"):
            for p in range(REPEATS.get(stage, 1)):
                policies = (("rank_alt", "fixed_alt") if p % 2 == 0 else ("fixed_alt", "rank_alt")) if stage == "discovery" else ("",)
                for policy in policies:
                    subprocess.run([sys.executable, "-m", "fusionscope.phase3.experiment", stage,
                                    "--out", str(out), "--worker", str(p), "--policy", policy],
                                   cwd=ROOT, env=worker_env("cpu"), check=True)
        elif stage in ("lock", "choose"):
            from .models import lock, choose
            (lock if stage == "lock" else choose)(out)
        else:
            from .analysis import analyze, verify
            (analyze if stage == "analyze" else verify)(out)


if __name__ == "__main__":
    main()
