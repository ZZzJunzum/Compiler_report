"""Run each CPU checkpoint in order. The published directory is append-only during collection."""
import argparse
import gc
import json
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter_ns
from zipfile import ZipFile

from fusionscope.common import ROOT, dump, read, sha
from fusionscope.phase2.experiment import csv_text, zip_write
from run import pin, worker_env

PUBLISHED = ROOT / "results/cpu_cost"
SEEDS = {"development": (29, 4831), "test": (733, 4261, 9521)}
PROCESSES = {"development": 3, "test": 5}
ROUNDS = {"development": 12, "test": 20}


def catalogue(out, split):
    import jax
    from .features import source, stable_features
    from .workloads import arguments, cases, parameters, program, variants
    target = out / f"{split}/lowered.zip"
    if target.exists():
        raise FileExistsError(target)
    rows, files = {}, {}
    for c in cases(split):
        args = arguments(c, parameters(c), SEEDS[split][0])
        for name in variants(c):
            start = perf_counter_ns()
            lowered = jax.jit(program(c, name)).lower(*args)
            static_ir = lowered.compiler_ir(dialect="stablehlo")
            features = stable_features(static_ir)
            extraction = perf_counter_ns()-start
            key = c.name+"/"+name
            files[key+".mlir"] = str(static_ir)
            rows[key] = dict(source=source(c, name), stable=features, extraction_ns=extraction)
    files["features.json"] = json.dumps(rows, sort_keys=True)
    zip_write(target, files)


def inspect(out):
    import jax
    from .features import trigger
    from .workloads import CONTRACTS, arguments, cases, legal, parameters, program, site_shapes
    if (out/"gate.json").exists():
        raise FileExistsError(out/"gate.json")
    files, choices = {}, {}
    for c in cases("test"):
        args = arguments(c, parameters(c), SEEDS["test"][0])
        for contract in CONTRACTS:
            if not legal(contract, site_shapes(c)):
                continue
            name = contract+"_S"
            start = perf_counter_ns()
            ref = jax.jit(program(c,name)).lower(*args).compile()
            compiled = perf_counter_ns()
            hlo = ref.as_text()
            hits, fingerprint = trigger(hlo, c, contract)
            stop = perf_counter_ns()
            key = c.name+"/"+contract
            files[key+".hlo"] = hlo
            choices[key] = dict(choice="J" if hits else "S", anchors=hits, fingerprint=fingerprint,
                                separate_build_ns=compiled-start, inspect_ns=stop-compiled)
    zip_write(out/"test/separate_hlo.zip", files)
    dump(out/"gate.json", dict(predictions=sha(out/"predictions.json"), choices=choices,
                              listings=sha(out/"test/separate_hlo.zip")))


def validation(c, fns, by_seed):
    import jax
    import numpy as np
    from .workloads import oracle, program, site_shapes
    rows = []
    diag = jax.jit(program(c, "diagnostic"))
    for seed, args in by_seed.items():
        y, sites = diag(*args)
        arrays = [np.asarray(v) for v in sites]
        if [v.shape for v in arrays] != site_shapes(c):
            raise ValueError(f"Site declarations disagree on {c.name}")
        main = np.asarray(fns["V0"](*args)[0])
        refs = {}
        for name, fn in fns.items():
            outputs = fn(*args)
            key = name.split("_")[0]
            if key != "V0" and key not in refs:
                refs[key] = oracle(key, arrays)
            expected = [(main, "float32"), *refs.get(key, [])]
            valid = len(expected) == len(outputs)
            max_count_error, max_scaled = 0, 0.
            for actual, (target, dtype) in zip(outputs, expected, strict=True):
                actual = np.asarray(actual)
                valid &= actual.dtype.name == dtype and actual.shape == target.shape and np.isfinite(actual).all()
                if dtype == "int32":
                    max_count_error = max(max_count_error, int(np.max(np.abs(actual.astype(np.int64)-target))))
                else:
                    diff = np.abs(actual.astype(np.float64)-target)/(3e-4 + 3e-4*np.abs(target))
                    max_scaled = max(max_scaled, float(diff.max()))
            valid &= max_count_error == 0 and max_scaled <= 1
            rows.append(dict(case=c.name, variant=name, seed=seed, valid=bool(valid),
                             max_count_error=max_count_error, max_fraction_of_tolerance=max_scaled))
    return rows


def measure(out, split, idx):
    import jax
    from fusionscope.bench import environment
    from fusionscope.hlo import Module
    from fusionscope.phase3.experiment import block
    from .workloads import arguments, cases, parameters, program, variants
    target = out/split/f"p{idx}.zip"
    lock = sha(out/("gate.json" if split == "test" else "protocol.json"))
    if target.exists():
        with ZipFile(target) as z:
            meta = json.loads(z.read("environment.json"))
            if z.testzip() or meta["lock"] != lock or meta["environment"] != environment("cpu"):
                raise ValueError(f"Inconsistent process archive {target}")
        return
    rng = random.Random(f"17011:{split}:{idx}")
    graphs = cases(split)
    rng.shuffle(graphs)
    timings, checks, builds = [], [], []
    for c in graphs:
        params = parameters(c)
        by_seed = {s: arguments(c, params, s) for s in SEEDS[split]}
        jax.block_until_ready(by_seed)
        args = by_seed[SEEDS[split][0]]
        labels = variants(c)
        rng.shuffle(labels)
        fns = {}
        for name in labels:
            start = perf_counter_ns()
            compiled = jax.jit(program(c,name)).lower(*args).compile()
            elapsed = perf_counter_ns()-start
            fp = Module(compiled.as_text()).fingerprint()
            builds.append(dict(case=c.name, variant=name, component="main", process=idx,
                               build_wall_ns=elapsed, fingerprint=fp,
                               temp_bytes=compiled.memory_analysis().temp_size_in_bytes))
            fns[name] = compiled
        checks.extend(validation(c, fns, by_seed))
        timings.extend(block(c, idx, "candidate", fns, args, rng, ROUNDS[split]))
        if split == "test":
            for name in labels:
                timings.extend(block(c, idx, "AA_"+name, dict.fromkeys("ABCD", fns[name]), args, rng, ROUNDS[split]))
        del fns, params, by_seed, args, compiled
        jax.clear_caches()
        gc.collect()
    meta = dict(environment=environment("cpu"), lock=lock, process=idx, rounds=ROUNDS[split], warmups=6,
                case_order=[c.name for c in graphs], seeds=SEEDS[split])
    zip_write(target, {"timings.csv": csv_text(timings), "builds.csv": csv_text(builds),
                       "correctness.csv": csv_text(checks), "environment.json": json.dumps(meta)})
    bad = [r for r in checks if not r["valid"]]
    print(f"{split} process {idx}: {len(builds)} configurations, {len(checks)} checks, {len(bad)} failures", flush=True)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["develop","predict","inspect","test","report","verify","worker"])
    parser.add_argument("--out", type=Path, default=ROOT/"reproduction/cpu_cost")
    parser.add_argument("--job")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == "worker":
        pin()
        if args.job.startswith("lower_"):
            catalogue(out, args.job[6:])
        elif args.job in PROCESSES:
            measure(out, args.job, args.index)
        elif args.job == "inspect":
            inspect(out)
        else:
            raise ValueError(args.job)
        return
    def child(job, i=0):
        subprocess.run([sys.executable, "-B","-m",__package__+".experiment","worker",
                        "--out",str(out),"--job",job,"--index",str(i)], env=worker_env("cpu"), cwd=ROOT, check=True)
    if args.stage == "develop":
        import shutil
        out.mkdir(parents=True, exist_ok=True)
        if out != PUBLISHED and not (out/"protocol.json").exists():
            shutil.copyfile(PUBLISHED/"protocol.json", out/"protocol.json")
        if not (out/"development/lowered.zip").exists():
            child("lower_development")
        for i in range(PROCESSES["development"]):
            child("development", i)
    elif args.stage == "predict":
        from .model import predict
        if not (out/"test/lowered.zip").exists():
            child("lower_test")
        predict(out)
    elif args.stage == "inspect":
        child("inspect")
    elif args.stage == "test":
        gate = read(out/"gate.json")
        if gate["predictions"] != sha(out/"predictions.json") or gate["listings"] != sha(out/"test/separate_hlo.zip"):
            raise ValueError("Prediction or inspection lock changed")
        for i in range(PROCESSES["test"]):
            child("test", i)
    else:
        from .analysis import report
        report(out, verify=args.stage == "verify")


if __name__ == "__main__":
    cli()
