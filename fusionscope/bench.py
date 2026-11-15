"""The worker is imported only after run.py has set CPU affinity and the backend."""
import gc
import hashlib
import os
import platform
import random
import sys
from time import perf_counter_ns

import jax
import jaxlib
import numpy as np
import psutil

from .common import MODES, ROOT, SPEC, describe, dump, sha, sites, table
from .graphs import inputs, pilot, program, split_executable


def environment(backend):
    return {"backend": backend, "devices": [str(d) for d in jax.devices(backend)],
            "device_kind": jax.devices(backend)[0].device_kind,
            "processor": platform.processor(), "os": platform.platform(),
            "python": platform.python_version(), "jax": jax.__version__, "jaxlib": jaxlib.__version__,
            "numpy": np.__version__, "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False), "ram_bytes": psutil.virtual_memory().total,
            "affinity": psutil.Process().cpu_affinity(),
            "flags": {k: os.environ.get(k, "") for k in
                      ("JAX_PLATFORMS","JAX_ENABLE_X64","JAX_ENABLE_COMPILATION_CACHE","XLA_FLAGS",
                       "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}}


def write_ir(path, fn, args, lowered, compiled):
    path.mkdir(parents=True, exist_ok=True)
    texts = {"jaxpr.txt": str(jax.make_jaxpr(fn)(*args)), "stablehlo.mlir": lowered.as_text(),
             "optimized_hlo.txt": compiled.as_text()}
    for name, content in texts.items():
        (path / name).write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")
    return {name: sha(path / name) for name in texts}


def timed(fn, args):
    start = perf_counter_ns()
    out = fn(*args)
    jax.block_until_ready(out)
    return perf_counter_ns() - start


def collect(fns, args, case, process, rng, group):
    keys = list(fns)
    for _ in range(SPEC["timing"]["warmup"]):
        rng.shuffle(keys)
        for key in keys:
            jax.block_until_ready(fns[key](*args))
    rows = []
    for iteration in range(SPEC["timing"]["rounds"]):
        rng.shuffle(keys)
        for order, key in enumerate(keys):
            elapsed = timed(fns[key], args)
            rows.append({"case": case, "process": process, "group": group, "round": iteration,
                         "order": order, "mode": key, "elapsed_ns": elapsed,
                         "endpoint": "all_device_outputs_ready"})
    return rows


def validate(case, fn_by_mode, all_args, stage, failures):
    family = describe(case)[0]
    rtol, atol = SPEC["tolerances"][family]
    checks = []
    baseline = fn_by_mode.get("base") or fn_by_mode.get("only_y")
    diagnostic = jax.jit(program(case, "diagnostic")) if stage != "probe4" else None
    if baseline is None:
        baseline = jax.jit(program(case, "base"))
    for seed, args in all_args.items():
        ref_y = np.asarray(baseline(*args))
        if diagnostic:
            _, z1, z2 = diagnostic(*args)
            refs = {k: int(np.count_nonzero(np.asarray(z) > 0)) for k,z in zip(("u","v"),(z1,z2))}
        else:
            x, p = jax.device_get(args)
            z = x * p[0] + p[1]
            pilot_aux = {"sample": z[0, 0], "mean": np.mean(z), "full": z}
            ref_y = np.maximum(z, 0)
        for label, fn in fn_by_mode.items():
            mode = label.removeprefix("split_")
            out = fn(*args)
            y, *q = out if isinstance(out, tuple) else (out,)
            y = np.asarray(y)
            error = float(np.max(np.abs(y - ref_y)))
            ok_y = bool(y.dtype == ref_y.dtype == np.float32 and y.shape == ref_y.shape
                        and np.isfinite(y).all() and np.allclose(y, ref_y, rtol=rtol, atol=atol))
            if diagnostic:
                actual = [int(np.asarray(v)) for v in q]
                expected = [refs[site] for site in sites(mode)]
                ok_aux = actual == expected and all(v.shape == () and v.dtype == np.int32 for v in q)
            else:
                actual = expected = []
                ok_aux = not q if mode == "only_y" else bool(len(q) == 1 and
                         q[0].shape == pilot_aux[mode].shape and q[0].dtype == np.float32
                         and np.allclose(np.asarray(q[0]), pilot_aux[mode], rtol=rtol, atol=atol))
            checks.append({"case": case, "seed": seed, "mode": label, "y_max_abs": error,
                           "y_pass": ok_y, "counts": actual, "reference_counts": expected,
                           "aux_pass": bool(ok_aux), "valid": bool(ok_y and ok_aux)})
            if not ok_y or not ok_aux:
                path = failures / f"{case}_{seed}.npz"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, **{f"leaf{i}": np.asarray(v) for i,v in
                                                 enumerate(jax.tree.leaves(args))})
    return checks


def measure_case(root, phase, case, process, backend, env):
    raw = root / phase / "raw" / f"p{process}" / case
    marker = raw / "done.json"
    signature = {"protocol": sha(ROOT / "protocol.json"), "graphs": sha(ROOT / "fusionscope/graphs.py"),
                 "worker": sha(ROOT / "fusionscope/bench.py"), "env": env}
    if marker.exists():
        from .common import read
        saved = read(marker)
        if saved["signature"] != signature or any(sha(raw / n) != h for n,h in saved["files"].items()):
            raise RuntimeError(f"Changed protocol, machine or completed data at {raw}. Use a fresh --out.")
        return
    start = perf_counter_ns()
    all_args = {seed: jax.device_put(inputs(case, seed), jax.devices(backend)[0])
                for seed in SPEC["input_seeds"]}
    jax.block_until_ready(all_args)
    args = all_args[SPEC["timed_seed"]]
    prep_ns = perf_counter_ns() - start
    modes = MODES
    if phase == "probe4":
        modes = ["only_y", "sample", "mean", "full"]
    if phase == "boundary":
        modes = ["base", "pre_pre", "post_pre", "split_pre_pre", "split_post_pre"]
    rng = random.Random(f"{SPEC['timing']['order_seed']}:{backend}:{phase}:{process}:{case}")
    order = list(modes)
    rng.shuffle(order)
    fns, builds = {}, []
    for mode in order:
        ir = root / phase / "ir" / case / mode
        if mode.startswith("split_"):
            start = perf_counter_ns()
            fns[mode], parts = split_executable(case, mode[6:], args)
            build = {"lower_ns": None, "compile_ns": None, "split_setup_ns": perf_counter_ns()-start}
            if process == 0:
                ir.mkdir(parents=True, exist_ok=True)
                for name, part in zip(("front","back"), parts):
                    (ir / f"{name}_hlo.txt").write_text(part.as_text(), encoding="utf-8", newline="\n")
        else:
            fn = pilot(mode) if phase == "probe4" else program(case, mode)
            start = perf_counter_ns()
            lowered = jax.jit(fn).lower(*args)
            middle = perf_counter_ns()
            fns[mode] = lowered.compile()
            stop = perf_counter_ns()
            build = {"lower_ns": middle-start, "compile_ns": stop-middle, "split_setup_ns": None}
            if process == 0:
                dump(ir / "hashes.json", write_ir(ir, fn, args, lowered, fns[mode]))
        builds.append({"case": case, "mode": mode, "process": process, **build})
    checks = validate(case, fns, all_args, phase, root / "failures")
    gc.collect()
    times = collect(fns, args, case, process, rng, "candidate")
    base = fns["only_y"] if phase == "probe4" else fns["base"]
    times += collect(dict.fromkeys(SPEC["timing"]["aa_labels"], base), args, case, process, rng, "AA")
    table(raw / "times.csv", times)
    table(raw / "builds.csv", builds)
    dump(raw / "checks.json", checks)
    dump(marker, {"signature": signature, "pid": os.getpid(), "input_preparation_ns": prep_ns,
                  "files": {n: sha(raw / n) for n in ("times.csv","builds.csv","checks.json")}})
    print(f"{phase}: p{process} {case} | {len(modes)} builds, {len(times)} samples", flush=True)


def worker(root, phase, cases, process, backend):
    env = environment(backend)
    dump(root / phase / f"env_p{process}.json", env)
    random.Random(f"{phase}:{process}:{SPEC['timing']['order_seed']}").shuffle(cases)
    for case in cases:
        measure_case(root, phase, case, process, backend, env)
        jax.clear_caches()
