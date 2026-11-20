"""Collect fresh mixed-block CPU timings or independently capture compiler output."""
import argparse
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from zipfile import ZipFile

from fusionscope.common import ROOT, dump, sha
from fusionscope.phase2.experiment import csv_text, zip_write
from run import worker_env
from .design import (AXIS_GRAPHS, DESIGN, LOCAL, ROUNDS, SEEDS, TASK_GRAPHS, VARIANTS,
                     case_for, compiler_options, design_lock, function, graphs, lanes, now, settings)


def source_lock():
    files = ["axis_probe/collect.py", "axis_probe/design.py", "axis_probe/semantics.py",
             "cpu_cost/workloads.py", "graphs.py", "bench.py", "hlo.py", "evidence.py"]
    return {name: sha(ROOT/"fusionscope"/name) for name in files}


def identities(compiled):
    from fusionscope.hlo import Module
    from fusionscope.evidence import scheduled_text_digest
    hlo = compiled.as_text()
    return dict(fingerprint=Module(hlo).fingerprint(), named_schedule=scheduled_text_digest(hlo),
                temp_bytes=compiled.memory_analysis().temp_size_in_bytes)


def host(width):
    import psutil
    proc = psutil.Process()
    available = proc.cpu_affinity()
    if len(available) < width:
        raise ValueError(f"Requested {width} logical CPUs, available {available}")
    proc.cpu_affinity(available[:width])
    import jaxlib
    from jaxlib import version
    from fusionscope.bench import environment

    env = environment("cpu")
    env["jaxlib_git"] = version._git_hash
    lib = Path(jaxlib.__file__).parent
    binaries = [p for p in lib.iterdir() if "jax_common" in p.name and p.is_file()]
    if len(binaries) != 1:
        raise ValueError(f"Cannot identify the XLA library in {lib}")
    env["compiler_binary"] = binaries[0].name
    env["compiler_sha256"] = sha(binaries[0])
    env["cpu_model"] = os.environ.get("PROCESSOR_IDENTIFIER", env["processor"])
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            env["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return env


def capture_one(case, variant, setting, args, folder):
    import jax
    from fusionscope.hlo import Module
    from fusionscope.evidence import scheduled_text_digest

    dump_dir = Path("dump")
    before = set(dump_dir.glob("*.cpu_after_optimizations.txt"))
    lower = jax.jit(function(case, variant)).lower(*args)
    compiled = lower.compile(compiler_options=compiler_options(setting))
    identity = identities(compiled)
    new = set(dump_dir.glob("*.cpu_after_optimizations.txt")) - before
    if len(new) != 1:
        raise ValueError(f"Expected one new compiled module for {case.name}/{variant}/{setting}, got {new}")
    optimized = new.pop()
    prefix = optimized.name.removesuffix("cpu_after_optimizations.txt")
    text = optimized.read_text(encoding="utf8")
    if (Module(text).fingerprint() != identity["fingerprint"] or
            scheduled_text_digest(text) != identity["named_schedule"]):
        raise ValueError(f"Unmatched compiler dump {optimized}")
    files = {"input.stablehlo": str(lower.compiler_ir("stablehlo"))}
    for path in sorted(dump_dir.glob(prefix+"*")):
        name = path.name.removeprefix(prefix)
        if path.is_file() and (name.endswith((".ll", ".txt", ".o"))):
            files[name] = path.read_bytes()
    if not any(n.endswith("buffer-assignment.txt") for n in files):
        raise ValueError("Missing buffer assignment")
    if variant != "V0" and not all(any(n.endswith(suffix) for n in files)
                                   for suffix in ("ir-with-opt.ll", "ir-no-opt.ll", ".o")):
        raise ValueError("Missing backend code")
    files["identity.json"] = json.dumps(dict(**identity, graph=case.name, variant=variant,
                                             setting=setting, compiler_options=compiler_options(setting)))
    zip_write(folder/case.name/f"{variant}_{setting}.zip", files)


def capture(out, width):
    env = host(width)
    from fusionscope.cpu_cost.workloads import arguments, parameters
    import jax
    dest = out/"ir"/f"cpu{width}"
    if (dest/"environment.json").exists():
        raise FileExistsError(dest)
    for name in (list(dict.fromkeys(AXIS_GRAPHS + TASK_GRAPHS)) if width == 1 else TASK_GRAPHS):
        c = case_for(name)
        args = arguments(c, parameters(c), SEEDS[0])
        for opt in (settings("task") if name in TASK_GRAPHS else settings("axis")):
            for variant in VARIANTS:
                capture_one(c, variant, opt, args, dest)
                jax.clear_caches()
    dump(dest/"environment.json", dict(collected=now().isoformat(), lock=design_lock(), environment=env))


def worker(out, panel, session, index, width):
    env = host(width)
    import jax
    import psutil
    from fusionscope.bench import timed
    from fusionscope.cpu_cost.workloads import parameters, arguments
    from .semantics import fixed_sites, pressure_inputs, validate

    stamp = now()
    rng = random.Random(f"{session}/{index}/{panel}/{width}")
    order = list(graphs(panel))
    rng.shuffle(order)
    tim, builds, checks = [], [], []
    for name in order:
        c = case_for(name)
        params = parameters(c)
        by_seed = {seed: arguments(c, params, seed) for seed in SEEDS}
        jax.block_until_ready(by_seed)
        keys = [(v, opt) for opt in settings(panel) for v in VARIANTS]
        rng.shuffle(keys)
        fns = {}
        for v, opt in keys:
            start = perf_counter_ns()
            ex = jax.jit(function(c, v)).lower(*by_seed[SEEDS[0]]).compile(compiler_options=compiler_options(opt))
            elapsed = perf_counter_ns()-start
            fns[v, opt] = ex
            builds.append(dict(graph=name, variant=v, setting=opt, build_ns=elapsed, **identities(ex)))
        checks.extend(validate(c, fns, {**by_seed, **pressure_inputs(c, params)}))
        shuffled_lanes = lanes(panel)
        shuffled_seeds = list(SEEDS)
        for rnd in ROUNDS:
            rng.shuffle(shuffled_seeds)
            for block, seed in enumerate(shuffled_seeds):
                rng.shuffle(shuffled_lanes)
                for pos, (variant, opt, alias) in enumerate(shuffled_lanes):
                    elapsed = timed(fns[variant, opt], by_seed[seed])
                    tim.append(dict(graph=name, seed=seed, round=rnd, block=block, position=pos,
                                    variant=variant, setting=opt, alias=alias, elapsed_ns=elapsed))
        jax.clear_caches()
    end = now()
    meta = dict(session=session, panel=panel, process=index, pid=os.getpid(), start=stamp.isoformat(),
                end=end.isoformat(), local_date=stamp.date().isoformat(), timezone=stamp.tzinfo.key,
                threads_at_end=psutil.Process().num_threads(), environment=env, graph_order=order,
                input_seeds=SEEDS, weight_seed=3701, lock=design_lock(), sources=source_lock())
    path = out/"sessions"/session/f"p{index}.zip"
    zip_write(path, {"environment.json": json.dumps(meta), "timings.csv": csv_text(tim),
                     "builds.csv": csv_text(builds), "correctness.csv": csv_text(checks),
                     "fixed_sites.json": json.dumps(fixed_sites())})
    print(f"{session}/p{index}: {len(tim)} calls, {len(checks)} whole-program checks", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "capture-worker", "session", "worker"])
    parser.add_argument("--out", type=Path, default=LOCAL)
    parser.add_argument("--panel", choices=["axis", "task"], default="axis")
    parser.add_argument("--name", default="discovery")
    parser.add_argument("--cpus", type=int, choices=[1, 4], default=1)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    out = args.out.resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("Invalid session name")
    if args.action == "capture-worker":
        capture(out, args.cpus)
    elif args.action == "worker":
        worker(out, args.panel, args.name, args.index, args.cpus)
    else:
        env = worker_env("cpu")
        env["PYTHONPATH"] = str(ROOT)
        cmd = [sys.executable, "-B", "-m", "fusionscope.axis_probe.collect"]
        common = ["--out", str(out), "--cpus", str(args.cpus)]
        if args.action == "capture":
            # XLA's Windows dump writer needs a pre-existing directory under an ASCII cwd.
            with TemporaryDirectory(prefix="fsaxis-") as scratch:
                (Path(scratch)/"dump").mkdir()
                env["XLA_FLAGS"] = ("--xla_dump_to=dump --xla_dump_hlo_as_text=true "
                                    "--xla_dump_hlo_pass_re=fusion|tree|parallel.task")
                subprocess.run(cmd+["capture-worker", *common], cwd=scratch, env=env, check=True)
        else:
            folder = out/"sessions"/args.name
            folder.mkdir(parents=True, exist_ok=False)
            for index in range(DESIGN["processes_per_session"]):
                subprocess.run(cmd+["worker", *common, "--panel", args.panel, "--name", args.name,
                                   "--index", str(index)], env=env, cwd=ROOT, check=True)
            with ZipFile(folder/"p0.zip") as z:
                meta = json.loads(z.read("environment.json"))
            dump(folder/"session.json", dict(name=args.name, panel=args.panel, cpus=args.cpus,
                 date=meta["local_date"], timezone=meta["timezone"], lock=design_lock(),
                 archives={p.name: sha(p) for p in sorted(folder.glob("p*.zip"))}))


if __name__ == "__main__":
    main()
