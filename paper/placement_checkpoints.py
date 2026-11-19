"""Locate the first CPU algebraic-simplifier rewrite in the original ReLU placement control."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from fusionscope.common import DOUBLES, ROOT, SPEC, records, table
from fusionscope.hlo import Module, REF
from fusionscope.phase2.experiment import zip_write
from run import pin, worker_env

PUBLISHED = ROOT / "results/placement_pass"
OLD_BLOB = "2f9cc0e840c721233f42694d0c46a4facaf93813"
BEFORE = ".simplification.after_pipeline-start.before_algsimp.txt"


def reference():
    package = PUBLISHED / "checkpoints.zip"
    if package.is_file():
        with ZipFile(package) as z:
            return z.read("historical.csv")
    return subprocess.check_output(["git", "cat-file", "blob", OLD_BLOB], cwd=ROOT)


def body(text):
    # Remove the local source/stack tables. Preserve computation text and instruction scopes.
    lines = text.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith(("%", "ENTRY %")))
    return "\n".join([lines[0], "", *lines[first:]]) + "\n"


def read_probes(text):
    module = Module(text)
    ops = {op.name: op for op in module.comps[module.entry]}
    operands = []
    for site in ("u", "v"):
        matches = [
            op for op in ops.values()
            if op.kind == "compare" and "direction=GT" in op.attrs and f"/probe_{site}/" in op.scope
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one entry comparison for site {site}")
        lhs, _ = REF.findall(matches[0].operands)
        operands.append(ops[lhs].kind)
    return tuple(operands)


def capture(out, directory):
    import jax
    import numpy as np
    from fusionscope.bench import environment
    from fusionscope.graphs import inputs, program
    from fusionscope.lowering import variant
    pin()
    device = jax.devices("cpu")[0]
    files = {"historical.csv": reference()}
    info = {}
    for case in ("E0_chain", "R1"):
        batches = {s: jax.device_put(inputs(case, s), device) for s in SPEC["input_seeds"]}
        args = batches[SPEC["timed_seed"]]
        diagnostic = jax.jit(program(case, "diagnostic"))
        reference_outputs = {}
        for seed, batch in batches.items():
            y, u, v = diagnostic(*batch)
            counts = [np.count_nonzero(np.asarray(z) > 0) for z in (u, v)]
            reference_outputs[seed] = np.asarray(y), counts
        rtol, atol = SPEC["tolerances"]["R1" if case == "R1" else "E"]
        for name in ("maximum", "nn_relu"):
            for mode in DOUBLES:
                key = f"{case}/{name}/{mode}"
                pattern = "*.jit_workload.before_optimizations.txt"
                previous = set(directory.glob(pattern))
                function = variant(case, mode, name == "nn_relu")
                executable = jax.jit(function).lower(*args).compile()
                new = set(directory.glob(pattern)) - previous
                if len(new) != 1:
                    raise ValueError(f"Expected one new workload module for {key}")
                prefix = new.pop().name.removesuffix("before_optimizations.txt")
                before = min(directory.glob(prefix + "*" + BEFORE))
                index = int(before.name[len(prefix):].split(".")[0])
                after, = directory.glob(f"{prefix}{index + 1:04d}.simplification.after_algsimp.*.txt")
                files[f"{key}/before.hlo"] = body(before.read_text(encoding="utf8"))
                files[f"{key}/after.hlo"] = body(after.read_text(encoding="utf8"))
                files[f"{key}/scheduled.hlo"] = body(executable.as_text())
                valid = {}
                for seed, batch in batches.items():
                    y, *q = executable(*batch)
                    y = np.asarray(y)
                    ref, counts = reference_outputs[seed]
                    main_matches = (y.shape == ref.shape and y.dtype == ref.dtype == np.float32
                                    and np.isfinite(y).all() and np.allclose(y, ref, rtol=rtol, atol=atol))
                    counts_match = (all(z.shape == () and z.dtype == np.int32 for z in q)
                                    and list(map(int, q)) == counts)
                    valid[seed] = bool(main_matches and counts_match)
                info[key] = dict(before=before.name[len(prefix):], after=after.name[len(prefix):], checks=valid)
    host = environment("cpu")
    host["flags"]["XLA_FLAGS"] = host["flags"]["XLA_FLAGS"].replace(directory.as_posix(), "<temporary-directory>")
    files["manifest.json"] = json.dumps(dict(environment=host, modules=info), indent=2)
    zip_write(out / "checkpoints.zip", files)


def extract(path):
    rows = []
    with ZipFile(path) as z:
        raw = z.read("historical.csv")
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if z.testzip() or blob != OLD_BLOB:
            raise ValueError("Historical comparison or checkpoint archive differs")
        manifest = json.loads(z.read("manifest.json"))["modules"]
        history = list(csv.DictReader(io.StringIO(raw.decode())))
        if len(history) != 16 or len(manifest) != 16:
            raise ValueError("Expected both expressions and all four placements for each of two graphs")
        for old in history:
            case, name, mode = (old[k] for k in ("case", "relu_implementation", "mode"))
            key = f"{case}/{name}/{mode}"
            module = manifest[key]
            before, after = (read_probes(z.read(f"{key}/{n}.hlo").decode()) for n in ("before", "after"))
            fingerprint = Module(z.read(f"{key}/scheduled.hlo").decode()).fingerprint()
            if fingerprint != old["sha256"] or old["valid"] != "True":
                raise ValueError(f"Scheduled program differs from the historical reference for {key}")
            before_name, after_name = module["before"], module["after"]
            adjacent = int(after_name[:4]) == int(before_name[:4]) + 1
            if not (before_name.endswith(BEFORE) and ".simplification.after_algsimp." in after_name and adjacent):
                raise ValueError(f"Expected adjacent CPU algebraic-simplifier checkpoints for {key}")
            checks = module["checks"]
            if set(map(int, checks)) != set(SPEC["input_seeds"]) or not all(checks.values()):
                raise ValueError(f"Output validation failed for {key}")
            rows.append(dict(
                case=case, relu=name, placement=mode, before_u=before[0], before_v=before[1],
                after_u=after[0], after_v=after[1], scheduled_sha256=fingerprint, validated_seeds=len(checks),
            ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["record", "verify", "worker"])
    parser.add_argument("--out", type=Path, default=ROOT / "reproduction/placement_pass")
    parser.add_argument("--dumps", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.action == "worker":
        if args.dumps is None:
            parser.error("worker requires --dumps")
        capture(out, args.dumps)
        return
    if args.action == "record":
        if (out / "checkpoints.zip").exists():
            raise FileExistsError(out / "checkpoints.zip")
        with tempfile.TemporaryDirectory(prefix="fusionscope-passes-") as tmp:
            directory = Path(tmp).as_posix()
            env = worker_env("cpu")
            env["XLA_FLAGS"] = (f'--xla_dump_to="{directory}" --xla_dump_hlo_as_text '
                                '--xla_dump_hlo_module_re=jit_workload --xla_dump_hlo_pass_re=.*')
            subprocess.run([sys.executable, "-B", "-m", "paper.placement_checkpoints", "worker",
                            "--out", str(out), "--dumps", directory], env=env, cwd=ROOT, check=True)
    rows = extract(out / "checkpoints.zip")
    if args.action == "record":
        table(out / "comparisons.csv", rows)
    elif records(out / "comparisons.csv") != [{k: str(v) for k, v in r.items()} for r in rows]:
        raise ValueError("Published rows differ from the archived HLO and validation records")
    validated = sum(row["validated_seeds"] for row in rows)
    print(f"{len(rows)} scheduled fingerprints match history; {validated} output checks passed")


if __name__ == "__main__":
    main()
