"""Train the frozen MNIST classifier and measure three adjacent-layer fragments."""
import argparse
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import sys
from urllib.request import urlopen
from zipfile import ZipFile

from fusionscope.common import ROOT, dump, read, sha, table
from fusionscope.phase2.experiment import csv_text, zip_write
from run import worker_env
from .collect import host, identities
from .design import LOCAL, ROUNDS, VARIANTS, lanes, now

DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
DATA_HASH = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
GRAPH_NAMES = ["hidden_1_2", "hidden_2_3", "hidden_3_4"]
BATCHES = [0, 64, 128]  # Starting image indices in the official test split.


def dataset(path):
    import numpy as np
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        with urlopen(DATA_URL, timeout=60) as stream:
            tmp.write_bytes(stream.read())
        if sha(tmp) != DATA_HASH:
            raise ValueError("MNIST download checksum mismatch")
        tmp.replace(path)
    if sha(path) != DATA_HASH:
        raise ValueError("Wrong MNIST archive")
    return np.load(path, allow_pickle=False)


def network(x, params):
    import jax.numpy as jnp
    sites = []
    for w, b in params[:-1]:
        z = x @ w + b
        sites.append(z)
        x = jnp.maximum(z, jnp.float32(0))
    w, b = params[-1]
    return x @ w + b, tuple(sites)


def pixels(x):
    import numpy as np
    return x.reshape(len(x), 784).astype(np.float32) / np.float32(255)


def train(out, path):
    env = host(1)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    if (out/"model.npz").exists():
        raise FileExistsError(out/"model.npz")
    out.mkdir(parents=True, exist_ok=True)
    with dataset(path) as data:
        x, y = pixels(data["x_train"][:20000]), data["y_train"][:20000]
        xt, yt = pixels(data["x_test"]), data["y_test"]
    rng = np.random.default_rng(4189)
    dims = [784, 96, 96, 96, 96, 10]
    params = tuple((jnp.asarray(rng.standard_normal((a, b), dtype=np.float32)*np.float32((2/a)**0.5)),
                    jnp.zeros(b, jnp.float32)) for a, b in zip(dims, dims[1:]))
    adam = optax.adam(0.001)
    state = adam.init(params)

    def objective(p, batch, label):
        return optax.softmax_cross_entropy_with_integer_labels(network(batch, p)[0], label).mean()

    @jax.jit
    def step(p, s, batch, label):
        updates, s = adam.update(jax.grad(objective)(p, batch, label), s, p)
        return optax.apply_updates(p, updates), s

    for _ in range(10):
        order = rng.permutation(len(y))
        for i in range(0, len(y)-127, 128):
            ix = order[i:i+128]
            params, state = step(params, state, x[ix], y[ix].astype(np.int32))
    logits = jax.jit(lambda a, p: network(a, p)[0])
    predicted = np.concatenate([np.asarray(logits(xt[i:i+128], params)).argmax(1)
                                for i in range(0, len(yt), 128)])
    np.savez_compressed(out/"model.npz", **{f"{key}{i}": np.asarray(arr)
                        for i, pair in enumerate(params) for key, arr in zip(("w", "b"), pair, strict=True)})
    dump(out/"training.json", dict(model_sha256=sha(out/"model.npz"), dataset_sha256=DATA_HASH,
         seed=4189, layer_widths=dims, train_images=len(y), optimizer="Adam", learning_rate=0.001,
         epochs=10, batch_size=128, test_correct=int((predicted == yt).sum()), test_images=len(yt),
         timed_test_batch_starts=BATCHES, environment=env, optax=optax.__version__, trained=now().isoformat()))


def fragment(variant):
    import jax.numpy as jnp
    from fusionscope.cpu_cost.workloads import observe
    def forward(x, params):
        (w1, b1), (w2, b2) = params
        first = x @ w1 + b1
        second = jnp.maximum(first, jnp.float32(0)) @ w2 + b2
        y = jnp.maximum(second, jnp.float32(0))
        if variant == "V0":
            return (y,)
        request, strategy = variant.split("_")
        return (y, *observe((first, second), request, strategy == "J"))
    return forward


def worker(out, path, index):
    env = host(1)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from fusionscope.bench import timed
    from fusionscope.cpu_cost.workloads import oracle
    training = read(out/"training.json")
    if sha(out/"model.npz") != training["model_sha256"]:
        raise ValueError("MNIST weight hash mismatch")
    with np.load(out/"model.npz", allow_pickle=False) as data:
        params = tuple(tuple(jnp.asarray(data[f"{key}{i}"]) for key in ("w", "b")) for i in range(5))
    with dataset(path) as data:
        images = pixels(data["x_test"][:192])
    batches = {i: jnp.asarray(images[i:i+64]) for i in BATCHES}
    reference = {i: jax.jit(network)(x, params)[1] for i, x in batches.items()}
    times, checks, builds, evidence = [], [], [], {}
    rng = random.Random(f"mnist-axis/{index}")
    order = list(enumerate(GRAPH_NAMES))
    rng.shuffle(order)
    start = now()
    for layer, graph in order:
        args = {i: (batches[i] if layer == 0 else jnp.maximum(reference[i][layer-1], 0),
                    params[layer:layer+2]) for i in BATCHES}
        jax.block_until_ready(args)
        variants = list(VARIANTS)
        rng.shuffle(variants)
        fns = {v: jax.jit(fragment(v)).lower(*args[0]).compile() for v in variants}
        for v, fn in fns.items():
            evidence[f"{graph}/{v}.hlo"] = fn.as_text()
            builds.append(dict(graph=graph, variant=v, **identities(fn)))
        for i in BATCHES:
            acts = [np.asarray(z) for z in reference[i][layer:layer+2]]
            result = {v: tuple(map(np.asarray, fn(*args[i]))) for v, fn in fns.items()}
            for v, (y, *q) in result.items():
                request = v.split("_")[0]
                expected = [] if v == "V0" else [r for r, _ in oracle(request, acts)]
                separate = result.get(request+"_S", (y,))[1:]
                errors = [int(np.abs(a.astype(np.int64)-b).max()) for a, b in zip(q, expected, strict=True)]
                types = all(a.dtype == np.int32 and a.shape == b.shape for a,b in zip(q, expected, strict=True))
                main = (y.dtype == np.float32 and y.shape == acts[1].shape and np.isfinite(y).all()
                        and np.allclose(y, result["V0"][0], atol=3e-4, rtol=3e-4)
                        and np.allclose(y, np.maximum(acts[1], 0), atol=3e-4, rtol=3e-4))
                paired = all(np.array_equal(a,b) for a,b in zip(q, separate, strict=True))
                checks.append(dict(graph=graph, batch_start=i, variant=v, main_ok=bool(main), typed=types,
                    max_reference_error=max(errors, default=0), matches_separate=paired,
                    valid=bool(main and types and paired and not any(errors)),
                    site1_within_1e_minus5=int((np.abs(acts[0]) <= 1e-5).sum()),
                    site2_within_1e_minus5=int((np.abs(acts[1]) <= 1e-5).sum()),
                    site1_min_abs=float(np.abs(acts[0]).min()), site2_min_abs=float(np.abs(acts[1]).min())))
        slots, batch_order = lanes("axis"), list(BATCHES)
        for rnd in ROUNDS:
            rng.shuffle(batch_order)
            for block, i in enumerate(batch_order):
                rng.shuffle(slots)
                for pos, (v, opt, alias) in enumerate(slots):
                    elapsed = timed(fns[v], args[i])
                    times.append(dict(graph=graph, seed=i, round=rnd, block=block, position=pos, variant=v,
                                      setting=opt, alias=alias, elapsed_ns=elapsed))
    evidence.update({"timings.csv": csv_text(times), "correctness.csv": csv_text(checks),
                     "builds.csv": csv_text(builds), "environment.json": json.dumps(dict(
                         process=index, start=start.isoformat(), end=now().isoformat(), environment=env,
                         model_sha256=training["model_sha256"], dataset_sha256=DATA_HASH))})
    zip_write(out/f"p{index}.zip", evidence)


def report(out, verify):
    from fusionscope.evidence import scheduled_text_digest
    from fusionscope.hlo import Module
    from .analysis import effects, process_medians, require, rows
    training = read(out/"training.json")
    require(sha(out/"model.npz") == training["model_sha256"] and training["dataset_sha256"] == DATA_HASH, out)
    processes, validation = [], []
    hashes = {}
    for p in range(5):
        with ZipFile(out/f"p{p}.zip") as z:
            meta = json.loads(z.read("environment.json"))
            require(z.testzip() is None and meta["process"] == p and meta["model_sha256"] == training["model_sha256"]
                    and meta["dataset_sha256"] == DATA_HASH and len(meta["environment"]["affinity"]) == 1, p)
            raw = rows(z, "timings.csv")
            processes.append(process_medians(raw, "axis", GRAPH_NAMES, BATCHES))
            checks = rows(z, "correctness.csv")
            require(len(checks) == 45 and {(r["graph"], int(r["batch_start"]), r["variant"]) for r in checks}
                    == set(product(GRAPH_NAMES, BATCHES, VARIANTS)), p)
            validation.extend(dict(process=p, **r) for r in checks)
            builds = rows(z, "builds.csv")
            require(len(builds) == 15 and {(r["graph"],r["variant"]) for r in builds}
                    == set(product(GRAPH_NAMES, VARIANTS)), p)
            for r in builds:
                key = (r["graph"],r["variant"])
                hlo = z.read("/".join(key)+".hlo").decode()
                require(r["fingerprint"] == Module(hlo).fingerprint() and r["named_schedule"]
                        == scheduled_text_digest(hlo) == hashes.setdefault(key, r["named_schedule"]), r)
    summaries = [dict(graph=g, **effects("axis", g, processes, BATCHES)[0]) for g in GRAPH_NAMES]
    for file, data in (("contrasts", summaries), ("checks", validation)):
        path = out/(file+".csv")
        if verify:
            require(path.read_text(encoding="utf8") == csv_text(data), path)
        else:
            table(path, data)
    print(f"{sum(r['valid']=='True' for r in validation)}/{len(validation)} whole-fragment checks")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["train", "measure", "worker", "report", "verify"])
    ap.add_argument("--out", type=Path, default=LOCAL/"mnist")
    ap.add_argument("--data", type=Path, default=LOCAL/"mnist.npz")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()
    out, path = args.out.resolve(), args.data.resolve()
    if args.action == "train":
        train(out, path)
    elif args.action == "measure":
        if list(out.glob("p*.zip")):
            raise FileExistsError("Choose a fresh output directory with the frozen checkpoint")
        for i in range(5):
            subprocess.run([sys.executable, "-B", "-m", __package__+".mnist", "worker", "--out", str(out),
                            "--data", str(path), "--index", str(i)], cwd=ROOT, env=worker_env("cpu"), check=True)
    elif args.action == "worker":
        worker(out, path, args.index)
    else:
        report(out, args.action == "verify")


if __name__ == "__main__":
    main()
