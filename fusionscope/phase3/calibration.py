"""Device calibration from old shapes and unobserved-IR extraction."""
from math import log, prod
from statistics import median
from time import perf_counter_ns

import jax
import numpy as np

from fusionscope.bench import environment, timed
from fusionscope.common import dump, table
from fusionscope.graphs import layer
from fusionscope.phase2.cost_model import baseline_facts
from fusionscope.phase2.experiment import zip_write
from .design import development, heldout
from .workloads import arguments, build


def collect_static(out, stage):
    folder = out/"static"
    start = perf_counter_ns()
    if stage == "baselines":
        facts, modules = {}, {}
        for c in development()+heldout():
            args = arguments(c, 42 if c.group == "development" else 101)
            jax.block_until_ready(args)
            begin = perf_counter_ns()
            _, (part,) = build(c, "V0", args)
            hlo = part.compiled.as_text()
            f = baseline_facts(c.name, hlo=hlo)
            facts[c.name] = dict(facts=f, extraction_ns=perf_counter_ns()-begin,
                                 lower_ns=part.lower_ns, compile_ns=part.compile_ns)
            modules[c.name+"/optimized_hlo.txt"] = hlo
            modules[c.name+"/stablehlo.mlir"] = part.lowered.as_text()
            jax.clear_caches()
        zip_write(folder/"baseline_ir.zip", modules)
        dump(folder/"baseline.json", facts)
        return
    rows, stream, rates = [], [], []
    rng = np.random.default_rng(32041)
    # Two warm contiguous FP32 arrays. Logical traffic of a copy is one read plus one write.
    for power in range(13, 26):
        total = 2**power
        x = rng.standard_normal(total//8, dtype=np.float32)
        y = np.empty_like(x)
        for i in range(-6, 31):
            begin = perf_counter_ns()
            np.copyto(y, x)
            ns = perf_counter_ns()-begin
            rows.append(dict(kind="copy", shape=str(total), round=i, ns=ns))
        stream.append((total, total/median(r["ns"] for r in rows[-31:])))
    # Fit a single change in log effective bandwidth after discarding call-dominated tiny copies.
    points = [(n, log(b)) for n, b in stream if n >= 65536]
    def loss(k):
        return sum((b-np.mean([v for _, v in part]))**2 for part in (points[:k], points[k:]) for _, b in part)
    k = min(range(2, len(points)-1), key=loss)
    knee = points[k][0]
    small, large = [median(b for n, b in stream if valid(n)) for valid in
                    (lambda n: 65536 <= n < knee, lambda n: n >= knee)]
    scalar = jax.device_put(np.float32(1))
    call = jax.jit(lambda x: x+np.float32(1)).lower(scalar).compile()
    samples = [timed(call, (scalar,)) for _ in range(37)]
    rows += [dict(kind="dispatch_proxy", shape="()", round=i-6, ns=t) for i,t in enumerate(samples)]
    dispatch = median(samples[6:])/1000
    seen = set()
    for c in development():
        op = c.ops[0]
        if op not in "MC":
            continue
        x, p = arguments(c, 42)
        for t in c.tiles:
            shape = (t, *c.shape[1:])
            if (op, shape) in seen:
                continue
            seen.add((op, shape))
            args = (x[:t], p[0], p[1])
            jax.block_until_ready(args)
            fn = jax.jit(lambda x,w,b: layer(op,x,w,b)).lower(*args).compile()
            samples = [timed(fn, args) for _ in range(26)]
            rows.extend(dict(kind=op, shape=str(shape), round=i-6, ns=v) for i,v in enumerate(samples))
            flops = 2*prod(shape)*shape[-1]*(9 if op == "C" else 1)
            us = median(samples[6:])/1000
            rates.append(dict(op=op, shape=shape, flops=flops, layer_us=us,
                              mflop_per_us=flops/1e6/max(.1, us-dispatch)))
    elapsed = perf_counter_ns()-start
    table(folder/"calibration_samples.csv", rows)
    dump(folder/"calibration.json", dict(environment=environment("cpu"), dispatch_proxy_us=dispatch,
         knee_bytes=knee, small_bytes_ns=small, large_bytes_ns=large, copy_bytes_per_ns=stream,
         layer_rates=rates, calibration_ns=elapsed))
