"""JAX implementations. Each cohort imports this module after setting CPU affinity."""
from math import prod, sqrt

import jax
import jax.numpy as jnp
import numpy as np

from fusionscope.graphs import first_stage, inputs, remainder
from fusionscope.phase2.variants import compile_part, count, first_activation, second_activation


def arguments(case, seed):
    if case.group == "development":
        return inputs(case.name, seed)
    x = np.random.default_rng(seed).standard_normal(case.shape, dtype=np.float32)
    weights = np.random.default_rng(1103)
    params = []
    w = case.shape[-1]
    for op in case.ops:
        shape = {"M": (w, w), "C": (3, 3, w, w)}.get(op, (w,))
        params.extend((weights.standard_normal(shape, dtype=np.float32) * np.float32(1/sqrt(prod(shape[:-1]))),
                       weights.standard_normal(w, dtype=np.float32) * np.float32(.25)))
    return jnp.asarray(x), tuple(map(jnp.asarray, params))


def primal(case, mode):
    def workload(x, params):
        return remainder(case.ops[1], case.top, mode, first_stage(case.ops[0], x, params), x, params)
    return workload


def joint(case):
    def workload(x, params):
        a = first_activation(case.ops[0], x, params)
        y, b = second_activation(case.ops[1], case.top, a, x, params)
        masks = ((a > 0).astype(jnp.int32), (b > 0).astype(jnp.int32))
        zero = jnp.int32(0)
        with jax.named_scope("joint_counts"):
            u, v = jax.lax.reduce(masks, (zero, zero), lambda p, q: (p[0]+q[0], p[1]+q[1]), tuple(range(a.ndim)))
        return y, u, v
    return workload


def build(case, label, args):
    if label == "V3":
        def front(x, params):
            a = first_activation(case.ops[0], x, params)
            with jax.named_scope("probe_u"):
                return a, count(a)
        def back(a, x, params):
            y, b = second_activation(case.ops[1], case.top, a, x, params)
            with jax.named_scope("probe_v"):
                return y, count(b)
        p = compile_part("front", front, args)
        x = args[0]
        a = jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)
        q = compile_part("back", back, (a, *args))
        def execute(x, params):
            a, u = p.compiled(x, params)
            y, v = q.compiled(a, x, params)
            return y, u, v
        return execute, [p, q]
    kind = {"V0": "B", "V1": "S", "V2": "J"}.get(label, label[0])
    body = joint(case) if kind == "J" else primal(case, "base" if kind == "B" else "post_post")
    if label.startswith("V"):
        execute = body
    else:
        tile = int(label[1:])
        def execute(x, params):
            blocks = x.reshape((x.shape[0]//tile, tile, *x.shape[1:]))
            with jax.named_scope("tile_body"):
                out = jax.lax.map(lambda block: body(block, params), blocks)
            if kind == "B":
                return out.reshape(x.shape)
            ys, us, vs = out
            with jax.named_scope("merge_partials"):
                return ys.reshape(x.shape), jnp.sum(us, dtype=jnp.int32), jnp.sum(vs, dtype=jnp.int32)
    part = compile_part("main", execute, args)
    return part.compiled, [part]
