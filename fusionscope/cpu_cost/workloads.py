"""Whole forward programs, observation contracts and their independent host oracle."""
from dataclasses import dataclass
from functools import partial
from math import prod

import jax
import jax.numpy as jnp
import numpy as np

from fusionscope.graphs import layer


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    shape: tuple
    readout: str
    tier: int
    group: str


def cases(split):
    dev = split == "development"
    matrix = [(16, 48), (64, 96), (256, 144), (512, 192)] if dev else [(48, 80), (320, 160)]
    conv = [(1, 12, 12, 8), (2, 18, 18, 12), (4, 24, 24, 16), (8, 28, 28, 24)] if dev else [
        (3, 20, 20, 12), (6, 26, 26, 20)]
    result = []
    for family in "EMNC":
        for tier, shape in enumerate(conv if family == "C" else matrix):
            for top in ["chain", "branch"]:
                name = f"{split[0]}_{family}{tier}_{top}"
                result.append(Case(name, family, shape, top, tier, "development" if dev else "new_shape"))
    if not dev:
        for family, shapes in {"G": [(40, 72), (192, 128)], "R": [(1, 15, 15, 12), (3, 23, 23, 20)],
                               "B": [(2, 18, 18, 16), (3, 22, 22, 24)]}.items():
            for tier, shape in enumerate(shapes):
                result.append(Case(f"t_{family}{tier}", family, shape, "native", tier, "new_architecture"))
    return result


# (activation index, measurement, reduced axes); raw and sample have no reduction.
CONTRACTS = {
    "count": [(0, "count", "all"), (1, "count", "all")],
    "row": [(0, "count", "last"), (1, "count", "last")],
    "energy": [(0, "energy", "all"), (1, "energy", "all")],
    "mixed": [(0, "count", "all"), (1, "max", "all")],
    "three": [(i, "count", "all") for i in range(3)],
    "shared": [(0, "energy", "all"), (0, "max", "all")],
    "axes": [(0, "count", "all"), (1, "count", "last")],
    "full": [(0, "full", "none")],
    "sample": [(0, "sample", "none")],
}


def axes(which, shape):
    return tuple(range(len(shape))) if which == "all" else (len(shape) - 1,)


def legal(contract, shapes):
    spec = CONTRACTS[contract]
    return len(spec) > 1 and all(which != "none" for _, _, which in spec) and len({
        (tuple(shapes[site]), axes(which, shapes[site])) for site, _, which in spec}) == 1


def site_shapes(case):
    x = case.shape
    if case.family == "G":
        return [(*x[:-1], 2*x[-1]), (*x[:-1], 2*x[-1]), x]
    if case.family == "B":
        b, h, w, c = x
        return [x, (b, (h+1)//2, (w+1)//2, c), (b, (h+1)//2, (w+1)//2, c*4)]
    return [x, x, x]


def variants(case):
    return ["V0"] + [f"{c}_{s}" for c in CONTRACTS
                     for s in (["S", "J"] if legal(c, site_shapes(case)) else ["S"])]


def resblock(case, act):
    from flax import linen as nn
    from vendor.flax_imagenet.models import BottleneckResNetBlock, ResNetBlock
    bottle = case.family == "B"
    return (BottleneckResNetBlock if bottle else ResNetBlock)(
        filters=case.shape[-1], strides=(2, 2) if bottle else (1, 1),
        conv=partial(nn.Conv, use_bias=False, dtype=jnp.float32),
        norm=partial(nn.BatchNorm, use_running_average=True, epsilon=1e-5, dtype=jnp.float32), act=act)


def parameters(case):
    width = case.shape[-1]
    if case.family in ("R", "B"):
        from flax.traverse_util import flatten_dict, unflatten_dict
        params = []
        for i in range(2 if case.family == "R" else 1):
            net = resblock(case, jax.nn.relu)
            flat = flatten_dict(net.init(jax.random.key(3701+i), jnp.zeros(case.shape, jnp.float32)))
            key = ("params", "BatchNorm_2" if case.family == "B" else "BatchNorm_1", "scale")
            flat[key] = jnp.ones_like(flat[key])
            params.append(unflatten_dict(flat))
        return tuple(params)
    rng = np.random.default_rng(3701)
    if case.family == "G":
        sizes = [(width, 2*width), (width, 2*width), (2*width, width)]
    else:
        kernel = {"M": (width,width), "C": (3,3,width,width)}.get(case.family, (width,))
        sizes = [kernel]*3
    result = []
    for shape in sizes:
        result += [rng.standard_normal(shape, dtype=np.float32) / np.float32(prod(shape[:-1])**0.5),
                   rng.standard_normal(shape[-1], dtype=np.float32)*np.float32(0.25)]
    return tuple(map(jnp.asarray, result))


def arguments(case, params, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(case.shape, dtype=np.float32)), params


def primal(case, x, p):
    if case.family in ("R","B"):
        zs = []
        def act(z):
            zs.append(z)
            return jax.nn.relu(z)
        for params in p:
            x = resblock(case, act).apply(params, x)
        return x, (zs[0], zs[1], zs[-1])
    if case.family == "G":
        gate = x @ p[0] + p[1]
        hidden = jax.nn.silu(gate) * (x @ p[2] + p[3])
        down = hidden @ p[4] + p[5]
        return x + down, (gate, hidden, down)
    zs = []
    for i in range(3):
        z = layer(case.family, x, p[2*i], p[2*i+1])
        zs.append(z)
        x = jnp.maximum(z, jnp.float32(0))
    y = x*jnp.float32(0.75) + jnp.float32(0.125)
    if case.readout == "branch":
        y = y + jnp.float32(0.25)*jnp.maximum(zs[0], jnp.float32(0))
    return y, tuple(zs)


def observe(zs, contract, joint):
    spec = CONTRACTS[contract]
    if joint and not legal(contract, [z.shape for z in zs]):
        raise ValueError("Direct joint reduction requires identical domains and axes")
    vals, initials, reducers, dims = [], [], [], []
    for site, kind, which in spec:
        z = zs[site]
        if kind == "full":
            return (z,)
        if kind == "sample":
            return (z[(0,)*z.ndim],)
        with jax.named_scope("observation_" + kind):
            v = (z > 0).astype(jnp.int32) if kind == "count" else (z*z if kind == "energy" else z)
        vals.append(v)
        initials.append(jnp.array(-np.inf if kind == "max" else 0, dtype=v.dtype))
        reducers.append(jax.lax.max if kind == "max" else jax.lax.add)
        dims.append(axes(which, z.shape))
    if joint:
        def combine(left, right):
            return tuple(f(a,b) for f,a,b in zip(reducers, left, right, strict=True))
        return jax.lax.reduce(tuple(vals), tuple(initials), combine, dims[0])
    return tuple(jax.lax.reduce(v, init, reducer, dim) for v,init,reducer,dim in
                 zip(vals, initials, reducers, dims, strict=True))


def program(case, name):
    def forward(x, params):
        y, zs = primal(case, x, params)
        if name == "diagnostic":
            return y, zs
        if name == "V0":
            return (y,)
        contract, schedule = name.split("_")
        return (y, *observe(zs, contract, schedule == "J"))
    return forward


def oracle(contract, arrays):
    expected = []
    for site, kind, which in CONTRACTS[contract]:
        z = arrays[site]
        dim = axes(which, z.shape)
        if kind == "count":
            v = np.sum(z > 0, axis=dim, dtype=np.int32)
        elif kind == "energy":
            v = np.sum(np.square(z, dtype=np.float64), axis=dim, dtype=np.float64)
        elif kind == "max":
            v = z.max(axis=dim)
        elif kind == "sample":
            v = z[(0,)*z.ndim]
        else:
            v = z
        expected.append((np.asarray(v), "int32" if kind == "count" else "float32"))
    return expected
