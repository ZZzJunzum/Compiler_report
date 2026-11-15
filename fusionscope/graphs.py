"""Graph inputs, main result and instrumentation contract."""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

from .common import ROOT, SPEC, describe, sha, sites


def inputs(case, seed):
    family, shape, _ = describe(case)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape, dtype=np.float32)
    if family.startswith("R"):
        return jnp.asarray(x), external_weights(case, shape)
    width = shape[-1]
    kernel_shape = {"M": (width, width), "C": (3, 3, width, width)}.get(family, (width,))
    scale = np.float32(1 / math.sqrt(math.prod(kernel_shape[:-1])))
    # Separate random streams keep weights fixed across the three input batches.
    weights = np.random.default_rng(31)
    params = []
    for _ in range(2):
        kernel = weights.standard_normal(kernel_shape, dtype=np.float32) * scale
        bias = weights.standard_normal(width, dtype=np.float32) * np.float32(0.25)
        params.extend((kernel, bias))
    return jnp.asarray(x), tuple(map(jnp.asarray, params))


def layer(family, x, w, b):
    if family == "M":
        x = x @ w
    elif family == "C":
        x = jax.lax.conv_general_dilated(x, w, (1, 1), "SAME",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
    elif family == "N":
        x = x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + jnp.float32(1e-4))
        x = x * w
    else:
        x = x * w
    return x + b


def first_stage(family, x, p):
    with jax.named_scope("u_z"):
        return layer(family, x, p[0], p[1])


def activation(z, site, mode, observe, official=False):
    with jax.named_scope(f"{site}_relu"):
        a = jax.nn.relu(z) if official else jnp.maximum(z, jnp.float32(0))
    if mode == "diagnostic":
        observe.append(z)
    elif site in sites(mode):
        with jax.named_scope(f"probe_{site}"):
            source = z if sites(mode)[site] == "pre" else a
            observe.append(jnp.sum(source > 0, dtype=jnp.int32))
    return a


def remainder(family, top, mode, z, x, p):
    observe = []
    a = activation(z, "u", mode, observe)
    with jax.named_scope("v_z"):
        z2 = layer(family, a, p[2], p[3])
        if family == "C":
            z2 = z2 + x
    a2 = activation(z2, "v", mode, observe)
    with jax.named_scope("readout"):
        y = a2 * jnp.float32(0.75) + jnp.float32(0.125)
        if top == "branch":
            y = y + a * jnp.float32(0.25)
    return (y, *observe) if observe else y


def program(case, mode):
    family, _, top = describe(case)
    if family.startswith("R"):
        return external_program(case, mode)

    def workload(x, params):
        return remainder(family, top, mode, first_stage(family, x, params), x, params)
    return workload


def split_executable(case, mode, args):
    family, _, top = describe(case)
    front = jax.jit(partial(first_stage, family)).lower(*args).compile()
    z = front(*args)
    back = jax.jit(partial(remainder, family, top, mode)).lower(z, *args).compile()

    def execute(x, params):
        return back(front(x, params), x, params)
    return execute, (front, back)


def pilot(mode):
    def workload(x, p):
        with jax.named_scope("u_z"):
            z = x * p[0] + p[1]
        with jax.named_scope("u_relu"):
            y = jnp.maximum(z, jnp.float32(0))
        if mode == "only_y":
            return y
        with jax.named_scope("pilot_probe"):
            aux = {"sample": lambda: z[0, 0], "mean": lambda: jnp.mean(z), "full": lambda: z}[mode]()
        return y, aux
    return workload


def resnet_block(act):
    from flax import linen as nn
    from vendor.flax_imagenet.models import ResNetBlock
    return ResNetBlock(filters=64, conv=partial(nn.Conv, use_bias=False, dtype=jnp.float32),
                       norm=partial(nn.BatchNorm, use_running_average=True, momentum=0.9,
                                    epsilon=1e-5, dtype=jnp.float32, axis_name="batch"), act=act)


def external_weights(case, shape):
    from flax.traverse_util import flatten_dict, unflatten_dict
    if sha(ROOT / "vendor/flax_imagenet/models.py") != SPEC["external"]["file_sha256"]:
        raise ValueError("Upstream ResNet source differs from the registered version.")
    result = []
    for i in range(int(case[1])):
        values = resnet_block(jax.nn.relu).init(jax.random.fold_in(jax.random.key(31), i),
                                                jnp.zeros(shape, dtype=jnp.float32))
        flat = flatten_dict(values)
        # The stock zero final scale would initially silence the entire residual path.
        flat["params", "BatchNorm_1", "scale"] = jnp.ones_like(flat["params", "BatchNorm_1", "scale"])
        result.append(unflatten_dict(flat))
    return tuple(result)


def external_program(case, mode):
    blocks = int(case[1])

    def workload(x, params):
        observe = []
        for block in range(blocks):
            index = 0

            def act(z):
                nonlocal index
                index += 1
                if block == 0 and index == 1:
                    return activation(z, "u", mode, observe, official=True)
                if block == blocks - 1 and index == 2:
                    return activation(z, "v", mode, observe, official=True)
                return jax.nn.relu(z)

            with jax.named_scope(f"block{block}"):
                x = resnet_block(act).apply(params[block], x)
        return (x, *observe) if observe else x
    return workload
