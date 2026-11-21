from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from fusionscope.axis_probe.design import MINI, PRIMARY, case_for, function as original_function
from fusionscope.common import ROOT, read
from fusionscope.cpu_cost.workloads import Case, arguments, observe, parameters, primal

PROTOCOL = ROOT/"docs/plans/partition_boundary.json"
PLAN = read(PROTOCOL)
OUT = ROOT/"results/partition_boundary"
SEEDS = PLAN["timing"]["seeds"]
KEYS = ("J", "S", "P", "off", "global", "global_S", "global_off", "V0")
STRUCTURES = ("residual_square", "dual_input", "gated_product", "center_residual")
RESOURCES = ("one", "four", "default")


@dataclass(frozen=True)
class Work:
    name: str
    kind: str
    rows: int
    width: int


def cases(phase):
    if phase == "holdout":
        return [Work(f"{kind}_{r}x{w}", kind, r, w) for kind in STRUCTURES
                for r,w in ((96,192),(1024,768))]
    base = [Work(PRIMARY, "full", 320, 160), Work("two_sites_small", "minimal", 4, 160)]
    n = PLAN["discovery"]["iso_elements"]
    for r in PLAN["discovery"]["output_rows"]:
        name = "two_sites_large" if r == 288 else f"iso_{r}x{n//r}"
        base.append(Work(name, "minimal", r, n//r))
    return base + [Work("large_work", "minimal", *PLAN["discovery"]["large_positive_control"])]


def original(case):
    return case_for(PRIMARY) if case.kind == "full" else Case(
        case.name, "E", (case.rows,case.width), "identity", 0, "boundary")


def inputs(case):
    c = original(case)
    params = parameters(c)
    data = {seed:arguments(c, params, seed) for seed in SEEDS}
    if case.kind == "dual_input":
        for seed,(x,p) in data.items():
            y = np.random.default_rng(seed+17).standard_normal(x.shape,dtype=np.float32)
            data[seed] = (x, (*p, jnp.asarray(y)))
    return jax.block_until_ready(data)


def activations(case, x, p):
    if case.kind == "full":
        y,z = primal(original(case), x, p)
        return y, z[:2]
    a = x*p[0] + p[1]
    if case.kind == "minimal":
        b = jnp.maximum(a,jnp.float32(0))*p[2] + p[3]
        return x, (a,b)
    if case.kind == "residual_square":
        b = jnp.square(jnp.maximum(a,0)+x*jnp.float32(0.25))*p[2]+p[3]
        return jnp.maximum(b,0)+a*jnp.float32(0.125), (a,b)
    if case.kind == "dual_input":
        b = p[-1]*p[2]+p[3]
        return jnp.maximum(a,b), (a,b)
    if case.kind == "gated_product":
        b = jax.nn.sigmoid(a)*(x*p[2]+p[3])
        return x+b, (a,b)
    if case.kind == "center_residual":
        a = a-jnp.mean(a, axis=-1, keepdims=True)
        b = jnp.maximum(a,0)*p[2]+p[3]+x
        return jnp.maximum(b,0), (a,b)
    raise ValueError(case.kind)


def packed_counts(a, b):
    """Sum two nonnegative predicates in disjoint fields, with a checked carry bound."""
    width = a.shape[-1]
    shift = width.bit_length()
    if a.shape != b.shape or width < 1 or width+(width << shift) > 0xffffffff:
        raise ValueError("Packed row counts exceed the uint32 contract")
    x = (a > 0).astype(jnp.uint32)
    y = (b > 0).astype(jnp.uint32)
    total = jnp.sum(x + (y << jnp.uint32(shift)), axis=-1, dtype=jnp.uint32)
    return ((total & jnp.uint32((1 << shift)-1)).astype(jnp.int32),
            (total >> jnp.uint32(shift)).astype(jnp.int32))


def program(case, key):
    native_key = {"J":"row_J","off":"row_J","S":"row_S","global":"count_J","global_off":"count_J",
                  "global_S":"count_S","V0":"V0","diagnostic":"diagnostic"}.get(key)
    if (case.name == PRIMARY or case.name in MINI) and native_key:
        return original_function(original(case),native_key)

    def forward(x, p):
        y,z = activations(case,x,p)
        if key == "diagnostic":
            return y,z
        if key == "V0":
            return (y,)
        if key == "P":
            return (y,*packed_counts(*z))
        axis = "count" if key.startswith("global") else "row"
        return (y,*observe(z,axis,key not in ("S","global_S")))
    return forward


def compile_one(case, key, args):
    options = {"xla_disable_hlo_passes":"cpu-parallel-task-assigner"} if key in ("off","global_off") else {}
    return jax.jit(program(case,key)).lower(*args).compile(compiler_options=options)
