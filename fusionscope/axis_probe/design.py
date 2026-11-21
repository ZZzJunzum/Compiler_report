"""Frozen workloads, lanes and endpoints. Import JAX only inside pinned workers."""
from datetime import datetime
from zoneinfo import ZoneInfo

from fusionscope.common import ROOT, read, sha

PLAN = ROOT / "docs/plans/axis_probe_manifest.json"
EXECUTION = ROOT / "docs/plans/axis_probe_execution.json"
DESIGN = read(PLAN)
SEEDS = DESIGN["timed_input_seeds"]
ROUNDS = range(-DESIGN["warmup_rounds"], DESIGN["rounds_per_input"])
VARIANTS = DESIGN["variants"]
PRIMARY = DESIGN["primary_graph"]
PUBLISHED = ROOT / "results/axis_probe"
LOCAL = ROOT / "reproduction/axis_probe"
MINI = {"two_sites_large": (288, 160), "two_sites_small": (4, 160)}
TASK_GRAPHS = [PRIMARY, *MINI]
AXIS_GRAPHS = [item["name"] for item in DESIGN["graphs"]]
ALIASES = "ABCD"


def now():
    return datetime.now(ZoneInfo(read(EXECUTION)["calendar"]))


def design_lock():
    return {path.name: sha(path) for path in (PLAN, EXECUTION)}


def settings(panel):
    return ("on",) if panel == "axis" else ("on", "off")


def graphs(panel):
    return AXIS_GRAPHS if panel == "axis" else TASK_GRAPHS


def cells(panel):
    """(low-left, high-left, low-right, high-right) for (right diff - left diff)."""
    if panel == "axis":
        return (("count_S", "on"), ("count_J", "on"), ("row_S", "on"), ("row_J", "on"))
    return (("count_J", "on"), ("count_J", "off"), ("row_J", "on"), ("row_J", "off"))


def lanes(panel):
    primary = cells(panel)
    return [(v, opt, alias) for opt in settings(panel) for v in VARIANTS
            for alias in (ALIASES if (v, opt) in primary else "A")]


def case_for(name):
    from fusionscope.cpu_cost.workloads import Case, cases
    if name in MINI:
        return Case(name, "E", MINI[name], "identity", 0, "reduced")
    return next(c for c in cases("test") if c.name == name)


def function(case, variant):
    from fusionscope.cpu_cost.workloads import observe, program
    if case.name not in MINI:
        return program(case, variant)

    import jax.numpy as jnp

    def forward(x, params):
        if variant == "V0":
            return (x,)
        first = x * params[0] + params[1]
        second = jnp.maximum(first, jnp.float32(0)) * params[2] + params[3]
        if variant == "diagnostic":
            return x, (first, second)
        request, schedule = variant.split("_")
        return (x, *observe((first, second), request, schedule == "J"))

    return forward


def compiler_options(setting):
    return {"xla_disable_hlo_passes": "cpu-parallel-task-assigner"} if setting == "off" else {}
