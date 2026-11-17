"""Registered development, mechanism and held-out workload grids."""
from dataclasses import asdict, dataclass
from math import prod

from fusionscope.common import conditions, describe


@dataclass(frozen=True)
class Case:
    name: str
    ops: str
    shape: tuple
    top: str
    group: str
    size: int

    @property
    def tiles(self):
        grid = (1, 2, 4, 8, 16) if self.ops == "CC" else (1, 4, 16, 64, 256)
        return sorted({t for t in (*grid, self.shape[0]) if self.shape[0] % t == 0})

    @property
    def options(self):
        return ["V0", "V1", "V2", "V3", *[f"{s}{t}" for t in self.tiles for s in ("S", "J")]]

    @property
    def fixed_tile(self):
        return "S" + str(max(t for t in self.tiles if t <= 16))

    def manifest(self):
        return {**asdict(self), "candidates": self.options, "elements": prod(self.shape)}


def development():
    cases = []
    for name in conditions():
        family, shape, top = describe(name)
        cases.append(Case(name, family * 2, shape, top, "development", int(name[1])))
    return cases


def heldout():
    cases = []
    for family in "EMNC":
        shapes = [(2, 24, 24, 24), (4, 40, 40, 32), (8, 48, 48, 48)] if family == "C" else [
            (32, 192), (128, 320), (512, 640)]
        for i, shape in enumerate(shapes):
            for top in ("chain", "branch"):
                cases.append(Case(f"{family}H{i}_{top}", family * 2, shape, top, "primary", i))
    for ops in ("NM", "MN"):
        for i, shape in enumerate(((32, 320), (512, 192))):
            cases.append(Case(f"{ops}H{i}_chain", ops, shape, "chain", "transfer", i))
    return cases


MECHANISM = {"E3_branch": (16, 64), "N2_branch": (4, 16),
             "M3_chain": (16, 64), "C3_chain": (2, 8)}
REPEATS = {"mechanism": 5, "development": 3, "discovery": 2, "confirmation": 5}
ROUNDS = {"mechanism": 30, "development": 20, "discovery": 8, "confirmation": 30}


def mechanism_options(case):
    return ["V0", "V1", "V2", *[f"{s}{t}" for t in MECHANISM[case.name] for s in ("B", "S", "J")]]
