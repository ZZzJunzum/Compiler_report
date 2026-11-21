"""Check the dated manuscript numbers independently from chronological timing rows."""
from itertools import product
import json
from zipfile import ZipFile

import numpy as np
from fusionscope.common import records
from fusionscope.axis_probe.analysis import require, rows
from fusionscope.axis_probe.design import PRIMARY, PUBLISHED, SEEDS


def check_numbers(root, session, graph, four, expected):
    effects = []
    for p in range(5):
        with ZipFile(root/"sessions"/session/f"p{p}.zip") as archive:
            group = {(s, v, opt): [] for s in SEEDS for v, opt in four}
            for row in rows(archive, "timings.csv"):
                key = (int(row["seed"]), row["variant"], row["setting"])
                if row["graph"] == graph and row["alias"] == "A" and int(row["round"]) >= 0 and key in group:
                    group[key].append(int(row["elapsed_ns"]))
            require(all(len(values) == 20 for values in group.values()), (session, p))
            values = np.array([[group[s,v,opt] for v,opt in four] for s in SEEDS])
            m = np.median(values, axis=2)/1000
            per_input = (m[:,3] - m[:,2]) - (m[:,1] - m[:,0])
            effects.append(float(per_input.mean()))
    actual = float(np.median(effects))
    require(abs(actual-expected) < 0.0051, (session, graph, actual, expected))


def objects(root, graph, variant, mode):
    with ZipFile(root/"ir/cpu1"/graph/f"{variant}_{mode}.zip") as z:
        return {n:z.read(n) for n in z.namelist() if n.endswith(".o")}


def main():
    glob_row = list(product(("count_S","count_J","row_S","row_J"), ("on",)))
    intervention = list(product(("count_J","row_J"), ("on","off")))
    for root, session, value in ((PUBLISHED,"axis_1120_a",115.80), (PUBLISHED,"axis_1120_b",110.03),
                                 (PUBLISHED/"jax082","axis_1120_082",88.45),
                                 (PUBLISHED,"axis_20261121",348.10)):
        check_numbers(root, session, PRIMARY, glob_row, value)
    for root, session, values in ((PUBLISHED,"task_1120_one",(-63.62,-115.45,0.17)),
                                  (PUBLISHED,"task_1120_four",(-22.20,-37.67,0.15)),
                                  (PUBLISHED,"task_20261121_one",(-315.23,-279.72,-8.68)),
                                  (PUBLISHED/"jax082","task_1120_082",(-64.03,-67.88,1.03))):
        for graph, value in zip((PRIMARY,"two_sites_large","two_sites_small"), values, strict=True):
            check_numbers(root, session, graph, intervention, value)
    for root in (PUBLISHED, PUBLISHED/"jax082"):
        for graph, variant in product(("two_sites_small","two_sites_large"), ("row_J","count_J")):
            a,b = (objects(root, graph, variant, opt) for opt in ("on","off"))
            require(a.keys() == b.keys() and len(a) == 1, (graph,variant))
            require((a != b) == (graph == "two_sites_large" and variant == "row_J"), (graph,variant))
        fixed = json.loads((root/"analysis/fixed_sites.json").read_text(encoding="utf8"))
        for r in fixed:
            if r["local_date"] not in ("2026-11-20","2026-11-21"):
                continue
            require(r["exact"] == (r["input"] != "zero_nextafter"), r)
            if r["input"] == "zero_nextafter" and r["variant"].startswith("count"):
                require(r["actual"] == [0,0] and r["ieee_reference"] == [2,2], r)
    mnist = records(PUBLISHED/"mnist/contrasts.csv")
    for r, value in zip(mnist, (-3.27,-2.80,-2.70), strict=True):
        require(abs(float(r["row_diff_us"])-value) < .0051 and r["row_diff_clears"] == "True", r)
    require(all(r["valid"] == "True" for r in records(PUBLISHED/"mnist/checks.csv")), "MNIST semantics")
    print("Verified 16 paired endpoints across dates and releases, object-code controls, subnormal counts and trained fragments.")


if __name__ == "__main__":
    main()
