"""Independent pivot of raw latency/build samples for the paper's principal comparisons."""
import json
from math import isclose
import re
import numpy as np
from zipfile import ZipFile
from fusionscope.common import ROOT,records
from fusionscope.axis_probe.analysis import rows

ROOTS = ROOT/"results"


def raw_cells(base,resource,graph,left,right):
    by_process = []
    for i in range(5):
        with ZipFile(base/resource/f"p{i}.zip") as z:
            raw = [r for r in rows(z,"timings.csv") if r["graph"]==graph and r["key"] in (left,right)
                   and int(r["round"])>=0 and r["alias"]=="A"]
        seeds = sorted({r["seed"] for r in raw})
        assert len(seeds)==3
        def value(key,seed):
            samples = [int(r["ns"])/1000 for r in raw if (r["key"],r["seed"])==(key,seed)]
            assert len(samples)==12
            return np.median(samples)
        by_process.append(np.mean([value(left,s)-value(right,s) for s in seeds]))
    return np.median(by_process)


def main():
    base = ROOTS/"partition_boundary/discovery"
    for resource,expected in zip(("one","four","default"),(74.47,7.18,-23.73),strict=True):
        assert abs(raw_cells(base,resource,"t_E1_chain","J","S")-expected)<.0051
    for graph,expected in (("two_sites_large",32.20),("large_work",-852.63)):
        assert abs(raw_cells(base,"default",graph,"J","off")-expected)<.0051
    scoped = ROOTS/"scoped_assignment/discovery"
    for other,expected in (("J",-273.30),("split_on",546.63)):
        assert abs(raw_cells(scoped,"default","large_work","split_off",other)-expected)<.0051
    # Compare all emitted object sets across all three discovery affinity configurations.
    with ZipFile(base/"default_ir.zip") as ref:
        from fusionscope.partition.workloads import cases
        for case in cases("discovery"):
            if case.kind=="minimal":
                hlo = ref.read(f"{case.name}/J/cpu_after_optimizations.txt").decode()
                entry = hlo.split("ENTRY %")[-1]
                copied = re.search(r"(%[\w.-]+) = f32\[([0-9]+),([0-9]+)\]\{1,0\} copy\(%x\.1\)",entry)
                assert copied and (int(copied[2]),int(copied[3]))==(case.rows,case.width)
                assert f"tuple({copied[1]}," in entry
        for res in ("one","four"):
            with ZipFile(base/(res+"_ir.zip")) as z:
                indices = [n for n in ref.namelist() if n.endswith("/index.json")]
                assert len(indices)==88
                for n in indices:
                    a,b = (json.loads(q.read(n)) for q in (ref,z))
                    assert all(a[k]==b[k] for k in ("named_schedule","fingerprint","temp_bytes","objects"))
    for panel in ("partition_boundary","scoped_assignment"):
        directory = ROOTS/panel/"holdout/analysis"
        charges = records(directory/"charges.csv")
        assert len(charges)==24*5*4*6
        for r in charges:
            assert isclose(float(r["total_us"]),float(r["build_us"])+float(r["selection_us"])
                           +int(r["n"])*float(r["call_us"]),rel_tol=1e-13)
        verdict = json.loads((directory/"verdict.json").read_text(encoding="utf8"))
        subset = {(r["resource"],r["graph"],r["process"],r["policy"]):float(r["total_us"])
                  for r in charges if r["n"]=="1000"}
        ratios = [np.median([subset[resource,g,str(i),"shape_guard"]/subset[resource,g,str(i),verdict["strongest_simple"]]
                             for i in range(5)]) for resource,g in sorted({(r,g) for r,g,_,_ in subset})]
        assert len(ratios)==24 and isclose(float(np.exp(np.mean(np.log(ratios)))),verdict["ratio"],rel_tol=1e-12)
    print("Seven manuscript effects from raw rounds, 88 compiler artifacts at three affinities and two held-out cost endpoints verified.")


if __name__=="__main__":
    main()
