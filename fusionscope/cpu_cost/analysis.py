"""Recompute all comparisons from independent processes and replay the blind locks."""
from collections import Counter, defaultdict
from itertools import combinations
import json
from math import exp, log
from pathlib import Path
from statistics import median
import tempfile
from zipfile import ZipFile

import numpy as np

from fusionscope.common import dump, read, sha, table
from fusionscope.paired_repeat import contrast
from fusionscope.phase2.experiment import zip_rows
from .experiment import PROCESSES, ROUNDS, SEEDS
from .model import MODELS, archive_features, estimates
from .workloads import CONTRACTS, cases, variants


def test_data(out, graphs):
    data = defaultdict(list)
    expected = {(c.name,n) for c in graphs for n in variants(c)}
    checks, invalid, env, inspected = 0, set(), None, read(out/"gate.json")
    test_lock = sha(out/"gate.json")
    matching = 0
    for i in range(PROCESSES["test"]):
        path = out/f"test/p{i}.zip"
        with ZipFile(path) as z:
            meta = json.loads(z.read("environment.json"))
            assert z.testzip() is None and meta["process"] == i and meta["lock"] == test_lock
            assert sorted(meta["case_order"]) == sorted(c.name for c in graphs)
            assert meta["seeds"] == list(SEEDS["test"]) and meta["rounds"] == ROUNDS["test"] and meta["warmups"] == 6
            if env is None:
                env = meta["environment"]
            assert env == meta["environment"]
        flags = {}
        for r in zip_rows(path,"correctness.csv"):
            k = r["case"],r["variant"],int(r["seed"])
            assert k not in flags
            flags[k] = r["valid"] == "True"
            if not flags[k]:
                invalid.add(k[:2])
        assert flags.keys() == {(c,n,s) for c,n in expected for s in SEEDS["test"]}
        checks += len(flags)
        timings = defaultdict(dict)
        for r in zip_rows(path,"timings.csv"):
            k = r["case"],r["block"],r["variant"]
            j = int(r["round"])
            assert j not in timings[k] and int(r["process"]) == i and int(r["elapsed_ns"]) > 0
            timings[k][j] = int(r["elapsed_ns"])/1000
        def sample(c,b,n):
            v = timings.pop((c,b,n))
            assert set(v) == set(range(-meta["warmups"],meta["rounds"]))
            return median(v[j] for j in range(meta["rounds"]))
        builds = {(r["case"],r["variant"]):r for r in zip_rows(path,"builds.csv")}
        assert builds.keys() == expected and len(zip_rows(path,"builds.csv")) == len(expected)
        for c in graphs:
            row = dict(times={}, builds={}, aa={}, fingerprints={}, temps={})
            for n in variants(c):
                b = builds[c.name,n]
                assert b["component"]=="main" and int(b["process"])==i and int(b["build_wall_ns"])>0
                row["builds"][n] = int(b["build_wall_ns"])/1000
                row["fingerprints"][n] = b["fingerprint"]
                row["temps"][n] = int(b["temp_bytes"])
                row["times"][n] = sample(c.name,"candidate",n)
                alias = [sample(c.name,"AA_"+n,k) for k in "ABCD"]
                row["aa"][n] = [abs(a-b) for a,b in combinations(alias,2)]
                gk = c.name+"/"+n.removesuffix("_S")
                if n.endswith("_S") and gk in inspected["choices"]:
                    assert b["fingerprint"] == inspected["choices"][gk]["fingerprint"], (gk,i)
                    matching += 1
            data[c.name].append(row)
        assert not timings
    return data, invalid, checks, env, matching


def geomean(values):
    return exp(float(np.mean(np.log(values))))


def summarize_metrics(all_rows):
    metrics, increments = [], {}
    for group in ("new_shape","new_architecture"):
        rows = [r for r in all_rows if r["group"] == group and r["variant"] != "V0"]
        graph_errors = defaultdict(list)
        for mode in MODELS:
            for endpoint in ("runtime_us","build_us","H1_us","H1000_us"):
                err = np.array([log(r[f"{mode}_{endpoint}"]/r[endpoint]) for r in rows])
                q = np.exp(abs(err))
                med, p90 = float(np.median(q)), float(np.percentile(q,90))
                metrics.append(dict(group=group, endpoint=endpoint, model=mode, configurations=len(rows),
                                    median_q=med, p90_q=p90, mae_us=float(np.mean([
                                        abs(r[f"{mode}_{endpoint}"]-r[endpoint]) for r in rows])),
                                    median_pred_over_measured=exp(float(np.median(err))),
                                    accuracy_target=(med<=1.30 and p90<=2.) if endpoint=="runtime_us" else ""))
        for r in rows:
            graph_errors[r["case"]].append([abs(log(r[m+"_runtime_us"]/r["runtime_us"])) for m in ("source","combined")])
        graphs = {k:np.mean(v,axis=0) for k,v in graph_errors.items()}
        cluster = defaultdict(list)
        for c in cases("test"):
            if c.name in graphs:
                cluster[c.family,c.shape].append(graphs[c.name])
        # Equal graph weight; each cluster has the same number of readouts within each cohort.
        arr = np.array([np.mean(v,axis=0) for v in cluster.values()])
        means = arr.mean(axis=0)
        rng = np.random.default_rng(1871)
        draws = arr[rng.integers(len(arr),size=(10000,len(arr)))].mean(axis=1)
        interval = np.percentile(draws[:,1]-draws[:,0], [2.5,97.5])
        relative = float(1-means[1]/means[0])
        increments[group] = dict(clusters=len(arr), source=float(means[0]), combined=float(means[1]),
                                 difference=float(means[1]-means[0]), interval=interval.tolist(),
                                 relative_reduction=relative, passes=bool(relative>=.10 and interval[1]<0))
    return metrics, increments


def derive(out, destination):
    graphs = cases("test")
    frozen, gate = read(out/"predictions.json"), read(out/"gate.json")
    assert gate["predictions"] == sha(out/"predictions.json") and gate["listings"] == sha(out/"test/separate_hlo.zip")
    data, invalid, checks, env, matches = test_data(out,graphs)
    cases_by_name = {c.name:c for c in graphs}
    excluded = set(invalid)
    for c,n in invalid:
        if n == "V0":
            excluded.update((c,k) for k in variants(cases_by_name[c]))
        else:
            contract = n.split("_")[0]
            excluded.update((c,k) for k in variants(cases_by_name[c]) if k.startswith(contract+"_"))
    measured, pairs, boundaries = [], [], []
    for c in graphs:
        process = data[c.name]
        for n in variants(c):
            if (c.name,n) in excluded:
                continue
            def typical(kind):
                return median(r[kind][n] for r in process)
            actual = dict(runtime_us=typical("times"), build_us=typical("builds"))
            actual.update({f"H{h}_us":median(r["builds"][n]+h*r["times"][n] for r in process) for h in (1,1000)})
            inc = median(r["times"][n]-r["times"]["V0"] for r in process)
            row = dict(case=c.name,group=c.group,family=c.family,shape="x".join(map(str,c.shape)),variant=n,
                       **actual,delta_to_main_us=inc,temp_bytes=typical("temps"))
            for model,values in frozen["predictions"][c.name+"/"+n].items():
                row.update({model+"_"+k:v for k,v in values.items()})
                for h in (1,1000):
                    row[f"{model}_H{h}_us"] = values["build_us"] + h*values["runtime_us"]
            measured.append(row)
        for contract in CONTRACTS:
            key = c.name+"/"+contract
            s, j = contract+"_S", contract+"_J"
            if key not in gate["choices"]:
                boundaries.append(dict(case=c.name,group=c.group,contract=contract,implementation="separate",
                                       reason=("raw_output" if contract in ("full","sample") else
                                               "different_axes" if contract=="axes" else "different_shapes")))
                continue
            if (c.name,s) in excluded or (c.name,j) in excluded:
                continue
            g = gate["choices"][key]
            r = dict(case=c.name,group=c.group,family=c.family,contract=contract,**contrast(process,j,s),
                     gate=g["choice"],gate_hits=len(g["anchors"]),gate_inspect_us=g["inspect_ns"]/1000,
                     separate_delta_us=median(x["times"][s]-x["times"]["V0"] for x in process),
                     joint_delta_us=median(x["times"][j]-x["times"]["V0"] for x in process),
                     separate_temp_bytes=median(x["temps"][s] for x in process),
                     joint_temp_bytes=median(x["temps"][j] for x in process))
            for mode in ("source","stable","combined"):
                def prediction(n):
                    return frozen["predictions"][c.name+"/"+n][mode]["runtime_us"]
                r[mode] = "J" if prediction(j)<prediction(s) else "S"
            for h in (1,1000):
                def time_cost(x):
                    base = x["builds"][s] + g["inspect_ns"]/1000
                    if g["choice"] == "J":
                        base += x["builds"][j]
                    return base+h*x["times"][j if g["choice"]=="J" else s]
                r[f"gate_to_static_joint_H{h}"] = median(time_cost(x)/(x["builds"][j]+h*x["times"][j]) for x in process)
            pairs.append(r)
    metrics, ir_increment = summarize_metrics(measured)
    policies = []
    for group in ("all","new_shape","new_architecture"):
        subset = [r for r in pairs if group=="all" or r["group"]==group]
        for policy in ("separate","joint","gate","source","stable","combined"):
            ratios, regrets, wins, losses, choices = [], [], 0, 0, []
            for r in subset:
                choose_j = policy=="joint" or (policy!="separate" and r[policy]=="J")
                choices.append(choose_j)
                ratio = r["ratio"] if choose_j else 1.
                ratios.append(ratio)
                regrets.append(ratio/min(1.,r["ratio"]))
                wins += choose_j and r["outcome"]=="gain"
                losses += choose_j and r["outcome"]=="loss"
            policies.append(dict(group=group,policy=policy,pairs=len(subset),joint_choices=sum(choices),
                                 to_separate=geomean(ratios),to_best_of_two=geomean(regrets),
                                 resolved_wins_vs_separate=wins,resolved_losses_vs_separate=losses))
    by_contract = []
    for key in CONTRACTS:
        q = [r for r in pairs if r["contract"]==key]
        if q:
            counts = Counter(r["outcome"] for r in q)
            by_contract.append(dict(contract=key,pairs=len(q),gain=counts["gain"],loss=counts["loss"],
                                    unresolved=counts["unresolved"],joint_to_separate=geomean([r["ratio"] for r in q])))
    for name,rows in [("measurements",measured),("rewrite_pairs",pairs),("boundaries",boundaries),
                      ("accuracy",metrics),("policies",policies),("contracts",by_contract)]:
        table(destination/(name+".csv"),rows)
    changed = [r for r in pairs if r["source"]!=r["combined"]]
    summary = dict(development_configurations=frozen["development_valid"],development_rejected=frozen["development_rejected"],
                   graphs=len(graphs), configurations=sum(len(variants(c)) for c in graphs),
                   checks=checks,invalid=sorted(invalid),excluded=sorted(excluded), fingerprint_matches=matches,
                   legal_pairs=len(pairs),boundary_contracts=len(boundaries), outcomes=dict(Counter(r["outcome"] for r in pairs)),
                   ir_increment=ir_increment,changed_decisions=len(changed),
                   changed_resolved_helped=sum((r["combined"]=="J" and r["outcome"]=="gain") or
                                               (r["combined"]=="S" and r["outcome"]=="loss") for r in changed),
                   changed_resolved_hurt=sum((r["combined"]=="S" and r["outcome"]=="gain") or
                                             (r["combined"]=="J" and r["outcome"]=="loss") for r in changed),
                   gate_to_static_joint={str(h):geomean([r[f"gate_to_static_joint_H{h}"] for r in pairs]) for h in (1,1000)},
                   environment=env)
    dump(destination/"summary.json", summary)
    return summary


def verify_ir(out):
    from jax._src.interpreters.mlir import make_ir_context
    from jaxlib.mlir import ir
    from .features import source, stable_features, trigger
    for split in ("development","test"):
        path = out/f"{split}/lowered.zip"
        stored = archive_features(path)
        with ZipFile(path) as archive, make_ir_context():
            for c in cases(split):
                for n in variants(c):
                    key = c.name+"/"+n
                    module = ir.Module.parse(archive.read(key+".mlir").decode())
                    assert stored[key]["stable"] == stable_features(module) and stored[key]["source"] == source(c,n)
    plans = read(out/"gate.json")["choices"]
    with ZipFile(out/"test/separate_hlo.zip") as z:
        for c in cases("test"):
            for k in CONTRACTS:
                key = c.name+"/"+k
                if key in plans:
                    anchors, fp = trigger(z.read(key+".hlo").decode(),c,k)
                    g = plans[key]
                    assert g["anchors"]==anchors and g["fingerprint"]==fp and g["choice"]==("J" if anchors else "S")


def report(out, verify):
    if not verify:
        print(json.dumps(derive(out,out/"analysis"), indent=2))
        return
    verify_ir(out)
    rebuilt, lock = estimates(out), read(out/"predictions.json")
    for k in ("protocol","development_features","test_features","training","development_valid","development_rejected"):
        assert rebuilt[k] == lock[k]
    for k,pred in rebuilt["predictions"].items():
        for model,value in pred.items():
            for endpoint,x in value.items():
                np.testing.assert_allclose(x, lock["predictions"][k][model][endpoint],rtol=1e-9,atol=1e-9)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        derive(out,target)
        assert {p.name:sha(p) for p in target.iterdir()} == {p.name:sha(p) for p in (out/"analysis").iterdir()}
    print("Replayed both prediction endpoints, every typed IR record and gate decision, controls and all published tables.")
