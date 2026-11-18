"""Verify locked predictions, matched main-only controls and deployment on new graphs."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import geometric_mean, median
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from fusionscope.common import ROOT, dump, read, sha, table
from fusionscope.conv_selection import decide, ir_facts, skip_operand
from fusionscope.conv_transfer import (POLICIES, PROCESSES, ROUNDS, SEEDS, VARIANTS, cases,
                                       checked, frozen_design)
from fusionscope.hlo import Module
from fusionscope.paired_repeat import contrast, load_runs
from fusionscope.phase3.analysis import crossover
from paper.headroom_analysis import HORIZONS


def deployment(rows, policy, choice, evidence):
    if policy == "candidate_ir":
        build = median(r["builds"]["V2"] + r["builds"]["J1"] for r in rows)
        info = evidence["extraction_us"]
    else:
        build = median(r["builds"][choice] for r in rows)
        info = 0.
    return build + info + evidence["score_us"][policy]


def policy_metrics(c, data, ev):
    choices = ev["choices"]
    setup = {k: deployment(data, k, choices[k], ev) for k in POLICIES}
    times = {v: median(r["times"][v] for r in data) for v in ("V2","J1")}
    best = min(times,key=times.get)
    output = []
    for k in POLICIES:
        v = choices[k]
        for base in ("cap16","exception","source"):
            f = choices[base]
            d = contrast(data,v,f)
            extra = setup[k] - setup[base]
            nonworse, calls = crossover(extra, d["gain_us"])
            output.append(dict(case=c.name, group="target" if c.shape[0]<16 else "batch16",
                policy=k, reference=base, chosen=v, compared_with=f, changed=v!=f,
                gain_us=d["gain_us"], ratio=d["ratio"], floor_us=d["floor_us"], outcome=d["outcome"],
                fastest=best, regret=times[v]/times[best]-1, exact=times[v]==times[best],
                setup_us=setup[k], extra_setup_us=extra, nonworse=nonworse, crossover_calls=calls,
                **{f"net_{h}_us":h*d["gain_us"]-extra for h in HORIZONS}))
    return output


def analyze(root, output):
    design=checked(root)
    if json.loads(json.dumps(frozen_design(root))) != design:
        raise ValueError("Training or the independently frozen source policies changed.")
    pred=read(root/"predictions.json")
    if pred["design_sha256"]!=sha(root/"design.json") or pred["ir_sha256"]!=sha(root/"ir.zip"):
        raise ValueError("Transferred IR evidence or source policy lock differs.")
    lanes={c.name:VARIANTS for c in cases()}
    runs,checks,environment=load_runs(root/"repeat",lanes,sha(root/"predictions.json"),
                                     PROCESSES,ROUNDS,SEEDS)
    if environment!=pred["environment"]:
        raise ValueError("Confirmation and prediction environments differ.")
    policies,controls=[],[]
    with ZipFile(root/"ir.zip") as z:
        for c in cases():
            data=runs[c.name]
            ev=pred["cases"][c.name]
            facts={}
            for v in ("V2","J1"):
                hlo=z.read(f"{c.name}/{v}.hlo").decode()
                fp=Module(hlo).fingerprint()
                if any(r["fingerprints"][v]!=(("main",fp),) for r in data):
                    raise ValueError(f"The timed executable differs from inspected IR: {c.name}/{v}")
                facts[v]=ir_facts(hlo,ev["temp_bytes"][v],c.shape)
                if v=="V2" and skip_operand(hlo)!=ev["direct_skip"]:
                    raise ValueError(f"The skip operand trace changed on {c.name}")
            differences=[a-b for a,b in zip(facts["J1"],facts["V2"])]
            if differences!=ev["features"] or any(
                    decide(k,design["models"],c,differences)!=ev["choices"][k] for k in POLICIES):
                raise ValueError(f"The archived features do not reproduce {c.name}'s decisions")
            policies.extend(policy_metrics(c,data,ev))
            obs=contrast(data,"J1","V2")
            main=contrast(data,"B1","V0")
            whole_cost=[r["times"]["V2"]-r["times"]["V0"] for r in data]
            one_cost=[r["times"]["J1"]-r["times"]["B1"] for r in data]
            controls.append(dict(case=c.name,batch=c.shape[0],side=c.shape[1],channels=c.shape[-1],
                topology=c.top,**ev["direct_skip"],observed_ratio=obs["ratio"],observed_gain_us=obs["gain_us"],
                observed_outcome=obs["outcome"],main_only_ratio=main["ratio"],main_only_gain_us=main["gain_us"],
                whole_observer_us=median(whole_cost),one_image_observer_us=median(one_cost),
                observer_saving_us=median(a-b for a,b in zip(whole_cost,one_cost))))
    summary=[]
    for group in ("target","batch16","all"):
        for k in POLICIES:
            for ref in ("cap16","exception","source"):
                rows=[r for r in policies if r["policy"]==k and r["reference"]==ref and
                      (group=="all" or r["group"]==group)]
                counts=Counter(r["outcome"] for r in rows)
                summary.append(dict(group=group,policy=k,reference=ref,cases=len(rows),
                    ratio=geometric_mean(r["ratio"] for r in rows),changed=sum(r["changed"] for r in rows),
                    resolved_gains=counts["gain"],resolved_losses=counts["loss"],
                    losses_over_ten_percent=sum(r["outcome"]=="loss" and r["ratio"]>1.10 for r in rows),
                    exact_fastest=sum(r["exact"] for r in rows),median_regret=median(r["regret"] for r in rows),
                    median_extra_setup_us=median(r["extra_setup_us"] for r in rows),
                    **{f"net_positive_{h}":sum(r[f"net_{h}_us"]>0 for r in rows) for h in HORIZONS},
                    **{f"summed_net_{h}_us":sum(r[f"net_{h}_us"] for r in rows) for h in HORIZONS}))
    index={(r["group"],r["policy"],r["reference"]):r for r in summary}
    target=index["target","exception","cap16"]
    ir_source=index["target","candidate_ir","source"]
    ir_exception=index["target","candidate_ir","exception"]
    decision=dict(checks=checks, shapes=len({c.shape for c in cases()}), graphs=len(controls),
        matched_direct_skip=sum(r["origin"]==("input" if r["topology"]=="input" else "first_relu")
                                for r in controls),
        fingerprints_rechecked=2*len(controls)*PROCESSES,
        exception_passed=(target["ratio"]<=.95 and target["resolved_gains"]>=6 and
                          target["losses_over_ten_percent"]==0),
        ir_added_value_passed=(ir_source["resolved_gains"]>=1 and all(r["ratio"]<1 and
                              r["summed_net_1000_us"]>0 for r in (ir_source,ir_exception))),
        predictions_sha256=sha(root/"predictions.json"),
        repeat_archives={p.name:sha(p) for p in sorted((root/"repeat").glob("p*.zip"))})
    table(output/"decisions.csv",policies)
    table(output/"main_only.csv",controls)
    table(output/"policies.csv",summary)
    dump(output/"verdict.json",decision)
    return decision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"results/headroom/conv_transfer")
    parser.add_argument("--write",action="store_true")
    args=parser.parse_args()
    published=args.data/"analysis"
    if args.write:
        result=analyze(args.data,published)
    else:
        with TemporaryDirectory() as tmp:
            out=Path(tmp)
            result=analyze(args.data,out)
            for p in out.iterdir():
                if p.read_bytes()!=(published/p.name).read_bytes():
                    raise ValueError(f"Published result differs from raw confirmation: {p.name}")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
