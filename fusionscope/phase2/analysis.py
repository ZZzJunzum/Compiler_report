"""Join process-paired measurements, structural evidence and frozen predictions."""
from collections import defaultdict
import math
from statistics import mean, median

import numpy as np

from fusionscope.common import conditions, describe, records, table
from .audit_ir import PLAN
from .cost_model import verify_lock
from .stats import cohort, metrics
from .structure import components


def make_tables(out):
    data={p:cohort(out,p) for p in ("screen","train","confirm","external")}
    stats,proc,correctness,memory=[],[],[],[]
    archived=set()
    for phase,cases in data.items():
        for case,c in cases.items():
            grouped=defaultdict(list)
            for check in c["checks"]:
                grouped[check["variant"],check["seed"]].append(check)
            for (variant,seed),rr in grouped.items():
                correctness.append(dict(phase=phase,case=case,variant=variant,seed=seed,
                    reference=" | ".join(sorted({r["reference_counts"] for r in rr})),
                    actual=" | ".join(sorted({r["actual_counts"] for r in rr})),
                    valid_processes=sum(r["valid"]=="True" for r in rr),processes=len(rr),
                    max_y_error=max(float(r["y_max_abs"]) for r in rr)))
            for variant in sorted(c["t"]):
                builds=[b for b in c["builds"] if b["variant"]==variant]
                compile_us=[]
                lower_us=[]
                for i,p in enumerate(c["p"]):
                    parts=[b for b in builds if b["process"]==i]
                    compile_us.append(sum(int(b["compile_ns"]) for b in parts)/1000)
                    lower_us.append(sum(int(b["lower_ns"]) for b in parts)/1000)
                    proc.append(dict(phase=phase,case=case,process=i,variant=variant,
                                     us=p[variant],V0_us=p["V0"],V1_us=p["V1"],
                                     speedup=p["V1"]/p[variant],observed_overhead_us=p[variant]-p["V0"]))
                stats.append(dict(phase=phase,case=case,variant=variant,n_process=len(c["p"]),
                                  **metrics(c,variant),aa_v0_us=float(np.quantile(c["aa0"],.95)),
                                  aa_v1_us=float(np.quantile(c["aa1"],.95)),
                                  compile_us=median(compile_us),lower_us=median(lower_us)))
                for part in sorted({b["component"] for b in builds}):
                    group=[b for b in builds if b["component"]==part]
                    mem={k:int(median(int(b[k]) for b in group)) for k in
                         ("argument_bytes","output_bytes","temp_bytes","alias_bytes")}
                    planned=sum(mem[k] for k in ("argument_bytes","output_bytes","temp_bytes"))-mem["alias_bytes"]
                    memory.append(dict(phase=phase,case=case,variant=variant,component=part,
                                       planned_component_bytes=planned,**mem))
                archived.update((case,b["ir_archive"]) for b in builds)
    dst=out/"analysis"
    for name,rows in (("timings",stats),("process_medians",proc),("correctness",correctness),("compiler_memory",memory)):
        table(dst/(name+".csv"),rows)
    structure=components(out,archived)
    table(dst/"candidate_structure.csv",structure)
    return data,stats,structure,memory,correctness


def evaluate(out,data):
    predictions=verify_lock(out)["choices"]
    scores={(r["case"],r["variant"]):r for r in records(out/"model/predictions.csv")}
    decisions,errors=[],[]
    for case in conditions()+PLAN["external"]:
        external=case in PLAN["external"]
        c=data["external" if external else "confirm"][case]
        eligible=[v for v in sorted(c["t"]) if v!="V0" and v not in c["invalid"]]
        best=min(eligible,key=c["t"].get)
        choices=dict(predictions[case])
        if not external:
            discovery=data["train"][case]
            legal=[k for k in discovery["t"] if k.startswith("V4") and k not in discovery["invalid"]]
            if legal:
                choices["measured_tile"]=min(legal,key=discovery["t"].get)
        choices["oracle"]=best
        for name,v in choices.items():
            valid=v in eligible
            gap=c["t"][v]-c["t"][best] if valid else None
            decisions.append(dict(case=case,family=describe(case)[0],policy=name,chosen=v,best=best,
                                  chosen_us=c["t"][v],best_us=c["t"][best],gap_us=gap,
                                  regret=gap/c["t"][best] if valid else None,chosen_valid=valid,
                                  valid_candidates=len(eligible),available_candidates=len(c["t"])-1,
                                  within_aa=valid and gap<=c["noise"],aa_floor_us=c["noise"]))
        for v in sorted(c["t"]):
            if v=="V0":
                continue
            pred=scores[case,v]
            for name in ("analytic","ridge"):
                us=[float(pred["analytic_us"]) if name=="analytic" else float(pred["ridge_ratio"])*p["V0"] for p in c["p"]]
                abs_err=median(abs(u-p[v]) for u,p in zip(us,c["p"]))
                log_err=median(abs(math.log(u/p[v])) for u,p in zip(us,c["p"]))
                errors.append(dict(case=case,family=describe(case)[0],policy=name,variant=v,measured_us=c["t"][v],
                                   predicted_us=median(us),absolute_error_us=abs_err,abs_log_error=log_err,valid=v in eligible))
    summaries=[]
    groups={f:[c for c in conditions() if c[0]==f] for f in PLAN["families"]}
    groups.update(synthetic=conditions(),external=PLAN["external"])
    for family,cases in groups.items():
        comparable=[c for c in cases if all(r["chosen_valid"] for r in decisions if r["case"]==c)]
        for name in dict.fromkeys(d["policy"] for d in decisions):
            selected=[d for d in decisions if d["case"] in comparable and d["policy"]==name]
            if not selected:
                continue
            err=[e for e in errors if e["case"] in comparable and e["policy"]==name and e["valid"]]
            summaries.append(dict(group=family,policy=name,requests=len(cases),evaluated=len(selected),
                median_regret=median(d["regret"] for d in selected),p90_regret=float(np.quantile([d["regret"] for d in selected],.9)),
                max_regret=max(d["regret"] for d in selected),median_gap_us=median(d["gap_us"] for d in selected),
                within_aa=sum(d["within_aa"] for d in selected),oracle_hits=sum(d["chosen"]==d["best"] for d in selected),
                mae_us=mean(e["absolute_error_us"] for e in err) if err else None,
                mean_abs_log_error=mean(e["abs_log_error"] for e in err) if err else None,
                valid_predictions=len(err)))
    for name,rows in (("heldout_results",decisions),("prediction_errors",errors),("policy_summary",summaries)):
        table(out/"model"/(name+".csv"),rows)
    return decisions,errors,summaries


def screen_replay(out,discovery,confirm):
    """Compare the separate clean-clone screening cohort, when it accompanies the study."""
    folder=out/"screening/replay"
    if not folder.exists():
        return []
    replay=cohort(out,"screen",folder/"raw")
    rows=[]
    for case in PLAN["screen_cases"]:
        new=replay[case]
        c=confirm[case]
        for v in sorted(new["t"]):
            old=discovery[case]
            def digests(group):
                return {(r["component"],r["hlo_digest"]) for r in group["builds"] if r["variant"]==v}
            rows.append(dict(case=case,variant=v,same_compiled_components=digests(new)==digests(old),
                passed_checks=sum(r["valid"]=="True" for r in new["checks"] if r["variant"]==v),
                checks=sum(r["variant"]==v for r in new["checks"]),published_screen_speedup=metrics(old,v)["speedup"],
                independent_confirmation_speedup=metrics(c,v)["speedup"],replay_speedup=metrics(new,v)["speedup"],
                replay_latency_us=new["t"][v],replay_aa_floor_us=new["noise"],
                replay_resolved_gain=metrics(new,v)["resolved_gain"],replay_resolved_loss=metrics(new,v)["resolved_loss"]))
    table(folder/"comparison.csv",rows)
    return rows
