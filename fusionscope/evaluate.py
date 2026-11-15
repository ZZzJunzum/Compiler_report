"""Evaluate locked predictions only against fresh confirmation processes."""
import math
from statistics import median

import numpy as np

from .common import DOUBLES, MODES, SPEC, conditions, describe, read, records, table
from .model import verify_lock
from .stats import interactions, ranking, summarize


def paired_tables(root):
    dst=root/"report/tables"
    all_stats={}
    latency,per_process,inter,iper,noise,inventory=[],[],[],[],[],[]
    for phase,cases in (("matrix",conditions()),("confirm",conditions()),("external",["R1","R2"])):
        for case in cases:
            family,shape,top=describe(case)
            s=summarize(root,phase,case)
            all_stats[phase,case]=s
            order,tie=ranking(s)
            noise.append(dict(phase=phase,case=case,aa_p95_us=s["aa_us"],aa_pseudo_p95_us=s["aa_i_us"],
                              best=order[0] if order else "",unresolved=tie))
            for mode in MODES:
                tag=dict(phase=phase,case=case,mode=mode)
                vals=[p[mode] for p in s["processes"]]
                latency.append({**tag,"us":s["times"][mode],"paired_delta_us":s["delta"][mode],
                                "paired_ratio":s["ratios"][mode],"min_us":min(vals),"max_us":max(vals),
                                "n_process":len(vals),"valid":s["valid"][mode]})
                if phase!="confirm":
                    inventory.append(dict(config_id=f"{read(root/phase/'env_p0.json')['backend']}/{case}/{mode}",
                                          family=family,shape=str(shape),topology=top,mode=mode,
                                          inputs=str(SPEC["input_seeds"]),ir_folder=f"{phase}/ir/{case}/{mode}",
                                          env_file=f"{phase}/env_p0.json",processes=len(vals),valid=s["valid"][mode]))
                for i,p in enumerate(s["processes"]):
                    per_process.append({**tag,"process":i,"us":p[mode],"delta_us":p[mode]-p["base"],"ratio":p[mode]/p["base"]})
            for row in interactions(s):
                inter.append(dict(phase=phase,case=case,**row))
                mode=row["mode"]
                u,v=mode.split("_")
                for i,p in enumerate(s["processes"]):
                    pred=p[f"u_{u}"]+p[f"v_{v}"]-p["base"]
                    iper.append(dict(phase=phase,case=case,mode=mode,process=i,additive_us=pred,
                                     actual_us=p[mode],interaction_us=p[mode]-pred))
    for name,rows in (("configs",inventory),("latencies",latency),("process_medians",per_process),
                      ("interactions",inter),("process_interactions",iper),("noise",noise)):
        table(dst/f"{name}.csv",rows)
    return all_stats,inter


def policy_tables(root,data):
    lock=verify_lock(root)
    ext=read(root/"model/external_singles_lock.json")["predictions"]
    decisions,preds,costs=[],[],[]
    for case in conditions()+["R1","R2"]:
        external=case.startswith("R")
        phase="external" if external else "confirm"
        s=data[phase,case]
        methods=dict(lock["predictions"][case])
        if external:
            methods["single_add"]=ext[case]
        ranking_,tie=ranking(s)
        if not ranking_:
            raise ValueError(f"No valid candidate: {case}")
        best=ranking_[0]
        methods["oracle"]={"chosen":best,"log_ratios":{},"used":"evaluation_only","fallback":False}
        available={"fixed":"training_or_policy","measured_singles":"independent_single_trials",
                   "evaluation_only":"post_measurement"}
        for method,row in methods.items():
            chosen=row["chosen"]
            valid=s["valid"][chosen]
            gap=s["times"][chosen]-s["times"][best] if valid else None
            family=describe(case)[0]
            decisions.append(dict(case=case,family=family,method=method,chosen=chosen,oracle=best,
                             regret=gap/s["times"][best] if valid else None,gap_us=gap,
                             valid=valid,valid_candidates=len(ranking_),fallback=row["fallback"],
                             top_two_unresolved=tie,chosen_within_aa=valid and gap<=s["aa_us"],
                             aa_p95_us=s["aa_us"],available=available.get(row["used"],"baseline_only")))
            for m,logr in row["log_ratios"].items():
                ratio=math.exp(logr)
                abs_err=median(abs(p["base"]*ratio-p[m]) for p in s["processes"])
                preds.append(dict(case=case,family=family,method=method,mode=m,predicted_ratio=ratio,
                             measured_paired_ratio=s["ratios"][m],
                             abs_error_us=abs_err,abs_log_error=abs(logr-math.log(s["ratios"][m])),
                             valid=s["valid"][m],fallback=row["fallback"]))
        facts=read(root/"baselines"/case/"features.json")
        build=facts["baseline_lower_compile_ns"]/1000
        feat=facts["feature_extraction_ns"]/1000
        saved=s["times"]["post_post"]-s["times"][methods["M2"]["chosen"]]
        full_build,trial=[],[]
        for i in range(len(s["processes"])):
            raw=root/phase/"raw"/f"p{i}"/case
            builds=records(raw/"builds.csv")
            full_build.append(sum((int(b["lower_ns"])+int(b["compile_ns"]))/1000 for b in builds if b["mode"] in DOUBLES))
            trial.append(sum(int(t["elapsed_ns"])/1000 for t in records(raw/"times.csv") if t["mode"] in DOUBLES))
        costs.append(dict(case=case,one_baseline_lower_compile_us=build,baseline_feature_extraction_us=feat,
                          four_candidate_lower_compile_us=median(full_build),four_candidate_20_rounds_us=median(trial),
                          M2_saved_vs_post_us=saved,break_even_calls=math.ceil((build+feat)/saved) if saved>0 else None,
                          saving_above_aa=saved>s["aa_us"]))
    methods=list(dict.fromkeys(row["method"] for row in decisions))
    groups={"matrix":conditions(),"external":["R1","R2"]}
    groups.update({f:[c for c in conditions() if c[0]==f] for f in "EMNC"})
    summaries=[]
    for group,cases in groups.items():
        # All decision policies compared on exactly the same semantically valid requests.
        common={c for c in cases if all(d["valid"] for d in decisions if d["case"]==c)}
        for method in methods:
            sel=[r for r in decisions if r["case"] in common and r["method"]==method]
            err=[r for r in preds if r["case"] in common and r["method"]==method and r["valid"]]
            if not sel:
                continue
            reg=[r["regret"] for r in sel]
            summaries.append(dict(group=group,method=method,n=len(reg),requests=len(cases),
                             median_regret=median(reg),p90_regret=float(np.quantile(reg,.9)),max_regret=max(reg),
                             median_gap_us=median(r["gap_us"] for r in sel),correct=sum(r["chosen"]==r["oracle"] for r in sel),
                             within_aa=sum(r["chosen_within_aa"] for r in sel),ties=sum(r["top_two_unresolved"] for r in sel),
                             fallback=sum(r["fallback"] for r in sel),prediction_rows=len(err),
                             mae_us=float(np.mean([r["abs_error_us"] for r in err])) if err else None,
                             mae_log_ratio=float(np.mean([r["abs_log_error"] for r in err])) if err else None))
    dst=root/"report/tables"
    for name,rows in (("decisions",decisions),("predictions",preds),("policy_summary",summaries),("costs",costs)):
        table(dst/f"{name}.csv",rows)
    return decisions,summaries,costs
