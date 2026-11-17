"""Evaluate frozen decisions on independent processes and account for information costs."""
from collections import Counter, defaultdict
from itertools import combinations
from math import ceil, exp, floor, log
from pathlib import Path
from statistics import median
from zipfile import ZipFile
import json

import numpy as np

from fusionscope.common import ROOT, dump, read, records, sha, table
from fusionscope.phase2.experiment import zip_rows
from .data import cohort, direction
from .design import MECHANISM, REPEATS, ROUNDS, development, heldout, mechanism_options
from .models import MODELS, encode, legacy_domain, ordered, verify_lock
from .structure import equivalence, mechanism


def geomean(values):
    return float(np.exp(np.mean(np.log(values))))


def bootstrap(cases, held, static, rows):
    cases = [c for c in cases if c.group=="primary"]
    ratios, regret, joint_regret = [], [], []
    failures = []
    for c in cases:
        d,choice = held[c.name],static[c.name]["static"]
        if choice in d["invalid"] or "V2" in d["invalid"]:
            failures.append(c.name)
        ratios.append([p[choice]/p["V2"] for p in d["processes"]])
        rr = rows[c.name]
        regret.append(rr["static"]["regret"])
        joint_regret.append(rr["V2"]["regret"])
    if failures:
        return dict(pass_primary=False,invalid_cases=failures,primary_ratio=None,interval=None)
    ratios = np.array(ratios).reshape(4,3,2,5)
    # Each sampled size carries both topologies. Repetitions are paired across schedules within a case.
    rng = np.random.default_rng(32041)
    boot = []
    for _ in range(20):
        size = rng.integers(3,size=(500,4,3))
        paired = ratios[np.arange(4)[None,:,None],size]
        samples = rng.integers(5,size=(*paired.shape[:-1],5))
        med = np.median(np.take_along_axis(paired,samples,axis=-1),axis=-1)
        boot.extend(np.exp(np.mean(np.log(med),axis=(1,2,3))))
    central = np.median(ratios,axis=-1)
    ratio = geomean(central)
    ci = np.quantile(boot,[.025,.975]).tolist()
    excess = float(np.quantile(regret,.9)-np.quantile(joint_regret,.9))
    leave = {str(i):geomean(np.delete(central,i,axis=1)) for i in range(3)}
    one = {c.ops[0]+str(c.size):geomean([v for k,v in zip(cases,central.flatten())
           if (k.ops,k.size)!=(c.ops,c.size)]) for c in cases}
    return dict(primary_ratio=ratio,interval=ci,p90_regret_difference=excess,pass_primary=bool(
                ratio<=.95 and ci[1]<1 and excess<=.02),leave_size_level_out=leave,leave_one_family_size_out=one,
                family_ratios=dict(zip("EMNC",[geomean(r) for r in central])),invalid_cases=[])


def build_cost(data, variant, p=None):
    vals = {int(r["process"]):int(r["build_wall_ns"])/1000 for r in data["builds"] if r["variant"]==variant}
    return vals[p] if p is not None else median(vals.values())


def crossover(extra, saving):
    if saving>0:
        return "from", max(0,ceil(extra/saving))
    if saving<0 and extra<=0:
        return "through",floor(-extra/(-saving))
    if saving==0 and extra<=0:
        return "all",0
    return "never",None


def inference_cost(out, cases, static, bases, calibration):
    from time import perf_counter_ns
    from fusionscope.phase2.cost_model import features, predict
    from .models import score
    path=out/"accounting/selector_us.json"
    if path.exists():
        return read(path)
    fitted=read(out/"static/models.json")["final"]
    legacy=read(ROOT/"results/phase2/model/model.json")["external"]["parameters"]
    times={}
    for c in cases:
        times[c.name]={}
        for k in MODELS:
            samples=[]
            for _ in range(11):
                begin=perf_counter_ns()
                if k.endswith("_old"):
                    vals={v:predict(legacy,features(c.name,"V4_t"+v[1:] if v.startswith("S") else v,
                                bases[c.name]["facts"],shape=c.shape))[int(k=="ridge_old")] for v in legacy_domain(c)}
                else:
                    vals={v:score(fitted[k],encode(c,v,bases[c.name]["facts"],calibration)) for v in ordered(c)}
                ranking=sorted(vals,key=vals.get)
                samples.append((perf_counter_ns()-begin)/1000)
            if ranking[0]!=static["choices"][c.name][k]:
                raise ValueError(f"Deployment does not reproduce {c.name}/{k}")
            times[c.name][k]=median(samples[1:])
    dump(path,times)
    return times


def analyze(out):
    from run import pin
    pin()
    static=verify_lock(out)
    decision=verify_lock(out,discovery=True)
    folder=out/"analysis"
    held=cohort(out,"confirmation")
    old=cohort(out,"development")
    mech=cohort(out,"mechanism")
    contrasts=mechanism(out,mech)
    ids,pairs=equivalence(out,held)
    calibration=read(out/"static/calibration.json")
    bases=read(out/"static/baseline.json")
    inference=inference_cost(out,heldout(),static,bases,calibration)
    prediction=defaultdict(dict)
    for row in records(out/"static/predictions.csv"):
        prediction[row["case"],row["model"]][row["variant"]]=(float(row["score"]),row["unit"])
    policies, errors, timings, budgets, crossovers, rankings = [],[],[],[],[],[]
    index = {}
    discovery={k:cohort(out,"discovery",k) for k in ("rank_alt","fixed_alt")}
    for c in heldout():
        d=held[c.name]
        t=d["us"]
        valid=[v for v in ordered(c) if v not in d["invalid"]]
        oracle=min(valid,key=t.get)
        common=[v for v in legacy_domain(c) if v in valid]
        legacy_best=min(common,key=t.get)
        choices=dict(static["choices"][c.name])
        choices["fixed_tile"]=choices.pop("fixed_alt")
        choices.pop("rank_alt")
        for k,ds in decision["choices"][c.name].items():
            choices["timed_"+k]=ds
        # Postcompile diagnostic: a source alternative in an equal-code class inherits its
        # best frozen score. Use the earliest representative in V2-first source order.
        cls=ids[c.name][choices["static"]]
        choices["candidate_ir"]=next(v for v in ordered(c) if ids[c.name][v]==cls)
        choices["oracle"]=oracle
        choices["V1"]="V1"
        index[c.name]={}
        for v in c.options:
            timings.append(dict(case=c.name,group=c.group,variant=v,valid=v not in d["invalid"],
                                median_us=t[v],process_us=str([p[v] for p in d["processes"]]),floor_us=d["floor"]))
        for policy,v in choices.items():
            ok=v in valid
            gap, resolved=direction(d,v,"V2")
            bestgap,_=direction(d,oracle,v)
            row=dict(case=c.name,group=c.group,family=c.ops,top=c.top,size=c.size,policy=policy,chosen=v,
                     valid=ok,oracle=oracle,legacy_oracle=legacy_best,us=t[v],oracle_us=t[oracle],floor_us=d["floor"],
                     gap_us=t[v]-t[oracle] if ok else None,regret=t[v]/t[oracle]-1 if ok else None,
                     common_regret=t[v]/t[legacy_best]-1 if v in common and ok else None,
                     paired_ratio_to_joint=median(p[v]/p["V2"] for p in d["processes"]),
                     paired_gain_us=gap,resolved_vs_joint=resolved,within_floor=ok and abs(bestgap)<=d["floor"])
            policies.append(row)
            index[c.name][policy]=row
        for model in MODELS:
            pred=prediction[c.name,model]
            valid_pred={v:s for v,(s,unit) in pred.items() if v in valid}
            for v,sc in valid_pred.items():
                rankings.append(dict(case=c.name,group=c.group,model=model,variant=v,score=sc,
                     score_unit=pred[v][1],latency_us=t[v],predicted_rank=1+sum(x<sc-1e-10 for x in valid_pred.values()),
                     measured_rank_on_domain=1+sum(t[k]<t[v] for k in valid_pred),
                     measured_rank_expanded=1+sum(t[k]<t[v] for k in valid)))
            corr=total=ties=0
            for a,b in combinations([v for v in pred if v in valid],2):
                gap,resolved=direction(d,a,b)
                if resolved:
                    diff=pred[b][0]-pred[a][0]
                    ties+=int(abs(diff)<1e-10)
                    corr+=.5 if abs(diff)<1e-10 else int(diff*gap>0)
                    total+=1
            abs_errors,log_errors=[],[]
            for v,(s,unit) in pred.items():
                if v in valid:
                    us=s if unit=="us" else t["V0"]*(s if unit=="ratio_to_V0" else exp(s))
                    abs_errors.append(abs(us-t[v]))
                    log_errors.append(abs(log(us/t[v])))
            errors.append(dict(case=c.name,group=c.group,model=model,resolvable_pairs=total,correct=corr,
                               predicted_ties=ties,mae_us=float(np.mean(abs_errors)),mean_abs_log_error=float(np.mean(log_errors)),
                               domain=len(pred),valid_domain=len(abs_errors)))
        base=bases[c.name]["extraction_ns"]/1000
        calibration_us=calibration["calibration_ns"]/1000
        per_case={}
        for policy,v in choices.items():
            compiled=build_cost(d,v)
            exploratory=feature=cal=0.
            scoring=inference[c.name][policy] if policy in MODELS else inference[c.name][
                static["ranker"] if policy=="timed_rank_alt" else static["primary"]]
            if policy.startswith("timed_"):
                k=policy.removeprefix("timed_")
                dd=discovery[k][c.name]
                alt=static["choices"][c.name][k]
                compiled=sum(build_cost(dd,u,p) for p in range(2) for u in ("V2",alt))
                exploratory=sum(dd["blocks"])
                if k=="rank_alt":
                    feature,cal=base+scoring,calibration_us
            elif policy=="oracle":
                # Full observed search. Its cost is a reference implementation charged at confirmation depth.
                compiled=sum(build_cost(d,k,p) for p in range(5) for k in c.options)
                exploratory=sum(d["blocks"])
            elif policy=="candidate_ir":
                compiled=sum(build_cost(d,k) for k in ordered(c))
                feature,cal=base+scoring,calibration_us
                exploratory=read(out/"accounting/ir_inspection_us.json")[c.name]
            elif policy in (*MODELS,"static"):
                feature=base+scoring
                cal=0. if policy.endswith("_old") else calibration_us
            per_case[policy]=(compiled+exploratory+feature,cal,t[v],v in valid)
            budgets.append(dict(case=c.name,group=c.group,policy=policy,chosen=v,valid=v in valid,
                         compiled_us=compiled,baseline_us=base if feature else 0,features_and_scoring_us=scoring if feature else 0,
                         exploration_us=exploratory,cold_calibration_us=cal,
                         reused_setup_us=compiled+exploratory+feature,cold_setup_us=compiled+exploratory+feature+cal,
                         **{f"{mode}_H{h}_us":compiled+exploratory+feature+charge+h*t[v] for
                            mode,charge in (("cold",cal),("reused",0)) for h in (1,10,100,1000)}))
        for policy,(setup,cal,us,ok) in per_case.items():
            comparators=["V2"]+(["timed_fixed_alt"] if policy=="timed_rank_alt" else [])
            for ref in comparators:
                rsetup,rcal,rus,rok=per_case[ref]
                for mode,extra in (("cold",setup+cal-rsetup-rcal),("reused",setup-rsetup)):
                    sign,h=crossover(extra,rus-us) if ok and rok else ("invalid",None)
                    crossovers.append(dict(case=c.name,group=c.group,policy=policy,against=ref,mode=mode,
                                           extra_setup_us=extra,per_call_saving_us=rus-us,nonworse=sign,horizon=h))
    for name,rows in (("decisions",policies),("prediction_quality",errors),("timings",timings),
                      ("setup",budgets),("break_even",crossovers),("rankings",rankings)):
        table(folder/(name+".csv"),rows)
    summary=[]
    for group in ("primary","transfer"):
        for policy in index[heldout()[0].name]:
            rows=[r for r in policies if r["group"]==group and r["policy"]==policy]
            vals=[r for r in rows if r["valid"]]
            if not vals:
                continue
            regrets=[r["regret"] for r in vals]
            summary.append(dict(group=group,policy=policy,valid=len(vals),total=len(rows),
                 ratio_to_joint=geomean([r["paired_ratio_to_joint"] for r in vals]),
                 median_regret=float(np.median(regrets)),p90_regret=float(np.quantile(regrets,.9)),
                 worst_regret=max(regrets),median_gap_us=median(r["gap_us"] for r in vals),
                 exact=sum(r["chosen"]==r["oracle"] for r in vals),within_floor=sum(r["within_floor"] for r in vals),
                 changed=sum(r["chosen"]!="V2" for r in vals),helped=sum(r["paired_gain_us"]>0 for r in vals),
                 harmed=sum(r["paired_gain_us"]<0 for r in vals),
                 resolved_helped=sum(r["resolved_vs_joint"] and r["paired_gain_us"]>0 for r in vals),
                 resolved_harmed=sum(r["resolved_vs_joint"] and r["paired_gain_us"]<0 for r in vals)))
    table(folder/"policy_summary.csv",summary)
    dominance=[]
    for c in development()+heldout():
        feat={v:encode(c,v,bases[c.name]["facts"],calibration)[1] for v in ordered(c)}
        for m,cols in (("volume",[0,1,2,4,5]),("throughput",[0,1,2,3,4,5])):
            dom=[v for v in feat if v!="V2" and all(feat["V2"][i]<=feat[v][i] for i in cols)]
            dominance.append(dict(case=c.name,group=c.group,model=m,alternatives=len(feat)-1,dominated=len(dom),
                                  all_joint_dominated=len(dom)==len(feat)-1))
    table(folder/"dominance.csv",dominance)
    result=bootstrap(heldout(),held,static["choices"],index)
    result.update(primary_model=static["primary"],shortlist_model=static["ranker"],
                  confirmation_invalid={c:sorted(d["invalid"]) for c,d in held.items() if d["invalid"]},
                  development_invalid={c:sorted(d["invalid"]) for c,d in old.items() if d["invalid"]},
                  mechanism_resolved=sum(r["resolved"] for r in contrasts),
                  matched_one_tile=sum(r["equal_hlo"] for r in pairs),
                  resolved_matching=sum(r["equal_hlo"] and r["resolved"] for r in pairs))
    dump(folder/"endpoint.json",result)
    from .figures import render
    render(out,policies,budgets)
    print(json.dumps(result,indent=2),flush=True)


def compare_replay(out):
    from fusionscope.hlo import Module
    a,b=cohort(out,"mechanism"),cohort(out/"replay","mechanism")
    def fingerprints(root, d, v):
        hashes=set()
        for archive,part in {(r["archive"],r["component"]) for r in d["builds"] if r["variant"]==v}:
            with ZipFile(root/archive) as z:
                hashes.add(Module(z.read(f"{v}/{part}/optimized_hlo.txt").decode()).fingerprint())
        return hashes
    rows=[]
    for c in a:
        for v in a[c]["us"]:
            first=fingerprints(out,a[c],v)
            second=fingerprints(out/"replay",b[c],v)
            rows.append(dict(case=c,variant=v,original_us=a[c]["us"][v],replay_us=b[c]["us"][v],
                             original_floor_us=a[c]["floor"],replay_floor_us=b[c]["floor"],
                             original_valid=v not in a[c]["invalid"],replay_valid=v not in b[c]["invalid"],
                             hlo_match=first==second,original_sha="|".join(sorted(first)),replay_sha="|".join(sorted(second))))
    path=out/"analysis/replay.csv"
    if path.exists():
        normalized=[{k:str(v) for k,v in r.items()} for r in rows]
        if records(path)!=normalized:
            raise ValueError("Stored replay comparison differs from the two raw cohorts")
    else:
        table(path,rows)
    return rows


def verify(out):
    from fusionscope.evidence import scheduled_text_digest
    from .experiment import signature
    checksums=out/"SHA256SUMS.json"
    def inventory():
        return {p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p!=checksums}
    if checksums.exists() and read(checksums)!=inventory():
        raise ValueError("Published evidence inventory differs")
    static=verify_lock(out)
    frozen=verify_lock(out,discovery=True)
    manifest=read(out/"static/manifest.json")
    expected=[c.manifest() for c in heldout()]
    if manifest["workloads"]!=json.loads(json.dumps(expected)):
        raise ValueError("Held-out grid differs from the static lock")
    if sum(len(c["candidates"]) for c in expected)!=368:
        raise ValueError("Wrong size of confirmation grid")
    stages=[(out,s,"") for s in ("mechanism","development","confirmation")]
    stages.extend((out,"discovery",p) for p in ("rank_alt","fixed_alt"))
    stages.append((out/"replay","mechanism",""))
    stats=[]
    ir_checked={}
    for root,stage,policy in stages:
        group=stage if root==out else "replay"
        cs=heldout() if stage in ("confirmation","discovery") else development()
        if stage=="mechanism":
            cs=[c for c in cs if c.name in MECHANISM]
        cases={c.name:c for c in cs}
        paths=sorted((root/stage/policy).glob("p*.zip"))
        if len(paths)!=REPEATS[stage]:
            raise ValueError(f"Missing {stage}/{policy} repeats")
        total=valid=failed=calls=0
        for i,path in enumerate(paths):
            with ZipFile(path) as z:
                meta=json.loads(z.read("environment.json"))
                if z.testzip() or meta["process"]!=i or meta["signature"]!=signature() or meta["environment"]!=static["environment"]:
                    raise ValueError(f"Invalid process provenance: {path}")
            options={c.name:(["V2",static["choices"][c.name][policy]] if policy else
                          mechanism_options(c) if stage=="mechanism" else c.options) for c in cs}
            blocks=defaultdict(list)
            for r in zip_rows(path,"timings.csv"):
                if int(r["elapsed_ns"])<=0:
                    raise ValueError("Nonpositive dispatch time")
                blocks[r["case"],r["block"],int(r["round"])].append((int(r["order"]),r["variant"]))
                calls+=1
            wanted={}
            for c in cs:
                lanes=("V2",) if policy else ("V0","V1","V2")
                for r in range(-6,ROUNDS[stage]):
                    wanted[c.name,"candidate",r]=options[c.name]
                    for lane in lanes:
                        wanted[c.name,"AA_"+lane,r]=list("ABCD")
            if set(blocks)!=set(wanted):
                raise ValueError(f"Missing measurement cell in {path}")
            for key,vals in blocks.items():
                if sorted(v for _,v in vals)!=sorted(wanted[key]) or sorted(i for i,_ in vals)!=list(range(len(vals))):
                    raise ValueError(f"Invalid block {key} in {path}")
            build_cells=set()
            for r in zip_rows(path,"builds.csv"):
                archive=root/r["archive"]
                key=archive,r["variant"],r["component"]
                build_cells.add((r["case"],r["variant"],r["component"]))
                if key not in ir_checked:
                    prefix=f"{r['variant']}/{r['component']}/"
                    with ZipFile(archive) as z:
                        hlo=z.read(prefix+"optimized_hlo.txt").decode()
                        if not z.read(prefix+"stablehlo.mlir"):
                            raise ValueError(f"Missing StableHLO in {key}")
                    ir_checked[key]=scheduled_text_digest(hlo)
                if ir_checked[key]!=r["digest"]:
                    raise ValueError(f"Changed scheduled HLO {key}")
            required={(case,v,part) for case,vs in options.items() for v in vs for part in
                      (("front","back") if v=="V3" else ("main",))}
            if build_cells!=required:
                raise ValueError(f"Incomplete candidate compilation in {path}")
            if not policy:
                checks=zip_rows(path,"correctness.csv")
                seeds=manifest["validation_seeds"] if stage=="confirmation" else [42,1729,2026]
                cells={(r["case"],r["variant"],int(r["seed"])) for r in checks}
                if cells!={(c,v,s) for c,vs in options.items() for v in vs for s in seeds} or len(cells)!=len(checks):
                    raise ValueError(f"Incomplete validation in {path}")
                for r in checks:
                    ok=r["valid"]=="True"
                    if ok!=(r["counts_valid"]=="True" and r["y_valid"]=="True"):
                        raise ValueError(f"Inconsistent correctness row in {path}")
                    if not ok and not (root/"failures"/f"{r['case']}-{r['seed']}.npz").exists():
                        raise ValueError(f"Missing failure input in {path}")
                    total+=1
                    valid+=int(ok)
                    failed+=int(not ok)
        stats.append(dict(stage=group,policy=policy,cases=len(cases),processes=len(paths),calls_with_warmup=calls,
                          checks=total,passed=valid,failed=failed))
    measured=cohort(out,"confirmation")
    by_case=defaultdict(dict)
    for r in records(out/"analysis/decisions.csv"):
        r["regret"]=float(r["regret"]) if r["regret"] else None
        by_case[r["case"]][r["policy"]]=r
    endpoint=bootstrap(heldout(),measured,static["choices"],by_case)
    saved=read(out/"analysis/endpoint.json")
    for k,v in endpoint.items():
        if saved[k]!=v:
            raise ValueError(f"Primary endpoint changed: {k}")
    for k in ("rank_alt","fixed_alt"):
        discovery=cohort(out,"discovery",k)
        for c in heldout():
            alternative=static["choices"][c.name][k]
            gain,resolved=direction(discovery[c.name],alternative,"V2")
            expected=alternative if resolved and gain>0 else "V2"
            if expected!=frozen["choices"][c.name][k]:
                raise ValueError("Discovery decision differs from the paired two-process evidence")
            if by_case[c.name]["timed_"+k]["chosen"]!=frozen["choices"][c.name][k]:
                raise ValueError("Published choice differs from discovery lock")
    replay=compare_replay(out)
    # Write verification outputs only on first publication. On later runs verify the archived bytes.
    verification=dict(cohorts=stats,unique_ir_components=len(ir_checked),replay_configurations=len(replay),
                      replay_matching=sum(r["hlo_match"] for r in replay),
                      primary=endpoint["pass_primary"],profile=read(out/"profile/coverage.json"))
    if checksums.exists():
        if verification!=read(out/"verification.json"):
            raise ValueError("Published evidence inventory or verification summary differs")
    else:
        dump(out/"verification.json",verification)
        dump(checksums,inventory())
    print(json.dumps(verification,indent=2),flush=True)
