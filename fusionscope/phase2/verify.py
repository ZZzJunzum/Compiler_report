"""Verify each published schedule, semantic check, executable, and frozen decision."""
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from zipfile import ZipFile

from fusionscope.common import ROOT, conditions, dump, read, records, sha
from fusionscope.evidence import scheduled_text_digest
from fusionscope.hlo import Module
from .audit_ir import ARCHIVE, PLAN
from .cost_model import baseline_facts, features, predict, verify_lock
from .experiment import PHASES, cases_for
from .stats import cohort, metrics
from .variants import options


def require(ok, where):
    if not ok:
        raise ValueError(f"Phase II evidence check failed: {where}")


def csv_member(archive, member):
    return list(csv.DictReader(io.StringIO(archive.read(member).decode("utf8"))))


def check_process(out, phase, process, ir_refs, refs_by_seed, raw_dir=None):
    cases=cases_for(phase)
    folder=raw_dir if raw_dir is not None else out/PHASES[phase]/"raw"
    path=folder/f"p{process}.zip"
    with ZipFile(path) as z:
        require(z.testzip() is None,path)
        meta=json.loads(z.read("environment.json"))
        times=csv_member(z,"timings.csv")
        checks=csv_member(z,"correctness.csv")
        builds=csv_member(z,"builds.csv")
    env=meta["environment"]
    require(meta["phase"]==phase and meta["process"]==process and sorted(meta["case_order"])==sorted(cases),path)
    require(env["backend"]=="cpu" and len(env["affinity"])==1 and env["flags"]["JAX_ENABLE_COMPILATION_CACHE"].lower() in ("false","0"),path)
    for name,file in {"protocol":ARCHIVE/"protocol.json","graphs":ROOT/"fusionscope/graphs.py",
                      "variants":ROOT/"fusionscope/phase2/variants.py","worker":ROOT/"fusionscope/phase2/experiment.py"}.items():
        require(meta["measurement_signature"][name]==sha(file),f"{path} {name}")
    labels={c:options(c) for c in cases}
    groups=defaultdict(list)
    for row in times:
        require(int(row["elapsed_ns"])>0 and int(row["process"])==process,path)
        groups[row["case"],row["block"],int(row["round"])].append((int(row["order"]),row["variant"]))
    counts=Counter()
    for case in cases:
        for block,names in (("candidate",labels[case]),("AA_V0","ABCD"),("AA_V1","ABCD")):
            for rnd in range(-PLAN["warmups"],PLAN["rounds"]):
                group=groups.pop((case,block,rnd))
                require(sorted(i for i,_ in group)==list(range(len(names))) and
                        sorted(v for _,v in group)==sorted(names),f"{case} {block} round {rnd}")
                key="warmup_calls" if rnd<0 else "formal_candidate_calls" if block=="candidate" else "formal_aa_calls"
                counts[key]+=len(group)
    require(not groups,path)
    expected={(c,v,str(s)) for c in cases for v in labels[c] for s in PLAN["input_seeds"]}
    for row in checks:
        case,v,seed=(row[k] for k in ("case","variant","seed"))
        expected.remove((case,v,seed))
        actual,ref=map(json.loads,(row["actual_counts"],row["reference_counts"]))
        require(len(ref)==(0 if v=="V0" else 2) and all(isinstance(q,int) and q>=0 for q in ref),row)
        if ref:
            key=case,seed
            require(ref==refs_by_seed.setdefault(key,ref),key)
        valid=row["valid"]=="True"
        require(valid==(row["y_valid"]=="True" and row["counts_valid"]=="True") and
                (row["counts_valid"]!="True" or actual==ref),row)
        if not valid:
            require((out/"failures"/f"{case}-{seed}.npz").is_file(),row)
        counts["semantic_checks"]+=1
        counts["failed_checks"]+=int(not valid)
    require(not expected,path)
    parts={(c,v,k) for c in cases for v in labels[c] for k in (("front","back") if v=="V3" else ("main",))}
    for b in builds:
        c,v,k=(b[x] for x in ("case","variant","component"))
        parts.remove((c,v,k))
        for key in ("lower_ns","compile_ns","argument_bytes","output_bytes","temp_bytes","alias_bytes"):
            require(int(b[key])>=0,b)
        file=b["ir_archive"]
        require(file.startswith("ir/"+c+"-"),file)
        member=f"{v}/{k}"
        require(ir_refs[file].setdefault(member,b["hlo_digest"])==b["hlo_digest"],file)
        counts["compiled_component_records"]+=1
    require(not parts,path)
    return counts,meta["pid"]


def check_ir(out,refs):
    require({p.relative_to(out).as_posix() for p in (out/"ir").glob("*.zip")}==set(refs),"IR archives")
    for file,parts in refs.items():
        with ZipFile(out/file) as z:
            require(z.testzip() is None,file)
            expected={f"{p}/{s}" for p in parts for s in ("optimized_hlo.txt","stablehlo.mlir")}
            require(set(z.namelist())==expected,file)
            for p,digest in parts.items():
                require(scheduled_text_digest(z.read(p+"/optimized_hlo.txt").decode())==digest,(file,p))
                require(b"module " in z.read(p+"/stablehlo.mlir"),(file,p))
    diag=out/"ir/diagnostic/E3_branch"
    manifest=read(diag/"manifest.json")
    for file,digest in manifest["files"].items():
        require(sha(diag/file)==digest,file)
    old=ROOT/"results/cpu/matrix/ir/E3_branch/post_post/optimized_hlo.txt"
    require(Module(old.read_text(encoding="utf8")).fingerprint()==manifest["hlo_signature"],"frozen E3 source")
    with ZipFile(diag/"fusion-pass-snapshots.zip") as z:
        for row in records(diag/"pipeline_steps.csv"):
            require(hashlib.sha256(z.read(row["snapshot"])).hexdigest()==row["sha256"],row)
    paths=records(out/"analysis/fusion_paths.csv")
    for case in PLAN["screen_cases"]:
        for mode,outputs in (("base",{"y"}),("post_post",{"y","q1","q2"})):
            require({r["output"] for r in paths if r["case"]==case and r["mode"]==mode}==outputs,(case,mode))


def check_predictions(out):
    lock=verify_lock(out)
    models=read(out/"model/model.json")
    all_cases=conditions()+PLAN["external"]
    pred={(r["case"],r["variant"]):r for r in records(out/"model/predictions.csv")}
    stored={(r["case"],r["variant"]):r for r in records(out/"model/features.csv")}
    require(set(pred)=={(c,v) for c in all_cases for v in options(c)[1:]},"prediction matrix")
    require(set(stored)=={(c,v) for c in all_cases for v in options(c)},"feature matrix")
    for fold,model in models.items():
        test=PLAN["external"] if fold=="external" else [c for c in conditions() if c.startswith(fold)]
        require(set(model["test"])==set(test) and set(model["train"])==set(conditions())-set(test),fold)
        for c in test:
            base=baseline_facts(c)
            values={}
            for v in options(c):
                f=features(c,v,base)
                original=stored[c,v]
                require(original["baseline_hlo_sha256"]==base["baseline_hlo_sha256"] and
                        all(math.isclose(float(original[k]),w) for k,w in f.items()),(c,v))
                if v=="V0":
                    continue
                us,ratio=predict(model["parameters"],f)
                expected=pred[c,v]
                require(expected["fold"]==fold and math.isclose(us,float(expected["analytic_us"])) and
                        math.isclose(ratio,float(expected["ridge_ratio"])),(c,v))
                values[v]=(us,ratio)
            for name,col in (("analytic",0),("ridge",1)):
                require(min(values,key=lambda v:values[v][col])==lock["choices"][c][name],(c,name))
    timings={(r["case"],r["variant"]):r for r in records(out/"analysis/timings.csv") if r["phase"] in ("confirm","external")}
    decisions=records(out/"model/heldout_results.csv")
    for c in all_cases:
        measured={v:float(timings[c,v]["latency_us"]) for v in options(c)[1:] if timings[c,v]["valid"]=="True"}
        oracle=min(measured,key=lambda v:(measured[v],v))
        chosen=[r for r in decisions if r["case"]==c]
        policies=set(lock["choices"][c])|{"oracle"}
        if c in conditions():
            policies.add("measured_tile")
        require({r["policy"] for r in chosen}==policies,(c,"coverage"))
        for r in chosen:
            require(r["best"]==oracle and r["chosen"]==lock["choices"][c].get(r["policy"],r["chosen"]),r)
            if r["chosen_valid"]=="True":
                diff=measured[r["chosen"]]-measured[oracle]
                require(math.isclose(diff,float(r["gap_us"]),abs_tol=1e-10) and
                        math.isclose(diff/measured[oracle],float(r["regret"]),abs_tol=1e-12),r)
    return len(pred)


def check_statistics(out):
    saved={(r["phase"],r["case"],r["variant"]):r for r in records(out/"analysis/timings.csv")}
    for phase in PHASES:
        for case,group in cohort(out,phase).items():
            for variant in options(case):
                row=saved.pop((phase,case,variant))
                for key,want in metrics(group,variant).items():
                    value=row[key]
                    if want is None:
                        ok=value==""
                    elif isinstance(want,bool):
                        ok=value==str(want)
                    else:
                        ok=math.isclose(float(value),want,rel_tol=1e-11,abs_tol=1e-11)
                    require(ok,(phase,case,variant,key))
    require(not saved,"raw/summary coverage")


def verify(out):
    refs=defaultdict(dict)
    totals=Counter()
    phase_counts={}
    refs_by_seed={}
    for phase in PHASES:
        tally=Counter()
        pids=[]
        for process in range(PLAN["processes"][phase]):
            counts,pid=check_process(out,phase,process,refs,refs_by_seed)
            tally.update(counts)
            pids.append(pid)
        require(len(pids)==len(set(pids)),f"{phase} process independence")
        phase_counts[phase]=dict(processes=len(pids),**tally)
        totals.update(tally)
    replay=None
    folder=out/"screening/replay"
    if folder.exists():
        info=read(folder/"provenance.json")
        hashes=info["raw_sha256"]
        tally=Counter()
        pids=set()
        for i in range(PLAN["processes"]["screen"]):
            path=folder/"raw"/f"p{i}.zip"
            require(sha(path)==hashes[path.name],path)
            counts,pid=check_process(out,"screen",i,refs,refs_by_seed,folder/"raw")
            pids.add(pid)
            tally.update(counts)
        require(len(pids)==PLAN["processes"]["screen"] and len(info["checkout_commit"])==40,"clean replay")
        rows=records(folder/"comparison.csv")
        require({(r["case"],r["variant"]) for r in rows}==
                {(c,v) for c in PLAN["screen_cases"] for v in options(c)},"replay coverage")
        verified=cohort(out,"screen",folder/"raw")
        initial=cohort(out,"screen")
        for r in rows:
            c,v=r["case"],r["variant"]
            require(math.isclose(float(r["replay_speedup"]),metrics(verified[c],v)["speedup"]),r)
            hashes_seen=[{(b["component"],b["hlo_digest"]) for b in group[c]["builds"] if b["variant"]==v}
                         for group in (initial,verified)]
            require((r["same_compiled_components"]=="True")==(hashes_seen[0]==hashes_seen[1]),r)
        replay=dict(checkout_commit=info["checkout_commit"],processes=len(pids),configurations=len(rows),
                    matching_ir_configurations=sum(r["same_compiled_components"]=="True" for r in rows),**tally)
    check_ir(out,refs)
    check_statistics(out)
    predictions=check_predictions(out)
    require(read(out/"screening/gate.json")["structural_change"],"screening")
    report=out/"report/REPORT.md"
    for link in re.findall(r"\]\(([^)]+)\)",report.read_text(encoding="utf8")):
        if not link.startswith(("http","#")):
            require((report.parent/link.split("#")[0]).exists(),link)
    audit=dict(backend="CPU",cohorts=phase_counts,totals=dict(totals),
               distinct_synthetic_configurations=sum(len(options(c)) for c in conditions()),
               distinct_external_configurations=sum(len(options(c)) for c in PLAN["external"]),
               case_ir_archives=len(refs),archived_components=sum(len(d) for d in refs.values()),
               frozen_candidate_predictions=predictions,
               prediction_lock_sha256=sha(out/"model/prediction_lock.json"),
               checks="All scheduled rows, output references, IR fingerprints, phase-zero files, train/test split, frozen predictions, selection regret, and report links verified.")
    if replay is not None:
        audit["additional_clean_clone_screening"]=replay
    dump(out/"report/audit.json",audit)
    print(json.dumps(audit["totals"],sort_keys=True))
    return audit
