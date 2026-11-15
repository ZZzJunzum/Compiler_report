"""Verify published data, hashes, pairing, semantic coverage and frozen decisions."""
from collections import Counter,defaultdict
from itertools import product

from .common import DOUBLES, FAMILIES, MODES, ROOT, SPEC, conditions, dump, read, records, sha
from .features import blocks
from .hlo import Module
from .model import estimates, require_query, verify_lock


def require(ok,detail):
    if not ok:
        raise ValueError(str(detail))


def raw_audit(root,phase,cases,nproc,modes):
    count=Counter()
    phase_dir=root/phase
    require({p.name for p in (phase_dir/"raw").iterdir() if p.is_dir()}=={f"p{i}" for i in range(nproc)},phase)
    for i in range(nproc):
        proc=phase_dir/"raw"/f"p{i}"
        require({p.name for p in proc.iterdir() if p.is_dir()}==set(cases),proc)
        for case in cases:
            raw=proc/case
            stamp=read(raw/"done.json")
            for filename,h in stamp["files"].items():
                require(sha(raw/filename)==h,raw/filename)
            if "signature" in stamp:
                sig=stamp["signature"]
                for k,p in (("protocol","protocol.json"),("graphs","fusionscope/graphs.py"),("worker","fusionscope/bench.py")):
                    require(sig[k]==sha(ROOT/p),(raw,k))
                require(sig["env"]==read(phase_dir/f"env_p{i}.json"),raw)
            groups=defaultdict(list)
            for row in records(raw/"times.csv"):
                require(row["case"]==case and int(row["process"])==i and int(row["elapsed_ns"])>0
                        and row["endpoint"]=="all_device_outputs_ready",raw)
                groups[row["group"],int(row["round"])].append(row)
                count[row["group"]]+=1
            for group,labels in (("candidate",modes),("AA",SPEC["timing"]["aa_labels"])):
                for j in range(SPEC["timing"]["rounds"]):
                    rr=groups[group,j]
                    require(sorted(r["mode"] for r in rr)==sorted(labels) and
                            sorted(int(r["order"]) for r in rr)==list(range(len(labels))),(raw,group,j))
            require(len(groups)==2*SPEC["timing"]["rounds"],raw)
            require(sorted(r["mode"] for r in records(raw/"builds.csv"))==sorted(modes),raw)
            checks=read(raw/"checks.json")
            expected=sorted(product(SPEC["input_seeds"],modes))
            require(sorted((c["seed"],c["mode"]) for c in checks)==expected,raw)
            for c in checks:
                count["semantic_checks"]+=1
                count["valid"]+=int(c["valid"])
                require(c["valid"]==(c["y_pass"] and c["aux_pass"]),raw)
                if not c["valid"]:
                    require((root/"failures"/f"{case}_{c['seed']}.npz").exists(),c)
            if phase=="external_singles":
                require(all(v["passed"] for v in read(raw/"upstream.json")),raw)
    return dict(count)


def verify(root):
    lock=verify_lock(root)
    t=SPEC["timing"]
    phases={
        "probe4":(["E0_chain"],t["pilot_processes"],["only_y","sample","mean","full"]),
        "pilot":(["E0_chain","E3_chain","M0_chain","M3_chain"],t["pilot_processes"],MODES),
        "matrix":(conditions(),t["matrix_processes"],MODES),
        "confirm":(conditions(),t["confirmation_processes"],MODES),
        "external":(["R1","R2"],t["external_processes"],MODES),
        "external_singles":(["R1","R2"],t["confirmation_processes"],MODES[:5]),
        "boundary":(lock["boundary_cases"],t["boundary_processes"],["base","pre_pre","post_pre","split_pre_pre","split_post_pre"])}
    raw={phase:raw_audit(root,phase,*args) for phase,args in phases.items()}
    ir_files=0
    for phase,(cases,_,modes) in phases.items():
        if phase=="external_singles":
            continue
        for case,mode in product(cases,modes):
            folder=root/phase/"ir"/case/mode
            if mode.startswith("split_"):
                for name in ("front_hlo.txt","back_hlo.txt"):
                    Module((folder/name).read_text(encoding="utf8"))
                    ir_files+=1
            else:
                hashes=read(folder/"hashes.json")
                require(set(hashes)=={"jaxpr.txt","stablehlo.mlir","optimized_hlo.txt"},folder)
                for name,digest in hashes.items():
                    require(sha(folder/name)==digest,folder/name)
                    ir_files+=1
    points=conditions()+["R1","R2"]
    model=read(root/"model/models.json")
    require(sha(ROOT/"vendor/flax_imagenet/models.py")==SPEC["external"]["file_sha256"],"upstream hash")
    for case in points:
        phase="external" if case.startswith("R") else "matrix"
        facts=read(root/"baselines"/case/"features.json")
        for name,digest in read(root/"baselines"/case/"hashes.json").items():
            require(sha(root/"baselines"/case/name)==digest,name)
        # The predictor's uninstrumented baseline used implicit placement. Confirm that explicit
        # placement in the timed baseline yields exactly the locked feature vector and fallback.
        graph=Module((root/phase/"ir"/case/"base/optimized_hlo.txt").read_text(encoding="utf8"))
        require(blocks(facts["source"],graph)==facts["features"],("feature transfer",case))
        require(facts["fallback"]==any(graph.relu(site) is None for site in ("u","v")),case)
        fold="external" if phase=="external" else case[0]
        require(case not in model[fold]["training_conditions"] and case in model[fold]["heldout_conditions"],case)
        require(all(c[0]!=fold for c in model[fold]["training_conditions"]),("fold",fold))
        pred=estimates(model[fold]["ridge"],facts)
        for name,p in pred.items():
            saved=lock["predictions"][case][name]
            require(p["chosen"]==saved["chosen"] and
                    max(abs(p["log_ratios"][m]-saved["log_ratios"][m]) for m in DOUBLES)<1e-9,(case,name))
    try:
        require_query("minimum_before_relu")
    except ValueError:
        pass
    else:
        raise ValueError("Unsupported query accepted")
    require(len(read(root/"validation/numpy.json")["checks"])==72,"NumPy references")
    singles=root/"model/external_singles_lock.json"
    require(sha(singles)==singles.with_suffix(".sha256").read_text().strip(),"external single-site lock")
    for sub in ("lowering","lowering_aligned"):
        extra=records(root/sub/"comparison.csv")
        require(len(extra)==16,sub)
        for row in extra:
            p=root/sub/"ir"/row["case"]/row["relu_implementation"]/row["mode"]
            for name,digest in read(p/"hashes.json").items():
                require(sha(p/name)==digest,p/name)
            require(row["valid"]==str(all(c["valid"] for c in read(p/"checks.json"))),p)
            require(Module((p/"optimized_hlo.txt").read_text(encoding="utf8")).fingerprint()==row["sha256"],p)
            native="nn_relu" if row["case"]=="R1" else "maximum"
            if sub=="lowering_aligned" and row["relu_implementation"]==native:
                require(row["matches_primary_hlo"]=="True",("matched-control",p))
    tb=root/"report/tables"
    require(len(records(tb/"configs.csv"))==306 and len(records(tb/"structure.csv"))==1836,"output inventory")
    require(len(records(tb/"decisions.csv"))==34*9 and len(records(tb/"predictions.csv"))==34*5*4,"policy reports")
    require(len(records(tb/"interactions.csv"))==4*(32+32+2),"interactions")
    require(len(list((root/"report/figures").glob("*.png")))==4 and (root/"report/REPORT.md").exists(),"report")
    out={"verified":True,"registered_matrix_conditions":32,"matrix_configurations":288,
         "external_configurations":18,"semantic_reference_cases":72,"primary_ir_files_checked":ir_files,
         "baseline_feature_vectors_matching_timed_baselines":len(points),"raw_by_stage":raw,
         "timed_samples_total":sum(s["candidate"]+s["AA"] for s in raw.values()),
         "candidate_samples_total":sum(s["candidate"] for s in raw.values()),
         "aa_samples_total":sum(s["AA"] for s in raw.values()),"prediction_lock_sha256":sha(root/"model/prediction_lock.json")}
    dump(root/"report/audit.json",out)
    print(f"Verified 288 matrix configurations and 18 external configurations, {out['timed_samples_total']:,} raw timings.")
    return out
