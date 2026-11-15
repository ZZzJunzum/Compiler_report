"""Leave-whole-family-out ridge models and the baseline-only decision interface."""
from datetime import datetime, timezone
import hashlib
import json
import math
from statistics import median

import numpy as np

from .common import DOUBLES, FAMILIES, ROOT, SPEC, conditions, describe, dump, read, sha, table
from .stats import summarize

LEVELS=["B1","B2","M1","M2"]


def vector(blocks,level):
    return {k:v for group in LEVELS[:LEVELS.index(level)+1] for k,v in blocks[group].items()}


def fit(rows,level):
    names=list(vector(rows[0][0],level))
    x=np.array([[vector(f,level)[k] for k in names] for f,_ in rows],dtype=float)
    y=np.array([y for _,y in rows])
    mean,std=x.mean(axis=0),x.std(axis=0)
    std[std<1e-12]=1
    x=(x-mean)/std
    intercept=y.mean()
    w=np.linalg.solve(x.T@x+np.eye(len(names)),x.T@(y-intercept))
    return {"level":level,"names":names,"mean":mean.tolist(),"scale":std.tolist(),
            "weights":w.tolist(),"intercept":float(intercept),"alpha":1.0,"training_rows":len(y)}


def predict(model, features):
    values=vector(features,model["level"])
    x=np.array([values[k] for k in model["names"]])
    return float(np.dot((x-model["mean"])/model["scale"],model["weights"])+model["intercept"])


def estimates(models, facts):
    rows={}
    for level in LEVELS:
        used="B2" if facts["fallback"] and level in ("M1","M2") else level
        scores={m:predict(models[used],facts["features"][m]) for m in DOUBLES}
        rows[level]={"chosen":min(DOUBLES,key=scores.get),"log_ratios":scores,
                     "fallback":used!=level,"used":used}
    return rows


def fixed(chosen):
    return {"chosen":chosen,"log_ratios":{},"fallback":False,"used":"fixed"}


def additive(processes):
    ratios={}
    for mode in DOUBLES:
        u,v=mode.split("_")
        ratios[mode]=median((p[f"u_{u}"]+p[f"v_{v}"]-p["base"])/p["base"] for p in processes)
    # Negative estimates from noisy single timings still participate in the ranking.
    return {"chosen":min(DOUBLES,key=ratios.get),"ratios":ratios,"log_ratios":
            {m:math.log(r) for m,r in ratios.items() if r>0},"fallback":False,"used":"measured_singles"}


def lock_predictions(root):
    path=root/"model/prediction_lock.json"
    if path.exists():
        verify_lock(root)
        return
    if any((root/p/"raw").exists() for p in ("confirm","external")):
        raise RuntimeError("Independent measurements already exist without this lock. Choose a fresh output directory.")
    data={c:summarize(root,"matrix",c) for c in conditions()}
    if any(len(s["processes"])!=SPEC["timing"]["matrix_processes"] for s in data.values()):
        raise RuntimeError("Incomplete discovery matrix")
    facts={c:read(root/"baselines"/c/"features.json") for c in [*conditions(),"R1","R2"]}
    predictions, models={},{}
    for fold in FAMILIES+["external"]:
        train=[c for c in conditions() if describe(c)[0]!=fold]
        test=[c for c in conditions() if describe(c)[0]==fold] if fold!="external" else ["R1","R2"]
        # Only training-family eligibility and targets enter the fitted models and fixed-rule selection.
        full=[c for c in train if all(data[c]["valid"][m] for m in DOUBLES)]
        rows=[(facts[c]["features"][m],math.log(data[c]["ratios"][m])) for c in full for m in DOUBLES]
        best=min(DOUBLES,key=lambda m:np.mean([math.log(data[c]["ratios"][m]) for c in full]))
        fitted={level:fit(rows,level) for level in LEVELS}
        models[fold]={"training_conditions":full,"heldout_conditions":test,"ridge":fitted,"fixed":best}
        for c in test:
            row=estimates(fitted,facts[c])
            row.update({"B0_pre":fixed("pre_pre"),"B0_post":fixed("post_post"),"B0_train":fixed(best)})
            if fold!="external":
                row["single_add"]=additive(data[c]["processes"])
            predictions[c]=row
    eligible=[c for c in conditions() if all(data[c]["valid"][m] for m in DOUBLES)]
    def spread(c):
        t=[data[c]["times"][m] for m in DOUBLES]
        return max(t)-min(t)
    weak=min(eligible,key=lambda c:(spread(c),c))
    strong=min(eligible,key=lambda c:(-spread(c),c))
    modelpath=root/"model/models.json"
    dump(modelpath,models)
    lock={"locked_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":sha(ROOT/"protocol.json"),
          "model_sha256":sha(modelpath),"feature_input":"baseline Jaxpr and baseline optimized HLO",
          "baseline_feature_hashes":{c:sha(root/"baselines"/c/"features.json") for c in facts},
          "boundary_cases":[weak,strong],"predictions":predictions}
    dump(path,lock)
    (path.parent/"prediction_lock.sha256").write_text(sha(path)+"\n",encoding="ascii",newline="\n")
    print("Locked baseline-only recommendations and discovery single-add comparison.",flush=True)


def verify_lock(root):
    path=root/"model/prediction_lock.json"
    lock=read(path)
    if sha(path)!=(path.parent/"prediction_lock.sha256").read_text().strip():
        raise ValueError("Prediction lock digest mismatch")
    if sha(root/"model/models.json")!=lock["model_sha256"] or sha(ROOT/"protocol.json")!=lock["protocol_sha256"]:
        raise ValueError("Model or registered protocol digest mismatch")
    for case,digest in lock["baseline_feature_hashes"].items():
        if sha(root/"baselines"/case/"features.json")!=digest:
            raise ValueError(f"Changed baseline features: {case}")
    return lock


def recommendation(models,facts):
    pred=estimates(models,facts)["M2"]
    return {"chosen":pred["chosen"],"predicted_ratios":{m:math.exp(s) for m,s in pred["log_ratios"].items()},
            "model_used":pred["used"],"fallback":pred["fallback"],
            "reason":"a baseline ReLU lacks a unique HLO anchor" if pred["fallback"] else "baseline HLO anchors available"}


def require_query(query):
    if query!="positive_count":
        raise ValueError("Only positive_count on finite ReLU inputs has a registered equivalence proof.")


def recommend_program(fn,args,models,query="positive_count"):
    """New annotated baseline. No instrumented executable or candidate timing is required."""
    require_query(query)
    from .features import extract
    return recommendation(models,extract(fn,args))


def recommend(root,case,query):
    require_query(query)
    verify_lock(root)
    facts=read(root/"baselines"/case/"features.json")
    family=describe(case)[0]
    fold=family if family in FAMILIES else "external"
    models=read(root/"model/models.json")[fold]["ridge"]
    return json.dumps({"case":case,"query":query,**recommendation(models,facts)},indent=2,ensure_ascii=False)
