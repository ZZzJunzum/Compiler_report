"""Select and freeze policies using old-case timings and unobserved baseline IR only."""
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
import json
from math import ceil, log, log2, prod
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import expit

from fusionscope.common import ROOT, dump, read, sha, table
from fusionscope.phase2.cost_model import features as old_features, predict as old_predict
from .data import cohort, direction
from .design import REPEATS, ROUNDS, development, heldout
from .experiment import signature


FAMILIES = tuple("EMNC")
MODELS = ("analytic_old", "ridge_old", "volume", "throughput", "log_main", "log_cross",
          "pair_main", "pair_cross")
GRID = (.1, 1., 10.)
BASE_COLS = ("log_n", "width", "op_m", "op_c", "op_n", "branch", "base_work", "fanout", "overlap",
             "joint", "staged", "tiled", "rows", "steps", "live", "visits", "boundary", "tree",
             "compute_us", "transfer_us", "kernel_us", "fraction")
CROSS_COLS = tuple(f"{s}_{op}" for s in ("joint","tiled","rows","fraction") for op in ("op_m","op_c","op_n")) + (
    "joint_fraction", "joint_rows", "tiled_fraction", "staged_fraction")


def legacy_domain(c):
    original = (1,2,4,8,16) if c.ops == "CC" else (1,4,16,64,256)
    return ["V2","V1","V3", *[f"S{b}" for b in c.tiles if b in original]]


def encode(c, label, base, cal):
    batch, width = c.shape[0], c.shape[-1]
    n = prod(c.shape)
    tiled = label[0] in "SJ"
    rows = int(label[1:]) if tiled else batch
    trips = batch//rows
    # A single mapped iteration is predicted to be removable. This uses only the source shape.
    joint, staged = label == "V2" or label[0] == "J", label == "V3"
    tiled = tiled and trips > 1
    elements = n//trips
    masks = 0 if joint else (4 if staged else 8)
    live = elements * (12+masks)
    visits = (12+masks*2)*n
    boundary = 8*n if staged else 0
    partial = 16*trips if tiled else 0
    fraction = live/cal["knee_bytes"]
    # Smooth transition between the two calibrated bandwidth regimes.
    blend = fraction/(1+fraction)
    bandwidth = (1-blend)*cal["small_bytes_ns"] + blend*cal["large_bytes_ns"]
    traffic = (visits+boundary+partial+base["base_bytes"])/bandwidth/1000
    main_mflop = base["base_work"]/1e6
    kernel = 0.
    for op in c.ops:
        if op in "MC":
            shape = (rows, *c.shape[1:])
            def distance(r):
                return sum(abs(log2(x/y)) for x,y in zip(shape, r["shape"]))
            choices = [r for r in cal["layer_rates"] if r["op"]==op]
            rate = min(choices,key=distance)["mflop_per_us"]
            kernel += 2*n*width*(9 if op=="C" else 1)/1e6/rate
    tree = (1 if joint else 2)*ceil(log(max(elements,2),32)) + (2*ceil(log(trips,32)) if tiled else 0)
    v = dict(log_n=log2(n), width=log2(width), op_m=c.ops.count("M")/2, op_c=c.ops.count("C")/2,
             op_n=c.ops.count("N")/2, branch=int(c.top=="branch"), base_work=np.log1p(main_mflop),
             fanout=base["fanout"], overlap=base["shared_prefix_ops"], joint=int(joint), staged=int(staged),
             tiled=int(tiled), rows=log2(rows), steps=log2(trips), live=np.log1p(live), visits=np.log1p(visits),
             boundary=np.log1p(boundary), tree=tree, compute_us=np.log1p(main_mflop),
             transfer_us=np.log1p(traffic), kernel_us=np.log1p(kernel), fraction=np.log1p(fraction))
    for s in ("joint","tiled","rows","fraction"):
        for op in ("op_m","op_c","op_n"):
            v[f"{s}_{op}"] = v[s]*v[op]
    for a,b in (("joint","fraction"),("joint","rows"),("tiled","fraction"),("staged","fraction")):
        v[f"{a}_{b}"] = v[a]*v[b]
    return v, [1+int(staged), main_mflop, traffic, kernel, tree, trips-1]


def ordered(c):
    return ["V2", *[v for v in c.options if v not in ("V0","V2")]]


def learn(kind, alpha, groups):
    cols = BASE_COLS + (CROSS_COLS if kind.endswith("cross") else ())
    rows = [item for group,_ in groups for item in group]
    X = np.array([[f[k] for k in cols] for f, roof, t, base in rows])
    y = np.array([log(t/b) for f,roof,t,b in rows])
    mean, sd = X.mean(0), X.std(0)
    sd[sd<1e-8]=1
    x = (X-mean)/sd
    scale, bias = 1., float(y.mean())
    if kind.startswith("log"):
        w = np.linalg.solve(x.T@x + alpha*np.eye(x.shape[1]), x.T@(y-bias))
    else:
        diffs, targets = [], []
        offset = 0
        for group, floor in groups:
            for i,j in combinations(range(len(group)),2):
                delta = group[i][2]-group[j][2]
                if abs(delta)>floor:
                    diffs.append(x[offset+i]-x[offset+j])
                    targets.append(float(delta>0))
            offset += len(group)
        d, t = np.asarray(diffs), np.asarray(targets)
        def objective(w):
            margins = d@w
            return float(np.sum(np.logaddexp(0.,margins)-t*margins)+alpha*(w@w)/2), d.T@(expit(margins)-t)+alpha*w
        fit = minimize(objective, np.zeros(len(cols)), jac=True, method="L-BFGS-B",
                       options={"maxiter":400,"ftol":1e-10})
        if not fit.success:
            raise RuntimeError(fit.message)
        w = fit.x
        # Log-latency calibration is diagnostic. The margin alone determines the ranking.
        z = x@w
        scale = max(1e-9, float(np.dot(z,y-bias)/max(np.dot(z,z),1e-12)))
    return dict(kind=kind, columns=cols, mean=mean.tolist(), sd=sd.tolist(), w=w.tolist(),
                bias=bias, scale=scale, alpha=alpha)


def roof_predict(kind, a, r):
    calls, base, transfer, kernel, tree, steps = np.moveaxis(np.asarray(r), -1, 0)
    d,f,b,h,s = a
    compute = f*base + (kernel if kind=="throughput" else 0.)
    return d*calls + np.maximum(compute,b*transfer) + h*tree + s*steps


def fit_roof(kind, groups):
    rows = [item for group,_ in groups for item in group]
    r = np.array([item[1] for item in rows])
    y = np.log([item[2] for item in rows])
    result = least_squares(lambda w: np.log(roof_predict(kind,np.exp(w),r))-y,
                           np.log([7.,1.,1.,.1,.1]),bounds=(-14,14),max_nfev=400)
    return dict(kind=kind, coefficients=np.exp(result.x).tolist(), fit_success=bool(result.success))


def score(model, item):
    f, roof = item
    if "coefficients" in model:
        return float(roof_predict(model["kind"],model["coefficients"],roof))
    x = (np.array([f[k] for k in model["columns"]])-model["mean"])/model["sd"]
    return float(model["bias"] + model["scale"]*np.dot(x,model["w"]))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def verify_lock(out, discovery=False):
    for folder in (("static","decision") if discovery else ("static",)):
        path = out/folder/"lock.json"
        data = read(path)
        if sha(path) != (path.parent/"lock.sha256").read_text().strip():
            raise ValueError(f"Changed {folder} lock")
        for rel,h in data["inputs"].items():
            if sha(out/rel) != h:
                raise ValueError(f"Changed lock input: {rel}")
        for rel,h in data.get("sources",{}).items():
            if sha(ROOT/rel) != h:
                raise ValueError(f"Changed frozen source: {rel}")
    return data


def seal(folder, data):
    path = folder/"lock.json"
    dump(path, data)
    (folder/"lock.sha256").write_text(sha(path)+"\n",encoding="ascii",newline="\n")


def lock(out):
    static = out/"static"
    if (static/"lock.json").exists():
        return verify_lock(out)
    if (out/"discovery").exists() or (out/"confirmation").exists():
        raise RuntimeError("A prospective lock requires an output directory without held-out timings")
    dev, fresh = development(), heldout()
    cal, bases = read(static/"calibration.json"), read(static/"baseline.json")
    old = read(ROOT/"results/phase2/model/model.json")
    devdata = cohort(out,"development")
    encoded, oldfeat, groups = {}, {}, {}
    feature_rows = []
    for c in dev+fresh:
        for v in ordered(c):
            f,r = encode(c,v,bases[c.name]["facts"],cal)
            encoded[c.name,v] = (f,r)
            feature_rows.append(dict(case=c.name,variant=v,**f,roof_inputs=json.dumps(r)))
            if v in legacy_domain(c):
                alias = "V4_t"+v[1:] if v.startswith("S") else v
                oldfeat[c.name,v] = old_features(c.name,alias,bases[c.name]["facts"],shape=c.shape)
        if c in dev:
            data = devdata[c.name]
            groups[c.name] = ([
                (*encoded[c.name,v], data["us"][v], data["us"]["V0"])
                for v in ordered(c) if v not in data["invalid"]],data["floor"])
    table(static/"features.csv",feature_rows)
    @lru_cache(None)
    def fit(kind, families, alpha):
        gs = [groups[c.name] for c in dev if c.ops[0] in families]
        return fit_roof(kind,gs) if kind in ("volume","throughput") else learn(kind,alpha,gs)
    def scores(c, kind, model, fold):
        if kind.endswith("_old"):
            idx = 0 if kind=="analytic_old" else 1
            return {v:old_predict(old[fold]["parameters"],oldfeat[c.name,v])[idx] for v in legacy_domain(c)}
        return {v:score(model,encoded[c.name,v]) for v in ordered(c)}
    def loss(c, pred):
        d = devdata[c.name]
        chosen = min(pred,key=pred.get)
        feasible = [v for v in ordered(c) if v not in d["invalid"]]
        return log(d["us"][chosen]/min(d["us"][v] for v in feasible)) if chosen in feasible else 10.
    inner = []
    def tune(kind, families):
        if kind in ("volume","throughput"):
            return 0.
        values = []
        for alpha in GRID:
            regrets = []
            for hold in families:
                m = fit(kind,tuple(f for f in families if f!=hold),alpha)
                regrets += [loss(c,scores(c,kind,m,hold)) for c in dev if c.ops[0]==hold]
            avg = float(np.mean(regrets))
            inner.append(dict(model=kind,train_families="".join(families),alpha=alpha,mean_log_regret=avg))
            values.append((avg,alpha))
        return min(values)[1]
    fold_rows,params = [],{}
    for hold in FAMILIES:
        families = tuple(f for f in FAMILIES if f!=hold)
        for kind in MODELS:
            m = None if kind.endswith("_old") else fit(kind,families,tune(kind,families))
            params[f"{hold}/{kind}"] = dict(train=list(families), held=hold, model=m)
            for c in dev:
                if c.ops[0]!=hold:
                    continue
                pred = scores(c,kind,m,hold)
                fold_rows.append(dict(case=c.name,fold=hold,model=kind,choice=min(pred,key=pred.get),
                                      log_regret=loss(c,pred)))
    means = {k:float(np.mean([r["log_regret"] for r in fold_rows if r["model"]==k])) for k in MODELS}
    # The principal static selector must support the complete candidate grid.
    eligible = [k for k in MODELS if not k.endswith("_old")]
    primary = min(eligible,key=means.get)
    ranker = min(("pair_main","pair_cross"),key=means.get)
    allmodels = {k:fit(k,FAMILIES,tune(k,FAMILIES)) for k in eligible}
    choices, predictions, score_us = {}, [], {}
    for c in fresh:
        start = perf_counter_ns()
        estimates = {k:scores(c,k,allmodels.get(k),"external") for k in MODELS}
        pick = {k:min(p,key=p.get) for k,p in estimates.items()}
        ranking = sorted(estimates[ranker],key=estimates[ranker].get)
        choices[c.name] = dict(**pick,static=pick[primary],rank_alt=next(v for v in ranking if v!="V2"),
                               fixed_alt=c.fixed_tile, V2="V2")
        score_us[c.name] = (perf_counter_ns()-start)/1000
        for k,p in estimates.items():
            unit = "us" if k in ("volume","throughput","analytic_old") else "ratio_to_V0" if k=="ridge_old" else "log_ratio_to_V0"
            predictions += [dict(case=c.name,model=k,variant=v,score=s,unit=unit) for v,s in p.items()]
    table(static/"outer_folds.csv",fold_rows)
    table(static/"inner_folds.csv",inner)
    table(static/"predictions.csv",predictions)
    dump(static/"models.json",dict(outer=params,final=allmodels))
    manifest = dict(workloads=[c.manifest() for c in fresh], weight_seed=1103, validation_seeds=[101,4471,8231],
                    timed_seed=101, processes=REPEATS, timed_rounds=ROUNDS, warmup_rounds=6,
                    bootstrap=dict(reps=10000,seed=32041), margin_ratio=.95, p90_allowance=.02,
                    tie_order="V2, V1, V3, then separate/joint in ascending legal extent",
                    primary_selection="Minimum whole-family outer mean log regret among full-domain static models",
                    discovery="Strict majority of two paired processes and median gain above V2 four-alias p95")
    dump(static/"manifest.json",manifest)
    inputs=[*static.glob("*.csv"), *static.glob("*.json"), static/"baseline_ir.zip", *(out/"development").glob("p*.zip")]
    sources = signature()
    for f in ("models.py","calibration.py","data.py"):
        p = Path(__file__).with_name(f)
        sources[p.relative_to(ROOT).as_posix()] = sha(p)
    for f in ("fusionscope/phase2/cost_model.py","results/phase2/model/model.json"):
        sources[f] = sha(ROOT/f)
    seal(static,dict(utc=stamp(), primary=primary, ranker=ranker, outer_scores=means, choices=choices,
                     inputs={p.relative_to(out).as_posix():sha(p) for p in inputs},sources=sources,
                     all_model_scoring_us=score_us, environment=cal["environment"]))
    print("Locked primary:", primary, "ranker:", ranker, flush=True)


def choose(out):
    static = verify_lock(out)
    folder = out/"decision"
    if (folder/"lock.json").exists():
        return verify_lock(out,discovery=True)
    if (out/"confirmation").exists():
        raise RuntimeError("Discovery choices must precede confirmation")
    rows, choices = [],{}
    inputs = [out/"static/lock.json"]
    for policy in ("rank_alt","fixed_alt"):
        records = cohort(out,"discovery",policy)
        inputs.extend((out/"discovery"/policy).glob("p*.zip"))
        for c in heldout():
            alt = static["choices"][c.name][policy]
            d = records[c.name]
            gain,resolved = direction(d,alt,"V2")
            choice = alt if resolved and gain>0 else "V2"
            choices.setdefault(c.name,{})[policy] = choice
            rows.append(dict(case=c.name,policy=policy,alternative=alt,chosen=choice,gain_us=gain,
                             floor_us=d["floor"],process_gains_us=str([p["V2"]-p[alt] for p in d["processes"]])))
    table(folder/"discovery.csv",rows)
    inputs.append(folder/"discovery.csv")
    seal(folder,dict(utc=stamp(),choices=choices,inputs={p.relative_to(out).as_posix():sha(p) for p in inputs}))
