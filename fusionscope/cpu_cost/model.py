"""Fit and replay predictions using development observations and untimed candidate IR."""
from collections import defaultdict
import json
from statistics import median
from zipfile import ZipFile

import numpy as np

from fusionscope.common import dump, sha
from fusionscope.phase2.experiment import zip_rows
from .experiment import PROCESSES, ROUNDS, SEEDS
from .workloads import cases, variants

MODELS = ("constant", "source", "stable", "combined")
ALPHAS = (0.01, 0.1, 1., 10., 100.)


def archive_features(path):
    with ZipFile(path) as z:
        if z.testzip():
            raise ValueError(path)
        return json.loads(z.read("features.json"))


def development_targets(out):
    directory = out/"development"
    expected = {c.name+"/"+v for c in cases("development") for v in variants(c)}
    measures = defaultdict(list)
    rejected = set()
    for i in range(PROCESSES["development"]):
        p = directory/f"p{i}.zip"
        with ZipFile(p) as z:
            meta = json.loads(z.read("environment.json"))
            if z.testzip() or meta["process"] != i or meta["lock"] != sha(out/"protocol.json"):
                raise ValueError(f"Incorrect development archive {p}")
        timing, builds, checks = defaultdict(dict), {}, defaultdict(dict)
        for r in zip_rows(p,"timings.csv"):
            k = r["case"]+"/"+r["variant"]
            j = int(r["round"])
            if j in timing[k] or r["block"] != "candidate" or int(r["elapsed_ns"]) <= 0:
                raise ValueError(f"Invalid timing {k}")
            timing[k][j] = int(r["elapsed_ns"])/1000
        for r in zip_rows(p,"builds.csv"):
            k = r["case"]+"/"+r["variant"]
            if k in builds or int(r["build_wall_ns"]) <= 0:
                raise ValueError(k)
            builds[k] = int(r["build_wall_ns"])/1000
        for r in zip_rows(p,"correctness.csv"):
            k = r["case"]+"/"+r["variant"]
            if int(r["seed"]) in checks[k]:
                raise ValueError(k)
            checks[k][int(r["seed"])] = r["valid"] == "True"
        if not expected == timing.keys() == builds.keys() == checks.keys():
            raise ValueError("Incomplete development data")
        for k in expected:
            if set(timing[k]) != set(range(-6,ROUNDS["development"])) or set(checks[k]) != set(SEEDS["development"]):
                raise ValueError(k)
            if not all(checks[k].values()):
                rejected.add(k)
            measures[k].append([median(v for j,v in timing[k].items() if j>=0), builds[k]])
    return {k:np.median(v,axis=0) for k,v in measures.items() if k not in rejected}, sorted(rejected)


def vector(row, mode):
    groups = ("source","stable") if mode == "combined" else (mode,)
    return {f"{g}.{k}":v for g in groups for k,v in row[g].items()}


def matrix(rows, keys, mode, columns):
    return np.log1p([[row.get(col,0) for col in columns] for row in (vector(rows[k],mode) for k in keys)])


def fit(x, y, alpha):
    center, spread = x.mean(axis=0), x.std(axis=0)
    use = spread > 1e-8
    scale = np.where(use, spread, 1)
    z = ((x-center)/scale)[:,use]
    intercept = y.mean()
    weights = np.zeros_like(center)
    weights[use] = np.linalg.solve(z.T@z + alpha*np.eye(z.shape[1]), z.T@(y-intercept))
    return dict(center=center.tolist(), scale=scale.tolist(), weights=weights.tolist(), intercept=float(intercept))


def score(x, fit_model):
    return (x-fit_model["center"])/fit_model["scale"] @ np.array(fit_model["weights"]) + fit_model["intercept"]


def estimates(out):
    dev_file, test_file = (out/f"{s}/lowered.zip" for s in ["development","test"])
    dev, test = archive_features(dev_file), archive_features(test_file)
    target, rejected = development_targets(out)
    keys, testkeys = sorted(target), sorted(test)
    if set(testkeys) != {c.name+"/"+n for c in cases("test") for n in variants(c)}:
        raise ValueError("Test features do not match the registered menu")
    folds = {c.name:c.tier for c in cases("development")}
    fold = np.array([folds[k.split("/")[0]] for k in keys])
    predicted = {k:{} for k in testkeys}
    fits = {}
    for mode in MODELS:
        columns = [] if mode == "constant" else sorted({t for k in keys for t in vector(dev[k],mode)})
        if columns:
            x, xt = matrix(dev,keys,mode,columns), matrix(test,testkeys,mode,columns)
        fits[mode] = dict(columns=columns)
        for idx, endpoint in enumerate(["runtime_us","build_us"]):
            y = np.log([target[k][idx] for k in keys])
            if mode == "constant":
                v = np.full(len(testkeys), np.median(y))
                fits[mode][endpoint] = dict(log_median=float(v[0]))
            else:
                cv = {}
                for a in ALPHAS:
                    errors = []
                    for i in sorted(set(fold)):
                        hold = fold == i
                        errors.extend(abs(score(x[hold], fit(x[~hold],y[~hold],a))-y[hold]))
                    cv[a] = float(np.mean(errors))
                alpha = min(cv, key=cv.get)
                fitted = fit(x,y,alpha)
                fits[mode][endpoint] = dict(alpha=alpha, cv=cv, **fitted)
                v = score(xt,fitted)
            for k, estimate in zip(testkeys, np.exp(v), strict=True):
                predicted[k].setdefault(mode,{})[endpoint] = float(estimate)
    return dict(protocol=sha(out/"protocol.json"), development_features=sha(dev_file), test_features=sha(test_file),
                training={f"p{i}":sha(out/f"development/p{i}.zip") for i in range(PROCESSES["development"])},
                development_valid=len(keys), development_rejected=rejected, fits=fits, predictions=predicted)


def predict(out):
    p = out/"predictions.json"
    if p.exists():
        raise FileExistsError(p)
    dump(p, estimates(out))
