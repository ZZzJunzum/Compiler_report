"""Three prespecified regressions for the two-schedule CPU convolution test."""
from collections import Counter
from math import log, log2, prod
from statistics import median
from zipfile import ZipFile

import numpy as np

from fusionscope.headroom import ARCHIVE
from fusionscope.hlo import Module, ancestors
from fusionscope.phase3.data import cohort
from fusionscope.phase3.design import development


MODELS = ("shape", "source", "candidate_ir")
OPCODES = ("fusion", "reduce", "while", "convolution", "custom-call")


def ir_facts(text, temp_bytes, shape):
    module = Module(text)
    counts = Counter(op.kind for comp in module.comps.values() for op in comp)
    return [log2(1 + temp_bytes/(4*prod(shape))), *[log2(1+counts[k]) for k in OPCODES]]


def skip_operand(text):
    """Trace the named skip multiply on the direct main-output path."""
    m = Module(text)
    hits = [v for v in ancestors(m.outputs[0]) if v.op.kind=="multiply" and "audit_skip" in v.op.scope]
    if len(hits) == 1:
        hit = hits[0]
        dynamic = [p for p in hit.parents if not all(n.op.kind in ("constant","broadcast")
                                                      for n in ancestors(p))]
        if len(dynamic) == 1:
            parent = dynamic[0]
            views = {"bitcast", "copy", "reshape"}
            while parent.op.kind in views and len(parent.parents)==1:
                parent=parent.parents[0]
            if parent.op.kind=="maximum" and "u_relu" in parent.op.scope:
                return dict(origin="first_relu", multiply_line=hit.op.line, operand_line=parent.op.line)
            if parent.op.kind=="parameter" and parent.op.operands=="0":
                return dict(origin="input", multiply_line=hit.op.line, operand_line=parent.op.line)
    return dict(origin="unresolved", multiply_line=hits[0].op.line if len(hits)==1 else None,
                operand_line=None)


def vector(kind, shape, shared, differences=None):
    b,h,w,c = shape
    batch = [float(b==n) for n in (1,4,8,16)]
    result = [*batch, log2(h*w), log2(c)]
    if kind in ("source", "candidate_ir"):
        result.extend((float(shared), *[float(shared)*x for x in batch]))
    if kind == "candidate_ir":
        result.extend(differences)
    return result


def training_data():
    data = cohort(ARCHIVE, "development")
    rows = []
    for c in development():
        if c.ops != "CC":
            continue
        d = data[c.name]
        if {"V2","J1"} & d["invalid"]:
            raise ValueError(f"Invalid training pair: {c.name}")
        facts = {}
        for v in ("V2","J1"):
            r = next(r for r in d["builds"] if r["variant"]==v and r["process"]==0)
            with ZipFile(ARCHIVE/r["archive"]) as z:
                text = z.read(f"{v}/{r['component']}/optimized_hlo.txt").decode()
            facts[v] = ir_facts(text, int(r["temp_bytes"]), c.shape)
        ratios = [log(row["J1"]/row["V2"]) for row in d["processes"]]
        rows.append(dict(case=c.name, shape=c.shape, shared=c.top=="branch", target=median(ratios),
                         differences=[a-b for a,b in zip(facts["J1"],facts["V2"])]))
    return rows


def fit(rows, kind):
    x=np.asarray([vector(kind,r["shape"],r["shared"],r["differences"]) for r in rows])
    y=np.asarray([r["target"] for r in rows])
    center=x.mean(axis=0)
    scale=x.std(axis=0)
    scale[scale < 1e-9] = 1.
    x=(x-center)/scale
    coef=np.linalg.solve(x.T@x + np.eye(x.shape[1]), x.T@(y-y.mean()))
    return dict(center=center.tolist(), scale=scale.tolist(), weights=coef.tolist(), intercept=float(y.mean()))


def decide(policy, models, case, differences=None):
    if policy == "cap16":
        return "V2"
    if policy == "exception":
        return "J1" if case.shape[0] in (4,8) else "V2"
    v=vector(policy,case.shape,case.top=="shared",differences)
    m=models[policy]
    prediction=m["intercept"] + sum(w*(x-u)/s for w,x,u,s in zip(m["weights"],v,m["center"],m["scale"]))
    return "J1" if prediction < 0 else "V2"
