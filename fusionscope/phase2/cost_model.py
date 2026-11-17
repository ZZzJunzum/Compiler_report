"""Baseline-only analytic and ridge estimates for exact-observation strategies."""
from datetime import datetime, timezone
import math
from statistics import median

import numpy as np
from scipy.optimize import least_squares

from fusionscope.common import ROOT, conditions, describe, dump, read, sha, table
from fusionscope.hlo import Module, ancestors
from .audit_ir import ARCHIVE, PLAN, dims
from .stats import cohort


def size(shape):
    dimension=dims(shape)
    return math.prod(dimension) if dimension is not None else 0


def work(node):
    kind=node.op.kind
    n=size(node.op.shape)
    if kind=="dot":
        # All registered dense layers contract the innermost activation dimension.
        return 2*n*dims(node.parents[0].op.shape)[-1]
    if kind=="convolution":
        return 2*n*math.prod(dims(node.parents[1].op.shape)[:-1])
    if kind in ("reduce","reduce-window") and node.parents:
        return max(0,size(node.parents[0].op.shape)-n)
    return n if kind in ("maximum","add","subtract","multiply","divide","rsqrt","sqrt") else 0


def prefix(node):
    todo=[node]
    found=set()
    while todo:
        n=todo.pop()
        if n not in found and n.group==node.group:
            found.add(n)
            todo.extend(n.parents)
    return {n for n in found if work(n)>0}


def baseline_facts(case, *, hlo=None):
    path=ROOT/"results/cpu/baselines"/case/"optimized_hlo.txt"
    if hlo is None:
        text, digest = path.read_text(encoding="utf8"), sha(path)
    else:
        import hashlib
        text, digest = hlo, hashlib.sha256(hlo.encode("utf8")).hexdigest()
    graph=Module(text)
    main=graph.main
    main_flops=sum(work(n) for n in main)
    inputs=[op for op in graph.comps[graph.entry] if op.kind=="parameter"]
    base_bytes=4*(sum(size(op.shape) for op in inputs)+size(graph.outputs[0].op.shape))
    candidates=[graph.relu_candidates(s) for s in ("u","v")]
    chosen=[]
    for anchors in candidates:
        # Explicit policy for duplicate baseline ReLUs: the largest local producer slice.
        sets=[prefix(n) for n in sorted(anchors,key=lambda n:(n.op.line,n.group))]
        chosen.append(max(sets,key=lambda nodes:sum(work(n) for n in nodes),default=set()))
    u,v=chosen
    return dict(base_work=main_flops,base_bytes=base_bytes,
                u_work=sum(work(n) for n in u),v_work=sum(work(n) for n in v),
                shared_work=sum(work(n) for n in u&v),shared_prefix_ops=len(u&v),
                producer_ops=len(u)+len(v),ambiguous_sites=sum(len(a)!=1 for a in candidates),
                fanout=sum(len(set(graph.children[n])) for seq in candidates for n in seq),
                dots=sum(n.op.kind=="dot" for n in main),convs=sum(n.op.kind=="convolution" for n in main),
                base_reductions=sum(n.op.kind.startswith("reduce") for n in main),
                baseline_hlo_sha256=digest)


def features(case,variant,facts,*,shape=None):
    if shape is None:
        _,shape,_=describe(case)
    n=math.prod(shape)
    tiled=variant.startswith("V4")
    tile=int(variant[4:]) if tiled else 0
    tiles=shape[0]//tile if tile else 1
    v0,v1,v2,v3=[int(variant==v) for v in ("V0","V1","V2","V3")]
    observed=1-v0
    dup=observed*(facts["u_work"]+facts["v_work"])-(v2+v3)*facts["shared_work"]
    mask_capacity=8*n/tiles if v1 or tiled else 4*n if v3 else 0
    material=4*n*v3
    partial=8*tiles if tiled else 0
    red_levels=0 if v0 else (1 if v2 else 2)*max(1,math.ceil(math.log(max(n/tiles,2),32)))
    if tiled and tiles>1:
        red_levels+=2*math.ceil(math.log(tiles,32))
    return dict(log_elements=math.log2(n),rank=len(shape),producer_ops=facts["producer_ops"]*observed,
                prefix_overlap=facts["shared_prefix_ops"],fanout=facts["fanout"],
                ambiguous_sites=facts["ambiguous_sites"],dot_sites=facts["dots"],conv_sites=facts["convs"],
                base_reduce_sites=facts["base_reductions"],v1=v1,v2=v2,v3=v3,v4=int(tiled),
                rows_per_tile=tile,tile_count=tiles,host_calls=1+v3,loop_steps=tiles-1,
                mask_capacity_bytes=mask_capacity,stage_material_bytes=material,partial_bytes=partial,
                compare_elements=2*n*observed,reduction_input_elements=(2*n+2*tiles*int(tiled))*observed,
                reduce_levels=red_levels,dup_flops_est=dup,
                total_mops=(facts["base_work"]+dup+4*n*observed)/1e6,
                transfer_mb=(facts["base_bytes"]+2*(mask_capacity+material+partial))/1e6)


def roof(f,coef):
    dispatch,compute,bandwidth,reduction,loop=coef
    return (dispatch*f["host_calls"]+max(compute*f["total_mops"],bandwidth*f["transfer_mb"])
            +reduction*f["reduce_levels"]+loop*f["loop_steps"])


def fit(rows):
    names=sorted(rows[0][0])
    observed=[r for r in rows if r[3]!="V0"]
    def encode(f):
        return [math.log1p(f[k]) for k in names]
    x=np.asarray([encode(f) for f,_,_,_ in observed])
    y=np.log([lat/base for _,lat,base,_ in observed])
    avg=x.mean(0)
    scale=x.std(0)
    scale[scale<1e-10]=1
    x=(x-avg)/scale
    bias=float(np.mean(y))
    weights=np.linalg.solve(x.T@x+np.eye(len(names)),x.T@(y-bias))
    def residual(log_coef):
        coef=np.exp(log_coef)
        return [math.log(roof(f,coef)/lat) for f,lat,_,_ in rows]
    analytic=least_squares(residual,np.log([20,10,50,1,.2]),bounds=(-14,14),max_nfev=500)
    return dict(feature_names=names,mean=avg.tolist(),scale=scale.tolist(),weights=weights.tolist(),
                intercept=bias,alpha=1,analytic_coefficients=np.exp(analytic.x).tolist(),
                analytic_fit_success=bool(analytic.success),number_training_candidates=len(observed))


def predict(model,f):
    x=np.log1p([f[k] for k in model["feature_names"]])
    log_ratio=model["intercept"]+np.dot((x-model["mean"])/model["scale"],model["weights"])
    return roof(f,model["analytic_coefficients"]),math.exp(float(log_ratio))


def verify_lock(out):
    path=out/"model/prediction_lock.json"
    lock=read(path)
    if sha(path)!=(path.parent/"prediction_lock.sha256").read_text().strip():
        raise ValueError("Changed prediction lock")
    for file,digest in lock["inputs"].items():
        if sha(out/file)!=digest:
            raise ValueError(f"Changed model input: {file}")
    if sha(ARCHIVE/"protocol.json")!=lock["protocol_sha256"]:
        raise ValueError("Changed registered protocol")
    return lock


def lock(out):
    from .variants import options
    modeldir=out/"model"
    if (modeldir/"prediction_lock.json").exists():
        return verify_lock(out)
    if any((out/x/"raw").exists() for x in ("confirmation","external")):
        raise ValueError("Independent outcome files already exist without a prediction lock")
    train=cohort(out,"train")
    all_cases=conditions()+PLAN["external"]
    feat={}
    rows=[]
    for case in all_cases:
        base=baseline_facts(case)
        for name in options(case):
            f=features(case,name,base)
            feat[case,name]=f
            rows.append(dict(case=case,variant=name,baseline_hlo_sha256=base["baseline_hlo_sha256"],**f))
    table(modeldir/"features.csv",rows)
    models,policies,preds={}, {}, []
    for fold in [*PLAN["families"],"external"]:
        training=[c for c in conditions() if c[0]!=fold]
        testing=PLAN["external"] if fold=="external" else [c for c in conditions() if c[0]==fold]
        fit_rows=[]
        for c in training:
            for v in options(c):
                if v in train[c]["invalid"]:
                    continue
                p=train[c]["p"]
                # Paired ratios are formed before aggregation. Use the same case baseline to restore units.
                b=train[c]["t"]["V0"]
                fit_rows.append((feat[c,v],b*median(t[v]/t["V0"] for t in p),b,v))
        model=fit(fit_rows)
        models[fold]={"train":training,"test":testing,"parameters":model}
        for c in testing:
            candidates=options(c)[1:]
            vals={v:predict(model,feat[c,v]) for v in candidates}
            fixed_tiles=[v for v in candidates if v.startswith("V4") and int(v[4:])<=16]
            choices={"fixed_V1":"V1","fixed_V2":"V2","fixed_tile":fixed_tiles[-1] if fixed_tiles else "V1",
                     "analytic":min(vals,key=lambda v:vals[v][0]),"ridge":min(vals,key=lambda v:vals[v][1])}
            policies[c]=choices
            for v,(us,ratio) in vals.items():
                preds.append(dict(case=c,fold=fold,variant=v,analytic_us=us,ridge_ratio=ratio))
    dump(modeldir/"model.json",models)
    table(modeldir/"predictions.csv",preds)
    inputs={f"model/{n}":sha(modeldir/n) for n in ("model.json","features.csv","predictions.csv")}
    inputs.update({p.relative_to(out).as_posix():sha(p) for p in sorted((out/"matrix/raw").glob("p*.zip"))})
    data={"locked_utc":datetime.now(timezone.utc).isoformat(),"inputs":inputs,
          "protocol_sha256":sha(ARCHIVE/"protocol.json"),"choices":policies}
    path=modeldir/"prediction_lock.json"
    dump(path,data)
    (modeldir/"prediction_lock.sha256").write_text(sha(path)+"\n",encoding="ascii",newline="\n")
    return data
