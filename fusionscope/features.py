"""Features observable from an uninstrumented program and its compiled baseline."""
from collections import Counter, defaultdict, deque
import math
from time import perf_counter_ns

from .common import DOUBLES, MODES, ROOT, SPEC, dump, sha
from .hlo import Module


def source_features(closed):
    eqs = closed.jaxpr.eqns
    ops = Counter(e.primitive.name for e in eqs)
    uses = defaultdict(set)
    scopes = []
    for index,e in enumerate(eqs):
        scopes.append(str(e.source_info.name_stack).split("/"))
        for var in e.invars:
            if not hasattr(var,"val"):
                uses[var].add(index)
    site_vars, indices = {}, {}
    for site in ("u","v"):
        ix = [i for i in range(len(eqs)) if f"{site}_relu" in scopes[i]]
        if len(ix)!=1:
            raise ValueError(f"Expected one {site}_relu annotation in the baseline Jaxpr, found {ix}")
        i=indices[site]=ix[0]
        e=eqs[i]
        site_vars[site]={"pre":next(v for v in e.invars if not hasattr(v,"val")), "post":e.outvars[0]}
    distance = {indices["u"]:0}
    todo = deque([indices["u"]])
    while todo:
        i=todo.popleft()
        for v in eqs[i].outvars:
            for k in uses[v]:
                if k not in distance:
                    distance[k]=distance[i]+1
                    todo.append(k)
    facts={}
    for site,vs in site_vars.items():
        facts[site]={}
        for side,var in vs.items():
            shape=var.aval.shape
            facts[site][side]={"log2_elements":math.log2(math.prod(shape)), "rank":len(shape),
                    "source_users":len(uses[var])+int(var in closed.jaxpr.outvars),
                    "shape":list(shape),"source_eqn":indices[site]+1}
    return {"sites":facts, "dot_count":ops["dot_general"], "conv_count":ops["conv_general_dilated"],
            "reduce_count":sum(v for k,v in ops.items() if k.startswith("reduce")),
            "source_distance":distance.get(indices["v"],-1)}


def blocks(source, graph):
    result={}
    for mode in DOUBLES:
        u,v=mode.split("_")
        byte={"scalar_count":2, "auxiliary_bytes":8}
        src={"dot_count":source["dot_count"],"conv_count":source["conv_count"],
             "reduce_count":source["reduce_count"],"source_distance":source["source_distance"]}
        per_ir={}
        for site,side in (("u",u),("v",v)):
            sf=source["sites"][site][side]
            byte.update({f"{site}_log2_elements":sf["log2_elements"],f"{site}_rank":sf["rank"]})
            src.update({f"{site}_post":int(side=="post"),f"{site}_source_users":sf["source_users"]})
            per_ir.update({f"{site}_{k}":value for k,value in graph.site_features(site,side).items()})
        pair=graph.pair_features(u,v)
        pair["both_post_same_fusion"]=int(u==v=="post")*pair["same_fusion"]
        result[mode]={"B1":byte,"B2":src,"M1":per_ir,"M2":pair}
    return result


def extract(fn, args, ir_dir=None):
    import jax
    start=perf_counter_ns()
    jitted=jax.jit(fn)
    lowered=jitted.lower(*args)
    compiled=lowered.compile()
    compiled_ns=perf_counter_ns()-start
    start=perf_counter_ns()
    traced=jax.make_jaxpr(fn)(*args)
    source=source_features(traced)
    hlo=compiled.as_text()
    graph=Module(hlo)
    features=blocks(source,graph)
    feature_ns=perf_counter_ns()-start
    if ir_dir is not None:
        from .bench import write_ir
        dump(ir_dir/"hashes.json",write_ir(ir_dir,fn,args,lowered,compiled))
    return {"source":source,"features":features,"fallback":any(graph.relu(s) is None for s in ("u","v")),
            "baseline_lower_compile_ns":compiled_ns,"feature_extraction_ns":feature_ns}


def export_baselines(root,cases):
    import jax
    from .graphs import inputs,program
    for case in cases:
        folder=root/"baselines"/case
        if (folder/"features.json").exists():
            continue
        args=inputs(case,SPEC["timed_seed"])
        jax.block_until_ready(args)
        facts=extract(program(case,"base"),args,folder)
        dump(folder/"features.json",facts)
        print(f"baseline features {case}, fallback={facts['fallback']}",flush=True)
        jax.clear_caches()
