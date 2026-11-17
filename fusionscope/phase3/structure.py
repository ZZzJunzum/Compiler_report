"""Read candidate IR after confirmation. Preserve loop-body and output-path evidence."""
from collections import defaultdict
from functools import cache
from math import prod
import re
from statistics import median
from time import perf_counter_ns
from zipfile import ZipFile

from fusionscope.common import dump, table
from fusionscope.hlo import Module, ancestors
from fusionscope.phase2.structure import inspect
from .data import direction
from .design import MECHANISM, development, heldout


def mechanism(out, cohort):
    summary, contrasts, paths, detail = [], [], [], []
    for c in development():
        if c.name not in MECHANISM:
            continue
        d = cohort[c.name]
        for t in (0, *MECHANISM[c.name]):
            labels = [f"{s}{t}" for s in "BSJ"] if t else ["V0","V1","V2"]
            b,s,j = labels
            sep = [p[s]-p[b] for p in d["processes"]]
            joint = [p[j]-p[b] for p in d["processes"]]
            delta = [p[s]-p[j]-(p["V1"]-p["V2"]) for p in d["processes"]]
            med = median(delta)
            signs = sum(x>0 for x in delta)
            contrasts.append(dict(case=c.name,extent=t or "whole",separate_overhead_us=median(sep),
                 joint_overhead_us=median(joint),separate_minus_joint_us=median(x-y for x,y in zip(sep,joint)),
                 interaction_us=med,interaction_floor_us=d["interaction_floor"],positive_processes=signs,
                 resolved=bool(abs(med)>d["interaction_floor"] and sum(x*med>0 for x in delta)>len(delta)/2),
                 process_interactions=str(delta)))
            for v in labels:
                builds = [r for r in d["builds"] if r["variant"]==v]
                triples = {(r["archive"],r["component"]) for r in builds}
                for archive,part in sorted(triples):
                    member = f"{v}/{part}/optimized_hlo.txt"
                    with ZipFile(out/archive) as z:
                        hlo = z.read(member).decode()
                    graph = Module(hlo)
                    ops = [op for comp in graph.comps.values() for op in comp]
                    alias = f"V4_t{t}" if t else v
                    facts = inspect(hlo,c.name,alias)
                    trips = c.shape[0]//t if t else 1
                    obs = v not in ("V0",) and not v.startswith("B")
                    @cache
                    def depth(node):
                        return int(node.op.kind in ("reduce","reduce-window"))+max((depth(p) for p in node.parents),default=0)
                    roots=list(graph.outputs)
                    loop_depths=[]
                    for op in ops:
                        if op.kind=="while":
                            body=re.search(r"body=%([\w.$-]+)",op.attrs)[1]
                            loop_depths.append(depth(graph.expand(body,(),f"/while_L{op.line}",False)))
                    summary.append(dict(case=c.name,variant=v,extent=t or "whole",us=d["us"][v],
                         floor_us=d["floor"],temp_bytes=int(builds[0]["temp_bytes"]),
                         trips=trips,domain_per_body=prod(c.shape)//trips,
                         main_and_mask_live_proxy_bytes=(12+(8 if s==v else 0))*prod(c.shape)//trips,
                         entry_path_reduction_depth=max(map(depth,roots)),max_loop_body_reduction_depth=max(loop_depths,default=0),
                         while_sites=len(loop_depths),host_executable_calls=1,
                         count_element_visits=2*prod(c.shape) if obs else 0,
                         exported_mask_bytes_over_trips=facts["exported_s32_bytes_per_body_sum"]*trips,
                         max_body_exported_mask_bytes=facts["exported_s32_bytes_per_body_sum"],
                         logical_partial_bytes=8*trips if obs and t else 0,
                         archive=archive,member=member,**facts))
                    # Entry output roots can point into a while tuple. Enumerate the actual body update sites.
                    updates = [op for op in ops if op.kind in ("dynamic-update-slice","dynamic-slice","while")]
                    for i,name in enumerate(("y","q1","q2")):
                        nodes = ancestors(graph.outputs[i]) if i<len(graph.outputs) else set()
                        tagged = sorted({n.op.line for n in nodes if n.op.scope and n.op.kind in
                                        ("add","maximum","reduce","reduce-window","while","get-tuple-element")})
                        paths.append(dict(case=c.name,variant=v,path=name,archive=archive,member=member,
                             entry_or_anchor=graph.outputs[i].anchor() if nodes else "absent",
                             HLO_lines="|".join(map(str,tagged)),loop_slice_and_update_lines="|".join(str(op.line) for op in updates)))
                    for name,seq in (("first producer",[op for op in ops if "/u_z/" in op.scope]),
                                     ("mask or partial",[op for op in ops if op.kind in ("reduce","reduce-window","convert")])):
                        paths.append(dict(case=c.name,variant=v,path=name,archive=archive,member=member,
                             entry_or_anchor="see listed instructions" if seq else "absent",
                             HLO_lines="|".join(str(op.line) for op in seq),loop_slice_and_update_lines=""))
                    for op in ops:
                        if op.scope or op.kind=="while":
                            line = hlo.splitlines()[op.line-1]
                            loc = re.search(r'source_file="([^"]*)"(?: source_line=(\d+))?',line)
                            detail.append(dict(case=c.name,variant=v,archive=archive,member=member,HLO_line=op.line,
                                 scope=op.scope,operation=op.kind,shape=op.shape,
                                 source_file=loc[1] if loc else "",source_line=loc[2] if loc else ""))
    folder=out/"analysis"
    for name,rows in (("mechanism",summary),("interactions",contrasts),("paths",paths),("instructions",detail)):
        table(folder/(name+".csv"),rows)
    return contrasts


def equivalence(out, held):
    groups, pairs = [], []
    classes, accounting = {}, {}
    for c in heldout():
        d = held[c.name]
        by_variant = defaultdict(lambda: defaultdict(list))
        seen = {}
        diagnostic_ns = 0
        for r in d["builds"]:
            key = (r["archive"],r["variant"],r["component"])
            if key not in seen:
                start = perf_counter_ns()
                with ZipFile(out/key[0]) as z:
                    seen[key] = Module(z.read(f"{key[1]}/{key[2]}/optimized_hlo.txt").decode()).fingerprint()
                if key[1] != "V0":
                    diagnostic_ns += perf_counter_ns()-start
            by_variant[r["variant"]][int(r["process"])].append((r["component"],seen[key]))
        ids = {}
        for v,runs in by_variant.items():
            fingerprints = {tuple(sorted(parts)) for parts in runs.values()}
            ids[v] = tuple(sorted(fingerprints))
            groups.append(dict(case=c.name,variant=v,distinct_programs=len(fingerprints),
                               process_fingerprints=str(ids[v]),archive=next(r["archive"] for r in d["builds"] if r["variant"]==v)))
        classes[c.name] = ids
        accounting[c.name] = diagnostic_ns/1000
        for v,ref in ((f"S{c.shape[0]}","V1"),(f"J{c.shape[0]}","V2")):
            delta,resolved = direction(d,v,ref)
            pairs.append(dict(case=c.name,tile=v,direct=ref,equal_hlo=ids[v]==ids[ref] and len(ids[v])==1,
                              paired_direct_minus_tile_us=delta,floor_us=d["floor"],resolved=resolved))
    table(out/"analysis/equivalence.csv",pairs)
    table(out/"analysis/fingerprints.csv",groups)
    if not (out/"accounting/ir_inspection_us.json").exists():
        dump(out/"accounting/ir_inspection_us.json",accounting)
    return classes,pairs
