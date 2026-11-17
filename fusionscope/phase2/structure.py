"""Inspect every called fusion and map-loop body in each distinct compiled program."""
from collections import defaultdict
import math
import re
from zipfile import ZipFile

from fusionscope.common import describe, dump, table
from fusionscope.hlo import Module, ancestors
from .audit_ir import ARITH, PLAN, dims, expression_keys, source_stage
from .experiment import zip_rows


def inspect(hlo,case,variant):
    graph=Module(hlo)
    ops=[op for body in graph.comps.values() for op in body]
    loops=[op for op in ops if op.kind=="while"]
    # The old parser treats while as an opaque value. Audit its actual body as a separate invocation.
    contexts=[(set.union(*(ancestors(v) for v in graph.outputs)),1)]
    tile=int(variant[4:]) if variant.startswith("V4_t") else 0
    ntiles=describe(case)[1][0]//tile if tile else 0
    for op in loops:
        name=re.search(r"body=%([\w.$-]+)",op.attrs)[1]
        loop_root=graph.expand(name,(),f"/while_L{op.line}",False)
        contexts.append((ancestors(loop_root),ntiles or 1))
    elements=math.prod(describe(case)[1])
    copies=flops=export_bytes=full_bytes=0
    matches=[]
    export_lines=[]
    for nodes,multiplier in contexts:
        signatures=expression_keys(nodes)
        grouped=defaultdict(list)
        for n in sorted(nodes,key=lambda value:(value.op.line,value.group)):
            op=n.op
            if source_stage(op.scope) and op.kind in ARITH:
                grouped[(op.scope,op.kind,op.shape,signatures[n])].append(n)
            if n in graph.exports and op.kind=="convert" and op.shape.startswith("s32[") and (
                    any(p.op.kind=="compare" for p in n.parents)):
                nelem=math.prod(dims(op.shape))
                export_bytes+=4*nelem
                full_bytes+=4*nelem if nelem==elements else 0
                export_lines.append(str(op.line))
        for (_,kind,shape,_),seq in grouped.items():
            extra=len({n.group for n in seq})-1
            copies+=extra
            if extra:
                matches.append("|".join(str(n.op.line) for n in sorted(seq,key=lambda n:n.op.line)))
                if kind in ("add","subtract","multiply","divide"):
                    flops+=multiplier*extra*math.prod(dims(shape))
    reds=[op for op in ops if op.kind in ("reduce","reduce-window")]
    joint=[op for op in reds if op.kind=="reduce" and op.shape.startswith("(")]
    return {"canonical_hlo_sha256":graph.fingerprint(),"static_extra_tagged_ops":copies,"estimated_extra_simple_flops":flops,
            "duplicate_lines":";".join(matches),"source_scope_coverage":"partial_official" if case.startswith("R") else "synthetic",
            "full_exported_s32_mask_bytes":full_bytes,"exported_s32_bytes_per_body_sum":export_bytes,
            "export_lines":"|".join(export_lines),"reduce_sites":len(reds),
            "reduce_window_sites":sum(op.kind=="reduce-window" for op in reds),
            "joint_reduce_lines":"|".join(str(op.line) for op in joint),
            "reduction_lines_and_shapes":";".join(f"L{op.line} {op.kind} {op.shape}" for op in reds),
            "while_lines":"|".join(str(op.line) for op in loops),
            "multioutput_fusion_lines":"|".join(str(op.line) for op in ops if op.kind=="fusion" and op.shape.startswith("(")),
            "planned_tiles":ntiles,"logical_partial_scalars":2*ntiles}


def components(out,archives):
    rows=[]
    for case,file in sorted(archives):
        with ZipFile(out/file) as z:
            for member in sorted(z.namelist()):
                if not member.endswith("optimized_hlo.txt"):
                    continue
                variant,part,_=member.split("/")
                ir=z.read(member).decode()
                rows.append(dict(case=case,variant=variant,component=part,ir_archive=file,member=member,
                                 **inspect(ir,case,variant)))
    baseline={r["ir_archive"]:r["canonical_hlo_sha256"] for r in rows if r["variant"]=="V1"}
    for r in rows:
        r["same_single_executable_as_V1"]=(r["component"]=="main" and
                                           r["canonical_hlo_sha256"]==baseline[r["ir_archive"]])
    return rows


def screen_summary(out):
    folder=out/"screening"
    raw=sorted((folder/"raw").glob("p*.zip"))
    if len(raw)!=PLAN["processes"]["screen"]:
        raise ValueError("Incomplete screening")
    archives={(r["case"],r["ir_archive"]) for p in raw for r in zip_rows(p,"builds.csv")}
    rows=components(out,archives)
    table(folder/"structure.csv",rows)
    checks=[r for p in raw for r in zip_rows(p,"correctness.csv")]
    invalid={(r["case"],r["variant"]) for r in checks if r["valid"]!="True"}
    evidence=[dict(case=r["case"],variant=r["variant"],joint_reduce_lines=r["joint_reduce_lines"],
                   ir_archive=r["ir_archive"]) for r in rows if r["variant"]=="V2" and
              r["joint_reduce_lines"] and (r["case"],"V2") not in invalid]
    dump(folder/"gate.json",{"structural_change":bool(evidence),"evidence":evidence,
                             "invalid":sorted(invalid),"checked_config_seed_process_rows":len(checks)})
