"""Trace phase-I outputs to distinct HLO values and capture a matching CPU dump."""
import argparse
from collections import defaultdict
from functools import cache
import hashlib
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from fusionscope.common import ROOT, describe, dump, read, sha, table
from fusionscope.hlo import Module, REF, ancestors

ARCHIVE = ROOT / "results/phase2"
PLAN = read(ARCHIVE / "protocol.json")
ARITH = {"add", "subtract", "multiply", "divide", "maximum", "minimum", "rsqrt",
         "sqrt", "dot", "convolution", "reduce", "reduce-window"}


def dims(shape):
    match = re.match(r"[\w]+\[([0-9,]*)\]", shape)
    return tuple(int(v) for v in match[1].split(",") if v) if match else None


def source_stage(scope):
    return next((s for s in ("u_z","u_relu","v_z","v_relu") if f"/{s}/" in scope), "")


def expression_keys(nodes):
    @cache
    def signature(node):
        op = node.op
        key = (op.kind, op.shape, REF.sub("%", op.operands), op.attrs,
               tuple(signature(p) for p in node.parents))
        return hashlib.sha256(repr(key).encode()).hexdigest()
    return {n: signature(n) for n in nodes}


def audit_case(case, mode):
    path = ROOT / "results/cpu/matrix/ir" / case / mode / "optimized_hlo.txt"
    graph = Module(path.read_text(encoding="utf8"))
    paths = {label: ancestors(v) for label,v in zip(("y","q1","q2"),graph.outputs)}
    nodes = set.union(*paths.values())
    keys = expression_keys(nodes)
    owner = {op.line:name for name,ops in graph.comps.items() for op in ops}
    common = {"case":case, "mode":mode, "hlo":path.relative_to(ROOT).as_posix()}
    path_rows, reductions, masks, duplicates = [],[],[],[]
    grouped = defaultdict(list)
    for node in sorted(nodes,key=lambda n:(n.op.line,n.group)):
        op = node.op
        labels = [p for p, reachable in paths.items() if node in reachable]
        if op.kind in ARITH and source_stage(op.scope):
            grouped[op.scope,op.kind,op.shape,keys[node]].append(node)
        for label in labels:
            path_rows.append(dict(**common, output=label, body=owner[op.line], call_path=node.group,
                                  line=op.line, opcode=op.kind, shape=op.shape, source=op.scope,
                                  dependencies=" ".join(str(v.op.line) for v in node.parents)))
            if label!="y" and op.kind in ("compare","convert","reduce","reduce-window"):
                reductions.append(dict(**common,output=label,line=op.line,body=owner[op.line],opcode=op.kind,
                                       input_shapes=" | ".join(v.op.shape for v in node.parents),
                                       output_shape=op.shape,source=op.scope,call_path=node.group))
        if node in graph.exports and op.kind in ("compare","convert") and any(p!="y" for p in labels):
            shape = dims(op.shape)
            if shape is not None and op.shape.startswith("s32["):
                masks.append(dict(**common,outputs="|".join(labels),line=op.line,body=owner[op.line],
                                  shape=op.shape,logical_bytes=4*math.prod(shape),
                                  full_activation_shape=shape==describe(case)[1],call_path=node.group))
    for (scope,kind,shape,key),copies in grouped.items():
        groups = {n.group for n in copies}
        extra = len(groups)-1
        # FLOP proxy counts simple add/multiply/divide/subtract only; comparisons and rsqrt stay separate.
        flops = extra*math.prod(dims(shape)) if kind in ("add","multiply","subtract","divide") else 0
        duplicates.append(dict(**common,stage=source_stage(scope),source=scope,opcode=kind,shape=shape,
                               expression_sha256=key,distinct_groups=len(groups),extra_copies=extra,
                               extra_elementary_flops=flops,lines="|".join(str(n.op.line) for n in copies),
                               paths="; ".join(f"{n.group}="+",".join(p for p,s in paths.items() if n in s)
                                               for n in copies)))
    return path_rows,duplicates,reductions,masks


def capture(out):
    from run import pin
    pin()
    import jax
    from fusionscope.graphs import inputs, program
    args = jax.device_put(inputs("E3_branch",42),jax.devices("cpu")[0])
    exe = jax.jit(program("E3_branch","post_post")).lower(*args).compile()
    target = ROOT/"results/cpu/matrix/ir/E3_branch/post_post/optimized_hlo.txt"
    actual = Module(exe.as_text()).fingerprint()
    if actual != Module(target.read_text(encoding="utf8")).fingerprint():
        raise ValueError("CPU dump does not match the archived optimized computation")
    out.mkdir(parents=True,exist_ok=True)
    files = list(Path("dump").glob("*jit_workload*"))
    kept = {}
    for p in files:
        if p.name.endswith(("buffer-assignment.txt","memory-usage-report.txt")) or (
                "compare_convert_fusion" in p.name and p.name.endswith("ir-with-opt.ll")):
            name = p.name.split("jit_workload.",1)[1]
            (out/name).write_bytes(p.read_bytes().replace(b"\r\n",b"\n"))
            kept[name] = sha(out/name)
    if not any("buffer-assignment" in p for p in kept):
        raise ValueError("Missing buffer assignment")
    info = exe.memory_analysis()
    dump(out/"manifest.json",{"hlo_signature":actual,"files":kept,"temp_bytes":info.temp_size_in_bytes,
                              "argument_bytes":info.argument_size_in_bytes,"output_bytes":info.output_size_in_bytes})


def figure(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11.5,3.8),layout="constrained")
    ax.set(xlim=(-.72,6.4),ylim=(-.62,2.67))
    ax.axis("off")
    for i,(path,stages,color) in enumerate((
            ("y",["u_z/add","u_relu/max","v_z/add","v_relu/max","readout/add"],"#196579"),
            ("q1",["u_z/add","probe_u/gt","probe_u/convert_element_type"],"#a84c24"),
            ("q2",["u_z/add","u_relu/max","v_z/add","probe_v/gt","probe_v/convert_element_type"],"#604691"))):
        selected=[r for r in rows if r["case"]=="E3_branch" and r["mode"]=="post_post" and r["output"]==path]
        end = max((r for r in selected if r["opcode"]=="reduce"),key=lambda r:r["line"],default=None)
        col=0
        for stage in stages:
            hits=[r for r in selected if r["source"].endswith("/"+stage) and r["opcode"] in ARITH|{"compare","convert"}]
            if not hits: continue
            node=hits[-1]
            ax.text(col,2-i,stage.split("/")[0]+"\n"+node["opcode"]+" · L"+str(node["line"]),ha="center",va="center",
                    fontsize=9,bbox=dict(boxstyle="round,pad=.45",fc="white",ec=color))
            if col: ax.annotate("",xy=(col-.35,2-i),xytext=(col-.65,2-i),arrowprops=dict(arrowstyle="->",color=color))
            col+=1
        if end:
            ax.text(col+.3,2-i,"partial sums → count\nreduce · L"+str(end["line"]),ha="center",va="center",fontsize=9,color=color)
        ax.text(-.66,2-i,path,weight="bold",color=color)
    ax.set_title("E3 branch | Selected instructions on the three CPU HLO output paths",loc="left",weight="bold")
    fig.savefig(out,dpi=150)
    plt.close(fig)


def run(out):
    from run import worker_env
    dest=out/"analysis"
    tables=[[],[],[],[]]
    for case in PLAN["screen_cases"]:
        for mode in ("base","post_post"):
            for acc,rows in zip(tables,audit_case(case,mode)):
                acc.extend(rows)
    for name,rows in zip(("fusion_paths","duplicated_ops","reduction_structure","intermediate_bytes"),tables):
        table(dest/(name+".csv"),rows)
    (out/"figures").mkdir(parents=True,exist_ok=True)
    figure(out/"figures/phase0-paths.png",tables[0])
    with tempfile.TemporaryDirectory(prefix="fsxla-") as tmp:
        env=worker_env("cpu")
        env.update(XLA_FLAGS="--xla_dump_to=dump --xla_dump_hlo_as_text=true",PYTHONPATH=str(ROOT))
        subprocess.run([sys.executable,"-m","fusionscope.phase2.audit_ir","--capture","--out",
                        str((out/"ir/diagnostic/E3_branch").resolve())],cwd=tmp,env=env,check=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=ARCHIVE)
    ap.add_argument("--capture",action="store_true")
    opts=ap.parse_args()
    (capture if opts.capture else run)(opts.out)
