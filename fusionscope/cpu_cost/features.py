"""Source estimates, typed pre-optimization IR and a postcompile rewrite trigger."""
from collections import Counter
from math import prod
import re

from jax.extend.mlir import ir

from fusionscope.hlo import Module, ancestors
from .workloads import CONTRACTS, axes, site_shapes


def source(case, name):
    shape = case.shape
    n, w = prod(shape), shape[-1]
    zs = site_shapes(case)
    main = prod(zs[-1])
    dot = {"M": 6*n*w, "G": 12*n*w}.get(case.family, 0)
    conv = {"C": 54*n*w, "R": 72*n*w, "B": 2*w*(n+9*prod(zs[1])+2*main)}.get(case.family, 0)
    out = Counter(input=n, rows=n//w, width=w, rank=len(shape), main=main, dot=dot, conv=conv,
                  norm=n*3 if case.family == "N" else 0, shortcut=n if case.readout == "branch" else 0)
    if name == "V0":
        return dict(out)
    c, schedule = name.split("_")
    spec = CONTRACTS[c]
    out["joint"] = int(schedule == "J")
    out["measurements"] = len(spec)
    out["distinct_sites"] = len({s for s,_,_ in spec})
    for s, kind, which in spec:
        elements = prod(zs[s])
        out[kind] += elements
        out["aux_outputs"] += (elements if kind == "full" else 1) if which == "none" else (
            elements//prod(zs[s][i] for i in axes(which, zs[s])))
        if which != "none":
            out["joint_domain" if schedule == "J" else "separate_domain"] += elements
    return dict(out)


def shape(value):
    return tuple(ir.RankedTensorType(value.type).shape)


def size(value):
    return prod(shape(value))


def stable_features(module):
    from jaxlib.mlir.dialects import stablehlo
    counts = Counter()

    def visit(view):
        op = view.operation
        k = str(op.name).removeprefix("stablehlo.")
        if str(op.name).startswith("stablehlo.") and k != "return":
            counts["ops"] += 1
            volume = sum(size(v) for v in op.results)
            counts["op_"+k] += 1
            counts["volume_"+k] += volume
            if k == "dot_general":
                dims = stablehlo.DotDimensionNumbers(op.attributes["dot_dimension_numbers"])
                counts["dot_flops"] += 2*volume*prod(shape(op.operands[0])[i] for i in dims.lhs_contracting_dimensions)
            elif k == "convolution":
                dims = stablehlo.ConvDimensionNumbers(op.attributes["dimension_numbers"])
                kernel = shape(op.operands[1])
                counts["conv_flops"] += 2*volume*prod(v for i,v in enumerate(kernel)
                                                     if i != dims.kernel_output_feature_dimension)
            elif k == "reduce":
                lanes = len(op.results)
                domain = sum(size(v) for v in list(op.operands)[:lanes])
                counts["variadic_domain" if lanes > 1 else "unary_domain"] += domain
                counts["accumulators"] += lanes
            elif k == "convert":
                counts["int_convert"] += sum(size(v) for v in op.results if str(v.type).endswith("xi32>"))
        if str(op.name) == "func.func" and str(op.attributes["sym_name"]) == '"main"':
            block = op.regions[0].blocks[0]
            counts["inputs"] = sum(size(v) for v in block.arguments)
            counts["outputs"] = sum(size(v) for v in list(block.operations)[-1].operands)
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child)
    visit(module)
    return dict(counts)


def trigger(text, case, contract):
    module = Module(text)
    shapes = {site_shapes(case)[s] for s,_,_ in CONTRACTS[contract]}
    auxiliary = set().union(*(ancestors(v) for v in module.outputs[1:])) - module.main
    hits = []
    for v in module.exports & auxiliary:
        match = re.match(r"(?:s32|pred|f32)\[([0-9,]+)\]", v.op.shape)
        if not match or tuple(map(int, match[1].split(","))) not in shapes:
            continue
        mask = v.op.kind == "convert" and v.parents and v.parents[0].op.kind == "compare"
        square = v.op.kind == "multiply" and len(v.parents) == 2 and v.parents[0] is v.parents[1]
        if mask or v.op.kind == "compare" or square:
            hits.append(v.anchor())
    return sorted(hits), module.fingerprint()
