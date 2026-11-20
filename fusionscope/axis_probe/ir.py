"""Index saved backend evidence and verify the historical and timed HLO identities."""
import hashlib
import json
import re
from zipfile import ZipFile

from fusionscope.common import ROOT, read
from fusionscope.evidence import scheduled_text_digest
from fusionscope.hlo import Module
from fusionscope.phase2.experiment import zip_rows
from .design import AXIS_GRAPHS, TASK_GRAPHS, VARIANTS, case_for, compiler_options, design_lock


def kernel_record(file, code):
    # Distinguish instruction sites from declarations and from dynamic execution counts.
    widths = sorted({int(x) for x in re.findall(r"fcmp \w+ <(\d+) x float>", code)})
    sums = re.findall(r"call i32 @llvm\.vector\.reduce\.add\.v(\d+)i32", code)
    return dict(kernel=file.removesuffix("_kernel_module.ir-with-opt.ll"),
                float_compare_lanes="/".join(map(str, widths)),
                horizontal_i32_calls=len(sums), horizontal_i32_lanes="/".join(sorted(set(sums))),
                vector_i32_phi=len(re.findall(r"= phi <\d+ x i32>", code)))


def inspect(out):
    from fusionscope.cpu_cost.features import trigger
    histories = [zip_rows(ROOT/f"results/cpu_cost/test/p{i}.zip", "builds.csv") for i in range(5)]
    historical = [{(r["case"], r["variant"]): r["fingerprint"] for r in rows} for rows in histories]
    ir, structures, kernels = {}, [], []
    for folder in sorted((out/"ir").glob("cpu*")):
        width = int(folder.name.removeprefix("cpu"))
        meta = read(folder/"environment.json")
        if meta["lock"] != design_lock() or len(meta["environment"]["affinity"]) != width:
            raise ValueError(folder)
        env = meta["environment"]
        ir[width] = {"environment": env}
        for archive in sorted(folder.glob("*/*.zip")):
            with ZipFile(archive) as z:
                if z.testzip() is not None:
                    raise ValueError(archive)
                ref = json.loads(z.read("identity.json"))
                g, v, opt = (ref[k] for k in ("graph", "variant", "setting"))
                if (archive.parent.name != g or archive.stem != f"{v}_{opt}"
                        or ref["compiler_options"] != compiler_options(opt)):
                    raise ValueError(archive)
                hlo = z.read("cpu_after_optimizations.txt").decode()
                mod = Module(hlo)
                if mod.fingerprint() != ref["fingerprint"] or scheduled_text_digest(hlo) != ref["named_schedule"]:
                    raise ValueError(f"Wrong identities {archive}")
                ir[width][g, v, opt] = ref
                check_old = g in AXIS_GRAPHS and opt == "on" and env["jaxlib"] == "0.9.0.1" and width == 1
                historical_match = all(p[g, v] == ref["fingerprint"] for p in historical) if check_old else ""
                if check_old and not historical_match:
                    raise ValueError(f"Historical fingerprint mismatch {archive}")
                base = dict(cpus=width, graph=g, variant=v, setting=opt)
                members = z.namelist()
                for suffix in ("buffer-assignment.txt", "before_optimizations.txt", "input.stablehlo"):
                    if not any(n.endswith(suffix) for n in members):
                        raise ValueError((archive, suffix))
                obj = sorted(n for n in members if n.endswith(".o"))
                opt_ll = sorted(n for n in members if n.endswith("ir-with-opt.ll"))
                if v != "V0" and (not obj or not opt_ll or not any(n.endswith("ir-no-opt.ll") for n in members)):
                    raise ValueError(archive)
                for n in opt_ll:
                    kernels.append(dict(**base, **kernel_record(n, z.read(n).decode())))
                object_hash = hashlib.sha256()
                for n in obj:
                    object_hash.update(n.encode()+b"\0"+hashlib.sha256(z.read(n)).digest())
                partitions = []
                for op in mod.comps[mod.entry]:
                    m = re.search(r'"outer_dimension_partitions"\s*:\s*\[([^\]]+)\]', op.attrs)
                    if m:
                        partitions.append(op.name+"="+"x".join(re.findall(r"\d+", m[1])))
                windows = sorted(set(re.findall(r"window=\{size=([\dx]+)", hlo)))
                before = next((n for n in members if ".before_cpu-parallel-task-assigner." in n), None)
                after = next((n for n in members if ".after_cpu-parallel-task-assigner." in n), None)
                structures.append(dict(**base, historical_match=historical_match, temp_bytes=ref["temp_bytes"],
                    exports=len(trigger(hlo, case_for(g), v.split("_")[0])[0]) if v != "V0" else 0,
                    windows="/".join(windows), partitions="|".join(partitions),
                    assigned_before=z.read(before).count(b"outer_dimension_partitions") if before else "",
                    assigned_after=z.read(after).count(b"outer_dimension_partitions") if after else "",
                    kernel_objects=len(obj), objects_sha256=object_hash.hexdigest(),
                    named_schedule=ref["named_schedule"], archive=archive.relative_to(out).as_posix()))
        expected = {(g, v, opt) for g in (set(AXIS_GRAPHS+TASK_GRAPHS) if width == 1 else TASK_GRAPHS)
                    for v in VARIANTS for opt in (("on", "off") if g in TASK_GRAPHS else ("on",))}
        if set(ir[width]) - {"environment"} != expected:
            raise ValueError(f"Incomplete capture {folder}")
    return ir, structures, kernels
