"""Independent calibration budget for the measured-single comparison on ResNet."""
import random
from time import perf_counter_ns

from .common import MODES, SPEC, dump, read, sha, table


def calibrate(root,cases,process):
    import jax
    import numpy as np
    from .bench import collect, validate
    from .graphs import inputs, program, resnet_block
    for case in cases:
        raw=root/"external_singles/raw"/f"p{process}"/case
        if (raw/"done.json").exists():
            continue
        args={k:inputs(case,k) for k in SPEC["input_seeds"]}
        jax.block_until_ready(args)
        timed=args[SPEC["timed_seed"]]
        compiled,builds={},[]
        for mode in MODES[:5]:
            start=perf_counter_ns()
            compiled[mode]=jax.jit(program(case,mode)).lower(*timed).compile()
            builds.append(dict(mode=mode,lower_compile_ns=perf_counter_ns()-start))
        checks=validate(case,compiled,args,"external_singles",root/"failures")

        def plain(x,params):
            for block in params:
                x=resnet_block(jax.nn.relu).apply(block,x)
            return x
        upstream=jax.jit(plain)
        plain_checks=[]
        for seed,(x,p) in args.items():
            truth=np.asarray(upstream(x,p))
            measured=np.asarray(compiled["base"](x,p))
            rtol,atol=SPEC["tolerances"][case]
            plain_checks.append({"seed":seed,"max_abs":float(np.max(np.abs(truth-measured))),
                                 "passed":bool(np.allclose(truth,measured,rtol=rtol,atol=atol))})
        if not all(row["passed"] for row in plain_checks):
            raise ValueError(f"Baseline mismatch against uninstrumented upstream {case}: {plain_checks}")
        rng=random.Random(f"singles:{case}:{process}:{SPEC['timing']['order_seed']}")
        times=collect(compiled,timed,case,process,rng,"candidate")
        times+=collect(dict.fromkeys(SPEC["timing"]["aa_labels"],compiled["base"]),
                       timed,case,process,rng,"AA")
        table(raw/"times.csv",times)
        table(raw/"builds.csv",builds)
        dump(raw/"checks.json",checks)
        dump(raw/"upstream.json",plain_checks)
        dump(raw/"done.json",{"files":{n:sha(raw/n) for n in ("times.csv","builds.csv","checks.json","upstream.json")}})
        print(f"independent single-site calibration {case} p{process}",flush=True)
        jax.clear_caches()


def seal(root):
    from datetime import datetime,timezone
    from .model import additive
    from .stats import summarize
    path=root/"model/external_singles_lock.json"
    if path.exists():
        if sha(path)!=(path.parent/"external_singles_lock.sha256").read_text().strip():
            raise ValueError("External additive lock mismatch")
        return
    if (root/"external/raw").exists():
        raise RuntimeError("External confirmation already exists without independent singles lock")
    preds={}
    for case in ("R1","R2"):
        data=summarize(root,"external_singles",case)
        if len(data["processes"])!=SPEC["timing"]["confirmation_processes"]:
            raise ValueError("Incomplete external calibration")
        preds[case]=additive(data["processes"])
    dump(path,{"locked_utc":datetime.now(timezone.utc).isoformat(),"predictions":preds})
    (path.parent/"external_singles_lock.sha256").write_text(sha(path)+"\n",encoding="ascii",newline="\n")
