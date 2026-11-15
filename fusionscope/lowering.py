"""Post-hoc, compile-only crossed control motivated by the two ResNet holdouts."""
import jax
import jax.numpy as jnp
import numpy as np

from .bench import write_ir
from .common import DOUBLES, SPEC, dump, sites, table
from .graphs import activation, first_stage, inputs, layer, program, resnet_block
from .hlo import Module


def variant(case,mode,use_nn):
    def workload(x,p):
        observed=[]
        if case=="R1":
            site_names=iter(("u","v"))
            def act(z):
                return activation(z,next(site_names),mode,observed,official=use_nn)
            with jax.named_scope("block0"):
                y=resnet_block(act).apply(p[0],x)
        else:
            z0=first_stage("E",x,p)
            a=activation(z0,"u",mode,observed,official=use_nn)
            with jax.named_scope("v_z"):
                z1=layer("E",a,p[2],p[3])
            a2=activation(z1,"v",mode,observed,official=use_nn)
            with jax.named_scope("readout"):
                y=a2*jnp.float32(.75)+jnp.float32(.125)
        return (y,*observed)
    return workload


def run(root):
    rows=[]
    for case in ("E0_chain","R1"):
        args={seed:inputs(case,seed) for seed in SPEC["input_seeds"]}
        jax.block_until_ready(args)
        timed=args[SPEC["timed_seed"]]
        native_phase="external" if case=="R1" else "matrix"
        for name in ("maximum","nn_relu"):
            for mode in DOUBLES:
                fn=variant(case,mode,name=="nn_relu")
                low=jax.jit(fn).lower(*timed)
                compiled=low.compile()
                folder=root/"lowering/ir"/case/name/mode
                dump(folder/"hashes.json",write_ir(folder,fn,timed,low,compiled))
                ref=jax.jit(program(case,mode))
                checks=[]
                for seed,xp in args.items():
                    y,*q=compiled(*xp)
                    ref_y,*ref_q=ref(*xp)
                    valid=bool(np.allclose(y,ref_y,rtol=SPEC["tolerances"][case if case=="R1" else "E"][0],
                                           atol=SPEC["tolerances"][case if case=="R1" else "E"][1])
                               and [int(v) for v in q]==[int(v) for v in ref_q])
                    checks.append({"seed":seed,"counts":[int(v) for v in q],"valid":valid})
                dump(folder/"checks.json",checks)
                graph=Module(compiled.as_text())
                sig=graph.fingerprint()
                native=Module((root/native_phase/"ir"/case/mode/"optimized_hlo.txt").read_text(encoding="utf8"))
                row={"case":case,"relu_implementation":name,"mode":mode,"sha256":sig,
                     "matches_primary_hlo":sig==native.fingerprint(),"valid":all(c["valid"] for c in checks)}
                for i,site in enumerate(("u","v")):
                    probe=graph.edge_rows(site,sites(mode)[site],i+1)[-1]
                    row[site+"_relation"]=probe["relation"]
                    row[site+"_source"]=probe["source"]
                rows.append(row)
    table(root/"lowering/comparison.csv",rows)
