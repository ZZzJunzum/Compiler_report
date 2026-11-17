"""Execution strategies with the same per-invocation observation contract."""
from dataclasses import dataclass
from time import perf_counter_ns

import jax
import jax.numpy as jnp

from fusionscope.common import describe
from fusionscope.graphs import first_stage, layer, program, resnet_block
from .audit_ir import PLAN


def options(case):
    family,shape,_ = describe(case)
    tiles = [] if family.startswith("R") else [
        t for t in PLAN["conv_tile_batches" if family=="C" else "tile_rows"] if shape[0]%t==0]
    return ["V0","V1","V2","V3"]+[f"V4_t{t}" for t in tiles]


def count(a):
    return jnp.sum(a>0,dtype=jnp.int32)


def first_activation(family,x,params):
    z=first_stage(family,x,params)
    with jax.named_scope("u_relu"):
        return jnp.maximum(z,jnp.float32(0))


def second_activation(family,top,a,x,params):
    with jax.named_scope("v_z"):
        z=layer(family,a,params[2],params[3])
        if family=="C":
            z=z+x
    with jax.named_scope("v_relu"):
        b=jnp.maximum(z,jnp.float32(0))
    with jax.named_scope("readout"):
        y=b*jnp.float32(.75)+jnp.float32(.125)
        if top=="branch":
            y=y+a*jnp.float32(.25)
    return y,b


def official_activations(x,params):
    watched=[]
    for block,p in enumerate(params):
        site=iter(("first","last"))
        def act(z):
            which=next(site)
            a=jax.nn.relu(z)
            if (block==0 and which=="first") or (block==len(params)-1 and which=="last"):
                watched.append(a)
            return a
        x=resnet_block(act).apply(p,x)
    return x,*watched


def joint(case):
    family,_,top=describe(case)
    def execute(x,params):
        if family.startswith("R"):
            y,a,b=official_activations(x,params)
        else:
            a=first_activation(family,x,params)
            y,b=second_activation(family,top,a,x,params)
        masks=((a>0).astype(jnp.int32),(b>0).astype(jnp.int32))
        zero=jnp.int32(0)
        with jax.named_scope("joint_counts"):
            counts=jax.lax.reduce(masks,(zero,zero),
                                  lambda p,q:(p[0]+q[0],p[1]+q[1]),tuple(range(a.ndim)))
        return y,*counts
    return execute


def official_sublayer(x,p,index):
    # Use the same Flax module factories and variable collections as ResNetBlock.__call__.
    block=resnet_block(jax.nn.relu)
    conv=f"Conv_{index}"
    bn=f"BatchNorm_{index}"
    x=block.conv(64,(3,3)).apply({"params":p["params"][conv]},x)
    return block.norm().apply({k:p[k][bn] for k in ("params","batch_stats")},x)


def stages(case):
    family,_,top=describe(case)
    official=family.startswith("R")
    def front(x,params):
        a=jax.nn.relu(official_sublayer(x,params[0],0)) if official else first_activation(family,x,params)
        with jax.named_scope("probe_u"):
            return a,count(a)

    def back(a,x,params):
        if official:
            y=jax.nn.relu(x+official_sublayer(a,params[0],1))
            if len(params)==2:
                y=resnet_block(jax.nn.relu).apply(params[1],y)
            b=y
        else:
            y,b=second_activation(family,top,a,x,params)
        with jax.named_scope("probe_v"):
            return y,count(b)
    return front,back


def tiled(case,tile):
    body=program(case,"post_post")
    def execute(x,params):
        blocks=x.reshape((x.shape[0]//tile,tile,*x.shape[1:]))
        with jax.named_scope("tile_body"):
            ys,us,vs=jax.lax.map(lambda block:body(block,params),blocks)
        with jax.named_scope("merge_partials"):
            return ys.reshape(x.shape),jnp.sum(us,dtype=jnp.int32),jnp.sum(vs,dtype=jnp.int32)
    return execute


@dataclass
class Component:
    name: str
    compiled: object
    lowered: object
    lower_ns: int
    compile_ns: int


def compile_part(name,fn,args):
    start=perf_counter_ns()
    lowered=jax.jit(fn).lower(*args)
    middle=perf_counter_ns()
    compiled=lowered.compile()
    return Component(name,compiled,lowered,middle-start,perf_counter_ns()-middle)


def build(case,mode,args):
    if mode not in options(case):
        raise ValueError(f"Unsupported {case}/{mode}")
    if mode=="V3":
        f,g=stages(case)
        front=compile_part("front",f,args)
        # Match the runtime tensor's sharding while avoiding an extra execution during compilation.
        x=args[0]
        abstract=jax.ShapeDtypeStruct(x.shape,x.dtype,sharding=x.sharding)
        back=compile_part("back",g,(abstract,*args))
        def execute(x,p):
            a,u=front.compiled(x,p)
            y,v=back.compiled(a,x,p)
            return y,u,v
        return execute,[front,back]
    if mode=="V2":
        fn=joint(case)
    elif mode.startswith("V4"):
        fn=tiled(case,int(mode.split("_t")[1]))
    else:
        fn=program(case,"base" if mode=="V0" else "post_post")
    part=compile_part("main",fn,args)
    return part.compiled,[part]
