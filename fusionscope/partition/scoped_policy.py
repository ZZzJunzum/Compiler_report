"""Choose a scoped observer implementation using only source dimensions and OS affinity."""
from time import perf_counter_ns
import psutil
from fusionscope.common import read
from .scoped import SEAL

SPEC = read(SEAL)
FIXED = {"native_J":"J","separate":"S","scoped_off":"split_off"}
POLICIES = (*FIXED,"resource","grain","shape_guard")


def choice(policy,rows,width,cpus):
    if policy in FIXED:
        return FIXED[policy]
    if policy=="resource":
        return SPEC["cpu_map"].get(str(cpus),"J")
    values = (cpus,rows*width/cpus,rows/cpus)
    node = SPEC[policy]
    while isinstance(node,dict):
        node = node["low"] if values[node["feature"]]<=node["at"] else node["high"]
    return node


def measure_choices(case):
    result = []
    for policy in POLICIES:
        if policy in FIXED:
            ns,chosen = 0,FIXED[policy]
        else:
            start = perf_counter_ns()
            chosen = choice(policy,case.rows,case.width,len(psutil.Process().cpu_affinity()))
            ns = perf_counter_ns()-start
        result.append(dict(graph=case.name,policy=policy,chosen=chosen,selection_ns=ns))
    return result
