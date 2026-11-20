"""Deployment decision before compilation. No candidate HLO or trial builds are inputs."""
from time import perf_counter_ns

import psutil
from fusionscope.common import read
from .collect import SEAL

SPEC = read(SEAL)
POLICIES = ("native_J","separate","packed","resource","grain","shape_guard")
FIXED = {"native_J":"J","separate":"S","packed":"P"}


def eligible(width):
    return width>0 and width+(width<<width.bit_length())<=0xffffffff


def choice(policy, rows, width, cpus):
    if policy in FIXED:
        key = FIXED[policy]
    elif policy=="resource":
        key = SPEC["cpu_map"].get(str(cpus),SPEC["unknown_affinity_default"])
    else:
        features = (cpus,rows*width/cpus,rows/cpus)
        key = SPEC[policy]
        while isinstance(key,dict):
            key = key["low"] if features[key["feature"]]<=key["at"] else key["high"]
    return "J" if key=="P" and not eligible(width) else key


def measure_choices(case):
    records = []
    for policy in POLICIES:
        if policy in ("native_J","separate"):
            elapsed,key = 0,FIXED[policy]
        else:
            start = perf_counter_ns()
            cpus = 1 if policy=="packed" else len(psutil.Process().cpu_affinity())
            key = choice(policy,case.rows,case.width,cpus)
            elapsed = perf_counter_ns()-start
        records.append(dict(graph=case.name,policy=policy,chosen=key,selection_ns=elapsed))
    return records
