from collections import Counter
import hashlib

from .common import DOUBLES, MODES, conditions, sites, table
from .hlo import META, Module


def scheduled_text_digest(text):
    lines=text.splitlines()
    start=next(i for i,line in enumerate(lines) if line.startswith(("%","ENTRY %")))
    body="\n".join([lines[0]]+[META.sub("",line) for line in lines[start:]])
    return hashlib.sha256(body.encode()).hexdigest()


def evidence(root):
    structures,hashes,classes=[],[],[]
    for case in conditions()+["R1","R2"]:
        phase="external" if case.startswith("R") else "matrix"
        by_mode={}
        for mode in MODES:
            path=root/phase/"ir"/case/mode/"optimized_hlo.txt"
            text=path.read_text(encoding="utf8")
            graph=Module(text)
            canon,strict=graph.fingerprint(),scheduled_text_digest(text)
            by_mode[mode]=(canon,strict)
            hashes.append({"case":case,"mode":mode,"canonical_sha256":canon,"named_schedule_sha256":strict,
                           "hlo_file":path.relative_to(root).as_posix()})
            positions=sites(mode)
            for site in ("u","v"):
                idx=list(positions).index(site)+1 if site in positions else None
                for row in graph.edge_rows(site,positions.get(site),idx):
                    structures.append({"case":case,"mode":mode,**row,"hlo_file":path.relative_to(root).as_posix()})
        classes.append({"case":case,"double_canonical_classes":len({by_mode[m][0] for m in DOUBLES}),
                        "double_named_schedule_classes":len({by_mode[m][1] for m in DOUBLES}),
                        "u_single_equal":by_mode["u_pre"]==by_mode["u_post"],
                        "v_single_equal":by_mode["v_pre"]==by_mode["v_post"]})
    dst=root/"report/tables"
    table(dst/"structure.csv",structures)
    table(dst/"hlo_fingerprints.csv",hashes)
    table(dst/"hlo_classes.csv",classes)
    return structures,classes
