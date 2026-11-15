import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
MODES = SPEC["placements"]
DOUBLES = MODES[5:]
FAMILIES = list(SPEC["shapes"])


def conditions():
    return [f"{family}{i}_{top}" for family in FAMILIES for i in range(4)
            for top in SPEC["topologies"]]


def describe(case):
    if case in ("R1", "R2"):
        return case, tuple(SPEC["external"][case]), "official"
    code, top = case.split("_")
    return code[0], tuple(SPEC["shapes"][code[0]][int(code[1:])]), top


def sites(mode):
    if mode == "base":
        return {}
    left, right = mode.split("_")
    if left in ("u", "v"):
        return {left: right}
    if mode not in DOUBLES:
        raise ValueError(mode)
    return {"u": left, "v": right}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")
    temp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path, rows):
    if not rows:
        raise ValueError(f"empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def records(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
