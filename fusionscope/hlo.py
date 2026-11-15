"""Conservative reader for scheduled XLA HLO. Anchors are one-based text lines."""
from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import re

NAME = r"[A-Za-z_.$][\w.$-]*"
HEADER = re.compile(rf"^(ENTRY )?%({NAME})\s*\(")
INSTR = re.compile(rf"^\s+(ROOT )?%({NAME}) = (.+?) ([\w-]+)\(([^)]*)\)(.*)$")
REF = re.compile(rf"%({NAME})")
META = re.compile(r', metadata=\{(?:[^{}"]|"(?:\\.|[^"\\])*")*\}')


@dataclass(frozen=True)
class Op:
    name: str
    shape: str
    kind: str
    operands: str
    attrs: str
    scope: str
    line: int
    root: bool


@dataclass(eq=False)
class Value:
    op: Op
    group: str
    fused: bool
    parents: tuple

    def anchor(self):
        return f"L{self.op.line} %{self.op.name} [{self.group}]"


def ancestors(v):
    seen, q = set(), [v]
    while q:
        node = q.pop()
        if node not in seen:
            seen.add(node)
            q.extend(node.parents)
    return seen


def relation(left, right):
    if left is None or right is None:
        return "unknown"
    return "same_explicit_fusion" if left.fused and left.group == right.group else "separate_hlo_groups"


def distances(start, children):
    dist = {start:0}
    todo = deque([start])
    while todo:
        v = todo.popleft()
        for child in children[v]:
            step = int(v.group != child.group)
            if child not in dist or dist[child] > dist[v]+step:
                dist[child] = dist[v]+step
                todo.append(child)
    return dist


class Module:
    def __init__(self, text):
        self.comps, self.entry, current = {}, None, None
        lines = text.splitlines()
        self.header = lines[0].partition(",")[2]
        self.computation_headers = {}
        for line_num, line in enumerate(lines,1):
            head = HEADER.match(line)
            if head:
                current = head[2]
                self.comps[current] = []
                self.computation_headers[current] = line
                if head[1]:
                    self.entry = current
                continue
            if line == "}":
                current = None
            if current is None or not line.strip():
                continue
            m = INSTR.match(line)
            if not m:
                raise ValueError(f"Cannot parse HLO line {line_num}: {line}")
            root, name, shape, kind, operands, attrs = m.groups()
            scope = re.search(r'op_name="([^"]*)"', attrs)
            self.comps[current].append(Op(name,shape,kind,operands,META.sub("",attrs),
                                           scope[1] if scope else "",line_num,bool(root)))
        if self.entry is None:
            raise ValueError("Missing HLO ENTRY")
        self.exports = set()
        self.opaque = []
        root = self.expand(self.entry, (), "", False)
        self.outputs = root.parents if root.op.kind=="tuple" else (root,)
        self.main = ancestors(self.outputs[0])
        self.children = defaultdict(list)
        for v in sorted(self.main, key=lambda n:n.op.line):
            for p in v.parents:
                self.children[p].append(v)

    def expand(self, comp, args, group, fused):
        env = {}
        root = None
        for op in self.comps[comp]:
            parents = tuple(env[n] for n in REF.findall(op.operands))
            enclosing = group if fused else f"{group}/L{op.line}"
            v = Value(op, enclosing, fused, parents)
            if op.kind == "parameter" and args:
                v = args[int(op.operands)]
            elif op.kind in ("fusion","call"):
                match = re.search(r"(?:calls|to_apply)=%(" + NAME + ")", op.attrs)
                if match and match[1] in self.comps:
                    v = self.expand(match[1], parents, f"{group}/L{op.line}", op.kind=="fusion")
                    if op.kind=="fusion":
                        self.exports.update(v.parents if v.op.kind=="tuple" else [v])
                else:
                    self.opaque.append(op.line)
            elif op.kind=="get-tuple-element" and parents and parents[0].op.kind=="tuple":
                idx = int(re.search(r"index=(\d+)",op.attrs)[1])
                v = parents[0].parents[idx]
            elif op.kind in ("custom-call","while","conditional","async-start"):
                self.opaque.append(op.line)
            env[op.name] = v
            if op.root:
                root = v
        if root is None:
            raise ValueError(f"Missing root in {comp}")
        return root

    def fingerprint(self):
        """Keep schedule, typed operands, constants, call bodies and backend attributes."""
        order, ids = [], {}
        def visit(name):
            if name in ids:
                return
            ids[name] = f"comp{len(order)}"
            order.append(name)
            for op in self.comps[name]:
                for ref in REF.findall(op.attrs):
                    if ref in self.comps:
                        visit(ref)
        visit(self.entry)
        for name in self.comps:
            visit(name)
        canonical = [self.header]
        for name in order:
            ops = self.comps[name]
            local = {op.name:f"v{i}" for i,op in enumerate(ops)}
            def rename(m):
                token=m[1]
                return "%"+local.get(token, ids.get(token, token))
            # Argument names in the signature repeat parameter ops. Retain their declared types/order.
            header = self.computation_headers[name]
            signature = header[header.index("("):]
            signature = re.sub(r"(?<=[(,])\s*[\w.$-]+:\s*", "", signature)
            canonical.append(signature)
            for op in ops:
                canonical.append((op.root,op.shape,op.kind,REF.sub(rename,op.operands),REF.sub(rename,op.attrs)))
        return hashlib.sha256(json.dumps(canonical,separators=(",",":")).encode()).hexdigest()

    def relu(self, site):
        matches = self.relu_candidates(site)
        return matches[0] if len(matches)==1 else None

    def relu_candidates(self, site):
        return [v for v in self.main if v.op.kind=="maximum" and f"/{site}_relu/" in v.op.scope]

    def producer(self, relu):
        if relu is None:
            return None
        candidates = [v for v in relu.parents if not is_zero(v)]
        return candidates[0] if len(candidates)==1 else None

    def site_features(self, site, side):
        a = self.relu(site)
        z = self.producer(a)
        v = z if side=="pre" else a
        if v is None:
            return {"located":0,"in_fusion":0,"fusion_output":0,"users":0,"same_next":0,
                    "near_reduce":0,"near_dot_conv_call":0,"entry_output":0}
        kids = self.children[v]
        nearby = {v,*v.parents,*kids}
        for parent in v.parents:
            nearby.update(parent.parents)
        return {"located":1,"in_fusion":int(v.fused),"fusion_output":int(v in self.exports),
                "users":len(set(kids)),"same_next":int(any(relation(v,k)=="same_explicit_fusion" for k in kids)),
                "near_reduce":int(any(n.op.kind.startswith("reduce") for n in nearby)),
                "near_dot_conv_call":int(any(n.op.kind in ("dot","convolution","custom-call") for n in nearby)),
                "entry_output":int(v is self.outputs[0])}

    def pair_features(self, u_side, v_side):
        u,v = self.relu("u"),self.relu("v")
        if u_side=="pre":
            u=self.producer(u)
        if v_side=="pre":
            v=self.producer(v)
        if u is None or v is None:
            return {"same_fusion":0,"group_crossings":-1,"shared_parent":0,"pair_known":0}
        return {"same_fusion":int(relation(u,v)=="same_explicit_fusion"),
                "group_crossings":distances(u,self.children).get(v,-1),
                "shared_parent":int(bool(set(u.parents)&set(v.parents))),"pair_known":1}

    def edge_rows(self, site, request, output_index):
        a = self.relu(site)
        z = self.producer(a)
        main_kids = self.children[a] if a is not None else []
        matches = self.relu_candidates(site)
        missing = f"{len(matches)} tagged maxima on y path; producer mapping ambiguous" if a is None else (
                  "producer hidden" if z is None else "")
        def anchors(values):
            return " | ".join(v.anchor() for v in sorted(set(values),key=lambda x:(x.op.line,x.group)))
        def row(edge, state, start, end, detail="", repeat="", multi=""):
            return {"site":site,"edge":edge,"relation":state,"source":anchors(start),
                    "target":anchors(end),"detail":detail,"repeated_hlo_expression":repeat,
                    "multioutput_fusion_shared_with_y":multi,"opaque_lines":",".join(map(str,self.opaque))}
        aa, zz = matches, ([z] if z else [])
        rows = [row("z_to_relu",relation(z,a),zz,aa,missing)]
        state = "entry_output" if a is self.outputs[0] else "|".join(sorted({relation(a,c) for c in main_kids})) or "unknown"
        rows.append(row("relu_to_main",state,aa,main_kids,missing))
        if request is None:
            return rows+[row("probe","absent",[],[])]
        path = ancestors(self.outputs[output_index])
        cmps = [n for n in path if n.op.kind=="compare" and f"/probe_{site}/" in n.op.scope]
        reductions = [n for n in path if n.op.kind.startswith("reduce") and "s32" in n.op.shape]
        if len(cmps)!=1 or not cmps[0].parents:
            return rows+[row("probe","unknown",[],reductions,"no unique tagged comparison on auxiliary path")]
        cmp = cmps[0]
        values = [v for v in cmp.parents if not is_zero(v)]
        operand = values[0] if len(values)==1 else None
        signatures = {}
        def expression(v):
            if v is None or v.op.kind not in {"constant","parameter","add","multiply","subtract","divide",
                    "maximum","broadcast","reshape","bitcast","transpose","rsqrt","sqrt","dot","convolution"}:
                return None
            if v not in signatures:
                subs = [expression(n) for n in v.parents]
                if None in subs:
                    signatures[v] = None
                else:
                    item=(v.op.shape,v.op.kind,REF.sub("%",v.op.operands),v.op.attrs,subs)
                    signatures[v]=hashlib.sha256(repr(item).encode()).hexdigest()
            return signatures[v]
        actual = "unknown"
        repeated = ""
        for label,target in (("z",z),("relu",a)):
            if operand is not None and (operand is target or
                    (expression(operand) is not None and expression(operand)==expression(target))):
                actual = label
                repeated = str(operand is not target and operand.group!=target.group)
                break
        groups_y={v.group for v in self.main if v.fused}
        # Two different exported tensor values of the same fused call on the two distinct paths.
        y_exports={n for n in self.main if n in self.exports and n.fused}
        multi=any(n in self.exports and n.fused and n.group in groups_y and
                  any(y.group==n.group and y is not n for y in y_exports) for n in path)
        return rows+[row("probe",f"{request}_reads_{actual}",[operand] if operand else [],[cmp,*reductions],
                         f"compare in {cmp.group}, q root {self.outputs[output_index].anchor()}",repeated,str(multi))]


def is_zero(v):
    while v.op.kind in ("broadcast","reshape","bitcast") and len(v.parents)==1:
        v=v.parents[0]
    return v.op.kind=="constant" and v.op.operands in ("0","0.0","-0","-0.0")
