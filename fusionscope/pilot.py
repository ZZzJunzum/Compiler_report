from statistics import median

from .common import DOUBLES, MODES, table
from .stats import interactions, ranking, summarize


def excerpt(path, filters):
    numbered = list(enumerate(path.read_text(encoding="utf-8").splitlines(),1))
    matches = [f"{i:4d}  {line.strip()}" for i,line in numbered if all(f in line for f in filters)]
    return "\n".join(matches[:4])


def first_report(root):
    case = "E0_chain"
    s = summarize(root, "pilot", case)
    p = summarize(root, "probe4", case)
    dst = root / "pilot"
    rows = []
    for k in MODES:
        row = next(c for c in s["checks"] if c["seed"]==42 and c["mode"]==k)
        rows.append(dict(mode=k,seed42_counts=str(row["counts"]),latency_us=s["times"][k],
                         delta_us=s["delta"][k],ratio=s["ratios"][k],valid=s["valid"][k]))
    table(dst / "first_nine.csv", rows)
    table(dst / "first_interactions.csv", interactions(s))
    table(dst / "four_probes.csv", [dict(mode=k,latency_us=p["times"][k],delta_us=p["delta"][k],
                                         auxiliary_bytes=b,valid=p["valid"][k])
                                    for k,b in zip(("only_y","sample","mean","full"),(0,4,4,16384))])
    ir = dst / "ir/E0_chain/pre_pre/optimized_hlo.txt"
    blocks = [excerpt(ir, [" compare(",f"/probe_{site}/gt"]) for site in ("u","v")]
    roots = excerpt(ir, ["ROOT %tuple", " tuple("])
    order, tie = ranking(s)
    if not order:
        raise RuntimeError("No equivalent double placement in pilot")
    verdict = "前两名落入波动范围或未获进程多数，先导无法稳定排序。" if tie else "第一名在这个先导中超出噪声门槛并获进程多数，仍需独立复核。"
    out = f"""# E0 双点观测先导

CPU，4096 个 FP32 元素，链式读出。每案 3 个独立进程，每进程预热 6 次，随机交错测 20 轮。各配置的逐次原始观测在 `raw/p*/E0_chain/times.csv`。同步端点覆盖全部输出。

| 配置 | seed 42 的计数 | 延迟 µs | 配对增量 µs |
|---|---|---:|---:|
"""
    for row in rows:
        out += f"| {row['mode']} | {row['seed42_counts']} | {row['latency_us']:.3f} | {row['delta_us']:.3f} |\n"
    out += f"""
同一可执行文件 A/A 的差值门槛为 {s['aa_us']:.3f} µs。双点观测的实测顺序为 {'、'.join(order)}。{verdict} 后续保留全部四种组合，并优先核对这两条比较分支是否重新计算了 affine 加法。

## 从返回值找回两个计数

以下行号对应 [pre_pre 优化后 HLO](ir/E0_chain/pre_pre/optimized_hlo.txt)。ENTRY 的 tuple 顺序是 y、q1、q2。

```text
{roots}
```

第一处比较读取第一段加法的结果，第二处比较读取第二段加法的结果。它们在两个单独的融合体内。两个归约结果都沿 ENTRY 的 reduce-window 和 reduce 调用进入返回元组。

```text
{blocks[0]}
{blocks[1]}
```

双点交互见 [四项配对计算](first_interactions.csv)。四种不同信息量的先导见 [only_y / sample / mean / full](four_probes.csv)。其中 sample 与 mean 都多返回 4 字节，mean 还需要读取全张量做归约。
"""
    (dst / "first_report.md").write_text(out,encoding="utf-8",newline="\n")
