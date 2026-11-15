"""Rebuild the Chinese course report and its evidence tables from sealed data."""
from collections import Counter
from statistics import mean, median

from .common import DOUBLES, FAMILIES, MODES, SPEC, conditions, describe, read, records, table
from .evaluate import paired_tables,policy_tables
from .evidence import evidence
from .figures import figures
from .model import verify_lock
from .stats import summarize


def md_table(header,rows):
    def line(row):
        return "| "+" | ".join(str(x).replace("|"," / ") for x in row)+" |"
    return "\n".join([line(header),line(["---"]*len(header)),*[line(r) for r in rows]])


def fmt(x,d=2):
    return f"{x:,.{d}f}"


def boundary_table(root,lock):
    rows=[]
    for case in lock["boundary_cases"]:
        s=summarize(root,"boundary",case)
        for prefix,kind in (("","单体"),("split_","分段")):
            first,second=prefix+"pre_pre",prefix+"post_pre"
            diffs=[p[second]-p[first] for p in s["processes"]]
            rows.append(dict(case=case,execution=kind,pre_pre_us=s["times"][first],
                             post_pre_us=s["times"][second],paired_post_minus_pre_us=median(diffs),
                             min_us=min(diffs),max_us=max(diffs),aa_p95_us=s["aa_us"],
                             valid=s["valid"][first] and s["valid"][second]))
    table(root/"report/tables/boundaries.csv",rows)
    return rows


def report(root):
    lock=verify_lock(root)
    data,inter=paired_tables(root)
    decisions,summaries,costs=policy_tables(root,data)
    structure,classes=evidence(root)
    bounds=boundary_table(root,lock)
    figures(root,data,summaries)
    hw=read(root/"hardware.json")
    env=read(root/"matrix/env_p0.json")
    idx={(x["group"],x["method"]):x for x in summaries}
    main=[x for x in classes if not x["case"].startswith("R")]
    equal=sum(x["double_named_schedule_classes"]==1 for x in main)
    ratio=[data["confirm",c]["ratios"][m] for c in conditions() for m in DOUBLES]
    fullvalid=sum(all(data["confirm",c]["valid"].values()) for c in conditions())
    ties=idx["matrix","M2"]["ties"]
    family_labels={"E":"逐元素仿射","M":"两层 MLP","N":"按行归一化","C":"残差卷积"}
    mainrows=[d for d in decisions if d["case"] in conditions()]
    text=f"""# 用编译器 IR 预测 XLA 中间激活观测的增量开销

*Predicting the Cost of Observing Intermediate Activations in XLA Using Compiler IR*

## 研究结论

在 {hw['cpu']} 的 {env['backend'].upper()} 后端上，两处 ReLU 各增加一个正元素计数后，128 个双点配置相对原程序的延迟倍率中位数为 **{median(ratio):.3f}×**，范围为 **{min(ratio):.3f}–{max(ratio):.3f}×**。对同一个查询改变取数位置，{equal}/32 个合成条件的四种写法得到逐条相同的优化后计算体。独立复测中，{ties}/32 个条件的前两名无法按预登记标准稳定排序。[逐配置时延](tables/latencies.csv) · [HLO 等价类](tables/hlo_classes.csv)

官方 Flax ResNet 块给出了不同的 IR 结果。下文从实际返回的计数反向追踪比较、归约和主输出，再用同一张图交换 ReLU 实现，检验这种差异的来源。本组贡献是可复查的同信息配对基准、双点交互数据和只读取基线 IR 的观测位置选择工具。

## 1　问题、等价性与实验单位

设中间值为有限 FP32 张量 z，a = max(z, 0)。每个元素均满足 (z > 0) ⇔ (a > 0)，包括正零和负零。对指定两层 u、v 做严格比较、再用 int32 求和，就能在四种接入方式中返回相同的 (y, q₁, q₂)；每个计数占 4 字节。对负值最小值的查询，ReLU 后的张量已经丢失负值，选择工具会拒绝该迁移。每个归约的元素数均小于 2³¹。

逐元素 E、全连接 M、归一化 N 和卷积 C 各有四档规模。每档各测链式和分支读出，分支版让第一处激活直接参与最终输出。每条件编译 1 个基线、4 个单点和 4 个双点配置，合计 32 个决策、288 个配置。E 的四档为 4,096 至 1,048,576 个元素；M 的宽度为 256；N 固定每行 256 个通道；C 使用 32×32、16 通道的两个 3×3 卷积。所有权重作为运行参数传入。输入种子为 42、1729、2026，计时使用 42。小图另有 72 项 NumPy 对照；在正式复测中，{fullvalid}/32 个条件的九版在三个种子上全部通过 y 与 q 校验。[预登记条件与容差](../../../protocol.json) · [配置清单](tables/configs.csv) · [NumPy 核验](../validation/numpy.json)

JAX / jaxlib 均为 {env['jax']}，Python {env['python']}。进程在导入 JAX 前绑定到逻辑 CPU {env['affinity']}。数据预先常驻设备，编译和预热单独进行。每个进程中每案先预热 6 次，正式计时 20 轮，轮内九版随机等次数出现。以 `jax.block_until_ready` 等待整个返回结构，以纳秒保存原始读数。五个训练批次后锁定四轮整族留出预测，再用三个新进程复测全部主矩阵；两个 ResNet 外测用五个新进程。[环境记录](../matrix/env_p0.json) · [预测封存](../model/prediction_lock.json) · [JAX 官方计时说明](https://docs.jax.dev/en/latest/benchmarking.html)

T 表示进程内 20 个读数的中位数；配对做差 T(c) − T₀ 或取比 T(c)/T₀ 后再跨进程取中位数。图中的绝对延迟是进程中位数的中位数，两种汇总的运算次序不同。负增量保留。A/A 把同一个基线可执行文件分成四个标签独立交错测量。其配对差绝对值的 95 分位是每条件的排序噪声参考；前两名落入门槛，或最快者未获进程严格多数，就记为无法稳定排序。它是实用判据，不提供总体显著性检验。[进程级数据](tables/process_medians.csv) · [噪声与平局](tables/noise.csv)

## 2　RQ1：观测位置如何影响生成的计算？

先导的 sample 和 mean 均额外返回 4 字节，但 mean 要访问全张量完成归约。随后固定同一个正数计数请求，沿 ENTRY 的 y、q₁、q₂ 分别回溯融合调用、参数及元组元素。下图展示 E0 链式图的 post/post 版。每个框都给出优化后 HLO 行号，中间的转换、广播和部分算术步骤已省略。[四种先导](../pilot/four_probes.csv) · [九行先导与 HLO 摘录](../pilot/first_report.md) · [图中原始 HLO](../matrix/ir/E0_chain/post_post/optimized_hlo.txt)

![E0 的主输出与两个计数的 HLO 数据流](figures/01-hlo-paths.png)

"""
    selected=[]
    for c in ("E0_chain","M3_chain","N0_branch","C0_chain","R1","R2"):
        arr=[r for r in structure if r["case"]==c and r["mode"]=="post_post" and r["site"]=="u"]
        a,b,q=arr
        selected.append((c,a["relation"],b["relation"],q["relation"],q["source"] or a["target"]))
    text+=md_table(["抽查图","z→ReLU","ReLU→主消费者","第一处比较的实际输入","对应 HLO 行"],selected)
    text+=f"""

对于 {equal} 个合成条件，严格比较保留了操作顺序、原始临时名、常量、类型、布局、调用和后端配置，只移除调试元数据。另一种允许临时名重命名的比较同时保存在表中。源码 post 在 E0 的比较处直接读取 affine 加法结果；两个计数有自己的归约路径，主输出的 affine 和 ReLU 仍可在同一融合体内找到。N0 分支上同一源码 ReLU 在主路径有多处对应，结构表保留所有可见行号并标记歧义。CPU 上的显式 fusion 只能用于讨论 HLO 分组；本研究没有测量物理循环次数或 DRAM 流量。[逐边证据](tables/structure.csv) · [逐配置指纹](tables/hlo_fingerprints.csv)

![四个图族的候选延迟和 A/A 参考范围](figures/02-placement-and-noise.png)

## 3　RQ2：两个计数的成本可以相加吗？

对每一对位置在进程内计算 I = T(u,v) − T(u) − T(v) + T₀，单位为 µs。负值表示同时观测比单点增量之和便宜。四个 A/A 标签以同一公式构造伪交互，绝对值的 95 分位作为对应门槛。下表列出两个独立批次均越过各自门槛且符号相同的项数。每族有 8 个条件 × 4 个位置组合。[全量交互及范围](tables/interactions.csv) · [每进程加和预测与实测](tables/process_interactions.csv)

"""
    discovery={(r["case"],r["mode"]):r for r in inter if r["phase"]=="matrix"}
    confirm={(r["case"],r["mode"]):r for r in inter if r["phase"]=="confirm"}
    repeated=[key for key,r in confirm.items() if r["valid"] and discovery[key]["valid"]
              and r["outside_floor"] and discovery[key]["outside_floor"]
              and r["interaction_us"]*discovery[key]["interaction_us"]>0]
    text+=md_table(["图族","发现批越线 / 32","复测批越线 / 32","两批同向越线 / 32"],[
        (family_labels[f],sum(r["outside_floor"] for (c,_),r in discovery.items() if c[0]==f),
         sum(r["outside_floor"] for (c,_),r in confirm.items() if c[0]==f),sum(c[0]==f for c,_ in repeated)) for f in FAMILIES])
    nrow=confirm["N3_chain","pre_pre"]
    text+=f"""

两批同向越线共 {len(repeated)}/128 项。以含 1024 行的 N3 链式图为例，pre/pre 的复测交互为 **{fmt(nrow['interaction_us'])} µs**，该条件的伪交互门槛为 {fmt(nrow['aa_pseudo_p95_us'])} µs。这个差值衡量两处监控的非加性；计时覆盖运算、调度与同步，单项差值无法辨认各因素所占的比例。

![单点相加预测与双点实测](figures/03-interactions.png)

## 4　RQ3：只用基线 IR 能否选出更便宜的位置？

每次将一个图族所有尺寸、拓扑和候选放入测试，其余三族用于训练。B1 使用输出字节和形状；B2 增加源图算子、扇出、候选端及距离；M1 加入基线 HLO 的逐点属性；M2 再加入两点是否同组、跨组数和共享直接生产者。四者均用训练折标准化与岭回归，固定 α = 1，预测 log(T(c)/T₀)。M1/M2 遇到任一 ReLU 在基线 y 路径上没有唯一定位时退回 B2。固定规则和实测单点相加另作对照。单点实测使用额外的四次候选编译和试跑。外测的单点时间在独立校准批取得。模型构建和封存均先于独立复测与外部候选测时。[特征字典与复现方法](../../../docs/METHODS.md) · [模型参数与折划分](../model/models.json)

位置损失定义为 T(选中) / min T(通过校验的候选) − 1。绝对误差逐进程使用实测 T₀ 将预测倍率还原为微秒，先对每个候选取中位绝对误差，再对候选求均值。这个指标需要额外取得基线时间。实测最优值取自四个带噪声的测量，因此每个模型也报告选中方案距离最优在 A/A 门槛内的次数。

"""
    policies=["B0_pre","B0_post","B0_train","B1","B2","M1","M2","single_add","oracle"]
    policy_rows=[]
    for m in policies:
        s=idx["matrix",m]
        policy_rows.append((m,f"{100*s['median_regret']:.3f}%",f"{100*s['p90_regret']:.3f}%",
                            fmt(s["median_gap_us"]),f"{s['within_aa']}/{s['n']}",
                            fmt(s["mae_us"]) if s["mae_us"] is not None else "—",
                            fmt(s["mae_log_ratio"],3) if s["mae_log_ratio"] is not None else "—"))
    text+=md_table(["策略","损失中位数","P90 损失","中位差距 µs","落在最优 A/A 门槛内","开销 MAE µs","log 倍率 MAE"],policy_rows)
    p=idx["matrix","B0_post"]; m=idx["matrix","M2"]
    fixed_gap=sum(r["gap_us"] for r in mainrows if r["method"]=="B0_post" and r["valid"])
    ir_gap=sum(r["gap_us"] for r in mainrows if r["method"]=="M2" and r["valid"])
    gained=f"{100*(fixed_gap-ir_gap)/fixed_gap:.1f}%" if fixed_gap else "无可追回空间"
    text+=f"""

固定 post/post 与实测最优的中位距离为 {fmt(p['median_gap_us'])} µs，M2 为 {fmt(m['median_gap_us'])} µs，{m['fallback']}/32 个请求触发了源码回退。把每个条件各调用一次，按绝对微秒相加，M2 相对固定 post/post 追回的实测空间占比为 {gained}。排名噪声及每种图的调用频率都会影响这个数值。单点相加在这批留出数据上的开销 MAE 为 {fmt(idx['matrix','single_add']['mae_us'])} µs。各模型的误差、尾部损失、最坏条件和未经筛选的有效率可逐案核对。[每个家族的指标](tables/policy_summary.csv) · [所有决策及平局](tables/decisions.csv) · [逐项预测误差](tables/predictions.csv)

![整图族留出的位置选择损失](figures/04-held-out-regret.png)

"""
    worst=max((r for r in mainrows if r["method"]=="M2" and r["valid"]),key=lambda r:r["gap_us"])
    cmain=[c for c in costs if c["case"] in conditions()]
    text+=f"""M2 的最大绝对选择损失发生在 {worst['case']}，选择 {worst['chosen']}，实测最快为 {worst['oracle']}，差距 {fmt(worst['gap_us'])} µs，该条件 A/A 门槛为 {fmt(worst['aa_p95_us'])} µs。每个基线的降低与编译时间中位数为 {fmt(median(c['one_baseline_lower_compile_us'] for c in cmain)/1000)} ms，特征抽取为 {fmt(median(c['baseline_feature_extraction_us'] for c in cmain)/1000)} ms。四个双点候选全编译的中位数为 {fmt(median(c['four_candidate_lower_compile_us'] for c in cmain)/1000)} ms，其正式试跑另需 {fmt(median(c['four_candidate_20_rounds_us'] for c in cmain)/1000)} ms。成本表给出每案的 C/d，仅在相对固定 post 的观测节省 d > 0 时计算，并标明 d 是否超出噪声。该估计按单次新基线编译和特征抽取计费，省略一次查表、模型打分、文件 I/O 和训练成本。[部署成本与摊销](tables/costs.csv)

## 5　官方 ResNet 外测与追加的写法对照

Flax ImageNet 的 ResNetBlock 直接从固定上游提交载入。R1 运行单个块，R2 串联两个相邻块。输入均为一张 56×56、64 通道特征图，监控首块的第一个 ReLU 与末块的最后一个 ReLU。采用随机固定卷积权重、推理态批归一化，并将末端 BN 的初始零缩放设为 1，让残差分支实际参与计算。未改监控的原模块对三个种子独立核对前向输出。这两个子图与 C 族相关，测试的是外部代码和上下文；数据不支持对真实 ImageNet 激活率或分类精度作结论。[Flax 原模块](https://github.com/google/flax/blob/0fde6365a958e39af1b243f490bad92244894be6/examples/imagenet/models.py) · [外测逐轮数据](../external/raw)

"""
    rs=[]
    for c in ("R1","R2"):
        d=next(r for r in decisions if r["case"]==c and r["method"]=="M2")
        sc=data["external",c]
        ir=next(r for r in classes if r["case"]==c)
        rs.append((c,ir["double_canonical_classes"],d["chosen"],d["oracle"],fmt(sc["times"]["base"]),
                   fmt(d["gap_us"]),fmt(sc["aa_us"]),"未分出" if d["top_two_unresolved"] else "分出"))
    text+=md_table(["子图","四种位置的 HLO 类数","锁定推荐","实测最快","基线 µs","推荐差距 µs","A/A µs","前两名"],rs)
    ex=lock["predictions"]["R2"]["M2"]
    from math import exp
    text+=f"\n\n在 R2 上，M2 为选中的 {ex['chosen']} 预测 {exp(ex['log_ratios'][ex['chosen']]):.3f}× 基线延迟；独立实测的配对倍率是 {data['external','R2']['ratios'][ex['chosen']]:.3f}×。这个外测例子同时保留推荐和测量值。"
    text+="""

R1 的 post/post 归约分支仍含对应 ReLU 的 maximum，pre/pre 的同一比较读取加法结果。ENTRY 的融合调用数无法区分这两种取数边。[R1 pre/pre](../external/ir/R1/pre_pre/optimized_hlo.txt) · [R1 post/post](../external/ir/R1/post_post/optimized_hlo.txt)

看到上述差异后，追加一个仅编译与验算的交叉对照。保持 E0 或 R1 自身的图、权重、输入、显式设备放置和输出契约固定，分别令两处 ReLU 使用直接 `jnp.maximum` 或 `jax.nn.relu`。下表按同一个写法内部的四种位置统计。该对照在推荐封存之后设计，所有位置和结果均保留；没有为这 16 个模块采集运行时延。

"""
    extra=records(root/"lowering_aligned/comparison.csv")
    er=[]
    for c in ("E0_chain","R1"):
        for impl in ("maximum","nn_relu"):
            rr=[r for r in extra if r["case"]==c and r["relu_implementation"]==impl]
            er.append((c,impl,len({r["sha256"] for r in rr}),sum(r["valid"]=="True" for r in rr),
                       next(r["u_relation"] for r in rr if r["mode"]=="post_post")))
    text+=md_table(["固定的图","仅替换 ReLU 实现","双点 HLO 类数","三种子通过的配置 / 4","post/post 中 u 的比较读取"],er)
    text+="""

在这两个案例、这个编译版本中，源码的 ReLU 表达方式足以改变四种位置是否在优化 HLO 汇合。对照没有定位负责变换的单个编译 pass，也没有验证机器码是否逐字节相同。第一次对照使用了隐式设备放置；随后将放置方式与正式计时对齐。原生写法的四个规范化指纹可在表中逐项与正式数据核对，两次归档均可查阅。[对齐后的对照](../lowering_aligned/comparison.csv) · [首次对照](../lowering/comparison.csv)

## 6　显式执行边界与可解释范围

用发现批四候选绝对延迟跨度最小和最大的条件做预登记对照。在第一处 z 之后拆为两个可执行程序，只移动第一个计数的 pre/post 位置，第二个保持 pre。两次调用之间保留设备数据，合并同步仍在所有输出完成时结束。

"""
    text+=md_table(["条件","执行方式","pre/pre µs","post/pre µs","进程配对位置差 µs"],[
        [b["case"],b["execution"],fmt(b["pre_pre_us"]),fmt(b["post_pre_us"]),fmt(b["paired_post_minus_pre_us"])] for b in bounds])
    text+="""

分段改变了调度次数、中间值可见性和潜在布局；单体与分段间的延迟变化不能单独归因于融合。计时来自一台共享桌面机器的单逻辑核配置，没有硬件计数器、GPU 实测或全网推理数据。A/A 使用基线可执行文件，候选具有自身的抖动；同形 HLO 的微小时间差应结合独立重复和编译结构阅读。[分段全表](tables/boundaries.csv) · [分段 HLO 与原始计时](../boundary)

## 7　已有工作与本实验的边界

Snider 与 Liang 分析 XLA 的融合决策并在 Cartpole 上评估性能。Kaufman 等使用程序图学习 TPU 上的性能模型，研究过分块与融合选择。本实验研究的决策变量是同一个激活统计在两个位置上的接入组合，只允许推荐器读取未观测程序的 IR。一次本机测量能够报告该协议下的位置选择空间与失败案例；跨设备推广仍需要新的原始测量。[Snider & Liang](https://arxiv.org/abs/2301.13062) · [Kaufman 等](https://proceedings.mlsys.org/paper_files/paper/2021/hash/6bcfac823d40046dca25ef6d6d59cc3f-Abstract.html)

复现实验、运行选择工具和课堂演示请见[项目首页](../../../README.md)。[预登记原文](../../../docs/FusionScope_Research_Plan_v2.md)及[结果校验](audit.json)随仓库交付。
"""
    (root/"report/REPORT.md").write_text(text,encoding="utf8",newline="\n")
    print("Report, four figures and evidence tables rebuilt.",flush=True)
