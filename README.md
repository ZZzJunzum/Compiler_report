# FusionScope

项目从“用编译器中间表示（IR）预测 GPU 算子融合成本”出发，先在一台 Intel CPU 上研究一个可以严格对照的问题。推理必须返回同一主张量和指定的激活统计，XLA 为此多做了什么，能否用等价改写省时，又能否在编译带观测的程序之前预知这笔开销？最后一项实验允许编译普通程序以读取其优化后 HLO。

**[英文双栏论文 PDF](output/pdf/FusionScope.pdf)** · [LaTeX 源码](paper/FusionScope.tex) · [逐篇文献对照](docs/literature-review.md) · [基线 IR 预测观测增量](results/cpu_cost/increment/accuracy.csv) · [预测带来的实际选型收益](results/cpu_cost/increment/policies.csv) · [观测改写逐项结果](results/cpu_cost/analysis/contracts.csv) · [整次调用误差](results/cpu_cost/analysis/accuracy.csv)

## 从文献线索到我们的实验

| 文献提供的线索 | 我们做的对照 | 结果怎样决定下一步 |
|---|---|---|
| Snider 与 Liang 分析 XLA 的 GPU 融合决策，Kaufman 等用已形成的 TPU 内核预测运行时间。Cano 等发现，调试时拆开 TPU 融合核会恢复原本缺失的半精度舍入。 | 我们把问题收紧到在 CPU 推理中增加一个返回值。先导实验中，取一个元素和取均值都多返回 4 字节，耗时分别为 28.75 和 32.45 微秒。 | 返回字节数不足以排序，而且两种输出回答的问题不同。后续实验每次都保留原主输出，并固定两个正激活计数。[原始研究计划](https://github.com/ZZzJunzum/Compiler_report/blob/4346836/docs/FusionScope_Research_Plan_v2.md) · [TPU 案例图 4](https://arxiv.org/abs/2611.00624) |
| JAX 记录了打印 ReLU 前的数值可能迫使其物化。对于有限输入，ReLU 前后的正元素集合相同。 | 两层各在 ReLU 前或后计数，32 个合成条件内的四种放置均得到相同的优化后 HLO，观测版相对无观测版的耗时比中位数仍为 1.107。 | 移动位置没消掉这笔开销，于是沿每个返回值倒查生产者。后来的逐步编译复核又定位到两张图上的首轮代数化简，显式 maximum 的 8/8 个后置计数改变操作数，库函数为 0/8。[JAX 调试文档](https://docs.jax.dev/en/latest/debugging/print_breakpoint.html) · [16 个编译对照](results/placement_pass/comparisons.csv) |
| SystemML 已实测共享输入上的多个聚合并建模重算与物化。Weld 的一个循环同时生成向量与和，给这项技术提供了另一直接先例。 | 4096 × 256 仿射分支上，两张各 4 MiB 的整数掩码可在 HLO、缓冲区分配和 LLVM 写指令中定位。联合归约消去两张掩码，提速 3.07 倍，独立复跑为 2.92 倍。 | 换一种图或分块方式，收益可能换方向。归一化案例在固定分块下配对后发现，整次调用变快，计数本身的增量却从 11.75 升至 15.55 微秒。[SystemML](https://www.vldb.org/pvldb/vol11/p1755-boehm.pdf) · [Weld Listing 6](https://www.vldb.org/pvldb/vol11/p1002-palkar.pdf) · [无观测对照](results/phase3/cpu/analysis/interactions.csv) |
| 已有成本模型分别评估预测耗时和候选排序。前者误差较小，并不保证后者选对。 | 第一版五项正成本在 186 个替代方案上均被固定策略支配。后一版在 24 个新条件上优于整批联合，但删除指定的基线 HLO 特征后 28/28 个选择相同。 | 我们加入开发集选出的固定 16 规则，分别检验选型、取得 IR 的费用和回本次数。在另一批 11 张图上直接预测两种实现的耗时比，41 次中改选 17 次，仅 1 次明确变快、1 次明确变慢。[AutoTVM](https://proceedings.neurips.cc/paper/2018/file/8b5700012be65c9da25f49408d959ca0-Paper.pdf) · [模型诊断](paper/check_diagnostics.py) · [独立决策](results/cpu_cost/paired/policies.csv) |
| 前述工作给出了共享机制，实际适用范围要靠改变统计请求来检验。 | 22 张新图各测 9 种请求。198 对中有 120 对满足直接联合归约的条件。82 对消去了参考实现里的观测专用导出，其中 30 对明确加速、4 对明确减速、48 对未分辨。 | 同一张 320 × 160 仿射链上，三个全局计数变快，两个末轴计数变慢。这里同时改变了归约轴和观测点数。要解释逐行计数的减速，下一项 CPU 对照应固定两个点，直接检验归约轴与实现方式的交互。[逐请求结果](results/cpu_cost/analysis/rewrite_pairs.csv) · [78 对回退原因](results/cpu_cost/analysis/boundaries.csv) |

## 距离原定目标还有多少

| 目标 | 已有证据 | 差距 |
|---|---|---|
| 在 CPU 解释并降低观测成本 | 对特定图已从编译路径追到实际写入，并用等价改写复现了加速。扩展测试中 120 个合法比较有 30 胜、4 负、86 个未分辨。 | 能给出机制和适用条件。缓冲消失仍不足以确定耗时方向，四个减速案例的低层原因仍需验证。[机制记录](results/cpu_cost/mechanisms.csv) |
| 提前预测整次调用耗时 | 预定误差倍数门槛为中位数不超过 1.30 且第 90 百分位不超过 2.00。主盲测的纯 StableHLO 回归在新尺寸上为 1.46 / 2.11，在新架构上为 2.16 / 7.76。独立后续组用 IR 加统计请求校准后分别为 1.13 / 2.32 和 2.15 / 2.51。 | 后续校准在熟悉结构达到了中位数目标，尾部仍超线。主盲测中，源码加 IR 的按图平均绝对对数误差较纯源码上升 2.8% 和 22.6%，没有达到预定的至少降低 10%。[主盲测](results/cpu_cost/analysis/accuracy.csv) · [增益判定](results/cpu_cost/analysis/summary.json) · [后续校准](results/cpu_cost/paired/accuracy.csv) |
| 预测观测增量并选出划算的方案 | 已在 33 张新图上直接盲测 477 个增量。新尺寸上，源码模型误差为 69.64 微秒，加入基线 IR 后为 69.57 微秒。训练未见的架构上，相应误差为 106.38 和 107.42 微秒。 | 两组的学习模型都优于猜零。额外 IR 未达到预定的 10% 改善门槛。架构迁移中，模型选出的实现相对固定联合归约平均更快，但只看源码也取得几乎相同的收益。[精度](results/cpu_cost/increment/accuracy.csv) · [实际选型](results/cpu_cost/increment/policies.csv) |
| 用 IR 预测 GPU 算子融合成本 | 已定位编译器里的真实决策。OpenXLA 的 GPU 优先融合会比较生产者与消费者分开执行及融合执行的估计时间，按预计节省的时间调整融合顺序。 | 预定的 36 个机制配置加 8 个 ResNet 配置，**GPU 实测 0/44**。下一步在同一设备上对齐 GPU HLO、实际内核和设备时间。用输出相同的融合与分离程序检验局部决策，再用同步的整次调用时间检验监控开销。[GPU 协议第 7 节](docs/plans/FusionScope_Phase3_Research_Plan.md) · [XLA 对候选融合打分的代码](https://github.com/openxla/xla/blob/ddaa568d48ad24c16e08bff00d3038bcf824eab9/xla/backends/gpu/transforms/priority_fusion.cc#L816-L869) |

## 能提前知道观测会增加多少微秒吗

对同一张图、同一进程，先分别取带观测与普通调用的多轮耗时中位数，再相减。五个进程的差值取中位数，正数表示增加，负数也保留。训练只用原有 32 张开发图。测试换成 24 张熟悉架构的新尺寸图，以及 9 张属于门控 MLP、ResNet 基本块和步幅瓶颈块的图。后三类架构全部留在训练之外。各模型都知道要返回什么统计，以及使用独立还是联合归约。获取新图特征时仅编译普通程序。GitHub 先保存了全部预测，再开始编译带观测的新图。[实验协议](results/cpu_cost/increment/protocol.json) · [计时前提交](https://github.com/ZZzJunzum/Compiler_report/commit/2ee7503)

| 每图等权的平均绝对误差（微秒，越低越好） | 零增量 | 只看源码 | 只看基线 HLO | 源码加基线 HLO |
|---|---:|---:|---:|---:|
| 24 张新尺寸图，360 个预测 | 110.29 | 69.64 | 69.50 | 69.57 |
| 9 张架构迁移图，117 个预测 | 132.96 | 106.38 | 108.64 | 107.42 |

在主测试组，加入 HLO 的误差变化为 −0.07 微秒，按同族尺寸配对重采样的 95% 区间为 [−1.30, +1.14]。在架构迁移组为 +1.03 微秒，区间为 [−1.15, +2.93]。7,650 项输出校验均通过。477 个增量中有 65 个超过同一可执行程序的重复计时波动门槛，79 个负增量均在各自门槛内。[逐进程增量与各方法预测](results/cpu_cost/increment/measurements.csv) · [误差和尾部误差](results/cpu_cost/increment/accuracy.csv) · [原始进程归档](results/cpu_cost/increment/runs)

这一预测有两个具体用途。有普通调用时间时，加上预测增量可以估算监控后的延迟预算。必须返回相同统计时，用较低的预测值可以选择实现。按实际选中程序的总耗时计，源码加 HLO 相对固定联合归约在两组的比值为 1.0002 和 0.9577，只看源码分别为 1.0037 和 0.9583。组合模型改选的 58 对均未单独超过相应波动门槛。取得基线 HLO 本身也有成本，33 张图的普通程序编译耗时中位数为 103.6 毫秒，提取特征为 1.5 毫秒。论文的决策流程图把这些调用级用途与 GPU 编译器对合法融合候选的打分相对应。[实际选择与配对区间](results/cpu_cost/increment/policies.csv) · [基线编译记录](results/cpu_cost/increment/lock.json) · [GPU 决策依据](docs/literature-review.md#谁会用到成本预测)

## 早期分块与选型结果

在 24 个新 CPU 条件上，封存模型相对**整批联合归约**的耗时比为 **0.861**，配对 95% 区间为 **0.797–0.943**，达到预登记标准。相对开发集选出的**固定 16 规则**，耗时比为 **0.981 [0.957, 1.007]**。计入决策成本并复用校准，调用 1,000 次时，静态模型相对这两个基线分别在 **2/24** 和 **4/24** 个主条件上净省时间。后续新卷积图上，单图分块取得四个明确加速和四个明确减速，候选 IR 在全部 12 个决策中都给出了与源码模型相同的选择。[独立复测与新图结果](#评审后的收益空间复测与部署成本)

每图只试联合归约和一个模型提名方案时，**24/24** 个选择与最快合法方案之间的配对差距落在同一可执行程序的重复测量波动范围内，其中 13 个正好选中实测最快方案。这个策略在 1,000 次调用内回本的条件为 **0/24**。正式确认的 **5,520** 项逐种子、逐进程校验全部通过，干净检出复现了 **36/36** 个机制配置的 HLO 指纹。

## 前三阶段怎样逐步收紧研究问题

1. **给成本模型找到一个实际决策。** 选题从用中间表示（IR）分析 GPU 算子融合出发。调试时取一个内部值可能改变生成的程序，这引出了更具体的问题。若一次推理必须返回同样的张量和激活统计，工程师可以怎样写这段观测代码，额外花费多少？数值一致性随之成为所有候选的准入条件。已有研究分别做过稀疏度采样、共享聚合和编译器自动调优，此处固定采样频率、两个观测点和主输出，在同一目标 CPU 上做成对比较。[最初任务书](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/docs/FusionScope_Research_Plan_v2.md) · [第二阶段的历史文献记录](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/docs/phase2-literature.md)

2. **先导发现，输出大小不足以定义一个公平的比较。** 复用 Lab 01 的 JAX 编译产物导出流程，先测只取主结果、取一个元素、取均值和取完整激活。4,096 元素的示例中，取样和均值都增加 4 字节，中位耗时分别是 28.75 和 32.45 微秒。两者提供的信息不同。于是固定一个可以严格核对的需求：对两处指定 ReLU 分别返回正元素数，并保留原主张量。对有限输入，ReLU 前大于零与 ReLU 后大于零逐元素等价。第一轮九配置小图在四种双点位置都得到 2117 和 2119，领先的两个位置落在计时波动范围内。[四种先导的原始表](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/cpu/pilot/four_probes.csv) · [九配置先导](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/cpu/pilot/first_report.md)

3. **先问“位置有用吗”。** 四类程序（仿射、矩阵乘、按行归一化、残差卷积）各取四档尺寸和链式/分支两种拓扑，得到 32 个条件。每个条件有无观测、四种单点和四种双点版本，一共 288 个配置。官方 Flax 的一个及两个相邻 ResNet 块另有 18 个配置。三个输入种子核对主张量的容差和两个整数的精确相等。五个发现进程、三个新确认进程和同一可执行程序的重复测量控制帮助区分小差异与波动。每个合成条件的四种位置均生成相同的优化后 HLO。128 个双点配置相对无观测的耗时比中位数却为 1.107，干净检出为 1.110。位置模型与固定后置的微小排序也随复跑翻转。[第一阶段结果](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/cpu/report/REPORT.md)

4. **用反例限定这个结论。** 两个 ResNet 子图各保留四类 HLO，但最快位置仍未稳定分出。把小仿射图和真实模块各用两种 ReLU 表达式编译，直接写 maximum 时四种位置合一，用 jax.nn.relu 时保留四种指纹。本次复核补上了编译过程。在 CPU 首轮代数化简刚结束时，显式 maximum 的 8 个后置计数全部改读此前的加法，库函数的 8 个仍读 maximum。16/16 个最终 HLO 指纹匹配早期归档，48 项输出校验通过。又用独立单点与双点计时检验成本能否相加。最大归一化图的链式和分支两种拓扑、各四种放置，都重复出现双点成本低于单点增量之和。显式拆成两个可执行程序还会改变调用和物化，因此下一步沿输出路径检查实际工作。[外测和交叉对照](https://github.com/ZZzJunzum/Compiler_report/tree/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/cpu/lowering_aligned) · [独立复跑](https://github.com/ZZzJunzum/Compiler_report/tree/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/reproducibility) · [逐步编译对照](results/placement_pass/comparisons.csv)

5. **从三个返回值倒查是谁在重复工作。** 把 HLO 融合体的形参绑定到调用处的实参，沿主张量及两个计数各走一遍，核对重复算术、归约树和跨边界的值。在 4096 × 256 仿射分支上，第一层加法走了三条路径。两个独立归约各写出一个 4 MiB 整数掩码，得到分配记录与优化后 LLVM 写指令的双重印证。大卷积图则已经把比较放进归约融合体，完整掩码导出为零。这给出了同一优化应当收益不同的两个案例。[HLO、缓冲区及 LLVM 逐行审计](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/phase2/analysis/PHASE0.md)

6. **实际修改生成程序的方式。** 对同一份统计实现独立归约、联合归约、显式分段和分块部分和。先筛查五个案例，再做 250 个合成配置及 8 个外测配置。相对独立计数，大仿射分支联合后两个完整掩码都消失，额外带标签算术副本从七处降到五处，独立确认提速 3.07 倍，干净复跑为 2.92 倍。16 行独立分块取得 1.86 倍，仍保留七处副本。大卷积分成每块两个样本时速度仅有整批独立计数的 0.43 倍。改变调度确实改进了某些程序，也带来了需要选择方案的新问题。[调度实现和完整结果](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/phase2/report/REPORT.md)

7. **解释第一版成本模型为什么永远选同一个。** 按整个图族留出测试，原解析模型对全部 32 个合成条件都选整批联合。逐项检查发现，186 个同图替代方案在调用、算术、搬运、归约层数和循环步数五项上都大于或等于联合策略，归约层数还严格更大。正系数使排序在拟合前就已决定。岭回归降低了时间预测误差，却在 12 次改选中仅 2 次变快、10 次变慢。新的模型需要能表达真实的排序翻转，实验则需要强固定策略作对照。[冻结的特征和预测](results/phase2/model) · [诊断计算](paper/check_diagnostics.py)

8. **分离“主计算变快”和“监控本身变便宜”。** 在四张旧图上把三种分块范围与主程序单独运行、独立计数、联合计数交叉，共 36 个配置。8 个非平凡交互中有 6 个超过相应波动门槛。归一化分支按 16 行分块后，带联合计数从 198.90 降到 105.40 微秒，主程序单独运行也从 186.30 降到 89.95 微秒。两种分块下按进程配对的计数增量分别为 11.75 和 15.55 微秒。这个案例的大部分收益来自主程序。[同分块有无观测对照](results/phase3/cpu/analysis/interactions.csv)

9. **先锁定推荐，再用新图验收。** 32 个旧条件、388 个配置用来开发和选择模型。把联合分块加入候选，分开估计工作集、累计访问、内核吞吐和循环成本，引入算子类别与分块的交互。先封存模型及推荐，然后分别用两个新进程试跑两候选策略并再次封存，最后才用五个独立进程确认。24 个主条件和 4 个混合迁移图共 368 个配置，其中 340 个返回两个计数。静态模型在主条件上的几何平均耗时比为 0.861，达到预登记标准；混合图为 1.047。归一化受益最大，稠密与卷积平均退化。这轮验收的 5,520 项配置、种子和进程组合校验全部通过，干净检出也复现了 36/36 个机制配置的 HLO 指纹。[注册方案](docs/plans/FusionScope_Phase3_Research_Plan.md) · [静态封存](https://github.com/ZZzJunzum/Compiler_report/tree/a8fd22575ffc293c850b922b95e1feb5e5e68272/results/phase3/cpu/static) · [试跑封存](https://github.com/ZZzJunzum/Compiler_report/tree/ed3876906c9431baa2f9465d6449c36d94bd9042/results/phase3/cpu/decision)

10. **再问“模型多提供了什么”。** 以下是确认完成后的复核。只用旧开发数据选择一条固定联合分块规则，每块取不超过 16 的最大合法行数或样本数，整批可装下便直接用整批。它在主条件上的耗时比为 0.876。完整模型直接与它配对的比值为 0.981，95% 区间 [0.957, 1.007]。用源码重建字节数并清除 HLO 工作量、扇出和共享前缀，保留原权重后 28/28 个选择相同；连训练也移除这些特征，主耗时比为 0.869。同样是 128 × 320 的仿射图，模型都选 64 行，链式图慢 9.8%，分支图却快 54.9%。结构信息有用的具体候选就在这对图里，当前模型的加法项会在候选相减时抵消。[逐图决策](results/phase3/cpu/analysis/decisions.csv) · [确认后的复算入口](paper/check_diagnostics.py)

11. **实际改变及边界。** 已经交付可执行的替代计数方案、可复核的融合分析和一个能提前选择合法方案的 CPU 原型。联合计数对某些现有工作负载带来实测收益；模型相对更强固定规则的额外收益仍难以分辨。若从五个进程的单图中位数推算每图各调用一次，总和的中位比为 1.005。若把决策成本也计入，复用校准后，静态选策和双候选试跑在 1,000 次内分别只在 2/24 和 0/24 个条件回本。它们回答了什么时候值得多做编译和试跑。研究结论覆盖这台 Intel CPU、这些形状及两个精确计数；XLA 上游优化、GPU 的物理流量、完整训练模型和跨设备策略迁移是不同的验证任务。[部署成本](results/phase3/cpu/analysis/break_even.csv) · [原始计时](results/phase3/cpu/analysis/process_medians.csv)

## 评审后的收益空间、复测与部署成本

原始课题能支持一项关于编译行为的实证研究。两个精确计数把观测信息、输出字节数与频率固定下来，便于检验融合、物化和重复计算。现有应用文献支持监测激活稀疏度的实际用途，其中 Spartan 研究的是训练与自适应监测。生产系统是否需要逐次、逐层精确计数，仍要用真实使用记录验证。其余逐篇对照及当前能够支持的四项结论见[文献评审](docs/literature-review.md)。[Spartan 原论文](https://eprints.gla.ac.uk/240696/) · [SystemML 原论文](https://www.vldb.org/pvldb/vol11/p1755-boehm.pdf)

评审后先算当前候选菜单中还能节省多少。固定规则始终用联合计数，每块取不超过 16 的最大合法尺寸，能装下整批便直接使用整批。在已有 32 张开发图的 356 个合法观测候选中，逐图选最低中位数，相对固定规则的单次节省中位数仅 **6.73 微秒**。为避免从同一批噪声中挑赢家，我们先封存全部 32 个逐图选择，再用五个新进程复测。复测的单次配对节省中位数为 **5.10 微秒**，16 图超过各自的重复测量误差，16 图不能分辨。原先越过误差门槛的 16 图有 14 图再次越过。四张批量为 4 或 8 的卷积图占复测正向绝对节省的 **83.2%**，这个比例按每个图调用一次计算。[逐图上限与独立复测](results/headroom/analysis/headroom.csv) · [候选封存](https://github.com/ZZzJunzum/Compiler_report/tree/9aa6b689bb0df95a61da67a9092d6b58313ad70a/results/headroom)

理想选择器在有限调用次数下也需要考虑编译。我们对每个合法方案计算“调用次数 × 相对固定规则节省的单次时间 − 额外编译时间 − 额外信息成本”，取菜单中的最大值。旧开发集调用 1,000 次时，免费知道候选身份有 **19/32** 图净省，另付基线编译与 IR 提取费用降为 **10/32**，再计冷校准降为 **2/32**。这是经验上限。重新核算第三阶段已封存的真实策略，在其 24 个主条件上，复用校准、相对固定 16 规则，静态模型和双候选试跑分别在 **4/24** 和 **0/24** 图净省。原论文中的 2/24 与 0/24 以整批联合归约为分母。[逐图、逐调用次数的上限](results/headroom/analysis/oracle_budget.csv) · [已执行策略的实际账目](results/headroom/analysis/historical_deployment.csv)

**复现了机会所在，也测出了规则迁移失败的边界。** 四个旧卷积图上，联合计数每次处理一张图像，单次耗时均比固定 16 规则低约 45%–51%。于是按事先约定，在六个新形状上各造两张同算术量的读出图，仅改变最后一项读取输入还是第一层激活。先锁定固定规则、卷积例外、形状模型和源码结构模型，再编译两个候选以取得 IR 并锁定第五种策略，最后用五个新进程计时。原来那对 128 × 320 仿射图没有充当新证据。[源码策略封存](https://github.com/ZZzJunzum/Compiler_report/tree/45b3879b6536674dbcec7dd13ebd68374d16cd34/results/headroom/conv_transfer) · [计时前的候选 IR 与预测](https://github.com/ZZzJunzum/Compiler_report/tree/9ed13d9b8a1efd37a87ac767bd055b452e8affbe/results/headroom/conv_transfer)

| 卷积形状与样本范围 | 单图联合计数相对整批联合的耗时比 | 实验判断 |
|---|---:|---|
| 28 × 28、16 通道，批量 4 或 8，两种读出 | 0.531–0.587 | 四图均明确加速 |
| 36 × 36、24 通道，同样的批量与读出 | 1.214–2.443 | 四图均明确减速 |
| 八个目标图等权汇总 | 0.992 | 预定通过线为不高于 0.95、至少六图明确加速、无超过 10% 的明确减速，实际四赢四输 |
| 批量 16 的四张边界图 | 模型统一分单图，几何平均 1.413 | 预定例外在四图均回退到整批，耗时比为 1 |

四种批量与读出组合的符号按空间尺寸和通道数组合翻转。12 张图的同分块无计数对照均给出相同方向。两个维度同时改变，不能单独归因给其中一个维度或某个 XLA 优化步骤。旧训练卷积的空间尺寸和通道数只有一个取值，无法通过训练估计这两项变化的作用。即便在优化后 HLO 中逐图找到正确的读出操作数，形状、源码结构和候选 IR 三个模型仍在 12/12 图上都选单图。在八个目标图上，IR 相对源码模型单次节省为零，每图额外设置时间中位数约 151 毫秒。720 项输出检查全部通过，120 个独立编译指纹与计时前的记录吻合。按已封存标准，结束这组候选上的自动选型路线，保留已复现的具体机制、强固定对照和适用边界。[逐案主计算对照](results/headroom/conv_transfer/analysis/main_only.csv) · [各策略决策与成本](results/headroom/conv_transfer/analysis/decisions.csv) · [验收结论](results/headroom/conv_transfer/analysis/verdict.json)

算法选择研究把单一策略和各实例菜单内的最快项同时作为参照，并计入特征收集成本。最小化带噪估计还会产生选择偏差。这两点分别决定了本轮的比较对象和先封存再独立复测的做法。[ASlib 原论文](https://ada.liacs.nl/papers/BisEtAl16.pdf) · [Cawley 与 Talbot](https://www.jmlr.org/papers/v11/cawley10a.html)

## 第三阶段证据

| 要核对的问题 | 入口 |
|---|---|
| 模型在每个图上选了什么，离最快方案多远？ | [逐图决策、精确计数有效性及噪声阈值](results/phase3/cpu/analysis/decisions.csv) · [逐候选排名](results/phase3/cpu/analysis/rankings.csv) |
| 分块怎样改变观测的额外代价？ | [按进程配对的交互量](results/phase3/cpu/analysis/interactions.csv) · [输出路径和 HLO 行号](results/phase3/cpu/analysis/paths.csv) · [机制图](results/phase3/cpu/figures/mechanism.pdf) |
| 一次部署要编译、试跑多少，多少次调用能回本？ | [冷启动与复用成本](results/phase3/cpu/analysis/setup.csv) · [逐图回本次数](results/phase3/cpu/analysis/break_even.csv) · [复用曲线](results/phase3/cpu/figures/reuse.pdf) |
| 如何核对测量和编译产物？ | [逐进程中位数](results/phase3/cpu/analysis/process_medians.csv) · [实验实现](fusionscope/phase3) · [逐阶段验证](results/phase3/cpu/verification.json) · [干净检出对照](results/phase3/cpu/analysis/replay.csv) · [SHA-256 清单](results/phase3/cpu/SHA256SUMS.json) |

每个进程 ZIP 内有环境、候选编译时间、所有预热和计时样本。正式队列还含逐种子验证。编译表中的 archive 和 component 定位相应的 StableHLO 与优化后 HLO。阶段一、二的完整程序、散装 IR、早期计划和报告保存在固定历史版本，当前分支保留了复核旧模型所需的冻结输入。[两阶段完整快照](https://github.com/ZZzJunzum/Compiler_report/tree/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325)

## 复现

编译过程检查可以从每个配置的两份相邻 HLO 和最终 HLO 重建全部 16 行对照。要独立重做这 16 次 CPU 编译，使用下面的第二个命令，默认输出写入本地复现目录。

~~~powershell
.\.venv\Scripts\python -B -m paper.placement_checkpoints verify --out results/placement_pass
.\.venv\Scripts\python -B -m paper.placement_checkpoints record
~~~

在仓库根目录及已安装锁定依赖的 Python 环境中，用下列命令从已发布数据核验 CPU 结果。最后一条重读 65 份未观测基线 HLO，从 32 张开发图的原始配对时间重新拟合模型，比对全部封存预测，然后从五个新进程的计时与输出检查重算表格。

~~~powershell
$py = (Resolve-Path .venv/Scripts/python.exe).Path
& $py -B -m fusionscope.cpu_cost.experiment verify --out results/cpu_cost
& $py -B -m fusionscope.cpu_cost.mechanisms verify --out results/cpu_cost
& $py -B -m fusionscope.cpu_cost.paired_model verify --out results/cpu_cost
& $py -B -m fusionscope.cpu_cost.increment verify --out results/cpu_cost/increment
~~~

单独复跑“基线 IR 预测增加多少微秒”时，沿用已归档的开发数据。新文件默认写入 `reproduction/cpu_increment`。第一步只编译无观测程序，生成的预测锁固定后才进入新候选编译和计时。

~~~powershell
& $py -B -m fusionscope.cpu_cost.increment lock
& $py -B -m fusionscope.cpu_cost.increment test
& $py -B -m fusionscope.cpu_cost.increment report
& $py -B -m fusionscope.cpu_cost.increment verify
~~~

独立重采样默认写入 Git 忽略的 `reproduction/cpu_cost`。依次执行开发、预测封存、参考 HLO 检查与封存、独立计时。计时结束后的联合 HLO 检查用于解释已测差异。末尾的小组实验重用开发计时，所有测试图与主盲测互斥。

~~~powershell
& $py -B -m fusionscope.cpu_cost.experiment develop
& $py -B -m fusionscope.cpu_cost.experiment predict
& $py -B -m fusionscope.cpu_cost.experiment inspect
& $py -B -m fusionscope.cpu_cost.experiment test
& $py -B -m fusionscope.cpu_cost.experiment report
& $py -B -m fusionscope.cpu_cost.mechanisms collect
& $py -B -m fusionscope.cpu_cost.mechanisms analyze
& $py -B -m fusionscope.cpu_cost.paired_model lock
& $py -B -m fusionscope.cpu_cost.paired_model test
& $py -B -m fusionscope.cpu_cost.paired_model report
& $py -B -m paper.absolute_latency_figure --data reproduction/cpu_cost
~~~

在仓库根目录，用 Python 3.12 建立环境。下列命令在 Windows PowerShell 中运行。完整环境版本见 [Windows 依赖锁](requirements-windows-lock.txt)。

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-windows-lock.txt
.\.venv\Scripts\python -m paper.check_diagnostics
.\.venv\Scripts\python -m fusionscope.phase3.experiment verify --out results/phase3/cpu
.\.venv\Scripts\python -m paper.headroom_analysis
.\.venv\Scripts\python -m paper.conv_transfer_analysis
~~~

要独立重跑评审后的实验，以下命令从已有第三阶段开发归档取得训练数据，新的运行写入本地复现目录。用第一轮的结果重新判断是否进入新图测试。源策略在编译新候选前封存，IR 策略在试跑新候选前封存。默认复核命令从已归档原始数据重算并比对表格。[协议、比较对象和停止条件](docs/plans/CPU_Headroom_Audit.md)

~~~powershell
$py = (Resolve-Path .venv/Scripts/python.exe).Path
& $py -m fusionscope.headroom freeze
& $py -m fusionscope.headroom repeat
& $py -m paper.headroom_analysis --data reproduction/headroom --write
& $py -m fusionscope.conv_transfer freeze
& $py -m fusionscope.conv_transfer inspect
& $py -m fusionscope.conv_transfer measure
& $py -m paper.conv_transfer_analysis --data reproduction/headroom/conv_transfer --write
~~~

要重新采样，在空输出目录按顺序运行。第二、三行之间的两次封存分别固定静态预测与试跑选择。再运行确认时，两份锁及其输入会被校验。

~~~powershell
$py = (Resolve-Path .venv/Scripts/python.exe).Path
& $py -m fusionscope.phase3.experiment mechanism development calibrate baselines lock
& $py -m fusionscope.phase3.experiment discovery choose
& $py -m fusionscope.phase3.experiment confirmation analyze
& $py -m fusionscope.phase3.profile --out reproduction/phase3
~~~

复跑时从同一代码版本建立另一个干净检出，在该检出中用同一 Python 运行机制实验。把输出写入上面那次实验的 replay 子目录，然后在原仓库核验。

~~~powershell
$replay = Join-Path (Resolve-Path reproduction/phase3) 'replay'
git worktree add --detach .scratch/replay-current HEAD
Push-Location .scratch/replay-current
& $py -m fusionscope.phase3.experiment mechanism --out $replay
Pop-Location
& $py -m fusionscope.phase3.experiment verify --out reproduction/phase3
~~~

论文采用 ACM SIGPLAN 的 10 pt 匿名双栏审稿格式。机制图由 LaTeX 直接绘制，调用成本图来自已归档的实验。安装包含 acmart、TikZ 和 BibTeX 的 TeX 发行版后，从仓库根目录构建。构建中间文件写入本地复现目录，发布目录只需最终 PDF。[CC 投稿格式说明](https://conf.researchr.org/track/CC-2027/CC-2027-research-papers)

~~~powershell
New-Item -ItemType Directory -Force reproduction/paper | Out-Null
pdflatex -halt-on-error '-output-directory=reproduction/paper' paper/FusionScope.tex
bibtex reproduction/paper/FusionScope
foreach ($pass in 1..4) {
    pdflatex -halt-on-error '-output-directory=reproduction/paper' paper/FusionScope.tex
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX build failed' }
}
Copy-Item -LiteralPath reproduction/paper/FusionScope.pdf -Destination output/pdf/FusionScope.pdf -Force
~~~
