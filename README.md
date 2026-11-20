# FusionScope

**项目交接**　[英文论文 PDF](output/pdf/FusionScope.pdf) · [可编辑的 LaTeX](paper/FusionScope.tex) · [逐篇文献与贡献边界](docs/literature-review.md) · [复现方法](#复现)

英文稿的阅读顺序是研究动机（§1）、已有工作和留给本文的问题（§2）、实验口径（§3），接着依次读位置（§4）、改写与适用范围（§5）、方案选择（§6）、耗时预测（§7）。§8 回到 GPU 决策并交代验证范围，§9 给出结论。

我们最初想用编译器中间表示（IR）预测同一段计算在 GPU 上分开执行和融合执行各需多久，帮助编译器选择更快的方案。已完成的实证工作使用 JAX/jaxlib 0.9.0.1，在 Intel Core i5-13420H 的一个逻辑核上运行。普通推理只返回主张量 y，观测版还要返回两个激活计数。**额外的统计给激活增加了消费者。HLO 是 XLA 描述程序的中间表示，读取它能追踪这些消费者如何被编译。** 我们检验能否用更便宜的写法返回同一组结果。为预测而取得基线 HLO 所需的编译和提取时间另行计账。

| 要回答的问题 | 比较什么 | 结果的单位 |
|---|---|---|
| 增加观测贵多少？ | 同图带观测的调用与普通调用相减，两者使用相同输入。若主计算也改了分块，就用同分块的普通调用作对照。 | 每次调用增加的微秒 |
| 怎样便宜地返回同一份统计？ | 主张量、统计定义和输出顺序都固定，换观测位置、独立或联合归约、分块方式。 | 配对延迟比、超出波动的胜负，以及计入编译后多少次调用回本 |
| IR 能否指导 GPU 融合？ | 同一语义的合法候选，比较生产者与消费者分开和融合的设备执行时间，再看编译器选中了哪一个。 | 设备时间差与选型收益，等待 GPU 实测 |

文中用同一可执行程序的重复计时估计本机波动。在五进程确认中，“可分辨的变快/变慢”表示进程内配对差值的中位数绝对值超过对应 A/A 经验门槛。非零中位数已经保证严格多数同号，因此多数条件不提供额外稳健证据。A/A 差值共享别名中位数、彼此相关；其经验 95 分位数是重复性筛选门槛，不是校准过的 95% 显著性检验。历史两进程 discovery 保留其原有多数筛选。增量预测的 bootstrap 区间条件于固定家族与已观测进程，不能替代跨批次、跨日期复验。

### 当前研究收敛与第一步修正

[研究中心与分阶段实验协议](docs/plans/Observation_Mechanism_Research_Plan.md) 把下一轮工作集中到精确统计的联合归约何时获益、何时退化，以及能否从具体编译机制推导可验证的适用条件。该文档是待执行的研究计划，尚未证明新自动改写或新性能收益。现有预测、物化扫描与分块实验作为证据和强对照保留。

在此分支可运行 `python -B -m paper.check_repeatability`，从五份原始 ZIP 重建全部 477 项增量和 180 项选择比较，并检验奇数/偶数面板、零值、门槛相等与非法输入。该检查重用归档计时，不产生新的性能测量。

## 哪些环节先引用文献

| 文献已建立的事实 | 本文接着验证的问题 |
|---|---|
| Cano 等在 TPU 上强制拆开融合核时，窄类型写回恢复了缺失的舍入。JAX 文档也记录了打印 ReLU 前的值可能迫使中间结果物化。 | 若只要求返回可逐项验证的计数，移动计数输入能改变 CPU 上的优化后程序吗？加入计数实际多花多少时间？[TPU 论文 §4.3、图 4](https://arxiv.org/pdf/2611.00624#page=8) · [JAX 的物化示例](https://docs.jax.dev/en/latest/debugging/print_breakpoint.html#unnecessary-materialization) |
| SystemML 实测了共享输入的多个聚合，并建模重算与物化。Weld 在一次遍历中同时返回向量和标量和。Jung 等在卷积生产者处累计两项归一化统计量。 | 保留主张量、从不同层额外取统计时，XLA CPU 究竟写出哪些数组？已知的联合归约能消去哪一项写入，消去后是否更快？[SystemML 图 2、9](https://www.vldb.org/pvldb/vol11/p1755-boehm.pdf) · [Weld Listing 6](https://www.vldb.org/pvldb/vol11/p1002-palkar.pdf#page=6) · [Jung 等图 5](https://proceedings.mlsys.org/paper_files/paper/2019/file/9db64c20dee0011899dfdf200e61ef35-Paper.pdf) |
| Snider 与 Liang 分析 XLA 的 GPU 融合决策，Kaufman 等用已形成的 XLA TPU 核预测运行时间并评估融合、分块选择。OpenXLA 的 GPU 优先融合实现按预计节省的执行时间排列候选。 | 在观测实现还没有编译时，普通程序的 IR 比源码多提供多少预测信息？它能否在新图上选出实际更便宜的实现？[Snider 与 Liang](https://arxiv.org/abs/2301.13062) · [Kaufman 等](https://proceedings.mlsys.org/paper_files/paper/2021/file/6bcfac823d40046dca25ef6d6d59cc3f-Paper.pdf) · [OpenXLA 融合评分实现](https://github.com/openxla/xla/blob/ddaa568d48ad24c16e08bff00d3038bcf824eab9/xla/backends/gpu/transforms/priority_fusion.cc#L816-L869) |

## 论文怎样从一个问题走到下一个

论文按问题组织证据。第 5 节把后做的九类请求实验放回机制分析，第 6 节集中评估早期两计数选型。各研究仍使用各自的开发集、测试集与封存记录。版本和原始时间线可在[历次实验详录](https://github.com/ZZzJunzum/Compiler_report/blob/be37dc93d5f354e5f98a8e91a90906c0219def3e/README.md)中核对。

1. **先让性能比较回答同一个需求（§1–3）。** 在 4,096 个 FP32 元素上，普通调用为 27.55 微秒。取一个元素和取均值都只多返回 4 字节，分别耗时 28.75 和 32.45 微秒，返回整个张量反而是 30.20 微秒。这促使我们把比较固定为同一主张量、同两处 ReLU、每次调用各返回一个精确正元素计数。随后每种候选都核对输出，编译与预热单列，计时一直等到全部输出就绪。[先导原始数据](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/cpu/pilot/four_probes.csv) · [最初任务书](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/docs/FusionScope_Research_Plan_v2.md)

2. **先试改动最小的办法，移动观测位置（§4）。** 对有限输入，ReLU 前后大于零的是同一批元素，两层各选前或后就有四种等价写法。仿射、稠密乘、归一化和残差卷积共 32 个合成条件、288 个配置。每个条件内部的四种双点写法都得到同一个优化后 HLO 指纹，但 128 个双点配置相对普通调用的配对耗时比中位数仍为 **1.107**，干净复跑为 **1.110**。因此只移动这两条观测边，未消除这里约一成的开销。[位置与耗时结果](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/cpu/report/REPORT.md)

3. **追问位置策略的边界（§4）。** 两个 Flax ResNet 子图各自保留四种指纹，位置间的领先差异却落在计时波动内。我们在一张仿射链和一个 ResNet 块上交叉检查两种 ReLU 写法。首轮代数化简后，显式 maximum 的 8 个后置比较都改读前面的加法，库函数 jax.nn.relu 的 8 个仍读 maximum。16 次最终编译与归档指纹一致。这解释了这些图上比较指令的去向。要解释剩余延迟，还得顺着主输出与计数各自的路径找工作。[逐配置的操作数](results/placement_pass/comparisons.csv) · [相邻优化步骤的 HLO](results/placement_pass/checkpoints.zip)

4. **用已有的共享归约方法改程序，验证它消除了什么（§5）。** 在 4096 × 256 仿射分支上，第一层加法出现在主输出和两个计数的三条路径中，独立计数各写出一张 4 MiB 整数掩码。HLO、缓冲区分配和 LLVM 写指令给出三层证据。改为联合归约后，两张掩码消失，计划临时空间由 8,392,704 字节降至 16 字节，相对独立计数提速 **3.07 倍**，干净复跑 **2.92 倍**。大卷积的独立版本本来就没有导出完整掩码，改按两幅图像分块后速度仅为整批独立版本的 0.43 倍。这个反例要求把“减少写入”“改变主计算”和“调用变快”分别验证。[生成代码审计](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/phase2/analysis/PHASE0.md) · [改写与复测](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/phase2/report/REPORT.md)

5. **检验收益归谁，以及方法适用于哪些请求（§5）。** 匹配同一分块的无观测对照后，归一化分支的带计数调用从 198.90 降到 105.40 微秒，计数本身的配对增量却从 11.75 升至 15.55 微秒，主要收益来自主计算。再对 22 张新图各做 9 类请求，198 个图与请求组合中有 120 个满足直接联合归约的形状和归约轴条件。其中 82 个消除了目标导出，**30 个明确变快、4 个明确变慢、48 个未分辨**。这既限定了改写的适用条件，也解释了为什么还值得研究怎样选实现。[同分块对照](results/phase3/cpu/analysis/interactions.csv) · [逐对胜负](results/cpu_cost/analysis/rewrite_pairs.csv) · [其余 78 个组合的原因](results/cpu_cost/analysis/boundaries.csv)

6. **检验模型是否真能替用户做出更好的选择（§6）。** 第一版正权重模型的 186 个替代方案，在五项特征上都被整批联合归约支配，其中归约层数严格更多，因此怎样拟合这些正权重都会选同一方案。改进后的模型允许分块与算子类型交互，封存后在 24 张主测试图上相对整批联合归约的耗时比为 **0.861 [0.797, 0.943]**。换成只用开发数据挑出的固定 16 规则，耗时比为 **0.981 [0.957, 1.007]**。冻结权重、用源码重建字节数并移除图级 IR 特征后，**28/28** 个选择相同。后续 12 张卷积图的结果还显示，旧图上有收益的单图分块会在新形状上改变方向。至此，已验证的是个别图存在选型机会，通用选择收益仍取决于对照策略和部署成本。[模型审计](paper/check_diagnostics.py) · [预登记主结果](results/phase3/cpu/analysis/endpoint.json) · [收益空间与部署账目](results/headroom/analysis) · [卷积迁移](results/headroom/conv_transfer/analysis/verdict.json)

7. **回到成本预测，区分整次调用与观测增量（§7）。** 先给模型看带观测候选在后端优化前的 StableHLO，预测整次调用。纯 IR 在新尺寸和新架构上的中位误差倍数分别为 1.46 和 2.16。直接预测增量时，可用的编译器信息限于普通程序的优化后 HLO。我们沿用 32 张开发图，在编译任何带观测测试候选前封存 33 张新图的 477 个增量预测。下面是按图等权的平均绝对误差。所有学习模型都知道请求的统计与实现方式。[整次调用误差](results/cpu_cost/analysis/accuracy.csv) · [两种实现的独立排序试验](results/cpu_cost/paired/policies.csv) · [增量封存提交](https://github.com/ZZzJunzum/Compiler_report/commit/2ee7503fcd38455cc89c393d17a3fc94f3a5cb6b)

| 测试图 | 猜零增量 | 只看源码 | 只看基线 HLO | 源码加基线 HLO |
|---|---:|---:|---:|---:|
| 熟悉结构的新尺寸，24 张 | 110.29 µs | 69.64 µs | 69.50 µs | 69.57 µs |
| 训练未见的架构，9 张 | 132.96 µs | 106.38 µs | 108.64 µs | 107.42 µs |

源码加入基线 HLO 后，误差变化分别为 −0.07 微秒（95% 区间 [−1.30, +1.14]）和 +1.03 微秒（[−1.15, +2.93]）。新尺寸上约 0.1% 的改善低于预先设定的 10%。五个进程的 7,650 项输出校验全部通过，477 个实测增量中有 65 个超过各自的计时波动门槛。对于这台 CPU、这些图和特征，源码已经取得了相对零增量的大部分预测收益。[误差与区间](results/cpu_cost/increment/accuracy.csv) · [逐图预测与计时](results/cpu_cost/increment/measurements.csv) · [实际选型收益](results/cpu_cost/increment/policies.csv)

## 结论和下一位接手的工作

**本文贡献（§9）**。在固定输出语义的 XLA CPU 程序上，我们定位了两张交叉对照图上后置比较在哪轮优化改读前驱，追到了能用已知改写消去的实际写入，也给出了去掉导出仍会减速的实例。选型和预测结果都给出了强对照。相对于开发集选出的固定调度或仅用源码的预测，额外读取 IR 的收益仍未得到支持。

**CPU 上仍待解释的机制**。同一张仿射图上，三个全局计数采用联合归约后变快，两个末轴计数却变慢，这个比较同时改了观测点数和归约轴。若要解释四个减速实例，可以先固定两个点，只切换归约轴，在相同进程中配对比较联合与独立归约的耗时差，再检查循环和向量化。现有缓冲分配记录给出计划容量，实际缓存和 DRAM 流量仍待测量。[逐请求反例](results/cpu_cost/analysis/rewrite_pairs.csv) · [同点对照设计](docs/literature-review.md#据此怎样推进)

**GPU 接续（§8）**。协议的 36 个机制配置与 8 个 ResNet 配置目前实测 **0/44**。有 NVIDIA 服务器后，先复现固定输出的正反案例，把优化后 GPU HLO 对到实际内核启动，同时记录设备时间和同步的整次调用时间。要回答最初的融合预测问题，还需构造同一语义的合法融合与分离候选，在独立图上事前封存预测，与 XLA 自带估计和固定策略比较误差及所选实现的实测收益。44 个机制配置本身只承担跨设备复现。[GPU 协议第 7 节](docs/plans/FusionScope_Phase3_Research_Plan.md#7-gpu-validation-and-server-request) · [XLA 从 HLO 到设备工作](https://openxla.org/xla/hlo_to_thunks)

## 文件入口

当前版本包含论文、源码、复现命令与核心计时归档。阶段一、二的完整编译产物和早期任务书在下表所列的固定历史版本中。

| 交接内容 | 文件或目录 |
|---|---|
| 稿件、原始文献和写作依据 | [英文论文](output/pdf/FusionScope.pdf) · [LaTeX 源码](paper/FusionScope.tex) · [BibTeX](paper/references.bib) · [逐篇文献对照](docs/literature-review.md) |
| 最初为何改题、早期机制图怎样生成 | [两阶段历史快照及任务书](https://github.com/ZZzJunzum/Compiler_report/tree/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325) · [CPU 调度与对照实验计划](docs/plans/FusionScope_Phase3_Research_Plan.md) |
| 编译、计时、解析与验证 | [早期选型实现](fusionscope/phase3) · [观测请求与成本预测实现](fusionscope/cpu_cost) · [论文审计与作图](paper) |
| 原始样本及派生结果 | [位置改写](results/placement_pass) · [分块与部署](results/phase3/cpu) · [收益上限与复测](results/headroom) · [九类请求](results/cpu_cost/analysis) · [最终增量盲测](results/cpu_cost/increment) |

## 复现

在仓库根目录，用 Python 3.12 建立环境。下列命令在 Windows PowerShell 中运行。完整环境版本见 [Windows 依赖锁](requirements-windows-lock.txt)。

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-windows-lock.txt
~~~

### 核验已发布的结果

在同一个 PowerShell 会话中执行下列命令，可以从已归档的 HLO、冻结预测和逐进程耗时重建结果。第一条验证 16 次位置编译。第二条先运行封存时的选型审计，再计算事后用两次整次调用预测相减的误差。最后一条用 32 张开发图重新拟合直接增量模型并核对 477 个盲测预测。

~~~powershell
$py = (Resolve-Path .venv/Scripts/python.exe).Path
& $py -B -m paper.placement_checkpoints verify --out results/placement_pass
& $py -B -m paper.check_paper_results
& $py -B -m fusionscope.phase3.experiment verify --out results/phase3/cpu
& $py -B -m paper.headroom_analysis
& $py -B -m paper.conv_transfer_analysis
& $py -B -m fusionscope.cpu_cost.experiment verify --out results/cpu_cost
& $py -B -m fusionscope.cpu_cost.mechanisms verify --out results/cpu_cost
& $py -B -m fusionscope.cpu_cost.paired_model verify --out results/cpu_cost
& $py -B -m fusionscope.cpu_cost.increment verify --out results/cpu_cost/increment
~~~

### 独立重跑 CPU 实验

<details>
<summary>重新编译、封存预测和采集计时的完整命令</summary>

下面的命令沿用上面的 Python 环境，新产物默认写入本地复现目录。独立重做 16 次位置编译和相邻优化步骤留档使用以下命令。

~~~powershell
& $py -B -m paper.placement_checkpoints record
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

从含 Git 历史的本地克隆建立同一版本的干净检出，在该检出中用同一 Python 运行机制实验。把输出写入上面那次实验的 replay 子目录，然后在原仓库核验。

~~~powershell
$replay = Join-Path (Resolve-Path reproduction/phase3) 'replay'
git worktree add --detach .scratch/replay-current HEAD
Push-Location .scratch/replay-current
& $py -m fusionscope.phase3.experiment mechanism --out $replay
Pop-Location
& $py -m fusionscope.phase3.experiment verify --out reproduction/phase3
~~~

</details>

### 构建英文论文

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
