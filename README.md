# FusionScope

在 JAX/XLA 中，一次推理要同时返回主张量和两个精确的正激活计数。这个项目研究观测位置与归约方式怎样改变 CPU 可执行程序，以及编译前选择策略的收益能否抵消选择本身的开销。

**[英文双栏论文 PDF](output/pdf/FusionScope.pdf)** · [LaTeX 源码](paper/FusionScope.tex) · [参考文献](paper/references.bib) · [CPU 主结果](results/phase3/cpu/analysis/endpoint.json) · [各策略结果](results/phase3/cpu/analysis/policy_summary.csv)

24 个新 CPU 条件上，封存模型相对固定联合归约的耗时比为 **0.861**，配对 95% 区间为 **0.797–0.943**，达到预注册的三个标准。它在 8 个条件上更快、12 个更慢、4 个作出相同选择。归一化的耗时比为 0.598，稠密矩阵乘法为 1.077。4 个混合算子迁移条件的耗时比为 1.047。把选择过程也计入，复用已有校准且调用 1,000 次时，静态模型在 **2/24** 个主条件上回本。

每图只试联合归约和一个模型提名方案时，**24/24** 个选择与最快合法方案之间的配对差距落在同一可执行程序的重复测量波动范围内，其中 13 个正好选中实测最快方案。这个策略在 1,000 次调用内回本的条件为 **0/24**。正式确认的 **5,520** 项逐种子、逐进程校验全部通过，干净检出复现了 **36/36** 个机制配置的 HLO 指纹。

## 从最初的问题到最后的判断

1. **给成本模型找到一个实际决策。** 选题从用中间表示（IR）分析 GPU 算子融合出发。调试时取一个内部值可能改变生成的程序，这引出了更具体的问题。若一次推理必须返回同样的张量和激活统计，工程师可以怎样写这段观测代码，额外花费多少？数值一致性随之成为所有候选的准入条件。已有研究分别做过稀疏度采样、共享聚合和编译器自动调优，此处固定采样频率、两个观测点和主输出，在同一目标 CPU 上做成对比较。[最初任务书](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/docs/FusionScope_Research_Plan_v2.md) · [文献及与本研究的关系](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/docs/phase2-literature.md)

2. **先导发现，输出大小不足以定义一个公平的比较。** 复用 Lab 01 的 JAX 编译产物导出流程，先测只取主结果、取一个元素、取均值和取完整激活。4,096 元素的示例中，取样和均值都增加 4 字节，中位耗时分别是 28.75 和 32.45 微秒。两者提供的信息不同。于是固定一个可以严格核对的需求：对两处指定 ReLU 分别返回正元素数，并保留原主张量。对有限输入，ReLU 前大于零与 ReLU 后大于零逐元素等价。第一轮九配置小图在四种双点位置都得到 2117 和 2119，领先的两个位置落在计时波动范围内。[四种先导的原始表](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/cpu/pilot/four_probes.csv) · [九配置先导](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/cpu/pilot/first_report.md)

3. **先问“位置有用吗”。** 四类程序（仿射、矩阵乘、按行归一化、残差卷积）各取四档尺寸和链式/分支两种拓扑，得到 32 个条件。每个条件有无观测、四种单点和四种双点版本，一共 288 个配置。官方 Flax 的一个及两个相邻 ResNet 块另有 18 个配置。三个输入种子核对主张量的容差和两个整数的精确相等。五个发现进程、三个新确认进程和同一可执行程序的重复测量控制帮助区分小差异与波动。每个合成条件的四种位置均生成相同的优化后 HLO。128 个双点配置相对无观测的耗时比中位数却为 1.107，干净检出为 1.110。位置模型与固定后置的微小排序也随复跑翻转。[第一阶段结果](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/cpu/report/REPORT.md)

4. **用反例限定这个结论。** 两个 ResNet 子图各保留四类 HLO，但最快位置仍未稳定分出。把小仿射图和真实模块各用两种 ReLU 表达式编译，直接写 maximum 时四种位置合一，用 jax.nn.relu 时保留四种指纹。这个交叉实验定位到了源码写法，尚未定位第一个改变该关系的优化步骤。又用独立单点与双点计时检验成本能否相加。最大归一化图的链式和分支两种拓扑、各四种放置，都重复出现双点成本低于单点增量之和。显式拆成两个可执行程序还会改变调用和物化，因此下一步沿输出路径检查实际工作。[外测和交叉对照](https://github.com/ZZzJunzum/Compiler_report/tree/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/cpu/lowering_aligned) · [独立复跑](https://github.com/ZZzJunzum/Compiler_report/tree/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/reproducibility)

5. **从三个返回值倒查是谁在重复工作。** 把 HLO 融合体的形参绑定到调用处的实参，沿主张量及两个计数各走一遍，核对重复算术、归约树和跨边界的值。在 4096 × 256 仿射分支上，第一层加法走了三条路径。两个独立归约各写出一个 4 MiB 整数掩码，得到分配记录与优化后 LLVM 写指令的双重印证。大卷积图则已经把比较放进归约融合体，完整掩码导出为零。这给出了同一优化应当收益不同的两个案例。[HLO、缓冲区及 LLVM 逐行审计](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/phase2/analysis/PHASE0.md)

6. **实际修改生成程序的方式。** 对同一份统计实现独立归约、联合归约、显式分段和分块部分和。先筛查五个案例，再做 250 个合成配置及 8 个外测配置。相对独立计数，大仿射分支联合后两个完整掩码都消失，额外带标签算术副本从七处降到五处，独立确认提速 3.07 倍，干净复跑为 2.92 倍。16 行独立分块取得 1.86 倍，仍保留七处副本。大卷积分成每块两个样本时速度仅有整批独立计数的 0.43 倍。改变调度确实改进了某些程序，也带来了需要选择方案的新问题。[调度实现和完整结果](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/phase2/report/REPORT.md)

7. **解释第一版成本模型为什么永远选同一个。** 按整个图族留出测试，原解析模型对全部 32 个合成条件都选整批联合。逐项检查发现，186 个同图替代方案在调用、算术、搬运、归约层数和循环步数五项上都大于或等于联合策略，归约层数还严格更大。正系数使排序在拟合前就已决定。岭回归降低了时间预测误差，却在 12 次改选中仅 2 次变快、10 次变慢。新的模型需要能表达真实的排序翻转，实验则需要强固定策略作对照。[冻结的特征和预测](results/phase2/model) · [诊断计算](paper/check_diagnostics.py)

8. **分离“主计算变快”和“监控本身变便宜”。** 在四张旧图上把三种分块范围与主程序单独运行、独立计数、联合计数交叉，共 36 个配置。8 个非平凡交互中有 6 个超过相应波动门槛。归一化分支按 16 行分块后，带联合计数从 198.90 降到 105.40 微秒，主程序单独运行也从 186.30 降到 89.95 微秒。两种分块下按进程配对的计数增量分别为 11.75 和 15.55 微秒。这个案例的大部分收益来自主程序。[同分块有无观测对照](results/phase3/cpu/analysis/interactions.csv)

9. **先锁定推荐，再用新图验收。** 32 个旧条件、388 个配置用来开发和选择模型。把联合分块加入候选，分开估计工作集、累计访问、内核吞吐和循环成本，引入算子类别与分块的交互。先封存模型及推荐，然后分别用两个新进程试跑两候选策略并再次封存，最后才用五个独立进程确认。24 个主条件和 4 个混合迁移图共 368 个配置，其中 340 个返回两个计数。静态模型在主条件上的几何平均耗时比为 0.861，达到预登记标准；混合图为 1.047。归一化受益最大，稠密与卷积平均退化。这轮验收的 5,520 项配置、种子和进程组合校验全部通过，干净检出也复现了 36/36 个机制配置的 HLO 指纹。[注册方案](docs/plans/FusionScope_Phase3_Research_Plan.md) · [静态封存](https://github.com/ZZzJunzum/Compiler_report/tree/a8fd22575ffc293c850b922b95e1feb5e5e68272/results/phase3/cpu/static) · [试跑封存](https://github.com/ZZzJunzum/Compiler_report/tree/ed3876906c9431baa2f9465d6449c36d94bd9042/results/phase3/cpu/decision)

10. **再问“模型多提供了什么”。** 以下是确认完成后的复核。只用旧开发数据选择一条固定联合分块规则，每块取不超过 16 的最大合法行数或样本数，整批可装下便直接用整批。它在主条件上的耗时比为 0.876。完整模型直接与它配对的比值为 0.981，95% 区间 [0.957, 1.007]。用源码重建字节数并清除 HLO 工作量、扇出和共享前缀，保留原权重后 28/28 个选择相同；连训练也移除这些特征，主耗时比为 0.869。同样是 128 × 320 的仿射图，模型都选 64 行，链式图慢 9.8%，分支图却快 54.9%。结构信息有用的具体候选就在这对图里，当前模型的加法项会在候选相减时抵消。[逐图决策](results/phase3/cpu/analysis/decisions.csv) · [确认后的复算入口](paper/check_diagnostics.py)

11. **实际改变及边界。** 已经交付可执行的替代计数方案、可复核的融合分析和一个能提前选择合法方案的 CPU 原型。联合计数对某些现有工作负载带来实测收益；模型相对更强固定规则的额外收益仍难以分辨。若从五个进程的单图中位数推算每图各调用一次，总和的中位比为 1.005。若把决策成本也计入，复用校准后，静态选策和双候选试跑在 1,000 次内分别只在 2/24 和 0/24 个条件回本。它们回答了什么时候值得多做编译和试跑。研究结论覆盖这台 Intel CPU、这些形状及两个精确计数；XLA 上游优化、GPU 的物理流量、完整训练模型和跨设备策略迁移是不同的验证任务。[部署成本](results/phase3/cpu/analysis/break_even.csv) · [原始计时](results/phase3/cpu/analysis/process_medians.csv)

## 第三阶段证据

| 要核对的问题 | 入口 |
|---|---|
| 模型在每个图上选了什么，离最快方案多远？ | [逐图决策、精确计数有效性及噪声阈值](results/phase3/cpu/analysis/decisions.csv) · [逐候选排名](results/phase3/cpu/analysis/rankings.csv) |
| 分块怎样改变观测的额外代价？ | [按进程配对的交互量](results/phase3/cpu/analysis/interactions.csv) · [输出路径和 HLO 行号](results/phase3/cpu/analysis/paths.csv) · [机制图](results/phase3/cpu/figures/mechanism.pdf) |
| 一次部署要编译、试跑多少，多少次调用能回本？ | [冷启动与复用成本](results/phase3/cpu/analysis/setup.csv) · [逐图回本次数](results/phase3/cpu/analysis/break_even.csv) · [复用曲线](results/phase3/cpu/figures/reuse.pdf) |
| 如何核对测量和编译产物？ | [逐进程中位数](results/phase3/cpu/analysis/process_medians.csv) · [实验实现](fusionscope/phase3) · [逐阶段验证](results/phase3/cpu/verification.json) · [干净检出对照](results/phase3/cpu/analysis/replay.csv) · [SHA-256 清单](results/phase3/cpu/SHA256SUMS.json) |

每个进程 ZIP 内有环境、候选编译时间、所有预热和计时样本。正式队列还含逐种子验证。编译表中的 archive 和 component 定位相应的 StableHLO 与优化后 HLO。阶段一、二的完整程序、散装 IR、早期计划和报告保存在固定历史版本，当前分支保留了复核旧模型所需的冻结输入。[两阶段完整快照](https://github.com/ZZzJunzum/Compiler_report/tree/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325)

## 复现

在仓库根目录，用 Python 3.12 建立环境。下列命令在 Windows PowerShell 中运行。完整环境版本见 [Windows 依赖锁](requirements-windows-lock.txt)。

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-windows-lock.txt
.\.venv\Scripts\python -m paper.check_diagnostics
.\.venv\Scripts\python -m fusionscope.phase3.experiment verify --out results/phase3/cpu
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
pdflatex -halt-on-error '-output-directory=reproduction/paper' paper/FusionScope.tex
pdflatex -halt-on-error '-output-directory=reproduction/paper' paper/FusionScope.tex
Copy-Item -LiteralPath reproduction/paper/FusionScope.pdf -Destination output/pdf/FusionScope.pdf -Force
~~~
