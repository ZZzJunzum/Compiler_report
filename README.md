# FusionScope

**两个激活计数何时值得并行？** 本研究固定程序的主输出和两个计数，追踪 XLA 如何按归约输出形状分派 CPU 工作，并把局部改写带到事先留出的程序上检验。[英文论文 PDF](output/pdf/FusionScope.pdf) · [LaTeX 原稿](paper/FusionScope.tex) · [中文实验报告与部署账目](results/partition_boundary/README.md)

在一台有 12 个逻辑 CPU 的 i5-13420H 上，同一条仿射链的逐行联合计数在允许 1 个逻辑 CPU 时比独立计数慢 74.47 微秒，在默认亲和性下快 23.73 微秒。对只返回输入和统计的简化程序，默认亲和性下，288×160 的计数保留分区多花 32.20 微秒，1536×1024 的计数保留分区节省 852.63 微秒。固定 46,080 个元素改变行数，观察到编译器从无外层分区到 4 份、再到 12 份。[配对耗时与重复性门槛](results/partition_boundary/discovery/analysis/pairs.csv) · [三个亲和配置的编译证据](results/partition_boundary/discovery/analysis/kernels.csv)

2023 年的 AutoNumerics-Zero 已经记录了单线程下 XLA 并行分派的代价，以及拆分融合避开分派的实例。本研究的新证据是输出形状、可用 CPU 和下游主输出共同限定了这个反例的适用范围。论文进一步检验整数打包计数和只在独立观测子程序内关闭分区两种方案。两套规则及强基线均在首次编译四种留出结构前公开冻结。[重叠工作的原文 §S3](https://arxiv.org/html/2312.08472v1#S3) · [打包方案冻结](https://github.com/ZZzJunzum/Compiler_report/commit/19238ac) · [局部分区方案冻结](https://github.com/ZZzJunzum/Compiler_report/commit/79112c3)

| 需要核对什么 | 证据入口 |
|---|---|
| 完整图、简化图的干预及第二编译版本 | [跨版本与机器码对照](results/axis_probe/README.md) |
| 每个真实采样日期的复验 | [原始会话与日期](results/axis_probe/analysis/sessions.md) · [论文日期段落](paper/partition_dates.tex) |
| 局部打包、原生联合和独立归约 | [逐组独立验证](results/partition_boundary/holdout/analysis/pairs.csv) · [冻结规则的评价](results/partition_boundary/holdout/analysis/verdict.json) |
| 保留主输出编译、单独改变计数分区 | [主输出与观测目标文件](results/scoped_assignment/holdout/analysis/scoping.csv) · [逐组胜负](results/scoped_assignment/holdout/analysis/policy_effects.csv) |
| 新图构建、复用主程序与调用次数 | [总成本与逐进程回本区间](results/partition_boundary/deployment) |
| 真实 MNIST 权重与输入、阈值边界 | [训练模型与三个重叠片段](results/axis_probe/mnist) · [定宽整数与次正规数实测](results/partition_boundary/contract.json) |
| 早期 IR 预测、分块和 GPU 范围 | [附录的历史材料](https://github.com/ZZzJunzum/Compiler_report/tree/a0f9edf) · [文献对照](docs/literature-review.md) |

下面保留早期实验的完整复现命令。当前论文的独立复算与重新计时命令见[本轮实验报告](results/partition_boundary/README.md#复现)。

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
