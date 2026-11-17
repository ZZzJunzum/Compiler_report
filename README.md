# FusionScope

在 JAX/XLA 中，一次推理要同时返回主张量和两个精确的正激活计数。这个项目研究观测位置与归约方式怎样改变 CPU 可执行程序，以及编译前选择策略的收益能否抵消选择本身的开销。

**[英文论文 PDF](output/pdf/FusionScope.pdf)** · [LaTeX 源码](paper/FusionScope.tex) · [CPU 主结果](results/phase3/cpu/analysis/endpoint.json) · [各策略结果](results/phase3/cpu/analysis/policy_summary.csv)

24 个新 CPU 条件上，封存模型相对固定联合归约的耗时比为 **0.861**，配对 95% 区间为 **0.797–0.943**，达到预注册的三个标准。它在 8 个条件上更快、12 个更慢、4 个作出相同选择。归一化的耗时比为 0.598，稠密矩阵乘法为 1.077。4 个混合算子迁移条件的耗时比为 1.047。把选择过程也计入，复用已有校准且调用 1,000 次时，静态模型在 **2/24** 个主条件上回本。

每图只试联合归约和一个模型提名方案时，**24/24** 个选择与最快合法方案之间的配对差距落在同一可执行程序的重复测量波动范围内，其中 13 个正好选中实测最快方案。这个策略在 1,000 次调用内回本的条件为 **0/24**。正式确认的 **5,520** 项逐种子、逐进程校验全部通过，干净检出复现了 **36/36** 个机制配置的 HLO 指纹。

## 三阶段研究

1. **移动观测位置。** 在 32 个合成图上，将两个计数分别放在 ReLU 前后，四种组合均编译为同一规范化 HLO。[第一阶段报告及原始数据](https://github.com/ZZzJunzum/Compiler_report/tree/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/cpu)
2. **改变计数方式。** 独立归约、联合归约、分段执行和分块产生了不同实现。旧解析模型在 186 次比较中都把联合归约的五项成本估得更低或相等，正系数使它无法选择实测更快的例外。旧岭回归改选 12 次，其中 2 次变快、10 次变慢。[第二阶段报告](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/phase2/report/REPORT.md) · [原始论文与研究依据](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/docs/phase2-literature.md)
3. **在新图上先预测、再验证。** 开发集为 32 个旧条件。新样本含 24 个主测试条件和 4 个混合算子迁移条件，共 368 个配置，其中 340 个能同时返回两个计数。另有 36 个机制配置和独立干净检出复跑。硬件为单逻辑核 Intel Core i5-13420H。GPU 留待服务器可用时验证。[预注册方案](docs/plans/FusionScope_Phase3_Research_Plan.md) · [测试前静态预测](https://github.com/ZZzJunzum/Compiler_report/tree/a8fd22575ffc293c850b922b95e1feb5e5e68272/results/phase3/cpu/static) · [确认前试跑决策](https://github.com/ZZzJunzum/Compiler_report/tree/ed3876906c9431baa2f9465d6449c36d94bd9042/results/phase3/cpu/decision)

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
.\.venv\Scripts\python paper/check_diagnostics.py
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

从仓库根目录连续两次运行下列命令可生成英文论文。需要 pdfLaTeX，第三阶段的三张矢量图随仓库提供。

~~~powershell
pdflatex -halt-on-error '-output-directory=output/pdf' paper/FusionScope.tex
~~~
