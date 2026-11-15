# FusionScope · 用编译器 IR 预测 XLA 中间激活观测的开销

同一个正激活计数可以在 ReLU 前后计算。保持模型输出、输入权重和两个计数完全相同，只移动计数分支，编译器会生成什么？只看原程序的 IR，能提前选出更便宜的位置吗？

**[阅读中文研究报告](results/cpu/report/REPORT.md)** · [先看第一张九配置实表](results/cpu/pilot/first_report.md) · [独立复跑对照](results/reproducibility/README.md)

## 实测结果

在 Intel Core i5-13420H 上，逐元素、全连接、归一化与卷积四个图族共 32 个条件、288 个编译配置。对每个条件，四种双点位置都生成了相同的优化后 HLO 计算体。三个新进程的独立复测也未能为这 32 个条件的前两名稳定排序。增加计数本身有成本，128 个双点配置的延迟相对基线中位数为 **1.107×**。[配对时延](results/cpu/report/tables/latencies.csv) · [同一可执行文件的 A/A 波动](results/cpu/report/tables/noise.csv)

官方 Flax ResNet 的单块和相邻双块各保留了 **4 种** HLO，两个子图的时延排名仍在本次判据下无法分出。追加的编译对照把图和输入固定，在逐元素图和单个 ResNet 块上均观察到相同规律：直接使用 `jnp.maximum` 得到 1 类，使用 `jax.nn.relu` 得到 4 类。三组输入上的两个计数完全一致。[逐边证据](results/cpu/report/tables/structure.csv) · [ReLU 写法对照](results/cpu/lowering_aligned/comparison.csv)

IR 模型 M2 在未见图族上的选择损失中位数为 **0.956%**，固定在两处 ReLU 后观察为 **0.749%**。这批数据没有显示 IR 选址的稳定优势。用四个已测单点估算双点开销，绝对误差为 60.59 µs；32 个条件中的 16 个由于基线 HLO 定位歧义触发了 M2 的源码回退。[各策略与图族的完整比较](results/cpu/report/tables/policy_summary.csv)

从 GitHub 干净克隆后重跑整套实验，306 个正式及外测配置的 HLO 指纹全部复现，双点延迟倍率中位数为 **1.110×**。两次运行中，最大尺寸按行归一化图的 8 个双点配置均在发现与确认批次中呈现负交互，合算两个计数的增量低于分别测得的单点增量之和。两次各有 **75,640 条逐次计时**，原始读数可下载复核。[两次运行的逐项对照与完整归档](results/reproducibility/README.md)

![实测四种位置与噪声](results/cpu/report/figures/02-placement-and-noise.png)

## 在本机运行

下面的命令在仓库根目录的 PowerShell 中执行。正式结果使用 Python 3.12.5、JAX/jaxlib 0.9.0.1 和 Flax 0.12.3。Windows 可使用完整依赖锁；其他支持 JAX 的平台可从 `requirements.txt` 安装。[测量环境](results/cpu/matrix/env_p0.json)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-lock.txt
.\.venv\Scripts\python.exe run.py --stage verify
.\.venv\Scripts\python.exe run.py --stage recommend --case R2
```

最后一条命令读取未观测 R2 基线的特征和锁定的模型，列出四种候选、预测倍率、选择及回退原因。查询 `minimum_before_relu` 会被明确拒绝。自定义带有 `u_relu`、`v_relu` 标记的基线函数也可直接传给 `recommend_program`。[选择工具接口](docs/METHODS.md#对自己的基线调用选择工具)

从独立目录重做先导，或从头运行整套实验。两条命令共用恢复点。使用新的 `--out` 路径可保留多次运行。

```powershell
.\.venv\Scripts\python.exe run.py --stage first --out reproduction/cpu
.\.venv\Scripts\python.exe run.py --stage all --backend cpu --out reproduction/cpu
```

分阶段入口依次为 `pilot`、`matrix`、`model`、`confirm`、`external`、`controls`、`lowering`、`report`、`verify`。计时进程按顺序运行；同一次运行使用同一个输出目录。已有结果可直接重建报告和全表。CPU 是已验证后端。GPU 入口供具备兼容 JAX CUDA 环境的服务器使用，其计时、模型和证据应写入独立设备目录。

## 从研究问题查数据

| 想核对的内容 | 入口 |
|---|---|
| 原始计划、Lab 01、具体实验决定 | [任务书原文](docs/FusionScope_Research_Plan_v2.md) · [Lab 01](lab01/) · [预登记](protocol.json) · [实验方法](docs/METHODS.md) |
| 每个配置的图、IR 和逐轮时间 | [配置索引](results/cpu/report/tables/configs.csv) · [发现批原始记录](results/cpu/matrix/raw/) · [独立复测原始记录](results/cpu/confirm/raw/) |
| 两次观测有没有共享开销 | [所有交互及噪声参考](results/cpu/report/tables/interactions.csv) · [逐进程计算](results/cpu/report/tables/process_interactions.csv) |
| 预测是否在测试前锁定、错在何处 | [已封存推荐](results/cpu/model/prediction_lock.json) · [逐案决策](results/cpu/report/tables/decisions.csv) · [成本与摊销](results/cpu/report/tables/costs.csv) |
| 外测、显式执行边界和语义验证 | [外部子图](results/cpu/external/) · [两次可执行程序对照](results/cpu/report/tables/boundaries.csv) · [完整性检查](results/cpu/report/audit.json) |
