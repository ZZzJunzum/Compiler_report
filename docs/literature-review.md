# FusionScope 的文献对照与贡献边界

这份索引与英文论文第 2 节配套。每项文献后面给出它已经回答的问题，以及项目中需要用本机实验支持的判断。整篇论文的顺序和各节数据入口见[项目交接说明](../README.md#论文怎样从一个问题走到下一个)。

文献与实现核对至 2026 年 11 月 19 日。问题是一次推理必须返回原张量和指定的内部激活统计时，XLA 为这些输出生成哪些额外工作，哪种等价实现能节省时间，以及预测时可用的 IR 提供多少信息。读取编译器输出的 IR 用于分析。改变送进编译器的计算图的是新增的统计和返回值。

材料列出 25 项研究和两份工程文档。下文按论断逐项核对论文、作者稿或正式会议记录，并补查与精确统计、实际 XLA 融合决策及硬件归属有关的原始文献和实现。整次调用耗时与两种实现的耗时比来自先前两批盲测。随后针对“观测多花多少微秒”单独封存了普通 CPU 程序的优化后 HLO 和直接增量预测。[早期预测封存](https://github.com/ZZzJunzum/Compiler_report/commit/508ef3d) · [早期盲测结果](https://github.com/ZZzJunzum/Compiler_report/commit/859389a) · [直接增量预测封存](https://github.com/ZZzJunzum/Compiler_report/commit/2ee7503)

## 哪些工作与我们直接重合

| 研究 | 原文已经做到什么 | 对本项目的影响 |
|---|---|---|
| Amanda，ASPLOS 2024 | 第 7 节讨论融合可能消除原来的插桩点，提出记录优化前后对应关系、限制部分优化等方向。 | 给出了可观测性问题的先例。这里的跨优化映射是一项设想。我们的逐 pass 检查回答一个可实测的问题，即写在 ReLU 后的比较最终读谁。[论文 §7](https://horizon-lab.org/pubs/asplos24-amanda.pdf) |
| SystemML，PVLDB 2018 | 图 2 和 §4.2 建模多个消费者间的共享、重算与物化。图 9c–d 在稠密和稀疏输入上实测共享矩阵 X 的两个乘积和。 | 直接覆盖共同输入上的多重聚合。我们的同层能量与最大值是类似对照。新证据来自跨层依赖的生成代码、消失的缓冲及实测方向。[论文图 2、9](https://www.vldb.org/pvldb/vol11/p1755-boehm.pdf) |
| Weld，PVLDB 2018 | §2.5 允许多个结果共同求值。§4.1 的 Listing 6 在同一遍历中生成加一后的向量并对原输入求和，最后返回两个结果。 | “保留普通数组并同时产生标量统计”已有直接先例。本文比较同一请求在 XLA 中的具体实现和代价。[论文 Listing 6](https://www.vldb.org/pvldb/vol11/p1002-palkar.pdf) |
| Restructuring Batch Normalization，MLSys 2019 | §3.2、图 5 在生成卷积激活时累计和与平方和，供后续归一化。§5 报告了 CPU 和 GPU 实验，其中 GPU 实验使用 CUTLASS。 | 生产激活时同时累计统计量已有实现与实测。BN 使用模型自身所需的统计，本文增加调用者要求的返回值。材料将该论文的实验平台写成仅 CPU，需要纠正。[论文图 5 及 §5](https://proceedings.mlsys.org/paper_files/paper/2019/file/9db64c20dee0011899dfdf200e61ef35-Paper.pdf) |
| oneDNN Graph Compiler，CGO 2024 | 图 3–4 的融合锚点对应循环嵌套层级，分析每核工作集、调用次数与累计元素访问。 | ReLU 前后的源码位置、生成代码中的循环层级是两个决策。静态临时分配量也不能换算成实测 DRAM 字节数。[作者稿图 3–4](https://arxiv.org/abs/2301.01333) |
| AStitch，ASPLOS 2022 | §2.3.1 和 §3.2 分析 SIMT 上元素及算子间的一对多依赖，在不同存储层级复用中间结果。 | 多消费者导致融合后重算已有清楚分析。本项目能指出某次 CPU 编译中三条实际路径及改写后保留的路径。[论文图 4–5](https://jamesthez.github.io/files/astitch-asplos22.pdf) |
| FusionStitching，2018 预印本 | Long 等人的 *Deep Fusion and Code Generation for Tensorflow Computations on GPUs* 研究 GPU 融合、代码生成和跨算子复用。 | 材料只给出系列名。同名系列另有不同副标题的 2019 和 2020 年预印本。本文按版本引用。[2018 年稿](https://arxiv.org/abs/1811.05213) · [2019 年稿](https://arxiv.org/abs/1911.11576) · [2020 年稿](https://arxiv.org/abs/2009.10924) |
| KPerfIR，OSDI 2025 | §6.4 式 1 和表 5 比较原 GEMM 周期数加记录事件的估计成本与实际插桩后周期数，以检查优化干扰。 | 支持把观测操作和原运算变化分开。本文在每个分块尺寸重测无观测主程序，用差值估计增量，测量单位是 CPU 整次调用。[论文 §6.4](https://www.usenix.org/system/files/osdi25-guan.pdf) |
| DeepContext，ASPLOS 2025 | §4.1、图 4 在 JAX/XLA 记录融合算子与源算子的对应关系，供性能归属。 | 这是已实现的映射。它把耗时关联回源操作。本文必须实际返回激活统计，并追踪支持该输出的程序路径。[论文 §4.1](https://arxiv.org/pdf/2411.02797) |

## 哪些成本模型是合适的参照

比较时须记下输入在哪个编译阶段可用，被预测的是内核周期数、两个方案的相对性能，还是整个调用的墙钟时间。下面的研究不宜与本文 CPU 实验直接比较误差数字。

| 研究 | 预测对象和信息来源 | 本文应怎样使用 |
|---|---|---|
| AutoTVM，NeurIPS 2018 | *Learning to Optimize Tensor Programs* 从程序特征学习调度成本，比较执行时间回归和排序目标。 | 单列绝对误差、是否改选及改选后的实际收益。[原文](https://proceedings.neurips.cc/paper/2018/file/8b5700012be65c9da25f49408d959ca0-Paper.pdf) |
| Halide 自动调度器，TOG / SIGGRAPH 2019 | 对候选调度进行树搜索，并用随机程序训练成本模型。 | 选择存储和计算粒度已有成熟先例。本文检验有限观测方案中，额外取得 IR 是否值得。[作者论文页](https://halide-lang.org/papers/autoscheduler2019.html) |
| Kaufman 等，MLSys 2021 | *A Learned Performance Model for Tensor Processing Units* 的融合模型使用已形成的 XLA TPU 内核。§2 讨论执行重叠与缓存对累加耗时的限制。 | 最接近最初的 IR 预测目标。我们的绝对时间模型只看后端优化前的候选 StableHLO，标签包含主机发起至全部输出就绪。[原文](https://proceedings.mlsys.org/paper_files/paper/2021/file/6bcfac823d40046dca25ef6d6d59cc3f-Paper.pdf) |
| Tiramisu，MLSys 2021 | *A Deep Learning Based Cost Model for Automatic Code Optimization* 从程序和变换序列预测相对加速。 | 同图选型需要能随执行方案变化的特征。我们早期模型中的加性图级特征在同图排序时抵消。[会议论文](https://proceedings.mlsys.org/paper_files/paper/2021/hash/d9387b6d643efb25132be36f7b908d96-Abstract.html) |
| nn-Meter，MobiSys 2021 | 探测边缘设备的后端融合行为，对执行单元估时并组合出模型延迟。 | CPU 的同步整次调用和 GPU 的逐核时间应分别计量。[原文](https://air.tsinghua.edu.cn/pdf/nn-Meter-Towards-Accurate-Latency-Prediction-of-Deep-Learning-Model-Inference-on-Diverse-Edge-Devices.pdf) |
| TpuGraphs，NeurIPS Datasets and Benchmarks 2023 | 数据包含张量计算图、候选布局或分块配置与实测性能，支持同图候选排序。 | 分开报告新尺寸、新结构和同图排序。本文两组盲测按图隔离，增加同图候选不会增加独立架构数。[原文](https://proceedings.neurips.cc/paper_files/paper/2023/file/ded1a89e2b3b925444ada973af66336e-Paper-Datasets_and_Benchmarks.pdf) |
| Daydream，USENIX ATC 2020 | 从 GPU 运行轨迹和依赖图估计施加优化后的训练时间。 | 动态轨迹需要运行程序。若引入这类特征，应计入取得它们的费用。[会议论文](https://www.usenix.org/conference/atc20/presentation/zhu-hongyu) |
| XTAT，PACT 2021 | 在 XLA 搜索布局、融合、分块及代码生成等多阶段配置。 | 源码等价、优化后 HLO 相同、最终延迟接近分别需要证据。[原文](https://mangpo.net/papers/xla-autotuning-pact2021.pdf) |

## 其他相关研究及工程依据

| 研究 | 具体用途 |
|---|---|
| Halide，PLDI 2013 | 算法与调度的分离给出了不同生产、局部性及重算选择的共同表述，适合解释为何对照组必须固定输出。[原文](https://people.csail.mit.edu/jrk/halide-pldi13.pdf) |
| DNNFusion，PLDI 2021 | 算子间映射关系和表达式重写能改变融合机会，可参照其方法解释为何两种 ReLU 写法需要查看编译产物。[作者稿](https://arxiv.org/abs/2108.13342) |
| TENSAT，MLSys 2021 | §6.4 给出 SqueezeNet 反例，扩大允许的多模式改写后，估价选出的整图实测反而变慢。对照的是不同改写集合。[原文 §6.4](https://proceedings.mlsys.org/paper_files/paper/2021/file/cc427d934a7f6c0663e5923f49eba531-Paper.pdf) |
| Apollo，MLSys 2022 | 用低层循环优化反馈修正高层图划分，可作后续引入低层证据的参照。该反馈的可用时间晚于本文静态预测。[会议论文](https://proceedings.mlsys.org/paper_files/paper/2022/hash/e175e8a86d28d935be4f43719651f86d-Abstract.html) |
| Welder，OSDI 2023 | 用 tile-graph 和数据流量估计协调算子内、算子间复用。本文的缓冲分配记录只证明某个对象计划占用多少空间。[会议论文](https://www.usenix.org/conference/osdi23/presentation/shi) |
| Mirage，OSDI 2025 | 多层张量程序搜索同时考虑代数变换、调度与实现。本项目两种归约的胜者仅是该候选集中的胜者。[会议论文](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi) |
| Spartan，IEEE TPDS 2021 | 实测训练期间的稀疏度监控并自适应改变采样周期，支持激活统计有使用场景。每次推理都返回精确统计仍是本文定义的受控请求。[作者公开稿及出版记录](https://eprints.gla.ac.uk/240696/) |
| Amazon SageMaker Debugger，MLSys 2021 | §3.3 等处描述张量采集、保存间隔与异步处理。采集频率和日志系统开销与本项目的函数返回值对应不同使用方式。[原文](https://proceedings.mlsys.org/paper_files/paper/2021/file/5c7cb28f9db7a96951df3f06fd0ccfaf-Paper.pdf) |

材料中的两份官方文档是工程行为的直接依据。JAX 用线性层接 ReLU 举例说明打印 logits 可能迫使中间数组物化，并引入与主机通信等成本。TensorRT 区分标记特定 debug tensor 和收集优化后仍存在的张量，强制保留原本会被融合消去的张量可限制优化。两者均未测量本文用函数返回两个整数的开销。[JAX 调试文档](https://docs.jax.dev/en/latest/debugging/print_breakpoint.html#unnecessary-materialization) · [TensorRT 10.x 调试文档](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html)

补充检索厘清了选题的三个出发点。TensorFlow Debugger V2 的 FULL_HEALTH 已可记录张量的有限正值个数，进一步收紧了对统计本身的创新表述。Snider 与 Liang 已在 GPU 上研究过 XLA 算子融合。Cano 等的 TPU 正确性预印本图 4 把调试拆核与半精度舍入的恢复联系起来。它提供选题动机，CPU 计时无法验证该 TPU 故障。[TensorFlow Debugger V2](https://www.tensorflow.org/tensorboard/debugger_v2) · [XLA 算子融合预印本](https://arxiv.org/abs/2301.13062) · [TPU 正确性预印本](https://arxiv.org/abs/2611.00624)

对于最初的 GPU 选题，还有一个比学习模型更直接的工程先例。OpenXLA 自带的 GPU 性能模型接受设备信息、HLO 生产者和消费者，提供多输出融合耗时估计，普通融合接口还显式接收“生产者是否另写一个输出”。HloEnv 在 XLA 上暴露可选的代数化简与融合决策，并在 A100 上评价搜索结果。若进入 GPU 阶段，后端估计应作为逐核问题的对照。这两个先例都没有给出本项目 CPU 语义契约的整次调用时间。[OpenXLA 固定版本接口](https://github.com/openxla/xla/blob/ddaa568d48ad24c16e08bff00d3038bcf824eab9/xla/backends/gpu/cost_model/gpu_performance_model.h#L36-L60) · [HloEnv，NeurIPS 2022 ML for Systems Workshop](https://mlforsystems.org/assets/papers/neurips2022/paper14.pdf)

## 哪些贡献经得起上述比较

| 可核对的结论 | 证据及其边界 |
|---|---|
| 一个等价的位置变化在哪一步被改写 | 在小仿射链和 Flax ResNet 块中，将两种 ReLU 表达与四种放置完全交叉。在 CPU 的首轮代数化简前，所有后置比较都读 maximum。之后，显式 maximum 的 8/8 处改读 add，库函数的 0/8 处发生该变化。16 个最终 HLO 指纹匹配历史记录，48 项逐输入检查通过。这定位了两个程序中比较操作数变化的步骤。该步骤单独贡献的耗时仍未知。[逐配置结果](../results/placement_pass/comparisons.csv) · [相邻 HLO 与验证记录](../results/placement_pass/checkpoints.zip) · [OpenXLA 的记录方法](https://openxla.org/xla/hlo_dumps) |
| 一项已知改写真正消除了哪些开销 | 4096 × 256 仿射分支的两张 4 MiB 整数掩码可以在 HLO 返回路径、缓冲分配和 LLVM 存储中交叉定位。联合归约后两张掩码均消失，实测提速 3.07 倍，干净复跑为 2.92 倍。另一个归一化案例在同分块无观测对照下显示，分块虽让整次调用变快，计数增量却从 11.75 升至 15.55 微秒。[历史机制审计](https://github.com/ZZzJunzum/Compiler_report/blob/e5964158e339cb1ff866b5f1ca4d0ba3bc9b6325/results/phase2/analysis/PHASE0.md) · [同分块配对结果](../results/phase3/cpu/analysis/interactions.csv) |
| 消除导出可作机制证据，单独以此选实现会失手 | 22 张新图、9 种请求的 198 对中，120 对可以直接做联合归约。82 对消除了参考实现里的观测专用导出，其中 30 对明确变快，4 对明确变慢，48 对未分出差异。两版都没有该导出的其余 38 对全未分出差异。同一 320 × 160 仿射链上，三个全局计数的耗时比为 0.53，两个逐行计数为 1.29。这里同时改变了观测点数和归约轴。[逐请求时间、导出及门槛](../results/cpu_cost/analysis/rewrite_pairs.csv) |
| IR 的解释价值、绝对时间预测和同图选型有不同证据 | 第一版模型五项正成本在 186 个替代方案上均被固定策略支配。后一版删除特定基线 HLO 特征后 28/28 个选择相同。候选 StableHLO 的绝对延迟回归在新尺寸和新架构上的中位误差倍数分别为 1.46、2.16。在另一批图上，直接拟合成对耗时比的集成模型 41 次中改选 17 次，相对固定联合归约有 1 胜、1 负、15 次未分辨，按图等权耗时比为 0.983，95% 区间为 [0.947, 1.016]。目前数据没有确定为选型增加 IR 的额外收益。[模型审计](../paper/check_diagnostics.py) · [绝对误差](../results/cpu_cost/analysis/accuracy.csv) · [独立决策结果](../results/cpu_cost/paired/policies.csv) |
| 只用普通程序的 IR 能预告多少观测开销 | 在 33 张新图编译任何带观测的候选之前固定了 477 个直接增量预测。熟悉架构的新尺寸上，源码模型的平均绝对误差是 69.64 微秒，加入普通程序优化后的 HLO 为 69.57 微秒。训练未见的架构上分别为 106.38 与 107.42 微秒。两种方法在两组都优于猜零，额外 IR 未达到预登记改善标准。这项对照把先前文献中的“IR 可用于成本估计”收紧为本研究的可检验主张。[封存版本](https://github.com/ZZzJunzum/Compiler_report/commit/2ee7503fcd38455cc89c393d17a3fc94f3a5cb6b) · [按图等权误差](../results/cpu_cost/increment/accuracy.csv) |

## 谁会用到成本预测

| 使用者及决策 | 预测什么 | 怎样判断预测有用 |
|---|---|---|
| 模型使用者为每次推理安排观测预算 | 普通程序已知的调用耗时加上指定观测的预测增量。若统计内容已经确定，独立归约和联合归约仍可能提供相同输出。 | 预算需要微秒误差。选实现则比较模型实际选中的调用耗时与固定策略，还要计入取得额外 IR 的一次性成本。我们在新 CPU 图上分别检验精度和选择。[盲测前封存](https://github.com/ZZzJunzum/Compiler_report/commit/2ee7503) |
| XLA GPU 编译器决定哪一对生产者和消费者先做融合 | 正常的优先融合路径先检查候选能否生成合法实现，再比较融合前后估计执行时间，更新候选优先级。同一生产者供多个消费者时，重复计算会进入估计。 | 对同一语义的候选记录预测节省、实际发射的内核及设备时间，再检查被选实现的整次调用是否也变快。核实所用版本的优化路径和实际发射记录。[优先级定义](https://github.com/openxla/xla/blob/ddaa568d48ad24c16e08bff00d3038bcf824eab9/xla/backends/gpu/transforms/priority_fusion.h#L42-L65) · [实际打分位置](https://github.com/openxla/xla/blob/ddaa568d48ad24c16e08bff00d3038bcf824eab9/xla/backends/gpu/transforms/priority_fusion.cc#L816-L869) |
| 自动调优器决定有限实测预算花在哪些候选上 | 按估计运行时间挑选少量实现交给设备测量。Chen 等在含服务器 GPU 的试验中用学习模型指导张量程序搜索。Kaufman 等在 TPU 上检验融合及分块选择。 | 固定试跑次数或搜索时间，比较最终找到的最快程序和搜索代价。XLA 的 GPU 调优器也会针对特定矩阵乘或卷积实测多个实现。[Chen 等，NeurIPS 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/8b5700012be65c9da25f49408d959ca0-Abstract.html) · [Kaufman 等，MLSys 2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/6bcfac823d40046dca25ef6d6d59cc3f-Paper.pdf) · [OpenXLA 的设备调优说明](https://openxla.org/xla/determinism) |

## 据此怎样推进

1. **本稿按 XLA CPU 的编译行为实证研究定稿。** 相关工作逐项承认共享聚合、向量与统计共同输出、生产者侧采集、插桩干扰及 IR 排序的先例。正文区分未插桩基线的优化后 HLO、候选在后端优化前的 StableHLO、已编译候选的优化后 HLO。最后一种需要计入编译和检查成本。按 1,000 次调用摊销，导出检查策略的总成本是直接采用联合实现的 1.345 倍。[逐对成本](../results/cpu_cost/analysis/rewrite_pairs.csv)
2. **下一项 CPU 验证应测量固定观测点时，归约轴如何改变两种实现的耗时差。** 复核已有数据，按图配对两个全局计数和同两点的两个末轴计数，20 张图支持这两种请求。按原来的判据，3 张在两种请求下都加速，10 张都未分辨，7 张只在一种请求下有明确方向，没有出现两种请求各自有明确且相反方向的图。320 × 160 链上，两点全局计数节省 31.70 微秒，低于 49.29 微秒的门槛。后续可固定这两个点、形状、输入及主输出，预先登记同一进程内的交互量 (联合逐行 − 独立逐行) − (联合全局 − 独立全局)，交错重复四个实现和同一可执行程序的重复对照。在分辨出交互之后，再对比 LLVM 归约循环、向量化及可用的硬件计数器。判断单独两项是否变快与直接检验这个交互量是两种检验。[可复核的 count 和 row 逐图记录](../results/cpu_cost/analysis/rewrite_pairs.csv)
3. **GPU 单独验收。** 在一台已记录型号和软件栈的 NVIDIA 服务器上先运行同一对正反案例、无观测主程序和固定策略。记录实际 kernel 边界、设备时间与整次调用时间，将性能分析与正式计时分开。为检验最初的 GPU 融合成本预测目标，先封存候选、模型和强固定基线，在可对齐的逐核问题上加入 XLA 自带估计，并另报整次调用误差和选型收益。若要追查物理内存停顿，TenProf（ICS 2026）用硬件采样向 PyTorch 张量归属访存停顿，可作方法参照。现有 CPU 数字可作复现输入和对照设计，GPU 结论待取得硬件后验证。[TenProf 论文](https://doi.org/10.1145/3797905.3800515) · [作者实现](https://github.com/David-Dingle/TenProf)
