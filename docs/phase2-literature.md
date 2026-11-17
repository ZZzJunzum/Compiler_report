# 精确激活观测、重复计算与分块归约的文献依据

这里要解释三种不同的代价：每次取得精确统计要做的工作，同一生产者被多个消费者读取的代价，以及改变计算或存储粒度产生的代价。以下定位均来自论文原文。各文献的设备、基线和输出契约与本项目分别列出。

## 每次观测是否都有代价

**Dong 等，Spartan，IEEE TPDS 32(10)，2448–2463，2021。** §2.1 用训练中激活图的稀疏度变化构造动机；§3 的算法 1 调整观测周期，算法 2 在 Active 与 Hibernate 间切换。§7 表 7 用无稀疏度观测的训练作为基线。五种模型逐迭代采样的额外时间为 14.9% 至 94.2%，自适应采样为 0.3% 至 5.7%。这是包含采样频率与数据移动在内的 GPU 训练结果。本项目的每次推理都必须返回两个精确计数，因此固定输入、频率和输出契约，将 V1 作为优化基准，另用 V0 单独量化观测开销。[发表信息与 DOI](https://eprints.gla.ac.uk/240696/) · [算法与表 7 原文](https://eprints.gla.ac.uk/240696/2/240696.pdf)

## 多消费者与物化点

**Boehm 等，On Optimizing Operator Fusion Plans for Large-Scale Machine Learning in SystemML，PVLDB 11，2018。** 图 1(c) 展示多个聚合共享输入；§5.2 图 9(c,d) 比较共享 X 的两个聚合。§4.2 式 (3) 对每个融合区域使用“写入时间 + 读取与计算时间的较大者”，并明确处理相交子图的重复计算。我们据此分别测联合归约 V2 与物化第一层激活的 V3。本文场景中的第二个计数依赖第一层结果，同时要返回完整 y，因而只统计少了一次 reduce 无法回答上游计算是否共享。[论文图 1、图 9 与式 (3)](https://www.vldb.org/pvldb/vol11/p1755-boehm.pdf) · [DOI](https://doi.org/10.14778/3229863.3229865)

**Phothilimthana 等，A Flexible Approach to Autotuning Multi-Pass Machine Learning Compilers，PACT 2021。** 图 2 和原文第 7 个 PDF 页面的小节 *Operator Fusion* 解释，不同消费者可以有不同迭代空间，融合后同一个 add 可能在多个消费者内分别计算。示例还展示布局选择怎样改变可行的融合。XTAT 的评估覆盖 Google TPU 上的 150 个训练及推理模型。我们在 CPU 上逐条核对操作数和融合调用，把“两个输出引用同一条已计算的值”与“两个调用各有一份等价算术”分别计数。只有后者增加静态副本数。[作者论文，图 2 与 Operator Fusion](https://mangpo.net/papers/xla-autotuning-pact2021.pdf) · [Google Research](https://research.google/pubs/a-flexible-approach-to-autotuning-multi-pass-machine-learning-compilers/)

## 分块粒度与归约组织

**Ragan-Kelley 等，Halide，PLDI 2013。** 图 2 从计算与存储粒度描述调度空间，图 3 用两阶段模糊滤波比较广度优先、完全融合、滑动窗口及分块的并行度、最大复用距离和工作放大。这为保留所有 tile 尺寸，以及同时记录缓冲区变化和变慢案例提供了理由。本文 V4 沿整行或整批样本切分，保留完整归一化行和卷积空间邻域；单样本 ResNet 跳过 batch 分块。[作者论文图 2–3](https://users.cs.duke.edu/~lkw34/papers/halide-pldi2013.pdf) · [MIT 出版记录](https://dspace.mit.edu/entities/publication/a1eb8c3a-91fe-40d7-978d-16b31cf11178)

**Dodović、Veselinović、Mišić，Analyzing the Impact of Kernel Fusion on GPU Tensor Operation Performance，Electronics 15(5)，1034，2026。** §4.4 在 RTX 3080 Ti、T4、A100 上分析 CUDA map-reduce。逐元素全局 atomic 版本在较大输入上的融合收益接近 1.00–1.06×；换为块内归约并让每块提交一个局部和，表 5 的融合加速范围约 1.17–2.00×。它支持把归约方式作为独立自变量。我们的 V4 用 JAX map 和局部 int32 计数，CPU 的向量存储、循环及临时缓冲应从本机 HLO、LLVM 和分配表核对。[论文 §4.4、表 5](https://www.mdpi.com/2079-9292/15/5/1034)

## 成本模型与后端依据

**Li 等，oneDNN Graph Compiler，CGO 2024。** 作者预印本的图 3 在同一个 CPU 微内核模板的不同融合位置，分别列出单核一次接触的切片大小、调用次数和总元素访问量。往内层移动可以压小临时区，也可能多次执行相同的前置计算。图 4 展示 GEMM 前后的布局转换与 ReLU 在不同位置生成。图 8 及表 1 在 32 核 Xeon Platinum 8358 上比较 MLP 和注意力子图并做融合消融。本项目的 V4 同样需要区分“一块有多少字节”和“所有块累计读写多少”。解析模型只用容量代理搬运量，在合成条件上全部选了联合归约，遗漏了归一化图中实测较快的分块。这篇工作说明了访问次数能给模型增加什么信息，本项目的缓存命中率仍未测量。[作者预印本图 3、4、8](https://arxiv.org/pdf/2301.01333) · [CGO 会议记录](https://2024.cgo.org/details/cgo-2024-main-conference/26/oneDNN-Graph-Compiler-A-Hybrid-Approach-for-High-Performance-Deep-Learning-Compilati)

**Williams、Waterman、Patterson，Roofline。** 该模型将浮点计算吞吐、内存带宽与算术强度联系起来，给出计算受限或带宽受限的性能界限。我们借用计算时间与搬运时间取较大者的形式，外加宿主调用、局部循环与归约项，并在三个训练图族上校准各系数。本文特征使用基线 IR 估计的逻辑工作量和字节数，系数是有效拟合量；估计真实 DRAM 流量与硬件峰值需要独立计数器或微基准。[Berkeley 原始技术报告](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html) · [后续论文记录](https://escholarship.org/uc/item/5tz795vq)

**OpenXLA 与 JAX。** OpenXLA 的 GPU 流程文档区分成本驱动的 PriorityFusion 和可为其他消费者额外导出结果的 Multi-Output Fusion，并讨论依赖环。V4 表示应用用 map 明确返回每块 y、q₁、q₂ 的执行组织。它能否呈现为一个多输出 HLO fusion，须逐案检查。V2 的多输入 reduce、V4 的 map 语义按 JAX 接口实现；性能计时等待全部返回值就绪。[GPU fusion passes](https://openxla.org/xla/hlo_to_thunks#fusion-passes) · [PriorityFusion 源码](https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/priority_fusion.h) · [JAX reduce](https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce.html) · [JAX map](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html) · [JAX 计时](https://docs.jax.dev/en/latest/benchmarking.html)

本项目的贡献集中于同一精确观测契约下的三路径诊断、执行方案比较、整图族留出预测和可复核的负结果。多聚合、显式物化、分块调度和基于带宽的代价模型各有上述先例。
