# FusionScope Phase II：观测诱发重复计算与多输出分块融合研究计划

**项目仓库：** `ZZzJunzum/Compiler_report`\
**基线提交：** `f2df6a2`\
**建议分支：** `phase2-observer-recompute`\
**主环境：** JAX / XLA CPU；若能使用 GPU 服务器，再增加独立 GPU 验证。

------------------------------------------------------------------------

## 0. Codex 总指令

以提交 `f2df6a2` 为只读基线。创建新分支
`phase2-observer-recompute`，所有新代码、IR、计时和报告放入
`results/phase2/`。不得覆盖第一阶段的
`results/cpu/`、`protocol.json`、原始日志和审计文件。

本阶段不要一开始就写优化代码。先完成 Phase 0 的 HLO
审计，确认重复计算和中间张量的证据，再实现候选方案。

本项目不声称发明通用 Multi-Output Fusion，也不声称替代 XLA 的 Fusion
Cost
Model。自主工作定位为：针对精确激活观测场景，分析观测消费者引发的重复计算与归约中间结果，并实现、评估
observer-aware 的多输出分块执行策略和成本模型。

------------------------------------------------------------------------

# 一、研究方向与研究问题

## 1.1 建议题目

**Reducing Observer-Induced Recomputation in XLA: An IR-Guided Study of
Multi-Output Fusion and Tiled Partial Reduction**

中文：

**降低 XLA 激活观测开销：基于 IR 的重复计算分析与多输出分块归约研究**

## 1.2 从第一阶段到第二阶段

第一阶段已经完成：

-   两层神经网络的精确激活计数；
-   pre/post 观测位置对照；
-   Jaxpr、StableHLO、优化后 HLO 导出；
-   E/M/N/C 四类图、四档规模、Chain/Branch 两种拓扑；
-   计时、A/A 噪声控制、独立复跑；
-   基于原始 IR 的位置选择模型。

第一阶段的重要结论：

1.  双点观测确实有成本，且不同图族差异很大；
2.  32 个合成条件中，同一条件的四种观测位置被优化成相同 HLO
    计算体，前两名没有稳定排序；
3.  ResNet 子图和不同 ReLU 写法可以保留不同 HLO；
4.  某些归一化大图的双点观测成本存在可重复的非加性；
5.  仅靠原始 IR 选择 pre/post，没有稳定优于固定策略。

第二阶段不再把"哪个观测位置更好"作为主问题，而是研究：

> 加入 q1、q2 后，XLA 是否把同一个上游 Producer 分别复制到主输出、q1、q2
> 的不同 Fusion computation
> 中？能否通过共享计算、局部归约和多输出执行减少这种重复工作？

## 1.3 Research Questions

**RQ1：观测代码引入了哪些额外计算？**

-   y、q1、q2 三条输出路径分别包含哪些 HLO 运算？
-   同一个源算子是否在不同 Fusion computation 中重复出现？
-   compare/convert 是否生成完整尺寸的中间张量？
-   q1、q2 是否各自形成独立的归约树？

**RQ2：能否通过不同执行组织减少重复计算和中间结果？**

比较： - 当前 XLA 默认观测实现； - 变分/联合归约； -
显式物化共享激活； - tile-level multi-output execution； - 可选
scan/carry 局部累计。

**RQ3：能否构建 recomputation-aware Cost Model？**

模型需要根据基线 IR、图结构和张量尺寸，估计： - 重复计算成本； - 完整
Mask 和中间激活的物化成本； - 归约成本； - tile 数量和调度成本；
并判断哪种执行方案更合适。

------------------------------------------------------------------------

# 二、论文阅读结论与创新边界

## 2.1 Spartan --- 观测开销是真实问题，但它主要减少观测频率

Dong et al., *Spartan: A Sparsity-Adaptive Framework to Accelerate Deep
Neural Network Training on GPUs*, IEEE TPDS 2021.

论文：https://eprints.gla.ac.uk/240696/2/240696.pdf

重点阅读：§2.1、§3、Algorithm 1/2、Table 6/7、Conclusions。

Spartan 研究训练期间的激活稀疏度监控。作者发现，统计稀疏度和 CPU-GPU
数据移动会产生明显开销。不同层的稀疏度不同，但相邻迭代之间可能相对稳定。因此他们设计
Sparsity Monitor，利用历史记录、Active/Hibernate
状态和动态周期调整，避免每次迭代都完整测量。

论文报告 exhaustive profiling 在不同模型上的 overhead 为
14.9%--94.2%；使用 monitor 后为 0.3%--5.7%，平均 profiling overhead 约
1.7%。

**对本项目的启发：** 观测本身会成为性能成本。

**边界：** Spartan
通过降低观测频率节省成本；本项目要求每次执行都返回精确
q1/q2，不能用跳过观测来代替优化。我们要降低每一次观测的执行成本。

不要把 Spartan 的训练 profiling overhead 与本项目 CPU 推理计时直接比较。

## 2.2 Boehm --- Multi-Aggregates、物化点和重复计算

Boehm et al., *On Optimizing Operator Fusion Plans for Large-Scale
Machine Learning in SystemML*, PVLDB 2018.

论文：https://www.vldb.org/pvldb/vol11/p1755-boehm.pdf

重点阅读：Figure 1、§2、§3、§4.2、§4.3、§5.2、Conclusion。

论文把 ML 程序表示成
DAG，并研究融合计划如何减少中间结果物化、重复输入扫描和不必要计算。Figure
1(c) 展示多个聚合共享输入，例如
`sum(X^2)`、`sum(X*Y)`、`sum(Y^2)`。分开执行可能重复读取 X/Y；合适的
multi-aggregate 计划可以共享数据访问。

论文还定义 materialization
points：一个结果有多个消费者时，融合可能避免写回/读取，但也可能让
Producer 在多个融合区域中重复计算。其成本模型估计 read、write、compute
时间，并考虑冗余计算。

**创新边界：** 多聚合和共享输入不是新概念，不能把"合并两个
sum"作为原创算法。

**本项目的具体差异：** q1、q2
来自神经网络不同层，有先后依赖；同时必须保留主输出 y。我们研究 XLA
的输出路径是否复制上游 Producer，以及怎样在 exact-observation
约束下共享层计算和局部统计。

论文明确指出，联合优化 graph rewrites 与 fusion、以及扩展 GPU
代码生成，是未来方向之一。

## 2.3 2026 Kernel Fusion Study --- Reduction 策略决定融合收益

Dodović et al., *Analyzing the Impact of Kernel Fusion on GPU Tensor
Operation Performance: A Systematic Performance Study*, Electronics
2026.

论文：https://www.mdpi.com/2079-9292/15/5/1034

重点阅读：§3.3.1--3.3.4、§4.4、§5、§6 Limitations and Future Work。

作者比较 Element-wise、FMA、Linear+ReLU、Map-Reduce，在 RTX 3080
Ti、T4、A100 上测试不同输入规模和精度。

关键发现： - 内存密集和激活密集操作通常更适合融合； - Atomic-based
Map-Reduce 的融合收益有限； - 改成 block-level shared-memory aggregation
后，融合收益重新出现； - 说明融合收益受归约结构、同步方式和数据流影响。

作者指出的限制包括：独立微基准主要为一维 Tensor；GPU 主要覆盖
Turing/Ampere；更复杂的矩阵、卷积和高计算强度操作还需要进一步研究；优化归约策略和自动融合决策仍值得探索。

**对本项目的启发：** 不要只把两个 sum 写在一起，要研究局部归约、partial
result 和主计算之间的关系。

## 2.4 XTAT --- XLA 会在多个 Consumer 中重复计算 Producer

Phothilimthana et al., *A Flexible Approach to Autotuning Multi-Pass
Machine Learning Compilers*, PACT 2021.

论文：https://mangpo.net/papers/xla-autotuning-pact2021.pdf

重点阅读：Introduction、§II-B Operator Fusion、Figure
2、§III、Limitations、Evaluation。

这是本阶段最关键的理论参考。

论文明确指出：当一个 Producer 被融合到多个 Consumer 时，由于 Consumer
可能具有不同迭代空间，Producer 可能需要在每个 Consumer
中重新计算。编译器需要在"重复计算"和"减少内存通信"之间权衡。

XTAT 还强调，Fusion、Layout、Tiling
等优化会互相影响，不能总是孤立地优化单个 Pass。

**与当前项目的直接联系：** 当前 HLO 的主输出、q1、q2
路径可能包含相同的上游计算。XTAT 为这种现象提供了编译器层面的解释。

## 2.5 OpenXLA 实现文档

-   From HLO to Thunks：https://openxla.org/xla/hlo_to_thunks
-   GPU
    Architecture：https://github.com/openxla/xla/blob/main/docs/gpu_architecture.md
-   PriorityFusion：https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/priority_fusion.h

OpenXLA 文档说明： - PriorityFusion 根据成本模型估计融合收益； -
成本模型考虑 Producer 可能被重复计算； - GPU Multi-Output Fusion
能将共享操作组织为多输出 Fusion，增加额外输出供其他消费者使用； -
该过程必须避免引入 HLO 依赖环。

**后端限制：** 官方上述详细 Pipeline 描述的是 XLA:GPU。不能假定 CPU
后端执行完全相同的 Multi-Output Fusion Pass。CPU 实验必须先检查实际 Pass
和生成代码；GPU 服务器可用于直接验证 GPU Multi-Output Fusion。

------------------------------------------------------------------------

# 三、第一阶段 HLO 审计：先证明到底重复了什么

重点文件：

`results/cpu/matrix/ir/E3_branch/post_post/optimized_hlo.txt`

E3_branch 的输入为 `[4096,256]`，共有 1,048,576 个元素。报告中该案例的
post/post 双点观测延迟约为无观测基线的 3.96 倍。

在 HLO 中重点检查三个融合计算体：

1.  q1 的 compare/convert Fusion；
2.  q2 的 compare/convert Fusion；
3.  主输出 y 的 Fusion。

审计内容：

-   q1 路径是否重新计算第一层 affine；
-   q2 路径是否重新计算第一层 affine、ReLU 和第二层 affine；
-   主输出路径是否又计算同一段主干；
-   q1/q2 的 compare/convert 是否分别输出完整的 `s32[4096,256]`；
-   每个 count 后面是否存在独立的 reduce-window/reduce 链。

`s32[4096,256]` 按 int32 逻辑尺寸约为 4 MiB；两个计数路径共涉及约 8 MiB
的逻辑中间张量。

注意： - 这是 HLO 的逻辑张量大小； - 不能直接写成"实际 DRAM 写入 8
MiB"； - 必须进一步检查 Buffer Assignment、LLVM/汇编或 GPU profiler
才能确认物理数据流量。

**Phase 0 的输出：** - `fusion_paths.csv`：y/q1/q2 各自的 Fusion
路径； - `duplicated_ops.csv`：同一源算子在多个 Fusion body
中出现的次数； - `reduction_structure.csv`：比较、转换、归约的 shape
和层级； - `intermediate_bytes.csv`：逻辑中间张量大小； - 一张标出 HLO
行号的路径图。

至少对 E3_branch、E0_chain、M3_chain、N3_chain、C3_chain 做审计。

------------------------------------------------------------------------

# 四、固定语义与候选执行方案

所有候选必须返回相同的：

\[ (y,q_1,q_2) \]

其中 q1、q2
是每次执行的精确正激活数量。继续使用第一阶段的统计定义、输入种子和正确性容差。

## V0：No-Probe Baseline

只返回 y，作为原模型执行时间基线。

## V1：Current Instrumentation

沿用当前 `program(case, "post_post")`，分别计算 q1、q2。

这是第二阶段的主要对照组。

## V2：Variadic / Joint Reduction

尝试用 `jax.lax.reduce` 的多输入、多输出形式联合归约 q1、q2。

目标： - 检查是否能生成一个多输出 reduce； -
比较两个独立归约与联合归约； - 检查是否减少归约调度； - 检查同时保留两个
Mask 是否增加临时存储。

这是消融实验，不作为主要创新。

## V3：Explicit Materialization / Shared Stage

第一阶段单独计算并返回 `(a1,q1)`；第二阶段接收 a1，计算 `(y,q2)`。

目标： - 显式共享第一层激活； - 避免不同输出路径重复计算第一层； - 测量
a1 物化、额外阶段调用和读取成本。

这是"重复计算 vs 物化共享结果"的重要对照。

## V4：Tiled Multi-Output Observer（主要候选）

将输入沿 batch 维分块。每个 tile 内完成：

``` python
def tile_body(x_tile):
    z1 = layer1(x_tile)
    a1 = relu(z1)
    q1_partial = sum(a1 > 0)

    z2 = layer2(a1)
    a2 = relu(z2)
    q2_partial = sum(a2 > 0)

    y_tile = readout(a1, a2)

    return y_tile, q1_partial, q2_partial
```

之后： - 拼接所有 y_tile； - 汇总 q1_partial； - 汇总 q2_partial。

优先使用 `jax.lax.map`；必要时再比较 `jax.lax.scan`。

要求： - Chain / Branch 的原始语义都要保留； - MLP 按 batch 行分块； -
Normalization 不得拆散单行归约； - Conv 优先按 batch 维分块； -
无法安全分块的结构明确标记，不得偷偷改变数学语义。

核心观察： - y 和局部计数是否进入同一 tile body； - q1/q2
是否还生成完整尺寸 Mask； - 上游算术是否仍在多个 Fusion body 中重复； -
分块是否引入额外循环、调度或并行度损失。

## V5：GPU Multi-Output Fusion（可选）

如果有 GPU 服务器： - 检查 XLA:GPU 的 Multi-Output Fusion； - 比较 V1 和
V4 的优化后 HLO； - 使用 XProf / Nsight Systems 观察 Kernel 时间线； -
记录 Kernel 数量、执行时间和可用的内存指标。

CPU 与 GPU 数据必须分开汇报。

------------------------------------------------------------------------

# 五、实验矩阵

## 5.1 第一轮关键案例

先跑： - E0_chain：小型逐元素对照； -
E3_branch：观测开销较高的重点案例； - M3_chain：矩阵乘法； -
N3_chain：归一化和行级归约； - C3_chain：卷积。

每个案例先比较 V0--V4。

## 5.2 Tile Size 搜索

根据 batch 维选择有效值。

-   E3：1、4、16、64、256 行；
-   M3：1、4、16、64、256 行；
-   N3：1、4、16、64、256 行；
-   C3：1、2、4、8、16 个样本。

小尺寸图根据实际 batch 维缩小范围。

不要只保存最佳 tile size；所有候选都要保存，以便分析趋势。

## 5.3 第二轮完整矩阵

第一轮确认某个候选有结构优势后，扩展到全部 32 个合成条件：

E/M/N/C × 四档规模 × Chain/Branch。

优先比较： - V0； - V1； - V2； - V3； - 最佳 V4 配置。

## 5.4 外部 ResNet

继续保留 R1/R2 作为外部图验证。

由于 batch size 为 1，batch tiling 不适用。先比较 V1/V2/V3。空间 tile
涉及卷积 halo 和边界处理，列为扩展项，不作为第一轮强制任务。

------------------------------------------------------------------------

# 六、指标与计时协议

每个配置保存：

-   lowering time；
-   compile time；
-   warm-up；
-   稳态执行时间；
-   每进程中位数；
-   跨进程中位数；
-   A/A 噪声范围；
-   可获取的峰值内存和临时内存。

继续沿用： - 固定输入种子； - 固定权重； - 随机交错候选顺序； -
`jax.block_until_ready`； - 多进程独立复测； - A/A 噪声控制。

主要指标：

\[ Speedup=`\frac{T_{V1}}{T_{candidate}}`{=tex} \]

\[ Overhead=T\_{observed}-T\_{base} \]

\[ Recovery= `\frac{T_{V1}-T_{candidate}}`{=tex} {T\_{V1}-T\_{base}} \]

只有当 V1 的观测开销显著大于噪声时才计算 Recovery。

另外报告： - HLO 重复算子数量； - 估算重复 FLOPs； - 完整尺寸 Mask
的逻辑字节数； - reduction 层级； - partial result 数量； - tile
数量； - 编译时间和执行时间。

------------------------------------------------------------------------

# 七、Recomputation-aware Cost Model

第一阶段模型主要预测 pre/post 观测位置；第二阶段改为预测 V1/V2/V3/V4
的成本。

概念性分解：

\[ T\_{`\text{observed}`{=tex}} = T\_{`\text{main}`{=tex}}+
T\_{`\text{recompute}`{=tex}}+ T\_{`\text{reduce}`{=tex}}+
T\_{`\text{materialize}`{=tex}}+ T\_{`\text{dispatch}`{=tex}} \]

这里的各项是分析模型，不代表都能从总时间直接分离测得。

## 7.1 特征

从基线 Jaxpr/HLO 提取：

-   重复 Producer 数量；
-   重复算子对应的 shape 和 FLOPs；
-   y/q1/q2 路径共享前缀长度；
-   compare/convert 的输入 shape；
-   full-size mask 逻辑字节数；
-   reduction 输入元素数和层级；
-   Producer fanout；
-   中间激活物化字节数；
-   tile size 和 tile 数量；
-   dot/conv/reduce 数量。

## 7.2 第一版解析模型

\[ T\_{`\text{estimate}`{=tex}}= T\_{`\text{dispatch}`{=tex}}+
`\max`{=tex}`\left`{=tex}(
`\frac{F_{\text{total}}}{P_{\text{effective}}}`{=tex},
`\frac{B_{\text{total}}}{BW_{\text{effective}}}`{=tex}
`\right`{=tex})+T\_{`\text{reduction}`{=tex}} \]

分别对 CPU、GPU 校准，不能把两种硬件的数据混在一起。

## 7.3 预测实验

比较： - 固定 V1； - 固定 tile 规则； - 解析成本模型； - 轻量 Ridge
Regression； - 实测 Oracle（只作为上界）。

使用 leave-one-family-out，避免同一个图的不同 tile
配置同时进入训练和测试。

报告： - MAE； - 预测 regret； - 选择方案与实测最优的差距； - 是否落在
A/A 噪声范围； - 各图族覆盖率； - 失败案例。

------------------------------------------------------------------------

# 八、代码结构与结果归档

建议新增：

``` text
fusionscope/
    phase2/
        variants.py
        tiling.py
        joint_reduce.py
        staged.py
        hlo_recompute.py
        cost_model.py
        analysis.py

results/
    phase2/
        protocol.json
        screening/
            raw/
            configs.csv
            correctness.csv
            timings.csv
        ir/
            baseline/
            joint_reduce/
            staged/
            tiled/
        analysis/
            fusion_paths.csv
            duplicated_ops.csv
            reduction_structure.csv
            intermediate_bytes.csv
        model/
            features.csv
            predictions.csv
            model.json
            heldout_results.csv
        figures/
        report/
            REPORT.md
```

第一阶段文件必须保持不变。

------------------------------------------------------------------------

# 九、阶段验收

## Phase 0：HLO 审计

必须先完成 E3_branch 的 y/q1/q2 三条路径追踪，确认重复算子、比较 Mask
和归约结构。对 E/M/N/C 各补至少一个案例。

## Phase 1：候选实现

实现 V2、V3、V4，全部通过 y 与 q1/q2 正确性校验。

## Phase 2：筛选实验

完成 E0、E3_branch、M3、N3、C3 的对照和 tile size 搜索。

## Phase 3：完整矩阵

扩展到 32 条件，保存全部 IR 和原始计时。

## Phase 4：成本模型

完成特征抽取、模型训练、整图族留出和预测误差分析。

## Phase 5：可选 GPU

在兼容 GPU 上验证 Multi-Output Fusion 和实际 Kernel 时间线。

## Phase 6：报告

报告必须包含成功案例、失败案例、HLO 证据、成本模型误差和硬件范围限制。

------------------------------------------------------------------------

# 十、研究纪律

1.  HLO 重复算术不等于已经测得相同数量的物理 DRAM 访问。
2.  CPU Fusion 数量不等于 GPU Kernel 数量。
3.  Multi-Aggregate、Multi-Output Fusion
    本身已有先前研究，不得声称原创。
4.  不得把第三方 GPU 数据当成本地 XLA 实测。
5.  所有候选都必须返回相同的 y、q1、q2。
6.  不得只报告最快配置，必须保留退化结果。
7.  不得把噪声范围内的微小差异写成稳定提升。
8.  如果 XLA 将候选优化回相同 HLO，应记录为结果。
9.  如果 V4 更慢，应分析 tile 调度、并行度和物化成本，不要删除失败实验。
10. 不得覆盖第一阶段数据和协议。

------------------------------------------------------------------------

# 十一、最终预期贡献

**Contribution 1 --- IR-level Diagnosis**

分析精确激活观测引入的 Producer 重复计算、Fusion 路径和归约中间结果。

**Contribution 2 --- Observer-aware Execution Strategies**

实现并比较联合归约、显式物化共享、分块局部统计和多输出执行。

**Contribution 3 --- Recomputation-aware Cost Model**

结合重复计算、物化、归约、图依赖和 tile 参数预测不同观测方案的成本。

**Contribution 4 --- Reproducible Benchmark**

公开四类图、不同规模、不同拓扑的代码、IR、计时和分析结果。

不宣称提出了通用 Fusion
算法；贡献是对精确激活观测场景的机制分析、执行方案比较和可复现工具。

------------------------------------------------------------------------

# 十二、Codex 第一条执行指令

请从 `f2df6a2` 创建分支 `phase2-observer-recompute`。

第一步只做 Phase 0，不要先改算法：

1.  读取 `E3_branch/post_post/optimized_hlo.txt`；
2.  从 ENTRY 的 y、q1、q2 分别追踪 Fusion computation；
3.  输出每个 Fusion body 的源算子及 shape；
4.  识别重复的 `u_z`、`u_relu`、`v_z`、`v_relu`；
5.  识别 compare/convert 输出的完整张量；
6.  统计每级 reduction 的输入/输出 shape；
7.  对 E0、M3、N3、C3 做同样审计；
8.  能获取时检查 CPU LLVM IR 或 Buffer Assignment；
9.  生成
    `fusion_paths.csv`、`duplicated_ops.csv`、`reduction_structure.csv`、`intermediate_bytes.csv`
    和 HLO 路径图；
10. 最后区分"已由 HLO 证明的事实"和"需要性能实验验证的推测"。

完成 Phase 0 后，提交审计报告，再开始 V2/V3/V4 实现。

------------------------------------------------------------------------

# 十三、参考文献与技术资料

1.  Spartan, IEEE TPDS 2021\
    https://eprints.gla.ac.uk/240696/2/240696.pdf

2.  On Optimizing Operator Fusion Plans for Large-Scale Machine Learning
    in SystemML, PVLDB 2018\
    https://www.vldb.org/pvldb/vol11/p1755-boehm.pdf

3.  Analyzing the Impact of Kernel Fusion on GPU Tensor Operation
    Performance, Electronics 2026\
    https://www.mdpi.com/2079-9292/15/5/1034

4.  A Flexible Approach to Autotuning Multi-Pass Machine Learning
    Compilers, PACT 2021\
    https://mangpo.net/papers/xla-autotuning-pact2021.pdf

5.  OpenXLA: From HLO to Thunks\
    https://openxla.org/xla/hlo_to_thunks

6.  OpenXLA: GPU Architecture Overview\
    https://github.com/openxla/xla/blob/main/docs/gpu_architecture.md

7.  OpenXLA: PriorityFusion implementation\
    https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/priority_fusion.h

8.  FusionScope baseline commit\
    https://github.com/ZZzJunzum/Compiler_report/tree/f2df6a2

9.  First-stage CPU report\
    https://github.com/ZZzJunzum/Compiler_report/blob/f2df6a2/results/cpu/report/REPORT.md
