# E0 双点观测先导

CPU，4096 个 FP32 元素，链式读出。每案 3 个独立进程，每进程预热 6 次，随机交错测 20 轮。各配置的逐次原始观测在 `raw/p*/E0_chain/times.csv`。同步端点覆盖全部输出。

| 配置 | seed 42 的计数 | 延迟 µs | 配对增量 µs |
|---|---|---:|---:|
| base | [] | 38.800 | 0.000 |
| u_pre | [2117] | 40.550 | 2.000 |
| u_post | [2117] | 42.400 | 3.350 |
| v_pre | [2119] | 42.650 | 3.650 |
| v_post | [2119] | 42.600 | 3.550 |
| pre_pre | [2117, 2119] | 43.950 | 5.200 |
| pre_post | [2117, 2119] | 43.250 | 4.800 |
| post_pre | [2117, 2119] | 44.250 | 5.350 |
| post_post | [2117, 2119] | 43.900 | 5.450 |

同一可执行文件 A/A 的差值门槛为 2.407 µs。双点观测的实测顺序为 pre_post、post_post、pre_pre、post_pre。前两名落入波动范围或未获进程多数，先导无法稳定排序。 后续保留全部四种组合，并优先核对这两条比较分支是否重新计算了 affine 加法。

## 从返回值找回两个计数

以下行号对应 [pre_pre 优化后 HLO](ir/E0_chain/pre_pre/optimized_hlo.txt)。ENTRY 的 tuple 顺序是 y、q1、q2。

```text
 184  ROOT %tuple.1 = (f32[16,256]{1,0}, s32[], s32[]) tuple(%broadcast_add_fusion, %wrapped_reduce, %wrapped_reduce.1)
```

第一处比较读取第一段加法的结果，第二处比较读取第二段加法的结果。它们在两个单独的融合体内。两个归约结果都沿 ENTRY 的 reduce-window 和 reduce 调用进入返回元组。

```text
  92  %gt.1 = pred[16,256]{1,0} compare(%add.4, %broadcast.6), direction=GT, metadata={op_name="jit(workload)/probe_u/gt" stack_frame_id=13}
  78  %gt.0 = pred[16,256]{1,0} compare(%add.0, %broadcast.5), direction=GT, metadata={op_name="jit(workload)/probe_v/gt" stack_frame_id=20}
```

双点交互见 [四项配对计算](first_interactions.csv)。四种不同信息量的先导见 [only_y / sample / mean / full](four_probes.csv)。其中 sample 与 mean 都多返回 4 字节，mean 还需要读取全张量做归约。
