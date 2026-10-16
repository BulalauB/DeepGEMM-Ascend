# DeepSeek-V4-Flash：DeepGEMM-Ascend MegaMoE 分核静态分析

分析对象是本仓库的 `take_linear_task`、`aic_gemm_pipeline`、`aiv0_weight_pipeline` 和 `aiv1_epilogue_comm`，不套用 experiments/megamoe 的另一套 wave 预算。

官方 [DeepSeek-V4-Flash config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json) 给出的 MoE 参数为 H=4096、I=2048、E=256、top-k=6、shared=1，routed 专家权重为 FP4。下面只分析这个 MoE kernel；模型其他模块不在范围内。

用户尚未指定 EP 数、实际路由 counts 和 T/Tmax。示例假设每个源 rank 的 T 相同、每条 top-k 路由有效，并严格均匀落到256个专家；Tmax=T，EP分别取8和64。实际核映射公式不依赖均衡假设，总块数与有效 M 必须由实际 counts 确定。

## 1. 两层的 task 数恰好相等

| 阶段 | 逻辑输入/输出 | 每个 M-block 的 task 数 |
| --- | --- | --- |
| GMM1 | `[M,4096] -> [M,gate2048 + up2048]` | `N1=2048/128=16` |
| SwiGLU | 每个 task 的 gate128/up128 -> 中间128列 | 不属于独立 AIC task |
| GMM2 | `[M,2048] -> [M,4096]` | `N2=4096/256=16` |

GMM1 权重经过预排布，每个物理 N256 task 包含相同128个中间通道的 gate和up。第n个 task负责 gate/up各自的 `[128*n,128*(n+1))`，不能把物理N256直接解释成256个中间输出通道。GMM2第n个 task输出 `[256*n,256*(n+1))`。

来源：[调度器](../deep_gemm/include/deep_gemm/scheduler/mega_moe.hpp)、[权重预排布](../deep_gemm/mega/__init__.py)、[AIC流水](../deep_gemm/include/deep_gemm/mega_moe.hpp)。

## 2. 一波32 task如何落到32个AIC

某一层第j波的 task_base=32*j，core号c在[0,32)：

```text
task_idx    = 32*j + c
pool_block  = task_idx / 16 = 2*j + c/16
n_block     = task_idx % 16 = c%16
```

| AIC | 逻辑 M-block | N-block | 配对 AIV0 | 配对 AIV1 |
| --- | --- | --- | --- | --- |
| 0 | 2j | 0 | 准备此task权重 | 消费此task的FIX/epilogue |
| 1 | 2j | 1 | 同上 | 同上 |
| 15 | 2j | 15 | 同上 | 同上 |
| 16 | 2j+1 | 0 | 同上 | 同上 |
| 17 | 2j+1 | 1 | 同上 | 同上 |
| 31 | 2j+1 | 15 | 同上 | 同上 |

同一块的16个AIC读取相同的M行、计算不同N列。不是每个AIC只计算8行；每核8行是满M256块的AIV1通信行分工。

一波覆盖最多两个M-block，即最多512条路由行；不是512个原始输入token。尾波若只剩一块，core16~31跳过该层对应任务。块属于哪个expert，由ExpertCursor按本地专家的有效接收数推进；不同wave可以跨expert，核不固定绑定某个expert。

若把“wave”定义为一批已接收的256条同expert路由行，则一批只有16个G1任务和16个G2任务。它的16个计算核在本块Dispatched和权重ready后可以启动G1，不需要等下一批凑满32核。若下一块属于本轮已知任务但尚未ready，分配给它的另16核等该块；不存在全核凑齐后统一发射的条件。若本轮只有这一块，其余核跳过无效任务，不能把尾任务留到下一次kernel调用补满。

## 3. routed的GMM1/GMM2顺序

下一波GMM2起点为32*j，完整波覆盖M-block 2j和2j+1。因此：

```text
required_G1_end = ceil((32*j+32)/16)*16 = 32*(j+1)
G1需继续领先一波：G1_base < required_G1_end+32
```

任务足够时，初始顺序如下：

| 顺序 | 层 | task区间 | 覆盖M-block |
| --- | --- | --- | --- |
| 1 | G1 | [0,32) | 0、1 |
| 2 | G1 | [32,64) | 2、3 |
| 3 | G2 | [0,32) | 0、1 |
| 4 | G1 | [64,96) | 4、5 |
| 5 | G2 | [32,64) | 2、3 |
| 6 | G1 | [96,128) | 6、7 |
| 7 | G2 | [64,96) | 4、5 |

两层任务耗尽时，最后两波G2连续排出。若总块数很少，则先耗尽已有G1，再排G2。

这是每核独立重放的任务次序，并非全核在时间线上同时切阶段。没有wave末尾全核barrier；任务实际运行仍受输入ready、权重ready、UB credit和GM ring回收约束。G2读取某块前必须等该块16个G1 epilogue任务全部贡献Linear1Ready。

“本块G1可以立即启动”不等于本块G2一定立即启动。若本轮B>=3，core0~15的固定顺序是G1块0 -> G1块2 -> G2块0；块2尚未Dispatched时，AIC会等块2而暂时无法推进G2块0，即使块0的激活已ready。B>=4时core16~31对应G1块1 -> G1块3 -> G2块1。AIV1仍小步推进Dispatch以供给后续块。这是固定lookahead顺序导致的等待，不是为了凑满核；调度器不探测ready，也没有work stealing。B=1/2时不存在额外的有效G1块2任务，相关越界task会被跳过。

## 4. 代入EP和T后的严格均衡示例

若所有rank都有T个token且每个token有6条有效路由，则每专家平均接收 `6*P*T/256` 行。严格均衡且能整除时：

```text
本rank专家数 = 256/P
B = (256/P) * ceil((6*P*T/256)/256)
G1/G2各有16*B个task，各ceil(B/2)个wave
```

一般情况使用实际 `B=sum_local_experts ceil(received_rows[e]/256)`，不能直接对本rank总接收数取ceil后替代逐专家padding。

| EP P | 每rank T | 每本地expert行数 | 本地expert数 | routed B | 每层task数 | 每层wave数 |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 96 | 18 | 32 | 32 | 512 | 16 |
| 8 | 4096 | 768 | 32 | 96 | 1536 | 48 |
| 64 | 96 | 144 | 4 | 4 | 64 | 2 |
| 64 | 4096 | 6144 | 4 | 96 | 1536 | 48 |

EP8/T96：每expert只有一个块，第0波的两组16核分别计算本地expert0/1，第1波计算expert2/3。每个AIC各有16个G1、16个G2任务。

EP8/T4096：每expert三个满M256块。第0波块0/1均属于expert0；第1波块2属于expert0、块3属于expert1；第2波块4/5均属于expert1。每个AIC各有48个G1、48个G2任务。同expert跨M块会重复读取对应权重，代码有L2 hint，未实现整个expert权重跨块固定驻留L1。

## 5. task与实际MMAD指令分块

| routed块有效M | 配置 | 实际指令M | MAD_N | MAD_K | G1每task的MAD调用数 | G2每task的MAD调用数 |
| --- | --- | --- | --- | --- | --- | --- |
| <=128 | M128配置 | Align16(valid_m) | 128 | 256 | 4096/256*2=32 | 2048/256*2=16 |
| >128 | M256配置 | Align16(valid_m) | 256 | 128 | 4096/128=32 | 2048/128=16 |

M128配置的逻辑N256拆成两个N128，L0A在两次N子乘之间复用。M256配置每个K段做一个N256。两种配置对这个shape的MAD调用次数相同，但每次指令的M/N/K不同，性能不能仅看调用次数。

EP8/T96的18行选M128配置，实际指令M=32；不是固定算128或256行。EP64/T96的144行选M256配置，实际指令M=144。EP8/T4096的满块实际M=256。

GMM1两个M128 epilogue槽是FIX写出与向量后处理的粒度，不能据此把M256配置的矩阵计算也解释成两次M128。

## 6. AIV0和AIV1的分工

metadata worker数 `min(max(ceil(T*6/512),1),32)`：T96时为2，T4096时为32。仅这些AIV0执行512线程SIMT metadata构建；全部32个AIV0随后参与到达等待、块数汇合和routed权重流水。AIV0按同一routed任务顺序准备配对AIC需要的权重，不执行shared权重展开。

AIV1有两种不同的分工：epilogue按配对AIC的(M,N)任务执行；通信按 `m=core_idx+32*k` 分摊每个M-block的路由行。两套索引不能混用。

| 场景 | routed每块有效行 | 单AIV1每块pull/push行数 | 单AIV1整轮pull/push行数 |
| --- | --- | --- | --- |
| EP8/T96严格均衡 | 18 | core0~17各1行；18~31无有效行 | core0~17各32行；其余0行 |
| EP8/T4096严格均衡 | 256 | 全部core各8行 | 全部core各768行 |
| EP64/T96严格均衡 | 144 | core0~15各5行；16~31各4行 | 分别20行和16行 |

没有有效通信行的AIV1仍贡献该块的Dispatched/CombineSent证书，也仍执行自己配对AIC的epilogue任务。例如EP8/T96中core18~31无有效pull/push行，不等于对应AIC/AIV1没有计算任务。

Combine常规每4个逻辑块确认一批。小块并不保证每核实际发送32行：EP8/T96前18个AIV1每批只发4行，其余0行；代码按块数计数，不按本核实际行数触发。四个UB push槽与这个确认间隔是独立参数。

## 7. shared专家与尾部归并

shared=1且I=2048，所以shared G1/G2每块也各16个task。Q=ceil(T/256)：

| T | shared块数Q | 每层shared task数 | 每层wave数 | AIC分工 |
| --- | --- | --- | --- | --- |
| 96 | 1 | 16 | 1个不满波 | core0~15各1个task，16~31无shared task |
| 4096 | 16 | 256 | 8 | 每个core每层8个task |

每个AIC先重放自己的shared G1，再shared G2，随后routed；没有全核的shared阶段结束barrier。空闲半波核可以更早等metadata/推进routed，具体时间还取决于ready事件。shared输入本来就在本rank，shared GMM2直接写本地shared combine槽，没有routed的跨rankpull/push。

最终输出有6个routed槽和1个shared槽，由64个AIV分摊归并。若overlap关闭，token按连续区间分摊；若开启，按 `token_idx=2*core_idx+subblock_idx+64*k` 轮转分摊。

注册Tmax=T时，T96的combine容量为5.25MiB，kOverlapReduce关闭；T4096为224MiB，开启。若T96复用Tmax4096的布局，该编译开关仍开启，不能只看本轮T。

H4096选择最多128个GM ring块：EP8/Tmax96为R=50，EP8/Tmax4096为R=128。上述均衡B32/B96均不超过R，因此没有数据环绕，AIV0/AIV1主体中的提前reduce调用也不启用，因为这些调用另要求B>R；最终归并仍执行。EP64两个示例R均为128，B仍小于R。

## 8. 静态结论的范围

这里核对了当前源码中的分块公式、任务序列及条件，并用Python计算上述参数表。没有执行CANN编译、NPU计算或性能测量。

shape能够确定N列任务数、固定核映射和K分块。精确的expert/block映射、valid_m、通信有效核数则需要本轮expert_recv_count；上述严格均衡示例不能替代实际路由。32 task排满也不代表32个核时间利用率必然一致，实际仍可能等待metadata、权重、epilogue或远端通信。

## 9. EP8：按路由分布统计评估

当前仓库README的MegaMoE实测表没有H4096/I2048/E256的Flash条目，也未保存这个shape的逐专家接收counts。下面是统计模型估算，不是Flash模型真实路由或NPU实测。模型依据本仓库 `tests/test_mega_moe.py::make_topk`：各token独立生成同分布的随机expert分数，选择6个不同专家。因此对任一expert，接收行数X服从 `Binomial(8*T,6/256)`；expert之间有相关性，但期望块数可以用线性求和，不需要假设expert相互独立。

计算脚本：[analyze_flash_route_statistics.py](analyze_flash_route_statistics.py)。结果：[deepseek_v4_flash_route_statistics.json](deepseek_v4_flash_route_statistics.json)。需要NumPy/SciPy；在仓库根目录执行：

```powershell
python docs/analyze_flash_route_statistics.py --ep 8 --tokens 96 4096 --output docs/deepseek_v4_flash_route_statistics.json
```

| 统计项 | T96 | T4096 |
| --- | --- | --- |
| 单expert平均行数 | 18 | 768 |
| 单expert行数标准差 | 4.19 | 27.39 |
| 单expert行数5%/95%分位 | 11 / 25 | 723 / 813 |
| 每rank接收路由数均值/标准差 | 576 / 22.23 | 24576 / 145.20 |
| routed B期望 | 约32 | 111.69 |
| 每层task数期望 | 约512 | 1787.09 |
| 每层wave数 | 约16 | 约56；精确ceil奇偶修正小于0.5波 |
| 指令M按16对齐后的行效率 | 69.1% | 99.0% |

这里行效率是 `期望有效行数/期望Align16行数`，不代表硬件利用率。单expert分位也不代表整个rank或全部expert同时落在该区间。

T4096的768行恰好是3*256边界。模型中 `P(X>768)=49.04%`，这些expert通常需要4个块；其余通常需要3个块。因此：

```text
E[B] ≈ 32*(3+0.4904) = 111.69
相对严格均衡B96：块数/task数约增加16.35%
每层task数：1536 -> 1787.09
每层wave数：48 -> 约56
```

代码将16行组在expert的各块之间均分。769行不是256/256/256/1，而是208/192/192/177；最后一块对齐为192。增加块数会增加N/K任务及其权重处理次数，但实际矩阵M也随之减小，不能把task数增加16.35%直接解释成FLOPs或延迟增加16.35%。

一个完整M-block的routed两层packed FP4 data读取请求合计12MiB（G1 8MiB、G2 4MiB，未含scale）。按task逐块准备权重的逻辑请求量：T96约384MiB；T4096严格均衡为1152MiB，统计模型约1340.32MiB。这个量包含可命中L2的读取，不能当作实测HBM流量；重复FP4展开和UB/L1搬运仍是需要评估的工作。

AIV1通信分工的统计期望如下，pull和push各自使用同一行数分配：

| AIV1 core号 | T96整轮行数期望 | T4096整轮行数期望 |
| --- | --- | --- |
| 0 | 32.03 | 797.08 |
| 15 | 22.92 | 787.72 |
| 16 | 20.07 | 753.92 |
| 23 | 3.15 | 743.82 |
| 31 | 0.05 | 733.28 |

T96并非每轮只有固定前18个core有行：接收数随机波动，core是否有有效行取决于各块valid_m。但总体低编号core搬运更多，高编号core极少通信。所有core仍贡献完成证书并执行自己的epilogue列任务。T4096的行数差异较小，但16行组均分加上 `m=core+32*k` 仍产生结构性差异；这里最大/最小平均行数比约1.087，不能直接解释成kernel时间差。

每rank平均远端路由数分别为504和21504；本地路由也执行对应的行搬运路径，但不计为跨rank流量。没有使用其他shape的README延迟外推Flash耗时。

### 用每轮实际counts替换模型

完成一次kernel并在下次复用buffer前，可以保存当前rank的counts：

```python
import json
from pathlib import Path
import torch

torch.npu.synchronize()
counts = buffer.expert_recv_count.cpu().sum(dim=0).tolist()
Path("counts.json").write_text(json.dumps(counts), encoding="utf-8")
```

然后执行：

```powershell
python docs/analyze_flash_route_statistics.py --ep 8 --counts counts.json
```

脚本按实际expert顺序计算B、M128块数、32个AIC每层task数/对齐行工作量，以及32个AIV1各自pull/push行数。若使用 `cumulative_local_expert_recv_stats`，应取本轮前后差值；不要用多轮累计值或均值先代入ceil，因为 `E[ceil(X/256)]` 不等于 `ceil(E[X]/256)`。

这些counts能定位负载和分块开销；判断最终瓶颈仍需要逐rank耗时及PIPE/CV/远端等待数据，不能由counts单独确定。
