# FlashKDA Context Parallelism (CP) 优化

## 1. 概述

### 问题

FlashKDA 是一个带循环状态的注意力 kernel。对于长序列（如 64K tokens），单次前向的 kernel2 中状态递推是串行的，GPU SM 利用率低。

### 为什么需要 CP？—— FlashKDA vs fla_chunk_kda 基线对比

FlashKDA 的前向计算包含一个 **串行的状态递推过程**（`S_t = A_t · S_{t-1} + b_t`），这意味着：

- **无 CP 时**：整个序列的状态必须从头到尾**逐 chunk 串行计算**，GPU 大部分 SM 空闲等待
- **有 CP 时**：将长序列切分为多个 sub-segment，每个 sub-segment 可以**并行处理**，大幅提升 SM 利用率

#### 无 CP 时 FlashKDA vs fla_chunk_kda 对比（H20 GPU, D=128, 无 CP）

以下数据展示了 **不开启 CP** 时，FlashKDA 相对于 fla_chunk_kda（FlashAttention 风格的 chunk KDA 实现）的性能差距。这个差距正是 CP 优化要解决的核心动机。

| Batch | SeqLen | FlashKDA (ms) | fla_chunk_kda (ms) | Speedup vs fla |
|-------|--------|---------------|-------------------|----------------|
| 1 | 8K | 1.235 | 0.907 | **0.73×（慢 37%）** |
| 8 | 8K | 2.871 | 6.380 | **2.22×（快）** |
| 16 | 8K | 5.650 | 12.718 | **2.25×（快）** |
| 32 | 8K | 10.413 | 25.456 | **2.44×（快）** |
| 1 | 16K | 2.429 | 1.710 | **0.70×（慢 43%）** |
| 8 | 16K | 5.728 | 12.866 | **2.25×（快）** |
| 16 | 16K | 11.241 | 25.848 | **2.30×（快）** |
| 32 | 16K | 21.161 | 51.703 | **2.44×（快）** |
| 1 | 64K | 9.509 | 6.626 | **0.70×（慢 43%）** |
| 8 | 64K | 22.625 | 52.095 | **2.30×（快）** |
| 16 | 64K | 45.017 | 104.981 | **2.33×（快）** |
| 1 | 256K | 37.860 | 26.484 | **0.70×（慢 43%）** |
| 1 | 512K | 75.675 | 52.968 | **0.70×（慢 43%）** |

**关键观察**：
- **Batch=1 时 FlashKDA 比 fla 慢 ~30-43%**：因为单条长序列的串行状态递推无法充分利用 GPU 并行度，SM 利用率低
- **Batch≥8 时 FlashKDA 反超 fla 2.2-2.4x**：batch 足够大时，不同 batch 的序列可以并行处理，SM 利用率上来后 FlashKDA 的 kernel 效率优势显现
- **Batch=1 + 长序列是最坏场景**：T=512K, B=1 时 FlashKDA 需要 75.7ms，而 fla 只需 53.0ms，差距达 **43%**
- **Speedup vs fla 与 Batch 强相关，与 SeqLen 弱相关**：B=1 时恒定 ~0.70x，B≥8 时恒定 ~2.3x

> 💡 **这就是 CP 要解决的问题**：当 Batch 较小（尤其是 B=1）且序列较长时，FlashKDA 因串行瓶颈比 fla 还慢。CP 通过切分序列让多个 sub-segment 并行执行，使 B=1 场景下也能获得接近 B≥8 时的 SM 利用率。

> 📖 **CP 优化的数学原理详见博客**：[DeltaNet 状态递推与 Context Parallelism](https://yywangcs.notion.site/DeltaNet-2a9fc9f5d8058013a498f34e0b25bd52)

### 方案

| 方案 | 计算量 | 原理 |
|------|--------|------|
| 无 CP | 100% (基线) | 整个序列串行递推状态 |
| **CP (warmup + correct)** | **~105-110%** | 只跑末尾几个 chunk 的 state_only（计算 ht）+ 一次完整 fwd |

核心思路：
1. **Warmup 阶段**：对每个 sub-segment，只处理末尾少量 chunk，计算出"数据驱动的终态" `ht`
2. **修正阶段**：利用线性性质 `S_final = mt · h0 + ht`，链式求解每段的正确初始状态
3. **并行执行**：所有 sub-segment 用修正后的 h0 并行跑完整前向

---

## 2. 性能实测 (H20 GPU, B=1, H=16, D=128)

### 强 decay 对比 (lb=-5.0, 生产参数)

| 序列长度 | 无 CP | CP 优化 | 加速比 |
|----------|-------|---------|--------|
| T=4K | 0.63ms | 0.35ms | **1.82x** |
| T=8K | 1.24ms | 0.54ms | **2.29x** |
| T=16K | 2.45ms | 0.99ms | **2.47x** |
| T=32K | 4.87ms | 1.76ms | **2.77x** |
| T=64K | 9.64ms | 3.91ms | **2.46x** |

### Weak decay 对比 (lb=-1.0)

| 序列长度 | 无 CP | CP 优化 | 加速比 |
|----------|-------|---------|--------|
| T=4K | 0.63ms | 0.35ms | **1.82x** |
| T=8K | 1.23ms | 0.54ms | **2.29x** |
| T=16K | 2.44ms | 0.99ms | **2.47x** |
| T=32K | 4.84ms | 1.75ms | **2.76x** |
| T=64K | 9.65ms | 3.91ms | **2.47x** |

### Very weak decay 对比 (lb=-0.5)

| 序列长度 | 无 CP | CP 优化 | 加速比 |
|----------|-------|---------|--------|
| T=4K | 0.63ms | 0.35ms | **1.82x** |
| T=8K | 1.23ms | 0.54ms | **2.30x** |
| T=16K | 2.44ms | 0.99ms | **2.47x** |
| T=32K | 4.84ms | 1.76ms | **2.76x** |
| T=64K | 9.64ms | 3.91ms | **2.47x** |

**关键结论**：
- **CP 在所有 decay 强度下均稳定提供 ~1.8x-2.8x 加速**
- 序列越长，加速越明显（T=64K 时稳定 ~2.5x）
- 加速比不随 decay 强度退化，这是 warmup + correct 方案的核心优势
- 额外开销仅 ~5-10%（warmup chunks 的 state_only 计算 + 串行修正）

---

## 3. 数学原理

> 📖 **详细推导见博客**：[DeltaNet 状态递推与 Context Parallelism](https://yywangcs.notion.site/DeltaNet-2a9fc9f5d8058013a498f34e0b25bd52)

### 3.1 KDA 状态递推

每个 chunk（16 tokens）的状态更新：

```
S_new = diag(g_total) · S_old + k_restored^T · INV · ((v - k_decayed · S_old) · β)
```

整理为关于 `S_old` 的仿射变换 `S_new = A · S_old + b`：

```
A = diag(g_total) - k_restored^T · INV · diag(β) · k_decayed    [D×D]
b = k_restored^T · INV · diag(β) · v                            [D×D]
```

### 3.2 ht —— 数据驱动的终态

从零状态出发，经过所有 chunk 后的状态：

```
初始化: S = 0
每个 chunk: S = A · S + b
输出: ht = S_final
```

**含义**：如果这个 segment 从空白开始，纯靠 k/v/β 数据能积累出什么状态。

### 3.3 mt —— 初始状态的转移矩阵

从单位矩阵出发，只追踪 h0 如何传播：

```
初始化: M = I
每个 chunk: M = A · M = [diag(g) - kr^T · INV · diag(β) · kd] · M
输出: mt = A_n · A_{n-1} · ... · A_0
```

**含义**：初始状态 h0 经过该 segment 后变为 `mt · h0`。

### 3.4 线性性质

最终状态对 h0 是线性的：

```
S_final = mt · h0 + ht
```

这是 CP 能够工作的**核心数学基础**。

### 3.5 串行修正

CP 将序列分为 sub-segment 0, 1, ..., N-1，利用线性性质链式求解每段的 h0：

```
h0[0] = raw_h0                    (用户提供，或 0)
h0[1] = mt[0] · h0[0] + ht[0]
h0[2] = mt[1] · h0[1] + ht[1]
...
h0[i+1] = mt[i] · h0[i] + ht[i]
```

### 3.6 为什么 mt 通常不需要？

当 gate decay 强时（`lower_bound = -5`），per-chunk：

```
||mt|| ≈ exp(16 × (-5) × 1.4427 × 0.5) = exp(-57.7) ≈ 10^{-25} ≈ 0
```

此时 `h0[i+1] ≈ ht[i]`，不需要计算 mt。这就是生产环境下 CP 开销极低的原因。

---

## 4. 架构与调用链

```
用户代码
  │
  ▼
flash_kda/__init__.py               ← Python 入口
  ├── fwd()                         → C++ kernel（无 CP）
  ├── state_only(calc_mt=True/False)→ C++ kernel
  └── fwd_cp()                      → flash_kda/cp.py
        │
        ▼
flash_kda/cp.py                     ← CP 编排（纯 Python）
  ├── _calc_cp_seqs()               切分序列为 sub-segments
  ├── get_warmup_chunks()           确定 warmup 长度
  ├── correct_initial_states()      串行修正 h0
  ├── → state_only()                调 C++ 计算 ht (+ mt)
  └── → fwd()                       调 C++ 做完整前向
        │
        ▼
flash_kda_C (pybind11)              ← csrc/flash_kda.cpp
  ├── fwd()                         → launch_kernel1 + launch_kernel2
  └── state_only()                  → launch_kernel1_warmup_only
                                      + launch_state_only (ht)
                                      + launch_mt_only (mt, if requested)
        │
        ▼
csrc/smxx/                          ← CUDA Kernels
  ├── fwd_kernel1.cuh              workspace 预计算 (k_decayed, k_restored, g_total, INV)
  ├── fwd_kernel2.cuh              完整前向（输出 + 状态更新）
  └── fwd_state_only.cuh           仅状态递推 (ht/mt)
```

---

## 5. CP 流程详解 (`fwd_cp`)

```python
def fwd_cp(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, ...):
```

### Step 0: 判断是否启用 CP

```python
use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r = _calc_cp_seqs(cu_seqlens, H)
```

基于 SM 数量和序列长度，判断 CP 是否有收益。将长序列切分为 sub-segments。

例：T=65536, H=16 → 切为 64 个 sub-segment，每段 1024 tokens。

### Step 1: 确定 warmup 长度

```python
# 快速路径：lower_bound * 1.4427 * 0.25 * 16 < -10 → 1 chunk 就够
per_chunk_decay_bound = lower_bound * 1.4426950408889634 * 0.25 * CHUNK_SIZE
if per_chunk_decay_bound < -10.0:
    num_warmup = torch.ones(cp_N, ...)    # 每段只需 1 chunk
    fallback_mask = torch.zeros(cp_N, ...) # 不需要 mt
else:
    num_warmup, fallback_mask = get_warmup_chunks(...)
```

- **快速路径触发条件**：`lower_bound < -1.73`（覆盖所有生产场景）
- **不需要考虑 per-head**：因为 `exp(A_log) < 1` 只会把 sigmoid 推向 0.5

#### 0.25 的由来

gate 公式中的 sigmoid 输入为 `exp(A_log) × (g + dt_bias)`。代码中 `get_warmup_chunks` 使用 D=128 维上的均值：

```
sigmoid_input = exp(A_log) × mean_D(g + dt_bias)
```

由中心极限定理，D=128 个独立 bf16 值的均值标准差为：

```
std(mean_D(g)) ≈ 1/√128 ≈ 0.088
std(mean_D(dt_bias)) ≈ 1/√128 ≈ 0.088
合计: std(mean_D(g + dt_bias)) ≈ √(0.088² + 0.088²) ≈ 0.125
```

因此 `mean_D(g + dt_bias)` 的 99.7% 区间约 [-0.37, 0.37]，乘以 `exp(A_log) ≈ 0.9` 后 sigmoid 输入在 [-0.33, 0.33]：

```
sigmoid(-0.33) = 0.42
sigmoid(0)     = 0.50
sigmoid(0.33)  = 0.58
```

**实际 sigmoid 输出几乎恒在 [0.42, 0.58]**。取 0.25 = 0.5/2 作为下界，提供了 ~2x 安全余量。

即使在最极端情况下（sigmoid = 0.25 确实不成立），快速路径真正失败的条件是 sigmoid < 0.087（推导：`-10 / (lb × 1.4427 × 16)` 当 lb=-5），对应 sigmoid 输入 < -2.35。这要求 `mean_D(g + dt_bias) < -2.6`（距均值 >20σ），概率可忽略。

**总结**：0.25 不是严格数学下界，是 "典型值 0.5 的一半" 的工程选择，实际安全余量远超 2x。

### Step 2: State-only kernel

```python
if need_mt:
    ht_buffer, mt_buffer = state_only(..., calc_mt=True)   # C++ 内跑两次 kernel
else:
    ht_buffer = state_only(..., calc_mt=False)             # C++ 内跑一次 kernel
    mt_buffer = None
```

C++ 端实际执行：
1. `launch_kernel1_warmup_only`：只计算 warmup 后缀对应 tile 的 workspace
2. `launch_state_only`：CalcMt=false，从 0 开始递推 → ht
3. `launch_mt_only`（如需要）：CalcMt=true，从 I 开始递推 → mt

### Step 3: 串行修正

```python
cp_h0 = correct_initial_states(initial_state, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c)
```

三种路径：
- **无 mt, 无 h0（最常见）**：`cp_h0[i+1] = ht[i]`，一个 slice 赋值
- **无 mt, 有 h0**：第一段用 h0，后续 `cp_h0[i+1] = ht[i]`
- **有 mt（罕见）**：`cp_h0[i+1] = mt[i] @ cp_h0[i] + ht[i]`，逐段矩阵乘

### Step 4: 带修正 h0 的完整前向

```python
fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
    initial_state=cp_h0, cu_seqlens=cp_cu_seqlens)
```

使用 cp_cu_seqlens（多段）和 cp_h0（每段正确的初始状态），一次完整前向得到最终输出。

---

## 6. CUDA Kernel 详解 (`_flash_kda_state_only`)

### 6.1 Grid 与 Warp 分工

```
Grid: (N_segments, H_heads)     — 每个 CTA 处理一个 segment 的一个 head
Block: 160 threads = 128(MMA) + 32(LOAD)

Warp 0-3: MMA 计算（状态递推）
Warp 4:   TMA 加载（异步 HBM→SMEM）
```

### 6.2 共享内存布局

```cpp
struct SharedStorageStateOnly {
    state_acc[D×D] bf16;           // 状态累积器 (ht 或 mt)
    union {
        InputStorage input[3];     // 3-stage 流水线
        state_fp32_buf[D×D] f32;   // 最终 bf16→fp32 转换用
    };
};

struct InputStorage {
    v[16×128] bf16;        // value 数据
    beta[32] bf16;         // 门控 logit
    k_decayed[16×128] bf16; // kernel1 预计算
    k_restored[16×128] bf16;
    g_total[128] f32;      // per-dim chunk 累积 gate
    INV[16×16] bf16;       // chunk 内逆矩阵
};
```

### 6.3 初始化

```cpp
if constexpr (!CalcMt) {
    state_acc = 0;        // ht: 从零开始
} else {
    state_acc = I;        // mt: 从单位矩阵开始
    // 通过逻辑坐标 s_acc(d, d) = 1.0 设置对角线，自动处理 swizzle
}
```

### 6.4 每个 Chunk 的计算流水线

```
LOAD warp 异步加载 chunk t 的数据 → smem stage
  ↓ (pipeline barrier)
MMA warps 消费该 stage，执行 Phase 1→2/3→6
  ↓ (compute barrier)
释放 stage，LOAD warp 复用该 slot
```

### 6.5 Phase 1: `U = k_decayed @ S_acc`

```
[16, D] @ [D, D] → [16, D]
分 K_BLOCKS=8 次 16×16 GEMM 累积
每个 warp 负责 S_acc 的 2 列（共 4 warp × 2 × 16 = 128 = D）
```

含义：把当前状态通过 k_decayed "读出"。

### 6.6 Phase 2/3: 中间变量 + INV 变换

```cpp
// CalcMt=false (ht):
U = INV @ ((v - U) * β)     // 数据增量 × 门控 × 逆矩阵

// CalcMt=true (mt):
U = INV @ (U * β)           // 只有状态传播项，没有 v（v 和 h0 无关）
```

关键区别：mt 没有 `v -` 是因为 `v` 是纯数据项，不参与 h0 的传播。

### 6.7 Phase 6: 状态更新

```cpp
// 对 S_acc 的每 16 行块 (共 D/16 = 8 块):
R = k_restored^T @ U        // [D,16] @ [16,D/warp] → [D,D/warp]

// CalcMt=false (ht):
S_acc[d] = g_total[d] * S_acc[d] + R[d]     // 衰减 + 新贡献

// CalcMt=true (mt):
M_acc[d] = g_total[d] * M_acc[d] - R[d]     // A = diag(g) - kr^T·INV·β·kd
```

减号来源：`A = diag(g) - kr^T·INV·β·kd` 中的减号。

### 6.8 精度处理

```cpp
// Phase 6 的 mixed-precision:
BF16(bf16_to_f32(S_old) * g0_f32 + R_f32)
//    ↑ 提升到fp32       ↑ fp32乘   ↑ fp32加    ↑ 截断回bf16存储
```

计算在 fp32，存储在 bf16。最终输出时 `smem_cvt_bf16_to_fp32` 转换为 fp32 通过 TMA 写回 gmem。

---

## 7. `get_warmup_chunks` 算法

### 目标

对每个 CP sub-segment，确定从末尾开始需要处理多少个 chunk 才能让 h0 的影响衰减到阈值以下。

### Gate 公式

```
per_token_decay = lower_bound × log2(e) × sigmoid(exp(A_log) × (g + dt_bias))
per_chunk_decay = per_token_decay × 16   (累积 16 tokens)
```

### 快速路径（向量化）

```python
# 只看每段末尾 1 个 token，批量判断所有段
g_at_end = g[last_token_of_each_seg].mean(dim=-1)     # [N, H]
per_seg_decay = gate_scale * sigmoid(...) * chunk_size  # [N, H]
converges_in_1 = per_seg_decay.max(dim=-1).values < threshold  # [N]
```

如果最慢的 head 都已收敛 → 1 chunk 足够。

### 慢路径（逐 chunk 扫描，仅对未收敛的段）

```python
for c in range(num_chunks):           # 从末尾第 0 个 chunk 往前
    g_cumsum += g_activated * 16      # 累积 decay（负值）
    if g_cumsum.max() < threshold:    # 所有 head 都超过阈值
        warmup = c + 1
        break
# 扫完没收敛 → fallback_mask = True，需要 mt
```

---

## 8. CUDA `get_warmup_chunks` Kernel

### 动机

Python 版 `get_warmup_chunks` 在 weak decay (lb=-0.5) 下需要扫描大量 chunk，耗时 ~12-16ms（成为整条 CP 流水线的瓶颈）。CUDA 版将其降到恒定 ~0.03ms。

### 架构

```
文件: csrc/warmup_chunks.cu
Grid: (N_segments,)     — 每个 block 处理一个 segment
Block: (H_heads,)       — 每个 thread 处理一个 head
共享内存: float[H]      — 用于跨 head tree reduction 求 max
```

### 算法

```cuda
// 每个 thread (= 一个 head) 从 segment 末尾往前扫描
for (int c = 0; c < num_chunks && !found; c++) {
    // 1. 计算 g[chunk_end_token, h, :] 的 mean
    float g_mean = mean(g_ptr[token * H * D + h * D : +D]);

    // 2. Gate activation
    float x = exp(A_log[h]) * (g_mean + dt_bias_mean[h]);
    float decay = gate_scale * sigmoid(x) * chunk_size;
    g_cumsum += decay;  // 累积（负值）

    // 3. 跨 head 求 max (tree reduction in shared memory)
    smem[h] = g_cumsum;
    __syncthreads();
    for (stride = blockDim.x/2; stride > 0; stride >>= 1)
        smem[h] = fmax(smem[h], smem[h + stride]);

    // 4. 检查收敛: max(所有 head 累积) < threshold
    if (smem[0] < threshold) {
        result_warmup = c + 1;
        found = true;
    }
}
```

### 性能 (H20 GPU)

| 配置 | Python | CUDA | 加速 |
|------|--------|------|------|
| T=8K, H=16, N=16, lb=-1 | 1.2ms | 0.03ms | 39x |
| T=64K, H=16, N=64, lb=-0.5 | 16.1ms | 0.03ms | 623x |
| T=64K, H=32, N=64, lb=-0.5 | 17.8ms | 0.03ms | 694x |

### 集成

```python
# flash_kda/cp.py 中的调用:
from flash_kda_C import get_warmup_chunks as _get_warmup_chunks_cuda

def get_warmup_chunks_cuda(g, A_log, dt_bias, lower_bound, cu_seqlens, chunk_size):
    N = cu_seqlens.numel() - 1
    num_warmup = torch.empty(N, dtype=torch.int32, device=g.device)
    fallback = torch.empty(N, dtype=torch.bool, device=g.device)
    threshold = -10.0
    _get_warmup_chunks_cuda(g, A_log, dt_bias, lower_bound, cu_seqlens,
                            chunk_size, threshold, num_warmup, fallback)
    return num_warmup, fallback
```

---

## 9. `correct_initial_states` 算法

### 无 mt（生产常态，~0.1ms）

```python
# gate decay 足够强，mt ≈ 0，所以 h0[i+1] ≈ ht[i]
cp_h0[seg_start] = raw_h0[raw_idx]  或 0
cp_h0[seg_start+1 : seg_end] = ht_buffer[seg_start : seg_end-1]  # slice 赋值
```

### 有 mt（weak decay，罕见）

```python
for i in range(seg_start, seg_end - 1):
    if fallback_mask[i]:
        h = einsum('hdk,hkv->hdv', mt[i], h) + ht[i]   # H 个 [D,D]@[D,D] 矩阵乘
    else:
        h = ht[i]
    cp_h0[i + 1] = h
```

`einsum('hdk,hkv->hdv')` = 每个 head 独立做 128×128 矩阵乘。

---

## 10. Kernel1 Warmup-Only 优化

### 问题

kernel1 为每个 tile 预计算 workspace（k_decayed, k_restored, g_total, INV）。state_only 只需要 warmup 后缀的 tile。

### 解决

```cpp
template <..., bool WarmupOnly = false>
__global__ void _flash_kda_fwd_prepare(..., int const* num_warmup_chunks_ptr) {
    ...
    if constexpr (WarmupOnly) {
        int warmup = num_warmup_chunks_ptr[seq_idx];
        int t_start = t_tiles_this_seq - min(warmup, t_tiles_this_seq);
        if (local_t < t_start) return;  // 非 warmup tile 直接退出
    }
    ...
}
```

Grid 不变（所有 tile 都启动 CTA），但非 warmup 的 CTA 只执行一个 branch 后 return，开销可忽略。Workspace 索引布局不变，state_only 照常读取。

---

## 11. 使用方式

```python
from flash_kda import fwd_cp

# 自动 CP（推荐，对用户透明）
fwd_cp(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, auto_cp=True)

# 带初始/终态
fwd_cp(..., initial_state=h0, final_state=hn, auto_cp=True)

# varlen 模式
cu_seqlens = torch.tensor([0, 2048, 6144, 8192], dtype=torch.int64, device="cuda")
fwd_cp(..., cu_seqlens=cu_seqlens, auto_cp=True)

# 禁用 CP（等价于 fwd()）
fwd_cp(..., auto_cp=False)
```

### CP 自动生效条件

- `B = 1`（batch 维度为 1）
- 序列够长，使得 CP 切分后 SM 利用率提升
- `_calc_cp_seqs` 内部判断：`Be * H <= SM_COUNT / 4`

---

## 12. 文件清单

| 文件 | 修改内容 |
|------|----------|
| `csrc/smxx/fwd_kernel1.cuh` | `WarmupOnly` 模板参数，非 warmup tile early-exit |
| `csrc/smxx/fwd_state_only.cuh` | `CalcMt` 模板参数，mt 用 Identity 初始化 + 减号更新 |
| `csrc/smxx/fwd_launch.cu` | `launch_kernel1_warmup_only`、`launch_mt_only` |
| `csrc/warmup_chunks.cu` | CUDA `get_warmup_chunks` kernel (跨 head tree reduction) |
| `csrc/fwd.h` | 新函数声明 |
| `csrc/flash_kda.cpp` | `state_only` + `get_warmup_chunks` pybind11 接口 |
| `flash_kda/__init__.py` | `state_only` 返回 (ht, mt) |
| `flash_kda/cp.py` | CP 编排：warmup + correct + single fwd；CUDA scan 集成 |
| `tests/test_warmup_cuda.py` | CUDA vs Python 正确性测试 + 性能基准 |

---

## 13. 关键设计决策

| 决策 | 理由 |
|------|------|
| Warmup 1 chunk 快速路径 | 生产参数下 `lower_bound=-5`，decay 极强，1 chunk 必然收敛 |
| kernel1 early-exit 而非 compact grid | 简单正确，workspace 索引不变，多余 CTA 开销可忽略 |
| mt 单独 launch 而非合并到 ht kernel | 简化实现；生产中 mt 几乎不触发，ROI 低 |
| correct_initial_states 用 Python | 段数少（~64），串行开销 0.1ms，不值得写 CUDA kernel |
| state_acc 用 bf16 | 和 kernel2 共用 SharedStorage；mt 只在 weak decay 时触发，此时值接近 1 不会下溢 |
| get_warmup_chunks 用 CUDA | Python 版在 weak decay 下 12-16ms，成为流水线瓶颈；CUDA 恒定 0.03ms |
| Tree reduction 而非 atomicMax | 负浮点数的 atomicMax 需要 CAS 循环，tree reduction 更简洁高效 |
