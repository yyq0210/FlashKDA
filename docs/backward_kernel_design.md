# FlashKDA Backward CUDA Kernel 实现解析

## 目录

1. [总体架构](#1-总体架构)
2. [Forward 回顾与 Backward 数据依赖](#2-forward-回顾与-backward-数据依赖)
3. [Workspace 设计](#3-workspace-设计)
4. [K2_bwd: 反向递推 Kernel](#4-k2_bwd-反向递推-kernel)
5. [K1_bwd: 梯度还原 Kernel](#5-k1_bwd-梯度还原-kernel)
6. [Host Launch 与内存管理](#6-host-launch-与内存管理)
7. [Python 层集成](#7-python-层集成)
8. [精度工程](#8-精度工程)

---

## 1. 总体架构

### 1.1 Forward 与 Backward 的镜像关系

FlashKDA 的 forward 采用 2-kernel 设计:

```
Forward:  K1_fwd (prepare)  →  K2_fwd (recurrence, 从前到后)
Backward: K2_bwd (recurrence, 从后到前)  →  K1_bwd (prepare backward)
```

Backward 完美镜像 forward：

| 维度 | K1_fwd | K2_fwd | K2_bwd | K1_bwd |
|------|--------|--------|--------|--------|
| Grid | (total_tiles, H) | (N, H) | (N, H) | (total_tiles, H) |
| 职责 | q,k,g → 各种中间量 | 前向递推 S | 反向递推 dS | 从 d-中间量 → 最终梯度 |
| 迭代方向 | 独立 per-tile | 前→后 | 后→前 | 独立 per-tile |
| Threads | 256 | 192 (warp specialization) | 256 (纯计算) | 256 |

### 1.2 输入输出总览

**Backward 接受的输入:**
- 原始前向输入: `q, k, v, g, beta, A_log, dt_bias` (bf16/fp32)
- Forward workspace: `kd, qd, kr, ki, gT, INV, Mqk, gc` (bf16+fp32)
- Forward all_states: 每个 chunk 起始处的状态 `S_in[K,V]` (bf16)
- 输出梯度: `do [B,T,H,D]`, `dfinal_state [N,H,D,D]`

**Backward 产出的梯度:**
- `dq, dk, dv, dg` : `[B,T,H,D]` bf16
- `dbeta` : `[B,T,H]` bf16 (logit 空间)
- `dA_log` : `[H]` fp32 (atomicAdd 累加)
- `ddt_bias` : `[H,D]` fp32 (atomicAdd 累加)
- `d_initial_state` : `[N,H,D,D]` bf16

---

## 2. Forward 回顾与 Backward 数据依赖

### 2.1 Forward 数学 (per head, per chunk)

```
qn = l2norm(q);  kn = l2norm(k)                        # 特征归一化
g_nat = lb * σ(exp(A_log) * (g_raw + dt_bias))         # 门控 (lb = lower_bound)
gc = cumsum(g_nat, dim=time);  gT = gc[-1]              # 门控累积和

kd = kn * exp(gc)       # decayed key
qd = qn * exp(gc) * s   # decayed query (s = scale)
ki = kn * exp(-gc)      # inverse key
kr = kn * exp(gT - gc)  # restored key

L    = tril(kd @ ki^T, -1) * σ(β)[:, None]             # 严格下三角
Mqk  = tril(qd @ ki^T)                                  # 含对角线
INV  = (I + L)^{-1}                                      # Neumann 级数近似
vcorr = (v - kd @ S) * σ(β)[:, None]                    # 校正值
U    = INV @ vcorr                                       # 求解后的校正
o    = qd @ S + Mqk @ U                                  # 输出
S_new = S * exp(gT)[:, None] + kr^T @ U                 # 状态递推
```

### 2.2 反向传播的计算图

将 forward 展开成计算图，backward 沿图反向传播。关键路径:

```
         ┌─── o = qd @ S + Mqk @ U ───────────── do (给定)
         │
    ┌────┴────┐
    │ qd @ S  │  Mqk @ U
    │  (跨chunk) │  (chunk内)
    └────┬────┘────┬────┘
         │         │
    dqd, dS    dMqk, dU
         │         │
         │    ┌────┴────┐
         │    │ INV@vcorr│        S_new = S*exp(gT) + kr^T@U
         │    └────┬────┘            │
         │    dvcorr, dL        dkr, dU (额外贡献), dS (递推)
         │         │
         │    vcorr = (v-kd@S)*β
         │         │
         │    dv, dkd, dbeta, dS (额外贡献)
         │
    ┌────┴────────────────┐
    K1_bwd: dkd,dqd,dki,dkr → dq,dk,dg,dA_log,ddt_bias
```

### 2.3 Forward 保存的数据

为了避免在 backward 中重算，forward 将以下数据保存到 workspace:

| 数据 | 形状 per tile | dtype | 保存位置 |
|------|-------------|-------|---------|
| kd (k_decayed) | [C, D] | bf16 + fp32 | workspace |
| qd (q_decayed) | [C, D] | bf16 + fp32 | workspace |
| kr (k_restored)| [C, D] | bf16 + fp32 | workspace |
| ki (k_inv) | [C, D] | bf16 + fp32 | workspace |
| gT (exp2后) | [D] | fp32 | workspace |
| gc (cumsum) | [C, D] | fp32 | workspace |
| INV = (I+L)^{-1} | [C, C] | bf16 | workspace |
| Mqk | [C, C] | bf16 | workspace |
| S_in (每chunk起始状态) | [D, D] | bf16 | all_states |

其中 C=CHUNK=16, D=128。

---

## 3. Workspace 设计

### 3.1 Forward Workspace 布局

定义在 `utils.cuh` 的 `WorkspaceSizes<CHUNK, D>`:

```
Forward workspace (per tile):
┌─────────────────────────────────────────────┐
│ bf16 区域 (forward K2 通过 TMA 读写)         │
│  kd[C*D*2B=4096]  qd[4096]  kr[4096]       │
│  ki[4096]  gT[D*4B=512]                    │
│  INV[C*C*2B=512]  Mqk[512]                 │
│  gc[C*D*4B=8192]                           │
├─────────────────────────────────────────────┤
│ fp32 区域 (backward 精度保障)                │
│  kd_fp32[C*D*4B=8192]  qd_fp32[8192]      │
│  ki_fp32[8192]  kr_fp32[8192]              │
└─────────────────────────────────────────────┘
```

总计 per tile: 62,208 bytes。全局: `H × total_tiles × 62,208`。

### 3.2 Backward 临时 Workspace

K2_bwd 产出、K1_bwd 消费的中间梯度存于临时 workspace（通过 `cudaMallocAsync` 分配）:

| 名称 | 形状 per tile | dtype | 大小 | 用途 |
|------|-------------|-------|------|------|
| dkd | [C, D] | fp32 | 8192B | decayed key 梯度 |
| dqd | [C, D] | fp32 | 8192B | decayed query 梯度 |
| dki | [C, D] | fp32 | 8192B | inverse key 梯度 |
| dkr | [C, D] | fp32 | 8192B | restored key 梯度 |
| dv | [C, D] | bf16 | 4096B | value 梯度 (精度要求低) |
| dgT | [D] | fp32 | 512B | gT 总梯度 |
| dbeta | [C] | fp32 | 64B | beta chunk 内梯度 |

全局: `H × total_tiles × 37,568B`。运行后异步释放。

---

## 4. K2_bwd: 反向递推 Kernel

**文件**: `csrc/smxx/bwd_kernel2.cuh`

### 4.1 Launch 配置

```
Grid:  (N, H)          — 每个 block 处理一个序列的一个 head
Block: 256 threads      — 无 warp specialization，纯标量计算
Smem:  ~112 KB          — dS[D*D] + scratch 缓冲区
```

### 4.2 Shared Memory 布局

```
shared_mem (动态分配):
┌─────────────────────────────────────────────────────────────┐
│ dS_smem[D*D = 16384 floats = 65536B]                       │  ← 状态梯度累加器
├─────────────────────────────────────────────────────────────┤
│ scratch_f32:                                                │
│  gT[D=128]        → exp2(gT) 值                            │
│  beta[C=16]       → σ(beta) 激活值                          │
│  inv_smem[C*C=256]→ (I+L)^{-1}, 全程保持不被覆写            │
│  mqk_smem[C*C=256]→ Mqk, 后被 dMqk 覆写                    │
│  vcorr[C*D=2048]  → (v-kd@S)*β, 后被 dqd_local 覆写        │
│  U[C*D=2048]      → INV @ vcorr                            │
│  dU[C*D=2048]     → Mqk^T @ do + kr @ dS                   │
│  dkr[C*D=2048]    → dS @ U 的结果                           │
│  dvcorr[C*D=2048] → INV^T @ dU, 后被覆写                    │
│  dL[C*C=256]      → 独立区域，不与 inv_smem 混用             │
│                                                             │
│  总计: 128+16+256*3+2048*5 = 11152 floats = 44608B          │
└─────────────────────────────────────────────────────────────┘
总 smem: 65536 + 44608 = 110144B ≈ 108 KB (SM90 限制 227 KB)
```

**关键设计**: `dL_smem` 使用独立的内存区域（而非复用 `inv_smem`），保证 INV 在整个 chunk 迭代中始终可读。这避免了后续 dvcorr 重算时需要从 bf16 gmem 重新读取 INV 导致的精度损失。

### 4.3 主循环结构

```python
# 伪代码 — 从最后一个 chunk 倒序迭代到第一个 chunk
for t = t_tiles-1 downto 0:
    加载 workspace 数据: kd, qd, kr, ki, gT, INV, Mqk, v, beta, S_in, do

    # Step 1: 重算 vcorr 和 U (forward 未保存)
    kd_S = kd @ S_in                       # [C,D] = [C,D]×[D,D], 262K FMAs
    vcorr = (v - kd_S) * β                 # [C,D]
    U = INV @ vcorr                        # [C,D] = [C,C]×[C,D]

    # Step 2: 梯度计算 — 输出层
    dU  = Mqk^T @ do                       # [C,D]: Mqk 路径贡献
    dU += kr @ dS                          # [C,D]: 状态递推路径贡献
    dkr = dS @ U^T → 逐元素 sum_d         # [C,D]

    # Step 3: 三角求解伴随
    dvcorr = INV^T @ dU                    # [C,D]

    # Step 3b: 立即计算 dL (趁 dvcorr 未被覆写)
    dL[i][j] = -sum_d dvcorr[i][d] * U[j][d],  i > j   # [C,C] 严格下三角

    # Step 4: vcorr 反向
    dv     = dvcorr * β                    # → 写入 bwd workspace
    dbeta  = sum_d dvcorr * vcorr / β      # per-row 归约

    # Step 5-6: Mqk 反向
    dMqk = do @ U^T                        # [C,C] 下三角
    dqd = do @ S_in^T + tril(dMqk) @ ki   # 跨chunk + chunk内
    dki = tril(dMqk)^T @ qd               # → 写入 bwd workspace

    # Step 7: L 反向 (使用 step 3b 计算的 dL)
    dki += tril(dL * β)^T @ kd            # L backward 对 ki 的贡献
    dkd = -β * dvcorr_recomp @ S_in^T     # vcorr backward 对 kd 的贡献
    dkd += tril(dL * β) @ ki              # L backward 对 kd 的贡献
    dbeta += sum_j dL * F                  # L backward 对 beta 的贡献

    # → 全部写入 bwd workspace (fp32)

    # Step 8a: dgT 计算 (必须在 dS 更新之前)
    dgT[k] = gT[k] * sum_d dS[k][d] * S_in[k][d]     # 状态递推贡献
    dgT[k] += sum_c dkr[c][k] * kr[c][k]               # kr 依赖贡献

    # Step 8b: dS 向前传播 (为上一个 chunk)
    dS[k][d] = gT[k] * dS[k][d]                        # 衰减
             + sum_c qd[c][k] * do[c][d]                # 跨chunk: qd@S
             - sum_c kd[c][k] * β[c] * dvcorr_recomp[c][d]  # vcorr: -kd@S

# 存储最终 dS 为 d_initial_state
```

### 4.4 各步骤的计算复杂度

| 步骤 | 操作 | FMAs / tile | 说明 |
|------|------|-------------|------|
| Step 1: kd@S | GEMM [C,D]×[D,D] | C×D×D = 262,144 | 最贵的步骤 |
| Step 1: INV@vcorr | GEMM [C,C]×[C,D] | C×C×D = 32,768 | |
| Step 2: Mqk^T@do | GEMM [C,C]×[C,D] | C×C×D = 32,768 | |
| Step 2: kr@dS | GEMM [C,D]×[D,D] | C×D×D = 262,144 | 第二贵 |
| Step 3: INV^T@dU | GEMM [C,C]×[C,D] | C×C×D = 32,768 | |
| Step 5: dMqk, dqd | O(C²D + C²D) | ~65,536 | 带三角 mask |
| Step 7: dkd 重算 | 含 dvcorr 重算 O(C²D²) | ~524,288 | 三重循环，最慢 |
| Step 8b: dS 更新 | O(D²×C) + dvcorr 重算 | ~8M | 占主要时间 |

### 4.5 dvcorr 重算策略

`dvcorr[c][d] = sum_j INV^T[c][j] * dU[j][d]` 在 step 3 中计算一次后被写入 `dvcorr_smem`。
但后续 `dvcorr_smem` 会被 `dqd_local` 和 `dki_local` 覆写（smem 空间复用）。

在 step 7 (dkd) 和 step 8b (dS 更新) 中需要再次用到 dvcorr，因此需要**按需重算**:

```cuda
// 从 smem 中的 inv_smem 和 dU_smem 实时重算 dvcorr[c][d]
float dvcorr_cd = 0.0f;
for (int j = 0; j < CHUNK; ++j) {
    dvcorr_cd += inv_smem[j * CHUNK + c] * dU_smem[j * D + d];
}
```

**精度关键**: `inv_smem` 在 chunk 迭代开始时从 bf16 gmem 加载到 fp32 smem（第 174 行），
之后在整个 chunk 内保持不变。`dL_smem` 使用独立区域（而非覆写 `inv_smem`），
使得所有 dvcorr 重算都从 fp32 smem INV 读取，而非从 bf16 gmem 重新读取。

---

## 5. K1_bwd: 梯度还原 Kernel

**文件**: `csrc/smxx/bwd_kernel1.cuh`

### 5.1 Launch 配置

```
Grid:  (total_tiles, H)   — 每个 CTA 独立处理一个 chunk 的一个 head
Block: 256 threads          — 线程布局: 16 rows × 16 threads/row, 每线程 8 元素
Smem:  8 KB                 — dgc[C*D] fp32
```

### 5.2 职责

K1_bwd 从 K2_bwd 的输出 (dkd, dqd, dki, dkr, dgT, dv, dbeta) 出发，
逆转 forward K1 的变换链，还原到原始参数空间:

```
K2_bwd outputs → K1_bwd → dq, dk, dv, dg, dbeta, dA_log, ddt_bias
```

### 5.3 Phase 1+2: 融合 dgc + L2 norm backward

这是 K1_bwd 最核心的计算。将门控梯度 dgc 和 L2 归一化反向两个步骤融合在同一个 pass 中。

#### 5.3.1 dgc 的数值稳定公式

Forward K1 通过指数衰减将 qn, kn 变换为 qd, kd, ki, kr:

```
kd = kn * exp(gc)       →  dkd/dgc =  kd
qd = qn * exp(gc) * s   →  dqd/dgc =  qd
ki = kn * exp(-gc)      →  dki/dgc = -ki
kr = kn * exp(gT-gc)    →  dkr/dgc = -kr
```

朴素公式: `dgc = dkd*kd + dqd*qd - dki*ki - dkr*kr`

**问题**: 当 gc ≈ -30 (常见工况) 时，kd = kn*exp(-30) ≈ 1e-13 而 ki = kn*exp(30) ≈ 1e13。
但 dkd 和 dki 也会被 exp(gc) 调制，导致乘积其实是 O(1) 量级。朴素公式中的加减法在
bf16 精度下可能发生灾难性抵消。

**数值稳定重写**:

```
dgc = kn * (dkd*exp(gc) - dki*exp(-gc) - dkr*exp(gT-gc))
    + qn * scale * dqd * exp(gc)
```

每一项都是 O(1) 量级，减法条件良好。

#### 5.3.2 L2 norm backward

L2 归一化 `xn = x / ||x||` 的反向:

```
dx = (dxn - xn * dot(dxn, xn)) / ||x||
```

其中:
- `dqn = dqd * exp(gc) * scale` (从 chain rule)
- `dkn = dkd * exp(gc) + dki * exp(-gc) + dkr * exp(gT-gc)` (所有路径汇总)

#### 5.3.3 实现细节

256 个线程按 16 行 × 16 列布局，每个线程处理 8 个连续元素 (`ELEMS_PER_THREAD=8`)。
利用 warp shuffle (`__shfl_xor_sync`) 在同一行的 16 个线程间做归约（计算 L2 范数和内积），
与 forward K1 的线程布局完全匹配:

```cuda
// 16 线程内的 butterfly 归约
for (int delta = 8; delta >= 1; delta >>= 1) {
    q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, delta);
    k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, delta);
}
float q_inv_norm = rsqrtf(q_sq + 1e-6f);
```

dgc 结果写入 smem，供 Phase 4 消费。

### 5.4 Phase 3: dv 直通

直接从 K2_bwd workspace 拷贝 dv 到输出（K2_bwd 已计算好）:

```cuda
dv_out[idx] = dv_tile[idx];  // bf16 → bf16, 无计算
```

### 5.5 Phase 4: Reverse Cumsum + Gate Backward

dgc 是 gate cumsum 的梯度。为了得到原始门控值 g_nat 的梯度，需要做 **反向累积和** (reverse cumsum):

```
dg_nat[t] = sum_{s>=t} dgc[s]  +  dgT   (对每个特征维度)
```

dgT 是从 K2_bwd 传来的 gT 总梯度，对 chunk 内每个时间步都有贡献。

然后通过 sigmoid 的链式法则:

```
g_nat = lb * σ(a * (g_raw + dt_bias)),   a = exp(A_log)
dz = dg_nat * lb * σ(z) * (1-σ(z)) * ln2       # sigmoid backward, 含 log2→ln 转换
dg_raw = dz * a
ddt_bias += dz * a                               # atomicAdd 跨 batch 累加
dA_log += dz * (g_raw + dt_bias) * a             # atomicAdd, 含 exp(A_log) 的梯度
```

**实现**: 每个线程处理一列 (一个 D 维元素)，按行从 CHUNK-1 到 0 反向累加:

```cuda
for (int row = CHUNK - 1; row >= 0; --row) {
    rev_sum += dgc_smem[row * D + col];   // reverse cumsum
    float dg_nat = rev_sum + dgT_val;     // 加上 gT 贡献
    // ... sigmoid backward + atomicAdd
}
```

### 5.6 Phase 5: dbeta Logit 空间转换

K2_bwd 输出的 dbeta 是 σ(β) 激活空间的梯度。需要转换到 logit 空间:

```
dbeta_logit = dbeta_activated * σ(β) * (1 - σ(β))
```

只有 CHUNK=16 个线程参与。

---

## 6. Host Launch 与内存管理

**文件**: `csrc/smxx/bwd_launch.cu`

### 6.1 Workspace 指针解析

`launch_bwd` 接受 forward workspace 的 void* 指针，按 `WorkspaceSizes` 定义的偏移量
解析出各数据区的指针:

```cpp
// bf16 workspace (forward K2 TMA 数据)
BF16 const* ws_kd  = (BF16*)(ws + 0);
BF16 const* ws_qd  = (BF16*)(ws + n_ht * WS::kKDecayed);
// ... (INV, Mqk 等)

// fp32 workspace (backward 使用)
float const* ws_kd_fp32 = (float*)(ws + fp32_base);
float const* ws_qd_fp32 = (float*)(ws + fp32_base + n_ht * WS::kKDecayedFP32);
// ...
```

K2_bwd 和 K1_bwd 读取 **fp32** 版本的 kd/qd/ki/kr，而非 bf16 版本。

### 6.2 临时 Workspace 分配

```cpp
int64_t bwd_ws_per_tile =
    4LL * CHUNK * D * sizeof(float) +   // dkd + dqd + dki + dkr (fp32)
    CHUNK * D * sizeof(BF16) +          // dv (bf16)
    D * sizeof(float) +                 // dgT
    CHUNK * sizeof(float);              // dbeta

void* bwd_ws_raw;
cudaMallocAsync(&bwd_ws_raw, n_ht * bwd_ws_per_tile, stream);
// ... K2_bwd 写入 → K1_bwd 读取 ...
cudaFreeAsync(bwd_ws_raw, stream);    // 用完释放
```

### 6.3 Kernel Launch 顺序

```
1. K2_bwd<<<(N, H), 256, smem_k2, stream>>>  — 反向递推
   (K2_bwd 完成后 bwd workspace 就绪)
2. K1_bwd<<<(total_tiles, H), 256, smem_k1, stream>>>  — 梯度还原
   (隐式同步: 同一 stream 保证顺序执行)
3. cudaFreeAsync  — 释放临时 workspace
```

---

## 7. Python 层集成

### 7.1 autograd.py

`FlashKDAFunction` 是 `torch.autograd.Function` 子类:

**forward**:
1. 调用 `flash_kda.fwd(...)` 执行 CUDA forward
2. `ctx.save_for_backward(q, k, v, g, beta, A_log, dt_bias, initial_state, workspace, all_states)`
3. workspace 和 all_states 在整个训练 step 内存活

**backward**:
1. 从 `ctx.saved_tensors` 取出保存的数据
2. 分配梯度输出 tensor
3. 调用 `flash_kda.bwd(...)` 执行 CUDA backward
4. 返回所有 12 个参数的梯度 (无梯度的返回 None)

### 7.2 flash_kda.cpp (pybind11 绑定)

`bwd()` 函数的主要工作:

1. **Layout 转换**: PyTorch `[B,T,H,D]` → kernel 需要的 `[H, T_total, D]`
   ```cpp
   auto do_t = do_3d.permute({1, 0, 2}).contiguous();  // [T,H,D] → [H,T,D]
   ```

2. **dfinal_state 处理**: `[N,H,D,D]` → `[N*H,D,D]` bf16
   ```cpp
   ds_init_buf = dfs.to(kBFloat16).reshape({N*H, D, D}).contiguous();
   ```

3. **调用 `launch_bwd`**: 传递所有指针和维度参数

4. **结果转置**: `[H,T,D]` → `[T,H,D]` → `[B,T,H,D]`
   ```cpp
   auto dq_thd = dq_htd.permute({1, 0, 2}).contiguous();
   dq.copy_(dq_thd.reshape_as(dq));
   ```

---

## 8. 精度工程

### 8.1 精度挑战

FlashKDA backward 面临的核心精度问题来自**指数衰减**:
- `gc` 的典型值约 -30（lb=-5, sigmoid≈1, 16 步累积）
- `exp(gc)` ≈ 1e-13, `exp(-gc)` ≈ 1e13
- kd, ki 之间差距可达 26 个数量级
- bf16 只有约 3 位有效数字精度

### 8.2 采用的精度保障措施

| 措施 | 适用位置 | 效果 |
|------|---------|------|
| **fp32 workspace** | Forward K1 同时存 bf16 和 fp32 的 kd/qd/ki/kr | K2_bwd 读 fp32 避免 bf16 截断 |
| **fp32 中间梯度** | K2_bwd → K1_bwd 传递 dkd/dqd/dki/dkr 用 fp32 | 保持 dgc 计算精度 |
| **数值稳定 dgc** | K1_bwd Phase 1 | 避免大小量级相消 |
| **INV smem 常驻** | K2_bwd: dL 独立存储，不覆写 inv_smem | dvcorr 重算读 fp32 smem 而非 bf16 gmem |
| **dL 提前计算** | K2_bwd Step 3b: 在 dvcorr 被覆写前算好 dL | 修复 smem 别名 bug，dL 基于正确的 dvcorr |

### 8.3 精度结果

对标 FLA (flash-linear-attention) 参考实现，最终精度:

| 梯度 | g_scale=0.5 | g_scale=4.0 | 阈值 |
|------|-------------|-------------|------|
| dq | 0.33% | 0.34% | < 2% |
| dk | 0.32% | 0.33% | < 2% |
| dv | 0.53% | 0.53% | < 2% |
| dg | 0.62% | 0.50% | < 2% |
| dbeta | 0.48% | 0.52% | < 2% |
| dA_log | 0.26% | 0.33% | < 2% |
| ddt_bias | 1.76% | 0.79% | < 2% |
| dh0 | 0.22% | 0.24% | < 2% |

### 8.4 精度调试教训

开发过程中发现的最关键 bug: **共享内存别名**。

原始代码中，`dL_smem = inv_smem`（line 441）将 INV 的 smem 区域覆写为 dL。
但在计算 dL 时引用的 `dvcorr_smem` 实际上已经被更早的 `dqd_local` / `dki_local`
覆写（它们共享同一块 smem 地址）。因此 dL 的计算读取了错误的数据。

修复:
1. **dL 提前计算**: 在 Step 3b（dvcorr 刚计算完、尚未被覆写时）立即计算 dL
2. **dL 独立存储**: `dL_smem = dvcorr_smem + CD`，使用全新的 smem 区域
3. 额外 smem 开销: 仅 256 floats = 1 KB

这个修复将 dg 的误差从 6.5% 降至 0.5%。

---

## 附录 A: 文件结构

```
csrc/
├── bwd.h                    # launch_bwd 声明
├── fwd.h                    # launch_fwd 声明
├── flash_kda.cpp            # pybind11 绑定 (fwd + bwd)
└── smxx/
    ├── utils.cuh            # WorkspaceSizes, bf16_to_f32, sigmoid_tanh_approx 等
    ├── fwd_kernel1.cuh      # Forward K1: L2 norm, gate, decay, L/INV/Mqk
    ├── fwd_launch.cu         # Forward K2: 前向递推 + launch
    ├── bwd_kernel2.cuh      # Backward K2: 反向递推 (本文 §4)
    ├── bwd_kernel1.cuh      # Backward K1: 梯度还原 (本文 §5)
    └── bwd_launch.cu        # Backward launch + workspace 管理 (本文 §6)

flash_kda/
├── __init__.py              # fwd/bwd Python 包装
└── autograd.py              # FlashKDAFunction (autograd 集成)
```

## 附录 B: 数学推导索引

| 梯度 | 来源等式 | K2_bwd 步骤 | K1_bwd Phase |
|------|---------|-------------|-------------|
| dqd | o = qd@S + Mqk@U | Step 5 (cross + intra) | — |
| dkd | vcorr = (v-kd@S)β, L = tril(kd@ki^T,-1)β | Step 7 (vcorr + L) | — |
| dki | Mqk = tril(qd@ki^T), L = tril(kd@ki^T,-1)β | Step 6 + 7 | — |
| dkr | S_new = S*exp(gT) + kr^T@U | Step 2 | — |
| dv | vcorr = (v-kd@S)β | Step 4 | Phase 3 (copy) |
| dvcorr | U = INV@vcorr | Step 3 | — |
| dL | d/dL of (I+L)^{-1} | Step 3b | — |
| dMqk | o += Mqk@U | Step 5 (implicit) | — |
| dgT | S_new = S*exp(gT) + kr^T@U | Step 8a | — |
| dbeta | vcorr = (v-kd@S)β, L = ...β | Step 4 + 7 | Phase 5 (logit) |
| dgc | kd/qd/ki/kr = f(kn,gc) | — | Phase 1 |
| dq, dk | qn = l2norm(q), kn = l2norm(k) | — | Phase 1+2 |
| dg | g_nat = lb*σ(a*(g_raw+dt_bias)) | — | Phase 4 |
| dA_log | g_nat = lb*σ(exp(A_log)*z) | — | Phase 4 (atomicAdd) |
| ddt_bias | g_nat = lb*σ(a*(g_raw+dt_bias)) | — | Phase 4 (atomicAdd) |
| dS | S_new = S*exp(gT) + kr^T@U, o = qd@S, vcorr = (v-kd@S)β | Step 8b | — |
