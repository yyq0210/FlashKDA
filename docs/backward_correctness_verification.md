# FlashKDA Backward 正确性验证与已知限制

本文档记录 FlashKDA backward(反向传播)CUDA kernel 的正确性验证方法、
结论,以及已知的数值限制。目标:回答"这个 backward 能否用于生产训练环境"。

最后更新:2026-06-25

---

## 1. 结论速览(TL;DR)

| 模式 | 状态 | 依据 |
|------|------|------|
| **Batched 反向** | ✅ 生产可用,正确 | L0/L1/FLA 三层验证全部通过 |
| **Varlen 反向** | ✅ 修复并验证,与 batched 逐位一致 | 见 §4 的 bug 修复 |
| `lower_bound ≤ -6` | ⚠️ 已知限制(见 §5) | fp32 门控溢出,超出官方推荐区间 |

**支持的门控范围:`lower_bound ≥ -5`**,与上游 FLA 官方推荐值一致。

---

## 2. 三层验证体系

为了在**不依赖 FLA** 的前提下获得可信的梯度金标准(gold),并在有 FLA 时做交叉
校验,验证分三层:

### L0 — fp64 参考实现自检(FLA-independent gold)

对纯 torch 实现的前向 `_kda_chunk_torch_fwd`(`flash_kda/autograd.py`)做
`torch.autograd.gradcheck`(fp64,有限差分对比解析梯度)。

- 作用:**证明这个 torch 参考实现的解析反向 == 有限差分**,即它本身可以作为
  梯度金标准,而无需借助 FLA。
- 配置:小规模 `B=1,T=32,H=1,D=8,lb=-3`(便于有限差分),`eps=1e-6,
  atol=1e-4,rtol=1e-3`。
- 结果:**PASS**。

### L1 — CUDA kernel vs fp64 参考(多指标 sweep)

用同一批原始输入,分别跑 CUDA kernel 和 fp64 torch 参考,对每个梯度
(`dq,dk,dv,dg,dbeta,dA_log,ddt_bias,dh0`)报告四个指标:
相对 RMSE、最大绝对误差、最大相对误差、cosine 相似度。

- 误差地板:bf16 kernel vs fp64 参考,rel-RMSE 天然在 ~1e-3..5e-3;
  因此 **cosine≈1.0 是稳健的判定门槛**(单点最大相对误差在接近 0 的元素上会
  偏大,属正常)。
- 覆盖形状:
  - `batched` (B=2,T=256,H=3) — 多 batch 多 head
  - `batched_lb1` (lb=-1) — 弱门控
  - `tail_tile` (T=48=3×16) — 整除边界
  - `partial_tile` (T=40→pad 48) — 非整除尾块
  - `single_chunk` (T=16) — 最小单块
  - `large_gate` (lb=-8) — 强门控压力(见 §5)
  - `varlen` (seq=[192,64,304]) — 变长多段
  - `varlen_short` (seq=[16,33,7,88]) — 变长含极短段
- 结果:除 `large_gate(lb=-8)` 外全部 **PASS**,最差梯度 cosine = 0.99980。

### 交叉校验 — vs FLA chunk_kda(Triton gold)

与上游 FLA 的 `chunk_kda` Triton 实现对比,阈值 `r < 2e-2`。
batched / batched_lb0 / varlen 三组用例全部 **PASS**。

测试入口:
- `tests/test_bwd_gradcheck.py` — L0 + L1 sweep(无需 FLA)
- `tests/test_bwd.py` — FLA 金标准套件

运行:
```bash
source /home/fluentllmenv/bin/activate
# L0 + L1(不依赖 FLA)
PYTHONPATH=/home/yuyanqi/FlashKDA python tests/test_bwd_gradcheck.py
# FLA 交叉校验
PYTHONPATH=/home/yuyanqi/.fla_pkgs:/home/yuyanqi/FlashKDA python tests/test_bwd.py
```

---

## 3. 为什么这套验证是"solid"的

1. **金标准独立于被测对象之外**:L0 用有限差分锚定 torch 参考,L1 再用该参考
   验证 CUDA kernel。即使没有 FLA,也能得到可信结论;有 FLA 时再做第三方交叉
   校验,三者互不依赖。
2. **多指标而非单一阈值**:rel-RMSE 反映整体、cosine 反映方向、max-abs/max-rel
   定位离群点。bf16 与 fp64 的固有误差地板被显式区分,避免误判。
3. **覆盖边界条件**:整除/非整除 tile、最小单块、多 batch/多 head、弱/强门控、
   变长多段(含极短段)——backward 的索引/scatter 路径在这些边界最容易出错。
4. **逐位等价检验**:修复 varlen bug 后,直接验证 varlen 输出与 batched
   **逐位一致**(bit-for-bit),这是比"误差小"更强的判据。

---

## 4. Varlen 反向 bug(已修复)

### 现象
变长(varlen)模式下,batched 正确但 varlen 的 per-token 梯度
(`dq/dk/dv/dg/dbeta`)错误;而它们沿 token 维求和得到的 `ddt_bias` 却与金标准
**逐位一致**。

### 诊断签名
"per-token 梯度错、但其 token 求和位级正确" ⇒ 数值算对了,但被**错位/覆盖**
(scatter/permutation bug)。因为 `dg_out[i]` 与 `ddt_bias_acc` 累加的是同一个值
(`dz·a_log_exp`),求和正确说明值对,逐 token 错说明落点错。

### 根因
`bwd_kernel1.cuh`(K1_bwd,grid=`(total_tiles, H)`)写的是 **token 索引**的梯度。
- batched:`total_tiles = N·ceil(T_seq/CHUNK)`(精确)。
- varlen:`total_tiles = ceil(T_total/CHUNK) + N`(**上界**,含 phantom tile)。

phantom CTA(超出真实 tile 数的多余 block)在解析 tile→seq 映射时回退到
`seq_idx=0`,于是**重复处理 seq-0 的 chunk,并把 seq-0 的真实 per-token 梯度
竞争覆盖成 0**。

> 注:`(N,H)` 网格的两个递推 kernel(K2_fwd/K2_bwd)没有 phantom tile;
> `fwd_kernel1.cuh` 虽有同样的 phantom 循环,但它写的是 **tile 索引**
> (`head*total_tiles+global_tile_idx`),落在无用的 phantom 槽位,无害 ——
> 这也是为何前向逐位正确。

### 修复
`csrc/smxx/bwd_kernel1.cuh`(约 L69-99):tile→seq 解析循环前置 `seq_idx = -1`,
解析后加一行 phantom 剔除守卫:

```cpp
if (seq_idx < 0) return;  // 剔除 varlen 上界产生的 phantom CTA
```

### 验证
修复后:varlen 输出与 batched **逐位一致**;L0+L1 套件、FLA 金标准全部 PASS。

---

## 5. 已知限制:`lower_bound ≤ -6` 门控溢出(采用方案一:文档化)

### 数值机制
门控:`g_nat = lower_bound · sigmoid(exp(A_log)·(g_raw + dt_bias)) ∈ [lb, 0]`;
累加 `gc = cumsum(g_nat)`;`ki = kn · exp(-gc)`。

当 `lb` 很负时,`gc` 单块最深可达 `lb·CHUNK`(CHUNK=16,故 lb=-8 时 ≈ -128),
`exp(-gc) = exp(128)` **超出 fp32 表示范围 → NaN**。

- FlashKDA kernel 与纯 torch fp32 参考在 `lb ≤ -6`(压力输入 gx=4,dt=4)
  **都会 NaN**。
- FLA(Triton)通过 log-space 累加/截断保持有限。
- 这是**数值鲁棒性差异,不是逻辑 bug**。

### 为什么文档化而非改 kernel(决策依据)
上游 FLA 对 `lower_bound` 的官方取值已经明确:

- `fla/models/kda/configuration_kda.py`:`safe_gate=True` 时必须设
  `lower_bound`,**"(recommended: -5)"**。
- `fla/layers/kda.py` docstring:在 `-5` 时每步最小衰减 `exp(g) ≈ 0.0067`,
  **"对质量影响可忽略"**;更负的值没有质量收益。
- `safe_gate=True` 正是启用 M=16 TensorCore 快速路径的开关,也正是
  FlashKDA 这个 kernel 对应的路径。

因此 **`lb = -5` 就是生产/推荐值,正好落在 FlashKDA 的安全区间内**;
`lb ≤ -6` 超出官方推荐操作范围,无质量收益。

### 结论(方案一)
- **支持范围:`lower_bound ≥ -5`**(覆盖整个上游推荐区间;此区间内即使压力输入
  也不 NaN,已验证)。
- `lower_bound ≤ -6` 列为已知限制,不投入 kernel 数值路径改造。
- 若未来确有 `lb ≤ -6` 的训练需求,再考虑将 `ki` 计算改为 log-space/截断
  (与 FLA 对齐),并重新走 §2 的三层验证。
