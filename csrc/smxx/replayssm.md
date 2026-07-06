# PR #28451 详解：ReplaySSM —— GDN/KDA 线性注意力的 buffered output-only decode

> 本文是对上游 PR `[GDN][KDA] ReplaySSM buffered output-only decode for Linear
> Attention`（sgl-project/sglang #28451，作者 yuan-luo，base `main`，13 文件
> +1722/-3，HEAD `b4b0325`）的完整解析。涵盖 motivation、数学原理、kernel 逐行、
> 框架集成流程、与本仓库 KV-buffer replay 的异同、layer/token 两个并行轴的分析，
> 以及具体例子。内容按对话讲解整理，力求一字不漏。

---

## 0. 速览（TL;DR）

- **针对瓶颈**：GDN/KDA 线性注意力的单 token decode 是**访存瓶颈**。packed decode
  kernel 每步把整个 recurrent state `S[HV,V,K]` 从 HBM **读出 + 写回**；H20-3e、
  batch≥64 时 **70–79% 卡在 HBM 带宽**上。
- **核心 idea（来自 Dao AI Lab 的 ReplaySSM）**：去掉每步的 state 写回。给每个 slot
  维护一个**最近 $L$ 步的小环形缓冲区** `(d, k, g)`，输出走"历史回放"重建，**每 $L$ 步
  才 flush 一次完整 state**。每步状态流量从 read+write($\sim 8dn$) 降到 **read-only**
  ($\sim 4dn$)，大致减半 → 理论 kernel 上限 $\sim 1.7\times$。
- **代价与权衡**：重建 state 的计算量是 baseline rank-1 更新的 $L$ 倍（$O(L\cdot K\cdot V)$），
  但这是**廉价的 tensor-core 计算**，能被减半的访存掩盖 → 本质是"用便宜的算力换贵
  的访存"。
- **落地形态**：新增 flag `--enable-linear-replayssm`（`--linear-replayssm-cache-len`
  默认 16）；一个 `IS_KDA` constexpr 化的 kernel 同时覆盖 GDN（per-head 标量门）和
  KDA（per-K-channel 门）；flag-OFF 时**字节一致**于现有路径。

---

## 1. Motivation：为什么做这件事

**针对的瓶颈：GDN/KDA 线性注意力的单 token decode 是访存瓶颈。**

GDN decode kernel 每个 decode step 都要把整个 recurrent state `S[HV, V, K]` 从 HBM
**读出 + 写回**。在 H20-3e、batch≥64 时，这个 packed decode **70–79% 卡在 HBM 带宽**
上——每步的 state read+write 流量占主导。对 fp32 state，每步状态访存 $\approx 8\cdot d\cdot n$ 字节
（读+写）。

**核心 idea（来自 Dao AI Lab 的 ReplaySSM，https://dao-lab.ai/blog/2026/replayssm/）：
去掉每步的 state 写回。**

不再每步写状态，而是给每个 slot 维护一个**最近 $L$ 步的小环形缓冲区** `(d, k, g)`，输出
走"历史回放"重建，**每 $L$ 步才 flush 一次完整 state**。于是每步状态流量从 read+write
($8dn$) 降到 **read-only**($4dn$，checkpoint `S0` 仍要读)→ 大致**减半**，理论 kernel
上限 $\sim 1.7\times$。

**代价与权衡**：重建 state 的计算量是 baseline rank-1 更新的 $L$ 倍（$O(L\cdot K\cdot V)$），但
这是**廉价的 tensor-core 计算**，能被减半的访存掩盖掉——所以本质是"用便宜的算力换贵
的访存"。

PR 对应的 issue 是 #28511。这个 PR 把 ReplaySSM decode 路径移植进 SGLang，藏在一个
flag 后面，并完整集成了 CUDA graph 和 radix 前缀缓存。它用**一个** gate-type 参数化的
kernel 同时覆盖**两种**线性注意力 gate 粒度：**GDN**（per-head scalar gate，radix +
CUDA graph）和 **KDA**（Kimi Delta Attention，per-K-channel gate）。

---

## 2. 数学原理

设 `a = exp(g)` 是 decay（GDN 是 per-head 标量，KDA 是 per-K 向量），`S` 是当前 token
**之前**的状态。单步 delta-rule：

$$
\begin{aligned}
d_{cur} &= \beta\cdot(v - (S\cdot \mathrm{Diag}(a))\cdot k) && \text{corrected delta 向量} \\
o &= (S\cdot \mathrm{Diag}(a))\cdot q + d_{cur}\cdot(k^\top q) && \text{输出} \\
S_{new} &= S\cdot \mathrm{Diag}(a) + d_{cur}\cdot k^\top && \text{只在 flush 时落盘}
\end{aligned}
$$

其中 $(\cdot)$ 是 K 维上的逐元素乘。对 GDN，$a$ 是标量，所以
$S\cdot(a\odot q) = a\cdot(S\cdot q)$（便宜的标量后乘）；对 KDA，per-K 的 $a$ 在 matvec 前折进 q/k。
$k^\top q$ 用**原始**当前 k/q（rank-1 项），两种 gate 类型一致。

**关键技巧——状态从 checkpoint + ring 重建，永不落 HBM**。缓冲了 $j=0..m-1$ 步
`(d_j, k_j, g_j)` 后，当前 token 前的状态：

$$
\begin{aligned}
S &= \mathrm{Diag}(A)\cdot S_0 + \sum_j d_j\cdot(W_j \odot k_j)^\top && \text{（per-K 形式）} \\
A[c] &= \exp\!\Big(\sum_j g_j[c]\Big) && \text{总 decay（用 cumsum 算，不需要逐步 state）} \\
W_j[c] &= \exp\!\Big(\sum_{i>j} g_i[c]\Big) = \prod_{i>j} a_i[c] && \text{回放 decay}
\end{aligned}
$$

对 GDN，$A$ 和 $W_j$ 是标量（$g$ 与 K 无关），$W_j$ 折到 $d_j$ 上（标量任一边都行）；对 KDA 折到
$k_j$ 上。`S` 按 **K-tile** 分块重建，每块算出来立刻和 `q`/`k` 收缩成标量 `Sq`/`Sk`，
**整块 $[V,K]$ state 永远不会 materialize 到 HBM**。

**边界自洽**：$L=1$ 时 ring 恒空、`write_pos==L-1` 每步成立 → 重建项为 0、总 decay 为
1，kernel **代数上退化为原 packed decode**（用作 bit-exact 正确性 oracle）。

### 2.1 为什么一次 matmul 就够、不用逐 token 递推（并行原理）

**把递推展开成闭式。** GDN 递推（$\alpha$ 是标量衰减，$d$、$k$ 是向量）：

$$
S_t = \alpha_t \cdot S_{t-1} + d_t \cdot k_t^\top
$$

从 checkpoint $S_0$ 展开 $m$ 步（手推前 3 步看规律）：

$$
\begin{aligned}
S_1 &= \alpha_0 S_0 + d_0 k_0^\top \\
S_2 &= \alpha_1 S_1 + d_1 k_1^\top = \alpha_1\alpha_0 S_0 + \alpha_1\cdot d_0 k_0^\top + d_1 k_1^\top \\
S_3 &= \alpha_2 S_2 + d_2 k_2^\top = \alpha_2\alpha_1\alpha_0 S_0 + \alpha_2\alpha_1\cdot d_0 k_0^\top + \alpha_2\cdot d_1 k_1^\top + d_2 k_2^\top
\end{aligned}
$$

通项：

$$
S_m = \Big(\prod_{i=0}^{m-1} \alpha_i\Big)\cdot S_0
      \;+\; \sum_{j=0}^{m-1} \Big(\underbrace{\prod_{i=j+1}^{m-1} \alpha_i}_{\text{回放权重 } W_j = \prod_{i>j} \alpha_i}\Big)\cdot d_j k_j^\top
$$

（前一项 $\prod_{i=0}^{m-1}\alpha_i$ 即总衰减 $A$。）

这正是 kernel 里那两行 cumsum 的来源：

$$
\begin{aligned}
A   &= \exp\!\Big(\sum g_i\Big)          &&= \texttt{b\_total\_decay} \\
W_j &= \exp\!\Big(\sum_{i>j} g_i\Big)    &&= \exp(\texttt{b\_g\_total} - \texttt{b\_g\_prefix}) = \texttt{b\_replay\_decay}
\end{aligned}
$$

**关键：标量衰减让各项"解耦"，可写成 matmul。** 通项里每一项 $W_j \cdot d_j k_j^\top$
**只依赖 $(d_j, k_j)$ 和一个标量 $W_j$，不依赖任何中间 $S$**。把 $m$ 项 rank-1 外积堆叠：

$$
\begin{aligned}
&\text{令}\quad D = [d_0\ d_1\ \dots\ d_{m-1}] && \text{形状 } [V, m] \\
&\phantom{\text{令}\quad} \tilde{K} = [W_0 k_0;\ W_1 k_1;\ \dots;\ W_{m-1} k_{m-1}] && \text{形状 } [m, K]\text{（第 } j \text{ 行是 } W_j\cdot k_j\text{）} \\
&\text{则}\quad \sum_j d_j (W_j k_j)^\top = D \,@\, \tilde{K} && \text{一次 } [V,m]\times[m,K] = [V,K] \text{ 的 matmul}
\end{aligned}
$$

于是整个重建是：

$$
S = A\cdot S_0 + D \,@\, \tilde{K} \qquad \leftarrow \text{一次 } \texttt{tl.dot}\text{（tensor core），无 token 轴串行}
$$

$m$ 个 rank-1 更新 → 一次矩阵乘。这就是 kernel 第 252 行
`b_h_c = b_h0_c*b_total_decay + tl.dot(b_d_tc, b_k_all_c)` 干的事。

**为什么标量/对角衰减是并行的"命门"。** 能把 $W_j = \prod_{i>j} \alpha_i$ 提到求和外面、用
**cumsum 一次性预算**所有位置的权重——靠的是 **$\alpha$ 是标量（GDN）或对角 per-K（KDA）**，
所以衰减因子**可交换、可因式分解**：

$$
\prod \alpha_i \ \text{在 log 域就是}\ \sum g_i \ \to\ \text{cumsum 一次出全部 } W_j
$$

如果衰减是**满矩阵** $S_t = A_t S_{t-1} + \dots$（一般线性 RNN），$\prod A_i$ 是矩阵连乘、
不可交换、不能 cumsum，每个 $W_j$ 都得串行算矩阵积——**重建就退回串行**。GDN/KDA 的 gate
恰好是标量/对角，才解锁了这个并行形式（这也是 chunked linear attention / 线性 RNN
parallel-scan 的同一个原理）。

### 2.2 为什么必须存 d、不能存 v

$d_j = \beta_j(v_j − \alpha_j S_{j-1} k_j)$ 本身依赖 $S_{j-1}$。但 $d_j$ 在 step $j$（它是当前
token 时）**已经算好了**，存下来即可。重建时 $W_j$ 只是把 step $j$ **之后**累积的衰减乘
上去——$d_j$ 对 $S_{j-1}$ 的依赖在它生成那刻就"焊死"了，后续步骤只是整体再衰减，正好
因式分解成 $W_j$。

若只存 $v$、重建时还原 $d_j$ 就要 $S_{j-1}$，$S_{j-1}$ 又要 $d_{j-1}$…… → **退回 $L$ 步串行
递推**。**存 $d$ 的全部意义，就是把串行依赖"切断"，让重建变成一次并行 matmul。**

### 2.3 存 d 的代价 vs 收益（量级账）

关键：**$S$ 是个矩阵 $[V,K]$，而 $(d,k)$ 是它的 rank-1 因子**。delta rule 的更新本质是一个
秩-1 外积更新 $S_t = \alpha_t\cdot S_{t-1} + d_t\cdot k_t^\top$。所以存 $d_t[V] + k_t[K] + g_t[1]$ 这
$V+K+1$ 个数，就完整记下了"这一步对 $S$ 做了什么"，**不需要存整个 $V\times K$ 的 $S$**。

拿 GDN 典型值 $K=128, V=128$ 算每步的 state 流量：

| | 写入量（每步每头） |
|---|---|
| 原版 packed decode：写回整个 S | $V\times K = 16384$ 个数 |
| ReplaySSM：append `(d,k,g)` | $V+K+1 = 257$ 个数 |
| 比值 | $\approx 64\times$ 更小 |

也就是说，存 d 的代价 $\approx$ 原本写 S 代价的 **1.5%**，却换掉了那一次 16384 个数的 HBM
写回。这就是为什么"耗空间"在量级上可以忽略——rank-1 因子远小于满矩阵。

并且，**d 依赖 S 不构成额外开销**：看 decode 每步必做的事（无论开不开 ReplaySSM），
$o = (\alpha\cdot S)\cdot q + d_{cur}\cdot(k^\top q)$、$d_{cur} = \beta\cdot(v − (\alpha\cdot S)\cdot k)$——`d_cur` 是产出 `o` 的必经
中间量，这一步本来就在寄存器里算好了。ReplaySSM 只是在写输出之前，顺手把这个已经算出
来的 `d_cur`（连同 k、g）`tl.store` 进 ring，**没有任何额外计算**，只多了一次小 store。

---

## 3. Kernel 逐行解析：`fused_recurrent_linear_replayssm_decode`

文件：`python/sglang/srt/layers/attention/fla/fused_recurrent_linear_replayssm.py`（628 行）。
分两部分：Python wrapper（:436-619）+ `@triton.jit` kernel 本体（:61-433）。
文件头注释明确：GDN 是 KDA 的特例（所有 per-K decay 相等），一个 `IS_KDA` constexpr 选
gate 路径，且 **GDN 路径与原 kernel 逐字节相同**（无回归）。Kernel 移植自
vllm `fused_recurrent_replayssm.py`（commit 3c85112），保留 vllm::persistent 风格便于
上游同步。

### 3.1 Wrapper 参数（:436-457）

| 参数 | 形状 / 类型 | 含义 |
|---|---|---|
| `mixed_qkv` | `[B, 2*H*K + HV*V]` | conv1d 之后的打包 q\|k\|v（每行一个 decode token） |
| `a` | GDN `[B, HV]` / KDA `[B, HV, K]` | gate 输入（送进 softplus 前的原始值） |
| `b` | `[B, HV]` | beta 输入（过 sigmoid 得 $\beta$） |
| `A_log` | `[HV]` | 每头标量的 log 空间 decay 参数（两种 gate 都是 per-head 标量） |
| `dt_bias` | GDN `[HV]` / KDA `[HV, K]` | time-step bias，加到 gate 输入上 |
| `scale` | float | q 的缩放系数（注意力 $1/\sqrt{d}$ 之类） |
| `initial_state` | `[num_slots, HV, V, K]` | **既是 checkpoint 读 (h0)，也是 flush 时写 (ht)**，原地 |
| `d_cache` | `[num_slots, HV, L, V]` | ring：corrected delta 向量 |
| `k_cache` | `[num_slots, H, L, K]` | ring：normed/scaled keys |
| `g_cache` | GDN `[num_slots, HV, L]` / KDA `[..,L,K]` | ring：log-decay gate（**必须 fp32**） |
| `out` | `[B, 1, HV, V]` | 输出（每步都写） |
| `ssm_state_indices` | `[B]` | 每个 decode 行的物理 slot 号 |
| `write_pos` | `[B]` int32 | 每行的 ring 游标（0..L-1） |
| `force_flush` | `[B]` int32 / None | !=0 强制本步 flush（radix 边界用） |
| `use_qk_l2norm_in_kernel` | bool | 是否在 kernel 内对 q/k 做 L2 归一化 |
| `is_kda` | bool | 选 KDA(per-K gate) 还是 GDN(标量 gate) |
| `block_v`/`num_warps`/`num_stages`/`nk` | | 调优参数 |

### 3.2 Wrapper 维度推导与校验（:477-561）

```python
B = mixed_qkv.shape[0]                          # batch（decode 行数）
num_state_slots, HV, V, K = initial_state.shape # 从 checkpoint 拿 HV/V/K
qkv_dim = mixed_qkv.shape[1]
q_dim = (qkv_dim - HV * V) // 2                 # 反推 q 段宽度：总宽 - v段(HV*V)，再均分给 q/k
H = q_dim // K                                  # k/q 头数 H（注意 H 可能 < HV，GQA）
max_cache_len = d_cache.shape[2]                # L（ring 深度）
```

然后是一堆形状 sanity check：KDA 要求 `a=[B,HV,K]`、`dt_bias=[HV,K]`、`g_cache=[..,L,K]`
（:512-533）；GDN 要求 `a=[B,HV]`、`dt_bias=[HV]`、`g_cache=[..,L]`。再校验
`d_cache/k_cache/g_cache` 的 per-slot 形状、`g_cache` 必须 fp32、`out` 必须
`[B,1,HV,V]`、`write_pos`/`ssm_state_indices` 长度都得是 B（:535-561）。

### 3.3 Wrapper block / grid 配置（:563-579）

```python
BK = triton.next_power_of_2(K)        # K 向上取 2 的幂
if triton.cdiv(K, BK) != 1: raise     # 要求 K 一个 block 装得下（NK_global=1）
if BK % nk != 0: raise
BKT = BK // nk                        # 每个 K-tile 的宽度（nk 个 tile）
if BKT < 16: raise                    # tl.dot 要求 ≥16
BV = block_v or min(next_pow2(V), 64) # V 方向 block，默认≤64
BC = max(16, next_pow2(max_cache_len))# ring 维度的 block（≥16，覆盖 L）
grid = (cdiv(V, BV), B, HV)           # 三维 grid：V-tile × batch × head
```

**grid 三轴 = (V 分块, batch 行, V-head)**。K 不分 grid（一个 program 内部用
`for kk in range(NK)` 串行扫 K-tile）。`num_warps=1`（递推天然低并行）。

### 3.4 Wrapper launch（:580-619）

注意两个细节：
- `h0=initial_state, ht=initial_state` —— **同一个 tensor**，读写同址（flush 时原地
  更新 checkpoint）。
- `force_flush=force_flush if not None else write_pos` —— None 时塞个占位
  （`write_pos`），同时 `HAS_FORCE_FLUSH=False` 让 kernel 不读它。
- 末尾 `:625-628` 是向后兼容别名：`fused_recurrent_gdn_replayssm_decode` = 本函数
  （`is_kda` 默认 False）。

### 3.5 Kernel：program 定位（:99-106）

```python
i_v  = tl.program_id(0)        # 第几个 V-tile
i_n  = tl.program_id(1)        # 第几个 decode 行（batch）
i_hv = tl.program_id(2)        # 第几个 V-head
i_h  = i_hv // (HV // H)       # 该 V-head 对应的 K/Q 头（GQA：多个 V-head 共享一个 K 头）

o_v = i_v * BV + tl.arange(0, BV)   # 本 program 负责的 V 维下标 [BV]
o_c = tl.arange(0, BC)              # ring 维下标 [BC]（覆盖 0..L-1）
mask_v = o_v < V                    # V 越界 mask
```

### 3.6 Kernel：取 slot + padded 行早退（:109-117）

```python
state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
p_o = o + (i_n * HV + i_hv) * V + o_v     # 本行本头的输出地址 [BV]
if state_idx < 0:                          # padding 行（CUDA graph 补齐用）
    tl.store(p_o, zeros, mask=mask_v)      # 写 0
    return                                 # 直接退出
```

### 3.7 Kernel：flush 标志 + 有效 ring 范围（:121-129）

```python
b_write_pos = tl.load(write_pos + i_n).to(tl.int64)
b_is_flush  = b_write_pos == MAX_CACHE_LEN - 1     # 自然 wrap：游标到 L-1 这步 flush
if HAS_FORCE_FLUSH:
    b_is_flush = b_is_flush | (tl.load(force_flush + i_n) != 0)  # radix 边界强制 flush
cache_valid = o_c < b_write_pos    # 只有 < write_pos 的 ring 槽是"已提交"的，可读
```

关键：`cache_valid` 用的是**真实** `write_pos`，即便 force-flush 在 ring 中途触发，也
只读已写入的项，不会读到脏数据。

### 3.8 Kernel：当前 token 的 gate（GDN 分支，:136-154）

```python
A_log_val = tl.load(A_log + i_hv)              # per-head 标量
b_val     = tl.load(b + i_n*stride_b_tok + i_hv)
beta_val  = tl.sigmoid(b_val)                  # β = sigmoid(b)
if not IS_KDA:                                 # GDN：标量 gate 在这里算
    a_val   = tl.load(a + i_n*stride_a_tok + i_hv)
    dt_bias_val = tl.load(dt_bias + i_hv)
    x = a_val + dt_bias_val
    softplus_x = where(x<=THRESH, log(1+exp(x)), x)   # 数值稳定 softplus
    g_val   = -exp(A_log_val) * softplus_x      # 当前步 log-decay g
    alpha_val = exp(g_val)                       # α = exp(g)，标量 decay

    # —— 回放 decay：从 ring 里缓存的历史 g 现算（无需逐步 state）——
    p_g_main = g_cache + (state_idx*HV + i_hv)*MAX_CACHE_LEN + o_c
    b_g_all  = tl.load(p_g_main, mask=cache_valid, other=0.0)  # 历史 g_j  [BC]
    b_g_prefix = tl.cumsum(b_g_all, axis=0)      # 含当前项的前缀和 Σ_{i≤j} g_i
    b_g_total  = tl.sum(b_g_all, axis=0)         # 全部和 Σ_i g_i
    b_replay_decay = where(cache_valid, exp(b_g_total - b_g_prefix), 0.0)  # W_j = ∏_{i>j}α_i
    b_total_decay  = exp(b_g_total)              # A = ∏_i α_i
```

`b_g_total - b_g_prefix` $= \sum_{i>j} g_i$，取 exp 就是 $W_j = \prod_{i>j} \alpha_i$——即 ring 中第
j 步那条记录衰减到"当前"还剩多少。`b_total_decay` 是 checkpoint S0 衰减到当前的总系数。

### 3.9 Kernel：取 ring 里的 d 向量 + tensor-core cast（:158-181）

```python
p_d_main = d_cache + (((state_idx*HV+i_hv)*MAX_CACHE_LEN + o_c[None,:])*V + o_v[:,None])
b_d_all  = tl.load(p_d_main, mask=mask_v[:,None]&cache_valid[None,:], other=0)  # [BV, BC]
if not IS_KDA:
    b_d_tc = (b_d_all * b_replay_decay[None,:]).to(p_o.dtype.element_ty)  # GDN：W_j 折到 d 上
else:
    b_d_tc = b_d_all.to(p_o.dtype.element_ty)                             # KDA：W_j 折到 k 上（循环内）
```

**这里 cast 到 IO dtype 是性能命门**（:164-173 注释）：让后面的 `tl.dot` 走
**tensor core**（fp32→TF32，bf16→bf16 TC）。若强行 IEEE fp32 dot → 关闭 tensor core
→ 慢 $10$-$20\times$，把 $1.5\times$ 收益变成 $15\times$ 退化。这是全 PR 最大的性能陷阱。GDN 把回放 decay
$W_j$ 乘到 `d` 上（标量任一边都行）；KDA 因 per-K 必须乘到 `k` 上。$L=1$ 时 buffer 空、
此 dot 恒零，无论精度都 bit-exact。

### 3.10 Kernel：当前 token 的 v + 可选 L2norm（:184-202）

```python
v_off = (2*H*K) + i_hv*V + o_v          # v 在 mixed_qkv 里的偏移：跳过 q段+k段(2*H*K)
b_v   = tl.load(mixed_qkv + i_n*stride + v_off, mask=mask_v)   # 当前 v [BV]

if USE_QK_L2NORM_IN_KERNEL:             # 算 q/k 整向量的 1/‖·‖（只算系数，不留向量）
    qf = load q全段; kf = load k全段
    q_rnorm = 1/sqrt(sum(qf²)+1e-6)
    k_rnorm = 1/sqrt(sum(kf²)+1e-6)
else:
    q_rnorm = k_rnorm = 1.0
```

### 3.11 Kernel：K-tile 主循环——重建 S 并立即读出（:208-320）

初始化累加器：

```python
b_state_q = zeros([BV])    # 累 Sq（S 和 q 收缩）
b_state_k = zeros([BV])    # 累 Sk（S 和 k 收缩）
cur_kq    = zeros([1])     # 累 kᵀq（rank-1 项）
write_k   = (not flush) and (i_v==0) and (i_hv == i_h*(HV//H))  # 只让一个 program 写 k ring（去重）
write_g_kda = IS_KDA and (not flush) and (i_v==0)               # KDA 写 g ring 的条件
```

循环体 `for kk in range(NK)`（:213）：

```python
o_kt = kk*BKT + arange(0,BKT); mask_kt = o_kt < K     # 本 K-tile 的 K 下标
q_c = load q[i_h, o_kt] * q_rnorm                      # 本 tile 的 q
k_c = load k[H*K + i_h, o_kt] * k_rnorm                # 本 tile 的 k（k 段偏移 H*K）
q_cs = q_c * scale
cur_kq += sum(k_c * q_cs)                              # 累加 kᵀq（用原始 k/q，gate 无关）

# 读 checkpoint S0 的本 tile [BV, BKT]
b_h0_c = load h0[state_idx, i_hv, o_v, o_kt]
# ring 里历史 k 的本 tile [BC, BKT]
p_k_c = k_cache + ((state_idx*H+i_h)*MAX_CACHE_LEN + o_c)*K + o_kt
```

GDN 分支（:248-255）：

```python
b_k_all_c = load(p_k_c)                                # 历史 k [BC, BKT]
b_h_c = b_h0_c * b_total_decay + tl.dot(b_d_tc, b_k_all_c)
#       └ checkpoint 衰减 ┘   └ Σ_j d_j (W_j k_j)ᵀ 的本 tile（tensor core）┘
q_eff = q_cs; k_eff = k_c                              # GDN：当前步标量 decay 循环后再乘
```

这就是 $S_{tile} = \mathrm{Diag}(A)\cdot S_{0,tile} + \sum_j d_j\cdot k_j^\top$ 的实现，重建出来的 `b_h_c [BV, BKT]`
是 S 的一个 K 切片，**只在寄存器/SRAM 里**。

KDA 分支（:256-304）：per-K gate 复杂些——本 tile 从 `g_cache` 读历史 g `[BC,BKT]`，
per-K 做 cumsum 得 `b_replay_decay_c`/`b_total_decay_c`，把回放 decay 折到历史 k
（`b_k_scaled`）、总 decay 折到 S0，再 `tl.dot`。当前步 per-K $\alpha$ 折进 `q_eff/k_eff`。
非 flush 时把当前 g tile 写进 ring（:298-304）。

读出（:307-308，两分支共用）：

```python
b_state_q += sum(b_h_c * q_eff[None,:], axis=1)   # 把这块 S-tile 和 q 收缩进累加器
b_state_k += sum(b_h_c * k_eff[None,:], axis=1)   # 和 k 收缩
```

**关键**：每个 K-tile 重建完立刻和 q/k 点乘累加，整块 `[V,K]` 的 S **永不在 HBM
materialize**。

写当前 k 进 ring（:310-320，仅非 flush）：

```python
if write_k:
    tl.store(k_cache[state_idx, i_h, b_write_pos, o_kt], k_c, mask=...)
```

### 3.12 Kernel：当前 token 输出（:326-331）

```python
if not IS_KDA:                # GDN：当前步标量 decay 在这统一乘
    b_state_q *= alpha_val    # Sq → (S·Diag(α))·q
    b_state_k *= alpha_val
b_d_cur = beta_val * (b_v - b_state_k)        # d_cur = β(v − (S·Diag(α))·k)
b_o     = b_state_q + b_d_cur * tl.sum(cur_kq)# o = (S·Diag(α))·q + d_cur·(kᵀq)
tl.store(p_o, b_o, mask=mask_v)               # 写输出
```

对应数学：$o = (S\cdot \mathrm{Diag}(a))\cdot q + d_{cur}\cdot(k^\top q)$。

### 3.13 Kernel：Flush 分支——把当前 token 折进 checkpoint 并落盘（:333-418）

```python
if b_is_flush:
    for kk in range(NK):                      # 重走 K-tile 再建一次 S（输出循环已耗掉寄存器值）
        ... 重建 b_h_c（同 3.11）...
        if not IS_KDA:
            b_h_new_c = alpha_val * b_h_c + b_d_cur[:,None] * k_c[None,:]   # S_new = αS + d_cur kᵀ
        else:
            b_h_new_c = b_h_c * alpha_cur_c[None,:] + b_d_cur[:,None]*k_c[None,:]  # per-K
        tl.store(ht[state_idx, i_hv, o_v, o_kt], b_h_new_c, mask=...)       # 写回 checkpoint
```

flush 后 ring 逻辑清空（caller 下步把 `write_pos` 置 0）。注意这里为 flush 多花一次
K-tile 重建（重算 S）——这就是"每 L 步一次"的较重那步，但只占 1/L。

### 3.14 Kernel：非 flush 分支——append 当前 d（和 GDN 的 g）（:419-433）

```python
else:
    tl.store(d_cache[state_idx, i_hv, b_write_pos, o_v], b_d_cur, mask=...)   # 存 d_cur
    if (not IS_KDA) and (i_v == 0):
        tl.store(g_cache[state_idx, i_hv, b_write_pos], g_val)               # GDN 存标量 g
```

（k 已在主循环里写；KDA 的 g 也在主循环写。`i_v==0` 保证 V 方向多个 program 只有一个
写、不重复。）

### 3.15 一步 decode 的数据流串起来

- **读**：$S_0$ (checkpoint, 1次) + ring[$(d,k,g) \times$ write_pos 项]
- **算**：$S = \mathrm{Diag}(A)\cdot S_0 + \sum_j d_j\cdot(W_j\odot k_j)^\top$ ← K-tile 重建，tensor core，不落 HBM
- **算**：$o = (S\cdot \mathrm{Diag}(\alpha))\cdot q + d_{cur}\cdot(k^\top q)$ ← 立即读出
- **写**：非 flush → 只 append $(d_{cur},k,g)$ 进 ring，S 不落盘
- **写**：flush(每 $L$ 步/radix 边界) → $S_{new}$ 写回 $S_0$

**净账**：每步省掉一次整 S（$HV\cdot V\cdot K$）的 HBM 写；代价是 ring 的小读写 + $L$ 倍的
tensor-core 重建计算（被省下的带宽掩盖）。$L=1$ 时 ring 恒空、`cache_valid` 全 False、
重建项为 0、`b_total_decay=1`，kernel 代数退化为原 packed decode（bit-exact oracle）。

---

## 4. 框架集成：装在哪、怎么串

### 4.1 整体定位

模型每个 GDN/KDA 线性注意力层在 decode 时，原本调
`fused_recurrent_gated_delta_rule_packed_decode`（每步读+写整个 state）。开了
`--enable-linear-replayssm` 后，**dispatch 改走** buffered kernel（每步只读
checkpoint，每 L 步才写 state）。其余模型结构、prefill、MoE、采样**完全不变**。

```
ServerArgs (--enable-linear-replayssm, --linear-replayssm-cache-len=16)
        │
        ▼
MambaPool  ──── 持久态：环形缓冲 + 游标（每个物理 slot 一份）
        │        replayssm_d/k/g  : [num_layers, num_slots, ...]  ← 所有层、所有 slot
        │        replayssm_write_pos : [num_slots] int32          ← 跨层共享的"decode 第几步"
        │
        ▼
HybridLinearAttnBackend._forward_metadata / _replay_metadata
        │        每个 forward 一次：snapshot 本步游标 → 算 force_flush → 推进游标
        │        产出 ForwardMetadata.replayssm_write_pos / replayssm_force_flush
        │
        ▼
GDNAttnBackend.forward_decode  ──── 每层调用：把本层 ring 切片 + 共享游标喂给 kernel
                                     fused_recurrent_linear_replayssm_decode(...)
```

### 4.2 持久状态在 `MambaPool`（`memory_pool.py`，+126）

三个环形缓冲 + 一个游标，**只在 flag 开时分配**（关时全 None → 旧路径字节一致）：

```python
replayssm_d : [L层, slots, HV, L, V]   # corrected delta 向量
replayssm_k : [L层, slots, H,  L, K]   # normed/scaled keys
replayssm_g : [L层, slots, HV, L]      # GDN 标量 gate；KDA 是 [..,L,K]
replayssm_write_pos : [slots] int32    # per-slot 的 decode 位置计数器（0..L-1）
```

关键设计：
- **`write_pos` 是 per-物理-slot，不是 per-layer**——所有 GDN 层在同一 decode step
  共享同一个游标（它表示"这个请求 decode 到第几步"）。`replayssm_is_kda` 记录 gate
  粒度，驱动 kernel 的 IS_KDA 路径 + g_cache 布局。
- **slot 分配时置 0**（`alloc` 里）：此刻 prefill 刚把"prefill 后的完整 state"写进
  `temporal[slot]`，正好是空 ring 的 checkpoint `S0`。注释：write_pos=0 表示"ring
  空"，decode kernel 忽略 ring 内容、只读 checkpoint。
- **COW（prefix 命中复制）时也置 0**（`copy_from` 里）：复制来的是已 flush 的快照
  （radix track 快照取在 force-flush 边界，不带 pending ring），目标 slot 的 ring 必须
  从空开始。覆盖 prefix-hit 的 deferred-COW 和任何 copy-into-slot 路径。
- `State.at_layer_idx` 改成：`v is None` 时直接传 None（让 flag-off 的 None 字段安全
  穿过）。`mem_usage_bytes` / `get_contiguous_buf_infos` / `get_state_dim_per_tensor`
  都加了 `None` 跳过 + 把 `replayssm_*` 列为不参与 RDMA 传输的派生 scratch。

### 4.3 每 forward 一次的游标管理（`_forward_metadata`，eager 路径）

不是每层做，而是**整批一次**：

```python
slots = mamba_cache_indices            # 本批每行的物理 slot
replayssm_write_pos = write_pos_buf[slots].clone()   # ① snapshot 本步游标给 kernel
# ② GDN 算 force_flush：和 radix track 完全相同的条件
force_flush = (seq_lens_cpu % mamba_track_interval == 0)
# ③ 推进持久游标：flush 了就归零，否则 (pos+1)%L
next_pos = where(flushed, 0, (pos+1)%L)
write_pos_buf[uniq_slots] = next_pos   # 用 unique 去重，避免 padded 行 clamp 到 slot0 的竞争
```

KDA 暂不做 radix 协调（`force_flush` 对 KDA 关闭，KimiLinear 上游本就禁了 radix），只在
自然 `write_pos==L-1` wrap 时 flush。

### 4.4 每层的 kernel 调用（`gdn_backend.forward_decode`，+16）

```python
layer_cache = mamba_pool.at_layer_idx(layer)   # 取本层的 ring 切片
fused_recurrent_..._packed_decode(             # dispatch 内部按 flag 选 replayssm
    ...,
    replayssm_d=layer_cache.replayssm_d,        # 本层 ring
    replayssm_k=..., replayssm_g=...,
    replayssm_write_pos=metadata.replayssm_write_pos,   # 共享游标
    replayssm_force_flush=metadata.replayssm_force_flush,
)
```

### 4.5 metadata 字段（`mamba2_metadata.py`，+10）

`ForwardMetadata` 新增两个 Optional 字段（flag-off 时为 None）：
- `replayssm_write_pos`：本 decode step 每行的 ring 写游标（从持久 per-slot buffer
  gather，再为下一步 advance）。int32，长度=batch。
- `replayssm_force_flush`：本 decode step 每行的 int32 flush 标志。!=0 强制 kernel 把
  partial ring + 当前 token 折进 checkpoint，恰好命中 radix track 快照的行，即
  `seq_lens_cpu % mamba_track_interval == 0`。

---

## 5. 一个请求完整的生命周期流程

**① Prefill**

正常跑，把 prefill 后的完整 SSM state 写入 `temporal[slot]`。
alloc 时 `write_pos[slot]=0` → ring 为空，`temporal[slot]` 就是 checkpoint $S_0$。

**② Decode step t（非 flush，$\text{write\_pos} < L-1$）**：

每个 GDN 层：
- 读 checkpoint $S_0$ + ring 里已提交的 `(d,k,g)`
- 按 K-tile 重建 S（tensor core），立刻和 q/k 收缩出 o ← 不落 HBM
- 把当前步的 `(d_cur, k, g)` append 进 `ring[write_pos]`
- state 不写回

forward 末尾：`write_pos += 1`

**③ Decode step t（flush，`write_pos==L-1` 或 force_flush）**：
- 重建 S，算出 o（同上）
- 额外：$S_{new} = \alpha\cdot S + d_{cur}\cdot k^\top$ 写回 `temporal[slot]` ← 唯一的 state 写

forward 末尾：`write_pos` 归零，ring 逻辑清空

**④ 与 radix 缓存对齐**：

radix 在 $\text{seq\_lens} \bmod \text{track\_interval} == 0$ 处快照 `temporal[slot]`。
`force_flush` 用同一条件算 → 保证快照那一步恰好 flush，`temporal[slot]` 是最新的。

**⑤ Prefix 命中（COW）**：

`copy_from` 把已 flush 的快照复制进新请求的私有 slot，并把该 slot 的 `write_pos` 置 0。

**净效果**：原本每步 1 次 state 读 + 1 次 state 写；现在每步 1 次 checkpoint 读 + 极小
的 ring append，每 $L$ 步才 1 次 state 写 → 状态访存约减半。

---

## 6. CUDA Graph 怎么活下来（最绕的一块）

decode forward 被 capture 一次反复 replay，所以游标不能每步新建 tensor。方案是**镜像
`mamba_cache_indices` 的处理方式**：
- `init_cuda_graph_state`：给每个 bs 预分配 **static** 的 `replayssm_write_pos_list[bs]`
  / `force_flush_list[bs]`（形如 `(i+1,)`，索引 `[bs-1]`），被 graph 按指针 capture。
  flag-off 时整个 list 为 None，dispatch 原样穿过。
- `_capture_metadata`：只是把 metadata 指向那两个 static buffer（capture 记录指针，
  不做 advance/snapshot，其零值会被 `_replay_metadata` 原地覆盖）。
- `_replay_metadata`（在 captured 区**外**）：每次 replay 用 `copy_()` **原地刷新**
  static buffer，从持久 `write_pos_buf` 取本步值；游标推进受 `not in_capture` 保护，
  避免 capture/warmup 污染真实 slot 计数（capture 跑在 dummy slot 上）。
- flush 是 **device 端分支**（`is_flush = write_pos==L-1`），一张 captured graph 能
  处理 ring 周期任意位置的所有行。
- `_replayssm_track_flush_mask`：用与 radix track **完全相同**的
  `seq_lens_cpu % mamba_track_interval == 0`、同一份 post-increment seq_lens_cpu，
  保证 force-flush 与快照无 off-by-one。

---

## 7. 与本仓库 KV-buffer replay 的异同

两者**同源**（都靠"存轻量信息 + 重建"避免存满的中间 SSM 态），但**并行的维度和重建
方式根本不同**。

### 7.1 最本质差别：并行 matmul vs 串行递推

| | **本仓库 KV buffer replay** | **ReplaySSM** |
|---|---|---|
| 存什么 | 原始输入 `(mixed_qkv, a, b)` | rank-1 因子 `(d, k, g)`，其中 **d 已含 S** |
| 重建怎么做 | **重跑 GDN 递推**：`fused_sigmoid_gating_delta_rule_update` 沿 accepted token 一步步推 | **一次 matmul**：$S = \mathrm{Diag}(A)\cdot S_0 + D@\tilde{K}$（`tl.dot`，tensor core） |
| token 轴 | **串行**（递推有 RAW 依赖，必须逐 token） | **并行**（L 步历史在一个 matmul 里规约掉，无串行依赖） |

- **KV buffer**：存 `(qkv,a,b)` ──重建──▶ `for t in accepted: d_t=f(S_{t-1}); S=αS+d_t k_tᵀ` ← 串行（即 $S = \alpha S + d_t k_t^\top$）
- **ReplaySSM**：存 `(d,k,g)` ──重建──▶ $S = \mathrm{Diag}(A) S_0 + D@\tilde{K}$ ← 并行 matmul

ReplaySSM 多付了"存 d"的代价，换来 token 轴可以并行；KV buffer 存更原始的输入（不存
d），省了那点存储，但代价是重建时必须把 d 现场串行重算一遍（重跑整个递推）。

### 7.2 相同的并行维度：layer × request × head（grid 同构）

- KV buffer fused kernel：`grid = (1, NV, num_layers·N·HV)` —— request(N) × value-head
  (HV) × **layer**（fusion 加的轴）并行。
- ReplaySSM decode：`grid = (cdiv(V,BV), B, HV)` —— request(B) × head(HV) × V-tile
  并行。

都把 request 和 head 铺在 grid 上、每个 program `num_warps=1`。区别在 **layer 轴**。

### 7.3 关键结构差异：能不能把 36 层打成一个 launch

| | KV buffer replay | ReplaySSM |
|---|---|---|
| 触发时机 | MTP **verify 之后**，作为独立后处理 | decode forward **之内**，逐层 |
| 36 层关系 | 同时可见 → **可融成一次 launch**（已做的 fusion） | 分散在 forward 的不同位置 → **天然只能逐层调** |
| launch 次数 | 1（融合后） | 每步每层各一次 |

本质约束：KV buffer replay 在模型 forward **外面**跑，36 层数据同时在手，所以能 batch
成单次 kernel；ReplaySSM 在 forward **里面**逐层执行，第 i 层 kernel 在计算图第 i 个
位置，**没法跨层 batch**——每步每层都是独立 launch（和 KV buffer 优化**之前**的逐层
replay 同形）。

### 7.4 频率 / 成本结构

| | KV buffer replay | ReplaySSM |
|---|---|---|
| 多久跑一次 | 每次 verify 一次（每个 accept 批） | 每个 decode step、每层 |
| 重建深度 | accept_len（≤ draft tokens，变长 cu_seqlens） | 固定窗口 L（默认 16），每 L 步 flush |
| 成本 ∝ | **$\text{batch} \times \text{accept\_len}$**（串行递推，随 batch 线性涨）| 每步固定 $L$ 深 matmul（被减半带宽掩盖）|
| 省的是什么 | verify 阶段不存 draft token 的中间 S（省显存）| decode 每步不写回整个 S（省带宽）|

### 7.5 layer 轴 vs token 轴：互补

| | layer 轴 | token 轴 |
|---|---|---|
| **KV buffer replay** | ✅ 可并行（post-hoc，层间 disjoint）→ 融单 launch | ❌ 串行（重跑递推，$\propto$ accept_len）|
| **ReplaySSM** | ❌ 串行（in-forward，层间链式依赖）| ✅ 并行（存 d → 闭式 → 一次 matmul）|

两者恰好在**相反的轴**上拿到并行：KV buffer 赢在 layer 轴（因为在 forward 外），
ReplaySSM 赢在 token 轴（因为存了 d）。**潜在改造**：若 KV buffer replay 也存 d，可在
保留 layer 并行的同时把 token 轴串行递推换成并行 matmul，两轴并行都吃到。

---

## 8. 效果与已知坑

- **正确性**：GDN fp32 在 $L=1$ bit-exact；$L>1$ 用 tensor-core 级容差。GSM8K 200 题
  flag-on 0.880 vs baseline 0.860（噪声内，无回归），GDN+KDA 统一后复测仍 parity；
  更早的 GDN-only 切片在 radix+CUDA graph 全开下 0.900 vs 0.850。GDN 的 $\alpha<1$ 会自然
  衰减掉旧重建误差，不跨 flush 累积。Greedy decode 与 baseline 逐位一致直到第一个近似
  平局 token，之后只因 benign TF32 字翻转分叉（两个续写都正确，非损坏）。
- **KDA 正确性**：kernel 单测对 `fused_recurrent_kda_packed_decode` 在同样 tensor-core
  容差下通过（L=1 / L>1 / forced-flush）；e2e plumbing 验证：server 起得来，MambaPool
  分配 per-K `g_cache[.,L,K]`（确认 is_kda 路由），decode 端到端不崩。
- **性能**：kernel microbench batch≥16 时 **$1.2$–$1.5\times$**（L=8）；batch1 略慢（overhead
  bound，非工作点）。e2e（128 并发，256/512 in/out）~2.3% TPOT 改善——model-dependent，
  这里小是因为 Qwen3.5-35B-A3B 是 MoE-heavy，GDN 线性层只占 decode 一小部分，GDN-dense
  模型收益更大。Roofline：H20-3e packed GDN decode 在 batch≥64 时 70–79% 带宽 bound，
  ReplaySSM 把 state 流量降到 $\sim 0.53\times$，高 batch 下 $\sim 1.7\times$ kernel 天花板。
- **已知坑（follow-up）**：`mamba_cache_per_req` **还没把 ReplaySSM ring 字节算进**
  显存预算 → KV pool 可能 over-size → 默认 `mem_fraction_static` 下 OOM，需手动调低
  （这与本仓库 §7.5 定位的 sizing 公式不感知 cache 差异是同一类问题）。KDA 的 radix
  协调也是 follow-up（目前只在自然 $L-1$ wrap flush）。需要
  `--linear-attn-decode-backend triton` + `mamba_scheduler_strategy=no_buffer`（默认）；
  extra_buffer/ping-pong 路径未覆盖、被 guard 掉。

---

## 9. 一句话总结

> ReplaySSM 在框架里就是给每个 mamba slot 挂了一个"最近 $L$ 步 (d,k,g) 环形缓冲 + 一个
> decode 位置游标"，把 GDN/KDA 的逐步状态写回，改成"每步从 checkpoint+ring 重建读出、
> 每 $L$ 步才落盘一次"；游标在 `_forward_metadata`/`_replay_metadata` 里每 forward 推进
> 一次、所有层共享，并用与 radix 完全相同的条件强制 flush 来保证缓存快照正确。
> 数学上靠"标量/对角衰减可因式分解 + cumsum 预算回放权重"把 $L$ 步递推规约成一次并行
> tensor-core matmul；存 d（而非 v）是为了切断串行依赖、解锁这个并行重建。

---

## 10. 补充问答：K-tile 主循环（:213）的变量与运算细节

> 本节针对 K-tile 主循环 `for kk in range(NK)`（:213-320）逐变量答疑，并澄清几个易混的
> 维度/运算类型问题。

### 10.1 主循环的整体作用

循环对应注释 :37-39 的"缓冲重建"公式：

$$
S = \mathrm{Diag}(A)\cdot S_0 + \sum_j d_j\cdot(W_j \odot k_j)^\top
$$

把 K 切成 `NK` 个分块（每块宽 `BKT`），逐块重建后立刻用 q、k 读出累加。好处：整个
$[V, K]$ 状态块**永不在 HBM 完整 materialize**，省内存省带宽。

### 10.2 进入循环前的初始化（:208-212）

| 变量 | 形状/类型 | 含义 |
|---|---|---|
| `b_state_q` | `[BV]` fp32, 初值0 | 累加 $S\cdot q$（输出主项 $o = S\cdot(a\odot q) + \dots$）|
| `b_state_k` | `[BV]` fp32, 初值0 | 累加 $S\cdot k$，用于 delta-rule 修正 $d_{cur} = \beta(v − S\cdot k)$ |
| `cur_kq` | `[1]` fp32, 初值0 | 累加当前 token 的 $k^\top q$（rank-1 项的标量系数）|
| `write_k` | bool | 本 program 是否负责把当前 key 写入 k ring（`非flush ∧ i_v==0 ∧ 是该 head 第一个 HV slot`，避免重复写）|
| `write_g_kda` | bool | KDA 下是否负责写当前 gate g（`IS_KDA ∧ 非flush ∧ i_v==0`）|

### 10.3 循环体逐变量（:214-320）

**① K-tile 索引（:214-216）**

| 变量 | 含义 |
|---|---|
| `kk` | 分块编号 0..`NK-1` |
| `o_kt` | `kk*BKT + arange(0,BKT)`，本块的 K 维绝对下标 `[BKT]` |
| `mask_kt` | `o_kt < K`，屏蔽超出 K 的 padding |
| `p_mix` | `mixed_qkv` 当前 token 行首指针 |

**② 当前 token 的 q/k（:217-229）**（`mixed_qkv` 是 `(q|k|v)` 打包，q 在前，k 偏移 `H*K`）

| 变量 | 含义 |
|---|---|
| `q_c` | 本块原始 query `[BKT]`，已乘 `q_rnorm` |
| `k_c` | 本块原始 key `[BKT]`，已乘 `k_rnorm` |
| `q_cs` | `q_c * scale` 缩放后 query |
| `cur_kq` | $+= \sum(k_c\cdot q_{cs})$，rank-1 项 $k^\top q$，**用原始 k/q**（gate 无关，两 gate 型一致，:32-33）|

**③ checkpoint S0（:231-241）**

| 变量 | 含义 |
|---|---|
| `p_h0_c` | `h0[state_idx, i_hv, o_v, o_kt]` 的 `[BV,BKT]` 块指针 |
| `b_h0_c` | S0 本块 `[BV,BKT]` fp32，**每步读但不写**（read-only 是本 kernel 关键，:21）|
| `p_k_c` | `k_cache[state_idx, i_h, o_c, o_kt]`，ring 内历史 key `[BC,BKT]` |

**④ GDN 重建（:248-255）**

| 变量 | 含义 |
|---|---|
| `b_k_all_c` | ring 内全历史 key `[BC,BKT]`，转 IO dtype（喂张量核）|
| `b_h_c` | `b_h0_c*b_total_decay + tl.dot(b_d_tc, b_k_all_c)`，重建状态块 `[BV,BKT]`；**该 `tl.dot` 跑张量核**是性能关键（:164-173）|
| `b_total_decay` | $\exp(\sum g_i)$，整个 ring 的总衰减**标量**（:154 算出）|
| `q_eff`/`k_eff` | GDN 标量衰减留到循环后乘，这里保持 `q_cs`/`k_c` |

**⑤ KDA 重建（:256-295）**（per-K 衰减，gate 在块内按 K 算）

| 变量 | 含义 |
|---|---|
| `b_g_all_c` | 本块 ring 内 gate `[BC,BKT]` |
| `b_g_prefix_c` | `cumsum(...,axis=0)` 包含式前缀和 `[BC,BKT]` |
| `b_g_total_c` | `sum(...,axis=0)` per-K 总衰减对数 `[BKT]` |
| `b_replay_decay_c` | `exp(b_g_total_c − b_g_prefix_c)`，per-K 回放权重 $W_j$ `[BC,BKT]` |
| `b_total_decay_c` | `exp(b_g_total_c)`，per-K 总衰减 `[BKT]` |
| `b_k_scaled` | `b_k_all_c * b_replay_decay_c`，回放衰减折进 key（GDN 折进 d，KDA 折进 k，:174-175）|
| `b_h_c` | `b_h0_c*b_total_decay_c + tl.dot(b_d_tc, b_k_scaled)` `[BV,BKT]` |
| `g_cur_c` | `-exp(A_log)*softplus(a+dt_bias)`，当前 token per-K 对数 gate `[BKT]` |
| `alpha_cur_c` | `exp(g_cur_c)`，当前 token per-K 衰减 |
| `q_eff`/`k_eff` | `q_cs*alpha_cur_c` / `k_c*alpha_cur_c`，KDA 把当前衰减折进 q/k |

KDA 分支内 `write_g_kda`（:298-304）：非 flush 时把 `g_cur_c` 写入 ring 的 `b_write_pos`。

**⑥ 读出（:307-308，两分支共用）**

```python
b_state_q += tl.sum(b_h_c * q_eff[None, :], axis=1)   # 跨块累加 S·q → [BV]
b_state_k += tl.sum(b_h_c * k_eff[None, :], axis=1)   # 跨块累加 S·k → [BV]
```

**⑦ 写当前 key（:310-320，仅非 flush）**：`write_k` 真时把 `k_c` 写到 `k_cache[..,b_write_pos,..]`。

### 10.4 `A_log` 与 `a` 的关系

二者是 gate 计算的**两个不同输入**，共同决定衰减 `alpha`（:135 / :142-145）：

$$
g = -\exp(\texttt{A\_log}) \cdot \mathrm{softplus}(a + \texttt{dt\_bias}); \qquad \alpha = \exp(g)
$$

| 参数 | 形状 | 性质 | 来源 |
|---|---|---|---|
| `A_log` | `[HV]` | **per-head 标量**学习参数，两 gate 型都一样（:66）。$\exp(\texttt{A\_log})$ 是恒正衰减率基底 | 模型权重 |
| `a` | GDN `[B,HV]` / KDA `[B,HV,K]` | **per-token** gate 输入（激活），随输入变化（:64）| 当前 token 激活 |
| `dt_bias` | GDN `[HV]` / KDA `[HV,K]` | time-step 偏置参数，加到 `a` 上 | 模型权重 |

- `A_log` 是全局、与 token 无关的衰减强度（每头一标量）；`a` 是动态的、随 token 变的输入。
- 关键：**`A_log` 永远是 per-head 标量，GDN/KDA 都一样**。GDN vs KDA 的差异只来自 `a`
  （和 `dt_bias`）的粒度：GDN 的 `a` 是标量 → `g`/`alpha` 标量；KDA 的 `a` 是 per-K 向量
  → 即便乘同一个标量 $\exp(\texttt{A\_log})$，`g`/`alpha` 也成 per-K 向量。所以 KDA 在 :284-293
  逐 K-tile 算 gate（因为 `a` 是 K-indexed）。

### 10.5 `b_total_decay` 的维度：KDA 是 `[BKT]`、GDN 是标量

- **KDA `b_total_decay_c` 是 `[BKT]`（per-K）**（:275）：`b_g_total_c = sum(b_g_all_c, axis=0)`，
  `b_g_all_c[BC,BKT]` 沿 ring 维 axis=0 求和后剩 `[BKT]`，每个 K 通道一个总衰减。:280 用
  时 broadcast 到 V 维：`b_h0_c * b_total_decay_c[None,:]`。
- **GDN `b_total_decay` 是标量**（:152-154，在循环**前**算）：`b_g_all[BC]` 只有一维
  （GDN 的 g_cache 是 `[slot,HV,L]` 无 K 维），sum 后是标量；:252 直接标量乘。

| | GDN | KDA |
|---|---|---|
| `g_cache` 形状 | `[slot,HV,L]` | `[slot,HV,L,K]` |
| 缓存 gate 加载 | `b_g_all` `[BC]` | `b_g_all_c` `[BC,BKT]` |
| 总衰减 | `b_total_decay` **标量** | `b_total_decay_c` **`[BKT]`** |
| 回放衰减 | `b_replay_decay` `[BC]`（循环前算）| `b_replay_decay_c` `[BC,BKT]`（循环内算）|
| 在哪算 | 循环**前**（:147-154，与 K 无关，只算一次）| 循环**内**（:256-275，K-indexed，每 tile 算）|

根本原因即 §10.4：GDN gate 与 K 无关 → 总衰减是标量、循环外算一次；KDA gate 是 per-K →
总衰减是 `[BKT]` 向量、每个 K-tile 内单独算。

### 10.6 `b_state_q`/`b_state_k` 的用途，以及 `cur_kq` 为何是逐元素乘

回顾单 token 输出公式（:28）：$o = S\cdot(a\odot q) + d_{cur}\cdot(k^\top q)$。

- **$b\_state\_q = S\cdot(a\odot q)$** 是输出主项。$S[V,K]\cdot q[K] \to [V]$ 是 matvec。代码 :307 用
  "逐元素乘 + 沿 axis=1（K）规约"实现，跨 `NK` 个 tile 累加成完整 $S\cdot q$。
- **$b\_state\_k = S\cdot(a\odot k)$** 用于 delta-rule 修正（:329 `b_d_cur = β·(v − b_state_k)`）：
  用 value 减去状态对 key 的"预测读出"得残差，再乘 $\beta$ = 写入状态的增量。

**`cur_kq += tl.sum(k_c * q_cs)` 是逐元素乘而非矩阵乘**，因为它算的是 $k^\top q$——两个向量的
**点积，结果是标量**：$k^\top q = \sum_i k[i]\cdot q[i]$ = 逐元素乘再求和。$k$、$q$ 都是 $[K]$ 向量，
没有矩阵参与，所以不需要 `tl.dot`。它是 rank-1 项的标量系数（:330
`b_o = b_state_q + b_d_cur*cur_kq`）。

**三种运算对比（关键判别法则）**：

| 代码位置 | 运算 | 数学含义 | 维度 |
|---|---|---|---|
| `cur_kq`（:229）| 逐元素乘+sum | 点积 $k^\top q$ | $[K]\cdot[K]\to$ 标量 |
| `b_state_q/k`（:307-308）| 逐元素乘+按轴规约 | matvec $S\cdot q$ | $[V,K]\cdot[K]\to[V]$ |
| `b_h_c` 里的 `tl.dot`（:252）| 矩阵乘（张量核）| 外积求和 $\sum d_j k_j^\top$ | $[V,L]@[L,K]\to[V,K]$ |

法则：**结果是标量 → 点积；结果是向量、只缩约一维 → matvec（也用逐元素乘+sum 实现）；
两矩阵相乘缩约公共维 → `tl.dot`**。`b_state_q/k` 因 q/k 是向量（matvec），广播逐元素乘
再沿 axis=1 规约即可，不必（也不便）调张量核。

### 10.7 为什么 decode 更新 S 和算 o 时没有矩阵乘

**核心：decode 每步只处理 1 个 token，时间维=1，所以状态更新天然是"矩阵×向量"和"向量
外积"，根本没有"矩阵×矩阵"。** 单 token 三个操作（:27-29）：

| 操作 | 公式 | 运算类型 | 维度 |
|---|---|---|---|
| 修正 δ | $d_{cur} = \beta(v − S\cdot k)$ | **matvec** | $[V,K]\cdot[K]\to[V]$ |
| 输出 | $o = S\cdot q + d_{cur}\cdot(k^\top q)$ | matvec + 点积 | $[V,K]\cdot[K]\to[V]$ |
| 状态更新 | $S_{new} = S\cdot \mathrm{Diag}(a) + d_{cur}\cdot k^\top$ | 逐元素缩放 + **rank-1 外积** | $[V]\otimes[K]\to[V,K]$ |

状态更新 $d_{cur}\cdot k^\top$：$d_{cur}[V]$ 外积 $k^\top[K]$ 得 $[V,K]$，是**秩 1** 更新（只有 1 token），
代码即逐元素广播乘（:370/:405 `alpha_val*b_h_c + b_d_cur[:,None]*k_c[None,:]`），无公共维
可缩约，所以不是 `tl.dot`。

**根本原因——token 维=1 让"矩阵乘"退化**。对比 prefill/chunked（一次处理 $T$ 个 token）：

$$
\begin{aligned}
\text{prefill:}\quad O &= Q \,@\, S^\top & Q[T,K] &\to [T,V] && \leftarrow \text{矩阵×矩阵} \\
S_{new} &= S + \Delta^\top \,@\, K & &\text{缩约 } T \text{ 维} && \leftarrow \text{矩阵×矩阵}
\end{aligned}
$$

prefill 时 token 维 $T$ 是真正要缩约的公共维 → 矩阵乘、compute-bound、适合张量核。decode 时
$T=1$，上面所有 $T$ 维坍缩成 1：$[T,K]@[K,V]\to$ matvec、$\Delta^\top @ K$ 缩约维 $T=1$ → 退化成单个 rank-1
外积。所以 decode 本质是 **matvec + rank-1 update**，算术强度低、**memory-bound**。这正是
所有 linear-attention/SSM decode 的固有性质，也是 ReplaySSM 去优化**带宽**而非算力的原因
（§1、§2.3）。

**那 kernel 里的 `tl.dot` 哪来的？** 是 ReplaySSM **人为制造**的矩阵乘：普通 decode 每步
做 1 个 rank-1 更新，ReplaySSM 攒 $L$ 步的 `(d_j,k_j)` 不立刻更新，重建时把这 **$L$ 个 rank-1
外积一次性合成一个 $[V,L]@[L,K]$ 矩阵乘**（:252），$L$（=`BC`，ring 深度）成了被缩约的公共维
→ 这才有了能跑张量核的矩阵乘（详见 §2.1）。

| 阶段 | token 维 | 状态更新本质 | 有无矩阵乘 |
|---|---|---|---|
| Prefill/Chunk | $T$ 个 | 矩阵乘（缩约 $T$ 维）| **有**，天然 compute-bound |
| 普通 decode | 1 个 | matvec + rank-1 外积 | **无**，memory-bound |
| ReplaySSM decode | 1 个，但攒 $L$ 步 | $L$ 个 rank-1 合成 `tl.dot` | **有**（重建处），人为造出来吃带宽红利 |

---

## 11. Code Review 实录：`copy_from` 的 "fully-flushed checkpoint" 不变量

> 来源：ReplaySSM PR 下 reviewer **@kaixih** 与作者 **@yuan-luo** 的一段 review 对话。
> 主题：ReplaySSM「**checkpoint(`temporal`) + ring**」双层状态在 **radix / prefix-cache
> 复用**时的一致性 bug，以及修复。这条不变量正是 memory_pool.py 里
> 「the SOURCE must be a fully-flushed checkpoint」注释的来由。

### 11.1 背景：为什么 `copy_from` 有前提

ReplaySSM 的状态分两层（见 §0、§4）：

- **`temporal`（checkpoint）**：只在 **flush** 时（每攒满 $L$ 步）更新一次。
- **ring `(d,k,g)`**：上次 flush 到现在的**增量**，`write_pos` 记录环里有几条未折叠的
  pending 更新。

**"当前真实状态" = `temporal` + ring**。只有 `write_pos==0`（刚 flush 完）时，`temporal`
才单独等于真实状态。

`copy_from`（COW 复制一个 slot）**只复制 `temporal`、不复制 ring**——它**假设源 slot 是
fully-flushed checkpoint**。若源 slot 还有 pending ring（`write_pos>0`），复制就**丢掉了
最后那几步增量**。

### 11.2 @kaixih 的担忧

`copy_from` 有两类调用方：

1. **radix-track 快照路径** —— 安全，因为快照前**强制 flush**，ring 已空。
2. **但还有调用方会复制"活跃请求"的 slot** —— 例如 prefix-cache insert 时
   `mamba_pool.copy_from(req.mamba_pool_idx, ...)` 保存当前请求状态。这些 slot 可能
   `write_pos>0`，于是**只保住 `temporal`、漏掉 buffered 更新**。

### 11.3 @yuan-luo 的逐 caller 排查结论

- **`cache_unfinished_req`（chunked prefill 触发）→ 安全**。`write_pos` 在 slot 分配时归
  0，**只在 decode 阶段才推进**；这条路径在 prefill 期间跑，ring 必为空，复制 `temporal`
  就够。

- **`cache_finished_req`（请求结束捐赠状态）→ 有 bug**。它在请求结束时按**完整 token 长度
  （非 flush 对齐）**把 slot 捐进 radix 树。而 finish 时 `write_pos` 通常 >0，于是捐进树
  的节点 **key 长度 > state 真实长度**。这是 `no_buffer` 下"decoded 状态进树"的唯一通道，
  普通 decode 因**每步都写 `temporal`** 而不受影响，唯独 ReplaySSM 的延迟写暴露此问题。

### 11.4 Bug 复现图解（L=16）

```
L = 16  (ring 最多攒 16 步 decode；flush 把 ring 折进 temporal，write_pos 归 0)

TURN 1   prompt=10, output=20, 结束于 token 30
──────────────────────────────────────────────────────────────
 tokens: [1..10]   [11.........26]   [27 28 29 30]
          prefill    decode 1..16      decode 17..20

  • prefill 后    : write_pos=0, temporal = state@10
  • decode 第16步 : ring 满 -> FLUSH -> temporal = state@26, write_pos=0
  • decode 17..20 : write_pos 0->1->2->3->4 (未满，不 flush)
  • 结束时        : temporal = state@26,  ring = {27,28,29,30}
                    └ temporal 落后真实状态 write_pos=4 个 token

 finish-donate 插入 radix 节点：
     key   = tokens[1..30]   -> 声称"30 个 token 后的状态"
     state = temporal        -> 实际只是 state@26
             key 说 30，但 state 是 26，ring 未复制 -> 27..30 丢失

TURN 2   prompt = [turn-1 prompt + turn-1 output + 新问题]
──────────────────────────────────────────────────────────────
 prefix = tokens[1..30] -> 精确命中 turn-1 的 finish 节点
 COW 复制 state@26 进 turn-2 的 slot (ring 被丢弃)
     -> turn-2 以为"在 token 30 之后"，实际是"token 26 之后"
     -> turn-1 答案最后 4 个 token 没进状态  => 输出错误
```

关键：bug 只在**多轮对话**才咬人——turn N+1 的 prefix 恰是 turn N 的完整 prompt+response，
正好落在那个长度对不上的 finish 节点上。

### 11.5 修复：把 finish-donate 截到上一个 flush 边界

```
cache_len = 30 - write_pos = 26
    key   = tokens[1..26]   ┐ 一致：state 长度 == key 长度
    state = state@26        ┘ token 27..30 不缓存，复用时重新 prefill
```

具体两步：
1. `cache_len -= write_pos`：捐赠长度砍掉未 flush 的尾巴。
2. **重置捐赠 slot 的游标**，让捐进树的 checkpoint 与其 key 长度精确相等。

代价：27..30 这几个 token 不进 prefix cache，turn-2 复用时对它们**重新 prefill**（正确但
略多算）。换来的是**绝不让"长度对不上的状态"进树**的正确性。

### 11.6 一句话总结

`copy_from` 只搬 `temporal` 不搬 ring，所以**只有 flush 对齐(`write_pos==0`)的 slot 才能
安全复制**。`cache_unfinished_req`（prefill 期 ring 空）没事；`cache_finished_req` 会把
`write_pos>0` 的 slot 按完整长度捐进 radix 树 → **key 长度 > state 真实长度** → 多轮复用
COW 出落后 `write_pos` 步的状态 → 输出错。修复是**捐赠长度截到上一个 flush 边界 + 重置游
标**，让 key 与 state 长度严格相等，丢掉的尾 token 靠复用时重新 prefill 补回。这条 review
确立了 memory_pool.py 中「源必须是 fully-flushed checkpoint」的不变量。

---

## 12. 显存流量估算：Baseline $8dn$ vs ReplaySSM $4dn$

> 量化 §1/§2.3 里"ReplaySSM 省的是带宽不是算力"——decode 是 memory-bound，主导成本是
> state 的 HBM 进出。这里给出每 head、每步的显存流量公式，说明 ReplaySSM 如何把主导项
> 减半。

### 12.1 符号与量级约定

- state $S$ 形状 $= d \times n$（$d$ = value head 维，$n$ = key/state 维），共 $d\cdot n$ 个元素。
- **state 是 4 字节（fp32）；激活（q/k/v、cached inputs、当前步 $d$）是 2 字节。**
- 量级关系：$d\cdot n$（矩阵）$\gg d$ 或 $n$（向量）$\gg$ 常数 →
  **只有 $d\cdot n$ 量级的项算"主导项"，$O(d+n)$ 向量项一律忽略。**

### 12.2 Baseline $= 8dn$

baseline 每步对 state 做"读 + 写"一整轮：

| 动作 | 量 | 字节 |
|---|---|---|
| **load state** $S$ | $d\cdot n \times 4\text{B}$ | $4dn$ |
| **store state** $S$（更新后写回） | $d\cdot n \times 4\text{B}$ | $4dn$ |
| load 输入 q/k/v | $\sim(n+n+d) \times 2\text{B}$ | $O(d+n)$，忽略 |

$$\text{Baseline}_{\text{dominant}} = \underbrace{4dn}_{\text{load}} + \underbrace{4dn}_{\text{store}} = 8dn$$

### 12.3 ReplaySSM $= 4dn$

ReplaySSM **不每步写回 full state**，只读 checkpoint + 缓存最近输入，把整状态写**推迟到每
$L$ 步 flush 一次**：

| 动作 | 量 | 字节 |
|---|---|---|
| **load state**（冻结 checkpoint，重建用） | $d\cdot n \times 4\text{B}$ | $4dn$ |
| load 缓存的最近 $k$ + 当前输入 | $\sim L(d+n) \times 2\text{B}$ | $O(d+n)$，忽略 |
| **store 当前步的 $d$**（rank-1 因子，非整状态） | $\sim d \times 2\text{B}$ | $O(d)$，忽略 |
| (整状态写仅在 flush 发生，每 $L$ 步一次 → 摊销 $4dn/L$) | | 忽略 |

$$\text{ReplaySSM}_{\text{dominant}} = \underbrace{4dn}_{\text{load}} + \underbrace{O(d+n) + \tfrac{4dn}{L}}_{\approx 0} \approx 4dn$$

### 12.4 为什么是"减半"

$$
\begin{aligned}
\text{baseline} &: \;\text{load } 4dn + \text{store } 4dn = 8dn \\
\text{ReplaySSM} &: \;\text{load } 4dn + \underbrace{(\text{只存小 } d)}_{\text{省掉每步 } 4dn \text{ 写回}} \approx 4dn
\end{aligned}
$$

**ReplaySSM 砍掉的正是 baseline 里"每步把整 state 写回 HBM"那 $4dn$**：整状态写被推迟到每
$L$ 步 flush 一次，摊销下来 $4dn/L$ 可忽略；缓存输入、当前 $d$ 都是 2 字节向量级，量级远
小于 $d\cdot n$。于是主导流量从 $8dn \to 4dn$，**对半砍**。

### 12.5 一句话

$d\cdot n$ 是 state 矩阵元素数、$4$ 是 fp32 字节数；baseline 每步对 state **读 $4dn$ + 写
$4dn = 8dn$**，ReplaySSM **只读 $4dn$、把写推迟到 flush 摊销掉**，主导流量减半到 $4dn$——
这正是它在 memory-bound 的 decode 上取胜的根源：省的是**带宽**，不是算力。

---

## 13. 缓存的精度（dtype）

> 来源：`python/sglang/srt/mem_cache/memory_pool.py`（ring 分配 `:443-471`、checkpoint
> `:435`）与 `python/sglang/srt/configs/mamba_utils.py`（`mamba2_state_dtype`，`:47-107`）。
> §12 的 "4 字节 state / 2 字节激活" 假设，对应的就是这里各张量的真实 dtype。

### 13.1 两个 dtype 来源

GDN 复用 Mamba2 cache 框架，所有缓存精度由 `mamba2_state_dtype()` 返回的两个 dtype 决定：

- **`ssm_dtype = dtype.temporal`**：SSM 递归状态精度。**代码默认 `float32`**。
- **`conv_dtype = dtype.conv`**：causal conv 状态精度。**默认 `bfloat16`**。

`ssm_dtype` 的三级优先级（低→高，`mamba_utils.py:70-103`）：
**默认 `float32`** ＜ 模型 config `mamba_ssm_dtype` ＜ 环境变量 `SGLANG_MAMBA_SSM_DTYPE`
（= CLI `--mamba-ssm-dtype`，可选 `float32 / bfloat16 / float16`）。

> 注意：**框架代码默认是 `float32`**；性能部署里常显式 `--mamba-ssm-dtype bfloat16`（甚至
> `float16`，见姊妹文档 spec-verify 的 fp16 修复）才变成 2 字节。§12 用 "4 字节" 估算正
> 是对着默认 fp32。

### 13.2 decode 版 ReplaySSM（#28451，`enable_linear_replayssm`）各张量精度

| 张量 | 含义 | dtype | 出处 |
|---|---|---|---|
| **checkpoint `h`**（`temporal_state`） | 冻结的完整状态 `[HV,V,K]` | **`ssm_dtype`**（默认 fp32） | `:435` |
| ring **`d`**（`replayssm_d`） | rank-1 写入因子 `[HV,L,V]` | **`ssm_dtype`**（默认 fp32） | `:451` |
| ring **`k`**（`replayssm_k`） | 历史 key `[H,L,K]` | **`ssm_dtype`**（默认 fp32） | `:456` |
| ring **`g`**（`replayssm_g`） | log-decay 门 | **`fp32`（始终）** | `:469` |

要点：

1. **ring 不单独存 `v`**。ReplaySSM 存的是已解耦的 $d = \beta(v − S\cdot\alpha k)$（rank-1 因子），不
   是原始 `v`——这也是它能把状态写省成 $O(d)$ 的原因（§12.3）。
2. **门 `g` 永远 fp32**。它要做 $\exp(\sum g)$ 这种长程累积衰减，对精度极敏感，绝不降精。
   GDN 是 per-head 标量门 → `[..,L]`；KDA 是 per-K 向量门 → `[..,L,K]`（`:462-466`）。
3. `d / k` 跟随 `ssm_dtype`：默认 fp32，部署常切 bf16。

### 13.3 为什么 SSM state 偏好 fp16 而非 bf16（精度选型）

同样 2 字节，比特分配不同（详见姊妹文档对 #26929 的分析）。规格化数有隐含前导 1 位，存储
$m$ 位尾数即有 $m+1$ 位有效精度，就近舍入（RNE）最大相对误差为半个 ulp：

$$\varepsilon_{\text{RNE}} = 2^{-(m+1)}$$

| 格式 | 尾数位 $m$ | 动态范围 | RNE 相对精度 $2^{-(m+1)}$ |
|---|---|---|---|
| fp32 | 23 | $\sim 10^{\pm 38}$ | $2^{-24} \approx 6\times10^{-8}$ |
| bf16 | 7 | $\sim 10^{\pm 38}$ | $2^{-8} \approx 3.9\times10^{-3}$ |
| fp16 | 10 | $6\times10^{-5} \sim 65504$ | $2^{-11} \approx 4.9\times10^{-4}$ |

SSM state **有界（门控收缩 + L2norm/sigmoid 限幅）但全程递归累积** → 用得上 fp16 的**尾
数精度**（$2^{-11}/2^{-8} = 2^{-3}$，比 bf16 细 8 倍），用不上 bf16 的**大动态范围**。所以
降精时 **fp16 优于 bf16**：实测长输出复读率 fp16 $1.7\%$ vs bf16 $7.9\%$。门 $g$ 因累积
衰减最敏感而保留 fp32。

### 13.4 一句话

decode 版 ReplaySSM 的缓存：**checkpoint `h`、ring `d`、ring `k` 跟随 `ssm_dtype`（代码默
认 fp32、部署常 bf16）；ring `g` 恒为 fp32；不存 `v`**。conv 状态另算（默认 bf16）。降精
时 SSM state 偏好 fp16（尾数精度）而非 bf16（动态范围用不上）。
