# K2_bwd 反向递推 kernel 逐行精讲（小白可推导版）

> 对应源码：`csrc/smxx/bwd_kernel2.cuh` 的 `_flash_kda_bwd_recurrence`
>
> 目标读者：没接触过自动微分 / CUDA kernel 的人。读完应能自己把每一条梯度公式推一遍，并理解代码为什么这么写。
>
> 约定：`d<量>` 表示 loss 对该量的梯度。代码里的 `exp(x)` 实际是 $2^x$（exp2），求导带的 $\ln 2$ 因子被统一推迟到 K1_bwd 处理，本文遇到时会注明。公式用 LaTeX，伪代码用代码块。

---

## 0. 先建立全局图景

### 0.1 FlashKDA 把序列切块做线性注意力

序列按 `CHUNK` 长度切块。**块内**用注意力，**块间**用一个状态矩阵 $S \in \mathbb{R}^{D\times D}$ 做递推。前向每块做：

$$
\begin{aligned}
v_{corr} &= (v - k_d\, S_{in}) \odot \beta \\
U &= \mathrm{INV}\, v_{corr}, \qquad \mathrm{INV} = (I+L)^{-1} \\
out &= q_d\, S_{in} + M_{qk}\, U \\
S_{new} &= S_{in}\odot \exp_2(g_T) + k_r^{\top} U
\end{aligned}
$$

反向就是对这一串求伴随（adjoint，即反向传播）。$\odot$ 表示按行/按元素广播乘。

### 0.2 形状与记号

| 符号 | 形状 | 含义 |
|------|------|------|
| `CHUNK` ($C$) | 标量 | 块长，例 16 |
| $D$ | 标量 | head 维度，例 128 |
| $S_{in}$ | $[D,D]$ | 进入本块时的状态。**存储是转置的**，见 0.3 |
| $k_d,q_d,k_r,k_i$ | $[C,D]$ | 前向 K1 生成的派生量（fp32） |
| $g_T$ | $[D]$ | 整块门控总和，workspace 里存 $\exp_2(g_T)$ |
| $\beta$ | $[C]$ | 每行一个门控标量（已过 sigmoid） |
| $\mathrm{INV}$ | $[C,C]$ | $(I+L)^{-1}$，下三角 |
| $M_{qk}$ | $[C,C]$ | $\mathrm{tril}(q_d k_i^{\top})$ |
| $v, do$ | $[C,D]$ | 输入 $v$；$do$ 是输出 $out$ 的上游梯度 |
| $U, v_{corr}$ | $[C,D]$ | 中间量（前向没存，反向重算） |
| $dS$ | $[D,D]$ | 跨块传播的状态梯度累加器（常驻 smem） |

### 0.3 状态的转置存储（极其重要，否则下标全错）

`all_states` / `ds_init` / `ds_out` 在显存里存的是 $S^{\top}$，布局 $[V,K]$。即：

$$
S[k][d] \quad\longleftrightarrow\quad \texttt{s\_in\_ptr}[\,d\cdot D + k\,]
$$

源码中凡出现 `s_in_ptr[d*D + k]`，语义都是数学上的 $S[k][d]$。

### 0.4 为什么必须从后往前

第 $t$ 块入口的状态梯度 $dS_{in}(t)$ 依赖第 $t{+}1$ 块算出的 $dS$。所以反向必须 **从最后一块往第一块** 迭代（`for t = t_tiles-1 ... 0`）。

---

## 1. Kernel 启动配置与线程模型（line 29-70）

```cpp
template <int CHUNK, int D, int NumThreads, bool IsVarlen = true>
__global__ void __launch_bounds__(NumThreads) _flash_kda_bwd_recurrence(...)
```

- **Grid = (N, H)**：`blockIdx.x = seq_idx`，`blockIdx.y = head_idx`。每个 block 独占一条序列的一个 head。
- **Block = 256 线程**，无 warp specialization（注释 line 8）。
- 纯标量的 shared-memory GEMM 循环，不用 tensor core；逻辑直白但有冗余重算（用算力换显存）。

---

## 2. 序列定位（line 72-89）

把 `seq_idx` 映射到 token 维范围 $[bos, eos)$ 和全局 tile 偏移 `tile_base`。

- **Varlen**：用 `cu_seqlens` 取 `bos/eos`；`tile_base` 累加前面序列的块数。
- **定长**：直接乘除。

$$
\text{seq\_len} = eos - bos,\qquad
\text{t\_tiles} = \left\lceil \tfrac{\text{seq\_len}}{C} \right\rceil
$$

---

## 3. 共享内存布局与 dS 初始化（line 91-117）

```cpp
float* dS_smem     = (float*)shared_mem;   // [D*D], 整个 kernel 常驻
float* scratch_f32 = dS_smem + D*D;        // 后面所有临时 tile 堆这里
```

$dS$ 初值来自 `dfinal_state`（对最终状态的梯度），没有则为 0。转置加载：

$$
dS\_smem[k\cdot D + v] = \texttt{ds\_init}[v\cdot D + k]
$$

---

## 4. 主循环：加载本块数据（line 120-177）

```cpp
for (int t = t_tiles - 1; t >= 0; --t) {
    int ws_idx     = head_idx * total_tiles + tile_base + t;
    int actual_len = min(CHUNK, seq_len - t*CHUNK);   // 尾块可能不满
```

- `actual_len`：尾块真实行数；越界行读 $v/do$ 时取 0。
- 把 $g_T$（已 exp2）、$\beta$（读原始再过 `sigmoid_tanh_approx`）、$\mathrm{INV}$、$M_{qk}$ 搬进 smem。
- scratch 区堆叠顺序（贯穿全 kernel）：

```
gT[D] | beta[C] | INV[C*C] | Mqk[C*C] | vcorr[C*D] | U[C*D] | dU[C*D] | dkr[C*D] | dvcorr[C*D] | dL[C*C]
```

多数 buffer 复用别名，唯独 `dL` 独立（见第 8 节）。

---

## 5. Step 1：重算 U（line 179-235）

前向的 $U,v_{corr}$ 未存，反向重算。

$$
k_dS[c][d] = \sum_{k} k_d[c][k]\, S[k][d],\qquad S[k][d]=\texttt{s\_in\_ptr}[d D + k]
$$

$$
v_{corr}[c][d] = \big(v[c][d] - k_dS[c][d]\big)\,\beta[c]
$$

$$
U[c][d] = \sum_{j} \mathrm{INV}[c][j]\, v_{corr}[j][d]
$$

> 5.1 的 $k_d S$ 是 $C\cdot D\cdot D$ 次乘加，最重。`vcorr_smem` 复用 `kd_S_smem`。

---

## 6. Step 2：输出梯度反传（line 237-303）

前向 $out = q_d S_{in} + M_{qk} U$，分两路。

### 6.1 dU 来自 $M_{qk}$ 路（line 251-267）

由 $out[c][d] = \sum_j M_{qk}[c][j]\,U[j][d]$，对 $U[j][d]$ 求偏导得 $M_{qk}[c][j]$：

$$
dU^{(mqk)}[j][d] = \sum_{c} M_{qk}[c][j]\, do[c][d] \quad(= M_{qk}^{\top} do)
$$

### 6.2 dU 加状态路贡献（line 269-287）

由 $S_{new}[k][d] \mathrel{+}= \sum_c k_r[c][k]\,U[c][d]$，$U$ 也影响 loss：

$$
dU[c][d] \mathrel{+}= \sum_{k} k_r[c][k]\, dS[k][d] \quad(= k_r\, dS)
$$

其中 $dS$ 是**后一块传回**的状态梯度。

### 6.3 dkr（line 289-303）

对同一式中的 $k_r[c][k]$ 求偏导得 $U[c][d]$：

$$
dk_r[c][k] = \sum_{d} dS[k][d]\, U[c][d]
$$

---

## 7. Step 3：三角求解的伴随（line 305-321）—— 重点之一

前向 $U = \mathrm{INV}\,v_{corr}$，$\mathrm{INV}=(I+L)^{-1}$，等价解 $(I+L)U = v_{corr}$。

> 本节是全文最难的一处。下面 **7.0 是写给线性代数基础薄弱读者的从零详解**，已经熟悉矩阵微分的读者可直接跳到 7.1 看结论。

### 7.0 从零搭起：只用一条核心规则

#### (a) 三个积木

**积木 1 — 矩阵乘法** $Y = A\,X$：$Y$ 的第 $i$ 行第 $j$ 列 = "A 的第 $i$ 行"点乘"X 的第 $j$ 列"。

$$
Y[i][j] = \sum_k A[i][k]\,X[k][j]
$$

一句话：matmul 就是一堆点积。

**积木 2 — 转置** $A^{\top}$：行列互换，$A^{\top}[i][j] = A[j][i]$（沿对角线翻一下）。

**积木 3 — 逆矩阵** $A^{-1}$：矩阵版的"倒数"，定义 $A\,A^{-1}=I$（$I$ 是单位矩阵，相当于数字 1）。你不需要会手算它，只要知道：

$$
A\,U = b \quad\Longleftrightarrow\quad U = A^{-1} b
$$

即"解线性方程组"和"求逆再乘"是同一件事。本例 $U=\mathrm{INV}\,v_{corr}$ 就等价于解 $(I+L)U=v_{corr}$。

#### (b) 反向传播唯一要背的核心规则

$$
\boxed{\;\text{前向 } Y = A\,X \;\Rightarrow\; dX = A^{\top} dY,\qquad dA = dY\,X^{\top}\;}
$$

其中 $dY$ 是"loss 对 $Y$ 的梯度"（与 $Y$ 同形状）。**推导**（只用积木 1 的点积定义 + 链式法则）：

由 $Y[i][j]=\sum_k A[i][k]X[k][j]$，

$$
dX[k][j] = \sum_i dY[i][j]\,\frac{\partial Y[i][j]}{\partial X[k][j]} = \sum_i A[i][k]\,dY[i][j] = \sum_i A^{\top}[k][i]\,dY[i][j] = (A^{\top}dY)[k][j]
$$

$$
dA[i][k] = \sum_j dY[i][j]\,\frac{\partial Y[i][j]}{\partial A[i][k]} = \sum_j dY[i][j]\,X[k][j] = (dY\,X^{\top})[i][k]
$$

**转置就是这么冒出来的。** 口诀：求输入 $X$ 的梯度，就把另一个因子 $A$ 转置乘到 $dY$ 左边；求权重 $A$ 的梯度，就把 $X$ 转置乘到 $dY$ 右边。

#### (c) 标量类比（建立"为什么有逆/转置"的直觉）

把矩阵换成普通数字，看除法 $u=b/a=a^{-1}b$：

$$
db = \frac{du}{a} = a^{-1}du,\qquad da = -\frac{u}{a}\,du = -db\cdot u
$$

记住这两条标量结果，矩阵版长得一模一样（只是 $1/a$ 变 $A^{-1}$、注意左右与转置）。

#### (d) 第一步：求 $dv_{corr}$ —— 直接套核心规则

前向 $U=\mathrm{INV}\,v_{corr}$ 就是 $Y=A\,X$，对应 $A=\mathrm{INV},\ X=v_{corr},\ Y=U$。求"对输入 $X$ 的梯度"：

$$
dv_{corr} = \mathrm{INV}^{\top} dU \qquad\Longleftrightarrow\qquad db = a^{-1}du
$$

#### (e) 第二步：求 $dL$ —— 需要"逆矩阵的梯度"这块拼图

$L$ 藏在 $\mathrm{INV}=(I+L)^{-1}$ 里，要多走一步。

先用核心规则求"对权重 $\mathrm{INV}$ 的梯度"：

$$
d\mathrm{INV} = dU\,v_{corr}^{\top}
$$

再用**逆矩阵反向规则**（标量 $u=1/a\Rightarrow da=-(1/a)\,du\,(1/a)$ 的矩阵版，推导见 (g)）：

$$
M=A^{-1},\ \text{已知 } dM \;\Rightarrow\; dA = -A^{-\top}\,dM\,A^{-\top}
$$

套 $A=I+L,\ M=\mathrm{INV}$，且 $d(I+L)=dL$：

$$
dL = -\mathrm{INV}^{\top}\,d\mathrm{INV}\,\mathrm{INV}^{\top}
$$

#### (f) 化简到代码里的简洁式

把 $d\mathrm{INV}=dU\,v_{corr}^{\top}$ 代入并重新分组（结合律）：

$$
dL = -\big(\mathrm{INV}^{\top}dU\big)\big(v_{corr}^{\top}\mathrm{INV}^{\top}\big)
   = -\underbrace{\big(\mathrm{INV}^{\top}dU\big)}_{=\,dv_{corr}}\underbrace{\big(\mathrm{INV}\,v_{corr}\big)^{\top}}_{=\,U^{\top}}
   = -\,dv_{corr}\,U^{\top}
$$

（用了 $(AB)^{\top}=B^{\top}A^{\top}$，以及 $U=\mathrm{INV}\,v_{corr}$。）于是得到

$$
dL = -\,dv_{corr}\,U^{\top} \qquad\Longleftrightarrow\qquad da = -db\cdot u
$$

又和标量版结构完全对应。

#### (g)（选读）逆矩阵反向规则的来历

由 $A\,M=I$ 求微分：$(dA)M + A(dM)=0 \Rightarrow dM=-A^{-1}(dA)A^{-1}$；再转成伴随（内积配对，见 7.1）得 $dA=-A^{-\top}dM\,A^{-\top}$。

> **小结**：第一步 $dv_{corr}=\mathrm{INV}^{\top}dU$，第二步 $dL=-dv_{corr}\,U^{\top}$。全程只用了"matmul 反向"这一条核心规则 + 一次"逆矩阵反向"。和标量 $u=b/a$ 的 $db=du/a,\ da=-db\,u$ 一一对应。

### 7.1 通用公式（务必记住）

对 $Y = A^{-1}B$，已知 $\bar Y$（即 $dY$）：

$$
\boxed{\;\bar B = A^{-\top}\bar Y,\qquad \bar A = -\,\bar B\, Y^{\top}\;}
$$

**推导**（内积配对）。对 $Y=A^{-1}B$ 取全微分，用 $d(A^{-1})=-A^{-1}(dA)A^{-1}$：

$$
dY = -A^{-1}(dA)\,Y + A^{-1}\,dB
$$

loss 一阶变化 $\langle \bar Y, dY\rangle$，其中 $\langle X,Y\rangle=\sum_{ij}X_{ij}Y_{ij}$：

$$
\langle \bar Y, A^{-1}dB\rangle = \langle A^{-\top}\bar Y,\, dB\rangle
\;\Rightarrow\; \bar B = A^{-\top}\bar Y
$$

$$
\langle \bar Y, A^{-1}(dA)Y\rangle = \langle (A^{-\top}\bar Y)\,Y^{\top},\, dA\rangle
\;\Rightarrow\; \bar A = -(A^{-\top}\bar Y)\,Y^{\top}
$$

### 7.2 套进本例（$Y=U,\,B=v_{corr},\,A=I+L,\,\bar Y=dU$）

$$
dv_{corr} = \mathrm{INV}^{\top} dU,\qquad
dv_{corr}[c][d] = \sum_{j}\mathrm{INV}[j][c]\, dU[j][d]
$$

---

## 8. Step 3b：dL（line 324-338）—— 重点之二

由 7.1 的 $\bar A = -\bar B\,Y^{\top}$，且 $A=I+L \Rightarrow dA = dL$：

$$
dL = -\,dv_{corr}\,U^{\top},\qquad
dL[i][j] = -\sum_{d} dv_{corr}[i][d]\,U[j][d]
$$

（这条公式的完整从零推导见 7.0(e)(f)；直觉：它是"输出敏感度 $dv_{corr}$"与"解 $U$"的外积、带负号，正对应标量 $da=-db\cdot u$。）

$L$ 严格下三角（上三角与对角前向被 mask 成 0，那些位置根本不是自由参数），故梯度也 mask：

$$
dL[i][j] = \begin{cases} -\sum_d dv_{corr}[i][d]\,U[j][d], & i>j\\[2pt] 0, & i\le j\end{cases}
$$

**工程关键**：`dL` 依赖 `dvcorr`，而 `dvcorr_smem` 在 Step 5 会被覆盖。所以必须趁 `dvcorr`、`U` 还在时立刻把 `dL` 算进**独立不别名**的 `dL_smem`（line 309 注释 "not aliased"）。这就是它被命名为 "Step 3b"、紧贴 Step 3a 之后的原因。

---

## 9. Step 4：vcorr 反传到 dv、dbeta（line 341-382）

前向 $v_{corr} = (v - k_dS)\odot\beta$。

$$
dv[c][d] = dv_{corr}[c][d]\,\beta[c]
$$

$$
d\beta[c] = \sum_{d} dv_{corr}[c][d]\,(v-k_dS)[c][d]
          = \sum_{d} dv_{corr}[c][d]\,\frac{v_{corr}[c][d]}{\beta[c]}
$$

代码仅在 $\beta[c] > 10^{-8}$ 时做除法（line 375）。

---

## 10. Step 5-6：dMqk → dqd、dki（line 384-451）

前向 $M_{qk}=\mathrm{tril}(q_d k_i^{\top})$，$out \mathrel{+}= M_{qk}U$。

$$
dM_{qk}[c][j] = \sum_{d} do[c][d]\,U[j][d] \quad (c\ge j,\ \text{否则 }0)
$$

**dqd 两路**（line 411-437）：

$$
dq_d[c][k] = \underbrace{\sum_{d} do[c][d]\,S[k][d]}_{\text{跨块 } out=q_dS_{in}}
           + \underbrace{\sum_{j\le c} dM_{qk}[c][j]\,k_i[j][k]}_{\text{块内 } M_{qk}=\mathrm{tril}(q_dk_i^\top)}
$$

**dki 来自 Mqk 路**（line 439-451）：

$$
dk_i[j][k] = \sum_{c\ge j} dM_{qk}[c][j]\,q_d[c][k] \quad (= dM_{qk}^{\top} q_d)
$$

---

## 11. Step 7：L 路反传 → dki、dkd、dbeta（line 453-535）

前向 $L = \mathrm{tril}(k_d k_i^{\top}, -1)\odot\beta$。记 $F = \mathrm{tril}(k_d k_i^{\top},-1)$，则 $L = F\odot\beta_{[\text{行}]}$。

**dki 加 L 贡献**（line 472-485）：

$$
dk_i^{(L)}[j][k] = \sum_{c>j} dL[c][j]\,\beta[c]\,k_d[c][k]
$$

**dkd 两路**（line 487-518）：

$$
dk_d[c][k] = \underbrace{-\beta[c]\sum_{d} dv_{corr}[c][d]\,S[k][d]}_{v_{corr}=(v-k_dS)\beta}
           + \underbrace{\sum_{j<c} dL[c][j]\,\beta[c]\,k_i[j][k]}_{L=\mathrm{tril}(k_dk_i^\top,-1)\beta}
$$

> 此时 `dvcorr_smem` 已被覆盖，但 `inv_smem`、`dU_smem` 还在，于是就地重算 $dv_{corr}[c][d]=\sum_j \mathrm{INV}[j][c]\,dU[j][d]$（line 500-505）。

**dbeta 加 L 贡献**（line 520-534）：

$$
d\beta^{(L)}[c] = \sum_{j<c} dL[c][j]\,F[c][j],\qquad
F[c][j] = \sum_{k} k_d[c][k]\,k_i[j][k]
$$

---

## 12. Step 8：dgT 与 dS 递推更新（line 543-625）

### 12.1 dgT（必须在更新 dS 之前算，line 567-592）

$g_T$ 有两个去处：状态衰减 $S_{new}=S_{in}\exp_2(g_T)+\dots$，以及 $k_r=k_n\exp_2(g_T-g_c)$：

$$
dg_T[k] = \underbrace{\exp_2(g_T[k])\sum_{d} dS[k][d]\,S_{in}[k][d]}_{\text{状态衰减}}
        + \underbrace{\sum_{c} dk_r[c][k]\,k_r[c][k]}_{k_r \text{ 依赖}}
$$

代码里 `gT_smem[k]` 已是 $\exp_2(g_T[k])$。用的是**当前(后块传来) dS**，故必须先于 12.2。

> 下面 12.1.0 是从零详解（$g_T$ 是个长度 $D$ 的向量，$g_T[k]$ 扇出到很多地方，逐路收集）；已熟悉的读者可跳过。

#### 12.1.0 从零推导 dgT

**(a) 先看 $g_T[k]$ 在前向被用在哪。** $g_T$ 是长度 $D$ 的向量（每个 head 维一个数）。$g_T[k]$ 出现在两个地方，所以它的总梯度 = 两路回传之和（扇出 → 求和，多元链式法则）：

- **路 1 — 状态衰减**：$S_{new}[k][d] = S_{in}[k][d]\,\exp_2(g_T[k]) + \dots$（注意只有**第 $k$ 行**乘 $\exp_2(g_T[k])$，因为衰减是按 $k$ 这一维广播的）。
- **路 2 — 进入 $k_r$**：$k_r[c][k] = k_n[c][k]\,\exp_2(g_T[k]-g_c[c][k])$（每个 $c$ 都用到同一个 $g_T[k]$）。

**(b) 路 1 的偏导。** 固定 $k$，$g_T[k]$ 影响第 $k$ 行所有列 $d$ 的 $S_{new}[k][d]$。用指数求导 $\frac{d}{dx}2^{x}=2^{x}\ln 2$（这里同样把公因子 $\ln 2$ 按"dgT 约定"省略，留到 K1_bwd 统一处理）：

$$
\frac{\partial S_{new}[k][d]}{\partial g_T[k]} \sim S_{in}[k][d]\,\exp_2(g_T[k])
$$

乘上游 $dS[k][d]$ 并对所有列 $d$ 求和（因为 $g_T[k]$ 影响了整行）：

$$
dg_T^{(1)}[k] = \exp_2(g_T[k])\sum_{d} dS[k][d]\,S_{in}[k][d]
$$

代码里 `gT_smem[k]` 已经是 $\exp_2(g_T[k])$，所以源码先 `sum_d dS*S_in` 再 `*= gT_k`（line 581-583）。

**(c) 路 2 的偏导。** $g_T[k]$ 出现在每个 $c$ 的 $k_r[c][k]$ 里，同样 $\frac{\partial k_r[c][k]}{\partial g_T[k]} \sim k_r[c][k]$（因 $k_r\propto 2^{g_T[k]}$）。乘上游 $dk_r[c][k]$ 并对所有行 $c$ 求和：

$$
dg_T^{(2)}[k] = \sum_{c} dk_r[c][k]\,k_r[c][k]
$$

这里的 $dk_r$ 是本块第 6.3 节刚算出的（`dkr_smem`），$k_r$ 从 workspace 读。

**(d) 两路相加。**

$$
dg_T[k] = dg_T^{(1)}[k] + dg_T^{(2)}[k]
        = \exp_2(g_T[k])\sum_{d} dS[k][d]\,S_{in}[k][d] + \sum_{c} dk_r[c][k]\,k_r[c][k]
$$

**(e) 为什么必须"先于 12.2"。** 路 1 用的 $dS$ 是**进入本块时（后一块传回来）的** $dS$；而 12.2 会把 `dS_smem` 原地覆盖成"前一块的 $dS$"。若先跑 12.2，路 1 就会读到错的 $dS$。所以源码严格按 `Step 8a (dgT) → Step 8b (更新 dS)` 的顺序（line 567 注释 "compute dgT BEFORE updating dS"）。

> **符号小提醒**：$g_T$ 和 $g_c$ 都来自门控，$k_r$ 的指数是 $g_T-g_c$。所以 $g_T[k]$ 增大让 $k_r$ 变大（$+$ 号，本节路 2）；而 $g_c$ 增大让 $k_r$ 变小（$-$ 号，那部分梯度走 $dg_c$，在 K1_bwd 处理，见 3.2(b)）。同一个 $k_r$ 对 $g_T$ 和 $g_c$ 贡献符号相反，别搞混。

### 12.2 dS 反向递推（line 594-625）

把 $dS$ 更新为前一块的状态梯度，三项相加：

$$
dS_{prev}[k][d] = \underbrace{\exp_2(g_T[k])\,dS[k][d]}_{\text{衰减}}
                + \underbrace{\sum_{c} q_d[c][k]\,do[c][d]}_{out=q_dS_{in}}
                - \underbrace{\sum_{c} k_d[c][k]\,\beta[c]\,dv_{corr}[c][d]}_{v_{corr}\text{ 里的 }-k_dS}
$$

> 三项偏导依据：
> $\partial S_{new}[k][d]/\partial S_{in}[k][d]=\exp_2(g_T[k])$；
> $\partial out[c][d]/\partial S_{in}[k][d]=q_d[c][k]$；
> $\partial v_{corr}[c][d]/\partial S[k][d]=-k_d[c][k]\beta[c]$。

`dvcorr` 就地重算。写回 `dS_smem`，进入更前一块。

---

## 13. 收尾：写出 d_initial_state（line 628-641）

循环结束时 `dS_smem` 即第一块入口的状态梯度。转置回 $[V,K]$ 写出：

$$
\texttt{ds\_out}[v\cdot D + k] = dS\_smem[k\cdot D + v]
$$

---

## 14. 全部梯度公式速查表

$$
\begin{aligned}
k_dS[c][d] &= \textstyle\sum_k k_d[c][k]\,S[k][d] \\
v_{corr}[c][d] &= (v[c][d]-k_dS[c][d])\,\beta[c] \\
U[c][d] &= \textstyle\sum_j \mathrm{INV}[c][j]\,v_{corr}[j][d] \\[4pt]
dU[j][d] &= \textstyle\sum_c M_{qk}[c][j]\,do[c][d] + \sum_k k_r[j][k]\,dS[k][d] \\
dk_r[c][k] &= \textstyle\sum_d dS[k][d]\,U[c][d] \\
dM_{qk}[c][j] &= \textstyle\sum_d do[c][d]\,U[j][d] \quad(c\ge j) \\[4pt]
dv_{corr}[c][d] &= \textstyle\sum_j \mathrm{INV}[j][c]\,dU[j][d] \\
dL[i][j] &= -\textstyle\sum_d dv_{corr}[i][d]\,U[j][d] \quad(i>j) \\[4pt]
dv[c][d] &= dv_{corr}[c][d]\,\beta[c] \\
d\beta[c] &= \textstyle\sum_d dv_{corr}[c][d]\,\frac{v_{corr}[c][d]}{\beta[c]} + \sum_{j<c} dL[c][j]\,F[c][j] \\
dq_d[c][k] &= \textstyle\sum_d do[c][d]\,S[k][d] + \sum_{j\le c} dM_{qk}[c][j]\,k_i[j][k] \\
dk_i[j][k] &= \textstyle\sum_{c\ge j} dM_{qk}[c][j]\,q_d[c][k] + \sum_{c>j} dL[c][j]\,\beta[c]\,k_d[c][k] \\
dk_d[c][k] &= -\beta[c]\textstyle\sum_d dv_{corr}[c][d]\,S[k][d] + \sum_{j<c} dL[c][j]\,\beta[c]\,k_i[j][k] \\
dg_T[k] &= \exp_2(g_T[k])\textstyle\sum_d dS[k][d]\,S_{in}[k][d] + \sum_c dk_r[c][k]\,k_r[c][k] \\[4pt]
dS_{prev}[k][d] &= \exp_2(g_T[k])\,dS[k][d] + \textstyle\sum_c q_d[c][k]\,do[c][d] - \sum_c k_d[c][k]\,\beta[c]\,dv_{corr}[c][d]
\end{aligned}
$$

---

## 15. 设计要点总结

| 主题 | 做法 |
|------|------|
| 从后往前 | $dS$ 依赖后块，必须逆序迭代 |
| 重算换显存 | $U/v_{corr}/dv_{corr}$ 不存，靠 $\mathrm{INV},dU$ 在 smem 重算 |
| 状态转置存储 | all_states/ds_init/ds_out 都是 $[V,K]$；取 $S[k][d]$ 写 `s_in_ptr[d*D+k]` |
| buffer 复用与别名 | scratch 严格堆叠；dMqk 复用 Mqk，dqd/dki 复用 dvcorr；唯 dL 独立 |
| gT 已是 exp2 | workspace 存 $\exp_2(g_T)$ |
| 尾块掩码 | actual_len 控制越界行读 0 |
| 数值 | workspace fp32；dbeta 除 $\beta$ 时加 $10^{-8}$ |

> 配套阅读：K1_bwd 的逐行精讲见 [`k1_bwd_prepare_deep_dive.md`](./k1_bwd_prepare_deep_dive.md)。本 kernel 产出的 $dk_d/dq_d/dk_i/dk_r/dg_T/dv/d\beta$ 正是 K1_bwd 的输入。
