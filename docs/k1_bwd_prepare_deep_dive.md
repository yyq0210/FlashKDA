# K1_bwd "prepare" kernel 逐行精讲（小白可推导版）

> 对应源码：`csrc/smxx/bwd_kernel1.cuh` 的 `_flash_kda_bwd_prepare`
>
> 目标读者：没接触过自动微分 / CUDA kernel 的人。读完应能自己推出 L2 归一化反向、数值稳定的 $dg_c$、以及门控的逆向 cumsum，并理解代码为何这样写。
>
> 约定：`d<量>` 表示 loss 对该量的梯度。`exp(x)` 实际是 $2^x$（exp2）。公式用 LaTeX，伪代码用代码块。

---

## 0. 它在整条反向链里的位置

```
Backward:  K2_bwd (反向递推, 后->前)  ->  K1_bwd (本文, 还原输入梯度)
```

K2_bwd 算出了一堆"派生量的梯度" $dk_d, dq_d, dk_i, dk_r, dg_T, dv, d\beta_{chunk}$。
K1_bwd 是**前向 K1 的逆过程**：把这些梯度还原回**原始输入**的梯度——
$dq, dk, dv, dg$（门）、$d\beta$，以及参数 $d(A\_\mathrm{log})$、$d(dt\_\mathrm{bias})$。

### 0.1 前向 K1 做了什么（理解反向的前提）

设每行（一个 token）的原始 $q, k \in \mathbb{R}^D$，门控相关原始量 $g_{raw}$、参数 $A_{\log}$、$dt_{bias}$。前向：

$$
\begin{aligned}
q_n &= q / \lVert q\rVert, \qquad k_n = k / \lVert k\rVert \quad(\text{L2 归一化}) \\
a &= \exp(A_{\log}) \\
z &= a\,(g_{raw} + dt_{bias}) \\
g_{total} &= \texttt{gate\_scale}\cdot \sigma(z) \quad(\sigma=\text{sigmoid}) \\
g_c[\text{row}] &= \textstyle\sum_{r\le \text{row}} g_{total}[r] \quad(\text{块内前缀和 / cumsum}) \\
g_T &= \textstyle\sum_{r} g_{total}[r] \quad(\text{整块总和}) \\[4pt]
q_d &= q_n \cdot \exp_2(g_c)\cdot \texttt{scale} \\
k_d &= k_n \cdot \exp_2(g_c) \\
k_i &= k_n \cdot \exp_2(-g_c) \\
k_r &= k_n \cdot \exp_2(g_T - g_c)
\end{aligned}
$$

所以反向要做三件事：把 $dk_d/dq_d/dk_i/dk_r$ 收拢成 $dq_n/dk_n$ 和 $dg_c$；把 $dq_n/dk_n$ 过 L2 反向得 $dq/dk$；把 $dg_c$（+$dg_T$）过 cumsum 和门函数得 $dg/dA_{\log}/d(dt_{bias})$。

---

## 1. 线程模型（line 10-62）

```cpp
constexpr int ELEMS_PER_THREAD = 8;
constexpr int THREADS_PER_ROW  = D / 8;   // = 16
```

- **Grid = (total_tiles, H)**：`blockIdx.x = global_tile_idx`，`blockIdx.y = head_idx`。每个 CTA 处理一个 chunk 的一个 head（line 7）。
- **每行 16 个线程，每线程负责 8 个元素**：

$$
\text{my\_row} = \lfloor tid/16\rfloor,\qquad
\text{my\_col} = (tid \bmod 16)\cdot 8
$$

这个布局**和前向 K1 完全一致**（line 60），保证归约顺序相同 → 数值可复现。

---

## 2. tile → 序列定位（line 64-99）

把 `global_tile_idx` 映射到 `seq_idx / local_t`，并算出：

$$
\begin{aligned}
\text{ws\_idx} &= \text{head\_idx}\cdot \text{total\_tiles} + \text{global\_tile\_idx} \\
\text{t\_start} &= bos + \text{local\_t}\cdot C \\
\text{actual\_len} &= \min(C,\ \text{seq\_len} - \text{local\_t}\cdot C)
\end{aligned}
$$

越界 tile（`local_t >= t_tiles_this_seq`）直接 `return`（line 95）。
`a_log_exp = exp(A_log[head])`（line 101）。

shared memory 只有一块：`dgc_smem[C*D]`（line 105）。

---

## 3. Phase 1+2：融合算 dgc 与 dq/dk（line 129-218）

这是全 kernel 最核心、最讲数值稳定的部分。

### 3.1 重算 L2 范数（line 136-157）

每线程读自己负责的 8 个 $q,k$ 元素，累加平方和；尾块越界行读 0；再用 16 线程的 `__shfl_xor` 归约（$\delta=8,4,2,1$）把整行平方和汇总：

$$
q_{inv} = \frac{1}{\sqrt{\lVert q\rVert^2 + \epsilon}},\qquad
k_{inv} = \frac{1}{\sqrt{\lVert k\rVert^2 + \epsilon}},\qquad \epsilon=10^{-6}
$$

归约方式必须与前向逐位一致。

### 3.2 重算门控指数、算 dqn/dkn 与 dgc（line 163-193）

对每个元素（列 $\text{col}=\text{my\_col}+i$）：

$$
e_+ = \exp_2(g_c),\quad e_- = \exp_2(-g_c),\quad e_{T} = \exp_2(g_T)\cdot e_- = \exp_2(g_T-g_c)
$$

（代码里 `gt_tile[col]` 存的就是 $\exp_2(g_T)$，所以 `exp_gt_gc = gt_tile[col]*exp_neg_gc`。）

#### (a) dqn / dkn（供 L2 反向）

前向 $q_d = q_n e_+ \texttt{scale}$，所以对 $q_n$：

$$
dq_n = dq_d\cdot e_+\cdot \texttt{scale}
$$

$k_n$ 同时进了 $k_d, k_i, k_r$ 三个分支，梯度相加：

$$
dk_n = dk_d\, e_+ + dk_i\, e_- + dk_r\, e_T
$$

#### (b) dgc —— 数值稳定写法（关键 trick）

> 下面先给从零详解（b.0–b.4），已熟悉的读者可直接看 b.3 的结论框。

##### (b.0) 先搞清 $g_c$ 是什么、要求什么

$g_c$ 是"门控累加和"，一个标量（对每行每维）。前向它出现在四个派生量的**指数**里：

$$
q_d = q_n\,2^{g_c}\,\texttt{scale},\quad
k_d = k_n\,2^{g_c},\quad
k_i = k_n\,2^{-g_c},\quad
k_r = k_n\,2^{\,g_T-g_c}
$$

上游已经给了这四个量的梯度 $dq_d,dk_d,dk_i,dk_r$。一个变量被用到多处（**扇出 / fan-out**），它的总梯度 = 各处回传之**和**（多元链式法则）。所以：

$$
dg_c = dq_d\frac{\partial q_d}{\partial g_c} + dk_d\frac{\partial k_d}{\partial g_c} + dk_i\frac{\partial k_i}{\partial g_c} + dk_r\frac{\partial k_r}{\partial g_c}
$$

##### (b.1) 求每个偏导（指数函数求导）

对指数函数 $\frac{d}{dx}a^{x} = a^{x}\ln a$。这里底数 $a=2$，所以 $\frac{d}{dg_c}2^{g_c} = 2^{g_c}\ln 2$。本 kernel 约定**把公共因子 $\ln 2$ 推迟到 Phase 4 统一乘**（因为 $dg_c$ 对所有项线性，提一个常数出来最后乘即可），故这里先省略 $\ln 2$。导数符号跟着指数的 $\pm$ 号走：

$$
\frac{\partial q_d}{\partial g_c}\!\sim\! +q_d,\quad
\frac{\partial k_d}{\partial g_c}\!\sim\! +k_d,\quad
\frac{\partial k_i}{\partial g_c}\!\sim\! -k_i,\quad
\frac{\partial k_r}{\partial g_c}\!\sim\! -k_r
$$

（$k_i,k_r$ 指数是 $-g_c$，链式带出一个负号。）代回得**天真公式**（数学正确）：

$$
dg_c = dq_d\, q_d + dk_d\, k_d - dk_i\, k_i - dk_r\, k_r\tag{天真}
$$

##### (b.2) 为什么天真公式在 fp32 里会崩（灾难性抵消）

$k_d = k_n 2^{g_c}$，$k_i = k_n 2^{-g_c}$。当 $g_c$ 较大时这两者量级天差地别。举个具体例子，设 $g_c=40$、$k_n=0.5$：

$$
2^{40}\approx 1.1\times10^{12},\quad 2^{-40}\approx 9\times10^{-13}
\;\Rightarrow\; k_d\approx 5.5\times10^{11},\ k_i\approx 4.5\times10^{-13}
$$

两者相差约 $10^{24}$。fp32 只有约 7 位十进制有效数字（$\approx 2^{-23}$ 相对精度）。一旦把 $dk_d k_d$（可能上千万量级）和 $dk_i k_i$（极小）放进同一个加减式：

- **大 + 小**：小项落在大项的有效位之外，被直接舍掉 → 信息丢失；
- **大 − 大**：若两个大项接近，相减后高位全抵消、只剩低位舍入噪声 → **灾难性抵消（catastrophic cancellation）**。

结果 $dg_c$ 基本是垃圾。注释 line 134 说的 "kd, ki can differ by ~1e20" 就是这个意思。

##### (b.3) 稳定写法：把 $k_n,q_n$ 提到括号外

注意 $k_d\,dk_d = (k_n 2^{g_c})\,dk_d = k_n\,(dk_d\,2^{g_c})$，对 $k_i,k_r$ 同理。把公共的 $k_n$（和 $q_n$）提出来：

$$
\boxed{\;dg_c = k_n\,\underbrace{(dk_d\, e_+ - dk_i\, e_- - dk_r\, e_T)}_{=\ dk_n^{\text{signed}}} \; + \; q_n\,\underbrace{(dq_d\, e_+\,\texttt{scale})}_{=\ dq_n}\;}
$$

其中 $e_+=2^{g_c},\ e_-=2^{-g_c},\ e_T=2^{\,g_T-g_c}$。对应源码 line 186-187。

##### (b.4) 为什么这样就稳了

关键观察：括号里的乘积 $dk_d\,e_+$ **本身就是 $O(1)$**（正常大小）。原因是上游梯度 $dk_d$ 是按 $\propto 1/e_+$ 的比例传下来的（前向 $k_d\propto e_+$，反向链式正好带一个 $1/e_+$），二者相乘把 $e_+$ 抵消掉。于是：

- 括号内是"几个正常大小的数相减" → 良态，没有大数吞小数；
- 再乘一个 $O(1)$ 的 $k_n$（单位向量的分量）→ 仍然正常。

全程不出现 $10^{12}$ 这种怪物。对比一句话：

$$
\text{天真：}\ dk_d\cdot \underbrace{k_d}_{\sim10^{11}}\quad\text{vs}\quad \text{稳定：}\ k_n\cdot\underbrace{(dk_d\,e_+)}_{O(1)}
$$

两者数学恒等，只是**运算顺序**不同，浮点结果天差地别。这是典型的"重新结合以避免中间量爆炸"的数值技巧。

> **注意区分两个量**（来源相同、符号不同）：
> $$dk_n = dk_d e_+ + dk_i e_- + dk_r e_T \quad(\text{全 }+,\ \text{对 }k_n\text{ 的梯度，给 L2 反向})$$
> $$dk_n^{\text{signed}} = dk_d e_+ - dk_i e_- - dk_r e_T \quad(\text{带符号，对 }g_c\text{ 的梯度})$$
> $k_n$ 进 $k_d/k_i/k_r$ **不带** $g_c$ 的符号（前向 $k_d=k_n e_+$ 里 $k_n$ 是线性因子，故 $dk_n$ 全加）；而 $g_c$ 进**指数**带 $\pm$ 号（故 $dk_n^{\text{signed}}$ 带符号）。两者只差中间两项的正负号，别混用。

### 3.3 L2 归一化的反向 → dq/dk（line 195-216）

**前向** $y = x/\lVert x\rVert$。**要求** 已知 $dy$ 求 $dx$。

> 下面 3.3.0 是写给线代基础薄弱读者的从零详解；已熟悉的读者可直接看 3.3.1 的结论。

#### 3.3.0 从零推导

**为什么不简单**：若 $y_i = 5x_i$ 这种逐元素映射，求导就完事。但归一化里 $n=\lVert x\rVert$ **装着所有分量** $x_0,\dots,x_{D-1}$，所以改动任一 $x_j$ 会通过 $n$ 影响**每一个** $y_i$。这就是最终会冒出耦合项 $(dy\!\cdot\!y)$ 的根源。

**第一块：$n$ 对 $x_j$ 的导数。** 设 $S=\sum_k x_k^2+\epsilon$，$n=\sqrt{S}$。只有第 $j$ 项含 $x_j$，故 $\partial S/\partial x_j=2x_j$，再用 $n=S^{1/2}$ 的链式：

$$
\frac{\partial n}{\partial x_j} = \tfrac12 S^{-1/2}\cdot 2x_j = \frac{x_j}{n}\tag{★}
$$

**第二块：$y_i$ 对 $x_j$ 的导数（核心）。** $y_i = x_i\, n^{-1}$，用乘积法则，$\partial x_i/\partial x_j=\delta_{ij}$（克罗内克符号，$i=j$ 为 1 否则 0），$\partial(1/n)/\partial x_j = -n^{-2}\,(x_j/n)$（代入 ★）：

$$
\frac{\partial y_i}{\partial x_j}
= \frac{\delta_{ij}}{n} - \frac{x_i x_j}{n^3}\tag{☆}
$$

两部分含义：$\delta_{ij}/n$ 是"自己对自己"的直接项（仅 $i=j$）；$-x_ix_j/n^3$ 是通过 $n$ 产生的"所有分量互相耦合"项（$i,j$ 任意都有）。

**第三块：链式法则汇总。** $x_j$ 影响所有 $y_i$，把贡献全加起来，并代入 (☆)：

$$
dx_j = \sum_i dy_i\frac{\partial y_i}{\partial x_j}
= \underbrace{\sum_i dy_i\frac{\delta_{ij}}{n}}_{=\,dy_j/n}
- \frac{x_j}{n^3}\sum_i dy_i x_i
= \frac{dy_j}{n} - \frac{x_j}{n^3}\sum_i dy_i x_i\tag{◇}
$$

第一项里 $\delta_{ij}$ 使求和塌缩成单项 $dy_j$。

**第四块：用 $y$ 化简。** 关键代换 $x_i = y_i n$：

$$
\sum_i dy_i x_i = n\sum_i dy_i y_i = n\,(dy\!\cdot\!y),\qquad \frac{x_j}{n^3}=\frac{y_j}{n^2}
$$

代回 (◇)：

$$
dx_j = \frac{dy_j}{n} - \frac{y_j}{n^2}\,n\,(dy\!\cdot\!y) = \frac{1}{n}\big(dy_j - y_j\,(dy\!\cdot\!y)\big)
$$

#### 3.3.1 结论

$$
\boxed{\;dx = \frac{1}{n}\big(\,dy - y\,(dy\!\cdot\! y)\,\big)\;},\qquad (dy\!\cdot\!y)=\sum_i dy_i\,y_i\ (\text{标量})
$$

对应代码：

$$
dq = (dq_n - q_n\,(dq_n\!\cdot\! q_n))\,q_{inv},\qquad
dk = (dk_n - k_n\,(dk_n\!\cdot\! k_n))\,k_{inv}
$$

点积 $(dq_n\!\cdot\! q_n)=\sum_i dq_n[i]q_n[i]$ 用 16 线程 shfl 归约出整行（line 195-205），仅 `my_row < actual_len` 才写出（line 207）。这个归约就是在算第二块说的"耦合标量"——所以**归一化反向必须先做一次行内归约，无法纯逐元素**。

> **直觉**：$y$ 是单位向量，沿 $y$ 方向拉伸 $x$ 不改变 $y$（只改长度），所以梯度里"沿 $y$ 的径向分量"无效。$(dy\!\cdot\!y)$ 是 $dy$ 在 $y$ 上的投影长度，$y(dy\!\cdot\!y)$ 是该径向分量，减掉它只留切向分量；再乘 $1/n$（$x$ 越长，同样的 $dx$ 对 $y$ 影响越小）。一句话：**把梯度投影到 $\perp y$ 方向，再按 $1/$长度 缩放**。

#### 3.3.2 小数字验证（2 维）

取 $x=(3,4)$，则 $n=5$，$y=(0.6,0.8)$。设 $dy=(1,0)$：

$$
(dy\!\cdot\!y)=0.6,\quad y(dy\!\cdot\!y)=(0.36,0.48),\quad dx=\frac{(1,0)-(0.36,0.48)}{5}=(0.128,-0.096)
$$

**自检**：$dx$ 恒垂直于 $x$（沿 $x$ 推不改变 $y$）：$dx\!\cdot\!x = 0.128\cdot3 + (-0.096)\cdot4 = 0$。一般地

$$
dx\!\cdot\!x = \tfrac1n\big[\textstyle\sum_j dy_j x_j - (dy\!\cdot\!y)\sum_j y_j x_j\big] = \tfrac1n\big[n(dy\!\cdot\!y)-(dy\!\cdot\!y)n\big]=0
$$

可作为写单测时的快速断言。

---

## 4. Phase 3：dv 直接搬运（line 220-224）

$dv$ 在 K2_bwd 已算好（bf16），这里只是从 workspace 的 tile 布局拷到最终 $[H,T,D]$ 布局，仅拷 `actual_len*D` 个元素。

---

## 5. Phase 4：门控反向 = dgc 的逆向 cumsum + dgT（line 226-261）

> 下面 5.0 是写给基础薄弱读者的从零详解（核心就一句：**前缀和的反向是后缀和**）；已熟悉的读者可直接看 5.1。

### 5.0 从零理解"cumsum 的反向"

#### (a) 什么是 cumsum（前缀和）

前向把一串数 $g_{total}[0],g_{total}[1],\dots,g_{total}[C-1]$ 累加成"到当前为止的和"：

$$
g_c[\text{row}] = \sum_{r\le\text{row}} g_{total}[r]
$$

展开看就是（以 $C=4$ 为例）：

$$
\begin{aligned}
g_c[0] &= g_{total}[0] \\
g_c[1] &= g_{total}[0] + g_{total}[1] \\
g_c[2] &= g_{total}[0] + g_{total}[1] + g_{total}[2] \\
g_c[3] &= g_{total}[0] + g_{total}[1] + g_{total}[2] + g_{total}[3]
\end{aligned}
$$

#### (b) 关键观察：每个输入"扇出"到哪些输出

竖着看上面这张表：$g_{total}[r]$ 出现在**第 $r$ 行及其下面所有行**里。比如 $g_{total}[1]$ 出现在 $g_c[1],g_c[2],g_c[3]$（不在 $g_c[0]$）。用偏导写：

$$
\frac{\partial\, g_c[\text{row}]}{\partial\, g_{total}[r]} = \begin{cases} 1, & \text{row}\ge r \\ 0, & \text{row}< r\end{cases}
$$

#### (c) 反向：把扇出的梯度收回来 = 后缀和

多元链式法则：$g_{total}[r]$ 被用到多处，它的总梯度 = 这些输出回传的梯度之**和**，每个偏导都是 1，所以就是**直接相加**：

$$
dg_{total}[r] = \sum_{\text{row}} dg_c[\text{row}]\cdot\frac{\partial g_c[\text{row}]}{\partial g_{total}[r]} = \sum_{\text{row}\ge r} dg_c[\text{row}]
$$

即"从 $r$ 到末尾"的求和——这叫**后缀和（suffix sum）**。还是 $C=4$：

$$
\begin{aligned}
dg_{total}[3] &= dg_c[3] \\
dg_{total}[2] &= dg_c[3] + dg_c[2] \\
dg_{total}[1] &= dg_c[3] + dg_c[2] + dg_c[1] \\
dg_{total}[0] &= dg_c[3] + dg_c[2] + dg_c[1] + dg_c[0]
\end{aligned}
$$

**一句话规律：前向是前缀和（从前往后累加），反向就是后缀和（从后往前累加）。** 方向正好反过来，所以叫"逆向 cumsum"。

#### (d) 怎么 $O(C)$ 算出来：一个滚动变量

不用对每个 $r$ 都重新求和（那是 $O(C^2)$）。从最后一行往前走，维护一个累加器 `rev_sum`，每步把当前行的 $dg_c$ 加进去：

```
rev_sum = 0
for row = C-1 down to 0:          // 从后往前
    rev_sum += dgc[row]           // 此刻 rev_sum == sum_{r>=row} dgc[r]
    dg_total[row] = rev_sum
```

走到 `row` 时，`rev_sum` 恰好就是 $\sum_{r\ge\text{row}}dg_c[r]$，正是我们要的后缀和。对应源码 line 237-238。

#### (e) 数字小例（$C=4$）

设 $dg_c=[\,2,\ 5,\ 1,\ 3\,]$（下标 0..3）。从后往前滚：

| 步 row | 加入 $dg_c[\text{row}]$ | rev_sum | 即 $dg_{total}[\text{row}]$ |
|--------|------|---------|------|
| 3 | 3 | 3  | 3 |
| 2 | 1 | 4  | 4 |
| 1 | 5 | 9  | 9 |
| 0 | 2 | 11 | 11 |

可手动验证 $dg_{total}[1]=dg_c[1]+dg_c[2]+dg_c[3]=5+1+3=9$ ✓。

#### (f) 再叠加 $g_T$ 那条路

$g_{total}[r]$ 其实还流进了整块总和 $g_T=\sum_r g_{total}[r]$（见 5.1）。$\partial g_T/\partial g_{total}[r]=1$ 对**所有** $r$，所以每行都再加一份相同的 $dg_T$：

$$
\texttt{dg\_nat}[r] = \underbrace{\texttt{rev\_sum}[r]}_{\text{来自 }g_c\text{（后缀和）}} + \underbrace{dg_T}_{\text{来自 }g_T\text{（常数）}}
$$

> 小结：cumsum 反向 = 后缀和（一个滚动累加器搞定）；$g_T$ 再贡献一个对每行都相同的常数。下面 5.1 把这两件事正式写出来。

### 5.1 g_total 扇出到两个去处

前向 $g_{total}[r]$ 同时进入两处（计算图里是**并列**两条边）：

$$
g_c[\text{row}] = \sum_{r\le \text{row}} g_{total}[r], \qquad
g_T = \sum_{r} g_{total}[r]
$$

多元链式法则：一个量被用到多处（扇出），其总梯度 = 各下游路径回传之**和**。

$$
dg_{total}[r] = \underbrace{\sum_{\text{row}\ge r} dg_c[\text{row}]}_{\text{来自 }g_c}
              + \underbrace{dg_T}_{\text{来自 }g_T}
$$

- **gc 这条**：因 $\partial g_c[\text{row}]/\partial g_{total}[r] = 1$ 仅当 $\text{row}\ge r$，所以是"后缀和"。代码从 `row=C-1` 往下走，维护 `rev_sum += dgc[row]`，到 $r$ 时 `rev_sum` 正是 $\sum_{\text{row}\ge r}dg_c$（这就是"逆向 cumsum"）。
- **gT 这条**：因 $\partial g_T/\partial g_{total}[r]=1$ 对**所有** $r$，所以每行都加同一个常数 `dgT_val`（line 232 循环外读一次，line 239 每行加）。

$$
\texttt{dg\_nat} = \texttt{rev\_sum} + \texttt{dgT\_val}
$$

（此处仍是"未乘 $\ln 2$"约定值。）

### 5.2 过门函数反传到 dg_raw 与参数（line 241-260）

把 $dg_{total}$（=`dg_nat`）沿 $g_{total}=\texttt{gate\_scale}\,\sigma(z),\ z=a(g_{raw}+dt_{bias}),\ a=\exp(A_{\log})$ 往回传，并补上延后的 $\ln 2$。

> 下面 5.2.0 是从零详解（单变量链式法则 + sigmoid 求导 + 一个量被多处使用怎么办）；已熟悉的读者可直接看 (a)(b)(c)。

#### 5.2.0 从零理解这条链

**(i) 前向是一串"嵌套函数"。** 从最里到最外：

$$
A_{\log}\ \xrightarrow{\exp}\ a\ \xrightarrow{\ \cdot(g_{raw}+dt)\ }\ z\ \xrightarrow{\ \sigma\ }\ \sigma(z)\ \xrightarrow{\ \times\texttt{gate\_scale}\ }\ g_{total}
$$

原始输入 $g_{raw}$ 和两个参数 $A_{\log}, dt_{bias}$ 都在这条链上。反向就是从右端的已知梯度 $dg_{total}$ 一站一站往左乘"每站的局部导数"。

**(ii) 单变量链式法则（唯一要会的）。** 若 $u=f(w)$，则 $dw = du\cdot f'(w)$。也就是"上游梯度 × 本站导数"。一站一站接力即可。

**(iii) sigmoid 的导数为什么是 $\sigma(1-\sigma)$。** $\sigma(z)=\dfrac{1}{1+e^{-z}}$。求导：

$$
\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2}
= \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}}
= \sigma(z)\,\big(1-\sigma(z)\big)
$$

（用了 $\dfrac{e^{-z}}{1+e^{-z}} = 1-\dfrac{1}{1+e^{-z}} = 1-\sigma$。）这就是代码里的 `dsig = sig*(1-sig)`。

**(iv) $\ln 2$ 从哪冒出来。** 回忆 3.2(b)：算 $dg_c$ 时我们**故意省略了** $2^{g_c}$ 求导该带的公因子 $\ln 2$，约定"最后统一补"。$g_{total}$ 是 $g_c$ 的来源（$g_c$ 是 $g_{total}$ 的前缀和），所以传到这里的 `dg_nat` 也是"少乘了一个 $\ln 2$"的版本。现在就把它补回来：真正的梯度 = `dg_nat` $\times\ln 2$。因为整条链对 `dg_nat` 是线性的，补在哪一步都行，代码补在算 $dz$ 这一步。

**(v) 一个量被用在多处怎么办（$a$ 的情形）。** $a=\exp(A_{\log})$ 同时乘进了 $z=a(g_{raw}+dt)$——它只出现一次在 $z$ 里，简单。但反过来看 $z$ 对三个东西都有依赖：$g_{raw}$、$dt_{bias}$、$a$。求 $z$ 对它们各自的偏导就是把另外的当常数：

$$
\frac{\partial z}{\partial g_{raw}} = a,\qquad
\frac{\partial z}{\partial dt_{bias}} = a,\qquad
\frac{\partial z}{\partial a} = g_{raw}+dt_{bias}
$$

每个偏导再乘上游 $dz$，就是各自的梯度。下面 (a)(b)(c) 正是把 (ii) 的接力走完。

**(a) 过 $g_{total}=\texttt{gate\_scale}\,\sigma(z)$**，$\sigma'(z)=\sigma(1-\sigma)$：

$$
dz = (\,dg_{nat}\cdot \ln 2\,)\cdot \texttt{gate\_scale}\cdot \sigma(1-\sigma)
$$

（代码 line 247 把 $\ln 2$ 写在最后，等价。）

**(b) 过 $z = a\,(g_{raw}+dt_{bias})$**：

$$
dg_{raw} = dz\cdot a,\qquad d(dt_{bias}) = dz\cdot a \quad(\partial z/\partial g_{raw}=\partial z/\partial dt = a)
$$

**(c) 过 $a = \exp(A_{\log})$**，$\partial z/\partial a = g_{raw}+dt$，$da/dA_{\log}=a$：

$$
dA_{\log} = \big(dz\,(g_{raw}+dt)\big)\cdot a
$$

代码 line 252-253 先累加 `ddt_bias_acc += dz*a`、`dA_log_acc += dz*(g_raw+dt)`，循环后：

$$
d(dt_{bias})[\text{head,col}] \mathrel{+}= \texttt{ddt\_bias\_acc},\qquad
dA_{\log}[\text{head}] \mathrel{+}= \texttt{dA\_log\_acc}\cdot a
$$

**为什么用 atomicAdd**（line 259-260）：$A_{\log}$ 是每 head 一个标量、$dt_{bias}$ 是每 head 每维，同一 head 的很多 chunk-CTA 会并发写同一地址，必须原子加。而 $dq/dk/dv/dg/d\beta$ 是每 token 独立位置，直接写。

**尾块掩码**：`row >= actual_len` 的行 `dg_out=0`（line 255），且不累加参数梯度。

---

## 6. Phase 5：dbeta 从 chunk 空间转回 logit 空间（line 263-272）

K2_bwd 给的 $d\beta$ 是对"已过 sigmoid 的 $\beta$"的梯度。前向 $\beta = \sigma(\beta_{raw})$，反传过 sigmoid 回 logit：

$$
d\beta_{logit} = d\beta_{chunk}\cdot \sigma(\beta_{raw})\,(1-\sigma(\beta_{raw}))
$$

仅前 `actual_len` 个线程各写一个 token 行。

---

## 7. 全部梯度公式速查表

$$
\begin{aligned}
&\textbf{重算:}\quad q_{inv}=\big(\lVert q\rVert^2+\epsilon\big)^{-1/2},\ k_{inv}=\dots,\quad q_n=q\,q_{inv},\ k_n=k\,k_{inv} \\
&e_+=\exp_2(g_c),\ e_-=\exp_2(-g_c),\ e_T=\exp_2(g_T-g_c) \\[4pt]
&\textbf{收拢:}\quad dq_n = dq_d\,e_+\,\texttt{scale},\qquad dk_n = dk_d e_+ + dk_i e_- + dk_r e_T \\
&dg_c = k_n(dk_d e_+ - dk_i e_- - dk_r e_T) + q_n\,dq_n \\[4pt]
&\textbf{L2 反向:}\quad dq = (dq_n - q_n(dq_n\!\cdot\! q_n))\,q_{inv},\qquad dk = (dk_n - k_n(dk_n\!\cdot\! k_n))\,k_{inv} \\[4pt]
&\textbf{门控:}\quad dg_{total}[r] = \textstyle\sum_{\text{row}\ge r} dg_c[\text{row}] + dg_T \\
&dz = dg_{total}\cdot \texttt{gate\_scale}\cdot \sigma(1-\sigma)\cdot \ln 2 \\
&dg_{raw} = dz\cdot a,\quad d(dt_{bias}) = dz\cdot a,\quad dA_{\log} = dz\,(g_{raw}+dt)\cdot a,\quad a=\exp(A_{\log}) \\[4pt]
&\textbf{dbeta:}\quad d\beta_{logit} = d\beta_{chunk}\,\sigma(\beta_{raw})(1-\sigma(\beta_{raw})) \\[4pt]
&\textbf{dv:}\quad \text{直接从 K2\_bwd workspace 拷贝}
\end{aligned}
$$

---

## 8. 设计要点总结

| 主题 | 做法 |
|------|------|
| 数值稳定 dgc | 把量级悬殊（$\sim10^{20}$）的 $k_d/k_i/k_r$ 改写成 $k_n\cdot(O(1)\text{ 项相减})+q_n\cdot dq_n$，避免 fp32 抵消 |
| 16 线程/行 + shfl | 严格对齐前向 K1 的归约顺序，保证 bit 级可复现 |
| dkn vs dkn_signed | 前者全加（对 $k_n$，给 L2 反向）；后者带符号（对 $g_c$） |
| 门控逆向 cumsum | $g_c$ 是前缀和 → 反向是后缀和（rev_sum），再叠加常数 $dg_T$ |
| ln2 延后 | $2^x$ 求导的 $\ln 2$ 统一在 Phase 4 乘入 |
| atomicAdd | $dA_{\log}, d(dt_{bias})$ 跨 chunk-CTA 累加；其余每 token 独立直接写 |
| 尾块掩码 | `actual_len` 控制越界行读 0 / 写 0 / 不累加参数 |

> 配套阅读：上游 K2_bwd 的逐行精讲见 [`k2_bwd_recurrence_deep_dive.md`](./k2_bwd_recurrence_deep_dive.md)。本 kernel 的输入 $dk_d/dq_d/dk_i/dk_r/dg_T/dv/d\beta$ 全部来自那里。
