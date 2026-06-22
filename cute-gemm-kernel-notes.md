# CuTe GEMM Kernel 完整知识点笔记

## 目录
1. [CuTe Layout 基础](#1-cute-layout-基础)（含 1.2 Index vs Offset 区别）
2. [Layout Inverse（逆）](#2-layout-inverse逆)
3. [Layout Compose & Inverse 实战：C-layout 到 A-layout 转换](#3-layout-compose--inverse-实战c-layout-到-a-layout-转换)（含 3.1 坐标变换的含义与用途）
4. [多阶段流水线 GEMM Kernel 完整解析](#4-多阶段流水线-gemm-kernel-完整解析)

---

## 1. CuTe Layout 基础

### 1.1 Layout 是什么

Layout 是一个函数，将多维坐标映射到一维 offset：

```
Layout A: (m, n) -> offset
```

由 shape 和 stride 组成，例如 `(2, 4):(4, 1)` 表示 row-major 的 2x4 矩阵：

```
offset = m * 4 + n * 1

(0,0)->0  (0,1)->1  (0,2)->2  (0,3)->3
(1,0)->4  (1,1)->5  (1,2)->6  (1,3)->7
```

### 1.2 Index 和 Offset 的区别

- **index（坐标/编号）**：逻辑上的序号，用来"找元素"的标签。可以是一维 index（如 `layout(3)`）或多维 index（如 `layout(1, 2)`）。
- **offset（偏移量）**：物理上离数据起始地址的距离，`data[offset]` 才是真正访问内存。

**Layout 的本质就是 index → offset 的映射函数：** `offset = layout(index)`

以 `layoutA = (3,8):(8,1)` (row-major) 为例：

```
二维 index (i,j)  →  一维 index  →  offset（物理位置）
    (0, 0)        →      0       →    0     →  data[0]
    (0, 1)        →      1       →    1     →  data[1]
    (1, 0)        →      3       →    8     →  data[8]
```

不同 layout 下，**同一个 index 可以对应不同的 offset**：

```
layoutA(3) = 8×1 + 0×1 = 8    ← index 3 在 offset 8
layoutC(3) = 1×0 + 3×1 = 1    ← 换了 layout，同一个 index 3 指向 offset 1
```

### 1.3 Tensor = data_ptr + layout

CuTe 中一个 Tensor 由两部分组成：

```
Tensor = (data_ptr, layout)
          ^           ^
        数据在哪     怎么索引
```

这种分离设计意味着可以在不搬运数据的情况下，只改变 layout 来改变数据的索引方式。

---

## 2. Layout Inverse（逆）

### 2.1 定义

给定函数 `f: x -> y`，其逆函数 `g = f^{-1}` 满足：`g(f(x)) = x`。

对于 Layout：
- 原始：`A: (m, n) -> offset`
- Inverse：`A^{-1}: offset -> (m, n)`

### 2.2 CuTe 中的两种 Inverse

代码位于 `include/cute/layout.hpp`：

**right_inverse**（第 1265 行）：
```
layout(result(i)) == i    对所有 i < size(result)
```
即 `composition(layout, right_inverse(layout))` 等于 identity。当 layout 是满射时存在。

**left_inverse**（第 1324 行）：
```
layout(result(layout(i))) == layout(i)    对所有 i < size(layout)
```
更宽松，即使 layout 不是双射也能用。

### 2.3 Inverse 的构造过程

以 `(2, 4):(4, 1)` 为例：

**第一步：按 stride 升序排列维度**

| 维度 | shape | stride |
|------|-------|--------|
| 0    | 2     | 4      |
| 1    | 4     | 1      |

按 stride 从小到大排：先 stride=1（shape=4），再 stride=4（shape=2）

排序后 shape 变成 `(4, 2)`

**第二步：inverse 的 stride 用 col-major（前缀积）**

shape `(4, 2)` 的 col-major stride = `(1, 4)`

最终结果 `(4, 2):(1, 4)`

**为什么按 stride 排序？**

因为 inverse 要把 offset 分解回坐标，需要从最小的 stride 开始逐级提取：
```
p = offset % 4     -> 提取 stride=1 那个维度（原来的 j）
q = offset / 4     -> 提取 stride=4 那个维度（原来的 i）
```

这个逐级取模/除法的过程，正好就是 col-major layout `(4, 2):(1, 4)` 做的事情。

**注意**：并非简单地"shape 和 stride 互换"。如果原始就是 col-major（stride 已升序），inverse 就是自身。

### 2.4 with_shape 就是 compose

代码 `layout.hpp:206-207`：
```cpp
auto with_shape(OtherShape const& shape) const {
    return composition(*this, make_layout(shape));
}
```

`make_layout(shape)` 创建以给定 shape 为形状、col-major stride 的 layout，然后 composition 串联。

整个 inverse + with_shape 的流程：
```
原始：    (m, n) --[Layout A]--> offset
逆操作：  offset --[A^{-1}]--> (m, n)
重塑(with_shape)：
          (p, q) --[col-major layout]--> offset --[A^{-1}]--> (m, n)
                   \__________ composition __________/
```

---

xuxcc

## 4. 多阶段流水线 GEMM Kernel 完整解析

### 4.1 整体功能

计算矩阵乘法 `D = A x B^T`，其中 A 是 `(M, K)`，B 是 `(N, K)`，D 是 `(M, N)`。
把大矩阵切成小 tile，每个 thread block 负责一个 `(128, 128)` 的输出 tile，沿 K 方向循环累加。

### 4.2 数据流全景

```
Global Memory (A, B)
  |
  | cp.async 128bit，异步，kStage 个缓冲环形轮转
  v
Shared Memory (sA, sB)    <-- Swizzle 消除 bank conflict
  |
  | ldmatrix (S2R)，每个 MMA 步之前预取
  v
Register (tCrA, tCrB)     <-- MMA 和 Copy 有不同视图，retile 切换
  |
  | cute::gemm -> MMA 指令，累加到 tCrD
  v
Register (tCrD)            <-- 累加器
  |
  | 类型转换 float->half，分批写出
  v
Shared Memory (sC，复用 sA 空间)
  |
  | 128bit copy
  v
Global Memory (D)
```

---

### 4.3 Config 结构体详解

#### 4.3.1 Tile 尺寸

```cpp
kTileM = 128, kTileN = 128, kTileK = 32, kStage = 3
```

一个 block 计算 D 的 128x128 子块，每次从 K 方向取 32 列，smem 中同时缓存 3 份（3 个 stage 做流水线）。

#### 4.3.2 Shared Memory Layout（带 Swizzle）

```cpp
using SmemLayoutAtom = composition(
    Swizzle<3, 3, 3>{},
    make_layout(make_shape(Int<8>{}, Int<32>{}),
                make_stride(Int<32>{}, Int<1>{})));
```

- 基础 layout 是 `(8, 32)` 的 row-major（8 行 32 列）
- 外面套了 `Swizzle<3,3,3>` 来消除 shared memory 的 bank conflict
- 用 `tile_to_shape` 扩展到 `(128, 32, 3)` = `(kTileM, kTileK, kStage)`

#### 4.3.3 MMA 配置——两层扩展机制

**底层 MMA Atom：**

```cpp
using mma_op = SM80_16x8x16_F16F16F16F16_TN;
// Shape_MNK = (16, 8, 16)，需要 32 个线程（1 个 warp）
```

这是一条 PTX 指令，不可再分的最小单元。

**第一层扩展：AtomLayoutMNK（EU Repeat）——用更多 warp 覆盖更大面积**

```cpp
kMmaEURepeatM = 2, kMmaEURepeatN = 2, kMmaEURepeatK = 1;
using MMA_EU_RepeatT = Layout<Shape<_2, _2, _1>>;
```

把 MMA atom 在 M/N 方向各复制 2 份，每份由不同的 warp 负责：
```
一个 atom 需要 32 个线程（1 个 warp）
复制 2x2x1 = 4 份
总共需要 32 x 4 = 128 个线程（4 个 warp）
```

可视化：
```
               N 方向
          +----------+----------+
          | warp 0   | warp 1   |
          | 16x8     | 16x8     |
    M     +----------+----------+
    方    | warp 2   | warp 3   |
    向    | 16x8     | 16x8     |
          +----------+----------+
          覆盖：32x16（MxN），K=16
```

**第二层扩展：PermutationMNK（P Tile）——让每个线程多算几轮，不增加线程数**

```cpp
kMmaPM = 1 * 2 * 16 = 32
kMmaPN = 2 * 2 * 8  = 32
kMmaPK = 1 * 1 * 16 = 16
using MMA_P_T = Tile<Int<32>, Int<32>, Int<16>>;
```

这是最终期望的 tile 尺寸。第一层后已覆盖 `(32, 16, 16)`，P Tile 为 `(32, 32, 16)`：
- M 方向：32 = 32，刚好
- N 方向：32 / 16 = 2，每个线程在 N 方向再多算一倍
- K 方向：16 = 16，刚好

```
第一层扩展后 (32x16):        Permutation 后 (32x32):

+----------+----------+     +----------+----------+----------+----------+
| warp 0   | warp 1   |     | warp 0   | warp 1   | warp 0   | warp 1   |
| 16x8     | 16x8     | --> | 16x8     | 16x8     | 16x8     | 16x8     |
+----------+----------+     +----------+----------+----------+----------+
| warp 2   | warp 3   |     | warp 2   | warp 3   | warp 2   | warp 3   |
| 16x8     | 16x8     |     | 16x8     | 16x8     | 16x8     | 16x8     |
+----------+----------+     +----------+----------+----------+----------+
                              <-- 第一层覆盖 -->  <-- 同样的线程再算一遍 -->
```

**两层扩展的区别：**

| | AtomLayoutMNK (EU Repeat) | PermutationMNK (P Tile) |
|---|---|---|
| 做什么 | 用更多 warp 覆盖更大面积 | 让每个线程多执行几次 MMA |
| 是否增加线程数 | 是，32->128 | 否，还是 128 |
| 代价 | 需要更多线程 | 每个线程做更多工作 |

**最终效果：**
```
atom:             (16,  8, 16)  x 32 线程
第一层 x(2,2,1):  (32, 16, 16)  x 128 线程
第二层 P(32,32,16): (32, 32, 16) x 128 线程，每线程多算一轮

要覆盖整个 tile (128, 128, 32):
  MMA_M = 128/32 = 4 次
  MMA_N = 128/32 = 4 次
  MMA_K = 32/16  = 2 次 (代码中的 nk)
```

#### 4.3.4 Copy 配置

| 阶段 | Copy 类型 | 说明 |
|------|----------|------|
| G2S | `CP_ASYNC_CACHEGLOBAL<uint128_t>` | 全局->共享，128bit 异步拷贝 |
| S2R | `SM75_U32x4_LDSM_N` | 共享->寄存器，ldmatrix 指令 |
| R2S | `UniversalCopy<int>` | 寄存器->共享（epilogue） |
| S2G | `UniversalCopy<uint128_t>` | 共享->全局（epilogue） |

---

### 4.4 Kernel 主体逐段解析

#### 4.4.1 创建全局 Tensor 并切 tile

```cpp
Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k),
                       make_stride(k, Int<1>{}));  // (M, K) row-major
```

用原始指针 + shape + stride 创建 CuTe Tensor。stride `(k, 1)` 即 row-major。

```cpp
Tensor gA = local_tile(A, make_tile(Int<128>{}, Int<32>{}), make_coord(iy, _));
// 结果 shape: (128, 32, num_tile_k)
```

`local_tile` 按 `(128, 32)` 切成网格，取第 `iy` 行，K 方向全部保留（`_`）。
得到 shape = `(128, 32, K/32)`：这个 block 负责的 128 行，每次取 32 列，共 K/32 个 tile。

同理 gB 是 `(128, 32, K/32)`，gD 是 `(128, 128)`。

#### 4.4.2 Shared Memory Tensor

```cpp
extern __shared__ T shm_data[];
T *Ashm = shm_data;
T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});  // (128, 32, 3)
auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});  // (128, 32, 3)
```

在共享内存上创建 tensor，第三维是 stage 编号。A 和 B 的 smem 空间紧挨排列。

#### 4.4.3 MMA Partition——分配到线程

```cpp
TiledMMA tiled_mma;
auto thr_mma = tiled_mma.get_slice(idx);  // 取当前线程的切片

auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
auto tCrD = thr_mma.partition_fragment_C(gD);           // (MMA, MMA_M, MMA_N)
```

在寄存器上创建 fragment：

| 变量 | 含义 | shape |
|------|------|-------|
| tCrA | 当前线程持有的 A 寄存器 | (MMA, MMA_M, MMA_K) |
| tCrB | 当前线程持有的 B 寄存器 | (MMA, MMA_N, MMA_K) |
| tCrD | 当前线程持有的累加器 | (MMA, MMA_M, MMA_N) |

- MMA：一次 mma 指令中该线程负责的元素数
- MMA_M/MMA_N：在 M/N 方向要执行多少次 MMA
- MMA_K：在 K 方向要执行多少次（nk=2，因为 kTileK=32, 每次 MMA 处理 16）

#### 4.4.4 Copy Partition——建立搬运映射

**S2R (Shared -> Register)：**

```cpp
auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
auto tAsA = s2r_thr_copy_a.partition_S(sA);          // smem 中该线程要读的部分
auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);      // 寄存器的"copy视图"
```

为什么需要 `retile_D`？因为 MMA 和 Copy 对寄存器的切分方式不同：
```
MMA 视角：tCrA 的 layout 是 (MMA, MMA_M, MMA_K)
Copy 视角：tCrA_view 的 layout 是 (CPY, CPY_M, CPY_K)
```
底层是同一组寄存器，retile_D 只是换了一种索引方式，让 `cute::copy(tAsA, tCrA_view)` 源和目的 layout 匹配。

**G2S (Global -> Shared)：**

```cpp
auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);   // 全局内存中该线程要读的部分
auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);    // smem 中该线程要写的部分
```

`partition_S` = 切分 Source，`partition_D` = 切分 Destination。

---

### 4.5 多阶段流水线详解

#### 4.5.1 核心问题：为什么要流水线？

GPU 上数据搬运很慢，计算很快。串行做的话，计算单元大部分时间在空等。
流水线目标是让搬运和计算重叠。
要做到这一点，smem 需要多份缓冲区（多个 stage），这样一边往 stage X 写新数据，一边从 stage Y 读旧数据做计算。

#### 4.5.2 三级存储

```
Global Memory  --cp.async-->  Shared Memory  --ldmatrix-->  Register  --mma-->  累加器
     慢(~100 cycles)               中(~30 cycles)              快(~几 cycles)
```

#### 4.5.3 关键变量

```cpp
int itile_to_read = 0;   // 下一个要从 global 读的 K-tile 编号
int ismem_read = 0;      // 当前从 smem 哪个 stage 读
int ismem_write = 0;     // 当前往 smem 哪个 stage 写
```

三个指针在 kStage=3 个 stage 上做环形缓冲。

#### 4.5.4 预填充阶段（Prologue）

```cpp
for (int istage = 0; istage < kStage - 1; ++istage) {  // 填 2 个 stage
    cute::copy(g2s, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
    cute::copy(g2s, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
    cp_async_fence();
    ++itile_to_read;   // 0->1->2
    ++ismem_write;      // 0->1->2
}
```

`cp_async` 是异步的——发出拷贝指令后立刻返回，GPU 在后台搬运。
`cp_async_fence()` 给每批拷贝打一个标记，后面可以用 `cp_async_wait<N>` 等待。

执行完后状态：
```
smem stage:  [0: tile0 搬运中]  [1: tile1 搬运中]  [2: 空]
itile_to_read = 2, ismem_write = 2, ismem_read = 0

                    fence0              fence1
异步队列:    ---|-- tile0 --|-- tile1 --|
```

#### 4.5.5 等待 + 首次预取

```cpp
cp_async_wait<kStage - 2>();   // wait<1>: 等到队列中最多剩 1 个未完成
__syncthreads();
```

`cp_async_wait<1>` 意思：等到未完成的异步拷贝组最多剩 1 个。
队列中有 fence0、fence1 两组，等完后 fence0（tile0）保证完成，fence1（tile1）可能还在搬。

然后预取第一批 S->R：
```cpp
cute::copy(s2r, tAsA(_, _, 0, ismem_read), tCrA_view(_, _, 0));
cute::copy(s2r, tBsB(_, _, 0, ismem_read), tCrB_view(_, _, 0));
```

#### 4.5.6 主循环逐步追踪

每个 K-tile 内部有 `nk=2` 个 MMA 步（kTileK=32，每次 MMA 处理 K=16）。

**itile=0, ik=0：**

```cpp
ik_next = 1;

// ik != nk-1，跳过 wait

// (1) 预取下一步 S->R：把 ik=1 的数据从 smem 搬到寄存器
cute::copy(s2r, tAsA(_, _, 1, 0), tCrA_view(_, _, 1));

// (2) ik==0，发起下一个 tile 的 G->S
cute::copy(g2s, tAgA_copy(_, _, _, 2), tAsA_copy(_, _, _, 2));  // tile2 -> stage2
cp_async_fence();
// itile_to_read: 2->3, ismem_write: 2->0

// (3) 执行 MMA
cute::gemm(tiled_mma, tCrD, tCrA(_, _, 0), tCrB(_, _, 0), tCrD);
```

此刻同时发生三件事（三级重叠）：
```
计算:  MMA 用 ik=0 的寄存器数据做乘加          <-- 计算单元忙
S->R:  ik=1 的数据从 smem 搬到寄存器            <-- smem 读端口忙
G->S:  tile2 从 global 异步搬到 smem stage[2]   <-- 全局内存忙
```

**itile=0, ik=1（最后一步）：**

```cpp
ik_next = 0;

// (1) ik == nk-1，需要切换 smem 读缓冲
cp_async_wait<kStage - 2>();   // 确保下一个 stage 的数据就绪
__syncthreads();
ismem_read = (0 + 1) % 3 = 1;  // 切到 stage 1

// (2) 预取下一个 tile 的 ik=0（从新的 stage 读）
cute::copy(s2r, tAsA(_, _, 0, 1), tCrA_view(_, _, 0));  // stage1, ik=0

// (3) ik != 0，跳过 G->S

// (4) 执行 MMA
cute::gemm(tiled_mma, tCrD, tCrA(_, _, 1), tCrB(_, _, 1), tCrD);
```

#### 4.5.7 `cp_async_wait<kStage-2>` 而不是 `wait<0>` 的原因

`wait<N>` 意思：等到未完成的异步拷贝组最多剩 N 个。

如果用 `wait<0>`（等所有都完成），G->S 搬运就完全不能和计算重叠。
用 `wait<1>` 允许还有 1 组在后台搬，只确保马上要读的那组已经完成。

#### 4.5.8 环形缓冲的完整轮转（kStage=3）

```
时刻          stage[0]    stage[1]    stage[2]    正在计算
--------------------------------------------------------------
预填充后       tile0 ok    tile1       -           -
itile=0       tile0(读)   tile1       tile2(写)   tile0
itile=1       tile3(写)   tile1(读)   tile2       tile1
itile=2       tile3       tile4(写)   tile2(读)   tile2
itile=3       tile3(读)   tile4       tile5(写)   tile3
              ...
```

每一行中：一个 stage 在被读（做计算），一个在被写（接收新数据），一个是缓冲（已写完等着被读）。
三个 stage 保证了读和写永远不会冲突。

#### 4.5.9 一个 stage 内部不会被覆盖

主循环期间累加器 `tCrD` 完全住在寄存器里，不经过 smem。smem 只有 sA 和 sB。

一个 stage 存一个完整的 K-tile (kTileK=32)，其中 ik=0 和 ik=1 是不同位置的数据：
```
smem stage[0] 的数据布局：
       kTileK = 32
  |--- ik=0 (K=0..15) ---|--- ik=1 (K=16..31) ---|
```

`ismem_read` 和 `ismem_write` 始终指向不同 stage，永远不会重合。
一个 stage 的数据从写入完成到该 tile 计算结束，整个期间都不会被覆盖。

#### 4.5.10 流水线时间线图

```
时间 ->

Global->Smem:  |tile0|tile1|     |tile2|     |tile3|     |tile4|
                                  <->          <->          <->
Smem->Reg:           |t0,k0|t0,k1|t1,k0|t1,k1|t2,k0|t2,k1|
                            <->          <->          <->
MMA计算:                    |t0,k0|t0,k1|t1,k0|t1,k1|t2,k0|

                     <--- 三级重叠，流水不停 --->
```

---

### 4.6 Epilogue——结果写回

MMA 计算完后，累加器 tCrD 在寄存器中。写回路径：Reg -> Smem -> Global。

```cpp
// 复用 A 的 smem 空间（主循环已经结束，sA 不需要了）
auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});
```

分批写出（smem 不够一次性放下所有结果，用 kSmemLayoutCBatch=2 做 pipeline）：

```cpp
int step = size<3>(tCsC_r2s);  // pipe = kSmemLayoutCBatch = 2

for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    // 第一步：Reg -> Smem
    for (int j = 0; j < step; ++j) {
        auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
        cute::copy(tCrC_r2sx(_, i + j), t);            // float -> half 类型转换
        cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));  // half -> smem
    }
    __syncthreads();  // 确保所有线程都写完 smem

    // 第二步：Smem -> Global
    for (int j = 0; j < step; ++j) {
        cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }
    __syncthreads();  // 确保所有线程都读完 smem，再进入下一批
}
```

为什么需要临时 tensor t？因为 MMA 累加器类型可能是 float，而输出需要 half。
`make_tensor_like<T>` 创建 half 类型的临时 tensor 来做类型转换。

---

### 4.7 Main 函数流程

```
1. 分配内存（host + device），随机初始化 A、B
2. 跑 11 次（warmup + benchmark）：cuBLAS / cuBLASLt / 自定义 kernel
3. gpu_compare 对比结果正确性
4. 打印左上角 8x8 子矩阵，目视检查
```
