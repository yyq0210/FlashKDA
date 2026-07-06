// Timing probe: does K-side TMA multicast + per-chunk cluster_sync actually BEAT
// naive per-rank K reload for the FlashKDA K2 V-split, net of the cluster_sync
// cost on the serial critical path? Realistic K-side byte volume per chunk:
// beta+kd+qd+kr+gt+INV+Mqk ~= (3*CHUNK*D + 2*CHUNK*CHUNK)*2 + CHUNK*D*4 + 32*2 bytes.
// We emulate that with a single [CHUNK, KWIDTH] bf16 K-side tile whose byte size
// matches, plus a value-split V tile, over T_TILES chunks, and time both paths.
//
// Two kernels, same grid/cluster (VS,1):
//   mc  = multicast K (1 DRAM read of the K tile broadcast to VS ranks) + local V
//         + per-chunk cluster_sync (buffer reuse).
//   nv  = every rank reloads the FULL K tile from DRAM itself (Step-1 naive) + local
//         V, NO cluster (grid z=VS), NO cluster_sync.
// Both do a trivial reduction so the compiler can't DCE the loads. We compare
// wall time across T_TILES to see if multicast's DRAM saving outweighs the
// cluster_sync serial cost.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90a -DNDEBUG --expt-relaxed-constexpr \
//   --expt-extended-lambda -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
//   -Icutlass/include -Icutlass/tools/util/include time_probe.cu -o time_probe

#include <cstdio>
#include <cstdint>
#include <vector>
#include <functional>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

using namespace cute;
using BF16 = cutlass::bfloat16_t;

static constexpr int D      = 128;
static constexpr int CHUNK  = 16;
static constexpr int VS     = 2;
static constexpr int VCOLS  = D / VS;   // 64
// K-side byte volume per chunk in the real K2 (beta+kd+qd+kr+gt+INV+Mqk):
//   3*CHUNK*D bf16 (kd,qd,kr) + 2*CHUNK*CHUNK bf16 (INV,Mqk) + CHUNK*D*? ...
// gt is [D] fp32 = D*4; beta 32*2. Total bytes:
//   3*(16*128)*2 + 2*(16*16)*2 + 128*4 + 32*2 = 12288 + 1024 + 512 + 64 = 13888 B
// Emulate with a bf16 [CHUNK, KWIDTH] tile of the same bytes: KWIDTH = 13888/(16*2)=434 -> round 448.
static constexpr int KROWS  = 48;       // emulate ~3 [16,128] K-side operands (12288 B)
static constexpr int KC     = 128;
static constexpr int STAGES = 3;

using KTileLayout = Layout<Shape<Int<KROWS>, Int<KC>>, Stride<Int<KC>, _1>>;
using VTileLayout = Layout<Shape<Int<CHUNK>, Int<VCOLS>>, Stride<Int<VCOLS>, _1>>;

// ---------------- Multicast + cluster_sync kernel ----------------
struct SharedMC {
    alignas(128) BF16 k_smem[STAGES][KROWS * KC];
    alignas(128) BF16 v_smem[STAGES][CHUNK * VCOLS];
    alignas(8) uint64_t k_full[STAGES];
    alignas(8) uint64_t v_full[STAGES];
};

template <class TmaK, class TmaV, class ClusterShape>
__global__ void __launch_bounds__(160) mc_kernel(
    CUTE_GRID_CONSTANT TmaK const tma_k,
    CUTE_GRID_CONSTANT TmaV const tma_v,
    ClusterShape cluster_shape, float* out, int T_tiles)
{
    extern __shared__ char smem_raw[];
    SharedMC& ss = *reinterpret_cast<SharedMC*>(smem_raw);
    int rank = cute::block_rank_in_cluster();
    int warp = threadIdx.x / 32;
    bool lane0 = cute::elect_one_sync();
    int tid = threadIdx.x;
    enum { COMPUTE_WARPS = 4 };
    bool is_load = (warp == COMPUTE_WARPS), is_compute = (warp < COMPUTE_WARPS);
    int compute_threads = COMPUTE_WARPS * 32;

    if (is_load && lane0)
        for (int s = 0; s < STAGES; ++s) {
            cute::initialize_barrier(ss.k_full[s], 1);
            cute::initialize_barrier(ss.v_full[s], 1);
        }
    cutlass::pipeline_init_arrive_relaxed(size(cluster_shape));
    cutlass::pipeline_init_wait(size(cluster_shape));

    Tensor mK = tma_k.get_tma_tensor(make_shape(Int<KROWS>{}, Int<KC>{}));  // reused every t
    Tensor mV = tma_v.get_tma_tensor(make_shape(Int<VS>{} * Int<CHUNK>{}, Int<VCOLS>{}));
    Tensor gK = zipped_divide(mK, Shape<Int<KROWS>, Int<KC>>{});
    constexpr uint32_t kBytes = KROWS * KC * sizeof(BF16);
    constexpr uint32_t vBytes = CHUNK * VCOLS * sizeof(BF16);
    uint16_t kmask = uint16_t((1u << size(cluster_shape)) - 1);
    auto cta_v = tma_v.get_slice(Int<0>{});

    int k_phase[STAGES], v_phase[STAGES];
    for (int s = 0; s < STAGES; ++s) { k_phase[s] = 0; v_phase[s] = 0; }
    float acc = 0.f;

    for (int t = 0; t < T_tiles; ++t) {
        int stage = t % STAGES;
        if (is_load && lane0) {
            Tensor sKst = make_tensor(make_smem_ptr(ss.k_smem[stage]), KTileLayout{});
            Tensor sKst_x = make_tensor(sKst.data(), make_layout(sKst.layout(), Layout<_1>{}));
            auto [tKgK, tKsK] = tma_partition(tma_k, rank, make_layout(cluster_shape),
                                              sKst_x, group_modes<0,1>(gK));
            cute::set_barrier_transaction_bytes(ss.k_full[stage], kBytes);
            cute::copy(tma_k.with(ss.k_full[stage], kmask), tKgK(_, 0), tKsK(_, 0));

            int vrow = rank * CHUNK;
            auto voff = mV.layout()(vrow, 0);
            Tensor gVtile = make_tensor(mV.data() + voff,
                make_layout(make_shape(Int<CHUNK>{}, Int<VCOLS>{}), stride(mV.layout())));
            Tensor sV = make_tensor(make_smem_ptr(ss.v_smem[stage]), VTileLayout{});
            cute::set_barrier_transaction_bytes(ss.v_full[stage], vBytes);
            cute::copy(tma_v.with(ss.v_full[stage]), cta_v.partition_S(gVtile), cta_v.partition_D(sV));
        }
        if (is_compute) {
            cute::wait_barrier(ss.k_full[stage], k_phase[stage]);
            cute::wait_barrier(ss.v_full[stage], v_phase[stage]);
            BF16* ks = ss.k_smem[stage]; BF16* vs = ss.v_smem[stage];
            for (int e = tid; e < KROWS * KC; e += compute_threads) acc += float(ks[e]) * 1e-6f;
            for (int e = tid; e < CHUNK * VCOLS; e += compute_threads) acc += float(vs[e]);
        }
        k_phase[stage] ^= 1; v_phase[stage] ^= 1;
        cute::cluster_sync();
    }
    if (is_compute && tid == 0) out[rank] = acc;
    cute::cluster_sync();
}

// ---------------- Naive per-rank reload kernel (grid z=VS, no cluster) ----------------
struct SharedNV {
    alignas(128) BF16 k_smem[STAGES][KROWS * KC];
    alignas(128) BF16 v_smem[STAGES][CHUNK * VCOLS];
    typename cutlass::PipelineTmaAsync<STAGES>::SharedStorage pipe;
};

template <class TmaK, class TmaV>
__global__ void __launch_bounds__(160) nv_kernel(
    CUTE_GRID_CONSTANT TmaK const tma_k,
    CUTE_GRID_CONSTANT TmaV const tma_v,
    float* out, int T_tiles)
{
    extern __shared__ char smem_raw[];
    SharedNV& ss = *reinterpret_cast<SharedNV*>(smem_raw);
    int warp = threadIdx.x / 32;
    bool lane0 = cute::elect_one_sync();
    int tid = threadIdx.x;
    enum { COMPUTE_WARPS = 4 };
    bool is_load = (warp == COMPUTE_WARPS), is_compute = (warp < COMPUTE_WARPS);
    int compute_threads = COMPUTE_WARPS * 32;
    int vsplit = blockIdx.z;

    using Pipe = cutlass::PipelineTmaAsync<STAGES>;
    typename Pipe::Params pp;
    pp.transaction_bytes = KROWS * KC * sizeof(BF16) + CHUNK * VCOLS * sizeof(BF16);
    pp.role = is_load ? Pipe::ThreadCategory::Producer
              : (is_compute ? Pipe::ThreadCategory::Consumer : Pipe::ThreadCategory::NonParticipant);
    pp.is_leader = is_load && lane0;
    pp.num_consumers = compute_threads;
    pp.num_producers = 1;
    Pipe pipe(ss.pipe, pp, Shape<_1,_1>{});
    cutlass::pipeline_init_wait(1);

    Tensor mK = tma_k.get_tma_tensor(make_shape(Int<KROWS>{}, Int<KC>{}));
    Tensor mV = tma_v.get_tma_tensor(make_shape(Int<VS>{} * Int<CHUNK>{}, Int<VCOLS>{}));
    auto cta_k = tma_k.get_slice(Int<0>{});
    auto cta_v = tma_v.get_slice(Int<0>{});

    auto wr = cutlass::make_producer_start_state<Pipe>();
    auto rd = cutlass::PipelineState<STAGES>();
    float acc = 0.f;

    for (int t = 0; t < T_tiles; ++t) {
        if (is_load) {
            pipe.producer_acquire(wr);
            if (lane0) {
                auto* bar = pipe.producer_get_barrier(wr);
                int stage = wr.index();
                Tensor gK = make_tensor(mK.data(),
                    make_layout(make_shape(Int<KROWS>{}, Int<KC>{}), stride(mK.layout())));
                Tensor sK = make_tensor(make_smem_ptr(ss.k_smem[stage]), KTileLayout{});
                cute::copy(tma_k.with(*bar), cta_k.partition_S(gK), cta_k.partition_D(sK));
                int vrow = vsplit * CHUNK;
                auto voff = mV.layout()(vrow, 0);
                Tensor gV = make_tensor(mV.data() + voff,
                    make_layout(make_shape(Int<CHUNK>{}, Int<VCOLS>{}), stride(mV.layout())));
                Tensor sV = make_tensor(make_smem_ptr(ss.v_smem[stage]), VTileLayout{});
                cute::copy(tma_v.with(*bar), cta_v.partition_S(gV), cta_v.partition_D(sV));
            }
            ++wr;
        }
        if (is_compute) {
            pipe.consumer_wait(rd);
            int stage = rd.index();
            BF16* ks = ss.k_smem[stage]; BF16* vs = ss.v_smem[stage];
            for (int e = tid; e < KROWS * KC; e += compute_threads) acc += float(ks[e]) * 1e-6f;
            for (int e = tid; e < CHUNK * VCOLS; e += compute_threads) acc += float(vs[e]);
            pipe.consumer_release(rd);
            ++rd;
        }
    }
    if (is_load) pipe.producer_tail(wr);
    if (is_compute && tid == 0) out[blockIdx.x * VS + vsplit] = acc;
}

static float bench(std::function<void()> launch, int iters = 100, int warmup = 20) {
    for (int i = 0; i < warmup; ++i) launch();
    cudaDeviceSynchronize();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters * 1e3f;  // us
}

int main(int argc, char** argv) {
    cudaSetDevice(0);
    int Nblk = argc > 1 ? atoi(argv[1]) : 1;   // emulate N*H blocks (grid-starvation knob)
    std::vector<int> Ts = {16, 64, 256, 1024};

    // host data (small; reused every chunk in the mc kernel, per-rank in nv)
    std::vector<BF16> hK(KROWS * KC), hV(VS * CHUNK * VCOLS);
    for (size_t i = 0; i < hK.size(); ++i) hK[i] = BF16(float((i * 7 % 13) - 6) * 0.1f);
    for (size_t i = 0; i < hV.size(); ++i) hV[i] = BF16(float((i * 5 % 11) - 5) * 0.1f);
    BF16 *dK, *dV; float* dOut;
    cudaMalloc(&dK, hK.size() * sizeof(BF16));
    cudaMalloc(&dV, hV.size() * sizeof(BF16));
    cudaMalloc(&dOut, (size_t)Nblk * VS * sizeof(float) + 16);
    cudaMemcpy(dK, hK.data(), hK.size() * sizeof(BF16), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV.data(), hV.size() * sizeof(BF16), cudaMemcpyHostToDevice);

    Tensor mK = make_tensor(make_gmem_ptr(dK), make_shape(Int<KROWS>{}, Int<KC>{}), make_stride(Int<KC>{}, _1{}));
    Tensor mV = make_tensor(make_gmem_ptr(dV), make_shape(VS * CHUNK, VCOLS), make_stride(VCOLS, 1));
    auto cluster_shape = Shape<Int<VS>, _1>{};
    auto tma_k_mc = make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, mK, KTileLayout{}, size(cluster_shape));
    auto tma_v_mc = make_tma_copy(SM90_TMA_LOAD{}, mV, VTileLayout{});
    auto tma_k_nv = make_tma_copy(SM90_TMA_LOAD{}, mK, KTileLayout{});
    auto tma_v_nv = make_tma_copy(SM90_TMA_LOAD{}, mV, VTileLayout{});

    int smem_mc = sizeof(SharedMC), smem_nv = sizeof(SharedNV);
    auto* mcp = (void*)mc_kernel<decltype(tma_k_mc), decltype(tma_v_mc), decltype(cluster_shape)>;
    auto* nvp = (void*)nv_kernel<decltype(tma_k_nv), decltype(tma_v_nv)>;
    cudaFuncSetAttribute(mcp, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_mc);
    cudaFuncSetAttribute(nvp, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_nv);

    printf("Nblk=%d  K-side=%dx%d (~%d B/chunk)  smem mc=%d nv=%d\n",
           Nblk, KROWS, KC, KROWS * KC * 2, smem_mc, smem_nv);
    printf("  T   mc(us)  nv(us)  mc/nv\n");
    for (int T : Ts) {
        auto mc_launch = [&]() {
            // mc grid: (VS, Nblk) cluster (VS,1) -> Nblk clusters
            dim3 grid(VS, Nblk, 1), block(160), cluster(VS, 1, 1);
            cutlass::ClusterLaunchParams lp{grid, block, cluster, (size_t)smem_mc};
            cutlass::launch_kernel_on_cluster(lp, mcp, tma_k_mc, tma_v_mc, cluster_shape, dOut, T);
        };
        auto nv_launch = [&]() {
            dim3 grid(Nblk, 1, VS), block(160);
            nv_kernel<decltype(tma_k_nv), decltype(tma_v_nv)>
                <<<grid, block, smem_nv>>>(tma_k_nv, tma_v_nv, dOut, T);
        };
        // warm correctness/launch check
        mc_launch(); nv_launch();
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("RUN ERR T=%d: %s\n", T, cudaGetErrorString(e)); return 1; }
        float tmc = bench(mc_launch), tnv = bench(nv_launch);
        printf(" %4d  %6.2f  %6.2f  %.3f\n", T, tmc, tnv, tmc / tnv);
    }
    return 0;
}
