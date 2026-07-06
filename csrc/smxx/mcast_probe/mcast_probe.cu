// Standalone de-risk microkernel for FlashKDA K2 Step-2 K-side multicast.
//
// FINDINGS (validated here):
//  (A) A "dual PipelineTmaAsync multicast" design is unworkable:
//      (1) PipelineTmaAsync multicast requires num_consumers==32 or a multiple
//          of 128 (VS=2's 64-thread compute group is unsupported), and
//      (2) its empty-arrive (consumer->producer) cross-CTA routing assumes every
//          block owns its OWN tile; a single shared multicast buffer consumed by
//          all ranks DEADLOCKS on the empty side.
//  (B) Even a LOCAL PipelineTmaAsync (built with Shape<1,1>) launched inside a
//      real (VS,1) cluster is WRONG: cluster_size==1 => dst_blockid_=0, so every
//      rank's consumer_release routes its empty-arrive to CTA 0 (mapa cta_id=0).
//      Rank!=0's producer never gets its empty release => deadlock on first
//      stage reuse (t==STAGES). (Confirmed: hang at V-wait t=2 with STAGES=2.)
//
// WORKING DESIGN (this file): HAND-ROLLED raw mbarriers for BOTH K and V, with a
// per-stage cute::cluster_sync() as the single "empty" (buffer-free) mechanism:
//   * K-side: raw uint64_t full-barrier per stage; the elected thread on EACH
//     rank issues its SM90_TMA_LOAD_MULTICAST copy into the shared K buffer; the
//     barrier's transaction-byte expectation is the FULL tile; consumers wait on
//     the local full-barrier. (testbed pattern.)
//   * V-side: raw uint64_t full-barrier per stage; the elected LOAD thread on
//     each rank issues its OWN value-column slice via a plain SM90_TMA_LOAD.
//   * "empty"/buffer-reuse: a cute::cluster_sync() at the end of each t. Because
//     the recurrence is chunk-SERIAL this per-chunk cluster barrier is on the
//     critical path anyway; it guarantees all ranks finished reading stage s
//     before it is overwritten STAGES iters later. NO PipelineTmaAsync empty
//     machinery => none of its cross-CTA routing bugs.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90a -DNDEBUG --expt-relaxed-constexpr \
//   --expt-extended-lambda -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
//   -Icutlass/include -Icutlass/tools/util/include mcast_probe.cu -o mcast_probe

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/pipeline/pipeline.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

using namespace cute;
using BF16 = cutlass::bfloat16_t;

static constexpr int D       = 128;
static constexpr int CHUNK   = 16;
static constexpr int VS      = 2;
static constexpr int VCOLS   = D / VS;   // 64
static constexpr int T_TILES = 5;
static constexpr int STAGES  = 2;

using KTileLayout  = Layout<Shape<Int<CHUNK>, Int<D>>,     Stride<Int<D>, _1>>;
using VTileLayout  = Layout<Shape<Int<CHUNK>, Int<VCOLS>>, Stride<Int<VCOLS>, _1>>;

struct SharedStorage {
    alignas(128) BF16 k_smem[STAGES][CHUNK * D];
    alignas(128) BF16 v_smem[STAGES][CHUNK * VCOLS];
    alignas(8) uint64_t k_full[STAGES];   // raw mbarrier: K tile ready
    alignas(8) uint64_t v_full[STAGES];   // raw mbarrier: V tile ready
};

template <class TmaK, class TmaV, class ClusterShape>
__global__ void __launch_bounds__(160) probe_kernel(
    CUTE_GRID_CONSTANT TmaK const tma_k,
    CUTE_GRID_CONSTANT TmaV const tma_v,
    ClusterShape cluster_shape,
    float* out,          // [VS][CHUNK*VCOLS]
    int    T_tiles)
{
    extern __shared__ char smem_raw[];
    SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);

    int rank = cute::block_rank_in_cluster();
    int warp = threadIdx.x / 32;
    bool lane0 = cute::elect_one_sync();
    int  tid  = threadIdx.x;

    enum { COMPUTE_WARPS = 4 };
    bool is_load    = (warp == COMPUTE_WARPS);
    bool is_compute = (warp < COMPUTE_WARPS);
    int  compute_threads = COMPUTE_WARPS * 32;

    // ---- raw mbarrier init (both K and V) : one elected thread inits ----
    if (is_load && lane0) {
        for (int s = 0; s < STAGES; ++s) {
            cute::initialize_barrier(ss.k_full[s], /*arrive_count=*/1);
            cute::initialize_barrier(ss.v_full[s], /*arrive_count=*/1);
        }
    }

    // Cluster-wide init handshake (fence the barrier inits across the cluster).
    cutlass::pipeline_init_arrive_relaxed(size(cluster_shape));
    cutlass::pipeline_init_wait(size(cluster_shape));

    Tensor mK = tma_k.get_tma_tensor(make_shape(Int<T_TILES>{} * Int<CHUNK>{}, Int<D>{}));
    Tensor mV = tma_v.get_tma_tensor(make_shape(Int<VS>{} * Int<T_TILES>{} * Int<CHUNK>{}, Int<VCOLS>{}));
    Tensor gK = zipped_divide(mK, Shape<Int<CHUNK>, Int<D>>{});   // ((CHUNK,D),(1,T))

    constexpr uint32_t kTmaBytes = CHUNK * D     * sizeof(BF16);  // full mcast tile
    constexpr uint32_t vTmaBytes = CHUNK * VCOLS * sizeof(BF16);  // local V slice
    uint16_t kmask = uint16_t((1u << size(cluster_shape)) - 1);

    auto cta_v = tma_v.get_slice(Int<0>{});   // local V descriptor slice

    // per-stage phase bits (barrier flips its parity each completion).
    int k_phase[STAGES], v_phase[STAGES];
    for (int s = 0; s < STAGES; ++s) { k_phase[s] = 0; v_phase[s] = 0; }

    float acc[CHUNK * VCOLS / (COMPUTE_WARPS * 32) + 1];
    for (auto& a : acc) a = 0.f;

    for (int t = 0; t < T_tiles; ++t) {
        int stage = t % STAGES;

        // ===== K multicast LOAD + V local LOAD (elected load thread / rank) =====
        if (is_load && lane0) {
            // K: every rank issues its multicast partition into shared k_smem[stage]
            Tensor sKst = make_tensor(make_smem_ptr(ss.k_smem[stage]), KTileLayout{});
            Tensor sKst_x = make_tensor(sKst.data(), make_layout(sKst.layout(), Layout<_1>{}));
            auto [tKgK, tKsK] = tma_partition(tma_k, rank, make_layout(cluster_shape),
                                              sKst_x, group_modes<0,1>(gK));
            cute::set_barrier_transaction_bytes(ss.k_full[stage], kTmaBytes);
            cute::copy(tma_k.with(ss.k_full[stage], kmask), tKgK(_, t), tKsK(_, 0));

            // V: this rank's own value-column slice into shared v_smem[stage]
            int vrow = (rank * T_TILES + t) * CHUNK;
            auto voff = mV.layout()(vrow, 0);
            Tensor gVtile = make_tensor(mV.data() + voff,
                make_layout(make_shape(Int<CHUNK>{}, Int<VCOLS>{}), stride(mV.layout())));
            Tensor sV = make_tensor(make_smem_ptr(ss.v_smem[stage]), VTileLayout{});
            cute::set_barrier_transaction_bytes(ss.v_full[stage], vTmaBytes);
            cute::copy(tma_v.with(ss.v_full[stage]), cta_v.partition_S(gVtile),
                       cta_v.partition_D(sV));
        }

        // ===== consume (all compute threads) =====
        if (is_compute) {
            cute::wait_barrier(ss.k_full[stage], k_phase[stage]);
            cute::wait_barrier(ss.v_full[stage], v_phase[stage]);
            BF16* ks = ss.k_smem[stage];
            BF16* vs = ss.v_smem[stage];
            int idx = 0;
            for (int e = tid; e < CHUNK * VCOLS; e += compute_threads, ++idx) {
                int row = e / VCOLS;
                int col = e % VCOLS;
                float ksum = 0.f;
                for (int d = 0; d < D; ++d) ksum += float(ks[row * D + d]);
                acc[idx] += ksum / float(D) + float(vs[row * VCOLS + col]);
            }
        }
        k_phase[stage] ^= 1;
        v_phase[stage] ^= 1;

        // buffer-free / "empty": chunk-serial cluster barrier. Guarantees all
        // ranks finished reading stage s before it is overwritten STAGES later.
        cute::cluster_sync();
    }

    if (is_compute) {
        int idx = 0;
        for (int e = tid; e < CHUNK * VCOLS; e += compute_threads, ++idx)
            out[rank * (CHUNK * VCOLS) + e] = acc[idx];
    }
    cute::cluster_sync();
}

int main() {
    cudaSetDevice(0);
    int Trows = T_TILES * CHUNK;
    std::vector<float> hK(Trows * D), hV(VS * Trows * VCOLS);
    for (int i = 0; i < Trows * D; ++i) hK[i] = float((i * 7 % 13) - 6) * 0.1f;
    for (int i = 0; i < VS * Trows * VCOLS; ++i) hV[i] = float((i * 5 % 11) - 5) * 0.1f;

    std::vector<BF16> hKb(Trows * D), hVb(VS * Trows * VCOLS);
    for (size_t i = 0; i < hKb.size(); ++i) hKb[i] = BF16(hK[i]);
    for (size_t i = 0; i < hVb.size(); ++i) hVb[i] = BF16(hV[i]);

    BF16 *dK, *dV; float* dOut;
    cudaMalloc(&dK, hKb.size() * sizeof(BF16));
    cudaMalloc(&dV, hVb.size() * sizeof(BF16));
    cudaMalloc(&dOut, VS * CHUNK * VCOLS * sizeof(float));
    cudaMemcpy(dK, hKb.data(), hKb.size() * sizeof(BF16), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hVb.data(), hVb.size() * sizeof(BF16), cudaMemcpyHostToDevice);

    Tensor mK = make_tensor(make_gmem_ptr(dK), make_shape(Trows, D), make_stride(D, 1));
    Tensor mV = make_tensor(make_gmem_ptr(dV), make_shape(VS * Trows, VCOLS),
                            make_stride(VCOLS, 1));

    auto cluster_shape = Shape<Int<VS>, _1>{};
    auto tma_k = make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, mK, KTileLayout{}, size(cluster_shape));
    auto tma_v = make_tma_copy(SM90_TMA_LOAD{}, mV, VTileLayout{});

    dim3 grid(VS, 1, 1);
    dim3 block(160, 1, 1);
    dim3 cluster(VS, 1, 1);
    int smem = sizeof(SharedStorage);
    printf("smem bytes = %d\n", smem);
    cudaFuncSetAttribute(
        (void*)probe_kernel<decltype(tma_k), decltype(tma_v), decltype(cluster_shape)>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    void* kernel = (void*)probe_kernel<decltype(tma_k), decltype(tma_v), decltype(cluster_shape)>;
    cutlass::ClusterLaunchParams lp{grid, block, cluster, smem};
    cutlass::Status st = cutlass::launch_kernel_on_cluster(
        lp, kernel, tma_k, tma_v, cluster_shape, dOut, T_TILES);
    cudaError_t err = cudaDeviceSynchronize();
    if (st != cutlass::Status::kSuccess || err != cudaSuccess) {
        printf("LAUNCH/RUN ERROR: status=%d cuda=%s\n", int(st), cudaGetErrorString(err));
        return 1;
    }

    std::vector<float> hOut(VS * CHUNK * VCOLS);
    cudaMemcpy(hOut.data(), dOut, hOut.size() * sizeof(float), cudaMemcpyDeviceToHost);

    auto refK = [&](int t, int row, int d) { return float(BF16(hK[(t * CHUNK + row) * D + d])); };
    auto refV = [&](int blk, int t, int row, int col) {
        return float(BF16(hV[((blk * Trows) + t * CHUNK + row) * VCOLS + col]));
    };
    double maxerr = 0;
    for (int blk = 0; blk < VS; ++blk)
        for (int row = 0; row < CHUNK; ++row)
            for (int col = 0; col < VCOLS; ++col) {
                float ref = 0.f;
                for (int t = 0; t < T_TILES; ++t) {
                    float ksum = 0.f;
                    for (int d = 0; d < D; ++d) ksum += refK(t, row, d);
                    ref += ksum / float(D) + refV(blk, t, row, col);
                }
                float got = hOut[blk * (CHUNK * VCOLS) + row * VCOLS + col];
                maxerr = fmax(maxerr, fabs(double(ref - got)));
            }
    printf("maxerr = %.3e  ->  %s\n", maxerr, maxerr < 1e-2 ? "PASS" : "FAIL");
    return maxerr < 1e-2 ? 0 : 2;
}
