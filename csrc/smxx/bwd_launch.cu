#include "bwd.h"
#include "bwd_kernel1.cuh"
#include "bwd_kernel2.cuh"

// ==================== launch_bwd ====================
template <int D, bool IsVarlen>
void launch_bwd(
    // Forward inputs
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float scale,
    float gate_scale,
    // Forward workspace
    void const* workspace_ptr,
    // do
    cutlass::bfloat16_t const* do_ptr,
    // all_states
    cutlass::bfloat16_t const* all_states_ptr,
    // dS_init
    cutlass::bfloat16_t const* ds_init_ptr,
    // Output gradients
    cutlass::bfloat16_t* dq_ptr,
    cutlass::bfloat16_t* dk_ptr,
    cutlass::bfloat16_t* dv_ptr,
    cutlass::bfloat16_t* dg_ptr,
    cutlass::bfloat16_t* dbeta_ptr,
    float* dA_log_ptr,
    float* ddt_bias_ptr,
    cutlass::bfloat16_t* ds_out_ptr,
    // Dimensions
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens_ptr,
    cudaStream_t stream
) {
    using BF16 = cutlass::bfloat16_t;
    constexpr int CHUNK = 16;
    using WS = WorkspaceSizes<CHUNK, D>;

    // --- Parse forward workspace pointers ---
    int64_t n_ht = int64_t(H) * total_tiles;
    char const* ws = reinterpret_cast<char const*>(workspace_ptr);
    BF16 const* ws_kd  = reinterpret_cast<BF16 const*>(ws);
    BF16 const* ws_qd  = reinterpret_cast<BF16 const*>(ws + n_ht * WS::kKDecayed);
    BF16 const* ws_kr  = reinterpret_cast<BF16 const*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed));
    BF16 const* ws_ki  = reinterpret_cast<BF16 const*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored));
    float const* ws_gt = reinterpret_cast<float const*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kKInv));
    BF16 const* ws_inv = reinterpret_cast<BF16 const*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kKInv + WS::kGTotal));
    BF16 const* ws_mqk = reinterpret_cast<BF16 const*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kKInv + WS::kGTotal + WS::kINV));
    float const* ws_gc = reinterpret_cast<float const*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kKInv + WS::kGTotal + WS::kINV + WS::kMqk));

    // fp32 workspace for backward precision
    int64_t fp32_base = n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kKInv + WS::kGTotal + WS::kINV + WS::kMqk + WS::kGCumsum);
    float const* ws_kd_fp32 = reinterpret_cast<float const*>(ws + fp32_base);
    float const* ws_qd_fp32 = reinterpret_cast<float const*>(ws + fp32_base + n_ht * WS::kKDecayedFP32);
    float const* ws_ki_fp32 = reinterpret_cast<float const*>(ws + fp32_base + n_ht * (WS::kKDecayedFP32 + WS::kQDecayedFP32));
    float const* ws_kr_fp32 = reinterpret_cast<float const*>(ws + fp32_base + n_ht * (WS::kKDecayedFP32 + WS::kQDecayedFP32 + WS::kKInvFP32));

    // --- Allocate backward workspace (K2_bwd → K1_bwd) ---
    // We use cudaMalloc on the stream. Alternatively, caller could pre-allocate.
    // For simplicity, allocate here as temporary memory.
    // Per tile: dkd[C*D*4] + dqd[C*D*4] + dki[C*D*4] + dkr[C*D*4] + dv[C*D*2] + dgT[D*4] + dbeta[C*4]
    // dkd/dqd/dki/dkr stored as fp32 to preserve precision for dgc computation.
    int64_t bwd_ws_per_tile =
        4LL * CHUNK * D * sizeof(float) +  // dkd + dqd + dki + dkr (fp32)
        CHUNK * D * sizeof(BF16) +         // dv (bf16 ok, not used in dgc)
        D * sizeof(float) +                // dgT
        CHUNK * sizeof(float);             // dbeta
    int64_t bwd_ws_total = n_ht * bwd_ws_per_tile;

    void* bwd_ws_raw;
    cudaMallocAsync(&bwd_ws_raw, bwd_ws_total, stream);

    char* bws = reinterpret_cast<char*>(bwd_ws_raw);
    int64_t off = 0;
    float* bwd_dkd  = reinterpret_cast<float*>(bws + off); off += n_ht * CHUNK * D * sizeof(float);
    float* bwd_dqd  = reinterpret_cast<float*>(bws + off); off += n_ht * CHUNK * D * sizeof(float);
    float* bwd_dki  = reinterpret_cast<float*>(bws + off); off += n_ht * CHUNK * D * sizeof(float);
    float* bwd_dkr  = reinterpret_cast<float*>(bws + off); off += n_ht * CHUNK * D * sizeof(float);
    BF16* bwd_dv    = reinterpret_cast<BF16*>(bws + off); off += n_ht * CHUNK * D * sizeof(BF16);
    float* bwd_dgt  = reinterpret_cast<float*>(bws + off); off += n_ht * D * sizeof(float);
    float* bwd_dbeta = reinterpret_cast<float*>(bws + off); off += n_ht * CHUNK * sizeof(float);

    // ===== Launch K2_bwd (backward recurrence) =====
    {
        constexpr int kK2BwdThreads = 256;
        // Shared memory: dS[D*D] fp32 + scratch
        // scratch: gT[D] + beta[C] + INV[C*C] + Mqk[C*C] + vcorr[C*D] + U[C*D] + dU[C*D] + dkr[C*D] + dvcorr[C*D] + dL[C*C]
        //        = D + C + 3*C*C + 5*C*D floats
        constexpr int dS_floats = D * D;
        constexpr int scratch_floats = D + CHUNK + 3 * CHUNK * CHUNK + 5 * CHUNK * D;
        constexpr int smem_size_k2 = (dS_floats + scratch_floats) * sizeof(float);

        auto kernel2 = _flash_kda_bwd_recurrence<CHUNK, D, kK2BwdThreads, IsVarlen>;
        cudaFuncSetAttribute(kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k2);

        dim3 grid_k2(N, H);
        dim3 block_k2(kK2BwdThreads);

        kernel2<<<grid_k2, block_k2, smem_size_k2, stream>>>(
            do_ptr, v_ptr, beta_ptr,
            scale,
            ws_kd_fp32, ws_qd_fp32, ws_kr_fp32, ws_ki_fp32, ws_gt, ws_inv, ws_mqk,
            all_states_ptr,
            bwd_dkd, bwd_dqd, bwd_dki, bwd_dkr, bwd_dgt, bwd_dv, bwd_dbeta,
            ds_init_ptr, ds_out_ptr,
            T_total, H, N, cu_seqlens_ptr, total_tiles
        );
    }

    // ===== Launch K1_bwd (prepare backward) =====
    {
        constexpr int kK1BwdThreads = 256;
        // Shared memory: dgc[C*D]
        constexpr int smem_floats = CHUNK * D;
        constexpr int smem_size_k1 = smem_floats * sizeof(float);

        auto kernel1 = _flash_kda_bwd_prepare<CHUNK, D, kK1BwdThreads, IsVarlen>;
        cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k1);

        dim3 grid_k1(total_tiles, H);
        dim3 block_k1(kK1BwdThreads);

        kernel1<<<grid_k1, block_k1, smem_size_k1, stream>>>(
            q_ptr, k_ptr, g_bf16_ptr, beta_ptr, A_log_ptr, dt_bias_ptr,
            ws_kd_fp32, ws_qd_fp32, ws_kr_fp32, ws_ki_fp32, ws_gt, ws_gc,
            bwd_dkd, bwd_dqd, bwd_dki, bwd_dkr, bwd_dgt, bwd_dv, bwd_dbeta,
            dq_ptr, dk_ptr, dv_ptr, dg_ptr, dbeta_ptr, dA_log_ptr, ddt_bias_ptr,
            scale, gate_scale, T_total, H, N, cu_seqlens_ptr, total_tiles
        );
    }

    // Free backward workspace
    cudaFreeAsync(bwd_ws_raw, stream);
}

// Explicit instantiations
#define INSTANTIATE_LAUNCH_BWD(D, VL) \
    template void launch_bwd<D, VL>( \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, float const*, float const*, \
        float, float, void const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t*, cutlass::bfloat16_t*, cutlass::bfloat16_t*, \
        cutlass::bfloat16_t*, cutlass::bfloat16_t*, float*, float*, \
        cutlass::bfloat16_t*, int, int, int, int, \
        int64_t const*, cudaStream_t);

INSTANTIATE_LAUNCH_BWD(128, true)
INSTANTIATE_LAUNCH_BWD(128, false)
