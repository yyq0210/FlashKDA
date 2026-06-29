#pragma once

#include "utils.cuh"

// ==================== Kernel 1 Backward: Prepare Backward ====================
//
// Grid: (total_tiles, H) — each CTA handles one chunk's one head
// Block: 256 threads (simple, no warp specialization)

template <int CHUNK, int D, int NumThreads, bool IsVarlen = true>
__global__ void __launch_bounds__(NumThreads) _flash_kda_bwd_prepare(
    // Forward inputs (for backward computation)
    cutlass::bfloat16_t const* __restrict__ q_ptr,         // [H, T_total, D]
    cutlass::bfloat16_t const* __restrict__ k_ptr,         // [H, T_total, D]
    cutlass::bfloat16_t const* __restrict__ g_bf16_ptr,    // [H, T_total, D] (raw gate, bf16)
    cutlass::bfloat16_t const* __restrict__ beta_ptr,      // [H, T_total] (transposed, pre-sigmoid)
    float const* __restrict__ A_log_ptr,                    // [H]
    float const* __restrict__ dt_bias_ptr,                  // [H, D]
    // Forward workspace (fp32 for backward precision)
    float const* __restrict__ ws_kd_ptr,     // [H*total_tiles, CHUNK, D] fp32
    float const* __restrict__ ws_qd_ptr,
    float const* __restrict__ ws_kr_ptr,
    float const* __restrict__ ws_ki_ptr,
    float const* __restrict__ ws_gt_ptr,                   // [H*total_tiles, D] (g_total fp32)
    float const* __restrict__ ws_gc_ptr,                   // [H*total_tiles, CHUNK, D] (gate cumsum fp32)
    // Backward workspace from K2_bwd (fp32 for precision)
    float const* __restrict__ bwd_ws_dkd_ptr,
    float const* __restrict__ bwd_ws_dqd_ptr,
    float const* __restrict__ bwd_ws_dki_ptr,
    float const* __restrict__ bwd_ws_dkr_ptr,
    float const* __restrict__ bwd_ws_dgt_ptr,               // [H*total_tiles, D]
    cutlass::bfloat16_t const* __restrict__ bwd_ws_dv_ptr,  // [H*total_tiles, CHUNK, D]
    float const* __restrict__ bwd_ws_dbeta_ptr,              // [H*total_tiles, CHUNK]
    // Output gradients
    cutlass::bfloat16_t* __restrict__ dq_ptr,               // [H, T_total, D]
    cutlass::bfloat16_t* __restrict__ dk_ptr,
    cutlass::bfloat16_t* __restrict__ dv_ptr,
    cutlass::bfloat16_t* __restrict__ dg_ptr,               // [H, T_total, D] (raw gate grad)
    cutlass::bfloat16_t* __restrict__ dbeta_ptr,            // [H, T_total] (logit space grad)
    float* __restrict__ dA_log_ptr,                         // [H] (atomicAdd)
    float* __restrict__ ddt_bias_ptr,                       // [H, D] (atomicAdd)
    // Dimensions
    float scale,
    float gate_scale,
    int T_total,
    int H,
    int N,
    int64_t const* __restrict__ cu_seqlens,
    int total_tiles
) {
    using BF16 = cutlass::bfloat16_t;
    constexpr int CD = CHUNK * D;
    constexpr int ELEMS_PER_THREAD = 8;
    constexpr int THREADS_PER_ROW = D / ELEMS_PER_THREAD; // 16

    int global_tile_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    // Thread layout matching forward K1 (16 threads per row, 8 elems each)
    int my_row = tid / THREADS_PER_ROW; // 0..15
    int my_col = (tid % THREADS_PER_ROW) * ELEMS_PER_THREAD; // 0, 8, 16, ..., 120

    // --- Resolve tile → sequence mapping
    int seq_idx, local_t;
    int64_t bos, eos;
    int seq_len, t_tiles_this_seq;

    if constexpr (IsVarlen) {
        seq_idx = -1;
        int tiles_before = 0;
        for (int i = 0; i < N; i++) {
            int slen = int(cu_seqlens[i + 1] - cu_seqlens[i]);
            int n_tiles = (slen + CHUNK - 1) / CHUNK;
            if (tiles_before + n_tiles > global_tile_idx) {
                seq_idx = i;
                break;
            }
            tiles_before += n_tiles;
        }
        // total_tiles is an upper bound for varlen, so phantom CTAs whose tile
        // index lands past the last real tile must be culled: otherwise they
        // fall through with seq_idx=0 and race the real CTAs on seq-0's outputs.
        if (seq_idx < 0) return;
        local_t = global_tile_idx - tiles_before;
        bos = cu_seqlens[seq_idx];
        eos = cu_seqlens[seq_idx + 1];
    } else {
        int T_seq = T_total / N;
        int tiles_per_seq = (T_seq + CHUNK - 1) / CHUNK;
        seq_idx = global_tile_idx / tiles_per_seq;
        int tiles_before = seq_idx * tiles_per_seq;
        local_t = global_tile_idx - tiles_before;
        bos = seq_idx * T_seq;
        eos = bos + T_seq;
    }
    seq_len = int(eos - bos);
    t_tiles_this_seq = (seq_len + CHUNK - 1) / CHUNK;
    if (local_t >= t_tiles_this_seq) return;

    int ws_idx = head_idx * total_tiles + global_tile_idx;
    int t_start = int(bos) + local_t * CHUNK;
    int actual_len = min(CHUNK, seq_len - local_t * CHUNK);

    float a_log_exp = expf(A_log_ptr[head_idx]);

    // Shared memory: dgc[C*D]
    extern __shared__ __align__(128) unsigned char shared_mem[];
    float* dgc_smem = reinterpret_cast<float*>(shared_mem);

    BF16 const* g_tile_ptr = g_bf16_ptr + (int64_t(head_idx) * T_total + t_start) * D;
    BF16 const* q_tile_ptr = q_ptr + (int64_t(head_idx) * T_total + t_start) * D;
    BF16 const* k_tile_ptr = k_ptr + (int64_t(head_idx) * T_total + t_start) * D;

    float const* gc_tile = ws_gc_ptr + int64_t(ws_idx) * CHUNK * D;
    float const* gt_tile = ws_gt_ptr + int64_t(ws_idx) * D;
    float const* dkd_tile = bwd_ws_dkd_ptr + int64_t(ws_idx) * CHUNK * D;
    float const* dqd_tile = bwd_ws_dqd_ptr + int64_t(ws_idx) * CHUNK * D;
    float const* dki_tile = bwd_ws_dki_ptr + int64_t(ws_idx) * CHUNK * D;
    float const* dkr_tile = bwd_ws_dkr_ptr + int64_t(ws_idx) * CHUNK * D;
    float const* dgt_tile = bwd_ws_dgt_ptr + int64_t(ws_idx) * D;
    BF16 const* dv_tile = bwd_ws_dv_ptr + int64_t(ws_idx) * CHUNK * D;
    float const* dbeta_tile = bwd_ws_dbeta_ptr + int64_t(ws_idx) * CHUNK;

    float const* kd_tile = ws_kd_ptr + int64_t(ws_idx) * CHUNK * D;
    float const* qd_tile = ws_qd_ptr + int64_t(ws_idx) * CHUNK * D;
    float const* ki_tile = ws_ki_ptr + int64_t(ws_idx) * CHUNK * D;
    float const* kr_tile = ws_kr_ptr + int64_t(ws_idx) * CHUNK * D;

    BF16* dq_out = dq_ptr + (int64_t(head_idx) * T_total + t_start) * D;
    BF16* dk_out = dk_ptr + (int64_t(head_idx) * T_total + t_start) * D;

    // ========== Phase 1+2: Fused dgc + dq/dk via L2 norm backward ==========
    // Numerically stable dgc formula:
    //   dgc = kn * (dkd*exp(gc) - dki*exp(-gc) - dkr*exp(gT-gc)) + qn*scale*dqd*exp(gc)
    // This avoids catastrophic cancellation in the original dkd*kd + dqd*qd - dki*ki - dkr*kr
    // because dkd*exp(gc), dki*exp(-gc), dkr*exp(gT-gc) are all O(1) magnitude,
    // while kd, ki can differ by ~1e20 making the original formula lose precision.
    {
        float q_vals[ELEMS_PER_THREAD], k_vals[ELEMS_PER_THREAD];
        float q_sq = 0.0f, k_sq = 0.0f;

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            float qv = (my_row < actual_len) ? bf16_to_f32(q_tile_ptr[my_row * D + my_col + i]) : 0.0f;
            float kv = (my_row < actual_len) ? bf16_to_f32(k_tile_ptr[my_row * D + my_col + i]) : 0.0f;
            q_vals[i] = qv;
            k_vals[i] = kv;
            q_sq += qv * qv;
            k_sq += kv * kv;
        }

        // 16-thread reduction matching forward K1 exactly
        #pragma unroll
        for (int delta = 8; delta >= 1; delta >>= 1) {
            q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, delta);
            k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, delta);
        }

        float q_inv_norm = rsqrtf(q_sq + 1e-6f);
        float k_inv_norm = rsqrtf(k_sq + 1e-6f);

        // Compute dqn, dkn, dgc using fp32 gc from workspace
        float dqn_vals[ELEMS_PER_THREAD], dkn_vals[ELEMS_PER_THREAD];
        float qn_vals[ELEMS_PER_THREAD], kn_vals[ELEMS_PER_THREAD];

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            int col = my_col + i;
            float gc_v = gc_tile[my_row * D + col];
            float exp_gc = ex2_approx_ftz_f32(gc_v);
            float exp_neg_gc = ex2_approx_ftz_f32(-gc_v);
            float exp_gt_gc = gt_tile[col] * exp_neg_gc;

            float dqd_v = dqd_tile[my_row * D + col];
            float dkd_v = dkd_tile[my_row * D + col];
            float dki_v = dki_tile[my_row * D + col];
            float dkr_v = dkr_tile[my_row * D + col];

            // dqn, dkn (for L2 norm backward)
            float dqn_v = dqd_v * exp_gc * scale;
            float dkn_v = dkd_v * exp_gc + dki_v * exp_neg_gc + dkr_v * exp_gt_gc;

            float qn_v = q_vals[i] * q_inv_norm;
            float kn_v = k_vals[i] * k_inv_norm;

            // Numerically stable dgc:
            // dkn_signed = dkd*exp(gc) - dki*exp(-gc) - dkr*exp(gT-gc)
            // Each term is O(1), subtraction is well-conditioned
            float dkn_signed = dkd_v * exp_gc - dki_v * exp_neg_gc - dkr_v * exp_gt_gc;
            dgc_smem[my_row * D + col] = kn_v * dkn_signed + qn_v * dqn_v;

            dqn_vals[i] = dqn_v;
            dkn_vals[i] = dkn_v;
            qn_vals[i] = qn_v;
            kn_vals[i] = kn_v;
        }

        float dot_dqn_qn = 0.0f, dot_dkn_kn = 0.0f;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            dot_dqn_qn += dqn_vals[i] * qn_vals[i];
            dot_dkn_kn += dkn_vals[i] * kn_vals[i];
        }
        #pragma unroll
        for (int delta = 8; delta >= 1; delta >>= 1) {
            dot_dqn_qn += __shfl_xor_sync(0xFFFFFFFF, dot_dqn_qn, delta);
            dot_dkn_kn += __shfl_xor_sync(0xFFFFFFFF, dot_dkn_kn, delta);
        }

        if (my_row < actual_len) {
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                int col = my_col + i;
                float dq_v = (dqn_vals[i] - qn_vals[i] * dot_dqn_qn) * q_inv_norm;
                float dk_v = (dkn_vals[i] - kn_vals[i] * dot_dkn_kn) * k_inv_norm;
                dq_out[my_row * D + col] = BF16(dq_v);
                dk_out[my_row * D + col] = BF16(dk_v);
            }
        }
    }
    __syncthreads();

    // ========== Phase 3: dv — copy from bwd workspace ==========
    BF16* dv_out = dv_ptr + (int64_t(head_idx) * T_total + t_start) * D;
    for (int idx = tid; idx < actual_len * D; idx += NumThreads) {
        dv_out[idx] = dv_tile[idx];
    }

    // ========== Phase 4: Reverse cumsum of dgc + dgT → gate backward ==========
    BF16* dg_out = dg_ptr + (int64_t(head_idx) * T_total + t_start) * D;

    for (int col = tid; col < D; col += NumThreads) {
        
        float dt = dt_bias_ptr[head_idx * D + col];
        float rev_sum = 0.0f;
        float dgT_val = dgt_tile[col];

        float dA_log_acc = 0.0f;
        float ddt_bias_acc = 0.0f;

        for (int row = CHUNK - 1; row >= 0; --row) {
            rev_sum += dgc_smem[row * D + col];
            float dg_nat = rev_sum + dgT_val;

            if (row < actual_len) {
                float g_raw = bf16_to_f32(g_tile_ptr[row * D + col]);
                float z = a_log_exp * (g_raw + dt);
                float sig = sigmoid_tanh_approx_f32(z);
                float dsig = sig * (1.0f - sig);
                constexpr float kLn2 = 0.6931471805599453f;
                float dz = dg_nat * gate_scale * dsig * kLn2;
                float dg_raw = dz * a_log_exp;

                dg_out[row * D + col] = BF16(dg_raw);

                ddt_bias_acc += dz * a_log_exp;
                dA_log_acc += dz * (g_raw + dt);
            } else {
                dg_out[row * D + col] = BF16(0.0f);
            }
        }

        atomicAdd(&ddt_bias_ptr[head_idx * D + col], ddt_bias_acc);
        atomicAdd(&dA_log_ptr[head_idx], dA_log_acc * a_log_exp);
    }

    // ========== Phase 5: dbeta — convert from chunk-space to logit-space ==========
    BF16 const* beta_tile_ptr = beta_ptr + int64_t(head_idx) * T_total + t_start;
    BF16* dbeta_out = dbeta_ptr + int64_t(head_idx) * T_total + t_start;
    if (tid < actual_len) {
        float dbeta_c = dbeta_tile[tid];
        float beta_raw = bf16_to_f32(beta_tile_ptr[tid]);
        float sig = sigmoid_tanh_approx_f32(beta_raw);
        float dbeta_logit = dbeta_c * sig * (1.0f - sig);
        dbeta_out[tid] = BF16(dbeta_logit);
    }
}
