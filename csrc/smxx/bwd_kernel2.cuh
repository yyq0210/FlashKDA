#pragma once

#include "utils.cuh"

// ==================== Kernel 2 Backward: Reverse Recurrence ====================
//
// Grid: (N, H) — each block handles one sequence, one head
// Block: 256 threads (no warp specialization — simpler than fwd K2)
//
// Iterates chunks from back to front.
// Reads from: do [T,H,D], workspace (kd,qd,kr,ki,gT,INV,Mqk), v, beta, all_states
// Writes to: bwd_workspace (dkd,dqd,dki,dkr,dv,dbeta_chunk,dgT per tile)
//            dS propagated in registers/smem across chunks
//
// Math per chunk (reverse order):
//   Load: do, kd, qd, kr, ki, gT, INV, Mqk, v, beta, S_in from all_states
//   Recompute: vcorr = (v - kd@S_in) * beta; U = INV @ vcorr
//   dout = do (incoming grad)
//   dqd_cross = dout @ S_in^T          contribution to dqd from cross-chunk
//   dS += qd^T @ dout                   cross-chunk state grad
//   dMqk = dout @ U^T                   within-chunk attention grad
//   dU_mqk = Mqk^T @ dout              contribution to dU from Mqk
//   dU_cross = kr @ dS                  contribution to dU from state update
//   dkr = dU_total^T @ ... => actually kr^T @ U contributes to S_new
//   ... (full derivation in comments below)
//
//   dS_prev = exp(gT) * dS + ... (propagate backward)

template <int CHUNK, int D, int NumThreads, bool IsVarlen = true>
__global__ void __launch_bounds__(NumThreads) _flash_kda_bwd_recurrence(
    // Input tensors (read-only)
    cutlass::bfloat16_t const* __restrict__ do_ptr,       // [T_total, H, D] (reordered as [H, T_total, D] for TMA compat)
    cutlass::bfloat16_t const* __restrict__ v_ptr,        // [H, T_total, D]
    cutlass::bfloat16_t const* __restrict__ beta_ptr,     // [H, T_total] (transposed)
    float scale,
    // Workspace from forward K1 (read-only, fp32 for backward precision)
    float const* __restrict__ ws_kd_ptr,    // [H*total_tiles, CHUNK, D] fp32
    float const* __restrict__ ws_qd_ptr,
    float const* __restrict__ ws_kr_ptr,
    float const* __restrict__ ws_ki_ptr,
    float const* __restrict__ ws_gt_ptr,                  // [H*total_tiles, D]
    cutlass::bfloat16_t const* __restrict__ ws_inv_ptr,   // [H*total_tiles, CHUNK, CHUNK]
    cutlass::bfloat16_t const* __restrict__ ws_mqk_ptr,
    // All states from forward K2
    cutlass::bfloat16_t const* __restrict__ all_states_ptr, // [N*H*max_tiles, D, D]
    // Backward workspace (write) — fp32 to preserve precision for dgc in K1_bwd
    float* __restrict__ bwd_ws_dkd_ptr,                    // [H*total_tiles, CHUNK, D] fp32
    float* __restrict__ bwd_ws_dqd_ptr,
    float* __restrict__ bwd_ws_dki_ptr,
    float* __restrict__ bwd_ws_dkr_ptr,
    float* __restrict__ bwd_ws_dgt_ptr,                    // [H*total_tiles, D]
    cutlass::bfloat16_t* __restrict__ bwd_ws_dv_ptr,       // [H*total_tiles, CHUNK, D]
    float* __restrict__ bwd_ws_dbeta_ptr,                  // [H*total_tiles, CHUNK]
    // dS_init (from dfinal_state, or zero)
    cutlass::bfloat16_t const* __restrict__ ds_init_ptr,   // [N*H, D, D] or nullptr
    // dS_out (d initial_state output)
    cutlass::bfloat16_t* __restrict__ ds_out_ptr,          // [N*H, D, D] or nullptr
    // Dimensions
    int T_total,
    int H,
    int N,
    int64_t const* __restrict__ cu_seqlens,
    int total_tiles
) {
    using BF16 = cutlass::bfloat16_t;
    constexpr int kWarpSize = 32;

    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    int64_t bos, eos;
    int tile_base;

    if constexpr (IsVarlen) {
        bos = cu_seqlens[seq_idx];
        eos = cu_seqlens[seq_idx + 1];
        tile_base = 0;
        for (int i = 0; i < seq_idx; i++) {
            tile_base += (int(cu_seqlens[i + 1] - cu_seqlens[i]) + CHUNK - 1) / CHUNK;
        }
    } else {
        int T_seq = T_total / N;
        bos = seq_idx * T_seq;
        eos = bos + T_seq;
        tile_base = seq_idx * ((T_seq + CHUNK - 1) / CHUNK);
    }
    int seq_len = int(eos - bos);
    int t_tiles = (seq_len + CHUNK - 1) / CHUNK;

    // --- Shared memory for dS accumulator [D, D] in fp32
    // Also used for loading workspace tiles
    extern __shared__ __align__(128) unsigned char shared_mem[];
    float* dS_smem = reinterpret_cast<float*>(shared_mem);  // [D*D]
    // After dS: scratch for loading tiles
    constexpr int dS_size = D * D;
    float* scratch_f32 = dS_smem + dS_size;

    // Initialize dS from ds_init or zero
    // ds_init (dfinal_state) has layout [V, K] (transposed state), but dS_smem is [K, V]
    // So we transpose on load: dS_smem[k*D + v] = src[v*D + k]
    {
        int state_linear = seq_idx * H + head_idx;
        if (ds_init_ptr != nullptr) {
            BF16 const* src = ds_init_ptr + int64_t(state_linear) * D * D;
            for (int i = tid; i < dS_size; i += NumThreads) {
                int k = i / D;
                int v_idx = i % D;
                dS_smem[i] = bf16_to_f32(src[v_idx * D + k]);
            }
        } else {
            for (int i = tid; i < dS_size; i += NumThreads) {
                dS_smem[i] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Iterate chunks from back to front
    for (int t = t_tiles - 1; t >= 0; --t) {
        int ws_idx = head_idx * total_tiles + tile_base + t;
        int actual_len = min(CHUNK, seq_len - t * CHUNK);
        constexpr int CD = CHUNK * D;

        // --- Load all needed data for this chunk into registers ---
        // We use a simple approach: each thread loads elements assigned to it
        // and we do the backward math element-wise or with simple reductions.

        // Pointers for this tile's workspace data
        float const* kd_tile  = ws_kd_ptr + int64_t(ws_idx) * CHUNK * D;
        float const* qd_tile  = ws_qd_ptr + int64_t(ws_idx) * CHUNK * D;
        float const* kr_tile  = ws_kr_ptr + int64_t(ws_idx) * CHUNK * D;
        float const* ki_tile  = ws_ki_ptr + int64_t(ws_idx) * CHUNK * D;
        float const* gt_tile = ws_gt_ptr + int64_t(ws_idx) * D;
        BF16 const* inv_tile = ws_inv_ptr + int64_t(ws_idx) * CHUNK * CHUNK;
        BF16 const* mqk_tile = ws_mqk_ptr + int64_t(ws_idx) * CHUNK * CHUNK;

        // v and do tiles: layout is [H, T_total, D], tile starts at [head_idx, bos + t*CHUNK, 0]
        int t_start = int(bos) + t * CHUNK;
        BF16 const* v_tile_ptr = v_ptr + (int64_t(head_idx) * T_total + t_start) * D;
        BF16 const* do_tile_ptr = do_ptr + (int64_t(head_idx) * T_total + t_start) * D;

        // Beta: [H, T_total], linear index = head_idx * T_total + t_start
        BF16 const* beta_tile_ptr = beta_ptr + int64_t(head_idx) * T_total + t_start;

        // S_in for this chunk: indexed same as workspace
        int state_idx = head_idx * total_tiles + tile_base + t;
        BF16 const* s_in_ptr = all_states_ptr + int64_t(state_idx) * D * D;

        // ========== Load tile data into shared memory ==========
        // We need: kd[C,D], qd[C,D], kr[C,D], ki[C,D], v[C,D], do[C,D],
        //          INV[C,C], Mqk[C,C], gT[D], beta[C], S_in[D,D]
        // Total smem needed beyond dS: quite a lot. Let's use registers where possible.

        // Load gT[D] into shared scratch
        float* gT_smem = scratch_f32; // [D]
        for (int i = tid; i < D; i += NumThreads) {
            gT_smem[i] = gt_tile[i];
        }
        __syncthreads();

        // Load beta[CHUNK] into first CHUNK floats of scratch after gT
        float* beta_smem = scratch_f32 + D; // [CHUNK]
        if (tid < CHUNK) {
            float b_raw = bf16_to_f32(beta_tile_ptr[tid]);
            beta_smem[tid] = sigmoid_tanh_approx_f32(b_raw);
        }
        __syncthreads();

        // Load INV[C,C] and Mqk[C,C] into shared
        float* inv_smem = scratch_f32 + D + CHUNK; // [C*C]
        float* mqk_smem = inv_smem + CHUNK * CHUNK; // [C*C]
        for (int i = tid; i < CHUNK * CHUNK; i += NumThreads) {
            inv_smem[i] = bf16_to_f32(inv_tile[i]);
            mqk_smem[i] = bf16_to_f32(mqk_tile[i]);
        }
        __syncthreads();

        // ========== Step 1: Recompute U ==========
        // vcorr = (v - kd @ S_in) * beta
        // U = INV @ vcorr
        // We compute these in shared memory using simple GEMM loops.

        // First: kd @ S_in -> tmp[C, D] (stored in smem)
        // S_in is [D, D], kd is [C, D], result is [C, D]
        // This is expensive: C*D*D = 16*128*128 = 262144 FMAs
        // With 256 threads, each thread does ~1024 FMAs

        // We'll compute kd_S_in[c][d] = sum_k kd[c][k] * S_in[k][d]
        // Thread assignment: each thread handles a subset of (c,d) pairs
        float* kd_S_smem = mqk_smem + CHUNK * CHUNK; // [C*D]

        // all_states stores S^T[V,K], so s_in_ptr[v*D+k] = S^T[v][k] = S[k][v]
        // We need kd_S[c][d] = sum_k kd[c][k] * S[k][d]
        // S[k][d] = s_in_ptr[d*D + k]
        for (int idx = tid; idx < CD; idx += NumThreads) {
            int c = idx / D;
            int d = idx % D;
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) {
                acc += kd_tile[c * D + k] * bf16_to_f32(s_in_ptr[d * D + k]);
            }
            kd_S_smem[idx] = acc;
        }
        __syncthreads();

        // vcorr[c][d] = (v[c][d] - kd_S[c][d]) * beta[c]
        float* vcorr_smem = kd_S_smem; // reuse same buffer
        for (int idx = tid; idx < CD; idx += NumThreads) {
            int c = idx / D;
            float v_val = (c < actual_len) ? bf16_to_f32(v_tile_ptr[c * D + idx % D]) : 0.0f;
            float beta_c = beta_smem[c];
            vcorr_smem[idx] = (v_val - kd_S_smem[idx]) * beta_c;
        }
        __syncthreads();

        // U = INV @ vcorr: U[c][d] = sum_j INV[c][j] * vcorr[j][d]
        float* U_smem = vcorr_smem + CD; // need new buffer for U since vcorr is still needed
        // Actually we can't reuse vcorr yet. Let's use a different region.
        // Let's reorganize: put U after the scratch we've used.
        // scratch layout: gT[D] | beta[C] | INV[C*C] | Mqk[C*C] | vcorr[C*D] | U[C*D]
        // That's D + C + 2*C*C + 2*C*D = 128+16+512+4096 = 4752 floats = 19008 bytes
        // Plus dS[D*D] = 16384 floats = 65536 bytes. Total ~84KB, within SM90 smem limits.
        U_smem = vcorr_smem + CD;  // [C*D]

        for (int idx = tid; idx < CD; idx += NumThreads) {
            int c = idx / D;
            int d = idx % D;
            float acc = 0.0f;
            for (int j = 0; j < CHUNK; ++j) {
                acc += inv_smem[c * CHUNK + j] * vcorr_smem[j * D + d];
            }
            U_smem[idx] = acc;
        }
        __syncthreads();

        // ========== Step 2: Compute gradients ==========
        // do_tile[C, D] is the output gradient for this chunk

        // dout = do[C, D]
        // out = qd @ S_in + Mqk @ U
        //
        // From out = qd @ S_in:
        //   dqd_cross[c][k] += sum_d do[c][d] * S_in[k][d]  (= do @ S_in^T)
        //   dS[k][d] += sum_c qd[c][k] * do[c][d]           (= qd^T @ do)
        //
        // From out += Mqk @ U:
        //   dMqk[c][j] = sum_d do[c][d] * U[j][d]           (= do @ U^T)
        //   dU[j][d] += sum_c Mqk[c][j]^T * do[c][d]        (= Mqk^T @ do)

        // First compute dU from Mqk path: dU_mqk = Mqk^T @ do
        float* dU_smem = U_smem + CD; // [C*D]
        // scratch: gT[D] | beta[C] | INV[C*C] | Mqk[C*C] | vcorr[C*D] | U[C*D] | dU[C*D]
        // = D + C + 2*C*C + 3*C*D = 128+16+512+6144 = 6800 floats = 27200B
        // plus dS = 65536B. Total ~92KB. Tight but should be OK on SM90.

        for (int idx = tid; idx < CD; idx += NumThreads) {
            int j = idx / D;  // row of dU (chunk dim)
            int d = idx % D;
            float acc = 0.0f;
            for (int c = 0; c < CHUNK; ++c) {
                float do_val = (c < actual_len) ? bf16_to_f32(do_tile_ptr[c * D + d]) : 0.0f;
                acc += mqk_smem[c * CHUNK + j] * do_val;  // Mqk^T: [j][c] = Mqk[c][j]
            }
            dU_smem[idx] = acc;
        }
        __syncthreads();

        // Add dU contribution from state update: S_new = S * exp(gT) + kr^T @ U
        // dkr[c][k] = sum_d U[c][d] * dS[k][d]  ... but actually:
        // S_new[k][d] += sum_c kr[c][k] * U[c][d] => kr^T @ U
        // dkr[c][k] = sum_d dS_new[k][d] * U[c][d]  = dS @ U^T then transpose? No:
        // dkr^T[k][c] = sum_d dS[k][d] * U[c][d]  => dkr^T = dS @ U^T => dkr = (dS @ U^T)^T = U @ dS^T
        // dU[c][d] += sum_k kr[c][k] * dS_new[k][d] = kr @ dS_new  (from the kr^T @ U term)
        // But dS_new = dS (current dS, since we iterate backward and dS comes from next chunk)

        // Add kr @ dS to dU
        for (int idx = tid; idx < CD; idx += NumThreads) {
            int c = idx / D;
            int d = idx % D;
            float acc = 0.0f;
            for (int k = 0; k < D; ++k) {
                acc += kr_tile[c * D + k] * dS_smem[k * D + d];
            }
            dU_smem[idx] += acc;
        }
        __syncthreads();

        // Now compute dkr from dS and U: dkr[c][k] = sum_d U[c][d] * dS[k][d]
        // = (U @ dS^T)  ... but we want dkr, and the update was S += kr^T @ U
        // More precisely: d(kr^T @ U)/dkr = ... dkr[c][k] = sum_d dS[k][d]*U[c][d]
        float* dkr_smem = dU_smem + CD; // [C*D]

        for (int idx = tid; idx < CD; idx += NumThreads) {
            int c = idx / D;
            int k = idx % D;
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) {
                acc += dS_smem[k * D + d] * U_smem[c * D + d];
            }
            dkr_smem[idx] = acc;
        }
        __syncthreads();

        // ========== Step 3: Triangular solve adjoint ==========
        // U = INV @ vcorr, so dvcorr = INV^T @ dU
        // (since U = A^{-1} @ vcorr where A = I+L, d/dvcorr = A^{-T} @ dU = INV^T @ dU)
        float* dvcorr_smem = dkr_smem + CD; // [C*D]
        float* dL_smem = dvcorr_smem + CD; // [C*C] — dedicated region, not aliased
        // scratch: gT[D]|beta[C]|INV[C*C]|Mqk[C*C]|vcorr[C*D]|U[C*D]|dU[C*D]|dkr[C*D]|dvcorr[C*D]|dL[C*C]
        // = D+C+2*C*C+5*C*D+C*C floats

        for (int idx = tid; idx < CD; idx += NumThreads) {
            int c = idx / D;
            int d = idx % D;
            float acc = 0.0f;
            for (int j = 0; j < CHUNK; ++j) {
                acc += inv_smem[j * CHUNK + c] * dU_smem[j * D + d]; // INV^T[c][j] = INV[j][c]
            }
            dvcorr_smem[idx] = acc;
        }
        __syncthreads();

        // ========== Step 3b: Compute dL while dvcorr is still alive ==========
        // dL[i][j] = -sum_d dvcorr[i][d] * U[j][d], strictly lower triangular (i > j)
        for (int idx = tid; idx < CHUNK * CHUNK; idx += NumThreads) {
            int i = idx / CHUNK;
            int j = idx % CHUNK;
            if (i > j) {
                float acc = 0.0f;
                for (int d = 0; d < D; ++d) {
                    acc += dvcorr_smem[i * D + d] * U_smem[j * D + d];
                }
                dL_smem[idx] = -acc;
            } else {
                dL_smem[idx] = 0.0f;
            }
        }
        __syncthreads();

        // ========== Step 4: Backward through vcorr = (v - kd@S) * beta ==========
        // dv[c][d] = dvcorr[c][d] * beta[c]
        // d(kd@S)[c][d] = -dvcorr[c][d] * beta[c]
        // dbeta_contrib[c] = sum_d dvcorr[c][d] * (v[c][d] - kd_S[c][d])
        //                  = sum_d dvcorr[c][d] * vcorr[c][d] / beta[c]  (if beta != 0)

        // Store dv to bwd workspace
        BF16* dv_out = bwd_ws_dv_ptr + int64_t(ws_idx) * CHUNK * D;
        for (int idx = tid; idx < CD; idx += NumThreads) {
            int c = idx / D;
            int d = idx % D;
            float dv_val = dvcorr_smem[idx] * beta_smem[c];
            dv_out[idx] = BF16(dv_val);
        }

        // dbeta_chunk[c] = sum_d dvcorr[c][d] * vcorr_orig[c][d] (where vcorr_orig = (v-kd@S))
        // We already overwrote vcorr_smem. We need vcorr_orig = vcorr / beta for non-zero beta.
        // Actually, vcorr = (v - kd@S) * beta, so (v - kd@S) = vcorr / beta.
        // dbeta[c] = sum_d dvcorr[c][d] * (v-kd@S)[c][d]
        //          = sum_d dvcorr[c][d] * vcorr[c][d] / beta[c]
        // But vcorr was overwritten... Let me recompute from U:
        // vcorr was overwritten but we have U = INV @ vcorr, so vcorr = (I+L) @ U
        // Actually wait, let me check what vcorr_smem contains at this point...
        // vcorr_smem was computed early, then we put U_smem after it. vcorr_smem should still be valid.
        // Let me trace: vcorr_smem = kd_S_smem = scratch_f32 + D + CHUNK + 2*C*C
        // After computing vcorr, we computed U into vcorr + CD, then dU into vcorr + 2*CD, etc.
        // vcorr_smem itself should still contain vcorr values. Good.

        // Compute dbeta per chunk row
        float* dbeta_out = bwd_ws_dbeta_ptr + int64_t(ws_idx) * CHUNK;
        if (tid < CHUNK) {
            int c = tid;
            float beta_c = beta_smem[c];
            float sum = 0.0f;
            if (beta_c > 1e-8f) {
                for (int d = 0; d < D; ++d) {
                    sum += dvcorr_smem[c * D + d] * vcorr_smem[c * D + d] / beta_c;
                }
            }
            dbeta_out[c] = sum;
        }
        __syncthreads();

        // d(kd@S)/dkd[c][k] = -beta[c] * sum_d dvcorr[c][d] * S_in[k][d]  (from -kd@S*beta part)
        // = -dvcorr_beta[c][d] @ S_in^T
        // dkd[c][k] = -sum_d (dvcorr[c][d]*beta[c]) * S_in[k][d]
        // Also from the Mqk@U backward, we need dMqk contributions. Let's handle dMqk:
        // dMqk[c][j] = sum_d do[c][d] * U[j][d]
        // But Mqk = tril(qd @ ki^T), so dqd += tril(dMqk) @ ki, dki += tril(dMqk)^T @ qd
        // For now, compute dMqk and apply tril mask.

        // Compute dMqk[C, C] = do @ U^T (only lower triangular matters since Mqk was tril)
        // Reuse some scratch. We can rewrite mqk_smem since we don't need Mqk anymore.
        float* dMqk_smem = mqk_smem; // reuse [C*C]
        for (int idx = tid; idx < CHUNK * CHUNK; idx += NumThreads) {
            int c = idx / CHUNK;
            int j = idx % CHUNK;
            if (c >= j) { // tril including diagonal
                float acc = 0.0f;
                for (int d = 0; d < D; ++d) {
                    float do_val = (c < actual_len) ? bf16_to_f32(do_tile_ptr[c * D + d]) : 0.0f;
                    acc += do_val * U_smem[j * D + d];
                }
                dMqk_smem[idx] = acc;
            } else {
                dMqk_smem[idx] = 0.0f;
            }
        }
        __syncthreads();

        // ========== Step 5: Accumulate dqd from cross-chunk and within-chunk ==========
        // dqd[c][k] = sum_d do[c][d] * S_in[k][d]  (cross-chunk: qd@S)
        //           + sum_j tril(dMqk)[c][j] * ki[j][k]  (within-chunk: Mqk@U, where Mqk=tril(qd@ki^T))
        float* dqd_local = dvcorr_smem; // reuse [C*D]
        for (int idx = tid; idx < CD; idx += NumThreads) {
            int c = idx / D;
            int k = idx % D;
            // Cross-chunk contribution: do[c] @ S_in[k]^T (dot over D)
            // S[k][d] = s_in_ptr[d*D + k] (since all_states stores S^T)
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) {
                float do_val = (c < actual_len) ? bf16_to_f32(do_tile_ptr[c * D + d]) : 0.0f;
                acc += do_val * bf16_to_f32(s_in_ptr[d * D + k]);
            }
            // Within-chunk: sum_j dMqk[c][j] * ki[j][k] (j <= c since tril)
            for (int j = 0; j <= c && j < CHUNK; ++j) {
                acc += dMqk_smem[c * CHUNK + j] * ki_tile[j * D + k];
            }
            dqd_local[idx] = acc;
        }
        __syncthreads();

        // Store dqd
        float* dqd_out = bwd_ws_dqd_ptr + int64_t(ws_idx) * CHUNK * D;
        for (int idx = tid; idx < CD; idx += NumThreads) {
            dqd_out[idx] = dqd_local[idx];
        }

        // ========== Step 6: dki from Mqk backward ==========
        // dki[j][k] = sum_c dMqk^T[j][c] * qd[c][k] = sum_{c>=j} dMqk[c][j] * qd[c][k]
        float* dki_local = dqd_local; // reuse [C*D]
        for (int idx = tid; idx < CD; idx += NumThreads) {
            int j = idx / D;
            int k = idx % D;
            float acc = 0.0f;
            for (int c = j; c < CHUNK; ++c) {
                acc += dMqk_smem[c * CHUNK + j] * qd_tile[c * D + k];
            }
            dki_local[idx] = acc;
        }
        __syncthreads();

        // ========== Step 7: Use dL (computed in step 3b while dvcorr was alive) ==========
        // L = tril(kd @ ki^T, -1) * beta[:,None]
        // dF = dL (where F = tril(kd@ki^T, -1), and L = F * beta)
        // d(F*beta)/dF = dL * beta[i], d(F*beta)/dbeta[i] = sum_j dL[i][j]*F[i][j]
        // dkd_L[c][k] += sum_{j<c} (dL[c][j] * beta[c]) * ki[j][k]  = sum_{j<c} dF[c][j] * ki[j][k]
        // dki_L[j][k] += sum_{c>j} (dL[c][j] * beta[c]) * kd[c][k]  = sum_{c>j} dF^T[j][c] * kd[c][k]

        // Compute dkd from L backward
        // Also accumulate dbeta from L: dbeta_L[c] = sum_{j<c} dL[c][j] * F[c][j]
        // F[c][j] = sum_k kd[c][k]*ki[j][k] for j < c

        // dkd contributions from the L-backward (kd@S term):
        // dkd_from_vcorr[c][k] = -beta[c] * sum_d dvcorr[c][d] * S_in[k][d]
        // dkd_from_L[c][k] = sum_{j<c} dL[c][j]*beta[c] * ki[j][k]
        float* dkd_local = dki_local + CD; // need fresh buffer
        // Hmm, we're running low on scratch. Let me reconsider the layout.
        // Actually dki_local already reused dqd_local which reused dvcorr_smem.
        // Let me use a simpler approach: store dki first, then reuse for dkd.

        // Store dki
        float* dki_out = bwd_ws_dki_ptr + int64_t(ws_idx) * CHUNK * D;
        for (int idx = tid; idx < CD; idx += NumThreads) {
            // Add dki from L backward: dki_L[j][k] = sum_{c>j} dL[c][j]*beta[c] * kd[c][k]
            int j = idx / D;
            int k = idx % D;
            float extra = 0.0f;
            for (int c = j + 1; c < CHUNK; ++c) {
                extra += dL_smem[c * CHUNK + j] * beta_smem[c] * kd_tile[c * D + k];
            }
            float dki_total = dki_local[idx] + extra;
            dki_out[idx] = dki_total;
        }
        __syncthreads();

        // Now compute dkd
        // dkd[c][k] = -beta[c] * sum_d dvcorr[c][d] * S_in[k][d]   (from vcorr = (v-kd@S)*beta)
        //           + sum_{j<c} dL[c][j]*beta[c] * ki[j][k]          (from L = tril(kd@ki^T,-1)*beta)
        // dvcorr_smem was overwritten, but INV is still in smem (we used dMqk_smem for dL).
        // Recompute dvcorr on the fly from smem INV and dU_smem:
        float* dkd_out = bwd_ws_dkd_ptr + int64_t(ws_idx) * CHUNK * D;
        for (int idx = tid; idx < CD; idx += NumThreads) {
            int c = idx / D;
            int k = idx % D;

            // dkd from vcorr backward: dvcorr[c][d] = sum_j INV^T[c][j] * dU[j][d]
            // INV is still in smem (preserved), dU_smem is still valid.
            float dvcorr_S_sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                // Recompute dvcorr[c][d] = sum_j INV^T[c][j] * dU[j][d]
                float dvcorr_cd = 0.0f;
                for (int j = 0; j < CHUNK; ++j) {
                    dvcorr_cd += inv_smem[j * CHUNK + c] * dU_smem[j * D + d]; // INV^T[c][j] = INV[j][c], from smem
                }
                dvcorr_S_sum += dvcorr_cd * bf16_to_f32(s_in_ptr[d * D + k]); // S[k][d] = s_in^T[d][k]
            }

            float dkd_val = -beta_smem[c] * dvcorr_S_sum;

            // Add L backward contribution: sum_{j<c} dL[c][j]*beta[c] * ki[j][k]
            for (int j = 0; j < c; ++j) {
                dkd_val += dL_smem[c * CHUNK + j] * beta_smem[c] * ki_tile[j * D + k];
            }

            dkd_out[idx] = dkd_val;
        }
        __syncthreads();

        // Add dbeta from L backward: sum_j dL[c][j] * F[c][j]
        // F[c][j] = sum_k kd[c][k]*ki[j][k] for j < c
        if (tid < CHUNK) {
            int c = tid;
            float dbeta_L = 0.0f;
            for (int j = 0; j < c; ++j) {
                float F_cj = 0.0f;
                for (int k = 0; k < D; ++k) {
                    F_cj += kd_tile[c * D + k] * ki_tile[j * D + k];
                }
                dbeta_L += dL_smem[c * CHUNK + j] * F_cj;
            }
            // Add to existing dbeta
            dbeta_out[c] += dbeta_L;
        }
        __syncthreads();

        // Store dkr
        float* dkr_out = bwd_ws_dkr_ptr + int64_t(ws_idx) * CHUNK * D;
        for (int idx = tid; idx < CD; idx += NumThreads) {
            dkr_out[idx] = dkr_smem[idx];
        }

        // ========== Step 8: Update dS for previous chunk ==========
        // dS_prev = exp(gT) * dS + qd^T @ do
        // From state update: S_new = S * exp(gT) + kr^T @ U
        // dS += exp(gT) * dS_next (already in dS_smem)
        // Also: dS += qd^T @ do (from out = qd @ S)
        // And: dS -= beta * dvcorr @ kd (from vcorr = (v-kd@S)*beta, d(kd@S)/dS = kd^T)
        // Wait, let me re-derive:
        // vcorr = (v - kd@S)*beta => d_loss/dS from this path = -kd^T @ (dvcorr * beta_vec)
        // But dvcorr already accounts for the beta multiplication? No:
        // vcorr[c][d] = (v[c][d] - sum_k kd[c][k]*S[k][d]) * beta[c]
        // d_loss/dS[k][d] = sum_c -kd[c][k] * beta[c] * d_loss/d_vcorr[c][d]
        //                 = sum_c -kd[c][k] * dvcorr_beta[c][d]
        // where dvcorr_beta[c][d] = beta[c] * d_vcorr/d_input? No, dvcorr is already the gradient w.r.t. vcorr.
        // dS[k][d] += sum_c (-kd[c][k] * beta[c]) * dvcorr[c][d]

        // For qd@S: dS[k][d] += sum_c qd[c][k] * do[c][d]

        // For S * exp(gT): dS_prev[k][d] = exp(gT[k]) * dS_next[k][d]

        // Recompute dvcorr for dS update from smem INV and dU_smem.
        // dS[k][d] = exp(gT[k]) * dS_current[k][d]
        //          + sum_c qd[c][k] * do[c][d]
        //          + sum_c (-kd[c][k] * beta[c]) * dvcorr[c][d]

        // ========== Step 8a: Compute dgT BEFORE updating dS ==========
        // dgT has TWO contributions:
        // 1) State update: S_new = S_in * exp2(gT) + kr^T @ U
        //    => dgT_state[k] = exp2(gT[k]) * sum_d dS_next[k][d] * S_in[k][d]
        // 2) kr dependency: kr = kn * exp2(gT - gc), so d(kr)/d(gT) = kr * ln2
        //    => dgT_kr[k] = sum_c dkr[c][k] * kr[c][k]   (without ln2, matching dgc convention)
        //    Note: dgc already has -dkr*kr for d(kr)/d(gc), but d(kr)/d(gT) is +dkr*kr
        float* dgT_out = bwd_ws_dgt_ptr + int64_t(ws_idx) * D;
        for (int k = tid; k < D; k += NumThreads) {
            float gT_k = gT_smem[k];
            // Contribution 1: state update
            float dgT_k = 0.0f;
            for (int d = 0; d < D; ++d) {
                // S[k][d] = s_in_ptr[d*D + k] (since all_states stores S^T)
                dgT_k += dS_smem[k * D + d] * bf16_to_f32(s_in_ptr[d * D + k]);
            }
            dgT_k *= gT_k;

            // Contribution 2: kr dependency — sum_c dkr[c][k] * kr[c][k]
            for (int c = 0; c < CHUNK; ++c) {
                dgT_k += dkr_smem[c * D + k] * kr_tile[c * D + k];
            }

            dgT_out[k] = dgT_k;
        }
        __syncthreads();

        // ========== Step 8b: Update dS for previous chunk ==========
        for (int idx = tid; idx < dS_size; idx += NumThreads) {
            int k = idx / D;
            int d = idx % D;

            float dS_val = gT_smem[k] * dS_smem[idx];  // gT_smem already stores exp2(gT)

            // qd^T @ do contribution
            float qd_do = 0.0f;
            for (int c = 0; c < actual_len; ++c) {
                float qd_ck = qd_tile[c * D + k];
                float do_cd = bf16_to_f32(do_tile_ptr[c * D + d]);
                qd_do += qd_ck * do_cd;
            }
            dS_val += qd_do;

            // -kd^T @ (beta * dvcorr) contribution
            float kd_dv = 0.0f;
            for (int c = 0; c < CHUNK; ++c) {
                float kd_ck = kd_tile[c * D + k];
                // Recompute dvcorr[c][d] = sum_j INV[j][c] * dU[j][d]
                float dvcorr_cd = 0.0f;
                for (int j = 0; j < CHUNK; ++j) {
                    dvcorr_cd += inv_smem[j * CHUNK + c] * dU_smem[j * D + d]; // from smem, not bf16 gmem
                }
                kd_dv += kd_ck * beta_smem[c] * dvcorr_cd;
            }
            dS_val -= kd_dv;

            dS_smem[idx] = dS_val;
        }
        __syncthreads();
    }

    // Store final dS (this is dS for the first chunk = d_initial_state)
    // dS_smem is [K, V] row-major: dS_smem[k*D + v] = dS[k][v]
    // d_initial_state layout is [V, K] (transposed state), so we store transposed:
    // dst[v*D + k] = dS[k][v]
    if (ds_out_ptr != nullptr) {
        int state_linear = seq_idx * H + head_idx;
        BF16* dst = ds_out_ptr + int64_t(state_linear) * D * D;
        for (int i = tid; i < dS_size; i += NumThreads) {
            int k = i / D;
            int v_idx = i % D;
            dst[v_idx * D + k] = BF16(dS_smem[i]);
        }
    }
}
