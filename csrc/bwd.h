#pragma once
#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>

template <int D, bool IsVarlen = true>
void launch_bwd(
    // Forward inputs
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,      // [H, T_total] transposed
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float scale,
    float gate_scale,
    // Forward workspace (from K1 fwd)
    void const* workspace_ptr,
    // do (output gradient)
    cutlass::bfloat16_t const* do_ptr,
    // all_states from forward K2
    cutlass::bfloat16_t const* all_states_ptr,
    // dS_init (from dfinal_state, or nullptr)
    cutlass::bfloat16_t const* ds_init_ptr,
    // Output gradients
    cutlass::bfloat16_t* dq_ptr,
    cutlass::bfloat16_t* dk_ptr,
    cutlass::bfloat16_t* dv_ptr,
    cutlass::bfloat16_t* dg_ptr,
    cutlass::bfloat16_t* dbeta_ptr,
    float* dA_log_ptr,
    float* ddt_bias_ptr,
    cutlass::bfloat16_t* ds_out_ptr,        // d_initial_state output
    // Dimensions
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens_ptr,
    cudaStream_t stream
);
