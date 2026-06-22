import torch
from flash_kda_C import fwd as _fwd_raw, bwd as _bwd_raw, get_workspace_size


def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=None, final_state=None, cu_seqlens=None, all_states=None):
    """FlashKDA forward (Flash Kimi Delta Attention).

    Args:
        q (torch.Tensor): Query, bf16, shape ``[B, T, H, K]``.
        k (torch.Tensor): Key, bf16, shape ``[B, T, H, K]``.
        v (torch.Tensor): Value, bf16, shape ``[B, T, H, V]``.
        g (torch.Tensor): Gate before activation, bf16, shape ``[B, T, H, K]``.
        beta (torch.Tensor): Beta logits (pre-activation; sigmoid is applied
            internally), bf16, shape ``[B, T, H]``.
        scale (float): Scaling factor.
        out (torch.Tensor): Output buffer, bf16, shape ``[B, T, H, V]``. Written
            in place.
        A_log (torch.Tensor): Log-gate parameter, fp32, shape ``[H]``.
        dt_bias (torch.Tensor): Gate bias, fp32, shape ``[H, K]``.
        lower_bound (float): Gate lower bound, expected in ``[-5.0, 0]``.
        initial_state (torch.Tensor, optional): Initial recurrent state, bf16
            or fp32. Shape ``[B, H, V, K]`` for batched mode, or ``[N, H, V, K]``
            for varlen mode. ``None`` means start from zero.
        final_state (torch.Tensor, optional): Output buffer for the final
            recurrent state. Same dtype/shape rules as ``initial_state``.
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths, int64,
            shape ``[N+1]``. When provided, ``B`` must be 1.
        all_states (torch.Tensor, optional): Buffer to save all chunk states
            for backward pass. Shape ``[N*H*total_tiles, D, D]`` bf16.
            If ``None``, states are not saved.

    Notes:
        * Currently requires ``K = V = 128``.
        * All input tensors must be CUDA, contiguous, and have the dtypes
          listed above.
    """
    B, T_seq, H = q.shape[0], q.shape[1], q.shape[2]
    T_total = B * T_seq
    N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B

    workspace = torch.empty(get_workspace_size(T_total, H, N), dtype=torch.uint8, device=q.device)

    _fwd_raw(q, k, v, g, beta, float(scale), out, workspace, A_log, dt_bias, lower_bound,
             initial_state=initial_state, final_state=final_state, cu_seqlens=cu_seqlens,
             all_states=all_states)

    return workspace


def bwd(q, k, v, g, beta, scale, workspace, all_states, do_tensor,
        A_log, dt_bias, lower_bound,
        dq, dk, dv, dg, dbeta, dA_log, ddt_bias,
        dfinal_state=None, dinitial_state=None, cu_seqlens=None):
    """FlashKDA backward (CUDA kernel).

    All gradient tensors (dq, dk, dv, dg, dbeta) must be pre-allocated.
    dA_log and ddt_bias are accumulated (atomicAdd), so they should be zeroed before calling.
    """
    _bwd_raw(q, k, v, g, beta, float(scale), workspace, all_states, do_tensor,
             A_log, dt_bias, lower_bound,
             dq, dk, dv, dg, dbeta, dA_log, ddt_bias,
             dfinal_state=dfinal_state, dinitial_state=dinitial_state,
             cu_seqlens=cu_seqlens)


from .autograd import FlashKDAFunction, flash_kda_func  # noqa: E402

__all__ = ["fwd", "bwd", "get_workspace_size", "FlashKDAFunction", "flash_kda_func"]
