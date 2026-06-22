"""Trainable (autograd) wrapper for FlashKDA.

``FlashKDAFunction`` is a ``torch.autograd.Function`` whose forward calls the
fast CUDA ``flash_kda.fwd`` kernel, and whose backward calls the CUDA
``flash_kda.bwd`` kernel using saved workspace and all_states from the forward
pass.

``flash_kda_func`` is the convenience entry point returning ``(out, final_state)``.

Math (per head, per chunk of size ``CHUNK``; all fp32):
    qn = l2norm(q); kn = l2norm(k)                       # over feature dim
    g  = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))
    gc = cumsum(g)  over time within the chunk;  gT = gc[-1]
    kd = kn*exp(gc);  qd = qn*exp(gc)*scale
    ki = kn*exp(-gc); kr = kn*exp(gT - gc)
    L  = tril(kd @ ki^T, -1) * sigmoid(beta)[:, None]     # strictly lower
    Mqk = tril(qd @ ki^T)                                 # incl diagonal
    vcorr = (v - kd @ S) * sigmoid(beta)[:, None]         # S is [K, V]
    U  = (I - L)^{-1} @ vcorr
    o  = qd @ S + Mqk @ U
    S  = S * exp(gT)[:, None] + kr^T @ U
"""

import torch
import torch.nn.functional as F

import flash_kda
from flash_kda_C import get_workspace_size

CHUNK = 16
_L2_EPS = 1e-6


def _l2norm(x):
    """L2-normalize over the last dim, matching FLA (eps inside the sqrt)."""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + _L2_EPS)


def _gate(g, A_log, dt_bias, lower_bound):
    """FlashKDA lower-bound sigmoid gate -> natural-log decay values [..., H, D]."""
    H, D = dt_bias.shape
    a = torch.exp(A_log).view(*([1] * (g.dim() - 2)), H, 1)
    return lower_bound * torch.sigmoid(a * (g + dt_bias.view(*([1] * (g.dim() - 2)), H, D)))


def _run_segment(qn, kn, v, g_nat, beta_logit, S0, scale):
    """One sequence. Tensors: qn/kn/g_nat [L,H,K], v [L,H,V], beta_logit [L,H],
    S0 [H,K,V]. Returns (o [L,H,V], S_final [H,K,V])."""
    L_len, H, K = qn.shape
    V = v.shape[-1]
    pad = (CHUNK - L_len % CHUNK) % CHUNK
    if pad:
        zk = qn.new_zeros(pad, H, K)
        qn = torch.cat([qn, zk], 0)
        kn = torch.cat([kn, zk], 0)
        g_nat = torch.cat([g_nat, zk], 0)
        v = torch.cat([v, v.new_zeros(pad, H, V)], 0)
        beta_logit = torch.cat([beta_logit, beta_logit.new_zeros(pad, H)], 0)
    NC = (L_len + pad) // CHUNK

    # -> [NC, H, CHUNK, *]
    def chunkify(x):
        return x.view(NC, CHUNK, H, -1).permute(0, 2, 1, 3).contiguous()

    qc = chunkify(qn)            # [NC,H,CHUNK,K]
    kc = chunkify(kn)
    gc = chunkify(g_nat)
    vc = chunkify(v)             # [NC,H,CHUNK,V]
    bc = chunkify(beta_logit.unsqueeze(-1)).squeeze(-1)  # [NC,H,CHUNK]

    gcum = gc.cumsum(dim=2)                      # [NC,H,CHUNK,K]
    gT = gcum[:, :, -1, :]                       # [NC,H,K]
    e_gc = gcum.exp()
    kd = kc * e_gc
    qd = qc * e_gc * scale
    ki = kc * (-gcum).exp()
    kr = kc * (gT[:, :, None, :] - gcum).exp()
    beta_s = bc.sigmoid()                         # [NC,H,CHUNK]

    Lmat = torch.tril(kd @ ki.transpose(-1, -2), -1) * beta_s[..., None]
    Mqk = torch.tril(qd @ ki.transpose(-1, -2))
    eye = torch.eye(CHUNK, dtype=qn.dtype, device=qn.device).expand(NC, H, CHUNK, CHUNK)
    IpL = eye + Lmat                              # unit lower-triangular; U = (I+L)^{-1} vcorr

    outs = []
    S = S0                                        # [H,K,V]
    for n in range(NC):
        vcorr = (vc[n] - kd[n] @ S) * beta_s[n][..., None]      # [H,CHUNK,V]
        U = torch.linalg.solve_triangular(IpL[n], vcorr, upper=False, unitriangular=True)
        o_n = qd[n] @ S + Mqk[n] @ U                            # [H,CHUNK,V]
        S = S * gT[n].exp()[..., None] + kr[n].transpose(-1, -2) @ U
        outs.append(o_n)

    o = torch.stack(outs, 0).permute(0, 2, 1, 3).reshape(NC * CHUNK, H, V)
    return o[:L_len], S


def _kda_chunk_torch_fwd(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound,
                         initial_state, output_final_state, cu_seqlens):
    """Pure-PyTorch differentiable KDA chunk forward (fp32 inside).

    Inputs: q,k,v,g [B,T,H,D]; beta [B,T,H]; A_log [H]; dt_bias [H,D];
    initial_state [N,H,V,K] or None. Returns (out [B,T,H,D], final_state).
    """
    B, T, H, D = q.shape
    cdt = torch.float64 if q.dtype == torch.float64 else torch.float32
    qf, kf, vf = q.to(cdt), k.to(cdt), v.to(cdt)
    qn = _l2norm(qf)
    kn = _l2norm(kf)
    g_nat = _gate(g.to(cdt), A_log.to(cdt), dt_bias.to(cdt), lower_bound)   # [B,T,H,D]
    beta_l = beta.to(cdt)

    if initial_state is not None:
        S0_all = initial_state.to(cdt).transpose(-1, -2).contiguous()       # [N,H,K,V]

    if cu_seqlens is None:
        outs, finals = [], []
        for b in range(B):
            S0 = S0_all[b] if initial_state is not None else qn.new_zeros(H, D, D)
            o_b, S_b = _run_segment(qn[b], kn[b], vf[b], g_nat[b], beta_l[b], S0, scale)
            outs.append(o_b)
            finals.append(S_b)
        out = torch.stack(outs, 0)                                          # [B,T,H,V]
        final = torch.stack(finals, 0).transpose(-1, -2) if output_final_state else None
    else:
        assert B == 1, "varlen requires B==1"
        cs = cu_seqlens.tolist()
        N = len(cs) - 1
        out = qn.new_zeros(1, T, H, D)
        finals = []
        for s in range(N):
            bos, eos = cs[s], cs[s + 1]
            S0 = S0_all[s] if initial_state is not None else qn.new_zeros(H, D, D)
            o_s, S_s = _run_segment(qn[0, bos:eos], kn[0, bos:eos], vf[0, bos:eos],
                                    g_nat[0, bos:eos], beta_l[0, bos:eos], S0, scale)
            out[0, bos:eos] = o_s
            finals.append(S_s)
        final = torch.stack(finals, 0).transpose(-1, -2) if output_final_state else None

    return out, final


def _compute_total_tiles(T_total, N, cu_seqlens, CHUNK=16):
    """Compute total_tiles matching the C++ logic."""
    if cu_seqlens is not None:
        return (T_total + CHUNK - 1) // CHUNK + N  # upper bound for varlen
    else:
        # Batched: N sequences, each T_total/N length
        T_seq = T_total // N
        return N * ((T_seq + CHUNK - 1) // CHUNK)


class FlashKDAFunction(torch.autograd.Function):
    """Fast CUDA forward + CUDA backward using saved workspace."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, A_log, dt_bias, lower_bound,
                initial_state, output_final_state, cu_seqlens):
        B, T, H, D = q.shape
        N = (cu_seqlens.numel() - 1) if cu_seqlens is not None else B
        T_total = B * T

        out = torch.empty(B, T, H, D, dtype=torch.bfloat16, device=q.device)
        final_state = None
        if output_final_state:
            fs_dtype = initial_state.dtype if initial_state is not None else torch.float32
            final_state = torch.empty(N, H, D, D, dtype=fs_dtype, device=q.device)

        init_bf = None
        if initial_state is not None:
            if output_final_state:
                init_bf = initial_state if initial_state.dtype == final_state.dtype else \
                    initial_state.to(final_state.dtype)
            else:
                init_bf = initial_state.to(torch.bfloat16)
            init_bf = init_bf.contiguous()

        # Prepare inputs as bf16
        q_bf = q.to(torch.bfloat16).contiguous()
        k_bf = k.to(torch.bfloat16).contiguous()
        v_bf = v.to(torch.bfloat16).contiguous()
        g_bf = g.to(torch.bfloat16).contiguous()
        beta_bf = beta.to(torch.bfloat16).contiguous()
        A_log_f = A_log.float().contiguous()
        dt_bias_f = dt_bias.float().contiguous()

        # Compute total tiles for all_states allocation
        total_tiles = _compute_total_tiles(T_total, N, cu_seqlens)

        # Allocate all_states buffer: [N*H*total_tiles_per_seq, D, D]
        # The K2 kernel stores state per (seq_idx * H + head_idx) * t_tiles + t
        # Total entries = sum over sequences of (H * t_tiles_for_that_seq)
        # Upper bound: H * total_tiles (same as workspace indexing)
        all_states = torch.empty(H * total_tiles, D, D, dtype=torch.bfloat16, device=q.device)

        workspace = flash_kda.fwd(
            q_bf, k_bf, v_bf, g_bf, beta_bf, float(scale), out,
            A_log=A_log_f, dt_bias=dt_bias_f,
            lower_bound=float(lower_bound),
            initial_state=init_bf, final_state=final_state, cu_seqlens=cu_seqlens,
            all_states=all_states,
        )

        ctx.save_for_backward(q_bf, k_bf, v_bf, g_bf, beta_bf, A_log_f, dt_bias_f,
                              initial_state, workspace, all_states)
        ctx.scale = float(scale)
        ctx.lower_bound = float(lower_bound)
        ctx.output_final_state = output_final_state
        ctx.cu_seqlens = cu_seqlens
        ctx.N = N
        ctx.total_tiles = total_tiles
        return out, final_state

    @staticmethod
    def backward(ctx, do, dfinal_state):
        (q_bf, k_bf, v_bf, g_bf, beta_bf, A_log_f, dt_bias_f,
         initial_state, workspace, all_states) = ctx.saved_tensors
        B, T, H, D = q_bf.shape

        # Allocate output gradient tensors
        dq = torch.zeros_like(q_bf)
        dk = torch.zeros_like(k_bf)
        dv = torch.zeros_like(v_bf)
        dg = torch.zeros_like(g_bf)
        dbeta = torch.zeros(B, T, H, dtype=torch.bfloat16, device=q_bf.device)
        dA_log = torch.zeros_like(A_log_f)
        ddt_bias = torch.zeros_like(dt_bias_f)

        # d_initial_state
        dinitial_state = None
        if initial_state is not None:
            dinitial_state = torch.zeros(ctx.N, H, D, D, dtype=initial_state.dtype, device=q_bf.device)

        flash_kda.bwd(
            q_bf, k_bf, v_bf, g_bf, beta_bf,
            ctx.scale, workspace, all_states,
            do.to(torch.bfloat16).contiguous(),
            A_log_f, dt_bias_f, ctx.lower_bound,
            dq, dk, dv, dg, dbeta, dA_log, ddt_bias,
            dfinal_state=dfinal_state if dfinal_state is not None else None,
            dinitial_state=dinitial_state,
            cu_seqlens=ctx.cu_seqlens,
        )

        # forward args: q,k,v,g,beta,scale,A_log,dt_bias,lower_bound,initial_state,ofs,cu_seqlens
        return dq, dk, dv, dg, dbeta, None, dA_log, ddt_bias, None, dinitial_state, None, None


def flash_kda_func(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound,
                   initial_state=None, output_final_state=False, cu_seqlens=None):
    """Trainable FlashKDA: fast CUDA forward + CUDA backward.

    Args mirror ``flash_kda.fwd`` (q,k,v,g,beta bf16 ``[B,T,H,128]``; beta logits
    ``[B,T,H]``; A_log fp32 ``[H]``; dt_bias fp32 ``[H,128]``; initial_state
    ``[N,H,V,K]``). Returns ``(out, final_state)`` where ``final_state`` is
    ``None`` unless ``output_final_state=True``.
    """
    return FlashKDAFunction.apply(
        q, k, v, g, beta, scale, A_log, dt_bias, lower_bound,
        initial_state, output_final_state, cu_seqlens,
    )
