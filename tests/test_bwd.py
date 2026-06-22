"""Backward correctness for FlashKDA's trainable wrapper (flash_kda_func).

Validates:
  1. forward value of flash_kda_func vs the pure-fp32 torch reference and vs the
     CUDA kernel (sanity that the recompute math matches the kernel).
  2. all input gradients (dq, dk, dv, dg, dbeta, dA_log, ddt_bias, dh0) against
     FLA's chunk_kda (which has a verified Triton backward), on identical raw
     inputs with matching gate semantics (lower-bound sigmoid).

Run:
    source /home/fluentllmenv/bin/activate && \
        python tests/test_bwd.py
"""

import math

import torch
import torch.nn.functional as F

import flash_kda
from flash_kda.autograd import _kda_chunk_torch_fwd


def err_ratio(x, y):
    x, y = x.detach().float(), y.detach().float()
    return ((x - y).square().mean().sqrt() / (y.square().mean().sqrt() + 1e-8)).item()


def make_inputs(B, T, H, D, lower_bound, seed=0, device="cuda", cu_seqlens=None):
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, D, device=device), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, D, device=device), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=device)
    g = (torch.randn(B, T, H, D, device=device) * 4).to(torch.bfloat16)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device=device)
    A_log = torch.rand(H, dtype=torch.float32, device=device)
    dt_bias = (torch.rand(H, D, dtype=torch.float32, device=device) - 0.5) * 4
    N = (cu_seqlens.numel() - 1) if cu_seqlens is not None else B
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, g, beta, A_log, dt_bias, h0, scale


def run_fla(q, k, v, g, beta, A_log, dt_bias, h0, scale, lower_bound, cu_seqlens):
    from fla.ops.kda import chunk_kda
    o, ht = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
        transpose_state_layout=True, cu_seqlens=cu_seqlens,
    )
    return o, ht


def _clone_leaf(t):
    return t.detach().clone().requires_grad_(True)


def check(label, B, T, H, D, lower_bound, cu_seqlens=None):
    print(f"\n=== {label}: B={B} T={T} H={H} D={D} lb={lower_bound} "
          f"varlen={cu_seqlens is not None} ===")
    q, k, v, g, beta, A_log, dt_bias, h0, scale = make_inputs(
        B, T, H, D, lower_bound, cu_seqlens=cu_seqlens)

    # ---- forward sanity: torch ref vs flash kernel (inference path) ----
    out_ref, ht_ref = _kda_chunk_torch_fwd(
        q, k, v, g, beta, scale, A_log, dt_bias, lower_bound,
        h0, True, cu_seqlens)
    with torch.no_grad():
        out_fk, ht_fk = flash_kda.flash_kda_func(
            q, k, v, g, beta, scale, A_log, dt_bias, lower_bound,
            initial_state=h0, output_final_state=True, cu_seqlens=cu_seqlens)
    print(f"  fwd  o  torch-ref vs kernel : {err_ratio(out_fk, out_ref):.3e}")
    print(f"  fwd  ht torch-ref vs kernel : {err_ratio(ht_fk, ht_ref):.3e}")

    do = torch.randn_like(out_ref)
    dht = torch.randn_like(ht_ref)

    # ---- FlashKDA trainable wrapper grads ----
    fk_in = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log, dt_bias, h0)]
    o_fk, ht_fk2 = flash_kda.flash_kda_func(
        fk_in[0], fk_in[1], fk_in[2], fk_in[3], fk_in[4], scale,
        fk_in[5], fk_in[6], lower_bound,
        initial_state=fk_in[7], output_final_state=True, cu_seqlens=cu_seqlens)
    ((o_fk.float() * do).sum() + (ht_fk2.float() * dht).sum()).backward()
    fk_grads = [x.grad for x in fk_in]

    # ---- FLA chunk_kda grads (gold). fla returns ddt_bias flattened [H*K]. ----
    fl_in = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log)]
    dt_flat = dt_bias.detach().reshape(-1).clone().requires_grad_(True)
    h0_fl = _clone_leaf(h0)
    o_fl, ht_fl = run_fla(
        fl_in[0], fl_in[1], fl_in[2], fl_in[3], fl_in[4],
        fl_in[5], dt_flat, h0_fl, scale, lower_bound, cu_seqlens)
    ((o_fl.float() * do).sum() + (ht_fl.float() * dht).sum()).backward()
    fl_grads = [x.grad for x in fl_in] + [dt_flat.grad.reshape(dt_bias.shape), h0_fl.grad]

    print(f"  fwd  o  flash vs fla        : {err_ratio(o_fk, o_fl):.3e}")
    names = ["dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias", "dh0"]
    ok = True
    for nm, gfk, gfl in zip(names, fk_grads, fl_grads):
        r = err_ratio(gfk, gfl)
        flag = "OK" if r < 2e-2 else "**FAIL**"
        if r >= 2e-2:
            ok = False
        print(f"  grad {nm:<9} flash vs fla : {r:.3e}  {flag}")
    assert ok, f"{label}: gradient mismatch"
    print(f"  {label}: PASS")


def main():
    check("batched", B=2, T=256, H=3, D=128, lower_bound=-5.0)
    check("batched_lb0", B=1, T=512, H=2, D=128, lower_bound=-1.0)
    seq_lens = [192, 64, 304]
    cu = torch.tensor([0] + torch.cumsum(torch.tensor(seq_lens), 0).tolist(),
                      dtype=torch.long, device="cuda")
    check("varlen", B=1, T=sum(seq_lens), H=2, D=128, lower_bound=-5.0, cu_seqlens=cu)
    print("\nAll backward tests PASS")


if __name__ == "__main__":
    main()
