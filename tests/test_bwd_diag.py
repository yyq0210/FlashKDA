"""Minimal diagnostic test for backward kernel correctness.

Tests single chunk (T=16) with various configurations to isolate issues.
"""
import math
import torch
import torch.nn.functional as F
import flash_kda
from flash_kda.autograd import _kda_chunk_torch_fwd


def err_ratio(x, y):
    x, y = x.detach().float(), y.detach().float()
    return ((x - y).square().mean().sqrt() / (y.square().mean().sqrt() + 1e-8)).item()


def _clone_leaf(t):
    return t.detach().clone().requires_grad_(True)


def test_single_chunk(B=1, T=16, H=1, D=128, lower_bound=-5.0, seed=0,
                      large_gate=False, zero_h0=False, label=""):
    """Single chunk test — eliminates cross-chunk state propagation issues."""
    print(f"\n=== {label}: B={B} T={T} H={H} D={D} lb={lower_bound} "
          f"large_gate={large_gate} zero_h0={zero_h0} ===")
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, D, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, D, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    if large_gate:
        g = (torch.ones(B, T, H, D, device="cuda") * 5.0).to(torch.bfloat16)
    else:
        g = (torch.randn(B, T, H, D, device="cuda") * 0.5).to(torch.bfloat16)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = (torch.rand(H, D, dtype=torch.float32, device="cuda") - 0.5) * 0.5

    if zero_h0:
        h0 = torch.zeros(B, H, D, D, dtype=torch.float32, device="cuda")
    else:
        h0 = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") * 0.01
    scale = 1.0 / math.sqrt(D)

    do = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    dht = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda")

    # FLA reference
    from fla.ops.kda import chunk_kda
    fl_in = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log)]
    dt_flat = dt_bias.detach().reshape(-1).clone().requires_grad_(True)
    h0_fl = _clone_leaf(h0)
    o_fl, ht_fl = chunk_kda(
        q=fl_in[0], k=fl_in[1], v=fl_in[2], g=fl_in[3], beta=fl_in[4], scale=scale,
        initial_state=h0_fl, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        A_log=fl_in[5], dt_bias=dt_flat, lower_bound=lower_bound,
        transpose_state_layout=True, cu_seqlens=None,
    )
    ((o_fl.float() * do.float()).sum() + (ht_fl.float() * dht).sum()).backward()
    fl_grads = [x.grad for x in fl_in] + [dt_flat.grad.reshape(dt_bias.shape), h0_fl.grad]

    # FlashKDA CUDA
    fk_in = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log, dt_bias, h0)]
    o_fk, ht_fk = flash_kda.flash_kda_func(
        fk_in[0], fk_in[1], fk_in[2], fk_in[3], fk_in[4], scale,
        fk_in[5], fk_in[6], lower_bound,
        initial_state=fk_in[7], output_final_state=True, cu_seqlens=None)
    ((o_fk.float() * do.float()).sum() + (ht_fk.float() * dht).sum()).backward()
    fk_grads = [x.grad for x in fk_in]

    print(f"  fwd o : {err_ratio(o_fk, o_fl):.3e}")
    print(f"  fwd ht: {err_ratio(ht_fk, ht_fl):.3e}")
    names = ["dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias", "dh0"]
    for nm, gfk, gfl in zip(names, fk_grads, fl_grads):
        r = err_ratio(gfk, gfl)
        flag = "OK" if r < 2e-2 else "**FAIL**"
        print(f"  {nm:<9}: {r:.3e}  {flag}")


def test_two_chunks(B=1, T=32, H=1, D=128, lower_bound=-5.0, seed=0, label=""):
    """Two chunks — tests cross-chunk dS propagation and dgT."""
    print(f"\n=== {label}: B={B} T={T} H={H} D={D} lb={lower_bound} ===")
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, D, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, D, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    g = (torch.randn(B, T, H, D, device="cuda") * 0.5).to(torch.bfloat16)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = (torch.rand(H, D, dtype=torch.float32, device="cuda") - 0.5) * 0.5
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32, device="cuda")
    scale = 1.0 / math.sqrt(D)

    do = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    dht = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda")

    from fla.ops.kda import chunk_kda
    fl_in = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log)]
    dt_flat = dt_bias.detach().reshape(-1).clone().requires_grad_(True)
    h0_fl = _clone_leaf(h0)
    o_fl, ht_fl = chunk_kda(
        q=fl_in[0], k=fl_in[1], v=fl_in[2], g=fl_in[3], beta=fl_in[4], scale=scale,
        initial_state=h0_fl, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        A_log=fl_in[5], dt_bias=dt_flat, lower_bound=lower_bound,
        transpose_state_layout=True, cu_seqlens=None,
    )
    ((o_fl.float() * do.float()).sum() + (ht_fl.float() * dht).sum()).backward()
    fl_grads = [x.grad for x in fl_in] + [dt_flat.grad.reshape(dt_bias.shape), h0_fl.grad]

    fk_in = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log, dt_bias, h0)]
    o_fk, ht_fk = flash_kda.flash_kda_func(
        fk_in[0], fk_in[1], fk_in[2], fk_in[3], fk_in[4], scale,
        fk_in[5], fk_in[6], lower_bound,
        initial_state=fk_in[7], output_final_state=True, cu_seqlens=None)
    ((o_fk.float() * do.float()).sum() + (ht_fk.float() * dht).sum()).backward()
    fk_grads = [x.grad for x in fk_in]

    print(f"  fwd o : {err_ratio(o_fk, o_fl):.3e}")
    print(f"  fwd ht: {err_ratio(ht_fk, ht_fl):.3e}")
    names = ["dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias", "dh0"]
    for nm, gfk, gfl in zip(names, fk_grads, fl_grads):
        r = err_ratio(gfk, gfl)
        flag = "OK" if r < 2e-2 else "**FAIL**"
        print(f"  {nm:<9}: {r:.3e}  {flag}")


if __name__ == "__main__":
    # Test 1: Single chunk, small gates, zero h0 — isolates within-chunk math
    test_single_chunk(B=1, T=16, H=1, D=128, lower_bound=-5.0,
                      zero_h0=True, label="1chunk_zeroh0_smallg")

    # Test 2: Single chunk, small gates, nonzero h0
    test_single_chunk(B=1, T=16, H=1, D=128, lower_bound=-5.0,
                      zero_h0=False, label="1chunk_h0_smallg")

    # Test 3: Single chunk, large gates (exp(gc) → large, exp(-gc) → 0)
    test_single_chunk(B=1, T=16, H=1, D=128, lower_bound=-5.0,
                      large_gate=True, zero_h0=True, label="1chunk_zeroh0_largeg")

    # Test 4: Two chunks, small gates, zero h0
    test_two_chunks(B=1, T=32, H=1, D=128, lower_bound=-5.0,
                    label="2chunks_zeroh0")

    # Test 5: Single chunk, lb=-1 (small decay)
    test_single_chunk(B=1, T=16, H=1, D=128, lower_bound=-1.0,
                      zero_h0=True, label="1chunk_zeroh0_lb1")

    print("\n--- Done ---")
