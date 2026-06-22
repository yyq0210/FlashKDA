"""Minimal test: check dv for a single chunk with zero initial state.
Compares CUDA backward vs torch autograd through the torch reference forward.
"""
import math
import torch
import torch.nn.functional as F
import flash_kda
from flash_kda.autograd import _kda_chunk_torch_fwd, _l2norm, _gate

CHUNK = 16

def err_ratio(x, y):
    x, y = x.detach().float(), y.detach().float()
    return ((x - y).square().mean().sqrt() / (y.square().mean().sqrt() + 1e-8)).item()


def _clone_leaf(t):
    return t.detach().clone().requires_grad_(True)


def main():
    B, T, H, D = 1, 16, 1, 128
    lb = -5.0
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(42)

    q = F.normalize(torch.randn(B, T, H, D, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, D, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    g = (torch.randn(B, T, H, D, device="cuda") * 0.5).to(torch.bfloat16)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = (torch.rand(H, D, dtype=torch.float32, device="cuda") - 0.5) * 0.5
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32, device="cuda")

    do = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    dht = torch.zeros(B, H, D, D, dtype=torch.float32, device="cuda")  # zero to focus on do

    # --- Torch reference backward ---
    fk_in_ref = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log, dt_bias, h0)]
    o_ref, ht_ref = _kda_chunk_torch_fwd(
        fk_in_ref[0], fk_in_ref[1], fk_in_ref[2], fk_in_ref[3], fk_in_ref[4],
        scale, fk_in_ref[5], fk_in_ref[6], lb,
        fk_in_ref[7], True, None)
    (o_ref.float() * do.float()).sum().backward()
    ref_grads = {nm: x.grad for nm, x in zip(
        ["q", "k", "v", "g", "beta", "A_log", "dt_bias", "h0"], fk_in_ref)}

    # --- CUDA backward ---
    fk_in = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log, dt_bias, h0)]
    o_fk, ht_fk = flash_kda.flash_kda_func(
        fk_in[0], fk_in[1], fk_in[2], fk_in[3], fk_in[4], scale,
        fk_in[5], fk_in[6], lb,
        initial_state=fk_in[7], output_final_state=True, cu_seqlens=None)
    (o_fk.float() * do.float()).sum().backward()
    cuda_grads = {nm: x.grad for nm, x in zip(
        ["q", "k", "v", "g", "beta", "A_log", "dt_bias", "h0"], fk_in)}

    print(f"fwd o: {err_ratio(o_fk, o_ref):.3e}")

    for nm in ["q", "k", "v", "g", "beta", "A_log", "dt_bias", "h0"]:
        r = err_ratio(cuda_grads[nm], ref_grads[nm])
        flag = "OK" if r < 2e-2 else "**FAIL**"
        # Also check magnitude
        cuda_mag = cuda_grads[nm].float().abs().mean().item()
        ref_mag = ref_grads[nm].float().abs().mean().item()
        ratio = cuda_mag / (ref_mag + 1e-12)
        print(f"  d{nm:<8}: err={r:.3e} {flag}  |cuda|={cuda_mag:.3e}  |ref|={ref_mag:.3e}  mag_ratio={ratio:.3f}")

    # Check dv per-element for first few
    print("\n--- dv sample (first 5 elements of v[0,0,0,:]) ---")
    for i in range(5):
        c = cuda_grads["v"][0, 0, 0, i].float().item()
        r = ref_grads["v"][0, 0, 0, i].float().item()
        print(f"  idx={i}: cuda={c:.6f}  ref={r:.6f}  ratio={c/(r+1e-12):.3f}")

    print("\n--- dk sample (first 5 elements of k[0,0,0,:]) ---")
    for i in range(5):
        c = cuda_grads["k"][0, 0, 0, i].float().item()
        r = ref_grads["k"][0, 0, 0, i].float().item()
        print(f"  idx={i}: cuda={c:.6f}  ref={r:.6f}  ratio={c/(r+1e-12):.3f}")


if __name__ == "__main__":
    main()
