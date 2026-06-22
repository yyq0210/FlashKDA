"""Debug dh0 (d_initial_state) for single chunk."""
import math
import torch
import torch.nn.functional as F
import flash_kda
from flash_kda.autograd import _kda_chunk_torch_fwd, _l2norm, _gate

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

    # Use nonzero h0 so dh0 has a signal
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") * 0.01

    # Only use do (no dht) to simplify
    do = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")

    # Torch reference
    h0_ref = _clone_leaf(h0)
    out_ref, _ = _kda_chunk_torch_fwd(
        q, k, v, g, beta, scale, A_log, dt_bias, lb, h0_ref, True, None)
    (out_ref.float() * do.float()).sum().backward()
    dh0_ref = h0_ref.grad

    # CUDA
    h0_cuda = _clone_leaf(h0)
    o_fk, ht_fk = flash_kda.flash_kda_func(
        q, k, v, g, beta, scale, A_log, dt_bias, lb,
        initial_state=h0_cuda, output_final_state=True, cu_seqlens=None)
    (o_fk.float() * do.float()).sum().backward()
    dh0_cuda = h0_cuda.grad

    print(f"fwd o: {err_ratio(o_fk, out_ref):.3e}")
    r = err_ratio(dh0_cuda, dh0_ref)
    print(f"dh0 err: {r:.3e}")
    print(f"|dh0_cuda|: {dh0_cuda.float().abs().mean():.3e}")
    print(f"|dh0_ref|:  {dh0_ref.float().abs().mean():.3e}")
    print(f"mag ratio:  {dh0_cuda.float().abs().mean() / dh0_ref.float().abs().mean():.3f}")

    # Sample elements
    print("\n--- dh0 sample [0,0,0,:5] ---")
    for i in range(5):
        c = dh0_cuda[0,0,0,i].item()
        r = dh0_ref[0,0,0,i].item()
        print(f"  i={i}: cuda={c:.8f} ref={r:.8f} ratio={c/(r+1e-12):.4f}")

    # Note: h0 layout for FlashKDA is [N,H,D,D] where the matrix is S^T.
    # The torch ref uses S = h0.transpose(-1,-2), so S[H,K,V].
    # d/d(h0[i][j]) = d/d(S[j][i])
    # Let's check if there's a transpose mismatch.

    # The dS computation in K2_bwd stores dS[K, D] where K=D=128
    # Then it stores to ds_out with layout [N*H, D, D]
    # The initial_state input is [N, H, D, D] and is transposed in the C++ layer?

    # Check cross-correlation: does dh0_cuda match dh0_ref transposed?
    dh0_ref_T = dh0_ref.transpose(-1, -2)
    r_T = err_ratio(dh0_cuda, dh0_ref_T)
    print(f"\ndh0 err (transposed ref): {r_T:.3e}")

    # Also check: does the CUDA output match a "wrong" layout?
    # Maybe the CUDA stores dS[k][d] but h0 expects [d][k]
    print("\n--- Check element correspondence ---")
    for i in range(3):
        for j in range(3):
            c = dh0_cuda[0,0,i,j].item()
            r = dh0_ref[0,0,i,j].item()
            rT = dh0_ref[0,0,j,i].item()
            print(f"  [{i},{j}]: cuda={c:.6f} ref={r:.6f} ref^T={rT:.6f}")


if __name__ == "__main__":
    main()
