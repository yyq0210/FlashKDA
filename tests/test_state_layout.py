"""Verify the layout of all_states by comparing against torch reference."""
import math
import torch
import torch.nn.functional as F
import flash_kda
from flash_kda.autograd import _l2norm, _gate

CHUNK = 16
_L2_EPS = 1e-6

def main():
    B, T, H, D = 1, 32, 1, 128  # 2 chunks
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

    # Use a simple non-zero h0 for testing
    h0_vals = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") * 0.01
    # h0 is [N, H, V, K] = [1, 1, 128, 128], represents S^T

    # Run forward
    out = torch.empty(B, T, H, D, dtype=torch.bfloat16, device=q.device)
    T_total = B * T
    N = B
    total_tiles = (T_total + CHUNK - 1) // CHUNK
    all_states = torch.empty(H * total_tiles, D, D, dtype=torch.bfloat16, device=q.device)

    workspace = flash_kda.fwd(
        q, k, v, g, beta, float(scale), out,
        A_log=A_log, dt_bias=dt_bias, lower_bound=float(lb),
        initial_state=h0_vals.to(torch.bfloat16), final_state=None, cu_seqlens=None,
        all_states=all_states,
    )

    # Torch reference: compute S after chunk 0
    qf, kf, vf = q.float(), k.float(), v.float()
    qn = _l2norm(qf)
    kn = _l2norm(kf)
    g_nat = _gate(g.float(), A_log, dt_bias, lb)

    # S0 = h0.transpose(-1,-2) = [H, K, V]
    S0 = h0_vals.float()[0].transpose(-1, -2)  # [H, K, V] = [1, 128, 128]

    # Chunk 0: t=0..15
    gc = g_nat[0, :CHUNK, :, :].cumsum(dim=0)  # [16, H, D]
    gT = gc[-1]  # [H, D]

    kd = kn[0, :CHUNK] * gc.exp()
    qd = qn[0, :CHUNK] * gc.exp() * scale
    ki = kn[0, :CHUNK] * (-gc).exp()
    kr = kn[0, :CHUNK] * (gT - gc).exp()
    beta_sig = beta[0, :CHUNK].float().sigmoid()

    L = torch.tril(kd.permute(1,0,2) @ ki.permute(1,0,2).transpose(-1,-2), -1) * beta_sig.unsqueeze(-1).permute(1,0,2)
    Mqk = torch.tril(qd.permute(1,0,2) @ ki.permute(1,0,2).transpose(-1,-2))

    vcorr = (vf[0,:CHUNK].permute(1,0,2) - kd.permute(1,0,2) @ S0) * beta_sig.unsqueeze(-1).permute(1,0,2)
    eye = torch.eye(CHUNK, dtype=torch.float32, device="cuda")
    IpL = eye + L
    U = torch.linalg.solve_triangular(IpL, vcorr, upper=False, unitriangular=True)

    S1 = S0 * gT.exp().unsqueeze(-1) + kr.permute(1,0,2).transpose(-1,-2) @ U
    # S1 is [H, K, V]

    print(f"S0 shape: {S0.shape}")
    print(f"S1 shape: {S1.shape}")

    # all_states[0] should be S0 (initial state for chunk 0)
    # all_states[1] should be S1 (state after chunk 0, input to chunk 1)

    # Check: does all_states store S[K,V] or S^T[V,K]?
    s0_stored = all_states[0].float()  # [D, D]
    s1_stored = all_states[1].float()  # [D, D]

    # Compare with S0[0] = [K, V]
    def err(x, y):
        return ((x-y).square().mean().sqrt() / (y.square().mean().sqrt()+1e-8)).item()

    print(f"\nall_states[0] vs S0[K,V]:   {err(s0_stored, S0[0]):.4e}")
    print(f"all_states[0] vs S0^T[V,K]: {err(s0_stored, S0[0].T):.4e}")
    print(f"all_states[1] vs S1[K,V]:   {err(s1_stored, S1[0]):.4e}")
    print(f"all_states[1] vs S1^T[V,K]: {err(s1_stored, S1[0].T):.4e}")

    # Check element-wise
    print(f"\n--- all_states[0] sample ---")
    for i in range(3):
        for j in range(3):
            print(f"  [{i},{j}]: stored={s0_stored[i,j]:.6f}  S0={S0[0,i,j]:.6f}  S0^T={S0[0,j,i]:.6f}")


if __name__ == "__main__":
    main()
