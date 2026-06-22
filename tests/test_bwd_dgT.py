"""Debug: check dgT value from K2_bwd for single chunk with zero h0.
Should be zero since S_in=0 and dht=0.
"""
import math
import torch
import torch.nn.functional as F
import flash_kda
from flash_kda_C import get_workspace_size

CHUNK = 16

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

    # Run forward to get workspace
    out = torch.empty_like(q)
    T_total = B * T
    N = B
    total_tiles = (T_total + CHUNK - 1) // CHUNK  # 1 tile for T=16
    all_states = torch.empty(H * total_tiles, D, D, dtype=torch.bfloat16, device=q.device)

    workspace = flash_kda.fwd(
        q, k, v, g, beta, float(scale), out,
        A_log=A_log, dt_bias=dt_bias, lower_bound=float(lb),
        initial_state=h0.to(torch.bfloat16), final_state=None, cu_seqlens=None,
        all_states=all_states,
    )

    # Allocate bwd output tensors
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dg = torch.zeros_like(g)
    dbeta = torch.zeros(B, T, H, dtype=torch.bfloat16, device=q.device)
    dA_log = torch.zeros_like(A_log)
    ddt_bias = torch.zeros_like(dt_bias)

    # Run bwd with dfinal_state=None (0) and dinitial_state buffer
    dinitial_state = torch.zeros(N, H, D, D, dtype=torch.float32, device=q.device)

    flash_kda.bwd(
        q, k, v, g, beta,
        float(scale), workspace, all_states,
        do,
        A_log, dt_bias, float(lb),
        dq, dk, dv, dg, dbeta, dA_log, ddt_bias,
        dfinal_state=None,
        dinitial_state=dinitial_state,
        cu_seqlens=None,
    )

    # Now I need to inspect the bwd workspace to see dgT.
    # But the bwd workspace is allocated inside the C++ bwd function and not returned.
    # Let me instead check the dg pattern.

    # The pattern: dg_nat[c] = reverse_cumsum(dgc)[c] + dgT
    # For single chunk with S_in=0, dgT should come from:
    #   S_new = S_in * exp(gT) + kr^T @ U
    #   dgT = d(S_new)/d(gT) * dS_new
    # Since S_in=0 and dfinal_state=None (dS_new=0):
    #   dgT should be 0
    # But what does K2_bwd actually compute?

    # From line 680-709 in bwd_kernel2.cuh:
    # It computes dgT by recovering dS_old from dS_new, which should also be 0 here.
    # Then dgT = gT * sum_d dS_old * S_in
    # With S_in=0, dgT = 0 regardless of dS_old. Good.

    # So the dgT contribution should be zero. The error must be in dgc.
    # Let me check if the error pattern is consistent with cumulative dgc bias.

    # For that, let me compute torch reference dgc:
    from flash_kda.autograd import _l2norm, _gate, _clone_leaf
    g_leaf = g.float().clone().requires_grad_(True)
    qf, kf, vf = q.float(), k.float(), v.float()
    qn = _l2norm(qf)
    kn = _l2norm(kf)
    g_nat = _gate(g_leaf, A_log, dt_bias, lb)

    # Single chunk: gc[t] = cumsum(g_nat)[t]
    gc = g_nat.cumsum(dim=1)
    gT = gc[:, -1:, :, :]

    # Compute kd, qd, ki, kr
    kd = kn * gc.exp()
    qd = qn * gc.exp() * scale
    ki = kn * (-gc).exp()
    kr = kn * (gT - gc).exp()

    # Now compute dkd, dqd, dki, dkr from the forward output grad
    # out = qd @ S + Mqk @ U (with S=0 => out = Mqk @ U)

    # Actually, let me just check what dg[t=15] looks like
    # dg_nat[15] = dgc[15] + 0  (last timestep, reverse cumsum only has 1 term)
    # dgc[15] = dkd[15]*kd[15] + dqd[15]*qd[15] - dki[15]*ki[15] - dkr[15]*kr[15]

    print(f"dg shape: {dg.shape}")
    print(f"dg abs mean: {dg.float().abs().mean():.3e}")

    # Print dg pattern
    print("\n--- dg[0,:,0,0] ---")
    for t in range(T):
        print(f"  t={t}: {dg[0,t,0,0].float().item():.8f}")

    # Print dg[0,:,0,64] to check a different column
    print("\n--- dg[0,:,0,64] ---")
    for t in range(T):
        print(f"  t={t}: {dg[0,t,0,64].float().item():.8f}")

    # Check if all_states[0] is zero (it should be for h0=0)
    print(f"\nall_states[0] abs max: {all_states[0].float().abs().max():.3e}")


if __name__ == "__main__":
    main()
