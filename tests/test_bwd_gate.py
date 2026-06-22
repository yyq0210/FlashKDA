"""Debug dg by computing torch reference gate backward step by step."""
import math
import torch
import torch.nn.functional as F
import flash_kda
from flash_kda.autograd import _l2norm, _gate
from flash_kda_C import get_workspace_size

CHUNK = 16
_L2_EPS = 1e-6

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
    g_raw = (torch.randn(B, T, H, D, device="cuda") * 0.5).to(torch.bfloat16)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = (torch.rand(H, D, dtype=torch.float32, device="cuda") - 0.5) * 0.5
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32, device="cuda")

    do = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")

    # ---- Torch reference: manually trace the backward through the gate ----
    # Forward:
    # g_nat = lb * sigmoid(a * (g_raw + dt_bias))  -- in natural log
    # The CUDA kernel uses base-2: g_nat_log2 = (lb/ln2) * sigmoid(a * (g_raw+dt))
    # gc_log2 = cumsum(g_nat_log2)
    # kd = kn * exp2(gc_log2), qd = qn * exp2(gc_log2) * scale
    # ki = kn * exp2(-gc_log2), kr = kn * exp2(gT_log2 - gc_log2)

    # The torch reference uses natural base:
    # g_nat = lb * sigmoid(a * (g_raw + dt_bias))
    # gc = cumsum(g_nat)  -- in natural log
    # kd = kn * exp(gc), etc.

    # So g_nat_log2 = g_nat / ln2, and gc_log2 = gc / ln2
    # And exp2(gc_log2) = exp(gc), so the math is consistent.

    # For backward:
    # dgc[c][k] = dkd*kd + dqd*qd - dki*ki - dkr*kr  (base doesn't matter since exp(gc) * gc' = exp(gc))
    # Actually, let me think again...
    # If using exp(gc) (natural base):
    #   d/d(gc) exp(gc) = exp(gc)
    #   dgc = dkd*kd + dqd*qd + (-dki)*(-ki) + (-dkr)*(-kr)
    #       = dkd*kd + dqd*qd - dki*ki - dkr*kr  (same formula regardless of base, because the
    #         kd/qd/ki/kr values are the same either way)

    # If using exp2(gc_log2) (base 2):
    #   d/d(gc_log2) exp2(gc_log2) = exp2(gc_log2) * ln2
    #   dgc_log2 = ln2 * (dkd*kd + dqd*qd - dki*ki - dkr*kr)
    # Then dg_nat_log2 = reverse_cumsum(dgc_log2) + dgT_log2
    # And dg_raw = dg_nat_log2 * d(g_nat_log2)/d(g_raw)
    # g_nat_log2 = (lb/ln2) * sig(z), z = a*(g_raw+dt)
    # d(g_nat_log2)/d(g_raw) = (lb/ln2) * sig'(z) * a = gate_scale * sig'(z) * a
    # So dg_raw = dg_nat_log2 * gate_scale * dsig * a
    #           = ln2*(sum...) * (lb/ln2) * dsig * a = lb * (sum...) * dsig * a
    # This matches the natural-base result:
    # dg_nat = (sum...), dg_raw = dg_nat * lb * dsig * a

    # Conclusion: the ln2 factors should cancel. The formula should work as:
    # dgc = dkd*kd + dqd*qd - dki*ki - dkr*kr  (no ln2)
    # dg_nat = revcumsum(dgc) + dgT
    # dg_raw = dg_nat * gate_scale * dsig * ln2 * a = dg_nat * lb * dsig * a
    # OR equivalently:
    # dgc = ln2*(dkd*kd + ...), dg_nat = revcumsum(dgc) + dgT, dg_raw = dg_nat * gate_scale * dsig * a

    # The current code uses: dgc (no ln2) + gate_scale*dsig*ln2*a
    # This gives: (revcumsum(dgc) + dgT) * gate_scale * dsig * ln2 * a
    # = (revcumsum(dgc) + dgT) * (lb/ln2) * dsig * ln2 * a
    # = (revcumsum(dgc) + dgT) * lb * dsig * a
    # This is CORRECT only if dgT also doesn't have the ln2 factor.
    # If dgT has the ln2 factor, the dgT part would be doubled.

    # Let me compute the torch reference backward for dg manually:
    g_raw_leaf = _clone_leaf(g_raw)
    qf = q.float()
    kf = k.float()
    vf = v.float()
    qn = _l2norm(qf)
    kn = _l2norm(kf)
    g_nat = _gate(g_raw_leaf.float(), A_log, dt_bias, lb)  # [B,T,H,D]
    beta_sig = beta.float().sigmoid()

    # Single chunk
    gc = g_nat.cumsum(dim=1)  # [B,T,H,D]
    gT = gc[:, -1:, :, :]    # [B,1,H,D]

    kd = kn * gc.exp()
    qd = qn * gc.exp() * scale
    ki = kn * (-gc).exp()
    kr = kn * (gT - gc).exp()

    S = h0.float().transpose(-1, -2)  # [B,H,K,V] = [1,1,128,128]

    L = torch.tril(kd.permute(0,2,1,3) @ ki.permute(0,2,1,3).transpose(-1,-2), -1) * beta_sig.unsqueeze(-1).permute(0,2,1,3)
    Mqk = torch.tril(qd.permute(0,2,1,3) @ ki.permute(0,2,1,3).transpose(-1,-2))

    # Reshape for matmul: [B,H,T,D]
    kd_bht = kd.permute(0,2,1,3)
    qd_bht = qd.permute(0,2,1,3)
    ki_bht = ki.permute(0,2,1,3)
    kr_bht = kr.permute(0,2,1,3)
    v_bht = vf.permute(0,2,1,3)
    beta_bh = beta_sig.permute(0,2,1)

    vcorr = (v_bht - kd_bht @ S) * beta_bh.unsqueeze(-1)
    eye = torch.eye(T, dtype=torch.float32, device="cuda")
    IpL = eye + L
    U = torch.linalg.solve_triangular(IpL, vcorr, upper=False, unitriangular=True)
    out = qd_bht @ S + Mqk @ U

    # Backward
    do_bht = do.float().permute(0,2,1,3)
    loss = (out * do_bht).sum()
    loss.backward()

    dg_ref = g_raw_leaf.grad
    print(f"Torch ref dg: mean={dg_ref.float().abs().mean():.3e}")

    # ---- CUDA backward ----
    fk_in = [_clone_leaf(x) for x in (q, k, v, g_raw, beta, A_log, dt_bias, h0)]
    o_fk, ht_fk = flash_kda.flash_kda_func(
        fk_in[0], fk_in[1], fk_in[2], fk_in[3], fk_in[4], scale,
        fk_in[5], fk_in[6], lb,
        initial_state=fk_in[7], output_final_state=True, cu_seqlens=None)
    (o_fk.float() * do.float()).sum().backward()
    dg_cuda = fk_in[3].grad

    print(f"CUDA     dg: mean={dg_cuda.float().abs().mean():.3e}")
    print(f"dg err: {err_ratio(dg_cuda, dg_ref):.3e}")

    # Per-element comparison
    print("\n--- dg sample [0, :8, 0, 0] ---")
    for t in range(min(8, T)):
        c = dg_cuda[0, t, 0, 0].float().item()
        r = dg_ref[0, t, 0, 0].float().item()
        ratio = c / (r + 1e-12)
        print(f"  t={t}: cuda={c:.8f}  ref={r:.8f}  ratio={ratio:.4f}")

    # Now let me also check whether the issue is dgT vs dgc
    # For a single chunk, dgT = 0 (no next chunk), so dg_nat = revcumsum(dgc)
    # If T=16 = 1 chunk, there IS dht=0 so dgT should be 0.
    # Wait, but I passed dht=0 above for the do-only test.
    # However, the dht contribution goes through S_new = S * exp(gT) + kr^T @ U
    # With dht=0, the only dgT contribution comes from the S*exp(gT) path in K2_bwd.
    # For single chunk with zero h0, S_in=0, so there's no state to decay.
    # In that case dgT should indeed be 0.

    # Let me also dump the intermediate workspace values read by K1_bwd
    print("\n--- Checking CUDA workspace (bwd workspace might need inspection) ---")

    # Let me check that the CUDA dg values make sense per-column
    # If dg is wrong uniformly by some factor, that would indicate a scaling bug
    print("\n--- dg error per time step ---")
    for t in range(T):
        r = err_ratio(dg_cuda[0, t, 0, :], dg_ref[0, t, 0, :])
        print(f"  t={t}: err={r:.3e}")


if __name__ == "__main__":
    main()
