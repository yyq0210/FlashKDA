"""Debug: check precision of dgc computation.
For single chunk with h0=0, dg_nat = reverse_cumsum(dgc).
We compare CUDA dg vs torch reference dg.
"""
import math
import torch
import torch.nn.functional as F
import flash_kda
from flash_kda.autograd import _kda_chunk_torch_fwd, _l2norm, _gate

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
    g = (torch.randn(B, T, H, D, device="cuda") * 0.5).to(torch.bfloat16)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = (torch.rand(H, D, dtype=torch.float32, device="cuda") - 0.5) * 0.5
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32, device="cuda")

    do = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")

    # Torch reference backward
    fk_ref = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log, dt_bias, h0)]
    o_ref, ht_ref = _kda_chunk_torch_fwd(
        fk_ref[0], fk_ref[1], fk_ref[2], fk_ref[3], fk_ref[4],
        scale, fk_ref[5], fk_ref[6], lb, fk_ref[7], True, None)
    (o_ref.float() * do.float()).sum().backward()
    dg_ref = fk_ref[3].grad  # [B, T, H, D]

    # CUDA backward
    fk_cuda = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log, dt_bias, h0)]
    o_cuda, ht_cuda = flash_kda.flash_kda_func(
        fk_cuda[0], fk_cuda[1], fk_cuda[2], fk_cuda[3], fk_cuda[4], scale,
        fk_cuda[5], fk_cuda[6], lb,
        initial_state=fk_cuda[7], output_final_state=True, cu_seqlens=None)
    (o_cuda.float() * do.float()).sum().backward()
    dg_cuda = fk_cuda[3].grad

    print(f"fwd o: {err_ratio(o_cuda, o_ref):.3e}")
    print(f"dg overall: {err_ratio(dg_cuda, dg_ref):.3e}")

    # Per-timestep
    print("\n--- Per-timestep dg error (col 0) ---")
    for t in range(T):
        r = err_ratio(dg_cuda[0, t, 0, :], dg_ref[0, t, 0, :])
        c_val = dg_cuda[0, t, 0, 0].float().item()
        r_val = dg_ref[0, t, 0, 0].float().item()
        print(f"  t={t}: err={r:.3e}  cuda={c_val:.6e}  ref={r_val:.6e}")

    # Now let me understand: the gate backward in K1_bwd is:
    # dg_raw[t][k] = dg_nat[t][k] * gate_scale * sigmoid'(z) * ln2 * a
    # where dg_nat[t] = sum_{r=t}^{15} dgc[r] + dgT (dgT=0 for single chunk zero h0)
    # and dgc[r][k] = dkd[r]*kd[r] + dqd[r]*qd[r] - dki[r]*ki[r] - dkr[r]*kr[r]

    # The K1_bwd reads kd, qd, ki, kr from the forward workspace (bf16),
    # and dkd, dqd, dki, dkr from the backward workspace (also bf16).
    # The product of two bf16 values and summation in fp32 should have
    # ~0.5% per-element error. But the reverse cumsum accumulates 16 terms,
    # so the error at t=0 could be ~sqrt(16) * 0.5% = ~2%.

    # The actual error is ~8%, which suggests something more than just bf16 precision.
    # Let me check if the forward workspace and backward workspace values are correct.

    # Actually, let me directly check the gate backward formula.
    # The CUDA kernel uses gate_scale = lb / ln2, with:
    # dz = dg_nat * gate_scale * dsig * kLn2
    # dg_raw = dz * a
    # => dg_raw = dg_nat * (lb/ln2) * dsig * ln2 * a = dg_nat * lb * dsig * a

    # Torch reference:
    # g_nat = lb * sigmoid(a * (g_raw + dt_bias))
    # d(g_nat)/d(g_raw) = lb * sigmoid'(z) * a
    # dg_raw = dg_nat * lb * dsig * a

    # These match! So the formula is correct.

    # The error must be purely from bf16 quantization in the workspace values.
    # Let me verify by checking how sensitive dg_nat is to small perturbations.

    # Compute dg_nat from torch reference
    # dg_nat is the derivative of the loss w.r.t. g_nat (pre-cumsum gate values)
    g_nat_leaf = _gate(g.float(), A_log, dt_bias, lb).detach().clone().requires_grad_(True)
    qf, kf, vf = q.float(), k.float(), v.float()
    qn = _l2norm(qf)
    kn = _l2norm(kf)

    gc = g_nat_leaf.cumsum(dim=1)
    gT = gc[:, -1:, :, :]
    kd = kn * gc.exp()
    qd = qn * gc.exp() * scale
    ki = kn * (-gc).exp()
    kr = kn * (gT - gc).exp()
    beta_sig = beta.float().sigmoid()

    kd_bht = kd.permute(0,2,1,3)
    qd_bht = qd.permute(0,2,1,3)
    ki_bht = ki.permute(0,2,1,3)
    kr_bht = kr.permute(0,2,1,3)
    v_bht = vf.permute(0,2,1,3)
    beta_bh = beta_sig.permute(0,2,1)

    S = h0.float().transpose(-1, -2)
    vcorr = (v_bht - kd_bht @ S) * beta_bh.unsqueeze(-1)
    L = torch.tril(kd_bht @ ki_bht.transpose(-1,-2), -1) * beta_bh.unsqueeze(-1)
    Mqk = torch.tril(qd_bht @ ki_bht.transpose(-1,-2))
    eye = torch.eye(T, dtype=torch.float32, device="cuda")
    IpL = eye + L
    U = torch.linalg.solve_triangular(IpL, vcorr, upper=False, unitriangular=True)
    out = qd_bht @ S + Mqk @ U
    do_bht = do.float().permute(0,2,1,3)
    loss = (out * do_bht).sum()
    loss.backward()

    dg_nat_ref = g_nat_leaf.grad  # [B, T, H, D]
    print(f"\n|dg_nat_ref|: {dg_nat_ref.abs().mean():.3e}")
    print(f"|dg_ref|:     {dg_ref.abs().mean():.3e}")

    # The ratio of dg_nat to dg gives us the gate backward scaling factor
    # dg = dg_nat * lb * dsig * a
    # Let me check if the scaling is consistent
    a = torch.exp(A_log[0]).item()
    print(f"\na = exp(A_log) = {a:.4f}")
    print(f"lb = {lb}")

    # Compute dsig for each timestep
    g_raw = g.float()
    z = a * (g_raw[0,:,0,:] + dt_bias[0,:])  # [T, D]
    sig = torch.sigmoid(z)
    dsig = sig * (1 - sig)
    expected_factor = lb * dsig * a  # [T, D]

    print(f"\nExpected dg/dg_nat factor: mean={expected_factor.abs().mean():.3e}")

    actual_ratio = dg_ref[0,:,0,:].float() / (dg_nat_ref[0,:,0,:].float() + 1e-15)
    print(f"Actual dg/dg_nat ratio:   mean={actual_ratio.abs().mean():.3e}")


if __name__ == "__main__":
    main()
