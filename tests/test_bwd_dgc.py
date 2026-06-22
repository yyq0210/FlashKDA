"""Debug: compute dgc from torch reference and compare with what K1_bwd computes.
For single chunk, dg_nat[c] = sum_{r=c}^{C-1} dgc[r] + dgT.
With zero h0 and no dfinal_state, dgT=0, so dg_nat = reverse_cumsum(dgc).
"""
import math
import torch
import torch.nn.functional as F
import flash_kda
from flash_kda.autograd import _l2norm, _gate

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

    # Torch reference: compute dgc
    qf, kf, vf = q.float(), k.float(), v.float()
    qn = _l2norm(qf)
    kn = _l2norm(kf)

    g_raw_leaf = g.float().clone().requires_grad_(True)
    g_nat = _gate(g_raw_leaf, A_log, dt_bias, lb)

    gc = g_nat.cumsum(dim=1)  # [B, T, H, D]
    gT = gc[:, -1:, :, :]

    kd = kn * gc.exp()
    qd = qn * gc.exp() * scale
    ki = kn * (-gc).exp()
    kr = kn * (gT - gc).exp()

    beta_sig = beta.float().sigmoid()

    # Reshape for matmul: [B,H,T,D]
    kd_bht = kd.permute(0,2,1,3)
    qd_bht = qd.permute(0,2,1,3)
    ki_bht = ki.permute(0,2,1,3)
    kr_bht = kr.permute(0,2,1,3)
    v_bht = vf.permute(0,2,1,3)
    beta_bh = beta_sig.permute(0,2,1)

    S = h0.float().transpose(-1, -2)  # [B,H,K,V]

    vcorr = (v_bht - kd_bht @ S) * beta_bh.unsqueeze(-1)
    eye = torch.eye(T, dtype=torch.float32, device="cuda")
    Mqk = torch.tril(qd_bht @ ki_bht.transpose(-1,-2))

    # Actually for L, need to be more careful about beta dimension
    beta_bh_expand = beta_sig.permute(0,2,1).unsqueeze(-1)  # [B, H, T, 1]
    L_correct = torch.tril(kd_bht @ ki_bht.transpose(-1,-2), -1) * beta_bh_expand
    IpL = eye + L_correct
    U = torch.linalg.solve_triangular(IpL, vcorr, upper=False, unitriangular=True)
    out = qd_bht @ S + Mqk @ U

    do_bht = do.float().permute(0,2,1,3)
    loss = (out * do_bht).sum()

    # Compute gradient w.r.t. gc using autograd
    gc_leaf = gc.detach().clone().requires_grad_(True)
    # Redo the forward from gc
    kd2 = kn * gc_leaf.exp()
    qd2 = qn * gc_leaf.exp() * scale
    ki2 = kn * (-gc_leaf).exp()
    gT2 = gc_leaf[:, -1:, :, :]
    kr2 = kn * (gT2 - gc_leaf).exp()

    kd2_bht = kd2.permute(0,2,1,3)
    qd2_bht = qd2.permute(0,2,1,3)
    ki2_bht = ki2.permute(0,2,1,3)
    kr2_bht = kr2.permute(0,2,1,3)

    vcorr2 = (v_bht - kd2_bht @ S) * beta_bh.unsqueeze(-1)
    L2 = torch.tril(kd2_bht @ ki2_bht.transpose(-1,-2), -1) * beta_bh_expand
    Mqk2 = torch.tril(qd2_bht @ ki2_bht.transpose(-1,-2))
    IpL2 = eye + L2
    U2 = torch.linalg.solve_triangular(IpL2, vcorr2, upper=False, unitriangular=True)
    out2 = qd2_bht @ S + Mqk2 @ U2
    loss2 = (out2 * do_bht).sum()
    loss2.backward()
    dgc_ref = gc_leaf.grad  # [B, T, H, D]

    print(f"dgc_ref shape: {dgc_ref.shape}")
    print(f"|dgc_ref|: {dgc_ref.abs().mean():.3e}")

    # Compute what the CUDA kernel would compute for dgc
    # dgc[c][k] = dkd[c][k]*kd[c][k] + dqd[c][k]*qd[c][k] - dki[c][k]*ki[c][k] - dkr[c][k]*kr[c][k]
    # This formula comes from d(exp(gc))/d(gc) = exp(gc) for the natural exp version

    # Let me check: does the autograd dgc match the formula?
    # The formula dgc = dkd*kd + dqd*qd - dki*ki - dkr*kr
    # is the derivative of {kd, qd, ki, kr} w.r.t. gc

    # Let me compute dkd, dqd, dki, dkr from autograd too
    kd_leaf = kd2_bht.detach().clone().requires_grad_(True)
    qd_leaf = qd2_bht.detach().clone().requires_grad_(True)
    ki_leaf = ki2_bht.detach().clone().requires_grad_(True)
    kr_leaf = kr2_bht.detach().clone().requires_grad_(True)

    vcorr3 = (v_bht - kd_leaf @ S) * beta_bh.unsqueeze(-1)
    L3 = torch.tril(kd_leaf @ ki_leaf.transpose(-1,-2), -1) * beta_bh_expand
    Mqk3 = torch.tril(qd_leaf @ ki_leaf.transpose(-1,-2))
    IpL3 = eye + L3
    U3 = torch.linalg.solve_triangular(IpL3, vcorr3, upper=False, unitriangular=True)
    out3 = qd_leaf @ S + Mqk3 @ U3
    loss3 = (out3 * do_bht).sum()
    loss3.backward()

    dkd_ref = kd_leaf.grad  # [B, H, T, D]
    dqd_ref = qd_leaf.grad
    dki_ref = ki_leaf.grad
    dkr_ref = kr_leaf.grad

    # Compute dgc from the formula
    dgc_formula = dkd_ref * kd2_bht.detach() + dqd_ref * qd2_bht.detach() \
                  - dki_ref * ki2_bht.detach() - dkr_ref * kr2_bht.detach()

    # Compare: dgc_formula should match dgc_ref (both in [B,H,T,D] vs [B,T,H,D])
    dgc_formula_btHD = dgc_formula.permute(0,2,1,3)  # [B, T, H, D]
    print(f"dgc_formula vs dgc_ref: {err_ratio(dgc_formula_btHD, dgc_ref):.3e}")

    # Good. Now let me check what the CUDA workspace gives for kd, dkd etc.
    # The CUDA kernel computes dgc using bf16 values from the workspace.
    # Let me compute dgc using the bf16 workspace values and compare.

    # Run forward to get workspace
    out_buf = torch.empty_like(q)
    T_total = B * T
    N = B
    total_tiles = (T_total + CHUNK - 1) // CHUNK
    all_states = torch.empty(H * total_tiles, D, D, dtype=torch.bfloat16, device=q.device)

    workspace = flash_kda.fwd(
        q, k, v, g, beta, float(scale), out_buf,
        A_log=A_log, dt_bias=dt_bias, lower_bound=float(lb),
        initial_state=None, final_state=None, cu_seqlens=None,
        all_states=all_states,
    )

    # Parse workspace to get kd, qd, ki, kr
    from flash_kda_C import get_workspace_size
    ws_size = get_workspace_size(T_total, H, N)
    print(f"workspace size: {ws_size}")

    # Layout: kd[n_ht * CD], qd[n_ht * CD], kr[n_ht * CD], ki[n_ht * CD],
    #         gT[n_ht * D], INV[n_ht * CC], Mqk[n_ht * CC]
    n_ht = H * total_tiles
    CD = CHUNK * D
    CC = CHUNK * CHUNK
    bf16_ws = workspace.view(-1)[:n_ht * (4*CD) * 2].view(torch.bfloat16)
    kd_ws = bf16_ws[:n_ht * CD].view(n_ht, CHUNK, D)
    qd_ws = bf16_ws[n_ht*CD:n_ht*2*CD].view(n_ht, CHUNK, D)
    kr_ws = bf16_ws[n_ht*2*CD:n_ht*3*CD].view(n_ht, CHUNK, D)
    ki_ws = bf16_ws[n_ht*3*CD:n_ht*4*CD].view(n_ht, CHUNK, D)

    # For tile 0: ws_idx=0 (H=1, total_tiles=1)
    ws_kd = kd_ws[0].float()  # [CHUNK, D]
    ws_qd = qd_ws[0].float()
    ws_ki = ki_ws[0].float()
    ws_kr = kr_ws[0].float()

    # Compare with torch reference
    kd_ref = kd2_bht[0, 0].detach().float()  # [T, D]
    qd_ref_val = qd2_bht[0, 0].detach().float()
    ki_ref_val = ki2_bht[0, 0].detach().float()
    kr_ref_val = kr2_bht[0, 0].detach().float()

    print(f"\nWorkspace vs ref:")
    print(f"  kd: {err_ratio(ws_kd, kd_ref):.3e}")
    print(f"  qd: {err_ratio(ws_qd, qd_ref_val):.3e}")
    print(f"  ki: {err_ratio(ws_ki, ki_ref_val):.3e}")
    print(f"  kr: {err_ratio(ws_kr, kr_ref_val):.3e}")

    # Now the key question: is the CUDA dgc (computed from bf16 products) close enough?
    # Simulate the K1_bwd dgc computation using workspace bf16 values:
    # dgc_cuda_sim[c][k] = dkd[c][k]*kd[c][k] + dqd[c][k]*qd[c][k]
    #                     - dki[c][k]*ki[c][k] - dkr[c][k]*kr[c][k]
    # where dkd etc. come from K2_bwd (also bf16 workspace values)

    # I don't have easy access to the K2_bwd workspace values, but I can estimate
    # the error from the bf16 workspace values of kd/qd/ki/kr and the fp32 dkd/dqd/dki/dkr:

    # Use fp32 dkd but bf16 kd to simulate the mixed precision
    dgc_mixed = (dkd_ref[0,0] * ws_kd + dqd_ref[0,0] * ws_qd
                 - dki_ref[0,0] * ws_ki - dkr_ref[0,0] * ws_kr)
    dgc_ref_val = dgc_ref[0, :, 0, :]  # [T, D]

    print(f"\ndgc (mixed fp32_grad * bf16_ws) vs ref: {err_ratio(dgc_mixed, dgc_ref_val):.3e}")

    # What if we use all fp32?
    dgc_fp32 = (dkd_ref[0,0] * kd_ref + dqd_ref[0,0] * qd_ref_val
                - dki_ref[0,0] * ki_ref_val - dkr_ref[0,0] * kr_ref_val)
    print(f"dgc (all fp32)                vs ref: {err_ratio(dgc_fp32, dgc_ref_val):.3e}")

    # Also check the reverse cumsum
    # dg_nat_ref = reverse_cumsum(dgc_ref)
    dg_nat_ref = dgc_ref_val.flip(0).cumsum(0).flip(0)

    dg_nat_mixed = dgc_mixed.flip(0).cumsum(0).flip(0)
    print(f"\ndg_nat (mixed) vs ref: {err_ratio(dg_nat_mixed, dg_nat_ref):.3e}")

    # Per-timestep error
    print("\n--- dg_nat error per timestep (mixed precision dgc) ---")
    for t in range(T):
        r = err_ratio(dg_nat_mixed[t], dg_nat_ref[t])
        print(f"  t={t}: err={r:.3e}")


if __name__ == "__main__":
    main()
