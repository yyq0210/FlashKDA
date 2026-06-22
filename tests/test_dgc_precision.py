"""Diagnose dgc precision: compare magnitudes of individual terms."""
import math
import torch
import torch.nn.functional as F
from flash_kda.autograd import _l2norm, _gate

CHUNK = 16
_L2_EPS = 1e-6

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
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") * 0.01

    # Compute forward in fp32
    qf = q.float()
    kf = k.float()
    qn = _l2norm(qf)
    kn = _l2norm(kf)
    g_nat = _gate(g.float(), A_log, dt_bias, lb)  # [B,T,H,D] in natural log space

    # Convert to log2 space
    LN2 = 0.6931471805599453
    gate_scale = lb / LN2
    g_nat_log2 = g_nat / LN2  # in log2 units

    # Compute gc (cumsum in log2 space)
    gc_log2 = g_nat_log2[0, :, 0, :].cumsum(0)  # [T, D]
    gT_log2 = gc_log2[-1, :]  # [D]

    # Compute kd, qd, ki, kr in fp32
    e_gc = (2.0 ** gc_log2)
    e_neg_gc = (2.0 ** (-gc_log2))
    e_gt_gc = (2.0 ** (gT_log2[None, :] - gc_log2))

    kd = kn[0, :, 0, :] * e_gc
    qd = qn[0, :, 0, :] * e_gc * scale
    ki = kn[0, :, 0, :] * e_neg_gc
    kr = kn[0, :, 0, :] * e_gt_gc

    # Also compute bf16 versions
    kd_bf16 = kd.to(torch.bfloat16).float()
    qd_bf16 = qd.to(torch.bfloat16).float()
    ki_bf16 = ki.to(torch.bfloat16).float()
    kr_bf16 = kr.to(torch.bfloat16).float()

    # Simulate random d-values (like they would come from backward)
    dkd = torch.randn_like(kd) * 0.01
    dqd = torch.randn_like(qd) * 0.01
    dki = torch.randn_like(ki) * 0.01
    dkr = torch.randn_like(kr) * 0.01

    # Compute dgc using fp32 values (ground truth)
    dgc_fp32 = dqd * qd + dkd * kd - dki * ki - dkr * kr

    # Compute dgc using bf16 forward values (like K2_bwd)
    dgc_bf16fwd = dqd * qd_bf16 + dkd * kd_bf16 - dki * ki_bf16 - dkr * kr_bf16

    # Compute dgc using FLA-style decomposition
    # dgc = qn * (dqd * exp2(gc) * scale) + kn * (dkd*exp2(gc) - dki*exp2(-gc) - dkr*exp2(gT-gc))
    dgc_fla = (qn[0,:,0,:] * dqd * e_gc * scale
             + kn[0,:,0,:] * (dkd * e_gc - dki * e_neg_gc - dkr * e_gt_gc))

    def err_ratio(x, y):
        return ((x - y).square().mean().sqrt() / (y.square().mean().sqrt() + 1e-8)).item()

    print(f"dgc_bf16fwd vs dgc_fp32: {err_ratio(dgc_bf16fwd, dgc_fp32):.3e}")
    print(f"dgc_fla     vs dgc_fp32: {err_ratio(dgc_fla, dgc_fp32):.3e}")

    # Check magnitudes
    print(f"\nMagnitudes per term:")
    print(f"  |dqd*qd|:  {(dqd*qd).abs().mean():.3e}")
    print(f"  |dkd*kd|:  {(dkd*kd).abs().mean():.3e}")
    print(f"  |dki*ki|:  {(dki*ki).abs().mean():.3e}")
    print(f"  |dkr*kr|:  {(dkr*kr).abs().mean():.3e}")
    print(f"  |dgc|:     {dgc_fp32.abs().mean():.3e}")
    print(f"  ratio:     {(dqd*qd).abs().mean()/dgc_fp32.abs().mean():.1f}x")

    # Check if bf16 quantization of forward values matters
    print(f"\nBf16 quantization errors:")
    print(f"  kd: {err_ratio(kd_bf16, kd):.3e}")
    print(f"  qd: {err_ratio(qd_bf16, qd):.3e}")
    print(f"  ki: {err_ratio(ki_bf16, ki):.3e}")
    print(f"  kr: {err_ratio(kr_bf16, kr):.3e}")

    # Now simulate what K2_bwd does: fp32 d-values * bf16 workspace
    # vs K1_bwd: bf16 d-values * bf16 workspace
    dkd_bf16 = dkd.to(torch.bfloat16).float()
    dqd_bf16 = dqd.to(torch.bfloat16).float()
    dki_bf16 = dki.to(torch.bfloat16).float()
    dkr_bf16_d = dkr.to(torch.bfloat16).float()

    dgc_k2 = dqd * qd_bf16 + dkd * kd_bf16 - dki * ki_bf16 - dkr * kr_bf16  # K2: fp32 d * bf16 fwd
    dgc_k1 = dqd_bf16 * qd_bf16 + dkd_bf16 * kd_bf16 - dki_bf16 * ki_bf16 - dkr_bf16_d * kr_bf16  # K1: bf16 * bf16
    dgc_best = dqd * qd + dkd * kd - dki * ki_bf16 - dkr * kr_bf16  # best of k2: fp32 d, but fp32 fwd for qd/kd

    # Hybrid: fp32 d × fp32 recomputed forward
    # Use the ACTUAL kn (same as forward would compute)
    kn_bf16 = kf[0,:,0,:].to(torch.bfloat16).float()  # bf16 k
    kn_renorm = kn_bf16 * torch.rsqrt((kn_bf16 * kn_bf16).sum(-1, keepdim=True) + 1e-6)
    qn_bf16 = qf[0,:,0,:].to(torch.bfloat16).float()
    qn_renorm = qn_bf16 * torch.rsqrt((qn_bf16 * qn_bf16).sum(-1, keepdim=True) + 1e-6)

    # Recomputed forward values
    kd_recomp = kn_renorm * e_gc
    qd_recomp = qn_renorm * e_gc * scale
    ki_recomp = kn_renorm * e_neg_gc
    kr_recomp = kn_renorm * e_gt_gc

    dgc_recomp = dqd * qd_recomp + dkd * kd_recomp - dki * ki_recomp - dkr * kr_recomp

    print(f"\nDgc errors:")
    print(f"  K2 (fp32 d × bf16 fwd): {err_ratio(dgc_k2, dgc_fp32):.3e}")
    print(f"  K1 (bf16 d × bf16 fwd): {err_ratio(dgc_k1, dgc_fp32):.3e}")
    print(f"  FLA decomp (fp32 all):  {err_ratio(dgc_fla, dgc_fp32):.3e}")
    print(f"  Recomputed (fp32 d × fp32 fwd): {err_ratio(dgc_recomp, dgc_fp32):.3e}")
    print(f"\nForward value matching:")
    print(f"  kd_recomp vs kd_fp32: {err_ratio(kd_recomp, kd):.3e}")
    print(f"  ki_recomp vs ki_fp32: {err_ratio(ki_recomp, ki):.3e}")
    print(f"  qd_recomp vs qd_fp32: {err_ratio(qd_recomp, qd):.3e}")

    # Also test: bf16 d × fp32 recomputed forward
    dgc_k1_recomp = dqd_bf16 * qd_recomp + dkd_bf16 * kd_recomp - dki_bf16 * ki_recomp - dkr_bf16_d * kr_recomp
    print(f"  K1 recomp (bf16 d × fp32 fwd): {err_ratio(dgc_k1_recomp, dgc_fp32):.3e}")

    # Test with larger gates
    g_big = (torch.randn(B, T, H, D, device="cuda") * 4.0).to(torch.bfloat16)
    g_nat_big = _gate(g_big.float(), A_log, dt_bias, lb)
    g_nat_big_log2 = g_nat_big / LN2
    gc_big = g_nat_big_log2[0, :, 0, :].cumsum(0)
    gT_big = gc_big[-1, :]
    e_gc_big = 2.0 ** gc_big
    e_neg_gc_big = 2.0 ** (-gc_big)
    e_gt_gc_big = 2.0 ** (gT_big[None, :] - gc_big)

    kd_big = kn[0, :, 0, :] * e_gc_big
    qd_big = qn[0, :, 0, :] * e_gc_big * scale
    ki_big = kn[0, :, 0, :] * e_neg_gc_big
    kr_big = kn[0, :, 0, :] * e_gt_gc_big

    kd_big_bf16 = kd_big.to(torch.bfloat16).float()
    qd_big_bf16 = qd_big.to(torch.bfloat16).float()
    ki_big_bf16 = ki_big.to(torch.bfloat16).float()
    kr_big_bf16 = kr_big.to(torch.bfloat16).float()

    dgc_big_fp32 = dqd * qd_big + dkd * kd_big - dki * ki_big - dkr * kr_big
    dgc_big_bf16 = dqd * qd_big_bf16 + dkd * kd_big_bf16 - dki * ki_big_bf16 - dkr * kr_big_bf16
    dgc_big_fla = (qn[0,:,0,:] * dqd * e_gc_big * scale
                 + kn[0,:,0,:] * (dkd * e_gc_big - dki * e_neg_gc_big - dkr * e_gt_gc_big))

    print(f"\n=== With large gates (g_scale=4.0) ===")
    print(f"  |dgc|:     {dgc_big_fp32.abs().mean():.3e}")
    print(f"  |dqd*qd|:  {(dqd*qd_big).abs().mean():.3e}")
    print(f"  ratio:     {(dqd*qd_big).abs().mean()/dgc_big_fp32.abs().mean():.1f}x")
    print(f"  K2 error:  {err_ratio(dgc_big_bf16, dgc_big_fp32):.3e}")
    print(f"  FLA error: {err_ratio(dgc_big_fla, dgc_big_fp32):.3e}")

if __name__ == "__main__":
    main()
