"""Check the dgc computation values."""
import math
import torch
import torch.nn.functional as F
import flash_kda
from flash_kda.autograd import _l2norm

CHUNK = 16
_L2_EPS = 1e-6

def main():
    B, T, H, D = 1, 16, 1, 128
    lb = -5.0
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(42)

    q = F.normalize(torch.randn(B, T, H, D, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, D, device="cuda"), p=2, dim=-1).to(torch.bfloat16)

    # Check L2 norm
    q_f = q.float()
    k_f = k.float()
    q_norm = (q_f * q_f).sum(-1)  # [B, T, H]
    k_norm = (k_f * k_f).sum(-1)
    print(f"q norm^2 range: [{q_norm.min():.6f}, {q_norm.max():.6f}]")
    print(f"k norm^2 range: [{k_norm.min():.6f}, {k_norm.max():.6f}]")

    # After F.normalize, norm should be 1.0
    # But in bf16, there's some quantization
    qn = _l2norm(q_f)
    kn = _l2norm(k_f)
    print(f"qn norm^2 range: [{(qn*qn).sum(-1).min():.6f}, {(qn*qn).sum(-1).max():.6f}]")
    print(f"kn norm^2 range: [{(kn*kn).sum(-1).min():.6f}, {(kn*kn).sum(-1).max():.6f}]")

    # Compare: q_bf16 already normalized, so qn(q_bf16) = q_bf16 / ||q_bf16||
    # where ||q_bf16|| ≈ 1.0 but not exactly
    # So kn_from_code ≈ k_bf16 / ||k_bf16|| ≈ k_bf16 * rsqrt(||k_bf16||^2 + eps)
    # vs kn_from_ref = k_f32 / ||k_f32||_f32
    kn_code = k_f * torch.rsqrt((k_f * k_f).sum(-1, keepdim=True) + 1e-6)
    print(f"kn_code vs kn_ref: {((kn_code - kn).square().mean().sqrt() / kn.square().mean().sqrt()).item():.3e}")

    # Now let me compute gc using the CUDA kernel's approach
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = (torch.rand(H, D, dtype=torch.float32, device="cuda") - 0.5) * 0.5
    g = (torch.randn(B, T, H, D, device="cuda") * 0.5).to(torch.bfloat16)

    a_exp = torch.exp(A_log[0]).item()
    LN2 = 0.6931471805599453
    gate_scale = lb / LN2

    # Compute gc in log2 units (matching CUDA K1_bwd)
    gc_log2 = torch.zeros(T, D, device="cuda")
    g_nat_log2 = torch.zeros(T, D, device="cuda")
    for t in range(T):
        for col in range(D):
            g_raw = g[0, t, 0, col].float().item()
            z = a_exp * (g_raw + dt_bias[0, col].item())
            sig = 1.0 / (1.0 + math.exp(-z))  # sigmoid
            g_nat_val = gate_scale * sig
            g_nat_log2[t, col] = g_nat_val
            if t == 0:
                gc_log2[t, col] = g_nat_val
            else:
                gc_log2[t, col] = gc_log2[t-1, col] + g_nat_val

    # Now compute kd, qd, ki, kr in fp32 from gc_log2
    import math as m
    kd_fp32 = torch.zeros(T, D, device="cuda")
    qd_fp32 = torch.zeros(T, D, device="cuda")
    ki_fp32 = torch.zeros(T, D, device="cuda")
    kr_fp32 = torch.zeros(T, D, device="cuda")

    gT_log2 = gc_log2[-1, :]  # [D]

    for t in range(T):
        kn_t = kn[0, t, 0, :].float()  # [D]
        qn_t = qn[0, t, 0, :].float()

        exp_gc = (2.0 ** gc_log2[t, :].float())
        exp_neg_gc = (2.0 ** (-gc_log2[t, :].float()))
        exp_gt_gc = (2.0 ** (gT_log2.float() - gc_log2[t, :].float()))

        kd_fp32[t] = kn_t * exp_gc
        qd_fp32[t] = qn_t * exp_gc * scale
        ki_fp32[t] = kn_t * exp_neg_gc
        kr_fp32[t] = kn_t * exp_gt_gc

    # Compare with workspace kd
    out = torch.empty_like(q)
    T_total = B * T
    N = B
    total_tiles = (T_total + CHUNK - 1) // CHUNK
    all_states = torch.empty(H * total_tiles, D, D, dtype=torch.bfloat16, device=q.device)

    workspace = flash_kda.fwd(
        q, k, torch.randn_like(q), g, torch.randn(B,T,H, dtype=torch.bfloat16, device="cuda"),
        float(scale), out,
        A_log=A_log, dt_bias=dt_bias, lower_bound=float(lb),
        initial_state=None, final_state=None, cu_seqlens=None,
        all_states=all_states,
    )

    # Parse workspace
    n_ht = H * total_tiles
    CD = CHUNK * D
    bf16_ws = workspace.view(torch.bfloat16)
    ws_kd = bf16_ws[:n_ht * CD].view(n_ht, CHUNK, D)[0].float()
    ws_qd = bf16_ws[n_ht*CD:n_ht*2*CD].view(n_ht, CHUNK, D)[0].float()
    ws_kr = bf16_ws[n_ht*2*CD:n_ht*3*CD].view(n_ht, CHUNK, D)[0].float()
    ws_ki = bf16_ws[n_ht*3*CD:n_ht*4*CD].view(n_ht, CHUNK, D)[0].float()

    def err_ratio(x, y):
        x, y = x.detach().float(), y.detach().float()
        return ((x - y).square().mean().sqrt() / (y.square().mean().sqrt() + 1e-8)).item()

    print(f"\nkd_fp32 vs ws_kd: {err_ratio(kd_fp32, ws_kd):.3e}")
    print(f"qd_fp32 vs ws_qd: {err_ratio(qd_fp32, ws_qd):.3e}")
    print(f"ki_fp32 vs ws_ki: {err_ratio(ki_fp32, ws_ki):.3e}")
    print(f"kr_fp32 vs ws_kr: {err_ratio(kr_fp32, ws_kr):.3e}")

    # Check magnitude
    print(f"\n|kd_fp32|: {kd_fp32.abs().mean():.3e}  |ws_kd|: {ws_kd.abs().mean():.3e}  ratio: {kd_fp32.abs().mean()/ws_kd.abs().mean():.3f}")
    print(f"|qd_fp32|: {qd_fp32.abs().mean():.3e}  |ws_qd|: {ws_qd.abs().mean():.3e}  ratio: {qd_fp32.abs().mean()/ws_qd.abs().mean():.3f}")

    # Check a few elements
    print(f"\n--- kd_fp32[0,:5] vs ws_kd[0,:5] ---")
    for i in range(5):
        print(f"  i={i}: fp32={kd_fp32[0,i]:.6f}  ws={ws_kd[0,i]:.6f}")


if __name__ == "__main__":
    main()
