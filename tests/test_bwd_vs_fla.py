"""Compare FlashKDA backward vs FLA and vs torch reference."""
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

    # Torch reference
    tr = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log, dt_bias, h0)]
    o_tr, _ = _kda_chunk_torch_fwd(tr[0], tr[1], tr[2], tr[3], tr[4], scale, tr[5], tr[6], lb, tr[7], True, None)
    (o_tr.float() * do.float()).sum().backward()
    tr_grads = {nm: x.grad for nm, x in zip(["q","k","v","g","beta","A_log","dt_bias","h0"], tr)}

    # FLA
    from fla.ops.kda import chunk_kda
    fl = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log)]
    dt_fl = dt_bias.detach().reshape(-1).clone().requires_grad_(True)
    h0_fl = _clone_leaf(h0)
    o_fl, ht_fl = chunk_kda(
        q=fl[0], k=fl[1], v=fl[2], g=fl[3], beta=fl[4], scale=scale,
        initial_state=h0_fl, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        A_log=fl[5], dt_bias=dt_fl, lower_bound=lb,
        transpose_state_layout=True, cu_seqlens=None,
    )
    (o_fl.float() * do.float()).sum().backward()
    fl_grads = {nm: x.grad for nm, x in zip(["q","k","v","g","beta","A_log"], fl)}
    fl_grads["dt_bias"] = dt_fl.grad.reshape(dt_bias.shape)
    fl_grads["h0"] = h0_fl.grad

    # CUDA
    cu = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log, dt_bias, h0)]
    o_cu, _ = flash_kda.flash_kda_func(cu[0], cu[1], cu[2], cu[3], cu[4], scale, cu[5], cu[6], lb,
                                        initial_state=cu[7], output_final_state=True, cu_seqlens=None)
    (o_cu.float() * do.float()).sum().backward()
    cu_grads = {nm: x.grad for nm, x in zip(["q","k","v","g","beta","A_log","dt_bias","h0"], cu)}

    print(f"fwd o: cuda={err_ratio(o_cu, o_tr):.3e}  fla={err_ratio(o_fl, o_tr):.3e}")

    names = ["q", "k", "v", "g", "beta", "A_log", "dt_bias", "h0"]
    print(f"\n{'grad':<10} {'cuda_vs_ref':>12} {'fla_vs_ref':>12} {'cuda_vs_fla':>12}")
    for nm in names:
        cr = err_ratio(cu_grads[nm], tr_grads[nm])
        fr = err_ratio(fl_grads[nm], tr_grads[nm])
        cf = err_ratio(cu_grads[nm], fl_grads[nm])
        print(f"  d{nm:<8} {cr:12.3e} {fr:12.3e} {cf:12.3e}")

if __name__ == "__main__":
    main()
