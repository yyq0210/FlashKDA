"""Multi-chunk backward test: B=1, T=32 (2 chunks), H=1, D=128.
Compares CUDA backward vs FLA and vs torch reference.
Tests both with moderate and large gate magnitudes.
"""
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

def run_test(label, B, T, H, D, lb, g_scale, dt_scale, seed=42):
    print(f"\n=== {label}: B={B} T={T} H={H} D={D} lb={lb} g_scale={g_scale} dt_scale={dt_scale} ===")
    scale = 1.0 / math.sqrt(D)
    torch.manual_seed(seed)

    q = F.normalize(torch.randn(B, T, H, D, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, D, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    g = (torch.randn(B, T, H, D, device="cuda") * g_scale).to(torch.bfloat16)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = (torch.rand(H, D, dtype=torch.float32, device="cuda") - 0.5) * dt_scale
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") * 0.01

    do = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    dht = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") * 0.01

    # Torch reference
    tr = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log, dt_bias, h0)]
    o_tr, ht_tr = _kda_chunk_torch_fwd(tr[0], tr[1], tr[2], tr[3], tr[4], scale, tr[5], tr[6], lb, tr[7], True, None)
    ((o_tr.float() * do.float()).sum() + (ht_tr.float() * dht.float()).sum()).backward()
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
    ((o_fl.float() * do.float()).sum() + (ht_fl.float() * dht.float()).sum()).backward()
    fl_grads = {nm: x.grad for nm, x in zip(["q","k","v","g","beta","A_log"], fl)}
    fl_grads["dt_bias"] = dt_fl.grad.reshape(dt_bias.shape)
    fl_grads["h0"] = h0_fl.grad

    # CUDA
    cu = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log, dt_bias, h0)]
    o_cu, ht_cu = flash_kda.flash_kda_func(cu[0], cu[1], cu[2], cu[3], cu[4], scale,
                                            cu[5], cu[6], lb,
                                            initial_state=cu[7], output_final_state=True, cu_seqlens=None)
    ((o_cu.float() * do.float()).sum() + (ht_cu.float() * dht.float()).sum()).backward()
    cu_grads = {nm: x.grad for nm, x in zip(["q","k","v","g","beta","A_log","dt_bias","h0"], cu)}

    print(f"fwd o: cuda_vs_ref={err_ratio(o_cu, o_tr):.3e}  fla_vs_ref={err_ratio(o_fl, o_tr):.3e}")

    names = ["q", "k", "v", "g", "beta", "A_log", "dt_bias", "h0"]
    print(f"{'grad':<10} {'cuda_vs_ref':>12} {'fla_vs_ref':>12} {'cuda_vs_fla':>12}")
    for nm in names:
        cr = err_ratio(cu_grads[nm], tr_grads[nm])
        fr = err_ratio(fl_grads[nm], tr_grads[nm])
        cf = err_ratio(cu_grads[nm], fl_grads[nm])
        flag = "OK" if cf < 2e-2 else "FAIL"
        print(f"  d{nm:<8} {cr:12.3e} {fr:12.3e} {cf:12.3e}  {flag}")

if __name__ == "__main__":
    # Single chunk, small gates
    run_test("1chunk_small", B=1, T=16, H=1, D=128, lb=-5.0, g_scale=0.5, dt_scale=0.5)
    # 2 chunks, small gates
    run_test("2chunk_small", B=1, T=32, H=1, D=128, lb=-5.0, g_scale=0.5, dt_scale=0.5)
    # 2 chunks, moderate gates
    run_test("2chunk_mod", B=1, T=32, H=1, D=128, lb=-5.0, g_scale=2.0, dt_scale=2.0)
    # Multi-chunk, same as main test
    run_test("multi_main", B=2, T=256, H=3, D=128, lb=-5.0, g_scale=4.0, dt_scale=4.0)
