"""Validate V-split K2 forward vs fla chunk_kda gold + fused_recurrent; then bench K2 split.

Run:
    PYTHONPATH=/home/yuyanqi/FlashKDA:/home/yuyanqi/flash-linear-attention \
        python /home/yuyanqi/FlashKDA/benchmarks/check_vsplit.py
"""
import math
import torch
import torch.nn.functional as F
import flash_kda
from fla.ops.kda import chunk_kda, fused_recurrent_kda

LB = -5.0


def err(a, b):
    a, b = a.float(), b.float()
    return ((a - b).square().mean().sqrt() / (b.square().mean().sqrt() + 1e-8)).item()


def make(seq_lens, H, D, seed=0):
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    T = sum(seq_lens); N = len(seq_lens)
    varlen = N > 1
    q = F.normalize(torch.randn(1, T, H, D, device=dev), p=2, dim=-1).bfloat16()
    k = F.normalize(torch.randn(1, T, H, D, device=dev), p=2, dim=-1).bfloat16()
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
    g = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
    beta = torch.randn(1, T, H, device=dev, dtype=torch.bfloat16)
    A_log = torch.rand(H, device=dev)
    dt_bias = torch.rand(H, D, device=dev)
    h0 = torch.randn(N, H, D, D, device=dev, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, g, beta, A_log, dt_bias, h0, scale, N, T, varlen


def run_flash(inp):
    q, k, v, g, beta, A_log, dt_bias, h0, scale, N, T, varlen = inp
    out = torch.zeros_like(q)
    fs = torch.zeros_like(h0)
    extra = {}
    if varlen:
        seq_lens = [T // N] * N
        cu = torch.tensor([0] + torch.cumsum(torch.tensor(seq_lens), 0).tolist(),
                          dtype=torch.long, device=q.device)
        extra["cu_seqlens"] = cu
    flash_kda.fwd(q, k, v, g, beta, scale, out, A_log=A_log, dt_bias=dt_bias,
                  lower_bound=LB, initial_state=h0, final_state=fs, **extra)
    return out, fs


def gold_chunk(inp):
    q, k, v, g, beta, A_log, dt_bias, h0, scale, N, T, varlen = inp
    dt_flat = dt_bias.reshape(-1)
    h0f = h0.float()
    if varlen:
        seq_lens = [T // N] * N
        cu = torch.tensor([0] + torch.cumsum(torch.tensor(seq_lens), 0).tolist(),
                          dtype=torch.long, device=q.device)
        o, fs = chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_flat,
            scale=scale, initial_state=h0f, output_final_state=True, cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=LB, chunk_size=64)
    else:
        o, fs = chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_flat,
            scale=scale, initial_state=h0f, output_final_state=True,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=LB, chunk_size=64)
    return o, fs


def timeit(fn, iters=50, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def check(seq_lens, H, D):
    inp = make(seq_lens, H, D)
    o, fs = run_flash(inp)
    og, fsg = gold_chunk(inp)
    eo, ef = err(o, og), err(fs, fsg)
    N = len(seq_lens); T = sum(seq_lens)
    print(f"  N={N} T={T} H={H} D={D}: o {eo:.2e}  fs {ef:.2e}  "
          f"{'OK' if eo < 2e-2 and ef < 2e-2 else 'FAIL'}")
    return eo, ef


if __name__ == "__main__":
    print("== V-split correctness vs fla chunk(cs=64) ==")
    check([1024], 64, 128)
    check([8192], 96, 128)
    check([1024] * 8, 64, 128)
    print("\n== bench (ms/iter) ==")
    for sl, H, D in [([1024], 64, 128), ([8192], 96, 128), ([1024] * 8, 64, 128)]:
        inp = make(sl, H, D)
        t = timeit(lambda: run_flash(inp))
        N = len(sl); T = sum(sl)
        print(f"  N={N} T={T} H={H} D={D}: flash {t:.4f} ms")
