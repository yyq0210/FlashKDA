"""torch-profiler K1(prepare) vs K2(recurrence) kernel-time split for flash_kda.fwd."""
import math
import torch
import torch.nn.functional as F
import flash_kda
from torch.profiler import profile, ProfilerActivity


def make(seq_lens, H, D):
    dev = torch.device("cuda")
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
    fs = torch.zeros_like(h0)
    out = torch.zeros_like(q)
    extra = {}
    if varlen:
        cu = torch.tensor([0] + torch.cumsum(torch.tensor(seq_lens), 0).tolist(),
                          dtype=torch.long, device=dev)
        extra["cu_seqlens"] = cu
    scale = 1.0 / math.sqrt(D)

    def run():
        flash_kda.fwd(q, k, v, g, beta, scale, out, A_log=A_log, dt_bias=dt_bias,
                      lower_bound=-5.0, initial_state=h0, final_state=fs, **extra)
    return run


def split(seq_lens, H, D, iters=50):
    run = make(seq_lens, H, D)
    for _ in range(30):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
    k1 = k2 = other = 0.0
    rows = []
    for e in prof.key_averages():
        t = e.self_device_time_total  # us total across iters
        if t <= 0:
            continue
        nm = e.key
        if "prepare" in nm:
            k1 += t
        elif "recurrence" in nm:
            k2 += t
        else:
            other += t
        rows.append((t / iters, nm))
    tot = k1 + k2 + other
    N = len(seq_lens); T = sum(seq_lens)
    print(f"\n=== N={N} T_total={T} H={H} D={D}  (grid_k2 = N*H = {N*H} blocks) ===")
    print(f"  total GPU/iter  {tot/iters:8.3f} us")
    print(f"  K1 prepare      {k1/iters:8.3f} us  ({100*k1/tot:5.1f}%)")
    print(f"  K2 recurrence   {k2/iters:8.3f} us  ({100*k2/tot:5.1f}%)")
    if other > 0:
        print(f"  other           {other/iters:8.3f} us  ({100*other/tot:5.1f}%)")
    print("  top kernels (us/iter):")
    for t, nm in sorted(rows, reverse=True)[:6]:
        print(f"    {t:8.3f}  {nm[:80]}")


if __name__ == "__main__":
    split([8192], 96, 128)      # their headline: single long seq, N=1
    split([1024], 64, 128)      # my v-series config, N=1 (max underutilization)
    split([1024] * 8, 64, 128)  # N=8 varlen -> grid_k2 = 512 blocks
