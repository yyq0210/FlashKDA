"""FLA-independent backward verification for FlashKDA.

Two layers:
  L0  fp64 torch.autograd.gradcheck of the pure-torch reference
      (_kda_chunk_torch_fwd). This proves the reference's analytic
      (autograd) backward equals finite-difference, i.e. the reference is a
      trustworthy gold for gradients WITHOUT needing FLA.

  L1  CUDA flash_kda_func grads vs the fp64 torch-reference grads, on
      identical raw inputs, across a sweep of boundary shapes. Reports
      rel-RMSE / max-abs / max-rel / cosine for every gradient.

Run:
    source /home/fluentllmenv/bin/activate && \
        PYTHONPATH=/home/yuyanqi/FlashKDA python tests/test_bwd_gradcheck.py
"""

import math

import torch
import torch.nn.functional as F

import flash_kda
from flash_kda.autograd import _kda_chunk_torch_fwd


# ---------------- metrics ----------------
def metrics(x, y):
    x, y = x.detach().float(), y.detach().float()
    diff = (x - y)
    rms = (diff.square().mean().sqrt() / (y.square().mean().sqrt() + 1e-12)).item()
    mabs = diff.abs().max().item()
    mrel = (diff.abs() / (y.abs() + 1e-6)).max().item()
    cos = F.cosine_similarity(x.reshape(-1), y.reshape(-1), dim=0).item()
    return rms, mabs, mrel, cos


def _clone_leaf(t):
    return t.detach().clone().requires_grad_(True)


# ---------------- L0: fp64 gradcheck of the reference ----------------
def gradcheck_reference():
    print("=" * 72)
    print("L0  fp64 gradcheck of pure-torch reference _kda_chunk_torch_fwd")
    print("=" * 72)
    torch.manual_seed(0)
    B, T, H, D = 1, 32, 1, 8   # small D for tractable finite-diff
    lb = -3.0
    scale = 1.0 / math.sqrt(D)
    dev = "cuda"

    def mk(req):
        q = F.normalize(torch.randn(B, T, H, D, device=dev, dtype=torch.float64), p=2, dim=-1)
        k = F.normalize(torch.randn(B, T, H, D, device=dev, dtype=torch.float64), p=2, dim=-1)
        v = torch.randn(B, T, H, D, device=dev, dtype=torch.float64)
        g = torch.randn(B, T, H, D, device=dev, dtype=torch.float64) * 0.5
        beta = torch.randn(B, T, H, device=dev, dtype=torch.float64)
        A_log = torch.rand(H, device=dev, dtype=torch.float64)
        dt_bias = (torch.rand(H, D, device=dev, dtype=torch.float64) - 0.5) * 0.5
        h0 = torch.randn(B, H, D, D, device=dev, dtype=torch.float64)
        ts = [q, k, v, g, beta, A_log, dt_bias, h0]
        for t in ts:
            t.requires_grad_(req)
        return ts

    ts = mk(True)

    def fn(q, k, v, g, beta, A_log, dt_bias, h0):
        o, ht = _kda_chunk_torch_fwd(q, k, v, g, beta, scale, A_log, dt_bias,
                                     lb, h0, True, None)
        return o, ht

    ok = torch.autograd.gradcheck(
        fn, tuple(ts), eps=1e-6, atol=1e-4, rtol=1e-3,
        nondet_tol=0.0, raise_exception=False)
    print(f"  gradcheck (o, ht) : {'PASS' if ok else '**FAIL**'}")
    return ok


# ---------------- L1: CUDA vs fp64 reference ----------------
def make_inputs(B, T, H, D, lb, seed=0, cu=None):
    torch.manual_seed(seed)
    dev = "cuda"
    q = F.normalize(torch.randn(B, T, H, D, device=dev), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, D, device=dev), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=dev)
    g = (torch.randn(B, T, H, D, device=dev) * 4).to(torch.bfloat16)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device=dev)
    A_log = torch.rand(H, dtype=torch.float32, device=dev)
    dt_bias = (torch.rand(H, D, dtype=torch.float32, device=dev) - 0.5) * 4
    N = (cu.numel() - 1) if cu is not None else B
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=dev)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, g, beta, A_log, dt_bias, h0, scale


def cuda_vs_ref(label, B, T, H, D, lb, cu=None, seed=0):
    q, k, v, g, beta, A_log, dt_bias, h0, scale = make_inputs(B, T, H, D, lb, seed, cu)
    print(f"\n--- {label}: B={B} T={T} H={H} D={D} lb={lb} "
          f"varlen={cu is not None} seed={seed} ---")

    do = torch.randn(B, T, H, D, dtype=torch.float32, device="cuda")
    N = (cu.numel() - 1) if cu is not None else B
    dht = torch.randn(N, H, D, D, dtype=torch.float32, device="cuda")

    # fp64 reference grads (gold)
    rin = [q.double(), k.double(), v.double(), g.double(), beta.double(),
           A_log.double(), dt_bias.double(), h0.double()]
    rin = [_clone_leaf(x) for x in rin]
    o_r, ht_r = _kda_chunk_torch_fwd(rin[0], rin[1], rin[2], rin[3], rin[4], scale,
                                     rin[5], rin[6], lb, rin[7], True, cu)
    ((o_r * do.double()).sum() + (ht_r * dht.double()).sum()).backward()
    r_grads = [x.grad for x in rin]

    # CUDA grads
    cin = [_clone_leaf(x) for x in (q, k, v, g, beta, A_log, dt_bias, h0)]
    o_c, ht_c = flash_kda.flash_kda_func(cin[0], cin[1], cin[2], cin[3], cin[4], scale,
                                         cin[5], cin[6], lb, initial_state=cin[7],
                                         output_final_state=True, cu_seqlens=cu)
    ((o_c.float() * do).sum() + (ht_c.float() * dht).sum()).backward()
    c_grads = [x.grad for x in cin]

    # forward
    fo = metrics(o_c, o_r)
    fh = metrics(ht_c, ht_r)
    print(f"  fwd o    rms={fo[0]:.2e} maxabs={fo[1]:.2e} cos={fo[3]:.5f}")
    print(f"  fwd ht   rms={fh[0]:.2e} maxabs={fh[1]:.2e} cos={fh[3]:.5f}")

    names = ["dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias", "dh0"]
    print(f"  {'grad':<9} {'rel-rms':>9} {'max-abs':>9} {'max-rel':>9} {'cosine':>9}")
    worst_cos = 1.0
    for nm, gc, gr in zip(names, c_grads, r_grads):
        rms, mabs, mrel, cos = metrics(gc, gr)
        worst_cos = min(worst_cos, cos)
        print(f"  {nm:<9} {rms:9.2e} {mabs:9.2e} {mrel:9.2e} {cos:9.5f}")
    # bf16-kernel vs fp64-ref: rel-rms ~1e-2 expected; cosine is the robust gate.
    return worst_cos


def main():
    l0 = gradcheck_reference()

    print("\n" + "=" * 72)
    print("L1  CUDA flash_kda_func grads vs fp64 torch reference")
    print("=" * 72)
    cases = []
    cases.append(cuda_vs_ref("batched", 2, 256, 3, 128, -5.0))
    cases.append(cuda_vs_ref("batched_lb1", 1, 512, 2, 128, -1.0))
    cases.append(cuda_vs_ref("tail_tile", 1, 48, 2, 128, -5.0))      # 48 = 3*16 exact
    cases.append(cuda_vs_ref("partial_tile", 1, 40, 2, 128, -5.0))   # 40 -> pad to 48
    cases.append(cuda_vs_ref("single_chunk", 1, 16, 1, 128, -3.0))
    cases.append(cuda_vs_ref("large_gate", 1, 128, 2, 128, -8.0))
    seq = [192, 64, 304]
    cu = torch.tensor([0] + torch.cumsum(torch.tensor(seq), 0).tolist(),
                      dtype=torch.long, device="cuda")
    cases.append(cuda_vs_ref("varlen", 1, sum(seq), 2, 128, -5.0, cu=cu))
    seq2 = [16, 33, 7, 88]
    cu2 = torch.tensor([0] + torch.cumsum(torch.tensor(seq2), 0).tolist(),
                       dtype=torch.long, device="cuda")
    cases.append(cuda_vs_ref("varlen_short", 1, sum(seq2), 1, 128, -3.0, cu=cu2))

    print("\n" + "=" * 72)
    worst = min(cases)
    print(f"L0 reference gradcheck : {'PASS' if l0 else 'FAIL'}")
    print(f"L1 worst grad cosine   : {worst:.5f}  "
          f"({'PASS' if worst > 0.999 else 'CHECK'})")


if __name__ == "__main__":
    main()
