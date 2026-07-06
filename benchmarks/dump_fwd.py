"""Dump flash_kda.fwd o/fs for fixed inputs to a .pt file, using whichever
flash_kda_C .so is importable. Run twice (baseline vs new) then compare.

Usage:
    PYTHONPATH=... python dump_fwd.py <out.pt>
"""
import sys, math, torch, torch.nn.functional as F, flash_kda

LB = -5.0

def make(seq_lens, H, D, seed=0):
    dev = torch.device("cuda"); torch.manual_seed(seed)
    T = sum(seq_lens); N = len(seq_lens); varlen = N > 1
    q = F.normalize(torch.randn(1, T, H, D, device=dev), p=2, dim=-1).bfloat16()
    k = F.normalize(torch.randn(1, T, H, D, device=dev), p=2, dim=-1).bfloat16()
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
    g = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
    beta = torch.randn(1, T, H, device=dev, dtype=torch.bfloat16)
    A_log = torch.rand(H, device=dev); dt_bias = torch.rand(H, D, device=dev)
    h0 = torch.randn(N, H, D, D, device=dev, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(D)
    return q,k,v,g,beta,A_log,dt_bias,h0,scale,N,T,varlen

def run(inp):
    q,k,v,g,beta,A_log,dt_bias,h0,scale,N,T,varlen = inp
    out = torch.zeros_like(q); fs = torch.zeros_like(h0); extra={}
    if varlen:
        sl=[T//N]*N
        cu=torch.tensor([0]+torch.cumsum(torch.tensor(sl),0).tolist(),dtype=torch.long,device=q.device)
        extra["cu_seqlens"]=cu
    flash_kda.fwd(q,k,v,g,beta,scale,out,A_log=A_log,dt_bias=dt_bias,
                  lower_bound=LB,initial_state=h0,final_state=fs,**extra)
    return out,fs

if __name__ == "__main__":
    res = {}
    for tag,(sl,H,D) in {"a":([1024],64,128),"b":([8192],96,128),"c":([1024]*8,64,128)}.items():
        o,fs = run(make(sl,H,D))
        res[tag]=(o.float().cpu(), fs.float().cpu())
    torch.save(res, sys.argv[1])
    print("dumped", sys.argv[1])
