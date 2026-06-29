"""Localize the varlen backward bug. Gold = fp64 torch reference."""
import math, torch
import torch.nn.functional as F
import flash_kda
from flash_kda.autograd import _kda_chunk_torch_fwd

def rms(x, y):
    x, y = x.detach().float(), y.detach().float()
    return ((x - y).square().mean().sqrt() / (y.square().mean().sqrt() + 1e-12)).item()

def clone(t): return t.detach().clone().requires_grad_(True)

def run(label, seq, H=2, lb=-5.0, seed=0):
    T = sum(seq); B = 1; D = 128
    cu = torch.tensor([0]+torch.cumsum(torch.tensor(seq),0).tolist(), dtype=torch.long, device="cuda")
    N = len(seq)
    torch.manual_seed(seed); dev="cuda"
    q = F.normalize(torch.randn(B,T,H,D,device=dev),p=2,dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(B,T,H,D,device=dev),p=2,dim=-1).to(torch.bfloat16)
    v = torch.randn(B,T,H,D,dtype=torch.bfloat16,device=dev)
    g = (torch.randn(B,T,H,D,device=dev)*4).to(torch.bfloat16)
    beta = torch.randn(B,T,H,dtype=torch.bfloat16,device=dev)
    A_log = torch.rand(H,dtype=torch.float32,device=dev)
    dt_bias = (torch.rand(H,D,dtype=torch.float32,device=dev)-0.5)*4
    h0 = torch.randn(N,H,D,D,dtype=torch.float32,device=dev)
    scale = 1.0/math.sqrt(D)
    do = torch.randn(B,T,H,D,dtype=torch.float32,device=dev)
    dht = torch.randn(N,H,D,D,dtype=torch.float32,device=dev)

    rin=[clone(x.double()) for x in (q,k,v,g,beta,A_log,dt_bias,h0)]
    o_r,ht_r=_kda_chunk_torch_fwd(rin[0],rin[1],rin[2],rin[3],rin[4],scale,rin[5],rin[6],lb,rin[7],True,cu)
    ((o_r*do.double()).sum()+(ht_r*dht.double()).sum()).backward()
    rg=[x.grad for x in rin]

    cin=[clone(x) for x in (q,k,v,g,beta,A_log,dt_bias,h0)]
    o_c,ht_c=flash_kda.flash_kda_func(cin[0],cin[1],cin[2],cin[3],cin[4],scale,cin[5],cin[6],lb,initial_state=cin[7],output_final_state=True,cu_seqlens=cu)
    ((o_c.float()*do).sum()+(ht_c.float()*dht).sum()).backward()
    cg=[x.grad for x in cin]

    fo=rms(o_c,o_r)
    nm=["dq","dk","dv","dg","dbeta","dA_log","ddt_bias","dh0"]
    parts=" ".join(f"{n}={rms(c,r):.2e}" for n,c,r in zip(nm,cg,rg))
    print(f"{label:<22} seq={seq} fwd_o={fo:.2e} | {parts}")

print("=== varlen localization (gold=fp64 ref) ===")
run("single_seg_aligned", [256])       # 1 segment, mult of 16
run("single_seg_ragged",  [250])       # 1 segment, not mult of 16
run("two_seg_aligned",    [256,256])   # 2 segments, both mult of 16
run("two_seg_ragged",     [192,304])   # original first two (192,304 mult of 16)
run("three_seg_orig",     [192,64,304])# original failing case
run("two_seg_nonmult",    [200,120])   # neither mult of 16
