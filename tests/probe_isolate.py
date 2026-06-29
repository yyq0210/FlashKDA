import math, torch, flash_kda
import torch.nn.functional as F
def rms(x,y):
    x,y=x.detach().float(),y.detach().float()
    return ((x-y).square().mean().sqrt()/(y.square().mean().sqrt()+1e-12)).item()
def clone(t): return t.detach().clone().requires_grad_(True)

def case(T, use_dht, h0scale, seed=0):
    B,H,D=1,2,128; lb=-5.0; scale=1/math.sqrt(D); dev="cuda"
    torch.manual_seed(seed)
    q=F.normalize(torch.randn(B,T,H,D,device=dev),p=2,dim=-1).to(torch.bfloat16)
    k=F.normalize(torch.randn(B,T,H,D,device=dev),p=2,dim=-1).to(torch.bfloat16)
    v=torch.randn(B,T,H,D,dtype=torch.bfloat16,device=dev)
    g=(torch.randn(B,T,H,D,device=dev)*4).to(torch.bfloat16)
    beta=torch.randn(B,T,H,dtype=torch.bfloat16,device=dev)
    A_log=torch.rand(H,dtype=torch.float32,device=dev)
    dt_bias=(torch.rand(H,D,dtype=torch.float32,device=dev)-0.5)*4
    h0=(torch.randn(1,H,D,D,dtype=torch.float32,device=dev)*h0scale)
    do=torch.randn(B,T,H,D,dtype=torch.float32,device=dev)
    dht=torch.randn(1,H,D,D,dtype=torch.float32,device=dev) if use_dht else torch.zeros(1,H,D,D,dtype=torch.float32,device=dev)
    cu=torch.tensor([0,T],dtype=torch.long,device=dev)
    def run(cu_arg):
        cin=[clone(x) for x in (q,k,v,g,beta,A_log,dt_bias,h0)]
        o,ht=flash_kda.flash_kda_func(cin[0],cin[1],cin[2],cin[3],cin[4],scale,cin[5],cin[6],lb,
                                      initial_state=cin[7],output_final_state=True,cu_seqlens=cu_arg)
        loss=(o.float()*do).sum()+(ht.float()*dht).sum()
        loss.backward()
        return [x.grad for x in cin]
    bg=run(None); vg=run(cu)
    nm=["dq","dk","dv","dg","dbeta","dA_log","ddt_bias","dh0"]
    s=" ".join(f"{n}={rms(v_,b):.2e}" for n,v_,b in zip(nm,vg,bg))
    print(f"T={T:<4} dht={int(use_dht)} h0x={h0scale}: {s}")

print("=== isolate (varlen vs batched, identical inputs) ===")
case(16, True, 1.0)
case(16, False, 1.0)
case(16, True, 0.0)
case(32, True, 1.0)
case(32, False, 0.0)
case(256, False, 0.0)
