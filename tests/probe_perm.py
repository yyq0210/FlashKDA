"""Is varlen dq a permutation of the (correct) batched dq? Same input, B=1 T=256."""
import math, torch
import torch.nn.functional as F
import flash_kda
from flash_kda.autograd import _kda_chunk_torch_fwd

def rms(x,y):
    x,y=x.detach().float(),y.detach().float()
    return ((x-y).square().mean().sqrt()/(y.square().mean().sqrt()+1e-12)).item()
def clone(t): return t.detach().clone().requires_grad_(True)

torch.manual_seed(0)
B,T,H,D=1,256,2,128; lb=-5.0; scale=1/math.sqrt(D); dev="cuda"
q=F.normalize(torch.randn(B,T,H,D,device=dev),p=2,dim=-1).to(torch.bfloat16)
k=F.normalize(torch.randn(B,T,H,D,device=dev),p=2,dim=-1).to(torch.bfloat16)
v=torch.randn(B,T,H,D,dtype=torch.bfloat16,device=dev)
g=(torch.randn(B,T,H,D,device=dev)*4).to(torch.bfloat16)
beta=torch.randn(B,T,H,dtype=torch.bfloat16,device=dev)
A_log=torch.rand(H,dtype=torch.float32,device=dev)
dt_bias=(torch.rand(H,D,dtype=torch.float32,device=dev)-0.5)*4
h0=torch.randn(1,H,D,D,dtype=torch.float32,device=dev)
do=torch.randn(B,T,H,D,dtype=torch.float32,device=dev)
dht=torch.randn(1,H,D,D,dtype=torch.float32,device=dev)
cu=torch.tensor([0,256],dtype=torch.long,device=dev)

def run(cu_arg):
    cin=[clone(x) for x in (q,k,v,g,beta,A_log,dt_bias,h0)]
    o,ht=flash_kda.flash_kda_func(cin[0],cin[1],cin[2],cin[3],cin[4],scale,cin[5],cin[6],lb,
                                  initial_state=cin[7],output_final_state=True,cu_seqlens=cu_arg)
    ((o.float()*do).sum()+(ht.float()*dht).sum()).backward()
    return [x.grad for x in cin], o

bg,bo=run(None)   # batched (correct)
vg,vo=run(cu)     # varlen

nm=["dq","dk","dv","dg","dbeta","dA_log","ddt_bias","dh0"]
print("fwd o  batched-vs-varlen:", rms(vo,bo))
for n,b,vv in zip(nm,bg,vg):
    print(f"{n:<9} varlen-vs-batched rms={rms(vv,b):.3e}")

# Per-column T-permutation check: sort along T axis for each (H,D)
def tperm(gb, gv, name):
    b=gb[0].float()  # [T,H,D]
    v=gv[0].float()
    asis=rms(v,b)
    bs=b.sort(dim=0).values  # sort along T per (H,D)
    vs=v.sort(dim=0).values
    sortedrms=rms(vs,bs)
    print(f"{name:<6} as-is={asis:.3e}  T-sorted-per-col={sortedrms:.3e}")

print("\n--- per-column sort-along-T (perm-invariant if T-permuted) ---")
tperm(bg[0],vg[0],"dq")
tperm(bg[1],vg[1],"dk")
tperm(bg[3],vg[3],"dg")

# Where does dq differ? per-token-block rms to see which tokens are wrong
dq_b=bg[0][0,:,0,:].float()  # [T,D] head0
dq_v=vg[0][0,:,0,:].float()
print("\nper-chunk(16) dq rms head0:")
for c in range(0,256,16):
    print(f"  tok[{c:3d}:{c+16}] rms={rms(dq_v[c:c+16],dq_b[c:c+16]):.3e}")
