import math, torch
import torch.nn.functional as F
import flash_kda
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
do=torch.randn(B,T,H,D,dtype=torch.float32,device=dev)
cu=torch.tensor([0,256],dtype=torch.long,device=dev)
for tag,h0 in [("h0=0", torch.zeros(1,H,D,D,dtype=torch.float32,device=dev)),
               ("h0=rand", torch.randn(1,H,D,D,dtype=torch.float32,device=dev))]:
    dht=torch.randn(1,H,D,D,dtype=torch.float32,device=dev)
    def run(cu_arg):
        cin=[clone(x) for x in (q,k,v,g,beta,A_log,dt_bias,h0)]
        o,ht=flash_kda.flash_kda_func(cin[0],cin[1],cin[2],cin[3],cin[4],scale,cin[5],cin[6],lb,
                                      initial_state=cin[7],output_final_state=True,cu_seqlens=cu_arg)
        ((o.float()*do).sum()+(ht.float()*dht).sum()).backward()
        return [x.grad for x in cin]
    bg=run(None); vg=run(cu)
    print(f"{tag}: dq={rms(vg[0],bg[0]):.3e} dk={rms(vg[1],bg[1]):.3e} dg={rms(vg[3],bg[3]):.3e}")
# Also: does forward output_final_state=False change anything? (no h0)
