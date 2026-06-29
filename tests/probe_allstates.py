import math, torch, flash_kda
import torch.nn.functional as F
from flash_kda.autograd import _compute_total_tiles
torch.manual_seed(0)
B,T,H,D=1,256,2,128; lb=-5.0; scale=1/math.sqrt(D); dev="cuda"
q=F.normalize(torch.randn(B,T,H,D,device=dev),p=2,dim=-1).to(torch.bfloat16).contiguous()
k=F.normalize(torch.randn(B,T,H,D,device=dev),p=2,dim=-1).to(torch.bfloat16).contiguous()
v=torch.randn(B,T,H,D,dtype=torch.bfloat16,device=dev).contiguous()
g=(torch.randn(B,T,H,D,device=dev)*4).to(torch.bfloat16).contiguous()
beta=torch.randn(B,T,H,dtype=torch.bfloat16,device=dev).contiguous()
A_log=torch.rand(H,dtype=torch.float32,device=dev).contiguous()
dt_bias=(torch.rand(H,D,dtype=torch.float32,device=dev)-0.5*4).contiguous()
h0=torch.randn(1,H,D,D,dtype=torch.float32,device=dev).contiguous()
cu=torch.tensor([0,256],dtype=torch.long,device=dev)

def fwd(cu_arg):
    N=(cu_arg.numel()-1) if cu_arg is not None else B
    out=torch.empty(B,T,H,D,dtype=torch.bfloat16,device=dev)
    fs=torch.empty(N,H,D,D,dtype=torch.float32,device=dev)
    tt=_compute_total_tiles(B*T,N,cu_arg)
    allst=torch.zeros(H*tt,D,D,dtype=torch.bfloat16,device=dev)
    ws=flash_kda.fwd(q,k,v,g,beta,float(scale),out,A_log=A_log,dt_bias=dt_bias,
        lower_bound=float(lb),initial_state=h0.to(torch.float32).contiguous(),
        final_state=fs,cu_seqlens=cu_arg,all_states=allst)
    return out,fs,allst,tt

ob,fsb,ab,ttb=fwd(None)
ov,fsv,av,ttv=fwd(cu)
print("total_tiles batched=",ttb," varlen=",ttv)
def rms(x,y):
    x,y=x.float(),y.float(); return ((x-y).square().mean().sqrt()/(y.square().mean().sqrt()+1e-12)).item()
print("fwd out  rms:", rms(ov,ob))
print("final_st rms:", rms(fsv,fsb))
# all_states head0 chunk0 = index 0 in both
print("all_states[0] (h0,head0,chunk0) batched-vs-varlen rms:", rms(av[0],ab[0]))
# compare to actual h0 (transposed: all_states stores S^T[V,K] = state[k][v]? autograd transposes initial_state)
# initial_state passed is [N,H,D,D] = [1,2,128,128]; the kernel state is S[K,V], stored S^T[V,K]
print("all_states[0] batched vs h0[0,0]   rms:", rms(ab[0], h0[0,0]))
print("all_states[0] batched vs h0[0,0].T rms:", rms(ab[0], h0[0,0].t()))
# head1 chunk0
print("all_states head1 chunk0: batched idx", ttb, "varlen idx", ttv)
print("  rms:", rms(av[ttv], ab[ttb]))
