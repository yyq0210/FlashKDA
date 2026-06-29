import math, torch, flash_kda
import torch.nn.functional as F
from flash_kda.autograd import _kda_chunk_torch_fwd
def finite(t): return bool(torch.isfinite(t).all())
B,T,H,D=1,128,2,128; scale=1/math.sqrt(D); dev="cuda"
def mk(seed=0, gx=4.0, dtx=4.0):
    torch.manual_seed(seed)
    q=F.normalize(torch.randn(B,T,H,D,device=dev),p=2,dim=-1).to(torch.bfloat16)
    k=F.normalize(torch.randn(B,T,H,D,device=dev),p=2,dim=-1).to(torch.bfloat16)
    v=torch.randn(B,T,H,D,dtype=torch.bfloat16,device=dev)
    g=(torch.randn(B,T,H,D,device=dev)*gx).to(torch.bfloat16)
    beta=torch.randn(B,T,H,dtype=torch.bfloat16,device=dev)
    A_log=torch.rand(H,dtype=torch.float32,device=dev)
    dt_bias=(torch.rand(H,D,dtype=torch.float32,device=dev)-0.5)*dtx
    h0=torch.zeros(1,H,D,D,dtype=torch.float32,device=dev)
    return q,k,v,g,beta,A_log,dt_bias,h0
print("lb sweep (gx=4,dtx=4): kernel forward finite? + FLA finite?")
for lb in [-1,-2,-3,-4,-5,-6,-7,-8]:
    q,k,v,g,beta,A_log,dt_bias,h0=mk()
    o,ht=flash_kda.flash_kda_func(q,k,v,g,beta,scale,A_log,dt_bias,float(lb),
                                  initial_state=h0,output_final_state=True)
    kf=finite(o) and finite(ht)
    # FLA
    try:
        from fla.ops.kda import chunk_kda
        dtf=dt_bias.reshape(-1).clone()
        of,hf=chunk_kda(q=q,k=k,v=v,g=g,beta=beta,scale=scale,initial_state=h0,
            output_final_state=True,use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,A_log=A_log,dt_bias=dtf,lower_bound=float(lb),
            transpose_state_layout=True,cu_seqlens=None)
        ff=finite(of) and finite(hf)
    except Exception as e:
        ff=f"ERR {e}"
    # ref fp32
    orf,_=_kda_chunk_torch_fwd(q,k,v,g,beta,scale,A_log,dt_bias,float(lb),h0,True,None)
    rf=finite(orf)
    print(f"  lb={lb}: kernel={kf}  fla={ff}  torch_fp32ref={rf}")
print("\nlb=-8 with milder inputs (gx=1,dtx=0.5):")
for lb in [-8]:
    q,k,v,g,beta,A_log,dt_bias,h0=mk(gx=1.0,dtx=0.5)
    o,ht=flash_kda.flash_kda_func(q,k,v,g,beta,scale,A_log,dt_bias,float(lb),
                                  initial_state=h0,output_final_state=True)
    print(f"  lb={lb} mild: kernel finite={finite(o) and finite(ht)}")
