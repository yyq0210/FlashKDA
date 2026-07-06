"""K2 grid-starvation test: sweep N (grid_k2=N*H), see if K2 us/iter per-block-work drops as grid fills SMs."""
import math, torch, torch.nn.functional as F, flash_kda
from torch.profiler import profile, ProfilerActivity

def make(seq_lens, H, D):
    dev=torch.device("cuda"); T=sum(seq_lens); N=len(seq_lens)
    q=F.normalize(torch.randn(1,T,H,D,device=dev),p=2,dim=-1).bfloat16()
    k=F.normalize(torch.randn(1,T,H,D,device=dev),p=2,dim=-1).bfloat16()
    v=torch.randn(1,T,H,D,device=dev,dtype=torch.bfloat16)
    g=torch.randn(1,T,H,D,device=dev,dtype=torch.bfloat16)
    beta=torch.randn(1,T,H,device=dev,dtype=torch.bfloat16)
    A_log=torch.rand(H,device=dev); dt_bias=torch.rand(H,D,device=dev)
    h0=torch.randn(N,H,D,D,device=dev,dtype=torch.bfloat16); fs=torch.zeros_like(h0)
    out=torch.zeros_like(q); extra={}
    if N>1:
        cu=torch.tensor([0]+torch.cumsum(torch.tensor(seq_lens),0).tolist(),dtype=torch.long,device=dev)
        extra["cu_seqlens"]=cu
    scale=1.0/math.sqrt(D)
    def run(): flash_kda.fwd(q,k,v,g,beta,scale,out,A_log=A_log,dt_bias=dt_bias,lower_bound=-5.0,initial_state=h0,final_state=fs,**extra)
    return run

def k2_us(seq_lens,H,D,iters=50):
    run=make(seq_lens,H,D)
    for _ in range(30): run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters): run()
        torch.cuda.synchronize()
    k2=0.0
    for e in prof.key_averages():
        if e.self_device_time_total>0 and "recurrence" in e.key: k2+=e.self_device_time_total
    return k2/iters

H,D=64,128; Tseq=2048
print(f"H={H} D={D} T_seq={Tseq}  H20 ~78 SMs; grid_k2=N*H; each block runs Tseq/16={Tseq//16} serial chunks")
print(f"{'N':>4} {'grid=N*H':>9} {'K2 us/it':>9} {'us/N (per-block-cost)':>22}")
for N in [1,2,4,8,16]:
    us=k2_us([Tseq]*N,H,D)
    print(f"{N:>4} {N*H:>9} {us:>9.2f} {us/N:>22.3f}")
