"""Twelve prespecified shape/kernel cases: warmup1 + measured5, no sweep."""
import time
import torch
import torch.nn.functional as F
from .measurement import timing

PAIRS=(('linear_duplicate','linear_single'),('gradient_dense','gradient_direct'),('head_full','head_selected'))

def comparison(B,reference_kernel,candidate_kernel,reference_times,candidate_times):
    """Labels have their own namespace; timing reference/candidate stay dicts."""
    return dict(B=B,reference_kernel=reference_kernel,candidate_kernel=candidate_kernel,
                **timing(reference_times,candidate_times))

def assemble_summary(rows):
    comparisons=[]
    for B in (4,100):
        local={r['kernel']:r['cuda_seconds'] for r in rows if r['B']==B}
        if len(local)!=6 or any(len(v)!=5 for v in local.values()):raise ValueError('KERNEL_CASE_INVENTORY')
        for a,b in PAIRS:comparisons.append(comparison(B,a,b,local[a],local[b]))
    if len(rows)!=12:raise ValueError('KERNEL_CASE_COUNT')
    return dict(schema=2,cases=rows,comparisons=comparisons,cases_count=len(rows),warmups=12,measurements=60,max_allowed_cases=24)

def filter_direct_routes(options,summary):
    direct=[k for k in summary['comparisons'] if k['candidate_kernel']=='gradient_direct']
    allowed=len(direct)==2 and {k['B'] for k in direct}=={4,100} and all(k['status']=='TIMING_SEPARATED' for k in direct)
    return [r for r in options if 'DIRECT' not in r or allowed]

def run(model,width4,width100):
    device=next(model.parameters()).device;I=model.config.intermediate_size;O=model.config.hidden_size;V=model.config.vocab_size
    gen=torch.Generator(device=device).manual_seed(20261001);rows=[]
    for B,L in ((4,width4),(100,width100)):
        T=2*L
        x=torch.randn(T,I,device=device,generator=gen)*.01;g=torch.randn(T,O,device=device,generator=gen)*.01
        q=torch.randn(I,B,device=device,dtype=torch.float64,generator=gen)*.01
        w=torch.randn(O,I,device=device,generator=gen)*.01
        h=torch.randn(T,O,device=device,generator=gen)*.01;head=torch.randn(V,O,device=device,generator=gen)*.01
        cases={
          'linear_duplicate':lambda:(F.linear(x,w),F.linear(x,w)),
          'linear_single':lambda:F.linear(x,w),
          'gradient_dense':lambda:(g.T@x).double()@q,
          'gradient_direct':lambda:g.double().T@(x.double()@q),
          'head_full':lambda:F.linear(h,head),
          'head_selected':lambda:F.linear(h[:2],head)}
        local={}
        for name,fn in cases.items():
            times=[];wall=[]
            for repeat in range(6):
                torch.cuda.synchronize();t=time.monotonic();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();out=fn();b.record();torch.cuda.synchronize()
                seconds=time.monotonic()-t
                if repeat:times.append(a.elapsed_time(b)/1000);wall.append(seconds)
                else:warm=seconds
                del out
            row=dict(B=B,T=T,I=I,O=O,V=V,kernel=name,warmup_seconds=warm,cuda_seconds=times,wall_seconds=wall,
                     synthetic_model_free_tensors=True,logical_oracles=0)
            rows.append(row);local[name]=times
        del x,g,q,w,h,head
    return assemble_summary(rows)
