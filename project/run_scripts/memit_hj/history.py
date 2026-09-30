"""Current-key sample/drift and occurrence-preserving FP64 reconstruction."""
import copy,time
import torch
from .plan import sample_batch,digest
from .state import rng,rng_hash,restore_rng
from project.run_scripts.memit_history_lifelong.io import tensor_sha
class History:
 def __init__(self,module,model,tok,hp,H,rows):self.m,self.model,self.tok,self.hp,self.H,self.rows=module,model,tok,hp,H,rows
 def keys(self,ordinals,layer,context=False):
  reqs=[self.rows[i]['requested_rewrite'] for i in ordinals];ctx=self.m.CONTEXT_TEMPLATES_CACHE
  if not context:return self.m.compute_ks(self.model,self.tok,reqs,self.hp,layer,ctx).T
  raw=self.m.get_module_input_output_at_words(self.model,self.tok,layer,
   context_templates=[c.format(r['prompt']) for r in reqs for g in ctx for c in g],words=[r['subject'] for r in reqs for g in ctx for c in g],
   module_template=self.hp.rewrite_module_tmp,fact_token_strategy=self.hp.fact_token)[0]
  raw=raw.reshape(len(reqs),sum(map(len,ctx)),-1);means=[];j=0
  weights=[]
  for group in ctx:means.append(raw[:,j:j+len(group)].mean(1));j+=len(group);weights.extend([1/len(ctx)/len(group)]*len(group))
  bar=torch.stack(means,0).mean(0)
  # Preserve BOTH pinned definitions: native key is mean of group means;
  # the authoritative drift.py dispersion is an equal-context RMS around
  # its own context mean. Do not silently reweight the trigger threshold.
  center=raw.mean(1)
  spread=((raw-center[:,None]).square().sum(-1).mean(1)).sqrt()
  dispersion=spread/center.norm(dim=1)
  return bar.T,dispersion
 def record(self,meta,start,stop):
  ids=sample_batch(self.rows[start:stop],start);samples=meta.setdefault('samples',[])
  by={}
  for l in range(4,9):
   k,d=self.keys(ids,l,True);by[l]=(k.detach().cpu(),d.detach().cpu())
  for j,ordinal in enumerate(ids):samples.append(dict(ordinal=ordinal,birth=stop,origin={l:by[l][0][:,j].clone() for l in by},reference={l:by[l][0][:,j].clone() for l in by},dispersion={l:float(by[l][1][j]) if torch.isfinite(by[l][1][j]) else None for l in by}))
 def trigger(self,meta):
  ss=meta.get('samples',[]);ids=[x['ordinal'] for x in ss];rec={};layers=[]
  if not ids:return [],rec
  for l in range(4,9):
   chunks=[self.keys(ids[j:j+100],l).cpu() for j in range(0,len(ids),100)];current=torch.cat(chunks,1)
   reference=torch.stack([x['reference'][l] for x in ss],1);norm=reference.norm(dim=0);drift=(current-reference).norm(dim=0)/norm
   valid=[i for i in range(len(ss)) if float(norm[i])>0 and ss[i]['dispersion'][l] is not None and torch.isfinite(drift[i])]
   if valid:
    ds=drift[valid].double();disp=torch.tensor([ss[i]['dispersion'][l] for i in valid],dtype=torch.float64)
    # quantile .5 interpolates even populations (torch.median does not).
    m=float(torch.quantile(ds,.5));d=float(torch.quantile(disp,.5));fired=l>4 and m>d
    rec[str(l)]=dict(drift_median=m,dispersion_median=d,drift_p90=float(torch.quantile(ds,.9)),drift_p95=float(torch.quantile(ds,.95)),NA=len(ss)-len(valid),fired=fired)
    if fired:layers.append(l)
   else:rec[str(l)]=dict(status='TRIGGER_UNDEFINED',NA=len(ss),fired=False)
  return layers,rec
 def rebuild(self,meta,layers):
  start=time.monotonic();ids=meta['ledger'];assert ids==list(range(meta['cursor']))
  receipt=dict(membership=digest(ids),occurrences=len(ids),layers=[],seconds=None)
  for l in layers:
   h=torch.zeros(self.H.shape[1:],dtype=torch.float64,device='cuda');gen=torch.Generator(device='cpu');gen.manual_seed(20260930+l)
   u=torch.randn((h.shape[0],4),generator=gen,dtype=torch.float64).cuda();reference=torch.zeros(4,dtype=torch.float64,device='cuda')
   for j in range(0,len(ids),100):
    k=self.keys(ids[j:j+100],l).double();h.add_(k@k.T);reference+=((k.T@u)**2).sum(0)
   native=h.to(dtype=self.H.dtype);observed=(u*(native.double()@u)).sum(0)
   error=float(((observed-reference).abs()/reference.abs().clamp_min(torch.finfo(torch.float64).tiny)).max())
   if error>1e-5:raise ArithmeticError('HISTORY_REBUILD_PROBE')
   self.H[l-4].copy_(native.cpu());receipt['layers'].append(dict(layer=l,probe_relative_error=error,sha256=tensor_sha(self.H[l-4])))
   del native,h,u
   ss=meta.get('samples',[])
   for j in range(0,len(ss),100):
    group=ss[j:j+100];k,d=self.keys([s['ordinal'] for s in group],l,True)
    for i,s in enumerate(group):s['reference'][l]=k[:,i].detach().cpu().clone();s['dispersion'][l]=float(d[i]) if torch.isfinite(d[i]) else None
  receipt['seconds']=time.monotonic()-start;return receipt
