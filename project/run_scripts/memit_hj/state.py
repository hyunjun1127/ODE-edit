"""Explicit CPU clones, exact restoration and task-owned rolling temporary CP."""
import copy,os,random,time
from pathlib import Path
import numpy as np
import torch
from project.run_scripts.memit_history_lifelong.io import save,file_sha,tensor_sha,digest,signature,content

def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(r):random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['cpu']);torch.cuda.set_rng_state_all(r['cuda'])
def rng_hash(r):return digest(dict(python=r['python'],numpy=(r['numpy'][0],r['numpy'][1].tolist(),*r['numpy'][2:]),cpu=tensor_sha(r['cpu']),cuda=[tensor_sha(x) for x in r['cuda']]))
def clone(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:clone(v) for k,v in x.items()}
 if isinstance(x,list):return [clone(v) for v in x]
 if isinstance(x,tuple):return tuple(clone(v) for v in x)
 return copy.deepcopy(x)
def metadata_hash(x):
 if isinstance(x,torch.Tensor):return dict(sha256=tensor_sha(x),dtype=str(x.dtype),shape=list(x.shape))
 if isinstance(x,dict):return {str(k):metadata_hash(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [metadata_hash(v) for v in x]
 return x
class States:
 def __init__(self,weights,H,module,model,tok,rows):self.weights,self.H,self.module,self.model,self.tok,self.rows=weights,H,module,model,tok,rows
 def identity(self,meta):return dict(state=content(signature(self.weights,self.H)),rng=rng_hash(rng()),contexts=digest(self.module.CONTEXT_TEMPLATES_CACHE),ledger=digest(meta.get('ledger',[])),metadata=digest(metadata_hash(meta)))
 def snapshot(self,meta):
  torch.cuda.synchronize()
  return dict(weights=clone(self.weights),H=clone(self.H),rng=clone(rng()),contexts=copy.deepcopy(self.module.CONTEXT_TEMPLATES_CACHE),meta=clone(meta),identity=self.identity(meta),cache_binding={str(k):tensor_sha(v) for k,v in self.module.COV_CACHE.items()})
 def restore(self,s):
  with torch.no_grad():
   for k,w in self.weights.items():w.copy_(s['weights'][k].to(w.device))
   self.H.copy_(s['H'])
  restore_rng(s['rng']);self.module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(s['contexts'])
  m=clone(s['meta']);assert self.identity(m)==s['identity'],'RAM_RESTORE_IDENTITY'
  for k,v in self.module.COV_CACHE.items():
   if str(k) in s['cache_binding']:assert tensor_sha(v)==s['cache_binding'][str(k)]
  return m
 def probe(self):
  token=self.tok(self.rows[0]['requested_rewrite']['prompt'].format(self.rows[0]['requested_rewrite']['subject']),return_tensors='pt').to('cuda')
  with torch.no_grad():y=self.model(**token,use_cache=False).logits.detach()
  return tensor_sha(y)
class Checkpoints:
 def __init__(self,root,states,bindings):
  self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True);self.states,self.bindings=states,bindings;self.paths={};self.counter=0;self.event_counter=0
 def write(self,arm,cursor,meta,technical=False):
  if arm not in ['000','100','110','111']:raise ValueError('CP_NOT_AUTHORIZED_ARM')
  if not technical:assert cursor%1000==0
  start=time.monotonic();s=self.states.snapshot(meta);s['bindings']=self.bindings
  path=self.root/f'{arm}-{cursor:05d}-{self.counter:04d}.pt';self.counter+=1
  tmp=path.with_suffix('.partial');assert not path.exists() and not tmp.exists()
  before=self.states.probe()
  with tmp.open('xb') as f:torch.save(s,f);f.flush();os.fsync(f.fileno())
  os.link(tmp,path);tmp.unlink();sha=file_sha(path)
  loaded=torch.load(path,map_location='cpu',weights_only=False);assert loaded['bindings']==self.bindings
  self.states.restore(loaded);assert self.states.probe()==before,'CP_OUTPUT_PARITY'
  rec=dict(path=str(path),owner_uid=os.getuid(),bytes=path.stat().st_size,sha256=sha,arm=arm,cursor=cursor,refcount=0,reload_state_output=True,technical=technical,seconds=time.monotonic()-start)
  save(path.with_suffix('.manifest.json'),rec);self.paths[str(path)]=rec;del s,loaded
  same=[r for r in self.paths.values() if r['arm']==arm and not r.get('deleted')]
  for old in same[:-2]:
   if old['refcount']==0:self.delete(old['path'],'rolling_replaced_validated')
  return rec
 def delete(self,path,reason):
  p=Path(path).resolve();rec=self.paths[str(p)]
  assert p.parent==self.root and rec['refcount']==0 and p.stat().st_uid==rec['owner_uid'] and file_sha(p)==rec['sha256']
  save(p.with_suffix('.tombstone.json'),dict(**rec,reason=reason,permanent_deletion=True,recoverable=False))
  p.unlink();rec['deleted']=True
 def pin(self,path,delta,reason):
  rec=self.paths[str(Path(path).resolve())];assert delta in [-1,1] and not rec.get('deleted')
  rec['refcount']+=delta;assert rec['refcount']>=0
  save(self.root/f'refcount-{self.event_counter:05d}.json',dict(path=path,refcount=rec['refcount'],reason=reason));self.event_counter+=1
 def load(self,path,sha):
  p=Path(path).resolve();assert p.parent==self.root and file_sha(p)==sha
  s=torch.load(p,map_location='cpu',weights_only=False);assert s['bindings']==self.bindings
  meta=self.states.restore(s);return meta
 def complete(self,arm):
  for rec in list(self.paths.values()):
   if rec['arm']==arm and not rec.get('deleted') and not rec['refcount']:self.delete(rec['path'],'terminal_evaluation_manifest_dependencies_verified')
