"""Read-only CPU inspection of native saved weights and history prior."""
import argparse, hashlib, json, time, sys
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
sys.path.insert(0,'/mnt/raid5/janghj/ODE-edit/local/worktrees/fe-collapse-repair-20261011')
from official.diagnostics.fe_native.evaluate import validate_payload
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--batch',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--skip-eigenvalues',action='store_true');args=p.parse_args()
torch.set_num_threads(8)
c=json.loads(Path(args.config).read_text());cp=Path(c['output'])/'checkpoints'/f'B{args.batch:02d}.pt';receipt=json.loads(cp.with_suffix('.json').read_text());rows=json.loads(Path(c['original_stream']['path']).read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for s in iter(lambda:f.read(8*1024**2),b''):h.update(s)
 return h.hexdigest()
assert sha(cp)==receipt['sha256']
a=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);validate_payload(a,c,rows,args.batch)
hp=a['config']['llms'];layers=hp['layers'];H=a['history'];assert H.dtype==torch.float32 and len(H)==len(layers)
result=dict(job_source=c['ode_patch_source'],model=c['model'],checkpoint=receipt,identity='PASS',CPU_only=True,scope='Saved C0 plus rounded history; actual next-batch solve also adds its current key Gram',layers={})
for i,layer in enumerate(layers):
 started=time.time();h=H[i];w=a['weights'][hp['rewrite_module_tmp'].format(layer)+'.weight']
 assert torch.isfinite(w).all() and torch.isfinite(h).all();assert torch.equal(h,h.T)
 # Inspect the actual execution asset, including an explicitly derived C0.
 entry=c['covariances'][i]
 if 'execution' in entry:src=entry['execution']
 else:
  assert not entry['repair'];src=entry['original']
 assert Path(entry['link']).resolve()==Path(src['path']).resolve()
 assert sha(Path(src['path']))==src['sha256']
 with np.load(src['path']) as z:C=(torch.from_numpy(z['mom2.mom2'].copy())/int(z['mom2.count'])).double()
 A=C*hp['mom2_update_weight'][i]+h.double();del C
 with threadpool_limits(limits=8):L,info=torch.linalg.cholesky_ex(A)
 del L
 row=dict(weight_norm=float(w.double().norm()),history_trace=float(h.diagonal().double().sum()),history_frobenius=float(h.double().norm()),history_symmetric=True,history_finite=True,weights_finite=True,prior_cholesky_info=int(info),covariance=src,seconds=time.time()-started)
 if int(info)!=0 and not args.skip_eigenvalues:
  from scipy.linalg import eigh
  with threadpool_limits(limits=8):ev=eigh(A.numpy(),eigvals_only=True,subset_by_index=(0,1),check_finite=False,driver='evr')
  row['prior_smallest_eigenvalues']=ev.tolist()
 result['layers'][str(layer)]=row;del A
 print(layer,json.dumps(row),flush=True);Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
assert sha(cp)==receipt['sha256'];result['checkpoint_unchanged']=True;Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
