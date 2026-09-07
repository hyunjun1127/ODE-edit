"""Read-only CPU provenance, tensor and sequential-chain validation."""
import argparse
import gc
import re
import subprocess
from .common import *

def tsha(t):
 import torch
 x=t.detach().contiguous().cpu();h=hashlib.sha256(str((str(x.dtype),list(x.shape))).encode())
 a=x.view(torch.uint8).numpy().reshape(-1)
 for i in range(0,a.size,8<<20):h.update(memoryview(a[i:i+(8<<20)]))
 return h.hexdigest()
def content(s):return dict(weights={k:v['sha256'] for k,v in s['weights'].items()},cache=s['cache_sha256'])

def run(repo):
 import torch
 torch.set_num_threads(4)
 out=repo/OUT_REL; ref=repo/REF_REL
 sample=read(ref/'sample.lock.json');assert digest(sample['records'])==SAMPLE_ROOT
 ids=[r['case_id'] for r in sample['records']]
 byid={r['case_id']:r for r in read('/data/janghj/EasyEdit/data/counterfact/counterfact.json')}
 for i,r in enumerate(sample['records']):
  assert (r['ordinal'],r['batch_index'],r['batch_ordinal'])==(i,i//100+1,i%100)
  assert digest(byid[r['case_id']])==r['raw_record_sha256']
  assert digest(byid[r['case_id']]['requested_rewrite'])==r['request_sha256']
 locks={a:read(r.parent/'execution.lock.json') for a,r in BLUE_ROOTS.items()}
 members={m['path']:m for l in locks.values() for m in l['members']}
 # Source/asset files may be HF symlinks; compare actual bytes, retaining lexical path.
 inv=[]
 for p,m in members.items():
  assert sha(p)==m['sha256'] and Path(p).stat().st_size==m['bytes'],p
  inv.append(dict(path=p,bytes=m['bytes'],sha256=m['sha256'],symlink=Path(p).is_symlink(),status='FULL_REHASH_PASS'))
 csvwrite(out/'source-asset-inventory.csv',inv)
 allp=torch.load(locks['BLUE']['projector'],map_location='cpu',weights_only=True)
 audits=[];cp=[];compat=[]
 for arm,root in BLUE_ROOTS.items():
  l=locks[arm];rt=read(root/'runtime.json');t=read(root/'terminal.json');layers=l['hparams']['layers']
  assert sha(root.parent/'execution.lock.json')==rt['lock_sha256']
  assert digest(read(l['sample'])['records'])==SAMPLE_ROOT
  indices=[x-4 for x in layers]
  psha=tsha(allp[indices]);assert psha==rt['projector_sha256'],(arm,'P selection')
  assert rt['hparams']==l['hparams'] and rt['projector_layers']==layers
  prior=content(rt['W0']);cases=[]
  for b in range(1,11):
   d=root/f'B{b:02d}';entry=read(d/'entry.json');commit=read(d/'commit.json');obs=read(d/'native-observation.json')
   assert entry['request_ids']==ids[(b-1)*100:b*100]
   assert entry['request_hashes']==[digest(byid[i]['requested_rewrite']) for i in entry['request_ids']]
   assert content(entry['signature'])==prior==commit['entry']
   assert commit['status']=='BATCH_COMMITTED' and commit['nonfinite']==commit['evaluator_mutation']==0
   assert (commit['history_entries_in'],commit['history_entries_out'],commit['history_append_passes'])==((b-1)*100,b*100,1)
   assert obs['compute_z']==commit['compute_z']==100*len(layers)
   assert [(z['case_id'],z['layer']) for z in obs['z']]==[(i,layer) for layer in layers for i in entry['request_ids']]
   target=torch.load(d/'native-layer-targets.pt',map_location='cpu',weights_only=True)
   assert [tsha(v) for v in target['values']]==[z['sha256'] for z in obs['z']]
   del target
   assert set(commit['layer_updates'])=={f'model.layers.{x}.mlp.down_proj.weight' for x in layers}
   for name in ['current.json','seen-rewrite.json']+(['seen-full.json'] if b in [1,5,10] else []):
    ev=read(d/name);state=ev.get('state',dict(weights=ev.get('weight_state'),cache=ev.get('cache_sha256')))
    assert state==commit['endpoint'],(arm,b,name,'endpoint')
    if name=='current.json':assert ev['before_after_exact'] and ev['evaluator_controller_influence']==0
   if b in [1,5,10]:
    v=torch.load(d/'W-M.pt',map_location='cpu',weights_only=True)
    assert {k:tsha(x) for k,x in v['weights'].items()}==commit['endpoint']['weights']
    assert tsha(v['cache_c'])==commit['endpoint']['cache']
    assert v['metadata']['state']==commit['endpoint'] and v['metadata']['request_ids']==ids[:100*b]
    assert all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in v['weights'].values())
    assert v['cache_c'].dtype==torch.float32 and torch.isfinite(v['cache_c']).all()
    cp.append(dict(arm=arm,batch=b,path=str(d/'W-M.pt'),sha256=sha(d/'W-M.pt'),weights=len(v['weights']),cache_shape=str(list(v['cache_c'].shape)),restore_scope='selected W + dense M + pinned base model; no full-model snapshot',status='CPU_RELOAD_HASH_PASS'))
    del v;gc.collect()
   audits.append(dict(arm=arm,batch=b,W_M_chain=True,history_append=1,compute_z=obs['compute_z'],
    solve_count=obs.get('solve_calls','NOT_RECORDED_COUNTER_SOURCE_HAS_ONE_PER_LAYER'),expected_native_solves=len(layers),
    key_calls=len(obs['keys']),eval_endpoint_exact=True,nonfinite=0,request_count=100,
    nonedited_identity_exact=obs.get('nonedited_identity_exact','SOURCE_SELECTED_WEIGHTS_ONLY'),
    nonedited_full_bytes=obs.get('nonedited_full_bytes_checked','NOT_RECORDED'),
    recorded_P_index=obs.get('projector_asset_index','NOT_RECORDED'),actual_P_indices=str(indices),P_tensor_hash=psha,
    P_metadata_status='METADATA_ERROR_ONLY_ACTUAL_P0_HASH_MATCH' if arm=='BLUE_L4_ONLY' else 'CONSISTENT',
    pre_gate='SKIPPED_USER_DIRECTED' if arm=='BLUE_L4_ONLY' else 'PRIOR_SMOKE_RECEIPT'))
   cases+=entry['request_ids'];prior=commit['endpoint']
  assert cases==ids and len(set(cases))==1000
  assert t['W0_cache_bytes_pointer_restore'] and t['W_chain_links']==9 and t['cold_reset_count']==1
  c={k:rt.get(k,NA) for k in ['source_head','source_tree','blue_head','model_revision','torch','transformers','gpu','dtype','attention','tf32_matmul','tf32_cudnn','autocast','parameter_elements']}
  c.update(arm=arm,job=rt['slurm_job'],sample_root=SAMPLE_ROOT,config_sha=sha(l['config']),layers=str(layers),
    target_policy='native selected-layer current W; one z/request/layer',seed=l['seed'],context_hash=read(root/'B10/commit.json')['context_hash'],
    contexts_file_sha=sha(root/'B10/contexts.json'),projector_file_sha=sha(l['projector']),projector_selected_tensor_sha=psha,
    writer_tokenizer=json.dumps(rt['writer_tokenizer']),evaluator_tokenizer=json.dumps(rt['evaluator_tokenizer']),
    hparams=json.dumps(l['hparams'],sort_keys=True),lock_path=str(root.parent/'execution.lock.json'),lock_sha=sha(root.parent/'execution.lock.json'),
    archive_path=l.get('source_archive',{}).get('path','SOURCE_GIT_COMMIT'),archive_sha=l.get('source_archive',{}).get('sha256',l['source_tree']),
    z_decay=l['hparams']['v_weight_decay'],L2=l['hparams']['L2'],v_loss_layer=l['hparams']['v_loss_layer'],v_num_grad_steps=l['hparams']['v_num_grad_steps'],
    source_scope='BLUE source + inherited helper + local archive for single-layer; no tracked hook migration')
  if 'source_archive' in l:assert sha(l['source_archive']['path'])==l['source_archive']['sha256']
  compat.append(c)
 del allp
 csvwrite(out/'batch-integrity.csv',audits);csvwrite(out/'checkpoint-integrity.csv',cp);csvwrite(out/'blue-source-config-compatibility.csv',compat)
 save(out/'integrity-receipt.json',dict(status='POSTRUN_INTEGRITY_WITH_METADATA_WARNING',blue_terminal=3,batches=30,requests=3000,W_M_links=27,checkpoints=9,source_asset_members=len(inv),sample_root=SAMPLE_ROOT,
  projector_mapping='CPU reconstructed selected-P hashes match runtime all three arms; L4 actual index0, observer label4 is incorrect metadata only',
  L4_pre_gate='SKIPPED_USER_DIRECTED',model_load=0,GPU=0,evaluator=0,raw_mutation=0,version_restore='NOT_CLAIMED_INPLACE_COPY_INCREMENTS_VERSION',
  nonedited_scope='single-layer main pointer/version/shape/dtype; NOT full byte comparison; original BLUE selected-write source audit only'))
 print('CPU_AUDIT_COMPLETE',len(inv),len(audits),len(cp),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);run(p.parse_args().repo)
