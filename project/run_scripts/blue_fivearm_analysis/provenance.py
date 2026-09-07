"""Source/config differences and immutable exclusion registry; no runtime imports."""
import argparse
import subprocess
from .common import *

def run(repo):
 out=repo/OUT_REL;rows=csvread(out/'blue-source-config-compatibility.csv');ref=repo/REF_REL
 old=read(ref/'assets.lock.json')['models']['llama3-8b-inst']
 local=read(JROOT/'assets.lock.json')['models']['llama3-8b-inst']
 assert old['revision']==local['revision']
 for kind in ['projector']:
  assert old[kind]['expected_sha256']==local[kind]['expected_sha256']
 for arm,fn in [('O_NATIVE','chain-0-llama3-8b-inst-O_NATIVE.runtime.lock.json'),('JVP','chain-2-llama3-8b-inst-JV_NATIVE.runtime.lock.json'),('JVP_L8',None)]:
  rt=read(ref/fn) if fn else read(JCHAIN/'runtime.lock.json')
  head='44602a1a80554c67da0ef9646b43d843104785f2' if arm=='JVP_L8' else '77358b1546d1baf83b3e251afcce663b08d7bfd7'
  hpblob=subprocess.check_output(['git','-C',str(repo),'show',head+':project/run_scripts/fixed_z_nonuniqueness/config/alphaedit-llama3-8b.yaml'])
  assert hashlib.sha256(hpblob).hexdigest()==old['hparams']['AlphaEdit']['sha256']
  # Flat pinned YAML values inspected without importing writer configuration code.
  hp={k.strip():v.strip() for line in hpblob.decode().splitlines() if ':' in line for k,v in [line.split(':',1)]}
  rows.append(dict(arm=arm,source_head=head,source_tree=subprocess.check_output(['git','-C',str(repo),'rev-parse',head+'^{tree}'],text=True).strip(),
   blue_head='NOT_USED',easyedit_head='14cea8245f06715684592ab55184939b99d70784',easyedit_tree='9c52aadbc0883da422badf0a730fff21aaa3a8a7',
   model_revision=old['revision'],sample_root=SAMPLE_ROOT,config_sha=old['hparams']['AlphaEdit']['sha256'],layers='[8] support / [4,5,6,7,8] dictionary/history' if arm=='JVP_L8' else '[4,5,6,7,8]',
   target_policy='fixed batch-entry L8 z, once/request, no inner recompute',z_decay=hp['v_weight_decay'],L2=hp['L2'],v_num_grad_steps=hp['v_num_grad_steps'],v_loss_layer=hp['v_loss_layer'],
   context_hash=rt['contexts_sha256'],seed=20260906,projector_file_sha=old['projector']['expected_sha256'],dtype='model FP32; controller/native metric FP64',
   gpu=rt['cuda_device'],torch=NA,transformers=NA,attention='eager',tf32_matmul=False,tf32_cudnn=False,autocast=False,
   writer_tokenizer='right; default BOS; pad=eos if unset',evaluator_tokenizer='same tokenizer as writer',
   T=2 if arm!='O_NATIVE' else NA,N=4 if arm!='O_NATIVE' else NA,h=.5 if arm!='O_NATIVE' else NA,lambda_value=.1 if arm!='O_NATIVE' else NA,
   normalization='SOURCE_EXACT_N0' if arm!='O_NATIVE' else 'NONE',source_scope='local rehashed raw+publication' if arm=='JVP_L8' else 'Git publication rehash; remote raw NOT_AVAILABLE',
   job=rt['slurm_job'],hparams=json.dumps(hp,sort_keys=True)))
 csvwrite(out/'source-config-compatibility.csv',rows)
 findings=[]
 roots={a:read(r.parent/'execution.lock.json') for a,r in BLUE_ROOTS.items()}
 paths={Path(roots['BLUE']['blue_root'])/'AlphaEdit/AlphaEdit_main.py':['if hparams.blue:','z_layer = layer','resid = targets  #','torch.linalg.solve(','cache_c[i,:,:] +='],
  Path(roots['BLUE']['blue_root'])/'AlphaEdit/compute_z.py':['loss_layer = max','v_weight_decay','loss.item()'],
  BLUE_ROOTS['BLUE_L4_ONLY'].parent/'l4_adapter/runtime.py':['projector = allp[[0]]','assert signature(weights, cache) == endpoint','prior=content(endpoint)'],
  BLUE_ROOTS['BLUE_L4_ONLY'].parent/'l4_adapter/observer.py':['projector_asset_index=4','nonedited_full_bytes_checked=full_bytes'],
  BASE/'local/worktrees/alpha-jv-l8-takeover-v1/project/run_scripts/alpha_native_response_ode_v31_sequential/trajectory.py':['for node in range(N)','if arm==\'L8_ONLY_NATIVE\'','dictionary.whiten','nnls_response','H*c[i]/q[i].sqrt()','family.finalize'],
  BASE/'local/worktrees/alpha-jv-l8-takeover-v1/project/run_scripts/alpha_native_response_ode_v31_sequential/runtime.py':['f.compute_fixed_z()','hp,hppath=','manual_seed','write_compute[\'history_key_captures\']']}
 for p,terms in paths.items():
  body=p.read_text().splitlines()
  for term in terms:
   found=[(i,line) for i,line in enumerate(body,1) if term in line]
   for i,line in found:findings.append(dict(path=str(p),sha256=sha(p),line=i,source_expression=line.strip(),topic=term))
 csvwrite(out/'source-findings.csv',findings)
 exclude=[]
 for job,attempt in [(38929,'execution-original-v2'),(38932,'execution-tech-r1')]:
  p=BASE/'local/blue-alphaedit-sequential-comparison/attempt-v1'/attempt/'smoke-llama/failure.json';d=read(p)
  exclude.append(dict(job=job,classification='PURE_TECHNICAL_PRE_EDIT',denominator=d['final_denominator'],prior_batches=d['prior_committed_batches'],stage=d['stage'],error=d['error'],process_seconds=d['seconds'],path=str(p),sha256=sha(p),scheduler_gpu_elapsed='NOT_REQUERIED'))
 csvwrite(out/'blue-technical-exclusions.csv',exclude)
 # Validate source/sample/evaluator member identity and inspect available W0 metrics, not model parity claims.
 pre=csvread(out/'preedit.csv');cmp=[]
 ref0=next(x for x in pre if x['arm']=='PRE_EDIT_ORIGINAL_W0')
 for x in pre:
  if x['arm'] in BLUE_ROOTS:
   cmp.append(dict(arm=x['arm'],W0_RS=x['RS_num'],W0_PS=x['PS_num'],W0_NS=x['NS_num'],
    reference_RS=ref0['RS_num'],reference_PS=ref0['PS_num'],reference_NS=ref0['NS_num'],
    statement='SAME_MODEL_ASSET_AND_SAMPLE; observed score parity separately, no cross-host bitwise guarantee'))
 csvwrite(out/'W0-reference-comparison.csv',cmp)
 save(out/'source-audit-notes.json',dict(sample_root=SAMPLE_ROOT,raw_data_sha=read(ref/'sample.lock.json')['dataset_sha256'],
  Llama_decay_correction='Actual pinned Llama JVP YAML v_weight_decay=0.5, not0.001; all BLUE variants also0.5. Qwen config0.001 not used here.',
  L4_metadata='observer constant projector_asset_index=4 erroneous; runtime/index0 and selected P tensor SHA proved by CPU audit; no source/raw repair',
  context_difference='BLUE native first-batch generated contexts and seed20260907 vs JVP fixed native contexts seed20260906; hashes not equal',
  raw_broadcast='NOT_PERFORMED_LOCAL_ONLY; Git contains raw-free package only',scientific_promotion=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);run(p.parse_args().repo)
