"""Create-once local preparation for the user-approved four new Middle chains."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from .runtime import file_sha, save
from .sequential_prepare import member

ROOT = Path('/data/janghj/ODE-edit')
OLD = ROOT/'local/low-cost-write-donor-seq10/20260913-v1/attempt-v1'
POLICIES = ['FROZEN2', 'I2', 'FROZEN4', 'I4']
DOCS = {
 'project/proposals/2026-09-14-refit4-write-refresh-seq1000-gh-instruction.md': '55c8eb6ca5730b197b83852ed65f20d00519bf89247ec2a5afd7894a75d99f54',
 'plans/global/2026-09-14-refit4-write-refresh-seq1000-final-design.md': '0763b390163e50b6ce711743ced304c6ac035198d69f7d2afffdaa446d45af6e',
 'plans/global/2026-09-14-refit4-write-refresh-seq1000-cells.csv': 'a13bffa3f7d86e1c90cb20eff77d5184199a8bf591535f35088bcba1b139e0cb',
 'plans/global/2026-09-14-refit4-write-refresh-seq1000-contract.json': '5e47ec6f5d821c260bb2b1020f6786b61caea20a08a32fdd2a476fa88ee898fe',
 'audits/global/2026-09-14-lowcost-seq10-review-ko.md': '3494439c5b0463e71bbc90fea03939f3b929614f46394dd38a7cfcdba7339d59',
 'plans/global/2026-09-14-refit4-write-refresh-design-checks.json': 'baf019ba5c8500ac78abd39be2330d10922844bcb01b41229a985048629631f5',
 'messages/head/2026-09-14-sh4-refit4-write-refresh-seq1000.md': None,
 'PROTOCOL.md': None,
}

def prepare(attempt):
 import torch
 from scripts.fixed_counterfact import load_prefix
 from project.run_scripts.baseline_mechanism_first.contracts import digest
 from project.run_scripts.baseline_mechanism_first.fixtures import tensor_sha
 from .fitting import select_projector
 torch.set_num_threads(8)
 a=Path(attempt); a.mkdir(parents=True,mode=0o700,exist_ok=False)
 w=Path(__file__).resolve().parents[3]
 docs=[]
 for rel,sha in DOCS.items():
  ref=member(w/rel,sha);dest=a/'instructions'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('xb') as f:f.write((w/rel).read_bytes())
  docs.append(dict(source=ref,preserved=member(dest)))
 original_lock=member(OLD/'execution.lock.json','695a2d985d5abb8fae1cc6f0b1933022895a47a71142aa032a9da3b6be1bbb28')
 old=json.loads(Path(original_lock['path']).read_text())
 records=load_prefix(old['dataset_root'],10000)
 prepared=member(old['prepared']['path'],old['prepared']['sha256'])
 prep=torch.load(prepared['path'],map_location='cpu',weights_only=True,mmap=True)
 cp=torch.load(old['entry_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
 assert prep['metadata']['entry_n']==cp['metadata']['batch']*100==5000
 assert cp['metadata']['seen_ids']==[r['case_id'] for r in records[:5000]]
 expected=old['common_state'];hashes={}
 for name,t in [('W4',prep['weights'][4]),('M4',prep['M4'])]:
  assert t.dtype==torch.float32 and bool(torch.isfinite(t).all())
  hashes[name]=tensor_sha(t)
 assert hashes['W4']==expected['weights']['4']==tensor_sha(cp['weights']['model.layers.4.mlp.down_proj.weight'])
 assert hashes['M4']==expected['M4']==tensor_sha(cp['cache_c'])
 assert digest(prep['contexts'])==expected['contexts']==digest(cp['metadata']['contexts'])
 assert digest(prep['rng'])==expected['rng']==digest(cp['metadata']['rng'])
 stack=torch.load(old['projector'],map_location='cpu',weights_only=True,mmap=True)
 P,pmap=select_projector(stack,4)
 assert tensor_sha(P)==expected['P4'] and pmap==prep['metadata']['P4']
 del cp,stack,P,prep
 for item in old['batch_locks']:
  rows=records[(item['batch']-1)*100:item['batch']*100]
  assert item['records_sha256']==digest(rows) and item['case_ids']==[r['case_id'] for r in rows]
 refs={}
 for arm,cell in [('N4',0),('REFIT4',5)]:
  r=OLD/'output'/f'cell-{cell}'
  entry=json.loads((r/'entry.json').read_text());terminal=json.loads((r/'terminal.json').read_text())
  assert entry['state']==expected and entry['prepared']==prepared
  assert terminal['status']=='TEN_SEQUENTIAL_BATCHES_COMPLETE' and terminal['batches']==list(range(51,61))
  assert terminal['source_lock_sha256']==original_lock['sha256']
  refs[arm]=dict(status='REUSE_EXACT_COMMON_CAPSULE_AND_NATIVE_SOURCE',entry=member(r/'entry.json'),terminal=member(r/'terminal.json'),root=str(r),
   execution='5e96dcb3745977b1f273e3f5afbee61167248d49',logical_batches=10,new_batches=0,
   missing_optimizer_state='NOT_RECORDED_REFERENCE_NO_SYNTHESIS',
   additional_observation_gpu=0,strict_firstsuffix500='CPU_DERIVABLE_FROM_EXISTING_IDENTITY_ROWS',
   comparison='samehost/model/config/context/RNG/precision/evaluator/order/entry; new policy mechanism differs by design')
 # Bind original mathematical/evaluation code by file content; never import mutable worktrees.
 immutable=[]
 for rel in ['fitting.py','evaluation.py','sequential_evaluation.py']:
  ours=w/'project/run_scripts/low_cost_write_donor_pilot'/rel
  prior=OLD/'source/project/run_scripts/low_cost_write_donor_pilot'/rel
  assert file_sha(ours)==file_sha(prior),('REFERENCE_CODE_DRIFT',rel)
  immutable.append(member(prior))
 cost=dict(estimate_not_measured=True,new_main_batches=40,logical_batches=60,
  old_reference_GPU_hours_not_new_spending=(5237+6027)/3600,
  new_main_gpu_hours_range=[6,16],technical_gpu_hours_range=[0.5,2],
  per_policy_walltime='12:00:00',gpu_hour_hardcap=None,
  basis='prior N4 5237s and REFIT4 6027s each with everybatch general; new 24maxAdam but 2/4 fresh solves, target snapshot I/O and terminal general. 12h wall reserve is scheduler request, not user budget.',
  storage_estimate_bytes=120*(1<<30),storage_safety_free_bytes=50*(1<<30),
  storage_breakdown='12 W4/M4 checkpoints about13GB; 120 subwrite deltas about27GB; target/Adam/teacher/captures and technical snapshots conservative80GB',
  prior_prepared_M8_new_reconstruction=0)
 free=shutil.disk_usage(a).free
 assert free>cost['storage_estimate_bytes']+cost['storage_safety_free_bytes'],'DISK_CAPACITY_HOLD'
 receipt=save(a/'FULL_READ-CPU-reuse.json',dict(status='FULL_READ_CPU_EXACT_INPUT_REUSE_NOT_GPU_GATE',documents=docs,
  original_attachment_sha256='6dbd71d8a495afdbc958677540019d7a55d13fbb0d6dc5091f31891615b05b35',
  published_attachment_bytes_equal=False,attachment_normalized_equivalence='GH_ASSERTED_NOT_RAW_TRANSFERRED',
  source_main=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),
  source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=w,text=True).strip(),
  prepared=prepared,L4_tensor_hashes=hashes,P_mapping=pmap,references=refs,
  native_evaluation_immutable=immutable,cost_estimate=cost,free_disk_bytes=free,
  monitoring='TASK_SPECIFIC_CONTINUE_THROUGH_MIDDLE_COMPLETION',other_paused_tasks_unchanged=True))
 lock=copy.deepcopy(old)
 lock.update(instruction_id='ODEEDIT-S06-REFIT4-WRITE-REFRESH-SEQ1000-SH4-V1',
  array_mapping=POLICIES,prepared=prepared,reference_reuse=refs,prepared_check=receipt,
  authorized_execution='FOUR_NEW_MIDDLE_POLICIES_WITH_TWO_REUSED_REFERENCES',
  checkpoint_batches=[51,55,60],audit_submission=False,conditional_followup_submission=False,
  after_initial='CONTINUE_MIDDLE_COMPLETION_USER_OVERRIDE',resource=dict(server='server4',GPU=1,CPU=8,mem='60416M',project_cap=2,array='0-3%2',time='12:00:00',gpu_hour_cap=None),
  technical_parity=dict(native_I1_target='torch.equal required at chunk0 exact native path; mismatch typed HOLD with no posthoc relaxation',
   endpoint='torch.equal same Z/native solve',history='torch.equal same endpoint keys',
   fixedW_carry='torch.equal u/m/v/step with same totalactualupdates; extra loss observations accounted'),
  policies=json.loads((w/'plans/global/2026-09-14-refit4-write-refresh-seq1000-contract.json').read_text())['policies'],
  cost_estimate=cost,reference_lock=original_lock)
 lock['new_input_members']=[receipt,original_lock]+[d['preserved'] for d in docs]+[x for r in refs.values() for x in [r['entry'],r['terminal']]]
 save(a/'refresh-inputs.json',lock)
 print(json.dumps(dict(status='CPU_READY',receipt=receipt,references=refs,cost_estimate=cost)))

def freeze(attempt):
 a=Path(attempt);w=Path(__file__).resolve().parents[3]
 config=json.loads((a/'refresh-inputs.json').read_text());source=a/'source';source.mkdir(mode=0o700,exist_ok=False)
 tracked=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=w,text=True)
 assert not tracked.strip(),'FREEZE_REQUIRES_COMMITTED_CLEAN_SOURCE'
 paths=[]
 for folder in ['project/run_scripts/baseline_mechanism_first','project/run_scripts/low_cost_write_donor_pilot']:
  paths += [p for p in (w/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.sbatch')]
 paths += [w/'scripts/fixed_counterfact.py']
 for p in paths:
  dest=source/p.relative_to(w);dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('xb') as f:f.write(p.read_bytes())
 archive=a/'execution-source.tar'
 with tarfile.open(archive,'x') as tar:
  for p in sorted(source.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(source)))
 members={m['path']:m for m in config['members']}
 for m in config.pop('new_input_members'):members[m['path']]=m
 for p in source.rglob('*'):
  if p.is_file():members[str(p)]=member(p)
 config.update(members=list(members.values()),execution_source=str(source),execution_source_archive=member(archive),
  worktree_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),
  worktree_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=w,text=True).strip())
 print(json.dumps(save(a/'execution.lock.json',config)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze']);p.add_argument('--attempt',required=True)
 args=p.parse_args();globals()[args.action](args.attempt)
