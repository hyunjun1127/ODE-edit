"""Seal one execution closure. Source path changes are explicit, never live edits."""
import argparse,json,tarfile,shutil,subprocess
from pathlib import Path
from .runtime import file_sha,save

def freeze(root):
 C=Path(root);A=C/'attempt-v1';W=Path(__file__).resolve().parents[3]
 inputs=json.loads((A/'inputs.json').read_text());source=A/'source';source.mkdir(mode=0o700,exist_ok=False)
 paths=[]
 for folder in ('project/run_scripts/baseline_mechanism_first','project/run_scripts/low_cost_write_donor_pilot'):
  paths += [p for p in (W/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.sbatch')]
 paths += [W/'scripts/fixed_counterfact.py',W/'scripts/check-session-boundary.sh']
 for p in paths:
  dest=source/p.relative_to(W);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 archive=A/'execution-source.tar'
 with tarfile.open(archive,'x') as f:
  for p in sorted(source.rglob('*')):
   if p.is_file():f.add(p,arcname=str(p.relative_to(source)))
 old=json.loads(Path(inputs['prior_lock']).read_text());members={m['path']:m for m in old['members']}
 for m in inputs['members']:members[m['path']]=m
 # Prior lock is immutable reused identity; current stat now, full SHA in job.
 for m in members.values():assert Path(m['path']).is_file() and Path(m['path']).stat().st_size==m['bytes'],m['path']
 for p in source.rglob('*'):
  if p.is_file():members[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p))
 for p in [Path(inputs['dataset_root'])/'counterfact.json',Path(inputs['dataset_root'])/'source-sample.lock.json',Path(inputs['dataset_root'])/'receipt.json']:
  members[str(p)]=dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha(p))
 lock=dict(inputs,members=list(members.values()),execution_source=str(source),execution_source_archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=file_sha(archive)),adapter_base='b51dcf5ab825608bee81dd13549318d8d267e835',schema_repair='58f50a25809779918b22ad0aceded732c097eab4',worktree_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),resource=dict(server='server4',GPU=1,CPU=8,mem='60416M',project_cap=2,time='12:00:00',gpu_hour_cap=None),core=['N4','S875','S75','FULL8','RES8','REFIT4'],after_initial='MONITORING_PAUSED_AWAITING_USER',audit_or_suffix_submission=False)
 save(A/'execution.lock.json',lock)
 save(A/'evidence-reuse-manifest.json',dict(prior_completed_source_and_assets=len(old['members']),new_CPU=['CP fullSHA/schema','Historical128 exact selection','Wiki128 token identity','MMLU32/68','audit128 exclusion seal'],new_GPU=['same-host N4 request-z100','secondfit request-z300','M8 common We B100x50 reconstruction','fixed endpoint observations'],unavailable=['exact same-host completed capsule','pure writer cost without instrumentation before execution'],source_archive=lock['execution_source_archive'],shared_D4_z_M8_study_cost_count=1,all_conditional_followup_unsubmitted=True))
 print('EXECUTION_FROZEN',file_sha(A/'execution.lock.json'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();freeze(a.root)
