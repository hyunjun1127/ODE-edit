"""Bind frozen source to exact existing assets; no downloads or environment writes."""
import argparse,importlib.metadata,json,os,sys
from pathlib import Path
from project.run_scripts.memit_history_lifelong.io import save,file_sha,digest

BASE=Path('/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/attempt-repair-r1/execution.lock.json')
def freeze(source,commit,archive,attempt):
 source=Path(source).resolve();attempt=Path(attempt).resolve();attempt.mkdir(parents=True,exist_ok=False)
 old=json.loads(BASE.read_text());members={}
 def add(p,expected=None):
  p=Path(p);st=p.stat();sha=file_sha(p)
  if expected:assert sha==expected,'PINNED_BYTES_CHANGED:'+str(p)
  members[str(p)]=dict(path=str(p),bytes=st.st_size,sha256=sha,stat_identity=dict(device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns))
  return sha
 for m in old['members']:
  p=Path(m['path'])
  if p.is_relative_to(old['source_root']):p=source/p.relative_to(old['source_root'])
  add(p,m['sha256'])
 # New callable closure and authorities are separately sealed. The broad
 # package inventory is source only, never a claim every file executed.
 for sub in ['memit_hj','single_layer_mechanism_first','low_cost_write_donor_pilot']:
  for p in sorted((source/'project/run_scripts'/sub).rglob('*')):
   if p.is_file() and p.suffix in ['.py','.json','.sbatch']:add(p)
 authorities=source/'audits/global/2026-09-30-memit-hj-sh3-dispatch/input-manifest.json'
 authority=json.loads(authorities.read_text());add(authorities,'c5d31907e36e94b1d57e9bacaf37ce2c2c7dd70e8a84388716b660b519b355f6')
 for m in authority['members']:
  assert (source/m['path']).stat().st_size==m['bytes'];add(source/m['path'],m['sha256'])
 for name in ['2026-09-30-memit-hj-sh3.md','2026-09-30-memit-hj-sh3-all-implementation-dependencies.md']:add(source/'messages/head'/name)
 from scripts.fixed_counterfact import load_prefix
 assert len(load_prefix(old['dataset_root'],10000))==10000
 resource=json.loads((source/'audits/servers/server3/memit-hj-20260930-v2/resource-plan.json').read_text())
 v=os.statvfs(attempt);assert v.f_bavail*v.f_frsize>=resource['storage']['required_free_bytes'],'INSUFFICIENT_STORAGE_RESERVE'
 plan=json.loads((source/'project/run_scripts/memit_hj/plan.json').read_text())
 source_sha=file_sha(archive)
 bindings=dict(source_commit=commit,source_archive_sha256=source_sha,authority_manifest_sha256=file_sha(authorities),
  hparams_sha256=file_sha(source/'project/run_scripts/memit_history_lifelong/hparams.json'),dataset=old['dataset'],context_sha256=file_sha(old['context']),
  model_revision=old['revision'],BLUE_commit=old['blue_commit'],resource_cap=1,plan_sha256=file_sha(source/'project/run_scripts/memit_hj/plan.json'))
 lock={k:old[k] for k in ['blue_root','blue_commit','entrypoint','entrypoint_sha256','snapshot','revision','dataset_root','dataset','context','stats_root','stats_paths','seed']}
 lock.update(schema='odeedit.memit-hj.v2',instruction='ODEEDIT-GH-SH3-MEMIT-HJ-V2-20260930-R1',additional_nonce='ODEEDIT-GH-SH3-MEMIT-HJ-ALL-IMPL-AUDIT-DAG-20260930-R1',
  source_commit=commit,source_root=str(source),source_archive=str(Path(archive).resolve()),source_archive_sha256=source_sha,
  hparams=str(source/'project/run_scripts/memit_history_lifelong/hparams.json'),cells=str(source/'plans/global/2026-09-30-memit-hj-experiment-design-v1/cells.csv'),
  output=str(attempt/'output'),resource=resource,plan=plan,bindings=bindings,members=list(members.values()),
  launchers={name:file_sha(source/'project/run_scripts/memit_hj'/name) for name in ['run.sbatch','collect.sbatch']},
  save_permanent_checkpoints=False,temporary_checkpoint_exception=['000','100','110','111'],
  python=sys.version,packages={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','scipy','safetensors','tokenizers','accelerate','matplotlib']})
 assert lock['packages']['torch']=='2.9.1+cu128' and lock['packages']['transformers']=='4.44.2'
 save(attempt/'execution.lock.json',lock)
 save(attempt/'preflight.json',dict(status='CPU_ASSET_SOURCE_VERIFIED',members=len(members),bytes=sum(x['bytes'] for x in members.values()),bindings=bindings,
  resource_free_bytes=v.f_bavail*v.f_frsize,actual_model_T0='NOT_OBSERVED',existing_54007_runtime_bytes_reused=True))
 print(json.dumps(dict(lock=str(attempt/'execution.lock.json'),sha256=file_sha(attempt/'execution.lock.json'),members=len(members),source=commit)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--commit',required=True);p.add_argument('--archive',required=True);p.add_argument('--attempt',required=True);a=p.parse_args();freeze(a.source,a.commit,a.archive,a.attempt)
