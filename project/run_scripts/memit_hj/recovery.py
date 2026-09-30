"""Exact frozen long-state recovery into a NEW output namespace.

Never infer edited state from scalar receipts. Only an owned, hash-verified
task CP supplies W/H/RNG. Completed scalar artifacts and immutable inputs can
be hard-linked; uncheckpointed partial trajectories stay in the old attempt.
"""
import json,os
from pathlib import Path
import torch
from project.run_scripts.memit_history_lifelong.io import file_sha,save
from .reducer import verify_cell

TASK_ROOT=Path('/data/janghj/ODE-edit/local/memit-hj/20260930-v2').resolve()
def payload(path,bindings):
 p=Path(path).resolve();assert p.is_relative_to(TASK_ROOT) and p.parent.name=='temporary-checkpoints'
 manifest=json.loads(p.with_suffix('.manifest.json').read_text())
 assert p.stat().st_uid==manifest['owner_uid']==os.getuid() and p.stat().st_size==manifest['bytes']
 assert file_sha(p)==manifest['sha256'] and manifest['reload_state_output']
 s=torch.load(p,map_location='cpu',weights_only=False)
 assert s['bindings']==bindings and s['meta']['cell'] in ['main_000','main_100','main_110','main_111']
 assert s['meta']['cursor']%1000==0 and s['meta']['ledger']==list(range(s['meta']['cursor']))
 return s,manifest

def link_file(src,dst):
 dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert file_sha(src)==file_sha(dst)
 else:os.link(src,dst)

def link_tree(src,dst):
 for p in sorted(Path(src).rglob('*')):
  if p.is_file():
   assert p.suffix not in ['.pt','.partial'];link_file(p,Path(dst)/p.relative_to(src))

def prepare(engine,orchestrator,checkpoint):
 s,manifest=payload(checkpoint,engine.lock['bindings']);old=Path(checkpoint).resolve().parent.parent
 assert old!=engine.output.resolve(),'RECOVERY_MUST_USE_NEW_OUTPUT_NAMESPACE'
 arm=s['meta']['cell'];cursor=s['meta']['cursor']
 # W0 evaluation, model T0, and calibration are source-bound immutable inputs.
 for name in ['readiness.json','W0-all10k.json']:link_file(old/name,engine.output/name)
 if (old/'T0').exists():link_tree(old/'T0',engine.output/'T0')
 if (old/'calibration/lock.json').exists():link_tree(old/'calibration',engine.output/'calibration')
 if (engine.output/'calibration/lock.json').exists():
  orchestrator.zlock=json.loads((engine.output/'calibration/lock.json').read_text());assert orchestrator.zlock['bindings']==engine.lock['bindings']
 def prefix(name,n):
  src=old/'cells'/name;dst=engine.output/'cells'/name
  link_file(src/'lineage.json',dst/'lineage.json')
  for d in src.glob('C*'):
   if d.is_dir() and int(d.name[1:])<=n:link_tree(d,dst/d.name)
  for p in src.glob('*.json'):
   if p.name.startswith(('trigger-','refresh-')) and int(p.stem.rsplit('-',1)[1])<=n:link_file(p,dst/p.name)
 prefix(arm,cursor)
 lineage=json.loads((old/'cells'/arm/'lineage.json').read_text())
 parent_checkpoint=lineage.get('resume_parent_checkpoint') if arm=='main_111' else None
 if lineage.get('parent'):prefix(lineage['parent'],lineage['prefix_requests'])
 for c in orchestrator.cells.values():
  name=c['cell_id'];term=old/'cells'/name/'terminal.json'
  if name==arm or not term.exists():continue
  eligible=(arm=='main_000' and c['family']!='main' and int(c['anchor_requests'])<=cursor) or (name=={'main_000':'main_001','main_100':'main_101','main_110':'main_111'}.get(arm))
  if not eligible:continue
  record=json.loads(term.read_text());assert record['status'] in ['COMPLETED','NOT_FIRED']
  if record['status']=='COMPLETED':verify_cell(old,c)
  link_tree(old/'cells'/name,engine.output/'cells'/name);engine.completed[name]=record
 engine.restore(s)
 save(engine.output/'recovery.json',dict(checkpoint=manifest,source_output=str(old),source_config_input_bound=True,
  next_cursor=cursor,old_partial_after_cursor_preserved=True,linked_completed_cells=sorted(engine.completed),
  new_attempt_recomputed_prefix_requests=0,parent_checkpoint=parent_checkpoint))
 return s,parent_checkpoint

def resume(engine,orchestrator,checkpoint):
 s,parent=prepare(engine,orchestrator,checkpoint);arm=s['meta']['cell'].removeprefix('main_')
 orchestrator.run_main(arm,from_snapshot=s,resume=True)
 if parent:
  # The long child's parent stayed pinned when RAM was lost. Restore that
  # exact state, skip its now-completed child, and continue the parent chain.
  ps,manifest=payload(parent,engine.lock['bindings'])
  engine.completed['main_111']=json.loads((engine.output/'cells/main_111/terminal.json').read_text())
  orchestrator.run_main('110',from_snapshot=ps,resume=True)
