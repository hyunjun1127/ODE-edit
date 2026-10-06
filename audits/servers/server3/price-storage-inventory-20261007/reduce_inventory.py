"""Reduce frozen metadata only. Classifications are retention evidence, never deletion approval."""
import collections,csv,datetime,hashlib,json,os,re
from pathlib import Path
ROOT=Path('/data/janghj/ODE-edit'); BASE=ROOT/'local/price-storage-inventory-20261007'; WT=BASE/'worktree'
AUDIT=WT/'audits/servers/server3/price-storage-inventory-20261007'
AUTH='messages/head/2026-10-07-price-storage-inventory-all-sh.json'
CLASSES=['PRICE_REQUIRED','OTHER_TASK_REQUIRED','REPRODUCTION_KEEP','UNREFERENCED_CANDIDATE','UNKNOWN']
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def small_json(p):
 assert not p.is_symlink() and p.stat().st_size<=3*1024**2
 return json.loads(p.read_text())
def member(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def main():
 os.nice(15)
 os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
 coverage=json.loads((BASE/'scan-coverage.json').read_text());supp=json.loads((BASE/'supplement-coverage.json').read_text())
 rows=[]
 for f in ['full-stat.jsonl','supplement-stat.jsonl']:
  rows.extend(json.loads(l) for l in (BASE/f).open())
 bypath={r['path']:r for r in rows}
 assets_path=ROOT/'local/state/sh3-experiment-ready-20260919-v1/asset-inventory.json'
 assetrows=small_json(assets_path);assetindex={r.get('consumed_path',''):r for r in assetrows}
 refs={};ref_members=[];read_bytes=0;ref_skips=[]
 # Only fixed metadata names, excluding payload/result/credential files and source copies.
 allowed_names={'execution.lock.json','execution-full-sha.lock.json','inputs.lock.json','asset-inventory.json','input-manifest.json','asset-manifest.json','source.lock.json'}
 refpaths=[Path(r['path']) for r in rows if r['type']=='regular' and (Path(r['path']).name in allowed_names or ('temporary-checkpoints/' in r['path'] and r['path'].endswith('.manifest.json'))) and '/worktree/' not in r['path'] and '/source/' not in r['path']]
 refpaths+= [ROOT/'local/en-adaptive-nullspace/20260920-v1/inputs/pstar-derived-v1/receipt.json']
 def index(v,evidence):
  if isinstance(v,dict):
   for k,x in v.items():
    if any(t in k.lower() for t in ['credential','secret','api_key','password']):continue
    index(x,evidence)
  elif isinstance(v,list):
   for x in v:index(x,evidence)
  elif isinstance(v,str) and v.startswith('/data/janghj/') and v in bypath:
   refs.setdefault(v,[])
   if evidence not in refs[v] and len(refs[v])<4:refs[v].append(evidence)
 for p in sorted(set(refpaths)):
  s=p.lstat()
  if s.st_size>3*1024**2 or read_bytes+s.st_size>64*1024**2:ref_skips.append(str(p));continue
  try:d=small_json(p)
  except (ValueError,OSError):ref_skips.append(str(p));continue
  read_bytes+=s.st_size;index(d,str(p));ref_members.append(member(p))
 evidence={'readiness':'agents/server3/experiment-ready-paths-20260919-v1.json','bootstrap':'audits/servers/server3/2026-09-19-bootstrap/environment-observations.json','checkpoint':'experiment-reports/servers/server3/memit-hj-20260930-v2/review-1k-20261002-v1/report-ko.md#11','past_KEEP':AUTH+' protected + source/raw/failure KEEP','GSS':'messages/head/2026-09-20-sh3-en-adapt-gss-history-10k.md','basis':'/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/inputs/pstar-derived-v1/receipt.json'}
 def classify(r):
  p=r['path'];path=Path(p);ref=';'.join(refs.get(p,[]));reason=r['root_reason']
  if reason.startswith('bootstrap_registered'):
   return 'REPRODUCTION_KEEP','historical_base_model_or_projector',evidence['bootstrap'],'Historic registered GPT-J original model/projector; current PRICE usage not established; original protected asset', 'GPT-J registered cache'
  if reason!='repository_root':
   return 'OTHER_TASK_REQUIRED',reason,evidence['readiness'],'Explicit protected base model/C0/five-layer projector/hparams; same logical assets used by PRICE on S4, S3 live dependency not established',reason
  if '/worktree/' in p or not path.is_relative_to(ROOT/'local'):
   return 'OTHER_TASK_REQUIRED','source_or_worktree',AUTH,'Git/worktree/original dirty files explicitly protected; no live-use assertion','repository and worktrees'
  rel=path.relative_to(ROOT/'local'); task=rel.parts[0]
  if task in {'runtime','datasets','wandb-setup'} or 'dependencies' in p:
   return 'OTHER_TASK_REQUIRED','dataset_context_runtime_or_logging',ref or evidence['readiness'],'Reusable runtime/dataset/context or W&B local files; unsynced status not inspected; preserve',task
  if task in {'memit-hj','memit-history-fixed10k','memit-h-failure-audit','jlz-realized-subject-v10','jlz-v12-alphaedit-writer','jlz-v12-shared-budget','jlz-v14-budget15-r2','en-adaptive-nullspace','en-adapt-gss-history','state','storage-cleanup'}:
   kind='historical_source_raw_or_receipt';why=evidence['past_KEEP']
   if 'temporary-checkpoints' in p and path.suffix=='.pt':kind='edited_checkpoint';why=evidence['checkpoint']
   elif 'pstar-derived-v1/basis.npy' in p:kind='derived_projection_basis';why=evidence['basis']
   elif '/cold-history/' in p:kind='history_teacher_or_sketch_map';why=evidence['GSS']
   elif r['tensor_extension_or_name']:kind='historical_tensor'
   return 'REPRODUCTION_KEEP',kind,ref or why,'Original task/raw/failure/reproduction retained; stopped/noCP does not authorize removal',task
  return 'UNKNOWN','unresolved_local_file',ref,'No sufficient current consumer/retention classification proof in bounded scope; preserve',task
 classified=[]
 for r in rows:
  if r['type']!='regular':continue
  cls,kind,ref,why,group=classify(r)
  old=assetindex.get(r['path']);oldsha=None;statmatch=None
  if old:
   oldsha=old.get('observed_sha256') or old.get('sha256');s=old.get('stable_stat')
   statmatch=bool(s and list(s[:4])==[r['dev'],r['inode'],r['logical_bytes'],r['mtime_ns']])
  out=dict(server='server3',absolute_path=r['path'],artifact_kind=kind,logical_bytes=r['logical_bytes'],allocated_bytes=r['allocated_bytes'],count=1,device_inode_or_hardlink_group=f"{r['dev']}:{r['inode']}",nlink=r['nlink'],owner_uid=r['uid'],mtime=r['mtime_utc'],referenced_task_or_lock=ref,classification=cls,classification_evidence=why,uniqueness_or_reproducibility='dev/inode dedup only; no content comparison or resume verification',uncertainty='No process/open-file check; no current use absence proof; old SHA not recomputed',delete_authorized=False,group=group,tensor_scope=r['tensor_extension_or_name'] or r['root_reason'] in {'registered_HF_model_cache','bootstrap_registered_model_cache'},historical_sha256=oldsha,historical_stat_match=statmatch)
  classified.append(out)
 unique={};pathcounts=collections.Counter()
 for r in classified:
  key=r['device_inode_or_hardlink_group'];pathcounts[key]+=1
  unique.setdefault(key,r)
 def summarize(items):
  rr=list(items);return dict(file_count=len(rr),logical_bytes=sum(r['logical_bytes'] for r in rr),allocated_bytes=sum(r['allocated_bytes'] for r in rr))
 categories={c:summarize(r for r in unique.values() if r['classification']==c) for c in CLASSES}
 tensor_categories={c:summarize(r for r in unique.values() if r['classification']==c and r['tensor_scope']) for c in CLASSES}
 groups=[]
 for group in sorted({r['group'] for r in unique.values()}):
  vals=[r for r in unique.values() if r['group']==group]
  groups.append(dict(group=group,**summarize(vals),classes=sorted({r['classification'] for r in vals})))
 top=sorted(unique.values(),key=lambda r:r['allocated_bytes'],reverse=True)[:20]
 tensor_groups=[]
 for task,match in [('MEMIT-HJ edited CP',lambda p:'temporary-checkpoints/' in p and p.endswith('.pt')),('EN derived basis',lambda p:'pstar-derived-v1/basis.npy' in p),('GSS cold history/maps',lambda p:'/en-adapt-gss-history/' in p and '/cold-history/' in p and p.endswith(('.npz','.npy','.pt'))),('W&B local files excluding SDK/worktree',lambda p:'/wandb-setup/' in p and '/worktree/' not in p)]:
  vals=[r for r in unique.values() if match(r['absolute_path'])]
  tensor_groups.append(dict(label=task,**summarize(vals),classification='OTHER_TASK_REQUIRED' if task.startswith('W&B') else 'REPRODUCTION_KEEP',sample_paths=[r['absolute_path'] for r in sorted(vals,key=lambda r:-r['allocated_bytes'])[:2]],delete_authorized=False))
 price=[]
 for task in ['jlz-price-cap-base-repair-2k','jlz-price-alpha-writer-2k']:
  path=WT/f'runs/{task}/submission.json';d=small_json(path)
  price.append(dict(task=task,source=d['source'],lock_sha256=d['lock']['sha256'],registered_owner='server4',nodes=sorted({arg.split('=',1)[1] for v in d['mapping'].values() for arg in v['argv'] if arg.startswith('--nodelist=')}),local_lock_path_exists=Path(d['lock']['path']).exists(),submission_receipt=member(path),current_scheduler_state='NOT_QUERIED'))
 cp=ROOT/'local/memit-hj/20260930-v2/attempt-v1/output/temporary-checkpoints/000-01000-0000.manifest.json'
 d=small_json(cp);cp_info={k:d[k] for k in ['path','bytes','sha256','cursor','refcount','reload_state_output']};cp_info.update(manifest=member(cp),current_metadata_match_size=bypath[d['path']]['logical_bytes']==d['bytes'],current_content_hash='NOT_RECOMPUTED',deletion_permission=False)
 summary=dict(schema=1,nonce='USER-GH-ALL-SH-PRICE-STORAGE-INVENTORY-20261007-SERVER3',status='INVENTORY_COMPLETE_NO_DELETE',delete_authorized=False,actual_host='ubuntu',actual_user='janghj',session='01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3',repository_root=str(ROOT),registry_cwd_matches=True,authority_commit='4ea217b5258ff786e83a5dcf617d311265a4ff94',snapshot_utc=coverage['started_utc'],supplement_utc=supp['started_utc'],scope_totals=summarize(unique.values()),regular_path_count=len(classified),hardlink_duplicate_paths_removed=len(classified)-len(unique),hardlink_multi_path_groups=sum(n>1 for n in pathcounts.values()),nlink_gt1_unique=sum(r['nlink']>1 for r in unique.values()),categories=categories,tensor_categories=tensor_categories,directory_groups=sorted(groups,key=lambda r:-r['allocated_bytes']),top20=top,special_groups=tensor_groups,price_reference=price,price_local_data='NO_PRICE_EXECUTION_LOCK_OR_LOCAL_TASK_ROOT_FOUND_IN_SCOPE; tracked Git references exist',checkpoint=cp_info,confirmed_reclaimable_bytes=0,candidate_bytes=categories['UNREFERENCED_CANDIDATE']['allocated_bytes'],candidate_is_deletion_permission=False,df_first=coverage['df'],df_supplement=supp['df'],content_hashes_recomputed_on_models_or_tensors=0,symlink_follow=False,models_loaded=0,Slurm_queries=0,Slurm_writes=0,wandb_runs=0,owner_audit=True,independent_agent_review=False)
 dump(AUDIT/'summary.json',summary)
 # Compact CSV: top20 individual files and unique-inode directory aggregates.
 fields=['server','absolute_path','artifact_kind','logical_bytes','allocated_bytes','count','device_inode_or_hardlink_group','nlink','owner_uid','mtime','referenced_task_or_lock','classification','classification_evidence','uniqueness_or_reproducibility','uncertainty','delete_authorized']
 csvrows=list(top)
 for r in groups:
  csvrows.append(dict(server='server3',absolute_path='GROUP:'+r['group'],artifact_kind='directory_aggregate',logical_bytes=r['logical_bytes'],allocated_bytes=r['allocated_bytes'],count=r['file_count'],device_inode_or_hardlink_group='per-file dev/inode deduplicated; membership local full inventory',mtime=coverage['started_utc'],referenced_task_or_lock='classification rules in reduce_inventory.py',classification=';'.join(r['classes']),classification_evidence='Aggregate of explicit per-file retention evidence',uniqueness_or_reproducibility='hardlink dedup; no same-name/size content dedup',uncertainty='Group rows overlap top20; do not sum CSV rows',delete_authorized=False))
 with (AUDIT/'inventory.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(csvrows)
 with (BASE/'classified-full.jsonl').open('w') as f:
  for r in classified:f.write(json.dumps(r,separators=(',',':'))+'\n')
 dump(BASE/'reference-metadata-members.json',ref_members)
 cov=dict(schema=1,scan=coverage,supplement=supp,metadata_reference_files_read=len(ref_members),metadata_reference_bytes_read=read_bytes,reference_read_limits={'file_bytes':3*1024**2,'total_bytes':64*1024**2},reference_files_skipped=ref_skips,coverage_notes=['Selected project files and exact registered HF/model/projector/C0 roots only, not whole filesystem usage.','No symlink targets followed. Package env/cache/private dirs and own new inventory/worktree excluded.','Snapshot is lstat metadata, not frozen filesystem; concurrent file changes possible. No science progress or result monitoring.','Group sums dedup across scanned dev/inode. nlink may include links outside scope; removing a name is not guaranteed reclaim.','Directory inode blocks, excluded environments, Git objects, other projects/users, unregistered cache locations not counted.','Qwen registered model directory missing; no alternative path search/home sweep.','Existing SHA/stat receipts retained, not fresh tensor/model hashes or byte-equivalence proof.','W&B sync status not queried; its local spool/logs remain protected.','No Slurm/open-file snapshot required: entire identified historical/current-source data protected; no idle/no-use assertion.'],delete_authorized=False)
 dump(AUDIT/'coverage.json',cov)
 # Arithmetic re-check independent of class-loop iteration.
 assert sum(x['allocated_bytes'] for x in categories.values())==sum(r['allocated_bytes'] for r in unique.values())
 assert sum(x['file_count'] for x in categories.values())==len(unique)
 assert all(not r['delete_authorized'] for r in classified)
 print(json.dumps({k:summary[k] for k in ['scope_totals','categories','special_groups','hardlink_duplicate_paths_removed']},ensure_ascii=False))
if __name__=='__main__':main()
