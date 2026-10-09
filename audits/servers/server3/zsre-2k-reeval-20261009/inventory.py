"""Read-only metadata eligibility check. Does not open checkpoint payloads."""
import pathlib,os,json,hashlib,datetime
ROOT=pathlib.Path('/data/janghj/ODE-edit/local');OUT=pathlib.Path(__file__).parent
prune={'worktree','source','.git','runtime','datasets','tracking','wandb','venv','__pycache__','post','pre','W0','inputs','cache','blobs','snapshots'}
records=[];errors=[];oversized=[]
for base,dirs,files in os.walk(ROOT,followlinks=False,onerror=lambda e:errors.append(str(e))):
 dirs[:]=[d for d in dirs if d not in prune and not pathlib.Path(base,d).is_symlink()]
 if len(pathlib.Path(base).relative_to(ROOT).parts)>6:dirs[:]=[];continue
 for n in files:
  if n not in {'terminal.json','result.json','completion.json','submission.json','status.json'}:continue
  p=pathlib.Path(base,n)
  if p.is_symlink():continue
  if p.stat().st_size>1000000:oversized.append(str(p));continue
  b=p.read_bytes()
  try:d=json.loads(b)
  except Exception as e:errors.append(str(p)+':'+type(e).__name__);continue
  records.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),mentions_zsre='zsre' in b.decode().lower(),status=d.get('status'),dataset=d.get('dataset'),job_id=d.get('job_id')))
assert not any(r['mentions_zsre'] for r in records)
evidence=[]
for p in [ROOT/'official-baselines-20261008/submission-request-preflight-20261009.json',pathlib.Path('audits/servers/server3/official-baselines-20261008/zsre-loc-review-20261009/inventory.json')]:
 b=p.read_bytes();evidence.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
r=dict(nonce='USER-GH-ZSRE-SAVED-WEIGHTS-2K-REEVAL-20261009-R1',accepted_turn='current direct SH3 owner turn',session='01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3',server='server3',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='NOT_APPLICABLE',reason='NO_OWN_COMPLETED_ZSRE_W20_CHECKPOINT_FOUND',eligible_methods=[],eligible_checkpoints=[],new_job_ids=[],new_job_names=[],completed_raw_metrics=None,checkpoint_payload_SHA_status='NOT_APPLICABLE_NO_ELIGIBLE_CP',source_pending='GH common evaluator READY pending at dispatch; not a blocker for this nojob disposition',metadata=records,evidence=evidence,coverage=dict(root=str(ROOT),depth=6,max_metadata_bytes=1000000,pruned=sorted(prune),errors=errors,oversized=oversized,scope='Known own local execution metadata; excludes code copies and external replicas; not filesystem-wide absence proof'),inventory_source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),CPU_only=True,model_checkpoint_load=False,Slurm_operations=0,other_server_replica_evaluation=False,preservation='source/raw/checkpoint/frozen jobs KEEP')
(OUT/'inventory.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':r['status'],'metadata_count':len(records),'errors':errors,'oversized':oversized}))
