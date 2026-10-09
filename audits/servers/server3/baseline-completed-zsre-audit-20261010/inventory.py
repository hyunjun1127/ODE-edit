"""Bounded metadata audit: no checkpoint/raw tensor load, forward or scheduler mutation."""
from pathlib import Path
import os,json,hashlib,datetime,subprocess
OUT=Path(__file__).resolve().parent;REPO=OUT.parents[3];ROOT=Path('/data/janghj/ODE-edit/local')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prune={'worktree','source','src','.git','runtime','datasets','tracking','wandb','venv','__pycache__','post','pre','W0','inputs','cache','blobs','snapshots','reference','node_modules'}
files=[];errors=[];oversized=[]
roots=[ROOT/n for n in ['official-baselines','official-baselines-20261008','qwen-baselines-12-20261009','fixed10k-native-baselines','qwen-blue-sweep-1k-20261010']]
for root in roots:
 for base,dirs,names in os.walk(root,followlinks=False,onerror=lambda e:errors.append(str(e))):
  dirs[:]=[d for d in dirs if d not in prune and not Path(base,d).is_symlink()]
  if len(Path(base).relative_to(root).parts)>7:dirs[:]=[];continue
  for n in names:
   if n not in {'terminal.json','result.json','completion.json','submission.json','status.json','summary.json','progress.json','receipt.json','W20.json','latest.json'}:continue
   p=Path(base,n)
   if p.is_symlink():continue
   if p.stat().st_size>1000000:oversized.append(str(p));continue
   b=p.read_bytes()
   try:d=json.loads(b)
   except Exception as e:errors.append(str(p)+':'+type(e).__name__);continue
   fields={k:d.get(k) for k in ['status','dataset','method','job_id','job_name','source','source_commit','batches'] if isinstance(d,dict) and k in d}
   files.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),mentions_zsre='zsre' in b.decode().lower(),**fields))
old=REPO/'audits/servers/server3/zsre-2k-reeval-20261009/inventory.json';prior=json.loads(old.read_text());unchanged=[];changed=[];absent=[]
for x in prior['metadata']:
 p=Path(x['path'])
 if not p.exists():absent.append(str(p))
 elif p.stat().st_size==x['bytes'] and sha(p)==x['sha256']:unchanged.append(str(p))
 else:changed.append(str(p))
snapshot=json.loads(subprocess.check_output(['squeue','--json','--user=janghj'],text=True))
owned=[]
for j in snapshot['jobs']:
 if j.get('nodes')=='ubuntu' or 'ubuntu' in str(j.get('required_nodes','')) or str(j.get('name','')).startswith('s3-'):
  owned.append({k:j.get(k) for k in ['job_id','name','job_state','nodes','required_nodes','current_working_directory','dependency','tres_per_node']})
receipt={'nonce':'USER-GH-ALL-SH-BASELINE-COMPLETED-ZSRE-AUDIT-20261010-R1','accepted_turn':'01a122d0-2dca-7152-b5ba-b513af14972d','observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'server':'server3','authority':'7dacd3fb','metadata':files,'prior_inventory_sha256':sha(old),'prior_unchanged_count':len(unchanged),'prior_changed_paths':changed,'prior_absent_paths':absent,'own_queue_snapshot':owned,'coverage':{'roots':list(map(str,roots)),'depth':7,'max_bytes':1000000,'pruned':sorted(prune),'errors':errors,'oversized':oversized,'limitation':'Known own baseline roots and prior receipts only, not filesystem-wide absence proof'},'reducer_sha256':sha(Path(__file__))}
(OUT/'inventory.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'metadata':files,'prior_unchanged':len(unchanged),'changed':changed,'absent':absent,'queue':owned,'errors':errors,'oversized':oversized},ensure_ascii=False,indent=2))
