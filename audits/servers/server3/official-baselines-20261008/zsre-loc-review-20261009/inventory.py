"""Bounded CPU metadata inventory only; no performance reducer needed without zsRE raw."""
import os,json,hashlib,pathlib,datetime,csv
root=pathlib.Path('/data/janghj/ODE-edit/local');out=pathlib.Path(__file__).parent
skip={'worktree','source','.git','runtime','datasets','tracking','wandb','venv','__pycache__','post','pre','W0','inputs','cache','blobs','snapshots'}
files=[];errors=[]
for base,dirs,names in os.walk(root,followlinks=False,onerror=lambda e:errors.append(str(e))):
 dirs[:]=[d for d in dirs if d not in skip and not pathlib.Path(base,d).is_symlink()]
 if len(pathlib.Path(base).relative_to(root).parts)>6:dirs[:]=[];continue
 for n in names:
  if n not in {'terminal.json','result.json','completion.json','submission.json','status.json'}:continue
  p=pathlib.Path(base,n)
  if p.is_symlink() or p.stat().st_size>1000000:continue
  b=p.read_bytes()
  try:d=json.loads(b)
  except Exception as e:errors.append(str(p)+':'+type(e).__name__);continue
  files.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),zsre_mention='zsre' in b.decode().lower(),**{k:d[k] for k in ['status','dataset','arm','model','job_id','source','batches'] if k in d}))
assert not any(x['zsre_mention'] for x in files)
evidence=[]
for name in [root/'official-baselines-20261008/submission-request-preflight-20261009.json',pathlib.Path('audits/servers/server3/official-baselines-20261008/zsre-wandb-metrics-r1.json'),pathlib.Path('experiment-reports/servers/server3/official-baselines-20261008/report-ko.md')]:
 b=name.read_bytes();evidence.append(dict(path=str(name),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
r=dict(nonce='USER-GH-ZSRE-LOC-RECALCULATE-MAIN-20261009-R1',server='server3',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='NO_OWN_COMPLETED_ZSRE_FOUND',completed_zsre_count=0,metric_policy='Loc=100*mean_requests(mean_loc_ans_tokens(predicted==target)); W0agreement auxiliary only',future_adoption='NEXT_NEW_SOURCE_FREEZE; shared GH implementation not modified',scope=dict(root=str(root),maximum_parent_depth=6,metadata_names=['terminal.json','result.json','completion.json','submission.json','status.json'],maximum_file_bytes=1000000,pruned_dirs=sorted(skip),symlink_follow=False,metadata_count=len(files),errors=errors,limitation='Bounded known-project metadata and own official submission records; not a filesystem-wide proof. Copied historical/worktree/source results are not owned executions.'),evidence=evidence,files=files,inventory_script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),reducer_sha256=None,reducer_status='NOT_APPLICABLE_NO_ZSRE_RAW',before_W0agreement=None,after_Loc=None,case_denominator=None,token_denominator=None,Eff=None,Gen=None,CF='UNCHANGED; Qwen61813 excluded as CF',GPU_forward_jobs_mutations=0,raw_CP_frozen='KEEP')
(out/'inventory.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
with (out/'table.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['server','dataset','status','completed_count','before_W0agreement','after_Loc','case_denominator','token_denominator','reducer_sha256']);w.writerow(['server3','zsRE',r['status'],0,'','','','',''])
print(json.dumps({'count':len(files),'zsre':0,'errors':errors}))
