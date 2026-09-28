"""Independent CPU-only review of immutable MEMIT receipts; no runtime imports.
Usage: python review_completed.py --attempt ... --dataset ... --report ... --audit ... --local ... --baseline ...
"""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
import numpy as np

FULL=[1,5,10,20,30,40,50,60,70,80,90,100]
TAGS=['RS','PS','NS']; WIDTH={'RS':1,'PS':2,'NS':10}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def table(p,rows):
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def summarize(rows,tag):
 d='true' if tag=='NS' else 'new';n=len(rows)
 out=dict(numerator=sum(r['success'] for r in rows),denominator=n,rate=sum(r['success'] for r in rows)/n,
 tf_token_correct=sum(r[d+'_token_correct'] for r in rows),tf_token_count=sum(r[d+'_token_count'] for r in rows),
 tf_prompt_macro=sum(r[d+'_token_correct']/r[d+'_token_count'] for r in rows)/n,
 tf_strict_numerator=sum(r[d+'_strict'] for r in rows),tf_strict=sum(r[d+'_strict'] for r in rows)/n,
 new_nll=sum(r['new_nll'] for r in rows)/n,true_nll=sum(r['true_nll'] for r in rows)/n,
 desired_nll=sum(r[d+'_nll'] for r in rows)/n,margin_true_minus_new=sum(r['margin'] for r in rows)/n,
 ties=sum(r['new_nll']==r['true_nll'] for r in rows),near_tie_1e4=sum(abs(r['margin'])<1e-4 for r in rows))
 out['tf_token_micro']=out['tf_token_correct']/out['tf_token_count']
 return out

def main():
 ap=argparse.ArgumentParser()
 for k in ['attempt','dataset','report','audit','local','baseline']:ap.add_argument('--'+k,type=Path,required=True)
 a=ap.parse_args();out=a.attempt/'output'
 for p in [a.report,a.audit,a.local]:p.mkdir(parents=True,exist_ok=True)
 data=read(a.dataset);assert len(data)==10000
 assert sha(a.dataset)=='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
 terminal=read(out/'terminal.json');runtime=read(out/'runtime.json');lock=read(a.attempt/'execution.lock.json')
 assert terminal['status']=='COMPLETED' and terminal['W0_H0_restored'] and terminal['batches']==100
 assert runtime['lock_sha256']==sha(a.attempt/'execution.lock.json')
 assert runtime['native_entrypoint']['blue_git_commit']=='311b076a92e4ed0f14f5c8b4909732da781bc5f7'
 assert runtime['hparams']['blue'] is False and runtime['hparams']['layers']==[4,5,6,7,8]
 inventory=[]
 for m in terminal['manifest']:
  p=out/m['path'];assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256'],str(p)
  inventory.append(m)
 assert not list(out.rglob('failure.json')) and not list(out.rglob('rollback.json'))
 assert set(p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file())=={m['path'] for m in inventory}|{'terminal.json'}
 assert all(Path(m['path']).suffix in ['.json','.md'] for m in inventory)
 save(a.audit/'terminal-manifest-verification.json',dict(status='PASS',members=len(inventory),bytes=sum(m['bytes'] for m in inventory),terminal_sha256=sha(out/'terminal.json'),no_failure_or_rollback=True,no_tensor_artifacts=True))
 expected={};casebatch={int(r['case_id']):i//100+1 for i,r in enumerate(data)}
 for r in data:
  x=r['requested_rewrite'];cid=int(r['case_id'])
  for tag,ps in [('RS',[x['prompt'].format(x['subject'])]),('PS',r['paraphrase_prompts']),('NS',r['neighborhood_prompts'])]:
   assert len(ps)==WIDTH[tag]
   for j,p in enumerate(ps):expected[(tag,cid,j)]=digest([cid,j,p,x['target_new']['str'],x['target_true']['str']])
 def validate(m,tag,records):
  rows=m['rows'];wanted=[(int(r['case_id']),i) for r in records for i in range(WIDTH[tag])]
  assert [(r['case_id'],r['prompt_index']) for r in rows]==wanted
  for r in rows:
   assert r['identity']==expected[tag,r['case_id'],r['prompt_index']]
   assert all(math.isfinite(r[k]) for k in ['new_nll','true_nll','margin'])
   assert r['margin']==r['true_nll']-r['new_nll']
   assert r['success']==(r['true_nll']<r['new_nll'] if tag=='NS' else r['new_nll']<r['true_nll'])
   for d in ['new','true']:
    assert 0<=r[d+'_token_correct']<=r[d+'_token_count'] and r[d+'_token_count']>0
    assert r[d+'_strict']==(r[d+'_token_correct']==r[d+'_token_count'])
  s=summarize(rows,tag)
  for k,v in s.items():
   if k in m:assert math.isclose(v,m[k],rel_tol=1e-12,abs_tol=1e-12),(tag,k,v,m[k])
  return s
 costs=[];curve=[];currentcurve=[];links=[];at={t:[] for t in TAGS};commits=[];full={};layerrows=[]
 prev=None;checked_rows=0
 for b in range(1,101):
  root=out/f'B{b:03d}';e=read(root/'entry.json');c=read(root/'commit.json');n=read(root/'native-observation.json')
  assert e['request_ids']==[int(r['case_id']) for r in data[(b-1)*100:b*100]]
  assert e['request_hashes']==[digest(r['requested_rewrite']) for r in data[(b-1)*100:b*100]]
  assert c['batch']==b and c['status']=='BATCH_COMMITTED' and c['seen_requests']==b*100
  assert c['compute_z']==100 and c['solve_calls']==c['history_append_layers']==5
  assert not c['save_checkpoints'] and c['exact_resume']=='NOT_AVAILABLE'
  assert c['finite_W_H'] and c['nonfinite']==c['observer_mutation']==0 and c['cache_c_returned_same_object']
  assert c['observer_auxiliary_before']==c['observer_auxiliary_after'] and c['observer_context_rng_ledger_unchanged']
  assert c['auxiliary_entry']==e['auxiliary']
  assert c['entry']==dict(cache=e['signature']['cache_sha256'],weights={k:v['sha256'] for k,v in e['signature']['weights'].items()})
  if prev:
   assert c['entry']==prev['endpoint'] and e['auxiliary']==prev['auxiliary_endpoint']
   assert e['history_norms']==prev['history_norms'] and c['covariance_guard']==prev['covariance_guard']
  else:assert e['history_norms']==[0]*5
  assert [(z['case_id'],z['layer']) for z in n['z']]==[(cid,8) for cid in e['request_ids']]
  assert [(k['layer'],k['phase']) for k in n['keys']]==[(l,'pre_layer') for l in range(4,9)]+[(l,'post_all_layers') for l in range(4,9)]
  assert n['temporary_restore_exact'] and n['nonselected_pointer_version_exact']
  assert all(s['dtype']=='torch.float64' for s in n['solves'])
  for i,l in enumerate(range(4,9)):
   assert math.isfinite(c['history_norms'][i]) and c['history_norms'][i]>e['history_norms'][i]
   layerrows.append(dict(batch=b,layer=l,H_before=e['history_norms'][i],H_after=c['history_norms'][i],update_norm=c['layer_update_norms'][f'model.layers.{l}.mlp.down_proj.weight']))
  links.append(dict(batch=b,W_H_link=True,auxiliary_link=True,observer_unchanged=True,finite=True,post_all_layers=True))
  costs.append(dict(batch=b,**{k:c[k] for k in ['edit_seconds','target_seconds','key_seconds','solve_seconds','evaluation_seconds','peak_gpu_bytes']}))
  cur=read(root/'current.json');assert cur['before_after_exact'] and cur['evaluator_controller_influence']==0
  assert cur['cache_sha256']==c['endpoint']['cache'] and cur['weight_state']==c['endpoint']['weights']
  for t in TAGS:
   s=validate(cur['metrics'][t],t,data[(b-1)*100:b*100]);checked_rows+=s['denominator']
   at[t].extend(cur['metrics'][t]['rows']);currentcurve.append(dict(batch=b,metric=t,**s))
  seen=read(root/('seen-full.json' if b in FULL else 'seen-rewrite.json'));assert seen['requests']==b*100
  for t,m in seen['metrics'].items():
   s=validate(m,t,data[:b*100]);checked_rows+=s['denominator'];curve.append(dict(batch=b,metric=t,**s))
  if b in FULL:full[b]=seen
  commits.append(c);prev=c
  if b%20==0:print('CPU_REVIEW_BATCH',b,flush=True)
 assert sum(c['compute_z'] for c in commits)==terminal['z_calls']==10000
 assert sum(c['solve_calls'] for c in commits)==terminal['solve_calls']==500
 assert sum(c['history_append_layers'] for c in commits)==terminal['history_append_layers']==500
 for name,rows in [('batch-cost.csv',costs),('history-layer.csv',layerrows),('chain-checks.csv',links),('current-metrics.csv',currentcurve),('all-seen-metrics.csv',curve)]:table(a.report/name,rows)
 final={t:full[100]['metrics'][t]['rows'] for t in TAGS};summary=read(out/'summary.json')
 finalsummary={t:summarize(final[t],t) for t in TAGS}
 for t in TAGS:
  for k,v in finalsummary[t].items():
   if k in summary['final'][t]:assert math.isclose(v,summary['final'][t][k],rel_tol=1e-12,abs_tol=1e-12)
 paired=[];ids={};cohorts=[];stratarows=[];quantiles=[]
 for t in TAGS:
  d='true' if t=='NS' else 'new';fm={r['identity']:r for r in final[t]}
  for scope,before in [('at_write',at[t]),('first500_W5_to_W100',full[5]['metrics'][t]['rows'])]:
   after=[fm[r['identity']] for r in before]
   for field in ['success',d+'_strict']:
    lost=[r['identity'] for r,s in zip(before,after) if r[field] and not s[field]];gained=[r['identity'] for r,s in zip(before,after) if not r[field] and s[field]]
    row=dict(scope=scope,metric=t,criterion='preference' if field=='success' else 'TF_strict',denominator=len(before),before=sum(r[field] for r in before),after=sum(r[field] for r in after),lost=len(lost),gained=len(gained),lost_sha256=digest(lost),gained_sha256=digest(gained))
    paired.append(row);ids[scope+'_'+t+'_'+field]=dict(lost=lost,gained=gained)
    ref=summary[scope][t] if field=='success' else summary[scope][t].get('tf_strict')
    if ref:assert ref['lost']==len(lost) and ref['gained']==len(gained) and ref['after_success']==row['after']
  for b in range(1,101):
   rows=[r for r in final[t] if casebatch[r['case_id']]==b]
   cohorts.append(dict(birth_batch=b,age=100-b,metric=t,**summarize(rows,t)))
  for k in ['new_nll','true_nll','margin']:
   v=np.array([r[k] for r in final[t]]);q=np.quantile(v,[0,.25,.5,.75,.9,.99,1])
   quantiles.append(dict(metric=t,value=k,**dict(zip(['min','q25','median','q75','p90','p99','max'],map(float,q)))))
 latest={}
 for r in data:
  x=r['requested_rewrite'];latest[x['subject'],x['relation_id']]=x['target_new']['str']
 active={int(r['case_id']) for r in data if latest[r['requested_rewrite']['subject'],r['requested_rewrite']['relation_id']]==r['requested_rewrite']['target_new']['str']}
 for t in TAGS:
  for name,flag in [('active',True),('superseded',False)]:
   s=summarize([r for r in final[t] if (r['case_id'] in active)==flag],t)
   for k,v in s.items():
    if k in summary['strata'][t][name]:assert math.isclose(v,summary['strata'][t][name][k],rel_tol=1e-12,abs_tol=1e-12)
   stratarows.append(dict(stratum=name,metric=t,**s))
 table(a.report/'paired-retention.csv',paired);save(a.local/'paired-identities.json',ids)
 table(a.report/'final-cohorts.csv',cohorts);table(a.report/'active-superseded.csv',stratarows);table(a.report/'nll-distributions.csv',quantiles)
 base=list(csv.DictReader((a.baseline/'MEMIT-cumulative-metrics.csv').open()));base=[r for r in base if r['arm']=='BASE_MEMIT' and r['batch']=='100'];assert len(base)==3
 comparison=[]
 for r in base:
  t=r['metric'];s=finalsummary[t];d='true' if t=='NS' else 'new';assert int(r['denominator'])==s['denominator']
  comparison.append(dict(metric=t,baseline_job=42658,history_job=54007,baseline_rate=float(r['rate']),history_rate=s['rate'],difference_pp=100*(s['rate']-float(r['rate'])),baseline_TF_micro=int(r[d+'_token_correct'])/int(r[d+'_token_den']),history_TF_micro=s['tf_token_micro'],baseline_TF_strict=int(r[d+'_strict_num'])/int(r[d+'_strict_den']),history_TF_strict=s['tf_strict'],baseline_desired_nll=float(r[d+'_nll_prompt_mean']),history_desired_nll=s['desired_nll']))
 table(a.report/'baseline-comparison.csv',comparison)
 cost={k:sum(c[k] for c in costs) for k in ['edit_seconds','target_seconds','key_seconds','solve_seconds','evaluation_seconds']}
 accounting=list(csv.DictReader((a.audit/'accounting.txt').open(),delimiter='|'))
 parent=next(r for r in accounting if r['JobID']=='54007');batch=next(r for r in accounting if r['JobID']=='54007.batch')
 assert parent['State']=='COMPLETED' and parent['ExitCode']=='0:0' and 'gres/gpu=1' in parent['AllocTRES']
 elapsed=int(parent['ElapsedRaw']);assert batch['MaxRSS'].endswith('M')
 cost.update(runtime_seconds=terminal['seconds'],allocated_GPU_seconds=elapsed,allocated_GPU_hours=elapsed/3600,failed_parent_GPU_seconds=451,total_attempts_GPU_seconds=elapsed+451,total_attempts_GPU_hours=(elapsed+451)/3600,peak_GPU_GiB=max(c['peak_gpu_bytes'] for c in costs)/2**30,batch_MaxRSS_MiB=float(batch['MaxRSS'][:-1]),raw_output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
 cost['edit_other_seconds']=cost['edit_seconds']-sum(cost[k] for k in ['target_seconds','key_seconds','solve_seconds'])
 cost['load_hash_io_reducer_unseparated_seconds']=terminal['seconds']-cost['edit_seconds']-cost['evaluation_seconds']
 report=dict(status='COMPLETED_CPU_REVIEW_PASS',job=54007,source_commit='3a904be9261d162239c6b780a62c52f3e59a9142',lock_sha256=sha(a.attempt/'execution.lock.json'),terminal_sha256=sha(out/'terminal.json'),validated_metric_rows=checked_rows,manifest_members=len(inventory),commits=100,chain_links=99,z_calls=10000,solves=500,history_appends=500,active_requests=len(active),superseded_requests=10000-len(active),final=finalsummary,cost=cost,save_checkpoints=False,exact_resume='NOT_AVAILABLE',baseline_paired='NOT_AVAILABLE_NO_LOCAL_BASELINE_ROWS',independent_red_agent=False,CPU_reducer='independent arithmetic/identity checks; existing raw only; no GPU',gpu=runtime['gpu'])
 save(a.report/'summary.json',report)
 save(a.audit/'review-input-manifest.json',dict(inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in [a.dataset,a.attempt/'execution.lock.json',out/'terminal.json',out/'summary.json',a.baseline/'MEMIT-cumulative-metrics.csv',a.baseline/'new-source-config-compatibility.csv']],script_sha256=sha(__file__)))
 # Static scientific figures exported alongside report, no interactive/live reads.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,3,figsize=(13,3.6))
 for z,t in zip(ax,TAGS):
  r=[r for r in curve if r['metric']==t];z.plot([r['batch'] for r in r],[100*r['rate'] for r in r],label='History all-seen')
  r=[r for r in currentcurve if r['metric']==t];z.plot([r['batch'] for r in r],[100*r['rate'] for r in r],alpha=.5,label='History current')
  r=[r for r in csv.DictReader((a.baseline/'MEMIT-cumulative-metrics.csv').open()) if r['arm']=='BASE_MEMIT' and r['metric']==t]
  z.plot([int(r['batch']) for r in r],[100*float(r['rate']) for r in r],linestyle='--',label='BASE_MEMIT all-seen')
  z.set(title=t,xlabel='Batch',ylabel='NLL preference (%)',ylim=(0,102));z.grid(alpha=.2)
 ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(a.report/'trajectory.png',dpi=160);plt.close(fig)
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
