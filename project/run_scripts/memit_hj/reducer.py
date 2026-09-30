"""CPU-only independent scalar/identity/coverage reducer. No model imports."""
import argparse,csv,hashlib,json,math
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,allow_nan=False,sort_keys=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_metric(m,tag):
 rows=m['rows'];n=len(rows);desired='true' if tag=='NS' else 'new'
 assert n and n==m['denominator'] and len({r['identity'] for r in rows})==n
 success=correct=count=strict=0
 for r in rows:
  assert all(math.isfinite(r[k]) for k in ['new_nll','true_nll','margin'])
  good=r['true_nll']<r['new_nll'] if tag=='NS' else r['new_nll']<r['true_nll']
  assert r['success']==good and r['margin']==r['true_nll']-r['new_nll']
  c,t=r[desired+'_token_correct'],r[desired+'_token_count'];assert 0<=c<=t and t>0
  assert r[desired+'_strict']==(c==t)
  success+=good;correct+=c;count+=t;strict+=c==t
 assert success==m['numerator'] and correct==m['tf_token_correct'] and count==m['tf_token_count']
 assert strict==m['tf_strict_numerator']
 assert abs(m['rate']-success/n)<1e-14 and abs(m['tf_token_micro']-correct/count)<1e-14
 assert abs(m['tf_prompt_macro']-sum(r[desired+'_token_correct']/r[desired+'_token_count'] for r in rows)/n)<1e-14
 for target in ['true','new']:
  expected=sum(r[target+'_nll'] for r in rows)/n
  assert abs(m[target+'_nll']-expected)<=1e-12*max(1.,abs(expected)),'NLL_REDUCER_MISMATCH'
 assert m['desired_nll']==m[desired+'_nll']
 return dict(numerator=success,denominator=n,rate=success/n,tf_token_micro=correct/count,
  tf_prompt_macro=m['tf_prompt_macro'],tf_strict=strict/n,new_nll=m['new_nll'],true_nll=m['true_nll'],desired_nll=m['desired_nll'],
  desired_nll_quantiles=quantiles([r[desired+'_nll'] for r in rows]))
def quantiles(v):
 v=sorted(v)
 def q(p):
  a=p*(len(v)-1);i=int(a);j=min(i+1,len(v)-1);return v[i]+(v[j]-v[i])*(a-i)
 return {str(p):q(p) for p in [.05,.5,.9,.95,.99]}
def expected_identities(records):
 out={tag:{} for tag in ['RS','PS','NS']}
 for r in records:
  q=r['requested_rewrite'];case=int(r['case_id'])
  groups={'RS':[q['prompt'].format(q['subject'])],'PS':r['paraphrase_prompts'],'NS':r['neighborhood_prompts']}
  for tag,prompts in groups.items():
   for j,prompt in enumerate(prompts):
    payload=[case,j,prompt,q['target_new']['str'],q['target_true']['str']]
    out[tag][case,j]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
 return out

def verify_identities(result,expected):
 for tag,m in result['metrics'].items():
  for r in m['rows']:assert r['identity']==expected[tag][r['case_id'],r['prompt_index']],'RAW_INPUT_IDENTITY'

def verify_cell(output,cell,terminal_record=None,expected=None):
 root=Path(output)/'cells'/cell['cell_id'];terminal=terminal_record or read(root/'terminal.json')
 if terminal['status']!='COMPLETED':return terminal['status']
 lineage=read(root/'lineage.json');bs=int(cell['batch_size']);end=int(cell['total_requests'])
 start=lineage.get('prefix_requests',lineage.get('anchor',0));stop=end if cell['family']=='main' else start+end
 paths=sorted(root.glob('C*/commit.json'));assert len(paths)==(stop-start)//bs,(cell['cell_id'],len(paths),stop,start)
 for n,p in enumerate(paths):
  r=read(p);assert r['start']==start+n*bs and r['stop']==start+(n+1)*bs
  assert r['history_append_layers']==5 and r['finite_W_H'] and r['observer_nonmutating']
  m=read(p.parent/'current.json')
  assert m['requests']==bs
  if expected:verify_identities(m,expected)
  entry=read(p.parent/'entry.json')
  assert [x['case_id'] for x in m['metrics']['RS']['rows']]==entry['request_ids'],'CURRENT_ORDER'
  for tag,denom in [('RS',bs),('PS',bs*2),('NS',bs*10)]:assert verify_metric(m['metrics'][tag],tag)['denominator']==denom
  if bs==100:
   seen=read(p.parent/'all-seen.json');full=r['stop'] in [100,500]+list(range(1000,10001,1000))
   if expected:verify_identities(seen,expected)
   assert set(seen['metrics'])==({'RS','PS','NS'} if full else {'RS'})
   for tag,m in seen['metrics'].items():assert verify_metric(m,tag)['denominator']==r['stop']*{'RS':1,'PS':2,'NS':10}[tag]
  elif r['stop']-start in [10,100,500,1000]:
   continuation=read(p.parent/'continuation.json')
   if expected:verify_identities(continuation,expected)
   for tag,m in continuation['metrics'].items():assert verify_metric(m,tag)['denominator']==(r['stop']-start)*{'RS':1,'PS':2,'NS':10}[tag]
   if start:
    panel=read(p.parent/'past400.json')
    if expected:verify_identities(panel,expected)
    for tag,m in panel['metrics'].items():assert verify_metric(m,tag)['denominator']==400*{'RS':1,'PS':2,'NS':10}[tag]
 return dict(status='PASS',physical_requests=stop-start,physical_writes=len(paths))
def transitions(before,after,subset=None):
 b={r['identity']:r for r in before};a={r['identity']:r for r in after};assert b.keys()==a.keys()
 ids=list(b) if subset is None else [k for k in b if k in subset]
 out=dict(denominator=len(ids),lost=[],gained=[],retained=[],initial_failed=[])
 for k in ids:
  if b[k]['success'] and not a[k]['success']:out['lost'].append(k)
  elif not b[k]['success'] and a[k]['success']:out['gained'].append(k)
  elif b[k]['success']:out['retained'].append(k)
  else:out['initial_failed'].append(k)
 out['retention_given_before_success']=len(out['retained'])/(len(out['retained'])+len(out['lost'])) if out['retained'] or out['lost'] else None
 return out
def prefix_currents(output,cell_id):
 root=Path(output)/'cells'/cell_id;lin=read(root/'lineage.json');parts=[]
 if lin.get('parent') and lin.get('prefix_requests'):
  parts=[r for r in prefix_currents(output,lin['parent']) if r[0]<=lin['prefix_requests']]
 for p in sorted(root.glob('C*/current.json')):parts.append((int(p.parent.name[1:]),read(p)))
 return parts
def endpoint(output,cell_id,n=None):
 root=Path(output)/'cells'/cell_id;t=read(root/'terminal.json')
 if t['status']=='NOT_FIRED':return endpoint(output,t['parent'],t['endpoint'])
 if n is None:n=t['cursor']
 p=root/f'C{n:05d}'/'all-seen.json'
 if not p.exists():
  l=read(root/'lineage.json');assert l.get('parent') and n<=l['prefix_requests'];return endpoint(output,l['parent'],n)
 return read(p)
def reduce(output,cells_path,dataset_path):
 output=Path(output);cells=list(csv.DictReader(Path(cells_path).open()));records=read(dataset_path)
 w0=read(output/'W0-all10k.json');w0map={k:{r['identity']:r for r in m['rows']} for k,m in w0['metrics'].items()}
 expected=expected_identities(records);verify_identities(w0,expected)
 coverage={c['cell_id']:verify_cell(output,c,expected=expected) for c in cells};save(output/'reduced/coverage.json',coverage)
 summary=[];paired={};cost={};allend={};atmaps={}
 for c in cells:
  name=c['cell_id'];root=output/'cells'/name;term=read(root/'terminal.json')
  if term['status'].startswith('BLOCKED'):continue
  if c['family']=='main':
   end=endpoint(output,name);allend[name]=end
   at={tag:{} for tag in ['RS','PS','NS']}
   owner=term['parent'] if term['status']=='NOT_FIRED' else name
   for cursor,current in prefix_currents(output,owner):
    if cursor>int(c['total_requests']):continue
    for tag,m in current['metrics'].items():at[tag].update({r['identity']:r for r in m['rows']})
   atmaps[name]=at
   detail={}
   for tag,m in end['metrics'].items():
    stats=verify_metric(m,tag);summary.append(dict(cell=name,family='main',tag=tag,**{k:v for k,v in stats.items() if k!='desired_nll_quantiles'}))
    before=[at[tag][r['identity']] for r in m['rows']];base=[w0map[tag][r['identity']] for r in m['rows']]
    detail[tag]=dict(at_write_to_current=transitions(before,m['rows']),W0_to_current=transitions(base,m['rows']),
     atwrite_to_current_among_W0_correct=transitions(before,m['rows'],{r['identity'] for r in base if r['success']}),tails=stats['desired_nll_quantiles'])
    first500={int(r['case_id']) for r in records[:500]};latest={}
    for r in records[:int(c['total_requests'])]:
     q=r['requested_rewrite'];latest[(q['subject'],q['relation_id'])]=q['target_new']['str']
    active={r['case_id'] for r in records[:int(c['total_requests'])] if latest[(r['requested_rewrite']['subject'],r['requested_rewrite']['relation_id'])]==r['requested_rewrite']['target_new']['str']}
    detail[tag]['strata']={label:dict(n=len(v),success=sum(r['success'] for r in v)) for label,v in {
     'first500':[r for r in m['rows'] if r['case_id'] in first500],
     'active':[r for r in m['rows'] if r['case_id'] in active],
     'superseded':[r for r in m['rows'] if r['case_id'] not in active]}.items()}
    birth={int(r['case_id']):i//100+1 for i,r in enumerate(records[:int(c['total_requests'])])}
    detail[tag]['birth_batch']={str(b):dict(age_batches=int(c['total_requests'])//100-b,
     paired=transitions(before,m['rows'],{r['identity'] for r in m['rows'] if birth[r['case_id']]==b})) for b in sorted(set(birth.values()))}
    early=endpoint(output,owner,500)['metrics'][tag]['rows']
    current500=[r for r in m['rows'] if r['case_id'] in first500]
    detail[tag]['W5_first500_to_endpoint']=transitions(early,current500)
   save(output/'reduced'/f'{name}-retention.json',detail)
  else:
   anchor=int(c['anchor_requests']);last=root/f'C{anchor+1000:05d}'
   continuation=read(last/'continuation.json');entry=read(root/'entry-next1000.json');detail={}
   for population,result,baseline in [('continuation1000',continuation,entry)]+([('past400',read(last/'past400.json'),read(root/'entry-past400.json'))] if anchor else []):
    detail[population]={}
    for tag,m in result['metrics'].items():
     stats=verify_metric(m,tag);summary.append(dict(cell=name,family=c['family']+':'+population,tag=tag,**{k:v for k,v in stats.items() if k!='desired_nll_quantiles'}))
     base=[w0map[tag][r['identity']] for r in m['rows']]
     detail[population][tag]=dict(entry_to_current=transitions(baseline['metrics'][tag]['rows'],m['rows']),W0_to_current=transitions(base,m['rows']),
      entry_to_current_among_W0_correct=transitions(baseline['metrics'][tag]['rows'],m['rows'],{r['identity'] for r in base if r['success']}))
   save(output/'reduced'/f'{name}-retention.json',detail)
  commits=[read(p) for p in root.glob('C*/commit.json')]
  cost[name]=dict(writes=len(commits),requests=sum(r['requests'] for r in commits),step_seconds=sum(r['total_step_seconds'] for r in commits),
   write_seconds=sum(r['write_seconds'] for r in commits),observer_seconds=sum(r['observer_seconds'] for r in commits),nested_timers_are_not_additive=True)
 for a,b in [('main_100','main_000'),('main_110','main_100'),('main_111','main_110')]:
  if a not in allend or b not in allend:continue
  paired[a+'-'+b]={}
  for tag in ['RS','PS','NS']:
   x,y=allend[a]['metrics'][tag],allend[b]['metrics'][tag]
   assert x['denominator']==y['denominator']
   t=transitions(y['rows'],x['rows'])
   common={k for k,r in atmaps[a][tag].items() if r['success'] and atmaps[b][tag][k]['success']}
   paired[a+'-'+b][tag]=dict(delta_pp=100*(x['rate']-y['rate']),paired=t,common_at_write_success=transitions(y['rows'],x['rows'],common))
 primary=paired.get('main_100-main_000')
 decision=bool(primary and primary['PS']['delta_pp']>=2 and primary['RS']['delta_pp']>=-1 and primary['NS']['delta_pp']>=-1)
 save(output/'reduced/paired.json',paired);save(output/'reduced/cost.json',cost)
 save(output/'reduced/primary-contract.json',dict(available=bool(primary),proposed_practical_effect_met=decision,performance_execution_gate=False,independent_seed_claim=False))
 p=output/'reduced/summary.csv'
 with p.open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
 text=['# MEMIT HJ v2 실행 사실 보고','',f'검산된 cell: {len(coverage)}. BLOCKED_Z/NOT_FIRED는 독립 완료 실행과 구분한다.','',
  '|cell|population|metric|분모|preference %|TF micro %|TF strict %|','|---|---|---|---:|---:|---:|---:|']
 text += [f"|{r['cell']}|{r['family']}|{r['tag']}|{r['denominator']}|{100*r['rate']:.3f}|{100*r['tf_token_micro']:.3f}|{100*r['tf_strict']:.3f}|" for r in summary]
 text += ['', 'TF는 teacher-forced 정확도이며 자유생성 평가가 아니다. 과거54007/AlphaEdit/BLUE는 이 matched 대조의 새 실행이 아니다.',
  '통계적 독립 재학습/순서 강건성은 검증하지 않았다. 상세 과학적 해석은 GH에 맡긴다.']
 (output/'reduced/report-ko.md').write_text('\n'.join(text)+'\n')
 return dict(coverage=coverage,primary=decision,physical_requests=sum(x['requests'] for x in cost.values()),physical_writes=sum(x['writes'] for x in cost.values()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cells',required=True);p.add_argument('--dataset',required=True);a=p.parse_args();reduce(a.output,a.cells,a.dataset)
