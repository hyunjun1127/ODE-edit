"""Report-only matched 2k factorial and diagnostic panels; no model execution."""
import csv,json
from pathlib import Path
from .reducer import read,save,transitions,verify_metric
from project.run_scripts.memit_history_lifelong.metrics import summarize

ARMS=['000','100','010','110','001','101','011','111']

def endpoint_at(output,arm,n):
 root=Path(output)/'cells'/('main_'+arm);term=read(root/'terminal.json')
 if term['status'].startswith('BLOCKED'):return None,dict(status=term['status'])
 if term['status']=='NOT_FIRED':
  result,origin=endpoint_at(output,term['parent'].removeprefix('main_'),n)
  return result,dict(status='NOT_FIRED',alias=term['parent'],endpoint=n,artifact=origin.get('artifact'),independent_replication=False)
 p=root/f'C{n:05d}/all-seen.json'
 if not p.exists():
  lineage=read(root/'lineage.json');assert lineage.get('parent') and n<=lineage['prefix_requests']
  result,origin=endpoint_at(output,lineage['parent'].removeprefix('main_'),n)
  return result,dict(status='SHARED_PREFIX',parent=lineage['parent'],endpoint=n,artifact=origin.get('artifact'),independent_replication=False)
 result=read(p)
 assert result['requests']==n and {k:m['denominator'] for k,m in result['metrics'].items()}=={'RS':n,'PS':2*n,'NS':10*n}
 return result,dict(status='OBSERVED',artifact=str(p),endpoint=n,independent_replication=False)

def metric(rows,tag):return dict(rows=rows,**summarize(rows,tag))
def subset(result,ids):
 return {tag:[r for r in m['rows'] if r['case_id'] in ids] for tag,m in result['metrics'].items()}
def compare(before,after,tag):
 b,a=metric(before,tag),metric(after,tag);verify_metric(b,tag);verify_metric(a,tag)
 desired='true' if tag=='NS' else 'new'
 return dict(denominator=a['denominator'],preference_delta_pp=100*(a['rate']-b['rate']),
  TF_token_micro_delta_pp=100*(a['tf_token_micro']-b['tf_token_micro']),TF_prompt_macro_delta_pp=100*(a['tf_prompt_macro']-b['tf_prompt_macro']),
  TF_strict_delta_pp=100*(a['tf_strict']-b['tf_strict']),new_NLL_difference=a['new_nll']-b['new_nll'],true_NLL_difference=a['true_nll']-b['true_nll'],desired_NLL_difference=a['desired_nll']-b['desired_nll'],
  preference_paired=transitions(before,after),TF_strict_paired=transitions([dict(r,success=r[desired+'_strict']) for r in before],[dict(r,success=r[desired+'_strict']) for r in after]))

def factorial(output,dest):
 endpoints={};origins={};table=[]
 for arm in ARMS:
  result,origin=endpoint_at(output,arm,2000);origins[arm]=origin
  if result is None:
   for tag in ['RS','PS','NS']:table.append(dict(arm=arm,status=origin['status'],tag=tag,denominator=None,preference=None,TF_micro=None,TF_macro=None,TF_strict=None,new_NLL=None,true_NLL=None,desired_NLL=None))
   continue
  endpoints[arm]=result
  for tag,m in result['metrics'].items():
   v=verify_metric(m,tag);table.append(dict(arm=arm,status=origin['status'],tag=tag,denominator=v['denominator'],preference=v['rate'],TF_micro=v['tf_token_micro'],TF_macro=v['tf_prompt_macro'],TF_strict=v['tf_strict'],new_NLL=v['new_nll'],true_NLL=v['true_nll'],desired_NLL=v['desired_nll']))
 contrasts={}
 for bit,label in [(0,'J'),(1,'Z'),(2,'R')]:
  for before in ARMS:
   if before[bit]!='0':continue
   after=before[:bit]+'1'+before[bit+1:];key=after+'-'+before
   if before not in endpoints or after not in endpoints:contrasts[key]=dict(status='BLOCKED',factor=label);continue
   contrasts[key]=dict(status='OBSERVED_OR_ALIAS',factor=label,endpoint=2000,metrics={tag:compare(endpoints[before]['metrics'][tag]['rows'],endpoints[after]['metrics'][tag]['rows'],tag) for tag in ['RS','PS','NS']},origin_before=origins[before],origin_after=origins[after])
 interactions={}
 for first,second,label in [(0,1,'JxZ'),(0,2,'JxR'),(1,2,'ZxR')]:
  third=next(i for i in range(3) if i not in [first,second])
  for fixed in ['0','1']:
   arms=[]
   for a,b in [(0,0),(1,0),(0,1),(1,1)]:
    bits=['0']*3;bits[first]=str(a);bits[second]=str(b);bits[third]=fixed;arms.append(''.join(bits))
   key=label+'_remaining_'+fixed
   interactions[key]=dict(arms=arms,status='BLOCKED')
   if all(a in endpoints for a in arms):
    interactions[key]=dict(arms=arms,status='DESCRIPTIVE_FIXED_TRAJECTORY',difference_in_differences_pp={tag:100*sum(sign*endpoints[arm]['metrics'][tag]['rate'] for arm,sign in zip(arms,[1,-1,-1,1])) for tag in ['RS','PS','NS']},independent_order_generalization=False)
 save(dest/'factorial-2k.json',dict(origins=origins,contrasts=contrasts,interactions=interactions,refresh_nonfiring_not_evidence_of_no_refresh_effect=True))
 write_csv(dest/'factorial-2k.csv',table)
 return table

def write_csv(path,rows):
 if not rows:return
 with Path(path).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def diagnostic_panels(output,dest,records,cells):
 rows=[];pairings={};byordinal={int(r['case_id']):i for i,r in enumerate(records)}
 for cell in cells:
  if cell['family']=='main':continue
  name=cell['cell_id'];root=Path(output)/'cells'/name;anchor=int(cell['anchor_requests']);lineage=read(root/'lineage.json')
  entry=read(root/'entry-next1000.json')
  for n in [0,10,100,500,1000]:
   current=entry if n==0 else read(root/f'C{anchor+n:05d}/continuation.json')
   for tag,m in current['metrics'].items():
    v=verify_metric(m,tag);rows.append(dict(cell=name,anchor=anchor,after_requests=n,population='next1000_entry' if n==0 else 'continuation',quartile='ALL',tag=tag,denominator=v['denominator'],preference=v['rate'],TF_micro=v['tf_token_micro'],TF_macro=v['tf_prompt_macro'],TF_strict=v['tf_strict'],desired_NLL=v['desired_nll']))
   if not anchor:continue
   past=read(root/'entry-past400.json') if n==0 else read(root/f'C{anchor+n:05d}/past400.json')
   for quartile in range(4):
    ids={int(records[i]['case_id']) for i in lineage['past_panel_ordinals'] if 4*i//anchor==quartile};assert len(ids)==100
    for tag,rr in subset(past,ids).items():
     v=verify_metric(metric(rr,tag),tag);rows.append(dict(cell=name,anchor=anchor,after_requests=n,population='past400',quartile=quartile,tag=tag,denominator=v['denominator'],preference=v['rate'],TF_micro=v['tf_token_micro'],TF_macro=v['tf_prompt_macro'],TF_strict=v['tf_strict'],desired_NLL=v['desired_nll']))
 # Same anchor, same endpoints and same request/prompt identities only.
 comparisons=[]
 for anchor in [0,1000,5000]:
  comparisons += [(f'writer_{anchor}_divisor',f'writer_{anchor}_{mode}') for mode in ['joint','frozen_upper_joint','energy_matched_divisor']]
  comparisons += [(f'writer_{anchor}_joint',f'writer_{anchor}_frozen_upper_joint')]
 for anchor in [1000,3000,5000,7000]:comparisons += [(f'history_{anchor}_stale',f'history_{anchor}_forced_refresh')]
 for before,after in comparisons:
  a=next(c for c in cells if c['cell_id']==before);anchor=int(a['anchor_requests'])
  for n in [10,100,500,1000]:
   for population in ['continuation']+(['past400'] if anchor else []):
    b=read(Path(output)/'cells'/before/f'C{anchor+n:05d}'/(population+'.json'));v=read(Path(output)/'cells'/after/f'C{anchor+n:05d}'/(population+'.json'))
    pairings[f'{after}-vs-{before}@{n}:{population}']={tag:compare(b['metrics'][tag]['rows'],v['metrics'][tag]['rows'],tag) for tag in ['RS','PS','NS']}
 write_csv(dest/'diagnostic-panels.csv',rows);save(dest/'diagnostic-paired.json',pairings)
 return len(pairings)

def make(output,cells,records):
 dest=Path(output)/'reduced/secondary';dest.mkdir(parents=True,exist_ok=False)
 table=factorial(output,dest);pairs=diagnostic_panels(output,dest,records,cells)
 lines=['# 동일 endpoint 보조 분석','', '2k 8논리arm을 같은2000/4000/20000 분모로 비교한다. 미발동 alias는 독립 반복이 아니다.', '', '|arm|상태|metric|분모|preference %|TF strict %|','|---|---|---|---:|---:|---:|']
 for r in table:
  pref='NA' if r['preference'] is None else f"{100*r['preference']:.3f}";strict='NA' if r['TF_strict'] is None else f"{100*r['TF_strict']:.3f}"
  lines.append(f"|{r['arm']}|{r['status']}|{r['tag']}|{r['denominator']}|{pref}|{strict}|")
 lines += ['',f'같은anchor의 writer/history paired 관측 {pairs}개. Past400의 네분위는 각각R100/P200/N1000이며 같은비중으로 집계한다.',
  '2k J/Z/R contrast와 interaction은 고정 모델·순서에 대한 기술적 요약이다. 장기 전체factorial 또는 독립seed 효과로 해석하지 않는다.',
  'TF는 teacher-forced 지표이며 자유생성 정확도가 아니다. 통계적 bootstrap CI를 추가로 주장하지 않았다.']
 (dest/'report-ko.md').write_text('\n'.join(lines)+'\n')
 return dict(factorial_logical_cells=8,endpoint=2000,diagnostic_pairings=pairs,CPU_only=True)
