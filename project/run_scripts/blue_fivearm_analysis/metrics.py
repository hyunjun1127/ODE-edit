from collections import defaultdict
from .common import *

TAGS={'RS':'rewrite','PS':'rephrase','NS':'locality'}
MULT={'RS':1,'PS':2,'NS':10}

def success(new,true,tag):return true<new if tag=='NS' else new<true

def normalize_j(raw):
 groups={}
 for tag,category in TAGS.items():
  kinds={side:{(r['case_id'],r['prompt_index']):r for r in raw['rows'] if r['kind']==category+'_target_'+side} for side in ('new','true')}
  assert len(kinds['new'])==sum(r['kind']==category+'_target_new' for r in raw['rows'])
  assert kinds['new'].keys()==kinds['true'].keys()
  rows=[]
  for key,a in kinds['new'].items():
   b=kinds['true'][key]
   rows.append(dict(case_id=key[0],prompt_index=key[1],identity=digest([key,a['input_identity_sha256'],b['input_identity_sha256']]),
    new_nll=a['nll'],true_nll=b['nll'],success=success(a['nll'],b['nll'],tag),
    new_strict=a['all_tokens_correct'],true_strict=b['all_tokens_correct'],
    new_token_correct=a['correct_token_count'],new_token_count=a['target_token_count'],
    true_token_correct=b['correct_token_count'],true_token_count=b['target_token_count']))
  groups[tag]=dict(rows=rows)
 return dict(metrics=groups)

def reduce_eval(raw,arm,batch,scope,expected_ids):
 result=dict(arm=arm,batch=batch,scope=scope,request_count=len(expected_ids),evidence='LOCAL_PROMPT_PAIRS_INDEPENDENT_REDUCER')
 for tag,metric in raw['metrics'].items():
  rows=metric['rows'];category=TAGS[tag]
  expected=[(i,p) for i in expected_ids for p in range(MULT[tag])]
  assert [(r['case_id'],r['prompt_index']) for r in rows]==expected,(arm,batch,tag,'order')
  assert len({r['identity'] for r in rows})==len(rows)
  bits=[success(r['new_nll'],r['true_nll'],tag) for r in rows]
  assert bits==[r['success'] for r in rows]
  n=sum(bits);den=len(rows)
  if 'numerator' in metric:assert (n,den)==(metric['numerator'],metric['denominator'])
  result.update({tag+'_num':n,tag+'_den':den,tag+'_rate':n/den,tag+'_bit_order_sha256':digest([(r['identity'],b) for r,b in zip(rows,bits)])})
  grouped=defaultdict(list)
  for r,b in zip(rows,bits):grouped[r['case_id']].append(b)
  result[tag+'_strict_num']=sum(all(v) for v in grouped.values());result[tag+'_strict_den']=len(grouped)
  margin=[(r['new_nll']-r['true_nll']) if tag=='NS' else (r['true_nll']-r['new_nll']) for r in rows]
  result.update({category+'_margin_'+k:v for k,v in stats(margin).items()})
  for side in ('new','true'):
   prefix=category+'_target_'+side
   result.update({prefix+'_nll_'+k:v for k,v in stats([r[side+'_nll'] for r in rows]).items()})
   result[prefix+'_row_count']=den
   result[prefix+'_all_tokens_correct_count']=sum(r[side+'_strict'] for r in rows)
   result[prefix+'_correct_token_count']=sum(r[side+'_token_correct'] for r in rows)
   result[prefix+'_target_token_denominator']=sum(r[side+'_token_count'] for r in rows)
 return result

def headline(rows):
 result=[]
 for arm in ARMS:
  r=next(r for r in rows if r['arm']==arm)
  z={'arm':arm,'state':'FINAL_W10_ALL1000'}
  for t in TAGS:z[t]=f"{r[t+'_num']}/{r[t+'_den']} ({100*float(r[t+'_rate']):.2f}%)"
  z['rewrite_acc']=f"{r['rewrite_target_new_all_tokens_correct_count']}/{r['RS_den']}"
  z['rephrase_acc']=f"{r['rephrase_target_new_all_tokens_correct_count']}/{r['PS_den']}"
  result.append(z)
 return result

def first(repo,out):
 inventory=[];final=[]
 ref=repo/REF_REL;pub=repo/REVIEW_REL
 for p,m in [(ref,'package-manifest.json'),(pub,'analysis-manifest.json')]:inventory+=verify_members(p,read(p/m)['members'],str(p))
 sample=read(ref/'sample.lock.json');assert digest(sample['records'])==SAMPLE_ROOT
 ids=[r['case_id'] for r in sample['records']]
 for arm,root in BLUE_ROOTS.items():
  terminal=read(root/'terminal.json')
  assert terminal['status']=='TERMINAL_VALID' and (terminal['batches'],terminal['requests'])==(10,1000)
  inventory+=verify_members(root,terminal['manifest_members'],arm)
  final.append(reduce_eval(read(root/'B10/seen-full.json'),arm,10,'FINAL_W10_ALL1000',ids))
 # Rehash the local Llama JVP-L8 chain against its previous inventory; never claim remote raw access.
 old=read(pub/'raw-member-inventory.json')['members']
 selected=[m for m in old if m['path'].startswith('chain-4-llama3-8b-inst-L8_ONLY_NATIVE/')]
 inventory+=verify_members(JROOT,selected,'JVP_L8')
 j=reduce_eval(normalize_j(read(JCHAIN/'batch-10/seen-full.json')),'JVP_L8',10,'FINAL_W10_ALL1000',ids);final.append(j)
 for r in csvread(pub/'final_metrics.csv'):
  if r['alias']=='llama3-8b-inst' and r['arm'] in ('JV_NATIVE','O_NATIVE'):
   q=dict(r);q['arm']='JVP' if r['arm']=='JV_NATIVE' else 'O_NATIVE';final.append(q)
  if r['alias']=='llama3-8b-inst' and r['arm']=='L8_ONLY_NATIVE':
   for tag in TAGS:assert (int(r[tag+'_num']),int(r[tag+'_den']))==(j[tag+'_num'],j[tag+'_den'])
 csvwrite(out/'raw-member-inventory.csv',inventory)
 csvwrite(out/'final_metrics.csv',final)
 text='# Llama 동일 JVP 표본 — 최종 W10 전체1,000 첫 비교표\n\n'+table(headline(final),['arm','state','RS','PS','NS','rewrite_acc','rephrase_acc'])
 text+='\nRS/PS=new NLL<true NLL; NS=true NLL<new NLL; tie failure. Acc는 teacher-forced all-target-token prompt accuracy. Current pooling 아님. O/JVP는 sealed Git publication 재해시, 나머지4개는 local NLL pair 독립 재계산. Source/config/context가 달라 method-only 인과비교 아님. L4 사전 gate는 SKIPPED_USER_DIRECTED. L4 observer P index metadata=4이나 runtime allp[[0]]; 별도 CPU projector 검산 예정.\n'
 (out/'first-fivearm-table-ko.md').write_text(text)
 save(out/'first-table-receipt.json',dict(status='FIRST_TABLE_READY',member_count=len(inventory),member_root=digest(inventory),report_sha256=sha(out/'first-fivearm-table-ko.md'),sample_root=SAMPLE_ROOT))
 print(text,flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
 o=a.repo/OUT_REL;o.mkdir(parents=True,exist_ok=False);first(a.repo,o)
