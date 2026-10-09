"""One bounded read-only audit, CPU token/query/scalar work only."""
import sys,math,csv
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
from datetime import datetime,timezone
B=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')
R=B/'zsre-2k-reeval-20261009/registration-r1'
sys.path.insert(0,str(R/'source'))
from official.runners.server1.common import read,member,verify
from official.runners.server1.assets import member as asset_member
from official.experiments.prepare import digest,write_new
from official.evaluation.zsre_query_parity import compare_queries
from transformers import AutoTokenizer
O=Path(__file__).parent
oldpath=Path('audits/servers/server1/flucon-paper-scale-20261010/completed-rows.json')
old=read(oldpath);oldby={r['job_id']:r for r in old['rows']}
rows=[]
for r in old['rows']:
    if r['dataset']!='cf':continue
    verify(r['endpoint']);verify(r['terminal']);verify(r['config'])
    copy=dict(r);copy.update(reused_audit=member(oldpath),status='W20_FACTUAL_CPU_VERIFIED_UNCHANGED',scheduler_state='COMPLETED')
    rows.append(copy)
s=read(R/'submission.json');lock=read(verify(s['execution_lock']))
lines=Path('README.md').read_text().splitlines()
start=next(i for i,line in enumerate(lines) if line.startswith('| FT |') and '62259' in line)
table={line.split('|')[1].strip():[v.strip() for v in line.split('|')[2:-1]] for line in lines[start:start+6]}
labels={'FT':'FT','MEMIT':'MEMIT','ALPHAEDIT':'AlphaEdit','ALPHAEDIT_BLUE':'AlphaEdit-BLUE','MEMIT_FE':'MEMIT-FE','SPHERE':'AlphaEdit+SPHERE'}
for m in lock['source_members']:verify(m)
first=read(verify(s['jobs']['FT']['config']));a=read(verify(first['assets']))
for m in a['model']['tokenizer_files'].values():asset_member(m['path'],allow_symlink=True,expected_sha=m['sha256'],expected_bytes=m['bytes'])
tok=AutoTokenizer.from_pretrained(a['model']['tokenizer_path'],local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
records=read(verify(first['stream']));proof=compare_queries(tok,records,model_family='llama3')
assert proof==read(verify(first['parity'])) and proof['requests']==2000
assert proof['input_mismatches']==proof['target_mismatches']==0 and proof['model_forward_calls']==0
write_new(O/'query-proof.json',dict(proof=proof,stream=first['stream'],tokenizer_sha256=first['original']['identity']['tokenizer_sha256'],
    frozen_evaluator=member(R/'source/official/evaluation/zsre_paper.py'),
    frozen_query_module=member(R/'source/official/evaluation/zsre_query_parity.py'),
    frozen_public_lock=member(R/'source/official/evaluation/zsre_public_sources/lock.json'),
    semantics='public loader/evaluator AST: native prefix/space/BOS/decode-retokenize exact; not pretrained forward parity'))
for method,job in s['jobs'].items():
    if method=='collector':continue
    c=read(verify(job['config']));assert c['parity']==first['parity'] and c['stream']==first['stream'] and c['assets']==first['assets']
    assert c['evaluator']['sha256']==member(R/'source/official/evaluation/zsre_paper.py')['sha256']
    folder=Path(c['output']);done=read(folder/'COMPLETE.json');raw=read(verify(done['endpoint']))
    verify(oldby[job['job_id']]['endpoint']);verify(oldby[job['job_id']]['terminal'])
    assert done['config']==job['config'] and done['source']==s['source']==raw['identity']['new_source']
    assert raw['identity']['original_checkpoint']==c['original']['checkpoint'] and raw['identity']['config_sha256']==c['config_sha256']
    assert len(raw['cases'])==raw['summary']['requests']==2000 and raw['query_sha256']==proof['query_sha256']
    assert digest([x['case_id'] for x in raw['cases']])==c['original']['ordered_case_ids_sha256']
    metrics={};counts={}
    for group,label in [('rewrite','Efficacy'),('paraphrase','Generalization'),('neighborhood','Specificity')]:
        rates=[];n=correct=0
        for case in raw['cases']:
            obs=case[group+'_observations'];bits=[x['predicted_token_id']==x['target_token_id'] for x in obs]
            assert bits and all(type(x['predicted_token_id']) is int and type(x['target_token_id']) is int for x in obs)
            assert bits==case[group+'_prompts_correct'] and all(x['correct']==b for x,b in zip(obs,bits))
            rates.append(sum(bits)/len(bits));n+=len(bits);correct+=sum(bits)
        value=100*math.fsum(rates)/2000;assert math.isfinite(value)
        assert abs(value-raw['summary'][label])<1e-10 and n==raw['token_denominators'][group]==proof['token_denominators'][group]
        metrics[label]=value;counts[group]=dict(requests=2000,tokens=n,correct=correct,missing_requests=0)
    # Stored alias must be exact; independent fsum has the same pre-existing
    # 1e-10 pp comparison above, not a bitwise equality claim across reducers.
    assert raw['summary']['Specificity']==raw['summary']['Specificity_loc_ans']
    prev=oldby[job['job_id']]['metrics'];delta={k:metrics[k]-prev[k] for k in metrics}
    assert all(v==0 for v in delta.values())
    displayed=[str(Decimal(str(metrics[k])).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)) for k in ('Efficacy','Generalization','Specificity')]
    assert displayed==table[labels[method]][-3:],'README_NUMERIC_MISMATCH'
    rows.append(dict(model='llama3',method=method,dataset='zsre',job_id=job['job_id'],job_name=job['name'],scheduler_state='COMPLETED',
        status='W20_PUBLIC_QUERY_CPU_VERIFIED',complete=True,metrics=metrics,counts=counts,delta_vs_previous_pp=delta,
        source=s['source'],config=job['config'],endpoint=done['endpoint'],terminal=member(folder/'COMPLETE.json'),
        original_edit_job=c['original']['job_id'],original_checkpoint=c['original']['checkpoint'],
        ordered_case_ids_sha256=c['original']['ordered_case_ids_sha256'],query_proof=member(O/'query-proof.json'),
        evaluation_profile='zsre-public-query-W20-only-v1',README_values=displayed,README_match=True,previous_endpoint_status='PUBLIC_QUERY_REEVALUATION_REQUIRED_SUPERSEDED',
        evaluation_not_new_edit_chain=True,generation='NOT_APPLICABLE'))
rows.append(read(O/'qwen-history.json'))
gen=read(B/'flucon-paper-scale-20261010/registration-r1/submission.json');pending=[]
for key,j in gen['jobs'].items():
    jid=j['job_id'];state='RUNNING' if jid in ('62259','62260','62261') else 'PENDING'
    pending.append(dict(job_id=jid,job_name=j['name'],state=state,source=gen['source'],final_scores='NOT_OBSERVED',
        completed_receipt_exists=False if key=='collector' else (Path(read(verify(j['config']))['output'])/'COMPLETE.json').exists()))
assert not any(r['completed_receipt_exists'] for r in pending),'NEW_GENERATION_COMPLETION_REQUIRES_CPU_AUDIT'
result=dict(nonce='USER-GH-ALL-SH-BASELINE-COMPLETED-ZSRE-AUDIT-20261010-R1',server='server1',
    recorded_at=datetime.now(timezone.utc).isoformat(),rows=rows,completed_rows=len(rows),generation_jobs=pending,
    scheduler_queries=1,snapshot_basis='single preceding sacct read: all factual/eval-only candidates COMPLETED; generation states as listed',
    query_parity='PASS_CPU_ONLY_NOT_PRETRAINED_FORWARD_PARITY',README_snapshot=member('README.md'),new_GPU_forwards=0,checkpoint_loads=0,
    old_wrong_context_61975_excluded=True,README_owner='GH',broadcast='NO_BROADCAST_NOT_REQUIRED',reducer=member(__file__))
write_new(O/'table-rows.json',result)
with (O/'table-rows.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['model','method','dataset','job_id','status','Eff','Gen','Loc','Score','generation'])
    for r in rows:
        m=r.get('metrics',r.get('summary',{}));w.writerow([r['model'],r['method'],r['dataset'],r['job_id'],r['status'],m.get('Efficacy'),m.get('Generalization'),m.get('Specificity'),m.get('Score'),r.get('generation','DEFERRED')])
print('completed rows',len(rows),'zsre6 all unchanged; query parity',proof['status'])
