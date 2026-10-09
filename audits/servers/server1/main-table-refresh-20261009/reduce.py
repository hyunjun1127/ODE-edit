"""Bounded CPU raw audit only; no GPU, restore, scheduler mutation or online log."""
import argparse, json, sys, math, csv, datetime, hashlib
from pathlib import Path

LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')
OUT=Path('audits/servers/server1/main-table-refresh-20261009')
p=argparse.ArgumentParser();p.add_argument('--history',action='store_true');a=p.parse_args()
if a.history:
    sys.path.insert(0,str(LOCAL/'memit-fe-history-three-model-2k/registration-r1/source'))
else:
    sys.path.insert(0,str(Path.cwd()))
from official.runners.server1.common import read,verify,member
from official.experiments.prepare import digest,write_new

if a.history:
    from official.runners.server1.audit import audit_factual
    from official.baselines import registry
    from transformers import AutoTokenizer
    reg=LOCAL/'memit-fe-history-three-model-2k/registration-r1'
    submit=read(reg/'submission.json');lock=read(verify(submit['execution_lock']))
    for m in lock['source_members']:verify(m)
    cm=submit['jobs']['gptj']['config'];cfg=read(verify(cm));folder=Path(cfg['output'])
    assets=read(verify(cfg['assets']));records=read(verify(cfg['stream_member']))
    if isinstance(records,dict):records=records['records']
    assert len(records)==2000 and digest(records)==cfg['stream_sha256']
    done=read(folder/'COMPLETE.json');identity=done['identity']
    assert done['requests']==2000 and done['native_apply_calls']==done['history_appends_per_layer']==20
    assert identity['code_commit']==lock['source_commit'] and identity['config_sha256']==cfg['config_sha256']
    assert identity['stream_sha256']==cfg['stream_sha256'] and done['checkpoint_W20_preserved']
    previous=None
    for b in range(1,21):
        c=read(folder/'commits'/f'batch-{b:02d}.json');e=c['cursor']['edit']
        assert c['batch']==c['cursor']['completed_batch']==c['checkpoint']['batch']==b and c['identity']==identity
        assert e['requests']==100 and e['history_appends_per_layer']==1
        assert e['request_sha256']==digest(registry.requests(records[(b-1)*100:b*100],'MEMIT_FE_HISTORY','gptj'))
        assert e['before']['successful_calls']==b-1 and e['after']['successful_calls']==b
        if previous is not None:assert e['before']==previous
        previous=e['after']
    assert c['cursor']==done['final_cursor']
    pointer=read(folder/'checkpoint/latest.json');assert pointer==c['checkpoint'] and pointer['final_W20']
    cp=member(folder/'checkpoint'/pointer['file']);assert cp['sha256']==pointer['sha256']
    raw_member=c['cursor']['factual'];raw=read(verify(raw_member))
    tok=AutoTokenizer.from_pretrained(assets['model']['snapshot'],local_files_only=True)
    tok.pad_token=tok.eos_token;tok.padding_side='right'
    external=dict(identity,instruction_id=cfg['instruction_id'],method=cfg['method'],model=cfg['model'])
    audited=audit_factual(raw,records,'cf',tok,external)
    result=dict(model='gptj',dataset='cf',method='MEMIT_FE_HISTORY',job_id='61927',complete=True,
        status='W20_FACTUAL_CPU_VERIFIED',summary=raw['summary'],identity=identity,
        ordered_case_ids_sha256=digest([r['case_id'] for r in records]),source=lock['source_commit'],
        config=cm,endpoint=raw_member,terminal=member(folder/'COMPLETE.json'),raw_audit=audited,
        checkpoint_metadata=member(folder/'checkpoint/latest.json'),checkpoint_SHA_verified=True,
        generation='DEFERRED',variant_separate_from_native_FE=True,commits=20)
    write_new(LOCAL/'main-table-refresh-20261009/history-review.json',result)
    print(json.dumps(result['summary']));sys.exit()

states={'61769':'CANCELLED','61770':'COMPLETED','61771':'COMPLETED','61772':'CANCELLED','61773':'COMPLETED',
        '61927':'COMPLETED','61928':'RUNNING','61929':'FAILED','61975':'RUNNING',
        '61932':'COMPLETED','61933':'COMPLETED','61934':'PENDING','61935':'COMPLETED','61936':'PENDING','61937':'COMPLETED'}
old=read('audits/servers/server1/official-baselines-20261008/results-review-20261009/results.json')
old_by_id={r['job_id']:r for r in old['rows']}
rows=read(LOCAL/'main-table-refresh-20261009/cf-review.json')['rows']
for r in rows:
    r['job_name']='official-s1-cf-'+r['method'].lower()
    r['previous_metrics']=old_by_id.get(r['job_id'],{}).get('summary')
    if r.get('complete'):
        r['metrics']={k:r['summary'][k] for k in ('Score','Efficacy','Generalization','Specificity')}
        r['delta_pp']=({k:r['metrics'][k]-r['previous_metrics'][k] for k in r['metrics']}
                       if r['previous_metrics'] else None)
zs=read(LOCAL/'main-table-refresh-20261009/zsre-review.json')
reg=LOCAL/'zsre-2k-reeval-20261009/registration-r1';submitted=read(reg/'submission.json')
for r in zs['rows']:
    j=submitted['jobs'][r['method']];c=read(verify(j['config']))
    r.update(job_id=j['job_id'],job_name=j['name'],model='llama3',dataset='zsre',source=submitted['source'],
        config=j['config'],ordered_case_ids_sha256=c['original']['ordered_case_ids_sha256'],
        complete=r['status']=='W20_EVAL_ONLY_CPU_REDUCED',generation='NOT_APPLICABLE',
        evaluation_profile='zsre-public-query-W20-only-v1',source_run_id=c['original']['job_id'])
    if r['complete']:
        done_path=Path(c['output'])/'COMPLETE.json';r['terminal']=member(done_path)
        r['identity']=read(verify(r['endpoint']))['identity']
        r['status']='W20_PUBLIC_QUERY_CPU_VERIFIED'
    else:
        r['status']='EVALUATION_PENDING';r.pop('error',None)
    rows.append(r)
history=read(LOCAL/'main-table-refresh-20261009/history-review.json')
history['job_name']='official-s1-cf-gptj-memit-fe-history';history['metrics']={k:history['summary'][k] for k in ('Score','Efficacy','Generalization','Specificity')}
rows.append(history)
for tag,jid,rel in [('llama3','61928','preparation-r1'),('qwen25','61975','oom-repair-r1/preparation')]:
    root=LOCAL/'memit-fe-history-three-model-2k';cpath=root/rel/'configs'/f'{tag}.json';c=read(cpath)
    rows.append(dict(model=tag,dataset='cf',method='MEMIT_FE_HISTORY',job_id=jid,
        job_name=f'official-s1-cf-{tag}-memit-fe-history',complete=False,status='IN_PROGRESS_NO_W20',
        config=member(cpath),ordered_case_ids_sha256=c['ordered_case_ids_sha256'],
        checkpoint_metadata=member(Path(c['output'])/'checkpoint/latest.json'),generation='DEFERRED',
        source='eaf78c331dee72093b2ccdf29f31962799475e34' if tag=='llama3' else '5d6dfd58773d97ca50525224439a7a06372b69a0'))
for r in rows:
    r['server']='server1';r['observed_state']=states[r['job_id']]
    r['observed_at']='2026-10-09T14:05:52+00:00'
    r['observed_at_semantics']='UTC receipt recording bound for the single preceding scheduler snapshot; not a new query'
    r['eligible_metrics']=r['complete']
    r['report_path']='experiment-reports/servers/server1/main-table-refresh-20261009/report-ko.md'
result=dict(instruction='USER-GH-ALL-SH-COMPLETED-MAIN-TABLE-REFRESH-20261009-R1-SERVER1',
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,eligible_metrics=sum(r['complete'] for r in rows),
    scheduler_snapshot=states,scheduler_queries=1,CPU_forward_calls=0,checkpoint_restore=False,
    source_reducer=member(__file__),policy=member('control/main-results-policy.json'),README_editor='GH',
    old_failed_replaced=dict(job_id='61929',state='FAILED',replacement='61975'),
    preserved_historical=['Llama CF MEMIT42658','Llama CF AlphaEdit42657','Llama BLUE39283_1','Llama PRICE FREE10060103'],
    historical_promoted=False,GPT2_appendix_new_claims=0,broadcast='NO_BROADCAST_NOT_REQUIRED')
write_new(OUT/'table-rows.json',result)
with (OUT/'table-rows.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['model','method','dataset','job_id','job_name','state','raw_verified','Score','Eff','Gen','Loc','Flu','Con'])
    for r in rows:
        m=r.get('metrics',{});w.writerow([r['model'],r['method'],r['dataset'],r['job_id'],r['job_name'],r['observed_state'],r['complete'],
            m.get('Score',''),m.get('Efficacy',''),m.get('Generalization',''),m.get('Specificity',''),
            'DEFERRED' if r['dataset']=='cf' else '', 'DEFERRED' if r['dataset']=='cf' else ''])
print(json.dumps(dict(eligible=result['eligible_metrics'],rows=len(rows))))
