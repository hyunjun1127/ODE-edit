"""CPU raw audit; no forward, checkpoint load, scheduler or experiment writes."""
import csv,math
from pathlib import Path
from datetime import datetime,timezone
from official.runners.server1.common import read,verify,member
from official.experiments.prepare import write_new,digest
from official.runners.server1.audit import audit_factual
from official.baselines import registry
from transformers import AutoTokenizer
O=Path(__file__).parent;B=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1')
prior=Path('audits/servers/server1/baseline-refresh-s2-flucon-20261010/table-rows.json')
rows=[]
for r in read(prior)['rows']:
    if not r.get('complete',True):continue
    for key in ('endpoint','terminal','config'):verify(r[key])
    r.update(previous_audit=member(prior),raw_SHA_unchanged=True,scheduler_state='COMPLETED')
    if r['method']=='MEMIT_FE':r['table_destination']='LEGACY_NATIVE_FE_NOT_AUTHOR'
    if r['dataset']=='zsre':
        proof=read(verify(r['query_proof']));raw=read(verify(r['endpoint']))
        for k in ('frozen_evaluator','frozen_query_module','frozen_public_lock','stream'):verify(proof[k])
        assert proof['proof']['query_sha256']==raw['query_sha256'] and len(raw['cases'])==2000
        for g,m in [('rewrite','Efficacy'),('paraphrase','Generalization'),('neighborhood','Specificity')]:
            values=[];n=0
            for case in raw['cases']:
                obs=case[g+'_observations'];bits=[x['predicted_token_id']==x['target_token_id'] for x in obs]
                assert bits and bits==case[g+'_prompts_correct']==[x['correct'] for x in obs]
                values.append(sum(bits)/len(bits));n+=len(bits)
            assert n==raw['token_denominators'][g]==r['counts'][g]['tokens']
            assert abs(100*math.fsum(values)/2000-r['metrics'][m])<1e-10
    gen=r.get('generation')
    if isinstance(gen,dict) and gen.get('endpoint'):
        for k in ('endpoint','terminal','config'):verify(gen[k])
    rows.append(r)
reg=B/'completed-table-flucon-cap3-20261010/registration-r1'
s=read(reg/'submission.json');proofpath=reg/'collector-results.json';collector=read(proofpath)
lock=read(verify(s['execution_lock']))
for m in lock['source_members']:verify(m)
q=next(x for x in collector['rows'] if x['original_job']=='62061');assert q['status']=='CPU_RAW_VERIFIED'
j=s['jobs'][q['key']];c=read(verify(j['config']));donepath=Path(c['output'])/'COMPLETE.json';done=read(donepath)
verify(q['endpoint']);assert done['endpoint']==q['endpoint'] and done['summary']==q['summary']
assert q['summary']['planned_count']==q['summary']['fluency_count']==q['summary']['consistency_count']==2000
assert done['weights_unchanged'] and done['RNG_unchanged'] and done['source']==s['source']
for k in ('config','pointer','terminal','checkpoint'):verify(c['original'][k])
r=next(x for x in rows if x['job_id']=='62061')
r['generation']=dict(q,job_id=j['job_id'],job_name=j['name'],config=j['config'],terminal=member(donepath),collector_verification=member(proofpath),source=s['source'],scheduler_state='COMPLETED',validation='EXACT_REFERENCE_BOUND_COLLECTOR_CPU_PROOF_REUSED')

reg=B/'fe-author-hparams-2k-20261010/registration-r1';s=read(reg/'submission.json');lock=read(verify(s['execution_lock']))
for m in lock['source_members']:verify(m)
for dataset,j in s['jobs'].items():
    c=read(verify(j['config']));out=Path(c['output']);records=read(verify(c['stream_member']))
    records=records['records'] if isinstance(records,dict) else records
    assert len(records)==2000 and digest(records)==c['stream_sha256']
    row=dict(model='llama3',method='MEMIT-FE (FE author hparams + history)',native_method='MEMIT_FE_HISTORY',dataset=dataset,
        profile_sha256=c['author_profile_sha256'],hparams=c['hparams'],job_id=j['job_id'],job_name=j['name'],source=s['source'],config=j['config'],
        ordered_case_ids_sha256=c['ordered_case_ids_sha256'],stream=c['stream_member'],generation='DEFERRED' if dataset=='cf' else 'NOT_APPLICABLE',table_destination='MODEL_MAIN_AUTHOR_REPLACES_NATIVE_FE')
    if not (out/'COMPLETE.json').exists():
        row.update(complete=False,scheduler_state='RUNNING',status='INCOMPLETE_NO_W20',observed_commits=len(list((out/'commits').glob('batch-*.json'))),query_proof=c['zsre_query_parity'])
        rows.append(row);continue
    done=read(out/'COMPLETE.json');identity=done['identity'];previous=None
    assert done['requests']==2000 and done['native_apply_calls']==done['history_appends_per_layer']==20
    assert identity['config_sha256']==c['config_sha256'] and identity['code_commit']==s['source']
    for b in range(1,21):
        commit=read(out/'commits'/f'batch-{b:02d}.json');e=commit['cursor']['edit']
        assert commit['batch']==commit['cursor']['completed_batch']==commit['checkpoint']['batch']==b and commit['identity']==identity
        assert e['requests']==100 and e['history_appends_per_layer']==1
        assert e['request_sha256']==digest(registry.requests(records[(b-1)*100:b*100],'MEMIT_FE_HISTORY','llama3'))
        if previous is not None:assert e['before']==previous
        previous=e['after']
    assert done['final_cursor']==commit['cursor']
    pointer=read(out/'checkpoint/latest.json');assert pointer==commit['checkpoint'] and pointer['final_W20']
    cp=member(out/'checkpoint'/pointer['file']);assert cp['sha256']==pointer['sha256']
    assets=read(verify(c['assets']));tok=AutoTokenizer.from_pretrained(assets['model']['snapshot'],local_files_only=True)
    tok.pad_token=tok.eos_token;tok.padding_side='right'
    rawmember=done['final_cursor']['factual'];raw=read(verify(rawmember))
    external=dict(identity,model=c['model'],method='MEMIT_FE_HISTORY',instruction_id=c['instruction_id'])
    audited=audit_factual(raw,records,'cf',tok,external)
    row.update(complete=True,scheduler_state='COMPLETED',status='W20_CPU_RAW_VERIFIED',commits=20,requests=2000,
        metrics={k:raw['summary'][k] for k in ('Score','Efficacy','Generalization','Specificity')},summary=raw['summary'],raw_audit=audited,
        endpoint=rawmember,terminal=member(out/'COMPLETE.json'),identity=identity,checkpoint=cp,checkpoint_pointer=member(out/'checkpoint/latest.json'),
        future_generation_consumer_pending=True)
    rows.append(row);print('AUTHOR_CF',row['metrics'],flush=True)
inventory=dict(nonce='USER-GH-S1-S2-FE-AUTHOR-MAIN-REFRESH-20261010-R1',at=datetime.now(timezone.utc).isoformat(),server='server1',cap=2,
    scheduler_queries=1,scheduler_snapshot_summary={'62529':'COMPLETED','62530':'RUNNING','62583':'COMPLETED','62584':'COMPLETED','other_previously_completed':'COMPLETED'},
    old_native_FE_relabel=False,new_GPU=0,checkpoint_load=0,job_mutations=0,README_owner='GH',collector_proof=member(proofpath),broadcast='NO_BROADCAST_NOT_REQUIRED')
write_new(O/'inventory.json',inventory)
write_new(O/'table-rows.json',dict(nonce=inventory['nonce'],rows=rows,reducer=member(__file__),inventory=member(O/'inventory.json'),CPU_query_not_GPU_forward_parity=True))
with (O/'table-rows.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['model','method','dataset','job_id','state','Eff','Gen','Loc','Score','Flu_x100','Con_x100'])
    for r in rows:
        m=r.get('metrics',{});g=r.get('generation');g=g if isinstance(g,dict) else {}
        w.writerow([r['model'],r['method'],r['dataset'],r['job_id'],r['scheduler_state'],m.get('Efficacy'),m.get('Generalization'),m.get('Specificity'),m.get('Score'),g.get('Flu',{}).get('paper_display_x100'),g.get('Con',{}).get('paper_display_x100')])
print('PASS',len(rows),'rows',flush=True)
