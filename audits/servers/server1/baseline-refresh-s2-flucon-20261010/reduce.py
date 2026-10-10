"""SH1 bounded result refresh and SH2 input handoff; no forward/CP load/job writes."""
import csv,math
from pathlib import Path
from datetime import datetime,timezone
from official.runners.server1.common import read,verify,member
from official.experiments.prepare import write_new,digest
from official.runners.server1 import flucon_eval as e
from official.evaluation.generation.assets import load_assets
from official.evaluation.generation.native_observer import read_observed

NONCE='USER-GH-S1-S2-BASELINE-REFRESH-S2-FLUCON-20261010-R1'
O=Path(__file__).parent;B=e.BASE
snap=Path('/mnt/raid5/janghj/ODE-edit/local/baseline-refresh-s2-flucon-20261010/server1')
acct=read(snap/'accounting.json');queue=read(snap/'queue.json')
states={s.split('|')[0]:s.split('|')[2] for s in acct['rows'].splitlines() if s}
prior_path=Path('audits/servers/server1/completed-table-flucon-cap3-20261010/table-rows.json')
prior=read(prior_path);rows=[]
for old in prior['rows']:
    if not old.get('complete',True):continue
    r=dict(old)
    for k in ('config','endpoint','terminal'):verify(r[k])
    assert states[r['job_id']]=='COMPLETED'
    r.update(reused_audit=member(prior_path),scheduler_state=states[r['job_id']],raw_terminal_SHA_unchanged=True)
    if r['dataset']=='zsre':
        proof=read(verify(r['query_proof']));p=proof['proof'];raw=read(verify(r['endpoint']))
        for k in ('frozen_evaluator','frozen_query_module','frozen_public_lock','stream'):verify(proof[k])
        assert p['requests']==len(raw['cases'])==2000 and p['input_mismatches']==p['target_mismatches']==0
        assert raw['query_sha256']==p['query_sha256'] and digest([x['case_id'] for x in raw['cases']])==r['ordered_case_ids_sha256']
        metrics={};counts={}
        for g,label in [('rewrite','Efficacy'),('paraphrase','Generalization'),('neighborhood','Specificity')]:
            rates=[];n=correct=0
            for case in raw['cases']:
                obs=case[g+'_observations'];bits=[x['predicted_token_id']==x['target_token_id'] for x in obs]
                assert bits and bits==case[g+'_prompts_correct']==[x['correct'] for x in obs]
                assert all(type(x['predicted_token_id']) is int and type(x['target_token_id']) is int for x in obs)
                rates.append(sum(bits)/len(bits));n+=len(bits);correct+=sum(bits)
            value=100*math.fsum(rates)/2000;assert math.isfinite(value) and abs(value-raw['summary'][label])<1e-10
            assert n==raw['token_denominators'][g]==p['token_denominators'][g]
            metrics[label]=value;counts[g]=dict(requests=2000,tokens=n,correct=correct,missing_requests=0)
        assert metrics==r['metrics'];assert raw['summary']['Specificity']==raw['summary']['Specificity_loc_ans']
        r.update(metrics=metrics,counts=counts,CPU_independent_reduction='SAVED_TOKEN_REQUEST_MACRO_RECOMPUTED_UNCHANGED')
    oldgen=r.get('generation')
    if isinstance(oldgen,dict):
        for k in ('endpoint','terminal','config','collector_verification'):verify(oldgen[k])
        # Same reference-bound collector proof/raw SHA checked in preceding audit.
        assert states[oldgen['job_id']]=='COMPLETED'
        oldgen=dict(oldgen,reused_audit=member(prior_path),scheduler_state='COMPLETED')
        r['generation']=oldgen
    rows.append(r)
print('UNCHANGED_COMPLETED_ROWS',len(rows),'zsre6 token macro PASS',flush=True)

registration=B/'completed-table-flucon-cap3-20261010/registration-r1'
s=read(registration/'submission.json');lock=read(verify(s['execution_lock']))
for m in lock['source_members']:verify(m)
refs=load_assets(e.REFERENCE);evaluations=[];handoff=[]
for key,j in s['jobs'].items():
    if not j['config']:
        evaluations.append(dict(job_id=j['job_id'],job_name=j['name'],state=states[j['job_id']],GPU=0));continue
    c=read(verify(j['config']));out=Path(c['output']);original=c['original']
    for k in ('checkpoint','pointer','terminal','config'):verify(original[k])
    endpoint_row=next(r for r in rows if r['job_id']==original['job_id'])
    assert original['ordered_case_ids_sha256']==endpoint_row['ordered_case_ids_sha256']
    source_config=read(verify(original['config']));source_done=read(verify(original['terminal']))
    assert source_done['identity']==original['identity'] and source_done['requests']==2000 and source_done['native_apply_calls']==20
    folder=Path(original['terminal']['path']).parent
    for b in range(1,21):
        commit=read(folder/'commits'/f'batch-{b:02d}.json')
        assert commit['identity']==original['identity'] and commit['batch']==commit['cursor']['completed_batch']==b
        assert commit['cursor']['edit']['requests']==100
    assert commit['cursor']==source_done['final_cursor']
    records=read(verify(c['stream']));records=records['records'] if isinstance(records,dict) else records
    assert len(records)==2000 and digest([x['case_id'] for x in records])==original['ordered_case_ids_sha256']
    record=dict(job_id=j['job_id'],job_name=j['name'],state=states[j['job_id']],config=j['config'],source=s['source'])
    binding=dict(model=c['model'],method=original['method'],dataset='cf',original_job=original['job_id'],
        checkpoint=original['checkpoint'],pointer=original['pointer'],original_config=original['config'],
        original_source=original['identity']['code_commit'],original_identity=original['identity'],
        cohort_sha256=original['ordered_case_ids_sha256'],stream=c['stream'],reference=c['reference'],reference_identity=c['reference_identity'],
        protocol='cf-cake-native-casebatch-kv-total100-globalrng-v1',seed=20261007,
        loader=member(Path(lock['source_directory'])/'official/runners/server1/zsre_reeval_restore.py'),
        eval_job=j['job_id'],eval_config=j['config'],payload_full_SHA_rechecked=True,SH2_submit=False)
    if states[j['job_id']]=='COMPLETED':
        done=read(out/'COMPLETE.json');assert done['config']==j['config'] and done['source']==s['source']
        assert done['original_checkpoint']==original['checkpoint'] and done['weights_unchanged'] and done['RNG_unchanged']
        raw=read_observed(verify(done['endpoint']),assets=refs)
        assert raw['summary']==done['summary'] and len(raw['rows'])==raw['summary']['planned_count']==2000
        assert [x['case_id'] for x in raw['rows']]==[x['case_id'] for x in records]
        gen=dict(record,status='W20_GENERATION_CPU_RAW_VERIFIED',summary=raw['summary'],endpoint=done['endpoint'],terminal=member(out/'COMPLETE.json'),
            Flu=e.display(raw['summary']['ngram_entropy'],'bits'),Con=e.display(raw['summary']['reference_score'],'cosine'),
            scoring='ALL_CASES_CPU_REFERENCE_BOUND_RECHECK',reference_identity=refs.sha)
        endpoint_row['generation']=gen;record.update(gen);binding.update(status='COMPLETE_NO_DUPLICATE',endpoint=done['endpoint'])
        print('NEW_GENERATION_VERIFIED',j['job_id'],gen['Flu']['paper_display_x100'],gen['Con']['paper_display_x100'],flush=True)
    else:
        assert states[j['job_id']] in ('RUNNING','PENDING','COMPLETING')
        record.update(status='REGISTERED_EVALUATION_NO_DUPLICATE',final_metrics=None)
        endpoint_row['generation']=record;binding.update(status='REGISTERED_NO_DUPLICATE')
    evaluations.append(record);handoff.append(binding)

author=read(B/'fe-author-hparams-2k-20261010/registration-r1/submission.json')
for dataset,j in author['jobs'].items():
    c=read(verify(j['config']));out=Path(c['output']);done=out/'COMPLETE.json'
    assert not done.exists(),'NEW_AUTHOR_COMPLETION_REQUIRES_ADDITIONAL_CPU_AUDIT'
    commits=sorted((out/'commits').glob('batch-*.json'))
    rows.append(dict(model='llama3',method='MEMIT_FE_HISTORY (FE author hparams)',dataset=dataset,job_id=j['job_id'],job_name=j['name'],
        source=author['source'],config=j['config'],status='INCOMPLETE_NO_W20_CHECKPOINT',scheduler_state=states[j['job_id']],
        complete=False,observed_commits=len(commits),generation='NO_W20_CHECKPOINT' if dataset=='cf' else 'NOT_APPLICABLE'))
    if dataset=='cf':handoff.append(dict(model='llama3',method='MEMIT_FE_HISTORY (FE author hparams)',original_job=j['job_id'],
        status='NO_W20_CHECKPOINT',SH2_submit=False,source=author['source'],config=j['config'],observed_commits=len(commits)))

native=[dict(original_job=r['job_id'],eval_job=r['generation']['job_id'],status='COMPLETED_PREVIOUS_AUDIT_SHA_UNCHANGED',
             SH2_submit=False,endpoint=r['generation']['endpoint']) for r in rows if r['dataset']=='cf' and r['method'] in ('FT','SPHERE','MEMIT_FE')]
handoff+=native
inventory=dict(nonce=NONCE,server='server1',at=acct['at'],queue=queue,accounting=acct,cap=3,
    evaluations=evaluations,SH2_inputs=handoff,eligible_unregistered_local_CF_checkpoints=0,
    historical_exclusions='MEMIT/AlphaEdit/BLUE remote historical CP: no verified current local payload/cohort candidate; existing table exceptions KEEP',
    new_GPU_jobs=0,transfers=0,deletions=0,all_existing_jobs_unchanged=True)
write_new(O/'inventory.json',inventory)
result=dict(nonce=NONCE,server='server1',at=datetime.now(timezone.utc).isoformat(),rows=rows,completed_rows=sum(r.get('complete',True) for r in rows),
    inventory=member(O/'inventory.json'),reducer=member(__file__),source_query_verification='CPU_ONLY_NOT_PRETRAINED_FORWARD_PARITY',
    new_GPU_forwards=0,checkpoint_loads=0,README_owner='GH',broadcast='NO_BROADCAST_NOT_REQUIRED')
write_new(O/'table-rows.json',result)
with (O/'table-rows.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['model','method','dataset','job_id','state','Eff','Gen','Loc','Score','Flu_x100','Con_x100','eval_job'])
    for r in rows:
        m=r.get('metrics',r.get('summary',{}));g=r.get('generation');g=g if isinstance(g,dict) else {}
        w.writerow([r['model'],r['method'],r['dataset'],r['job_id'],r['scheduler_state'],m.get('Efficacy'),m.get('Generalization'),m.get('Specificity'),m.get('Score'),g.get('Flu',{}).get('paper_display_x100'),g.get('Con',{}).get('paper_display_x100'),g.get('job_id')])
print('ROWS',len(rows),'eligible SH2 inputs0',flush=True)
