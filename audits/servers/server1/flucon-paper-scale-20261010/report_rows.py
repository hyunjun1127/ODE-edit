"""Compact local evidence extraction; no model loads, raw text or tensor publication."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from official.experiments.prepare import write_new, digest
from official.runners.server1.common import member, read, verify
from official.runners.server1.flucon_eval import LOCAL, INSTRUCTION

def main():
    prep_path=LOCAL/'preparation-r2/preparation.json'
    submission_path=LOCAL/'registration-r1/submission.json'
    prep,sub=read(prep_path),read(submission_path)
    rows=[]
    w=prep['W0']
    rows.append(dict(model='llama3',method='W0',dataset='cf',endpoint='W0',status='CPU_EXISTING_RAW_VERIFIED',
        raw=w['raw'],identity_sha256=digest(w['identity']),
        identity={k:v for k,v in w['identity'].items() if k not in ('observation_identities','ordered_occurrences')},
        summary=w['summary'],Flu=w['Flu'],Con=w['Con'],new_generation=False))
    for cm in prep['configs']:
        c=read(cm['path']);job=sub['jobs'][c['key']];snap=sub['initial_snapshot'][c['key']]
        rows.append(dict(model=c['model'],method=c['original']['method'],dataset='cf',endpoint='W20',
            job_id=job['job_id'],job_name=job['name'],status=snap['JobState'],dependency=job['dependencies'],
            source=sub['source'],config=cm,checkpoint=c['original']['checkpoint'],original_job=c['original']['job_id'],
            original_identity=c['original']['identity'],stream=c['stream'],
            ordered_cohort_sha256=c['original']['ordered_case_ids_sha256'],planned_requests=2000,
            Flu='EVALUATION_REGISTERED_NOT_OBSERVED',Con='EVALUATION_REGISTERED_NOT_OBSERVED',
            factual_cells_unchanged=True,edits_in_this_job=0))
    out=Path(__file__).parent
    base=LOCAL.parent
    prior_path=Path('audits/servers/server1/main-table-refresh-20261009/table-rows.json')
    prior=read(prior_path)
    completed=[]
    for r in prior['rows']:
        if r.get('complete') and r['dataset']=='cf' and r['job_id'] in ('61770','61771','61773','61927'):
            verify(r['endpoint']);verify(r['terminal'])
            item=dict(r);item.pop('observed_state',None);item.pop('observed_at',None);item.pop('observed_at_semantics',None)
            item.update(evidence_reused=member(prior_path),state_basis='IMMUTABLE_W20_RAW_AND_TERMINAL_REVERIFIED_NO_NEW_SCHEDULER_QUERY')
            completed.append(item)
    completed.append(read(LOCAL/'llama-history-refresh.json'))
    zs=read(LOCAL/'zsre-refresh.json')
    zsub=read(base/'zsre-2k-reeval-20261009/registration-r1/submission.json')
    for r in zs['rows']:
        item=dict(r);job=zsub['jobs'][r['method']];cfg=read(verify(job['config']))
        item.update(model='llama3',dataset='zsre',job_id=job['job_id'],job_name=job['name'],
            complete=r['status']=='W20_EVAL_ONLY_CPU_REDUCED',
            ordered_case_ids_sha256=cfg['original']['ordered_case_ids_sha256'],
            terminal=member(Path(cfg['output'])/'COMPLETE.json'),generation='NOT_APPLICABLE',
            evaluation_profile='zsre-public-query-W20-only-v1')
        completed.append(item)
    write_new(out/'completed-rows.json',dict(rows=completed,count=len(completed),
        reducer=member(__file__),zsre_independent_reducer=member('official/runners/server1/zsre_reeval.py'),
        llama_history_reducer=member(out/'history_refresh.py'),extra_GPU_forwards=0,extra_scheduler_queries=0,
        historical_exceptions_preserved=['MEMIT42658','ALPHAEDIT42657','BLUE39283_1','PRICE_FREE10060103']))
    report=dict(instruction=INSTRUCTION,recorded_at=datetime.now(timezone.utc).isoformat(),
        preparation=member(prep_path),submission=member(submission_path),execution_source=sub['source'],
        official_tree=sub['official_tree'],rows=rows,completed_rows=member(out/'completed-rows.json'),excluded=prep['excluded'],
        collector=sub['jobs']['collector'],CPU_tests=52,source_files_verified=166,
        actual_GPU_evaluation='NOT_OBSERVED_AT_REGISTRATION',online_readback='NOT_OBSERVED_AT_REGISTRATION',
        broadcast='NO_BROADCAST_NOT_REQUIRED_SAME_HOST_CHECKPOINTS_AND_REFERENCE',
        preserved=dict(old_jobs=True,old_raw=True,checkpoints=True,WandB_raw_keys=True),
        README_owner='GH',display_rule='raw * 100, decimal HALF_UP 2 places, no prior rounding')
    write_new(out/'table-rows.json',report)
    with (out/'table-rows.csv').open('x',newline='') as f:
        writer=csv.writer(f);writer.writerow(['model','method','dataset','endpoint','job_id','status','Flu_raw','Flu_x100','Con_raw','Con_x100'])
        for r in rows:
            flu,con=r['Flu'],r['Con']
            writer.writerow([r['model'],r['method'],r['dataset'],r['endpoint'],r.get('job_id',''),r['status'],
                flu['raw_value'] if isinstance(flu,dict) else flu,flu['paper_display_x100'] if isinstance(flu,dict) else flu,
                con['raw_value'] if isinstance(con,dict) else con,con['paper_display_x100'] if isinstance(con,dict) else con])
    print(json.dumps(dict(rows=len(rows),jobs={k:v['job_id'] for k,v in sub['jobs'].items()},source=sub['source'])))

if __name__=='__main__':main()
