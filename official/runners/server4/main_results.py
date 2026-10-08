"""Compact owner evidence for GH's main table; never edits README or polls."""
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from official.experiments.prepare import digest, read, write_new, file_sha

POLICY='USER-GH-MAIN-TABLE-FRESH-RERUN-20261009-R1'
REPORT='experiment-reports/servers/server4/official-baselines-20261008/main-table-fresh-rerun-ko.md'
EXCLUDED={'qualification','W0_preparation','collector','heldout_tuning','historical_replay'}

def identity(config,assets,ready,attempt,records,*,job_id=None,job_name=None,cold=None,role='main'):
    if role!='main':raise ValueError('NOT_A_MAIN_RESULT_ROLE')
    if len(records)!=2000:raise ValueError('MAIN_ORDERED_FIRST2000_REQUIRED')
    ids=[r['case_id'] for r in records]
    if len(set(ids))!=2000:raise ValueError('MAIN_OCCURRENCE_IDENTITY')
    if config['model']!='llama3' or config['method'] not in ('ALPHAEDIT','ALPHAEDIT_BLUE','SPHERE'):
        raise ValueError('SERVER4_MAIN_SCOPE')
    if config['dataset'] not in ('cf','zsre'):raise ValueError('MAIN_DATASET')
    if (job_id is None)!=(job_name is None):raise ValueError('INCOMPLETE_ACTUAL_JOB_IDENTITY')
    if job_id is not None and (not str(job_id).isdigit() or int(job_id)<=0 or not job_name):
        raise ValueError('ACTUAL_JOB_IDENTITY_REQUIRED')
    return dict(policy=POLICY,rerun_attempt=attempt,model=config['model'],method=config['method'],
        dataset=config['dataset'],ordered_sample_sha256=digest(ids),
        ordered_stream_sha256=assets['streams'][config['dataset']]['sha256'],
        cold_state_identity=cold,source_commit=ready['code_commit'],
        config_sha256=ready['config_sha256'],server='server4',job_id=job_id,job_name=job_name,
        role=role,fresh_cold_required=True,historical_checkpoint_substitution=False,report_path=REPORT)

def event(binding,state,*,reason=None,metrics=None,batch=None,generation_deferred=False):
    if binding['role'] in EXCLUDED or binding['role']!='main':raise ValueError('NOT_A_MAIN_RESULT_ROLE')
    if state not in ('NOT_SUBMITTED','PENDING','RUNNING','COMPLETING','FAILED','CANCELLED','W20_COMPLETE'):
        raise ValueError('MAIN_STATE')
    if state!='NOT_SUBMITTED' and not binding.get('job_id'):raise ValueError('STATE_REQUIRES_ACTUAL_JOB')
    if metrics is not None and (state!='W20_COMPLETE' or batch!=20 or not binding.get('cold_state_identity')):
        raise ValueError('ONLY_OBSERVED_FRESH_W20_RESULTS')
    if state=='W20_COMPLETE' and (batch!=20 or not binding.get('cold_state_identity')):
        raise ValueError('FRESH_W20_IDENTITY_REQUIRED')
    if generation_deferred and binding['dataset']!='cf':raise ValueError('CF_GENERATION_ONLY')
    display=('' if state=='NOT_SUBMITTED' else
        ('ING: ' if state=='RUNNING' else state+': ')+binding['job_name'])
    row=dict(binding,observed_state=state,observed_at=datetime.now(timezone.utc).isoformat(),
        table_status=display,not_submitted_reason=reason,
        generation_status='DEFERRED' if generation_deferred else 'NOT_REPORTED')
    if metrics is not None:row['W20_measured_metrics']=metrics
    return row

def save_event(output,binding,state,**kwargs):
    value=event(binding,state,**kwargs)
    path=Path(output)/'main-table-events'/f'{time.time_ns()}-{state}.json'
    write_new(path,value)
    return path

def cold_origin(args,config,assets,ready,records,cold_state):
    """Resume can continue this same fresh chain, never adopt an older checkpoint."""
    binding=identity(config,assets,ready,args.attempt,records,
        job_id=os.environ.get('SLURM_JOB_ID'),job_name=os.environ.get('SLURM_JOB_NAME'),
        cold=digest(cold_state))
    if not binding['job_id']:raise ValueError('MAIN_MUST_BIND_ACTUAL_SLURM_JOB')
    path=args.output/'main-cold-origin.json'
    if args.resume:
        old=read(path)
        stable=('policy','model','method','dataset','ordered_sample_sha256',
                'ordered_stream_sha256','source_commit','config_sha256','server')
        if any(old[k]!=binding[k] for k in stable) or not old.get('cold_state_identity'):
            raise ValueError('HISTORICAL_CHECKPOINT_NOT_FRESH_MAIN')
        binding.update(cold_state_identity=old['cold_state_identity'],rerun_attempt=old['rerun_attempt'])
    else:
        write_new(path,binding)
    return binding

def unsubmitted_inventory(preparation,source_commit):
    """No scheduler action. Explicit null IDs/cold measurements are not fake PENDING."""
    preparation=Path(preparation);assets=read(preparation/'assets.json');rows=[]
    for path in sorted((preparation/'configs').glob('*.json')):
        config=read(path);stream=assets['streams'][config['dataset']]
        if file_sha(stream['path'])!=stream['sha256']:raise ValueError('STREAM_BYTES_CHANGED')
        records=read(stream['path'])
        binding=identity(config,assets,dict(code_commit=source_commit,config_sha256=file_sha(path)),
            'fresh-first2k-20261009-r1-'+config['run_id'],records)
        value=event(binding,'NOT_SUBMITTED',
            reason='SH1_PORTABLE_W0_REPAIR_READY_PENDING; LOGGER_DISPLAY_MISMATCH; ACTUAL_NATIVE_RESUME_PARITY_NOT_RUN')
        value.update(execution_source_frozen=False,cold_state_measurement='NOT_OBSERVED',
            metrics={},CF_columns=['Score','Eff','Gen','Loc','Flu','Con'] if config['dataset']=='cf' else [],
            zsRE_columns=['Eff','Gen','Loc'] if config['dataset']=='zsre' else [])
        rows.append(value)
    if len(rows)!=6:raise ValueError('SIX_OWN_ROWS_REQUIRED')
    return dict(policy=POLICY,server='server4',source_candidate_commit=source_commit,
        rows=rows,job_ids=[],README_edited=False,old_checkpoints='KEEP',
        source_publication_is_execution=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    write_new(args.output,unsubmitted_inventory(args.preparation,args.source_commit))
