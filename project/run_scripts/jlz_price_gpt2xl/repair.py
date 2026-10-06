"""Explicit owner-invoked one-attempt repair; no daemon/automatic retry.

Only failed zero-commit cells and exact unstarted pending descendants qualify.
Context and W0 observations are references, not checkpoint continuation.
"""
import argparse
import copy
import getpass
import json
from pathlib import Path
from .common import *
from .submit import ROLES, command, job_name, verify_frozen

PARENT=LOCAL/'attempt-execution-r1'
NEW=LOCAL/'attempt-checkpoint-repair-r1'
AUTHORIZATION='USER_DIRECT_REPAIR_DEPENDENT_RUNS_20261007'

def accounting(ids):
    fields=['JobIDRaw','JobName','User','State','ExitCode','Start','End','ElapsedRaw','AllocTRES']
    raw=command(['sacct','-j',','.join(ids),'--format='+','.join(
        f+'%120' if f=='JobName' else f for f in fields),'--parsable2'])
    lines=raw.splitlines();header=lines[0].split('|')
    rows={r['JobIDRaw']:r for line in lines[1:] if line
          for r in [dict(zip(header,line.split('|'))) ] if r['JobIDRaw'] in ids}
    require(set(rows)==set(ids),'EXACT_ACCOUNTING_CARDINALITY')
    return rows

def reconcile(out):
    require(not out.exists(),'CREATE_ONCE_RECONCILIATION')
    lock,c=verify_frozen(PARENT)
    sub=json.loads((PARENT/'submission.json').read_text());ids=sub['jobs']
    require(set(ids)==set(ROLES) and sub['source']==lock['source_commit'],'OLD_SUBMISSION_SOURCE')
    before=accounting(list(ids.values()));actions=[]
    # Collector first: upstream cancellation satisfies afterany.
    for role in reversed(ROLES):
        job=ids[role];row=accounting([job])[job]
        require(row['User']==getpass.getuser() and row['JobName']==job_name(role),'OLD_EXACT_OWNER_NAME')
        if row['State']=='PENDING':
            detail=command(['scontrol','show','job',job,'--oneliner'])
            require('JobState=PENDING ' in detail and 'RunTime=00:00:00 ' in detail
                and 'StartTime=Unknown ' in detail and 'AllocTRES=(null)' in detail
                and 'Command='+str(PARENT/(role+'.sh'))+' ' in detail
                and 'WorkDir='+str(PARENT/'source')+' ' in detail,'EXACT_UNSTARTED_PENDING_ONLY')
            expected=next(r for r in lock['launchers'] if Path(r['path']).name==role+'.sh');verify(expected)
            require(command(['scontrol','write','batch_script',job,'-']).strip()==verify(expected).read_text().strip(),
                    'OLD_EXACT_LAUNCHER_BYTES')
            result=command(['scancel','--state=PENDING',job])
            actions.append(dict(role=role,job=job,action='CONDITIONAL_PENDING_CANCEL',before=detail,result=result))
        else:
            require(row['State'].split()[0] in ('FAILED','CANCELLED'),'KEEP_RUNNING_OR_COMPLETED_NO_DUPLICATE')
            actions.append(dict(role=role,job=job,action='ALREADY_TERMINAL_NO_CANCEL'))
    after=accounting(list(ids.values()))
    failures=[]
    for role,job in ids.items():
        row=after[job]
        require(row['State'].split()[0] in ('FAILED','CANCELLED'),'WAIT_EXACT_TERMINAL_RELEASE')
        if row['State'].startswith('FAILED') and role!='collector':
            terminal=json.loads((PARENT/role/'terminal.json').read_text())
            error=json.loads((PARENT/role/'first-error.json').read_text())
            require(terminal['source']==lock['source_commit'] and terminal['commits']==0
                and terminal['status']=='TECHNICAL_BLOCKED' and error['type']=='CheckpointError',
                'FAILED_ZERO_COMMIT_CHECKPOINT_SCOPE')
            failures.append(dict(cell=role,job=job,terminal=member(PARENT/role/'terminal.json'),
                error=member(PARENT/role/'first-error.json'),commits=0,cause='lookup pos closure overwritten by parity positions'))
        elif role!='collector':
            require(before[job]['State']=='PENDING' and before[job]['Start']=='Unknown'
                and before[job]['ElapsedRaw']=='0' and before[job]['AllocTRES']=='','CANCELLED_UNSTARTED_ONLY')
    remaining=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%b|%N|%R'])
    require(not remaining,'OLD_TARGETS_STILL_ACTIVE')
    value=dict(status='OLD_TARGETS_TERMINAL_RELEASED',authorization=AUTHORIZATION,
        user_quotes=['fail되었으니 확인해. 나머지 run들도 다시 올려야할 것 같다. fail 지점까진 모니터링 해',
                     'dependency 걸어놓은 다른 job들도 계속 fail중이니 repair 후 제출해'],
        parent_attempt=str(PARENT),new_attempt=str(NEW),parent_submission=member(PARENT/'submission.json'),
        parent_lock=member(PARENT/'execution.lock.json'),parent_source=lock['source_commit'],jobs=ids,
        before=before,actions=actions,after=after,failures=failures,remaining_exact_queue=remaining,
        allocated_GPU_seconds=sum(int(after[ids[r]]['ElapsedRaw']) for r in CELLS
            if 'gres/gpu=1' in after[ids[r]]['AllocTRES']),
        all_old_files_KEEP=True,physical_allocation_release='TERMINAL_ACCOUNTING_AND_EMPTY_EXACT_QUEUE',
        other_jobs_mutated=0,automatic_retry=False)
    write(out,value);return dict(status=value['status'],jobs=ids,allocated_GPU_seconds=value['allocated_GPU_seconds'])

def verify_registration(c):
    """Never bypass duplicate guard merely because an attempt is labelled repair."""
    rec=json.loads(verify(c['repair']['reconciliation']).read_text())
    require(rec['authorization']==AUTHORIZATION and rec['status']=='OLD_TARGETS_TERMINAL_RELEASED'
        and rec['new_attempt']==c['attempt'] and rec['parent_attempt']==str(PARENT),'EXPLICIT_REPAIR_SCOPE')
    require(Path(c['attempt'])==NEW and not NEW.exists(),'ONE_NEW_IMMUTABLE_REPAIR')
    parent_sub=json.loads(verify(rec['parent_submission']).read_text())
    parent_lock=json.loads(verify(rec['parent_lock']).read_text())
    require(parent_sub['jobs']==rec['jobs'] and parent_sub['source']==rec['parent_source']==parent_lock['source_commit'],
        'REPAIR_PARENT_MAPPING_SOURCE')
    submissions=list(LOCAL.glob('*/submission.json'));partials=list(LOCAL.glob('*/submitted-*.json'))
    require(submissions==[PARENT/'submission.json'] and all(p.parent==PARENT for p in partials),
        'UNMATCHED_PREVIOUS_REGISTRATION')
    rows=accounting(list(rec['jobs'].values()))
    require(all(rows[j]['State'].split()[0] in ('FAILED','CANCELLED')
        and rows[j]['User']==getpass.getuser() and rows[j]['JobName']==job_name(role)
        for role,j in rec['jobs'].items()),'PARENT_NOT_TERMINAL_OR_OWNER_NAME')

def prepare(out,preflight,reconciliation):
    require(not out.exists() and not NEW.exists(),'CREATE_ONCE_REPAIR_PREPARATION')
    oldlock,old=verify_frozen(PARENT)
    rec=json.loads(verify(member(reconciliation)).read_text())
    require(rec['status']=='OLD_TARGETS_TERMINAL_RELEASED' and rec['authorization']==AUTHORIZATION,
        'RECONCILIATION_REQUIRED')
    c=copy.deepcopy(old)
    c.update(attempt=str(NEW),run_instance=dict(date='2026-10-07',attempt=NEW.name),
        cpu_preflight=member(preflight))
    c['tracking']['cpu_review']=member(preflight)
    c['repair']=dict(authorization=AUTHORIZATION,reconciliation=member(reconciliation),
        parent_attempt=str(PARENT),parent_source=oldlock['source_commit'],
        native_reexecution='FRESH_COLD_W0_H0_NO_COMMIT_TO_RESUME',science_changed=False,
        failure_point='B1 candidate0 owner-group checkpoint backward',noCP=True,exact_resume='NOT_AVAILABLE')
    from .prepare import authority
    c['authority_members']=authority()+[member(reconciliation)]
    ready=json.loads((PARENT/'inputs/ready.json').read_text())
    require(ready['RNG_restored'] and ready['reference_fits']==ready['updates']==ready['history_appends']==0,
        'NATIVE_CONTEXT_NO_FIT_STATE')
    c['input_reuse_ready']=member(PARENT/'inputs/ready.json')
    for model in c['models'].values():
        from .inputs import augment
        bound=augment(dict(model,ordered_ids_sha256=c['ordered_ids_sha256']),ready)
        require(bound['cold_W0_H0']==old['models']['MEMIT']['cold_W0_H0'],'COMMON_W0_H0')
        # The repair changes only masked parity-local variables, not evaluator.
        for row in model['evaluator_sources']:
            rel=str(row['path']).split('/project/',1)[1]
            require(sha(ROOT/'project'/rel)==row['sha256'],'UNCHANGED_W0_EVALUATOR')
        model.update({k:bound[k] for k in ('packs','contexts','native_input_alignment','native_full_input_binding',
            'observer_identity','observation_identity','input_ready_sha256')})
    src=PARENT/'MEMIT_CAP075';folder=src/'W0'
    from .inputs import augment
    original_cell=augment(cell_config(old,'MEMIT_CAP075'),ready)
    original_runtime=json.loads((src/'runtime.json').read_text())
    require(original_runtime['source']==oldlock['source_commit']
        and original_runtime['cold_W0_H0']==original_cell['cold_W0_H0']
        and original_runtime['config']==digest(original_cell),'ORIGINAL_W0_EXECUTION_PROVENANCE')
    value=dict(status='QUALIFIED_EXACT_REUSE',cold_state=c['models']['MEMIT']['cold_W0_H0'],
        chunks=[member(p) for p in sorted(folder.glob('chunk-*.json'))],summary=member(folder/'summary.json'),
        runtime=member(src/'runtime.json'),observation_identity=ready['observation_identity'],source_folder=str(folder))
    require(len(value['chunks'])==40,'W0_FULL40_CHUNKS')
    rows=[]
    for row in value['chunks']:
        obj=json.loads(verify(row).read_text())
        require(obj['state']==value['cold_state'] and not obj['optimizer_feedback'],'W0_RAW_STATE')
        rows.extend(obj['rows'])
    refs=json.loads(verify(ready['observer_identity']).read_text())['rows']
    ids=[case for p in ready['packs'] for case in p['ids']]
    reduced=validate_rows(rows,refs,ids,'W0');summary=json.loads(verify(value['summary']).read_text())
    from .w0 import verify_summary
    verify_summary(summary,value['cold_state'],rows)
    require(len(rows)==26000 and len(ids)==2000 and reduced==summary['summary'],'W0_FULL_IDENTITY_COUNTS')
    for model in c['models'].values():model['W0_reuse']=copy.deepcopy(value)
    c['repair']['reuse']=dict(input_READY=c['input_reuse_ready'],context=ready['contexts_member'],
        W0_requests=2000,W0_prompt_pairs=26000,W0_summary=value['summary'],runtime_match='RECHECK_EACH_NEW_GPU_STARTUP',
        raw_copy=False,context_generation=0,W0_new_forwards=0,native_fit_resume=False)
    write(out,c)
    return dict(config=str(out),attempt=str(NEW),reuse=c['repair']['reuse'],cells=list(CELLS),
        cold_chains='FRESH_W0_H0',checkpoint_saved=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('reconcile','prepare'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',type=Path)
    p.add_argument('--reconciliation',type=Path);a=p.parse_args()
    result=reconcile(a.out.resolve()) if a.action=='reconcile' else prepare(
        a.out.resolve(),a.preflight.resolve(),a.reconciliation.resolve())
    print(json.dumps(result))
