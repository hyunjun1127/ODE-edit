"""One-shot prospective W20 archive: independent receiver verify before unlink."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime,timezone
from project.run_scripts.checkpoint_archive import archive as a

TASK='qwen-baselines-server4-20261009'
ADAPTER_MODULE='project.run_scripts.server4_qwen_archive'


def read(p):return json.loads(Path(p).read_text())
def member(p):
    value=a.inspect_file(p)
    return {k:value[k] for k in ('path','bytes','sha256')}
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):a.write_once(Path(p),x)


def replay(kind,subject,doc,expected,original_member,contract):
    """Replay concrete runner/scheduler schemas, not caller Boolean assertions."""
    identity=expected['checkpoint_identity'];run=expected['run_identity']
    if kind=='checkpoint_authority':
        a.require(doc.get('schema')=='server4-qwen-new-submission-authority-v1','AUTHORITY_SCHEMA')
        a.require(doc['candidate_identity']=={k:v for k,v in run.items() if k!='actual_job_id'}
                  and doc['checkpoint_identity']==identity and doc['endpoint']=='W20'
                  and doc['checkpoint_creation_authorized'] is True,'AUTHORITY_SCOPE')
    elif kind=='final_calculation':
        a.require(doc.get('schema')=='server4-qwen-official-terminal-v1','TERMINAL_SCHEMA')
        a.require(doc['actual_job_id']==run['actual_job_id'] and doc['logical_main_row']==run['run_id']
                  and doc['checkpoint_identity']==identity and doc['status']=='W20_COMPLETE'
                  and doc['completed_edits']==2000 and doc['checkpoint']['batch']==20
                  and doc['checkpoint']['final_W20'] is True
                  and doc['checkpoint']['sha256']==expected['checkpoint_sha256']
                  and len(doc['commits'])==20,'TERMINAL_IDENTITY_ENDPOINT')
        a.require(subject in doc['calculation_evidence']
                  and doc['calculation_evidence'][subject]['state']=='COMPLETE'
                  and doc['calculation_evidence'][subject]['observed_requests']==2000
                  and doc['calculation_evidence'][subject]['sha256'],'ACTUAL_CALCULATION_REQUIRED')
    elif kind in ('writer_termination','consumer_clearance'):
        a.require(doc.get('schema')=='server4-qwen-scheduler-clearance-v1','SCHEDULER_SCHEMA')
        a.require(doc['run_identity']==run and doc['checkpoint_identity']==identity
                  and doc['checkpoint_sha256']==expected['checkpoint_sha256']
                  and doc['latest_pointer_sha256']==expected['latest_pointer_sha256']
                  and doc['scheduler_job_id']==run['actual_job_id']
                  and doc['scheduler_state']=='COMPLETED' and doc['scheduler_exit_code']=='0:0'
                  and doc['scheduler_raw'].split('|')[:3]==[run['actual_job_id'],'COMPLETED','0:0']
                  and doc['writer_processes_returned_zero'] is True,'SCHEDULER_TERMINAL_REQUIRED')
        if kind=='consumer_clearance':
            a.require(subject in ('writer','evaluation') and expected['required_state']=='TERMINAL'
                      and a.timestamp(doc['observed_at_utc'])>=a.timestamp(expected['checked_at_utc']),
                      'CONSUMER_NOT_CLEAR')
    else:raise a.ArchiveError('UNRECOGNIZED_ORIGINAL_EVIDENCE')
    return a._replay_result(dict(kind=kind,subject=subject,member=original_member,expected=expected),doc,contract)


def adopt(root,logical,identity,*,adapter_source=None):
    root=Path(root).resolve();folder=root/'archive'/logical
    cutover=member(root/'cutover.json');policy=member(root/'archive-policy.json')
    candidate=dict(origin_server='server4',task_id=TASK,run_id=logical,attempt=root.name,
                   registration_attempt_id=root.name+'-'+logical)
    contract=dict(module=ADAPTER_MODULE,function='replay',source_member=member(Path(adapter_source) if adapter_source else Path(__file__).absolute()))
    authority=dict(schema='server4-qwen-new-submission-authority-v1',candidate_identity=candidate,
                   checkpoint_identity=identity,endpoint='W20',checkpoint_creation_authorized=True,
                   instruction='USER transfers Qwen twelve cold2k baselines to server4; archive on server1')
    write(folder/'authority.json',authority)
    adoption=dict(schema='final-checkpoint-archive-adoption-v1',scope=a.SCOPE,instruction_id=a.INSTRUCTION,
                  policy_sha256=policy['sha256'],candidate_identity=candidate,
                  cutover_receipt_sha256=cutover['sha256'],adopted_at_utc=now())
    write(folder/'adoption.json',adoption)
    write(folder/'submission-lock.json',dict(schema='final-checkpoint-archive-submission-lock-v1',
        scope=a.SCOPE,policy_sha256=policy['sha256'],candidate_identity=candidate,
        checkpoint_identity=identity,adoption_sha256=member(folder/'adoption.json')['sha256'],
        sealed_at_utc=now(),checkpoint_directory=str(root/'runs'/logical/'checkpoint'),
        artifact_role='APPROVED_FINAL_EXPERIMENT_CHECKPOINT',checkpoint_creation_authorized=True,
        checkpoint_authority_member=member(folder/'authority.json'),evidence_adapter=contract,
        required_final_calculations=['factual','generation'] if '-cf-' in logical else ['factual'],
        planned_consumers=dict(writer='writer',evaluation='evaluation')))


def registered(root,logical,job,submitted_at):
    folder=Path(root)/'archive'/logical;lock=read(folder/'submission-lock.json')
    write(folder/'submission.json',dict(schema='final-checkpoint-archive-actual-submission-v1',
        run_identity=dict(lock['candidate_identity'],actual_job_id=str(job)),
        checkpoint_identity=lock['checkpoint_identity'],actual_registered_job_id=str(job),
        registration_response='Submitted batch job '+str(job),
        adoption_sha256=lock['adoption_sha256'],submission_lock_sha256=member(folder/'submission-lock.json')['sha256'],
        scheduler_submitted_at_utc=submitted_at,receipt_sealed_at_utc=now(),
        consumer_bindings=dict(writer=str(job),evaluation=str(job))))


def remote(root,action,payload):
    cfg=read(Path(root)/'receiver.json')
    # This is the SH1-owned receiver, not a local substitute for verification.
    program="""import sys,json,hashlib
from pathlib import Path
c=json.loads(sys.stdin.readline());p=json.load(sys.stdin)
sys.path.insert(0,c['source'])
for name,sha in c['helper_members'].items():
 assert hashlib.sha256((Path(c['source'])/name).read_bytes()).hexdigest()==sha,'RECEIVER_SOURCE_CHANGED'
assert hashlib.sha256(Path(c['policy']).read_bytes()).hexdigest()==c['policy_sha256'],'RECEIVER_POLICY_CHANGED'
from project.run_scripts.checkpoint_archive.archive import Receiver
r=Receiver.from_policy_file(Path(c['policy']),cutover_members=json.loads(Path(c['cutovers']).read_text()))
if c['action']=='admit': result=r.admit(p['manifest'])
elif c['action']=='verify':result=r.verify(p['admission'],p['manifest'])
elif c['action']=='recheck':result=r.recheck(p['receipt'])
else:raise RuntimeError('UNKNOWN_ACTION')
print(json.dumps(result))
"""
    import shlex
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',cfg['host'],
                           shlex.quote(cfg['python'])+' -c '+shlex.quote(program)],
                          input=json.dumps(dict(cfg,action=action))+'\n'+json.dumps(payload),
                          text=True,capture_output=True,timeout=3600)
    if result.returncode:raise RuntimeError('RECEIVER_'+action.upper()+'_FAILED_KEEP_SOURCE')
    return json.loads(result.stdout)


def archive(root,logical):
    from official.runners.server4.qwen_submission_plan import require_execution_enabled
    require_execution_enabled()
    root=Path(root).resolve();folder=root/'archive'/logical;out=root/'runs'/logical
    a.require(not (folder/'manifest.json').exists(),'NO_AUTOMATIC_ARCHIVE_RETRY_KEEP_SOURCE')
    submission=read(folder/'submission.json');lock=read(folder/'submission-lock.json')
    run=submission['run_identity'];identity=lock['checkpoint_identity']
    terminal=read(out/'terminal.json');pointer=read(out/'checkpoint/latest.json')
    checkpoint=out/'checkpoint'/pointer['file']
    a.require(checkpoint.parent==out/'checkpoint' and checkpoint.name==pointer['file'],'PAYLOAD_SCOPE')
    a.require(terminal['checkpoint']==pointer and terminal['checkpoint_identity']==identity,'FINAL_IDENTITY')
    sha=member(checkpoint)['sha256'];a.require(sha==pointer['sha256'],'FINAL_PAYLOAD_CHANGED')
    for entry in terminal['commits']:
        a.require(member(entry['path'])['sha256']==entry['sha256'],'COMMIT_CHANGED')
    for entry in terminal['calculation_evidence'].values():
        a.require(member(entry['path'])['sha256']==entry['sha256'],'CALCULATION_CHANGED')
    shared=dict(run_identity=run,checkpoint_identity=identity,checkpoint_sha256=sha,
                latest_pointer_sha256=member(out/'checkpoint/latest.json')['sha256'])
    companions=dict(latest_pointer=out/'checkpoint/latest.json',adoption=folder/'adoption.json',
        submission_lock=folder/'submission-lock.json',submission_receipt=folder/'submission.json')
    write(folder/'terminal-proof.json',dict(schema='final-checkpoint-scientific-terminal-v1',**shared,
        actual_complete=True,final_W20=True,completed_batch=20,completed_edits=2000,
        commit_batches=list(range(1,21)),actual_final_calculations_complete=True,
        completed_at_utc=terminal['completed_at_utc'],required_calculations={name:dict(
            state='COMPLETE',endpoint='W20',actual_measurement=True,checkpoint_identity_sha256=a.digest(identity),
            original_receipt_member=member(out/'terminal.json')) for name in lock['required_final_calculations']}))
    companions['scientific_terminal']=folder/'terminal-proof.json'
    companions.update(clearance(folder,'before',shared,terminal))
    manifest=a.seal_manifest(run=run,identity=identity,checkpoint=checkpoint,companions=companions,
        provenance_references=dict(evidence_adapter=lock['evidence_adapter'],
            pinned_source=member(root/'source-lock.json'),native_hparams=member(root/'configs'/f'{logical}.json'),
            base_model=member(root/'assets.json'),tokenizer=member(root/'assets.json'),
            input=member(root/'streams'/('cf-stream.lock.json' if '-cf-' in logical else 'zsre-stream.lock.json'))),
        policy_sha256=lock['policy_sha256'],cutover_member=member(root/'cutover.json'),evidence_adapter=replay)
    write(folder/'manifest.json',manifest)
    admission=remote(root,'admit',dict(manifest=manifest));write(folder/'admission.json',admission)
    cfg=read(root/'receiver.json')
    for role,row in manifest['files'].items():
        destination=admission['staging']+'/'+admission['incoming_names'][role]
        subprocess.run(['rsync','--protect-args','--checksum','--ignore-existing','--',row['path'],
                        cfg['host']+':'+destination],check=True,timeout=7200)
    receipt=remote(root,'verify',dict(admission=admission,manifest=manifest))
    write(folder/'verified-destination.json',receipt)
    fresh=dict(companions,**clearance(folder,'after',shared,terminal))
    gate=a.source_delete_gate(receipt,manifest,fresh_companions=fresh,
        destination_verify=lambda value:remote(root,'recheck',dict(receipt=value)),evidence_adapter=replay)
    a.require(gate['stage']=='ELIGIBLE_SOURCE_OWNER_FINAL_RECHECK_REQUIRED','SOURCE_KEEP')
    scheduler_terminal(run['actual_job_id'])
    before=manifest['files']['checkpoint']
    a.require(a.inspect_file(checkpoint)==before,'SOURCE_CHANGED_BEFORE_UNLINK')
    with a.parent_fd(checkpoint) as (fd,name):
        observed=os.stat(name,dir_fd=fd,follow_symlinks=False)
        a.require(a._stat_row(observed)=={k:before[k] for k in a._stat_row(observed)},'SOURCE_FINAL_STAT_CHANGED')
        os.unlink(name,dir_fd=fd);os.fsync(fd)
    write(folder/'source-removed.json',dict(stage='SOURCE_REMOVED_ARCHIVE_VERIFIED',
        source=str(checkpoint),destination=receipt['destination'],bytes=before['bytes'],
        sha256=before['sha256'],removed_at_utc=now(),metadata_raw_preserved=True))


def scheduler_terminal(job):
    raw=subprocess.check_output(['sacct','-n','-P','-X','-j',job,'--format=JobIDRaw,State,ExitCode'],text=True,timeout=30)
    lines=[line.strip() for line in raw.splitlines() if line.split('|')[0]==job]
    a.require(len(lines)==1 and lines[0].split('|')[:3]==[job,'COMPLETED','0:0'],'WRITER_NOT_TERMINAL_KEEP_SOURCE')
    return lines[0]


def clearance(folder,label,shared,terminal):
    job=shared['run_identity']['actual_job_id'];raw=scheduler_terminal(job);observed=now()
    original=dict(schema='server4-qwen-scheduler-clearance-v1',**shared,observed_at_utc=observed,
        scheduler_job_id=job,scheduler_state='COMPLETED',scheduler_exit_code='0:0',scheduler_raw=raw,
        writer_processes_returned_zero=terminal['status']=='W20_COMPLETE')
    path=folder/f'scheduler-{label}.json';write(path,original);ref=member(path)
    writer=folder/f'writer-{label}.json';consumer=folder/f'consumers-{label}.json'
    write(writer,dict(schema='final-checkpoint-writer-stopped-v1',**shared,
        state='STOPPED_NO_FUTURE_MUTATION',actual_writer_job_id=job,writer_lease_closed=True,
        stopped_at_utc=observed,original_writer_evidence_member=ref))
    write(consumer,dict(schema='final-checkpoint-all-consumers-clear-v1',**shared,
        inventory_complete=True,no_unlisted_dependents=True,state='ALL_CONSUMERS_VERIFIED_CLEAR',
        checked_at_utc=observed,consumers=[dict(consumer_id=k,kind=k,actual_job_id=job,
            state='TERMINAL',not_using_checkpoint=True,original_evidence_member=ref) for k in ('writer','evaluation')]))
    return dict(writer_termination=writer,consumer_clearance=consumer)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cell',required=True)
    args=p.parse_args()
    from project.run_scripts.server4_qwen_archive import archive as canonical_archive
    canonical_archive(args.root,args.cell)
