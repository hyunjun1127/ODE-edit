"""INCOMPLETE preparation draft; execution disabled until reviewed integration.

Open gates: final-calculation evidence and post-destination fresh clearance.
This file is not an operational transfer/deletion receipt or executable approval.
"""
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


def adopt(root,logical,identity):
    root=Path(root).resolve();folder=root/'archive'/logical
    cutover=member(root/'cutover.json');policy=member(root/'archive-policy.json')
    candidate=dict(origin_server='server4',task_id=TASK,run_id=logical,attempt=root.name,
                   registration_attempt_id=root.name+'-'+logical)
    contract=dict(module=ADAPTER_MODULE,function='replay',source_member=member(Path(__file__).absolute()))
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
                           'python3 -c '+shlex.quote(program)],
                          input=json.dumps(dict(cfg,action=action))+'\n'+json.dumps(payload),
                          text=True,capture_output=True,timeout=3600)
    if result.returncode:raise RuntimeError('RECEIVER_'+action.upper()+'_FAILED_KEEP_SOURCE')
    return json.loads(result.stdout)


def archive(root,logical):
    from official.runners.server4.qwen_submission_plan import require_execution_enabled
    require_execution_enabled()
    raise RuntimeError('ARCHIVE_CALLER_INTEGRATION_NOT_VALIDATED_KEEP_SOURCE')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cell',required=True)
    args=p.parse_args();archive(args.root,args.cell)
