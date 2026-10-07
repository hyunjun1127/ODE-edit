from pathlib import Path
from project.run_scripts.base_model_eval.gpt2xl_server1_common import read,write,write_bytes
from project.run_scripts.jlz_realization.common import require,sha,digest,member,tensor_sha
from project.run_scripts.jlz_shared_budget.common import verify

ROOT=Path(__file__).resolve().parents[3]
PARENT_TASK='gpt2xl-baselines-fluency-consistency-2k'
PARENT_NONCE='USER-GH-ALL-SH-BASELINE-FLUENCY-CONSISTENCY-RERUN-20261007-R1'
TASK='gpt2xl-baselines-generation-cache-repair'
NONCE='USER-GH-SH1-SH2-BASELINE-GENERATION-KV-BATCH-REPAIR-20261008-R1'
REPAIR_ENVELOPE='messages/head/2026-10-08-baseline-generation-kv-batch-repair.json'
REPAIR_ENVELOPE_SHA='6902af1cb866fde3e6e01a0cd21cac7642db9c20de1bbdb86d18d72646371546'
SESSION='01a04939-f93a-7b50-bca0-65438eab2062'
ENVELOPE='messages/head/2026-10-07-baseline-fluency-consistency-rerun-server1.json'
ENVELOPE_SHA='8cf431e73196ecf199ba670dd7299c6b60b946f6b10d96b2178e199486f887da'
CONTRACT='plans/global/2026-10-07-baseline-fluency-consistency-rerun/contract.json'
CONTRACT_SHA='663c3c596613d413fbfb3d409d7eb24faaaf97e4ebd68b57b046f94f7947f357'
GENERATION_POLICY='control/generation-metric-policy.json'
GENERATION_POLICY_SHA='a0f7b8ac864f9d5f0cfa41a0d76d6441583972607c8b9327c893dd827be66b62'
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/gpt2xl-baselines-fluency-consistency-2k/20261007-v1')
ARMS=('BASE_MEMIT','BASE_ALPHAEDIT','CAKE','ALPHAEDIT_BLUE','PRUNE','RECT')
LAYERS=(13,14,15,16,17)
MILESTONES=(5,10,15,20)
SOURCE_ENV='GPT2_GENERATION_BASELINE_SOURCE_COMMIT'
PYTHON='/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
ORDERED_SHA='0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'
MANUAL_RECALL='USER-DIRECT-SH1-GPT2XL-GENERATION-CACHE-REPAIR-20261008-R2'
MANUAL_USER_QUOTE='fail 된거 다시 처리해'
TERMINAL_STATES=frozenset(('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY',
    'NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'))

def authority():
    require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA and sha(ROOT/CONTRACT)==CONTRACT_SHA
            and sha(ROOT/GENERATION_POLICY)==GENERATION_POLICY_SHA,'GENERATION_AUTHORITY')
    e,c,p=read(ROOT/ENVELOPE),read(ROOT/CONTRACT),read(ROOT/GENERATION_POLICY)
    require(e['instruction_id']==c['instruction_id']==p['instruction_id']==PARENT_NONCE
            and e['owner']['session']==SESSION and e['model']=='gpt2xl'
            and c['science']['BS']==100 and c['science']['batches']==20
            and c['science']['edits_per_baseline']==2000,'GENERATION_AUTHORIZED_SCOPE')
    require(sha(ROOT/REPAIR_ENVELOPE)==REPAIR_ENVELOPE_SHA,'CACHE_REPAIR_AUTHORITY_SHA')
    repair=read(ROOT/REPAIR_ENVELOPE);owner=repair['owners']['server1']
    require(repair['instruction_id']==NONCE and owner['session']==SESSION
        and owner['task_id']==TASK and owner['parent_task']==PARENT_TASK
        and repair['resources']['project_task_cap_server1']==2,'CACHE_REPAIR_AUTHORIZED_SCOPE')
    return c,p

def registered_repair_attempts():
    """Old attempts are history, not this new nonce's duplicate guard."""
    found=[]
    for path in LOCAL.glob('*/config.json'):
        value=read(path)
        if value.get('instruction_id')==NONCE and ((path.parent/'submission.json').exists()
            or list(path.parent.glob('submitted-*.json'))):found.append(path.parent)
    return found

def manual_retry_authority(attempt,row):
    """Verify this one explicit USER recall; never waive the original nonce.

    The prior immutable submission and a root's bounded terminal reconciliation
    are evidence, not a scientific resume or an automatic transport retry.
    Runtime/collector may verify the same receipt without querying Slurm.
    """
    attempt=Path(attempt)
    value=read(verify(row))
    require(value.get('schema')==1 and value.get('instruction_id')==NONCE
        and value.get('task_id')==TASK and value.get('manual_recall_id')==MANUAL_RECALL
        and value.get('authority_type')=='USER_EXPLICIT_MANUAL_REPAIR'
        and value.get('user_quote')==MANUAL_USER_QUOTE
        and value.get('automatic_retry') is False
        and value.get('owner')==dict(server='server1',session=SESSION),
        'MANUAL_RETRY_EXPLICIT_USER_AUTHORITY')
    prior=Path(value['prior_attempt'])
    require(attempt==LOCAL/'attempt-cache-repair-r2'
        and value['target_attempt']==str(attempt)
        and prior==LOCAL/'attempt-cache-repair-r1' and prior!=attempt,
        'MANUAL_RETRY_EXACT_ATTEMPT')
    bound={}
    for key,name in (('prior_config_member','config.json'),
        ('prior_lock_member','execution.lock.json'),('prior_submission_member','submission.json')):
        require(value[key]['path']==str(prior/name),'MANUAL_RETRY_PRIOR_MEMBER_PATH')
        bound[key]=read(verify(value[key]))
    config,lock,submission=(bound[key] for key in
        ('prior_config_member','prior_lock_member','prior_submission_member'))
    require(config['instruction_id']==lock['instruction_id']==submission['instruction_id']==NONCE
        and config['task_id']==lock['task_id']==submission['task_id']==TASK
        and config['attempt']==str(prior)
        and lock['source_commit']==submission['source_commit']==value['prior_source_commit']
        and lock['config_sha256']==value['prior_config_member']['sha256'],
        'MANUAL_RETRY_PRIOR_SOURCE_CONFIG')
    for key,row_key in (('config','prior_config_member'),('lock','prior_lock_member')):
        require(all(submission[key][field]==value[row_key][field]
            for field in ('path','bytes','sha256')),'MANUAL_RETRY_PRIOR_SUBMISSION_BINDING')
    jobs=value['prior_jobs'];roles=set(ARMS)|{'collector'}
    require(set(jobs)==roles and jobs==submission['jobs']
        and all(type(job) is str and job.isdigit() and int(job)>0 for job in jobs.values())
        and len(set(jobs.values()))==7,'MANUAL_RETRY_PRIOR_SEVEN_IDS')
    terminal=read(verify(value['terminal_reconciliation_member']))
    require(terminal['status']=='TERMINAL_RECONCILED'
        and terminal['prior_attempt']==str(prior)
        and terminal['source_commit']==value['prior_source_commit']
        and terminal['owner']==value['owner']
        and type(terminal['snapshot_at']) is str and bool(terminal['snapshot_at'])
        and set(terminal['jobs'])==roles,'MANUAL_RETRY_TERMINAL_RECONCILIATION')
    require(all(terminal['jobs'][role]['job']==jobs[role]
        and terminal['jobs'][role]['state'] in TERMINAL_STATES for role in roles),
        'MANUAL_RETRY_PRIOR_NOT_TERMINAL')
    return value

def registration_authority(attempt,row=None):
    """Only the specifically reconciled prior r1 may coexist with a new r2."""
    registered=set(registered_repair_attempts())
    if row is None:
        require(not registered,'NO_DUPLICATE_NONCE')
        return None
    value=manual_retry_authority(attempt,row)
    require(registered=={Path(value['prior_attempt'])},'MANUAL_RETRY_NO_OTHER_REGISTERED_ATTEMPT')
    return value

def bound_manual_authority(attempt,config,lock):
    row=config.get('manual_retry_authority_member')
    require(row==lock.get('manual_retry_authority_member')
        and config.get('manual_recall_id')==lock.get('manual_recall_id'),
        'MANUAL_RETRY_CONFIG_LOCK_BINDING')
    if row is None:return None
    value=manual_retry_authority(attempt,row)
    require(config['manual_recall_id']==value['manual_recall_id']
        and lock.get('automatic_retry') is False,'MANUAL_RETRY_RUNTIME_BINDING')
    return value

def batches(records):
    require(len(records)==2000 and digest([r['case_id'] for r in records])==ORDERED_SHA,'EXACT_ORDERED_FIRST2K')
    for n in range(20):yield n+1,records[n*100:(n+1)*100],records[:(n+1)*100]
