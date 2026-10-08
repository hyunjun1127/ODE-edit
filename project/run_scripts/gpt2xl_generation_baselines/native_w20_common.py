"""Explicit new MEMIT/AlphaEdit/CAKE authority; previous W20 profile immutable."""
from pathlib import Path
from .common import (read,write,write_bytes,require,sha,digest,member,tensor_sha,verify,
    LOCAL,SESSION,PYTHON,ORDERED_SHA,CONTRACT,CONTRACT_SHA,PARENT_TASK)

ROOT=Path(__file__).resolve().parents[3]
TASK='gpt2xl-memit-alphaedit-cake-w20-generation'
NONCE='USER-GH-SH1-GPT2XL-MEMIT-ALPHAEDIT-CAKE-W20-GENERATION-20261008-R1'
ENVELOPE='messages/head/2026-10-08-gpt2xl-memit-alphaedit-cake-w20-generation.json'
ENVELOPE_SHA='c865c14ba386cacac7380b89bbdbf6c800c194b9d363f1c869256b970da9eccb'
ARMS=('BASE_MEMIT','BASE_ALPHAEDIT','CAKE')
MILESTONES=(5,10,15,20)
SOURCE_ENV='GPT2_NATIVE_W20_BASELINE_SOURCE_COMMIT'
SCHEDULE='W20_ONLY_FIRST2000'
GENERATION_POLICY='control/generation-metric-policy.json'
SCHEDULE_INSTRUCTION='USER-GH-SH1-GPT2XL-BLUE-PRUNE-RECT-W20-GENERATION-20261008-R1'
POLICY_SHA={GENERATION_POLICY:'bb147bfff33d86f9a7a31d9b3193fd00a6075b3ab2df7496cad727cd9dbcf412',
    'control/wandb-policy.json':'02a15a637991320461c5ce8fd560f7001f2219a4c716ba2f8466c7e2f01fbf7d',
    'control/wandb-method-metric-schema.json':'122320e9e9e1f10cb72bf540f22b3b34c60435f3144155a699882f5fc24fb11a',
    'control/gpu-concurrency-policy.tsv':'00736e8bbdb649e1ec5fd2e8913b9a1db80aeab9e71dcf845509a749505084a4'}
OLD_TARGET_IDS=dict(BASE_MEMIT='61167',BASE_ALPHAEDIT='61168',CAKE='61169')
PROTECTED_IDS=dict(ALPHAEDIT_BLUE='61436',PRUNE='61437',RECT='61438',collector='61439')
TERMINAL_STATES=frozenset(('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY',
    'NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'))

def authority():
    require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA and sha(ROOT/CONTRACT)==CONTRACT_SHA,
        'NATIVE_W20_AUTHORITY_BYTES')
    for path,expected in POLICY_SHA.items():require(sha(ROOT/path)==expected,'NATIVE_W20_POLICY_BYTES:'+path)
    value=read(ROOT/ENVELOPE);policy=read(ROOT/GENERATION_POLICY);scope=value['scope']
    require(value['instruction_id']==NONCE and value['task_id']==TASK
        and value['owner']['server']=='server1' and value['owner']['session']==SESSION
        and scope['methods']==list(ARMS)
        and scope['batch_size']==100 and scope['batches']==20 and scope['edits_per_method']==2000
        and scope['generation_schedule']==policy['generation_schedule']==SCHEDULE
        and policy['instruction_id']==SCHEDULE_INSTRUCTION
        and policy['generation']['profile']=='cf-cake-prompt-inclusive-total100-eos-corrected-v1'
        and scope['new_BLUE_PRUNE_RECT_OURS_baseline_sweep_or_pilot']==0
        and value['resources']['server1_combined_project_GPU_cap']==2,
        'NATIVE_W20_AUTHORITY_EXPLICIT_SCOPE_AND_SCHEDULE_INHERITANCE')
    return value,policy

def registered_attempts():
    found=[]
    for path in LOCAL.glob('*/config.json'):
        value=read(path)
        if value.get('instruction_id')==NONCE and ((path.parent/'submission.json').exists()
            or (path.parent/'registration-pass-started.json').exists()
            or list(path.parent.glob('submitted-*.json'))):found.append(path.parent)
    return found

def validate_config(value):
    require(value['instruction_id']==NONCE and value['task_id']==TASK
        and set(value['source_configs'])==set(ARMS) and value['arms']==list(ARMS)
        and Path(value['attempt']).parent==LOCAL,'NATIVE_W20_CONFIG_EXACT_THREE')
    generation=value['generation']
    require(generation['schedule']==generation['generation_schedule']==SCHEDULE
        and generation['phase']=='W20_generation'
        and not any(key in generation for key in ('shared_W0_root','old_w0_reuse','primary_arm'))
        and value['W0_generation_required'] is False
        and value['generation_endpoints']==[dict(endpoint='all_seen/post',state='W20',edits=2000,requests=2000)],
        'NATIVE_W20_ONLY_NO_W0_GENERATION_PREREQUISITE')
    require(value['noCP'] is True and value['z_disk_cache'] is False
        and value['exact_resume']=='NOT_AVAILABLE','NATIVE_W20_NOCP_NATIVE_Z_CACHE_OFF')
    return value

def transition_receipt(row,source_attempt):
    value=read(verify(row));owner=dict(server='server1',session=SESSION)
    require(value['instruction_id']==NONCE and value['task_id']==TASK and value['owner']==owner
        and value['source_attempt']==str(source_attempt)
        and value['protected_jobs_preserved'] is True and value['old_source_raw_preserved'] is True
        and type(value['snapshot_at']) is str and value['snapshot_at']
        and set(value['target_jobs'])==set(ARMS),'NATIVE_W20_EXACT_TRANSITION_SCOPE')
    require(all(value['target_jobs'][arm]['job']==OLD_TARGET_IDS[arm]
        and value['target_jobs'][arm]['state'] in TERMINAL_STATES for arm in ARMS),
        'NATIVE_W20_OLD_TARGETS_RECONCILED_TERMINAL')
    return value

def session_boundary():
    path=ROOT/'servers/local/session-boundary.env'
    values=dict(line.split('=',1) for line in path.read_text().splitlines()
        if line and not line.startswith('#'))
    require(values==dict(ODEEDIT_REPOSITORY_ID='hyunjun1127/ODE-edit',ODEEDIT_REPOSITORY_CWD=str(ROOT),
        ODEEDIT_CODEX_SESSION_ID=SESSION,ODEEDIT_CURRENT_SERVER='server1',
        ODEEDIT_APP_CWD='/mnt/raid5/janghj/ODE-edit'),'NATIVE_W20_OWN_SESSION_BOUNDARY')
    return member(path),dict(session=SESSION,server='server1',repository='hyunjun1127/ODE-edit',
        actual_app_CWD=values['ODEEDIT_APP_CWD'],dedicated_worktree_CWD=str(ROOT),
        legacy_registry_CWD='/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit',
        legacy_CWD_is_historical=True,root_dirty_preserved=True)

def batches(records):
    require(len(records)==2000 and digest([r['case_id'] for r in records])==ORDERED_SHA,
        'NATIVE_W20_EXACT_ORDERED_FIRST2K')
    for index in range(20):yield index+1,records[index*100:(index+1)*100],records[:(index+1)*100]
