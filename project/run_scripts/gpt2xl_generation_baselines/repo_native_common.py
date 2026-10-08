"""Task-private direct USER native-generation override; old profiles immutable."""
from pathlib import Path
from .common import (read,write,write_bytes,require,sha,digest,member,tensor_sha,verify,
    LOCAL,SESSION,PYTHON,ORDERED_SHA,CONTRACT,CONTRACT_SHA,PARENT_TASK)

ROOT=Path(__file__).resolve().parents[3]
TASK='gpt2xl-baselines-native-generation-repair'
NONCE='USER-DIRECT-NATIVE-FLUCON-REPAIR-20261008-R1'
ENVELOPE='plans/updates/server1/gpt2xl-baselines-native-generation-repair/user-authority.json'
ENVELOPE_SHA='fb974ed38808198aa0b95fd2eb49b2bb238efd5f0a98eb0a7eca13ed7e2a91bd'
ARMS=('BASE_MEMIT','BASE_ALPHAEDIT','ALPHAEDIT_BLUE')
MILESTONES=(5,10,15,20)
SOURCE_ENV='GPT2_REPO_NATIVE_BASELINE_SOURCE_COMMIT'
SCHEDULE='W20_ONLY_FIRST2000'
PROFILE='cf-cake-native-casebatch-kv-total100-globalrng-v1'
GENERATION_POLICY='control/generation-metric-policy.json'
POLICY_SHA={GENERATION_POLICY:'bb147bfff33d86f9a7a31d9b3193fd00a6075b3ab2df7496cad727cd9dbcf412',
    'control/wandb-policy.json':'02a15a637991320461c5ce8fd560f7001f2219a4c716ba2f8466c7e2f01fbf7d',
    'control/wandb-method-metric-schema.json':'122320e9e9e1f10cb72bf540f22b3b34c60435f3144155a699882f5fc24fb11a',
    'control/gpu-concurrency-policy.tsv':'00736e8bbdb649e1ec5fd2e8913b9a1db80aeab9e71dcf845509a749505084a4'}
OLD_TARGET_IDS=dict(ALPHAEDIT_BLUE='61436',PRUNE='61437',RECT='61438',collector='61439')
PROTECTED_IDS={}
TERMINAL_STATES=frozenset(('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY',
    'NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'))

def authority():
    require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA,'REPO_NATIVE_DIRECT_USER_AUTHORITY_BYTES')
    require(sha(ROOT/CONTRACT)==CONTRACT_SHA,'REPO_NATIVE_PARENT_CONTRACT_BYTES')
    for path,expected in POLICY_SHA.items():require(sha(ROOT/path)==expected,'REPO_NATIVE_POLICY_BYTES:'+path)
    value=read(ROOT/ENVELOPE);policy=read(ROOT/GENERATION_POLICY)
    require(value['instruction_id']==NONCE and value['task_id']==TASK
        and value['owner']==dict(server='server1',session=SESSION)
        and value['arms']==list(ARMS) and value['generation_schedule']==SCHEDULE
        and value['project_GPU_cap']==2 and value['noCP'] is True
        and value['raw_preserve'] is True and value['automatic_retry'] is False
        and value['generation_override']=='CAKE-native per-case padded KV batch, prompt-inclusive total100, topk5, no EOS early stop; remove derived equal-length MB8 qualification and no-cache fallback',
        'REPO_NATIVE_EXPLICIT_USER_SCOPE')
    require(policy['generation_schedule']==SCHEDULE,'REPO_NATIVE_W20_SCHEDULE_PRESERVED')
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
        and Path(value['attempt']).parent==LOCAL,'REPO_NATIVE_CONFIG_EXACT_THREE')
    generation=value['generation']
    require(generation['schedule']==generation['generation_schedule']==SCHEDULE
        and generation['phase']=='W20_generation' and generation['profile']==PROFILE
        and not any(key in generation for key in ('shared_W0_root','old_w0_reuse','primary_arm',
            'qualification_plan_member','qualification_plan_sha256','qualification_receipts',
            'qualification_receipt_member'))
        and value['W0_generation_required'] is False
        and value['generation_endpoints']==[dict(endpoint='all_seen/post',state='W20',edits=2000,requests=2000)],
        'REPO_NATIVE_W20_NO_OLD_QUALIFICATION_OR_W0_PREREQUISITE')
    require(value['noCP'] is True and value['z_disk_cache'] is False
        and value['exact_resume']=='NOT_AVAILABLE','REPO_NATIVE_NOCP')
    return value

def transition_receipt(row,source_attempt=None):
    value=read(verify(row))
    require(value['instruction_id']==NONCE and value['owner']==dict(server='server1',session=SESSION)
        and value['unrelated_jobs_mutated'] is False and value['raw_source_preserved'] is True
        and value['order']==['collector','RECT','PRUNE','ALPHAEDIT_BLUE']
        and set(value['before'])==set(OLD_TARGET_IDS),'REPO_NATIVE_TRANSITION_EXACT_SCOPE')
    require(all(value['before'][role]['JobId']==job for role,job in OLD_TARGET_IDS.items())
        and all(action.get('response',{}).get('returncode',0)==0 for action in value['actions']),
        'REPO_NATIVE_TRANSITION_ACTION_RECEIPTS')
    return value

def session_boundary():
    path=ROOT/'servers/local/session-boundary.env'
    values=dict(line.split('=',1) for line in path.read_text().splitlines()
        if line and not line.startswith('#'))
    require(values==dict(ODEEDIT_REPOSITORY_ID='hyunjun1127/ODE-edit',ODEEDIT_REPOSITORY_CWD=str(ROOT),
        ODEEDIT_CODEX_SESSION_ID=SESSION,ODEEDIT_CURRENT_SERVER='server1',
        ODEEDIT_APP_CWD='/mnt/raid5/janghj/ODE-edit'),'REPO_NATIVE_SESSION_BOUNDARY')
    return member(path),dict(session=SESSION,server='server1',repository='hyunjun1127/ODE-edit',
        actual_app_CWD=values['ODEEDIT_APP_CWD'],dedicated_worktree_CWD=str(ROOT),
        legacy_registry_CWD='/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit',
        legacy_CWD_is_historical=True,root_dirty_preserved=True)

def batches(records):
    require(len(records)==2000 and digest([r['case_id'] for r in records])==ORDERED_SHA,
        'REPO_NATIVE_EXACT_ORDERED_FIRST2K')
    for index in range(20):yield index+1,records[index*100:(index+1)*100],records[:(index+1)*100]
