"""Explicit execution horizons. Default S4 production remains 20 batches."""
from . import TASK, NONCE, MILESTONES, require

B1_TASK = 'causal-allocation-editing-b1'
B1_NONCE = 'USER-GH-SH2-CAUSAL-ALLOCATION-EDITING-B1'

def execution(c):
    if c.get('execution_profile', 'production2k') == 'production2k':
        return dict(task=TASK, nonce=NONCE, batches=20, requests=2000,
                    milestones=MILESTONES, status='W20_COMPLETE', no_next='no_B21')
    require(c['execution_profile'] == 'server2-b1', 'UNKNOWN_EXECUTION_PROFILE')
    require(c['task_id'] == B1_TASK and c['instruction_id'] == B1_NONCE, 'B1_AUTHORITY')
    return dict(task=B1_TASK, nonce=B1_NONCE, batches=1, requests=100,
                milestones=(1,), status='W1_COMPLETE', no_next='no_B2')

def check_horizon(c):
    p=execution(c);s=c['settings']
    require((s['B'],s['batches'],s['requests']) == (100,p['batches'],p['requests'])
            and len(c['packs']) == p['batches'], 'EXPLICIT_PROFILE_HORIZON')
    return p
