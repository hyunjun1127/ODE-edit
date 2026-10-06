"""New six-cell task identity; historical PRICE namespace remains untouched."""
from project.run_scripts.jlz_interference_l1 import *
TASK='jlz-price-gpt2xl-2k'
NONCE='USER-GH-SH1-JLZ-PRICE-GPT2XL-SIXARM-2K-20261007-R1'
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/jlz-price-gpt2xl-2k')
DESIGN='project/proposals/jlz-price-gpt2xl-2k'
ENVELOPE='messages/head/2026-10-07-price-gpt2xl-sixarm-sh1.json'
ARMS=('CAP075','CAP100','FREE100')
MODELS=('MEMIT','ALPHAEDIT')
CELLS=tuple(m+'_'+a for m in MODELS for a in ARMS)
# Explicit execution recall supersedes the historical method-only source.
# Bind the exact authority bytes in both the preparation and frozen runtime.
EXECUTION_AUTHORIZED=True
EXECUTION_NONCE='USER-GH-PRICE-MODEL-RUNS-TRACKING-20261007-SERVER1'
EXECUTION_ENVELOPE='messages/head/2026-10-07-price-model-runs-tracking.json'
EXECUTION_ENVELOPE_SHA='f978c411ae5a0001ee155214825fbe1b6223bb8630624e381ba5dd150d503320'
def require_execution_authority():
    require(EXECUTION_AUTHORIZED,'EXECUTION_NOT_AUTHORIZED')
    require(sha(ROOT/EXECUTION_ENVELOPE)==EXECUTION_ENVELOPE_SHA,'EXECUTION_AUTHORITY_BYTES')
    value=json.loads((ROOT/EXECUTION_ENVELOPE).read_text())
    require(value['instruction_id']=='USER-GH-PRICE-MODEL-RUNS-TRACKING-20261007','EXECUTION_AUTHORITY_ID')
    target=next(t for t in value['targets'] if t['server']=='server1')
    require(target['session']=='01a04939-f93a-7b50-bca0-65438eab2062'
        and target['models']==['gpt2xl'] and target['task_ids']==[TASK],'EXECUTION_AUTHORITY_SCOPE')
def history_expected(arm):
    require(arm in ARMS,'ARM');return 100

def cell_config(config,cell):
    import copy
    require(cell in CELLS,'CELL')
    model,arm=cell.split('_',1)
    c=copy.deepcopy(config)
    c.update(c.pop('models')[model]);c['model_profile']='GPT2XL';c['writer']='memit' if model=='MEMIT' else 'alphaedit';c['arm']=arm;c['cell']=cell
    c['arm_profiles']={cell:c.pop('profiles')[arm]}
    return c

def rows_from(folder,expected_state=None):
    from project.run_scripts.jlz_interference_l1 import rows_from as original
    folder=Path(folder)
    if not (folder/'cap-reuse.json').exists():return original(folder,expected_state)
    from .w0 import chunks
    rows=[]
    for path in chunks(folder,expected_state):
        obj=json.loads(path.read_text());require(not obj['optimizer_feedback'],'OBSERVATION_FEEDBACK')
        rows.extend(obj['rows'])
    return rows
