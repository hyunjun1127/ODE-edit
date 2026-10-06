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
# Latest direct user instruction: method implementation only. A future explicit
# execution recall requires a new immutable source/lock; never reuse old grant.
EXECUTION_AUTHORIZED=False
def require_execution_authority():
    require(EXECUTION_AUTHORIZED,'METHOD_ONLY_USER_DIRECTED_NO_SUBMIT_OR_GPU')
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
