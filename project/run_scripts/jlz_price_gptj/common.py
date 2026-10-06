"""New six-cell task identity; historical PRICE namespace remains untouched."""
from project.run_scripts.jlz_interference_l1 import *
TASK='jlz-price-gptj-2k'
NONCE='USER-SH4-GPTJ-MEMIT-ALPHA-2K-20261007'
LOCAL=Path('/data/janghj/ODE-edit/local/jlz-price-gptj-2k')
DESIGN='project/proposals/jlz-price-cap-budget-review'
ENVELOPE='messages/head/2026-10-06-price-cap-base-repair-2k-sh4.json'
ARMS=('CAP075','CAP100','FREE100')
MODELS=('MEMIT','ALPHA')
CELLS=tuple(m+'_'+a for m in MODELS for a in ARMS)
def history_expected(arm):
    require(arm in ARMS,'ARM');return 20*6

def cell_config(config,cell):
    import copy
    require(cell in CELLS,'CELL')
    model,arm=cell.split('_',1)
    c=copy.deepcopy(config)
    c.update(c.pop('models')[model]);c['model_profile']='GPTJ';c['writer']='memit' if model=='MEMIT' else 'alphaedit';c['arm']=arm;c['cell']=cell
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
