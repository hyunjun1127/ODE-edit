"""New six-cell task identity; historical PRICE namespace remains untouched."""
from . import *
TASK='jlz-price-cap-base-repair-2k'
NONCE='USER-GH-SH4-JLZ-PRICE-CAP-BASE-REPAIR-2K-20261006-R1'
LOCAL=Path('/data/janghj/ODE-edit/local/jlz-price-cap-base-repair-2k')
DESIGN='project/proposals/jlz-price-cap-budget-review'
ENVELOPE='messages/head/2026-10-06-price-cap-base-repair-2k-sh4.json'
ARMS=('CAP075','CAP100','FREE100')
MODELS=('LLAMA','QWEN')
CELLS=tuple(m+'_'+a for m in MODELS for a in ARMS)
def history_expected(arm):
    require(arm in ARMS,'ARM');return 100

def cell_config(config,cell):
    import copy
    require(cell in CELLS,'CELL')
    model,arm=cell.split('_',1)
    c=copy.deepcopy(config)
    c.update(c.pop('models')[model]);c['model_profile']=model;c['arm']=arm;c['cell']=cell
    c['arm_profiles']={cell:c.pop('profiles')[arm]}
    return c

def rows_from(folder,expected_state=None):
    from . import rows_from as original
    folder=Path(folder)
    if not (folder/'cap-reuse.json').exists():return original(folder,expected_state)
    from .cap_w0 import chunks
    rows=[]
    for path in chunks(folder,expected_state):
        obj=json.loads(path.read_text());require(not obj['optimizer_feedback'],'OBSERVATION_FEEDBACK')
        rows.extend(obj['rows'])
    return rows
