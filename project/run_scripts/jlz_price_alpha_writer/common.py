"""New six-cell task identity; historical PRICE namespace remains untouched."""
from project.run_scripts.jlz_interference_l1 import *
TASK='jlz-price-alpha-writer-2k'
NONCE='USER-GH-SH4-JLZ-PRICE-ALPHA-WRITER-2K-20261007-R1'
LOCAL=Path('/data/janghj/ODE-edit/local/jlz-price-alpha-writer-2k')
DESIGN='project/proposals/jlz-alpha-writer-review'
ENVELOPE='messages/head/2026-10-07-price-alpha-writer-2k-sh4.json'
ARMS=('CAP075','CAP100','FREE100')
MODELS=('LLAMA','QWEN')
CELLS=tuple(m+'_AE_'+a for m in MODELS for a in ARMS)
def history_expected(arm):
    require(arm in ARMS,'ARM');return 100

def cell_config(config,cell):
    import copy
    require(cell in CELLS,'CELL')
    model,_,arm=cell.split('_',2)
    c=copy.deepcopy(config)
    c.update(c.pop('models')[model]);c['model_profile']=model;c['arm']=arm;c['cell']=cell
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
