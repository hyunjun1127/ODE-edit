"""Exact scalar W0 references only; never state/checkpoint continuation."""
import json
from pathlib import Path
from .common import require,member,verify,digest,rows_from,validate_rows
from .storage import write

RUNTIME_FIELDS=('device','torch','transformers','model','FP32','geometry_FP64','eager','TF32','autocast','CPU_threads','cold_W0_H0')

def choose_reuse(c,attempt):
    current=json.loads((attempt/c['cell']/'runtime.json').read_text())
    candidates=[]
    if c.get('W0_reuse',{}).get('status')=='QUALIFIED_EXACT_REUSE':candidates.append(c['W0_reuse'])
    from .common import CELLS
    for cell in CELLS:
        if cell==c['cell']:break
        root=attempt/cell;folder=root/'W0'
        if not (folder/'summary.json').exists() or not (root/'runtime.json').exists():continue
        runtime=json.loads((root/'runtime.json').read_text())
        if any(runtime[k]!=current[k] for k in RUNTIME_FIELDS):continue
        if (folder/'cap-reuse.json').exists():
            value=json.loads((folder/'cap-reuse.json').read_text())['manifest']
        else:
            value=dict(status='QUALIFIED_EXACT_REUSE',cold_state=c['cold_W0_H0'],
                chunks=[member(p) for p in sorted(folder.glob('chunk-*.json'))],summary=member(folder/'summary.json'),
                runtime=member(root/'runtime.json'),observation_identity=c['observation_identity'],source_folder=str(folder))
        candidates.append(value)
    for value in candidates:
        if value['observation_identity']!=c['observation_identity']:continue
        old=json.loads(verify(value['runtime']).read_text())
        if all(old[k]==current[k] for k in RUNTIME_FIELDS):return value
    return None

def chunks(folder,expected_state=None):
    ref=json.loads((Path(folder)/'cap-reuse.json').read_text());value=ref['manifest']
    require(digest(value)==ref['manifest_sha256'],'W0_REFERENCE_HASH')
    require(value['cold_state']==ref['actual_cold_state'],'W0_REFERENCE_STATE')
    if expected_state is not None:require(expected_state==value['cold_state'],'W0_ACTUAL_COLD_STATE')
    require(len(value['chunks'])==40,'W0_FULL_CHUNKS')
    return [verify(row) for row in value['chunks']]

def install(folder,c,actual_cold_state,value):
    require(actual_cold_state==value['cold_state']==c['cold_W0_H0'],'W0_COLD_IDENTITY')
    require(value['observation_identity']==c['observation_identity'],'W0_MODEL_OBSERVER_IDENTITY')
    rows=[]
    for row in value['chunks']:
        obj=json.loads(verify(row).read_text());require(obj['state']==actual_cold_state and not obj['optimizer_feedback'],'W0_RAW_IDENTITY')
        rows.extend(obj['rows'])
    identities=json.loads(verify(c['observer_identity']).read_text())['rows']
    ids=[i for p in c['packs'] for i in p['ids']]
    reduced=validate_rows(rows,identities,ids,'W0');old=json.loads(verify(value['summary']).read_text())
    require(old['summary']==reduced and len(rows)==26000,'W0_INDEPENDENT_COUNTS')
    write(folder/'cap-reuse.json',dict(manifest=value,manifest_sha256=digest(value),actual_cold_state=actual_cold_state,
        no_forward=True,no_raw_copy=True,no_checkpoint=True))
    write(folder/'summary.json',dict(old,seconds=0.,new_forwards=0,reference_only=True,
        original_evaluation_seconds=old.get('original_evaluation_seconds',old['seconds']),reused_from=value['source_folder']))
