"""Same-server cold observation reuse with explicit selected-state expansion.

The old L13 task captured the unedited full model. We retain its raw rows,
verify all five original weight hashes against the same model asset binding,
and explicitly distinguish the old observed H/W subset from the new editor.
No forward, model load, rewritten observation or W0 generation is performed.
"""
import json
from pathlib import Path
from project.run_scripts.jlz_price_gpt2xl.common import verify,require,digest,validate_rows
from project.run_scripts.jlz_price_gpt2xl.storage import write


def validate(c):
    value=c['W0_reuse'];old=json.loads(verify(value['summary']).read_text());rows=[]
    for item in value['chunks']:
        obj=json.loads(verify(item).read_text())
        require(obj['state']==value['cold_state'] and obj['optimizer_feedback'] is False,'SAVED_COLD_RAW_STATE')
        rows.extend(obj['rows'])
    identities=json.loads(verify(c['observer_identity']).read_text())['rows']
    reduced=validate_rows(rows,identities,[i for p in c['packs'] for i in p['ids']],'W0')
    from project.run_scripts.jlz_price_gpt2xl.w0 import verify_summary
    verify_summary(old,value['cold_state'],rows)
    require(old['summary']==reduced and len(rows)==26000,'SAVED_COLD_FULL_2K_REDUCTION')
    return old


def install_runtime_bindings(parent,c):
    from project.run_scripts.jlz_price_gpt2xl import inputs,w0,tracking
    original_rows=parent.rows_from
    def bind(config,attempt,model,tok,records):
        require(digest([r['case_id'] for r in records])==config['ordered_ids_sha256'],'REUSED_INPUT_ORDER')
        for key in ('contexts_member','native_input_alignment','native_full_input_binding','observer_identity'):
            verify(config[key])
        require(config['contexts']==config['contexts_member']['path'],'REUSED_CONTEXT_POINTER')
    def choose(config,attempt):
        value=config['W0_reuse'];old=json.loads(verify(value['runtime']).read_text())
        current=json.loads((attempt/config['cell']/'runtime.json').read_text())
        for key in ('device','torch','transformers','model','FP32','geometry_FP64','eager','TF32','autocast','CPU_threads'):
            require(old[key]==current[key],'SAVED_W0_RUNTIME_DIFFERENCE_'+key)
        require(old['cold_W0_H0']==value['cold_state'],'ORIGINAL_COLD_STATE')
        require(current['cold_W0_H0']==config['cold_W0_H0'],'ACTUAL_FULL_WINDOW_COLD')
        require(config['full_cold_W']==current['cold_W0_H0']['W'],'FULL_COLD_MODEL_WEIGHT_BINDING')
        require(value['observation_identity']==config['observation_identity'],'SAVED_W0_OBSERVATION_IDENTITY')
        return value
    def install(folder,config,state,value):
        old=validate(config)
        require(state==config['cold_W0_H0'],'NEW_EDITOR_COLD_STATE')
        write(folder/'task-w0-reference.json',dict(manifest=value,manifest_sha256=digest(value),
            original_observed_state=value['cold_state'],actual_editor_state=state,
            full_cold_weight_binding=config['full_cold_W'],no_forward=True,no_raw_copy=True))
        write(folder/'summary.json',dict(old,state=state,seconds=0,new_forwards=0,reference_only=True,
            original_observed_state=value['cold_state'],reused_from=value['source_folder']))
    def rows(folder,expected_state=None):
        path=Path(folder)/'task-w0-reference.json'
        if not path.exists():return original_rows(folder,expected_state)
        ref=json.loads(path.read_text());value=ref['manifest']
        require(digest(value)==ref['manifest_sha256'],'W0_REFERENCE_CONTENT')
        require(expected_state is None or expected_state==ref['actual_editor_state'],'W0_READER_EDITOR_STATE')
        result=[]
        for item in value['chunks']:
            obj=json.loads(verify(item).read_text());require(obj['state']==ref['original_observed_state'],'W0_READER_ORIGIN')
            result.extend(obj['rows'])
        return result
    # Only the new task process uses these imported old runner entrypoints.
    # Existing jobs/archives/shared tracking and generation modules stay intact.
    inputs.bind=bind;w0.choose_reuse=choose;w0.install=install;parent.rows_from=rows
    # tracking imported rows_from into its own globals: rebinding the runner
    # alone does not affect w0_rows() used by drive(). Same validated reader.
    tracking.rows_from=rows
