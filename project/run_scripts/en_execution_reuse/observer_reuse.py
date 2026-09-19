"""Explicit same-host prior W0/N4 raw observer bridge, after all decisions seal."""
import copy
import json
from pathlib import Path
import torch
from .preparation import member, sha, create_json
from project.run_scripts.single_layer_edit_preserving_correction.common import digest


def bind_prior(rt, observer, weight, seal, role, nonselected, output):
    assets=rt.lock['prior_observer_reuse']
    for item in assets.values():
        if Path(item['path']).stat().st_size!=item['bytes'] or sha(item['path'])!=item['sha256']:
            raise ValueError('PRIOR_OBSERVER_SEAL_CHANGED')
    old=json.loads(Path(assets['runtime']['path']).read_text())
    if old['model_gpu_name']!=torch.cuda.get_device_name():raise ValueError('PRIOR_OBSERVER_GPU_CLASS_MISMATCH')
    keys=('W0','M0','P4','contexts','context_tokens','rng','records_digest','torch','transformers','physical_layer','canonical_microbatch')
    if any(old['identity'][k]!=rt.identity[k] for k in keys):raise ValueError('PRIOR_OBSERVER_RUNTIME_MISMATCH')
    if nonselected is not None:
        for key in ('nonselected_before','nonselected_after'):
            if json.loads(Path(assets[key]['path']).read_text())!=nonselected:
                raise ValueError('PRIOR_OBSERVER_MODEL_BYTES_MISMATCH')
    old_lock=json.loads(Path(assets['lock']['path']).read_text())
    if any(old_lock[k]!=rt.lock[k] for k in ('snapshot','model_revision','seed','config4','sample_order','torch','transformers')):
        raise ValueError('PRIOR_OBSERVER_MODEL_TOKENIZER_INPUT_CONFIG')
    # Actual physical/canonical evaluator source bytes are unchanged; the new
    # generated-reference adapter is not an evaluator input or endpoint change.
    current_source=Path(__file__).parents[1]/'single_layer_edit_preserving_correction/observer.py'
    if sha(current_source)!=assets['observer_source']['sha256']:raise ValueError('PRIOR_OBSERVER_SOURCE_MISMATCH')
    raw=assets[role];prior=json.loads(Path(raw['path']).read_text())
    current=observer.compatibility_for(rt.records,weight,selection_seal=seal)
    if prior['compatibility']['runtime_identity']!=digest(old['identity']):raise ValueError('PRIOR_OBSERVER_RUNTIME_DIGEST')
    for key in current:
        if key!='runtime_identity' and current[key]!=prior['compatibility'][key]:
            raise ValueError('PRIOR_OBSERVER_COMPATIBILITY_'+key)
    proof=dict(status='RAW_OBSERVER_COMPATIBILITY_BRIDGED',role=role,source=raw,
        old_runtime=old['identity'],new_runtime=rt.identity,matched_fields=list(keys),
        old_compatibility=prior['compatibility'],new_compatibility=current,
        complete_nonselected_parameter_bytes_equal=True if nonselected is not None else None,
        nonselected_byte_validation='CHECKED' if nonselected is not None else 'SKIPPED_USER_DIRECTED',
        observer_source=assets['observer_source'],
        differences_excluded_from_canonical_evaluator=['task/reference data identity','controller/source publication identity'],
        old_KL_or_choice_objective_reused=False,new_canonical_forwards=0,prior_work=prior['work'])
    create_json(output/f'prior-{role}-proof.json',proof)
    bridged=copy.deepcopy(prior);bridged['compatibility']=current
    bridged['same_endpoint_provenance_bridge']=proof
    return bridged
