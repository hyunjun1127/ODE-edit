"""Conditional clamp .5 final-CF generation; old evaluator files stay unchanged."""
import argparse
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
from official.experiments.prepare import digest
from official.runners.server1.common import read, verify, require
from official.diagnostics.fe_qwen_current import flucon as base, flucon_prepare
from official.diagnostics.fe_qwen_current import rescue_clamp as rescue

SPEC = rescue.SPEC
TASK = 'fe-qwen-current-z-clamp-0p5-final-cf-flucon-20261012'
_audit = base.audit_final
_tracking = base.tracking_values


def native_identity(identity, native):
    require(identity.get('clamp_norm_factor') == .5 and identity.get('L2') == 100
        and identity.get('author_hparam_profile') == SPEC['protocol']
        and identity.get('selection_contract_sha256') == digest(SPEC), 'CLAMP_FINAL_IDENTITY')
    require(native['llms']['clamp_norm_factor'] == .5 and native['algs']['L2'] == 100,
        'CLAMP_FINAL_NATIVE_CONFIG')


def audit_final(member):
    # The unchanged independent CF/zsRE raw/query/20-commit auditor runs first.
    result = _audit(member)
    final = read(verify(member))
    for row in final['rows']:
        c = read(verify(row['config'])); rescue.validate_profile(c)
        out = Path(c['output'])
        require(not (out/'QUALITY_EARLY_STOP.json').exists()
            and not (out/'FAILURE.json').exists(), 'FAILED_FINAL_FORBIDDEN')
        done = read(out/'COMPLETE.json'); runtime = read(out/'runtime.json')
        require(runtime['identity'] == done['identity'], 'FINAL_RUNTIME_IDENTITY')
        native_identity(done['identity'], runtime['resolved'])
        # Metadata only through mmap: no H traversal, clone or restore.
        cp = row['checkpoint']
        p = torch.load(verify(cp), map_location='cpu', mmap=True, weights_only=False)
        require(p['identity'] == done['identity'] and p['batch'] == 20
            and p['cursor']['edits'] == 2000, 'BOTH_FINAL_CP_W20')
        native_identity(p['identity'], p['cursor']['native_config'])
        del p
        verify(cp)
    return result


def validate_payload(p, original):
    # Explicit new validator, never relabel clamp .5 as clamp1 to bypass e87.
    require(set(p) == {'weights','cache_c','contexts','z_member','batch','cursor','identity','rng','integrity_sha256'}, 'CURRENT_PAYLOAD_SCHEMA')
    require(p['batch'] == 20 and p['identity'] == original['identity'], 'CURRENT_PAYLOAD_IDENTITY')
    i = p['identity']; cursor = p['cursor']; native = cursor['native_config']
    native_identity(i, native)
    require(i['stage'] == 'final_benchmark' and i['dataset'] == 'cf'
        and i['split'] == 'original_cf_first2000' and i['z_reference'] == 'batch_entry'
        and i['dtype'] == 'bfloat16', 'FINAL_CF_NOT_HELDOUT')
    require(cursor['edits'] == 2000 and digest(cursor['case_ids']) == base.ORDER
        and cursor['raw'] == read(verify(original['commit']))['raw'], 'CURRENT_W20_CURSOR')
    require(native['bs'] == 100 and native['num_edits'] == 2000 and native['seed'] == 0
        and native['model_dtype'] == 'bfloat16' and native['algs']['add_old_keys'] is True
        and {k:native['llms'][k] for k in base.HP} == base.HP, 'CURRENT_NATIVE_CONFIG')
    required = dict(clamp_norm_factor=.5,v_num_grad_steps=35,v_lr=.5,v_weight_decay=.001,
        kl_factor=.0625,v_loss_layer=27,mom2_update_weight=[15000]*5)
    require(all(native['llms'][k] == v for k,v in required.items()), 'CURRENT_AUTHOR_HPARAMS')
    require(p['z_member'] == dict(policy='batch_entry_current100',durable_target_cache=False), 'CURRENT_Z_POLICY')
    integrity = p['integrity_sha256']
    require(type(integrity) is str and len(integrity) == 64
        and all(x in '0123456789abcdef' for x in integrity), 'PAYLOAD_INTEGRITY_FIELD')
    if 'payload_integrity_sha256' in original:
        require(integrity == original['payload_integrity_sha256'], 'PAYLOAD_INTEGRITY_BINDING')
    names = {base.HP['rewrite_module_tmp'].format(l)+'.weight' for l in base.HP['layers']}
    require(set(p['weights']) == names, 'CURRENT_SELECTED_NAMES')
    for v in p['weights'].values():
        require(v.dtype == torch.bfloat16 and v.device.type == 'cpu' and v.is_contiguous()
            and tuple(v.shape) == base.SELECTED_SHAPE and base.finite_tensor(v), 'CURRENT_SELECTED_BF16')
    return p['weights'], integrity


def tracking_values(c, source):
    values = _tracking(c, source)
    values.update(baseline='FE-tuned-L2-100-clamp-0p5-selected-final-CF')
    return base.validate_tracking(values)


@contextmanager
def bindings():
    # Process-local opt-in; neither editing source nor original gate is changed.
    with ExitStack() as stack:
        for name,value in dict(SPEC=SPEC,TASK=TASK,validated_selection=rescue.validated_selection,
            audit_final=audit_final,validate_payload=validate_payload,tracking_values=tracking_values).items():
            stack.enter_context(patch.object(base,name,value))
        yield


def prepare(output, inputs):
    with bindings(): return flucon_prepare.prepare(output, inputs)


def run(config, lock):
    with bindings(): return base.run(config, lock)


def collect(config, lock, output):
    with bindings(): return base.collect(config, lock, output)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','collect'])
    p.add_argument('first');p.add_argument('second');p.add_argument('--output');a=p.parse_args()
    if a.mode=='prepare':print(prepare(a.first,a.second))
    elif a.mode=='run':run(a.first,a.second)
    else:collect(a.first,a.second,a.output)
