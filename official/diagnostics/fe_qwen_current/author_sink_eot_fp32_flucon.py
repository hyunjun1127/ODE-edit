"""FP32 author-hparams sink-EOT FE (author_sink_eot_fp32) final CF W20 generation, source-only adapter.

prepare NEW_ROOT BINDINGS | run CONFIG LOCK | collect CONFIG LOCK --output FILE
The shared generator, reference, prompt order, sampling profile and Flu/Con scoring (flucon.py) are unchanged; only
four hooks are substituted for one fresh evaluation process:

* audit_final: the final_result is a small binding {collect_report, collector_python} for one run.  The run's own
  sink-variant collector is run again in a separate CPU process from the run's frozen source package (derived from
  the run bindings) under the run's bound library versions (read-only on the run, temporary report only) and must
  equal the stored report; the model assets are the model-assets-resolved.json next to the run bindings and must match the
  bound snapshot/config/tokenizer.
* validate_payload: the W20 checkpoint is the cold FP32 author run (stage FE_AUTHOR_HPARAMS_CURRENT_Z_SINK_EOT_FP32,
  L2=0, clamp 1, batch_entry z, author numeric settings audited by the FP32 audit) with FP32 selected weights.
* restore: the FP32 selected weights are copied exactly (no cast; evaluation stays FP32 eager, TF32 off).
* tracking: same W&B group as the run (its task_id) with the run as parent_run_id.
No editing, selection, submission, downloads, or editing-runtime mutation.
"""
import argparse
import json
import os
import subprocess
from contextlib import contextmanager, ExitStack
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import author_basic_eot_benchmark as author
from official.diagnostics.fe_qwen_current import author_basic_eot_fp32 as fp32
from official.diagnostics.fe_qwen_current import author_basic_eot_flucon as author_flucon
from official.diagnostics.fe_qwen_current import author_sink_eot_fp32 as sink
from official.diagnostics.fe_qwen_current import flucon as base, flucon_prepare
from official.diagnostics.fe_qwen_current import failure_eot_probe as eot
from official.experiments.prepare import digest
from official.runners.fe_cake_flucon import bind_weights
from official.runners.server1.common import read, verify, member, require

TASK = 'fe-qwen-author-sink-eot-fp32-final-cf-flucon-20261013'
_tracking = base.tracking_values


FINAL_SCHEMA = 'author-sink-eot-fp32-flucon-final-v1'
_REAUDIT = '''import json,sys
import numpy,tokenizers,torch,transformers
from official.diagnostics.fe_qwen_current import author_sink_eot_fp32 as m
runtime,result,out=json.loads(sys.argv[1]),sys.argv[2],sys.argv[3]
actual=dict(torch=torch.__version__,transformers=transformers.__version__,numpy=numpy.__version__,tokenizers=tokenizers.__version__)
assert all(runtime[k]==v for k,v in actual.items()),('BOUND_COLLECTOR_RUNTIME',actual)
assert m.__file__.startswith(sys.argv[4]),'RUN_FROZEN_SOURCE'
print(m.collect(result,out))
'''


def reaudit(report, b, python):
    """The run's own collector from its frozen source, separate CPU process; returns the fresh report."""
    source = Path(b['source']['path']).resolve().parents[3]
    require(source / 'official/diagnostics/fe_qwen_current/author_sink_eot_fp32.py' in
            {Path(m['path']).resolve() for m in b['official_members']}, 'RUN_FROZEN_SINK_COLLECTOR')
    env = dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               FE_NATIVE_AUTHOR_ROOT=b['native']['path'], FE_AUTHOR_ROOT=b['native']['path'], PYTHONDONTWRITEBYTECODE='1')
    with tempfile.TemporaryDirectory(prefix='fe-sink-fp32-flucon-audit-') as folder:
        out = Path(folder) / 'reaudit.json'
        subprocess.run([python, '-B', '-c', _REAUDIT, json.dumps(b['runtime']), report['result']['path'], str(out),
                        str(source)], env=env, cwd=folder, check=True)
        return read(out)


def audit_final(final_member):
    """Re-run the run's sink-variant collector (temporary report only) and require the stored report."""
    final = read(verify(final_member))
    require(final.get('schema') == FINAL_SCHEMA and set(final) == {'schema', 'collect_report', 'collector_python'},
            'SINK_FP32_FINAL_BINDING')
    report = read(verify(final['collect_report']))
    require(report.get('status') == 'AUTHOR_BASIC_EOT_FP32_W20_CPU_AUDITED' and report.get('dataset') == 'cf'
            and report.get('profile') == sink.PROFILE and report.get('commits') == 20, 'SINK_FP32_CF_W20_REPORT')
    result = verify(report['result']);proof = read(result)
    b = read(verify(proof['bindings']))
    require(reaudit(report, b, final['collector_python']) == report, 'SINK_FP32_INDEPENDENT_REAUDIT')
    require(proof['status'] == fp32.STATUS and proof['dataset'] == b['dataset'] == 'cf' and b['schema'] == sink.SCHEMA
            and b['eot_policy'] in sink.POLICIES and b['main_stream']['sha256'] == base.STREAM, 'SINK_FP32_CF_SCOPE')
    for name in ('QUALITY_EARLY_STOP.json', 'FAILURE.json'):
        require(not (Path(result).parent / name).exists(), 'FAILED_FINAL_FORBIDDEN')
    assets = member(Path(proof['bindings']['path']).parent / 'model-assets-resolved.json');a = read(verify(assets))
    model = {Path(m['path']).resolve(): m['sha256'] for m in b['model']['members']}
    listed = {Path(m['path']).resolve(): m['sha256'] for m in a['model']['members']}
    config = Path(b['model']['snapshot'], 'config.json').resolve()
    require(a['model']['snapshot'] == b['model']['snapshot'] and a['model']['identity']['revision'] in b['model']['snapshot']
            and listed.get(config) == model.get(config) == b['model']['config']['sha256']
            and all(model.get(path) == sha for path, sha in listed.items() if path in model)
            and Path(b['model']['snapshot'], 'tokenizer.json').resolve() in listed.keys() & model.keys(), 'RUN_MODEL_ASSETS')
    return dict(row=dict(checkpoint=proof['checkpoint'], config=proof['bindings']),
                config=dict(stream=b['main_stream'], model_assets=assets), identity=proof['identity'],
                terminal=member(result), pointer=proof['pointer'], commit=proof['commits'][-1])


def validate_payload(p, original):
    require(set(p) == {'weights', 'cache_c', 'contexts', 'z_member', 'batch', 'cursor', 'identity', 'rng', 'integrity_sha256'},
            'CURRENT_PAYLOAD_SCHEMA')
    require(p['batch'] == 20 and p['identity'] == original['identity'], 'SINK_FP32_CF_W20_IDENTITY')
    i = p['identity']
    require(i['stage'] == sink.PROFILE and i['dataset'] == 'cf' and i['model_dtype'] == 'float32' and i['L2'] == 0
            and i['clamp_norm_factor'] == 1. and i['z_reference'] == 'batch_entry' and i['eot_policy'] in sink.POLICIES
            and i['input_policy'] == sink.POLICIES[i['eot_policy']] and i['stream']['sha256'] == base.STREAM
            and i['author_commit'] == author.AUTHOR, 'SINK_FP32_IDENTITY')
    cur = p['cursor'];native = cur['native_config']
    require(cur['edits'] == 2000 and digest(cur['case_ids']) == base.ORDER
            and cur['raw'] == read(verify(original['commit']))['raw'], 'SINK_FP32_W20_CURSOR')
    b = read(verify(original['config']))
    resolved = fp32.audit_settings(OmegaConf.create(native), author.author_profiles(b))
    require(resolved == native and i['author_settings_sha256'] == digest(native)
            and {k: native['llms'][k] for k in base.HP} == base.HP, 'SINK_FP32_AUTHOR_SETTINGS')
    require(cur['input_adapter']['sha256'] == i['adapter']['sha256'] == fp32.ADAPTER_SHA == member(eot.__file__)['sha256'],
            'FROZEN_C36_ADAPTER')
    require(p['z_member'] == dict(policy='BASIC_EOT_CURRENT_ENTRY', durable_target_cache=False), 'CURRENT_Z_POLICY')
    integrity = p['integrity_sha256']
    require(type(integrity) is str and len(integrity) == 64 and all(x in '0123456789abcdef' for x in integrity), 'PAYLOAD_INTEGRITY_FIELD')
    if 'payload_integrity_sha256' in original:require(integrity == original['payload_integrity_sha256'], 'PAYLOAD_INTEGRITY_BINDING')
    names = {base.HP['rewrite_module_tmp'].format(l) + '.weight' for l in base.HP['layers']}
    require(set(p['weights']) == names, 'SELECTED_NAMES')
    for w in p['weights'].values():
        require(w.dtype == torch.float32 and w.device.type == 'cpu' and w.is_contiguous()
                and tuple(w.shape) == base.SELECTED_SHAPE and base.finite_tensor(w), 'SELECTED_FP32')
    # No H/context/RNG traversal, cloning, restoration or consumption here.
    return p['weights'], integrity


def restore(model, original):
    weights, integrity = base.payload_weights(original)
    restored = bind_weights(model, weights, original, torch.float32)
    del weights
    return dict(checkpoint=original['checkpoint'], identity=original['identity'],
        payload_integrity_sha256=integrity, restored_selected_weights_fp32_sha256=restored,
        cast='NONE_FP32_CHECKPOINT_EXACT', evaluation='FP32_EAGER_TF32_OFF',
        nonselected_unchanged=True, history_contexts_rng_consumed=False, edit_calls=0)


def tracking_values(c, source):
    values = _tracking(c, source);o = c['original']
    run = read(Path(o['terminal']['path']).parent / 'tracking' / 'receipt.json')
    require(str(run['job_identity']['job_id']) == str(o['identity']['job_id']), 'PARENT_RUN_JOB')
    values.update(task_id='fe-qwen-author-sink-eot-fp32-20261013',
                  arm=f'AUTHOR-cf-FP32-L2-0-clamp1-{o["identity"]["eot_policy"]}EOT-W20-flucon',
                  baseline='FE-author-hparams-current-z-sinkEOT-FP32', writer='fe_native_batch_entry_z',
                  parent_run_id=run['run_id'])
    return base.validate_tracking(values)


@contextmanager
def bindings():
    # Fresh evaluation process ONLY; scoped substitutions restore on exit.
    with ExitStack() as stack:
        for name, value in dict(TASK=TASK, audit_final=audit_final, validate_payload=validate_payload, restore=restore,
                                load_reference=author_flucon.load_reference, tracking_values=tracking_values).items():
            stack.enter_context(patch.object(base, name, value))
        stack.enter_context(patch.object(base, 'EDITING_PRECISION', 'FP32_UNCHANGED', create=True))
        yield


def prepare(output, inputs):
    with bindings():return flucon_prepare.prepare(output, inputs)


def run(config, lock):
    with bindings():return base.run(config, lock)


def collect(config, lock, output):
    with bindings():return base.collect(config, lock, output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('mode', choices=['prepare', 'run', 'collect'])
    p.add_argument('first');p.add_argument('second');p.add_argument('--output');a = p.parse_args()
    if a.mode == 'prepare':print(prepare(a.first, a.second))
    elif a.mode == 'run':run(a.first, a.second)
    else:collect(a.first, a.second, a.output)
