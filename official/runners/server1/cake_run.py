"""CAKE native sequential 2K; frozen official source and retained 500-edit states."""
import argparse
import copy
import importlib.metadata
import os
from pathlib import Path
import random
import time
import traceback
import uuid

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from official.baselines import registry
from official.evaluation.factual import evaluate_counterfact
from official.evaluation import zsre_paper
from official.experiments import checkpoint, milestones
from official.experiments.prepare import digest, write_new
from official.tracking import init, official_zsre_metrics
from .common import read, verify, member, factual_payload
from .assets import member as asset_member

TASK = 'cake-six-2k-server1-20261011'
INSTRUCTION = 'USER-CAKE-THREE-MODELS-TWO-DATASETS-TWO-LANES-500-CHECKPOINT-20261011'
MODELS = ('llama3', 'gptj', 'qwen25')
CONTEXT_SHA = dict(llama3='cf14b8571136b0413ded1ea8aa283a004ecbddbd818a6a8744b0296b63f72191',
    gptj='b454d83e8ad3656a2e4beaed7cbc73b74a6bc50b7d5cc8fa7db0c99a01b9c842',
    qwen25='5c01bc1a91c0890af2897f18b7a5f95de011badc1eca5e27f44199acc0cb6515')
HPARAMS = Path(__file__).resolve().parents[2] / 'hparams' / 'CAKE'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(c):
    require(c['instruction'] == INSTRUCTION and c['server'] in ('server1', 'server4'), 'CAKE_SCOPE')
    if c['server'] == 'server4':
        require(c['model'] == 'qwen25' and c['dataset'] in ('cf', 'zsre'), 'CAKE_SERVER4_SCOPE')
    if c.get('checkpoint_archive'):
        require((c['server'], c['model'], c['dataset']) == ('server4', 'qwen25', 'zsre'), 'CAKE_ARCHIVE_SCOPE')
    require(c['model'] in MODELS and c['dataset'] in ('cf', 'zsre'), 'CAKE_MODEL_DATASET')
    require((c['requests'], c['batch_size'], c['seed'], c['dtype']) == (2000, 100, 0, 'float32'), 'CAKE_STREAM_PRECISION')
    require(c['milestones'] == [5, 10, 15, 20] and c['history'] is True, 'CAKE_CHECKPOINT_HISTORY_POLICY')
    require(c['generation'] == ('DEFERRED_TO_W20_CHECKPOINT' if c['dataset'] == 'cf' else 'NOT_APPLICABLE'), 'CAKE_GENERATION_POLICY')
    require(c['hparams'] == read(HPARAMS / (c['model'] + '.json')), 'CAKE_HPARAMS_CHANGED')
    require(c['context_sha256'] == CONTEXT_SHA[c['model']], 'COMMON_CONTEXT_REQUIRED')
    require(c['config_sha256'] == digest({k: v for k, v in c.items() if k != 'config_sha256'}), 'CONFIG_DIGEST')
    return c


def verify_asset(row):
    """Use a prior full hash only while the exact local inode/stat still agrees."""
    path = Path(row['path']); stat = path.stat()
    fields = dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                  mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)
    require(row.get('sha256') and str(path.resolve()) == row['real_path'], 'MODEL_ASSET_IDENTITY')
    if all(row.get(k) == v for k, v in fields.items()):
        return 'PRIOR_FULL_SHA_AND_UNCHANGED_STAT'
    asset_member(path, allow_symlink=True, expected_sha=row['sha256'], expected_bytes=row['bytes'])
    return 'FRESH_FULL_SHA'


def versions(model, excluded=()):
    return {k: (p._version, str(p.dtype), tuple(p.shape)) for k, p in model.named_parameters() if k not in excluded}


def validate_projector(P, layers, width):
    require(isinstance(P, torch.Tensor) and tuple(P.shape) == (len(layers), width, width)
            and P.dtype == torch.float32 and P.device.type == 'cpu', 'PROJECTOR_SHAPE_DTYPE')
    # Layer-sized checks avoid a full multi-GiB boolean temporary.
    for slot in P:
        require(bool(torch.isfinite(slot).all()), 'PROJECTOR_NONFINITE')


def payload(result, dataset, prefix, edits, values):
    if dataset == 'zsre':
        return official_zsre_metrics(result['summary'], config_values=values,
            endpoint=prefix, edits=edits, pre_state_edits=edits, post_state_edits=edits)
    require(prefix in ('W0_first2000', 'current/post', 'all_seen/post'), 'CAKE_EVALUATION_ENDPOINT')
    if prefix == 'current/post':
        require(len(result['cases']) == 100, 'CURRENT100_COHORT')
        return {k.replace('all_seen/post/', 'current/post/'): v for k, v in
                factual_payload(result, 'all_seen/post', edits).items()}
    return factual_payload(result, prefix, edits)


def run(config_path, lock_path, resume=False):
    c = validate(read(config_path)); lock = read(lock_path)
    source = Path(lock['source_directory']).resolve()
    require(Path(__file__).resolve().is_relative_to(source), 'FROZEN_OFFICIAL_RUNTIME_REQUIRED')
    require(any(member(config_path) == m for m in lock['configs']), 'UNBOUND_CONFIG')
    require(os.environ.get('OFFICIAL_CODE_COMMIT') == lock['source_commit']
            and os.environ.get('OFFICIAL_TREE_SHA256') == lock['official_tree'], 'FROZEN_SOURCE_ENV')
    for m in lock['source_members']:
        verify(m)
    assets = read(verify(c['assets']))
    require(digest({k: v for k, v in assets.items() if k != 'assets_sha256'}) == assets['assets_sha256'], 'ASSET_MANIFEST_DIGEST')
    checks = [verify_asset(m) for m in assets['model']['members']]
    for name, version in assets['runtime']['packages'].items():
        require(importlib.metadata.version(name) == version, 'RUNTIME_PACKAGE_CHANGED_' + name)
    records = read(verify(c['stream']))
    records = records['records'] if isinstance(records, dict) else records
    require(len(records) == 2000 and digest(records) == c['stream_sha256'], 'STREAM_IDENTITY')
    contexts = read(verify(c['context']))
    require(digest(contexts) == c['context_sha256'] and contexts[0] == ['{}']
            and len(contexts) == 2 and len(contexts[1]) == 5, 'CONTEXT_PACKAGE')
    verify(c['projector']); layers = c['hparams']['layers']
    P = torch.load(c['projector']['path'], map_location='cpu', weights_only=True)
    width = assets['model']['identity']['intermediate']
    validate_projector(P, layers, width)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'ONE_ALLOCATED_GPU_REQUIRED')
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(assets['model']['snapshot'], local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager').to('cuda:0').eval()
    tok = AutoTokenizer.from_pretrained(assets['model']['snapshot'], local_files_only=True)
    tok.pad_token = tok.eos_token; tok.padding_side = 'right'
    require(all(p.dtype == torch.float32 for p in model.parameters()), 'MODEL_FP32_REQUIRED')
    for p in model.parameters():
        p.requires_grad_(False)
    native, _, apply = registry.implementation('CAKE', c['model'])
    hp = registry.hparams('CAKE', c['model'])
    weights = {hp.rewrite_module_tmp.format(l) + '.weight': native.nethook.get_parameter(model,
               hp.rewrite_module_tmp.format(l) + '.weight') for l in layers}
    expected_shape = (assets['model']['identity']['hidden'], width)
    require(all(tuple(w.shape) == expected_shape for w in weights.values()), 'ACTUAL_LINEAR_REWRITE_SHAPE')
    H = torch.zeros((len(layers), width, width), dtype=torch.float32, device='cpu')
    native.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(contexts); native.COV_CACHE.clear()
    out = Path(c['output']); out.mkdir(parents=True, exist_ok=True)
    if resume:
        observations = out / 'resumes' / uuid.uuid4().hex
        observations.mkdir(parents=True)
    else:
        require(not (out / 'runtime.json').exists() and not (out / 'checkpoint/latest.json').exists(), 'COLD_OUTPUT_ALREADY_USED')
        observations = out
    identity = dict(config_sha256=c['config_sha256'], stream_sha256=c['stream_sha256'],
        code_commit=lock['source_commit'], official_tree_sha256=lock['official_tree'],
        model_revision=assets['model']['identity']['revision'], tokenizer_sha256=assets['tokenizer_sha256'],
        assets_sha256=assets['assets_sha256'])
    values = dict(server=c['server'], task_id=(TASK if c['server'] == 'server1'
        else f"cake-qwen25-{c['dataset']}-server4-20261011"), model=c['model'],
        model_family={'llama3': 'llama', 'gptj': 'gptj', 'qwen25': 'qwen2'}[c['model']],
        writer='cake_native', baseline='CAKE', role='scientific', arm=c['arm'],
        attempt=c['attempt'] + ('-resume-' + observations.name if resume else ''), source_sha=lock['source_commit'],
        config_sha=c['config_sha256'], dataset=c['dataset'], metric_schema='official-baselines-scalar-v1',
        instruction_id='USER-OFFICIAL-BASELINES-20261008-R1')
    if c['dataset'] == 'cf':
        values['generation_schedule'] = 'DEFERRED_CHECKPOINT_EVALUATION'
    write_new(observations / 'runtime.json', dict(identity=identity, config=c, resume=resume,
        torch=torch.__version__, transformers=__import__('transformers').__version__,
        attention_backend=model.config._attn_implementation, precision='float32', TF32=False,
        model_asset_checks=checks, GPU=torch.cuda.get_device_name(), job_id=os.environ.get('SLURM_JOB_ID'),
        causal_layer_weights=native.compute_layer_weights(hp.causal_scores, hp.temperature)))
    tracker = init(env_file=c['tracking_env_file'], spool=observations / 'tracking', config=values)
    def evaluate(rows, prefix, batch, label):
        before = versions(model); rng = checkpoint.rng_snapshot()
        try:
            result = (evaluate_counterfact(model, tok, rows, identity=identity, batch_size=16, device='cuda:0')
                if c['dataset'] == 'cf' else zsre_paper.evaluate(model, tok, rows,
                model_family=c['model'], identity=identity, batch_size=16, device='cuda:0'))
        finally:
            checkpoint.rng_restore(rng)
        require(before == versions(model), 'EVALUATION_MUTATED_PARAMETERS')
        path = observations / 'factual' / (label + '.json'); write_new(path, result)
        require(tracker.log(payload(result, c['dataset'], prefix, batch * 100, values)) is not False, 'TRACKING_REJECTED')
        return member(path)
    def save(batch, cursor):
        saver = milestones.save
        extra = {}
        if c.get('checkpoint_archive'):
            from official.experiments import remote_milestones
            saver = remote_milestones.save
            extra['archive'] = c['checkpoint_archive']
        return saver(out, lock_path=c['checkpoint_lock'], reserve_bytes=c['storage_min_free_bytes'], **extra,
            batch=batch, weights=weights, cache_c={str(l): H[i] for i, l in enumerate(layers)},
            contexts=contexts, evaluation_cursor=cursor, identity=identity, method='CAKE', evaluation_complete=True)
    exit_code = 1
    try:
        if resume:
            if c.get('checkpoint_archive'):
                from official.experiments import remote_milestones
                saved = remote_milestones.load(out, identity, c['checkpoint_archive'])
            else:
                saved = checkpoint.load(out / 'checkpoint', identity)
            start = saved['batch']
            require(start in (0, 5, 10, 15), 'RESUME_ALREADY_COMPLETE_OR_INVALID_BATCH')
            require(saved['method'] == 'CAKE' and saved['contexts'] == contexts
                and set(saved['weights']) == set(weights) and set(saved['cache_c']) == {str(l) for l in layers}, 'RESUME_STATE_SCHEMA')
            with torch.no_grad():
                for name, weight in weights.items():
                    require(saved['weights'][name].dtype == weight.dtype and saved['weights'][name].shape == weight.shape, 'RESUME_WEIGHT_SCHEMA')
                    weight.copy_(saved['weights'][name])
                for i, layer in enumerate(layers):
                    require(saved['cache_c'][str(layer)].shape == H[i].shape, 'RESUME_HISTORY_SCHEMA')
                    H[i].copy_(saved['cache_c'][str(layer)])
            checkpoint.rng_restore(saved['rng']); del saved
            if not c.get('checkpoint_archive'):
                milestones.retain(out / 'checkpoint', out / 'milestones')
            write_new(observations / 'resume-parent.json', dict(batch=start, pointer=member(out / 'checkpoint/latest.json')))
        else:
            start = 0
            w0 = evaluate(records, 'W0_first2000', 0, 'W0-first2000')
            save(0, dict(batch=0, edits=0, W0=w0))
        nonedited = versions(model, weights)
        requests = registry.requests(records, 'CAKE', c['model'])
        for batch in range(start + 1, 21):
            begin = time.monotonic()
            with torch.enable_grad():
                returned, history = apply(model, tok, requests[(batch-1)*100:batch*100], hp,
                    cache_template=None, cache_c=H, P=P)
            require(returned is model and history is H, 'NATIVE_INPLACE_STATE')
            require(nonedited == versions(model, weights) and native.CONTEXT_TEMPLATES_CACHE == contexts, 'NATIVE_SCOPE_OR_CONTEXT_CHANGED')
            require(all(bool(torch.isfinite(w).all()) for w in weights.values())
                    and all(bool(torch.isfinite(h).all()) for h in H), 'NONFINITE_EDIT_STATE')
            cursor = dict(batch=batch, edits=batch*100, request_sha256=digest(requests[(batch-1)*100:batch*100]),
                current=evaluate(records[(batch-1)*100:batch*100], 'current/post', batch, f'current-{batch:02d}'))
            if batch in c['milestones']:
                cursor['all_seen'] = evaluate(records[:batch*100], 'all_seen/post', batch, f'all-seen-{batch:02d}')
                cursor['checkpoint'] = save(batch, cursor.copy())
            cursor['elapsed_seconds'] = time.monotonic() - begin
            write_new(observations / 'commits' / f'batch-{batch:02d}.json', dict(identity=identity, cursor=cursor))
        retained = [member(out / 'milestones' / f'edit-{b*100:04d}' / 'latest.json') for b in c['milestones']]
        write_new(observations / 'COMPLETE.json', dict(identity=identity, requests=2000, batches=20,
            checkpoint=member(out / 'checkpoint/latest.json'), milestones=retained,
            generation=c['generation'], final_cursor=cursor))
        exit_code = 0
    except BaseException as error:
        write_new(observations / 'FAILURE.json', dict(type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), checkpoints_retained=True))
        raise
    finally:
        tracker.finish(exit_code=exit_code, timeout=45)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--lock', required=True); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); run(args.config, args.lock, args.resume)
