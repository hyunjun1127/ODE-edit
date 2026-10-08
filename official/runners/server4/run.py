"""Server4 native CF/zsRE chain, latest-only checkpoint and explicit resume.

The shared factual API and review READY are required, not simulated. This
runner cannot submit or silently fall back to historical task evaluators.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

from official.experiments.prepare import digest, file_sha, read, write_new, ROOT


def validate(config, assets, ready):
    if config['model'] != 'llama3' or config['method'] not in (
            'ALPHAEDIT', 'ALPHAEDIT_BLUE', 'SPHERE'):
        raise ValueError('SERVER4_SCOPE')
    if config['dataset'] not in ('cf', 'zsre') or config['edit_seed'] != 0:
        raise ValueError('DATASET_SEED')
    repo = ROOT.parent
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    tree = subprocess.check_output(['git','rev-parse','HEAD:official'], cwd=repo, text=True).strip()
    if head != ready['code_commit'] or tree != ready['official_tree']:
        raise ValueError('REVIEWED_SOURCE_IDENTITY')
    if ready['config_sha256'] != file_sha(config['_path']) or ready['assets_sha256'] != digest(assets):
        raise ValueError('REVIEWED_CONFIG_ASSETS_IDENTITY')
    if not ready.get('main_integrated') or not ready.get('source_review_pass'):
        raise ValueError('MAIN_INTEGRATION_REVIEW_REQUIRED')
    if not ready.get('tracking_online_adapter_pass'):
        raise ValueError('OFFICIAL_TRACKING_ADAPTER_NOT_BOUND')
    if not all(assets.get('w0', {}).get(d) for d in (config['dataset'],)):
        raise ValueError('SHARED_W0_IDENTITY_NOT_BOUND')
    if subprocess.check_output(['git','status','--porcelain','--','official'],cwd=repo,text=True).strip():
        raise ValueError('UNFROZEN_OFFICIAL_SOURCE')
    if config['method'] == 'ALPHAEDIT_BLUE' and not ready.get('blue_tensor_tuple_review_pass'):
        raise ValueError('BLUE_TENSOR_TUPLE_REVIEW_REQUIRED')
    if config['dataset'] == 'cf' and not ready.get('native_generation_llama_review_pass'):
        raise ValueError('NATIVE_GENERATION_LLAMA_COMPATIBILITY_REQUIRED')
    for member in assets['members']:
        path = Path(member['path'])
        stat = path.stat()
        if stat.st_size != member['bytes']:
            raise ValueError('ASSET_SIZE: '+str(path))
        if member.get('mtime_ns') != stat.st_mtime_ns or member.get('inode') != stat.st_ino:
            if file_sha(path) != member['sha256']:
                raise ValueError('ASSET_SHA: '+str(path))
    factual = importlib.import_module('official.evaluation.factual')
    # Proposed SH1 API handshake; execution remains blocked until coordinated.
    if getattr(factual, 'SERVER_RUNNER_API', None) != 'official-factual-observer-v1':
        raise ValueError('SH1_FACTUAL_API_NOT_BOUND')
    return factual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','assets','ready','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, choices=range(1,21), default=20)
    args = parser.parse_args()
    config, assets, ready = read(args.config), read(args.assets), read(args.ready)
    config['_path'] = str(args.config)
    factual = validate(config, assets, ready)
    if args.stop_after == 20 and not ready.get('native_resume_and_parity_pass'):
        raise ValueError('ACTUAL_NATIVE_RESUME_PARITY_REQUIRED')
    if config['dataset'] == 'zsre' and args.stop_after > 1 and not ready.get('zsre_smoke_pass'):
        raise ValueError('ACTUAL_ZSRE_SMOKE_REQUIRED')
    if shutil.disk_usage(args.output.parent).free < ready['disk_reserve_bytes']:
        raise ValueError('RESOURCE_BLOCKED_STORAGE')
    records = read(assets['streams'][config['dataset']]['path'])
    if len(records) != 2000 or file_sha(assets['streams'][config['dataset']]['path']) != assets['streams'][config['dataset']]['sha256']:
        raise ValueError('STREAM_IDENTITY')
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .native import NativeState
    from .native import rng_hash
    from official.experiments import checkpoint
    from official.evaluation.generation.generator import rng_equal, rng_snapshot
    from official.evaluation.generation.native_observer import NativeGenerationObserver
    from official.evaluation.generation.assets import load_assets
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(assets['model_snapshot'], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(assets['model_snapshot'], local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
    model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'
    state = NativeState(model, tokenizer, config['method'], assets)
    identity = dict(config_sha256=file_sha(args.config), stream_sha256=assets['streams'][config['dataset']]['sha256'],
        code_commit=ready['code_commit'], official_tree_sha256=ready['official_tree'],
        model_revision=assets['model_revision'], tokenizer_sha256=assets['tokenizer_sha256'],
        assets_sha256=digest(assets))
    args.output.mkdir(parents=True, exist_ok=True)
    cp = args.output/'checkpoint'
    if not args.resume and (cp/'latest.json').exists():
        raise ValueError('EXISTING_CHAIN_REQUIRES_EXPLICIT_RESUME')
    # W0 data are shared by model/dataset and must be identity-verified by SH1.
    observer = factual.FactualObserver(model, tokenizer, dataset=config['dataset'],
        identity=identity, output=args.output/'factual', w0=assets['w0'][config['dataset']])
    generation = None
    if config['dataset'] == 'cf':
        os.environ['NLTK_DATA'] = assets['nltk_data']
        generation = NativeGenerationObserver(model, tokenizer, load_assets(assets['generation']),
            dict(model_identity=dict(revision=assets['model_revision'], tokenizer=assets['tokenizer_sha256']),
                 generation_source_sha=ready['code_commit']), args.output/'generation',
            state_callback=state.signature)
    if args.resume:
        start, cursor = state.restore(cp, identity)
    else:
        state.contexts()
        start, cursor = 0, dict(completed_batch=0, evaluated_endpoints=['W0'])
        observer.verify_w0(records)
        state.save(cp, 0, cursor, identity)
    for batch in range(start+1, args.stop_after+1):
        if shutil.disk_usage(args.output).free < ready['disk_reserve_bytes']:
            raise ValueError('RESOURCE_BLOCKED_STORAGE')
        state.edit(records[(batch-1)*100:batch*100])
        signature = state.signature()
        rng = rng_snapshot()
        if batch in (5,10,15,20) or batch == args.stop_after:
            result = observer.observe(records[:batch*100], endpoint=f'W{batch}')
            write_new(args.output/f'W{batch}-factual.json', result)
        if batch == 20 and generation is not None:
            generation.observe(records, 'W20', state_identity=digest(signature))
        if signature != state.signature() or not rng_equal(rng):
            raise ValueError('OBSERVER_STATE_OR_RNG_MUTATION')
        cursor = dict(completed_batch=batch, case_ids=[r['case_id'] for r in records[:batch*100]])
        ref = state.save(cp, batch, cursor, identity)
        write_new(args.output/f'batch-{batch:02d}-commit.json', dict(
            checkpoint=ref, state=signature, rng_sha256=rng_hash(),
            job_id=os.environ.get('SLURM_JOB_ID')))
    write_new(args.output/f'terminal-B{args.stop_after}.json', dict(
        status='W20_COMPLETE' if args.stop_after == 20 else 'VALIDATION_PREFIX_COMPLETE',
        batch=args.stop_after, identity=identity))


if __name__ == '__main__':
    main()
