"""Server4 native CF/zsRE chain, latest-only checkpoint and explicit resume.

The shared factual API and review READY are required, not simulated. This
runner cannot submit or silently fall back to historical task evaluators.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

from official.experiments.prepare import digest, file_sha, read, write_new, ROOT, load_plan, build_matrix


def validate(config, assets, ready, qualification=False):
    if config['model'] != 'llama3' or config['method'] not in (
            'ALPHAEDIT', 'ALPHAEDIT_BLUE', 'SPHERE'):
        raise ValueError('SERVER4_SCOPE')
    if config['dataset'] not in ('cf', 'zsre') or config['edit_seed'] != 0:
        raise ValueError('DATASET_SEED')
    expected=next((row for row in build_matrix(*load_plan()) if row['run_id']==config['run_id']),None)
    if expected is None or {k:v for k,v in config.items() if k!='_path'}!=expected:
        raise ValueError('CANONICAL_CONFIG_PROFILE_CHANGED')
    repo = ROOT.parent
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    tree = subprocess.check_output(['git','rev-parse','HEAD:official'], cwd=repo, text=True).strip()
    if head != ready['code_commit'] or tree != ready['official_tree']:
        raise ValueError('REVIEWED_SOURCE_IDENTITY')
    if ready['config_sha256'] != file_sha(config['_path']) or ready['assets_sha256'] != digest(assets):
        raise ValueError('REVIEWED_CONFIG_ASSETS_IDENTITY')
    if (not qualification and not ready.get('main_integrated')) or not ready.get('source_review_pass'):
        raise ValueError('MAIN_INTEGRATION_REVIEW_REQUIRED')
    if not ready.get('tracking_cpu_adapter_pass'):
        raise ValueError('OFFICIAL_TRACKING_ADAPTER_NOT_BOUND')
    if not qualification and not assets.get('w0', {}).get(config['dataset']):
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
    from .observe import verify_api
    api=verify_api()
    if ready.get('factual_api')!=api:
        raise ValueError('SH1_FACTUAL_SOURCE_API_RECEIPT_REQUIRED')
    return api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','assets','ready','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--qualification', action='store_true')
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--stop-after', type=int, choices=range(1,21), default=20)
    args = parser.parse_args()
    from official.tracking.schema import identifier
    identifier(args.attempt)
    config, assets, ready = read(args.config), read(args.assets), read(args.ready)
    config['_path'] = str(args.config)
    validate(config, assets, ready, args.qualification)
    if args.qualification and (args.stop_after not in (2,3) or config['dataset']!='cf'):
        raise ValueError('QUALIFICATION_BUDGET')
    if args.stop_after == 20 and not ready.get('native_resume_and_parity_pass'):
        raise ValueError('ACTUAL_NATIVE_RESUME_PARITY_REQUIRED')
    if config['dataset'] == 'zsre' and args.stop_after > 1 and not ready.get('zsre_smoke_pass'):
        raise ValueError('ACTUAL_ZSRE_SMOKE_REQUIRED')
    if shutil.disk_usage(args.output.parent).free < ready['disk_reserve_bytes']:
        raise ValueError('RESOURCE_BLOCKED_STORAGE')
    records = read(assets['streams'][config['dataset']]['path'])
    if len(records) != 2000 or file_sha(assets['streams'][config['dataset']]['path']) != assets['streams'][config['dataset']]['sha256']:
        raise ValueError('STREAM_IDENTITY')
    from . import tracking
    # Cheap mandatory online identity check before loading any pretrained model.
    # Main owns finish even if model setup, restore, editing or evaluation fails.
    with tracking.start(config,assets,ready,args.output,args.attempt) as tracker:
        run_chain(args,config,assets,ready,records,tracker)


def run_chain(args,config,assets,ready,records,tracker):
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
    from .observe import FactualObserver, subset, scores, numeric_receipt
    from . import tracking
    observer = FactualObserver(model,tokenizer,config['dataset'],assets,
        args.output/'factual'/args.attempt,qualification=args.qualification)
    generation = None
    if args.resume:
        start, cursor = state.restore(cp, identity)
        write_new(args.output/('resume-'+args.attempt+'.json'),dict(
            start_batch=start,checkpoint=read(cp/'latest.json'),state=state.signature(),rng_sha256=rng_hash()))
    else:
        state.contexts()
        start, cursor = 0, dict(completed_batch=0,
            evaluated_endpoints=['W0'] if observer.w0 is not None else [],
            W0_observation='REUSED' if observer.w0 is not None else 'NOT_MEASURED_QUALIFICATION')
        observer.verify_w0(records)
        state.save(cp, 0, cursor, identity)
    # Tracking is already initialized outside the seeded/restored edit RNG.
    # Main alone closes this run; queued scalars are not remote certification.
    with nullcontext(tracker):
        if start==0 and observer.w0 is not None:
            tracker.log(scores(observer.w0,'W0_first2000',0))
        for batch in range(start+1, args.stop_after+1):
            if shutil.disk_usage(args.output).free < ready['disk_reserve_bytes']:
                raise ValueError('RESOURCE_BLOCKED_STORAGE')
            current=records[(batch-1)*100:batch*100]
            pre=observer.observe(current,endpoint=f'B{batch}-pre')
            tracker.log(scores(pre,'current/pre',batch))
            state.edit(current)
            signature = state.signature()
            rng = rng_snapshot()
            if batch in (5,10,15,20):
                all_seen=observer.observe(records[:batch*100],endpoint=f'W{batch}')
                tracker.log(scores(all_seen,'all_seen/post',batch))
                result=subset(all_seen,current,config['dataset'])
            else:
                result=observer.observe(current,endpoint=f'B{batch}-post')
            tracker.log(scores(result,'current/post',batch))
            write_new(args.output/f'W{batch}-factual.json',numeric_receipt(result))
            if batch == 20 and config['dataset']=='cf':
                # Keep large reference assets out of the host edit working set.
                os.environ['NLTK_DATA'] = assets['nltk_data']
                generation = NativeGenerationObserver(model, tokenizer, load_assets(assets['generation']),
                    dict(model_identity=dict(revision=assets['model_revision'], tokenizer=assets['tokenizer_sha256']),
                         generation_source_sha=ready['code_commit']), args.output/'generation',
                    state_callback=state.signature)
                generation.progress_callback=tracking.progress(tracker,'W20')
                gen=generation.observe(records,'W20',state_identity=digest(signature))
                tracker.log(tracking.generation(gen,'W20'))
            if signature != state.signature() or not rng_equal(rng):
                raise ValueError('OBSERVER_STATE_OR_RNG_MUTATION')
            cursor = dict(completed_batch=batch,case_ids=[r['case_id'] for r in records[:batch*100]])
            ref = state.save(cp,batch,cursor,identity)
            write_new(args.output/f'batch-{batch:02d}-commit.json',dict(
                checkpoint=ref,state=signature,rng_sha256=rng_hash(),
                job_id=os.environ.get('SLURM_JOB_ID')))
    write_new(args.output/f'terminal-B{args.stop_after}.json', dict(
        status='W20_COMPLETE' if args.stop_after == 20 else 'VALIDATION_PREFIX_COMPLETE',
        batch=args.stop_after, identity=identity))


if __name__ == '__main__':
    main()
