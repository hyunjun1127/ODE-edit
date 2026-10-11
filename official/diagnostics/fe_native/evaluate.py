"""Read-only common CF evaluation of checkpoints produced by native FE main.py.

This program is never called by training and never computes targets or edits.
Native training remains independent of the ODE runner and evaluator.
"""
import argparse
import json
from pathlib import Path
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from official.evaluation.factual import evaluate_counterfact
from official.experiments.prepare import digest, file_sha, write_new
from official.runners.server1.common import read, verify


def validate_payload(payload, config, records, batch):
    reference = config.get('z_reference', 'batch_entry')
    policies = {'batch_entry': 'CURRENT_BATCH_ENTRY_NO_FUTURE_TARGETS',
                'initial_model_on_arrival': 'CURRENT_BATCH_ONLY_INITIAL_MODEL'}
    if reference not in policies:
        raise ValueError('NATIVE_EVAL_Z_REFERENCE')
    if (payload['schema'] != 'native-fe-sequential-diagnostic-v1'
            or payload['identity_sha256'] != config['identity_sha256']
            or payload['z_policy'] != policies[reference]
            or payload['completed_batches'] != batch or batch not in (0, 5, 10, 15, 20)
            or payload['completed_edits'] != batch * 100):
        raise ValueError('NATIVE_EVAL_CHECKPOINT_IDENTITY')
    ids = [r['case_id'] for r in records[:batch*100]]
    if payload['seen_case_ids'] != ids:
        raise ValueError('NATIVE_EVAL_STREAM_ORDER')
    native = payload['config']
    if (native['model_dtype'] != 'bfloat16' or native['bs'] != 100
            or native['num_edits'] != 2000 or not native['algs']['add_old_keys']
            or native.get('diagnostic_z_reference', 'batch_entry') != reference
            or native['diagnostic_identity_sha256'] != config['identity_sha256']):
        raise ValueError('NATIVE_EVAL_CONFIG')
    expected = {native['llms']['rewrite_module_tmp'].format(layer)+'.weight'
                for layer in native['llms']['layers']}
    if set(payload['weights']) != expected:
        raise ValueError('NATIVE_EVAL_WEIGHT_SET')
    return expected


def restore_weights(model, weights):
    parameters = dict(model.named_parameters())
    # Validate the complete set before assigning any tensor.
    for name, value in weights.items():
        if (name not in parameters or value.shape != parameters[name].shape
                or value.dtype != parameters[name].dtype or not torch.isfinite(value).all()):
            raise ValueError('NATIVE_EVAL_WEIGHT_SHAPE_DTYPE_FINITE')
    with torch.no_grad():
        for name, value in weights.items():
            parameters[name].copy_(value)


def run(native_config, output, lock_path):
    lock = read(lock_path)
    if not Path(__file__).resolve().is_relative_to(Path(lock['source_directory']).resolve()):
        raise ValueError('NATIVE_EVAL_FROZEN_SOURCE')
    for member in lock['source_members'] + lock['configs']:
        verify(member)
    config = read(native_config)
    if digest({k: v for k, v in config.items() if k != 'identity_sha256'}) != config['identity_sha256']:
        raise ValueError('NATIVE_EVAL_CONFIG_IDENTITY')
    records = read(verify(config['original_stream']))
    if len(records) != 2000 or config['requests'] != 2000 or not config['current_batch_targets_only']:
        raise ValueError('NATIVE_EVAL_CF2K_STREAM')
    assets = read(verify(config['model_assets']))
    snapshot = assets['model']['snapshot']
    for member in assets['model']['members']:
        path = Path(member['path'])
        if path.stat().st_size != member['bytes'] or file_sha(path) != member['sha256']:
            raise ValueError('NATIVE_EVAL_PRETRAINED_ASSET')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        receipts = []
        for batch in (0, 5, 10, 15, 20):
            receipt_path = Path(config['output'])/'checkpoints'/f'B{batch:02d}.json'
            if receipt_path.exists():
                receipt = read(receipt_path)
                if receipt['identity_sha256'] != config['identity_sha256'] or receipt['completed_batches'] != batch:
                    raise ValueError('NATIVE_EVAL_RECEIPT_IDENTITY')
                receipts.append((batch, receipt))
        if not receipts:
            raise ValueError('NATIVE_EVAL_NO_COMMITTED_CHECKPOINTS')
        model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
            trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda:0').eval()
        tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = 'right'
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        results = {}
        for batch, receipt in receipts:
            checkpoint = verify(receipt)
            before_stat = checkpoint.stat()
            payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
            validate_payload(payload, config, records, batch)
            restore_weights(model, payload['weights'])
            del payload
            rows = records if batch == 0 else records[:batch*100]
            identity = dict(native_config=config['identity_sha256'], checkpoint_sha256=receipt['sha256'],
                native_source=config['ode_patch_source'], author_source=config['author_commit'],
                evaluation_source=lock['source_commit'], model=config['model'], dataset='cf',
                completed_edits=batch*100, evaluation_cases=len(rows), precision='bfloat16',
                z_reference=config.get('z_reference', 'batch_entry'),
                attention_backend=model.config._attn_implementation)
            observed = evaluate_counterfact(model, tok, rows, identity=identity, batch_size=16, device='cuda:0')
            after_stat = checkpoint.stat()
            if (before_stat.st_ino, before_stat.st_size, before_stat.st_mtime_ns) != (after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns):
                raise ValueError('NATIVE_EVAL_CHECKPOINT_CHANGED')
            if file_sha(checkpoint) != receipt['sha256']:
                raise ValueError('NATIVE_EVAL_CHECKPOINT_CONTENT_CHANGED')
            target = output/f'B{batch:02d}.json'
            write_new(target, observed)
            results[str(batch)] = dict(summary=observed['summary'], path=str(target), sha256=file_sha(target),
                                      checkpoint=receipt, evaluation_identity=identity)
            write_new(output/f'B{batch:02d}-receipt.json', results[str(batch)])
            print('NATIVE_CHECKPOINT_CF_METRICS', batch, json.dumps(observed['summary']), flush=True)
        write_new(output/'COMPLETE.json', dict(results=results, W20_available='20' in results,
            scope='Native FE checkpoint observation only; current baseline table is not overwritten',
            missing_milestones=[b for b in (0,5,10,15,20) if str(b) not in results],
            evaluation_source=lock['source_commit'], native_config=config['identity_sha256']))
    except BaseException:
        write_new(output/'FAILURE.json', dict(error=traceback.format_exc(), native_config=config['identity_sha256']))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lock', required=True)
    args = parser.parse_args()
    run(args.native_config, args.output, args.lock)
