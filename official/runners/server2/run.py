"""Actual GPT-J official baseline execution, batch checkpoints and resume proof.

Algorithms/hparams/factual/generation are exclusively the pinned official
distribution. This connector never downloads a model, recomputes C0/P, or
interprets low scientific scores as a technical failure.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from official.experiments import checkpoint
from official.experiments.prepare import build_matrix, digest, file_sha, load_plan, read, write_new
from official.runners.server2 import assets, generation
from official.runners.server2.native import METHODS, NativeEngine

INSTRUCTION = 'USER-OFFICIAL-BASELINES-20261008-R1'
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1')


def require(value, code):
    if not value:
        raise ValueError(code)


def member(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path))


def configuration(method, dataset):
    require(method in METHODS and dataset in ('cf', 'zsre'), 'OFFICIAL_CELL_SCOPE')
    contract, profiles = load_plan()
    return next(row for row in build_matrix(contract, profiles)
                if row['model'] == 'gptj' and row['method'] == method and row['dataset'] == dataset)


def checkpoint_identity(manifest, method, dataset):
    config = configuration(method, dataset)
    return dict(config_sha256=config['config_sha256'],
        stream_sha256=manifest['streams'][dataset]['lock']['stream_sha256'],
        code_commit=manifest['code_commit'], official_tree_sha256=manifest['official_tree_sha256'],
        model_revision=manifest['model_revision'], tokenizer_sha256=manifest['tokenizer_sha256'],
        assets_sha256=manifest['assets_identity_sha256'])


def equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def tensor_hash(value):
    array = value.detach().to('cpu').contiguous().numpy()
    sha = hashlib.sha256()
    sha.update(str(array.dtype).encode())
    sha.update(json.dumps(list(array.shape)).encode())
    sha.update(memoryview(array).cast('B'))
    return sha.hexdigest()


def physical_state(engine):
    return dict(completed_batch=engine.batch, method=engine.method,
        selected_weights={key:tensor_hash(value) for key, value in engine.weights().items()},
        native_history={key:tensor_hash(value) for key, value in engine.history().items()},
        context_sha256=digest(engine.contexts(False)))


def signature(model, engine=None):
    values = {}
    for kind, iterator in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
        for name, value in iterator:
            values[kind + ':' + name] = (value.data_ptr(), value._version,
                list(value.shape), str(value.dtype), str(value.device), value.requires_grad)
    modules = {name:dict(training=module.training,
        hooks={key:list(getattr(module, key).keys()) for key in
               ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks')})
        for name, module in model.named_modules()}
    return dict(tensors=values, modules=modules, config=model.config.to_dict(),
                native=None if engine is None else engine.state_identity())


def verify_manifest(manifest):
    require(manifest['instruction_id'] == INSTRUCTION and manifest['model'] == 'gptj'
        and manifest['owner']['server'] == 'server2', 'OFFICIAL_SERVER2_EXECUTION_IDENTITY')
    assets.verify(manifest['asset_manifest'])
    require(file_sha(manifest['asset_manifest']) == manifest['asset_manifest_sha256'],
            'OFFICIAL_ASSET_MANIFEST_SHA')
    require(os.environ.get('OFFICIAL_CODE_COMMIT') == manifest['code_commit']
        and os.environ.get('OFFICIAL_TREE_SHA256') == manifest['official_tree_sha256'],
        'OFFICIAL_FROZEN_SOURCE_ENVIRONMENT')
    root = Path(__file__).resolve().parents[2]
    expected = manifest['source_members']
    actual = {str(path.relative_to(root)):file_sha(path) for path in sorted(root.rglob('*'))
              if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    require(actual == expected, 'OFFICIAL_IMPORTED_SOURCE_BYTES_CHANGED')
    runtime = manifest['runtime']
    import transformers
    require(torch.__version__ == runtime['torch'] and transformers.__version__ == runtime['transformers'],
            'OFFICIAL_SCIENTIFIC_RUNTIME_CHANGED')
    return True


def tracker(manifest, config, mode, out):
    """Only the common owner's exact official transport API may initialize W&B."""
    binding = manifest.get('tracking')
    require(binding and binding.get('namespace', '').startswith('official.'),
            'OFFICIAL_TRACKING_SHARED_API_NOT_BOUND')
    module = importlib.import_module(binding['namespace'])
    require(file_sha(module.__file__) == binding['source_sha256'], 'OFFICIAL_TRACKING_SOURCE_CHANGED')
    values = dict(server='server2', task_id='official-baselines-20261008',
        arm='W0_BASE_MODEL' if mode == 'w0' else config['method'],
        attempt=out.name + '-' + mode, source_sha=manifest['code_commit'],
        config_sha=config['config_sha256'], model='gptj', model_family='gptj',
        writer='none' if mode == 'w0' else config['method'],
        role='scientific', metric_schema=binding['metric_schema'],
        dataset=config['dataset'], execution_mode=mode)
    # The common helper reads only the whitelisted real Slurm IDs and uses the
    # user's existing local credential. No token/key/full environment is copied.
    common = module.init(env_file=binding['env_file'], spool=str(out/'tracking'), config=values)
    return TransportAudit(common, out)


class TransportAudit:
    """Receipt-only adapter, not another SDK/logger implementation."""
    def __init__(self, common, out):
        self.common, self.out = common, out
        self.accepted = self.rejected = self.errors = 0
        self.finished = False
        self.finish_status = 'NOT_FINISHED'

    def log(self, values):
        try:
            result = self.common.log(values)
        except Exception:
            self.errors += 1
            return False
        if result is False:
            self.rejected += 1
        else:
            self.accepted += 1
        return result

    def finish(self, exit_code=0):
        try:
            result = self.common.finish(exit_code=exit_code)
            self.finish_status = 'BOUNDED_COMMON_FINISH_RETURNED_NOT_SCIENTIFIC_COMPLETION'
            return result
        except Exception:
            self.errors += 1
            self.finish_status = 'LOGGING_FINISH_FAILED_SANITIZED'
        finally:
            self.finished = True
            write_new(self.out/'logging-transport.json', self.receipt())

    def receipt(self):
        return dict(accepted_calls=self.accepted, rejected_calls=self.rejected,
            sanitized_errors=self.errors, bounded_finish=self.finished, finish_status=self.finish_status,
            status='LOGGING_DEGRADED' if self.rejected or self.errors else 'SDK_ACCEPTED_NOT_REMOTE_ACK',
            common_spool=str(self.out/'tracking'), no_scientific_retry=True,
            immutable_identity_and_remote_readback='COMMON_OWNER_RECEIPTS_AUTHORITATIVE')


def log(tracking, values):
    accepted = tracking.log(values)
    if accepted is False:
        # Existing raw results remain authoritative; transport degradation never
        # authorizes another model forward, fit or scientific retry.
        return 'LOGGING_DEGRADED_NOT_REMOTE_ACK'
    return 'SDK_ASYNC_NOT_REMOTE_ACK'


def load_model(manifest):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'ONE_ALLOCATED_GPU_REQUIRED')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tok = AutoTokenizer.from_pretrained(manifest['model_snapshot'], local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(manifest['model_snapshot'], local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager').to('cuda:0').eval()
    require(model.config.model_type == 'gptj' and all(value.dtype == torch.float32
        for value in model.parameters()), 'OFFICIAL_GPTJ_FP32_MODEL')
    return model, tok


def factual_identity(manifest, dataset):
    return dict(model_revision=manifest['model_revision'], tokenizer_sha256=manifest['tokenizer_sha256'],
        stream_sha256=manifest['streams'][dataset]['lock']['stream_sha256'],
        runtime=manifest['runtime'], source=manifest['code_commit'])


def factual(model, tok, manifest, records, dataset, out, endpoint, engine=None, w0_reference=None):
    api = importlib.import_module('official.evaluation.factual')
    before, rng = signature(model, engine), checkpoint.rng_snapshot()
    observed = api.evaluate(model, tok, records, dataset, w0_reference=w0_reference,
                            batch_size=16, device='cuda:0', identity=factual_identity(manifest, dataset))
    require(signature(model, engine) == before and equal(checkpoint.rng_snapshot(), rng),
            'OFFICIAL_FACTUAL_NATIVE_STATE_RNG_MUTATED')
    require(len(observed['cases']) == len(records)
        and [case['occurrence_index'] for case in observed['cases']]
            == [record['occurrence_index'] for record in records], 'OFFICIAL_FACTUAL_COHORT_IDENTITY')
    value = dict(observed, endpoint=endpoint, dataset=dataset, requests=len(records),
                 observer_no_mutation=True, RNG_restored=True)
    path = out/'factual'/(endpoint+'.json')
    write_new(path, value)
    return value, member(path)


def w0_reference(model, tok, records, manifest, out):
    api = importlib.import_module('official.evaluation.factual')
    before, rng = signature(model), checkpoint.rng_snapshot()
    value = api.build_zsre_w0_reference(model, tok, records,
        identity=factual_identity(manifest, 'zsre'), device='cuda:0', batch_size=16)
    require(signature(model) == before and equal(checkpoint.rng_snapshot(), rng),
            'OFFICIAL_ZSRE_W0_STATE_RNG_MUTATED')
    write_new(out/'W0-reference.json', value)
    return value


def save(engine, out, identity, evaluations, checkpoint_folder=None):
    return checkpoint.save(checkpoint_folder or out/'checkpoints', batch=engine.batch, weights=engine.weights(),
        cache_c=engine.history(), contexts=engine.contexts(False),
        evaluation_cursor=evaluations, identity=identity, method=engine.method, evaluation_complete=True)


def evaluate_payload(observed, endpoint, edits):
    # This schema reports request-macro E/G/S, not old prompt-pair R/P/N or
    # paper free-generation accuracy. The shared official whitelist binds it.
    prefix = 'W0_first2000' if endpoint == 'W0' else 'all_seen/post'
    value = {prefix + '/' + key:score for key, score in observed['summary'].items()
             if type(score) in (int, float)}
    return dict(value, edits=edits, pre_state_edits=max(0, edits-100), post_state_edits=edits)


def qualification(model, tok, engine, manifest, records, out, tracking):
    """Same native B3 versus actual durable B2->B3; no mock qualification."""
    require(engine.batch == 0, 'QUALIFICATION_COLD_NATIVE_REQUIRED')
    identity = checkpoint_identity(manifest, engine.method, 'cf')
    engine.contexts()
    cursor = {}
    save(engine, out, identity, cursor)
    commits = []
    for batch in (1, 2):
        receipt = engine.apply(records[(batch-1)*100:batch*100], batch)
        log(tracking, dict(batch=batch, edits=batch*100))
        observed, observed_member = factual(model, tok, manifest, records[:batch*100], 'cf', out,
                                            'qualification-W'+str(batch), engine)
        cursor[str(batch)] = observed_member
        ref = save(engine, out, identity, cursor)
        commits.append(dict(receipt, checkpoint=ref))
    durable_B2 = read(out/'checkpoints/latest.json')
    receipt3 = engine.apply(records[200:300], 3)
    continuous, continuous_member = factual(model, tok, manifest, records[200:300], 'cf', out,
                                           'continuous-B3', engine)
    continuous_weights = {key:value.detach().cpu().clone() for key,value in engine.weights().items()}
    continuous_history = {key:value.clone() for key,value in engine.history().items()}
    continuous_contexts, continuous_rng = engine.contexts(False), checkpoint.rng_snapshot()
    # Load the real fsync+SHA-bound B2 payload. Nothing is inferred from raw
    # observations, and native target fits/FT optimizer restart naturally at B3.
    saved = checkpoint.load(out/'checkpoints', identity)
    require(saved['batch'] == 2, 'QUALIFICATION_DURABLE_B2_REQUIRED')
    engine.restore(saved)
    checkpoint.rng_restore(saved['rng'])
    del saved
    resumed_receipt = engine.apply(records[200:300], 3)
    resumed, resumed_member = factual(model, tok, manifest, records[200:300], 'cf', out, 'resumed-B3', engine)
    flags = dict(weights_equal=equal(continuous_weights,
                                    {key:value.detach().cpu() for key,value in engine.weights().items()}),
        history_equal=equal(continuous_history, engine.history()),
        contexts_equal=equal(continuous_contexts, engine.contexts(False)),
        rng_equal=equal(continuous_rng, checkpoint.rng_snapshot()),
        metrics_equal=equal(continuous['cases'], resumed['cases'])
                      and equal(continuous['summary'], resumed['summary']))
    require(all(flags.values()), 'ACTUAL_B2_B3_NATIVE_RESUME_PARITY_FAILED')
    cursor['3'] = resumed_member
    final = save(engine, out, identity, cursor)
    value = dict(status='PASS_ACTUAL_QUALIFICATION', actual_GPU=True, model='gptj', dataset='cf',
        method=engine.method, code_commit=manifest['code_commit'],
        official_tree_sha256=manifest['official_tree_sha256'],
        manifest_sha256=manifest['base_manifest_sha256'], checkpoint_identity=identity,
        continuous_batches=3, resume_after_batch=2, resumed_batches=[3], **flags,
        durable_B2=durable_B2, checkpoint=final, continuous_metric=continuous_member,
        resumed_metric=resumed_member, native_commits=commits+[receipt3, resumed_receipt],
        actual_native_batch_calls=4, actual_native_request_applications=400,
        no_quality_selection=True, CPU_fixture_is_not_actual_proof=True)
    write_new(out/'qualification.json', value)
    return value


def cold_w0(model, tok, manifest, records, dataset, out, tracking):
    if dataset == 'zsre':
        reference = w0_reference(model, tok, records, manifest, out)
        observed = reference['evaluation']
        path = out/'factual/W0.json'
        write_new(path, dict(observed, endpoint='W0', dataset=dataset, requests=len(records)))
        factual_member = member(path)
        generation_member = None
    else:
        observed, factual_member = factual(model, tok, manifest, records, dataset, out, 'W0')
        result = generation.observe(model, tok, manifest, records, out/'generation', 'W0',
            dict(completed_batch=0), lambda:signature(model), lambda values:log(tracking, values))
        generation_member = member(result['rows_path'])
    log(tracking, evaluate_payload(observed, 'W0', 0))
    ready = dict(status='READY_COLD_W0_COMPLETE', actual_GPU=True, model='gptj', dataset=dataset,
        code_commit=manifest['code_commit'], official_tree_sha256=manifest['official_tree_sha256'],
        manifest_sha256=manifest['base_manifest_sha256'],
        model_revision=manifest['model_revision'], tokenizer_sha256=manifest['tokenizer_sha256'],
        stream_sha256=manifest['streams'][dataset]['lock']['stream_sha256'],
        factual=factual_member, generation=generation_member,
        generation_READY=member(out/'generation/READY.json') if dataset == 'cf' else None,
        w0_reference=member(out/'W0-reference.json') if dataset == 'zsre' else None)
    write_new(out/'READY.json', ready)
    return ready


def read_w0(manifest, dataset, records):
    path = Path(manifest['W0_'+dataset+'_ready_path'])
    ready = read(path)
    require(ready['status'] == 'READY_COLD_W0_COMPLETE' and ready['actual_GPU'] is True
        and ready['model_revision'] == manifest['model_revision']
        and ready['tokenizer_sha256'] == manifest['tokenizer_sha256']
        and ready['code_commit'] == manifest['code_commit']
        and ready['official_tree_sha256'] == manifest['official_tree_sha256']
        and ready['stream_sha256'] == manifest['streams'][dataset]['lock']['stream_sha256'],
        'OFFICIAL_W0_READY_IDENTITY')
    require(member(ready['factual']['path']) == ready['factual'], 'OFFICIAL_W0_FACTUAL_RAW_CHANGED')
    if dataset == 'cf':
        generation.reuse_w0(ready['generation_READY']['path'], manifest, records)
        reference = None
    else:
        require(member(ready['w0_reference']['path']) == ready['w0_reference'], 'OFFICIAL_ZSRE_W0_RAW_CHANGED')
        reference = read(ready['w0_reference']['path'])
    return ready, reference


def chain(model, tok, engine, manifest, records, dataset, out, tracking, resume=None, smoke=False):
    identity = checkpoint_identity(manifest, engine.method, dataset)
    endpoints, cursor, commits = {}, {}, []
    ready, reference = read_w0(manifest, dataset, records)
    endpoints['W0'] = ready['factual']
    checkpoint_folder = Path(resume).resolve() if resume else out/'checkpoints'
    require(checkpoint_folder.name == 'checkpoints' and LOCAL in checkpoint_folder.parents
        and not checkpoint_folder.is_symlink(), 'OWN_TASK_CHECKPOINT_FOLDER_SCOPE')
    if resume:
        saved = checkpoint.load(resume, identity)
        engine.restore(saved)
        checkpoint.rng_restore(saved['rng'])
        cursor = saved['evaluation_cursor']
        # New attempt has fresh logs/run identity, but only the original owned
        # checkpoint folder advances. Historical failure/terminal files remain.
        original_out = checkpoint_folder.parent
        for key, value in cursor.items():
            if key in ('W0', 'W5', 'W10', 'W15', 'W20'):
                endpoints[key] = value
        commits = list(cursor.get('commit_members', []))
        current_path = cursor.get('current_commit_path')
        if current_path:
            current = Path(current_path)
            if not current.exists():
                # The durable checkpoint includes the actual completed native
                # receipt/evaluation cursor. Recover only this metadata commit
                # if a process died between checkpoint fsync and ledger write.
                current_receipt = cursor['current_commit_receipt']
                require(current_receipt['batch'] == saved['batch'], 'CHECKPOINT_COMMIT_CURSOR_BATCH')
                write_new(current, dict(current_receipt, checkpoint=read(checkpoint_folder/'latest.json'),
                                        metadata_recovered_from_durable_checkpoint=True))
            require(read(current)['batch'] == saved['batch'], 'RESUME_LAST_COMMIT_BATCH')
            commits.append(member(current))
        elif not commits:
            for path in sorted((original_out/'commits').glob('batch-*.json')):
                if read(path)['batch'] <= saved['batch']:
                    commits.append(member(path))
        require(len(commits) == saved['batch'], 'RESUME_COMPLETED_COMMIT_PROVENANCE_REQUIRED')
        del saved
    else:
        engine.contexts()
        cursor['W0'] = ready['factual']
        save(engine, out, identity, cursor)
    end = 1 if smoke else 20
    for batch in range(engine.batch+1, end+1):
        previous = engine.batch
        started = time.monotonic()
        receipt = engine.apply(records[(batch-1)*100:batch*100], batch)
        log(tracking, dict(batch=batch, edits=batch*100, **{'time/phase_seconds':time.monotonic()-started}))
        if batch in (5, 10, 15, 20) or smoke:
            observed, observed_member = factual(model, tok, manifest, records[:batch*100], dataset, out,
                                                'W'+str(batch), engine, reference)
            endpoints['W'+str(batch)] = cursor['W'+str(batch)] = observed_member
            log(tracking, evaluate_payload(observed, 'W'+str(batch), batch*100))
        generation_endpoint = None
        if dataset == 'cf' and batch == 20 and not smoke:
            observed_generation = generation.observe(model, tok, manifest, records, out/'generation',
                'W20', physical_state(engine), lambda:signature(model, engine),
                lambda values:log(tracking, values))
            generation_endpoint = member(observed_generation['rows_path'])
            cursor['generation_W20'] = generation_endpoint
            cursor['generation_READY'] = member(out/'generation/READY.json')
        # Bind actual current native/evaluation metadata inside the checkpoint.
        # Its post-save external ledger member cannot be circularly hashed.
        path = out/'commits'/('batch-'+str(batch).zfill(2)+'.json')
        cursor['commit_members'] = list(commits)
        cursor['current_commit_path'] = str(path)
        current_receipt = dict(receipt, previous_batch=previous,
            checkpoint_identity=identity, native_state_identity=engine.state_identity(),
            factual_endpoint=endpoints.get('W'+str(batch)),
            generation_endpoint=cursor.get('generation_W20') if batch == 20 else None)
        cursor['current_commit_receipt'] = current_receipt
        ref = save(engine, out, identity, cursor, checkpoint_folder)
        write_new(path, dict(current_receipt, checkpoint=ref))
        commits.append(member(path))
    if smoke:
        value = dict(status='PASS_ACTUAL_SMOKE', actual_GPU=True, model='gptj', dataset='zsre',
            method=engine.method, batches=1, code_commit=manifest['code_commit'],
            official_tree_sha256=manifest['official_tree_sha256'], manifest_sha256=manifest['base_manifest_sha256'],
            factual_endpoints=endpoints, commits=commits, checkpoint_identity=identity)
        write_new(out/'smoke.json', value)
        return value
    value = dict(status='SCIENTIFIC_COMPLETE', actual_GPU=True, model='gptj', dataset=dataset,
        method=engine.method, batches=20, code_commit=manifest['code_commit'],
        official_tree_sha256=manifest['official_tree_sha256'], manifest_sha256=manifest['base_manifest_sha256'],
        checkpoint_identity=identity, checkpoint_folder=str(checkpoint_folder),
        factual_endpoints=endpoints, commits=commits, ownstate_links=19,
        history_appends=sum(read(item['path'])['history_appends_expected'] for item in commits),
        generation_endpoint=cursor.get('generation_W20'), W0_READY=member(manifest['W0_'+dataset+'_ready_path']),
        generation_ready=cursor.get('generation_READY') if dataset == 'cf' else None,
        w0_ready=ready['generation_READY'] if dataset == 'cf' else None,
        native_only=True, requested_quality_not_selection_gate=True)
    write_new(out/'result.json', value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--dataset', choices=('cf', 'zsre'), required=True)
    parser.add_argument('--mode', choices=('qualification', 'chain', 'smoke', 'w0'), required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    manifest, out = read(args.manifest), Path(args.out).resolve()
    require(LOCAL in out.parents and not out.is_symlink(), 'OWN_TASK_OUTPUT_SCOPE')
    out.mkdir(parents=True, exist_ok=True)
    require(not (out/'terminal.json').exists() and not (out/'failure.json').exists(),
            'ATTEMPT_ALREADY_EXECUTED_USE_NEW_EXPLICIT_RESUME_ATTEMPT')
    started, tracking, result, error = time.monotonic(), None, None, None
    try:
        verify_manifest(manifest)
        config = configuration(args.method, args.dataset)
        records = read(manifest['streams'][args.dataset]['path'])
        require(len(records) == 2000 and [r['occurrence_index'] for r in records] == list(range(1,2001)),
                'OFFICIAL_ORDERED_STREAM_2000')
        # Shared input/source must be ready before allocating model memory. This
        # is not a fake online PASS or CPU substitution for actual GPU proof.
        importlib.import_module('official.evaluation.factual')
        tracking = tracker(manifest, config, args.mode, out)
        model, tok = load_model(manifest)
        if args.mode == 'w0':
            result = cold_w0(model, tok, manifest, records, args.dataset, out, tracking)
        else:
            engine = NativeEngine(model, tok, args.method, manifest)
            if args.mode == 'qualification':
                require(args.dataset == 'cf' and not args.resume, 'ACTUAL_QUALIFICATION_CF_ONLY')
                result = qualification(model, tok, engine, manifest, records, out, tracking)
            else:
                require(args.mode != 'smoke' or args.dataset == 'zsre', 'ZSRE_SMOKE_ONLY')
                result = chain(model, tok, engine, manifest, records, args.dataset, out, tracking,
                               resume=args.resume, smoke=args.mode == 'smoke')
    except BaseException as exc:
        error = exc
        write_new(out/'failure.json', dict(status='TECHNICAL_FAILURE', exception_type=type(exc).__name__,
            error_code=str(exc)[:160] if isinstance(exc, ValueError) else 'EXCEPTION_DETAILS_LOCAL_STDERR',
            elapsed_seconds=time.monotonic()-started, preserved_source_raw_checkpoint=True,
            new_fit_or_automatic_retry=False))
        raise
    finally:
        if tracking is not None:
            try:
                tracking.finish(exit_code=1 if error else 0)
            except Exception:
                # Never hide the scientific exception or block result persistence.
                pass
        write_new(out/'terminal.json', dict(status='FAILED' if error else 'COMPLETED',
            scientific_status=None if result is None else result['status'],
            elapsed_seconds=time.monotonic()-started,
            actual_GPU=torch.cuda.is_initialized(), checkpoint_policy='LATEST1_FINAL_W20',
            tracking=None if tracking is None else tracking.receipt(),
            logger_finish_is_not_scientific_completion=True, automatic_retry=False))


if __name__ == '__main__':
    main()
