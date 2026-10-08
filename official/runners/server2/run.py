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
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from official.experiments import checkpoint
from official.evaluation import reduce as factual_reduce
from official.experiments.prepare import build_matrix, digest, file_sha, load_plan, read, write_new
from official.runners.server2 import assets, generation, oracle, parity
from official.runners.server2.telemetry import NativeTelemetry
from official.runners.server2.native import METHODS, NativeEngine
from official.runners.server2 import no_gpu_qualification as noqual

INSTRUCTION = 'USER-OFFICIAL-BASELINES-20261008-R1'
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1')


def deferred(manifest):
    """Caller schedule only, outside the unchanged native qualification closure."""
    from official.runners.server2.checkpoint_profile import deferred as validate_schedule
    return noqual.enabled(manifest) or validate_schedule(manifest)


def require(value, code):
    if not value:
        raise ValueError(code)


def member(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path))


def failure_checkpoint_metadata(folder):
    """Read metadata only; a damaged receipt must not replace the first error."""
    path = Path(folder)/'latest.json'
    if not path.is_file():
        return None
    try:
        return read(path)
    except Exception as error:
        return dict(status='CHECKPOINT_METADATA_UNREADABLE',
                    exception_type=type(error).__name__, original_error_preserved=True)


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
    if manifest.get('registration_stage') == 'cf':
        from official.runners.server2.submit import gate
        bound = manifest.get('qualification_receipt')
        require(isinstance(bound, dict) and member(bound['path']) ==
            {key:bound[key] for key in ('path', 'bytes', 'sha256')},
            'CF_NATIVE_RESUME_PRODUCER_INPUT_CHANGED')
        require(gate(bound['path'], manifest=manifest, kind='qualification') == bound,
                'CF_NATIVE_RESUME_CONSUMER_BINDING_CHANGED')
    return True


def tracking_config(manifest, config, mode, out):
    """Strict shared config mapping; CPU-testable without SDK/auth or upload."""
    binding = manifest['tracking']
    values = dict(server='server2', task_id='official-baselines-20261008',
        arm='W0_BASE_MODEL' if mode == 'w0' else
            ('QUALIFICATION_' if mode == 'qualification' else 'ZSRE_SMOKE_' if mode == 'smoke' else '')
            + config['method'],
        attempt=out.name + '-' + mode, source_sha=manifest['code_commit'],
        config_sha=config['config_sha256'], model='gptj', model_family='gptj',
        writer='none' if mode == 'w0' else config['method'],
        role='scientific', metric_schema=binding['metric_schema'],
        baseline='none' if mode == 'w0' else config['method'],
        instruction_id=INSTRUCTION, dataset=config['dataset'])
    if config['dataset'] == 'cf' and deferred(manifest):
        values.update(config_sha=digest(dict(native_config_sha256=config['config_sha256'],
                      checkpoint_only_profile=manifest.get('checkpoint_only_profile'))),
                      generation_schedule='DEFERRED_CHECKPOINT_EVALUATION')
    elif config['dataset'] == 'cf':
        measured = generation.configuration(manifest)
        values.update(generation_metric_schema='counterfact-cake-generation-metrics-v1',
            generation_profile=measured['profile'], generation_eval_seed=measured['eval_seed'],
            reference_assets_sha256=measured['reference_assets_sha256'],
            generation_source_sha=measured['generation_source_sha'],
            generation_repair_instruction=INSTRUCTION,
            generation_schedule='W0_AND_W20_FIRST2000')
    if noqual.enabled(manifest):
        values['config_sha'] = digest(dict(native_config_sha256=config['config_sha256'],
            user_overlay=manifest['no_gpu_qualification_profile']))
    if manifest.get('cf_display_repair'):
        values['config_sha'] = digest(dict(parent_config_sha=values['config_sha'],
            cf_display_repair=manifest['cf_display_repair']))
    return values


def tracker(manifest, config, mode, out):
    """Only the common owner's exact official transport API may initialize W&B."""
    binding = manifest.get('tracking')
    require(binding and binding.get('namespace') == 'official.tracking',
            'OFFICIAL_TRACKING_SHARED_API_NOT_BOUND')
    module = importlib.import_module(binding['namespace'])
    require(file_sha(module.__file__) == binding['source_sha256'], 'OFFICIAL_TRACKING_SOURCE_CHANGED')
    values = tracking_config(manifest, config, mode, out)
    # The common helper reads only the whitelisted real Slurm IDs and uses the
    # user's existing local credential. No token/key/full environment is copied.
    common = module.init(env_file=binding['env_file'], spool=str(out/'tracking'), config=values)
    return TransportAudit(common, out, values)


class TransportAudit:
    """Receipt-only adapter, not another SDK/logger implementation."""
    def __init__(self, common, out, config_values=None):
        self.common, self.out = common, out
        self.config_values = config_values
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


def factual(model, tok, manifest, records, dataset, out, endpoint, engine=None, w0_reference=None,
            parity_plan=None):
    api = importlib.import_module('official.evaluation.factual')
    before, rng = signature(model, engine), checkpoint.rng_snapshot()
    def evaluate_call():
        return api.evaluate(model, tok, records, dataset, w0_reference=w0_reference,
            batch_size=16, device='cuda:0', identity=factual_identity(manifest, dataset))
    parity_member = None
    if parity_plan is not None:
        require(engine is not None and engine.batch == 1, 'NATIVE_PARITY_FIRST_ACTUAL_B1_ONLY')
        observed, proof = parity.observe(model, tok, engine, manifest, records, parity_plan, evaluate_call)
        write_new(out/'native-parity'/('B1-'+dataset+'.json'), proof)
        parity_member = member(out/'native-parity'/('B1-'+dataset+'.json'))
    else:
        observed = evaluate_call()
    require(signature(model, engine) == before and equal(checkpoint.rng_snapshot(), rng),
            'OFFICIAL_FACTUAL_NATIVE_STATE_RNG_MUTATED')
    require(len(observed['cases']) == len(records)
        and [case['occurrence_index'] for case in observed['cases']]
            == [record['occurrence_index'] for record in records], 'OFFICIAL_FACTUAL_COHORT_IDENTITY')
    value = dict(observed, endpoint=endpoint, dataset=dataset, requests=len(records),
                 observer_no_mutation=True, RNG_restored=True, native_owner_formula_parity=parity_member)
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


def native_apply(engine, records, batch, tracking, cursor):
    """Numeric native stdout only; no added tensor conversion/model work."""
    fit_cursor = cursor.setdefault('fit_telemetry', {})
    with NativeTelemetry(tracking.log, batch, cursor=fit_cursor, method=engine.method) as bridge:
        receipt = engine.apply(records, batch)
    return dict(receipt, fit_telemetry=bridge.receipt)


def evaluate_payload(observed, endpoint, edits, *, current=False, config_values=None):
    # This schema reports request-macro E/G/S, not old prompt-pair R/P/N or
    # paper free-generation accuracy. The shared official whitelist binds it.
    prefix = 'W0_first2000' if endpoint == 'W0' else 'current/post' if current else 'all_seen/post'
    if observed.get('dataset', observed.get('identity', {}).get('dataset')) == 'zsre':
        from official.tracking import official_zsre_metrics
        return official_zsre_metrics(observed['summary'], config_values=config_values,
            endpoint=prefix, edits=edits, pre_state_edits=max(0,edits-100), post_state_edits=edits)
    fields = ('Efficacy', 'Generalization', 'Specificity', 'Score',
              'Score_AlphaEdit_display', 'Specificity_loc_ans', 'requests')
    value = {'official/' + prefix + '/' + key:score for key, score in observed['summary'].items()
             if key in fields and type(score) in (int, float)}
    if observed.get('dataset', observed.get('identity', {}).get('dataset')) == 'cf':
        if 'Score_AlphaEdit_display' in observed['summary']:
            for kind, label in (('rewrite','Efficacy'),('paraphrase','Generalization'),('neighborhood','Specificity')):
                rates = []
                for case in observed['cases']:
                    bits = [float(row['target_true']['mean_nll'] < row['target_new']['mean_nll']
                        if kind == 'neighborhood' else
                        row['target_new']['mean_nll'] < row['target_true']['mean_nll'])
                        for row in case[kind+'_observations']]
                    require(bool(bits), 'DISPLAY_EMPTY_REQUEST')
                    rates.append(np.mean(bits))
                value['official/'+prefix+'/'+label+'_AlphaEdit_display'] = float(np.around(np.mean(rates)*100,2))
        value.update(cf_diagnostics(observed['cases'], prefix))
    return dict(value, edits=edits, pre_state_edits=max(0, edits-100), post_state_edits=edits)


def cf_diagnostics(cases, prefix):
    """Already measured native prompt-pair/token rows, not request-macro aliases."""
    value, successes = {}, []
    for kind, label in (('rewrite', 'R'), ('paraphrase', 'P'), ('neighborhood', 'N')):
        rows = [row for case in cases for row in case[kind+'_observations']]
        require(bool(rows), 'EMPTY_CF_DIAGNOSTICS')
        desired = [row['target_true' if kind == 'neighborhood' else 'target_new'] for row in rows]
        require(all(row['desired_target'] == ('true' if kind == 'neighborhood' else 'new')
                    for row in rows), 'CF_DIAGNOSTICS_DESIRED_TARGET')
        count = len(rows)
        success = sum((row['target_true']['mean_nll'] < row['target_new']['mean_nll'])
            if kind == 'neighborhood' else
            (row['target_new']['mean_nll'] < row['target_true']['mean_nll']) for row in rows)
        token_count = sum(row['token_count'] for row in desired)
        true_nll = math.fsum(row['target_true']['mean_nll'] for row in rows)/count
        new_nll = math.fsum(row['target_new']['mean_nll'] for row in rows)/count
        item = dict(count=count, success_count=success, success_pct=100*success/count,
            token_acc_pct=100*sum(row['token_correct_count'] for row in desired)/token_count,
            prompt_acc_pct=100*math.fsum(row['token_correct_count']/row['token_count']
                                        for row in desired)/count,
            strict_acc_pct=100*sum(row['strict_correct'] for row in desired)/count,
            true_nll=true_nll, new_nll=new_nll, margin_true_minus_new=true_nll-new_nll)
        value.update({prefix+'/'+label+'/'+key:scalar for key, scalar in item.items()})
        successes.append(item['success_pct'])
    value[prefix+'/success_harmonic_pct'] = factual_reduce.harmonic(successes)
    return value


def current_subset(observed, records, dataset):
    """Last incoming 100 rows of the same actual milestone, no extra forward."""
    require(len(records) == 100, 'CURRENT_COHORT_BATCH100')
    cases = observed['cases'][-100:]
    require([(row['case_id'], row['occurrence_index']) for row in cases]
        == [(row['case_id'], row['occurrence_index']) for row in records],
        'CURRENT_SUBSET_ORDERED_OCCURRENCE')
    summary = factual_reduce.counterfact(cases) if dataset == 'cf' else factual_reduce.zsre(cases)
    return dict(cases=cases, summary=summary, dataset=dataset,
        parent_identity_sha256=observed['identity_sha256'], work=dict(new_model_forward_calls=0),
        current_pre_availability='NOT_MEASURED_BY_OFFICIAL_MILESTONE_SCHEDULE')


def qualification(model, tok, engine, manifest, records, out, tracking):
    """Same native B3 versus actual durable B2->B3; no mock qualification."""
    require(engine.batch == 0, 'QUALIFICATION_COLD_NATIVE_REQUIRED')
    identity = checkpoint_identity(manifest, engine.method, 'cf')
    engine.contexts()
    cursor = {}
    save(engine, out, identity, cursor)
    commits = []
    for batch in (1, 2):
        receipt = native_apply(engine, records[(batch-1)*100:batch*100], batch, tracking, cursor)
        log(tracking, dict(batch=batch, edits=batch*100))
        observed, observed_member = factual(model, tok, manifest, records[:batch*100], 'cf', out,
            'qualification-W'+str(batch), engine,
            parity_plan=manifest['native_parity_plans']['cf'][engine.method] if batch == 1 else None)
        if batch == 1:
            native_formula_parity = observed['native_owner_formula_parity']
        cursor[str(batch)] = observed_member
        ref = save(engine, out, identity, cursor)
        commits.append(dict(receipt, checkpoint=ref))
    durable_B2 = read(out/'checkpoints/latest.json')
    receipt3 = native_apply(engine, records[200:300], 3, tracking, cursor)
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
    # Keep this run's transport axis monotone, independently of restored edit
    # RNG/state; the second B3 is a qualification replay, not another endpoint.
    resumed_receipt = native_apply(engine, records[200:300], 3, tracking, cursor)
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
        native_owner_formula_parity=native_formula_parity,
        CF_original_evaluator_parity='NOT_OBSERVED_NOT_SUBSTITUTED_BY_RESUME_OR_FORMULA_PROOF',
        no_quality_selection=True, CPU_fixture_is_not_actual_proof=True)
    write_new(out/'qualification.json', value)
    return value


def cf_native_oracle(model, tok, manifest, records, out):
    """One locked first4 engineering observation; never hotpatch old jobs.

    The preregistered plan is part of the future execution manifest, not an
    actual receipt invented during held inspection. Physical state is captured
    before the fresh canonical observation. Raw/state/reference remain local.
    """
    frozen = manifest.get('cf_native_reference_plan')
    require(frozen is not None and frozen == oracle.plan(manifest, records[:4]),
            'PREMEASUREMENT_CF_NATIVE_ORACLE_PLAN_REQUIRED')
    folder = out/'native-oracle'
    write_new(folder/'plan.json', frozen)
    before = oracle.capture_state(model, tok)
    # The literal public scorer calls model(ids, mask), without a use_cache
    # kwarg. Supply its declared native precondition for this call only, and
    # restore before full W0/generation. No model-name/tokenizer spoofing.
    previous_cache = model.config.use_cache
    original_error = None
    try:
        model.config.use_cache = False
        state = oracle.capture_state(model, tok)
        write_new(folder/'state-before-canonical.json', state)
        canonical, canonical_member = factual(model, tok, manifest, records[:4], 'cf', out,
            'native-reference-canonical-first4')
        proof = oracle.compare(model, tok, manifest, records[:4], canonical,
            frozen_plan=frozen, canonical_member=canonical_member, state_identity=state)
    except BaseException as error:
        original_error = error
        raise
    finally:
        model.config.use_cache = previous_cache
        restored = oracle.capture_state(model, tok) == before
        if not restored and original_error is not None:
            original_error.add_note('ORACLE_CALL_LOCAL_CONFIG_STATE_NOT_RESTORED; FIRST_ERROR_PRESERVED')
        else:
            require(restored, 'ORACLE_CALL_LOCAL_CONFIG_STATE_NOT_RESTORED')
    # Retain the complete typed comparison before a mismatch blocks READY.
    write_new(folder/'comparison.json', proof)
    checked = oracle.verify(frozen, proof, canonical_member)
    require(checked['status'] == 'PASS_ACTUAL_GPU_SMOKE' and checked['actual_GPU'] is True,
            'ACTUAL_INDEPENDENT_ORIGINAL_CF_SMOKE_NOT_PASSED')
    return dict(plan_sha256=frozen['plan_sha256'], proof=member(folder/'comparison.json'),
        canonical=canonical_member, state=member(folder/'state-before-canonical.json'),
        scope=checked['evidence_scope'], full_2k_parity='NOT_APPLICABLE_GPTJ; NOT_OBSERVED')


def verify_cf_native_oracle(manifest, binding):
    """Stored local proof verification only, no model/GPU/extra observation."""
    require(isinstance(binding, dict), 'W0_ACTUAL_ORIGINAL_ORACLE_BINDING_REQUIRED')
    frozen = manifest.get('cf_native_reference_plan')
    require(frozen is not None and binding.get('plan_sha256') == frozen['plan_sha256'],
            'W0_ORIGINAL_ORACLE_FROZEN_PLAN_BINDING')
    for key in ('proof', 'canonical', 'state'):
        require(member(binding[key]['path']) == binding[key], 'W0_ORACLE_MEMBER_CHANGED:'+key)
    proof = read(binding['proof']['path'])
    checked = oracle.verify(frozen, proof, binding['canonical'])
    require(checked['status'] == 'PASS_ACTUAL_GPU_SMOKE' and checked['actual_GPU'] is True
        and binding.get('scope') == checked['evidence_scope']
        and proof['binding']['physical_state_sha256'] == digest(read(binding['state']['path']))
        and binding.get('full_2k_parity') == 'NOT_APPLICABLE_GPTJ; NOT_OBSERVED',
        'W0_ORIGINAL_CF_ACTUAL_SOURCE_STATE_SCOPE_PROOF')
    return checked


def cold_w0(model, tok, manifest, records, dataset, out, tracking):
    native_reference = None
    if dataset == 'zsre':
        reference = w0_reference(model, tok, records, manifest, out)
        observed = reference['evaluation']
        path = out/'factual/W0.json'
        write_new(path, dict(observed, endpoint='W0', dataset=dataset, requests=len(records)))
        factual_member = member(path)
        generation_member = None
    else:
        native_reference = (dict(status=noqual.DISABLED, instruction_id=noqual.INSTRUCTION)
            if noqual.enabled(manifest) else cf_native_oracle(model, tok, manifest, records, out))
        observed, factual_member = factual(model, tok, manifest, records, dataset, out, 'W0')
        generation_member = None
        if not deferred(manifest):
            result = generation.observe(model, tok, manifest, records, out/'generation', 'W0',
                dict(completed_batch=0), lambda:signature(model), lambda values:log(tracking, values))
            generation_member = member(result['rows_path'])
    log(tracking, evaluate_payload(observed, 'W0', 0, config_values=getattr(tracking,'config_values',None)))
    ready = dict(status='READY_COLD_W0_COMPLETE', actual_GPU=True, model='gptj', dataset=dataset,
        code_commit=manifest['code_commit'], official_tree_sha256=manifest['official_tree_sha256'],
        manifest_sha256=manifest['base_manifest_sha256'],
        model_revision=manifest['model_revision'], tokenizer_sha256=manifest['tokenizer_sha256'],
        stream_sha256=manifest['streams'][dataset]['lock']['stream_sha256'],
        factual=factual_member, generation=generation_member,
        generation_READY=member(out/'generation/READY.json') if dataset == 'cf' and not deferred(manifest) else None,
        generation_status='DEFERRED_NOT_MEASURED' if deferred(manifest) else 'SCHEDULED',
        w0_reference=member(out/'W0-reference.json') if dataset == 'zsre' else None,
        original_native_reference=native_reference)
    write_new(out/'READY.json', ready)
    return ready


def read_w0(manifest, dataset, records):
    if dataset == 'cf' and manifest.get('cf_display_repair'):
        from official.runners.server2.cf_display_repair import reused_w0
        return reused_w0(manifest, records), None
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
        if noqual.enabled(manifest):
            require(ready.get('original_native_reference') == dict(status=noqual.DISABLED,
                instruction_id=noqual.INSTRUCTION), 'EXPLICIT_USER_DISABLED_ORACLE_NOT_PASS')
        else:
            verify_cf_native_oracle(manifest, ready.get('original_native_reference'))
        if deferred(manifest):
            require(ready.get('generation_status') == 'DEFERRED_NOT_MEASURED'
                and ready.get('generation') is None and ready.get('generation_READY') is None,
                'DEFERRED_W0_NO_GENERATION_OR_RELABELED_RAW')
        else:
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
    if manifest.get('cf_display_repair'):
        require(dataset == 'cf' and not resume and not smoke, 'CF_DISPLAY_COLD_MAIN_ONLY')
        write_new(out/'W0-provenance.json', dict(original_ready=member(manifest['W0_cf_ready_path']),
            original_factual=ready['factual'], consumer_source=manifest['code_commit'],
            compatibility=manifest['cf_display_repair']['W0_binding'], historical_online_overwrite=False))
        log(tracking, evaluate_payload(read(ready['factual']['path']), 'W0', 0,
            config_values=getattr(tracking,'config_values',None)))
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
        receipt = native_apply(engine, records[(batch-1)*100:batch*100], batch, tracking, cursor)
        log(tracking, dict(batch=batch, edits=batch*100, **{'time/phase_seconds':time.monotonic()-started}))
        if batch in (5, 10, 15, 20) or smoke:
            observed, observed_member = factual(model, tok, manifest, records[:batch*100], dataset, out,
                'W'+str(batch), engine, reference,
                parity_plan=manifest['native_parity_plans']['zsre'][engine.method] if smoke else None)
            endpoints['W'+str(batch)] = cursor['W'+str(batch)] = observed_member
            if not smoke:
                log(tracking, evaluate_payload(observed, 'W'+str(batch), batch*100,
                    config_values=getattr(tracking,'config_values',None)))
            incoming = current_subset(observed, records[(batch-1)*100:batch*100], dataset)
            log(tracking, evaluate_payload(incoming, 'W'+str(batch), batch*100, current=True,
                config_values=getattr(tracking,'config_values',None)))
        generation_endpoint = None
        if dataset == 'cf' and batch == 20 and not smoke and not deferred(manifest):
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
        generation_status='DEFERRED_NOT_MEASURED' if deferred(manifest) else 'SCHEDULED',
        completion_scope='EDIT_FACTUAL_CHECKPOINT_ONLY' if deferred(manifest) else 'FULL_SCHEDULE',
        checkpoint_evaluation_consumer_pending=deferred(manifest),
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
    started, tracking, result, error, engine = time.monotonic(), None, None, None, None
    try:
        verify_manifest(manifest)
        noqual.check_mode(manifest, args.mode)
        if noqual.enabled(manifest):
            require(not args.resume, 'NEW_COLD_NO_OLD_PARTIAL_RESUME')
            write_new(out/'qualification-status.json', noqual.profile())
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
                from official.runners.server2.zsre_profile import enabled as zsre_enabled
                if zsre_enabled(manifest):
                    require(args.dataset=='zsre' and not args.resume,'NEW_COLD_ZSRE_PIPELINE_ONLY')
                    if args.mode=='chain':
                        from official.runners.server2.zsre_pipeline import verify_smoke
                        proof=verify_smoke(Path(args.manifest).resolve().parent,manifest)
                        require(read(Path(args.manifest).resolve().parent/'zsre-smoke-verified.json')==proof,
                                'ACTUAL_ZSRE_SMOKE_REQUIRED_BEFORE_CHAIN')
                if deferred(manifest) and not noqual.enabled(manifest):
                    require(args.mode == 'chain' and args.dataset == 'cf' and not args.resume,
                            'NEW_COLD_CHECKPOINT_PIPELINE_ONLY')
                    from official.runners.server2.checkpoint_pipeline import verify_qualification
                    proof = verify_qualification(out.parent/'qualification', manifest, args.method)
                    require(read(out.parent/'qualification-verified.json')['member'] == proof,
                            'ACTUAL_QUALIFICATION_REQUIRED_BEFORE_CHAIN')
                result = chain(model, tok, engine, manifest, records, args.dataset, out, tracking,
                               resume=args.resume, smoke=args.mode == 'smoke')
    except BaseException as exc:
        error = exc
        write_new(out/'failure.json', dict(status='TECHNICAL_FAILURE', exception_type=type(exc).__name__,
            error_code=str(exc)[:160] if isinstance(exc, ValueError) else 'EXCEPTION_DETAILS_LOCAL_STDERR',
            elapsed_seconds=time.monotonic()-started, preserved_source_raw_checkpoint=True,
            completed_native_batches=None if engine is None else engine.batch,
            latest_checkpoint_metadata=failure_checkpoint_metadata(
                Path(args.resume) if args.resume else out/'checkpoints'),
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
