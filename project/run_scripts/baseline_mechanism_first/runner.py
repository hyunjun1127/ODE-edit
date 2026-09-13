"""One source-exact cold singleton batch; no E2/controller or automatic cascade."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

from .contracts import ContractBoundary, digest, file_sha, member, save


def tensor_artifact(path, value):
    import torch
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + '.partial-' + str(os.getpid()))
    with temporary.open('xb') as handle:
        torch.save(value, handle); handle.flush(); os.fsync(handle.fileno())
    os.link(temporary, p)  # Atomic, refuses an existing endpoint.
    temporary.unlink()
    return member(p)


def run(lock_path, output):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    from .assets import bind_native
    from .evaluation import bind_evaluation_sources, diagnostic_margin, evaluate_records
    from .fixtures import (FixtureTransaction, SingletonSpec, capture_rng, restore_rng,
                           select_projector, tensor_sha)
    from .native_runner import run_native_batch
    from .observer import observe_native
    from .signed_response import AllPositionContraction, finite_difference_audit

    output = Path(output).absolute(); output.mkdir(parents=True, exist_ok=False)
    lock = json.loads(Path(lock_path).read_text())
    stage = 'SOURCE_BINDING'; started = time.monotonic(); component = []
    tx = None; forward_handle = None; forward_counts = {}
    count = {'native_batch': 0, 'diagnostic_forward': 0, 'evaluation_endpoint': 0}
    try:
        for row in lock['source_members']:
            # Models were fully hashed by CPU preflight. Re-read all bytes here
            # as well: no inferred HF cache hit or shared-memory model reuse.
            member(row['path'], expected=row['sha256'])
        if set(lock.get('companion_members', {})) != set(lock['companions']):
            raise ContractBoundary('COMPANION_LOCK_INCOMPLETE')
        for name, row in lock['companion_members'].items():
            if row['path'] != lock['companions'][name]:
                raise ContractBoundary('COMPANION_PATH_DRIFT', name=name)
            member(row['path'], expected=row['sha256'])
        if transformers.__version__ != lock['transformers'] or torch.__version__ != lock['torch']:
            raise ContractBoundary('DEPENDENCY_IDENTITY', torch=torch.__version__, transformers=transformers.__version__)
        if lock['cell']['layer'] != 4 or lock['cell']['entry_n'] != 0:
            raise ContractBoundary('FIRST_WAVE_COLD_L4_ONLY')
        records = load_prefix(lock['dataset_root'], 100)
        if [r['case_id'] for r in records] != [r['case_id'] for r in lock['records']]:
            raise ContractBoundary('SAMPLE_ORDER')
        native, hp_type = bind_native(lock['native_root'])
        hp = hp_type.from_json(lock['config']); spec = SingletonSpec(4)
        bind_evaluation_sources(lock['historical_root'], helper_root=lock['helper_root'])
        random.seed(lock['seed']); np.random.seed(lock['seed']); torch.manual_seed(lock['seed'])
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = lock['tf32_matmul']
        torch.backends.cudnn.allow_tf32 = lock['tf32_cudnn']
        stage = 'MODEL_LOAD'; then = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(lock['snapshot'], local_files_only=True,
                    low_cpu_mem_usage=True, attn_implementation='eager').cuda().eval()
        tok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        tok.add_bos_token = False; tok.pad_token_id = tok.eos_token_id
        evaltok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        evaltok.pad_token_id = evaltok.eos_token_id
        if tok.padding_side != 'right' or evaltok.padding_side != 'right' or any(p.dtype != torch.float32 for p in model.parameters()):
            raise ContractBoundary('MODEL_TOKENIZER_DTYPE')
        component.append(dict(component='model_load', wall_seconds=time.monotonic()-then))
        def count_forward(_module, args, kwargs):
            begin = time.monotonic()
            ids = kwargs.get('input_ids', args[0] if args else None)
            mask = kwargs.get('attention_mask')
            row = forward_counts.setdefault(stage,dict(calls=0,input_positions=0,
                nonpadding_positions=0,attention_mask_missing_calls=0,counter_wall_seconds=0.))
            row['calls'] += 1
            if ids is not None: row['input_positions'] += ids.numel()
            if mask is not None: row['nonpadding_positions'] += int(mask.sum())
            else: row['attention_mask_missing_calls'] += 1
            row['counter_wall_seconds'] += time.monotonic()-begin
            return None
        forward_handle = model.register_forward_pre_hook(count_forward, with_kwargs=True)
        weights = dict(model.named_parameters()); weight = weights[spec.weight_name]
        w0 = weight.detach().cpu().clone()
        stack = torch.load(lock['projector'], map_location='cpu', weights_only=True, mmap=True)
        projector, mapping = select_projector(stack, spec); del stack
        history = torch.zeros_like(projector)
        original_entry = json.loads(Path(lock['companions']['entry.json']).read_text())
        reference_w0 = original_entry['signature']['weights'][spec.weight_name]['sha256']
        if tensor_sha(weight) != reference_w0 or tensor_sha(history) != original_entry['signature']['cache_sha256']:
            raise ContractBoundary('W0_HISTORY_REFERENCE_BYTES')
        save(output/'runtime.json', dict(input_lock=member(lock_path), source_head=subprocess.check_output(
            ['git','rev-parse','HEAD'], text=True).strip(), model_revision=lock['model_revision'],
            model_dtype='torch.float32', attention=model.config._attn_implementation,
            torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
            writer_tokenizer=dict(padding=tok.padding_side, add_bos_token=tok.add_bos_token, pad=tok.pad_token_id),
            evaluator_tokenizer=dict(padding=evaltok.padding_side, add_bos_token=getattr(evaltok,'add_bos_token','NOT_EXPOSED'),pad=evaltok.pad_token_id),
            projector_mapping=mapping, source_init_seed=lock['seed'], source_W0_bytes_exact=True,
            W0_RNG_reference='NOT_RECORDED_SOURCE_INIT_ONLY', slurm_job=os.environ.get('SLURM_JOB_ID')))
        tx = FixtureTransaction(model, native, history)
        with tx:
            stage = 'ORIGINAL_CONTEXT_GENERATION'
            if native.CONTEXT_TEMPLATES_CACHE is not None:
                raise ContractBoundary('NONCOLD_NATIVE_CONTEXT_CACHE')
            contexts = native.get_context_templates(model, tok)
            reference_contexts = json.loads(Path(lock['companions']['contexts.json']).read_text())
            save(output/'context-reproduction.json', dict(actual=contexts,
                 actual_sha256=digest(contexts), reference_sha256=digest(reference_contexts),
                 exact=contexts==reference_contexts, source_generation_called_once=True,
                 original_rng_consumption='ORIGINAL_GENERATION_EXECUTED_NOT_CACHED_INJECTION'))
            if contexts != reference_contexts:
                raise ContractBoundary('COLD_CONTEXT_REPRODUCTION_UNRESOLVED')
            stage = 'NATIVE_B100'; then = time.monotonic()
            requests = [dict(r['requested_rewrite'], case_id=int(r['case_id'])) for r in records]
            def event(name, row):
                save(output/'progress'/f'{name}.json', dict(stage=name, observed_utc=time.time(), **row))
                print('E01_PROGRESS', name, flush=True)
            result = run_native_batch(model, tok, native, hp, history, projector, requests, spec,
                                      observer=observe_native, on_stage=event)
            count['native_batch'] += 1
            torch.cuda.synchronize(); component.append(dict(component='native_B100', wall_seconds=time.monotonic()-then))
            native_rng = capture_rng()
            raw = result['observer'].pop('_tensors')
            raw_receipt = tensor_artifact(output/'native-tensors.pt', raw)
            target_reference = torch.load(lock['companions']['native-targets.pt'], map_location='cpu', weights_only=False)
            actual_targets = raw['targets']
            target_compare = dict(status='OBSERVED_RECOMPUTATION_NOT_CACHED', count=len(actual_targets),
                reference_count=len(target_reference['values']), comparison='SHAPES_AND_FP32_DIFFERENCE')
            reference_values = target_reference['values']
            if len(actual_targets) == len(reference_values):
                target_compare['rows'] = [dict(index=i, exact=torch.equal(a,b), max_abs=float((a-b).abs().max()),
                    difference_norm=float((a.double()-b.double()).norm())) for i,(a,b) in enumerate(zip(actual_targets,reference_values))]
            save(output/'target-reproduction.json', target_compare)
            endpoint = result['weight']; endpoint_history = result['history']
            endpoint_receipt = tensor_artifact(output/'native-endpoint.pt', dict(weights={spec.weight_name:endpoint},
                cache_c=endpoint_history, metadata=dict(batch=1, seen_ids=[r['case_id'] for r in records],
                base_model_revision=lock['model_revision'],contexts=deepcopy(contexts),rng=native_rng,
                state=dict(weights={spec.weight_name:tensor_sha(endpoint)},cache=tensor_sha(endpoint_history)),
                covariance={}, method='AlphaEdit_BLUE_SINGLETON',cache_c_is_history=True,
                original_checkpoint_equivalence='NOT_YET_VERIFIED')))
            save(output/'native-observation.json', dict(receipt=result['receipt'], observer=result['observer'],
                raw=raw_receipt, endpoint=endpoint_receipt))
            del raw, target_reference, actual_targets, reference_values
            stage = 'SIGNED_PROBE_AND_SELECTED_RESTORE'; then = time.monotonic()
            model.requires_grad_(False)
            with torch.no_grad(): weight.copy_(w0.to(weight)); history.zero_()
            if tensor_sha(weight) != reference_w0 or tensor_sha(history) != original_entry['signature']['cache_sha256']:
                raise ContractBoundary('SELECTED_W0_RESTORE')
            # Actual endpoint operand difference, not replacement low-rank writer.
            delta64 = endpoint.double()-w0.double(); direction = delta64.float().to(weight.device)
            rounding = float((direction.cpu().double()-delta64).norm()); del delta64
            module = model.get_submodule(hp.rewrite_module_tmp.format(4))
            with AllPositionContraction(module, direction) as capture:
                scalar = diagnostic_margin(model, evaltok, records[0]); count['diagnostic_forward'] += 1
                contraction = capture.compute(scalar)
            alpha = lock['diagnostics']['fd_alpha']
            with torch.no_grad():
                zero = float(diagnostic_margin(model,evaltok,records[0])); repeated = float(diagnostic_margin(model,evaltok,records[0]))
                values = []
                try:
                    for sign in (-1,1):
                        weight.copy_(w0.to(weight) + (sign*alpha)*direction)
                        values.append(float(diagnostic_margin(model,evaltok,records[0])))
                finally:
                    weight.copy_(w0.to(weight))
                count['diagnostic_forward'] += 4
            noise = abs(repeated-zero)
            absolute = (8*torch.finfo(torch.float32).eps*max(1.,abs(zero)) + noise)/alpha
            fd = finite_difference_audit(contraction['event_derivative'],values[0],zero,values[1],
                    alpha=alpha,forward_noise=noise,absolute_tolerance=absolute,
                    relative_tolerance=lock['diagnostics']['fd_relative_tolerance'])
            save(output/'signed-initial.json',dict(contraction=contraction, finite_difference=fd,
                physical_endpoint_difference_FP32_rounding_norm=rounding, observer_role='DIAGNOSTIC_ONLY',
                target_identity=digest(records[0]), no_alpha_performance_selection=True))
            if fd['status'] in {'NONFINITE','DERIVATIVE_MISMATCH'}:
                raise ContractBoundary('SIGNED_DERIVATIVE_TECHNICAL_BOUNDARY', fd=fd)
            if tensor_sha(weight) != reference_w0:
                raise ContractBoundary('FD_SELECTED_RESTORE')
            restore_rng(native_rng)
            torch.cuda.synchronize(); component.append(dict(component='signed_initial',wall_seconds=time.monotonic()-then))
            save(output/'INITIAL_VALID.json',dict(status='INITIAL_VALID', native_batches=1,
                source_W0_bytes_exact=True, original_contexts_exact=True, native_endpoint_finite=True,
                history_append=result['observer'].get('history_append_passes'),
                selected_W0_restore_exact=True, nonselected_native_bytes_versions_exact=True,
                signed_validation=fd['status'], derivative_unresolved_limits_claim_only=fd['status']=='NUMERICALLY_UNRESOLVED',
                full_E0_continuation_or_20cell_complete=False, checkpoint_fidelity='REFERENCE_NOT_YET_RECEIVED',
                agent_policy='MONITORING_PAUSED_AWAITING_USER', elapsed_seconds=time.monotonic()-started))
            print('E01_INITIAL_VALID', flush=True)
            del direction, scalar
            # Already sealed single-batch program may finish after the agent pauses.
            for label, value, cache in [('W0', w0, torch.zeros_like(endpoint_history)),
                                         ('NATIVE', endpoint, endpoint_history)]:
                stage = 'EVALUATION_'+label; then = time.monotonic()
                with torch.no_grad(): weight.copy_(value.to(weight)); history.copy_(cache)
                before = tensor_sha(weight),tensor_sha(history)
                metrics = evaluate_records(model,evaltok,records)
                if before != (tensor_sha(weight),tensor_sha(history)):
                    raise ContractBoundary('EVALUATOR_MUTATION')
                save(output/(label+'-current.json'), metrics); count['evaluation_endpoint'] += 1
                torch.cuda.synchronize(); component.append(dict(component='evaluation_'+label,wall_seconds=time.monotonic()-then))
        save(output/'restore.json',tx.receipt)
        save(output/'terminal.json',dict(status='COLD_L4_FIRST_BATCH_FINITE_OBSERVED',counts=count,
            checkpoint_equivalence='NOT_YET_VERIFIED', remaining_E01_cells=19, general='NOT_YET_MEASURED',
            all_E01_complete=False, restore=tx.receipt, elapsed_seconds=time.monotonic()-started,
            components=component, forward_counts=forward_counts,
            backward_observation=dict(signed_backward_calls=1,native_target_backward_calls='NOT_OBSERVED'),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), scientific_promotion=False))
    except BaseException as exc:
        save(output/'failure.json',dict(status='TECHNICAL_OR_INPUT_BOUNDARY',stage=stage,error=repr(exc),
            original_receipt=getattr(exc,'receipt',{}), traceback=traceback.format_exc(), counts=count,
            restore=None if tx is None else tx.receipt, elapsed_seconds=time.monotonic()-started,
            components=component, forward_counts=forward_counts,
            accepted_E0_equivalence_denominator=0, scientific_promotion=False))
        raise
    finally:
        if forward_handle is not None: forward_handle.remove()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--lock', required=True); p.add_argument('--output', required=True)
    a=p.parse_args(); run(a.lock,a.output)
