"""Source-pinned two-model S binding, not a new target/writer implementation.

The model is loaded only after launch.validate_launch. All scientific kernels
come from the pinned native-v3.1 packages; seven independent paths share one
cold entry and one stock fixed-z capture. D recovery is not a dependency.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback

from .contracts import BindingBoundary, INSTRUCTION_ID, RUNTIME_HEAD
from .launch import validate_launch
from .publication import digest, member, safe_path, write_once


def bound_dev_records(sample, dataset):
    """Load only sealed S_DEV rows; reserved audit outcomes are never opened."""
    from project.run_scripts.ordered_response_barrier_ode import runtime as old
    from .sampling import canonical_hash, evaluation_inventory
    body = {k: v for k, v in sample.items() if k != 'manifest_identity'}
    if canonical_hash(body) != sample.get('manifest_identity'):
        raise BindingBoundary('SAMPLE_MANIFEST_IDENTITY')
    cohort = sample['cohorts']['S_DEV']
    entries = cohort['records']
    if (len(entries) != 100 or len({r['case_id'] for r in entries}) != 100
            or any(r['fixture'] != 'S_DEV' for r in entries)
            or canonical_hash(entries) != cohort['ordered_root']
            or canonical_hash([r['request_sha256'] for r in entries]) != cohort['request_order_sha256']):
        raise BindingBoundary('S_DEV_ORDER_COUNT_IDENTITY')
    fact = member(dataset)
    if fact['sha256'] != sample['dataset']['sha256'] or fact['bytes'] != sample['dataset']['bytes']:
        raise BindingBoundary('S_DATASET_BYTES_DRIFT')
    raw = old._raw_records(Path(dataset), {int(r['case_id']) for r in entries})
    rows = [raw[int(r['case_id'])] for r in entries]
    for sealed, row in zip(entries, rows, strict=True):
        if (canonical_hash(row) != sealed['raw_record_sha256']
                or canonical_hash(row['requested_rewrite']) != sealed['request_sha256']):
            raise BindingBoundary('S_DEV_RAW_REQUEST_DRIFT')
    if evaluation_inventory(rows) != cohort['evaluation']:
        raise BindingBoundary('S_DEV_EVALUATOR_INPUT_DRIFT')
    return rows, cohort


def first_request_logits(family):
    """Outcome-blind first DEV row, all positions/vocabulary; validation only."""
    request = family.requests[0]
    text = str(request['prompt']).format(str(request['subject']))
    batch = family.tokenizer([text], padding=True, return_tensors='pt').to(family.device)
    return family.model(**batch, use_cache=False).logits.detach().float().cpu()


def _family_type():
    # Delayed import keeps launch rejection entirely ahead of torch/model setup.
    from project.run_scripts.native_response_ode_v31.runtime import ObservedFamily

    class CampaignFamily(ObservedFamily):
        record_node_semantic = True

        def compute_fixed_z(self):
            from project.run_scripts.ordered_response_barrier_ode import runtime as old
            old._sync()
            started, before = time.perf_counter(), old._model_forward_count(self.model)
            value = super().compute_fixed_z()
            old._sync()
            self.target_generation_accounting = dict(wall_seconds=time.perf_counter()-started,
                model_forward_invocations=old._model_forward_count(self.model)-before,
                stock_compute_z_request_count=len(self.requests), cohort_capture_count=1)
            return value

        def _capture_endpoint_for_persistence(self):
            super()._capture_endpoint_for_persistence()
            if self._capture_persistent_endpoint:
                self._captured_endpoint_cache_c_new = bool(self.module.cache_c_new)

        def evaluate_endpoint(self):
            result = super().evaluate_endpoint()
            # Captured while actual endpoint weights are live, including prefix
            # finalize; the parent resets last_terminal after prefix observation.
            self.last_actual_activation = self.last_terminal.detach().clone()
            return result

    return CampaignFamily


def _validate_endpoint_counts(endpoint, cohort, entry=None):
    from project.run_scripts.native_response_ode_v31.analysis import summarize
    facts, _rows, _public = summarize(endpoint, cohort['records'], entry)
    expected = cohort['evaluation']['denominators']
    # Explicit counts come from the input schema, never a hardcoded locality10.
    for label in ('RS', 'PS', 'NS'):
        if facts[label + '_d'] != expected[label]:
            raise BindingBoundary('ENDPOINT_DENOMINATOR_' + label)
    return facts


def run_cell(repo, root, cell):
    premodel = validate_launch(repo, root, cell)
    repo, root = Path(repo), Path(root)
    output = safe_path(root / f'cell-{cell}', root=root)
    output.mkdir(mode=0o700, exist_ok=False)
    save = lambda name, value: write_once(output / name, value, root=output)
    save('premodel.json', premodel)
    started = time.perf_counter()
    stage = 'PREMODEL_ASSETS'
    entry = model = module = None
    completed = []
    component_clock = {}
    compute = []

    def terminate(signum, _frame):
        raise RuntimeError(f'EXTERNAL_RESOURCE_TERMINATION_SIGNAL_{signum}')
    signal.signal(signal.SIGTERM, terminate)

    try:
        import torch
        from project.run_scripts.ordered_response_barrier_ode import runtime as old
        from project.run_scripts.ordered_response_barrier_ode import preflight as assets
        from project.run_scripts.ordered_response_barrier_ode.contracts import assert_full_fp32
        from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256, tensor_sha256
        from project.run_scripts.native_response_ode_v31.runtime import raw_requests
        from .fixture import EntrySnapshot
        from .raw_store import LocalRawStore
        from .gpu_fidelity import run_gpu_fidelity
        from .sweep_driver import run_model_sweep, SweepCallbacks

        sample = json.loads((root / 'sample.lock.json').read_text())
        rows, cohort = bound_dev_records(sample, assets.EASYEDIT_ARTIFACT_ROOT / assets.DATASET_RELATIVE)
        alias = premodel['model_alias']
        pinned_assets = json.loads((root / 'assets.lock.json').read_text())
        if pinned_assets.get('status') != 'PINNED_NATIVE_ASSET_HASH_PASS' or not pinned_assets.get('deep_hash'):
            raise BindingBoundary('PINNED_DEEP_ASSETS_REQUIRED')
        # Pre-submission full hashes bind large assets; verify sizes/source and
        # config/tokenizer again inside the allocated process, no new model call.
        live_assets = assets.validate_model_artifacts(repo, assets.EASYEDIT_ARTIFACT_ROOT,
                                                      assets.HF_HUB_CACHE_ROOT, deep_hash=False)
        expected = pinned_assets['models'][alias]
        observed = live_assets['models'][alias]
        for key in ('revision', 'snapshot_config_sha256', 'snapshot_tokenizer_sha256'):
            if observed[key] != expected[key]:
                raise BindingBoundary('LIVE_ASSET_IDENTITY_' + key)
        official = old._bootstrap_easyedit(assets.OFFICIAL_EASYEDIT_ROOT)
        if official['head'] != assets.OFFICIAL_EASYEDIT_HEAD or not official['tracked_clean']:
            raise BindingBoundary('EASYEDIT_SOURCE_DRIFT')
        source = json.loads((root / 'source.lock.json').read_text())
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                          TOKENIZERS_PARALLELISM='false', WANDB_DISABLED='true')
        random.seed(20260906)
        torch.manual_seed(20260906)
        if torch.cuda.device_count() != 1:
            raise BindingBoundary('SINGLE_GPU_BOUNDARY')
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(20260906)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision('highest')
        stage = 'MODEL_LOAD'
        load_start = time.perf_counter()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        snapshot = old._model_snapshot(assets.HF_HUB_CACHE_ROOT, alias)
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True,
            device_map={'': 'cuda:0'}, attn_implementation='eager')
        tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                            trust_remote_code=False, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        tok.padding_side = 'right'
        model.config.pad_token_id = tok.pad_token_id
        model.config.use_cache = False
        model.eval()
        old.seal_eager_attention(model)
        old._install_model_forward_counter(model)
        dtype = assert_full_fp32(model)
        old._sync()
        load_seconds = time.perf_counter() - load_start
        hp, hp_path = old._load_hparams(repo, 'AlphaEdit', alias)
        hp.device = 0
        hp.stats_dir = str(assets.EASYEDIT_ARTIFACT_ROOT / 'examples/data/stats')
        hp.P_loc = str(assets.EASYEDIT_ARTIFACT_ROOT / assets.MODEL_BINDINGS[alias]['projector'][0])
        module = old._method_module('AlphaEdit')
        module.CONTEXT_TEMPLATES_CACHE = None
        with old._model_name(model, str(hp.model_name)):
            contexts = module.get_context_templates(model, tok)
        save('runtime.lock.json', dict(model_alias=alias, family='AlphaEdit', source=source,
            official=official, hf_snapshot=str(snapshot), hparams=member(hp_path),
            dtype=dtype, parameter_inventory=old._parameter_inventory(model),
            contexts_sha256=assets.canonical_hash(contexts), sample_root=cohort['ordered_root'],
            request_order_sha256=cohort['request_order_sha256'],
            cuda_device=torch.cuda.get_device_name(0), cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'), array_task=premodel['task_id'],
            load_seconds=load_seconds, model_load_count=1, autocast=False, tf32=False,
            controller_dtype='float64', model_storage_forward_dtype='float32',
            runtime_parent=RUNTIME_HEAD, independent_cold_S_DEV=True, D_restore_dependency=False,
            FD_logits_scope='SEALED_FIRST_S_DEV_REQUEST_ALL_POSITIONS_FULL_VOCAB',
            FD_raw_activation_scope='ALL_S_DEV_REQUESTS',
            timing='Leading CUDA sync excluded; trailing wait included; entire residency charged',
            per_node_training_predicate_observation=True, predicate_dynamics_influence=0,
            audit_evaluation_count=0, scientific_promotion=False))

        family_type = _family_type()
        kwargs = dict(tokenizer=tok, family='AlphaEdit', hparams=hp,
                      requests=old._official_requests(raw_requests(rows)), endpoint_records=rows,
                      request_order_sha256=cohort['request_order_sha256'], contexts=contexts,
                      capture_persistent_endpoint=True)
        stage = 'COLD_ENTRY'
        cold = family_type(model=model, module=module, **kwargs)
        cold.prepare_method_state()
        entry = EntrySnapshot.capture(cold.parameters, module)
        del cold
        store = LocalRawStore(output / 'local-raw', contexts=contexts,
            source_identity=dict(head=source['head'], tree=source['tree'], model_alias=alias,
                sample_root=cohort['ordered_root'], order=cohort['request_order_sha256']))

        def component_sink(identity, payload):
            nonlocal stage
            component, phase = identity['component'], identity['phase']
            old._sync()
            now, forwards = time.perf_counter(), old._model_forward_count(model)
            stage = f'{component}/{phase}'
            if phase == 'begin':
                component_clock[component] = (now, forwards)
            else:
                start, before = component_clock.pop(component)
                fact = dict(component=component, wall_seconds=now-start,
                            model_forward_invocations=forwards-before,
                            includes_nested_components=component == 'fixed_target_and_entry',
                            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved())
                compute.append(fact)
                save(f'components/{component}.json', dict(identity=identity, accounting=fact, payload=payload))
                if component == 'first_fidelity':
                    save('first-valid-fidelity.json', dict(status='PASS', cell=cell, model=alias,
                        actual_GPU_fidelity=True, W0_restore=True, fixed_z_recompute_count=0,
                        source_head=source['head'], request_order_sha256=cohort['request_order_sha256']))
                elif component == 'fixed_target_and_entry':
                    facts = _validate_endpoint_counts(payload['entry_evaluation'], cohort)
                    save('entry-metrics.json', facts)
                elif component == 'O_NATIVE':
                    save('O_NATIVE-metrics.json', _validate_endpoint_counts(payload['endpoint']['evaluation'], cohort))
                    completed.append(component)
                else:
                    for label, endpoint in payload['result']['endpoints'].items():
                        facts = _validate_endpoint_counts(endpoint['evaluation'], cohort)
                        save(f'metrics/{label}.json', dict(candidate_id=label, **facts))
                        completed.append(label)
                save(f'progress/{len(compute):02}.json', dict(stage=stage, completed=completed.copy(),
                    process_residency_seconds=time.perf_counter()-started,
                    source_head=source['head'], scientific_promotion=False))

        callbacks = SweepCallbacks(
            first_fidelity=lambda family, dictionary, normalization: run_gpu_fidelity(
                family, dictionary, normalization, output / 'fidelity',
                logits_observer=first_request_logits,
                logits_observer_identity='SEALED_FIRST_S_DEV_REQUEST_ALL_POSITIONS_FULL_VOCAB'),
            fixed_target_sink=store.fixed_target, component_sink=component_sink)
        result = run_model_sweep(model_alias=alias, model=model, module=module,
            entry_snapshot=entry, family_factory=family_type, family_kwargs=kwargs,
            output=output / 'paths', callbacks=callbacks, raw_sink=store.node,
            endpoint_sink=store.endpoint)
        stage = 'TERMINAL'
        save('sweep-result.json', result)
        save('compute-accounting.json', dict(model_load_seconds=load_seconds, components=compute,
            model_forward_invocations=old._model_forward_count(model),
            process_residency_seconds=time.perf_counter()-started,
            scheduler_GPU_seconds='AUTHORITATIVE_SACCT_RECONCILIATION_AFTER_EXIT',
            summed_nested_components_are_not_total=True))
        save('terminal.json', dict(status='S_NINE_ENDPOINTS_COMPLETE', instruction_id=INSTRUCTION_ID,
            source_head=source['head'], source_tree=source['tree'], model_alias=alias,
            JV_endpoints=9, actual_paths=7, completed_nodes=34, official_endpoints=1,
            requests_per_endpoint=100, completed=completed,
            S_DEV_order_sha256=cohort['request_order_sha256'], entry_W_sha256=entry.W_sha256,
            entry_M_sha256=entry.M_sha256, W0_restore=tensor_set_sha256({k:model.get_parameter(k) for k in entry.weights}) == entry.W_sha256,
            M_restore=tensor_sha256(module.cache_c) == entry.M_sha256,
            fixed_z_capture_count=1, fixed_z_recompute_count=0, qref_capture_count=1,
            raw_journals='STORED_NOT_RECONSTRUCTED', audit_outcomes_opened=0,
            D_status='HISTORICAL_EXACT_STATE_UNAVAILABLE_ON_SERVER1_NO_RECONSTRUCTION',
            source_science_change_count=0, tolerance_change_count=0, scientific_promotion=False))
    except BaseException as exc:
        restore_error = None
        if entry is not None:
            try:
                entry.restore(model, module)
            except BaseException as restore:
                restore_error = repr(restore)
        restore_fact = None
        if entry is not None:
            restore_fact = dict(W0_bytes=tensor_set_sha256({k:model.get_parameter(k) for k in entry.weights}) == entry.W_sha256,
                                M_bytes=tensor_sha256(module.cache_c) == entry.M_sha256,
                                cache_flag=bool(module.cache_c_new) == entry.cache_c_new)
        save('failure-boundary.json', dict(status=getattr(exc, 'status', 'TECHNICAL_RUNTIME_BOUNDARY'),
            stage=stage, exception_type=type(exc).__name__, exception=str(exc),
            original_exception_receipt=getattr(exc, 'receipt', None), traceback=traceback.format_exc(),
            completed=completed, restore_error=restore_error, entry_restore=restore_fact,
            process_residency_seconds=time.perf_counter()-started,
            science_change_count=0, tolerance_change_count=0, automatic_retry_count=0,
            scientific_promotion=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--cell', type=int, required=True)
    args = parser.parse_args()
    run_cell(args.repo, args.run_root, args.cell)


if __name__ == '__main__':
    main()
