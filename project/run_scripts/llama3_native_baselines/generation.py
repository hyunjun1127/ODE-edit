"""Task caller of SH1's immutable shared generator; no alternate generator.

Generation state is the actual full-five-site weight identity. Native histories
are guarded separately because cold scalar observations do not resume an editor.
Raw texts/tokens and endpoint rows remain exclusively in ignored task-local paths.
"""
import copy
import json
from pathlib import Path

from .common import (METHODS, ORDERED_SHA, PROFILE, digest, member, require,
                     sha, verify, write)
from .producer import REASONS


def normalize_summary(summary):
    """Normalize the shared reducer's sums, never score missing values as zero."""
    reasons = summary['missing_reason_counts']
    require(set(reasons) == set(REASONS) | {'asset_not_available', 'tokenizer_not_available'},
            'GENERATION_SHARED_REASON_SCHEMA')
    require(reasons['asset_not_available'] == reasons['tokenizer_not_available'] == 0,
            'GENERATION_READY_ASSET_RUNTIME_LOST')
    result = {key: summary[key] for key in ('planned_count', 'fluency_count',
        'consistency_count', 'fluency_sum', 'consistency_sum',
        'generation_prompt_count', 'generated_token_count')}
    result['reason_counts'] = {key: reasons[key] for key in REASONS}
    return result


def endpoint_ref(observed):
    """The shared immutable endpoint contains ordered row and raw-file hashes."""
    return dict(member=member(observed['rows_path']), identity=observed['identity'],
        identity_sha256=observed['identity_sha256'], summary=observed['summary'],
        work=observed['work'])


def read_observed(ref, config=None, *, observer=None):
    """CPU read only; validation requires raw rows, never just aggregate metrics.

    With no observer this imports SH1's reader, not its generator. The shared
    reader checks every referenced per-case immutable raw payload and reduction.
    Exact endpoint/model/profile/source/reference runtime is additionally bound
    below. CPU collectors may use this function without a model or GPU.
    """
    from project.run_scripts.experiment_generation_eval.observer import read_observed as shared_read
    row = ref['member'] if 'member' in ref else ref
    path = verify(row)
    value = observer.read_observed(path) if observer is not None else shared_read(path)
    if 'identity' in ref:
        require(value['identity'] == ref['identity'] and
                value['identity_sha256'] == ref['identity_sha256'] and
                value['summary'] == ref['summary'], 'GENERATION_REFERENCE_IDENTITY')
    if config is not None:
        generation = config['generation']
        runtime = runtime_identity(config)
        require(value['identity']['runtime'] == digest(runtime),
                'GENERATION_REFERENCE_RUNTIME_BINDING')
        require(runtime['profile'] == PROFILE and runtime['eval_seed'] == 20261007 and
                runtime['generation_source_sha'] == generation['source_sha'] and
                runtime['reference_assets_sha256'] == generation['assets_sha256'],
                'GENERATION_REFERENCE_SOURCE_ASSET_BINDING')
    normalize_summary(value['summary'])
    return value


def runtime_identity(config):
    return dict(schema='counterfact-cake-generation-metrics-v1', profile=PROFILE,
        eval_seed=20261007, model_identity=model_identity(config),
        generation_source_sha=config['generation']['source_sha'],
        reference_assets_sha256=config['generation']['assets_sha256'],
        route='UNPADDED_FULL_PREFIX_NO_CACHE')


def model_identity(config):
    # Method/job/attempt are intentionally not part of a paired generation seed.
    return dict(model='llama3', revision=config['model_revision'],
                observation_identity=config['observation_identity'])


def require_binding(config):
    generation = config['generation']
    require(generation['profile'] == PROFILE and generation['W0_producer'] == 'MEMIT',
            'GENERATION_PROFILE_W0_OWNER')
    require(len(generation['source_sha']) == 40 and len(generation['assets_sha256']) == 64,
            'GENERATION_SOURCE_ASSET_SHA')
    require(bool(generation['source_members']), 'GENERATION_SOURCE_CLOSURE_MISSING')
    for row in generation['source_members']:
        verify(row)
    ready = json.loads(verify(generation['READY']).read_text())
    require(ready['status'] == 'READY' and
            ready['identity_sha256'] == generation['assets_sha256'],
            'GENERATION_REFERENCE_READY')
    manifest = json.loads(verify(generation['reference_manifest']).read_text())
    require(manifest['status'] == 'READY' and
            manifest['identity_sha256'] == generation['assets_sha256'],
            'GENERATION_REFERENCE_MANIFEST_READY')
    require(ready['manifest']['sha256'] == generation['reference_manifest']['sha256'],
            'GENERATION_READY_MANIFEST_JOIN')
    return generation


class NativeGeneration:
    """All endpoints/subsets share SH1's one authoritative observation cache."""
    def __init__(self, config, view, engine, bench, records, out, cursor):
        from project.run_scripts.experiment_generation_eval.assets import load_assets
        from project.run_scripts.experiment_generation_eval.observer import GenerationObserver
        generation = require_binding(config)
        assets = load_assets(dict(reference_assets=generation['reference_manifest'],
                                 asset_paths=generation['asset_paths']))
        require(assets.sha == generation['assets_sha256'], 'GENERATION_LOADED_ASSET_IDENTITY')
        require(len(records) == 2000 and digest([r['case_id'] for r in records]) == ORDERED_SHA,
                'GENERATION_FULL_OCCURRENCE_ORDER')
        require(len({r['case_id'] for r in records}) == 2000,
                'GENERATION_UNIQUE_CASE_MAPPING_REQUIRED')
        self.config, self.out, self.records = config, Path(out), records
        self.view, self.engine, self.bench, self.cursor = view, engine, bench, cursor
        self.W = copy.deepcopy(config['cold_W'])
        def external():
            return dict(history={str(layer): dict(pointer=value.data_ptr(),
                version=value._version, shape=list(value.shape), dtype=str(value.dtype))
                for layer, value in engine.history().items()},
                native_context=digest(engine.contexts()), context=digest(bench.contexts),
                cursor=digest(cursor), next_batch=engine.next_batch)
        self.observer = GenerationObserver(view.model, bench.tokenizer, assets,
            dict(model_identity=model_identity(config), profile=PROFILE, eval_seed=20261007,
                generation_source_sha=generation['source_sha'],
                occurrence_by_case_id={str(r['case_id']): index for index, r in enumerate(records)}),
            self.out, state_callback=external)
        require(self.observer.runtime_identity == runtime_identity(config),
                'GENERATION_ACTUAL_RUNTIME_IDENTITY')

    def set_weights(self, state):
        require(set(state['W']) <= set(self.W), 'GENERATION_SELECTED_PHYSICAL_WEIGHTS')
        self.W.update(state['W'])

    def weights_identity(self):
        return dict(W=self.W, model_identity=model_identity(self.config))

    def observe(self, records, endpoint, cohort=None):
        value = self.observer.observe(records, endpoint, cohort=cohort,
                                      state_identity=self.weights_identity())
        normalize_summary(value['summary'])
        return value

    def subset(self, observed, records, endpoint, cohort=None):
        value = self.observer.subset(observed, records, endpoint, cohort=cohort)
        normalize_summary(value['summary'])
        return value

    def W0(self, method, source_commit, config_sha):
        generation = self.config['generation']
        ready_path = Path(generation['W0_ready_path'])
        require(self.W == self.config['cold_W'], 'GENERATION_W0_ACTUAL_COLD_WEIGHT_IDENTITY')
        if method == generation['W0_producer']:
            require(not ready_path.exists(), 'GENERATION_W0_ALREADY_PRODUCED_NO_DUPLICATE')
            value = self.observe(self.records, 'W0_first2000', cohort='first2000')
            write(ready_path, dict(schema='native-baseline-generation-W0-ready-v1', status='READY',
                ordered_ids_sha256=ORDERED_SHA, requests=2000,
                runtime=self.observer.runtime_identity, runtime_sha256=self.observer.runtime_sha,
                cold_W=self.W, endpoint=endpoint_ref(value), source_commit=source_commit,
                config_sha256=config_sha, editor_history_not_shared=True,
                native_history_identity_separately_guarded=True, raw_local_only=True,
                checkpoint_saved=False))
            return value
        require(method in METHODS and ready_path.is_file(), 'GENERATION_W0_READY_NOT_AVAILABLE')
        ready = json.loads(ready_path.read_text())
        require(ready['status'] == 'READY' and ready['requests'] == 2000 and
                ready['ordered_ids_sha256'] == ORDERED_SHA and ready['cold_W'] == self.W and
                ready['runtime'] == self.observer.runtime_identity and
                ready['runtime_sha256'] == self.observer.runtime_sha and
                ready['source_commit'] == source_commit and ready['config_sha256'] == config_sha and
                ready['editor_history_not_shared'] is True,
                'GENERATION_W0_IDENTITY_MISMATCH')
        value = read_observed(ready['endpoint'], self.config, observer=self.observer)
        require(value['identity']['ordered_occurrences'] == list(range(2000)),
                'GENERATION_W0_COMPLETE_ORDER')
        write(self.out/'W0-reuse.json', dict(ready=member(ready_path), endpoint=ready['endpoint'],
            full_five_actual_cold_W=self.W, native_history_not_resumed=True,
            new_generation_forwards=0, raw_copied=False))
        return value
