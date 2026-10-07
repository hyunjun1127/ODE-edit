"""Generation and scoring share one local observation, with nonmutation guards."""
import copy
import json
from pathlib import Path
import time

import torch

from .common import SCHEMA, PROFILE, EVAL_SEED, digest, immutable_write, require
from .generator import generate_row, isolated_rng, rng_snapshot, rng_equal
from .metrics import score_case, reduce_cases


def _cache_signature(value, depth=0):
    """Inspect only structure/tensor versions, not cache data or secret strings."""
    if torch.is_tensor(value):
        return ('tensor', id(value), value.data_ptr(), value._version,
                tuple(value.shape), str(value.dtype), str(value.device))
    if value is None or isinstance(value, (bool, int, float)):
        return (type(value).__name__, value)
    if depth < 4 and isinstance(value, dict):
        return ('dict', id(value), tuple((str(k), _cache_signature(v, depth+1))
                                         for k, v in value.items()))
    if depth < 4 and isinstance(value, (list, tuple)):
        return (type(value).__name__, id(value), tuple(_cache_signature(v, depth+1) for v in value))
    if depth < 2 and hasattr(value, '__dict__'):
        return (type(value).__name__, id(value), _cache_signature(vars(value), depth+1))
    return (type(value).__name__, id(value))


def read_observed(path, expected_runtime=None):
    """Load an immutable local endpoint; never accepts a summary without rows."""
    value = json.loads(Path(path).read_text())
    require(value['identity_sha256'] == digest(value['identity']), 'GENERATION_ENDPOINT_IDENTITY')
    require(value['RNG_restored'] is True and value['observer_no_mutation'] is True,
            'GENERATION_ENDPOINT_OBSERVER_GUARD')
    if expected_runtime is not None:
        require(value['identity']['runtime'] == expected_runtime, 'GENERATION_RUNTIME_IDENTITY')
    rows = value['rows']
    require(value['identity']['ordered_occurrences'] == [r['occurrence'] for r in rows]
        and len({r['occurrence'] for r in rows}) == len(rows)
        and value['identity']['observation_identities'] == [r['identity_sha256'] for r in rows],
        'GENERATION_ENDPOINT_ROWS_IDENTITY')
    require(value['summary'] == reduce_cases(rows), 'GENERATION_ENDPOINT_REDUCTION')
    for row in rows:
        raw = json.loads(Path(row['observation_path']).read_text())
        require(raw['identity_sha256'] == row['identity_sha256'] == digest(raw['identity'])
            and raw['identity']['runtime'] == value['identity']['runtime']
            and digest(raw['identity']['state_identity']) == value['identity']['state_sha256']
            and raw['payload_sha256'] == row['payload_sha256']
            and raw['payload_sha256'] == digest({k:v for k,v in raw.items() if k!='payload_sha256'})
            and raw['occurrence'] == row['occurrence'] and raw['case_id'] == row['case_id']
            and raw['identity']['record_identity']['ordered_occurrence'] == row['occurrence']
            and raw['identity']['record_identity']['case_id'] == row['case_id']
            and raw['metrics'] == row['metrics'], 'GENERATION_OBSERVATION_BYTES_IDENTITY')
    return dict(**value, rows_path=str(Path(path).resolve()), work=dict(new_case_observations=0,
        cached_case_observations=len(rows), generation_forwards=0, full_prefix_token_work=0, seconds=0.0))


def model_signature(model):
    """No giant weight copy/hash: versions/pointers plus all hooks/training modes."""
    params = [(n, id(p), p.data_ptr(), p._version, tuple(p.shape), str(p.dtype), str(p.device),
               _cache_signature(p.grad))
              for n, p in model.named_parameters()]
    buffers = [(n, id(p), p.data_ptr(), p._version, tuple(p.shape), str(p.dtype), str(p.device))
               for n, p in model.named_buffers()]
    modules = []
    for name, module in model.named_modules():
        hooks = []
        for field in ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks',
                      '_forward_hooks_with_kwargs', '_forward_pre_hooks_with_kwargs',
                      '_forward_hooks_always_called'):
            values = getattr(module, field, {})
            hooks.append((field, tuple((str(k), id(v)) for k, v in values.items())))
        caches = []
        for field in ('past_key_values', '_past_key_values', '_cache', '_kv_cache'):
            if hasattr(module, field):
                value = getattr(module, field)
                caches.append((field, _cache_signature(value)))
        modules.append((name, bool(module.training), tuple(hooks), tuple(caches)))
    return dict(parameters=params, buffers=buffers, modules=modules,
        config_use_cache=getattr(getattr(model, 'config', None), 'use_cache', None))


def occurrence(record, mapping=None):
    values = [record[field] for field in ('ordered_occurrence', 'occurrence_index', 'occurrence', 'ordinal')
              if field in record]
    case = str(record.get('case_id'))
    if mapping is not None and case in mapping:
        values.append(mapping[case])
    require(values and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in values)
            and len(set(values)) == 1, 'GENERATION_ORDERED_OCCURRENCE_REQUIRED')
    return values[0]


def _record_identity(record, ordinal):
    rewrite = record.get('requested_rewrite', {})
    target = rewrite.get('target_new', {})
    prompts = record.get('generation_prompts', [])
    require(isinstance(prompts, list) and all(isinstance(x, str) for x in prompts),
            'GENERATION_PROMPTS_SCHEMA')
    require(isinstance(target, dict), 'GENERATION_TARGET_SCHEMA')
    return dict(ordered_occurrence=ordinal, case_id=record.get('case_id'),
        generation_prompts=prompts, relation_id=rewrite.get('relation_id'),
        target_new_id=target.get('id'))


class GenerationObserver:
    """The caller supplies exact state identity and optional W/H/context guard.

    config requires model_identity. occurrence_by_case_id is permitted only for
    a unique-case schedule; repeated case IDs need an explicit occurrence field.
    All files below out are raw ignored-local, not W&B payloads or Git artifacts.
    """
    def __init__(self, model, tokenizer, assets, config, out, state_callback=None):
        require(config.get('model_identity'), 'GENERATION_MODEL_IDENTITY_REQUIRED')
        require(config.get('profile', config.get('generation_profile', PROFILE)) == PROFILE,
                'GENERATION_PROFILE_CHANGED')
        require(config.get('eval_seed', config.get('generation_eval_seed', EVAL_SEED)) == EVAL_SEED,
                'GENERATION_SEED_CHANGED')
        self.model, self.tokenizer, self.assets = model, tokenizer, assets
        self.config = copy.deepcopy(config)
        self.out = Path(out)
        self.state_callback = state_callback
        self.model_identity = config['model_identity']
        self.assets_sha = getattr(assets, 'sha', 'ASSET_NOT_AVAILABLE')
        self.source_identity = config.get('generation_source_sha', config.get('source_identity'))
        require(self.source_identity, 'GENERATION_SOURCE_IDENTITY_REQUIRED')
        self.runtime_identity = dict(schema=SCHEMA, profile=PROFILE, eval_seed=EVAL_SEED,
            model_identity=self.model_identity, generation_source_sha=self.source_identity,
            reference_assets_sha256=self.assets_sha, route='UNPADDED_FULL_PREFIX_NO_CACHE')
        self.runtime_sha = digest(self.runtime_identity)
        immutable_write(self.out/'observer-identity.json', dict(identity=self.runtime_identity,
            identity_sha256=self.runtime_sha, raw_local_only=True, checkpoint_saved=False))

    def _external(self):
        return copy.deepcopy(self.state_callback()) if self.state_callback is not None else None

    def _raw_case(self, record, ordinal, state_identity):
        record_identity = _record_identity(record, ordinal)
        identity = dict(runtime=self.runtime_sha, state_identity=state_identity,
                        record_identity=record_identity)
        key = digest(identity)
        path = self.out/'observations'/(key+'.json')
        if path.exists():
            value = json.loads(path.read_text())
            require(value['identity'] == identity and value['identity_sha256'] == key,
                    'GENERATION_CACHE_IDENTITY')
            payload = {k: v for k, v in value.items() if k != 'payload_sha256'}
            require(value['payload_sha256'] == digest(payload), 'GENERATION_CACHE_BYTES')
            return value, path, True
        observations = [generate_row(self.model, self.tokenizer, prompt,
            model_identity=self.model_identity, occurrence=ordinal, prompt_index=index)
            for index, prompt in enumerate(record_identity['generation_prompts'])]
        references = None if self.assets is None else self.assets.snippets_for(
            record_identity['relation_id'], record_identity['target_new_id'])
        metrics = score_case(observations, references,
            getattr(self.assets, 'vectorizer', None), getattr(self.assets, 'word_tokenize', None))
        value = dict(identity=identity, identity_sha256=key, occurrence=ordinal,
            case_id=record.get('case_id'), observations=observations, metrics=metrics,
            raw_local_only=True, checkpoint_saved=False)
        value['payload_sha256'] = digest(value)
        immutable_write(path, value)
        return value, path, False

    def observe(self, records, endpoint, cohort=None, state_identity=None):
        records = list(records)
        mapping = self.config.get('occurrence_by_case_id')
        ordinals = [occurrence(r, mapping) for r in records]
        require(len(set(ordinals)) == len(ordinals), 'GENERATION_OCCURRENCE_DUPLICATE')
        before, external = model_signature(self.model), self._external()
        state_identity = state_identity if state_identity is not None else external
        require(state_identity is not None, 'GENERATION_STATE_IDENTITY_REQUIRED')
        # Explicit JSON identity, not a tensor dump or serialized cache object.
        state_identity = json.loads(json.dumps(state_identity, sort_keys=True, allow_nan=False))
        state_sha = digest(state_identity)
        row_cache = []
        started = time.monotonic()
        cache_hits = forwards = tokens = 0
        saved_rng = rng_snapshot()
        try:
            with isolated_rng():
                for record, ordinal in zip(records, ordinals):
                    value, path, reused = self._raw_case(record, ordinal, state_identity)
                    cache_hits += int(reused)
                    if not reused:
                        forwards += sum(x['model_forwards'] for x in value['observations'])
                        tokens += sum(x['full_prefix_token_work'] for x in value['observations'])
                    row_cache.append(dict(occurrence=ordinal, case_id=value['case_id'],
                        identity_sha256=value['identity_sha256'], observation_path=str(path.resolve()),
                        payload_sha256=value['payload_sha256'], metrics=value['metrics']))
        finally:
            require(rng_equal(saved_rng), 'GENERATION_RNG_RESTORE')
            require(model_signature(self.model) == before, 'GENERATION_MODEL_HOOK_CACHE_MUTATION')
            require(self._external() == external, 'GENERATION_NATIVE_STATE_MUTATION')
        summary = reduce_cases(row_cache)
        identity = dict(runtime=self.runtime_sha, state_sha256=state_sha,
            endpoint=endpoint, cohort=cohort, ordered_occurrences=ordinals,
            observation_identities=[r['identity_sha256'] for r in row_cache])
        key = digest(identity)
        # Timing/transport stays separate from immutable scientific evidence.
        receipt = dict(identity=identity, identity_sha256=key, summary=summary, rows=row_cache,
            RNG_restored=True, observer_no_mutation=True, raw_local_only=True)
        path = self.out/'endpoints'/(key+'.json')
        immutable_write(path, receipt)
        return dict(**receipt, rows_path=str(path.resolve()),
            work=dict(new_case_observations=len(records)-cache_hits, cached_case_observations=cache_hits,
                generation_forwards=forwards, full_prefix_token_work=tokens,
                seconds=time.monotonic()-started))

    def subset(self, observed, records, endpoint, cohort=None):
        """Identity-checked CPU subset. No second generation or model invocation."""
        records = list(records)
        require(observed['identity']['runtime'] == self.runtime_sha, 'GENERATION_RUNTIME_IDENTITY')
        require(observed['identity_sha256'] == digest(observed['identity'])
            and observed['identity']['ordered_occurrences'] == [r['occurrence'] for r in observed['rows']]
            and len({r['occurrence'] for r in observed['rows']}) == len(observed['rows'])
            and observed['identity']['observation_identities'] == [r['identity_sha256'] for r in observed['rows']]
            and observed['summary'] == reduce_cases(observed['rows'])
            and observed['RNG_restored'] is True and observed['observer_no_mutation'] is True,
            'GENERATION_SUBSET_ENDPOINT_IDENTITY')
        lookup = {r['occurrence']: r for r in observed['rows']}
        ordinals = [occurrence(r, self.config.get('occurrence_by_case_id')) for r in records]
        require(len(set(ordinals)) == len(ordinals) and set(ordinals) <= set(lookup),
                'GENERATION_SUBSET_IDENTITY')
        selected = []
        for record, ordinal in zip(records, ordinals):
            row = lookup[ordinal]
            require(row['case_id'] == record.get('case_id'), 'GENERATION_SUBSET_CASE_IDENTITY')
            raw = json.loads(Path(row['observation_path']).read_text())
            require(raw['identity']['record_identity'] == _record_identity(record, ordinal),
                    'GENERATION_SUBSET_PROMPT_IDENTITY')
            require(raw['identity_sha256'] == row['identity_sha256'] == digest(raw['identity'])
                and raw['identity']['runtime'] == self.runtime_sha
                and digest(raw['identity']['state_identity']) == observed['identity']['state_sha256']
                and raw['occurrence'] == row['occurrence'] and raw['case_id'] == row['case_id']
                and raw['payload_sha256'] == row['payload_sha256']
                and raw['metrics'] == row['metrics'], 'GENERATION_SUBSET_ROW_IDENTITY')
            require(raw['payload_sha256'] == digest({k:v for k,v in raw.items() if k!='payload_sha256'}),
                    'GENERATION_CACHE_BYTES')
            selected.append(copy.deepcopy(row))
        identity = dict(runtime=self.runtime_sha, state_sha256=observed['identity']['state_sha256'],
            endpoint=endpoint, cohort=cohort, ordered_occurrences=ordinals,
            observation_identities=[r['identity_sha256'] for r in selected])
        key = digest(identity)
        receipt = dict(identity=identity, identity_sha256=key, summary=reduce_cases(selected), rows=selected,
            RNG_restored=True, observer_no_mutation=True, raw_local_only=True)
        path = self.out/'endpoints'/(key+'.json')
        immutable_write(path, receipt)
        return dict(**receipt, rows_path=str(path.resolve()), work=dict(new_case_observations=0,
            cached_case_observations=len(selected), generation_forwards=0, full_prefix_token_work=0,
            seconds=0.0))

    def read_observed(self, path):
        return read_observed(path, expected_runtime=self.runtime_sha)
