"""Generation and scoring share one local observation, with nonmutation guards."""
import copy
import json
from pathlib import Path
import time

import torch

from .common import SCHEMA, PROFILE, EVAL_SEED, digest, immutable_write, require
from .generator import generate_row, isolated_rng, rng_snapshot, rng_equal
from .metrics import score_case, reduce_cases
from .compatibility import (W0Compatibility, qualification_binding, member,
                            verify_member, load_compatibility, runtime_identity,
                            verify_raw_complete, verify_scored)
from .progress import GenerationProgress


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


def read_observed(path, expected_runtime=None, *, allow_cpu_fixture=False):
    """Load an immutable local endpoint; never accepts a summary without rows."""
    value = json.loads(Path(path).read_text())
    require(value['identity_sha256'] == digest(value['identity']), 'GENERATION_ENDPOINT_IDENTITY')
    require(value['RNG_restored'] is True and value['observer_no_mutation'] is True,
            'GENERATION_ENDPOINT_OBSERVER_GUARD')
    if expected_runtime is not None:
        require(value['identity']['runtime'] == expected_runtime, 'GENERATION_RUNTIME_IDENTITY')
    rows = value['rows']
    compatibility = None
    eligible = {}
    if 'compatibility_member' in value:
        compatibility = load_compatibility(value['compatibility_member'],
            expected_runtime=value['identity']['runtime'],
            expected_qualification=value['identity']['qualification_receipt_sha256'])
        require(value['identity']['compatibility_sha256'] == compatibility['identity_sha256'],
                'GENERATION_COMPATIBILITY_ENDPOINT_SHA')
        eligible = {r['occurrence']:r for r in compatibility['identity']['original_entries']}
    if 'qualification_receipt_member' in value:
        verify_member(value['qualification_receipt_member'])
        require(value['identity']['qualification_receipt_sha256'] == value['qualification_receipt_member']['sha256'],
                'GENERATION_ENDPOINT_QUALIFICATION_SHA')
        from .kv_qualification import verify_actual_receipt
        verify_actual_receipt(value['qualification_receipt_member'], allow_cpu_fixture=allow_cpu_fixture)
    require(value['identity']['ordered_occurrences'] == [r['occurrence'] for r in rows]
        and len({r['occurrence'] for r in rows}) == len(rows)
        and value['identity']['observation_identities'] == [r['identity_sha256'] for r in rows],
        'GENERATION_ENDPOINT_ROWS_IDENTITY')
    require(value['summary'] == reduce_cases(rows), 'GENERATION_ENDPOINT_REDUCTION')
    for row in rows:
        raw = json.loads(Path(row['observation_path']).read_text())
        raw_runtime = raw['identity']['runtime']
        runtime_ok = raw_runtime == value['identity']['runtime']
        if not runtime_ok and compatibility is not None:
            entry = eligible.get(row['occurrence'])
            require(entry is not None and entry['original_raw_member']['path'] == row['observation_path']
                and entry['original_runtime_sha256'] == raw_runtime
                and entry['original_identity_sha256'] == row['identity_sha256']
                and entry['original_payload_sha256'] == row['payload_sha256'],
                'GENERATION_COMPATIBILITY_ROW_ORIGIN')
            verify_member(entry['original_raw_member'])
            runtime_ok = True
        if 'provenance' in row:
            provenance = row['provenance']
            require(provenance['raw_member']['path'] == row['observation_path']
                    and provenance['runtime_sha256'] == raw_runtime,
                    'GENERATION_ROW_PROVENANCE')
        require(raw['identity_sha256'] == row['identity_sha256'] == digest(raw['identity'])
            and runtime_ok
            and digest(raw['identity']['state_identity']) == value['identity']['state_sha256']
            and raw['payload_sha256'] == row['payload_sha256']
            and raw['payload_sha256'] == digest({k:v for k,v in raw.items() if k!='payload_sha256'})
            and raw['occurrence'] == row['occurrence'] and raw['case_id'] == row['case_id']
            and raw['identity']['record_identity']['ordered_occurrence'] == row['occurrence']
            and raw['identity']['record_identity']['case_id'] == row['case_id']
            and raw['metrics'] == row['metrics'], 'GENERATION_OBSERVATION_BYTES_IDENTITY')
        if 'provenance' in row:
            verify_member(row['provenance']['raw_member'])
    if 'provenance_sha256' in value['identity']:
        require(value['identity']['provenance_sha256'] == digest([r['provenance'] for r in rows]),
                'GENERATION_ENDPOINT_PROVENANCE_IDENTITY')
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
    def __init__(self, model, tokenizer, assets, config, out, state_callback=None,
                 progress_callback=None):
        require(config.get('model_identity'), 'GENERATION_MODEL_IDENTITY_REQUIRED')
        require(config.get('profile', config.get('generation_profile', PROFILE)) == PROFILE,
                'GENERATION_PROFILE_CHANGED')
        require(config.get('eval_seed', config.get('generation_eval_seed', EVAL_SEED)) == EVAL_SEED,
                'GENERATION_SEED_CHANGED')
        self.model, self.tokenizer, self.assets = model, tokenizer, assets
        self.config = copy.deepcopy(config)
        self.out = Path(out)
        self.state_callback = state_callback
        self.progress_callback = progress_callback
        self._progress_step = 0
        self.model_identity = config['model_identity']
        self.assets_sha = getattr(assets, 'sha', 'ASSET_NOT_AVAILABLE')
        self.source_identity = config.get('generation_source_sha', config.get('source_identity'))
        require(self.source_identity, 'GENERATION_SOURCE_IDENTITY_REQUIRED')
        self.runtime_identity = runtime_identity(config, self.assets_sha)
        self.qualification_member = None
        if 'generation_route' in config:
            self.qualification_member = qualification_binding(config)
        self.runtime_sha = digest(self.runtime_identity)
        immutable_write(self.out/'observer-identity.json', dict(identity=self.runtime_identity,
            identity_sha256=self.runtime_sha, raw_local_only=True, checkpoint_saved=False))

    def _row(self, value, path, *, origin='NEW_CURRENT_RUNTIME', entry=None):
        provenance = dict(origin=origin, raw_member=member(path),
            runtime_sha256=value['identity']['runtime'],
            generation_source_sha=self.source_identity,
            route=self.runtime_identity['route'])
        if entry is not None:
            provenance.update(generation_source_sha=entry['original_generation_source_sha'],
                route=entry['original_route'], compatibility_sha256=self._compatibility.sha)
        return dict(occurrence=value['occurrence'], case_id=value['case_id'],
            identity_sha256=value['identity_sha256'], observation_path=str(path.resolve()),
            payload_sha256=value['payload_sha256'], metrics=value['metrics'], provenance=provenance)

    def _assemble(self, identity, observations, record):
        ri = identity['record_identity']
        references = None if self.assets is None else self.assets.snippets_for(ri['relation_id'], ri['target_new_id'])
        metrics = score_case(observations, references, getattr(self.assets, 'vectorizer', None),
                             getattr(self.assets, 'word_tokenize', None))
        value = dict(identity=identity, identity_sha256=digest(identity),
            occurrence=ri['ordered_occurrence'], case_id=record.get('case_id'),
            observations=observations, metrics=metrics, raw_local_only=True, checkpoint_saved=False)
        value['payload_sha256'] = digest(value)
        path = self.out/'observations'/(value['identity_sha256']+'.json')
        immutable_write(path, value)
        return value, path

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
            verify_raw_complete(value, record_identity, runtime=self.runtime_sha,
                state=state_identity, model_identity=self.model_identity,
                expected_route=self.runtime_identity['route'])
            verify_scored(value, self.assets)
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
        self._compatibility = None
        if self.config.get('old_w0_reuse') and endpoint == 'W0':
            self._compatibility = W0Compatibility(self, records, ordinals, state_identity)
        progress = GenerationProgress(len(records), sum(len(_record_identity(r,o)['generation_prompts'])
            for r,o in zip(records,ordinals)), self.progress_callback,
            phase='W0_generation' if endpoint == 'W0' else
                'generation_evaluation', first_step=self._progress_step)
        saved_rng = rng_snapshot()
        try:
            with isolated_rng():
                progress.emit('start', force=True)
                if self.qualification_member is None:
                    for record, ordinal in zip(records, ordinals):
                        value, path, reused = self._raw_case(record, ordinal, state_identity)
                        cache_hits += int(reused)
                        if not reused:
                            forwards += sum(x['model_forwards'] for x in value['observations'])
                            tokens += sum(x['full_prefix_token_work'] for x in value['observations'])
                        row_cache.append(self._row(value, path, origin='CURRENT_RUNTIME_CACHE' if reused else 'NEW_CURRENT_RUNTIME'))
                        progress.complete_case(value, reused=reused)
                else:
                    # Work is collected in bounded groups of complete cases. The
                    # generator buckets only exact equal input-token lengths and
                    # restores request order; incomplete cases are never persisted.
                    from .generator import generate_rows
                    pending = []
                    lookup = {}
                    def flush():
                        nonlocal forwards, tokens
                        if not pending:
                            return
                        requests = []
                        for record, ordinal, identity in pending:
                            requests.extend(dict(prompt=prompt, occurrence=ordinal, prompt_index=i,
                                model_identity=self.model_identity) for i,prompt in
                                enumerate(identity['record_identity']['generation_prompts']))
                        generated = generate_rows(self.model, self.tokenizer, requests,
                            route=self.runtime_identity['route'], microbatch=self.config['generation_microbatch'])
                        require(len(generated) == len(requests), 'GENERATION_BATCH_ROW_COUNT')
                        cursor = 0
                        for record, ordinal, identity in pending:
                            count = len(identity['record_identity']['generation_prompts'])
                            values = generated[cursor:cursor+count]; cursor += count
                            value, path = self._assemble(identity, values, record)
                            forwards += sum(x['model_forwards'] for x in values)
                            tokens += sum(x['full_prefix_token_work'] for x in values)
                            lookup[ordinal] = self._row(value, path)
                            progress.complete_case(value)
                        pending.clear()
                    for record, ordinal in zip(records, ordinals):
                        identity = dict(runtime=self.runtime_sha, state_identity=state_identity,
                                        record_identity=_record_identity(record,ordinal))
                        path = self.out/'observations'/(digest(identity)+'.json')
                        compatible = self._compatibility.reuse(ordinal) if self._compatibility else None
                        if path.exists():
                            value = json.loads(path.read_text())
                            require(value['identity'] == identity and value['identity_sha256'] == digest(identity)
                                and value['payload_sha256'] == digest({k:v for k,v in value.items() if k!='payload_sha256'}),
                                'GENERATION_CACHE_BYTES')
                            verify_raw_complete(value, identity['record_identity'], runtime=self.runtime_sha,
                                state=state_identity, model_identity=self.model_identity,
                                expected_route=self.runtime_identity['route'])
                            verify_scored(value, self.assets)
                            flush(); cache_hits += 1
                            lookup[ordinal] = self._row(value, path, origin='CURRENT_RUNTIME_CACHE')
                            progress.complete_case(value, reused=True)
                        elif compatible is not None:
                            flush(); value, path, entry = compatible; cache_hits += 1
                            lookup[ordinal] = self._row(value, path, origin='COMPATIBLE_ORIGINAL_W0', entry=entry)
                            progress.complete_case(value, reused=True)
                        else:
                            pending.append((record,ordinal,identity))
                            if sum(len(x[2]['record_identity']['generation_prompts']) for x in pending) >= 64:
                                flush()
                    flush()
                    row_cache = [lookup[o] for o in ordinals]
                progress.finish()
        except BaseException:
            # Preserve the original scientific error even if error logging fails;
            # final RNG/model/state guards still apply.
            try:
                with isolated_rng():
                    progress.emit('error', force=True)
            except Exception:
                pass
            raise
        finally:
            self._progress_step = progress.step
            require(rng_equal(saved_rng), 'GENERATION_RNG_RESTORE')
            require(model_signature(self.model) == before, 'GENERATION_MODEL_HOOK_CACHE_MUTATION')
            require(self._external() == external, 'GENERATION_NATIVE_STATE_MUTATION')
        summary = reduce_cases(row_cache)
        identity = dict(runtime=self.runtime_sha, state_sha256=state_sha,
            endpoint=endpoint, cohort=cohort, ordered_occurrences=ordinals,
            observation_identities=[r['identity_sha256'] for r in row_cache])
        identity['provenance_sha256'] = digest([r['provenance'] for r in row_cache])
        if self.qualification_member is not None:
            identity['qualification_receipt_sha256'] = self.qualification_member['sha256']
        if self._compatibility is not None:
            identity['compatibility_sha256'] = self._compatibility.sha
        key = digest(identity)
        # Timing/transport stays separate from immutable scientific evidence.
        receipt = dict(identity=identity, identity_sha256=key, summary=summary, rows=row_cache,
            RNG_restored=True, observer_no_mutation=True, raw_local_only=True)
        if self.qualification_member is not None:
            receipt['qualification_receipt_member'] = self.qualification_member
        if self._compatibility is not None:
            receipt['compatibility_member'] = self._compatibility.member
        path = self.out/'endpoints'/(key+'.json')
        immutable_write(path, receipt)
        return dict(**receipt, rows_path=str(path.resolve()),
            work=dict(new_case_observations=len(records)-cache_hits, cached_case_observations=cache_hits,
                generation_forwards=forwards, full_prefix_token_work=tokens,
                physical_forward_calls=progress.counts['physical_forward_calls'],
                prefill_query_tokens=progress.counts['prefill_query_tokens'],
                decode_query_tokens=progress.counts['decode_query_tokens'],
                completed_prompts=progress.counts['completed_prompts'],
                generated_tokens=progress.counts['generated_tokens'],
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
        compatible_entries = {}
        if 'compatibility_member' in observed:
            compat = load_compatibility(observed['compatibility_member'],
                expected_runtime=self.runtime_sha,
                expected_qualification=observed['identity']['qualification_receipt_sha256'])
            require(compat['identity_sha256'] == observed['identity']['compatibility_sha256'],
                    'GENERATION_COMPATIBILITY_ENDPOINT_SHA')
            compatible_entries = {r['occurrence']:r for r in compat['identity']['original_entries']}
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
            runtime_ok = raw['identity']['runtime'] == self.runtime_sha
            if not runtime_ok:
                entry = compatible_entries.get(ordinal)
                require(entry is not None and entry['original_raw_member']['path'] == row['observation_path']
                    and entry['original_identity_sha256'] == row['identity_sha256']
                    and entry['original_payload_sha256'] == row['payload_sha256']
                    and entry['original_runtime_sha256'] == raw['identity']['runtime'],
                    'GENERATION_SUBSET_ROW_IDENTITY')
                verify_member(entry['original_raw_member'])
                runtime_ok = True
            require(raw['identity_sha256'] == row['identity_sha256'] == digest(raw['identity'])
                and runtime_ok
                and digest(raw['identity']['state_identity']) == observed['identity']['state_sha256']
                and raw['occurrence'] == row['occurrence'] and raw['case_id'] == row['case_id']
                and raw['payload_sha256'] == row['payload_sha256']
                and raw['metrics'] == row['metrics'], 'GENERATION_SUBSET_ROW_IDENTITY')
            require(raw['payload_sha256'] == digest({k:v for k,v in raw.items() if k!='payload_sha256'}),
                    'GENERATION_CACHE_BYTES')
            if 'provenance' in row:
                require(row['provenance']['runtime_sha256'] == raw['identity']['runtime']
                    and row['provenance']['raw_member']['path'] == row['observation_path'],
                    'GENERATION_SUBSET_ROW_IDENTITY')
                verify_member(row['provenance']['raw_member'])
            selected_row = copy.deepcopy(row)
            if 'provenance' not in selected_row:
                # Strict old same-runtime reader remains usable; the new subset
                # adds original-byte provenance without changing that old raw.
                selected_row['provenance'] = dict(origin='CURRENT_RUNTIME_CACHE',
                    raw_member=member(row['observation_path']), runtime_sha256=self.runtime_sha,
                    generation_source_sha=self.source_identity, route=self.runtime_identity['route'])
            selected.append(selected_row)
        identity = dict(runtime=self.runtime_sha, state_sha256=observed['identity']['state_sha256'],
            endpoint=endpoint, cohort=cohort, ordered_occurrences=ordinals,
            observation_identities=[r['identity_sha256'] for r in selected])
        identity['provenance_sha256'] = digest([r['provenance'] for r in selected])
        for key in ('qualification_receipt_sha256', 'compatibility_sha256'):
            if key in observed['identity']:
                identity[key] = observed['identity'][key]
        key = digest(identity)
        receipt = dict(identity=identity, identity_sha256=key, summary=reduce_cases(selected), rows=selected,
            RNG_restored=True, observer_no_mutation=True, raw_local_only=True)
        for key in ('qualification_receipt_member', 'compatibility_member'):
            if key in observed:
                receipt[key] = copy.deepcopy(observed[key])
        path = self.out/'endpoints'/(key+'.json')
        immutable_write(path, receipt)
        return dict(**receipt, rows_path=str(path.resolve()), work=dict(new_case_observations=0,
            cached_case_observations=len(selected), generation_forwards=0, full_prefix_token_work=0,
            seconds=0.0))

    def read_observed(self, path):
        return read_observed(path, expected_runtime=self.runtime_sha,
            allow_cpu_fixture=self.config.get('qualification_allow_cpu_fixture', False))
