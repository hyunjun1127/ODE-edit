"""Native case-batch adapter for the existing endpoint/subset observation API.

Only an exact complete endpoint is a cache hit. A global endpoint RNG stream
cannot be resumed from arbitrary per-case rows or borrowed from another cohort.
Raw partial rows remain local and are never relabelled as complete evidence.
"""
import copy
import json
from pathlib import Path
import time

from .common import EVAL_SEED, digest, immutable_write, require
from .compatibility import member, verify_member, verify_scored
from .generator import isolated_rng, rng_snapshot, rng_equal
from .metrics import reduce_cases
from .native_generator import generate_case
from .native_profile import PROFILE, ROUTE, runtime_identity
from .observer import GenerationObserver, _record_identity, model_signature, occurrence
from .observer import read_observed as read_existing_observed
from .progress import GenerationProgress


def verify_native_raw(raw, *, expected_runtime=None, assets=None, expected_stream=None):
    """Validate native profile completeness, not the old per-prompt/EOS profile."""
    identity = raw['identity']
    require(raw['identity_sha256'] == digest(identity)
        and raw['payload_sha256'] == digest({k:v for k,v in raw.items() if k != 'payload_sha256'})
        and raw.get('raw_local_only') is True and raw.get('checkpoint_saved') is False,
        'NATIVE_GENERATION_RAW_IDENTITY')
    if expected_runtime is not None:
        require(identity['runtime'] == expected_runtime, 'GENERATION_RUNTIME_IDENTITY')
    stream = identity.get('sampling_stream_sha256')
    require(type(stream) is str and len(stream) == 64
        and all(char in '0123456789abcdef' for char in stream), 'NATIVE_GENERATION_STREAM_IDENTITY')
    if expected_stream is not None:
        require(stream == expected_stream, 'NATIVE_GENERATION_STREAM_IDENTITY')
    ri, observations = identity['record_identity'], raw['observations']
    prompts = ri['generation_prompts']
    require(raw['occurrence'] == ri['ordered_occurrence'] and raw['case_id'] == ri['case_id']
        and type(observations) is list and len(observations) == len(prompts),
        'NATIVE_GENERATION_CASE_INCOMPLETE')
    inputs = [row['input_token_ids'] for row in observations]
    width = max(map(len, inputs), default=0)
    minimum = min(map(len, inputs), default=0)
    forwards = 0 if width >= 100 else 100-minimum
    for index, (prompt, row) in enumerate(zip(prompts, observations)):
        input_ids, continuation, full = row['input_token_ids'], row['continuation_token_ids'], row['full_token_ids']
        padded_input, padded_full = row['padded_input_token_ids'], row['padded_decode_token_ids']
        require(row['profile'] == PROFILE and row['route'] == ROUTE and row['seed'] == EVAL_SEED
            and row['sampling_scope'] == 'ENDPOINT_GLOBAL_BATCH_STREAM'
            and row['prompt'] == prompt and row['occurrence'] == raw['occurrence']
            and row['prompt_index'] == index and row['endpoint_RNG_restore_guard_required'] is True
            and row['EOS_stop'] is False and row['case_batch_prompt_count'] == len(prompts)
            and row['initial_batch_width'] == width and row['model_forwards'] == forwards
            and row['physical_forward_calls'] == (forwards if index == 0 else 0)
            and row['sampling'] == dict(top_k=5, temperature=1, top_p=1,
                                      max_total_tokens=100, n_gen_per_prompt=1),
            'NATIVE_GENERATION_PROMPT_PROFILE')
        require(type(input_ids) is list and bool(input_ids) and type(continuation) is list
            and full == input_ids+continuation and padded_input[:len(input_ids)] == input_ids
            and len(padded_input) == width and padded_full[:len(full)] == full
            and len(padded_full) == (width if width >= 100 else 100)
            and all(type(token) is int and token >= 0 for token in padded_input+padded_full)
            and row['input_token_count'] == len(input_ids)
            and row['continuation_token_count'] == len(continuation)
            and (not continuation if width >= 100 else len(full) == 100)
            and row['stop_reason'] == ('length_cap_no_continuation' if width >= 100 else 'length_cap')
            and type(row['text']) is str, 'NATIVE_GENERATION_TOKEN_COMPLETENESS')
        expected_prefill = minimum if forwards else 0
        expected_decode = forwards-1 if forwards else 0
        expected_dense = sum(range(minimum, 100)) if forwards else 0
        require(row['prefill_query_tokens'] == expected_prefill
            and row['decode_query_tokens'] == expected_decode
            and row['full_prefix_token_work'] == expected_dense,
            'NATIVE_GENERATION_WORK_COMPLETENESS')
    if assets is not None:
        verify_scored(raw, assets)
    return raw


def _zero_work(count):
    return dict(new_case_observations=0, cached_case_observations=count,
        generation_forwards=0, full_prefix_token_work=0, physical_forward_calls=0,
        prefill_query_tokens=0, decode_query_tokens=0, completed_prompts=0,
        generated_tokens=0, seconds=0.0)


def read_observed(path, expected_runtime=None, *, assets=None, _depth=0):
    """CPU-only native endpoint reader; original endpoint/raw bytes are retained."""
    require(_depth < 32, 'NATIVE_GENERATION_SUBSET_PARENT_DEPTH')
    require(Path(path).is_file() and not Path(path).is_symlink(), 'NATIVE_GENERATION_ENDPOINT_REGULAR_FILE')
    value = read_existing_observed(path, expected_runtime=expected_runtime)
    require(Path(path).stem == value['identity_sha256'], 'NATIVE_GENERATION_ENDPOINT_PATH_IDENTITY')
    streams = set()
    work = dict(physical_forward_calls=0,prefill_query_tokens=0,decode_query_tokens=0)
    for row in value['rows']:
        raw = json.loads(Path(row['observation_path']).read_text())
        verify_native_raw(raw, expected_runtime=value['identity']['runtime'], assets=assets,
            expected_stream=value['identity'].get('sampling_stream_sha256'))
        streams.add(raw['identity']['sampling_stream_sha256'])
        for name in work:
            work[name] += sum(observation[name] for observation in raw['observations'])
    require(len(streams) <= 1, 'NATIVE_GENERATION_MIXED_STREAM_SUBSET')
    if 'native_execution_member' in value:
        execution = json.loads(verify_member(value['native_execution_member']).read_text())
        require(execution['identity_sha256'] == digest(execution['identity'])
            and execution['identity'] == value['identity']
            and execution['profile'] == PROFILE and execution['route'] == ROUTE
            and execution['native_execution_complete'] is True
            and execution['qualification_performed'] is False
            and execution['no_fallback'] is True
            and execution['RNG_restored'] is True and execution['observer_no_mutation'] is True
            and execution['raw_local_only'] is True
            and all(execution[name] == count for name,count in work.items()),
            'NATIVE_GENERATION_EXECUTION_RECEIPT')
    elif 'parent_endpoint_member' in value:
        parent_path = verify_member(value['parent_endpoint_member'])
        parent = read_observed(parent_path, expected_runtime=value['identity']['runtime'],
                               assets=assets, _depth=_depth+1)
        require(value['identity']['parent_endpoint_identity_sha256'] == parent['identity_sha256']
            and value['identity']['sampling_stream_sha256'] == parent['identity']['sampling_stream_sha256']
            and value['identity']['state_sha256'] == parent['identity']['state_sha256'],
            'NATIVE_GENERATION_SUBSET_PARENT_IDENTITY')
        lookup = {row['occurrence']:row for row in parent['rows']}
        require(all(row == lookup.get(row['occurrence']) for row in value['rows']),
                'NATIVE_GENERATION_SUBSET_PARENT_ROWS')
    else:
        require(False, 'NATIVE_GENERATION_EXECUTION_RECEIPT_REQUIRED')
    value['work'] = _zero_work(len(value['rows']))
    return value


class NativeGenerationObserver(GenerationObserver):
    """Same constructor/observe/subset/read API, distinct native source identity."""
    def __init__(self, model, tokenizer, assets, config, out, state_callback=None,
                 progress_callback=None):
        require(config.get('model_identity'), 'NATIVE_GENERATION_MODEL_IDENTITY_REQUIRED')
        self.model, self.tokenizer, self.assets = model, tokenizer, assets
        self.config = copy.deepcopy(config)
        self.out = Path(out)
        self.state_callback, self.progress_callback = state_callback, progress_callback
        self._progress_step = 0
        self.model_identity = config['model_identity']
        self.assets_sha = getattr(assets, 'sha', 'ASSET_NOT_AVAILABLE')
        self.runtime_identity = runtime_identity(config, self.assets_sha)
        self.runtime_sha = digest(self.runtime_identity)
        self.source_identity = self.runtime_identity['generation_source_sha']
        # This is an actual native execution, never a synthetic MB8 qualification.
        self.qualification_member = None
        self._compatibility = None
        immutable_write(self.out/'observer-identity.json', dict(identity=self.runtime_identity,
            identity_sha256=self.runtime_sha, raw_local_only=True, checkpoint_saved=False))

    def observe(self, records, endpoint, cohort=None, state_identity=None):
        records = list(records)
        ordinals = [occurrence(record, self.config.get('occurrence_by_case_id')) for record in records]
        require(len(set(ordinals)) == len(ordinals), 'GENERATION_OCCURRENCE_DUPLICATE')
        record_ids = [_record_identity(record, ordinal) for record, ordinal in zip(records, ordinals)]
        before, external = model_signature(self.model), self._external()
        state_identity = state_identity if state_identity is not None else external
        require(state_identity is not None, 'GENERATION_STATE_IDENTITY_REQUIRED')
        state_identity = json.loads(json.dumps(state_identity, sort_keys=True, allow_nan=False))
        stream_sha = digest(dict(runtime=self.runtime_sha, eval_seed=EVAL_SEED,
                                 ordered_record_identities=record_ids))
        identities = [dict(runtime=self.runtime_sha, state_identity=state_identity,
            record_identity=ri, sampling_stream_sha256=stream_sha) for ri in record_ids]
        endpoint_identity = dict(runtime=self.runtime_sha, state_sha256=digest(state_identity),
            endpoint=endpoint, cohort=cohort, ordered_occurrences=ordinals,
            observation_identities=[digest(identity) for identity in identities],
            sampling_stream_sha256=stream_sha)
        key = digest(endpoint_identity)
        path = self.out/'endpoints'/(key+'.json')
        progress = GenerationProgress(len(records), sum(len(ri['generation_prompts']) for ri in record_ids),
            self.progress_callback, phase='W0_generation' if endpoint == 'W0' else 'generation_evaluation',
            first_step=self._progress_step)
        saved_rng, started = rng_snapshot(), time.monotonic()
        row_cache, cached, logical_forwards, token_work = [], False, 0, 0
        try:
            with isolated_rng(EVAL_SEED):
                progress.emit('start', force=True)
                if path.exists():
                    observed = self.read_observed(path)
                    require(observed['identity'] == endpoint_identity, 'NATIVE_GENERATION_ENDPOINT_CACHE_IDENTITY')
                    row_cache, cached = observed['rows'], True
                    for row in row_cache:
                        progress.complete_case(json.loads(Path(row['observation_path']).read_text()), reused=True)
                else:
                    # Never skip an isolated cached case: sampling for earlier
                    # cases is part of the original global RNG stream. Incomplete
                    # raw rows can only be reproduced from the start and byte-
                    # compared by immutable_write, not used as resume evidence.
                    for record, identity in zip(records, identities):
                        ri = identity['record_identity']
                        observations = generate_case(self.model, self.tokenizer,
                            ri['generation_prompts'], occurrence=ri['ordered_occurrence'])
                        value, raw_path = self._assemble(identity, observations, record)
                        verify_native_raw(value, expected_runtime=self.runtime_sha,
                            assets=self.assets, expected_stream=stream_sha)
                        row_cache.append(self._row(value, raw_path))
                        logical_forwards += sum(row['model_forwards'] for row in observations)
                        token_work += sum(row['full_prefix_token_work'] for row in observations)
                        progress.complete_case(value)
                progress.finish()
        except BaseException:
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
        receipt = dict(identity=endpoint_identity, identity_sha256=key, summary=reduce_cases(row_cache),
            rows=row_cache, RNG_restored=True, observer_no_mutation=True, raw_local_only=True)
        if not cached:
            execution = dict(identity=endpoint_identity, identity_sha256=key,
                profile=PROFILE, route=ROUTE, native_execution_complete=True,
                qualification_performed=False, no_fallback=True, RNG_restored=True,
                observer_no_mutation=True, raw_local_only=True,
                physical_forward_calls=progress.counts['physical_forward_calls'],
                prefill_query_tokens=progress.counts['prefill_query_tokens'],
                decode_query_tokens=progress.counts['decode_query_tokens'])
            execution_path = self.out/'native-executions'/(key+'.json')
            immutable_write(execution_path, execution)
            receipt['native_execution_member'] = member(execution_path)
            immutable_write(path, receipt)
        else:
            receipt = {name:value for name,value in observed.items() if name not in ('work', 'rows_path')}
        return dict(**receipt, rows_path=str(path.resolve()), work=dict(
            new_case_observations=0 if cached else len(records),
            cached_case_observations=len(records) if cached else 0,
            generation_forwards=logical_forwards, full_prefix_token_work=token_work,
            physical_forward_calls=progress.counts['physical_forward_calls'],
            prefill_query_tokens=progress.counts['prefill_query_tokens'],
            decode_query_tokens=progress.counts['decode_query_tokens'],
            completed_prompts=progress.counts['completed_prompts'],
            generated_tokens=progress.counts['generated_tokens'], seconds=time.monotonic()-started))

    def read_observed(self, path):
        return read_observed(path, expected_runtime=self.runtime_sha, assets=self.assets)

    def subset(self, observed, records, endpoint, cohort=None):
        """No generation; preserve an immutable link to the guarded parent."""
        records = list(records)
        loaded = self.read_observed(observed['rows_path'])
        scientific = {key:value for key,value in loaded.items() if key not in ('rows_path','work')}
        require(scientific == {key:value for key,value in observed.items() if key not in ('rows_path','work')},
                'GENERATION_SUBSET_ENDPOINT_IDENTITY')
        lookup = {row['occurrence']:row for row in loaded['rows']}
        ordinals = [occurrence(record,self.config.get('occurrence_by_case_id')) for record in records]
        require(len(set(ordinals)) == len(ordinals) and set(ordinals) <= set(lookup),
                'GENERATION_SUBSET_IDENTITY')
        selected = []
        for record, ordinal in zip(records,ordinals):
            row = lookup[ordinal]
            raw = json.loads(Path(row['observation_path']).read_text())
            require(raw['identity']['record_identity'] == _record_identity(record,ordinal),
                    'GENERATION_SUBSET_PROMPT_IDENTITY')
            selected.append(copy.deepcopy(row))
        identity = dict(runtime=self.runtime_sha, state_sha256=loaded['identity']['state_sha256'],
            endpoint=endpoint, cohort=cohort, ordered_occurrences=ordinals,
            observation_identities=[row['identity_sha256'] for row in selected],
            sampling_stream_sha256=loaded['identity']['sampling_stream_sha256'],
            parent_endpoint_identity_sha256=loaded['identity_sha256'],
            provenance_sha256=digest([row['provenance'] for row in selected]))
        key = digest(identity)
        receipt = dict(identity=identity,identity_sha256=key,summary=reduce_cases(selected),rows=selected,
            parent_endpoint_member=member(loaded['rows_path']), RNG_restored=True,
            observer_no_mutation=True,raw_local_only=True)
        path = self.out/'endpoints'/(key+'.json')
        immutable_write(path,receipt)
        return dict(**receipt,rows_path=str(path.resolve()),work=_zero_work(len(selected)))
