"""Server2 CF generation bridge: one cold W0, one W20 per official chain.

All sampling/scoring is implemented by official.evaluation.generation. This
bridge supplies exact physical state/cohort/source identity and immutable local
READY evidence. No old W20-only profile or incomplete raw row is adopted.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from official.experiments.prepare import ROOT, file_sha, read
from official.evaluation.generation.assets import load_assets
from official.evaluation.generation.common import digest, immutable_write, require
from official.evaluation.generation.compatibility import member, verify_member
from official.evaluation.generation.metrics import generation_payload
from official.evaluation.generation.native_observer import NativeGenerationObserver, read_observed
from official.evaluation.generation.native_profile import PROFILE, ROUTE, runtime_identity
from official.evaluation.generation.progress import FIELDS, PHASES
from official.tracking import official_generation_progress

SCHEMA = 'official-server2-native-generation-v1'
STUDY = 'USER-OFFICIAL-BASELINES-20261008-R1'
COHORT = 'OFFICIAL_CF_FIRST2000'


def source_identity():
    """Actual imported official generator/scorer bytes, not an old source label."""
    package = ROOT / 'evaluation/generation'
    files = {str(path.relative_to(ROOT)): file_sha(path) for path in sorted(package.glob('*.py'))
             if not path.name.startswith('test_')}
    return dict(namespace='official.evaluation.generation', members=files,
                sources_sha256=file_sha(ROOT / 'SOURCES.json'))


def model_identity(manifest):
    require(manifest['model'] == 'gptj', 'OFFICIAL_GENERATION_MODEL_FAMILY')
    payload = [row for row in manifest['model_assets']
               if Path(row.get('snapshot_path', row['path'])).name == 'pytorch_model.bin']
    require(len(payload) == 1, 'OFFICIAL_GENERATION_MODEL_PAYLOAD_IDENTITY')
    return dict(model_id=manifest['model_id'], model_revision=manifest['model_revision'],
        model_payload_sha256=payload[0]['sha256'], tokenizer_sha256=manifest['tokenizer_sha256'],
        scientific_runtime=dict(torch=manifest['runtime']['torch'],
                                transformers=manifest['runtime']['transformers']),
        precision=dict(weights='float32', attention='eager', autocast=False, tf32=False))


def configuration(manifest):
    generation = manifest['generation']
    require(generation['profile'] == PROFILE and generation['eval_seed'] == 20261007
        and generation['generation_route'] == ROUTE
        and generation['schedule'] == 'CF_W0_ONCE_PER_MODEL_AND_W20_PER_CHAIN',
        'OFFICIAL_GENERATION_PROFILE_SCHEDULE')
    source = source_identity()
    return dict(model_identity=model_identity(manifest), profile=PROFILE, eval_seed=20261007,
        generation_route=ROUTE, generation_source_sha=digest(source),
        reference_assets_sha256=generation['reference_assets_sha256'])


def _records(manifest, records):
    records = list(records)
    require(len(records) == 2000 and [row.get('occurrence_index') for row in records]
            == list(range(1, 2001)), 'OFFICIAL_GENERATION_FULL_FIRST2000_REQUIRED')
    stream = manifest['streams']['cf']
    require(file_sha(stream['path']) == stream['lock']['stream_sha256'],
            'OFFICIAL_GENERATION_STREAM_BYTES_CHANGED')
    require(records == read(stream['path']), 'OFFICIAL_GENERATION_RECORD_TOKEN_TARGET_ORDER_IDENTITY')
    return records


def _state_batch(state):
    require(type(state) is dict, 'OFFICIAL_GENERATION_PHYSICAL_STATE_REQUIRED')
    values = [state[name] for name in ('batch', 'completed_batch') if name in state]
    require(values and all(type(value) is int for value in values) and len(set(values)) == 1,
            'OFFICIAL_GENERATION_STATE_BATCH_REQUIRED')
    return values[0]


def _w0_state(manifest):
    return dict(physical_model=model_identity(manifest), completed_batch=0,
        actual_model_edits=0, actual_applied_edits=0, model_state='COLD_W0')


def _progress(log, *, endpoint):
    keys = {'generation_progress/' + name for name in FIELDS} | {'phase'}
    def callback(value):
        require(set(value) == keys and value['phase'] in PHASES,
                'OFFICIAL_GENERATION_PROGRESS_SCHEMA')
        require(all(type(item) in (int, float) and math.isfinite(item) and item >= 0
                    for key, item in value.items() if key != 'phase'),
                'OFFICIAL_GENERATION_PROGRESS_PRIVACY')
        # The common owner adapts only W20's observational phase name. Native
        # sampling, raw callback counts/axis and endpoint identity stay intact.
        mapped = official_generation_progress(value, endpoint=endpoint)
        if log is not None:
            log(mapped)
    return callback


def payload(observed, endpoint):
    require(endpoint in ('W0', 'W20'), 'OFFICIAL_GENERATION_ENDPOINT')
    require(observed['summary']['planned_count'] == 2000, 'OFFICIAL_GENERATION_PARTIAL_METRIC_FORBIDDEN')
    prefix, edits = ('W0_first2000', 0) if endpoint == 'W0' else ('all_seen/post', 2000)
    return dict(generation_payload(prefix, observed['summary']), edits=edits,
                pre_state_edits=0 if endpoint == 'W0' else 1900, post_state_edits=edits)


def _complete(observed, records, endpoint, runtime_sha):
    require(observed['identity']['runtime'] == runtime_sha
        and observed['identity']['endpoint'] == endpoint
        and observed['identity']['cohort'] == COHORT
        and observed['identity']['ordered_occurrences'] == list(range(1, 2001))
        and len(observed['rows']) == 2000 and observed['summary']['planned_count'] == 2000
        and [row['occurrence'] for row in observed['rows']] == list(range(1, 2001))
        and [row['case_id'] for row in observed['rows']] == [row['case_id'] for row in records]
        and observed['RNG_restored'] is True and observed['observer_no_mutation'] is True
        and 'native_execution_member' in observed and 'parent_endpoint_member' not in observed
        and observed['raw_local_only'] is True, 'OFFICIAL_GENERATION_COMPLETE_ENDPOINT')


def observe(model, tok, manifest, records, out, endpoint, state_identity, state_callback, log=None):
    """Run one native endpoint and return its unmodified full receipt.

    ``state_callback`` guards actual W/H/context/cache during sampling. The W0
    observation identity is canonical cold-model bytes (not method pointers), so
    a complete new-study W0 can be reused by all six cold official chains.
    """
    require(endpoint in ('W0', 'W20'), 'OFFICIAL_GENERATION_W0_OR_W20_ONLY')
    require(callable(state_callback), 'OFFICIAL_GENERATION_NONMUTATION_CALLBACK_REQUIRED')
    require(_state_batch(state_identity) == (0 if endpoint == 'W0' else 20),
            'OFFICIAL_GENERATION_W20_REQUIRES_TWENTY_COMMITS')
    records = _records(manifest, records)
    out = Path(out).absolute()
    require(not out.is_symlink(), 'OFFICIAL_GENERATION_OUTPUT_SYMLINK')
    config = configuration(manifest)
    native_state = _w0_state(manifest) if endpoint == 'W0' else state_identity
    if (out / 'READY.json').exists():
        # A completed native endpoint is a cache hit; partial rows are not.
        observed = _reuse_ready(out / 'READY.json', manifest, records, endpoint,
                                native_state=native_state)
        if log is not None:
            log(payload(observed, endpoint))
        return observed
    started = time.monotonic()
    assets = load_assets(manifest['generation'])
    assets_seconds = time.monotonic() - started
    require(assets.sha == config['reference_assets_sha256'], 'OFFICIAL_GENERATION_ASSETS_IDENTITY')
    observer = NativeGenerationObserver(model, tok, assets, config, out / 'observations',
        state_callback=state_callback, progress_callback=_progress(log, endpoint=endpoint))
    observed = observer.observe(records, endpoint, COHORT, native_state)
    _complete(observed, records, endpoint, observer.runtime_sha)
    # Full raw/execution reader establishes completeness before publishing READY.
    loaded = observer.read_observed(observed['rows_path'])
    _complete(loaded, records, endpoint, observer.runtime_sha)
    identity = dict(schema=SCHEMA, study_instruction=STUDY, endpoint=endpoint,
        model_identity=config['model_identity'], runtime_identity=runtime_identity(config, assets.sha),
        runtime_sha256=observer.runtime_sha, source_identity=source_identity(),
        reference_assets_sha256=assets.sha, stream_sha256=manifest['streams']['cf']['lock']['stream_sha256'],
        physical_state_sha256=digest(native_state), full_requests=2000,
        generation_schedule='CF_W0_ONCE_PER_MODEL_AND_W20_PER_CHAIN',
        old_W20_only_or_partial_reused=False)
    receipt = dict(identity=identity, identity_sha256=digest(identity), status='READY_COMPLETE_ENDPOINT',
        endpoint_member=member(observed['rows_path']), native_execution_member=observed['native_execution_member'],
        actual_state_before_observer=state_identity, summary=observed['summary'],
        work=observed['work'], asset_loading_seconds=assets_seconds,
        RNG_restored=True, observer_no_mutation=True, raw_local_only=True,
        new_generation_observation_performed=observed['work']['new_case_observations'] > 0,
        SDK_acceptance_is_not_remote_delivery=True)
    immutable_write(out / 'READY.json', receipt)
    if log is not None:
        log(payload(observed, endpoint))
    return observed


def _reuse_ready(path, manifest, records, endpoint, *, native_state=None,
                 expected_model_identity=None, expected_runtime=None):
    path = Path(path)
    if path.is_dir():
        path = path / 'READY.json'
    require(path.name == 'READY.json' and path.is_file() and not path.is_symlink(),
            'OFFICIAL_GENERATION_COMPLETE_READY_REQUIRED')
    ready = read(path)
    identity = ready['identity']
    require(ready['identity_sha256'] == digest(identity)
        and ready['status'] == 'READY_COMPLETE_ENDPOINT' and identity['schema'] == SCHEMA
        and identity['study_instruction'] == STUDY and identity['endpoint'] == endpoint
        and identity['generation_schedule'] == 'CF_W0_ONCE_PER_MODEL_AND_W20_PER_CHAIN'
        and identity['old_W20_only_or_partial_reused'] is False
        and identity['source_identity'] == source_identity()
        and identity['stream_sha256'] == manifest['streams']['cf']['lock']['stream_sha256'],
        'OFFICIAL_GENERATION_W0_STUDY_SOURCE_IDENTITY')
    config = configuration(manifest)
    require(expected_model_identity is None or expected_model_identity == config['model_identity'],
            'OFFICIAL_GENERATION_W0_MODEL_IDENTITY')
    runtime = runtime_identity(config, config['reference_assets_sha256'])
    require(identity['model_identity'] == config['model_identity'] and identity['runtime_identity'] == runtime
        and identity['runtime_sha256'] == digest(runtime)
        and (native_state is None or identity['physical_state_sha256'] == digest(native_state))
        and identity['full_requests'] == 2000
        and (expected_runtime is None or expected_runtime in (digest(runtime), runtime)),
        'OFFICIAL_GENERATION_W0_RUNTIME_OR_PHYSICAL_STATE')
    assets = load_assets(manifest['generation'])
    observed = read_observed(verify_member(ready['endpoint_member']),
                             expected_runtime=digest(runtime), assets=assets)
    _complete(observed, records, endpoint, digest(runtime))
    require(observed['identity']['state_sha256'] == identity['physical_state_sha256']
        and observed['summary'] == ready['summary'], 'OFFICIAL_GENERATION_W0_STATE_REDUCTION')
    return observed


def reuse_w0(path, manifest, records, model_identity=None, expected_runtime=None):
    """CPU-only complete new-study W0 adoption; rejects old or partial endpoints."""
    records = _records(manifest, records)
    return _reuse_ready(path, manifest, records, 'W0', native_state=_w0_state(manifest),
                        expected_model_identity=model_identity, expected_runtime=expected_runtime)


def subset(observed, manifest, records, endpoint, cohort=None, out=None):
    """CPU-only native subset; raw provenance retains its complete W0/W20 parent."""
    config = configuration(manifest)
    assets = load_assets(manifest['generation'])
    observer = NativeGenerationObserver(None, None, assets, config,
        Path(out) if out is not None else Path(observed['rows_path']).parent.parent)
    return observer.subset(observed, records, endpoint, cohort)
