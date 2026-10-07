"""Explicit old-cold-W0 reuse; original raw bytes/provenance are never rewritten."""
import copy
import hashlib
import json
from pathlib import Path

from .common import SCHEMA as METRIC_SCHEMA, PROFILE, EVAL_SEED, case_seed, digest, immutable_write, require

SCHEMA = 'generation-cold-W0-compatibility-v1'
REFERENCE_ROUTE = 'UNPADDED_FULL_PREFIX_NO_CACHE'


def runtime_identity(config, assets_sha=None):
    """Pure source/route builder; qualification verification is separate.

    This API does not import torch, inspect a model, or access a GPU. Legacy
    callers retain the exact old no-cache identity bytes.
    """
    source = config.get('generation_source_sha', config.get('source_identity'))
    identity = dict(schema=METRIC_SCHEMA, profile=PROFILE, eval_seed=EVAL_SEED,
        model_identity=config['model_identity'], generation_source_sha=source,
        reference_assets_sha256=assets_sha if assets_sha is not None else config['reference_assets_sha256'],
        route=config.get('generation_route', REFERENCE_ROUTE))
    if 'qualification_receipt_member' in config:
        identity.update(generation_microbatch=config['generation_microbatch'],
            qualification_receipt_sha256=config['qualification_receipt_member']['sha256'])
    require(identity['route'] != 'QUALIFICATION_REQUIRED', 'GENERATION_ACTUAL_QUALIFICATION_REQUIRED')
    return identity


def member(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), 'GENERATION_MEMBER_REGULAR_FILE')
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'GENERATION_MEMBER_CHANGED')
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def verify_member(row):
    core = {'path', 'bytes', 'sha256'}
    metadata = {'inode', 'mtime_ns'}
    require(type(row) is dict and core <= set(row) <= core | metadata
            and type(row.get('path')) is str and Path(row['path']).is_absolute()
            and type(row.get('bytes')) is int and row['bytes'] >= 0
            and type(row.get('sha256')) is str and len(row['sha256']) == 64
            and all(char in '0123456789abcdef' for char in row['sha256'])
            and all(type(row[key]) is int and row[key] >= 0 for key in metadata if key in row),
            'GENERATION_MEMBER_SCHEMA')
    path = Path(row['path'])
    # Native caller members contain the same core plus optional inode/mtime.
    # These metadata fields strengthen the identity; they are not extra content
    # keys against which the three-field content member should be compared.
    require(path.is_file() and not path.is_symlink(), 'GENERATION_MEMBER_REGULAR_FILE')
    before = path.stat()
    actual = member(path)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'GENERATION_MEMBER_CHANGED')
    require(actual == {key: row[key] for key in core}, 'GENERATION_MEMBER_BYTES_IDENTITY')
    for key, value in (('inode', after.st_ino), ('mtime_ns', after.st_mtime_ns)):
        require(key not in row or row[key] == value, 'GENERATION_MEMBER_METADATA_IDENTITY')
    return path


def qualification_binding(config):
    item = config.get('qualification_receipt_member')
    require(item is not None, 'GENERATION_ACTUAL_QUALIFICATION_REQUIRED')
    from .kv_qualification import verify_actual_receipt
    actual = verify_actual_receipt(item, expected_plan_sha256=config.get('qualification_plan_sha256'),
        expected_model_identity=config['model_identity'],
        allow_cpu_fixture=config.get('qualification_allow_cpu_fixture', False))
    require(actual.get('qualification_pass') is True
            and actual.get('selected_route') == config['generation_route']
            and actual.get('fixed_microbatch') == config['generation_microbatch'],
            'GENERATION_ACTUAL_QUALIFICATION_SELECTION')
    return copy.deepcopy(item)


def verify_raw_complete(raw, record_identity, *, runtime, state, model_identity,
                        expected_route=None):
    require(type(raw) is dict and raw.get('identity') == dict(runtime=runtime,
            state_identity=state, record_identity=record_identity)
            and raw.get('identity_sha256') == digest(raw['identity'])
            and raw.get('payload_sha256') == digest({k:v for k,v in raw.items() if k != 'payload_sha256'})
            and raw.get('occurrence') == record_identity['ordered_occurrence']
            and raw.get('case_id') == record_identity['case_id']
            and raw.get('raw_local_only') is True and raw.get('checkpoint_saved') is False,
            'GENERATION_COMPATIBILITY_RAW_IDENTITY')
    observations = raw.get('observations')
    prompts = record_identity['generation_prompts']
    require(type(observations) is list and len(observations) == len(prompts),
            'GENERATION_COMPATIBILITY_CASE_INCOMPLETE')
    for index, (prompt, row) in enumerate(zip(prompts, observations)):
        require(row.get('profile') == PROFILE and row.get('prompt') == prompt
                and row.get('occurrence') == record_identity['ordered_occurrence']
                and row.get('prompt_index') == index
                and row.get('seed') == case_seed(model_identity, record_identity['ordered_occurrence'], index, EVAL_SEED)
                and row.get('RNG_restored') is True
                and (expected_route is None or row.get('route') == expected_route),
                'GENERATION_COMPATIBILITY_PROMPT_SEED_ROUTE')
        inputs, continuation, full = row.get('input_token_ids'), row.get('continuation_token_ids'), row.get('full_token_ids')
        require(type(inputs) is list and bool(inputs) and type(continuation) is list
                and full == inputs+continuation
                and all(type(token) is int and token >= 0 for token in full)
                and row.get('input_token_count') == len(inputs)
                and row.get('continuation_token_count') == len(continuation)
                and type(row.get('text')) is str and row.get('sampling') ==
                    dict(top_k=5, temperature=1, top_p=1, max_total_tokens=100),
                'GENERATION_COMPATIBILITY_TOKEN_SCHEMA')
        stop = row.get('stop_reason')
        eos = row.get('eos_ids')
        require(type(eos) is list and all(type(x) is int for x in eos)
            and ((stop == 'length_cap_no_continuation' and len(inputs) >= 100 and not continuation)
                or (stop == 'length_cap' and len(inputs) < 100 and len(full) == 100
                    and not any(x in eos for x in continuation))
                or (stop == 'eos' and bool(continuation) and len(full) <= 100
                    and continuation[-1] in eos and not any(x in eos for x in continuation[:-1]))),
            'GENERATION_COMPATIBILITY_COMPLETED_STOP')
    return raw


def verify_scored(raw, assets):
    from .metrics import score_case
    ri = raw['identity']['record_identity']
    refs = None if assets is None else assets.snippets_for(ri['relation_id'], ri['target_new_id'])
    actual = score_case(raw['observations'], refs, getattr(assets, 'vectorizer', None),
                        getattr(assets, 'word_tokenize', None))
    expected = raw['metrics']
    require(set(actual) == set(expected), 'GENERATION_COMPATIBILITY_METRIC_SCHEMA')
    for key, value in actual.items():
        if key in ('ngram_entropy', 'reference_score') and value is not None:
            require(type(expected[key]) in (int, float) and abs(value-expected[key]) <= 1e-6,
                    'GENERATION_COMPATIBILITY_METRIC_VALUE')
        else:
            require(value == expected[key], 'GENERATION_COMPATIBILITY_METRIC_VALIDITY')


class W0Compatibility:
    """Built once for an exact cold cohort after an actual qualification receipt.

    Unknown/corrupt/missing rows are recorded as unavailable and regenerated, not
    relabelled. Matching original rows remain at their original paths and runtimes.
    """
    def __init__(self, observer, records, ordinals, state):
        spec = observer.config['old_w0_reuse']
        require(spec.get('allowed_only_state_W') == state.get('W') and state.get('H') == {},
                'GENERATION_COMPATIBILITY_COLD_W0_ONLY')
        observer_member = spec['observer_identity_member']
        old_identity = json.loads(verify_member(observer_member).read_text())
        runtime = old_identity['identity']
        require(old_identity['identity_sha256'] == digest(runtime)
            and runtime.get('route') == REFERENCE_ROUTE
            and runtime.get('model_identity') == observer.model_identity
            and runtime.get('profile') == PROFILE and runtime.get('eval_seed') == EVAL_SEED
            and runtime.get('reference_assets_sha256') == observer.assets_sha,
            'GENERATION_COMPATIBILITY_RUNTIME_BINDING')
        old_config_member = spec['config_member']
        old_config = json.loads(verify_member(old_config_member).read_text())
        cfg = old_config['generation']
        require(cfg['generation_source_sha'] == runtime['generation_source_sha']
            and cfg['model_identity'] == runtime['model_identity']
            and cfg['reference_assets_sha256'] == runtime['reference_assets_sha256']
            and cfg['profile'] == runtime['profile'] and cfg['eval_seed'] == runtime['eval_seed'],
            'GENERATION_COMPATIBILITY_OLD_SOURCE_CONFIG')
        # This original reference manifest binds exact scoring versions/resources.
        old_reference = json.loads(verify_member(cfg['assets_manifest_member']).read_text())
        require(old_reference['identity_sha256'] == observer.assets_sha,
                'GENERATION_COMPATIBILITY_SCORING_RUNTIME')
        actual_manifest = getattr(observer.assets, 'manifest', None)
        if actual_manifest is not None:
            require(old_reference['versions'] == actual_manifest['versions']
                and old_reference['tokenizer'] == actual_manifest['tokenizer'],
                'GENERATION_COMPATIBILITY_SCORING_RUNTIME')
        # Source cancellation audit must establish that these atomically completed
        # case rows belong to the original cold W0 phase, not a RAM edit trajectory.
        guard_member = spec['cold_observation_guard_member']
        guard = json.loads(verify_member(guard_member).read_text())
        require(guard.get('source_commit') == spec['source_commit']
            and guard.get('phase') == 'W0_generation' and guard.get('commits') == 0
            and guard.get('history_appends') == 0 and guard.get('model_W') == state['W']
            and guard.get('old_generation_runtime') == old_identity['identity_sha256'],
            'GENERATION_COMPATIBILITY_OLD_COLD_PHASE_GUARD')
        require(guard.get('whole_endpoint_guard_recorded') is False
            and guard.get('proof_basis') == 'FROZEN_SOURCE_CONTROL_FLOW_COLD_RUNTIME_AND_RPN'
            and guard.get('partial_rows_authorized') is True,
            'GENERATION_COMPATIBILITY_PARTIAL_GUARD_DISCLOSURE')
        root = Path(spec['observations_root'])
        require(root.is_dir() and not root.is_symlink(), 'GENERATION_COMPATIBILITY_ROOT')
        entries, exclusions = [], []
        self.lookup = {}
        from .observer import _record_identity
        for record, ordinal in zip(records, ordinals):
            ri = _record_identity(record, ordinal)
            identity = dict(runtime=old_identity['identity_sha256'], state_identity=state, record_identity=ri)
            path = root/(digest(identity)+'.json')
            if not path.exists():
                exclusions.append(dict(occurrence=ordinal, reason='OLD_COMPLETE_CASE_NOT_AVAILABLE'))
                continue
            try:
                file_member = member(path)
                raw = json.loads(path.read_text())
                verify_raw_complete(raw, ri, runtime=old_identity['identity_sha256'], state=state,
                    model_identity=observer.model_identity, expected_route=REFERENCE_ROUTE)
                verify_scored(raw, observer.assets)
            except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
                exclusions.append(dict(occurrence=ordinal, reason='OLD_ROW_UNVERIFIED',
                                       error_type=type(error).__name__))
                continue
            entry = dict(occurrence=ordinal, original_raw_member=file_member,
                original_identity_sha256=raw['identity_sha256'], original_payload_sha256=raw['payload_sha256'],
                original_runtime_sha256=old_identity['identity_sha256'],
                original_generation_source_sha=runtime['generation_source_sha'], original_route=REFERENCE_ROUTE)
            entries.append(entry)
            self.lookup[ordinal] = (raw, path, entry)
        identity = dict(schema=SCHEMA, physical_state=state, model_identity=observer.model_identity,
            old_observer_member=observer_member, old_config_member=old_config_member,
            old_reference_member=cfg['assets_manifest_member'], old_cold_guard_member=guard_member,
            new_runtime_sha256=observer.runtime_sha, reference_assets_sha256=observer.assets_sha,
            qualification_receipt_member=qualification_binding(observer.config),
            ordered_record_identity_sha256=digest([_record_identity(r,o) for r,o in zip(records,ordinals)]),
            original_entries=entries, excluded=exclusions, eligible_cases=len(entries),
            planned_cases=len(records), raw_local_only=True, edited_trajectory_resume=False)
        identity.update(old_whole_endpoint_guard_recorded=False,
            old_partial_evidence='SOURCE_CONTROL_FLOW_COLD_RUNTIME_RPN_AND_ATOMIC_COMPLETE_ROWS',
            new_native_qualification_and_observer_guard_required=True)
        if 'runtime_member' in spec:
            verify_member(spec['runtime_member'])
            identity['old_runtime_member'] = copy.deepcopy(spec['runtime_member'])
        key = digest(identity)
        path = observer.out/'compatibility'/(key+'.json')
        immutable_write(path, dict(identity=identity, identity_sha256=key))
        self.member, self.sha = member(path), key

    def reuse(self, ordinal):
        row = self.lookup.get(ordinal)
        if row is None:
            return None
        raw, path, entry = row
        verify_member(entry['original_raw_member'])
        return copy.deepcopy(raw), path, copy.deepcopy(entry)


def load_compatibility(item, expected_runtime=None, expected_qualification=None):
    value = json.loads(verify_member(item).read_text())
    identity = value['identity']
    require(value['identity_sha256'] == digest(identity) and identity['schema'] == SCHEMA
        and identity['raw_local_only'] is True and identity['edited_trajectory_resume'] is False,
        'GENERATION_COMPATIBILITY_MANIFEST_IDENTITY')
    if expected_runtime is not None:
        require(identity['new_runtime_sha256'] == expected_runtime, 'GENERATION_RUNTIME_IDENTITY')
    q = identity['qualification_receipt_member']
    verify_member(q)
    if expected_qualification is not None:
        require(q['sha256'] == expected_qualification, 'GENERATION_COMPATIBILITY_QUALIFICATION_SHA')
    for key in ('old_observer_member', 'old_config_member', 'old_reference_member', 'old_cold_guard_member'):
        verify_member(identity[key])
    if 'old_runtime_member' in identity:
        verify_member(identity['old_runtime_member'])
    require(len({x['occurrence'] for x in identity['original_entries']}) == len(identity['original_entries']),
            'GENERATION_COMPATIBILITY_OCCURRENCE_DUPLICATE')
    return value


def verified_endpoint_row(row, endpoint, *, expected_record_identity=None,
                          expected_state=None):
    """Independent CPU collector primitive; validates original row and provenance.

    No observer/model/torch import is necessary. It returns the original raw
    document, never a relabelled copy. The collector still independently reduces
    the original metric values/counts and validates its native cohort identities.
    """
    require(endpoint['identity_sha256'] == digest(endpoint['identity']),
            'GENERATION_ENDPOINT_IDENTITY')
    path = Path(row['observation_path'])
    raw = json.loads(path.read_text())
    identity = endpoint['identity']
    require(raw['identity_sha256'] == row['identity_sha256'] == digest(raw['identity'])
        and raw['payload_sha256'] == row['payload_sha256']
        and raw['payload_sha256'] == digest({k:v for k,v in raw.items() if k != 'payload_sha256'})
        and raw['occurrence'] == row['occurrence'] and raw['case_id'] == row['case_id']
        and raw['metrics'] == row['metrics']
        and digest(raw['identity']['state_identity']) == identity['state_sha256'],
        'GENERATION_OBSERVATION_BYTES_IDENTITY')
    if expected_record_identity is not None:
        require(raw['identity']['record_identity'] == expected_record_identity,
                'GENERATION_SUBSET_PROMPT_IDENTITY')
    if expected_state is not None:
        require(raw['identity']['state_identity'] == expected_state, 'GENERATION_STATE_IDENTITY')
    provenance = row.get('provenance')
    if provenance is not None:
        require(provenance['raw_member']['path'] == row['observation_path']
            and provenance['runtime_sha256'] == raw['identity']['runtime'],
            'GENERATION_ROW_PROVENANCE')
        verify_member(provenance['raw_member'])
    if raw['identity']['runtime'] != identity['runtime']:
        require('compatibility_member' in endpoint, 'GENERATION_RUNTIME_IDENTITY')
        value = load_compatibility(endpoint['compatibility_member'],
            expected_runtime=identity['runtime'],
            expected_qualification=identity['qualification_receipt_sha256'])
        require(value['identity_sha256'] == identity['compatibility_sha256'],
                'GENERATION_COMPATIBILITY_ENDPOINT_SHA')
        lookup = {entry['occurrence']:entry for entry in value['identity']['original_entries']}
        entry = lookup.get(row['occurrence'])
        require(entry is not None
            and entry['original_raw_member']['path'] == row['observation_path']
            and entry['original_runtime_sha256'] == raw['identity']['runtime']
            and entry['original_identity_sha256'] == raw['identity_sha256']
            and entry['original_payload_sha256'] == raw['payload_sha256'],
            'GENERATION_COMPATIBILITY_ROW_ORIGIN')
        verify_member(entry['original_raw_member'])
        require(provenance is not None and provenance['origin'] == 'COMPATIBLE_ORIGINAL_W0'
            and provenance['generation_source_sha'] == entry['original_generation_source_sha']
            and provenance['route'] == entry['original_route'], 'GENERATION_ROW_PROVENANCE')
    return raw
