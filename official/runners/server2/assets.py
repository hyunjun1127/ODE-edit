"""Bind existing Server2 assets to the official package without loading a model.

Large payload SHA/schema evidence is reused only with an exact unchanged stat.
No EasyEdit algorithm is imported, no assets are downloaded or recomputed, and
input streams/tokenizer audits are written only under the ignored local root.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import zipfile
from pathlib import Path

from official.experiments.prepare import (
    ROOT, audit_tokenizer, digest, file_sha, load_plan, prepare_stream, read, write_new,
)

PROJECT = Path('/mnt/raid5/janghj/ODE-edit')
LOCAL = PROJECT / 'local/official-baselines-server2/20261008-r1'
DEFAULT_OUT = LOCAL / 'preparation-r1'
EASYEDIT = Path('/mnt/raid5/janghj/EasyEdit')
REVISION = '47e169305d2e8376be1d31e765533382721b2cc1'
SNAPSHOT = Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots') / REVISION
PRIOR_CONFIG = PROJECT / 'local/gptj-baselines-fluency-consistency-2k/native-repo-repair-r1/preparation-r1/config.json'
PRIOR_CONFIG_SHA = '0df5795a4d89805620ee2271ff2b355530d0fe98ccfe5a6209160472d80afb02'
PRIOR_ASSET_CHECKS = PROJECT / 'local/gptj-easyedit-native-baselines-2k/preparation-r1/asset-checks.json'
PRIOR_ASSET_CHECKS_SHA = 'a13853fe679b8bfad4ea4d461a1d4face4656e2318c2349e0b93ff38d083ddbc'
REFERENCE_ROOT = PROJECT / 'local/gptj-baselines-fluency-consistency-2k/inputs/reference-r1'
REFERENCE_RECEIPT = REFERENCE_ROOT / 'receipt.json'
REFERENCE_RECEIPT_SHA = '61bea03318f8f80ad65a074a25109912e65c4165451d08e7edd80ded30cb54da'
REFERENCE_MANIFEST = REFERENCE_ROOT / 'reference-ready-r1/manifest.json'
REFERENCE_READY = REFERENCE_ROOT / 'reference-ready-r1/READY.json'
LAYERS = (3, 4, 5, 6, 7, 8)
COUNT = 54924275
SHAPE = [16384, 16384]
TOKENIZER_FILES = ('added_tokens.json', 'merges.txt', 'special_tokens_map.json',
                   'tokenizer.json', 'tokenizer_config.json', 'vocab.json')


class AssetBindingError(RuntimeError):
    def __init__(self, code, path=None):
        self.code, self.path = code, None if path is None else str(path)
        super().__init__(code + (': ' + self.path if self.path else ''))


def require(value, code, path=None):
    if not value:
        raise AssetBindingError(code, path)


def member(path):
    path = Path(path).absolute()
    require(path.is_file(), 'ASSET_MISSING', path)
    resolved = path.resolve()
    before = resolved.stat()
    checksum = file_sha(resolved)
    after = resolved.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'ASSET_CHANGED_DURING_HASH', path)
    return dict(path=str(resolved), bytes=after.st_size, sha256=checksum,
                inode=after.st_ino, mtime_ns=after.st_mtime_ns, device=after.st_dev,
                verification='CURRENT_FULL_SHA256')


def verify_member(row, *, hash_now=False):
    """Validate a sealed file; do not silently convert stat reuse to full SHA."""
    path = Path(row['path'])
    require(path.is_file(), 'SEALED_ASSET_MISSING', path)
    stat = path.stat()
    require(stat.st_size == row['bytes'] and stat.st_ino == row['inode'] and
            stat.st_mtime_ns == row['mtime_ns'], 'SEALED_ASSET_STAT_CHANGED', path)
    if hash_now:
        require(file_sha(path) == row['sha256'], 'SEALED_ASSET_SHA_CHANGED', path)
    return dict(row, device=stat.st_dev, verification=(
        'CURRENT_FULL_SHA256' if hash_now else 'PRIOR_FULL_SHA256_PLUS_CURRENT_UNCHANGED_STAT'))


def read_locked(path, checksum):
    row = member(path)
    require(row['sha256'] == checksum, 'RECEIPT_SHA_CHANGED', path)
    return read(path), row


def stats_metadata(path):
    """Read only the NPZ header and scalar count, not the 1 GiB matrix."""
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == {'mom2.constructor.npy', 'mom2.count.npy',
                    'mom2.mom2.npy', 'sample_size.npy'}, 'C0_NPZ_MEMBERS', path)
        with archive.open('mom2.mom2.npy') as stream:
            version = np.lib.format.read_magic(stream)
            reader = { (1, 0): np.lib.format.read_array_header_1_0,
                       (2, 0): np.lib.format.read_array_header_2_0 }.get(version)
            require(reader is not None, 'C0_NPY_HEADER_VERSION', path)
            shape, fortran, dtype = reader(stream)
        with archive.open('mom2.count.npy') as stream:
            count = np.lib.format.read_array(stream, allow_pickle=False)
        with archive.open('sample_size.npy') as stream:
            sample_size = np.lib.format.read_array(stream, allow_pickle=False)
    require(list(shape) == SHAPE and str(dtype) == 'float32' and not fortran,
            'C0_MATRIX_HEADER', path)
    require(count.size == 1 and float(count.item()) == COUNT, 'C0_COUNT_CHANGED', path)
    require(sample_size.size == 1 and int(sample_size.item()) == 100000,
            'C0_SAMPLE_SIZE_CHANGED', path)
    return dict(shape=SHAPE, dtype='float32', count=COUNT, sample_size=100000,
                stored='mom2_sum', native_covariance='mom2_sum / count',
                content_verification='CURRENT_NPZ_HEADER_AND_SCALARS_PRIOR_FINITE_MATRIX_CHECK')


def validate_stream_lock(actual, expected):
    require(actual == expected, 'OFFICIAL_STREAM_LOCK_MISMATCH')
    require(actual['requests'] == 2000 and actual['batch_size'] == 100 and
            len(actual['batches']) == 20, 'STREAM_HORIZON_CHANGED')
    return actual


def validate_tokenizer_audit(actual, expected):
    require(actual == expected, 'OFFICIAL_TOKENIZER_AUDIT_MISMATCH')
    require(actual['model_forward_calls'] == 0 and actual['lookup_offset_disagreements'] == 0,
            'TOKENIZER_SUBJECT_SPAN_MISMATCH')
    return actual


def defaults():
    return dict(easyedit_root=str(EASYEDIT), model_snapshot=str(SNAPSHOT),
        cf_source=str(PROJECT / 'local/datasets/counterfact-fixed-10k-v1/counterfact.json'),
        zsre_source=str(EASYEDIT / 'data/zsre/zsre_mend_eval.json'),
        generation_reference_manifest=str(REFERENCE_MANIFEST))


def _output(path):
    path = Path(path).absolute()
    require(path == LOCAL or LOCAL in path.parents, 'OUTPUT_OUTSIDE_AUTHORIZED_LOCAL_ROOT', path)
    require(not path.is_symlink(), 'OUTPUT_SYMLINK', path)
    return path


def prepare(out=DEFAULT_OUT, *, overrides=None, tokenizer_audit=True):
    """Create assets.json and canonical CF/zsRE streams, model-free and CPU-only.

    Changed repeated preparation is refused. An existing exact manifest can be
    validated with ``verify``; this is not a native GPU qualification receipt.
    """
    out = _output(out)
    require(tokenizer_audit, 'ACTUAL_TOKENIZER_AUDIT_REQUIRED')
    options = defaults()
    options.update(overrides or {})
    require(set(options) == set(defaults()), 'UNKNOWN_ASSET_PATH_OVERRIDE')
    easyedit, snapshot = Path(options['easyedit_root']), Path(options['model_snapshot'])
    require(easyedit == EASYEDIT and snapshot == SNAPSHOT, 'SERVER2_ASSET_ROOT_CHANGED')
    require(Path(options['generation_reference_manifest']) == REFERENCE_MANIFEST,
            'REFERENCE_MANIFEST_PATH_CHANGED')
    contract, _ = load_plan()
    require(contract['models']['gptj']['revision'] == REVISION, 'OFFICIAL_MODEL_REVISION_CHANGED')
    prior, prior_member = read_locked(PRIOR_CONFIG, PRIOR_CONFIG_SHA)
    checks, checks_member = read_locked(PRIOR_ASSET_CHECKS, PRIOR_ASSET_CHECKS_SHA)
    require(prior['model_revision'] == REVISION and prior['model'] == str(snapshot),
            'PRIOR_MODEL_IDENTITY')
    require(checks['finite'] and checks['GPU'] == 0 and checks['model_loads'] == 0,
            'PRIOR_ASSET_SCHEMA_FINITE_RECEIPT')
    prior_assets = {row['path']: row for row in prior['assets']}
    model_assets = []
    for row in prior['model_assets']:
        checked = verify_member(row, hash_now=row['bytes'] < (32 << 20))
        require(Path(row['snapshot_path']).resolve() == Path(row['path']), 'MODEL_SNAPSHOT_LINK_CHANGED')
        checked['snapshot_path'] = row['snapshot_path']
        model_assets.append(checked)
    tokenizer_files = {name: file_sha(snapshot / name) for name in TOKENIZER_FILES}
    tokenizer_sha = digest(tokenizer_files)
    tokenizer_lock = read(ROOT / 'hparams/tokenizers.lock.json')
    require(tokenizer_sha == tokenizer_lock['audits']['gptj-cf']['tokenizer_sha256'],
            'TOKENIZER_SHA_CHANGED')
    model_config = read(snapshot / 'config.json')
    require(model_config['model_type'] == 'gptj' and model_config['n_layer'] == 28
            and model_config['n_embd'] == 4096 and model_config['n_inner'] is None
            and model_config['vocab_size'] == 50400 and model_config['rotary_dim'] == 64
            and model_config.get('tie_word_embeddings') is False, 'GPTJ_NATIVE_STRUCTURE')
    stats_root = easyedit / 'examples/data/stats/gpt-j-6b/wikipedia_stats'
    assets = {}
    for slot, layer in enumerate(LAYERS):
        path = stats_root / f'transformer.h.{layer}.mlp.fc_out_float32_mom2_100000.npz'
        require(str(path) in prior_assets, 'C0_PRIOR_SHA_MISSING', path)
        row = verify_member(prior_assets[str(path)])
        row.update(stats_metadata(path), layer=layer, physical_slot=slot,
                   provenance_member=checks_member, full_sha_source=prior_member)
        check = next((x for x in checks['stats'] if x['layer'] == layer), None)
        require(check is not None and check['shape'] == SHAPE and check['count'] == COUNT
                and check['dtype'] == 'float32', 'C0_PRIOR_SCHEMA_MAPPING')
        assets[f'C0_L{layer}'] = row
    projector = easyedit / 'examples/null_space_project_gpt-j-6b.pt'
    require(str(projector) in prior_assets, 'PROJECTOR_PRIOR_SHA_MISSING', projector)
    slots = [dict(slot=slot, physical_layer=layer, shape=SHAPE) for slot, layer in enumerate(LAYERS)]
    require(checks['projector_slots'] == slots, 'PROJECTOR_SIX_SLOT_MAPPING')
    assets['projector'] = dict(verify_member(prior_assets[str(projector)]),
        shape=[6, 16384, 16384], dtype='float32', slot_layers=list(LAYERS), slots=slots,
        threshold=0.02, provenance_member=checks_member, full_sha_source=prior_member,
        threshold_provenance='NATIVE_HPARAMS_AND_HISTORICAL_TASK_CONTRACT; ORIGINAL_CONSTRUCTION_RECEIPT_NOT_AVAILABLE',
        spectral_recomputed=False, finite_verification='PRIOR_CHECK_WITH_UNCHANGED_FILE_STAT',
        BLUE_physical_slots=[0, 5], BLUE_history_slots=[0, 1])
    streams = {}
    for dataset in ('cf', 'zsre'):
        source = Path(options[dataset + '_source'])
        locked = read(ROOT / 'hparams' / f'{dataset}-stream.lock.json')
        actual = prepare_stream(source, dataset, out / 'streams', locked['source_sha256'])
        validate_stream_lock(actual, locked)
        stream = out / 'streams' / f'{dataset}-stream.json'
        audit_path = out / 'tokenizers' / f'gptj-{dataset}.json'
        audit = audit_tokenizer(stream, snapshot, audit_path, 'gptj')
        validate_tokenizer_audit(audit, tokenizer_lock['audits'][f'gptj-{dataset}'])
        streams[dataset] = dict(path=str(stream), member=member(stream), lock=actual,
            lock_member=member(out / 'streams' / f'{dataset}-stream.lock.json'),
            source=member(source), tokenizer_audit=member(audit_path), tokenizer_summary=audit)
    receipt, receipt_member = read_locked(REFERENCE_RECEIPT, REFERENCE_RECEIPT_SHA)
    require(receipt['status'] == 'READY_VERIFIED_RECEIVER', 'REFERENCE_RECEIVER_STATUS')
    reference_members = {Path(row['path']).name: verify_member(row) for row in receipt['files']}
    gen_lock = read(ROOT / 'hparams/generation.lock.json')
    manifest = read(REFERENCE_MANIFEST)
    require(manifest['identity_sha256'] == gen_lock['reference_identity_sha256']
            and read(REFERENCE_READY)['identity_sha256'] == manifest['identity_sha256'],
            'GENERATION_REFERENCE_IDENTITY_CHANGED')
    paths = {}
    for name, locked in gen_lock['reference_files'].items():
        row = reference_members[name]
        require(row['bytes'] == locked['bytes'] and row['sha256'] == locked['sha256'],
                'GENERATION_REFERENCE_BYTES_CHANGED')
        paths[name] = row['path']
    scoring_versions = {key: importlib.metadata.version(package) for key, package in
        (('numpy', 'numpy'), ('scipy', 'scipy'), ('sklearn', 'scikit-learn'), ('nltk', 'nltk'))}
    require(scoring_versions == manifest['versions'], 'GENERATION_SCORING_RUNTIME_CHANGED')
    for row in manifest['tokenizer']['required_resources']:
        require(member(row['path'])['sha256'] == row['sha256'], 'NLTK_RESOURCE_CHANGED')
    runtime = dict(prior['runtime'], python=str(easyedit / '.venv/bin/python'),
                   python_version=sys.version.split()[0])
    runtime['members'] = [verify_member(row, hash_now=True) for row in runtime['members']]
    require(importlib.metadata.version('torch') == runtime['torch'].split('+')[0]
            and importlib.metadata.version('transformers') == runtime['transformers'],
            'SCIENTIFIC_RUNTIME_VERSION_CHANGED')
    scientific_files = [member(path) for folder in ('baselines', 'hparams', 'evaluation', 'experiments')
        for path in sorted((ROOT / folder).rglob('*')) if path.is_file() and path.suffix in ('.py', '.json')]
    result = dict(schema='official-server2-assets-v1', status='ASSETS_BOUND_CPU_ONLY', server='server2',
        instruction_id='USER-OFFICIAL-BASELINES-20261008-R1', easyedit_root=str(easyedit),
        python=runtime['python'], model='gptj', model_id=contract['models']['gptj']['model_id'],
        model_snapshot=str(snapshot), model_revision=REVISION, model_identity=contract['models']['gptj'],
        model_assets=model_assets, tokenizer_files_sha256=tokenizer_files, tokenizer_sha256=tokenizer_sha,
        stats_root=str(stats_root), stats_dir=str(easyedit / 'examples/data/stats'), assets=assets,
        streams=streams, runtime=runtime,
        generation=dict(generation_assets=reference_members['manifest.json'], asset_paths=paths,
            reference_identity_sha256=manifest['identity_sha256'], reference_assets_sha256=manifest['identity_sha256'],
            reference_READY=reference_members['READY.json'], receive_receipt=receipt_member,
            profile=gen_lock['profile']['profile'], eval_seed=20261007,
            generation_route='NATIVE_CASE_PADDED_KV_GLOBAL_RNG', scoring_versions=scoring_versions,
            schedule='CF_W0_ONCE_PER_MODEL_AND_W20_PER_CHAIN', new_model_observation_required=True,
            old_W20_only_observation_reused=False),
        provenance=dict(prior_asset_checks=checks_member, prior_config=prior_member,
            official_scientific_members=scientific_files, official_sources=member(ROOT / 'SOURCES.json')),
        GPU=0, model_loads=0, model_forward_calls=0, EasyEdit_algorithm_imports=0,
        downloads=0, generated_C0_P=0, ready_to_submit=False,
        remaining=['native actual GPU smoke/resume', 'projector original construction receipt unavailable'])
    result['assets_sha256'] = digest(result)
    write_new(out / 'assets.json', result)
    return result


def verify_official_source(row, original_root):
    """Exact source bytes in the imported archive; inode/mtime are not portable."""
    try:
        relative = Path(row['path']).relative_to(original_root)
    except ValueError as error:
        raise AssetBindingError('OFFICIAL_SOURCE_MEMBER_SCOPE', row['path']) from error
    require(not relative.is_absolute() and '..' not in relative.parts,
            'OFFICIAL_SOURCE_MEMBER_SCOPE', row['path'])
    path = ROOT/relative
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
            and file_sha(path) == row['sha256'], 'IMPORTED_OFFICIAL_SOURCE_SHA_CHANGED', path)


def verify(manifest):
    """Cheap reentry check used by frozen runners before opening model/assets."""
    value = read(manifest) if isinstance(manifest, (str, Path)) else manifest
    require(value.get('schema') == 'official-server2-assets-v1', 'ASSET_MANIFEST_SCHEMA')
    identity = dict(value)
    checksum = identity.pop('assets_sha256', None)
    require(checksum == digest(identity), 'ASSET_MANIFEST_IDENTITY')
    for row in value['model_assets'] + list(value['assets'].values()):
        verify_member(row, hash_now=row['bytes'] < (32 << 20))
    for stream in value['streams'].values():
        verify_member(stream['member'], hash_now=True)
        verify_member(stream['source'], hash_now=True)
        verify_member(stream['lock_member'], hash_now=True)
    # An immutable production Git archive has different source inode/mtime.
    # Validate its actual imported relative member bytes, not a mutable WT.
    original_root = Path(value['provenance']['official_sources']['path']).parent
    for row in [*value['provenance']['official_scientific_members'],
                value['provenance']['official_sources']]:
        verify_official_source(row, original_root)
    for row in value['runtime']['members']:
        verify_member(row, hash_now=True)
    generation = value['generation']
    for row in (generation['generation_assets'], generation['reference_READY'],
                generation['receive_receipt']):
        verify_member(row, hash_now=True)
    receipt = read(generation['receive_receipt']['path'])
    for row in receipt['files']:
        verify_member(row)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    try:
        value = prepare(args.out)
    except AssetBindingError as error:
        print(json.dumps(dict(status='ASSET_BLOCKED', code=error.code, path=error.path)))
        raise SystemExit(2) from error
    print(json.dumps(dict(status=value['status'], manifest=str(args.out / 'assets.json'),
        assets_sha256=value['assets_sha256'], model='gptj', streams=['cf', 'zsre'], GPU=0,
        ready_to_submit=False), indent=2))


if __name__ == '__main__':
    main()
