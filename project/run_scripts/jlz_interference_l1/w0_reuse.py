"""Hash-bound, read-only reuse of job 59721's complete cold W0 observation.

Only scalar rows are read: no model construction, forward, tensor checkpoint,
raw copying, hardlink or symlink. Qualification covers this observation, not the
failed fit, repaired projection, or a whole-source GPU qualification.
"""
import argparse
import ast
import json
from pathlib import Path

from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader, validate_rows, reduce_rows, compare_summary, active_flags)
from scripts.fixed_counterfact import load_prefix
from . import ROOT, TASK, ORDERED_SHA, require, digest, member, sha, expected_rows
from .storage import write, CHUNK_BYTES, CHUNK_ROWS

PRIOR_SOURCE = 'a9905b9fccbdb48b9b17e768368afa9e663f0bf5'
PRIOR_CONFIG_SHA = 'ad6ca0f4987ecfd400c4b637682ba2dc1a8a2626f22de8594b8d7cadb1b7511d'
PRIOR_LOCK_SHA = 'c66173da14839e078660a51166fa2169d8d5220f403d72192f1289411bb49f7a'
STATUS = 'QUALIFIED_EXACT_REUSE'
REVISION = '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
NAMESPACE = 'project/run_scripts/jlz_interference_l1/'


def _file_identity(row, path=False, stat=False):
    keys = ['bytes', 'sha256'] + (['path'] if path else []) + (['inode', 'mtime_ns'] if stat else [])
    return {key: row[key] for key in keys}


def observation_fingerprint(c):
    """Exclude repaired optimizer/controller, attempt path and resource metadata."""
    runtime = c['runtime']
    profile = c['arm_profiles']['PRICE']
    return digest(dict(
        model=c['model'], revision=REVISION, stream=c['stream'], contexts=c['contexts'],
        seed=c['seed'], cold=c['cold_W0_H0'], ordered_ids=c['ordered_ids_sha256'],
        observer_microbatch=c['settings']['observer_microbatch'],
        requests=c['settings']['requests'], B=c['settings']['B'], packs=c['packs'],
        profile=profile,
        assets=[_file_identity(row, path=True, stat=True) for row in c['assets']],
        input_receipts={key: _file_identity(c[key]) for key in
                        ('observer_identity', 'native_input_alignment', 'native_full_input_binding')},
        scoring=[_file_identity(row, path=True) for row in c['readonly_input_evaluator_sources']],
        native=[_file_identity(row, path=True) for row in c['native_reference']],
        hparams=c['native_hparams'],
        runtime={key: runtime[key] for key in ('python', 'python_version', 'torch', 'torch_cuda',
            'transformers', 'attention', 'autocast', 'cudnn_TF32', 'matmul_TF32',
            'model_dtype', 'geometry_dtype', 'source_root_sha256')},
        runtime_sources=[_file_identity(row, path=True) for row in runtime['source_members']],
    ))


def _bound(reader, row):
    data = reader.bytes(row['path'])
    require(len(data) == row['bytes'] and sha(row['path']) == row['sha256'], 'W0_REUSE_MEMBER_CHANGED')
    return data


def _assets(c):
    for row in c['assets']:
        st = Path(row['path']).stat()
        require((st.st_size, st.st_ino, st.st_mtime_ns) ==
                (row['bytes'], row['inode'], row['mtime_ns']), 'W0_REUSE_ASSET_STAT_CHANGED')
    # Prior full hashes are reused, not recomputed for multi-GB model/C0 assets.


def _functions(path):
    tree = ast.parse(Path(path).read_text())
    names = ('setup', 'observer', 'rng_identity')
    return {name: digest(ast.dump(next(node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name),
        include_attributes=False)) for name in names}


def _source_closure(reader, prior, lock):
    root = prior / 'source'
    rows = []
    for row in lock['source_members']:
        relative = Path(row['path']).relative_to(root).as_posix()
        # Source changes in the repair orchestration/projection are intentional.
        # All inherited Python import bytes plus the actual scoring writer remain
        # exact; design/policy documents are not observation executable closure.
        eligible = relative.endswith('.py') and (relative.startswith('project/run_scripts/')
                    or relative.startswith('scripts/'))
        if not eligible or (relative.startswith(NAMESPACE) and relative != NAMESPACE + 'observer_io.py'):
            continue
        _bound(reader, row)
        current = ROOT / relative
        require(current.is_file() and not current.is_symlink() and
                current.stat().st_size == row['bytes'] and sha(current) == row['sha256'],
                'W0_REUSE_SCORING_SOURCE_CHANGED:' + relative)
        rows.append(dict(relative=relative, prior=member(row['path']), current=member(current)))
    require(any(row['relative'] == NAMESPACE + 'observer_io.py' for row in rows), 'W0_REUSE_OBSERVER_CLOSURE')
    functions = _functions(root / (NAMESPACE + 'run.py'))
    require(_functions(ROOT / (NAMESPACE + 'run.py')) == functions, 'W0_REUSE_SETUP_OBSERVER_AST_CHANGED')
    return rows, functions


def seal_w0_reuse(prior_attempt, new_config):
    prior = Path(prior_attempt).resolve()
    require(sha(prior/'config.json') == PRIOR_CONFIG_SHA and
            sha(prior/'execution.lock.json') == PRIOR_LOCK_SHA, 'W0_REUSE_PRIOR_IDENTITY')
    reader = Reader()
    old = reader.json(prior/'config.json'); lock = reader.json(prior/'execution.lock.json')
    require(lock['source_commit'] == PRIOR_SOURCE and lock['config_sha256'] == PRIOR_CONFIG_SHA
            and old['task_id'] == new_config['task_id'] == TASK, 'W0_REUSE_PRIOR_SOURCE')
    require(not list((prior/'PRICE').glob('batch-*/commit.json')), 'W0_REUSE_PRIOR_NOT_COLD_ONLY')
    require(Path(old['model']).name == REVISION and old['settings']['observer_microbatch'] == 2,
            'W0_REUSE_MODEL_OR_MICROBATCH')
    fingerprint = observation_fingerprint(old)
    require(observation_fingerprint(new_config) == fingerprint, 'W0_REUSE_OBSERVATION_CONTRACT_CHANGED')
    _assets(new_config)
    closure, functions = _source_closure(reader, prior, lock)
    for row in old['runtime']['source_members'] + old['native_reference'] + old['readonly_input_evaluator_sources']:
        _bound(reader, row)
    _bound(reader, lock['native_hparams'])
    for key in ('native_input_alignment', 'native_full_input_binding', 'observer_identity'):
        _bound(reader, old[key])
    runtime = reader.json(prior/'PRICE/runtime.json')
    require(runtime['source'] == PRIOR_SOURCE and runtime['config'] == digest(old)
            and runtime['job'] == '59721' and runtime['arm'] == 'PRICE'
            and runtime['profile'] == digest(old['arm_profiles']['PRICE'])
            and runtime['cold_W0_H0'] == old['cold_W0_H0'] and runtime['model'] == old['model']
            and runtime['torch'] == old['runtime']['torch']
            and runtime['transformers'] == old['runtime']['transformers']
            and runtime['FP32'] and runtime['geometry_FP64'] and runtime['eager']
            and runtime['TF32'] is False and runtime['autocast'] is False
            and runtime['checkpoint_saved'] is False, 'W0_REUSE_ACTUAL_RUNTIME')
    records = load_prefix(Path(old['stream']).parent, 2000)
    ids = [r['case_id'] for r in records]
    require(digest(ids) == old['ordered_ids_sha256'] == ORDERED_SHA, 'W0_REUSE_FIXED_ORDER')
    identities = reader.json(old['observer_identity']['path'])['rows']
    folder = prior/'PRICE/W0'
    paths = sorted(folder.glob('chunk-*.json'))
    require([p.name for p in paths] == [f'chunk-{start:04d}.json' for start in range(0,2000,50)],
            'W0_REUSE_EXACT_40_CHUNKS')
    rows = []
    for path in paths:
        require(path.stat().st_size <= CHUNK_BYTES, 'W0_REUSE_CHUNK_BYTES')
        obj = reader.json(path)
        require(obj['state'] == old['cold_W0_H0'] and obj['optimizer_feedback'] is False
                and len(obj['rows']) == CHUNK_ROWS, 'W0_REUSE_RAW_STATE_CARDINALITY')
        rows.extend(obj['rows'])
    validate_rows(rows, expected_rows(identities, ids), 'W0')
    require(all(r['margin_new_minus_true'] == r['new_nll']-r['true_nll'] for r in rows), 'W0_REUSE_MARGIN_SIGN')
    flags = active_flags(records)
    require(all(r['active_at_endpoint'] == flags[r['case_id']] for r in rows), 'W0_REUSE_ACTIVE_VERSION')
    summary = reader.json(folder/'summary.json'); reduced = reduce_rows(rows)
    require(summary['endpoint'] == 'W0' and summary['state'] == old['cold_W0_H0']
            and summary['requests'] == 2000 and summary['row_count'] == 26000
            and summary['row_order'] == digest([r['identity'] for r in rows])
            and summary['no_mutation'] and summary['optimizer_feedback'] is False and summary['replay'] is False,
            'W0_REUSE_SUMMARY_IDENTITY')
    require({k:v['denominator'] for k,v in reduced.items()} == dict(R=2000,P=4000,N=20000), 'W0_REUSE_DENOMINATORS')
    compare_summary(reduced, summary['summary']); compare_summary(reduced, summary['current'])
    return dict(schema='jlz-interference-l1-exact-w0-reuse-v1', status=STATUS, task=TASK,
        prior_attempt=str(prior), source_folder=str(folder), prior_source=PRIOR_SOURCE, prior_job=59721,
        observation_fingerprint=fingerprint, cold_state=old['cold_W0_H0'], row_count=26000,
        row_order=summary['row_order'], chunk_count=40, chunks=[member(p) for p in paths],
        summary=member(folder/'summary.json'), runtime=member(prior/'PRICE/runtime.json'),
        prior_config=member(prior/'config.json'), prior_lock=member(prior/'execution.lock.json'),
        closure=closure, unchanged_setup_observer_AST=functions,
        members=list(reader.files.values()), original_evaluation_seconds=summary['seconds'],
        new_evaluation_seconds=0., new_forwards=0, independent_reduction=reduced,
        asset_verification='PRIOR_FULL_SHA_PLUS_FRESH_SIZE_INODE_MTIME_NO_LARGE_REHASH',
        raw_reference_only=True, raw_copy=False, hardlink=False, symlink=False, checkpoint=False,
        reuse_scope='Complete cold W0 scalar observation only; no failed fit/state/production GPU qualification reuse')


def verify_manifest(value, config=None):
    require(value['status'] == STATUS and value['task'] == TASK and value['prior_source'] == PRIOR_SOURCE
            and value['prior_config']['sha256'] == PRIOR_CONFIG_SHA and value['prior_lock']['sha256'] == PRIOR_LOCK_SHA,
            'W0_REUSE_MANIFEST_IDENTITY')
    require(value['raw_reference_only'] and not value['raw_copy'] and not value['hardlink']
            and not value['symlink'] and not value['checkpoint'], 'W0_REUSE_NO_BUNDLE')
    require(value['chunk_count'] == 40 and value['row_count'] == 26000, 'W0_REUSE_MANIFEST_COVERAGE')
    folder=Path(value['prior_attempt'])/'PRICE/W0'
    require(Path(value['source_folder'])==folder and
        [row['path'] for row in value['chunks']]==[str(folder/f'chunk-{start:04d}.json') for start in range(0,2000,50)],
        'W0_REUSE_EXACT_RAW_ALLOWLIST')
    bound_members={row['path']:(row['bytes'],row['sha256']) for row in value['members']}
    for row in value['chunks']+[value[key] for key in ('summary','runtime','prior_config','prior_lock')]:
        require(bound_members.get(row['path'])==(row['bytes'],row['sha256']),'W0_REUSE_MEMBER_ALLOWLIST')
    reader = Reader()
    for row in value['members']:
        _bound(reader,row)
    for row in value['closure']:
        _bound(reader,row['current'])
    require(_functions(ROOT/(NAMESPACE+'run.py')) == value['unchanged_setup_observer_AST'], 'W0_REUSE_AST_CHANGED')
    if config is not None:
        require(observation_fingerprint(config) == value['observation_fingerprint'], 'W0_REUSE_CONFIG_BINDING')
        _assets(config)
    return value


def source_chunks(folder, expected_state=None):
    folder = Path(folder)
    receipt = json.loads((folder/'reuse.json').read_text())
    value = verify_manifest(receipt['manifest'])
    require(receipt['manifest_sha256'] == digest(value) and receipt['actual_cold_state'] == value['cold_state'],
            'W0_REUSE_LOCAL_RECEIPT')
    if expected_state is not None:
        require(expected_state == value['cold_state'], 'W0_REUSE_COLD_STATE_CHANGED')
    return [Path(row['path']) for row in value['chunks']]


def install(folder, config, actual_cold_state):
    value = verify_manifest(config['W0_reuse'], config)
    require(actual_cold_state == value['cold_state'], 'W0_REUSE_ACTUAL_COLD_STATE')
    folder = Path(folder)
    require(not folder.exists(), 'W0_REUSE_CREATE_ONCE')
    oldruntime=json.loads(Path(value['runtime']['path']).read_text())
    actualruntime=json.loads((folder.parent/'runtime.json').read_text())
    runtime_fields=('device','torch','transformers','model','FP32','geometry_FP64','eager','TF32','autocast','CPU_threads','cold_W0_H0')
    require(all(actualruntime[key]==oldruntime[key] for key in runtime_fields),'W0_REUSE_ACTUAL_RUNTIME_CHANGED')
    old = json.loads(Path(value['summary']['path']).read_text())
    write(folder/'reuse.json',dict(manifest=value,manifest_sha256=digest(value),actual_cold_state=actual_cold_state,
        actual_runtime=member(folder.parent/'runtime.json'),compared_runtime_fields=list(runtime_fields),
        no_raw_copy=True,no_forward=True,no_checkpoint=True))
    summary = dict(old,seconds=0.,new_forwards=0,reference_only=True,
        original_evaluation_seconds=value['original_evaluation_seconds'],
        reused_from=value['source_folder'],reuse_manifest_sha256=digest(value))
    write(folder/'summary.json',summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--prior-attempt',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    args = p.parse_args(); value = seal_w0_reuse(args.prior_attempt,json.loads(args.config.read_text()))
    write(args.out,value)
    print(json.dumps(dict(status=value['status'],rows=value['row_count'],chunks=value['chunk_count'],
        original_evaluation_seconds=value['original_evaluation_seconds'],new_forwards=0,manifest=member(args.out))))
