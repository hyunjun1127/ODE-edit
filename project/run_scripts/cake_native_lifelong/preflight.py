"""CPU-only CAKE source/data/storage preparation; never imports a model or submits.

The returned packet is NOT an execution lock. A complete runner, live admission,
asset content verification and execution closure freeze remain required.
"""
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
from datetime import datetime, timezone

UPSTREAM_HEAD = 'c8243e1d7e43ca9cf64d552f96221fcb9561aac2'
UPSTREAM_TREE = '4f59249bb23c7cacf0f6490bce8127c74ff7b111'
CHECKPOINTS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
LAYERS = [4, 5, 6, 7, 8]
SCORES = {'0': .4812439084, '1': .4743820429, '2': .4656370878,
          '3': .4440660179, '4': .4335190654}
TASK_ROOT = Path('/data/janghj/ODE-edit/local/cake-native-lifelong/20260915-v1/attempt-v1')
BASELINE_ROOT = Path('/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1')
DATA_ROOT = Path('/data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')
SESSION = '01a04939-b5c7-7a03-ba2d-ef3343d62cfd'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def file_identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def check_config(hp):
    expected = dict(layers=LAYERS, L2=10, v_weight_decay=.4, v_num_grad_steps=25,
                    v_lr=.1, clamp_norm_factor=.5, nullspace_threshold=.02,
                    temperature=.1, causal_scores=SCORES)
    for k, v in expected.items():
        require(hp[k] == v, 'ORIGINAL_CONFIG_DRIFT:' + k)
    require(sorted(map(int, hp['causal_scores'])) == list(range(5)), 'SCORE_INDEX_ORDER')


def check_import_only_patch(original, execution):
    removed = 'from notebooks.util import hparams\n'
    require(original.count(removed) == 1, 'EXPECTED_IMPORT_NOT_UNIQUE')
    expected = original.replace(removed, '')
    require(execution in (expected, expected + '\n'), 'PATCH_EXCEEDS_UNUSED_IMPORT')
    original_ast = ast.parse(original)
    filtered = [n for n in original_ast.body
                if not (isinstance(n, ast.ImportFrom) and n.module == 'notebooks.util')]
    original_ast.body = filtered
    require(ast.dump(original_ast) == ast.dump(ast.parse(execution)), 'NATIVE_AST_CHANGED')
    # The imported global is shadowed by function arguments; no module-level use.
    for n in filtered:
        if isinstance(n, ast.FunctionDef):
            if any(isinstance(x, ast.Name) and x.id == 'hparams' for x in ast.walk(n)):
                require('hparams' in [a.arg for a in n.args.args], 'GLOBAL_HPARAMS_USE')
        else:
            require(not any(isinstance(x, ast.Name) and x.id == 'hparams'
                            for x in ast.walk(n)), 'MODULE_HPARAMS_USE')


def storage_plan(hidden, intermediate, available, save_checkpoints=True):
    one = 5 * 4 * (hidden * intermediate + intermediate * intermediate)
    floor = len(CHECKPOINTS) * one
    raw_reserve = 16 * 1024**3  # planning allowance, NOT measured/allocated space
    total = floor + one + raw_reserve if save_checkpoints else raw_reserve
    return dict(weight_shape=[hidden, intermediate], history_shape=[intermediate, intermediate],
                selected_layers=5, bytes_per_scalar=4, checkpoints=12 if save_checkpoints else 0,
                tensor_checkpoint_storage=save_checkpoints,
                one_checkpoint_tensor_bytes=one, checkpoint_tensor_floor_bytes=floor,
                temporary_checkpoint_reserve_bytes=one if save_checkpoints else 0, raw_source_serialization_reserve_bytes=raw_reserve,
                planned_available_required_bytes=total, available_bytes=available,
                original_tensor_floor_shortfall_bytes=max(0, floor-available),
                tensor_floor_shortfall_bytes=max(0, floor-available) if save_checkpoints else 0,
                planned_shortfall_bytes=max(0, total-available),
                reserves='ESTIMATE_NOT_MEASUREMENT_NOT_EXCLUSIVE_RESERVATION',
                status='DISK_HOLD' if available < total else 'DISK_ARITHMETIC_ONLY_PASS')


def batch_manifest(records, sealed_rows):
    require(len(records) == len(sealed_rows) == 10000, 'EXPECTED_FULL_10K')
    result = []
    for i in range(100):
        rows = records[i*100:(i+1)*100]
        seals = sealed_rows[i*100:(i+1)*100]
        require([r['case_id'] for r in rows] == [r['case_id'] for r in seals], 'ORDER_DRIFT')
        require(all(s['ordinal'] == i*100+j for j, s in enumerate(seals)), 'ORDINAL_DRIFT')
        result.append(dict(batch=i+1, start_ordinal=i*100, end_ordinal=(i+1)*100,
                           records=100, case_order_sha256=digest([r['case_id'] for r in rows]),
                           sealed_record_order_sha256=digest(seals),
                           request_order_sha256=digest([r['requested_rewrite'] for r in rows]),
                           target_order_sha256=digest([(r['requested_rewrite']['target_new'],
                                                        r['requested_rewrite']['target_true']) for r in rows]),
                           prompt_inventory_sha256=digest([(r['requested_rewrite']['prompt'],
                                                            r['paraphrase_prompts'], r['neighborhood_prompts']) for r in rows])))
    return result


def function_ast(path, name):
    return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body
                         if isinstance(n, ast.FunctionDef) and n.name == name))


def prepare(repo, task):
    require(socket.gethostname() == 'server4', 'HOST_BOUNDARY')
    require(task.resolve() == TASK_ROOT, 'TASK_OUTPUT_BOUNDARY')
    require(git(repo, 'remote', 'get-url', 'origin') == 'https://github.com/hyunjun1127/ODE-edit.git', 'REPO_BOUNDARY')
    upstream, execution = task/'upstream/CAKE', task/'execution/CAKE'
    require(git(upstream, 'rev-parse', 'HEAD') == UPSTREAM_HEAD, 'UPSTREAM_HEAD')
    require(git(upstream, 'rev-parse', 'HEAD^{tree}') == UPSTREAM_TREE, 'UPSTREAM_TREE')
    require(git(upstream, 'status', '--porcelain') == '', 'UPSTREAM_DIRTY')
    require(git(execution, 'rev-parse', 'HEAD') == UPSTREAM_HEAD, 'EXEC_BASE_HEAD')
    require(git(execution, 'diff', '--name-only') == 'Cake/Cake_main.py', 'COMPATIBILITY_SCOPE')
    require(not git(execution, 'ls-files', '--others', '--exclude-standard'), 'UNTRACKED_EXEC_SOURCE')
    original = (upstream/'Cake/Cake_main.py').read_text()
    amended = (execution/'Cake/Cake_main.py').read_text()
    check_import_only_patch(original, amended)
    hp = json.loads((upstream/'hparams/Cake/Llama3-8B.json').read_bytes())
    check_config(hp)
    baseline = json.loads((BASELINE_ROOT/'execution.lock.json').read_bytes())
    baseline_runtime = json.loads((BASELINE_ROOT/'output/main-cell-1/runtime.json').read_bytes())
    spec = importlib.util.spec_from_file_location('fixed_counterfact', repo/'scripts/fixed_counterfact.py')
    fixed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixed)
    records = fixed.load_prefix(DATA_ROOT, 10000)
    sealed = json.loads((DATA_ROOT/'source-sample.lock.json').read_bytes())
    require(digest(sealed['records'][:1000]) == baseline['prefix1000_root'], 'PREFIX1000_ROOT')
    sample = batch_manifest(records, sealed['records'])
    config_path = Path(baseline['snapshot'])/'config.json'
    model_config = json.loads(config_path.read_bytes())
    fs = os.statvfs(task)
    space = storage_plan(model_config['hidden_size'], model_config['intermediate_size'], fs.f_bavail*fs.f_frsize, save_checkpoints=False)
    space.update(free_inodes=fs.f_favail, filesystem_block_bytes=fs.f_frsize)
    source_members = []
    for rel in git(upstream, 'ls-files').splitlines():
        p = upstream/rel
        require(p.is_file() and not p.is_symlink(), 'SOURCE_MEMBER_TYPE:' + rel)
        member = file_identity(p)
        member['relative_path'] = rel
        source_members.append(member)
    source_read = ['README.md', 'LICENSE', 'Cake/__init__.py', 'Cake/Cake_main.py',
                   'Cake/Cake_hparams.py', 'Cake/compute_z.py', 'Cake/compute_ks.py',
                   'hparams/Cake/Llama3-8B.json', 'experiments/evaluate.py', 'util/globals.py',
                   'globals.yml', 'util/hparams.py', 'util/__init__.py', 'util/nethook.py',
                   'util/generate.py', 'util/logit_lens.py', 'rome/__init__.py',
                   'rome/repr_tools.py', 'rome/layer_stats.py', 'rome/tok_dataset.py']
    read_receipt = [dict(file_identity(upstream/p), lines=len((upstream/p).read_text().splitlines()),
                         scope='FULL_READ_UPSTREAM_SOURCE') for p in source_read]
    read_receipt += [dict(file_identity(repo/p), lines=len((repo/p).read_text().splitlines()),
                          scope='FULL_READ_GOVERNING_POLICY') for p in ['PROTOCOL.md', 'plans/global/fixed-counterfact-10k-policy.md', 'scripts/fixed_counterfact.py']]
    alpha_path = Path(baseline['blue_root'])/'AlphaEdit/AlphaEdit_main.py'
    context_same = function_ast(upstream/'Cake/Cake_main.py', 'get_context_templates') == function_ast(alpha_path, 'get_context_templates')
    require(context_same, 'CONTEXT_SOURCE_SEMANTIC_DIFFERENCE')
    context = BASELINE_ROOT/'output/main-cell-1/B001/contexts.json'
    asset_paths = [baseline['projector']] + [r['path'] for r in baseline['stats_members']]
    assets = []
    for p in asset_paths:
        expected = next(x for x in baseline['members'] + baseline['stats_members'] if x['path'] == p)
        actual = Path(p).stat()
        require(actual.st_size == expected['bytes'], 'ASSET_SIZE_DRIFT:' + p)
        assets.append(dict(expected, current_size=actual.st_size, current_mtime_ns=actual.st_mtime_ns,
                           verification='PRIOR_SEALED_SHA_REFERENCE_PLUS_CURRENT_STAT; NEW_FULL_REHASH_NOT_PERFORMED'))
    differences = {k: dict(CAKE=v, BASE_ALPHAEDIT=baseline['cells'][1]['hparams'].get(k, 'NOT_PRESENT'))
                   for k, v in hp.items() if baseline['cells'][1]['hparams'].get(k) != v}
    packet = dict(schema='cake-preparation-v1-NOT_EXECUTION_LOCK', instruction_id='ODEEDIT-S06-CAKE-NATIVE-FIXED10K-LIFELONG-SH4-V1',
                  timestamp_utc=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(), session=SESSION,
                  registered_cwd='/data/janghj/ODE-edit', worktree=str(repo), task_root=str(task),
                  analysis_source_head=git(repo, 'rev-parse', 'HEAD'), observed_origin_main=git(repo, 'rev-parse', 'origin/main'),
                  upstream_head=UPSTREAM_HEAD, upstream_tree=UPSTREAM_TREE, hparams=hp,
                  original_hparams=file_identity(upstream/'hparams/Cake/Llama3-8B.json'),
                  original_main=file_identity(upstream/'Cake/Cake_main.py'), execution_main=file_identity(execution/'Cake/Cake_main.py'),
                  compatibility_patch='unused notebooks.util import removal plus final LF; native AST otherwise exact',
                  dataset=fixed.verify(DATA_ROOT), batches=sample, checkpoint_batches=CHECKPOINTS,
                  projector_mapping=[dict(physical_layer=l, asset_index=i, local_index=i) for i,l in enumerate(LAYERS)],
                  storage=space, baseline_lock=file_identity(BASELINE_ROOT/'execution.lock.json'),
                  baseline_runtime=file_identity(BASELINE_ROOT/'output/main-cell-1/runtime.json'),
                  baseline_source_head=baseline['source_head'], baseline_source_tree=baseline['source_tree'],
                  seed=baseline['seed'], revision=baseline['revision'], model_config=file_identity(config_path),
                  context=file_identity(context), context_function_ast_equal=context_same,
                  planned_environment={k:baseline_runtime[k] for k in ['torch','transformers','dtype','attention','tf32_matmul','tf32_cudnn','writer_tokenizer','evaluator_tokenizer']},
                  upstream_readme_environment=dict(torch='2.6.0', transformers='4.51.3'),
                  hparam_differences=differences, asset_references=assets,
                  evaluation_plan=dict(current_every_batch=True, all_seen_rewrite_every_batch=True,
                                       full_seen_at_checkpoints=True, final_fullseen_once=True,
                                       expected_final_denominators=dict(RS=10000, PS=20000, NS=100000),
                                       comparison='RS/PS new<true; NS true<new; ties=failure; strict secondary',
                                       evaluator_microbatch=16, W0_preedit='REUSE_SEALED_42673_NO_NEW_EVALUATION'),
                  compute_plan=dict(native_target_requests=10000, maximum_target_loss_evaluations=250000,
                                    maximum_Adam_updates=240000, native_solves=500,
                                    whole_batch_history_passes=100, layer_history_updates=500,
                                    runtime_seconds='NOT_ESTIMATED_FROM_CAKE_MEASUREMENT',
                                    allocated_gpu_seconds=0, gpu_hour_cap=None),
                  resource_plan=dict(job_name='odeedit_cake_native_lifelong_s4', gpus=1, cpus=8, mem_mib=60416,
                                     wall_hours=48, requeue=False, export='NONE', project_cap=2,
                                     current_admission='FRESH_SCHEDULER_CHECK_REQUIRED_BEFORE_SUBMISSION'),
                  user_override='CAKE 부분은 weight 저장 하지 말고 그냥 올려라',
                  storage_override='NO_PERSISTENT_W_M_TENSOR_CHECKPOINTS; live history and original evaluations preserved; exact restart unavailable',
                  state='CPU_PREPARATION_ONLY', job_id=None, runtime_implementation='SEPARATE_EXECUTION_FREEZE_REQUIRED',
                  monitoring_active=False, automatic_resume=False, resume_trigger='explicit_user_call',
                  scientific_promotion=False, artifact_broadcast='NO_BROADCAST_NOT_REQUIRED')
    return packet, source_members, read_receipt, ''.join(difflib.unified_diff(
        original.splitlines(True), amended.splitlines(True), fromfile='a/Cake/Cake_main.py', tofile='b/Cake/Cake_main.py'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--task-root', type=Path, default=TASK_ROOT)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    require(args.output.resolve().parent == TASK_ROOT, 'OUTPUT_BOUNDARY')
    packet, members, reads, patch = prepare(args.repo.resolve(), args.task_root)
    args.output.mkdir(mode=0o700, exist_ok=False)
    for name, value in [('preparation.json', packet), ('upstream-inventory.json', members), ('full-read.json', reads)]:
        with (args.output/name).open('x') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write('\n')
    with (args.output/'unused-notebooks-import.patch').open('x') as f:
        f.write(patch)
    print(json.dumps(dict(status=packet['state'], storage=packet['storage'], preparation=file_identity(args.output/'preparation.json'))))


if __name__ == '__main__':
    main()
