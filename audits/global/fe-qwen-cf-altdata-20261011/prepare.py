"""Prepare one preregistered, data-only contrast to native Qwen FE job 63385."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

WORKTREE = Path('/mnt/raid5/janghj/ODE-edit/local/worktrees/fe-qwen-cf-altdata-20261011')
TASK = Path(__file__).resolve().parent
ROOT = TASK / 'native-r1'
REF = Path('/mnt/raid5/janghj/ODE-edit/local/fe-collapse-repair-20261011')
sys.path.insert(0, str(WORKTREE))
from official.experiments.prepare import digest, file_sha, normalize_records, write_new
from official.runners.server1.common import member, read, verify
from omegaconf import OmegaConf


def copy_members(members, old_root, new_root):
    copied = []
    comparisons = []
    for entry in members:
        source = verify(entry)
        relative = source.relative_to(old_root)
        destination = new_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as handle:
            handle.write(source.read_bytes())
        actual = member(destination)
        assert actual['sha256'] == entry['sha256']
        copied.append(actual)
        comparisons.append(dict(relative=str(relative), sha256=entry['sha256']))
    return copied, comparisons


assert not ROOT.exists(), 'CREATE_ONCE_PACKAGE'
ROOT.mkdir(parents=True)
previous = read(REF / 'native-r2/configs/qwen25.json')
previous_config_member = member(REF / 'native-r2/configs/qwen25.json')
assert digest({k: v for k, v in previous.items() if k != 'identity_sha256'}) == previous['identity_sha256']
old_stream = read(verify(previous['original_stream']))
source = Path('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json')
source_member = member(source)
assert source_member['sha256'] == '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'
raw = read(source)
assert len(raw) == 10000 and normalize_records(raw, 'cf', 2000) == old_stream
rows = normalize_records(raw[2000:4000], 'cf', 2000)
assert len(rows) == 2000
ids = [row['case_id'] for row in rows]
assert len(set(ids)) == 2000 and not set(ids) & {row['case_id'] for row in old_stream}
stream_path = ROOT / 'data/cf-stream.json'
write_new(stream_path, rows)
converted = []
for row in rows:
    rewrite = row['requested_rewrite']
    converted.append(dict(case_id=row['case_id'], prompt=rewrite['prompt'], subject=rewrite['subject'],
                          target_new=rewrite['target_new']['str'], target_true=rewrite['target_true']['str'],
                          paraphrase_prompts=row['paraphrase_prompts'], neighborhood_prompts=row['neighborhood_prompts']))
converted_path = ROOT / 'data/counterfact_stream_2000.json'
write_new(converted_path, converted)
assert all(not row['target_new'].startswith(' ') for row in converted)
selection = dict(schema='fe-qwen-cf-altdata-selection-v1', registered_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 hypothesis='Does native current-model-z FE collapse persist on a different fixed CF cohort?',
                 reference_training_job=63385, reference_evaluation_job=63387, reference_config=previous_config_member,
                 source=source_member, original_stream=previous['original_stream'], stream=member(stream_path),
                 converted=member(converted_path), source_slice_zero_based=[2000, 4000],
                 selection='NEXT_2000_IN_SAME_FIXED_10K_FILE_ORDER_NO_SHUFFLE_NO_PERFORMANCE_FILTER',
                 case_overlap_with_reference=0, ordered_case_ids=ids, ordered_case_ids_sha256=digest(ids),
                 batches=[dict(batch=i//100+1, case_ids_sha256=digest(ids[i:i+100])) for i in range(0, 2000, 100)],
                 requests=2000, batch_size=100, training_seed=0, model_dtype='bfloat16',
                 target_reference='batch_entry', history=True, covariance_repair=False,
                 GPU_results_observed=False, replaces_main_baseline=False)
write_new(ROOT / 'stream-proof.json', selection)

config = copy.deepcopy(previous)
author = ROOT / 'author'
config['source_members'], author_equivalence = copy_members(previous['source_members'], Path(previous['root']), author)
config['root'] = str(author)
output = ROOT / 'runs/qwen25'
output.mkdir(parents=True)
config['output'] = str(output)
config['original_stream'] = member(stream_path)
config['stream'] = member(converted_path)
config['overrides'] = [value.replace(str(REF / 'native-r2'), str(ROOT)) for value in previous['overrides']]
config['data_contrast'] = dict(reference_job=63385, selection=member(ROOT / 'stream-proof.json'),
                             scientific_changes=['CF requests: fixed10k[2000:4000] instead of fixed10k[0:2000]'])
config.pop('identity_sha256')
for cov in config['covariances']:
    verify(cov['original'])
    link = output / 'cache/stats/Qwen-Qwen2.5-7B-Instruct' / f"layer-{cov['layer']}.npz"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(cov['original']['path'])
    cov['link'] = str(link)
    assert not cov['repair']
assets = read(verify(config['model_assets']))
for entry in assets['model']['members']:
    actual = member(Path(entry['path']).resolve())
    assert (actual['bytes'], actual['sha256']) == (entry['bytes'], entry['sha256'])
snapshot = Path(assets['model']['snapshot'])
hf_cache = ROOT / 'hf-cache'
hf_cache.mkdir()
alias = hf_cache / 'models--Qwen--Qwen2.5-7B-Instruct'
alias.symlink_to(snapshot.parent.parent, target_is_directory=True)
assert (alias / 'refs/main').read_text().strip() == snapshot.name
config['identity_sha256'] = digest(config)
config_path = ROOT / 'configs/qwen25.json'
write_new(config_path, config)
overrides = config['overrides'] + ['+diagnostic_identity_sha256=' + config['identity_sha256']]
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_CACHE=str(hf_cache), HF_HUB_OFFLINE='1',
           TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
resolved_text = subprocess.check_output([config['python'], '-B', 'main.py', '--cfg', 'job', *overrides],
                                       cwd=author, env=env, text=True, stderr=subprocess.STDOUT)
(ROOT / 'resolved.yaml').write_text(resolved_text)
resolved = OmegaConf.create(resolved_text[min(resolved_text.index('algs:'), resolved_text.index('llms:')):])
previous_resolved = OmegaConf.load(Path(previous['output']) / 'hydra-edit/.hydra/config.yaml')
allowed = {'data_dir', 'cache_dir', 'results_dir', 'zs_cache_dir', 'diagnostic_checkpoint_dir', 'diagnostic_identity_sha256'}
old_values = OmegaConf.to_container(previous_resolved, resolve=True)
new_values = OmegaConf.to_container(resolved, resolve=True)
assert set(old_values) == set(new_values)
changed = [key for key in old_values if old_values[key] != new_values[key]]
assert set(changed) == allowed, changed
assert resolved.model_dtype == 'bfloat16' and resolved.bs == 100 and resolved.num_edits == 2000
assert resolved.seed == 0 and resolved.algs.add_old_keys

old_lock = read(REF / 'native-eval-r1/execution-lock.json')
eval_source = ROOT / 'evaluation-source'
eval_members, eval_equivalence = copy_members(old_lock['source_members'], Path(old_lock['source_directory']), eval_source)
write_new(ROOT / 'execution-lock.json', dict(source_commit=old_lock['source_commit'], source_directory=str(eval_source),
                                           source_members=eval_members, configs=[member(config_path)]))
env_lines = ['#!/bin/bash', 'set -euo pipefail', 'export PYTHONDONTWRITEBYTECODE=1',
             'export TOKENIZERS_PARALLELISM=false', 'export HF_HUB_OFFLINE=1', 'export TRANSFORMERS_OFFLINE=1',
             'export OMP_NUM_THREADS=8', 'export MKL_NUM_THREADS=8', 'export OPENBLAS_NUM_THREADS=8']
eval_overrides = [value for value in overrides if not value.startswith('hydra.run.dir=')]
eval_overrides += ['test_only=True', f'hydra.run.dir={output}/hydra-eval']
training_script = env_lines + ['export HF_HUB_CACHE=' + shlex.quote(str(hf_cache)), 'cd ' + shlex.quote(str(author)),
    shlex.join([config['python'], '-B', '-u', 'main.py', *overrides]),
    shlex.join([config['python'], '-B', '-u', 'main.py', *eval_overrides])]
evaluation_script = env_lines + ['cd ' + shlex.quote(str(eval_source)),
    'exec ' + shlex.join([config['python'], '-B', '-u', '-m', 'official.diagnostics.fe_native.evaluate',
                         '--native-config', str(config_path), '--output', str(ROOT / 'evaluations/qwen25'),
                         '--lock', str(ROOT / 'execution-lock.json')])]
for name, lines in [('qwen25.sh', training_script), ('qwen25-eval.sh', evaluation_script)]:
    script = ROOT / name
    script.write_text('\n'.join(lines) + '\n')
    script.chmod(0o755)
    subprocess.run(['bash', '-n', str(script)], check=True)
free = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
assert free > 110 * 2**30, 'STORAGE_MUST_COVER_ALL_FIVE_CHECKPOINTS_AND_RESERVE'
write_new(ROOT / 'preparation.json', dict(preparation_source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORKTREE, text=True).strip(),
    reference_config=previous_config_member, config=member(config_path), selection=member(ROOT/'stream-proof.json'),
    author_source_commit=previous['author_commit'], native_patch_source=previous['ode_patch_source'],
    evaluator_source_commit=old_lock['source_commit'], author_files_equal=author_equivalence,
    evaluator_files_equal=eval_equivalence, resolved_config_changed_keys=changed,
    hparams_seed_dtype_writer_history_equal=True, GPU_execution=False, free_bytes=free,
    scripts=[member(ROOT/'qwen25.sh'), member(ROOT/'qwen25-eval.sh')],
    execution_lock=member(ROOT/'execution-lock.json'), preparer=member(Path(__file__))))
print(json.dumps(dict(prepared=str(ROOT), identity=config['identity_sha256'], requests=2000,
                      overlap=0, slice=[2000,4000], author_files_equal=len(author_equivalence),
                      evaluator_files_equal=len(eval_equivalence), free_GiB=free/2**30), indent=2), flush=True)
