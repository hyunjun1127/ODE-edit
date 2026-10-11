"""Server4 lane B: FE zsRE 2K, Llama3 then GPT-J, as one serial Slurm lane.

User instructions (2026-10-11): "FE zsRE는 server4에 제출하는걸로 옮겨", "gpt-j의 zsRE도 server4에서 제출해".
Contract of the README FE rows: author hparams + persistent writer history; z fitted on the CURRENT
model at each batch entry for that batch's 100 requests (never W0-precomputed); the first 2,000
official zsRE requests, 20 batches of 100, seed 0; public-query zsRE evaluator (request macro,
Loc = loc_ans).

* Llama3 = the implementation of the Llama FE CF row (63382): `official.runners.fe_author_history`
  (modified EasyEdit FE + history, FP32). The science config is 63357's unchanged; 63357 failed while
  streaming its B2 checkpoint to server1, so this config drops `checkpoint_archive` and keeps the
  ordinary local checkpoint (latest each batch, W20 kept) on server4 /data.
* GPT-J = the implementation of the GPT-J FE CF row (63389 -> 63390): the pinned author FE `main.py`
  through `official.diagnostics.fe_native` (BF16, native writer, persistent history, current-model z,
  L3 C0 `project_roundoff` repair recomputed here from the same stats bytes). Only the dataset changes:
  the official zsRE first 2,000 rows, which equal the author's `zsre_mend_eval_19086.json[:2000]` in
  every field the editor reads. Its snapshots are evaluated by `fe_native.evaluate` (public zsRE).
  Storage only: W0/W5/W10/W15 snapshots omit the history tensor (`diagnostic_history_batches=[20]`).

Every step runs on server4, CPU only, from an extracted `git archive <main commit> official`:

    EASYEDIT_PY -m official.runners.server4.fe_zsre_lane prepare-llama --output LOCAL/llama-zsre-local-r1 --wandb-env ENV
    NATIVE_PY -m official.diagnostics.fe_native.covariance --source L3_NPZ \\
        --destination LOCAL/native-derived-prior/gptj-layer-3.npz --expected-sha256 L3_SHA
    NATIVE_PY -m official.runners.server4.fe_zsre_lane prepare-gptj --output LOCAL/gptj-zsre-native-r1 \\
        --inputs LOCAL/inputs --derived LOCAL/native-derived-prior/gptj-layer-3.json --source-commit COMMIT
    EASYEDIT_PY -m official.runners.server4.fe_zsre_lane register --llama LLAMA_PREP --gptj GPTJ_PREP \\
        --attempt LOCAL/registration-r1 --source-commit COMMIT

NATIVE_PY is a Python 3.12 venv whose own site-packages hold the devbox FE runtime pins (transformers
4.51.3, tokenizers 0.21.4, numpy 1.26.4, hydra-core 1.3.2, ...) and whose `.pth` appends the existing
EasyEdit site-packages (torch 2.9.1+cu128), as on devbox; `native_runtime_receipt` compares both.
"""
import argparse
import copy
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile

from official.experiments.prepare import digest, file_sha, write_new
from official.runners.server1.common import member, read, verify
from official.runners.server1.assets import member as asset_member

INSTRUCTION = 'USER-FE-ZSRE-SERVER4-LANE-B-20261011'
TASK = 'fe-zsre-lane-b-s4-20261011'
LOCAL = Path('/data/janghj/ODE-edit/local') / TASK
REPOSITORY = Path('/data/janghj/ODE-edit')
NODE, QOS, PARTITION, CAP = 'server4', 'lab_gpu_s4', 'gpu', 2
EASYEDIT_PYTHON = '/data/janghj/EasyEdit/.venv/bin/python'
MEMORY_MiB = 59392  # server4 policy is 60416 MiB per GPU; 63357 used the same request.

LLAMA_TEMPLATE = Path('/data/janghj/ODE-edit/local/llama-zsre-fe-author-hparams-s4-20261011')
LLAMA_TEMPLATE_CONFIG_SHA = 'b35dde639824300ece65b38b518e7a112655f0c1600b83f5c077e2c043a53dc2'
LLAMA_STORAGE_MIN_FREE = 16 << 30  # one local checkpoint is ~5.3 GB (W + H, FP32)
LLAMA_OPERATIONAL = {'config_sha256', 'checkpoint_archive', 'checkpoint_policy', 'dispatch_instruction',
                     'task_id', 'assets', 'stream_member', 'runtime', 'output', 'tracking_env_file',
                     'zsre_query_parity', 'author_profile_sha256', 'covariance_policy',
                     'storage_min_free_bytes', 'rerun_of'}

AUTHOR_COMMIT = '478134dfb24b43f4e18b47e8500893ce3f9cc50f'
AUTHOR_CLONE = Path('/data/janghj/FE')
GPTJ_TEMPLATE_IDENTITY = '909c7a0762ce4a2640d74808f52a06a75cec47a8cea528ce07a09e517a2e7daf'  # 63389
GPTJ_REVISION = '47e169305d2e8376be1d31e765533382721b2cc1'
GPTJ_LAYERS = (3, 4, 5, 6, 7, 8)
ZSRE_STREAM_SHA = 'f42ee4bc6e98b1133e48dc81102201ccd85ccc917a97160ee3c6a0f3a38f378c'
ZSRE_RECORDS_DIGEST = '93925ec2f74e00a1b3fbdf261e557d10652ae0eca7265b4948958d39de4848b7'
ZSRE_ORDERED_IDS = '494b15107deb77d9741606b7bddfecb8b350a486bb5a751655b294d4ebc6f1a4'
ARM = 'gptj_zsre_repaired'
# The author's own test_only evaluator dispatches on "zsre_mend_eval" in the data name.
DATA_NAME = 'zsre_mend_eval_stream_2000'
NATIVE_RESERVE = 8 << 30
HISTORY_BATCHES = [20]
GPTJ_HIDDEN, GPTJ_INTERMEDIATE = 4096, 16384
PATH_MAP = (('/mnt/raid5/janghj/.cache/huggingface/hub/', '/data/janghj/.cache/huggingface/hub/'),
            ('/mnt/raid5/janghj/EasyEdit/examples/data/stats/', '/data/janghj/EasyEdit/examples/data/stats/'))
SCIENCE_FIELDS = ('model', 'author_commit', 'entrypoint', 'model_dtype', 'current_batch_targets_only',
                  'z_reference', 'history', 'covariance_repair', 'context_override', 'writer_override',
                  'requests', 'batch_size')
FIXED_OVERRIDES = ('llms=gpt-j-6b', 'algs=memit', 'bs=100', 'num_edits=2000', 'seed=0',
                   '+diagnostic_z_reference=batch_entry', 'hydra.job.chdir=False')
MISSING = object()


def mapped(path, path_map=PATH_MAP):
    for old, new in path_map:
        if path.startswith(old):
            return new + path[len(old):]
    raise ValueError('UNMAPPED_TEMPLATE_PATH ' + path)


def server4_cpu():
    assert socket.gethostname() == NODE, 'RUN_ON_SERVER4'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_ONLY_PREPARATION'


def git(*args, cwd=REPOSITORY, text=True):
    return subprocess.check_output(['git', *args], cwd=cwd, text=text)


def official_source(commit, control_root, repository=REPOSITORY):
    """The main commit's official/ tree; the running control copy must be exactly that archive."""
    tree = git('rev-parse', commit + ':official', cwd=repository).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'origin/main'], cwd=repository, check=True)
    archive = git('archive', '--format=tar', commit, 'official', cwd=repository, text=False)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for row in bundle.getmembers():
            if row.isfile():
                files[row.name] = hashlib.sha256(bundle.extractfile(row).read()).hexdigest()
    control = {str(p.relative_to(control_root)): file_sha(p) for p in sorted((Path(control_root)/'official').rglob('*'))
               if p.is_file() and '__pycache__' not in p.parts}
    assert control == files, 'CONTROL_SOURCE_IS_NOT_THE_COMMIT_ARCHIVE'
    return dict(commit=commit, official_tree=tree, files=len(files)), archive


def extract_official(archive, target):
    target = Path(target)
    assert not target.exists()
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        rows = bundle.getmembers()
        assert rows and all((r.isfile() or r.isdir()) and r.name.split('/')[0] == 'official'
                            and not Path(r.name).is_absolute() and '..' not in Path(r.name).parts for r in rows)
        bundle.extractall(target, filter='data')
    return [member(p) for p in sorted(target.rglob('*')) if p.is_file()]


def rehash_assets(template, path_map=None, runtime_info=None):
    """Same content at server4 paths; every member re-hashed here against the template SHA/bytes."""
    a = copy.deepcopy(template)
    a.pop('assets_sha256')
    move = (lambda p: mapped(p, path_map)) if path_map else (lambda p: p)
    model = a['model']
    model['snapshot'] = move(model['snapshot'])
    members = []
    for m in model['members']:
        got = asset_member(move(m['path']), allow_symlink=True, expected_sha=m['sha256'], expected_bytes=m['bytes'])
        if got['path'].endswith(('.safetensors', '.bin')):
            assert Path(got['real_path']).name == got['sha256'], 'HF_CONTENT_ADDRESSED_PAYLOAD_MISMATCH'
        members.append(got)
    model['members'] = members
    for row in a['C0'].values():
        old = row['member']
        row['member'] = asset_member(move(old['path']), allow_symlink=False, expected_sha=old['sha256'],
                                     expected_bytes=old['bytes'])
    if runtime_info is not None:
        a['runtime'] = runtime_info
    a['template_assets_sha256'] = template['assets_sha256']
    a['assets_sha256'] = digest(a)
    return a


# ---------------------------------------------------------------- Llama3 (EasyEdit FE + history)

def llama_config(template, *, assets, stream, runtime_info, output, tracking_env, parity, template_member,
                 profile_sha256):
    """63357's science config with local checkpoints; only operational fields change."""
    c = copy.deepcopy(template)
    c.pop('config_sha256')
    c.pop('checkpoint_archive')
    c.update(server='server4', task_id=TASK, dispatch_instruction=INSTRUCTION,
             checkpoint_policy='W0_THEN_LATEST_EACH_BATCH_KEEP_W20_LOCAL', covariance_policy='strict',
             assets=assets, stream_member=stream, runtime=runtime_info, output=str(Path(output)/'runs'/'zsre'),
             tracking_env_file=str(tracking_env), zsre_query_parity=parity, author_profile_sha256=profile_sha256,
             storage_min_free_bytes=LLAMA_STORAGE_MIN_FREE,
             rerun_of=dict(job='63357', config_sha256=template['config_sha256'], config=template_member,
                           reason='63357 failed at the B2 checkpoint save while streaming it to server1 '
                                  '(checkpoint_archive pipe, torch.save unexpected pos); this rerun keeps '
                                  'checkpoints local on server4 /data'))
    changed = {k for k in set(c) | set(template) if c.get(k, MISSING) != template.get(k, MISSING)}
    assert changed <= LLAMA_OPERATIONAL, sorted(changed - LLAMA_OPERATIONAL)
    c['config_sha256'] = digest(c)
    return c


def prepare_llama(output, wandb_env):
    from official.baselines.fe_author_profile import PROFILE, profile
    from official.evaluation.zsre_query_parity import compare_queries
    from official.runners.fe_author_history import validate_config, tracking_values
    from official.runners.server1.fe_history_prepare import runtime
    from official.tracking.schema import config as tracking_config, load_env
    server4_cpu()
    output = Path(output).absolute()
    assert output.is_relative_to(LOCAL) and not output.exists()
    template_path = LLAMA_TEMPLATE/'config.json'
    t = read(template_path)
    assert t['config_sha256'] == digest({k: v for k, v in t.items() if k != 'config_sha256'}) == LLAMA_TEMPLATE_CONFIG_SHA
    assert (t['model'], t['dataset'], t['server'], t['method']) == ('llama3', 'zsre', 'server4', 'MEMIT_FE_HISTORY')
    assert t.get('checkpoint_archive') and 'native_context' not in t and 'context_replay' not in t
    assert t['hparams'] == profile('llama3')['hparams'], 'AUTHOR_HPARAMS_CHANGED'
    current = runtime()
    assert (current['executable'], current['packages']) == (t['runtime']['executable'], t['runtime']['packages']), \
        'RUN_WITH_THE_63357_PYTHON'
    load_env(wandb_env)  # the runner's tracker accepts only the schema ENV_KEYS
    template_assets = read(verify(t['assets']))
    assert template_assets['assets_sha256'] == digest({k: v for k, v in template_assets.items() if k != 'assets_sha256'})
    records = read(verify(t['stream_member']))
    records = records['records'] if isinstance(records, dict) else records
    assert len(records) == 2000 and digest(records) == t['stream_sha256'] == ZSRE_RECORDS_DIGEST
    assert digest([r['case_id'] for r in records]) == t['ordered_case_ids_sha256'] == ZSRE_ORDERED_IDS
    output.mkdir(parents=True)
    stream = output/'inputs'/'zsre-stream.json'
    stream.parent.mkdir()
    shutil.copyfile(t['stream_member']['path'], stream)
    assert file_sha(stream) == t['stream_member']['sha256'] == ZSRE_STREAM_SHA
    env = output/'tracking.env'
    shutil.copyfile(wandb_env, env)
    load_env(env)
    assets_path = output/'assets-llama3.json'
    write_new(assets_path, rehash_assets(template_assets, runtime_info=runtime()))
    print('LLAMA_MODEL_AND_C0_FULL_SHA_PASS', flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(read(assets_path)['model']['snapshot'], local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    parity = compare_queries(tok, records, model_family='llama3')
    assert parity['requests'] == 2000 and parity['input_mismatches'] == parity['target_mismatches'] == 0
    assert parity['model_forward_calls'] == 0
    write_new(output/'zsre-query-parity.json', parity)
    c = llama_config(t, assets=member(assets_path), stream=member(stream), runtime_info=runtime(), output=output,
                     tracking_env=env, parity=member(output/'zsre-query-parity.json'),
                     template_member=member(template_path), profile_sha256=file_sha(PROFILE))
    validate_config(c)
    tracking_config(tracking_values(c, {'source_commit': 'a'*40}))
    config_path = output/'configs'/'zsre.json'
    write_new(config_path, c)
    changed = sorted(k for k in set(c) | set(t) if c.get(k, MISSING) != t.get(k, MISSING))
    result = dict(instruction=INSTRUCTION, server=NODE, model='llama3', dataset='zsre', config=member(config_path),
                  config_sha256=c['config_sha256'], template_config=member(template_path),
                  template_config_sha256=t['config_sha256'], changed_operational_fields=changed,
                  scientific_configuration_identical=True, hparams_sha256=digest(c['hparams']),
                  zsre_query_parity=dict(queries=parity['queries'], mismatches=0),
                  checkpoint='LOCAL_SERVER4_DATA_LATEST_EACH_BATCH_KEEP_W20', GPU_qualification='NOT_RUN_USER_DISABLED',
                  free_bytes=shutil.disk_usage(output).free)
    write_new(output/'preparation.json', result)
    print(json.dumps(result, indent=1), flush=True)


# ---------------------------------------------------------------- GPT-J (pinned author FE main.py)

def native_zsre_rows(records):
    """Official zsRE stream rows -> author request records; order and case_id are preserved."""
    rows = []
    for record in records:
        rewrite, neighbors = record['requested_rewrite'], record['neighborhood_prompts']
        assert rewrite['prompt'].count('{}') == 1 and len(neighbors) == 1 and record['paraphrase_prompts']
        row = dict(case_id=record['case_id'], prompt=rewrite['prompt'], subject=rewrite['subject'],
                   target_new=rewrite['target_new']['str'], target_true=rewrite['target_true']['str'],
                   paraphrase_prompts=list(record['paraphrase_prompts']),
                   neighborhood_prompts=[n['prompt'] for n in neighbors],
                   neighborhood_prompts_answers=[n['target'] for n in neighbors])
        # The author's apply_memit_to_model prepends the space itself.
        assert not row['target_new'].startswith(' ')
        rows.append(row)
    return rows


def compare_author_zsre(rows, author):
    """Every field the author editor and its zsRE evaluator read must equal the author's own zsRE rows."""
    fields = ('prompt', 'subject', 'target_new', 'paraphrase_prompts', 'neighborhood_prompts',
              'neighborhood_prompts_answers')
    assert len(author) >= len(rows)
    mismatches = {f: sum(r[f] != a[f] for r, a in zip(rows, author)) for f in fields}
    mismatches['case_id_vs_author_id'] = sum(r['case_id'] != a['id'] for r, a in zip(rows, author))
    assert not any(mismatches.values()), mismatches
    return dict(requests=len(rows), compared_fields=list(fields), mismatches=mismatches,
                target_true_differs=sum(r['target_true'] != a['target_true'] for r, a in zip(rows, author)),
                target_true_use='not read by the author editor or its zsRE evaluator')


def native_overrides(run, data_dir, reserve=NATIVE_RESERVE, history_batches=HISTORY_BATCHES):
    return ['llms=gpt-j-6b', 'algs=memit', 'bs=100', 'num_edits=2000', 'seed=0', f'data={DATA_NAME}',
            f'data_dir={data_dir}', f'cache_dir={run}/cache', f'results_dir={run}/results',
            f'zs_cache_dir={run}/unused-z-cache', f'save_name={ARM}', f'load_name={ARM}',
            f'+diagnostic_checkpoint_dir={run}/checkpoints', f'+diagnostic_reserve_bytes={reserve}',
            '+diagnostic_z_reference=batch_entry',
            '+diagnostic_history_batches=[' + ','.join(map(str, history_batches)) + ']',
            f'hydra.run.dir={run}/hydra-edit', 'hydra.job.chdir=False']


def native_disk_need(reserve=NATIVE_RESERVE, history_batches=HISTORY_BATCHES):
    """Bytes the native edit job writes: five snapshots, history only where selected, final weights."""
    weights = len(GPTJ_LAYERS) * GPTJ_HIDDEN * GPTJ_INTERMEDIATE * 2  # BF16
    history = len(GPTJ_LAYERS) * GPTJ_INTERMEDIATE * GPTJ_INTERMEDIATE * 4  # FP32 cache_c
    snapshots = sum(weights + (history if b in history_batches else 0) + (16 << 20) for b in (0, 5, 10, 15, 20))
    return snapshots + weights + reserve


def native_runtime_receipt():
    """Versions of every visible distribution; file hashes (RECORD) of the venv's own packages."""
    from importlib import metadata
    own = Path(sys.prefix).resolve()
    rows = {}
    for dist in metadata.distributions():
        name = dist.metadata['Name'].lower().replace('_', '-')
        location = Path(dist.locate_file('')).resolve()
        files = {str(f): f.hash.value for f in (dist.files or [])
                 if f.hash and '.dist-info/' not in str(f) and '__pycache__' not in str(f)
                 and not str(f).startswith('..')}
        local = location.is_relative_to(own)
        if name in rows:  # first match on sys.path wins, as at import time
            continue
        rows[name] = dict(version=dist.version, origin='venv' if local else 'external',
                          files_sha256=digest(files) if local else None, files=len(files) if local else None)
    import hydra
    import numpy
    import torch
    import transformers
    return dict(python=sys.version, executable=sys.executable, prefix=sys.prefix,
                versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                              hydra=hydra.__version__, numpy=numpy.__version__),
                modules={m.__name__: str(Path(m.__file__).resolve()) for m in (torch, transformers, hydra, numpy)},
                distributions=rows)


def compare_runtime(server4, devbox):
    expected = {k: v for k, v in devbox['distributions'].items() if k != 'pip'}
    actual = {k: v for k, v in server4['distributions'].items() if k != 'pip'}
    differences = sorted(k for k in set(expected) | set(actual)
                         if (expected.get(k) or {}).get('version') != (actual.get(k) or {}).get('version')
                         or (expected.get(k) or {}).get('origin') != (actual.get(k) or {}).get('origin')
                         or (expected.get(k) or {}).get('files_sha256') != (actual.get(k) or {}).get('files_sha256'))
    assert server4['versions'] == devbox['versions'], (server4['versions'], devbox['versions'])
    return dict(distributions=len(actual), differences=differences, versions=server4['versions'],
                python=dict(server4=server4['python'], devbox=devbox['python']), pip='devbox venv only; unused')


def compare_repair(server4, devbox):
    a, b = server4['repair'], devbox['repair']
    assert server4['source']['sha256'] == devbox['source']['sha256'] and server4['count'] == devbox['count']
    assert a['policy'] == b['policy'] == 'project_roundoff' and a['repaired'] and b['repaired']
    assert a['corrected_eigenvalues'] == b['corrected_eigenvalues'] and a['shape'] == b['shape']
    close = lambda x, y, rel: abs(x - y) <= rel * max(abs(y), 1e-300)
    eigen = max(abs(x - y) for x, y in zip(a['original_eigenvalues'], b['original_eigenvalues']))
    assert eigen <= 1e-3 * a['eigenvalue_floor'], eigen
    assert close(a['eigenvalue_floor'], b['eigenvalue_floor'], 1e-9) and close(a['matrix_frobenius'], b['matrix_frobenius'], 1e-9)
    assert close(a['correction_frobenius'], b['correction_frobenius'], 1e-3)
    return dict(corrected_eigenvalues=a['corrected_eigenvalues'], max_eigenvalue_difference=eigen,
                eigenvalue_floor=(a['eigenvalue_floor'], b['eigenvalue_floor']),
                correction_frobenius=(a['correction_frobenius'], b['correction_frobenius']),
                derived_sha256=dict(server4=server4['derived']['sha256'], devbox=devbox['derived']['sha256']),
                bitwise_equal=server4['derived']['sha256'] == devbox['derived']['sha256'])


def native_script(python, author, hf_cache, tmp, run, overrides, eval_overrides, need):
    check = ('import shutil,sys;free=shutil.disk_usage(%r).free;need=%d;'
             'sys.exit(0 if free>=need else "NATIVE_DISK_PREFLIGHT free=%%d need=%%d"%%(free,need))') % (str(run), need)
    lines = ['#!/bin/bash', 'set -euo pipefail', 'export PYTHONDONTWRITEBYTECODE=1', 'export TOKENIZERS_PARALLELISM=false',
             'export HF_HUB_OFFLINE=1', 'export TRANSFORMERS_OFFLINE=1', 'export HF_HUB_CACHE=' + shlex.quote(str(hf_cache)),
             'export OMP_NUM_THREADS=8', 'export MKL_NUM_THREADS=8', 'export OPENBLAS_NUM_THREADS=8',
             'export TMPDIR=' + shlex.quote(str(tmp)), 'cd ' + shlex.quote(str(author)),
             shlex.join([python, '-B', '-c', check]),
             shlex.join([python, '-B', '-u', 'main.py', *overrides]),
             shlex.join([python, '-B', '-u', 'main.py', *eval_overrides])]
    return '\n'.join(lines) + '\n'


def evaluation_script(python, source, tmp, config, output, lock):
    lines = ['#!/bin/bash', 'set -euo pipefail', 'export PYTHONDONTWRITEBYTECODE=1', 'export TOKENIZERS_PARALLELISM=false',
             'export HF_HUB_OFFLINE=1', 'export TRANSFORMERS_OFFLINE=1', 'export OMP_NUM_THREADS=8',
             'export MKL_NUM_THREADS=8', 'export OPENBLAS_NUM_THREADS=8', 'export TMPDIR=' + shlex.quote(str(tmp)),
             'cd ' + shlex.quote(str(source)),
             'exec ' + shlex.join([python, '-B', '-u', '-m', 'official.diagnostics.fe_native.evaluate',
                                   '--native-config', str(config), '--output', str(output), '--lock', str(lock)])]
    return '\n'.join(lines) + '\n'


def relative_members(rows, root):
    return {str(Path(m['path']).relative_to(root)): (m['sha256'], m['bytes']) for m in rows}


def loader_check(python, author, run, derived):
    """CPU: the author's own get_cov reads the derived L3 sum and normalizes it exactly, FP64, PD."""
    code = '''
import json, sys, numpy as np, torch
from types import SimpleNamespace
from omegaconf import OmegaConf
sys.path.insert(0, '.')
from locate_edit_utils.layer_stats import get_cov
cfg = OmegaConf.load('configs/config.yaml')
cfg.llms = OmegaConf.load('configs/llms/gpt-j-6b.yaml'); cfg.algs = OmegaConf.load('configs/algs/memit.yaml')
cfg.cache_dir = sys.argv[1]
model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048))
cov = get_cov(cfg, model, None, 3, cfg.llms.mom2_dataset, cfg.llms.mom2_n_samples, cfg.llms.mom2_dtype,
              force_recompute=False, cache_filename_suffix='')
with np.load(sys.argv[2], allow_pickle=False) as stored:
    expected = torch.from_numpy(stored['mom2.mom2'].copy()) / int(stored['mom2.count'])
info = int(torch.linalg.cholesky_ex(cov)[1])
print('NATIVE_LOADER_CHECK', json.dumps(dict(dtype=str(cov.dtype), shape=list(cov.shape),
      exact_derived_normalization=bool(torch.equal(cov, expected)), cholesky_info=info)))
'''
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8',
               MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
    text = subprocess.check_output([python, '-B', '-c', code, str(Path(run)/'cache'), str(derived)], cwd=author,
                                   env=env, text=True)
    line = next(x for x in text.splitlines() if x.startswith('NATIVE_LOADER_CHECK '))
    result = json.loads(line.split(' ', 1)[1])
    assert result == dict(dtype='torch.float64', shape=[GPTJ_INTERMEDIATE]*2, exact_derived_normalization=True,
                          cholesky_info=0), result
    return result


def prepare_gptj(output, inputs, derived, commit):
    from omegaconf import OmegaConf
    from official.diagnostics.fe_native.patch import install, AUTHOR_COMMIT as PATCH_AUTHOR
    from official.evaluation.zsre_query_parity import compare_queries
    server4_cpu()
    assert PATCH_AUTHOR == AUTHOR_COMMIT
    output, inputs = Path(output).absolute(), Path(inputs).absolute()
    assert output.is_relative_to(LOCAL) and not output.exists()
    control = Path(__file__).resolve().parents[3]
    source, archive = official_source(commit, control)
    reference = read(inputs/'reference.json')
    files = {name: verify(row) for name, row in reference['files'].items()}
    template = read(files['native_config'])
    assert digest({k: v for k, v in template.items() if k != 'identity_sha256'}) == template['identity_sha256'] == GPTJ_TEMPLATE_IDENTITY
    assert (template['model'], template['arm'], template['z_reference'], template['covariance_repair']) == ('gptj', 'gptj_repaired', 'batch_entry', True)
    devbox_patch, devbox_derived = read(files['patch_receipt']), read(files['derived_receipt'])
    devbox_assets, devbox_runtime = read(files['gptj_assets']), read(files['native_runtime'])
    assert devbox_assets['assets_sha256'] == digest({k: v for k, v in devbox_assets.items() if k != 'assets_sha256'})
    assert member(files['gptj_assets'])['sha256'] == template['model_assets']['sha256']

    runtime = native_runtime_receipt()
    runtime_check = compare_runtime(runtime, devbox_runtime)
    assert runtime_check['differences'] == [], runtime_check
    assert runtime['versions'] == {k: template['runtime'][k] for k in ('torch', 'transformers', 'hydra', 'numpy')}
    output.mkdir(parents=True)
    tmp = output/'tmp'
    tmp.mkdir()

    # Pinned author tree + the unchanged target-supply patch (native_checkpoint_observer differs: history option).
    author = output/'author'
    author.mkdir()
    tree = git('archive', '--format=tar', AUTHOR_COMMIT, cwd=AUTHOR_CLONE, text=False)
    with tarfile.open(fileobj=io.BytesIO(tree)) as bundle:
        bundle.extractall(author, filter='data')
    patch = install(author)
    for key in ('author_commit', 'main_sha256', 'upstream_fe_sha256', 'patched_fe_sha256', 'patch'):
        assert patch[key] == devbox_patch[key], key
    patch.update(ode_patch_source=commit)
    write_new(output/'patch-receipt.json', patch)
    source_members = [member(p) for p in sorted(author.rglob('*')) if p.is_file()]
    mine, theirs = relative_members(source_members, author), relative_members(template['source_members'], template['root'])
    differences = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    observer = 'algs/memit/native_checkpoint_observer.py'
    assert differences == [observer], differences
    assert mine[observer][0] == file_sha(control/'official/diagnostics/fe_native/checkpoint_observer.py')

    # Model/tokenizer and C0 stats: devbox manifest mapped to server4 paths, every byte re-hashed.
    assets_path = output/'assets-gptj.json'
    write_new(assets_path, rehash_assets(devbox_assets, PATH_MAP, runtime_info=runtime['versions']))
    assets = read(assets_path)
    snapshot = Path(assets['model']['snapshot'])
    assert snapshot.name == GPTJ_REVISION == assets['model']['identity']['revision']
    print('GPTJ_MODEL_AND_C0_FULL_SHA_PASS', flush=True)
    hf = output/'hf-cache'
    alias = hf/'models--EleutherAI--gpt-j-6B'  # author YAML name; server4's own cache folder has no refs/main
    (alias/'refs').mkdir(parents=True)
    (alias/'refs'/'main').write_text(GPTJ_REVISION)
    (alias/'snapshots').mkdir()
    (alias/'snapshots'/GPTJ_REVISION).symlink_to(snapshot, target_is_directory=True)
    assert (alias/'refs'/'main').read_text().strip() == snapshot.name

    # Official zsRE first 2,000 -> author records; verified against the author's own zsRE file.
    stream = output/'inputs'/'zsre-stream.json'
    stream.parent.mkdir()
    shutil.copyfile(files['zsre_stream'], stream)
    assert file_sha(stream) == ZSRE_STREAM_SHA == file_sha(LLAMA_TEMPLATE/'inputs'/'zsre-stream.json')
    records = read(stream)
    assert len(records) == 2000 and digest(records) == ZSRE_RECORDS_DIGEST
    assert digest([r['case_id'] for r in records]) == ZSRE_ORDERED_IDS
    rows = native_zsre_rows(records)
    data_check = compare_author_zsre(rows, read(author/'data'/'zsre_mend_eval_19086.json'))
    data_path = output/'data'/(DATA_NAME + '.json')
    write_new(data_path, rows)
    assert [r['case_id'] for r in read(data_path)] == [r['case_id'] for r in records]

    # Covariances: L4-L8 original NPZ, L3 the derived project_roundoff NPZ recomputed on server4.
    run = output/'runs'/ARM
    receipt = read(derived)
    repair_check = compare_repair(receipt, devbox_derived)
    verify(receipt['derived'])
    covariances = []
    template_layers = {row['layer']: row for row in template['covariances']}
    for layer in GPTJ_LAYERS:
        original = assets['C0'][str(layer)]['member']
        assert original['sha256'] == template_layers[layer]['original']['sha256']
        repair = receipt if layer == 3 else None
        if repair:
            assert repair['source']['sha256'] == original['sha256'] and Path(repair['source']['path']) == Path(original['path'])
        execution = repair['derived'] if repair else original
        link = run/'cache'/'stats'/'EleutherAI-gpt-j-6B'/f'layer-{layer}.npz'
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(execution['path'])
        assert file_sha(link) == execution['sha256']
        covariances.append(dict(layer=layer, original=original, execution=execution, link=str(link), repair=repair))
    assert [row['layer'] for row in covariances] == sorted(template_layers)
    loader = loader_check(sys.executable, author, run, receipt['derived']['path'])

    overrides = native_overrides(run, output/'data')
    template_fixed = [x for x in template['overrides'] if x in FIXED_OVERRIDES]
    assert template_fixed == [x for x in overrides if x in FIXED_OVERRIDES] and len(template_fixed) == len(FIXED_OVERRIDES)
    spec = dict(model='gptj', arm=ARM, dataset='zsre', server=NODE, task_id=TASK, author_commit=AUTHOR_COMMIT,
                ode_patch_source=commit, entrypoint='python main.py', root=str(author), runtime=runtime['versions'],
                python=sys.executable, source_members=source_members, model_assets=member(assets_path),
                covariances=covariances, stream=member(data_path), original_stream=member(stream),
                overrides=overrides, model_dtype='bfloat16', current_batch_targets_only=True, z_reference='batch_entry',
                history='native add_old_keys=True', covariance_repair=True, context_override=False,
                writer_override=False, requests=2000, batch_size=100, gpu_authorization=INSTRUCTION,
                output=str(run), tracking='native stdout and native evaluation files; no ODE tracker injected',
                template_native_config=dict(member(files['native_config']), job='63389', evaluation_job='63390',
                                            devbox_path=reference['devbox']['native_config']),
                scientific_difference='Dataset only: official zsRE first 2,000 (= author zsre_mend_eval_19086[:2000] '
                                      'in every field read) instead of CF; author hparams, BF16, native writer, history, '
                                      'current batch-entry z and the L3 project_roundoff C0 are those of 63389',
                storage_difference='diagnostic_history_batches=[20]: W0/W5/W10/W15 snapshots keep weights, contexts, '
                                   'RNG and case IDs without the history tensor; editing is unaffected')
    for key in SCIENCE_FIELDS:
        assert spec[key] == template[key], key
    spec['identity_sha256'] = digest(spec)
    config_path = output/'configs'/f'{ARM}.json'
    write_new(config_path, spec)
    actual = overrides + ['+diagnostic_identity_sha256=' + spec['identity_sha256']]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', HF_HUB_CACHE=str(hf),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TMPDIR=str(tmp))
    dry = subprocess.check_output([sys.executable, '-B', 'main.py', '--cfg', 'job', *actual], cwd=author, env=env,
                                  text=True, stderr=subprocess.STDOUT)
    (output/f'{ARM}-resolved.yaml').write_text(dry)
    resolved = OmegaConf.create(dry[min(dry.index('algs:'), dry.index('llms:')):])
    container = lambda value: OmegaConf.to_container(value, resolve=True)
    assert container(resolved.llms) == container(OmegaConf.load(author/'configs/llms/gpt-j-6b.yaml'))
    assert container(resolved.algs) == container(OmegaConf.load(author/'configs/algs/memit.yaml'))
    assert (resolved.model_dtype, resolved.bs, resolved.num_edits, resolved.seed, resolved.data) == ('bfloat16', 100, 2000, 0, DATA_NAME)
    assert resolved.diagnostic_z_reference == 'batch_entry' and list(resolved.diagnostic_history_batches) == HISTORY_BATCHES
    assert resolved.diagnostic_identity_sha256 == spec['identity_sha256'] and resolved.algs.add_old_keys

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'  # exactly the evaluator's tokenizer setup
    parity = compare_queries(tok, records, model_family='gptj')
    assert parity['requests'] == 2000 and parity['input_mismatches'] == parity['target_mismatches'] == 0
    write_new(output/'zsre-query-parity.json', parity)

    evaluation_source = output/'evaluation-source'
    evaluation_members = extract_official(archive, evaluation_source)
    lock_path = output/'evaluation-lock.json'
    write_new(lock_path, dict(source_commit=commit, official_tree=source['official_tree'],
                              source_directory=str(evaluation_source), source_members=evaluation_members,
                              configs=[member(config_path)]))
    need = native_disk_need()
    eval_overrides = [x for x in actual if not x.startswith('hydra.run.dir=')] + ['test_only=True', f'hydra.run.dir={run}/hydra-eval']
    scripts = {}
    for name, text in ((f'{ARM}.sh', native_script(sys.executable, author, hf, tmp, run, actual, eval_overrides, need)),
                       (f'{ARM}-eval.sh', evaluation_script(sys.executable, evaluation_source, tmp, config_path,
                                                            output/'evaluations'/ARM, lock_path))):
        path = output/name
        with path.open('x') as handle:
            handle.write(text)
        path.chmod(0o755)
        scripts[name] = member(path)
    result = dict(instruction=INSTRUCTION, server=NODE, model='gptj', dataset='zsre', source=source,
                  config=member(config_path), identity_sha256=spec['identity_sha256'],
                  native_script=scripts[f'{ARM}.sh'], evaluation_script=scripts[f'{ARM}-eval.sh'],
                  evaluation_lock=member(lock_path), patch_receipt=member(output/'patch-receipt.json'),
                  author_tree=dict(files=len(mine), differences_from_63389=differences),
                  runtime_check=runtime_check, data_check=data_check, repair_check=repair_check, loader_check=loader,
                  zsre_query_parity=dict(member(output/'zsre-query-parity.json'), queries=parity['queries']),
                  disk=dict(edit_job_need_bytes=need, free_bytes=shutil.disk_usage(output).free),
                  template=spec['template_native_config'], GPU_execution=False)
    write_new(output/'preparation.json', result)
    print(json.dumps({k: result[k] for k in ('identity_sha256', 'author_tree', 'repair_check', 'disk')}, indent=1), flush=True)


# ---------------------------------------------------------------- registration (held -> inspect -> release)

def lane_rows(existing, keys=('LANE_B_LLAMA', 'LANE_B_GPTJ', 'LANE_B_GPTJ_EVAL')):
    """Planned serial lane appended to the current server4 GPU jobs (DAG width = max concurrent GPUs)."""
    rows = [dict(key=r['key'], gpus=r['gpus'], parents=list(r['parents'])) for r in existing]
    parent = []
    for key in keys:
        rows.append(dict(key=key, gpus=1, parents=parent))
        parent = [key]
    return rows


def register(llama_preparation, gptj_preparation, attempt, commit):
    from official.baselines.fe_author_profile import PROFILE
    from official.runners.fe_author_history import validate_config, tracking_values
    from official.runners.server1 import submit as control
    from official.runners.server1.fe_history_prepare import runtime
    from official.tracking.schema import config as tracking_config, load_env
    assert socket.gethostname() == NODE
    attempt = Path(attempt).absolute()
    assert attempt.is_relative_to(LOCAL) and not attempt.exists(), 'EXISTING_ATTEMPT_RECONCILE'
    control_root = Path(__file__).resolve().parents[3]
    source, _ = official_source(commit, control_root)
    llama_prep, gptj_prep = read(llama_preparation), read(gptj_preparation)
    assert llama_prep['instruction'] == gptj_prep['instruction'] == INSTRUCTION
    assert gptj_prep['source']['commit'] == commit
    cm = llama_prep['config']
    c = validate_config(read(verify(cm)))
    assert c['server'] == NODE and not c.get('checkpoint_archive')
    verify(c['assets']); verify(c['stream_member']); verify(c['zsre_query_parity'])
    assert c['author_profile_sha256'] == file_sha(PROFILE)
    tracking_config(tracking_values(c, {'source_commit': commit}))
    load_env(c['tracking_env_file'])
    assert runtime() == c['runtime'], 'REGISTER_WITH_THE_RUN_PYTHON'
    native = read(verify(gptj_prep['config']))
    assert digest({k: v for k, v in native.items() if k != 'identity_sha256'}) == native['identity_sha256']
    for row in (gptj_prep['native_script'], gptj_prep['evaluation_script'], native['stream'], native['original_stream'],
                native['model_assets']):
        verify(row)
    lock = read(verify(gptj_prep['evaluation_lock']))
    for row in lock['source_members'] + lock['configs']:
        verify(row)
    for row in native['covariances']:
        assert file_sha(row['link']) == row['execution']['sha256']
    for row in native['source_members']:
        verify(row)

    before = control.inventory(node=NODE)
    planned = control.graph_width(lane_rows(before['jobs']))
    assert planned <= CAP, ('SERVER4_CAP', planned, before)
    free = shutil.disk_usage(LOCAL).free
    budget = 2 * (5 << 30) + c['storage_min_free_bytes'] + native_disk_need() - NATIVE_RESERVE
    assert free >= budget, ('SERVER4_STORAGE', free, budget)
    attempt.mkdir(parents=True)
    (attempt/'logs').mkdir()
    (attempt/'scripts').mkdir()
    write_new(attempt/'admission.json', dict(instruction=INSTRUCTION, cap=CAP, planned_width=planned,
                                             inventory_before=before, free_bytes=free, budget_bytes=budget))
    frozen = control.freeze_source(dict(source=dict(main_commit=commit, official_tree=source['official_tree'])),
                                   attempt, repository=REPOSITORY)
    llama_lock = dict(instruction=INSTRUCTION, source_commit=commit, official_tree=source['official_tree'],
                      source_directory=frozen['directory'], source_members=frozen['members'], configs=[cm],
                      preparation=member(llama_preparation),
                      author_profile=member(Path(frozen['directory'])/'official/hparams/MEMIT_FE_HISTORY_AUTHOR/profiles.json'),
                      GPU_qualification='NOT_RUN_USER_DISABLED')
    lock_path = attempt/'execution-lock-llama.json'
    write_new(lock_path, llama_lock)
    env = dict(PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', OFFICIAL_CODE_COMMIT=commit,
               OFFICIAL_TREE_SHA256=source['official_tree'], WANDB_CONSOLE='off', WANDB_SAVE_CODE='false',
               TMPDIR=str(attempt/'tmp'))
    (attempt/'tmp').mkdir()
    argv = [EASYEDIT_PYTHON, '-B', '-u', '-m', 'official.runners.fe_author_history', '--config', cm['path'], '--lock', str(lock_path)]
    text = '#!/bin/bash\nset -euo pipefail\n' + ''.join('export ' + k + '=' + shlex.quote(v) + '\n' for k, v in env.items())
    text += 'cd ' + shlex.quote(frozen['directory']) + '\nexec ' + shlex.join(argv) + '\n'
    llama_script = attempt/'scripts'/'llama3-zsre.sh'
    with llama_script.open('x') as handle:
        handle.write(text)
    llama_script.chmod(0o755)
    jobs = [('llama3-zsre', 'official-s4-llama3-zsre-fe-author-history-local', llama_script, Path(frozen['directory']),
             MEMORY_MiB, 48*3600),
            ('gptj-zsre', 'official-s4-gptj-zsre-fe-native-repaired', Path(gptj_prep['native_script']['path']),
             Path(native['root']), MEMORY_MiB, 12*3600),
            ('gptj-zsre-eval', 'official-s4-gptj-zsre-fe-native-repaired-eval', Path(gptj_prep['evaluation_script']['path']),
             Path(lock['source_directory']), 49152, 8*3600)]
    submitted, records, released, parent = [], {}, False, None
    try:
        for key, name, script, workdir, memory, wall in jobs:
            argv = ['sbatch', '--parsable', '--hold', '--export=NONE', '--no-requeue', '--kill-on-invalid-dep=yes',
                    '--partition=' + PARTITION, '--qos=' + QOS, '--nodelist=' + NODE, '--gres=gpu:1', '--cpus-per-task=8',
                    f'--mem={memory}M', f'--time={wall // 60}', '--job-name=' + name, '--chdir=' + str(workdir),
                    '--output=' + str(attempt/'logs'/(key + '-%j.out')), '--error=' + str(attempt/'logs'/(key + '-%j.err'))]
            if parent:
                argv.append('--dependency=afterany:' + parent)
            argv.append(str(script))
            job = control.command(argv).split(';')[0]
            assert job.isdigit()
            submitted.append(job)
            d = control.metadata(control.command(['scontrol', 'show', 'job', job, '--oneliner']))
            assert d['JobState'] == 'PENDING' and d['Reason'] == 'JobHeldUser' and d['UserId'].split('(')[0] == getpass.getuser()
            assert d['JobName'] == name and d['Command'] == str(script) and d['WorkDir'] == str(workdir)
            assert d['ReqNodeList'] == NODE and d['QOS'] == QOS and d['Partition'] == PARTITION and d['Requeue'] == '0'
            assert control.gpu_count(d['ReqTRES']) == 1 and control.memory_MiB(d['MinMemoryNode']) == memory
            assert control.requested_cpu_matches(d, 8) and control.seconds(d['TimeLimit']) == wall
            assert control.dependencies(d['Dependency']) == ([('afterany', parent)] if parent else [])
            assert control.command(['scontrol', 'write', 'batch_script', job, '-']).strip() == Path(script).read_text().strip()
            records[key] = dict(job_id=job, name=name, parent=parent, argv=argv, script=member(script), held=d)
            parent = job
        held = control.inventory(node=NODE)
        assert held['admitted_DAG_width'] <= CAP, held
        write_new(attempt/'held-inspection.json', dict(jobs=records, inventory=held, llama_lock=member(lock_path)))
        for job in reversed(submitted):
            control.command(['scontrol', 'release', job])
        released = True
        after = control.inventory(node=NODE)
        assert after['admitted_DAG_width'] <= CAP, after
        write_new(attempt/'submission.json', dict(instruction=INSTRUCTION, source=source, jobs=records,
                  chain=submitted, inventory_after=after, gptj_preparation=member(gptj_preparation),
                  llama_preparation=member(llama_preparation), released=True, automatic_retry=False,
                  other_jobs_changed=0))
        print(json.dumps(dict(chain=submitted, width=after['admitted_DAG_width'])), flush=True)
    except BaseException:
        if not released:
            for job in submitted:
                subprocess.run(['scancel', job], check=False)
        write_new(attempt/'submission-failure.json', dict(submitted=submitted, released=released,
                                                          error=__import__('traceback').format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare-llama')
    a.add_argument('--output', required=True)
    a.add_argument('--wandb-env', required=True)
    b = sub.add_parser('prepare-gptj')
    b.add_argument('--output', required=True)
    b.add_argument('--inputs', required=True)
    b.add_argument('--derived', required=True)
    b.add_argument('--source-commit', required=True)
    r = sub.add_parser('register')
    r.add_argument('--llama', required=True)
    r.add_argument('--gptj', required=True)
    r.add_argument('--attempt', required=True)
    r.add_argument('--source-commit', required=True)
    x = parser.parse_args()
    if x.command == 'prepare-llama':
        prepare_llama(x.output, x.wandb_env)
    elif x.command == 'prepare-gptj':
        prepare_gptj(x.output, x.inputs, x.derived, x.source_commit)
    else:
        register(x.llama, x.gptj, x.attempt, x.source_commit)


if __name__ == '__main__':
    main()
