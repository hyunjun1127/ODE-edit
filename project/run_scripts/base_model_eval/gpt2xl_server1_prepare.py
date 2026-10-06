"""CPU-only exact W0 inputs/runtime binding, source freeze and launcher maker."""
import getpass
import importlib
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path

from .gpt2xl_server1_common import *


def cpu_imports():
    from scripts.fixed_counterfact import load_prefix
    from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
    from project.run_scripts.jlz_price_gpt2xl.scores import scores
    from project.run_scripts.jlz_realized_writer_sequential import review_completed
    from project.run_scripts.jlz_interference_l1.comparison_bridge import metric_row
    from project.run_scripts.experiment_tracking import schema
    return load_prefix, CounterFactAdapter, scores, review_completed, metric_row, schema


def runtime():
    import torch
    import transformers
    require(not torch.cuda.is_initialized(), 'CPU_PREP_ONLY')
    names = ('torch', 'torch.nn.functional', 'transformers',
        'transformers.models.gpt2.modeling_gpt2', 'transformers.pytorch_utils',
        'transformers.modeling_utils', 'transformers.tokenization_utils_base',
        'transformers.tokenization_utils_fast', 'transformers.models.gpt2.tokenization_gpt2_fast')
    require((str(torch.__version__), transformers.__version__) == ('2.9.1+cu128', '4.57.1'), 'W0_PINNED_RUNTIME')
    return dict(python=PYTHON, torch=str(torch.__version__), transformers=transformers.__version__,
        FP32=True, eager=True, TF32=False, autocast=False, geometry='NOT_APPLICABLE',
        source_members=[dict(module=name, **member(importlib.import_module(name).__file__)) for name in names])


def prepare(attempt, output=None):
    from transformers import AutoTokenizer
    import torch
    require(not torch.cuda.is_initialized(), 'W0_PREP_NO_GPU')
    attempt = Path(attempt).resolve()
    require(attempt.parent == LOCAL and not attempt.exists(), 'W0_CREATE_ONCE_NAMESPACE')
    require(not list(LOCAL.glob('*/submission*.json')), 'W0_NONCE_ALREADY_REGISTERED')
    require(sha(ROOT / ENVELOPE) == ENVELOPE_SHA, 'W0_AUTHORITY_BYTES')
    auth = read(ROOT / ENVELOPE)
    require(auth['instruction_id'] == NONCE and auth['owner']['session'] == SESSION
            and auth['scope']['fresh_cold_base_model'] and auth['scope']['edit_calls'] == 0,
            'W0_AUTHORITY_SCOPE')
    old = read(OLD_CONFIG)
    profile = old['models']['MEMIT']
    ready_member = old['input_reuse_ready']
    ready = read(verify(ready_member))
    require(ready['model_asset_identity'] == profile['model_asset_identity']
            and ready['ordered_ids_sha256'] == ORDER_SHA, 'W0_REFERENCE_INPUT_READY')
    inputs = [ready_member, ready['contexts_member'], ready['observer_identity'], member(old['stream'])]
    for row in inputs:
        verify(row)
    model = Path(profile['model'])
    revision = subprocess.check_output(['git', '-C', str(model), 'rev-parse', 'HEAD'], text=True).strip()
    require(revision == MODEL_REVISION, 'W0_MODEL_REVISION')
    assets = [row for row in profile['assets'] if Path(row['path']).parent == model]
    payload = next(row for row in assets if Path(row['path']).name == 'model.safetensors')
    require(payload['sha256'] == PAYLOAD_SHA, 'W0_MODEL_PAYLOAD_SEAL')
    for row in assets:
        verify(row, metadata_only=row is payload)
    cfg = read(model / 'config.json')
    require((cfg['n_layer'], cfg['n_embd'], cfg['n_positions'], cfg['vocab_size'])
            == (48, 1600, 1024, 50257), 'W0_MODEL_CONFIG')
    load_prefix, Adapter, scorer, independent, metric_row, schema = cpu_imports()
    references = {Path(row['path']).name:row for row in profile['evaluator_sources']}
    for module_name in ('project.run_scripts.jlz_price_gpt2xl.scores',
                        'project.run_scripts.jlz_realization.inputs', 'project.run_scripts.jlz_pilot.prompts'):
        row = member(importlib.import_module(module_name).__file__)
        require(row['sha256'] == references[Path(row['path']).name]['sha256'], 'W0_SCORER_REFERENCE_BYTES')
    records = load_prefix(Path(old['stream']).parent, 2000)
    require(digest([row['case_id'] for row in records]) == ORDER_SHA, 'W0_FIRST2000_ORDER')
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    bench = Adapter(tok, read(ready['contexts_member']['path']))
    identities = token_identity(records, bench)
    require(identities == read(ready['observer_identity']['path'])['rows'], 'W0_FULL_TOKEN_ROW_IDENTITY')
    require(len(identities) == 26000, 'W0_FULL_PAIR_DENOMINATOR')
    schema.config(dict(server='server1', task_id=TASK, arm='W0_BASE_MODEL', attempt=attempt.name,
        source_sha='a'*40, config_sha='b'*64, model='gpt2xl', model_family='gpt2',
        writer='none', role='scientific', metric_schema=SCHEMA))
    c = dict(schema=1, instruction_id=NONCE, task_id=TASK, attempt=str(attempt),
        run_instance=dict(attempt=attempt.name), model=str(model), model_revision=revision,
        model_assets=assets, model_asset_identity=digest(assets), original_asset_identity=profile['model_asset_identity'],
        cold_selected_W=profile['cold_W0_H0']['W'],
        input_members=inputs, observer_identity=ready['observer_identity'], contexts=ready['contexts_member'],
        stream=old['stream'], ordered_ids_sha256=ORDER_SHA, seed=20261002,
        runtime=runtime(), denominators=DENOMINATORS, prompt_pairs=26000, candidate_rows=52000,
        observer_microbatch=2, observer_request_chunk=50,
        evaluator_identity=dict(rows_sha256=digest(identities), tokenizer_padding='right',
            scorer_padding='left per original adjacent new/true MB2 group',
            GPT2_position='native default arange, no cumsum', final_LN='transformer once'),
        old_W0_CPU_crosscheck=profile.get('W0_reuse'), fresh_observation_required=True,
        source_reference=member(OLD_CONFIG), authority=member(ROOT / ENVELOPE),
        tracking=dict(env_file=old['tracking']['env_file'], metric_schema=SCHEMA,
            progress_axis='explicit step=processed requests; performance edits=0; no fit candidate axis'),
        resources=dict(gpu=1, cpu=8, host_mib=65536, wall='04:00:00',
            collector_cpu=8, collector_host_mib=24576, collector_wall='02:00:00',
            scoped_project_cap_exception=True, global_cap_changed=False,
            dependency=None, ETA='NOT_MEASURED;4h is wall request only', storage=capacity(LOCAL)),
        save_checkpoints=False, exact_resume='NOT_AVAILABLE', edit_calls=0, target_fits=0,
        solves=0, history_appends=0, stats_loads=0, projector_loads=0,
        model_payload_identity='prior exact SHA + unchanged inode/mtime/size; no second large payload hash',
        broadcast='NO_BROADCAST_NOT_REQUIRED; same-host raw local KEEP')
    attempt.mkdir(parents=True)
    output = Path(output or attempt / 'config.json')
    require(output.parent == attempt, 'W0_CONFIG_OWN_NAMESPACE')
    write(output, c)
    write(attempt / 'cpu-input-preflight.json', dict(status='CPU_PASS_NOT_MODEL_PASS',
        identity_rows=26000, candidate_rows=52000, requests=2000, runtime=c['runtime'],
        model_asset_seal_reused=True, C0_P_loaded=False, model_loaded=False,
        GPU=False, nonce=NONCE, config_sha256=sha(output)))
    return c


def source_files():
    """Only evaluator/reducer/tracking/import closure, never fitting runtime."""
    cpu_imports()
    import sys
    files = {Path(__file__).with_name(name) for name in (
        'gpt2xl_server1.py', 'gpt2xl_server1_common.py', 'gpt2xl_server1_prepare.py',
        'gpt2xl_server1_collect.py', 'gpt2xl_server1_tests.py','gpt2xl_server1_submit.py')}
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if file and (name.startswith('project.run_scripts.') or name == 'scripts.fixed_counterfact'):
            path = Path(file).resolve()
            if path.is_relative_to(ROOT) and path.suffix == '.py':
                files.add(path)
    files.update((ROOT / 'project/run_scripts/experiment_tracking').glob('*.py'))
    files.add(ROOT / ENVELOPE)
    files.update(ROOT / 'control' / name for name in ('wandb-policy.json', 'wandb-method-metric-schema.json'))
    return sorted(path.relative_to(ROOT).as_posix() for path in files)


def freeze(config_path, source_commit):
    config = read(config_path)
    attempt = Path(config['attempt'])
    source = attempt / 'source'
    require(not source.exists() and not (attempt / 'execution.lock.json').exists(), 'W0_SOURCE_CREATE_ONCE')
    files = source_files()
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--', *files], cwd=ROOT, text=True),
            'W0_COMMIT_SOURCE_BEFORE_FREEZE')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(head == source_commit, 'W0_SOURCE_HEAD')
    archive = attempt / 'source.tar'
    subprocess.run(['git', 'archive', '--format=tar', '--output=' + str(archive), head, *files], cwd=ROOT, check=True)
    source.mkdir()
    with tarfile.open(archive) as stream:
        items = stream.getmembers()
        require(len(items) == len({item.name for item in items}) and all(
            (item.isfile() or item.isdir()) and not Path(item.name).is_absolute()
            and '..' not in Path(item.name).parts for item in items), 'W0_SAFE_SOURCE_ARCHIVE')
        stream.extractall(source, filter='data')
    source_members = [member(source / relative) for relative in files]
    require(all(sha(ROOT / relative) == row['sha256'] for relative, row in zip(files, source_members)),
            'W0_FROZEN_SOURCE_IDENTITY')
    lock = dict(instruction_id=NONCE, task_id=TASK, source_commit=head,
        source_tree=subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip(),
        archive=member(archive), source_root=str(source), source_members=source_members,
        config_sha256=sha(config_path), owner=getpass.getuser(), server='server1', session=SESSION,
        resources=config['resources'], noCP=True, fresh_W0_required=True, no_edit_fit_solve_H_C0_P=True)
    lock['tracking_env']=member(config['tracking']['env_file'])
    write(attempt / 'execution.lock.json', lock)
    return lock


def make_launchers(config_path, lock_path, partition='gpu', node='devbox', qos='lab_gpu_s1'):
    config, lock = verify_config_lock(config_path, lock_path)
    attempt = Path(config['attempt'])
    scripts = {}
    for role in ('GPU', 'collector'):
        env = dict(PYTHONPATH=lock['source_root'], PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8',
            MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1', ODEEDIT_W0_SOURCE_COMMIT=lock['source_commit'])
        if role == 'collector':
            env['CUDA_VISIBLE_DEVICES'] = ''
        argv = [PYTHON, '-u', '-m', 'project.run_scripts.base_model_eval.gpt2xl_server1',
            'collect' if role == 'collector' else 'run', '--config', str(config_path), '--lock', str(lock_path)]
        script = '#!/bin/bash\nset -euo pipefail\n' + ''.join(
            'export ' + key + '=' + shlex.quote(value) + '\n' for key, value in env.items())
        script += 'cd ' + shlex.quote(lock['source_root']) + '\nexec ' + shlex.join(argv) + '\n'
        path = attempt / (role + '.sh')
        write_bytes(path, script.encode())
        path.chmod(0o755)
        scripts[role] = member(path)
    receipt = dict(scripts=scripts, partition=partition, node=node, qos=qos,
        GPU_dependency=None, collector_dependency='afterany:ACTUAL_NEW_W0_JOB_ID',
        resources=config['resources'], export='NONE', requeue=False,
        actual_job_ids=[], registered=False, scoped_project_cap_exception=True, no_cap_file_change=True)
    write(attempt / 'launchers.json', receipt)
    return receipt
