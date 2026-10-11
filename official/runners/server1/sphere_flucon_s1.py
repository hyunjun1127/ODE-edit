"""Server1 (devbox) resubmission of the cancelled 62873 CF Flu/Con of the Qwen SPHERE W20 (edit run 62079).

User 2026-10-11, "server1 cap 남으니 그냥 올려": resubmit on server1 the cancelled
`s2-flucon-qwen25-sphere-62079` (62873). The science is 62873's: the shared FP32 evaluator
official.runners.server1.flucon_eval.run, unchanged (FP32 eager, TF32 off, NativeGenerationObserver
PROFILE, eval seed 20261007, reference identity 75e595c7..., CF first 2,000 in the 62079 order, x100
display), and 62873's own restore, official.runners.server2.flucon_eval.restore (W20 pointer checks,
then the exact FP32 copy of the five selected down_proj weights; history/contexts/RNG not consumed).
Only the host labels and the restore are bound process-locally, as in server2/flucon_eval.

`prepare` derives the config from 62873's own config (copied from server2 with SHA-256 on both sides,
bound to the 62873 registration records) and changes only host bindings: the copied input paths,
this host's stat manifest of the same model files, this host's reference manifest of the same
reference identity, the evaluator path, the runtime record, the output and the authority. Any other
field difference is refused. No model forward, edit or checkpoint change happens here.
"""
import argparse
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import re

import torch

from official.evaluation.generation.assets import load_assets
from official.experiments.prepare import digest, write_new
from official.runners.server1 import flucon_eval as shared
from official.runners.server1.assets import member as asset_member
from official.runners.server1.common import member, read, verify
from official.runners.server1.fe_history_prepare import runtime
from official.runners.server1.sphere_s1_common import (AUTHORITY, LOCAL, TASK, differences, field_diff, host_only,
                                                       received, require, transfer_rows)
from official.runners.server2 import flucon_eval as server2
from official.tracking.schema import config as validate_tracking

KEY = 'qwen25-sphere-62079'
NODE = 'devbox'
OUTPUT = LOCAL / 'flucon'
RUN_BASE = '/mnt/raid5/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/registration-r1'
FLUCON_BASE = '/mnt/raid5/janghj/ODE-edit/local/baseline-refresh-s2-flucon-20261010'
FILES = dict(  # role: (transfer key, server2 base, relative); copied to inputs/<key>/<relative>
    checkpoint=('cf-sphere-62079-run', RUN_BASE, 'runs/qwen25-cf-sphere/checkpoint/batch-20-2fc52d2e87166ffd.pt'),
    pointer=('cf-sphere-62079-run', RUN_BASE, 'runs/qwen25-cf-sphere/checkpoint/latest.json'),
    terminal=('cf-sphere-62079-run', RUN_BASE, 'runs/qwen25-cf-sphere/terminal.json'),
    config=('cf-sphere-62079-run', RUN_BASE, 'configs/qwen25-cf-sphere.json'),
    stream=('cf-sphere-62079-run', RUN_BASE, 'streams/cf-stream.json'),
    eval_config=('flucon-62873-registration', FLUCON_BASE, 'preparation-r2/configs/qwen25-sphere-62079.json'),
    eval_assets=('flucon-62873-registration', FLUCON_BASE, 'preparation-r2/assets/qwen25.json'),
    eval_lock=('flucon-62873-registration', FLUCON_BASE, 'registration-r1/execution-lock.json'),
    eval_script=('flucon-62873-registration', FLUCON_BASE, 'registration-r1/scripts/qwen25-sphere-62079.sh'),
    eval_submitted=('flucon-62873-registration', FLUCON_BASE, 'registration-r1/submitted-qwen25-sphere-62079.json'))
CANCELLED = dict(job_id='62873', name='s2-flucon-qwen25-sphere-62079', server='server2', original_job_id='62079',
                 source='417a12e7db55af93346be60f79138a6923324de1',
                 config=('e8a192d08be30a27e0f8e53558e311cb180dc1af9e88e90d10a6b4ac3dece021', 4954),
                 config_sha256='deeb0f16d6ff71f0ab07569f265e22146b8bd46243fb1d7df289e17b7ac042d2',
                 lock=('665ddc7b5ef09d11100fd38c4dd1cd292677c7741a6f1c22d8d2507bfb85c65f', 127505),
                 script=('8801a6f88709598a97d6a5c259cc9a249981b4fac8bbc7a4845b0c8605ece711', 857),
                 cancelled='2026-10-11T07:05:24+09:00 audits/global/sphere-stop-20261011/cancellation.json')
CHECKPOINT = ('2fc52d2e87166ffd59ca420c4da4f8e130a384f613614e34d90a00b7962be1ef', 8535462041)
ORDER_SHA = '0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'
REFERENCE = Path('/mnt/raid5/janghj/ODE-edit/local/baseline-generation-eval-assets/20261007/reference-ready-r1/manifest.json')
HUB = Path('/mnt/raid5/janghj/.cache/huggingface/hub')
WANDB_ENV = Path('/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env')
WANDB_KEYS = frozenset(('WANDB_ENTITY', 'WANDB_PROJECT', 'WANDB_MODE', 'WANDB_CONSOLE', 'WANDB_SAVE_CODE',
                        'WANDB_BASE_URL', 'ODEEDIT_WANDB_PYTHON'))
MODEL_CONFIG_SHA = '7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c'
# 62873 config fields that are host bindings; every other field must be equal.
HOST_FIELDS = tuple(re.compile(p) for p in (
    r'registration_authority', r'config_sha256', r'output', r'original\.(checkpoint|pointer|terminal|config)\.path',
    r'assets\.(path|bytes|sha256)', r'stream\.path', r'reference\.(path|bytes|sha256)', r'evaluator\.path',
    r'runtime\.(python|platform|executable)'))
MANIFEST_HOST_FIELDS = tuple(re.compile(p) for p in (
    r'model\.members\[\d+\]\.(real_path|allocated_bytes|device|inode|nlink|mtime_ns|ctime_ns)',))
_tracking = shared.tracking_values


def tracking_values(config, source):
    value = _tracking(config, source)
    value.update(server='server1', task_id=TASK)
    return validate_tracking(value)


@contextmanager
def host_bindings():
    # Process-local dependency injection only: 62873's restore and this host's labels.
    # The shared evaluator source and every other job are never modified. Reset even on exception.
    previous = shared.tracking_values, shared.restore
    shared.tracking_values, shared.restore = tracking_values, server2.restore
    try:
        yield
    finally:
        shared.tracking_values, shared.restore = previous


def run(config, lock):
    c = read(config)
    require(c.get('registration_authority') == AUTHORITY and c.get('asset_schema') == 'history' and c.get('key') == KEY,
            'SPHERE_FLUCON_S1_SCOPE')
    node = os.environ.get('SLURMD_NODENAME')
    require(node is None or node == NODE, 'SPHERE_FLUCON_S1_NODE')
    with host_bindings():
        shared.run(config, lock)


def wandb_keys(path=WANDB_ENV):
    """Only the allowlisted non-secret tracker keys may be present."""
    keys = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            keys.add(line.split('=', 1)[0].strip())
    require(keys <= WANDB_KEYS, 'TRACKER_ENV_KEYS_NOT_ALLOWLISTED')
    return sorted(keys)


def located(rows, inputs, role):
    key, base, relative = FILES[role]
    local = Path(inputs) / key / relative
    received(rows, local, base + '/' + relative)
    return local


def check_payload(path, original):
    """CPU check of what 62873's restore will require (schema, identity, batch, method, FP32 finite weights)."""
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    hp = original['hparams']
    names = {hp['rewrite_module_tmp'].format(layer) + '.weight' for layer in hp['layers']}
    require(payload.get('schema') == 'official-baseline-checkpoint-v1' and payload.get('batch') == 20
            and payload.get('identity') == original['identity'] and payload.get('method') == original['method']
            and set(payload['weights']) == names and set(payload.get('cache_c') or {}) == {str(l) for l in hp['layers']},
            'W20_PAYLOAD_IDENTITY')
    shapes = {}
    for name in sorted(names):
        value = payload['weights'][name]
        require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()), 'W20_FP32_FINITE')
        shapes[name] = list(value.shape)
    return dict(schema=payload['schema'], batch=20, method=payload['method'], selected_weight_shapes=shapes,
                history_layers=sorted(payload['cache_c']), cursor_completed_batch=payload['evaluation_cursor'].get('completed_batch'))


def prepare(output, inputs, receipt):
    output = Path(output).absolute()
    require(output.is_relative_to(OUTPUT) and not output.exists(), 'NEW_OUTPUT_UNDER_TASK')
    rows = transfer_rows(receipt)
    path = {role: located(rows, inputs, role) for role in FILES}

    # 62873 itself: its config, lock, script and submission record, as registered on server2.
    old = read(path['eval_config'])
    require((member(path['eval_config'])['sha256'], member(path['eval_config'])['bytes']) == CANCELLED['config']
            and old['config_sha256'] == CANCELLED['config_sha256']
            == digest({k: v for k, v in old.items() if k != 'config_sha256'}), 'NOT_62873_CONFIG')
    require(old['key'] == KEY and old['registration_authority'] == server2.AUTHORITY
            and old['instruction'] == shared.INSTRUCTION and old['asset_schema'] == 'history', 'NOT_62873_SCOPE')
    server2_config = dict(path=FLUCON_BASE + '/' + FILES['eval_config'][2], bytes=CANCELLED['config'][1],
                          sha256=CANCELLED['config'][0])
    lock = read(path['eval_lock'])
    require((member(path['eval_lock'])['sha256'], member(path['eval_lock'])['bytes']) == CANCELLED['lock']
            and lock['source_commit'] == CANCELLED['source'] and server2_config in lock['configs'], 'NOT_62873_LOCK')
    script = path['eval_script'].read_text()
    require((member(path['eval_script'])['sha256'], member(path['eval_script'])['bytes']) == CANCELLED['script']
            and '-m official.runners.server2.flucon_eval run --config ' + server2_config['path'] in script, 'NOT_62873_SCRIPT')
    submitted = read(path['eval_submitted'])
    require(submitted['job_id'] == CANCELLED['job_id'] and submitted['name'] == CANCELLED['name']
            and submitted['config'] == server2_config and submitted['script']['sha256'] == CANCELLED['script'][0],
            'NOT_62873_SUBMISSION')

    # The 62079 W20 records: same bytes as 62873's binding, then the restore preconditions.
    original = copy.deepcopy(old['original'])
    for role in ('checkpoint', 'pointer', 'terminal', 'config'):
        found = member(path[role])
        require((found['sha256'], found['bytes']) == (original[role]['sha256'], original[role]['bytes']), 'COPY_NOT_62873_' + role.upper())
        original[role] = found
    require((original['checkpoint']['sha256'], original['checkpoint']['bytes']) == CHECKPOINT
            and original['job_id'] == CANCELLED['original_job_id'] and original['method'] == 'SPHERE', 'NOT_62079_W20')
    pointer = read(path['pointer'])
    require(pointer == dict(batch=20, file=path['checkpoint'].name, final_W20=True,
                            identity_sha256=digest(original['identity']), sha256=CHECKPOINT[0]), 'W20_POINTER')
    terminal = read(path['terminal'])
    require(terminal['actual_job_id'] == '62079' and terminal['status'] == 'W20_COMPLETE' and terminal['completed_edits'] == 2000
            and terminal['checkpoint'] == pointer and terminal['checkpoint_identity'] == original['identity'], 'W20_TERMINAL')
    run_config = read(path['config'])
    require(run_config['config_sha256'] == original['identity']['config_sha256'] and run_config['hparams'] == original['hparams'],
            'W20_RUN_CONFIG')
    payload = check_payload(path['checkpoint'], original)

    stream = member(path['stream'])
    require((stream['sha256'], stream['bytes']) == (old['stream']['sha256'], old['stream']['bytes'])
            and stream['sha256'] == original['identity']['stream_sha256'], 'CF_STREAM')
    records = read(path['stream']); records = records['records'] if isinstance(records, dict) else records
    require(len(records) == 2000 and digest([r['case_id'] for r in records]) == original['ordered_case_ids_sha256'] == ORDER_SHA,
            'CF_FIRST2000_ORDER')

    # Host bindings of the same identities.
    manifest = read(path['eval_assets'])
    require(member(path['eval_assets'])['sha256'] == old['assets']['sha256'], 'NOT_62873_ASSETS')
    snapshot = HUB / 'models--Qwen--Qwen2.5-7B-Instruct' / 'snapshots' / original['identity']['model_revision']
    require(str(snapshot) == manifest['model']['snapshot'], 'MODEL_SNAPSHOT_PATH')
    members = []
    for m in manifest['model']['members']:
        local = snapshot / Path(m['path']).name
        item = asset_member(local, allow_symlink=True, expected_sha=m['sha256'], expected_bytes=m['bytes'])
        item['path'] = str(local)
        members.append(item)
    model_config = asset_member(snapshot / 'config.json', allow_symlink=True, expected_sha=MODEL_CONFIG_SHA)
    assets = dict(model=dict(snapshot=str(snapshot), members=members))
    manifest_changes = host_only(field_diff(manifest, assets), MANIFEST_HOST_FIELDS)
    write_new(output / 'assets' / 'qwen25.json', assets)
    refs = load_assets(REFERENCE)
    require(refs.sha == old['reference_identity'], 'REFERENCE_IDENTITY')
    evaluator = member(Path(shared.__file__).resolve().parents[2] / 'evaluation/generation/native_observer.py')
    require(evaluator['sha256'] == old['evaluator']['sha256'], 'EVALUATOR_SOURCE')
    now = runtime()
    require(now['packages'] == old['runtime']['packages'], 'RUNTIME_PACKAGES')
    keys = wandb_keys(WANDB_ENV)
    require(old['tracking_env_file'] == str(WANDB_ENV), 'TRACKER_ENV_PATH')

    c = copy.deepcopy(old)
    c.update(registration_authority=AUTHORITY, original=original, assets=member(output / 'assets' / 'qwen25.json'),
             stream=stream, reference=member(REFERENCE), evaluator=evaluator, runtime=now,
             output=str(output / 'runs' / KEY), tracking_env_file=str(WANDB_ENV))
    c['config_sha256'] = digest({k: v for k, v in c.items() if k != 'config_sha256'})
    changed = host_only(field_diff(old, c), HOST_FIELDS)
    tracking_values(c, 'a' * 40)
    config_path = output / 'configs' / f'{KEY}.json'
    write_new(config_path, c)
    result = dict(instruction=shared.INSTRUCTION, registration_authority=AUTHORITY, server='server1', node=NODE,
        configs=[member(config_path)], reference=member(REFERENCE), transfer_receipt=member(receipt),
        resubmission_of=dict(CANCELLED, config_path=server2_config['path']),
        changed_fields=changed, scientific_fields_changed=[],
        differences=differences(old, c, changed),
        assets_manifest_changed_fields=manifest_changes, model_config_json=model_config,
        checkpoint_payload=payload, tracker_env_keys=keys, no_model_forward=True, no_edit=True, checkpoints_read_only=True)
    write_new(output / 'preparation.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'run'])
    p.add_argument('--config'); p.add_argument('--lock'); p.add_argument('--output'); p.add_argument('--inputs')
    p.add_argument('--receipt'); a = p.parse_args()
    if a.mode == 'run':
        run(a.config, a.lock)
    else:
        prepare(a.output, a.inputs, a.receipt)
