"""Cold GPT2-XL W0-only identities and create-once scalar evidence."""
import json
import math
import os
from pathlib import Path

from project.run_scripts.jlz_realization.common import digest, member, require, sha, tensor_sha

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local/base-model-gpt2xl-w0/20261007-v1')
TASK = 'base-model-gpt2xl-w0'
NONCE = 'USER-GH-SH1-SH2-BASEMODEL-W0-20261007-R1-SERVER1'
SESSION = '01a04939-f93a-7b50-bca0-65438eab2062'
ENVELOPE = 'messages/head/2026-10-07-base-model-w0-server1.json'
ENVELOPE_SHA = '8694d6b4492b5096ec1083f0814d5e6965dabb7e85c0524c8ab45e9f47344a94'
OLD_CONFIG = Path('/mnt/raid5/janghj/ODE-edit/local/jlz-price-gpt2xl-2k/attempt-checkpoint-repair-r1/config.json')
PYTHON = '/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
MODEL_REVISION = '15ea56dee5df4983c59b2538573817e1667135e2'
PAYLOAD_SHA = '0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a'
ORDER_SHA = '0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'
LAYERS = (13, 14, 15, 16, 17)
DENOMINATORS = {'R': 2000, 'P': 4000, 'N': 20000}
SCHEMA = 'price-first2k-scalar-v1'
RESERVE_BYTES = 2 * 1024**3


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'UNSAFE_READ:' + str(path))
    return json.loads(path.read_text())


def verify(row, *, metadata_only=False):
    path = Path(row['path'])
    require(path.is_file() and not path.is_symlink(), 'UNSAFE_BOUND_FILE:' + str(path))
    stat = path.stat()
    require(stat.st_size == row['bytes'], 'BOUND_SIZE:' + str(path))
    if metadata_only:
        require((stat.st_ino, stat.st_mtime_ns) == (row['inode'], row['mtime_ns']), 'BOUND_ASSET_STAT')
    else:
        require(sha(path) == row['sha256'], 'BOUND_SHA:' + str(path))
    return path


def write_bytes(path, data):
    """Atomic create-once; never replace an old result or scientific state."""
    path = Path(path)
    require(isinstance(data, bytes) and 0 < len(data) <= 32 * 1024**2, 'SCALAR_FILE_BOUND')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(not path.is_symlink() and path.read_bytes() == data, 'CREATE_ONCE_CONFLICT')
        return
    temp = path.with_name(path.name + '.tmp')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temp, path)
    temp.unlink()


def write(path, value):
    write_bytes(path, (json.dumps(value, sort_keys=True, ensure_ascii=False,
                                separators=(',', ':'), allow_nan=False) + '\n').encode())


def capacity(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    value = os.statvfs(path)
    free = value.f_bavail * value.f_frsize
    require(free >= RESERVE_BYTES and value.f_favail >= 256, 'W0_STORAGE_CAPACITY')
    return dict(free_bytes=free, free_inodes=value.f_favail, reserve_bytes=RESERVE_BYTES,
                exclusive_reservation=False)


def pair_specs(records, bench):
    """Canonical request/panel/new-then-true order, without model access."""
    specs, pairs = [], []
    for record in records:
        rewrite = record['requested_rewrite']
        panels = bench.panels(record)
        require({kind: len(prompts) for kind, prompts in panels.items()} == {'R': 1, 'P': 2, 'N': 10},
                'W0_PANEL_INVENTORY')
        for kind, prompts in panels.items():
            for index, prompt in enumerate(prompts):
                specs.append(dict(case_id=record['case_id'], kind=kind, prompt_index=index,
                    identity=digest([record['case_id'], kind, index, prompt,
                                     rewrite['target_new']['str'], rewrite['target_true']['str']]),
                    endpoint='W0'))
                pairs.extend([(prompt, rewrite['target_new']['str']),
                              (prompt, rewrite['target_true']['str'])])
    return specs, pairs


def token_identity(records, bench):
    specs, pairs = pair_specs(records, bench)
    for index, spec in enumerate(specs):
        for label, pair in zip(('new', 'true'), pairs[2*index:2*index+2]):
            prompt, target = bench.evaluation_ids(*pair)
            require(len(prompt) + len(target) - 1 <= 1024, 'W0_NO_TRUNCATION')
            spec[label + '_token_identity'] = digest([prompt, target])
    return [{key: value for key, value in row.items() if key != 'endpoint'} for row in specs]


def verify_config_lock(config_path, lock_path):
    config, lock = read(config_path), read(lock_path)
    require(config['instruction_id'] == lock['instruction_id'] == NONCE
            and config['task_id'] == TASK and sha(config_path) == lock['config_sha256'], 'W0_CONFIG_LOCK')
    require(lock['source_commit'] == os.environ.get('ODEEDIT_W0_SOURCE_COMMIT', lock['source_commit']),
            'W0_EXECUTION_SOURCE')
    for row in lock['source_members']:
        verify(row)
    verify(lock['archive'])
    if 'tracking_env' in lock:verify(lock['tracking_env'])
    for row in config['runtime']['source_members'] + config['input_members']:
        verify(row)
    for row in config['model_assets']:
        verify(row, metadata_only=row['path'].endswith('model.safetensors'))
    require(config['save_checkpoints'] is False and config['edit_calls'] == config['solves']
            == config['history_appends'] == 0 and config['observer_microbatch'] == 2, 'W0_ONLY_BOUNDARY')
    return config, lock


class W0View:
    """Evaluator-only native GPT2 view: no writer, C0/P/H, optimizer or hooks."""
    def __init__(self, model):
        import torch
        cfg = model.config
        require((cfg.model_type, cfg.n_layer, cfg.n_embd, cfg.n_positions, cfg.vocab_size)
                == ('gpt2', 48, 1600, 1024, 50257), 'W0_GPT2_ARCHITECTURE')
        require(cfg._attn_implementation == 'eager', 'W0_EAGER')
        require(model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr(), 'W0_TIED_HEAD')
        self.model = model
        self.device = next(model.parameters()).device
        self.weights = {layer: model.transformer.h[layer].mlp.c_proj.weight for layer in LAYERS}
        require(all(tuple(weight.shape) == (6400, 1600) and weight.dtype == torch.float32
                    for weight in self.weights.values()), 'W0_CONV1D_LAYOUT')
        require(all(weight.dtype == torch.float32 and not weight.requires_grad
                    for weight in model.parameters()), 'W0_FROZEN_FP32')
        self.calls = 0
        self.physical_candidate_rows = 0

    def observer_hidden(self, **tokens):
        self.calls += 1
        self.physical_candidate_rows += int(tokens['input_ids'].shape[0])
        # Exact existing GPT2 evaluator: native learned-position default arange,
        # not attention-mask cumsum; transformer applies final LN exactly once.
        return self.model.transformer(**tokens, use_cache=False).last_hidden_state

    def guard(self):
        return {name: (weight.data_ptr(), weight._version, tuple(weight.shape), str(weight.dtype))
                for name, weight in self.model.named_parameters(remove_duplicate=False)}

    def hooks(self):
        return {name: tuple(tuple(getattr(module, attr).keys()) for attr in
                ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks'))
                for name, module in self.model.named_modules()}

    def selected_state(self):
        return {str(layer): tensor_sha(weight) for layer, weight in self.weights.items()}


def check_guard(before, after):
    require(before == after, 'W0_PARAMETER_POINTER_VERSION_MUTATION')


def check_finite_rows(rows):
    require(all(math.isfinite(row[label + '_nll']) for row in rows for label in ('new', 'true')),
            'W0_NONFINITE_NLL')
