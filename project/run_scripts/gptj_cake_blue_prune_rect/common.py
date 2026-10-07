"""Task-only authority and immutable, scalar-only evidence helpers."""
import json
from pathlib import Path

from project.run_scripts.jlz_realization.common import (
    digest, member, require, sha, tensor_sha, write,
)
from project.run_scripts.jlz_shared_budget.common import verify

ROOT = Path(__file__).resolve().parents[3]
TASK = 'gptj-cake-blue-prune-rect-2k'
NONCE = 'USER-GH-SH2-GPTJ-CAKE-ALPHAEDIT-BLUE-PRUNE-RECT-2K-20261007-R1'
CONTRACT = 'plans/global/2026-10-07-gptj-cake-blue-prune-rect-2k/contract.json'
CONTRACT_SHA = '6fb4690971e92f5a90b3175ff8c4e95608a7b0596671381f34b230977696b395'
SESSION = '01a0493a-074c-7f91-9a13-769116326fef'
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local/gptj-cake-blue-prune-rect-2k')
ARMS = ('CAKE', 'ALPHAEDIT_BLUE', 'PRUNE', 'RECT')
LAYERS = (3, 4, 5, 6, 7, 8)
ARM_LAYERS = {arm: (3, 8) if arm == 'ALPHAEDIT_BLUE' else LAYERS for arm in ARMS}
MILESTONES = (5, 10, 15, 20)
ORDERED_SHA = '0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'
PYTHON = '/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
SOURCE_ENV = 'GPTJ_CAKE_BLUE_PRUNE_RECT_SOURCE_COMMIT'


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'UNSAFE_JSON_MEMBER:' + str(path))
    return json.loads(path.read_text())


def authority():
    require(sha(ROOT / CONTRACT) == CONTRACT_SHA, 'FOUR_ARM_AUTHORITY_BYTES')
    contract = read(ROOT / CONTRACT)
    scope, owner = contract['scope'], contract['owner']
    require(contract['instruction_id'] == NONCE and contract['task_id'] == TASK
            and owner['server'] == 'server2' and owner['session'] == SESSION,
            'FOUR_ARM_OWNER')
    require(scope['arms'] == list(ARMS) and scope['batch_size'] == 100
            and scope['batches'] == 20 and scope['edits_per_arm'] == 2000
            and scope['seed'] == 20261002 and scope['no_batch21']
            and scope['cold_independent'] and scope['fresh_history_when_native'],
            'FOUR_ARM_SCOPE')
    require(contract['assets']['P_slot_layers'] == list(LAYERS)
            and contract['assets']['P_shape'] == [6, 16384, 16384]
            and contract['input']['ordered_case_ID_sha256'] == ORDERED_SHA,
            'FOUR_ARM_PHYSICAL_ASSET_INPUT')
    return contract


def batches(records):
    require(len(records) == 2000
            and digest([record['case_id'] for record in records]) == ORDERED_SHA,
            'EXACT_FIRST2K')
    for index in range(20):
        yield index + 1, records[index * 100:(index + 1) * 100], records[:(index + 1) * 100]


def stat_seal(row):
    """Reuse prior full SHA evidence without opening a large payload again."""
    path = Path(row['path'])
    require(path.is_file() and not path.is_symlink(), 'ASSET_SAFE_FILE:' + str(path))
    stat = path.stat()
    require((stat.st_size, stat.st_ino, stat.st_mtime_ns)
            == (row['bytes'], row['inode'], row['mtime_ns']),
            'ASSET_STAT_CHANGED:' + str(path))
    require(type(row['sha256']) is str and len(row['sha256']) == 64,
            'ASSET_PRIOR_SHA_MISSING:' + str(path))
    return row


def small_member(path):
    """Explicit small-source/config full SHA; never hash a tensor or raw corpus."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size <= 4 * 1024 ** 2,
            'PREPARATION_SMALL_SOURCE_ONLY:' + str(path))
    return member(path)
