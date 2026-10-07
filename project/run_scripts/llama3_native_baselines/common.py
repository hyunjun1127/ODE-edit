"""Exact task boundaries and bounded CPU metadata primitives."""
from pathlib import Path

from project.run_scripts.jlz_realization.common import digest, member, require, sha, write
from project.run_scripts.jlz_shared_budget.common import verify

ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = Path('/data/janghj/ODE-edit')
LOCAL = APP_ROOT / 'local/llama3-baselines-fluency-consistency-2k'
TASK = 'llama3-baselines-fluency-consistency-2k'
NONCE = 'USER-GH-ALL-SH-BASELINE-FLUENCY-CONSISTENCY-RERUN-20261007-R1'
SESSION = '01a04939-b5c7-7a03-ba2d-ef3343d62cfd'
METHODS = ('MEMIT', 'PRUNE', 'RECT', 'ALPHAEDIT', 'ALPHAEDIT_BLUE', 'CAKE')
LAYERS = (4, 5, 6, 7, 8)
MILESTONES = (5, 10, 15, 20)
MODEL_REVISION = '8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
ORDERED_SHA = '0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4'
PROFILE = 'cf-cake-prompt-inclusive-total100-eos-corrected-v1'
ENVELOPE = 'messages/head/2026-10-07-baseline-fluency-consistency-rerun-server4.json'
AUTHORITY_FILES = {
    ENVELOPE: '47683aff67443a5ccaaf00d2cd15857b4fd1c6918a0747e73ea5a7a292dc454e',
    'plans/global/2026-10-07-baseline-fluency-consistency-rerun/contract.json':
        '663c3c596613d413fbfb3d409d7eb24faaaf97e4ebd68b57b046f94f7947f357',
    'control/generation-metric-policy.json':
        'a0f7b8ac864f9d5f0cfa41a0d76d6441583972607c8b9327c893dd827be66b62',
}


def authority():
    for relative, expected in AUTHORITY_FILES.items():
        require(sha(ROOT / relative) == expected, 'AUTHORITY_BYTES:' + relative)
    return {relative: member(ROOT / relative) for relative in AUTHORITY_FILES}


def batches(records):
    require(len(records) == 2000, 'EXACT_FIRST2000')
    require(digest([r['case_id'] for r in records]) == ORDERED_SHA, 'ORDERED_OCCURRENCES')
    for number in range(1, 21):
        yield number, records[(number - 1) * 100:number * 100], records[:number * 100]


def stat_seal(row):
    """Prior full SHA plus unchanged local metadata; never rehash huge assets."""
    p = Path(row['path'])
    s = p.stat()
    require((s.st_size, s.st_ino, s.st_mtime_ns) ==
            (row['bytes'], row['inode'], row['mtime_ns']), 'ASSET_STAT_CHANGED:' + str(p))
    require(isinstance(row.get('sha256'), str) and len(row['sha256']) == 64,
            'ASSET_PRIOR_SHA_MISSING')
    return row
