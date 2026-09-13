"""Read-only source/asset binding to the executed singleton BLUE closure."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import types

from .contracts import (Cell, ContractBoundary, MODEL_REVISION, execution_plan,
                        file_sha, member, save)

ABC = Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1')
HUB = Path('/mnt/raid5/janghj/.cache/huggingface/hub')
MODEL = HUB / 'models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots' / MODEL_REVISION
DATA = Path('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')
PROJECTOR = Path('/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt')


def bind_native(source):
    """Path-only namespace shim: no eager unrelated ROME writer initialization."""
    source = Path(source)
    sys.path.insert(0, str(source))
    for name in ('rome', 'util', 'AlphaEdit'):
        if name in sys.modules:
            if list(getattr(sys.modules[name], '__path__', [])) != [str(source / name)]:
                raise ContractBoundary('FOREIGN_NATIVE_NAMESPACE', name=name)
        else:
            mod = types.ModuleType(name)
            mod.__path__ = [str(source / name)]
            mod.__package__ = name
            sys.modules[name] = mod
    previous = os.getcwd()
    try:
        os.chdir(source)
        native = importlib.import_module('AlphaEdit.AlphaEdit_main')
        hp_type = importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
    finally:
        os.chdir(previous)
    return native, hp_type


def resolved_member(path, expected):
    """HF snapshot links are real links, not fabricated regular cache entries."""
    p = Path(path)
    row = member(p.resolve(strict=True), expected=expected)
    row['consumed_path'] = str(p.absolute())
    row['snapshot_symlink'] = p.is_symlink()
    return row


def build_lock(worktree, root, seal_id='cold-l4-r1'):
    """First wave is cold L4 only; warm inputs remain explicitly unverified."""
    from scripts.fixed_counterfact import load_prefix
    worktree, root = Path(worktree).absolute(), Path(root).absolute()
    records = load_prefix(DATA, 10000)
    plan = json.loads((root / 'imports/server4-allowlist-v1.json').read_text())
    def received(suffix):
        candidates = [x for x in plan['members'] if x['source_path'].endswith(suffix)]
        if len(candidates) != 1:
            raise ContractBoundary('AMBIGUOUS_SOURCE_MEMBER', suffix=suffix, count=len(candidates))
        x = candidates[0]
        p = x['local_reuse'] or x['destination']
        member(p, expected=x['sha256'])
        return p
    original_path = received('blue-lifelong-b100x100/attempt-checkpoint-r2/execution.lock.json')
    original = json.loads(Path(original_path).read_text())
    if original['blue_head'] != '311b076a92e4ed0f14f5c8b4909732da781bc5f7' or original['seed'] != 20260907:
        raise ContractBoundary('HISTORICAL_SOURCE_IDENTITY')
    rows = []
    # Only the consumed Alpha/ROME utility closure, never MEMIT writer imports.
    for x in original['members']:
        old = Path(x['path'])
        if x['path'].startswith(original['blue_root'] + '/'):
            rel = old.relative_to(original['blue_root'])
            if rel.parts[0] in {'AlphaEdit', 'rome', 'util'} or str(rel) == 'globals.yml':
                p = ABC / 'imports/blue-source' / rel
            else:
                continue
        elif '/snapshots/' in x['path']:
            p = MODEL / old.name
        elif x['path'].endswith('AlphaEdit-L4_ONLY.json'):
            p = ABC / 'imports/config.json'
        elif old.name == PROJECTOR.name:
            p = PROJECTOR
        elif '/source-tech-r2/' in x['path'] and any(x['path'].endswith(s) for s in (
                'alphaedit_strength_neutral_barrier/contracts.py',
                'alphaedit_strength_neutral_barrier/evaluator.py',
                'ordered_response_barrier_ode/counterfact_locality_evaluator.py')):
            p = worktree / old.relative_to(original['source_root'])
        else:
            continue
        row = resolved_member(p, x['sha256'])
        row.update(historical_path=x['path'], historical_sha256=x['sha256'])
        rows.append(row)
    config = ABC / 'imports/config.json'
    cfg = json.loads(config.read_text())
    if cfg['layers'] != [4] or cfg['L2'] != 1 or not cfg['blue']:
        raise ContractBoundary('SOURCE_SINGLETON_CONFIG')
    if not any(x['consumed_path'] == str(PROJECTOR) for x in rows):
        raise ContractBoundary('PROJECTOR_NOT_IN_ORIGINAL_LOCK')
    historical = ABC / 'imports/historical/blue_alphaedit_sequential_comparison'
    for name in ('evaluation.py', 'integrity.py'):
        # These were selectively preserved from the original helper; match the lock.
        x = next(x for x in original['members'] if x['path'].endswith('/blue_alphaedit_sequential_comparison/' + name))
        rows.append(resolved_member(historical / name, x['sha256']))
    cell = Cell(4, 0)
    companions = {name: received('blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3/B001/' + name)
                  for name in ('entry.json', 'contexts.json', 'native-targets.pt', 'native-observation.json', 'current.json')}
    companion_members = {name: member(path) for name, path in companions.items()}
    payload = dict(instruction_id=execution_plan()['instruction_id'], source_binding='ORIGINAL_SINGLETON_BLUE',
        original_execution_lock=member(original_path), original_blue_head=original['blue_head'],
        original_helper_head=original['source_head'], source_members=rows,
        native_root=str(ABC / 'imports/blue-source'), config=str(config), historical_root=str(historical),
        helper_root=str(worktree / 'project/run_scripts'), snapshot=str(MODEL), projector=str(PROJECTOR),
        dataset_root=str(DATA), model_revision=MODEL_REVISION, seed=20260907,
        torch='2.9.1+cu128', transformers='4.44.2', attention='eager',
        tf32_matmul=False, tf32_cudnn=True, model_dtype='torch.float32',
        cell=cell.plan(), records=[dict(case_id=r['case_id']) for r in records[:100]],
        companions=companions, companion_members=companion_members,
        original_W0_RNG_snapshot='NOT_RECORDED_SOURCE_INIT_REPRODUCTION',
        checkpoint_reference=None, checkpoint_reference_status='AWAITING_S2_OWNER_ALLOWLIST',
        target_policy='ORIGINAL_COMPUTE_Z_RECOMPUTE_NO_CACHE_SUBSTITUTION',
        native_C0_role='NOT_CONSUMED_BY_BLUE_P_PROVIDED_WRITER_DIAGNOSTIC_SEPARATE',
        diagnostics=dict(fd_alpha=2.**-8, fd_relative_tolerance=.05,
            fd_absolute_tolerance='8*FP32_eps*max(1,abs(margin))/alpha + repeated_forward_range/alpha',
            fd_unresolved='LIMIT_DERIVATIVE_CLAIM_NOT_EXCLUDE_NATIVE',
            initial_probe='CURRENT_FIRST_CANONICAL_REQUEST_REWRITE',
            general_panel='PENDING_SOURCE_SEAL_NO_GENERAL_MEASUREMENT_IN_FIRST_GATE',
            spectrum='NOT_MEASURED_YET_NO_IMPLICIT_RANK_CAP'),
        first_wave=dict(native_batches=1, next_submission_after_user_recall=True),
        after_initial_valid='MONITORING_PAUSED_AWAITING_USER', scientific_promotion=False)
    if not seal_id.replace('-', '').isalnum():
        raise ContractBoundary('INVALID_SEAL_ID')
    return save(root / 'locks' / seal_id / 'input.lock.json', payload)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--worktree', required=True); p.add_argument('--root', required=True)
    p.add_argument('--seal-id', default='cold-l4-r1')
    a = p.parse_args(); print(json.dumps(build_lock(a.worktree, a.root, a.seal_id)))
