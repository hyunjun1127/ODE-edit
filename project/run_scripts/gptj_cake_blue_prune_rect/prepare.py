"""CPU metadata/source preparation; no model/C0/P load, generation or cloning.

Large payloads retain the earlier SHA/schema/finite receipts and are sealed by
size/inode/mtime only. Small task-owned native closures and JSON parsers are
fully verified. Actual cold model/operator/observer checks belong to B1/B2.
"""
import argparse
import csv
import shutil
from dataclasses import asdict
from pathlib import Path

from .common import (
    ARMS, ARM_LAYERS, CONTRACT, LAYERS, LOCAL, NONCE, ORDERED_SHA, PYTHON,
    ROOT, TASK, authority, batches, digest, read, require, sha, small_member,
    stat_seal, write,
)

PARENT = Path('/mnt/raid5/janghj/ODE-edit/local/gptj-easyedit-native-baselines-2k/preparation-r1')
PARENT_CONFIG_SHA = 'a9ff4078e0839adcee07ce3ece0c26ed1d6f0d9ff71fadfd69dc203f1866decf'
PARENT_ASSET_CHECKS_SHA = 'a13853fe679b8bfad4ea4d461a1d4face4656e2318c2349e0b93ff38d083ddbc'


def validate_reused_assets(old, checks, contract):
    """Identity and prior receipts only: intentionally no numerical payload I/O."""
    require(old['model'] == contract['assets']['model_snapshot']
            and old['model_revision'] == contract['scope']['revision']
            and old['seed'] == contract['scope']['seed']
            and old['ordered_ids_sha256'] == ORDERED_SHA,
            'REUSED_MODEL_INPUT_IDENTITY')
    require(old['runtime']['python'] == PYTHON
            and old['runtime']['torch'] == '2.9.1+cu128'
            and old['runtime']['transformers'] == '4.57.1', 'REUSED_RUNTIME_IDENTITY')
    rows = old['assets'] + old['runtime']['members'] + [old['observer_identity']]
    for row in rows:
        stat_seal(row)
    expected_stats = {
        str(layer): str(Path(contract['assets']['stats_root']) /
                       f'transformer.h.{layer}.mlp.fc_out_float32_mom2_100000.npz')
        for layer in LAYERS
    }
    assets_by_path = {row['path']: row for row in old['assets']}
    require(len(assets_by_path) == len(old['assets']), 'REUSED_ASSET_DUPLICATE')
    require(all(path in assets_by_path for path in expected_stats.values()),
            'REUSED_C0_SIX_PHYSICAL_LAYERS')
    require(Path(old['native']['stats_dir']) / 'gpt-j-6b/wikipedia_stats'
            == Path(contract['assets']['stats_root'])
            and old['native']['logical_model_name'] == 'gpt-j-6b',
            'REUSED_NATIVE_LOGICAL_STATS_PATH')
    projector = old['native']['projector']
    require(projector['path'] == contract['assets']['projector']
            and assets_by_path.get(projector['path']) == projector,
            'REUSED_SIX_LAYER_PROJECTOR')
    weights = [row for row in old['model_assets']
               if row.get('snapshot_path', '').endswith('/pytorch_model.bin')]
    require(len(weights) == 1
            and weights[0]['sha256'] == contract['assets']['model_payload_SHA'],
            'REUSED_MODEL_PAYLOAD_SHA')
    require(all(assets_by_path.get(row['path']) == row for row in old['model_assets']),
            'REUSED_MODEL_ASSET_MEMBERS')
    require(set(old['cold_W']) == set(map(str, LAYERS))
            and all(type(value) is str and len(value) == 64 for value in old['cold_W'].values()),
            'REUSED_COLD_SIX_WEIGHTS')
    expected_slots = [dict(slot=index, physical_layer=layer, shape=[16384, 16384])
                      for index, layer in enumerate(LAYERS)]
    require(checks['projector_slots'] == expected_slots
            and checks['finite'] is True and checks['P_spectral_recompute'] is False,
            'REUSED_PROJECTOR_SCHEMA_FINITE_RECEIPT')
    require(checks['stats'] == [dict(layer=layer, shape=[16384, 16384], dtype='float32',
            count=54924275.0, native_moment='stored mom2 / count; no transformed cache saved')
            for layer in LAYERS], 'REUSED_RAW_MOM2_SCHEMA_COUNT')
    require(old['stream'] == contract['input']['dataset']
            and assets_by_path[old['stream']]['sha256'] == contract['input']['dataset_sha256'],
            'REUSED_DATASET_IDENTITY')
    return {layer: assets_by_path[path] for layer, path in expected_stats.items()}


def load_input_packs(old, contract, normalizers=()):
    """Read the existing corpus once; validate only frozen first2k occurrences."""
    rows = read(old['stream'])[:2000]
    chunks = list(batches(rows))
    require(len({row['case_id'] for row in rows}) == 2000,
            'EXISTING_EVALUATOR_OCCURRENCE_UNIQUENESS')
    schedule_path = ROOT / contract['input']['schedule']
    require(small_member(schedule_path)['sha256'] == contract['input']['schedule_sha256'],
            'SCHEDULE_EXACT_SHA')
    with schedule_path.open(newline='') as stream:
        scheduled = list(csv.DictReader(stream))
    require(len(scheduled) == 2000
            and [int(row['case_id']) for row in scheduled] == [row['case_id'] for row in rows]
            and [int(row['stream_index0']) for row in scheduled] == list(range(2000))
            and [int(row['batch1']) for row in scheduled] == [index // 100 + 1 for index in range(2000)]
            and [int(row['slot0']) for row in scheduled] == [index % 100 for index in range(2000)],
            'SCHEDULE_PHYSICAL_OCCURRENCES')
    require(len(old['packs']) == 20 and [pack['ids'] for pack in old['packs']]
            == [[row['case_id'] for row in current] for _, current, _ in chunks],
            'REUSED_EXACT_TWENTY_PACKS')
    packs = []
    for batch, current, _ in chunks:
        require(all(len(row['paraphrase_prompts']) == 2
                    and len(row['neighborhood_prompts']) == 10 for row in current),
                'INPUT_R100_P200_N1000')
        identities = [digest(normalize(current)) for normalize in normalizers]
        require(not identities or len(set(identities)) == 1, 'FOUR_ARM_REQUEST_SCHEMA_PARITY')
        packs.append(dict(batch=batch, ids=[row['case_id'] for row in current],
                          identity=identities[0] if identities else None,
                          counts=dict(requests=100, R=100, P=200, N=1000)))
    return rows, packs, small_member(schedule_path)


def evaluator_sources(old, w0):
    """Small source-only identity comparison, not evaluator execution."""
    members = []
    for relative in old['evaluator_sources']:
        current = small_member(ROOT / relative)
        frozen = small_member(w0 / 'source' / relative)
        require(current['sha256'] == frozen['sha256'], 'W0_SCORER_SOURCE:' + relative)
        members.append(dict(relative=relative, **current))
    return members


def bind_W0_reference(old):
    """Read-only complete reference identity; raw rows are validated in the runner."""
    source = Path(old['W0_reference'])
    require((source / 'result.json').is_file() and (source / 'terminal.json').is_file(),
            'W0_SCALAR_REFERENCE_NOT_READY')
    original = read(source / 'config.json')
    lock, terminal = read(source / 'execution.lock.json'), read(source / 'terminal.json')
    result = read(source / 'result.json')
    config_member, result_member = small_member(source / 'config.json'), small_member(source / 'result.json')
    require(config_member['sha256'] == lock['config_sha256']
            and result_member['sha256'] == terminal['result_sha256']
            and terminal['status'] == result['status'] == 'COMPLETED', 'W0_REFERENCE_SEAL')
    require(original['revision'] == old['model_revision'] and original['model'] == old['model']
            and original['seed'] == old['seed'] and original['assets'] == old['model_assets']
            and original['runtime'] == old['runtime'] and original['microbatch'] == 2,
            'W0_REFERENCE_MODEL_RUNTIME')
    require(original['ordered_ids'] == ORDERED_SHA
            and original['observer_manifest'] == old['observer_identity']
            and original['input_source']['path'] == old['stream'], 'W0_REFERENCE_INPUT_OBSERVER')
    stat_seal(original['input_source'])
    require(result['new_actual_evaluation'] and result['no_mutation']
            and result['edit_calls'] == result['fits'] == result['solves'] == result['history_appends'] == 0
            and result['requests'] == 2000 and result['prompt_pairs'] == 26000
            and result['candidates'] == 52000 and not result['checkpoint_saved'],
            'W0_REFERENCE_COLD_ONLY')
    raw = result['raw']
    require(len(raw) == 40 and len({row['path'] for row in raw}) == 40,
            'W0_REFERENCE_FORTY_RAW_CHUNKS')
    for row in raw:
        require(Path(row['path']).parent == source / 'raw', 'W0_REFERENCE_RAW_PATH')
        stat_seal(row)
    sources = evaluator_sources(old, source)
    return dict(status='COMPLETE_REFERENCE_IDENTITY_NOT_NEW_SCIENCE',
                source=str(source), config=config_member, result=result_member,
                terminal=small_member(source / 'terminal.json'),
                chunks=raw, requests=2000, row_count=26000,
                scalar_bridge_only=True, history_or_editor_resume=False,
                raw_validation='PRIOR_SHA_PLUS_CURRENT_STAT; RUNNER_VALIDATES_ROWS_TOKENS_ONCE',
                new_model_evaluations=0), sources


def prepare(out=None, attempt=None):
    contract = authority()
    out = (LOCAL / 'preparation-r1' if out is None else Path(out)).absolute()
    attempt = (LOCAL / 'attempt-r1' if attempt is None else Path(attempt)).absolute()
    require(out == LOCAL / 'preparation-r1' and attempt.parent == LOCAL,
            'TASK_LOCAL_PREPARATION_ATTEMPT_SCOPE')
    require(not out.exists() and not attempt.exists() and not out.is_symlink()
            and not LOCAL.is_symlink(),
            'CREATE_ONCE_PREPARATION')
    require(not list(LOCAL.glob('*/submission.json')), 'NONCE_NOT_REGISTERED')
    require(shutil.disk_usage(LOCAL.parent).free >= 8 * 1024 ** 3, 'STORAGE_RESERVE')
    prior_config = small_member(PARENT / 'config-submission.json')
    prior_checks = small_member(PARENT / 'asset-checks.json')
    require(prior_config['sha256'] == PARENT_CONFIG_SHA
            and prior_checks['sha256'] == PARENT_ASSET_CHECKS_SHA, 'PRIOR_PREPARATION_EXACT_SHA')
    old, checks = read(PARENT / 'config-submission.json'), read(PARENT / 'asset-checks.json')
    stats = validate_reused_assets(old, checks, contract)
    w0_binding, scorer_members = bind_W0_reference(old)
    # Delayed adapter imports are parser/source-only; the adapters never load assets here.
    from . import native_cake_blue, native_prune_rect
    modules = (native_cake_blue, native_prune_rect)
    records, packs, schedule = load_input_packs(old, contract,
        normalizers=tuple(module.native_requests for module in modules))
    native_bundle, parsed, imported = {}, {}, {}
    for module, family in zip(modules, ('cake-blue', 'prune-rect')):
        adopted = module.adopt_native(out / 'native-source' / family)
        for arm, binding in adopted.items():
            require(arm not in native_bundle and arm in ARMS, 'NATIVE_BUNDLE_ARM_DUPLICATE')
            bundle = module.load_native(binding, arm)
            native_bundle[arm] = binding
            parsed[arm] = asdict(module.parse_hparams(bundle))
            imported[arm] = module.closure(bundle)
    require(set(native_bundle) == set(ARMS), 'NATIVE_FOUR_ARM_CLOSURE')
    arm_slots = {spec['arm']: list(spec.get('P_slots', [])) for spec in contract['native']}
    config = dict(schema=1, instruction_id=NONCE, task_id=TASK, attempt=str(attempt),
        seed=old['seed'], model=old['model'], model_revision=old['model_revision'],
        model_assets=old['model_assets'], assets=old['assets'], runtime=old['runtime'],
        stream=old['stream'], ordered_ids_sha256=ORDERED_SHA, cold_W=old['cold_W'],
        observer_identity=old['observer_identity'], evaluator_sources=old['evaluator_sources'],
        evaluator_source_members=scorer_members, packs=packs,
        arm_layers={arm: list(layers) for arm, layers in ARM_LAYERS.items()},
        native_bundle=native_bundle,
        native=dict(projector=old['native']['projector'], stats=stats,
            stats_dir=old['native']['stats_dir'], logical_model_name='gpt-j-6b', device=0,
            effective_hparams=parsed, closure=imported, context_reuse=None,
            P_shape=[6, 16384, 16384], P_slot_layers=list(LAYERS), arm_P_slots=arm_slots),
        native_repair_labels={'PRUNE': 'PRUNE_TERMINAL_BASE_FIX'},
        native_provenance=dict(prior_config=prior_config, prior_asset_checks=prior_checks,
            large_payload_validation='PRIOR_FULL_SHA_SCHEMA_FINITE_PLUS_UNCHANGED_STAT',
            original_source_modified=False, shared_environment_modified=False),
        W0_reference=old['W0_reference'], W0_reuse=w0_binding,
        context_policy='OWN_NATIVE_COLD_GENERATOR_ONCE; no EasyEdit/GPT2 context transplant',
        authority=small_member(ROOT / CONTRACT), contract=small_member(ROOT / CONTRACT),
        tracking=old['tracking'], schedule=schedule,
        input_source=next(row for row in old['assets'] if row['path'] == old['stream']),
        resources=dict(cpu=6, gpu=1, host_mib=59392, wall='2-00:00:00',
            collector_cpu=6, collector_host_mib=24576, collector_wall='04:00:00',
            project_cap=2, task_cap=2, reserve_bytes=8 * 1024 ** 3, wall_is_eta=False,
            fresh_admission_required=True,
            host_budget='FP32 model + native per-layer C0/P + independent H/rollback RAM; actual peak measured',
            gpu_budget='FP32 native model/fit/solve/cast order; no scientific dtype/layer reduction'),
        noCP=True, z_disk_cache=False, exact_resume='NOT_AVAILABLE',
        broadcast='NO_BROADCAST_NOT_REQUIRED; same-host protected assets/raw KEEP')
    write(out / 'config.json', config)
    write(out / 'preparation.json', dict(status='CPU_METADATA_BOUND_NOT_GPU_PASS',
        config=small_member(out / 'config.json'), prior_config=prior_config, prior_asset_checks=prior_checks,
        input_requests=len(records), packs=20, R=2000, P=4000, N=20000,
        physical_layers=list(LAYERS), arm_layers=config['arm_layers'],
        projector_slots=checks['projector_slots'], arm_P_slots=arm_slots,
        numerical_finite_checks='REUSED_PRIOR_RECEIPT_NO_RESCAN',
        model_loads=0, tensor_loads=0, GPU=0, C0_P_recomputed=False,
        large_payload_SHA_passes=0, large_payload_copies=0,
        native_imports=imported, shared_environment_modified=False,
        W0_reference_status=w0_binding['status'], new_W0_evaluations=0,
        context='OWN_NATIVE_NOT_YET_GENERATED'))
    return out / 'config.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--attempt', type=Path)
    args = parser.parse_args()
    print(prepare(args.out, args.attempt))


if __name__ == '__main__':
    main()
