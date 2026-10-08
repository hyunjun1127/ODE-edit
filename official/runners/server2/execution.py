"""Seal an actual reviewed main source and Server2 execution inputs; no GPU."""
import argparse
import importlib
from pathlib import Path

from official.baselines import registry
from official.experiments.prepare import METHODS, digest, file_sha, read, write_new
from official.runners.server2 import assets
from official.runners.server2.run import checkpoint_identity, configuration, tracking_config
from official.runners.server2 import oracle, parity
from official.tracking import schema as tracking_schema
from official.runners.server2.submit import REPO, OUTPUT, SESSION, INSTRUCTION, sealed_source


def memory_disk_plan():
    # Native selected FP32 matrices. FT's fc_out bias is included. This is
    # payload arithmetic, not a measured host/GPU peak or an ETA.
    weight = 4096 * 16384 * 4
    history = 16384 * 16384 * 4
    cells = dict(FT=weight+4096*4, MEMIT=6*weight,
        ALPHAEDIT=6*(weight+history), ALPHAEDIT_BLUE=2*(weight+history),
        MEMIT_FE=6*weight, SPHERE=6*(weight+history))
    latest_six = sum(cells.values())
    return dict(per_method_payload_bytes=cells,
        twelve_final_checkpoints_bytes=2*latest_six,
        qualification_latest_checkpoints_bytes=latest_six,
        two_lane_atomic_extra_bytes=2*max(cells.values()),
        raw_error_metadata_reserve_bytes=16*1024**3,
        planned_reserve_bytes=128*1024**3,
        latest1_reclamation_only_this_new_task_CP_folders=True,
        old_asset_delete_authorized=False, checkpoint_exception=INSTRUCTION,
        measured_peak_RAM_VRAM=False, requested_wall_is_not_ETA=True,
        memory_components=['FP32 GPTJ GPU model', 'native layer solve/activations',
            '6 FP32 C0 CPU planes when required', '6-plane P readonly mmap when required',
            'native independent H6/2 CPU planes', 'checkpoint serialization CPU copy',
            'qualification continuous W/H snapshots and durable B2 restore',
            'CF reference scoring assets at W0/W20 only'])


def prepare(asset_manifest, out, source, official_tree, tracking_binding, local_caps,
            qualification_producer_attempt=None):
    source_binding = sealed_source(source, official_tree)
    assets.verify(asset_manifest)
    base = read(asset_manifest)
    module = importlib.import_module('official.evaluation.factual')
    require_api = ('evaluate', 'build_zsre_w0_reference')
    if not all(callable(getattr(module, key, None)) for key in require_api):
        raise ValueError('OFFICIAL_FACTUAL_API_NOT_READY')
    tracking = read(tracking_binding)
    if tracking['namespace'] != 'official.tracking' or tracking['metric_schema'] != tracking_schema.OFFICIAL_SCHEMA:
        raise ValueError('OFFICIAL_COMMON_TRACKING_NAMESPACE_REQUIRED')
    transport = importlib.import_module(tracking['namespace'])
    if not callable(getattr(transport, 'init', None)) or file_sha(transport.__file__) != tracking['source_sha256']:
        raise ValueError('OFFICIAL_SHARED_TRACKING_API_SOURCE_NOT_BOUND')
    root = REPO/'official'
    files = {str(path.relative_to(root)):file_sha(path) for path in sorted(root.rglob('*'))
             if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    # A sealed-source gate has already rejected uncommitted official bytes.
    value = dict(base, owner=dict(server='server2', session=SESSION), instruction_id=INSTRUCTION,
        asset_manifest=str(Path(asset_manifest).resolve()), asset_manifest_sha256=file_sha(asset_manifest),
        assets_identity_sha256=base['assets_sha256'], code_commit=source, official_tree_sha256=official_tree,
        source_binding=source_binding, source_members=files, tracking=tracking,
        local_caps_file=str(Path(local_caps).resolve()),
        resources=dict(gpu=1,cpu=6,host_mib=59392,wall='2-00:00:00',
            collector_cpu=6,collector_host_mib=24576,collector_wall='04:00:00',
            partition='gpu',qos='lab_gpu_s2',node='server2',reserve_bytes=128*1024**3),
        checkpoint_plan=memory_disk_plan(),
        qualification_plan=dict(methods=list(METHODS), dataset='cf', continuous_batches=[1,2,3],
            resume_from_actual_durable_batch=2,resumed_batches=[3], logical_batch=100,
            native_call_count_per_method=4, native_request_applications_per_method=400,
            comparison='EXACT_WEIGHTS_HISTORY_CONTEXT_RNG_CASE_METRICS',
            native_formula_parity='PASSIVE_FIRST_EXISTING_B1_FACTUAL_FORWARD; EXTRA_LM_FORWARD0',
            original_native_evaluator_parity='SEPARATE_REQUIRED_INPUT_NOT_REPLACED_BY_RESUME',
            quality_not_gate=True, actual_GPU_PASS=False, typed_failure_no_retry=True),
        actual_GPU_qualification='NOT_OBSERVED', ready_to_submit=True,
        native_source_sha256={method:[file_sha(registry.implementation(method,'gptj')[0].__file__)]
                             for method in METHODS})
    value['native_parity_plans'] = {dataset:{method:parity.plan(value,dataset,method,
        read(value['streams'][dataset]['path'])) for method in METHODS} for dataset in ('cf','zsre')}
    value['cf_native_reference_plan'] = oracle.plan(value, read(value['streams']['cf']['path'])[:4])
    value['qualification_plan']['original_native_evaluator_parity'] = (
        'SEPARATE_LOCKED_FIRST4_INDEPENDENT_ORIGINAL_FORWARD_IN_NEW_CF_W0; '
        'ACTUAL_PASS_REQUIRED_BEFORE_READY; NOT_A_PREREGISTRATION_PASS')
    tracking_schema.load_env(tracking['env_file'])
    for dataset in ('cf','zsre'):
        for method in METHODS:
            tracking_schema.config(tracking_config(value,configuration(method,dataset),
                                                   'chain',Path(method+'-'+dataset)))
    value['tracking_preflight'] = dict(strict_twelve_cell_CPU_config='PASS',
        credential_values_read=False, online_remote_identity='NOT_OBSERVED',
        source_route='SHARED_OFFICIAL_TRACKING_READONLY', duplicate_logger=False)
    value['checkpoint_identities'] = {dataset:{method:checkpoint_identity(value,method,dataset)
        for method in METHODS} for dataset in ('cf','zsre')}
    if qualification_producer_attempt is not None:
        from official.runners.server2 import qualification_input
        value['qualification_input_plan'] = qualification_input.plan(qualification_producer_attempt, value)
    value['qualification_plan_sha256'] = digest(value['qualification_plan'])
    out = Path(out).resolve()
    if OUTPUT not in out.parents or out.exists():
        raise ValueError('NEW_IMMUTABLE_TASK_EXECUTION_PREPARATION_REQUIRED')
    out.mkdir(parents=True)
    write_new(out/'manifest.json',value)
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--asset-manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--official-tree', required=True)
    parser.add_argument('--tracking-binding', required=True)
    parser.add_argument('--local-caps', required=True)
    parser.add_argument('--qualification-producer-attempt')
    args=parser.parse_args()
    prepare(args.asset_manifest,args.out,args.source,args.official_tree,args.tracking_binding,args.local_caps,
            qualification_producer_attempt=args.qualification_producer_attempt)


if __name__=='__main__':
    main()
