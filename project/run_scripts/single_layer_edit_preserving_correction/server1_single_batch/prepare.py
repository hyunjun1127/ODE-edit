"""CPU-only S1 path binding and reuse admission; frozen inputs never edited."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import torch
from ..common import member,write as create_once,sha,tensor_sha,digest
from .reuse import verify_endpoint,COMPLETED,MISSING,read

ORIGIN='/data/janghj/ODE-edit/local/'
E='single-layer-edit-preserving-correction/20260918-v1/'
BLUE=Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/blue-source')
HISTORY=BLUE.parent/'historical/blue_alphaedit_sequential_comparison'

def write(path,value):
    if Path(path).exists():
        if read(path)!=value:raise ValueError('CREATE_ONCE_BINDING_DIFF:'+str(path))
        return member(path)
    return create_once(path,value)

def prepare(repo,root):
    repo,root=Path(repo).resolve(),Path(root).resolve()
    torch.set_num_threads(8)
    imports=root/'imports/server4-b001-r1';sealed=imports/'sealed'
    received=read(imports/'receiver-seal.json')
    if received['status']!='FULL_SIZE_SHA_VERIFIED':raise ValueError('RECEIVER_NOT_SEALED')
    def path(value):
        p=Path(value)
        return sealed/p.relative_to(ORIGIN)
    original=read(sealed/E/'M/attempt-metadata-r1/execution.lock.json')
    if original['execution']['head']!='87f65ea2abcbe7e77e04367f73a001d63443734b':raise ValueError('M_SOURCE')
    frozen=sealed/E/'M/attempt-metadata-r1/episodes/b001/attempt-v1'
    runtime=read(frozen/'runtime-load.json');identity=runtime['identity']
    native=torch.load(path(original['reused_native_b1']),weights_only=True,map_location='cpu',mmap=True)
    if tensor_sha(native['weight'])!=original['reused_native_binding']['b1_endpoint_verified']:
        raise ValueError('NATIVE_CAPSULE')
    ids=original['sample_order'][:100]
    if [r['case_id'] for r in native['target_observations']]!=ids or native['target'].shape!=(4096,100):
        raise ValueError('NATIVE_TARGET_CASE_BINDING')
    rows=[]
    for arm in COMPLETED:
        weight,ledger,seal=verify_endpoint(frozen,arm,ids=ids,W0=identity['W0'],
            WN=tensor_sha(native['weight']),context_tokens=identity['context_tokens'])
        rows.append(dict(arm=arm,optimization='REUSE',observer='EVAL_ONLY',
            endpoint=member(frozen/'arms'/arm/'final-L4.pt'),weight_sha256=tensor_sha(weight),
            seal=member(frozen/'arms'/arm/'selection-seal.json'),stop=ledger['stop_reason'],
            new_optimization=0,source_hardware=runtime['model_gpu_name']))
        del weight
    for arm in MISSING:
        if (frozen/'arms'/arm/'selection-seal.json').exists():raise ValueError('UNEXPECTED_COMPLETED_ARM')
        rows.append(dict(arm=arm,optimization='RUN_MISSING',observer='EVAL_ONLY_AFTER_SELECTION',
            partial='REFERENCE_ONLY_NOT_RESUMABLE' if arm=='EN-F' else 'ABSENT',new_optimization=1))
    write(root/'locks/reuse.json',dict(arms=rows,native_fit_new=0,unique_requests=100,arm_requests=800,
        geometry='REUSE_WHEN_KEY_IDENTITY_EXACT_ELSE_AFFECTED_EXACT_RECOMPUTE',
        gradient='REUSE_WHEN_FIXED_TEACHER_AND_CACHE_IDENTITY_EXACT',
        original_cancelled_GPU_seconds=9768,original_SIGTERM_rollback='NOT_VERIFIED',
        historical_canonical_metrics='REFERENCE_ONLY_CROSS_HARDWARE_NOT_BITWISE',
        source=original['execution']['head'],import_seal=member(imports/'receiver-seal.json')))
    # Path-only teacher manifest derivative: full teacher payload SHA stays exact.
    teacher=read(path(original['teacher_manifest']['path']))
    derived=copy.deepcopy(teacher)
    for m in derived['cache_shards']:m['path']=str(path(m['path']))
    for key in ('lock','reference_tokens','splits'):derived[key]['path']=str(path(derived[key]['path']))
    teacher_ref=write(root/'locks/teacher-path-binding.json',derived)
    write(root/'locks/teacher-path-diff.json',dict(original=member(path(original['teacher_manifest']['path'])),
        derived=teacher_ref,changes='absolute paths only',payload_members=teacher['cache_shards'],
        original_bytes_preserved=True,teacher_recomputed=0,Report256_access=0))
    from project.run_scripts.single_layer_zflow.native_binding import verify_native_source
    blue=verify_native_source(BLUE)
    from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
    evalbind=bind_evaluation_sources(HISTORY,helper_root=repo/'project/run_scripts')
    prior=read(path(original['observer_reuse_binding']['N4_B1']['path']))
    if sorted(v['sha256'] for v in prior['source_binding'].values())!=sorted(v['sha256'] for v in evalbind['sources'].values()):
        raise ValueError('EVALUATOR_SOURCE_IDENTITY')
    # Keep exact used primitive closure; unrelated main/S additions aren't executable inputs.
    used=['common.py','alltoken.py','binding.py','geometry.py','optimizer.py','observer.py','retained_native.py']
    proof=[]
    for m in original['execution']['members']:
        rel=m['relative'];p=repo/rel
        required=(rel.startswith('project/run_scripts/single_layer_edit_preserving_correction/') and p.name in used) or any(
            rel.endswith(s) for s in ('baseline_mechanism_first/evaluation.py','baseline_mechanism_first/fixtures.py',
                'bg_tw_reference/ep_tw/model_adapter.py','low_cost_write_donor_pilot/fitting.py','single_layer_zflow/native_binding.py'))
        if required:
            if sha(p)!=m['sha256']:raise ValueError('USED_M_SOURCE_DIFF:'+rel)
            proof.append(member(p))
    lock=copy.deepcopy(original)
    for key in ('cold_capsule','P_star_basis','technical_evidence'):
        lock[key]=member(path(original[key]['path']))
    lock['config4']=str(path(original['config4']))
    lock['reused_native_b1']=str(path(original['reused_native_b1']))
    lock['original_teacher_manifest']=original['teacher_manifest'];lock['teacher_manifest']=teacher_ref
    lock['reference_root']=str(path(original['reference_root']))
    lock['blue_root']=str(BLUE);lock['historical_evaluator_root']=str(HISTORY)
    lock['helper_scripts_root']=str(repo/'project/run_scripts')
    lock['snapshot']='/mnt/raid5/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
    lock['projector']='/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt'
    lock['dataset_root']='/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1'
    lock.update(instruction_id='ODEEDIT-S06-ENFC-SINGLE-BATCH-M-RESUME-SH1-V1',runtime_node='devbox',
        allowed_stages=['M_B001_ONLY'],native_fit_new_allowed=False,storage_waiver_inherited=False,
        frozen_b001=str(frozen),M_root=str(root/'output-r1'),source_gpu_name=runtime['model_gpu_name'],
        M_native_new_max=0,full_numerical_validation='NOT_ESTABLISHED')
    # Observation refs preserve original bytes with only the outer path binding changed.
    for k in ('N4_B1','W0_first1000'):
        lock['observer_reuse_binding'][k]=member(path(original['observer_reuse_binding'][k]['path']))
    write(root/'locks/source-input.json',dict(blue=blue,evaluator=evalbind,unchanged_M_used_source=proof,
        original_execution=original['execution']['head'],original_tree=original['execution']['tree'],
        runtime_diff='expected node devbox; S1 paths; reuse-first B1 adapter; numerical constants unchanged',
        frozen_model_revision=Path(lock['snapshot']).name,
        fixed_model_config=member(Path(lock['snapshot'])/'config.json'),
        projector=member(lock['projector']),receiver=received))
    write(root/'locks/resource.json',dict(server='server1',node='devbox',project_cap=2,gpus=1,cpus=8,
        mem_MiB=182272,wall_hours=24,gpu_hour_budget=None,export='NONE',requeue=0,
        free_bytes=shutil.disk_usage(root).free,import_bytes=received['total_bytes'],
        additional_disk_reserve_bytes=32*(1<<30),S4_storage_waiver=False,
        requested_wall_not_actual_cost=True,
        cost_estimate='B1 only; 3 missing optimization arms <=40 candidate slots, 6 gradient rounds incl COV; guards/geometry/observers additional. S1 A6000 vs S4 Blackwell speed unmeasured; 24h allocation ceiling not scientific expansion.',
        initial_gate='new EN-F saved/reloaded + selection sealed + canonical observer + Dev128 + W0/M0 reset',
        initial_NOT_full_validation=True))
    write(root/'locks/runtime-draft.json',lock)
    print(json.dumps(dict(status='CPU_REUSE_BOUND',arms_reused=5,arms_missing=3,CUDA_initialized=torch.cuda.is_initialized())))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--root',required=True);a=p.parse_args()
    prepare(a.repo,a.root)
