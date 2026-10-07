"""CPU-only new immutable repair binding; model/stat/reference payloads reused.

Requires the final read SH1 API to have been bound in the task-private bridge.
It never qualifies a GPU route, generates text, fits a target or modifies old
source/config/locks. The actual qualification path is a planned output only.
"""
import argparse
import copy
import subprocess
from pathlib import Path

from .generation_common import (LOCAL, ROOT, small_member, digest, member,
    read, require, sha, stat_seal, verify, write)
from .generation_cache_common import (ATTEMPT, ENVELOPE, NONCE, PARENT_TASK,
    REPAIR_LOCAL, TASK, authority, ready)
from .generation_cache_qualification import (build_plan, freeze_plan,
    installed_native_binding)
from .generation_cache_reuse import (INVENTORY_SCHEMA, read_member, semantic_identity,
    build_old_w0_reuse_binding)

PACKAGE = 'project/run_scripts/experiment_generation_eval'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def bind(shared_source, package_tree, *, batch_microbatch=8, memory_admission_member=None):
    authority()
    require(len(shared_source) == len(package_tree) == 40
        and all(c in '0123456789abcdef' for c in shared_source + package_tree),
        'CACHE_REPAIR_EXACT_SHARED_GIT_IDENTITY')
    require(git('rev-parse', shared_source + ':' + PACKAGE)
        == git('rev-parse', 'HEAD:' + PACKAGE) == package_tree,
        'CACHE_REPAIR_RECEIVED_SHARED_SOURCE_TREE')
    require(batch_microbatch in (4, 8), 'CACHE_REPAIR_PREDECLARED_MB4_OR_MB8')
    if batch_microbatch == 4:
        require(memory_admission_member is not None, 'CACHE_REPAIR_MB4_MEMORY_EVIDENCE_REQUIRED')
        evidence = read(verify(memory_admission_member))
        require(evidence['status'] == 'PREDECLARED_MEMORY_MB4'
            and evidence['model'] == 'gptj' and evidence['before_actual_qualification'] is True,
            'CACHE_REPAIR_MB4_MEMORY_DECISION_BEFORE_TEST')
    # This is the actual owner adapter, not a guessed exported SH1 API name.
    from .generation_cache_bridge import SHARED_SOURCE, PACKAGE_TREE, api_binding
    require(SHARED_SOURCE == shared_source and PACKAGE_TREE == package_tree,
        'CACHE_REPAIR_ACTUAL_API_SOURCE_BOUND')
    api = api_binding()
    require(api['status'] == 'READ_BOUND_SHARED_API'
        and api['source_sha'] == shared_source, 'CACHE_REPAIR_ACTUAL_API_READ')
    out = REPAIR_LOCAL / 'preparation-r1'
    require(not out.exists() and not ATTEMPT.exists(), 'CACHE_REPAIR_BIND_CREATE_ONCE')
    old_path = LOCAL / 'preparation-r2/config.json'
    original = read(old_path)
    config = copy.deepcopy(original)
    for row in config['assets'] + config['runtime']['members'] + [config['observer_identity']]:
        stat_seal(row)
    old_inventory_path = REPAIR_LOCAL / 'inputs/old-complete-cases-r1.json'
    old_inventory, inventory_member = read_member(old_inventory_path)
    require(old_inventory['schema'] == INVENTORY_SCHEMA
        and old_inventory['identity_sha256'] == digest(old_inventory['identity'])
        and old_inventory['payload_sha256'] == digest({key: value for key, value in old_inventory.items()
            if key != 'payload_sha256'})
        and [entry['occurrence'] for entry in old_inventory['entries']] == list(range(1, 2001))
        and old_inventory['reusable_cases'] == sum(entry['status'] == 'REUSABLE_COMPLETE_CASE'
            for entry in old_inventory['entries'])
        and old_inventory['unknown_cases'] == 2000 - old_inventory['reusable_cases'],
        'CACHE_REPAIR_OLD_INVENTORY_INTERNAL_IDENTITY')
    require(old_inventory['identity']['semantic_inputs'] == semantic_identity(config)
        and old_inventory['planned_cases'] == 2000, 'CACHE_REPAIR_OLD_SEMANTIC_REUSE_BOUND')
    old_binding = build_old_w0_reuse_binding(inventory_member,
        out_directory=REPAIR_LOCAL / 'inputs/old-cold-phase-r1')
    config.update(task_id=TASK, instruction_id=NONCE, parent_task_id=PARENT_TASK,
        authority=small_member(ROOT / ENVELOPE), attempt=str(ATTEMPT))
    gen = config['generation']
    gen.update(source_sha=shared_source, package_tree=package_tree,
        shared_source_members=[small_member(p) for p in sorted((ROOT / PACKAGE).rglob('*.py'))],
        W0_cache=str(ATTEMPT / 'W0-generation-cache'), generator_route='RUNTIME_QUALIFIED_ROUTE_ONLY')
    # Local tokenizer load only, no AutoModel import, CUDA, forward or download.
    from transformers import AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    records = load_prefix(Path(config['stream']).parent, 2000)
    plan, cohort = build_plan(records, tokenizer, model_identity=gen['model_identity'],
        tokenizer_identity=digest(config['model_assets']),
        reference_identity=gen['reference_assets_sha256'], shared_source_sha=shared_source,
        native_source_binding=installed_native_binding(), batch_microbatch=batch_microbatch,
        admission_reason='DEFAULT_MB8' if batch_microbatch == 8 else 'PREDECLARED_MEMORY_MB4')
    if memory_admission_member is not None:
        plan['memory_admission_member'] = memory_admission_member
    plan_binding = freeze_plan(plan, cohort, REPAIR_LOCAL / 'qualification-plan-r1')
    gen['repair'] = dict(status='PLAN_BOUND_NOT_ACTUAL_PASS', actual_qualification_status='NOT_RUN',
        qualification_in_first_replacement_job=True,
        qualification_plan=plan_binding['qualification_plan'],
        qualification_plan_sha256=plan_binding['qualification_plan_sha256'],
        qualification_cohort=plan_binding['qualification_cohort'],
        shared_qualification_plan=plan_binding['shared_qualification_plan'],
        shared_qualification_plan_sha256=plan_binding['shared_qualification_plan_sha256'],
        qualification_receipt_path=str(ATTEMPT / 'BASE_MEMIT/qualification.json'),
        compatibility_manifest_path=str(ATTEMPT / 'BASE_MEMIT/compatibility.json'),
        old_complete_case_inventory=inventory_member,
        old_cold_observation_guard=old_binding['cold_observation_guard_member'],
        old_w0_reuse_binding=old_binding['binding_member'],
        old_w0_reuse=old_binding['old_w0_reuse'],
        shared_API=api, source_relabel=False, automatic_OOM_retry=False,
        fixed_microbatch_chosen_by_actual_qualification=True,
        old_attempt=str(LOCAL / 'attempt-r1'), actual_editor_resume=False)
    require(semantic_identity(config) == semantic_identity(original),
        'CACHE_REPAIR_SCIENCE_MODEL_INPUT_PRECISION_UNCHANGED')
    ready(config)
    write(out / 'config.json', config)
    write(out / 'binding.json', dict(status='PLAN_BOUND_NOT_ACTUAL_PASS',
        source=shared_source, package_tree=package_tree, shared_API=api,
        parent_config=member(old_path), config=member(out / 'config.json'),
        qualification_plan=plan_binding['qualification_plan'],
        shared_qualification_plan=plan_binding['shared_qualification_plan'],
        old_inventory=member(old_inventory_path),
        reusable_complete_cases=old_inventory['reusable_cases'],
        missing_or_unknown_cases=old_inventory['unknown_cases'],
        model_loads=0, generation_forwards=0, native_fits=0, checkpoint_saved=False,
        shared_environment_modified=False, actual_GPU_qualification=False))
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-source', required=True)
    parser.add_argument('--package-tree', required=True)
    parser.add_argument('--batch-microbatch', type=int, choices=(4, 8), default=8)
    parser.add_argument('--memory-admission-json', type=Path)
    args = parser.parse_args()
    bind(args.shared_source, args.package_tree, batch_microbatch=args.batch_microbatch,
         memory_admission_member=member(args.memory_admission_json) if args.memory_admission_json else None)


if __name__ == '__main__':
    main()
