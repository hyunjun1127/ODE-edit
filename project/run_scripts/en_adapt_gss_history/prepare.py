"""CPU-only fixed-input and conservative two-job resource planning."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.en_adaptive_nullspace.runtime import verify_ready_inputs
from .io import save

BASE = Path('/data/janghj/ODE-edit/local/en-adapt-gss-history/20260920-v1')
READY = Path('/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1')
PARENT_INPUTS = Path('/data/janghj/ODE-edit/local/en-adaptive-nullspace/20260920-v1/inputs')
ARMS = ('EN_ADAPT_H_RES', 'EN_ADAPT_H_GSS_REC')
FULL_BATCHES = [2,5,10,20,30,40,50,60,70,80,90,100]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_map_seal(preparation):
    import numpy as np
    from .factors import make_maps
    basis_path=PARENT_INPUTS/'pstar-derived-v1/basis.npy'
    started=time.monotonic();digest=sha(basis_path)
    if digest!='515f7d9947b5c7e177ca4c8d9ba0529823f06db9e714fb07c2b7eb63c36e4e05':
        raise ValueError('REUSABLE_PSTAR_BASIS_SHA')
    basis=np.load(basis_path,mmap_mode='r',allow_pickle=False)
    maps=make_maps(4096,14336,basis)
    arrays={}
    for i,parts in enumerate(maps):
        for name,value in zip(('output','input','Pstar_input'),parts):
            arrays[f'replica_{i}_{name}']=dict(shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
    st=basis_path.stat()
    seal=dict(seed=20260920,generator='numpy.Generator(PCG64)',
        draw_order='replica0_output_input_then_replica1_output_input',arrays=arrays,
        basis=dict(path=str(basis_path),size=st.st_size,sha256=digest,
            stat=[st.st_dev,st.st_ino,st.st_mtime_ns]),seconds=time.monotonic()-started,
        model_run=False,GPU_allocated=0,map_code_path='project/run_scripts/en_adapt_gss_history/factors.py',
        raw_map_values_not_Git=True)
    path=Path(preparation)/'fixed-map-logical-seal.json';save(path,seal)
    return path


def prepare(root=BASE):
    root = Path(root).resolve()
    if root != BASE:
        raise ValueError('EXACT_TASK_ROOT_REQUIRED')
    root.mkdir(parents=True, exist_ok=True)
    prep = root / 'preparation-v1'
    prep.mkdir(exist_ok=False)
    manifest = json.loads((READY/'manifest.json').read_text())
    ready_binding = verify_ready_inputs(manifest)
    records = load_prefix(manifest['dataset_root'], 10000)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(manifest['models']['llama3-8b-inst']['snapshot'], local_files_only=True)
    tok.pad_token_id = tok.eos_token_id
    spec = importlib.util.spec_from_file_location('sh3_gss_canonical_contracts', READY/'evaluator_source/project/run_scripts/alphaedit_strength_neutral_barrier/contracts.py')
    contracts = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = contracts
    spec.loader.exec_module(contracts)
    rows = []
    for i, rec in enumerate(records):
        rw = rec['requested_rewrite']
        prefix = contracts.prompt_token_ids(tok, rw['prompt'].format(rw['subject']))
        target = contracts.target_token_ids(tok, rw['target_new']['str'])
        ids = prefix + target[:-1]
        rows.append(dict(ordinal=i, case_id=rec['case_id'], batch=i//100+1,
            input_tokens=len(ids), target_tokens=len(target),
            input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            target_sha256=hashlib.sha256(json.dumps(target).encode()).hexdigest()))
    # Conservative upper bound: every occurrence gets a distinct teacher/cache,
    # even though repeated versions and terminal B100 need fewer actual files.
    vocabulary = 128256
    tokens = sum(r['input_tokens'] for r in rows)
    targets = sum(r['target_tokens'] for r in rows)
    teacher = targets * vocabulary * 4
    keys_residual = tokens * (14336+4096) * 4
    packed = tokens * 3 * 8
    maps = tokens * 32 * 2 * 8
    cold_each = teacher + keys_residual + packed + maps
    per_version_metadata = 16384
    cold_total = 2 * (cold_each + len(rows)*per_version_metadata)
    # Scalar observations, current aliases/frontiers, diagnostic/ledger receipts:
    # lossless gzip is used, but this is a budget, not an assumed compression ratio.
    outputs = 8 * 2**30
    temporary = 2 * 2**30
    margin = 8 * 2**30
    required = cold_total + outputs + temporary + margin
    stat = os.statvfs(root)
    free = stat.f_bavail * stat.f_frsize
    reference = json.loads((PARENT_INPUTS/'generated-v1/manifest.json').read_text())
    prior = {}
    for line in (PARENT_INPUTS.parent/'transfer-verified-members.jsonl').read_text().splitlines():
        row = json.loads(line)
        prior[row.get('relative', row.get('path', ''))] = row
    ref_bindings = []
    for doc in reference['documents']:
        for kind in ('capsule','keys','residual','logp'):
            member = doc[kind]
            path = PARENT_INPUTS/'generated-v1'/member['path']
            st = path.stat()
            old = prior.get(member['path'])
            if old is None:
                raise ValueError('PRIOR_FULL_VERIFICATION_MEMBER_MISSING:' + member['path'])
            if st.st_size != member['bytes'] or old['sha256'] != member['sha256']:
                raise ValueError('REFERENCE_SIZE_SHA_IDENTITY')
            if [st.st_dev, st.st_ino, st.st_mtime_ns] != [old['device'], old['inode'], old['mtime_ns']]:
                raise ValueError('REFERENCE_PRIOR_STABLE_IDENTITY_CHANGED')
            ref_bindings.append(dict(path=str(path), bytes=st.st_size,
                sha256=member['sha256'], stat=[st.st_dev, st.st_ino, st.st_mtime_ns]))
    save(prep/'sequence-token-identity.json', dict(rows=rows, records=10000,
        input_tokens=tokens,target_tokens=targets,max_input_tokens=max(r['input_tokens'] for r in rows),
        max_target_tokens=max(r['target_tokens'] for r in rows), padding='canonical right attribute, explicit unpadded TF input',
        reference_manifest_sha256=sha(PARENT_INPUTS/'generated-v1/manifest.json')))
    save(prep/'reference-stat-binding.json', ref_bindings)
    save(prep/'ready-binding.json', ready_binding)
    # Node/partition metadata is a preparation check, never a job/result poll.
    node = subprocess.check_output(['scontrol','show','node','ubuntu','--oneliner'], text=True)
    partition = subprocess.check_output(['scontrol','show','partition','gpu','--oneliner'], text=True)
    memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    full_requests = sum(FULL_BATCHES)*100
    ordinary_union_upper = (100-len(FULL_BATCHES))*(100+100+128)
    observer_union_upper = full_requests + ordinary_union_upper
    plan = dict(schema='SH3_GSS_TWOARM_RESOURCE_V1',created_unix=time.time(),
        authority='7216cc96c0436fcbd1d3b7cba2c69c09d613bd93',
        arms=list(ARMS), independent_jobs=2, shared_edited_state=False,
        cpu=dict(per_job=8,aggregate=16,node_observed=node),
        memory=dict(per_job_limit_MiB=121856,aggregate_limit_MiB=243712,
            per_job_peak_estimate_GiB=100,aggregate_peak_estimate_GiB=200, measured=False,
            prior_B300_MaxRSS_GiB=69.375, observed_meminfo_KiB=memory,
            components_GiB=dict(model_CPU_loading_transient=32,reference_prefix_cache=19,
                mapped_reference_page_working_set=12,P_basis_C0=8,TSQR_SVD_workspace=16,
                W_M_gradient_controller=8,history_hot_A_keys_teacher=1,
                observations_ledger_python=4,allocator_and_compression=8),
            note='component maxima are not all concurrent; conservative 100GiB planned peak, 119GiB requested; new path peak unmeasured'),
        storage=dict(free_bytes=free,free_inodes=stat.f_favail,exclusive_reserved=False,
            per_arm_cold_teacher_bytes=teacher,per_arm_keys_residual_bytes=keys_residual,
            per_arm_packed_bytes=packed,per_arm_projected_key_upper_bytes=maps,
            cold_two_arms_with_metadata_bytes=cold_total,
            scalar_output_budget_bytes=outputs,atomic_temp_budget_bytes=temporary,
            safety_margin_bytes=margin,required_free_bytes=required,
            headroom_bytes=free-required,immutable_reference_duplicate_bytes=0,
            checkpoint_bytes=0,old_artifacts_deleted=0,compression='observations gzip lossless; teacher FP32 uncompressed'),
        time=dict(expected_wall_hours_each=[28,72],expected_allocation_GPU_hours_two=[56,144],
            requested_wall_hours_each=168,partition_metadata=partition, measured=False,
            prior_measured_core_seconds_per_state=[652.49,672.72],
            extrapolated_core_hours_each=[652.49*100/3600,672.72*100/3600],
            extra_work='history short suffix KL/NLL VJPs, three observer endpoints, cold teacher I/O, CPU diagnostics/report, two-job resource contention',
            per_arm_observer_union_request_upper=observer_union_upper,
            per_arm_entry_native_selected_request_upper=3*observer_union_upper,
            W0_observer_requests_each=10000,
            GPUhour_hardcap=None),
        counts=dict(planned_native_batches=200,planned_native_requests=20000,
            planned_reference_gradients=200,max_candidates=400,
            GSS_REC_overflow_sweeps_no_overwrite_upper=94,
            GSS_REC_selection_NLL_fact_VJP_no_overwrite_upper=57516,
            RES_selection_NLL_fact_VJP=0),
        scopes=dict(no_checkpoint=True,post_release_monitoring=False,
            actual_initial='NOT_OBSERVED',actual_terminal='NOT_OBSERVED'))
    save(prep/'resource-plan.json',plan)
    if free < required or stat.f_favail < 150000:
        raise RuntimeError(f'STORAGE_BLOCKER deficit={max(0,required-free)} free={free} required={required}')
    map_seal=make_map_seal(prep)
    for arm in ARMS:
        short = arm.removeprefix('EN_ADAPT_H_').lower()
        attempt = root/short/'attempt-v1'
        inputs = attempt/'inputs'
        inputs.mkdir(parents=True,exist_ok=False)
        config = dict(schema='EN_ADAPT_GSS_TWOARM_RUNTIME_V1',arm=arm,
            authority=plan['authority'],batches=100,batch_size=100,bank_capacity=512,
            pending_capacity=100,hot_pool_capacity=612,seed=20260920,map_dimensions=[32,32],
            map_replicas=2,map_feature_dimension=2048,history_microbatch=1,
            half_life_batches=5.12,history_coefficient=1.,reference_coefficient=1.,
            primary_epsilon=.05,native_chunk_size=16,
            current_only_Q=True,full_reference_documents=512,history_target_truncation=None,
            save_checkpoints=False,exact_crash_resume='NOT_AVAILABLE',cross_job_prefix_sharing=False,
            full_observer_batches=FULL_BATCHES,history_panel_size=128,
            parent_precision='NOT_ESTABLISHED',sketch_precision='TO_BE_MEASURED_NO_GATE',
            basis=str(PARENT_INPUTS/'pstar-derived-v1/basis.npy'),
            map_seal=str(map_seal),
            output=str(attempt/'output'),history_root=str(attempt/'cold-history'),
            generated_root=str(PARENT_INPUTS/'generated-v1'),
            reference_inputs=str(PARENT_INPUTS/'reference-inputs.json'))
        m=dict(manifest,node='ubuntu',python_version=[3,12,3],
            generated_root=config['generated_root'], reference_inputs=config['reference_inputs'],
            maximum_batch=100,task_runtime_adapter='en_adapt_gss_history.runtime.Runtime')
        save(inputs/'manifest.json',m)
        emap={name:str(READY/name) for name in json.loads((Path(__file__).parents[1]/'en_adaptive_nullspace/metrics-contract.json').read_text())['source_files']}
        save(inputs/'evaluator-map.json',emap)
        save(inputs/'config.json',config)
    print(json.dumps(dict(preparation=str(prep),free_bytes=free,required_free_bytes=required,
                         headroom_bytes=free-required,cold_two_arms=cold_total)))


if __name__ == '__main__':
    prepare()
