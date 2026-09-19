"""One matched B100 pair. No sequential entry or submission facility."""
import argparse
import gc
import json
from pathlib import Path
import resource
import time
import traceback
import numpy as np
import torch
from .config import ARMS, Scope
from .model import Runtime, require_lock
from .preparation import create_json, member, sha
from .generated_teacher import GeneratedTeacherStore
from .generated_oracle import GeneratedReferenceOracle
from .schedule import run_schedule
from .transaction import commit
from .observer_reuse import bind_prior
from project.run_scripts.single_layer_edit_preserving_correction.common import digest, tensor_sha, save_tensor
from project.run_scripts.single_layer_edit_preserving_correction.geometry import RightSpace, edit_null_space
from project.run_scripts.single_layer_edit_preserving_correction.runner import selection_seal, reuse_observation
from project.run_scripts.single_layer_edit_preserving_correction.observer import CanonicalObserver
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng


def require_member(item):
    if Path(item['path']).stat().st_size!=item['bytes'] or sha(item['path'])!=item['sha256']:
        raise ValueError('SEALED_MEMBER_CHANGED:'+item['path'])


def load_teacher(rt,lock):
    require_member(lock['generated_ready'])
    ready=json.loads(Path(lock['generated_ready']['path']).read_text())
    if ready['status']!='GENERATED_REFERENCE_READY_NOT_CORRECTION_VALIDATION':raise ValueError('PREP_NOT_READY')
    require_member(ready['binding']);binding=json.loads(Path(ready['binding']['path']).read_text())
    expected=dict(model_revision=lock['model_revision'],tokenizer_revision=lock['model_revision'],
        model_config_sha256=lock['model_config_sha256'],model_weights_sha256=lock['model_weights_identity_sha256'],
        tokenizer_sha256=lock['tokenizer_identity_sha256'],w0_sha256=rt.identity['W0_header_bytes'])
    if any(binding[k]!=v for k,v in expected.items()):raise ValueError('GENERATED_TEACHER_RUNTIME_BINDING')
    for k in ('torch','transformers','dtype','attention','matmul_tf32','cudnn_tf32'):
        want={'dtype':'float32','attention':'eager','matmul_tf32':False,'cudnn_tf32':False}.get(k,lock.get(k))
        if binding['runtime'][k]!=want:raise ValueError('GENERATED_TEACHER_RUNTIME_'+k)
    manifest=ready['manifest'];require_member(manifest)
    return GeneratedTeacherStore(Path(manifest['path']).parent,manifest['path'],
        expected_manifest_sha256=manifest['sha256'],inputs_path=lock['reference_inputs']['path'],
        expected_binding=binding,require_upstream_cache=True,verify_payloads=False),ready


def run(lock):
    require_lock(lock);Scope().require_batch(0)
    if lock['stage']!='MATCHED_B1':raise ValueError('MATCHED_B1_ONLY')
    out=Path(lock['output']);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();stage='source';rt=None;timing={}
    def begin(name):
        nonlocal stage
        stage=name;print(json.dumps(dict(event='B1_STAGE',stage=name)),flush=True)
        return time.monotonic()
    try:
        for item in lock['execution']['members']+lock['external_members']:require_member(item)
        create_json(out/'execution-entry.json',dict(lock=lock,max_batches=1,sequential_authorized=False,auto_continue=False))
        t=begin('W0_load');rt=Runtime(lock,out);timing['model_load_seconds']=time.monotonic()-t
        skipped=dict(status='SKIPPED_USER_DIRECTED',numerical_validation='NOT_ESTABLISHED',
            teacher_payload_audit=False,reference_cache_rehash=False,endpoint_rehash=False,
            physical_AD_FD_parity=False,checkpoint_reload_test=False,nonselected_full_hash=False,
            gradient_accumulation='GPU_FP64_DOCUMENT_ORDER_FINAL_SUM_TO_CPU',
            method_guards_unchanged=True)
        create_json(out/'runtime-validation-policy.json',skipped)
        before=None
        t=begin('complete_generated_teacher_binding');store,ready=load_teacher(rt,lock)
        rt.generated_store=store
        create_json(out/'generated-input-binding.json',dict(store=store.receipt,preparation=lock['generated_ready'],
            prior_preparation_seconds=ready['seconds'],new_generated_documents=0,old_teacher_relabelled=False))
        timing['teacher_binding_seconds']=time.monotonic()-t
        t=begin('load_shared_W0_upstream');reference=GeneratedReferenceOracle(rt.model,store)
        create_json(out/'reference-cache-setup.json',reference.setup_receipt)
        timing['reference_cache_load_seconds']=time.monotonic()-t
        t=begin('matched_native_reuse');native=rt.native(rt.records,out/'native',reuse=True);WN=native['weight'].clone()
        # native.reset intentionally clears stale old oracles. Rebind retained
        # W0 upstream cache after exact reused WN is installed; no prefix refit.
        rt.oracles.append(reference);rt.sync_oracles()
        timing['native_reuse_binding_seconds']=time.monotonic()-t
        t=begin('protected_current_inputs');current,rows,K,meta=rt.protected_oracle(rt.records)
        create_json(out/'protected-provenance.json',meta)
        save_tensor(out/'geometry/protected-keys.pt',dict(K=K,meta_sha=digest(meta)))
        timing['current_cache_and_keys_seconds']=time.monotonic()-t
        t=begin('immutable_shared_geometry');require_member(lock['P_star_basis'])
        pstar=torch.load(lock['P_star_basis']['path'],weights_only=True,mmap=True,map_location='cpu')
        allowed=RightSpace(pstar['basis'].numpy(),np.empty((pstar['basis'].shape[1],0)),
            'RESOLVED' if pstar['basis'].shape[1] else 'REPAIR_SPACE_EMPTY',pstar['diagnostic'])
        space=edit_null_space(allowed,K)
        create_json(out/'geometry/EN-F.json',space.receipt())
        save_tensor(out/'geometry/EN-F-factors.pt',dict(shared_allowed_basis=lock['P_star_basis'],
            blocked=torch.from_numpy(space.blocked),status=space.status))
        timing['shared_geometry_seconds']=time.monotonic()-t
        technical=create_json(out/'technical/checks.json',skipped)
        timing['technical_seconds']=0.0
        results={};exacts={};work={};weights={'N4':WN};seals={}
        for arm in ARMS:
            begin(arm)
            result,exact,measured=run_schedule(rt,arm,WN,reference,current,rows,K,allowed,space,meta,out/'arms'/arm)
            results[arm]=result;exacts[arm]=exact;work[arm]=measured;weights[arm]=result.weight
            ledger=member(out/'arms'/arm/'selection-ledger.json')
            seals[arm]=selection_seal('B1',arm,result.weight,lock['sample_order'],ledger['sha256'])
            create_json(out/'arms'/arm/'selection-seal.json',seals[arm])
            print(json.dumps(dict(event='SCHEDULE_SELECTED',arm=arm,stop=result.stop_reason,
                accepted=result.counters['accepted_rounds'],trial_slots=result.counters['attempted_trial_slots'],
                weight_sha256=tensor_sha(result.weight),seconds=measured['wall_seconds'])),flush=True)
        seals['N4']=selection_seal('B1','N4',WN,lock['sample_order'],digest(lock['native_reuse_lineage']))
        allseals=create_json(out/'ALL_SELECTIONS_SEALED.json',dict(arms=seals,official_P_N_access_so_far=0,
            Dev128_candidate_or_observer_evaluations_so_far=0,
            Dev128_W0_immutable_teacher_cache_binding='READ_DURING_SETUP_NOT_CONTROLLER_OBSERVATION',
            method_gradient_shared=False,max_batches=1,auto_continue=False))
        timing['selected_physical_parity_seconds']=0.0
        t=begin('final_history_and_atomic_checkpoints');commits={}
        for arm in ('N4',*ARMS):
            commits[arm]=commit(rt,arm,weights[arm],seals[arm],store.receipt['manifest_sha256'],out/'checkpoints'/arm,
                               verify_reload=False)
            if arm in exacts:
                exacts[arm]['history_appends']=commits[arm]['history_appends']
                create_json(out/'arms'/arm/'execution-exactness.json',exacts[arm])
        parity=dict(status='SKIPPED_USER_DIRECTED',checkpoints_CPU_verified=False,
                    GPU_continuation='NOT_TESTED_B2_UNAUTHORIZED')
        create_json(out/'matched-exactness.json',parity)
        timing['final_history_checkpoint_reload_and_parity_seconds']=time.monotonic()-t
        # No protected cache reuse is allowed across physical observers.
        rt.oracles=[reference];del current,K,rows,pstar,allowed,space,native
        results.clear();gc.collect()
        t=begin('postseal_canonical_observers');observer=CanonicalObserver(rt.model,rt.etok,runtime_identity=digest(rt.identity))
        w0seal=selection_seal('B1','W0',rt.W0,lock['sample_order'],allseals['sha256'])
        prior_w0=bind_prior(rt,observer,rt.W0,w0seal,'W0',before,out/'observers')
        W0=observer.observe(rt.records,rt.W0,selection_seal=w0seal,greedy=False,reuse=prior_w0);rt.sync_oracles()
        create_json(out/'observers/W0.json',W0)
        observations={};observed_endpoints={};devs={};dev_cache={}
        for arm in ('N4',*ARMS):
            weight=weights[arm];h=tensor_sha(weight)
            if h in observed_endpoints:
                prior=observed_endpoints[h]
                value=reuse_observation(prior['value'],prior['member'],seals[arm],
                    observer.compatibility_for(rt.records,weight,selection_seal=seals[arm]))
            else:
                prior_native=bind_prior(rt,observer,weight,seals[arm],'N4',before,out/'observers') if arm=='N4' else None
                value=observer.observe(rt.records,weight,selection_seal=seals[arm],w0_result=W0,greedy=True,reuse=prior_native)
                rt.sync_oracles()
            saved=create_json(out/'observers'/f'{arm}.json',value)
            observations[arm]=saved;observed_endpoints.setdefault(h,dict(value=value,member=saved))
            entry=tensor_sha(rt.W.detach());rng=digest(capture_rng())
            if h in dev_cache:
                dev=dict(reused=dev_cache[h]['path'],new_forwards=0,weight_sha256=h,selection=seals[arm])
            else:
                L,_,docrows=reference.kl(weight,role='Dev128')
                dev=dict(loss=L,rows=docrows,receipt=reference.last_sweep,weight_sha256=h,selection=seals[arm])
            if entry!=tensor_sha(rt.W.detach()) or rng!=digest(capture_rng()):raise ValueError('DEV_OBSERVER_MUTATION')
            devs[arm]=create_json(out/'observers'/f'{arm}-Dev128.json',dev);dev_cache.setdefault(h,devs[arm])
            rt.guard()
        timing['canonical_Dev_generation_observers_seconds']=time.monotonic()-t
        t=begin('terminal_artifact_inventory')
        files=[member(p) for p in sorted(out.rglob('*')) if p.is_file()]
        inventory=create_json(out/'artifact-manifest.json',dict(members=files,no_model_copy=True,
            source=lock['execution']['commit'],prior_native=lock['native_reuse_lineage'],shared_teacher=lock['generated_ready']))
        timing['terminal_nonselected_hash_inventory_seconds']=time.monotonic()-t
        create_json(out/'terminal.json',dict(status='B1_COMPLETE',science_efficiency_parity=parity['status'],
            shared_native_new_fits=0,new_native_targets=0,prior_native_seconds=lock['native_reuse_lineage']['prior_seconds'],
            requests=100,unique_requests=100,arms=ARMS,commits=commits,observations=observations,Dev128=devs,
            preparation=lock['generated_ready'],input=store.receipt,technical=technical,
            numerical_validation='NOT_ESTABLISHED',runtime_validation=skipped,
            setup_timing=timing,arm_work=work,artifact_manifest=inventory,
            total_program_seconds=time.monotonic()-started,peak_gpu_allocated=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved=torch.cuda.max_memory_reserved(),peak_host_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            actual_gpu_continuation='NOT_TESTED',Past_actual='B1_EMPTY_NA',max_batches=1,
            sequential_authorized=False,auto_continue=False,next_status='WAITING_USER_APPROVAL_FOR_SEQUENTIAL'))
        print(json.dumps(dict(event='B1_COMPLETE',parity=parity['status'],B2_authorized=False)),flush=True)
    except BaseException as exc:
        create_json(out/'failure.json',dict(status='TECHNICAL_FAILURE',stage=stage,error=repr(exc),
            traceback=traceback.format_exc(),seconds=time.monotonic()-started,source=lock['execution']['commit'],
            partial_preserved=True,automatic_restart=False,method_threshold_changed=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--max-batches',type=int,required=True)
    a=p.parse_args()
    if a.max_batches!=1:raise ValueError('B1_ONLY')
    run(json.loads(a.lock.read_text()))
