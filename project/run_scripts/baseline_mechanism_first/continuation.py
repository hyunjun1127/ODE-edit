"""Independent source-exact warm window; first batch also supplies E1 data.

No context/RNG substitution, cached-z rescue, or new full chain. A missing
comparison checkpoint prevents an equivalence claim, not collection of an
explicitly labeled native diagnostic branch from a verified saved entry.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import time
import traceback

from .contracts import ContractBoundary, digest, member, save
from .runner import tensor_artifact
from .case_population import source_digest


def execute(lock_path, output):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    from .assets import bind_native
    from .evaluation import bind_evaluation_sources,evaluate_records
    from .fixtures import FixtureTransaction,SingletonSpec,capture_rng,restore_checkpoint,select_projector,tensor_sha
    from .native_runner import run_native_batch
    from .observer import observe_native
    from .panels import historical_ordinals
    root=Path(output).absolute();root.mkdir(parents=True,exist_ok=False)
    lock=json.loads(Path(lock_path).read_text());started=time.monotonic();stage='INPUTS';committed=[];tx=None
    forward_counts={};forward_handle=None
    try:
        for row in lock['source_members']:
            member(row['path'],expected=row['sha256'])
        for row in [lock['entry_checkpoint'],*lock['companion_members'].values()]:
            member(row['path'],expected=row['sha256'])
        for key in ('general_manifest','covariance','comparison_checkpoint'):
            if lock.get(key):member(lock[key]['path'],expected=lock[key]['sha256'])
        spec=SingletonSpec(lock['cell']['layer']);entry_n=lock['cell']['entry_n']
        if entry_n not in (1000,5000,9000):raise ContractBoundary('WARM_ENTRY_SCOPE')
        if lock['native_batches']!=list(range(entry_n//100+1,entry_n//100+11)):
            raise ContractBoundary('CONTINUATION_WINDOW_SCOPE')
        assert torch.__version__==lock['torch'] and transformers.__version__==lock['transformers']
        records=load_prefix(lock['dataset_root'],10000)
        cp=torch.load(lock['entry_checkpoint']['path'],map_location='cpu',weights_only=False,mmap=True)
        reference_entry=json.loads(Path(lock['companions']['entry.json']).read_text())
        current=records[entry_n:entry_n+100]
        assert [r['case_id'] for r in current]==reference_entry['request_ids']
        assert [source_digest(r['requested_rewrite']) for r in current]==reference_entry['request_hashes']
        native,hp_type=bind_native(lock['native_root']);hp=hp_type.from_json(lock['config'])
        assert hp.layers==[spec.layer]
        bind_evaluation_sources(lock['historical_root'],helper_root=lock['helper_root'])
        random.seed(lock['seed']);np.random.seed(lock['seed']);torch.manual_seed(lock['seed']);torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32=lock['tf32_matmul'];torch.backends.cudnn.allow_tf32=lock['tf32_cudnn']
        stage='MODEL_LOAD';load_start=time.monotonic()
        model=AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
        tok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
        evaltok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);evaltok.pad_token_id=evaltok.eos_token_id
        assert tok.padding_side==evaltok.padding_side=='right' and all(p.dtype==torch.float32 for p in model.parameters())
        model_load_seconds=time.monotonic()-load_start
        def forward_counter(_module,args,kwargs):
            row=forward_counts.setdefault(stage,dict(calls=0,input_positions=0,nonpadding_positions=0))
            row['calls']+=1;ids=kwargs.get('input_ids',args[0] if args else None);mask=kwargs.get('attention_mask')
            if ids is not None:row['input_positions']+=ids.numel()
            if mask is not None:row['nonpadding_positions']+=int(mask.sum())
        hook_state=[(m,dict(m._forward_hooks),dict(m._forward_pre_hooks),dict(m._backward_hooks)) for m in model.modules()]
        forward_handle=model.register_forward_pre_hook(forward_counter,with_kwargs=True)
        stack=torch.load(lock['projector'],map_location='cpu',weights_only=True,mmap=True)
        projector,mapping=select_projector(stack,spec);del stack
        history=torch.zeros_like(projector);weight=dict(model.named_parameters())[spec.weight_name]
        w0=weight.detach().cpu().clone();entry_weight=cp['weights'][spec.weight_name]
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        tx=FixtureTransaction(model,native,history)
        with tx:
            stage='EXACT_SAVED_ENTRY_RESTORE'
            restored=restore_checkpoint(model,native,history,cp,spec,expected_model_revision=lock['model_revision'],expected_seen_ids=[r['case_id'] for r in records[:entry_n]])
            assert tensor_sha(weight)==reference_entry['signature']['weights'][spec.weight_name]['sha256']
            assert tensor_sha(history)==reference_entry['signature']['cache_sha256']
            assert source_digest(native.CONTEXT_TEMPLATES_CACHE)==reference_entry['context_hash']
            save(root/'runtime.json',dict(input_lock=member(lock_path),source_head=source_head,entry_restore=restored,
                projector_mapping=mapping,model_dtype='torch.float32',torch=torch.__version__,transformers=transformers.__version__,
                GPU=torch.cuda.get_device_name(),attention=model.config._attn_implementation,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
                model_load_seconds=model_load_seconds,prior_gate_reused=lock['prior_gate'],repeat_FD=0,
                new_direction_FD=bool(lock['diagnostics'].get('initial_probe_new_direction',False)),
                original_trajectory_equivalence='NOT_YET_VERIFIED'))
            for batch in lock['native_batches']:
                stage=f'B{batch:03d}_NATIVE';broot=root/f'B{batch:03d}';rows=records[(batch-1)*100:batch*100]
                before=weight.detach().cpu().clone();before_m=history.clone();t=time.monotonic()
                requests=[dict(r['requested_rewrite'],case_id=int(r['case_id'])) for r in rows]
                save(broot/'entry-state.json',dict(batch=batch,weight_sha256=tensor_sha(before),
                    history_sha256=tensor_sha(before_m),rng_sha256=digest(capture_rng()),
                    contexts_sha256=source_digest(native.CONTEXT_TEMPLATES_CACHE),
                    request_ids=[r['case_id'] for r in rows],request_hashes=[source_digest(r['requested_rewrite']) for r in rows]))
                result=run_native_batch(model,tok,native,hp,history,projector,requests,spec,observer=observe_native)
                torch.cuda.synchronize();elapsed=time.monotonic()-t
                raw=result['observer'].pop('_tensors');raw_ref=tensor_artifact(broot/'native-tensors.pt',raw)
                endpoint=result['weight'];end_m=result['history'];rng=capture_rng();contexts=deepcopy(native.CONTEXT_TEMPLATES_CACHE)
                snapshot=tensor_artifact(broot/'W-method-state.pt',dict(weights={spec.weight_name:endpoint},cache_c=end_m,
                    metadata=dict(batch=batch,seen_ids=[r['case_id'] for r in records[:batch*100]],base_model_revision=lock['model_revision'],
                    contexts=contexts,rng=rng,covariance={},state=dict(weights={spec.weight_name:tensor_sha(endpoint)},cache=tensor_sha(end_m)),
                    method='AlphaEdit_BLUE_SINGLETON',original_equivalence='UNVERIFIED')))
                save(broot/'native-observation.json',dict(receipt=result['receipt'],observer=result['observer'],raw=raw_ref,endpoint=snapshot,seconds=elapsed))
                # Selected W/M rollback is real, then reinstalls the exact native
                # endpoint. No extra native application or history append.
                with torch.no_grad():weight.copy_(before.to(weight));history.copy_(before_m)
                assert tensor_sha(weight)==tensor_sha(before) and tensor_sha(history)==tensor_sha(before_m)
                with torch.no_grad():weight.copy_(endpoint.to(weight));history.copy_(end_m)
                assert tensor_sha(weight)==result['receipt']['endpoint_sha256'] and tensor_sha(history)==result['receipt']['history_sha256']
                if batch==lock['native_batches'][0]:
                    refs=torch.load(lock['companions']['native-targets.pt'],map_location='cpu',weights_only=False)
                    target_rows=[dict(index=i,exact=torch.equal(a,b),max_abs=float((a-b).abs().max())) for i,(a,b) in enumerate(zip(raw['targets'],refs['values']))]
                    assert len(raw['targets'])==len(refs['values'])==100
                    save(broot/'target-reproduction.json',dict(count=100,rows=target_rows,source_exact_equivalence='NOT_ASSUMED'))
                    query_initial=None
                    if lock.get('observations'):
                        stage='E1_NATIVE_FACTOR_DIAGNOSTIC';begin=time.monotonic()
                        from .geometry import native_write_diagnostics
                        K=raw['solves'][0]['K'];R=raw['solves'][0]['R'];D=raw['physical_updates'][0]
                        diagnostic=native_write_diagnostics(K.to(weight.device),R.to(weight.device),
                            projector[0].to(weight.device),before_m[0].to(weight.device),D.to(weight.device),
                            ridge=1.,weight_before=before.to(weight.device),return_factor=True)
                        F=diagnostic.pop('diagnostic_factor_F').cpu()
                        factor=tensor_artifact(broot/'diagnostic-factor.pt',dict(F=F,R=R,K=K,role='DIAGNOSTIC_ONLY_NOT_NATIVE_WRITE'))
                        save(broot/'native-write-diagnostic.json',dict(values=diagnostic,factor=factor,seconds=time.monotonic()-begin))
                        realised=(endpoint.double()-before.double()).float()
                        from .query_initial import validate as query_gate
                        query_initial=query_gate(model,evaltok,weight,hp.rewrite_module_tmp.format(spec.layer),before,
                            rows[0],json.loads(Path(lock['general_manifest']['path']).read_text()),K,F,realised)
                        save(broot/'query-initial.json',query_initial)
                    signed_initial=None
                    if lock['diagnostics'].get('initial_probe_new_direction'):
                        stage='FIRST_NATIVE_SIGNED_PROBE'
                        from .initial_probe import probe
                        signed_initial=probe(model,evaltok,weight,hp.rewrite_module_tmp.format(spec.layer),
                            before,endpoint,rows[0],lock['diagnostics'])
                        save(broot/'signed-initial.json',signed_initial)
                        if signed_initial['finite_difference']['status'] in ('NONFINITE','DERIVATIVE_MISMATCH'):
                            raise ContractBoundary('SIGNED_DERIVATIVE_TECHNICAL_BOUNDARY',fd=signed_initial['finite_difference'])
                    save(root/'INITIAL_VALID.json',dict(status='INITIAL_VALID',entry_checkpoint_pointer_bytes_rng=True,
                        first_batch=batch,native_history_append=1,endpoint_finite=True,selected_rollback_reinstall_exact=True,
                        nonselected_native_bytes_version_exact=True,prior_signed_gate=lock['prior_gate'],repeated_FD_count=0,
                        new_direction_signed_status=None if signed_initial is None else signed_initial['finite_difference']['status'],
                        query_general_initial_status=None if query_initial is None else query_initial['status'],
                        continuation_complete=False,source_equivalence='NOT_YET_VERIFIED',elapsed_seconds=time.monotonic()-started,
                        after_initial='MONITORING_PAUSED_AWAITING_USER'))
                    print('E01_WARM_INITIAL_VALID',batch,flush=True)
                    stage='E1_FIRST_BATCH_PANELS';eval_start=time.monotonic()
                    historical=[records[i] for i in historical_ordinals(entry_n)]
                    # Same panels at exact W0, saved entry and new native endpoint.
                    # No model/target reoptimization, teacher or writer feedback.
                    for label,view,cache in [('W0',w0,torch.zeros_like(history)),('ENTRY',entry_weight,cp['cache_c']),('NATIVE',endpoint,end_m)]:
                        with torch.no_grad():weight.copy_(view.to(weight));history.copy_(cache)
                        expected=(tensor_sha(weight),tensor_sha(history))
                        save(broot/(label+'-current.json'),evaluate_records(model,evaltok,rows))
                        save(broot/(label+'-historical.json'),evaluate_records(model,evaltok,historical))
                        assert expected==(tensor_sha(weight),tensor_sha(history))
                    with torch.no_grad():weight.copy_(endpoint.to(weight));history.copy_(end_m)
                    # Observers are deterministic, nevertheless reset to the exact
                    # captured native RNG to prevent observer consumption drift.
                    from .fixtures import restore_rng
                    restore_rng(rng)
                    save(broot/'E1-coverage.json',dict(current_requests=100,historical_requests=128,prompt_pairs_per_endpoint=2964,
                        general='SEPARATE_FOLLOWING_OBSERVATION_PHASE' if lock.get('observations') else 'NOT_YET_MEASURED',
                        all_position_exposure='SEPARATE_FOLLOWING_OBSERVATION_PHASE' if lock.get('observations') else 'CAPTURED_NATIVE_K_R_ONLY_PENDING_QUERY_OBSERVATION',
                        signed_backward='PRIORITY_FOLLOWING_PHASE' if lock.get('observations') else 'NOT_PRIORITY_CELL_PRIOR_GATE_REUSED',evaluation_seconds=time.monotonic()-eval_start))
                    if lock.get('observations'):
                        from .observation_panels import run_observations
                        stage='E1_QUERY_SIGNED_GENERAL'
                        observation=run_observations(model,evaltok,weight,hp.rewrite_module_tmp.format(spec.layer),records,
                            list(range(entry_n,entry_n+100)),historical_ordinals(entry_n),lock['general_manifest']['path'],
                            before,endpoint,realised,K,F,broot/'observations',lock['observations'],w0_weight=w0,diagnostic_R=R)
                        save(broot/'E1-observations-complete.json',dict(receipt=member(broot/'observations/observation-receipt.json'),
                            signed_initial=member(broot/'signed-initial.json'),native_inputs_changed=False))
                        if lock['observations'].get('full_geometry'):
                            stage='E1_CPU_FULL_GEOMETRY'
                            from .full_geometry import run as full_geometry
                            full_geometry(projector[0],before_m[0],K,realised,before,w0,lock['covariance']['path'],broot/'geometry')
                        restore_rng(rng)
                        assert tensor_sha(weight)==result['receipt']['endpoint_sha256'] and tensor_sha(history)==result['receipt']['history_sha256']
                        del K,R,D,F,realised,diagnostic,observation
                record=dict(batch=batch,requests=100,native_seconds=elapsed,endpoint=snapshot,history_append=1,status='FINITE_NATIVE_BATCH')
                save(broot/'commit.json',record);committed.append(record)
                print('E01_NATIVE_BATCH_COMMITTED',batch,flush=True)
                del raw,result,before,before_m,endpoint,end_m
            stage='COMPARISON_CHECKPOINT'
            comparison=lock.get('comparison_checkpoint')
            fidelity=dict(status='REFERENCE_NOT_YET_RECEIVED',source_equivalence=False)
            if comparison is not None:
                member(comparison['path'],expected=comparison['sha256'])
                ref=torch.load(comparison['path'],map_location='cpu',weights_only=False,mmap=True)
                dw=weight.detach().cpu().double()-ref['weights'][spec.weight_name].double()
                dm=history.double()-ref['cache_c'].double()
                fidelity=dict(status='EXACT' if not torch.count_nonzero(dw) and not torch.count_nonzero(dm) else 'NONEXACT_CAUSE_UNRESOLVED',
                    weight_max_abs=float(dw.abs().max()),weight_relative_norm=float(dw.norm()/ref['weights'][spec.weight_name].double().norm()),
                    history_max_abs=float(dm.abs().max()),history_relative_norm=float(dm.norm()/ref['cache_c'].double().norm()),
                    endpoint_weight_history_exact=bool(not torch.count_nonzero(dw) and not torch.count_nonzero(dm)),
                    source_equivalence=False,trajectory_equivalence='UNVERIFIED_REQUIRES_INTERMEDIATE_TARGET_CONTEXT_ORDER_RNG_COMPARISON',
                    reference=comparison)
            save(root/'resume_fidelity.json',fidelity)
        if forward_handle is not None:
            forward_handle.remove();forward_handle=None
        hooks_exact=all(dict(m._forward_hooks)==a and dict(m._forward_pre_hooks)==b and dict(m._backward_hooks)==c for m,a,b,c in hook_state)
        if not hooks_exact:raise ContractBoundary('FINAL_MODEL_HOOK_REGISTRY_RESTORE')
        save(root/'terminal.json',dict(status='NATIVE_WINDOW_FINITE_OBSERVED',batches=len(committed),requests=len(committed)*100,
            E1_reused_first_batch=True,E1_double_count=0,source_equivalence=fidelity,restore=tx.receipt,
            forward_counts=forward_counts,model_load_seconds=model_load_seconds,elapsed_seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            hook_registry_restore_exact=hooks_exact,
            general='OBSERVED_FIRST_BATCH' if lock.get('observations') else 'NOT_YET_MEASURED',
            full_E01_complete=False,scientific_promotion=False))
    except BaseException as exc:
        save(root/'failure.json',dict(stage=stage,error=repr(exc),receipt=getattr(exc,'receipt',{}),traceback=traceback.format_exc(),
            completed_batches=len(committed),restore=None if tx is None else tx.receipt,elapsed_seconds=time.monotonic()-started,
            original_equivalence_denominator=0,scientific_promotion=False))
        raise
    finally:
        if forward_handle is not None:forward_handle.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();execute(a.lock,a.output)
