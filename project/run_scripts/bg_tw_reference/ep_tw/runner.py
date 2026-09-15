"""EP-TW-1 only: cold W0/M0, native Vp anchor, persistent B1..B10.

Observers cannot enter policy arguments. No native baseline trajectory is run.
First episode contains bounded technical forwards, no extra target fitting.
All files are create-once. A failed open batch restores its own entry; already
committed batches are immutable. There is no automatic retry/resume/submission.
"""
import argparse
import copy
import importlib
import json
import os
import resource
from pathlib import Path
import shutil
import time
import traceback

from .control import identity, save, sha, verify_dispatch

def save_tensor(path,payload):
    import torch
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    return identity(p)

def validate_batch(records,batch,item):
    from scripts.fixed_counterfact import encoded,sha as digest
    cur=records[(batch-1)*100:batch*100]
    assert len(cur)==100 and item['batch']==batch and item['ordinals']==[(batch-1)*100,batch*100]
    assert [r['case_id'] for r in cur]==item['case_ids']
    assert digest(encoded(cur))==item['record_sha256']
    assert digest(encoded([r['requested_rewrite'] for r in cur]))==item['request_target_sha256']
    assert item['inventory']=={'RS':100,'PS':200,'NS':1000}
    return cur

class Chronology:
    def __init__(self,entry):
        self.previous=copy.deepcopy(entry);self.next_batch=1;self.history_count=0;self.open=False
    def begin(self,batch,state):
        assert not self.open and batch==self.next_batch and state==self.previous,'ENTRY_CONTINUATION_IDENTITY'
        self.open=True
    def commit(self,state,history):
        assert self.open and len(history)==1 and history[0]['layer']==4 and history[0]['history_append']==1
        self.previous=copy.deepcopy(state);self.history_count+=1;self.next_batch+=1;self.open=False

def run(lock_path,output):
    import sys
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng,restore_rng,tensor_sha
    from project.run_scripts.baseline_mechanism_first.contracts import digest
    from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
    from project.run_scripts.low_cost_write_donor_pilot.fitting import NativeSingletonFitter,select_projector
    from project.run_scripts.low_cost_write_donor_pilot.evaluation import counterfact
    from project.run_scripts.baseline_mechanism_first.performance_schema import subset
    from project.run_scripts.bg_tw_reference.native_map import FrozenNativeMap
    from .model_adapter import EpisodeAdapter,TeacherStore,capture_native_fit,ModelBoundary
    from .technical import self_kl_check,validate_episode
    from .policy import NumericalPolicy,build_correction,materialize_candidates,CandidateObservation,choose_candidate
    from .ledger import AcceptedLedger
    root=Path(output);root.mkdir(parents=True,exist_ok=False,mode=0o700)
    started=time.monotonic();stage='SOURCE_INPUT_VERIFY';completed=[];rollback=None;model=None
    try:
        lock=json.loads(Path(lock_path).read_text())
        repair_prior=None
        if 'repair_pass_path' in lock:
            from .repair_runtime import verify_members,validate_repair_pass,recorder,scientific_checks
            lock['_lock_path']=str(Path(lock_path).resolve())
            repair_prior,repair_reference=validate_repair_pass(lock)
            verification=verify_members(lock)
            save(root/'conditional-admission.json',dict(status='TECHNICAL_PASS_EXACT_LOCK_VERIFIED',
                technical=repair_reference,verification=verification,fresh_W0_science=True,
                failed_old_B1_not_resumed=True))
        verify_dispatch(lock['dispatch']['path']);assert lock['policy']=='EP-TW-1' and lock['new_scientific_chains']==1
        assert lock['warm_state_imports']==lock['M8_imports']==lock['baseline_reruns']==0 and lock['N4_calibration'] is False
        assert os.environ.get('SLURMD_NODENAME')=='server4'
        assert torch.__version__==lock['torch'] and transformers.__version__==lock['transformers']
        if repair_prior is None:
            for m in lock['members']:
                assert Path(m['path']).stat().st_size==m['bytes'] and sha(m['path'])==m['sha256'],('FROZEN_MEMBER_DRIFT',m['path'])
        assert identity(lock['source_archive']['path'])==lock['source_archive']
        records=load_prefix(lock['dataset_root'],1000)
        assert len(records)==len({r['case_id'] for r in records})==1000
        for b,item in enumerate(lock['batches'],1):validate_batch(records,b,item)
        torch.set_num_threads(8);transformers.set_seed(lock['seed'])
        torch.backends.cuda.matmul.allow_tf32=lock['tf32_matmul'];torch.backends.cudnn.allow_tf32=lock['tf32_cudnn']
        sys.path.insert(0,lock['blue_root']);os.chdir(lock['blue_root'])
        module=importlib.import_module('AlphaEdit.AlphaEdit_main')
        HP=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
        hp=HP.from_json(lock['config4']);assert hp.layers==[4] and hp.blue and hp.L2==1 and hp.v_num_grad_steps==25
        context=json.loads(Path(lock['contexts']).read_text());module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(context);module.COV_CACHE={}
        fullp=torch.load(lock['projector'],map_location='cpu',weights_only=True,mmap=True)
        P,pmap=select_projector(fullp,4);del fullp
        assert pmap==lock['projector_mapping'],'CPU_RUNTIME_PHYSICAL_PROJECTOR_BINDING'
        M=torch.zeros_like(P);assert pmap['source_index']==pmap['local_index']==0
        stage='MODEL_W0_LOAD';begin=time.monotonic()
        model=AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,
            torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
        assert model.config.vocab_size==128256
        writer_tok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        writer_tok.add_bos_token=False;writer_tok.pad_token_id=writer_tok.eos_token_id
        etok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);etok.pad_token_id=etok.eos_token_id
        assert writer_tok.padding_side==etok.padding_side=='right'
        params=dict(model.named_parameters());name='model.layers.4.mlp.down_proj.weight';weight=params[name]
        assert all(p.dtype==torch.float32 and p.grad is None for p in params.values())
        for p in params.values():p.requires_grad_(False)
        w0=weight.detach().cpu().clone();model_seconds=time.monotonic()-begin
        baseline_nonselected={k:(p,p.data_ptr(),p._version,tensor_sha(p)) for k,p in params.items() if k!=name}
        buffers={k:(b,b.data_ptr(),b._version,tensor_sha(b)) for k,b in model.named_buffers()}
        hooks={k:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks),tuple(m._backward_hooks)) for k,m in model.named_modules()}
        def nonguard(full=False):
            assert not model.training and all(not p.requires_grad and p.grad is None for p in params.values())
            assert dict(model.named_parameters())[name] is weight,'SELECTED_PARAMETER_OBJECT'
            for k,(p,ptr,v,h) in baseline_nonselected.items():
                assert dict(model.named_parameters())[k] is p and p.data_ptr()==ptr and p._version==v,('NON_L4_MUTATION',k)
                if full:assert tensor_sha(p)==h
            for k,(b,ptr,v,h) in buffers.items():
                assert b.data_ptr()==ptr and b._version==v,('BUFFER_MUTATION',k)
                if full:assert tensor_sha(b)==h
            assert all(not m.training and hooks[k]==(tuple(m._forward_hooks),tuple(m._forward_pre_hooks),tuple(m._backward_hooks)) for k,m in model.named_modules())
            assert not module.COV_CACHE and module.CONTEXT_TEMPLATES_CACHE==context
        ledger=AcceptedLedger()
        def state():
            return dict(W4=tensor_sha(weight),M4=tensor_sha(M),P4=tensor_sha(P),contexts=digest(context),
                rng=digest(capture_rng()),ledger=ledger.state_hash())
        def snapshot():
            return dict(weight=weight.detach().cpu().clone(),M=M.clone(),rng=capture_rng(),ledger=ledger.to_dict())
        def restore(s):
            nonlocal ledger
            with torch.no_grad():weight.copy_(s['weight'].to(weight.device));M.copy_(s['M'])
            restore_rng(s['rng']);ledger=AcceptedLedger.from_dict(s['ledger']);nonguard()
        def observe(fn):
            before=state();ptr=weight.data_ptr();version=weight._version
            result=fn()
            assert before==state() and weight.data_ptr()==ptr and weight._version==version,'OBSERVER_MUTATION'
            nonguard();return result
        bind=bind_evaluation_sources(lock['historical_evaluator_root'],helper_root=lock['helper_scripts_root'])
        teacher=TeacherStore(lock['reference_root'],lock['teacher_manifest']['path'],
            expected_manifest_sha=lock['teacher_manifest']['sha256'],verify_payload_hashes=False)
        adapter=EpisodeAdapter(model,etok,name,teacher)
        stage='W0_TEACHER_TECHNICAL'
        selfkl=observe(lambda:self_kl_check(adapter,numerics=lock['technical_numerics']))
        selfref=save(root/'technical/w0-teacher.json',selfkl)
        common=state();chronology=Chronology(common)
        entryref=save(root/'entry.json',dict(state=common,initial='PRETRAINED_W0_COLD_M0',M_zero=bool((M==0).all()),
            seed=lock['seed'],contexts=identity(lock['contexts']),source_lock=identity(lock_path),model_revision=lock['model_revision'],
            teacher=teacher.receipt,canonical_binding=bind,P_mapping=pmap,technical=selfref,model_seconds=model_seconds,
            base_W4_sha256=tensor_sha(w0),nonselected_parameter_hashes={k:h for k,(_,_,_,h) in baseline_nonselected.items()}))
        fitter=NativeSingletonFitter(module,expected_source_sha256=lock['editor_sha256'],contexts=context)
        config=NumericalPolicy(**lock['numerical_policy']);technical_ref=None
        for batch,item in enumerate(lock['batches'],1):
            stage=f'B{batch:03d}_ENTRY';bdir=root/f'B{batch:03d}'
            assert shutil.disk_usage(root).free>5*(1<<30),'OUTPUT_SPACE_TECHNICAL_HOLD'
            cur=validate_batch(records,batch,item);entry=state();chronology.begin(batch,entry)
            snap=snapshot();rollback=(snap,entry)
            save(bdir/'entry.json',dict(batch=batch,state=entry,order=item,prior_endpoint_exact=True,history_count=chronology.history_count))
            if batch==2:
                assert chronology.history_count==1 and ledger.next_ordinal==100 and technical_ref
                save(root/'G0_PASS.json',dict(status='G0_PASS',completed_requests=100,next_ordinal=100,history_finalizations=1,
                    first_commit=completed[0],B2_entry=identity(bdir/'entry.json'),state=entry,teacher=selfref,model_technical=technical_ref,
                    finite=True,original_denominator=100,actual_native_candidate_and_selected_verified=True,
                    durable_restore_cpu_and_live_selected_state=True,persistent_B2_B10_programmed=True,
                    not_full1000_complete=True,agent_next_state='WAITING_USER_RESUME'))
                print('EP_TW1_G0_PASS',flush=True)
            old_ids=set(ledger.accepted_ids)
            old_records=[r for r in records[:(batch-1)*100] if r['case_id'] in old_ids]
            def panel_observation(label):
                out={'current':observe(lambda:counterfact(model,etok,cur,panel=label+'_CURRENT'))}
                out['accepted_old']=observe(lambda:counterfact(model,etok,old_records,panel=label+'_ACCEPTED_OLD')) if old_records else None
                out['accepted_old_annotations']=[r for r in ledger.annotated_accepted() if r['case_id'] in old_ids]
                out['endpoint_state']=state();out['controller_access']=False
                return out
            stage=f'B{batch:03d}_ENTRY_OBSERVATION';entry_eval=save(bdir/'entry-evaluation.json',panel_observation('ENTRY'))
            stage=f'B{batch:03d}_NATIVE_FIT';begin=time.monotonic()
            requests=[dict(copy.deepcopy(r['requested_rewrite']),case_id=r['case_id']) for r in cur]
            fit=capture_native_fit(fitter,model,writer_tok,hp,M,P,requests,layer=4)
            torch.cuda.synchronize();fit_seconds=time.monotonic()-begin
            assert state()['M4']==entry['M4'] and state()['P4']==entry['P4'] and ledger.state_hash()==entry['ledger']
            nonguard();Vp=weight.detach().clone();We=snap['weight'].to(weight.device)
            Z=fit['target'].to(weight.device);K=fit['captures']['compute_ks'][0].T.to(weight.device)
            H=fit['captures']['get_module_input_output_at_words'][0].T.to(weight.device)
            stage=f'B{batch:03d}_FIXED_MAP';map_begin=time.monotonic()
            native_map=FrozenNativeMap.from_frozen_kpm(K,P[0].to(weight.device),M[0].to(weight.device),
                request_count=100,weight_shape=tuple(weight.shape),source_identity=lock['editor_sha256'])
            map_receipt=native_map.comparison(Z-H,We,captured_native_endpoint=Vp)
            assert map_receipt['rhs_captured_endpoint_exact'],'DIRECT_NATIVE_CAPTURE_MISMATCH'
            map_seconds=time.monotonic()-map_begin
            adapter.set_episode(Vp,native_map.A)
            save_tensor(bdir/'native-targets-map.pt',dict(captures=fit['captures'],target_observations=fit['target_observations'],
                anchors=fit['anchors'],radii=fit['radii'],A=native_map.A.cpu(),native_proposal=fit['weight']))
            save(bdir/'native-fit.json',dict(fit['receipt'],synchronized_seconds=fit_seconds,map=map_receipt,map_seconds=map_seconds))
            early=recorder(bdir/'early-diagnostics') if repair_prior is not None else None
            if early is not None:
                early('postfit-entry',dict(state=state(),rng=capture_rng(),native_targets=identity(bdir/'native-targets-map.pt'),
                    method_history_appends=0,committed=False))
            stage=f'B{batch:03d}_GRADIENTS';sweeps=observe(lambda:adapter.gradient_sweeps(cur,recorder=early))
            if batch==1:
                stage='B001_MODEL_TECHNICAL';begin=time.monotonic()
                if repair_prior is None:
                    tech=validate_episode(adapter,cur,sweeps,numerics=lock['technical_numerics'])
                else:
                    truth=copy.deepcopy(cur)
                    for row in truth:row['requested_rewrite']['target_new']=row['requested_rewrite']['target_true']
                    truth_observation=observe(lambda:adapter.current(truth))
                    save(bdir/'early-diagnostics/current-true-margin.json',dict(observation=truth_observation,
                        margins=[t['nll']-n['nll'] for t,n in zip(truth_observation['rows'],sweeps['current']['rows'])],
                        controller_access=False,technical_extra_forwards=7))
                    tech=scientific_checks(adapter,cur,sweeps,lock,root/'technical/fresh-episode-checks',repair_prior)
                nonguard();assert tensor_sha(weight)==tensor_sha(Vp) and state()['M4']==entry['M4']
                tech['seconds_total']=time.monotonic()-begin
                technical_ref=save(root/'technical/episode1.json',tech)
            stage=f'B{batch:03d}_CORRECTION'
            correction=build_correction(Vp,We,Z,fit['anchors'].to(weight.device),fit['radii'].to(weight.device),
                native_map.A,sweeps['gE'],sweeps['gD'],config)
            candidates=materialize_candidates(Vp,correction.C,native_map.A,
                native_delta_norm=correction.diagnostics['actual_native_delta_norm'],config=config)
            raw_panel=panel_observation('RAW')
            raw_rows=raw_panel['current']['metrics']['RS']['rows']
            assert [r['new_nll'] for r in raw_rows]==[r['nll'] for r in sweeps['current']['rows']],'CANONICAL_CURRENT_NLL_PARITY'
            assert [r['case_id'] for r in raw_rows if r['new_strict']]==sweeps['current']['strict_ids'],'CANONICAL_CURRENT_STRICT_PARITY'
            rawref=save(bdir/'raw-evaluation.json',raw_panel)
            candidate_evals={'RAW':dict(current=sweeps['current'],generic=sweeps['generic'])}
            obs={'RAW':CandidateObservation(sweeps['current']['E'],frozenset(sweeps['current']['strict_ids']),sweeps['generic']['D'],tuple(r['case_id'] for r in cur))}
            stage=f'B{batch:03d}_CANDIDATE_SCREEN';screen_begin=time.monotonic()
            for c in candidates[1:]:
                if c.duplicate_of or not c.trust_valid:continue
                with torch.no_grad():weight.copy_(c.weight)
                assert state()['M4']==entry['M4'] and state()['ledger']==entry['ledger'];nonguard()
                try:
                    ccurrent=observe(lambda:adapter.current(cur));cgeneric=observe(lambda:adapter.generic('S64'))
                    candidate_evals[c.id]=dict(current=ccurrent,generic=cgeneric)
                    obs[c.id]=CandidateObservation(ccurrent['E'],frozenset(ccurrent['strict_ids']),cgeneric['D'],tuple(r['case_id'] for r in cur))
                except ModelBoundary as error:
                    if not str(error).startswith('NONFINITE_'):raise
                    c.trust_valid=False;c.exclusion_reason='NONFINITE_CANDIDATE_OBSERVATION'
                    candidate_evals[c.id]={'status':c.exclusion_reason,'error':str(error)}
                finally:
                    with torch.no_grad():weight.copy_(Vp)
                assert tensor_sha(weight)==tensor_sha(Vp)
            by_id={c.id:c for c in candidates}
            for c in candidates:
                if c.duplicate_of and not by_id[c.duplicate_of].trust_valid:
                    c.trust_valid=False;c.exclusion_reason='DUPLICATE_OF_INVALID_OBSERVATION_'+c.duplicate_of
            selection=choose_candidate(candidates,obs,config)
            selected=next(c for c in candidates if c.id==selection.selected_id)
            with torch.no_grad():weight.copy_(selected.weight)
            chosen=candidate_evals[selected.duplicate_of or selected.id]
            screen_seconds=time.monotonic()-screen_begin
            route_ref=save_tensor(bdir/'route.pt',dict(gE=sweeps['gE'].cpu(),gD=sweeps['gD'].cpu(),C=correction.C.cpu(),
                map_A=native_map.A.cpu(),raw_weight_sha256=tensor_sha(Vp),selected_weight_sha256=tensor_sha(weight)))
            save(bdir/'candidate-evaluation.json',candidate_evals)
            save(bdir/'policy.json',dict(correction=correction.diagnostics,status=correction.status,selection=selection.receipt(),
                sweeps=sweeps['receipt'],gradient_current=sweeps['current']['counts'],gradient_S64=sweeps['generic']['counts'],
                route=route_ref,screen_seconds=screen_seconds))
            # Always materialize selected endpoint before finalizer. No parent
            # fallback; low performance/zero correction never resets to entry.
            stage=f'B{batch:03d}_FINALIZE';before_W=tensor_sha(weight);begin=time.monotonic()
            finalization=fitter.finalize(model,writer_tok,requests,[(4,hp,M,P)])
            final_seconds=time.monotonic()-begin
            assert tensor_sha(weight)==before_W and finalization[0]['before_sha256']==fit['receipt']['history_sha256']
            assert finalization[0]['history_append']==1 and len(finalization)==1
            ledger_receipt=ledger.commit_batch(cur,chosen['current']['strict_ids'],
                {r['case_id']:r['nll'] for r in chosen['current']['rows']},batch_index=batch,endpoint_identity={'W4':before_W})
            endpoint=state();nonguard();assert torch.isfinite(weight).all() and torch.isfinite(M).all()
            stage=f'B{batch:03d}_SAVE_RESTORE'
            cp=save_tensor(bdir/'checkpoint.pt',dict(weights={name:weight.detach().cpu().clone()},M4=M.clone(),
                contexts=copy.deepcopy(context),rng=capture_rng(),accepted_ledger=ledger.to_dict(),
                metadata=dict(batch=batch,next_batch=batch+1,next_ordinal=batch*100,history_count=batch,endpoint=endpoint,
                    seen_ids=[r['case_id'] for r in records[:batch*100]],model_revision=lock['model_revision'],
                    source_lock=identity(lock_path),P_mapping=pmap,sample_lock=lock['sample_lock'],entry=entryref,
                    full_GPU_continuation_replay='NOT_TESTED')))
            loaded=torch.load(cp['path'],map_location='cpu',weights_only=True,mmap=True)
            assert set(loaded['weights'])=={name} and tensor_sha(loaded['weights'][name])==endpoint['W4'] and tensor_sha(loaded['M4'])==endpoint['M4']
            assert AcceptedLedger.from_dict(loaded['accepted_ledger']).state_hash()==ledger.state_hash()
            saved_selected=snapshot()
            # Actual selected-state restoration with saved bytes, no model
            # forward/re-edit/re-finalize. This verifies materialization scope.
            with torch.no_grad():weight.copy_(snap['weight'].to(weight.device))
            restore(dict(weight=loaded['weights'][name],M=loaded['M4'],rng=loaded['rng'],ledger=loaded['accepted_ledger']))
            assert state()==endpoint,'CHECKPOINT_SELECTED_STATE_RESTORE'
            del loaded,saved_selected
            stage=f'B{batch:03d}_SELECTED_OBSERVATION';eval_begin=time.monotonic()
            selected_panel=panel_observation('SELECTED')
            selected_rs=selected_panel['current']['metrics']['RS']['rows']
            assert [r['new_nll'] for r in selected_rs]==[r['nll'] for r in chosen['current']['rows']]
            evals=dict(selected=selected_panel,requested_annotations=ledger.annotate_requested())
            if batch in (5,10):
                seen=records[:batch*100]
                evals['seen_full']=observe(lambda:counterfact(model,etok,seen,panel='WHOLE_SEEN_ACTUAL_ENDPOINT'))
                evals['current_reuse']=subset(evals['seen_full'],len(seen)-100,len(seen),seen)
                evals['current_reuse_layout_note']='Same actual W; whole-prefix MB16 group boundaries can differ from own-B100. Both saved, not double-counted or claimed byte-parity.'
                evals['Dev128']=observe(lambda:adapter.generic('Dev128'))
                if batch==10:evals['first500']=subset(evals['seen_full'],0,500,seen)
            evalref=save(bdir/'selected-evaluation.json',dict(evals,endpoint_state=endpoint,evaluator_nonmutation=True))
            evaluation_seconds=time.monotonic()-eval_begin
            chronology.commit(endpoint,finalization)
            commit=save(bdir/'commit.json',dict(batch=batch,entry=entry,endpoint=endpoint,next_ordinal=batch*100,
                selected=selected.id,history=finalization,history_count=chronology.history_count,ledger=ledger_receipt,
                checkpoint=cp,checkpoint_CPU_load='ALL_SELECTED_W_M_RNG_LEDGER_VERIFIED',selected_live_restore=True,
                nonselected_guard='POINTER_VERSION_GRAD_MODE_HOOK_BUFFERS; base/final full SHA',
                evaluations={'entry':entry_eval,'raw':rawref,'selected':evalref},source_lock=identity(lock_path),
                native=fit['receipt'],map=map_receipt,gradient=sweeps['receipt'],
                cost={'native_instrumented':fit_seconds,'map_setup_comparison':map_seconds,'screen':screen_seconds,
                      'current_gradient':sweeps['current']['seconds'],'generic_gradient':sweeps['generic']['seconds'],
                      'history':final_seconds,'selected_evaluation':evaluation_seconds,
                      'pure_writer':'NOT_SEPARATED_FROM_NATIVE_INSTRUMENTATION'},finite=True,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
            completed.append(commit);rollback=None
            del snap,fit,sweeps,candidates,correction,native_map,Vp,We,K,H,Z,raw_panel,selected_panel,evals,candidate_evals
        stage='TERMINAL';nonguard(full=True)
        assert chronology.history_count==10 and ledger.next_ordinal==1000 and len(completed)==10
        save(root/'terminal.json',dict(status='EP_TW1_TEN_BATCHES_COMPLETE',completed_requests=1000,commits=completed,
            state=state(),history_count=10,source_lock=identity(lock_path),seconds=time.monotonic()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            teacher_setup_reused_not_new_cost=98,audit_report_mmlu='DEFERRED_NOT_EVALUATED',scientific_promotion=False))
    except BaseException as exc:
        trace=traceback.format_exc();restored='NO_OPEN_BATCH'
        if rollback is not None:
            try:
                restore(rollback[0]);assert state()==rollback[1];restored='FAILED_BATCH_ENTRY_W_M_RNG_LEDGER_EXACT'
            except BaseException as error:restored={'status':'RESTORE_FAILURE','error':repr(error)}
        save(root/'failure.json',dict(status='TECHNICAL_FAILURE',stage=stage,error=repr(exc),traceback=trace,
            completed_commits=completed,seconds=time.monotonic()-started,rollback=restored,
            automatic_retry=False,scientific_rescue=False))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--output',required=True)
    x=p.parse_args();run(x.lock,x.output)
