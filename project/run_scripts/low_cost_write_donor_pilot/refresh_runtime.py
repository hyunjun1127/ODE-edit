"""Independent L4 write-refresh Middle chain; immutable reference code reused.

Every chunk finishes all B100 targets at one W before one native batch write.
No checkpoint from another policy and no score enters the target/writer path.
"""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import time
import traceback

from .runtime import file_sha, save, save_tensor
from .sequential_runtime import (SequentialLedger, batch_bounds, validate_batch_lock,
                                 evaluate_nonmutating, restore_and_verify)

POLICIES = {
    'FROZEN2': dict(caps=[24, 0], gammas=[.75, 1.], frozen=True),
    'I2': dict(caps=[12, 12], gammas=[.75, 1.], frozen=False),
    'FROZEN4': dict(caps=[24, 0, 0, 0], gammas=[.75, .75, .75, 1.], frozen=True),
    'I4': dict(caps=[6, 6, 6, 6], gammas=[.75, .75, .75, 1.], frozen=False),
}


def project_l4_state(state):
    return dict(weights={'4': state['weights']['4']}, M4=state['M4'],
                P4=state['P4'], contexts=state['contexts'], rng=state['rng'])


def assert_no_inner_history(before, after):
    for name in ('M4', 'P4', 'contexts'):
        assert before[name] == after[name], ('INNER_STATE_MUTATION', name)


class BatchBarrier:
    """Reject interleaved B1 writes, duplicate targets, or an extra finalizer."""
    def __init__(self, requests, chunks):
        self.requests, self.chunks = requests, chunks
        self.chunk = 0
        self.ready = []
        self.writes = 0
        self.finalized = False

    def target(self, request_index):
        assert not self.finalized and request_index == len(self.ready), 'TARGET_ORDER_OR_DUPLICATE'
        assert request_index < self.requests and self.chunk < self.chunks
        self.ready.append(request_index)

    def write(self):
        assert self.ready == list(range(self.requests)), 'WRITE_BEFORE_ALL_B100_TARGETS'
        self.ready = []
        self.writes += 1
        self.chunk += 1

    def finalize(self):
        assert not self.finalized and not self.ready and self.writes == self.chunks, 'FINALIZE_BARRIER'
        self.finalized = True


def run(lock_path, output, policy):
    import sys
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng, tensor_sha
    from project.run_scripts.baseline_mechanism_first.contracts import digest
    from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
    from .fitting import select_projector, materialize_alpha
    from .target_stepper import NativeTargetStepper, snapshot_state
    from .write_refresh_policy import ExternalTargetSingletonFitter
    from .refresh_evaluation import evaluate_batch

    config = POLICIES[policy]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    lock_path = str(Path(lock_path).resolve())
    lock = json.loads(Path(lock_path).read_text())
    lock_sha = file_sha(lock_path)
    started = time.monotonic()
    stage, model, weight, w0 = 'VERIFY_SOURCE_ASSETS', None, None, None
    flags, modes, process_rng, nonguard, rollback = {}, [], None, None, None
    completed = []
    try:
        for member in lock['members']:
            path = Path(member['path'])
            assert path.stat().st_size == member['bytes'] and file_sha(path) == member['sha256'], ('SOURCE_ASSET_DRIFT', str(path))
        assert lock['seq_batches'] == list(range(51, 61))
        assert lock['array_mapping'] == list(POLICIES), 'REFRESH_POLICY_MAPPING'
        if 'SLURM_ARRAY_TASK_ID' in os.environ:
            assert lock['array_mapping'][int(os.environ['SLURM_ARRAY_TASK_ID'])] == policy
        assert transformers.__version__ == lock['transformers'] and torch.__version__ == lock['torch']
        assert os.environ.get('SLURMD_NODENAME', 'server4') == 'server4'
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = lock['tf32_matmul']
        torch.backends.cudnn.allow_tf32 = lock['tf32_cudnn']
        records = load_prefix(lock['dataset_root'], 10000)
        batch_locks = {item['batch']: item for item in lock['batch_locks']}
        for batch in lock['seq_batches']:
            validate_batch_lock(records, batch, batch_locks[batch])
        prepared_ref = lock['prepared']
        assert file_sha(prepared_ref['path']) == prepared_ref['sha256']
        assert Path(prepared_ref['path']).stat().st_size == prepared_ref['bytes']
        prepared = torch.load(prepared_ref['path'], map_location='cpu', weights_only=True, mmap=True)
        assert prepared['metadata']['entry_n'] == 5000 and prepared['metadata']['sample_root'] == lock['sample_root']
        cp = torch.load(lock['entry_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
        assert cp['metadata']['batch'] == 50 and cp['metadata']['seen_ids'] == [r['case_id'] for r in records[:5000]]
        assert tensor_sha(cp['cache_c']) == tensor_sha(prepared['M4']), 'PREPARED_M4_ENTRY'
        del cp
        wiki_panel = json.loads(Path(lock['wiki_panel']).read_text())
        mmlu = json.loads(Path(lock['mmlu100']).read_text())
        dev_mmlu = [mmlu[i] for i in lock['mmlu_development_indices']]
        del mmlu
        sys.path.insert(0, lock['blue_root'])
        os.chdir(lock['blue_root'])
        module = importlib.import_module('AlphaEdit.AlphaEdit_main')
        zmodule = importlib.import_module('AlphaEdit.compute_z')
        HP = importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
        hp = HP.from_json(lock['config4'])
        assert hp.layers == [4] and hp.blue and hp.L2 == 1 and hp.v_num_grad_steps == 25
        fullp = torch.load(lock['projector'], map_location='cpu', weights_only=True, mmap=True)
        P4, pmap = select_projector(fullp, 4)
        del fullp
        assert prepared['metadata']['P4'] == pmap
        M4 = prepared['M4'].clone()
        stage = 'MODEL_LOAD'
        begin = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(lock['snapshot'], local_files_only=True,
                    low_cpu_mem_usage=True, attn_implementation='eager').cuda().eval()
        model_seconds = time.monotonic() - begin
        tok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        tok.add_bos_token = False
        tok.pad_token_id = tok.eos_token_id
        etok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        etok.pad_token_id = etok.eos_token_id
        assert tok.padding_side == etok.padding_side == 'right'
        parameters = dict(model.named_parameters())
        assert all(p.dtype == torch.float32 and p.grad is None for p in parameters.values())
        name = hp.rewrite_module_tmp.format(4) + '.weight'
        weight = parameters[name]
        w0 = weight.detach().cpu().clone()
        flags = {k:p.requires_grad for k,p in parameters.items()}
        modes = [(m,m.training) for m in model.modules()]
        process_rng = capture_rng()
        nonselected = {k:(v,v.data_ptr(),v._version,tensor_sha(v)) for k,v in parameters.items() if k != name}
        hook_baseline = {id(m):(tuple(m._forward_hooks), tuple(m._forward_pre_hooks), tuple(m._backward_hooks)) for m in model.modules()}

        def nonguard(full=False):
            assert all(p.grad is None for p in parameters.values()), 'MODEL_PARAMETER_GRAD'
            assert all(m.training == mode for m,mode in modes), 'MODEL_MODE_MUTATION'
            assert hook_baseline == {id(m):(tuple(m._forward_hooks),tuple(m._forward_pre_hooks),tuple(m._backward_hooks)) for m in model.modules()}, 'HOOK_LEAK'
            for key,(obj,ptr,version,sha) in nonselected.items():
                assert parameters[key] is obj and obj.data_ptr() == ptr and obj._version == version, ('NONSELECTED_MUTATION',key)
                if full:
                    assert tensor_sha(obj) == sha, ('NONSELECTED_BYTES',key)

        def state():
            return dict(weights={'4':tensor_sha(weight)}, M4=tensor_sha(M4), P4=tensor_sha(P4),
                        contexts=digest(module.CONTEXT_TEMPLATES_CACHE), rng=digest(capture_rng()))

        def snapshot():
            return dict(weights={4:weight.detach().cpu().clone()}, M4=M4.clone(),
                        contexts=copy.deepcopy(module.CONTEXT_TEMPLATES_CACHE), rng=capture_rng())

        def restore(snap):
            with torch.no_grad():
                weight.copy_(snap['weights'][4].to(weight.device))
                M4.copy_(snap['M4'])
            module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(snap['contexts'])
            restore_rng(snap['rng'])
            for key,p in parameters.items():
                assert p.grad is None
                p.requires_grad_(flags[key])
            for m,mode in modes:
                m.train(mode)
            nonguard()

        module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(prepared['contexts'])
        module.COV_CACHE = {}
        binding = bind_evaluation_sources(lock['historical_evaluator_root'], helper_root=lock['helper_scripts_root'])
        restore(prepared)
        common = state()
        assert common == project_l4_state(lock['common_state']), 'COMMON_L4_CAPSULE_MISMATCH'
        # Native compute_z freezes these flags before keys/readout/solve. New
        # carried target calls preserve incoming flags, so establish that same
        # read-only model state explicitly (u alone remains differentiable).
        for param in parameters.values():
            param.requires_grad_(False)
        contexts = copy.deepcopy(prepared['contexts'])
        del prepared
        writer = ExternalTargetSingletonFitter(module, expected_source_sha256=lock['editor_sha256'], contexts=contexts)
        save(root/'entry.json', dict(policy=policy, state=common, prepared=prepared_ref,
            source_lock_sha256=lock_sha, P_mapping=pmap, evaluation_binding=binding,
            target_policy=config, M8_setup=0, selected_layers=[4],
            model_parameter_gradients='EXPLICITLY_FROZEN_AS_NATIVE_COMPUTE_Z',
            prior_setup_cost='REUSED_NOT_NEW_ALLOCATION', gpu_continuation='NOT_TESTED'))
        ledger = SequentialLedger(common, [4])
        solves = target_losses = adam_updates = target_chunks = 0
        for batch in lock['seq_batches']:
            stage = f'B{batch:03d}_ENTRY'
            bdir = root/f'B{batch:03d}'
            entry_snap, entry = snapshot(), state()
            ledger.begin(batch, entry)
            rollback = lambda snap=entry_snap, expected=entry: restore_and_verify(restore,snap,state,expected)
            current = validate_batch_lock(records, batch, batch_locks[batch])
            requests = [r['requested_rewrite'] for r in current]
            save(bdir/'entry.json', dict(policy=policy,batch=batch,state=entry,order=batch_locks[batch],history_counts=ledger.history_counts.copy(),previous_commit_exact=True))
            if batch == 52:
                save(root/'INITIAL_VALID.json', dict(status='ACTUAL_B51_COMMIT_TO_B52_ENTRY_EXACT',
                    policy=policy, prior_commit=completed[-1], state=entry, history_between_chunks=0,
                    evaluation_nonmutation=True, complete=False,
                    after_initial='CONTINUE_THIS_TASK_TO_MIDDLE_TERMINAL_USER_OVERRIDE'))
                print('REFRESH_INITIAL_VALID', policy, flush=True)
            barrier = BatchBarrier(100, len(config['caps']))
            stepper = NativeTargetStepper(model, tok, hp, 4, contexts, zmodule)
            request_states = []
            subwrites = []
            batch_target_seconds = batch_write_seconds = batch_materialize_seconds = batch_io_seconds = 0.
            batch_losses = batch_adam = 0
            for chunk, (cap, gamma) in enumerate(zip(config['caps'], config['gammas'])):
                stage = f'B{batch:03d}_CHUNK{chunk}_TARGETS'
                cdir = bdir/f'chunk-{chunk}'
                chunk_entry = state()
                chunk_weight = weight.detach().cpu().clone()
                target_versions = {key:(p.data_ptr(),p._version) for key,p in parameters.items()}
                targets, summaries, state_refs = [], [], []
                begin = time.monotonic()
                for index, request in enumerate(requests):
                    if chunk == 0:
                        request_states.append(stepper.create_state(request, index))
                    reqstate = request_states[index]
                    if chunk and config['frozen']:
                        target = reqstate.absolute_target().detach()
                        evidence = snapshot_state(reqstate)
                        summary = dict(actual_adam_updates=0, target_loss_evaluations=0,
                                       status='FROZEN_TARGET_REUSE_NO_TARGET_FORWARD', chunk=chunk)
                    else:
                        result = stepper.run_chunk(reqstate, cap, chunk)
                        target, evidence, summary = result['target'], result['evidence'], result['summary']
                        target_chunks += 1
                    targets.append(target.detach().clone())
                    summaries.append(summary)
                    batch_losses += int(summary['target_loss_evaluations'])
                    batch_adam += int(summary['actual_adam_updates'])
                    io = time.monotonic()
                    state_refs.append(save_tensor(cdir/f'request-{index:03d}.pt', dict(
                        evidence=evidence,summary=summary,case_id=current[index]['case_id'],
                        request_index=index,batch=batch,chunk=chunk,policy=policy,
                        input_state=chunk_entry,request_sha256=digest(request))))
                    batch_io_seconds += time.monotonic()-io
                    barrier.target(index)
                    del evidence, target
                torch.cuda.synchronize()
                batch_target_seconds += time.monotonic()-begin
                assert state() == chunk_entry, 'TARGET_MODEL_HISTORY_CONTEXT_RNG_MUTATION'
                assert target_versions == {key:(p.data_ptr(),p._version) for key,p in parameters.items()}, 'TARGET_CHANGED_MODEL_WEIGHTS'
                nonguard()
                stage = f'B{batch:03d}_CHUNK{chunk}_WRITE'
                bound = writer.bind_targets(requests, targets)
                barrier.write()
                begin = time.monotonic()
                fit = writer.fit_targets(model,tok,hp,M4,P4,requests,bound,layer=4,capture=True)
                torch.cuda.synchronize()
                write_seconds = time.monotonic()-begin
                batch_write_seconds += write_seconds
                assert_no_inner_history(entry,state())
                assert fit['receipt']['solve'] == 1 and fit['receipt']['compute_z'] == 0
                solves += 1
                begin = time.monotonic()
                materialization = materialize_alpha(weight,chunk_weight,fit['weight'],gamma)
                torch.cuda.synchronize()
                materialize_seconds = time.monotonic()-begin
                batch_materialize_seconds += materialize_seconds
                after = state()
                assert_no_inner_history(entry,after)
                nonguard()
                io = time.monotonic()
                actual_delta = weight.detach().cpu()-chunk_weight
                tensorref = save_tensor(cdir/'subwrite.pt', dict(
                    actual_delta=actual_delta,
                    native_candidate_delta=fit['weight']-chunk_weight,
                    captures=fit['captures'], current_y=fit['current_y'], residual=fit['residual'],
                    targets=torch.stack([t.detach().cpu() for t in targets]),
                    entry=chunk_entry,endpoint=after,policy=policy,batch=batch,chunk=chunk,
                    reconstruction='ENTRY_PLUS_RECORDED_DELTAS_NOT_GPU_REPLAY_TESTED'))
                batch_io_seconds += time.monotonic()-io
                receipt = dict(policy=policy,batch=batch,chunk=chunk,cap=cap,gamma=gamma,
                    request_chunk_states=state_refs,target_summaries=summaries,
                    entry=chunk_entry,endpoint=after,fit=fit['receipt'],
                    materialization=materialization,tensors=tensorref,
                    actual_subwrite_frobenius=float(actual_delta.double().norm()),
                    write_seconds=write_seconds,materialization_seconds=materialize_seconds,
                    inner_history_appends=0)
                subwrites.append(save(cdir/'receipt.json',receipt))
                del fit,targets,chunk_weight,summaries,state_refs,actual_delta
            del request_states
            stage = f'B{batch:03d}_FINALIZE'
            barrier.finalize()
            before_finalize = state()
            begin = time.monotonic()
            finalization = writer.finalize(model,tok,requests,[(4,hp,M4,P4)])
            torch.cuda.synchronize()
            finalize_seconds = time.monotonic()-begin
            endpoint = state()
            assert endpoint['weights'] == before_finalize['weights'] and endpoint['P4'] == entry['P4']
            assert endpoint['contexts'] == entry['contexts']
            assert [r['layer'] for r in finalization] == [4] and finalization[0]['history_append'] == 1
            assert all(torch.isfinite(value).all().item() for value in (weight,M4,P4))
            nonguard()
            cpref = None
            checkpoint_io_seconds = 0.
            if batch in (51,55,60):
                cpbegin = time.monotonic()
                saved = snapshot()
                saved['metadata'] = dict(policy=policy,batch=batch,next_batch=batch+1,
                    seen_ids=[r['case_id'] for r in records[:batch*100]],selected_layers=[4],
                    state=endpoint,history_appends=ledger.history_counts[4]+1,
                    sample_root=lock['sample_root'],source_lock_sha256=lock_sha,
                    model_revision=lock['model_revision'],common_prepared=prepared_ref,
                    P_mapping=pmap,policy_contract=config,gpu_continuation='NOT_TESTED')
                cpref = save_tensor(bdir/'checkpoint.pt',saved)
                del saved
                loaded = torch.load(cpref['path'],map_location='cpu',weights_only=True,mmap=True)
                assert set(loaded['weights']) == {4} and tensor_sha(loaded['weights'][4]) == endpoint['weights']['4']
                assert tensor_sha(loaded['M4']) == endpoint['M4']
                assert digest(loaded['contexts']) == endpoint['contexts'] and digest(loaded['rng']) == endpoint['rng']
                cpref['cpu_weights_only_reload'] = 'SELECTED_W_M_CONTEXT_RNG_SHA_PASS'
                cpref['gpu_continuation'] = 'NOT_TESTED'
                del loaded
                checkpoint_io_seconds = time.monotonic()-cpbegin
            stage = f'B{batch:03d}_EVALUATION'
            begin = time.monotonic()
            results = evaluate_nonmutating(state,
                lambda:{key:(p.data_ptr(),p._version,p.requires_grad) for key,p in parameters.items()},
                lambda:evaluate_batch(model,etok,records,batch,lock['historical_ordinals'],wiki_panel,dev_mmlu,endpoint))
            evaluation_seconds = time.monotonic()-begin
            nonguard()
            evalref = save(bdir/'evaluation.json',dict(results,evaluation_nonmutation=True,endpoint_state=endpoint))
            ledger.commit(endpoint,finalization)
            target_losses += batch_losses
            adam_updates += batch_adam
            commit = dict(policy=policy,batch=batch,entry=entry,endpoint=endpoint,
                subwrites=subwrites,history=finalization,history_counts=ledger.history_counts.copy(),
                checkpoint=cpref,evaluation=evalref,evaluation_nonmutation=True,
                target_loss_evaluations=batch_losses,actual_adam_updates=batch_adam,
                target_seconds_including_request_state_IO=batch_target_seconds,
                native_writer_seconds=batch_write_seconds,materialization_seconds=batch_materialize_seconds,
                history_seconds=finalize_seconds,evaluation_seconds=evaluation_seconds,
                checkpoint_save_reload_seconds=checkpoint_io_seconds,
                nested_request_subwrite_IO_seconds=batch_io_seconds,
                instrumented_online_seconds=batch_target_seconds+batch_write_seconds+batch_materialize_seconds+finalize_seconds,
                pure_writer_without_instrumentation_seconds='NOT_SEPARATED',
                IO_timer_note='Nested in target timer where per-request save; do not sum nested timers')
            completed.append(save(bdir/'commit.json',commit))
            rollback = None
            del entry_snap,results
        stage = 'TERMINAL'
        assert solves == 10*len(config['caps']) and ledger.history_counts[4] == 10
        assert target_chunks == (1000 if config['frozen'] else 1000*len(config['caps']))
        assert adam_updates <= 24000 and target_losses <= (25000 if config['frozen'] else 24000+1000*len(config['caps']))
        nonguard(full=True)
        save(root/'terminal.json',dict(status='TEN_SEQUENTIAL_BATCHES_COMPLETE',policy=policy,
            batches=lock['seq_batches'],commits=completed,fit_solve_total=solves,
            request_target_chunks=target_chunks,target_loss_evaluations=target_losses,
            actual_adam_updates=adam_updates,history_counts=ledger.history_counts,
            entry_state=common,terminal_state=state(),source_lock_sha256=lock_sha,
            seconds=time.monotonic()-started,model_seconds=model_seconds,
            M8_setup=0,fullseen_requests=6000,suffix_unique_requests=1000,
            audit_executed=False,mmlu68_executed=False,futureN_executed=False,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),scientific_promotion=False))
    except BaseException as exc:
        trace = traceback.format_exc()
        rb = dict(status='NO_OPEN_BATCH')
        if rollback is not None:
            try:
                rollback()
                rb = dict(status='EXACT_FAILED_BATCH_ENTRY_W_M_CONTEXT_RNG_RESTORED')
            except BaseException as error:
                rb = dict(status='ROLLBACK_FAILURE',error=repr(error))
        save(root/'failure.json',dict(stage=stage,error=repr(exc),traceback=trace,
            seconds=time.monotonic()-started,completed_commits=completed,rollback=rb,scientific_rescue=0))
        raise
    finally:
        if model is not None and w0 is not None:
            with torch.no_grad():
                weight.copy_(w0.to(weight.device))
            for key,param in model.named_parameters():
                param.requires_grad_(flags[key])
            for m,mode in modes:
                m.train(mode)
            if process_rng is not None:
                restore_rng(process_rng)
            ok = tensor_sha(weight) == tensor_sha(w0)
            if nonguard is not None:
                nonguard(full=True)
            save(root/'process-restore.json',dict(selected_W0_exact=ok,
                RNG_restored=process_rng is not None,method_state_process_discarded=True,
                parameter_version_restore='NOT_CLAIMED_COPY_INCREMENTS'))
            assert ok, 'PROCESS_RESTORE_FAILURE'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--policy',required=True,choices=list(POLICIES))
    args = parser.parse_args()
    run(args.lock,args.output,args.policy)
