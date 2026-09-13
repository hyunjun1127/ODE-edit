"""One frozen policy/process, B51..B60; native fitting math is imported unchanged.

The only persistent scientific state is this process's committed W/M/context/RNG.
Evaluation never chooses a policy. Failed batches roll back to their own entry.
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

POLICIES = {'N4': (1., None), 'RES8': (.75, 8), 'S875': (.875, None),
            'S75': (.75, None), 'FULL8': (1., 8), 'REFIT4': (.75, 4)}


def batch_bounds(batch):
    if type(batch) is not int or batch not in range(51, 61):
        raise ValueError('BATCH_OUTSIDE_SEALED_SUFFIX')
    return (batch - 1) * 100, batch * 100


def selected_layers(arm):
    return [4, 8] if POLICIES[arm][1] == 8 else [4]


def validate_batch_lock(records, batch, item):
    from project.run_scripts.baseline_mechanism_first.contracts import digest
    start, stop = batch_bounds(batch)
    rows = records[start:stop]
    ids = [r['case_id'] for r in rows]
    assert len(rows) == len(set(ids)) == 100, 'BATCH_CARDINALITY'
    assert item['batch'] == batch and item['case_ids'] == ids, 'BATCH_ID_ORDER'
    assert item['request_order_sha256'] == digest(ids), 'BATCH_ORDER_SHA'
    assert item['records_sha256'] == digest(rows), 'BATCH_REQUEST_TARGET_SHA'
    return rows


def assert_fit_history_unchanged(before, after):
    assert before['M4'] == after['M4'] and before['M8'] == after['M8'], 'FIT_APPENDED_HISTORY'
    assert before['P4'] == after['P4'] and before['P8'] == after['P8'], 'FIT_PROJECTOR_MUTATION'
    assert before['contexts'] == after['contexts'], 'FIT_CONTEXT_MUTATION'


class SequentialLedger:
    """Production transaction ledger, also exercised with CPU toy state."""
    def __init__(self, entry, layers):
        self.previous = copy.deepcopy(entry)
        self.layers = list(layers)
        self.next_batch = 51
        self.history_counts = {4:0,8:0}
        self.open_batch = None

    def begin(self, batch, state):
        assert self.open_batch is None and batch == self.next_batch, 'BATCH_TRANSACTION_ORDER'
        assert state == self.previous, 'PREVIOUS_COMMIT_NEXT_ENTRY_MISMATCH'
        self.open_batch = batch

    def commit(self, state, finalization):
        assert self.open_batch is not None, 'NO_OPEN_BATCH'
        assert [x['layer'] for x in finalization] == self.layers and all(x['history_append']==1 for x in finalization), 'FINALIZE_COUNTS'
        for layer in self.layers:
            self.history_counts[layer] += 1
        self.previous = copy.deepcopy(state)
        self.next_batch += 1
        self.open_batch = None


def evaluate_nonmutating(state, versions, evaluate):
    before, pointers = state(), versions()
    result = evaluate()
    assert state() == before and versions() == pointers, 'ENDPOINT_EVALUATION_MUTATION'
    return result


def restore_and_verify(restore, snapshot, state, expected):
    restore(snapshot)
    assert state() == expected, 'BATCH_ROLLBACK_BYTES'
    return True


def run(lock_path, output, arm):
    import sys
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng, tensor_sha
    from project.run_scripts.baseline_mechanism_first.contracts import digest
    from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
    from .fitting import NativeSingletonFitter, select_projector, materialize_alpha
    from .sequential_evaluation import evaluate_batch

    alpha, second = POLICIES[arm]
    layers = selected_layers(arm)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    lock = json.loads(Path(lock_path).read_text())
    started = time.monotonic()
    stage = 'VERIFY_FROZEN_SOURCE'
    model = None
    weights, w0, flags = {}, {}, {}
    process_rng = None
    rollback = None
    nonguard = None
    completed = []
    try:
        for member in lock['members']:
            path = Path(member['path'])
            assert path.stat().st_size == member['bytes'] and file_sha(path) == member['sha256'], ('SOURCE_ASSET_DRIFT', str(path))
        assert lock['seq_batches'] == list(range(51, 61)), 'SEQUENTIAL_SCHEDULE'
        if 'SLURM_ARRAY_TASK_ID' in os.environ:
            assert lock['array_mapping'][int(os.environ['SLURM_ARRAY_TASK_ID'])] == arm, 'ARRAY_ARM_MAPPING'
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
        assert set(prepared['weights']) == {4, 8}
        cp = torch.load(lock['entry_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
        assert cp['metadata']['batch'] == 50 and cp['metadata']['seen_ids'] == [r['case_id'] for r in records[:5000]]
        assert tensor_sha(cp['cache_c']) == tensor_sha(prepared['M4']), 'PREPARED_M4_ENTRY'
        del cp
        wiki_panel = json.loads(Path(lock['wiki_panel']).read_text())
        mmlu = json.loads(Path(lock['mmlu100']).read_text())
        dev_mmlu = [mmlu[i] for i in lock['mmlu_development_indices']]
        sys.path.insert(0, lock['blue_root'])
        os.chdir(lock['blue_root'])
        module = importlib.import_module('AlphaEdit.AlphaEdit_main')
        HP = importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
        hp4, hp8 = HP.from_json(lock['config4']), HP.from_json(lock['config8'])
        assert hp4.layers == [4] and hp8.layers == [8]
        assert {k:v for k,v in vars(hp4).items() if k != 'layers'} == {k:v for k,v in vars(hp8).items() if k != 'layers'}
        fullp = torch.load(lock['projector'], map_location='cpu', weights_only=True, mmap=True)
        P4, pmap4 = select_projector(fullp, 4)
        P8, pmap8 = select_projector(fullp, 8)
        del fullp
        assert prepared['metadata']['P4'] == pmap4 and prepared['metadata']['P8'] == pmap8
        M4, M8 = prepared['M4'].clone(), prepared['M8'].clone()
        stage = 'MODEL_LOAD'
        begin = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(lock['snapshot'], local_files_only=True, low_cpu_mem_usage=True, attn_implementation='eager').cuda().eval()
        model_seconds = time.monotonic() - begin
        tok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        tok.add_bos_token = False
        tok.pad_token_id = tok.eos_token_id
        etok = AutoTokenizer.from_pretrained(lock['snapshot'], local_files_only=True)
        etok.pad_token_id = etok.eos_token_id
        assert tok.padding_side == etok.padding_side == 'right'
        parameters = dict(model.named_parameters())
        assert all(p.dtype == torch.float32 and p.grad is None for p in parameters.values())
        weights = {l:parameters[f'model.layers.{l}.mlp.down_proj.weight'] for l in (4, 8)}
        w0 = {l:w.detach().cpu().clone() for l,w in weights.items()}
        flags = {k:p.requires_grad for k,p in parameters.items()}
        modes = [(m,m.training) for m in model.modules()]
        process_rng = capture_rng()
        assert tensor_sha(w0[8]) == tensor_sha(prepared['weights'][8]), 'PRETRAINED_L8_MISMATCH'
        nonselected = {k:(v,v.data_ptr(),v._version,tensor_sha(v)) for k,v in parameters.items() if k not in [f'model.layers.{l}.mlp.down_proj.weight' for l in (4,8)]}
        def nonguard(full=False):
            assert all(p.grad is None for p in parameters.values()), 'MODEL_PARAMETER_GRAD_MUTATION'
            for k,(obj,ptr,version,sha) in nonselected.items():
                assert parameters[k] is obj and obj.data_ptr() == ptr and obj._version == version, ('NONSELECTED_MUTATION',k)
                if full:
                    assert tensor_sha(obj) == sha, ('NONSELECTED_BYTES',k)
        def state():
            return dict(weights={str(l):tensor_sha(w) for l,w in weights.items()}, M4=tensor_sha(M4), M8=tensor_sha(M8), P4=tensor_sha(P4), P8=tensor_sha(P8), contexts=digest(module.CONTEXT_TEMPLATES_CACHE), rng=digest(capture_rng()))
        def snapshot():
            return dict(weights={l:w.detach().cpu().clone() for l,w in weights.items()}, M4=M4.clone(), M8=M8.clone(), rng=capture_rng(), contexts=copy.deepcopy(module.CONTEXT_TEMPLATES_CACHE))
        def restore(snap):
            with torch.no_grad():
                for l,w in weights.items():
                    w.copy_(snap['weights'][l].to(w.device))
                M4.copy_(snap['M4'])
                M8.copy_(snap['M8'])
            module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(snap['contexts'])
            restore_rng(snap['rng'])
            for k,p in parameters.items():
                assert p.grad is None, 'PARAMETER_GRAD_MUTATION'
                p.requires_grad_(flags[k])
            for m,mode in modes:
                m.train(mode)
            nonguard()
        module.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(prepared['contexts'])
        module.COV_CACHE = {}
        binding = bind_evaluation_sources(lock['historical_evaluator_root'], helper_root=lock['helper_scripts_root'])
        restore(prepared)
        common_entry = state()
        assert common_entry == lock['common_state'], 'ACTUAL_PREPARED_RESTORE_IDENTITY'
        fitter = NativeSingletonFitter(module, expected_source_sha256=lock['editor_sha256'], contexts=prepared['contexts'])
        save(root/'entry.json',dict(arm=arm,state=common_entry,prepared=prepared_ref,source_lock_sha256=file_sha(lock_path),P_mapping=[pmap4,pmap8],evaluation_binding=binding,M8_reconstructed_here=0,prior_M8_setup_cost_reused_not_charged=True,RNG_policy='Own sequential optimizer chronology; unlike static branch reset, first-fit RNG is not reset before partial second fit',guard_scope='Selected W/M/P/context/RNG bytes and all parameter pointer/version; module buffers not separately byte-hashed'))
        previous = common_entry
        ledger = SequentialLedger(common_entry,layers)
        history_counts = ledger.history_counts
        z_total = solve_total = 0
        for batch in lock['seq_batches']:
            stage = f'B{batch:03d}_ENTRY'
            bdir = root/f'B{batch:03d}'
            batch_entry = snapshot()
            entry_state = state()
            ledger.begin(batch,entry_state)
            rollback = lambda snap=batch_entry, expected=entry_state: restore_and_verify(restore,snap,state,expected)
            current = validate_batch_lock(records, batch, batch_locks[batch])
            save(bdir/'entry.json',dict(arm=arm,batch=batch,state=entry_state,previous_commit_exact=True,order=batch_locks[batch],history_counts=history_counts.copy()))
            if batch == 52:
                save(root/'INITIAL_VALID.json',dict(status='ACTUAL_B51_COMMIT_TO_B52_ENTRY_EXACT',arm=arm,completed_batches=[51],entry_state=entry_state,previous_commit=completed[-1],finite=True,history_between_fits=0,evaluation_nonmutation=True,core_complete=False,after_initial='MONITORING_PAUSED_AWAITING_USER'))
                print('LOWCOST_SEQ10_INITIAL_VALID',flush=True)
            requests = [r['requested_rewrite'] for r in current]
            stage = f'B{batch:03d}_FRESH_FIRST_FIT'
            begin = time.monotonic()
            first = fitter.fit(model,tok,hp4,M4,P4,requests,layer=4,capture=True)
            torch.cuda.synchronize()
            first_seconds = time.monotonic()-begin
            after_first = state()
            assert_fit_history_unchanged(entry_state,after_first)
            assert after_first['weights']['8'] == entry_state['weights']['8']
            save_tensor(bdir/'first-target-key-readout.pt',first['captures'])
            save(bdir/'first-fit.json',dict(first['receipt'],synchronized_wall_seconds=first_seconds,actual_entry_state=entry_state))
            z_total += first['receipt']['compute_z']
            solve_total += first['receipt']['solve']
            # Restore native temporary W, not RNG: native optimizer RNG chronology
            # remains this batch's own. Partial-state second fit is always fresh.
            with torch.no_grad():
                weights[4].copy_(batch_entry['weights'][4].to(weights[4].device))
            begin = time.monotonic()
            materialization = materialize_alpha(weights[4],batch_entry['weights'][4],first['weight'],alpha)
            torch.cuda.synchronize()
            materialization_seconds = time.monotonic()-begin
            partial = state()
            assert_fit_history_unchanged(entry_state,partial)
            second_receipt = None
            second_seconds = 0.
            if second is not None:
                stage = f'B{batch:03d}_FRESH_SECOND_FIT'
                begin = time.monotonic()
                fit = fitter.fit(model,tok,hp8 if second==8 else hp4,M8 if second==8 else M4,P8 if second==8 else P4,requests,layer=second,capture=True)
                torch.cuda.synchronize()
                second_seconds = time.monotonic()-begin
                second_receipt = dict(fit['receipt'],actual_partial_state=partial,synchronized_wall_seconds=second_seconds)
                save_tensor(bdir/'second-target-key-readout.pt',fit['captures'])
                save(bdir/'second-fit.json',second_receipt)
                z_total += fit['receipt']['compute_z']
                solve_total += fit['receipt']['solve']
                after_second = state()
                assert_fit_history_unchanged(entry_state,after_second)
                other_layer = '4' if second == 8 else '8'
                assert after_second['weights'][other_layer] == partial['weights'][other_layer], 'SECOND_FIT_OTHER_LAYER_MUTATION'
                del fit
            stage = f'B{batch:03d}_FINALIZE_ONCE'
            bindings = [(4,hp4,M4,P4)]+([(8,hp8,M8,P8)] if second==8 else [])
            before_finalize_weights = state()['weights']
            begin = time.monotonic()
            finalization = fitter.finalize(model,tok,requests,bindings)
            torch.cuda.synchronize()
            finalize_seconds = time.monotonic()-begin
            assert [x['layer'] for x in finalization] == layers and all(x['history_append']==1 for x in finalization)
            endpoint = state()
            assert endpoint['weights'] == before_finalize_weights, 'FINALIZATION_WEIGHT_MUTATION'
            if second != 8:
                assert endpoint['weights']['8'] == common_entry['weights']['8'] and endpoint['M8'] == common_entry['M8'], 'UNUSED_L8_STATE_MUTATION'
            nonguard()
            assert all(torch.isfinite(t).all().item() for t in [*weights.values(),M4,M8])
            increments = save_tensor(bdir/'actual-increments.pt',dict(deltas={l:weights[l].detach().cpu()-batch_entry['weights'][l] for l in layers},entry=entry_state,endpoint=endpoint,exact_replay='NOT_TESTED',metadata=dict(arm=arm,batch=batch)))
            cpref = None
            if batch in (51,55,60):
                saved = dict(weights={l:weights[l].detach().cpu().clone() for l in layers},M4=M4.clone(),rng=capture_rng(),contexts=copy.deepcopy(module.CONTEXT_TEMPLATES_CACHE))
                if second == 8:
                    saved['M8'] = M8.clone()
                saved['metadata'] = dict(arm=arm,batch=batch,next_batch=batch+1,seen_ids=[r['case_id'] for r in records[:batch*100]],selected_layers=layers,state=endpoint,history_counts={l:n+int(l in layers) for l,n in history_counts.items()},policy=dict(alpha=alpha,second_layer=second),sample_root=lock['sample_root'],base_model_revision=lock['model_revision'],source_lock_sha256=file_sha(lock_path),common_prepared=prepared_ref,unselected_L8_from_prepared=second!=8,P_mapping=[pmap4,pmap8],gpu_continuation_replay='NOT_TESTED')
                cpref = save_tensor(bdir/'checkpoint.pt',saved)
                del saved
                restored_cp = torch.load(cpref['path'], map_location='cpu', weights_only=True, mmap=True)
                assert set(restored_cp['weights']) == set(layers)
                assert all(tensor_sha(restored_cp['weights'][l]) == endpoint['weights'][str(l)] for l in layers)
                assert tensor_sha(restored_cp['M4']) == endpoint['M4']
                if second == 8:
                    assert tensor_sha(restored_cp['M8']) == endpoint['M8']
                else:
                    assert 'M8' not in restored_cp and restored_cp['metadata']['common_prepared'] == prepared_ref
                assert digest(restored_cp['rng']) == endpoint['rng'] and digest(restored_cp['contexts']) == endpoint['contexts']
                cpref['cpu_weights_only_reload'] = 'TENSOR_CONTEXT_RNG_SHA_PASS'
                cpref['gpu_continuation'] = 'NOT_TESTED'
                del restored_cp
            stage = f'B{batch:03d}_ENDPOINT_EVALUATION'
            before = state()
            begin = time.monotonic()
            evaluation = evaluate_nonmutating(state,lambda:{k:(p.data_ptr(),p._version) for k,p in parameters.items()},lambda:evaluate_batch(model,etok,records,batch,lock['historical_ordinals'],wiki_panel,dev_mmlu,before))
            evaluation_seconds = time.monotonic()-begin
            nonguard()
            evalref = save(bdir/'evaluation.json',dict(evaluation,evaluation_nonmutation=True,endpoint_state=before))
            ledger.commit(endpoint,finalization)
            commit = dict(arm=arm,batch=batch,entry=entry_state,endpoint=endpoint,materialization=materialization,partial_state=partial,first_fit=first['receipt'],second_fit=second_receipt,history=finalization,history_counts=history_counts.copy(),checkpoint=cpref,increments=increments,evaluation=evalref,evaluation_nonmutation=True,first_fit_seconds=first_seconds,second_fit_seconds=second_seconds,materialization_seconds=materialization_seconds,finalization_seconds=finalize_seconds,evaluation_seconds=evaluation_seconds,policy_instrumented_online_seconds=first_seconds+second_seconds+materialization_seconds+finalize_seconds,pure_writer_without_instrumentation_seconds='NOT_SEPARATED')
            completed.append(save(bdir/'commit.json',commit))
            previous = endpoint
            rollback = None
            del batch_entry,first,evaluation
        stage = 'TERMINAL'
        assert z_total == (2000 if second is not None else 1000) and solve_total == (20 if second is not None else 10)
        nonguard(full=True)
        save(root/'terminal.json',dict(status='TEN_SEQUENTIAL_BATCHES_COMPLETE',arm=arm,batches=lock['seq_batches'],commits=completed,request_z_total=z_total,fit_solve_total=solve_total,history_counts=history_counts,entry_state=common_entry,terminal_state=previous,source_lock_sha256=file_sha(lock_path),model_seconds=model_seconds,seconds=time.monotonic()-started,M8_reconstruct_executions=0,audit_executed=False,fullseen_requests=6000,suffix_unique_requests=1000,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),scientific_promotion=False))
    except BaseException as exc:
        trace = traceback.format_exc()
        rb = dict(status='NO_OPEN_BATCH')
        if rollback is not None:
            try:
                equal = rollback()
                assert equal, 'BATCH_ROLLBACK_BYTES'
                rb = dict(status='EXACT_FAILED_BATCH_ENTRY_W_M_CONTEXT_RNG_RESTORED')
            except BaseException as error:
                rb = dict(status='ROLLBACK_FAILURE',error=repr(error))
        save(root/'failure.json',dict(stage=stage,error=repr(exc),traceback=trace,seconds=time.monotonic()-started,completed_commits=completed,rollback=rb,scientific_rescue=0))
        raise
    finally:
        if model is not None and w0:
            with torch.no_grad():
                for layer,weight in weights.items():
                    weight.copy_(w0[layer].to(weight.device))
            if flags:
                for name,param in model.named_parameters():
                    param.requires_grad_(flags[name])
            if process_rng is not None:
                restore_rng(process_rng)
            ok = all(tensor_sha(weight)==tensor_sha(w0[layer]) for layer,weight in weights.items())
            if nonguard is not None:
                nonguard(full=True)
            save(root/'process-restore.json',dict(selected_W0_exact=ok,parameter_version_restore='NOT_CLAIMED_COPY_INCREMENTS',RNG_restored=process_rng is not None,method_state_process_discarded=True))
            assert ok, 'PROCESS_RESTORE_FAILURE'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--arm',required=True,choices=list(POLICIES))
    args = parser.parse_args()
    run(args.lock,args.output,args.arm)
