"""Actual 8B Llama correctness and two-process durable-resume gate.

This is NOT a scientific chain or quality-selection gate. Fixed calibration
probes set numerical tolerances before any Current RS/PS/NS observation.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import time
import types
import torch
from scripts.fixed_counterfact import load_prefix
from .config import validate_main
from .durable import CheckpointStore,prepare_state,tensor_sha256
from .llama_adapter import WEIGHT,model_guard,LlamaAffineOracle
from .native_binding import build_training_sequences,select_l4_projector,verify_native_source,pack_sequences
from .runtime import (load_model,prepare_batch,run_flow,save,tensor_save,capture_rng,
                      restore_rng,state_fingerprint,digest,file_sha,probe_next_entry,check_parity)
from .transaction import materialized_cost


def source_exact_key_probe(model,tok,records,contexts,source):
    """Execute only exact source key/lookup functions; compute_z is not compiled."""
    import numpy as np
    source=Path(source);verify_native_source(source)
    spec=importlib.util.spec_from_file_location('sl_zflow_verified_nethook',source/'util/nethook.py')
    hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)
    ns=dict(torch=torch,np=np,nethook=hook,GPTJForCausalLM=type('NotLlama',(),{}))
    selections={'rome/repr_tools.py':['get_words_idxs_in_templates','get_reprs_at_idxs','get_reprs_at_word_tokens'],
                'AlphaEdit/compute_z.py':['get_module_input_output_at_words'],
                'AlphaEdit/compute_ks.py':['compute_ks']}
    for name,wanted in selections.items():
        tree=ast.parse((source/name).read_text())
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
        assert len(funcs)==len(wanted)
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+funcs,type_ignores=[])),str(source/name),'exec'),ns)
        ns['repr_tools']=types.SimpleNamespace(**{k:v for k,v in ns.items() if k.startswith('get_')})
    requests=[dict(r['requested_rewrite'],case_id=r['case_id']) for r in records]
    hp=types.SimpleNamespace(rewrite_module_tmp='model.layers.{}.mlp.down_proj',fact_token='subject_last')
    with torch.no_grad():keys=ns['compute_ks'](model,tok,requests,hp,4,contexts).T.contiguous().cpu()
    return keys


def load_inputs(lock_path):
    lock=json.loads(Path(lock_path).read_text())
    # Full input verification is also invoked by launcher; direct module calls
    # retain the official dataset gate before any model load.
    records=load_prefix(lock['dataset_root'],1000)
    contexts=json.loads(Path(lock['contexts_path']).read_text())
    repo=Path(__file__).resolve().parents[3]
    contract=validate_main(json.loads((repo/'plans/global/2026-09-16-single-layer-zflow-contract-v1.json').read_text()))
    return lock,records,contexts,contract


def initial(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if (root/'technical-runtime.json').exists():raise FileExistsError('TECHNICAL_ATTEMPT_CREATE_ONCE')
    lock,records,contexts,contract=load_inputs(args.lock)
    plan=dict(calibration='first two request caches; X=0 and fixed seeded X~N(0,.001); no quality scores',
        validation='separate fixed seed X~N(0,.002); full-write X gradient and all-token',
        tolerance_rule='max(8*calibration_error, declared FP32 floor); held fixed before flow',
        fp32_floors=dict(max_logit_abs=256*torch.finfo(torch.float32).eps,
                        logit_relative_l2=64*torch.finfo(torch.float32).eps,
                        edit_abs_error=64*torch.finfo(torch.float32).eps,
                        kl_abs_error=64*torch.finfo(torch.float32).eps,
                        gradient_relative_l2=256*torch.finfo(torch.float32).eps),
        hard_signal_boundary=dict(gradient_relative_l2=.02,logit_relative_l2=.002),
        cost_tolerance_rule='max(8*calibration_relative_cost_error,256*FP32_eps)',
        scientific_request_denominator=0,technical_requests=100,seed=20260907)
    save(root/'preregistered-numerical-plan.json',plan)
    begin=time.perf_counter();model,tok,_,runtime=load_model(lock)
    save(root/'technical-runtime.json',runtime)
    stack=torch.load(lock['projector_path'],map_location='cpu',weights_only=True,mmap=True)
    p,p_binding=select_l4_projector(stack);p=p.clone();del stack
    history=torch.zeros_like(p);weight=dict(model.named_parameters())[WEIGHT]
    entry=weight.detach().cpu().clone();m_before=tensor_sha256(history)
    binding,k,b,s,geom,oracle,prep=prepare_batch(model,tok,records[:100],contexts,p,history,direct_test=True)
    save(root/'preparation.json',prep)
    if prep['geometry']['direct_rhs_relative_error']>.002:raise RuntimeError('DIRECT_FACTORED_ORIENTATION_FAIL')
    native_keys=source_exact_key_probe(model,tok,records[:2],contexts,lock['blue_source'])
    keyerr=float((native_keys-k[:,:2]).double().norm()/native_keys.double().norm())
    save(root/'native-key-parity.json',dict(relative_error=keyerr,threshold=.002,
            stock_function='exact AST compute_ks/repr_tools with original128 microbatch',
            passed=keyerr<=.002,source=verify_native_source(lock['blue_source']),P=p_binding))
    if keyerr>.002:raise RuntimeError('ACTUAL_NATIVE_KEY_PARITY_FAIL')
    indices=list(range(min(7,len(oracle.caches))))
    generator=torch.Generator().manual_seed(20260907)
    x=torch.randn(oracle.shape,generator=generator).cuda()*.001
    zero=oracle.parity(torch.zeros_like(x),backward=True,cache_indices=indices)
    probe=oracle.parity(x,backward=True,cache_indices=indices)
    tolerance={name:max(floor,8*max(zero.get(name,0),probe[name])) for name,floor in plan['fp32_floors'].items()}
    candidate=(entry.cuda()+x@b).cpu()
    calibration_cost_begin=time.perf_counter()
    actual=materialized_cost(candidate.double()-entry.double(),history,request_count=100)
    calibration_cost_seconds=time.perf_counter()-calibration_cost_begin
    cost_relative=abs(actual-geom.cost(x))/max(actual,1e-30)
    tolerance['cost_relative']=max(256*torch.finfo(torch.float32).eps,8*cost_relative)
    save(root/'calibration.json',dict(zero=zero,fixed_probe=probe,cost_relative_error=cost_relative,
        actual_cost_seconds=calibration_cost_seconds))
    if tolerance['gradient_relative_l2']>.02 or tolerance['logit_relative_l2']>.002:
        raise RuntimeError('CALIBRATION_NUMERICAL_SIGNAL_UNRESOLVED')
    save(root/'parity-tolerance.json',tolerance)
    # Disjoint fixed perturbation; tolerance cannot depend on this result.
    xv=torch.randn(oracle.shape,generator=generator).cuda()*.002
    validation=check_parity(oracle.parity(xv,backward=True,cache_indices=indices),tolerance,gradient=True)
    save(root/'actual-llama-parity.json',validation)
    if not validation['passed']:raise RuntimeError('LLAMA_AFFINE_GRADIENT_PARITY_FAIL')
    # Actual two-request logical objective, physical microbatch1 vs2. Preserve
    # all contexts/target tokens and the same full100-coordinate writer B.
    tiny=build_training_sequences(tok,[dict(r['requested_rewrite'],case_id=r['case_id']) for r in records[:2]],contexts)
    partitions=[]
    for mb in (1,2):
        packs=[pack_sequences(tiny.sequences[i:i+mb],tok.pad_token_id) for i in range(0,len(tiny.sequences),mb)]
        test_oracle=LlamaAffineOracle(model,b,packs)
        loss,grad=test_oracle(xv);partitions.append((loss,grad))
        del test_oracle
    partition_error=float((partitions[0][1]-partitions[1][1]).double().norm()/partitions[0][1].double().norm().clamp_min(1e-30))
    partition=dict(logical_requests=2,physical_microbatches=[1,2],global_weights_preserved=True,
        loss_abs_error=abs(partitions[0][0]-partitions[1][0]),gradient_relative_l2=partition_error,
        passed=partition_error<=tolerance['gradient_relative_l2'] and abs(partitions[0][0]-partitions[1][0])<=tolerance['edit_abs_error'])
    save(root/'actual-microbatch-parity.json',partition)
    if not partition['passed']:raise RuntimeError('ACTUAL_GLOBAL_WEIGHT_PARTITION_PARITY_FAIL')
    del partitions,tiny
    result,flow=run_flow(oracle,geom,contract)
    save(root/'technical-flow.json',flow)
    candidate=(entry.cuda()+result.x@b).cpu() if result.accepted_steps else entry.clone()
    physical=check_parity(oracle.physical_terminal_parity(result.x,candidate),tolerance)
    save(root/'physical-parity.json',physical)
    if not physical['passed']:raise RuntimeError('COMMIT_PARITY_FAIL')
    terminal_cost_begin=time.perf_counter()
    actual=materialized_cost(candidate.double()-entry.double(),history,request_count=100)
    terminal_cost_seconds=time.perf_counter()-terminal_cost_begin
    predicted=geom.cost(result.x);cost_error=abs(actual-predicted)/max(actual,predicted,1e-30)
    cost=dict(actual_cost=actual,predicted_cost=predicted,relative_error=cost_error,
              tolerance=tolerance['cost_relative'],passed=cost_error<=tolerance['cost_relative'],
              actual_cost_seconds=terminal_cost_seconds)
    save(root/'physical-cost.json',cost)
    if not cost['passed']:raise RuntimeError('COMMIT_COST_FAIL')
    if tensor_sha256(history)!=m_before:raise RuntimeError('INNER_HISTORY_MUTATION')
    state_begin=time.perf_counter()
    state=prepare_state(entry,history,candidate,result.x.cpu(),b.cpu(),s.cpu(),k,
        accepted=result.accepted_steps,request_count=100,parity_evidence=physical,cost_evidence=cost)
    state_seconds=time.perf_counter()-state_begin
    store=CheckpointStore(root/'checkpoints')
    source=dict(input_lock_sha256=file_sha(args.lock),source_head=lock['source_head'],source_tree=lock['source_tree'])
    rng=capture_rng();kwargs=dict(config=contract,source=source,context=contexts,rng=rng,
        ledger=flow,cache_resume_fingerprint=digest(binding.metadata))
    receipt=store.publish('B001',None,2,state,**kwargs)
    replay=store.publish('B001',None,2,state,**kwargs)
    if not replay['replayed'] or replay['receipt_sha256']!=receipt['receipt_sha256']:raise RuntimeError('RETRY_NOT_NOOP')
    with torch.no_grad():weight.copy_(state.tensors['W'].cuda())
    resumed=store.load('B001');restore_rng(resumed['rng'])
    if not torch.equal(resumed['tensors']['W'],weight.cpu()) or not torch.equal(resumed['tensors']['M'],state.tensors['M']):
        raise RuntimeError('LOCAL_CHECKPOINT_RESTORE_PARITY')
    next_logits,next_binding=probe_next_entry(model,tok,records[100],contexts)
    tensor_save(root/'next-entry-reference.pt',dict(logits=next_logits,binding=next_binding,
        W=tensor_sha256(weight),M=tensor_sha256(state.tensors['M'])))
    save(root/'phase1-complete.json',dict(status='TECHNICAL_PHASE1_VALID_NOT_RESUME_PASS',
        checkpoint=receipt,duplicate_noop=True,inner_history_append=0,
        peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved(),
        total_seconds=time.perf_counter()-begin,scientific_batches=0,technical_batches=1,
        source=source,remaining='separate process model reload + complete W/M/RNG continuation'))
    save(root/'commit-cost-ledger.json',dict(calibration_actual_cost_seconds=calibration_cost_seconds,
        terminal_actual_cost_seconds=terminal_cost_seconds,state_prepare_seconds=state_seconds,
        state_prepare_includes_independent_actual_cost=bool(result.accepted_steps),
        actual_cost_evaluations=2+int(bool(result.accepted_steps))))


def resumed(args):
    root=Path(args.output);lock,records,contexts,contract=load_inputs(args.lock)
    first=json.loads((root/'phase1-complete.json').read_text())
    begin=time.perf_counter();model,tok,_,runtime=load_model(lock)
    loaded=CheckpointStore(root/'checkpoints').load('B001')
    if loaded['metadata']['next_batch_index']!=2 or loaded['metadata']['context']!=contexts:
        raise RuntimeError('RESUME_CONTEXT_OR_NEXT_INDEX')
    source=dict(input_lock_sha256=file_sha(args.lock),source_head=lock['source_head'],source_tree=lock['source_tree'])
    if loaded['metadata']['source']!=source or loaded['metadata']['config']!=contract:
        raise RuntimeError('RESUME_SOURCE_OR_CONFIG_MISMATCH')
    original_binding=build_training_sequences(tok,
        [dict(r['requested_rewrite'],case_id=r['case_id']) for r in records[:100]],contexts)
    if loaded['metadata']['cache_resume_fingerprint']!=digest(original_binding.metadata):
        raise RuntimeError('RESUME_NATIVE_CACHE_BINDING_MISMATCH')
    weight=dict(model.named_parameters())[WEIGHT]
    before=model_guard(model)
    with torch.no_grad():weight.copy_(loaded['tensors']['W'].cuda())
    restore_rng(loaded['rng'])
    rng_expected=state_fingerprint(loaded['rng']);rng_actual=state_fingerprint(capture_rng())
    if rng_expected!=rng_actual:raise RuntimeError('RESUME_RNG_BYTES_MISMATCH')
    actual,binding=probe_next_entry(model,tok,records[100],contexts)
    reference=torch.load(root/'next-entry-reference.pt',map_location='cpu',weights_only=True)
    if (tensor_sha256(weight)!=reference['W'] or tensor_sha256(loaded['tensors']['M'])!=reference['M']
        or binding!=reference['binding']):raise RuntimeError('RESUME_STATE_BINDING_MISMATCH')
    tolerance=json.loads((root/'parity-tolerance.json').read_text())
    error=float((actual-reference['logits']).abs().max())
    if error>tolerance['max_logit_abs']:raise RuntimeError('RESUME_NEXT_BATCH_LOGIT_PARITY_FAIL')
    if tuple(x for x in before if x[0]!=WEIGHT)!=tuple(x for x in model_guard(model) if x[0]!=WEIGHT):
        raise RuntimeError('RESUME_NONSELECTED_MUTATION')
    save(root/'TECHNICAL_VALID.json',dict(status='ACTUAL_LLAMA_TECHNICAL_VALID',
        phase1_sha256=file_sha(root/'phase1-complete.json'),separate_process_model_reload=True,
        W_M_context_RNG_restored=True,next_batch_index=2,next_entry_logits_max_abs=error,
        rng_expected_sha256=rng_expected,rng_restored_sha256=rng_actual,
        source_config_exact=True,cache_resume_binding_exact=True,
        exact_same_logits=bool(torch.equal(actual,reference['logits'])),
        no_writer_z_or_history_on_resume=True,source_input_lock_sha256=file_sha(args.lock),
        total_technical_seconds=first['total_seconds']+time.perf_counter()-begin,
        phase2_seconds=time.perf_counter()-begin,runtime=runtime,
        main_initial_state='MUST_LOAD_FRESH_PRETRAINED_W0_COLD_M0',scientific_batches=0,
        tolerance_sha256=file_sha(root/'parity-tolerance.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['initial','resume'],required=True)
    p.add_argument('--lock',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    try:(initial if args.phase=='initial' else resumed)(args)
    except Exception as error:
        save(Path(args.output)/('FAILURE-'+args.phase+'.json'),dict(status='TECHNICAL_FAILURE',
             exception=type(error).__name__,message=str(error),scientific_batches=0))
        raise

if __name__=='__main__':main()
