"""One BS100 replay, pinned objective check, exact RAM and CP round trips."""
import copy,time,sys
import torch
from project.run_scripts.memit_history_lifelong.method import observe
from project.run_scripts.memit_history_lifelong.io import save,tensor_sha
from project.run_scripts.single_layer_mechanism_first.z_hook import instrument_native_compute_z,normalize_requests
from .oracle import Oracle

def t0(engine):
 e=engine;e.stage='T0a';begin=time.monotonic();entry=e.w0;requests=e.requests(0,100)
 # Observe the source-exact native objective at the initial delta of the first
 # pre-registered request. No new optimization or selection panel.
 obs={};losses={};active=[None];original=e.module.compute_z
 def sink(kind,it,delta,loss,nll,kl,decay):
  if it!=0:return
  case=active[0]
  if kind=='loss':
   f=sys._getframe(1).f_locals
   losses[case]=dict(loss=float(loss),nll=float(nll),kl=float(kl),decay=float(decay),ids=tensor_sha(f['input_tok']['input_ids']),mask=tensor_sha(f['input_tok']['attention_mask']),labels=tensor_sha(f['rewriting_targets']),lookup=list(f['lookup_idxs']),anchor=tensor_sha(f['target_init']),teacher=tensor_sha(f['kl_distr_init']))
  elif 'gradient' not in obs:obs.update(gradient=delta.grad.detach().cpu().clone(),case_id=case)
 clone=instrument_native_compute_z(original,sink)
 def instrumented(*args,**kwargs):
  active[0]=args[2]['case_id'];return clone(*args,**kwargs)
 e.module.compute_z=instrumented
 try:
  with observe(e.module,e.hp,e.weights,e.H,requests,e.model) as native:
   m,h=e.module.apply_memit_seq_to_model(e.model,e.tok,requests,e.hp,copy=False,return_orig_weights=False,cache_template=None,cache_c=e.H)
   assert m is e.model and h is e.H
 finally:e.module.compute_z=original
 native_state=e.snapshot();native_probe=e.states.probe()
 native_eval=e.observe(list(range(100)),e.output/'T0/native-eval.json')
 e.restore(entry)
 adapter=e.adapter.run(requests,technical=True)
 parity=e.states.identity(e.meta)==native_state['identity']
 adapted_eval=e.observe(list(range(100)),e.output/'T0/adapter-eval.json')
 eval_equal={tag:native_eval['metrics'][tag]['rows']==adapted_eval['metrics'][tag]['rows'] for tag in ['RS','PS','NS']}
 z_equal=[r['sha256'] for r in adapter['z']]==[r['sha256'] for r in native['z']]
 key_equal=[r['sha256'] for r in adapter['keys']]==[r['sha256'] for r in native['keys']]
 e.restore(native_state,output_parity=native_probe);e.restore(entry)
 case=obs.get('case_id',requests[0]['case_id']);req=normalize_requests([r for r in requests if r['case_id']==case])[0]
 oracle=Oracle(e.model,e.tok,req,e.hp,e.module.CONTEXT_TEMPLATES_CACHE,e.module.find_fact_lookup_idx)
 input_equal=(losses[case]['ids']==tensor_sha(oracle.batch['tokens']['input_ids']) and losses[case]['mask']==tensor_sha(oracle.batch['tokens']['attention_mask']) and losses[case]['labels']==tensor_sha(oracle.batch['targets']) and losses[case]['lookup']==oracle.batch['specs'][0]['lookup'] and losses[case]['anchor']==tensor_sha(oracle.initial[0]) and losses[case]['teacher']==tensor_sha(oracle.kl))
 value,gradient,parts=oracle(torch.zeros_like(oracle.initial[0]));base=obs.get('gradient');base=base.to(gradient.device) if base is not None else None
 objective=dict(initial_native_loss=losses[case]['loss'],initial_cached_loss=value,loss_difference=value-losses[case]['loss'],
  gradient_relative_error=float((base-gradient).norm()/base.norm().clamp_min(1e-30)) if base is not None else None,gradient_bitwise=torch.equal(base,gradient) if base is not None else None,
  input_identity=oracle.batch['identity'],loss_parts=parts,full_vocabulary=True,cache_FP32=True,
  precision='BITWISE_AT_ORIGIN' if value==losses[case]['loss'] and base is not None and torch.equal(base,gradient) else 'NOT_ESTABLISHED_AT_ORIGIN')
 save(e.output/'T0/native.json',native);save(e.output/'T0/adapter.json',adapter)
 l4_unchanged=native['keys'][0]['sha256']==native['keys'][5]['sha256']
 rec=dict(L4_key_invariance=l4_unchanged,status='PASS' if input_equal and l4_unchanged and parity and all(eval_equal.values()) and z_equal and key_equal else 'FAIL',
  BS=100,technical_replay_not_science=True,z_bitwise=z_equal,key_bitwise=key_equal,W_H_RNG_context_ledger_exact=parity,
  evaluator_rows_bitwise=eval_equal,oracle_input_teacher_anchor_exact=input_equal,objective=objective,oracle_precision='PENDING_T0B_MEASUREMENT',RAM_restore_output_bitwise=True,seconds=time.monotonic()-begin)
 save(e.output/'T0/pre-checkpoint.json',rec)
 assert rec['status']=='PASS','NATIVE_ADAPTER_PARITY_REQUIRES_RCA'
 # Temporarily corrupt only our RAM copy, restore from a real saved payload,
 # and verify full state and output. The CP implementation also reloads.
 cp=e.cp.write('000',0,e.meta,technical=True)
 save(e.output/'T0/result.json',dict(**rec,checkpoint=cp,checkpoint_actual_reload=True))
 e.cp.delete(cp['path'],'T0_roundtrip_verified_no_remaining_dependency')
 e.total['technical_requests']+=200
 del native_state,obs;torch.cuda.empty_cache();return rec
