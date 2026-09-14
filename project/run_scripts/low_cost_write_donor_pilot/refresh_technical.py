"""Bounded actual B100 native/I1 checks; never a policy quality gate.

Two fresh independent processes use identical W50/M50/context/RNG. The CPU
comparison is byte exact and cannot relax tolerances after observing results.
"""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types
import inspect

from .runtime import file_sha, save, save_tensor

def run(lock_path, output, mode):
 import torch
 import transformers
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from scripts.fixed_counterfact import load_prefix
 from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng,restore_rng,tensor_sha
 from project.run_scripts.baseline_mechanism_first.contracts import digest
 from .fitting import NativeSingletonFitter,select_projector,materialize_alpha
 from .target_stepper import NativeTargetStepper
 from .write_refresh_policy import ExternalTargetSingletonFitter
 out=Path(output);out.mkdir(parents=True,exist_ok=False)
 lock=json.loads(Path(lock_path).read_text());started=time.monotonic();stage='LOCK'
 model=None;w=None;original=None
 try:
  for m in lock['members']:
   assert Path(m['path']).stat().st_size==m['bytes'] and file_sha(m['path'])==m['sha256'],('INPUT_DRIFT',m['path'])
  assert torch.__version__==lock['torch'] and transformers.__version__==lock['transformers']
  torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=lock['tf32_matmul'];torch.backends.cudnn.allow_tf32=lock['tf32_cudnn']
  records=load_prefix(lock['dataset_root'],10000)
  requests=copy.deepcopy([r['requested_rewrite'] for r in records[5000:5100]])
  for r in requests:
   if not r['target_new']['str'].startswith(' '):r['target_new']['str']=' '+r['target_new']['str']
  prep=torch.load(lock['prepared']['path'],map_location='cpu',weights_only=True,mmap=True)
  sys.path.insert(0,lock['blue_root']);os.chdir(lock['blue_root'])
  module=importlib.import_module('AlphaEdit.AlphaEdit_main');zmodule=importlib.import_module('AlphaEdit.compute_z')
  hp=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams.from_json(lock['config4'])
  module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(prep['contexts']);module.COV_CACHE={}
  P,pmap=select_projector(torch.load(lock['projector'],map_location='cpu',weights_only=True,mmap=True),4)
  M=prep['M4'].clone();entryM=M.clone()
  stage='MODEL_LOAD';t=time.monotonic()
  model=AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
  model_seconds=time.monotonic()-t
  tok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
  params=dict(model.named_parameters());w=params['model.layers.4.mlp.down_proj.weight'];original=w.detach().cpu().clone()
  assert all(p.dtype==torch.float32 for p in params.values())
  flags={k:p.requires_grad for k,p in params.items()};initial_rng=capture_rng()
  with torch.no_grad():w.copy_(prep['weights'][4].to(w.device))
  restore_rng(prep['rng']);entryw=w.detach().cpu().clone()
  nonguard={k:(p.data_ptr(),p._version,tensor_sha(p)) for k,p in params.items() if p is not w}
  native_losses=[]
  class ObservedNativeFitter(NativeSingletonFitter):
   def _functions(self,counts,capture):
    fit,final=super()._functions(counts,capture)
    native=zmodule.compute_z
    def observed(*args,**kwargs):
     losses=[]
     def observation_print(*values,**kw):
      if values and isinstance(values[0],str) and values[0].startswith('loss '):
       loc=inspect.currentframe().f_back.f_locals
       losses.append(dict(iteration=int(loc['it']),total=float(loc['loss'].item()),
        nll=float(loc['nll_loss'].item()),kl=float(loc['kl_loss'].item()),
        regularizer=float(loc['weight_decay'].item())))
      print(*values,**kw)
     ns=dict(native.__globals__);ns['print']=observation_print
     fn=types.FunctionType(native.__code__,ns,native.__name__,native.__defaults__,native.__closure__)
     t=time.monotonic();result=fn(*args,**kwargs)
     counts['compute_z']=counts.get('compute_z',0)+1
     counts['compute_z_seconds']=counts.get('compute_z_seconds',0.)+time.monotonic()-t
     capture.setdefault('compute_z',[]).append(result.detach().cpu().clone())
     native_losses.append(losses)
     return result
    fit.__globals__['compute_z']=observed
    return fit,final
  fitter=ObservedNativeFitter(module,expected_source_sha256=lock['editor_sha256'],contexts=prep['contexts'])
  external=ExternalTargetSingletonFitter(module,expected_source_sha256=lock['editor_sha256'],contexts=prep['contexts'])
  stage='B100_TARGETS_AND_WRITE';t=time.monotonic()
  chunk_refs=[]
  if mode=='native':
   result=fitter.fit(model,tok,hp,M,P,requests,layer=4,capture=True)
   targets=result['captures']['compute_z']
  else:
   for p in model.parameters():p.requires_grad_(False)
   stepper=NativeTargetStepper(model,tok,hp,4,prep['contexts'],zmodule)
   targets=[]
   for i,request in enumerate(requests):
    state=stepper.create_state(request,request_index=i)
    chunk=stepper.run_chunk(state,max_updates=24,chunk_index=0)
    targets.append(chunk['target'].detach().cpu().clone())
    chunk_refs.append(save_tensor(out/'requests'/f'{i:03d}.pt',chunk['evidence']))
    del state,chunk
   for p in model.parameters():p.requires_grad_(False)
   result=external.fit_targets(model,tok,hp,M,P,requests,external.bind_targets(requests,targets),layer=4,capture=True)
  torch.cuda.synchronize();fit_seconds=time.monotonic()-t
  assert torch.equal(M,entryM),'INNER_HISTORY_MUTATION'
  nativew=w.detach().cpu().clone()
  materialize_alpha(w,entryw,nativew,.75)
  partial=w.detach().cpu().clone()
  assert torch.equal(partial,entryw+.75*(nativew-entryw)),'S75_ROUNDING'
  materialize_alpha(w,entryw,nativew,1.)
  finalization=fitter.finalize(model,tok,requests,[(4,hp,M,P)])
  assert len(finalization)==1 and finalization[0]['history_append']==1
  ref=save_tensor(out/'native-I1-comparison.pt',dict(targets=torch.stack(targets),weight=nativew,partial75=partial,history=M.clone(),entryW=entryw,entryM=entryM,context=prep['contexts'],entry_rng=prep['rng']))
  # Fixed-W split vs uninterrupted uses the same u/Adam state; this is a
  # separate technical observation, not an additional scientific batch.
  carry=None
  if mode=='I1':
   stage='FIXED_W_PAUSE_RESUME'
   with torch.no_grad():w.copy_(entryw.to(w.device));M.copy_(entryM)
   restore_rng(prep['rng'])
   whole=stepper.create_state(requests[0],request_index=0)
   one=stepper.run_chunk(whole,max_updates=24,chunk_index=0)
   restore_rng(prep['rng'])
   split=stepper.create_state(requests[0],request_index=0)
   first=stepper.run_chunk(split,max_updates=12,chunk_index=0)
   second=stepper.run_chunk(split,max_updates=12,chunk_index=1)
   carry=save_tensor(out/'fixed-weight-carry.pt',dict(whole=one['evidence'],first=first['evidence'],second=second['evidence']))
   assert torch.equal(one['target'],second['target']),'FIXED_W_TARGET_PAUSE_RESUME'
  for k,(ptr,ver,sha) in nonguard.items():
   p=params[k];assert p.data_ptr()==ptr and p._version==ver and tensor_sha(p)==sha,('NONSELECTED_MUTATION',k)
  save(out/'terminal.json',dict(status='TECHNICAL_B100_COMPLETE_PENDING_INDEPENDENT_COMPARISON',mode=mode,comparison=ref,native_loss_observation=native_losses,native_observer='same function code; private globals print reads detached scalar locals; original module unchanged',request_chunks=chunk_refs,fixed_weight_carry=carry,fit_receipt=result['receipt'],history=finalization,request_count=100,source_lock_sha256=file_sha(lock_path),model_seconds=model_seconds,fit_seconds=fit_seconds,seconds=time.monotonic()-started,policy_quality_evaluation=0,P_mapping=pmap))
 except BaseException as exc:
  save(out/'failure.json',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started))
  raise
 finally:
  if w is not None and original is not None:
   with torch.no_grad():w.copy_(original.to(w.device))
   if 'flags' in locals():
    for k,p in model.named_parameters():p.requires_grad_(flags[k])
   if 'initial_rng' in locals():restore_rng(initial_rng)
   save(out/'restore.json',dict(W0_restored=tensor_sha(w)==tensor_sha(original),methodstate_discarded=True))

def compare(root):
 import torch
 a=Path(root);terms=[json.loads((a/m/'terminal.json').read_text()) for m in ['native','I1']]
 cps=[]
 for term in terms:
  r=term['comparison'];assert file_sha(r['path'])==r['sha256']
  cps.append(torch.load(r['path'],map_location='cpu',weights_only=True,mmap=True))
 checks={}
 for key in ['targets','weight','partial75','history','entryW','entryM']:
  x,y=[c[key] for c in cps]
  checks[key]=dict(equal=torch.equal(x,y),max_abs=float((x-y).abs().max()))
 assert len(terms[0]['native_loss_observation'])==len(terms[1]['request_chunks'])==100
 loss_equal=True
 for losses,ref in zip(terms[0]['native_loss_observation'],terms[1]['request_chunks']):
  assert file_sha(ref['path'])==ref['sha256']
  value=torch.load(ref['path'],map_location='cpu',weights_only=True)
  observed=[{k:row[k] for k in ('iteration','total','nll','kl','regularizer')} for row in value['losses']]
  loss_equal=loss_equal and losses==observed
 checks['native_loss_components']=dict(equal=loss_equal,requests=100)
 ref=terms[1]['fixed_weight_carry'];assert file_sha(ref['path'])==ref['sha256']
 value=torch.load(ref['path'],map_location='cpu',weights_only=True)
 for key in ['u','m','v','a0','teacher']:
  checks['carry_'+key]=dict(equal=torch.equal(value['whole'][key],value['second'][key]))
 checks['carry_t']=dict(equal=value['whole']['t']==value['second']['t'])
 checks['carry_last_loss']=dict(equal=value['whole']['losses'][-1]['total']==value['second']['losses'][-1]['total'])
 # Preserve mismatch evidence before failing; no adaptive tolerance.
 status='PASS' if all(v['equal'] for v in checks.values()) else 'TECHNICAL_PARITY_HOLD'
 r=save(a/'comparison-receipt.json',dict(status=status,checks=checks,inputs=[t['comparison'] for t in terms],criterion='EXACT_FP32_NO_POSTHOC_RELAXATION',scientific_gate=False))
 print(json.dumps(r));assert status=='PASS',checks

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--lock');p.add_argument('--output',required=True);p.add_argument('--mode',choices=['native','I1','compare'],required=True)
 args=p.parse_args();compare(args.output) if args.mode=='compare' else run(args.lock,args.output,args.mode)
