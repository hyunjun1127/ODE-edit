"""P0 and six fixed P1 endpoints, one isolated process. No audit/suffix launch."""
import argparse,copy,hashlib,importlib,json,os,time,traceback
from pathlib import Path

def file_sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def save(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
 return dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path))

def save_tensor(path,obj):
 import torch
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:torch.save(obj,f)
 return dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path))

def run(lock_path,output):
 import torch,numpy as np,transformers
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from scripts.fixed_counterfact import load_prefix
 from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng,restore_rng,tensor_sha,restore_checkpoint,SingletonSpec
 from project.run_scripts.baseline_mechanism_first.contracts import digest
 from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
 from .fitting import NativeSingletonFitter,select_projector,materialize_alpha
 from .evaluation import counterfact,wiki,mmlu_alternative
 from .panels import history_status
 root=Path(output);root.mkdir(parents=True,exist_ok=False)
 lock=json.loads(Path(lock_path).read_text());started=time.monotonic();stage='LOCK_VERIFY';model=None;weights={};w0={};initial_rng=None;entry=None;restore_entry=None;requires_grad=None;nonguard=None
 try:
  for e in lock['members']:
   assert Path(e['path']).stat().st_size==e['bytes'] and file_sha(e['path'])==e['sha256'],('SOURCE_ASSET_DRIFT',e['path'])
  assert transformers.__version__==lock['transformers'] and torch.__version__==lock['torch']
  assert os.environ.get('SLURMD_NODENAME','server4')=='server4'
  torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=lock['tf32_matmul'];torch.backends.cudnn.allow_tf32=lock['tf32_cudnn']
  records=load_prefix(lock['dataset_root'],10000)
  current=records[5000:5100];history_ord=lock['historical_ordinals'];historical=[records[i] for i in history_ord]
  annotations={e['case_id']:e for e in history_status(records,history_ord,entry_n=5000)}
  wiki_panel=json.loads(Path(lock['wiki_panel']).read_text())
  mmlu_rows=json.loads(Path(lock['mmlu100']).read_text());dev_mmlu=[mmlu_rows[i] for i in lock['mmlu_development_indices']]
  import sys
  sys.path.insert(0,lock['blue_root']);os.chdir(lock['blue_root'])
  module=importlib.import_module('AlphaEdit.AlphaEdit_main')
  HP=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
  hp4=HP.from_json(lock['config4']);hp8=HP.from_json(lock['config8'])
  assert hp4.layers==[4] and hp8.layers==[8]
  assert {k:v for k,v in vars(hp4).items() if k!='layers'}=={k:v for k,v in vars(hp8).items() if k!='layers'}
  cp=torch.load(lock['entry_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
  assert cp['metadata']['batch']==50 and cp['metadata']['seen_ids']==[r['case_id'] for r in records[:5000]]
  fullp=torch.load(lock['projector'],map_location='cpu',weights_only=True,mmap=True)
  P4,pmap4=select_projector(fullp,4);P8,pmap8=select_projector(fullp,8);del fullp
  M4=torch.zeros_like(cp['cache_c']);M8=torch.zeros_like(M4)
  stage='MODEL_LOAD';t=time.monotonic()
  model=AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
  tok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
  etok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);etok.pad_token_id=etok.eos_token_id
  assert tok.padding_side==etok.padding_side=='right' and all(p.dtype==torch.float32 for p in model.parameters())
  model_seconds=time.monotonic()-t;parameters=dict(model.named_parameters())
  weights={l:parameters[f'model.layers.{l}.mlp.down_proj.weight'] for l in (4,8)}
  w0={l:w.detach().cpu().clone() for l,w in weights.items()};initial_rng=capture_rng()
  requires_grad={k:p.requires_grad for k,p in parameters.items()}
  assert all(p.grad is None for p in parameters.values())
  training_modes=[(m,m.training) for m in model.modules()]
  nonselected={k:(v,v.data_ptr(),v._version,tensor_sha(v)) for k,v in parameters.items() if k not in [f'model.layers.{l}.mlp.down_proj.weight' for l in (4,8)]}
  binding=bind_evaluation_sources(lock['historical_evaluator_root'],helper_root=lock['helper_scripts_root'])
  def nonguard(full=False):
   for k,(obj,ptr,version,sha) in nonselected.items():
    assert parameters[k] is obj and obj.data_ptr()==ptr and obj._version==version,('NONSELECTED_MUTATION',k)
    if full:assert tensor_sha(obj)==sha,('NONSELECTED_BYTES',k)
  def state():
   return dict(weights={str(l):tensor_sha(w) for l,w in weights.items()},M4=tensor_sha(M4),M8=tensor_sha(M8),P4=tensor_sha(P4),P8=tensor_sha(P8),contexts=digest(module.CONTEXT_TEMPLATES_CACHE),rng=digest(capture_rng()))
  def evaluate_endpoint(label):
   before=state();versions={k:(v.data_ptr(),v._version) for k,v in parameters.items()};t=time.monotonic()
   result=dict(current=counterfact(model,etok,current,panel='Current'),historical=counterfact(model,etok,historical,panel='Historical',annotations=annotations),wiki=wiki(model,wiki_panel),mmlu=mmlu_alternative(model,etok,dev_mmlu))
   assert state()==before and all((v.data_ptr(),v._version)==versions[k] for k,v in parameters.items()),'EVALUATOR_MUTATION'
   nonguard();return save(root/label/'evaluation.json',dict(result,endpoint_state=before,seconds=time.monotonic()-t,evaluation_nonmutation=True))
  stage='W0_OBSERVATION';module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(cp['metadata']['contexts']);module.COV_CACHE={}
  # Preparation reference only, no writer/controller consumption of the values.
  w0_evaluation=evaluate_endpoint('W0')
  stage='ENTRY_RESTORE'
  restored=restore_checkpoint(model,module,M4,cp,SingletonSpec(4),expected_model_revision=lock['model_revision'],expected_seen_ids=[r['case_id'] for r in records[:5000]])
  entry_rng=capture_rng();entry_contexts=copy.deepcopy(module.CONTEXT_TEMPLATES_CACHE)
  entry_weights={l:w.detach().cpu().clone() for l,w in weights.items()};entry_M4=M4.clone()
  assert tensor_sha(entry_weights[8])==tensor_sha(w0[8])
  entry_eval=evaluate_endpoint('ENTRY');stage='M8_RECONSTRUCT';t=time.monotonic();history_steps=[]
  entry_weight_sha={l:tensor_sha(w) for l,w in weights.items()}
  for batch in range(50):
   ctx=json.loads(Path(lock['history_contexts'][batch]['path']).read_text())
   module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(ctx)
   requests=[copy.deepcopy(r['requested_rewrite']) for r in records[batch*100:(batch+1)*100]]
   for req in requests:
    if req['target_new']['str'][0]!=' ':req['target_new']['str']=' '+req['target_new']['str']
   begin=time.monotonic()
   with torch.no_grad():
    K=module.compute_ks(model,tok,requests,hp8,8,ctx).T.cpu()
    M8[0,:,:] += K @ K.T
   assert torch.isfinite(M8).all(),'NONFINITE_RECONSTRUCTED_M8'
   history_steps.append(dict(batch=batch+1,events=100,context_sha256=lock['history_contexts'][batch]['sha256'],key_sha256=tensor_sha(K),seconds=time.monotonic()-begin))
  assert {l:tensor_sha(w) for l,w in weights.items()}==entry_weight_sha and tensor_sha(M4)==tensor_sha(entry_M4)
  module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(entry_contexts);restore_rng(entry_rng);entry_M8=M8.clone();nonguard(full=True)
  history_seconds=time.monotonic()-t
  prepared=save_tensor(root/'prepared.pt',dict(weights=entry_weights,M4=entry_M4,M8=entry_M8,rng=entry_rng,contexts=entry_contexts,metadata=dict(entry_checkpoint=lock['entry_checkpoint'],sample_root=lock['sample_root'],entry_n=5000,P4=pmap4,P8=pmap8)))
  save(root/'history-provenance.json',dict(method='COMMON_WE_REENCODE_ALL5000_EVENTS_B100_CHRONOLOGICAL_FP32_CPU_GRAM',steps=history_steps,seconds=history_seconds,M8_sha256=tensor_sha(M8),alpha_reconstructions=1,history_per_suffix='APPEND_ONLY_NOT_EXECUTED',prepared=prepared))
  entry=state()
  def restore_entry():
   with torch.no_grad():
    for l,w in weights.items():w.copy_(entry_weights[l].to(w.device))
    M4.copy_(entry_M4);M8.copy_(entry_M8)
   module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(entry_contexts);restore_rng(entry_rng)
   for k,p in parameters.items():
    assert p.grad is None,'MODEL_PARAMETER_GRAD_MUTATION'
    p.requires_grad_(requires_grad[k])
   for m,mode in training_modes:m.train(mode)
   assert state()==entry,'BRANCH_ENTRY_RESTORE_FAILURE'
   nonguard()
  fitter=NativeSingletonFitter(module,expected_source_sha256=lock['editor_sha256'],contexts=entry_contexts)
  requests=[r['requested_rewrite'] for r in current];fits={};endpoints={};committed=set()
  stage='N4_FRESH_FIT';restore_entry();t=time.monotonic()
  fits['N4']=fitter.fit(model,tok,hp4,M4,P4,requests,layer=4,capture=True)
  torch.cuda.synchronize();native_fit_seconds=time.monotonic()-t
  native_weight=fits['N4']['weight'];save_tensor(root/'N4/native-target-key-readout.pt',fits['N4']['captures'])
  save(root/'N4/fit.json',fits['N4']['receipt']);restore_entry()
  arms=[('N4',1.,None),('S875',.875,None),('S75',.75,None),('FULL8',1.,8),('RES8',.75,8),('REFIT4',.75,4)]
  for arm,alpha,donor in arms:
   stage=arm+'_ENDPOINT';restore_entry();t=time.monotonic();materialization=materialize_alpha(weights[4],entry_weights[4],native_weight,alpha);torch.cuda.synchronize();materialization_seconds=time.monotonic()-t
   fit_receipt=None
   if donor is not None:
    partial=state();t=time.monotonic()
    fit=fitter.fit(model,tok,hp8 if donor==8 else hp4,M8 if donor==8 else M4,P8 if donor==8 else P4,requests,layer=donor,capture=True)
    torch.cuda.synchronize();fit_receipt=dict(fit['receipt'],synchronized_wall_seconds=time.monotonic()-t,actual_partial_state=partial)
    save_tensor(root/arm/'native-target-key-readout.pt',fit['captures']);save(root/arm/'fit.json',fit_receipt)
   assert arm not in committed
   bindings=[(4,hp4,M4,P4)]+([(8,hp8,M8,P8)] if donor==8 else [])
   t=time.monotonic();finalization=fitter.finalize(model,tok,requests,bindings);torch.cuda.synchronize();finalization_seconds=time.monotonic()-t
   assert sum(x['history_append'] for x in finalization)==len(bindings)
   committed.add(arm);nonguard(full=True)
   endpoint_state=state()
   cpref=save_tensor(root/arm/'endpoint.pt',dict(weights={l:w.detach().cpu().clone() for l,w in weights.items()},M4=M4.clone(),M8=M8.clone(),rng=capture_rng(),contexts=copy.deepcopy(module.CONTEXT_TEMPLATES_CACHE),metadata=dict(arm=arm,entry_n=5000,accepted_current=100,selected_layers=[b[0] for b in bindings],state=endpoint_state,sample_root=lock['sample_root'],base_model_revision=lock['model_revision'])))
   online_cost=native_fit_seconds+(fit_receipt['synchronized_wall_seconds'] if fit_receipt else 0)+finalization_seconds+materialization_seconds
   endpoints[arm]=dict(arm=arm,alpha=alpha,second_layer=donor,endpoint=cpref,state=endpoint_state,materialization=materialization,history=finalization,policy_instrumented_online_seconds=online_cost,pure_writer_without_instrumentation_seconds='NOT_SEPARATED',materialization_seconds=materialization_seconds,finalization_seconds=finalization_seconds,cost_note='phase timings include guards/hash/capture overhead; compute_z/keys/readout/solve passthrough timings in fit receipt; not a controlled speedup')
   save(root/arm/'commit.json',endpoints[arm]);restore_entry()
   if arm=='RES8':
    save(root/'INITIAL_VALID.json',dict(status='ACTUAL_NATIVE_AND_PARTIAL_SECOND_FIT_RESTORED',completed_endpoints=list(endpoints),native_fit=fits['N4']['receipt'],second_fit=fit_receipt,entry_restored=True,history_between_fits=0,endpoint_history_once=True,P_mapping=[pmap4,pmap8],evaluation_nonmutation_preparation=True,core_complete=False,after_initial='MONITORING_PAUSED_AWAITING_USER'))
    print('LOWCOST_INITIAL_VALID',flush=True)
  # Fixed complete six-arm evaluation is part of the already submitted program.
  for arm,_,_ in arms:
   stage=arm+'_STATIC_EVALUATION';e=torch.load(endpoints[arm]['endpoint']['path'],map_location='cpu',weights_only=True,mmap=True)
   with torch.no_grad():
    for l,w in weights.items():w.copy_(e['weights'][l].to(w.device))
    M4.copy_(e['M4']);M8.copy_(e['M8'])
   module.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(e['contexts']);restore_rng(e['rng']);assert state()==e['metadata']['state']
   endpoints[arm]['evaluation']=evaluate_endpoint(arm);restore_entry();del e
  save(root/'comparison-capsule.json',dict(entry=restored,prepared=prepared,entry_state=entry,source_lock_sha256=file_sha(lock_path),target_mode='SAME_HOST_FRESH_NATIVE',native_z_shared_first_fit=True,second_fit_fresh=3,request_z_total=400,original_target_replay=0,same_host_target_replay=0,evaluator_binding=binding))
  save(root/'core-terminal.json',dict(status='SIX_FIXED_ENDPOINTS_COMPLETE',endpoints=endpoints,preparation=dict(model_seconds=model_seconds,M8_seconds=history_seconds,W0_evaluation=w0_evaluation,entry_evaluation=entry_eval),study_shared_native_fit_seconds=native_fit_seconds,policy_selection=None,audit_executed=False,suffix_executed=False,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-started))
 except BaseException as exc:
  original_trace=traceback.format_exc();rollback=dict(status='CORE_ENTRY_NOT_YET_AVAILABLE')
  if restore_entry is not None:
   try:
    restore_entry();rollback=dict(status='EXACT_COMMON_ENTRY_W_M_RNG_RESTORED',state=state())
   except BaseException as restore_error:rollback=dict(status='ROLLBACK_FAILURE',error=repr(restore_error))
  save(root/'failure.json',dict(stage=stage,error=repr(exc),traceback=original_trace,seconds=time.monotonic()-started,entry_rollback=rollback,scientific_rescue=0))
  raise
 finally:
  if model is not None and w0:
   with torch.no_grad():
    for l,w in weights.items():w.copy_(w0[l].to(w.device))
   if requires_grad is not None:
    for k,p in model.named_parameters():p.requires_grad_(requires_grad[k])
   if initial_rng is not None:restore_rng(initial_rng)
   ok=all(tensor_sha(w)==tensor_sha(w0[l]) for l,w in weights.items())
   if nonguard is not None:nonguard(full=True)
   save(root/'process-restore.json',dict(selected_W0_exact=ok,parameter_version_restore='NOT_CLAIMED_COPY_INCREMENTS',RNG_restored=True,method_state_process_discarded=True))
   assert ok,'PROCESS_W0_RESTORE_FAILURE'

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.lock,a.output)
