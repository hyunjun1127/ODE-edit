"""Sealed whole-plan single GPU process. T0 and Z gates are internal stages."""
import argparse,json,os,random,sys,time,traceback,resource
from pathlib import Path
from project.run_scripts.memit_history_lifelong.io import save,file_sha

def verify(lock):
 assert lock['instruction']=='ODEEDIT-GH-SH3-MEMIT-HJ-V2-20260930-R1'
 assert lock['resource']['cap']==1 and lock['resource']['memory_mib']<=121856
 assert lock['plan']['cells']==28 and lock['plan']['diagnostic_batch_size']==10
 assert lock['blue_commit']=='311b076a92e4ed0f14f5c8b4909732da781bc5f7'
 for item in lock['members']:
  p=Path(item['path']);assert p.is_file() and p.stat().st_size==item['bytes'],'MISSING:'+str(p)
  st=p.stat();identity=dict(device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
  if item['bytes']>=100_000_000 and identity==item.get('stat_identity'):
   continue  # Freeze performed full SHA; stable exact file identity reuses it.
  assert file_sha(p)==item['sha256'],'CHANGED:'+str(p)

def pre_model_z_gate(lock,group,out,groupout):
 if group not in ['C','D']:return False
 p=out/'calibration/lock.json'
 calibration=json.loads(p.read_text()) if p.exists() else dict(status='BLOCKED',reasons=['UPSTREAM_CALIBRATION_NOT_AVAILABLE'],upstream_technical=True)
 if p.exists():assert calibration['bindings']==lock['bindings']
 if calibration['status']=='PASS':return False
 coverage={}
 for arm in (['010','011'] if group=='C' else ['110','111']):
  rec=dict(status='BLOCKED_Z_CALIBRATION',cell='main_'+arm,calibration=calibration)
  save(out/'cells'/('main_'+arm)/'terminal.json',rec);coverage['main_'+arm]=rec
 save(groupout/'terminal.json',dict(status='COMPLETED_GROUP',group=group,coverage=coverage,
  model_loaded=False,scientific_requests=0,bindings=lock['bindings'],allocation_cost_not_zero_claim=True))
 return True

def run(lock_path,group,resume_checkpoint=None):
 start=time.monotonic();lock=json.loads(Path(lock_path).read_text());out=Path(lock['output']);out.mkdir(parents=True,exist_ok=True);groupout=out/'groups'/group;groupout.mkdir(parents=True,exist_ok=False)
 e=None;stage='VERIFY_LOCK'
 try:
  if group!='P' and not resume_checkpoint:
   ready=json.loads((out/'readiness.json').read_text());assert ready['bindings']==lock['bindings'] and ready['status']=='PASS'
  verify(lock)
  if not resume_checkpoint and pre_model_z_gate(lock,group,out,groupout):return
  v=os.statvfs(out);assert v.f_bavail*v.f_frsize>=lock['resource']['storage']['required_free_bytes'],'INSUFFICIENT_STORAGE_RESERVE'
  import torch,numpy as np,transformers
  from transformers import AutoModelForCausalLM,AutoTokenizer
  from scripts.fixed_counterfact import load_prefix
  from project.run_scripts.memit_history_lifelong.method import bind
  from project.run_scripts.memit_history_lifelong.provenance import import_closure
  from .engine import Engine
  from .plan import cells
  from .technical import t0
  from .orchestrator import Orchestrator
  from .reducer import reduce
  assert torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.44.2'
  torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
  random.seed(lock['seed']);np.random.seed(lock['seed']);torch.manual_seed(lock['seed'])
  rows=load_prefix(lock['dataset_root'],10000);plan=cells(lock['cells'])
  sys.path.insert(0,lock['blue_root']);os.chdir(lock['blue_root']);stage='MODEL_LOAD'
  model=AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,low_cpu_mem_usage=True,torch_dtype=torch.float32,attn_implementation='eager').cuda().eval()
  model.requires_grad_(False)
  tok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
  evaltok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);evaltok.pad_token_id=evaltok.eos_token_id
  assert tok.padding_side==evaltok.padding_side=='right'
  module,hp,weights,H=bind(lock['hparams'],lock,model)
  e=Engine(model,tok,evaltok,module,hp,weights,H,rows,lock,out)
  for layer in hp.layers:module.get_cov(model,tok,hp.rewrite_module_tmp.format(layer),hp.mom2_dataset,hp.mom2_n_samples,hp.mom2_dtype)
  e.w0=e.snapshot()
  save(groupout/'runtime.json',dict(source_commit=lock['source_commit'],lock_sha256=file_sha(lock_path),torch=torch.__version__,transformers=transformers.__version__,
   model_revision=lock['revision'],device=torch.cuda.get_device_name(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
   dtype='float32',attention='eager',autocast=False,matmul_TF32=False,cudnn_TF32=True,seed=lock['seed'],hparams=vars(hp),
   actual_entrypoint=module.__file__+'::apply_memit_seq_to_model',entrypoint_sha256=file_sha(module.__file__),task_local_AST_execute_adapter=True,
   writer_tokens=tok('MEMIT HJ')['input_ids'],evaluator_tokens=evaltok('MEMIT HJ')['input_ids'],
   tokenizer_padding='right',evaluator_kernel_padding='explicit_left',evaluator_MB=16,
   CP_exception_scope=['000','100','110','111'],permanent_checkpoints=False,plan=lock['plan'],Slurm_job=os.environ.get('SLURM_JOB_ID')))
  if group=='P':
   result=t0(e);e.restore(e.w0)
   e.stage='W0_FULL10K_OBSERVER';e.observe(list(range(10000)),out/'W0-all10k.json')
   save(out/'readiness.json',dict(status='PASS',bindings=lock['bindings'],T0=result,W0_state=e.states.identity(e.meta),W0_probe=e.states.probe(),W0_eval_sha256=file_sha(out/'W0-all10k.json')))
  elif resume_checkpoint:
   from .recovery import resume
   resume(e,Orchestrator(e,plan),resume_checkpoint)
  else:
   assert e.states.probe()==ready['W0_probe'],'CROSS_JOB_W0_OUTPUT_PARITY'
   assert e.states.identity(e.meta)==ready['W0_state'],'CROSS_JOB_W0_STATE_PARITY'
   Orchestrator(e,plan).run_group(group)
  save(groupout/'actual-import-closure.json',import_closure(sys.modules,[lock['source_root'],lock['blue_root']]))
  e.restore(e.w0)
  save(groupout/'terminal.json',dict(status='COMPLETED_GROUP',group=group,coverage=e.completed,counts=e.total,
    seconds=time.monotonic()-start,host_max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved(),
    model_restored_W0=True,bindings=lock['bindings'],full_model_forward_cost=e.forward_cost))
 except BaseException as ex:
  save(groupout/'failure.json',dict(status='TECHNICAL_FAILURE',stage=e.stage if e else stage,error=repr(ex),traceback=traceback.format_exc(),
    seconds=time.monotonic()-start,counts=e.total if e else None,temporary_CP_preserved=True,other_tasks_untouched=True,bindings=lock['bindings']))
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--group',required=True,choices=list('PABCD'));p.add_argument('--resume-checkpoint');a=p.parse_args();run(a.lock,a.group,a.resume_checkpoint)
