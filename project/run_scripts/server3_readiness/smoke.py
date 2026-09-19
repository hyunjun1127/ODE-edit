"""Bounded, offline sequential model checks. No scientific chain/checkpoint writes."""
import argparse,copy,gc,hashlib,importlib,importlib.util,json,os,sys,time,traceback,resource,random
from pathlib import Path

def save(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def run_model(name,m,output):
 import numpy as np,torch,transformers
 from transformers import AutoModelForCausalLM,AutoTokenizer
 from scripts.fixed_counterfact import load_prefix
 t=time.monotonic();cfg=m['models'][name];result={'model':name,'revision':cfg['revision'],'job_id':os.environ['SLURM_JOB_ID'],'save_checkpoints':False,'status':'RUNNING'};save(output,result)
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.device_count()==1 and torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.44.2'
 torch.cuda.reset_peak_memory_stats();result['torch']=torch.__version__;result['transformers']=transformers.__version__;result['gpu']=torch.cuda.get_device_name(0);result['cuda']=torch.version.cuda
 loaded=time.monotonic();model=AutoModelForCausalLM.from_pretrained(cfg['snapshot'],local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
 for p in model.parameters():p.requires_grad_(False)
 result['model_load_seconds']=time.monotonic()-loaded;assert all(p.dtype==torch.float32 for p in model.parameters());assert model.config._attn_implementation=='eager'
 tok=AutoTokenizer.from_pretrained(cfg['snapshot'],local_files_only=True);tok.pad_token_id=tok.eos_token_id
 writer=AutoTokenizer.from_pretrained(cfg['snapshot'],local_files_only=True);writer.add_bos_token=False;writer.pad_token_id=writer.eos_token_id
 records=load_prefix(m['dataset_root'],1);request=copy.deepcopy(records[0]['requested_rewrite']);request['case_id']=records[0]['case_id']
 if not request['target_new']['str'].startswith(' '):request['target_new']['str']=' '+request['target_new']['str']
 # Capture only small input slices through the exact physical modules.
 captures={};hooks=[]
 def hook(layer):
  def observe(module,args):captures[str(layer)]={'shape':list(args[0].shape),'finite':bool(torch.isfinite(args[0]).all()),'dtype':str(args[0].dtype)}
  return observe
 modules=dict(model.named_modules())
 for l in range(4,9):hooks.append(modules[f'model.layers.{l}.mlp.down_proj'].register_forward_pre_hook(hook(l)))
 ids=writer(request['prompt'].format(request['subject']),return_tensors='pt').to('cuda')
 with torch.no_grad():logits=model(**ids).logits;assert bool(torch.isfinite(logits).all());baseline=logits[:,-1,:].clone()
 for h in hooks:h.remove()
 del logits
 for l in range(4,9):assert captures[str(l)]['shape'][-1]==cfg['assets']['projector']['shape'][-1] and captures[str(l)]['finite']
 result['key_module_capture']=captures
 # P/C0 CPU shape mapping was verified once; exercise the consumed operator paths.
 P=torch.load(cfg['assets']['projector']['path'],map_location='cpu',weights_only=True,mmap=True);mapping={}
 for l in range(4,9):
  item=cfg['assets']['covariance'][str(l)]
  with np.load(item['path'],allow_pickle=False) as z:
   C=z[item['npz_key']];assert C.dtype==np.float32 and list(C.shape)==item['shape']
   vector=np.zeros(C.shape[0],dtype=np.float32);vector[0]=1;cv=C@vector;pv=P[l-4]@torch.from_numpy(vector)
   assert np.isfinite(cv).all() and bool(torch.isfinite(pv).all());mapping[str(l)]={'physical_index':l-4,'P_probe_norm':float(pv.norm()),'C0_probe_norm':float(np.linalg.norm(cv))}
  del C,cv,pv
 result['P_C0_mapping']=mapping
 # Bind exact completed evaluator source without unrelated package initializers.
 spec=importlib.util.spec_from_file_location('server3_historical_evaluation_binding',m['evaluator_binding']);ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
 result['evaluator_binding']=ev.bind_evaluation_sources(m['evaluator_root'],helper_root=m['helper_root'])
 with torch.no_grad():evaluation=ev.evaluate_records(model,tok,records)
 result['evaluation']={k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in evaluation['metrics'].items()};result['evaluation_request_count']=1
 if name=='llama3-8b-inst':
  os.chdir(m['native_root']);sys.path.insert(0,m['native_root']);native=importlib.import_module('AlphaEdit.AlphaEdit_main');HP=importlib.import_module('AlphaEdit.AlphaEdit_hparams').AlphaEditHyperParams
  hp=HP.from_json(m['native_config']);assert hp.layers==[4] and hp.blue
  capsule=json.loads(Path(m['context_capsule']).read_text());contexts=capsule['contexts'];assert [[writer(c)['input_ids'] for c in g] for g in contexts]==capsule['context_tokens'];native.CONTEXT_TEMPLATES_CACHE=copy.deepcopy(contexts);native.COV_CACHE.clear()
  # The unchanged native singleton adapter is exercised only in RAM and restored.
  weight=modules['model.layers.4.mlp.down_proj'].weight;before=weight.detach().cpu().clone();wsha=tensor_sha(before);cache=torch.zeros((1,P.shape[1],P.shape[2]),dtype=torch.float32);cache_sha=tensor_sha(cache)
  rng_cpu=torch.get_rng_state();rng_gpu=torch.cuda.get_rng_state();rng_py=random.getstate();rng_np=np.random.get_state();other={k:(p,p._version,p.data_ptr()) for k,p in model.named_parameters() if k!='model.layers.4.mlp.down_proj.weight'}
  native_start=time.monotonic()
  try:
   native.apply_AlphaEdit_to_model(model,writer,[request],hp,cache_template=None,cache_c=cache,P=P[:1]);assert bool(torch.isfinite(weight).all());result['native_delta_norm']=float((weight.detach().cpu()-before).norm())
  finally:
   with torch.no_grad():weight.copy_(before.to(weight.device))
   cache.zero_();torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_gpu);random.setstate(rng_py);np.random.set_state(rng_np)
  assert tensor_sha(weight)==wsha and tensor_sha(cache)==cache_sha
  assert all(p._version==v and p.data_ptr()==ptr for p,v,ptr in other.values());assert native.CONTEXT_TEMPLATES_CACHE==contexts and not native.COV_CACHE
  for p in model.parameters():p.requires_grad_(False);p.grad=None
  with torch.no_grad():after=model(**ids).logits[:,-1,:];assert torch.equal(after,baseline),'post-restore forward drift'
  result['native_shadow']={'status':'PASS','requests':1,'layer':4,'v_steps':hp.v_num_grad_steps,'weight_restored':True,'history_restored':True,'nonselected_unchanged':True,'rng_restored':True,'checkpoint_writes':0,'seconds':time.monotonic()-native_start}
  del cache,before,weight,other
 else:result['native_shadow']={'status':'NOT_RUN','reason':'Qwen-specific completed context/native hparams not bound; model/assets/evaluator checked separately'}
 result.update(status='PASS',model_load_ready=True,seconds=time.monotonic()-t,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 save(output,result);print('MODEL_READY',name,flush=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest',required=True);a.add_argument('--output',required=True);a.add_argument('--model',choices=['llama3-8b-inst','qwen2.5-7b-inst'],required=True);args=a.parse_args()
 assert os.environ.get('SLURMD_NODENAME')=='ubuntu' and os.environ.get('SLURM_JOB_ID'),'Slurm server3 allocation required'
 assert os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1'
 m=json.loads(Path(args.manifest).read_text());out=Path(args.output)
 try:run_model(args.model,m,out)
 except BaseException as exc:
  save(out,{'status':'FAIL','model':args.model,'job_id':os.environ.get('SLURM_JOB_ID'),'error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc(),'save_checkpoints':False});raise
if __name__=='__main__':main()
