"""One-time CPU asset and offline tokenizer binding verification; no GPU loads."""
import argparse,hashlib,json,os,sys,time,importlib
from pathlib import Path
ROOT=Path('/data/janghj/ODE-edit'); RUNTIME=ROOT/'local/runtime/server3-experiment-ready-v1';STATE=ROOT/'local/state/sh3-experiment-ready-20260919-v1'
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def main():
 import numpy as np,torch,transformers
 from transformers import AutoTokenizer
 assert torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.44.2'
 sys.path.insert(0,str(ROOT));from scripts.fixed_counterfact import load_prefix,verify
 data=ROOT/'local/datasets/counterfact-fixed-10k-v1';dataset=verify(data)
 prefix={n:load_prefix(data,n) for n in [100,1000,3000,10000]}
 for n,recs in prefix.items():assert recs==prefix[10000][:n]
 batch=[hashlib.sha256(json.dumps(prefix[10000][i:i+100],sort_keys=True).encode()).hexdigest() for i in range(0,10000,100)]
 inv=json.loads((STATE/'asset-inventory.json').read_text());assert all(x['reuse'] for x in inv)
 manifest={'schema':'odeedit.server3.readiness/v1','instruction_id':'ODEEDIT-S06-SH3-EXPERIMENT-READY-ASSETS-20260919-V1','runtime_root':str(RUNTIME),'python':str(RUNTIME/'venv/bin/python'),'native_root':str(RUNTIME/'easyedit'),'evaluator_binding':str(RUNTIME/'contexts/evaluation.py'),'evaluator_root':str(RUNTIME/'evaluator_source/project/run_scripts/blue_alphaedit_sequential_comparison'),'helper_root':str(RUNTIME/'evaluator_source/project/run_scripts'),'dataset_root':str(data),'dataset':dataset,'prefix_counts':list(prefix),'batch100_count':len(batch),'batch100_sha256':batch,'models':{},'save_checkpoints':False,'tf32_matmul':False,'tf32_cudnn':False,'attention':'eager','dtype':'float32','asset_inventory':str(STATE/'asset-inventory.json'),'native_config':str(RUNTIME/'contexts/config4.json'),'context_capsule':str(RUNTIME/'contexts/cold-capsule.json'),'source_allowlist':str(STATE/'source-allowlist.json')}
 seals=json.loads((ROOT/'agents/server4/alphaedit-runtime-path-seal.json').read_text())['models'];hf=json.loads((ROOT/'agents/server4/p4-hf-consumed-closure-seal.json').read_text())['models']
 for model,seal in seals.items():
  rows=[x for x in inv if x['model']==model];assets={}
  for row in rows:
   p=Path(row['candidate']);st=p.stat();assert [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns]==row['stable_stat'],'post-SHA asset drift'
   if row['kind']=='projector':
    tensor=torch.load(p,map_location='cpu',mmap=True,weights_only=True);assert list(tensor.shape)==row['shape'] and str(tensor.dtype)=='torch.float32';assets['projector']={'path':str(p),'sha256':row['sha256'],'shape':list(tensor.shape),'dtype':'float32','layer_mapping':{str(l):i for i,l in enumerate(seal['layers'])}};del tensor
   if row['kind']=='covariance':
    with np.load(p,allow_pickle=False) as z:
     keys=z.files;key=next(k for k in keys if k.endswith('mom2') and not k.endswith('count'));a=z[key];assert list(a.shape)==row['shape'] and a.dtype==np.float32
     assets.setdefault('covariance',{})[str(row['layer'])]={'path':str(p),'sha256':row['sha256'],'shape':list(a.shape),'dtype':str(a.dtype),'npz_key':key};del a
   if row['kind']=='hparams':assets['canonical_hparams']={'path':str(p),'sha256':row['sha256']}
  snapshot=hf[model]['snapshot_path'];tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
  tk={'class':type(tok).__name__,'padding_side':tok.padding_side,'bos':tok.bos_token_id,'eos':tok.eos_token_id,'default_add_bos':getattr(tok,'add_bos_token',None)}
  manifest['models'][model]={'revision':seal['revision'],'snapshot':snapshot,'assets':assets,'asset_ready':True,'model_load_ready':False,'tokenizer':tk,'context_ready':model=='llama3-8b-inst','native_shadow_ready':False}
  if model=='llama3-8b-inst':
   capsule=json.loads((RUNTIME/'contexts/cold-capsule.json').read_text());tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
   actual=[[tok(c)['input_ids'] for c in g] for g in capsule['contexts']];assert actual==capsule['context_tokens'],'canonical context token drift'
   manifest['models'][model]['context_identity']=hashlib.sha256(json.dumps(capsule['contexts'],sort_keys=True).encode()).hexdigest()
  print('ASSET_TOKENIZER_PASS',model,flush=True)
 os.chdir(RUNTIME/'easyedit');sys.path.insert(0,str(RUNTIME/'easyedit'));native=importlib.import_module('AlphaEdit.AlphaEdit_main')
 assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()=='79da927aad5ab817fd008c5958768adcd00556989a8efbaa2c4bdc80d8fc842e'
 manifest['native_source_sha256']='79da927aad5ab817fd008c5958768adcd00556989a8efbaa2c4bdc80d8fc842e';manifest['cpu_verified_at']=time.time();save(RUNTIME/'manifest.json',manifest)
 print('CPU_READY',flush=True)
if __name__=='__main__':main()
