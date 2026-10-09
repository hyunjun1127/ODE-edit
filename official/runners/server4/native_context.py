"""Generate native editing contexts only, with separate immutable model identity."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

ORDER=('qwen25','gptj','llama3')
SNAPSHOTS={
 'qwen25':('Qwen--Qwen2.5-7B-Instruct','a09a35458c702b33eeacc393d103063234e8bc28'),
 'gptj':('EleutherAI--gpt-j-6b','47e169305d2e8376be1d31e765533382721b2cc1'),
 'llama3':('meta-llama--Meta-Llama-3-8B-Instruct','8afb486c1db24fe5011ec46dfbe5b5dccdb575c2')}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()

def prepare(root):
 repo=Path(__file__).resolve().parents[3];root.mkdir(parents=True,exist_ok=False)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)
 import tarfile
 subprocess.run(['git','archive','-o',str(root/'source.tar'),commit,'official'],cwd=repo,check=True)
 (root/'source').mkdir()
 with tarfile.open(root/'source.tar') as t:t.extractall(root/'source',filter='data')
 members={str(p.relative_to(root/'source')):sha(p) for p in (root/'source/official').rglob('*') if p.is_file()}
 assert members['official/baselines/easyedit/util/generate.py']=='35506690c41ecb7d59f11660da41dde50338f5a2ba9613735e045a8c7ab98db4'
 seed=read(repo/'official/hparams/contract.json')['edit_seed'];assert seed==0
 for family,(leaf,revision) in SNAPSHOTS.items():
  model=Path('/data/janghj/.cache/huggingface/hub')/('models--'+leaf)/'snapshots'/revision
  assert model.is_dir(),str(model)
  files=[]
  for p in sorted(model.iterdir()):
   if not p.is_file() or p.name.endswith(('.msgpack','.h5')):continue
   if p.suffix not in ('.json','.txt','.safetensors','.bin','.model'):continue
   weight=p.suffix in ('.safetensors','.bin')
   target=p.resolve()
   expected=target.name if weight and len(target.name)==64 else sha(p)
   assert len(expected)==64
   files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=expected,
      verification='CONTENT_ADDRESSED_HF_BLOB_RUNTIME_FULL_SHA_REQUIRED' if weight else 'CPU_FULL_SHA'))
  assert any(Path(f['path']).suffix in ('.safetensors','.bin') for f in files)
  c=dict(model_family=family,model=str(model),revision=revision,model_manifest=files,
      model_manifest_sha256=digest(files),source=commit,source_members=members,
      native_method='MEMIT',seed=seed,profile='official-native-edit-context-fiveprompts-total10-topk5',
      hparams_sha256=sha(repo/f'official/hparams/MEMIT/{family}.json'),
      out=str(root/'outputs'/family),order=list(ORDER),role='input_preparation_no_edit_no_eval')
  c['config_sha256']=digest(c);write(root/f'{family}.json',c)
 write(root/'freeze.json',dict(source=commit,archive_sha256=sha(root/'source.tar'),order=list(ORDER),
    configs={f:sha(root/f'{f}.json') for f in ORDER}))

def run(path):
 c=read(path);assert digest({k:v for k,v in c.items() if k!='config_sha256'})==c['config_sha256']
 root=Path(__file__).resolve().parents[3]
 for name,h in c['source_members'].items():assert sha(root/name)==h,'SOURCE_CHANGED'
 for f in c['model_manifest']:
  assert Path(f['path']).stat().st_size==f['bytes'] and sha(f['path'])==f['sha256'],'MODEL_CONTENT_CHANGED'
 out=Path(c['out']);out.mkdir(parents=True,exist_ok=False)
 import numpy as np
 import torch
 import transformers
 from official.baselines import registry
 assert torch.cuda.device_count()==1
 torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
 start=time.monotonic()
 tok=transformers.AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
 tok.pad_token=tok.eos_token;tok.padding_side='right'
 model=transformers.AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,
    dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
 module,_,_=registry.implementation('MEMIT',c['model_family'])
 module.CONTEXT_TEMPLATES_CACHE=None
 guard={n:(p.data_ptr(),p._version) for n,p in model.named_parameters()}
 contexts=module.get_context_templates(model,tok)
 assert len(contexts)==2 and contexts[0]==['{}'] and len(contexts[1])==5
 assert all(x.count('{}')==1 for group in contexts for x in group)
 assert all((p.data_ptr(),p._version)==guard[n] for n,p in model.named_parameters()),'WEIGHT_MUTATION'
 write(out/'contexts.json',contexts)
 tokens=[[tok(x,add_special_tokens=True)['input_ids'] for x in group] for group in contexts]
 write(out/'context-token-ids.json',tokens)
 write(out/'READY.json',dict(status='CONTEXT_GENERATED',model_family=c['model_family'],source=c['source'],
   config_sha256=c['config_sha256'],model_manifest_sha256=c['model_manifest_sha256'],
   model_weight_verification='RUNTIME_FULL_SHA_PASS',revision=c['revision'],
   context_sha256=sha(out/'contexts.json'),context_tokens_sha256=sha(out/'context-token-ids.json'),
   native_module=module.__name__,native_module_sha256=sha(module.__file__),
   generator_sha256=sha(__import__(module.generate_fast.__module__,fromlist=['x']).__file__),
   seed=c['seed'],profile=c['profile'],job_id=os.environ.get('SLURM_JOB_ID'),
   runtime=dict(torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name()),
   seconds=time.monotonic()-start,edits=0,fit=0,W0_evaluation=0,GPU_qualification='NOT_RUN'))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare');ap.add_argument('--config');a=ap.parse_args()
 if a.prepare:prepare(Path(a.prepare))
 else:run(a.config)
if __name__=='__main__':main()
