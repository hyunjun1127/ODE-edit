"""CPU first-layer ridge-prior audit over the unchanged CF stream; no z or edits."""
import json,sys,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,logging
from omegaconf import OmegaConf
logging.set_verbosity_error();torch.set_num_threads(8)
base=Path('/mnt/raid5/janghj/ODE-edit/local/fe-collapse-repair-20261011');root=base/'native-r2';author=root/'author';sys.path.insert(0,str(author))
from algs.memit import memit_main as native
from util import nethook
c=json.loads((root/'configs/gptj.json').read_text());a=json.loads(Path(c['model_assets']['path']).read_text());snap=a['model']['snapshot']
old=json.loads(Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/preparation-r2/configs/gptj-cf.json').read_text());context_path=Path(old['output'])/'contexts.json';contexts=json.loads(context_path.read_text())['contexts']
conf=AutoConfig.from_pretrained(snap,local_files_only=True);conf.n_layer=4
model=AutoModelForCausalLM.from_pretrained(snap,config=conf,torch_dtype=torch.bfloat16,local_files_only=True).eval();tok=AutoTokenizer.from_pretrained(snap,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
for p in model.parameters():p.requires_grad_(False)
cfg=OmegaConf.load(author/'configs/config.yaml');cfg.llms=OmegaConf.load(author/'configs/llms/gpt-j-6b.yaml');cfg.algs=OmegaConf.load(author/'configs/algs/memit.yaml')
Trace=nethook.Trace
class StopTrace(Trace):
 def __init__(self,*a,**kw):kw['stop']=True;super().__init__(*a,**kw)
nethook.Trace=StopTrace
records=json.loads(Path(c['stream']['path']).read_text())
with np.load(c['covariances'][0]['original']['path']) as z:C=(torch.from_numpy(z['mom2.mom2'].copy())/int(z['mom2.count'])).double()
with np.load(base/'native-derived-prior/gptj-layer-3.npz') as z:fixed=torch.from_numpy(z['mom2.mom2'].copy())/int(z['mom2.count'])
H=torch.zeros_like(C,dtype=torch.float32);out=base/'native-r2/gptj-l3-prior-stream-cpu.json'
result=dict(scope='CPU BF16 first-edited-layer keys, unchanged CF 2K; input to first edited layer is independent of higher edits; no z or edits, no GPU bit-parity/accuracy claim',context_path=str(context_path),context=contexts,rows=[])
for b in range(1,21):
 start=time.time();rows=records[(b-1)*100:b*100];K=native.compute_ks(model,tok,rows,cfg,3,contexts).T.double();gram=K@K.T;history=H.double();infos={}
 for arm,prior in [('original',C),('repaired',fixed)]:
  A=gram+history+15000*prior;L,info=torch.linalg.cholesky_ex(A);infos[arm]=int(info);del A,L
 H+=gram.cpu();row=dict(batch=b,cholesky_info=infos,history_frobenius=float(H.double().norm()),key_frobenius=float(K.norm()),seconds=time.time()-start);result['rows'].append(row);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
 if infos['repaired']!=0:raise RuntimeError('REPAIRED_PRIOR_STREAM_FAILED')
