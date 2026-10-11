"""Read-only CPU BF16 activation probe at native Qwen B5; not accuracy evaluation."""
import hashlib,json,sys,time
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,logging
logging.set_verbosity_error();torch.set_num_threads(8)
base=Path('/mnt/raid5/janghj/ODE-edit/local/fe-collapse-repair-20261011');root=base/'native-r2';author=root/'author';sys.path.insert(0,str(author))
from util.nethook import TraceDict
from algs.memit.compute_z import find_fact_lookup_idx
c=json.loads((root/'configs/qwen25.json').read_text());assets=json.loads(Path(c['model_assets']['path']).read_text());snapshot=assets['model']['snapshot'];cfg=AutoConfig.from_pretrained(snapshot,local_files_only=True);cfg.num_hidden_layers=9
model=AutoModelForCausalLM.from_pretrained(snapshot,config=cfg,torch_dtype=torch.bfloat16,local_files_only=True).eval();model.config.use_cache=False
for p in model.parameters():p.requires_grad_(False)
tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
rows=json.loads(Path(c['original_stream']['path']).read_text())[500:600];layers=list(range(4,9));names=[f'model.layers.{layer}' for layer in layers]
lookups=[find_fact_lookup_idx(r['requested_rewrite']['prompt'],r['requested_rewrite']['subject'],tok,'subject_last',verbose=False) for r in rows]
texts=[r['requested_rewrite']['prompt'].format(r['requested_rewrite']['subject']) for r in rows]
receipt=json.loads((root/'runs/qwen25/checkpoints/B05.json').read_text());payload=torch.load(receipt['path'],map_location='cpu',mmap=True,weights_only=False)
result=dict(scope='Read-only partial pretrained model CPU BF16 SDPA; incoming B6 100 bare prompts, no targets or edits, no GPU accuracy/parity claim',checkpoint=receipt,case_ids=[r['case_id'] for r in rows],layers={},attention_backend=model.config._attn_implementation)
saved={}
initial_weights={name:model.get_parameter(name).detach().clone() for name in payload['weights']}
for arm in ('w0','L4_only','B05','B05_L5_restored'):
 with torch.no_grad():
  for name,value in initial_weights.items():model.get_parameter(name).copy_(value)
  for name,value in payload['weights'].items():
   layer=int(name.split('.')[2])
   if arm=='B05' or (arm=='L4_only' and layer==4) or (arm=='B05_L5_restored' and layer!=5):model.get_parameter(name).copy_(value)
 observations={layer:dict(subject_norm=[],other_nonfirst_norm=[],first_norm=[]) for layer in layers};representations={layer:[] for layer in layers}
 for start in range(0,len(rows),8):
  inputs=tok(texts[start:start+8],padding=True,return_tensors='pt')
  with torch.no_grad(),TraceDict(model,layers=names,stop=True) as trace:model(**inputs)
  for layer in layers:
   output=trace[f'model.layers.{layer}'].output;output=output[0] if isinstance(output,tuple) else output
   for j,length in enumerate(inputs['attention_mask'].sum(dim=1).tolist()):
    lookup=lookups[start+j];h=output[j,:length].float();norms=h.norm(dim=-1)
    observations[layer]['subject_norm'].append(float(norms[lookup]));observations[layer]['first_norm'].append(float(norms[0]))
    indices=[k for k in range(1,length) if k!=lookup]
    observations[layer]['other_nonfirst_norm'].extend(norms[indices].tolist());representations[layer].append(h.clone())
 for layer,obs in observations.items():
  row={key:dict(median=float(torch.tensor(values).median()),p95=float(torch.quantile(torch.tensor(values),.95)),max=max(values)) for key,values in obs.items()}
  if arm!='w0':
   changes=[];extremes=[]
   for i,h in enumerate(representations[layer]):
    delta=(h-saved[layer][i]).norm(dim=-1);indices=[k for k in range(1,len(h)) if k!=lookups[i]];changes.extend(delta[indices].tolist())
    lookup=lookups[i];old=saved[layer][i];ratio=float(h[lookup].norm()/old[lookup].norm())
    extremes.append(dict(case_id=rows[i]['case_id'],request_index=500+i,lookup=lookup,subject=rows[i]['requested_rewrite']['subject'],w0_subject_norm=float(old[lookup].norm()),edited_subject_norm=float(h[lookup].norm()),subject_ratio=ratio,other_max_change=float(delta[indices].max()) if indices else 0.))
   row['largest_subject_growth']=sorted(extremes,key=lambda x:x['subject_ratio'],reverse=True)[:5]
   row['other_nonfirst_change_from_w0']=dict(median=float(torch.tensor(changes).median()),p95=float(torch.quantile(torch.tensor(changes),.95)),max=max(changes))
  result['layers'].setdefault(str(layer),{})[arm]=row
 if arm=='w0':saved=representations
 print(arm,json.dumps({layer:result['layers'][str(layer)][arm] for layer in layers}),flush=True)
(base/'native-r2/qwen-b05-activation-probe.json').write_text(json.dumps(result,indent=2)+'\n')
