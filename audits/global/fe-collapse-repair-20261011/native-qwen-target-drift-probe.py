"""Read-only CPU native target fit, one detected outlier and the first B6 case."""
import json,sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM,AutoTokenizer,logging
logging.set_verbosity_error();torch.set_num_threads(8)
base=Path('/mnt/raid5/janghj/ODE-edit/local/fe-collapse-repair-20261011');root=base/'native-r2';author=root/'author';sys.path.insert(0,str(author))
from algs.memit.compute_z import compute_z,find_fact_lookup_idx,get_module_input_output_at_words
from algs.memit.native_sequential_z import current_batch_targets
from util.nethook import TraceDict
c=json.loads((root/'configs/qwen25.json').read_text());assets=json.loads(Path(c['model_assets']['path']).read_text());snapshot=assets['model']['snapshot']
model=AutoModelForCausalLM.from_pretrained(snapshot,torch_dtype=torch.bfloat16,local_files_only=True).eval();tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
for p in model.parameters():p.requires_grad_(False)
receipt=json.loads((root/'runs/qwen25/checkpoints/B05.json').read_text());payload=torch.load(receipt['path'],map_location='cpu',mmap=True,weights_only=False);cfg=OmegaConf.create(payload['config']);contexts=payload['contexts']
allrows=json.loads(Path(c['stream']['path']).read_text());rows=[dict(allrows[i]) for i in (500,594)]
for r in rows:r['target_new']=' '+r['target_new']
result=dict(scope='CPU BF16 native compute_z and firstforward; current requests only; initial model vs B5 model; no updates or GPU-parity claim',selected_indices=[500,594],selection='First B6 case and case13617 detected as the largest B5 L5 subject activation outlier',case_ids=[r['case_id'] for r in rows],checkpoint=receipt,arms={})
for arm in ('w0','B05'):
 if arm=='B05':
  with torch.no_grad():
   for name,value in payload['weights'].items():model.get_parameter(name).copy_(value)
 z=current_batch_targets(model,tok,rows,cfg,contexts,compute_z=compute_z,find_fact_lookup_idx=find_fact_lookup_idx,TraceDict=TraceDict)
 summary={}
 for layer in cfg.llms.layers:
  y=get_module_input_output_at_words(model,tok,layer,context_templates=[r['prompt'] for r in rows],words=[r['subject'] for r in rows],module_template=cfg.llms.layer_module_tmp,fact_token_strategy=cfg.llms.fact_token)[1].T
  summary[str(layer)]=dict(plain_output_norm=y.float().norm(dim=0).tolist(),firstforward_target_norm=z[layer].float().norm(dim=0).tolist(),target_residual_norm=(z[layer].float()-y.float()).norm(dim=0).tolist())
 torch.save({l:t.detach().cpu() for l,t in z.items()},root/f'qwen-target-probe-{arm}.pt');result['arms'][arm]=summary
 (root/'qwen-target-drift-probe.json').write_text(json.dumps(result,indent=2)+'\n');print('NATIVE_TARGET_DRIFT',arm,json.dumps(summary),flush=True)
