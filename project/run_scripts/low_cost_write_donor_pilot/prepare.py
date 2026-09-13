"""CPU-only input admission and immutable panel/config creation."""
import argparse,io,json,pickle,hashlib,shutil
from pathlib import Path
from .runtime import file_sha,save

def prepare(root):
 import torch
 from scripts.fixed_counterfact import load_prefix
 from project.run_scripts.baseline_mechanism_first.contracts import digest
 from project.run_scripts.baseline_mechanism_first.fixtures import tensor_sha
 from project.run_scripts.baseline_mechanism_first.panels import historical_ordinals
 from .panels import bind_counterfact_panel,split_mmlu,seal_audit,validate_wiki
 C=Path(root);A=C/'attempt-v1';A.mkdir(mode=0o700,exist_ok=False)
 R=Path('/data/janghj/ODE-edit');old=json.loads((R/'local/blue-lifelong-b100x100/attempt-checkpoint-r2/execution.lock.json').read_text())
 def member(p,expected=None):
  p=Path(p);h=file_sha(p)
  if expected:assert h==expected,(p,h,expected)
  return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
 dataset_root=R/'local/datasets/counterfact-fixed-10k-v1';records=load_prefix(dataset_root,10000)
 cp_path=C/'imports/server2/attempt-v1/W-method-state.pt';cp_member=member(cp_path,'0ecc3a4b790ca7ab4d837cab785adc329f47c9b33c5b7092827c6ddac0571704')
 cp=torch.load(cp_path,map_location='cpu',weights_only=True,mmap=True)
 assert set(cp['weights'])=={'model.layers.4.mlp.down_proj.weight'} and cp['metadata']['batch']==50
 assert cp['metadata']['seen_ids']==[r['case_id'] for r in records[:5000]]
 assert tensor_sha(cp['cache_c'])==cp['metadata']['state']['cache']
 for k,v in cp['weights'].items():assert v.dtype==torch.float32 and torch.isfinite(v).all() and tensor_sha(v)==cp['metadata']['state']['weights'][k]
 assert torch.isfinite(cp['cache_c']).all() and cp['cache_c'].dtype==torch.float32
 assert len(cp['metadata']['rng']['cuda'])==1
 historical_path=C/'imports/server1/attempt-v1/historical128.json';history=json.loads(historical_path.read_text())
 ordinals=[e['ordinal'] for e in history['records']];assert ordinals==historical_ordinals(5000)
 for e in history['records']:assert e['case_id']==records[e['ordinal']]['case_id'] and e['raw_record_sha256']==digest(records[e['ordinal']])
 wiki_path=C/'imports/server1/attempt-v1/wiki128.json';wiki=json.loads(wiki_path.read_text());wiki_seal=validate_wiki(wiki)
 # Restricted primitive-only decode, no pickle globals/code execution.
 class Plain(pickle.Unpickler):
  def find_class(self,*a):raise ValueError('PICKLE_GLOBAL_FORBIDDEN')
  def persistent_load(self,*a):raise ValueError('PERSISTENT_ID_FORBIDDEN')
 mmlu=Path('/data/janghj/EasyEdit/glue_eval/dataset/mmlu.pkl');member(mmlu,'fda21bebe7a7cdbaaac856ddb6e214bb78808ab845b64f25c5bf111474c6e715')
 stream=io.BytesIO(mmlu.read_bytes());allrows=Plain(stream).load();assert stream.read()==b''
 fixed100=allrows[10:110];split=split_mmlu(fixed100);mmlu100=save(A/'mmlu100.json',fixed100)
 current_panel=bind_counterfact_panel(records,list(range(5000,5100)),expected_count=100)
 historical_panel=bind_counterfact_panel(records,ordinals,expected_count=128)
 audit=seal_audit(records,list(range(5000,5100))+ordinals)
 panels=save(A/'panel-lock.json',dict(Current=current_panel,Historical=historical_panel,Wiki=wiki_seal,MMLU=split,Audit=audit,audit_access='NOT_EVALUATED_UNTIL_GH_POLICY_LOCK',sample_root=old['sample_root']))
 config4=json.loads(Path(old['cells'][3]['config']).read_text());assert config4['layers']==[4] and config4['blue'] and config4['L2']==1
 config8=dict(config4,layers=[8]);c4=save(A/'config4.json',config4);c8=save(A/'config8.json',config8)
 histroot=R/'local/blue-lifelong-b100x100/attempt-checkpoint-r2/output/main-cell-3'
 term=json.loads((histroot/'terminal.json').read_text());assert term['status']=='TERMINAL_VALID'
 bypath={m['path']:m for m in term['manifest_members']};contexts=[]
 for b in range(1,51):
  rel=f'B{b:03d}/contexts.json';contexts.append(member(histroot/rel,bypath[rel]['sha256']))
 next_context=json.loads((histroot/'B051/contexts.json').read_text());assert next_context==cp['metadata']['contexts']
 helper=Path(old['source_root'])/'project/run_scripts';hist_eval=helper/'blue_alphaedit_sequential_comparison'
 editor=Path(old['blue_root'])/'AlphaEdit/AlphaEdit_main.py'
 inputs=dict(instruction_id='ODEEDIT-S06-LOW-COST-WRITE-DONOR-PILOT-SH4-V1',model_revision=old['revision'],snapshot=old['snapshot'],source_head_reference=old['source_head'],blue_root=old['blue_root'],blue_head=old['blue_head'],editor_sha256=file_sha(editor),projector=old['projector'],entry_checkpoint=str(cp_path),sample_root=old['sample_root'],dataset_root=str(dataset_root),config4=c4['path'],config8=c8['path'],history_contexts=contexts,historical_ordinals=ordinals,wiki_panel=str(wiki_path),mmlu100=mmlu100['path'],mmlu_development_indices=split['groups']['development']['indices'],historical_evaluator_root=str(hist_eval),helper_scripts_root=str(helper),torch='2.9.1+cu128',transformers='4.44.2',tf32_matmul=False,tf32_cudnn=True,dependencies=old['dependencies'],python='/data/janghj/EasyEdit/.venv/bin/python',panels=panels,prior_lock=str(R/'local/blue-lifelong-b100x100/attempt-checkpoint-r2/execution.lock.json'),members=[cp_member,member(historical_path),member(wiki_path),member(mmlu),c4,c8,mmlu100,panels]+contexts)
 save(A/'inputs.json',inputs)
 save(A/'P0-cpu-receipt.json',dict(status='INPUT_CPU_PASS_NOT_GPU_VALID',cp=cp_member,checkpoint_keys=list(cp['weights']),history_shape=list(cp['cache_c'].shape),contexts_exact_nextentry=True,RNG_saved_devices=1,panels=panels,gpu_calls=0,free_disk_bytes=shutil.disk_usage(C).free,estimated_storage_bytes=40*(1<<30),estimated_gpu_hours=[2,8],estimate_not_measured=True,hard_gpu_hour_cap=None,M8_reuse='NOT_SAME_HOST_CONFIRMED; RECONSTRUCT_ONCE_AT_WE',native_capsule='CREATE_SAME_HOST_FRESH_N4_ONCE'))
 print('P0_CPU_READY',A)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();prepare(a.root)
