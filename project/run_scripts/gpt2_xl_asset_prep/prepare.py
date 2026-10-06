"""Fresh input hashes, prior completion/source evidence, metadata-only model construction; no GPU."""
import ast
import importlib.metadata
import json
import shutil
import subprocess
import time
import torch
from .common import *

def main():
    root=LOCAL/'preparation-r1';root.mkdir(parents=True,exist_ok=False)
    oldlock=json.loads((OLD/'r2/execution-lock.json').read_text());terminal=json.loads((OLD/'r2/run-r2/terminal-manifest.json').read_text())
    require(terminal['status']=='COMPLETE' and terminal['all_documents_per_layer']==100000,'OLD_COMPLETION')
    require(sha(OLD/'r2/execution-lock.json')==terminal['source_lock_sha'],'OLD_LOCK')
    sources={}
    names=['easyeditor/models/rome/layer_stats.py','easyeditor/models/rome/tok_dataset.py','easyeditor/util/runningstats.py','easyeditor/util/nethook.py','easyeditor/util/globals.py','easyeditor/models/alphaedit/AlphaEdit_main.py','hparams/AlphaEdit/gpt2-xl.yaml','hparams/MEMIT/gpt2-xl.yaml']
    for name in names:
        sources[name]=member(EASY/name)
        dest=root/'native'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(EASY/name,dest)
    previous={Path(x['path']).relative_to(STOCK).as_posix():x for x in oldlock['source_members'] if x['path'].startswith(str(STOCK)+'/')}
    comparisons={name:sources[name]['sha256']==previous[name]['sha256'] for name in names if name in previous}
    for name in names[:3]:require(comparisons[name],'STATS_SOURCE_CHANGED')
    def funcs(path):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(Path(path).read_text()).body if isinstance(x,ast.FunctionDef)}
    a=funcs(STOCK/names[5]);b=funcs(EASY/names[5]);equal={name:a[name]==b[name] for name in ('get_cov','get_project')}
    require(all(equal.values()),'PROJECTOR_FORMULA_CHANGED')
    rev=subprocess.check_output(['git','-C',str(MODEL),'rev-parse','HEAD'],text=True).strip()
    require(rev==oldlock['model_revision'],'MODEL_REVISION')
    pointer=subprocess.check_output(['git','-C',str(MODEL),'show','HEAD:model.safetensors'],text=True)
    expected=pointer.split('oid sha256:')[1].splitlines()[0]
    payload=member(MODEL/'model.safetensors');require(payload['sha256']==expected,'MODEL_LFS_PAYLOAD')
    metadata=[]
    for prev in oldlock['model_metadata']:
        current=member(prev['path']);require(current['sha256']==prev['sha256'],'MODEL_TOKENIZER_METADATA');metadata.append(current)
    data=member(oldlock['dataset']['arrow_identity']['path']);require(data['sha256']==oldlock['dataset']['arrow_identity']['sha256'],'DATASET_BYTES')
    from datasets import Dataset
    ds=Dataset.from_file(data['path']);require(len(ds)==6078422,'DATASET_ROWS')
    rs=module(root/'native/easyeditor/util/runningstats.py','native_runningstats')
    indices=list(rs.FixedRandomSubsetSampler(range(len(ds)),end=100000,seed=1));require(len(indices)==len(set(indices))==100000,'SUBSET')
    write(root/'sample-indices.json',indices)
    from transformers import AutoConfig,AutoModelForCausalLM,GPT2Tokenizer
    cfg=AutoConfig.from_pretrained(str(MODEL),local_files_only=True)
    tok=GPT2Tokenizer.from_pretrained(str(MODEL),local_files_only=True);require(tok.padding_side=='right','TOKENIZER_PADDING')
    with torch.device('meta'):m=AutoModelForCausalLM.from_config(cfg,attn_implementation='sdpa')
    m.eval();m.requires_grad_(False)
    dims={str(l):dict(weight_shape=list(m.get_submodule(f'transformer.h.{l}.mlp.c_proj').weight.shape),module_class=type(m.get_submodule(f'transformer.h.{l}.mlp.c_proj')).__name__) for l in LAYERS}
    require(all(x['weight_shape']==[6400,1600] and x['module_class']=='Conv1D' for x in dims.values()),'CONV1D_LAYOUT')
    del m
    stats=[]
    for r in terminal['statistics']:
        cur=member(stats_path(r['layer']));require(cur['sha256']==r['filename']['sha256'],'C0_EXISTING_SHA')
        stats.append(dict(layer=r['layer'],asset=cur,recorded_count=r['token_count'],documents=r['documents'],action='REUSE',historical_completed_groups=r['completed_groups']))
    proj=member(terminal['P']['path']);require(proj['sha256']==terminal['P']['sha256'],'P_EXISTING_SHA')
    profile=dict(model_family='gpt2',model_revision=rev,model_payload=payload,tokenizer_class=type(tok).__name__,padding_side=tok.padding_side,
        layers=list(LAYERS),anchor=17,readout=47,hidden=cfg.n_embd,input_width=6400,vocab_size=cfg.vocab_size,max_positions=cfg.n_positions,
        rewrite_module_tmp='transformer.h.{}.mlp.c_proj',orientation='Conv1D stored weight[input,output]; key=input6400; native update orientation must match shape',
        layer_to_stack={str(l):i for i,l in enumerate(LAYERS)},modules=dims,live_binding='current meta-device module objects + historical loaded FP32 module receipt; no new model forward',
        C0_normalization='native FP32 sum/count; count=masked token vectors',P_threshold=.02,P_comparison='<',
        inherited_edit_hyperparameters=False,ours_method_parity='NOT_TESTED',context='no editing context generation; Wikipedia TokenizedDataset field=text',lookup='future native subject_last; no editing lookup applied here')
    free=shutil.disk_usage(LOCAL).free;require(free>=4*1024**3,'STORAGE_RESERVE')
    plan=dict(instruction_id=NONCE,task_id=TASK,mode='REUSE_ALL_STATS_AND_P_CPU_REVALIDATION',
        sources=sources,source_snapshot=str(root/'native'),source_comparison=comparisons,get_cov_get_project_AST_equal=equal,
        historical_manifest=member(OLD/'r2/run-r2/terminal-manifest.json'),historical_execution_lock=member(OLD/'r2/execution-lock.json'),
        historical_run_source=member(OLD/'r2/generate.py'),historical_launcher=member(OLD/'r2/job.sbatch'),
        dataset=dict(identity=data,configuration='20200501.en',split='train',rows=len(ds),old_loader_fingerprint=oldlock['dataset']['fingerprint'],current_direct_arrow_fingerprint=ds._fingerprint,
                     sample_size=100000,random_sample=1,sampler='native FixedRandomSubsetSampler',indices=member(root/'sample-indices.json'),
                     token_maxlen=1024,batch_tokens=3072,document_group_size=100,all_layers_shared=True),
        model=profile,metadata=metadata,stats=stats,projector=proj,historical_spectra=terminal['spectra'],
        historical_runtime=oldlock['versions'],runtime={n:importlib.metadata.version(n) for n in ('torch','transformers','datasets','numpy','safetensors')},
        historical_precision=dict(model='FP32',moment='FP32',attention='sdpa',explicit_TF32_runtime_flag='NOT_RECORDED',
                                 source_note='native same stats/tally/tokenizer bytes; prior launcher did not set TF32. No claim of retrospective flag measurement; no GPU recompute solely to fill absent telemetry.'),
        current_precision=dict(model_forward='NOT_RUN_REUSE',CPU='FP32 C0/P; TF32/autocast disabled'),
        tolerances=dict(c0_symmetry_relative=1e-6,P_symmetry_maxabs=1e-5,P_idempotence_relative=1e-4,P_trace_abs=.1,retained_RMS_slack=1e-4,c0_psd_relative=1e-6),
        work=dict(GPU_jobs=0,CPU_verify_lanes=[[13,14,15],[16,17]],CPU_lane_cpus=8,CPU_lane_memory_MiB=32768,CPU_lane_wall_hours=12,
                  exact_SVD_recomputed=False,reason='existing exact C0/P completion/source/SHA reused; full P symmetry/idempotence and C0P residual recomputed on CPU; no redundant forward/SVD'),
        storage=dict(reserve_bytes=4*1024**3,free_bytes=free,new_C0_or_P_copy=False,manifest_alias_only=True),
        save_model_checkpoints=False,authorized_assets_only='C0/P existing readonly reuse; READY manifest new',tracking_env=TRACKING)
    write(root/'input-lock.json',plan);write(root/'ours-profile.json',profile)
    print(json.dumps(dict(status='INPUT_SEALED_READY_FOR_CPU_VALIDATION',lock_sha256=sha(root/'input-lock.json'),mode=plan['mode'],counts=[x['recorded_count'] for x in stats]),indent=2))

if __name__=='__main__':main()
