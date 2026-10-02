"""CPU asset/runtime/native-input binding; no model load or other task writes."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from .common import ROOT,LOCAL,DESIGN,INSTRUCTION,sha,member,write,require
from .inputs import CounterFactAdapter

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=LOCAL/'preparation-v1');args=p.parse_args()
    require(not args.out.exists(),'CREATE_ONCE_PREPARATION')
    args.out.mkdir(parents=True)
    old=Path('/data/janghj/ODE-edit/local/jlz-twoarm/20261002-bs100x20-v1/preparation-v1/configuration.json')
    previous=json.loads(old.read_text());oldmembers={r['path']:r for r in previous['inputs']}
    model=Path(previous['model']);stats=Path(previous['stats'])
    stream=Path(previous['stream']);contexts=Path(previous['contexts'])
    require(sha(stream)=='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1','STREAM_SHA')
    require(sha(contexts)=='33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e','CONTEXT_SHA')
    paths=[stream,contexts]
    index=json.loads((model/'model.safetensors.index.json').read_text())
    paths += [model/n for n in sorted(set(index['weight_map'].values())|{p.name for p in model.glob('*.json')})]
    statpaths={str(l):str(stats/f'model.layers.{l}.mlp.down_proj_float32_mom2_100000.npz') for l in range(4,9)}
    paths += [Path(p) for p in statpaths.values()]
    assets=[]
    for path in paths:
        st=path.stat();prior=oldmembers.get(str(path.resolve()))
        if prior and (prior['bytes'],prior.get('inode'),prior.get('mtime_ns'))==(st.st_size,st.st_ino,st.st_mtime_ns):
            assets.append(dict(prior,verification='PRIOR_FULL_SHA_CURRENT_SIZE_INODE_MTIME',prior_receipt=member(old)))
        else:assets.append(dict(member(path),verification='NEW_FULL_SHA'))
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    bench=CounterFactAdapter(tok,json.loads(contexts.read_text()));data=json.loads(stream.read_text());packing=[]
    for phase,start,B,count in [('pilot',2000,2,2),('main',0,100,20)]:
        for i in range(count):
            records=data[start+B*i:start+B*(i+1)];s=bench.prepare(records)
            require(s['entry_key_prefix_exact'],'RUNTIME_TOKEN_PREFIX_PARITY')
            packing.append(dict(phase=phase,batch=i+1,ids=s['record_ids'],identity=s['identity'],
                                rows=len(s['row_request']),valid_tokens=int(s['tokens']['attention_mask'].sum()),
                                width=s['tokens']['input_ids'].shape[1],key_prefix_exact=True))
    schemas=[]
    for l,path in statpaths.items():
        with np.load(path,allow_pickle=False) as z:
            x=z['mom2.mom2'];n=int(z['mom2.count'])
            require(x.dtype==np.float32 and x.shape==(14336,14336) and n>0 and np.isfinite(x).all(),'C0_SCHEMA')
        schemas.append(dict(layer=int(l),count=n,shape=list(x.shape),normalization='FP32 raw/count -> FP64'));del x
    native=Path('/data/janghj/EasyEdit/easyeditor/models')
    reference=[native/'memit/compute_z.py',native/'memit/compute_ks.py',native/'rome/repr_tools.py']
    free=shutil.disk_usage(LOCAL).free;require(free>=30*1024**3,'OUTPUT_RESERVE_30GIB')
    config=dict(instruction_id=INSTRUCTION,authority='8ab5d0ad13d43e4594ff97bc3cea2c5673a501e2',
        model=str(model),stats=statpaths,stream=str(stream),contexts=str(contexts),assets=assets,
        native_reference=[member(x) for x in reference],stats_schema=schemas,packing=packing,
        profile=dict(adapter='llama_causal_down_proj',benchmark='counterfact_native',eligible_layers=[4,5,6,7,8],
                     nll_layer=31,kl_factor=.0625,norm_factor=.5,clamp_factor=.75,learning_rate=.1,lambda_C=15000),
        experiment=json.loads((DESIGN/'experiment-2k/experiment.json').read_text()),
        evaluation_schedule=json.loads((DESIGN/'experiment-2k/evaluation-schedule.json').read_text()),
        runtime=dict(python=os.path.realpath(os.sys.executable),torch=torch.__version__,transformers=transformers.__version__,
                     torch_cuda=torch.version.cuda,attention='eager',model_dtype='float32',geometry_dtype='float64',
                     matmul_TF32=False,cudnn_TF32=False,autocast=False),
        settings=dict(seed=20261002,fit_microbatch=4,observer_microbatch=2,route='strict_prefix',save_checkpoints=False),
        resources=dict(cap=2,cpu=8,gpu=1,host_mib=60416,wall='7-00:00:00',wall_not_ETA=True,
                       host_peak_estimate_gib=44,gpu_peak_estimate_gib=65,free_bytes=free,reserve_bytes=30*1024**3,
                       components='CPU: H+rollback7.66GiB, W rollback1.10GiB, factors7.66GiB, streamed loading/scratch/capture<24GiB; GPU model29.92GiB + one factor/system/solve scratch<9GiB + activations/KV/heads<26GiB; actual P2 pending'),
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE',new_baseline_runs=0,broadcast='NO_BROADCAST_NOT_REQUIRED')
    write(args.out/'configuration.json',config)
    print(json.dumps(dict(configuration=str(args.out/'configuration.json'),assets=len(assets),packing=len(packing),status='CPU_INPUT_BINDING_PASS_GPU_NOT_RUN')))

if __name__=='__main__':main()
