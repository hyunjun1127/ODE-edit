"""Read/hash and CPU-only readiness for Llama/GPT-J; no model load or submission."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import zipfile
from official.ours.config import resolve, plain
from official.ours.common import digest
from official.runners.server3.ours_prepare import sha, tracking_config
from official.tracking.schema import load_env, bind_job_identity

ROOT=Path(__file__).resolve().parents[3]
HOME=Path('/data/janghj')


def array_header(stream):
    import numpy as np
    version=np.lib.format.read_magic(stream)
    if version==(1,0):return np.lib.format.read_array_header_1_0(stream)
    if version==(2,0):return np.lib.format.read_array_header_2_0(stream)
    raise ValueError('UNSUPPORTED_NPY_HEADER')


def prepare(model, output, env_file):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU_ONLY')
    import torch
    import transformers
    import numpy as np
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    cfg=resolve(model)
    contract=json.loads((ROOT/'official/hparams/contract.json').read_text())['models'][model]
    snapshot=HOME/'.cache/huggingface/hub'/('models--'+contract['model_id'].replace('/','--'))/'snapshots'/contract['revision']
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    # Verify HF large-file blobs against their content-addressed SHA names.
    weights=sorted(snapshot.glob('*.safetensors')) or sorted(snapshot.glob('pytorch_model*.bin'))
    if not weights:raise ValueError('MODEL_WEIGHT_MISSING')
    members=[]
    for path in weights:
        target=path.resolve(strict=True); observed=sha(target)
        if len(target.name)!=64 or observed!=target.name:raise ValueError('MODEL_HF_BLOB_SHA')
        st=target.stat();members.append(dict(path=str(path),bytes=st.st_size,sha256=observed,
            inode=st.st_ino,mtime_ns=st.st_mtime_ns))
    index=snapshot/'model.safetensors.index.json'
    if index.exists():
        expected=set(json.loads(index.read_text())['weight_map'].values())
        if expected!={p.name for p in weights}:raise ValueError('MODEL_SHARD_COVERAGE')
    basename=contract['model_id'].split('/')[-1]
    module='transformer.h.{}.mlp.fc_out' if model=='gptj' else 'model.layers.{}.mlp.down_proj'
    historical={}
    if model=='llama3':
        historical=json.loads((ROOT/'agents/server3/experiment-ready-paths-20260919-v1.json').read_text())['models']['llama3-8b-inst']['assets']['covariance']
    stats={}
    for layer in cfg['eligible_layers']:
        path=HOME/'EasyEdit/examples/data/stats'/basename/'wikipedia_stats'/(module.format(layer)+'_float32_mom2_100000.npz')
        observed=sha(path)
        if historical and observed!=historical[str(layer)]['sha256']:raise ValueError('C0_HISTORICAL_SHA')
        with zipfile.ZipFile(path) as archive:
            with archive.open('mom2.mom2.npy') as f:shape,fortran,dtype=array_header(f)
            with archive.open('mom2.count.npy') as f:count=np.lib.format.read_array(f,allow_pickle=False)
        if shape!=(cfg['expected_intermediate'],)*2 or dtype!=np.dtype('float32') or count.size!=1 or not np.isfinite(count.item()) or count.item()<=0 or int(count.item())!=count.item():
            raise ValueError('C0_HEADER_COUNT')
        stats[str(layer)]=dict(path=str(path),sha256=observed,bytes=path.stat().st_size,
            shape=list(shape),dtype=str(dtype),count=int(count.item()),
            provenance='HISTORICAL_SHA_MATCH' if historical else 'LOCAL_EASYEDIT_HEADER_AND_FRESH_SHA; historical_byte_identity_not_compared')
    if model=='gptj':mods=['jlz_price_gptj.adapter','jlz_price_gptj.entry','jlz_price_gptj.fit']
    else:mods=['jlz_interference_l1.cap_adapter','jlz_v12r.entry','jlz_interference_l1.cap_fit']
    imports={m:importlib.import_module('official.ours.core.'+m).__file__ for m in mods}
    tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    tok.pad_token=tok.eos_token;tok.padding_side='right'
    dataset=HOME/'ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json'
    dataset_sha=sha(dataset)
    if dataset_sha!='3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1':raise ValueError('DATA_IDENTITY')
    records=json.loads(dataset.read_text())[:2000]
    # Only tokenizer coverage, not a substitute for model-specific native contexts.
    identities=[]
    for record in records:
        r=record['requested_rewrite'];text=r['prompt'].format(r['subject'])
        ids=tok(text)['input_ids'];new=tok.encode(' '+r['target_new']['str'].lstrip(),add_special_tokens=False)
        if not ids or not new:raise ValueError('EMPTY_TOKENIZATION')
        identities.append(digest([record['case_id'],ids,new]))
    packs=[];context=None
    if model=='llama3':
        from official.ours.core.jlz_realization.inputs import CounterFactAdapter
        path=HOME/'ODE-edit/local/memit-history-fixed10k/20260928-v1/inputs/baseline/contexts.json'
        value=json.loads(path.read_text());bench=CounterFactAdapter(tok,value)
        context=dict(path=str(path),sha256=sha(path),semantic_sha256=digest(value),status='EXISTING_REFERENCE_CPU_CHECKED_NOT_FUTURE_RUN_LOCK')
        for start in range(0,2000,100):
            p=bench.prepare(records[start:start+100]);packs.append(dict(batch=start//100+1,identity=p['identity'],key_prefix_exact=p['entry_key_prefix_exact']))
    else:context=dict(status='NATIVE_COLD_CONTEXT_PENDING_FINAL_RUN_SETTINGS',borrowed_other_model_context=False)
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    tracking=tracking_config(head,cfg);bind_job_identity(tracking,environ={});load_env(env_file)
    state_bytes=len(cfg['eligible_layers'])*4*(cfg['expected_hidden']*cfg['expected_intermediate']+cfg['expected_intermediate']**2)
    result=dict(model=model,status='ASSETS_AND_CPU_PREPARED_AWAITING_USER_SETTINGS',source_parent=head,
        provisional_config=plain(cfg),model_snapshot=str(snapshot),model_members=members,
        covariance=stats,context=context,packs=packs,tokenizer=dict(type=type(tok).__name__,requests_checked=len(records),
            identity=digest(identities),ordered_ids_sha256=digest([r['case_id'] for r in records])),
        dataset=dict(path=str(dataset),sha256=dataset_sha),imports=imports,
        runtime=dict(python=os.sys.executable,torch=str(torch.__version__),transformers=transformers.__version__),
        tracking_config=tracking,wandb_env=str(env_file),WandB_online_receipt='/data/janghj/ODE-edit/local/ours-qwen-readiness-20261009/online-result.json',
        W_and_H_FP32_bytes=state_bytes,storage_note='RAM state lower bound; not checkpoint permission or total run disk estimate',
        GPU_qualification='NOT_RUN',model_loads=0,submitted_jobs=[],execution_settings='AWAITING_USER',
        remaining=['Final user method/run configuration and native context identity','Own runtime/transaction/evaluator/collector binding and source freeze','Actual model qualification and cap1 admission only under execution scope'])
    (out/'preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(model=model,status=result['status'],C0_layers=len(stats),weights=len(members),CPU_tokenized=len(records))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['llama3','gptj'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--wandb-env',required=True)
    a=p.parse_args();prepare(a.model,a.output,a.wandb_env)
