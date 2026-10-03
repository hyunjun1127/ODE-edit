"""CPU-only canonical/input/runtime/lookup binding before any model results."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
from scripts.fixed_counterfact import load_prefix,verify
from project.run_scripts.jlz_realized_subject.inputs import CounterFactAdapter,make_rows
from .common import *
from .lookup import panel


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out;require(not out.exists(),'CREATE_ONCE_BIND');out.mkdir(parents=True)
    require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA,'ENVELOPE_SHA')
    envelope=json.loads((ROOT/ENVELOPE).read_text());receipt=json.loads((ROOT/envelope['bindings']['source_member_manifest']).read_text())
    canonical=[]
    for row in envelope['bindings']['files']:
        actual=member(ROOT/row['path']);require(actual['sha256']==row['sha256'] and actual['bytes']==row['bytes'],'CANONICAL_IDENTITY')
        canonical.append(actual)
    source=[]
    for row in receipt['source_members']:
        blob=subprocess.check_output(['git','show',FROZEN+':'+row['path']],cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest()==row['sha256'] and len(blob)==row['bytes'],'FROZEN31_GIT')
        original=Path(receipt['source_snapshot'])/'source'/row['path']
        require(sha(original)==row['sha256'],'FROZEN31_SNAPSHOT')
        current=member(ROOT/row['path'])
        nonruntime=('project/run_scripts/jlz_realized_subject/collect.py','project/run_scripts/jlz_realized_subject/test_operations.py')
        if row['path'] not in nonruntime+('project/run_scripts/jlz_realized_subject/optimize.py',):
            require(current['sha256']==row['sha256'],'FROZEN_SOURCE_CHANGED:'+row['path'])
        source.append(dict(reference=row,current=current,
            freeze_policy='frozen c2d5fb10 bytes except own instrumented optimize; main collector/tests not used'))
    backgrounds=[]
    for row in envelope['bindings']['background_reports']:
        m=member(row['path']);require(m['sha256']==row['sha256'],'BACKGROUND_SHA');backgrounds.append(m)
    write(out/'full-read.json',dict(nonce=NONCE,canonical=canonical,source31=source,background=backgrounds,
        owner_full_read=True,source_read_not_GPU_validation=True,actual_GPU='NOT_RUN',
        implementation_diff='optimize observer callback only; new diagnostics namespace',
        independent_reviewer=False,independent_CPU_reducer='separate implementation required'))
    ref=json.loads((ROOT/'plans/global/2026-10-03-jlz-realized-subject-v10/experiment-500/input-reference.json').read_text())['input_identity']
    require(sha(DATA)==ref['dataset']['sha256'] and sha(CONTEXT)==ref['native_contexts']['sha256'],'DATA_CONTEXT_IDENTITY')
    policy=verify(DATA.parent);records=load_prefix(DATA.parent,100);all_records=json.loads(DATA.read_text())
    input_expected=json.loads((ROOT/envelope['bindings']['input']).read_text())
    require([r['case_id'] for r in records]==input_expected['ids'],'FIRST100_ORDER')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    bench=CounterFactAdapter(tok,json.loads(CONTEXT.read_text()));pack=bench.prepare(records)
    require(pack['identity']==input_expected['identity'],'PACKING_IDENTITY')
    lookup=panel(records,all_records,bench);write(out/'lookup-local.json',lookup)
    oldpath=Path('/mnt/raid5/janghj/ODE-edit/local/jlz-sequential/20261001-v1/attempt-r1/execution.lock.json')
    require(sha(oldpath)=='3fb6601ac22352f214c17d3b0a3ebcf495c4897fba0e9523a9afa36bf8a396cf','PRIOR_ASSET_SEAL')
    old=json.loads(oldpath.read_text());assets=[];stats={}
    for prior in old['inputs']:
        path=Path(prior['path'])
        # Only this task's actual model/tokenizer/dataset/context/C0 allowlist.
        if path not in (DATA,CONTEXT) and path.parent not in (MODEL,STATS):continue
        m=member(path);require(m['bytes']==prior['bytes'] and m['sha256']==prior['sha256'],'FRESH_ASSET_SHA:'+str(path))
        assets.append(m|dict(logical_path=str(path),previous_seal=str(oldpath)))
    for l,r in zip(range(4,9),ref['C0_assets']):
        path=STATS/Path(r['path']).name
        item=next(a for a in assets if a['logical_path']==str(path))
        require(item['sha256']==r['sha256'],'V10_C0_IDENTITY');stats[str(l)]=str(path)
    require(len([r for r in assets if r['logical_path'].endswith('.safetensors')])==4,'MODEL_CLOSURE')
    runtime=[]
    for name in ['transformers.models.llama.modeling_llama','transformers.masking_utils','torch.utils.checkpoint','torch.optim.adam']:
        runtime.append(member(importlib.import_module(name).__file__))
    disk=os.statvfs(out);require(disk.f_bavail*disk.f_frsize>8*1024**3 and disk.f_favail>1000,'DISK_INODE_RESERVE')
    rows=make_rows(pack);tokens=sum(len(r['tokens']['input_ids']) for r in rows)
    modelbytes=32121044992;factorbytes=5*14336**2*8;copybytes=5*4096*14336*4
    # Whole-B graph persists across microbatches; this is an explicit estimate,
    # never an actual memory qualification or permission to truncate rows.
    base=modelbytes+factorbytes+2*copybytes
    config=dict(instruction=NONCE,task=TASK,model=str(MODEL),stream=str(DATA),contexts=str(CONTEXT),stats=stats,
        assets=assets,runtime_sources=runtime,python=sys.executable,
        runtime={k:importlib.metadata.version(k) for k in ('torch','transformers','numpy','tokenizers','safetensors','accelerate')},
        reference_runtime='S3 torch2.9.1+cu128/transformers4.44.2/H200; cross-platform equality NOT_ESTABLISHED',
        profile=dict(eligible_layers=[4,5,6,7,8],nll_layer=31,lambda_C=15000.,kl_factor=.0625),
        settings=dict(seed=20261002,fit_microbatch=1,observer_microbatch=1,fit_count=1,B=100,batches=1,
            candidates=25,updates=24,arm='A',snapshots=list(CAPTURE),save_checkpoints=False,exact_resume='NOT_AVAILABLE'),
        input_identity=pack['identity'],ids=pack['record_ids'],fixed10k=policy,
        lookup=member(out/'lookup-local.json'),canonical=canonical,source_receipt=member(out/'full-read.json'),
        resources=dict(GPU=1,CPU=8,host_memory_MiB=98304,wall_hours=6,task_cap=1,project_cap=2,
            node='devbox',export='NONE',Requeue=0,storage_reserve_bytes=8*1024**3,
            disk_free_bytes=disk.f_bavail*disk.f_frsize,inodes_free=disk.f_favail,
            model_FP32_bytes=modelbytes,five_factor_FP64_bytes=factorbytes,each_selected_copy_bytes=copybytes,
            snapshot_RAM_bytes=5*copybytes,base_GPU_bytes=base,native_valid_input_tokens=tokens,
            estimated_GPU_GiB_interval=[44,48],estimated_host_GiB=65,
            capacity='ESTIMATE_ONLY; actual same-allocation qualification/OOM typed block, no retry loop',
            timing='6h is wall cap, NOT measured ETA; all costs measured in runner'),
        no_other_task_action=True,broadcast='NO_BROADCAST_NOT_REQUIRED; same-host GH access')
    require(not torch.cuda.is_initialized(),'BIND_MUST_BE_CPU_ONLY')
    write(out/'config.json',config)
    print(json.dumps(dict(status='CPU_INPUTS_BOUND_NOT_GPU_PASS',config=member(out/'config.json'),
        lookup_identifiable=lookup['identifiable'],N=lookup['denominator'],input=pack['identity'],runtime=config['runtime'],resources=config['resources']),indent=2))


if __name__=='__main__':main()
