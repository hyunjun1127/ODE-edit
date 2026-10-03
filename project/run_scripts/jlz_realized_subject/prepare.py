"""CPU-only SH3 asset/native/runtime binding and finite resource plan."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer
from .common import ROOT,LOCAL,INSTRUCTION,TASK,require,sha,member,write,digest
from .inputs import CounterFactAdapter
from .observe import reduce_rows


def historical_digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def bind_w0(bench,data,out):
    source=Path('/data/janghj/ODE-edit/local/memit-hj/20260930-v2/attempt-v1/output/W0-all10k.json')
    require(source.is_file(),'W0_REUSE_SOURCE_MISSING')
    old=json.loads(source.read_text());rows=[];byid={r['case_id']:r for r in data[:500]}
    for kind,tag in [('R','RS'),('P','PS'),('N','NS')]:
        for row in old['metrics'][tag]['rows']:
            if row['case_id'] not in byid:continue
            record=byid[row['case_id']];rw=record['requested_rewrite'];prompt=bench.panels(record)[kind][row['prompt_index']]
            expected=historical_digest([row['case_id'],row['prompt_index'],prompt,rw['target_new']['str'],rw['target_true']['str']])
            require(row['identity']==expected,'W0_PROMPT_TARGET_IDENTITY')
            new={k:v for k,v in row.items() if k not in ('success','margin','identity')}
            new.update(kind=kind,identity=digest([row['case_id'],kind,row['prompt_index'],prompt,rw['target_new']['str'],rw['target_true']['str']]),
                endpoint=0,margin_true_minus_new=row['true_nll']-row['new_nll'])
            for label in ('new','true'):
                encoded=bench.evaluation_ids(prompt,rw['target_'+label]['str'])
                require(len(encoded[1])==row[label+'_token_count'],'W0_TOKEN_COUNT')
                new[label+'_token_identity']=digest(encoded)
            rows.append(new)
    summary=reduce_rows(rows);require({k:v['denominator'] for k,v in summary.items()}==dict(R=500,P=1000,N=5000),'W0_COUNTS')
    dst=out/'W0-first500.json';write(dst,dict(rows=rows,summary=summary,original=member(source),
        original_weight_state=old['weight_state'],forward_count=0,token_ID_validation='current tokenizer reconstruction + original source/token counts; old per-row token IDs were not stored',
        runtime_difference='Original cuDNN TF32=true vs v10=false; Llama graph contains no convolution. Same S3 FP32/eager/torch2.9.1/transformers4.44.2.',
        evaluator_grouping='Same left-padded target scoring semantics/MB16; new/true pair ordering differs; bitwise cross-grouping certification not claimed'))
    return member(dst)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    envelope=json.loads((ROOT/'messages/head/2026-10-03-jlz-v10-tprime-500-sh3.json').read_text())
    reference=json.loads((ROOT/envelope['bindings']['input_reference']).read_text())['input_identity']
    model=Path('/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2')
    stream=Path(reference['dataset']['path']);require(sha(stream)==reference['dataset']['sha256'],'DATASET_SHA')
    from scripts.fixed_counterfact import verify
    policy=verify(stream.parent)
    data=json.loads(stream.read_text());schedule=list(csv.DictReader((ROOT/'plans/global/2026-10-03-jlz-realized-subject-v10/experiment-500/case-schedule-first500.csv').open(newline='')))
    require([r['case_id'] for r in data[:500]]==[int(r['case_id']) for r in schedule],'FIXED_ORDER')
    require(hashlib.sha256(json.dumps([r['case_id'] for r in data[:500]],separators=(',',':')).encode()).hexdigest()==envelope['execution']['main']['first500_case_ids_sha256'],'FIRST500_SHA')
    require([r['case_id'] for r in data[2000:2004]]==envelope['execution']['Q1']['cases'],'Q1_CASE_ORDER')
    contexts=Path('/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/inputs/baseline/contexts.json')
    require(sha(contexts)==reference['native_contexts']['sha256'],'CONTEXT_SHA')
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
    bench=CounterFactAdapter(tok,json.loads(contexts.read_text()));packed=[]
    for b in range(1,6):
        pack=bench.prepare(data[(b-1)*100:b*100]);expected=reference['packed_native_batches'][b-1]
        require(pack['identity']==expected['identity'],'PACKED_NATIVE_IDENTITY_B'+str(b))
        packed.append(dict(batch=b,identity=pack['identity'],B=100,rows=len(pack['row_kind']),valid_tokens=int(pack['tokens']['attention_mask'].sum()),token_width=pack['tokens']['input_ids'].shape[1]))
    stats={};assets=[member(stream),member(contexts)];native=[]
    for l,r in zip(range(4,9),reference['C0_assets']):
        require(Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256'],'C0_SHA')
        stats[str(l)]=r['path'];assets.append(member(r['path']))
    # Snapshot symlink closure is bound to existing read-only blobs. No download.
    for path in sorted(model.iterdir()):
        if not path.is_file():continue
        rec=member(path);rec['logical_path']=str(path);assets.append(rec)
    blue=Path('/data/janghj/ODE-edit/local/memit-history-fixed10k/20260928-v1/blue-upstream')
    import subprocess
    require(subprocess.check_output(['git','-C',str(blue),'rev-parse','HEAD'],text=True).strip()=='311b076a92e4ed0f14f5c8b4909732da781bc5f7','BLUE_REVISION')
    for rel in ['memit/memit_seq_main.py','memit/compute_z.py','memit/compute_ks.py','rome/repr_tools.py']:
        native.append(member(blue/rel))
    disk=os.statvfs(out);require(disk.f_bavail*disk.f_frsize>20*1024**3,'DISK_RESERVE_20G')
    config=dict(instruction_id=INSTRUCTION,task_id=TASK,model=str(model),revision=model.name,stream=str(stream),contexts=str(contexts),stats=stats,
        assets=assets,native_reference=native,BLUE_commit='311b076a92e4ed0f14f5c8b4909732da781bc5f7',
        runtime={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','tokenizers','safetensors','accelerate']},
        python=sys.version,profile=dict(eligible_layers=[4,5,6,7,8],nll_layer=31,lambda_C=15000.,kl_factor=.0625),
        settings=dict(seed=20261002,fit_microbatch=2,observer_microbatch=16,save_checkpoints=False,exact_resume='NOT_AVAILABLE',
            candidates=25,updates=24,lambda_norm=.5,lambda_allocation=.1,main_batches=5,main_B=100),
        resource=dict(cap=1,GPU=1,CPU=8,host_memory_MiB=60416,host_peak_estimate_GiB=46,GPU_peak_estimate_GiB=100,
            output_reserve_bytes=2*1024**3,disk_margin_bytes=20*1024**3,disk_free_bytes=disk.f_bavail*disk.f_frsize,inodes_free=disk.f_favail,
            Q1_wall_hours=24,main_wall_hours=168,collector_wall_hours=4,total_GPUh='NOT_MEASURED',
            host_accounting='FP32 model load 29.9GiB transient; live model GPU. H3.83+rollbackH3.83+CPU priors7.66+W0/snapshot/entry3.28+prefix<2+buffers<10 GiB; phase loads not all simultaneous.',
            GPU_accounting='model29.9+FP64 factor7.66+entry/materialized weights2.19+checkpoint activations/wholeB keys<25+transient suffix/head/solve<30GiB; Q1 and Q2 measure actual.',
            runtime_ETA='Q1 H200 measurements required; 24h/168h are finite wall requests, not ETA or measured cost.'),
        packed_native=packed,fixed10k=policy,W0_reuse=bind_w0(bench,data,out),
        historical_comparisons='Existing MEMIT-H/AlphaEdit/BLUE compact tables can be reported as historical only; no baseline fit, no exact probe.')
    write(out/'config.json',config)
    print(json.dumps(dict(status='CPU_INPUT_RUNTIME_BOUND_NOT_GPU_PASS',config=member(out/'config.json'),packed=packed),indent=2))


if __name__=='__main__':main()
