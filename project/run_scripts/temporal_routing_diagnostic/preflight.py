"""Outcome-free CPU tokenizer/coverage, cache storage and native-import checks."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys
from .common import *
from .observations import build_rows


def run(binding,output):
    config=read(binding);sys.path.insert(0,str(DEPS));sys.path.insert(0,str(NATIVE));os.chdir(NATIVE)
    import torch
    from transformers import AutoTokenizer
    from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
    from .native import Trace
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
    rows=build_rows(config,tok)
    spec=importlib.util.spec_from_file_location('AlphaEdit.temporal_preflight_native',config['official_native']['path'])
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native);trace=Trace(native)
    hps=[]
    for layer in LAYERS:
        hp=AlphaEditHyperParams(**dict(config['hparams'],layers=[layer],blue=False,L2=10))
        require((hp.layers,hp.blue,hp.L2,hp.v_num_grad_steps)==([layer],False,10,25),'WRITER_CONFIG')
        hps.append(dict(layer=layer,parent_slot=layer-4,singleton_slot=0,L2=10,blue=False))
    estimates=[]
    for cp in config['checkpoints']:
        selected=[r for r in rows if r['checkpoint'] in ('all',cp)]
        require(len(selected)==844,'ROWS')
        for layer in LAYERS:
            # W0 + entry, K and down-proj output, only selected/downstream layers.
            tokens=sum(len(r['input_ids']) for r in selected)
            capture=2*tokens*(9-layer)*(14336+4096)*4
            teacher=2*sum(len(r['target_ids']) for r in selected if r['role'].startswith('base_') and r['label']=='true')*128256*4
            estimates.append(dict(branch=f'{cp}-L{layer}',rows=len(selected),valid_input_tokens=tokens,
                capture_payload_bytes=capture,teacher_payload_bytes=teacher,snapshot_payload_bytes=2*234881024))
    payload=sum(r['capture_payload_bytes']+r['teacher_payload_bytes']+r['snapshot_payload_bytes'] for r in estimates)
    reserve=payload+64*2**30  # additional raw JSON/atomic temp, serialization and nonexclusive safety
    free=shutil.disk_usage(ROOT).free;require(free>=reserve,'STORAGE_BLOCKED')
    config['resources'].update(output_reserve_bytes=reserve,planned_exact_payload_bytes=payload,
        observed_free_bytes=free,estimated_host_peak_gib=48,estimated_gpu_peak_gib=64,
        host_basis='transient original FP32 model CPU load ~30GiB; steady 4GiB M + mmap input P/M + W0 + bounded row and CPU solve-history scratch; limit59GiB',
        gpu_basis='FP32 model ~30GiB + native delta graph and vocabulary head + original FP32 14336-square solve + <=2GiB diagnostic FP64 scratch; RTX_PRO_6000',
        wall='1-00:00:00',wall_basis='24h conservative per100-fit branch registration, not measured ETA; no scientific output/time cutoff policy change')
    config['storage_estimates']=estimates
    config['token_rows_sha']=digest(rows);config['tokenizer_preflight']=dict(type=type(tok).__name__,
        backend_sha=digest(tok.backend_tokenizer.to_str()),assigned_add_bos=False,
        actual_prompt_starts_bos=sum(r['input_ids'][0]==tok.bos_token_id for r in rows),rows=len(rows),max_input_tokens=max(len(r['input_ids']) for r in rows))
    config['native_trace_preflight']=dict(loss_source_line=trace.loss_line,loss_source_sha=sha(Path(native.compute_z.__code__.co_filename)),
        duplicate_fits=0,model_loaded=False,GPU_calls=0,configurations=hps)
    dest=Path(output);cfg=save(dest/'configuration.json',config)
    receipt=save(dest/'cpu-preflight.json',dict(configuration=cfg,estimated_payload_bytes=payload,
        reserve_bytes=reserve,free_bytes=free,branches=15,rows=len(rows),GPU_calls=0,native_import='PASS',numerical_actual='NOT_TESTED'))
    print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.binding,a.output)
