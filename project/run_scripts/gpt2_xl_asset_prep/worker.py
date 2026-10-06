import argparse
import json
import os
from pathlib import Path
import resource
import time
import torch
from project.run_scripts.experiment_tracking import init
from project.run_scripts.experiment_tracking.schema import bind_job_identity,job_identity
from .common import *
from .native_adapter import cache_only
from .numerics import validate

def attempt_path(execution):
    path=Path(execution['attempt_root'])
    require(path.is_absolute() and path.parent==LOCAL and path.name.startswith('attempt-'),'ATTEMPT_PATH')
    require(path.is_dir() and not path.is_symlink(),'ATTEMPT_DIRECTORY')
    return path

def load_plan(lock):
    execution=json.loads(Path(lock).read_text())
    verify(execution['input_lock'],full=True)
    for row in execution['source_members']:verify(row,full=True)
    plan=json.loads(Path(execution['input_lock']['path']).read_text())
    for row in plan['sources'].values():
        # Runtime native source is frozen copy, not mutable EasyEdit import.
        rel=Path(row['path']).relative_to(EASY);require(sha(Path(plan['source_snapshot'])/rel)==row['sha256'],'FROZEN_NATIVE_SHA')
    for row in [plan['model']['model_payload'],plan['dataset']['identity'],plan['projector']]+[x['asset'] for x in plan['stats']]:verify(row)
    return execution,plan

def validate_lane(execution,plan,lane,tracking):
    root=attempt_path(execution);start=time.monotonic();torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p=torch.load(plan['projector']['path'],map_location='cpu',weights_only=True,mmap=True)
    require(type(p) is torch.Tensor and p.dtype==torch.float32 and list(p.shape)==[5,6400,6400],'P_SHAPE_DTYPE')
    layers=plan['work']['CPU_verify_lanes'][lane];records=[]
    for layer in layers:
        t=time.monotonic();statsrow=next(x for x in plan['stats'] if x['layer']==layer)
        verify(statsrow['asset'],full=True)
        c,count=cache_only(plan,layer)
        old=next(x for x in plan['historical_spectra'] if x['layer']==layer)
        require(old['threshold']==.02 and old['dtype']=='torch.float32' and old['device']=='cpu','OLD_SPECTRAL_BINDING')
        check=validate(c,p[LAYERS.index(layer)],old['nullity'],plan['tolerances'])
        verify(statsrow['asset']);verify(plan['projector'])
        row=dict(status='READY_REUSED_VALIDATED',layer=layer,stats=statsrow['asset'],projector=plan['projector'],
            stack_index=LAYERS.index(layer),documents=100000,masked_token_count=count,
            historical_native_FP32_SVD=old,checks=check,seconds=time.monotonic()-t,
            source_sha=execution['source_commit'],input_lock_sha256=execution['input_lock']['sha256'],
            new_model_forwards=0,new_stats_collection=0,new_SVD=0,new_projector_generation=0,
            readonly_reuse=True,weight_edits=0,CUDA_initialized=torch.cuda.is_initialized(),
            host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        require(not row['CUDA_initialized'],'CPU_LANE_GPU_INIT')
        write(root/f'output/layer-{layer}-READY.json',row);records.append(row)
        tracking.log({'phase_id':2,'step':layer,'batch':1000,
                      'time/phase_seconds':row['seconds'],'memory/host_rss_bytes':row['host_peak_RSS_bytes'],'status_code':1})
        print(json.dumps(dict(event='LAYER_READY',layer=layer,seconds=row['seconds'])),flush=True)
        del c
    write(root/f'output/lane-{lane}-COMPLETE.json',dict(status='COMPLETE',layers=layers,seconds=time.monotonic()-start,layer_ready=[str(root/f'output/layer-{l}-READY.json') for l in layers]))

def pack(execution,plan,tracking):
    root=attempt_path(execution)
    records=[json.loads((root/f'output/layer-{l}-READY.json').read_text()) for l in LAYERS]
    require(all(x['status']=='READY_REUSED_VALIDATED' and x['input_lock_sha256']==execution['input_lock']['sha256'] for x in records),'LAYER_READY_JOIN')
    require({x['masked_token_count'] for x in records}=={44068071},'SHARED_TOKEN_COUNT')
    for row in [plan['projector']]+[x['asset'] for x in plan['stats']]:verify(row)
    # Reuse means absolute native paths, not another .pt/.npz copy or a mutated shared alias.
    ready=dict(status='READY_ALL5_REUSED_CPU_VALIDATED',task_id=TASK,source=execution['source_commit'],
        input_lock=execution['input_lock'],model=plan['model'],dataset=plan['dataset'],
        stats=plan['stats'],projector=plan['projector'],layers=list(LAYERS),layer_checks=records,
        save_model_checkpoints=False,new_asset_tensor_copies=0,manifest_alias_only=True,
        ours_method_or_baseline_performance='NOT_TESTED',historical_precision=plan['historical_precision'],
        NO_BROADCAST_NOT_REQUIRED=True)
    write(root/'assets/READY.json',ready)
    tracking.log({'phase_id':3,'step':5,'status_code':1})

def collect(execution,plan,tracking):
    root=attempt_path(execution);ready=root/'assets/READY.json'
    layers=[l for l in LAYERS if (root/f'output/layer-{l}-READY.json').exists()]
    status='COMPLETE' if ready.exists() else 'PARTIAL_OR_FAILED'
    result=dict(status=status,ready_layers=layers,source=execution['source_commit'],input_lock=execution['input_lock'],
        readiness=member(ready) if ready.exists() else None,all5_claim=ready.exists(),GPU_seconds=0,new_stats_forwards=0,
        CPU_allocation='Slurm accounting to be collected only on user recall',checkpoints_saved=False,
        original_assets_preserved=True,missing_layers=[l for l in LAYERS if l not in layers])
    write(root/'output/collector-report.json',result)
    text=f"# GPT2-XL 자산 CPU 검산 수집\n\n상태: {status}. READY layers={layers}. 신규 GPU forward/통계 중복생성/원 자산 변경 0.\n\n기존 stats/P 재사용, 모델 실제 forward 동등성·ours 편집 성능은 NOT_TESTED. historical TF32 flag 미기록은 input lock에 보존.\n\nREADY: {ready if ready.exists() else '미완료'}.\n"
    # No publication from compute job; owner publishes compact reports on handoff/recall.
    target=root/'output/report-ko.md';target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as f:f.write(text)
    write(root/'output/TERMINAL.json',dict(status=status,report=member(target),result=member(root/'output/collector-report.json')))
    tracking.log({'phase_id':4,'step':len(layers),'status_code':1 if ready.exists() else -1})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',required=True);ap.add_argument('--role',choices=['verify','pack','collect'],required=True);ap.add_argument('--lane',type=int,default=0);a=ap.parse_args()
    execution,plan=load_plan(a.lock);name=f'{a.role}-{a.lane}';t=time.monotonic();tracker=None;root=attempt_path(execution)
    try:
        cfg=bind_job_identity(dict(server='server1',task_id=TASK,arm=name,attempt=execution['attempt'],source_sha=execution['source_commit'],config_sha=execution['input_lock']['sha256']))
        write(root/f'identity/{name}.json',dict(identity=job_identity(cfg),source=execution['source_commit']))
        tracker=init(env_file=plan['tracking_env'],spool=root/f'tracking/{name}',config=cfg)
        tracker.log({'phase_id':1,'status_code':0,'resource/gpus':0,'resource/cpus':4 if a.role=='collect' else 8})
        if a.role=='verify':validate_lane(execution,plan,a.lane,tracker)
        elif a.role=='pack':pack(execution,plan,tracker)
        else:collect(execution,plan,tracker)
    except BaseException as exc:
        # No SDK exception message/credential or token content in compact failure.
        write(root/f'output/{name}-FAILURE.json',dict(status='FAILED',type=type(exc).__name__,seconds=time.monotonic()-t,source=execution['source_commit'],partial_is_complete=False))
        if tracker:tracker.finish(exit_code=1)
        raise
    else:
        if tracker:tracker.finish()

if __name__=='__main__':main()
