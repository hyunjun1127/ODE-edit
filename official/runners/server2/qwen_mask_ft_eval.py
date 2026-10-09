"""One authorized Qwen FT61900 saved-W20 public-query evaluation; no native fit."""
import argparse
import os
from pathlib import Path
import time
from official.experiments.prepare import read,write_new,digest,file_sha
from official.runners.server2.zsre_reeval import member,validate_result,runtime,INSTRUCTION,EVALUATOR_SHA
from official.runners.server2.zsre_reeval_restore import require,verify_member,load_and_restore
COUNTS=dict(rewrite=6691,paraphrase=6691,neighborhood=11476)

def config(inputs,source,attempt):
    row=inputs['row']
    value=dict(server='server2',task_id='qwen-ft-zsre-2k-reeval-20261010',arm='FT',
        attempt=attempt,source_sha=source,model='qwen25',model_family='qwen2',writer='FT',
        role='eval_only',metric_schema='official-baselines-scalar-v1',dataset='zsre',
        instruction_id=INSTRUCTION,evaluation_profile='zsre-public-query-W20-only-v1',
        checkpoint_sha256=row['checkpoint']['sha256'],evaluator_sha256=EVALUATOR_SHA,
        stream_sha256=inputs['stream']['sha256'],tokenizer_sha256=inputs['tokenizer_sha256'],
        source_run_id='61900')
    value['config_sha']=digest(value)
    return value

def execute(root):
    import torch
    from official.runners.server2.qwen_run import _load_model
    from official.evaluation.zsre_paper import evaluate
    from official.tracking import init,official_zsre_metrics
    lock=read(root/'source-lock.json')
    for item in lock['members']:
        require(file_sha(root/'source'/item['relative'])==item['sha256'],'SOURCE_CHANGED')
    for item in read(root/'input-lock.json')['members']:
        require(file_sha(item['path'])==item['sha256'],'INPUT_CHANGED')
    inputs=read(root/'ft-eval-inputs.json');source=lock['code_commit']
    out=root/'ft-eval';out.mkdir(exist_ok=False)
    cfg=config(inputs,source,root.name+'-ft-eval-job'+os.environ['SLURM_JOB_ID'])
    write_new(out/'config.json',cfg);row=inputs['row'];tracker=None;code=1
    try:
        require(runtime()==inputs['runtime'],'RUNTIME_CHANGED')
        require(torch.cuda.device_count()==1,'ONE_GPU')
        records=read(verify_member(inputs['stream']))
        verify_member(row['latest']);verify_member(row['checkpoint'])
        hparams=read(verify_member(row['original_config']))['hparams']
        tracker=init(env_file='/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env',spool=out/'tracking',config=cfg)
        model,tok=_load_model(inputs['model_snapshot'],inputs['model_revision'])
        tok.padding_side='right';tok.pad_token=tok.eos_token
        require(model.config.model_type=='qwen2','QWEN_MODEL')
        restored=load_and_restore(model,row,hparams);write_new(out/'restore.json',restored)
        last=[-1e30]
        def progress(v):
            now=time.monotonic()
            if now-last[0]>=15 or v['completed_queries']==v['total_queries']:
                require(tracker.log({'eval_progress/'+k:x for k,x in v.items()}),'TRACKING_PROGRESS_REJECTED')
                last[0]=now
        identity=dict(original_identity=row['identity'],checkpoint=row['checkpoint'],source=source,
            config_sha256=cfg['config_sha'],job_id=os.environ['SLURM_JOB_ID'])
        with torch.autocast('cuda',enabled=False):
            raw=evaluate(model,tok,records,model_family='qwen25',batch_size=16,device='cuda:0',identity=identity,progress=progress)
        summary=validate_result(raw,inputs,model_family='qwen25',counts=COUNTS)
        verify_member(row['checkpoint']);verify_member(row['latest'])
        write_new(out/'evaluation.json',raw)
        require(tracker.log(official_zsre_metrics(raw['summary'],config_values=cfg,
            endpoint='all_seen/post',edits=2000,post_state_edits=2000)),'TRACKING_FINAL_REJECTED')
        write_new(out/'result.json',dict(status='EVAL_COMPLETE_W20_2000',job_id=os.environ['SLURM_JOB_ID'],
            source=source,config_sha256=cfg['config_sha'],summary=summary,raw=member(out/'evaluation.json'),
            original_checkpoint=row['checkpoint'],no_edits=True,no_W0=True,no_generation=True))
        code=0
    except BaseException as exc:
        write_new(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc)[:500],CP_KEEP=True))
        raise
    finally:
        if tracker is not None: write_new(out/'transport-finish.json',tracker.finish(exit_code=code,timeout=45))

def collect(root):
    try:
        result=read(root/'ft-eval/result.json');raw=read(verify_member(result['raw']))
        result['CPU_independent_summary']=validate_result(raw,read(root/'ft-eval-inputs.json'),model_family='qwen25',counts=COUNTS)
    except Exception as exc: result=dict(status='NOT_COMPLETE_OR_FAILED',error_type=type(exc).__name__)
    write_new(root/'collector/ft-eval.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();execute(a.root)
