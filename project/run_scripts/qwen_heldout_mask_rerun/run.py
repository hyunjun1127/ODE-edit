"""Native B100x5 with corrected native contexts and deferred generation."""
import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(v,f,sort_keys=True,allow_nan=False)
def tracking_config(c):
    return dict(server='server4',task_id='qwen-heldout-mask-rerun-20261010',arm=c['method']+'-heldout500',
        attempt='cold-r1',source_sha=c['source'],config_sha=c['config_sha256'],model='qwen25',model_family='QWEN',
        writer=c['method'].lower(),baseline=c['method'],role='scientific',metric_schema='official-baselines-scalar-v1',
        instruction_id='USER-OFFICIAL-BASELINES-20261008-R1',dataset='cf',
        generation_schedule='DEFERRED_CHECKPOINT_EVALUATION',observation_identity=c['stream_sha256'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    c=read(args.config); unsigned={k:v for k,v in c.items() if k!='config_sha256'}
    assert digest(unsigned)==c['config_sha256']
    source=Path(__file__).resolve().parents[3]
    for name,h in c['source_members'].items(): assert sha(source/name)==h, 'SOURCE_CHANGED'
    assert sha(c['stream'])==c['stream_sha256']
    for name,h in c['tokenizer_members'].items():assert sha(Path(c['model'])/name)==h, 'TOKENIZER_CHANGED'
    records=read(c['stream']);assert len(records)==500
    out=Path(c['out']);out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    import transformers
    from official.baselines import registry
    from official.runners.server3.native_state import NativeState
    from official.runners.server4.qwen_run import _factual_scalars, _accuracy_scalars, _editable_weights
    from official.evaluation.factual import evaluate
    from official.experiments import checkpoint
    from official.tracking import init
    from official.tracking.schema import metrics
    assert torch.__version__==c['runtime']['torch'] and transformers.__version__==c['runtime']['transformers']
    tracker=init(env_file=c['tracking_env'],spool=out/'tracking',config=tracking_config(c))
    status='FAILED';start=time.monotonic()
    try:
        assert torch.cuda.device_count()==1
        assert 'RTX PRO 6000 Blackwell Server Edition' in torch.cuda.get_device_name()
        torch.set_num_threads(8);random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        tok=transformers.AutoTokenizer.from_pretrained(c['model'],local_files_only=True)
        tok.pad_token=tok.eos_token;tok.padding_side='right'
        model=transformers.AutoModelForCausalLM.from_pretrained(c['model'],local_files_only=True,
            dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
        hp=registry.hparams(c['method'],'qwen25')
        actual=vars(hp).copy();assert actual==c['native_hparams'], 'HPARAM_CHANGED'
        native=NativeState(c['method'],model,hp,c['assets'])
        names=native.editable_parameter_names()
        guard={n:(p.data_ptr(),p._version) for n,p in model.named_parameters() if n not in names}
        # Baseline's own cold native generator, never restore_context(OURS).
        contexts=native.module.get_context_templates(model,tok)
        write(out/'contexts.json',contexts)
        context_sha=sha(out/'contexts.json')
        packs=[]
        for b in range(5):
            pack=registry.requests(records[b*100:(b+1)*100],c['method'],'qwen25')
            path=out/'packs'/f'batch-{b+1:02d}.json';write(path,pack)
            packs.append(dict(batch=b+1,path=str(path),sha256=sha(path),context_sha256=context_sha))
        write(out/'pack-lock.json',dict(stream_sha256=c['stream_sha256'],context_sha256=context_sha,packs=packs))
        identity=dict(config_sha256=c['config_sha256'],stream_sha256=c['stream_sha256'],
            code_commit=c['source'],official_tree_sha256=c['official_tree_sha256'],model_revision=c['revision'],
            tokenizer_sha256=c['tokenizer_sha256'],assets_sha256=digest(c['assets']))
        def observe(rows,path):
            result=evaluate(model,tok,rows,'cf',batch_size=16,identity=identity)
            write(out/path,result);return result
        def log_observed(observed,prefix,edits):
            v=_factual_scalars('cf',observed['cases'],observed['summary'],prefix,edits)
            if prefix=='current/pre':v['pre_state_edits']=edits-100
            v.update(_accuracy_scalars('cf',observed,prefix))
            metrics(v,scientific=True,config_values=tracker.config_values)
            tracker.log(v)
        w0=observe(records,'W0/factual.json')
        # Exact heldout500 W0 kept local; legacy scalar namespace avoids false first2000 label.
        tracker.log(dict(edits=0,**{'eval/'+k:float(w0['summary'][s]) for k,s in
            [('RS','Efficacy'),('PS','Generalization'),('NS','Specificity'),('harmonic','Score')]}))
        write(out/'initial.json',dict(job_id=os.environ['SLURM_JOB_ID'],context_sha256=context_sha,
            identity=identity,seed=c['seed'],cold=True,native_history_empty=True,role='heldout_baseline',requests=500))
        # New task-owned checkpoint folder only. Latest1 rotation is common policy; B5 kept.
        def save(b):
            payload_bytes=sum(model.get_parameter(n).numel()*4 for n in names)
            payload_bytes+=sum(t.numel()*4 for t in native.cache_for_checkpoint().values())
            assert shutil.disk_usage(out).free>=c['disk_save_reserve_bytes']+payload_bytes, 'RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE'
            return checkpoint.save(out/'checkpoint',batch=b,weights=_editable_weights(model,names),
                cache_c=native.cache_for_checkpoint(),contexts=contexts,evaluation_cursor=dict(edits=b*100),
                identity=identity,method=c['method'],evaluation_complete=True)
        save(0)
        for b in range(1,6):
            current=records[(b-1)*100:b*100]
            t=time.monotonic();native.apply(tok,read(packs[b-1]['path']))
            assert native.context_snapshot()==contexts, 'CONTEXT_CHANGED'
            assert all((p.data_ptr(),p._version)==guard[n] for n,p in model.named_parameters() if n in guard), 'NONEDITED_MUTATION'
            assert all(torch.isfinite(model.get_parameter(n)).all().item() for n in names), 'NONFINITE_COMMIT'
            post=observe(records if b==5 else current,f'batch-{b:02d}/post.json')
            if b==5:
                from official.evaluation.reduce import counterfact
                last=dict(cases=post['cases'][-100:],summary=counterfact(post['cases'][-100:]))
                from official.evaluation.factual import _accuracy
                last['accuracy']={kind:_accuracy([obs['target_'+obs['desired_target']]
                    for case in last['cases'] for obs in case[kind+'_observations']])
                    for kind in ('rewrite','paraphrase','neighborhood')}
                log_observed(last,'current/post',500);log_observed(post,'all_seen/post',500)
            else:log_observed(post,'current/post',b*100)
            cp=save(b)
            write(out/f'batch-{b:02d}/commit.json',dict(batch=b,edits=b*100,checkpoint=cp,
                context_sha256=context_sha,post_raw_sha256=sha(out/f'batch-{b:02d}/post.json')))
            tracker.log({'batch':b,'edits':b*100,'time/phase_seconds':time.monotonic()-t})
        write(out/'final.json',dict(status='B5_500_COMPLETE_GENERATION_DEFERRED',edits=500,commits=5,
            summary=post['summary'],checkpoint=cp,consumer_status='DEFERRED_PENDING_KEEP_SOURCE',identity=identity))
        status='B5_500_COMPLETE_GENERATION_DEFERRED'
    except BaseException as e:
        write(out/'error.json',dict(type=type(e).__name__,error=str(e)));raise
    finally:
        write(out/'terminal.json',dict(status=status,seconds=time.monotonic()-start,job_id=os.environ.get('SLURM_JOB_ID')))
        try:tracker.finish(exit_code=0 if status.startswith('B5_') else 1)
        except Exception as e:write(out/'logging-finish-error.json',dict(type=type(e).__name__))

if __name__=='__main__':main()
