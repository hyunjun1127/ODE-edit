"""Evaluation-only final T100 restore: R100/P200/N1000, no writer/history/z imports."""
import argparse
import random
import resource
import time
import traceback
from pathlib import Path
from .server2_entry import bind, MODEL, DEPS
bind()
from .common import read, record, save, sha, require, tensor_sha, digest, LAYERS
from .observations import Scorer, pair
from .reduce import summary

ARMS=('NATIVE','JOINT_STEP','JOINT_CUM')

def rows_for(tok, record_):
    q=record_['requested_rewrite']
    require(len(record_['paraphrase_prompts'])==2 and len(record_['neighborhood_prompts'])==10,'PROMPT_INVENTORY')
    specs=[('continuation','R',0,q['prompt'].format(q['subject']))]
    specs += [('continuation','P',i,p) for i,p in enumerate(record_['paraphrase_prompts'])]
    specs += [('neighborhood','N',i,p) for i,p in enumerate(record_['neighborhood_prompts'])]
    return sum([pair(tok,record_,role,'all',kind,i,p) for role,kind,i,p in specs],[])

class EvalRuntime:
    def __init__(self,cfg,snapshot):
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.torch=torch;self.config=cfg;self.np=np
        require(torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.44.2','RUNTIME')
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        random.seed(20260929);np.random.seed(20260929);torch.manual_seed(20260929)
        for r in cfg['model_members']:
            s=Path(r['path']).stat();require(s.st_size==r['bytes'] and s.st_mtime_ns==r['mtime_ns'],'MODEL_STAT')
        start=time.monotonic()
        self.model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,low_cpu_mem_usage=True,
            attn_implementation='eager',torch_dtype=torch.float32).cuda().eval()
        self.model.requires_grad_(False);self.model.config.use_cache=False
        require({p.dtype for p in self.model.parameters()}=={torch.float32},'FP32')
        self.tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);self.tok.add_bos_token=False;self.tok.pad_token_id=self.tok.eos_token_id
        require(self.tok.padding_side=='right','PADDING')
        self.model_load_seconds=time.monotonic()-start
        names={f'model.layers.{l}.mlp.down_proj.weight' for l in LAYERS}
        before=self.versions();start=time.monotonic()
        require(sha(snapshot['file']['path'])==snapshot['file']['sha256'],'SNAPSHOT_SHA')
        payload=torch.load(snapshot['file']['path'],map_location='cpu',weights_only=True,mmap=True)
        require(set(payload)=={'weights','metadata'} and set(payload['weights'])==names,'SNAPSHOT_SCHEMA')
        self.selected={};self.restore_hashes={}
        with torch.no_grad():
            for l in LAYERS:
                name=f'model.layers.{l}.mlp.down_proj.weight';w=payload['weights'][name]
                require(w.dtype==torch.float32 and tuple(w.shape)==(4096,14336) and bool(torch.isfinite(w).all()),'WEIGHT_SCHEMA')
                expected=snapshot['tensor_hashes'][str(l)];require(tensor_sha(w)==expected,'TENSOR_SHA')
                actual=self.model.get_parameter(name);actual.copy_(w);self.selected[str(l)]=actual
                self.restore_hashes[str(l)]=tensor_sha(actual);require(self.restore_hashes[str(l)]==expected,'RESTORE_EXACT')
        after=self.versions();require(all(after[k]==v for k,v in before.items() if k not in names),'NONSELECTED_RESTORE_MUTATION')
        self.restore_seconds=time.monotonic()-start;self.before_eval=after
        del payload
    def versions(self):return {n:(p.data_ptr(),p._version) for n,p in self.model.named_parameters()}
    def rng_get(self):return random.getstate(),self.np.random.get_state(),self.torch.get_rng_state(),self.torch.cuda.get_rng_state_all()

def run(lockfile,index):
    from scripts.fixed_counterfact import verify
    start=time.monotonic();lock=read(lockfile);require(0<=index<3,'INDEX');spec=lock['arms'][index]
    out=Path(lock['root'])/'output'/spec['arm'];out.mkdir(parents=True,exist_ok=False)
    rt=None;allrows=[]
    try:
        for r in lock['source_members']:require(sha(r['path'])==r['sha256'],'SOURCE_CHANGED')
        for key in ('configuration','snapshot_receipt','old_catalog','old_final'):
            r=spec[key] if key!='configuration' else lock[key]
            require(sha(r['path'])==r['sha256'],'INPUT_CHANGED:'+key)
        cfg=read(lock['configuration']['path']);verify(Path(cfg['dataset']['path']).parent)
        ids=cfg['execution_ids'];require(len(ids)==len(set(ids))==100,'REQUEST_ORDER')
        data={int(r['case_id']):r for r in read(cfg['dataset']['path'])}
        old={r['row_id']:r for r in read(spec['old_catalog']['path']) if r['role']=='continuation'}
        sr=read(spec['snapshot_receipt']['path']);require(sr['exact_logp'] and not sr['early_replacement'],'SNAPSHOT_COMPLETE')
        rt=EvalRuntime(cfg,sr);scorer=Scorer(rt)
        save(out/'restore.json',dict(source=lock['source'],snapshot=spec['snapshot_receipt'],weight_hashes=rt.restore_hashes,
            selected_exact=True,nonselected_pointer_version_unchanged=True,full_nonselected_bytehash='NOT_CLAIMED',
            fp32=True,eager=True,tf32_matmul=False,tf32_cudnn=True,model_load_seconds=rt.model_load_seconds,restore_seconds=rt.restore_seconds))
        for ordinal,cid in enumerate(ids,1):
            rows=rows_for(rt.tok,data[cid]);require(len(rows)==26,'ROW_COUNT')
            for row in rows:
                if row['role']=='continuation':require(old.get(row['row_id'])==row,'ORIGINAL_RP_TOKEN_BINDING')
            values=[scorer.metric(row,category='final_full_evaluation')[0] for row in rows]
            saved=save(out/'requests'/f'{ordinal:03d}.json',values);allrows.extend(values)
            if ordinal==1:
                save(out/'initial-valid.json',dict(status='INITIAL_VALID_NOT_TERMINAL',source=lock['source'],arm=spec['arm'],
                    first_request=saved,summary=summary(values),restore=record(out/'restore.json'),
                    weight_versions_unchanged=rt.versions()==rt.before_eval,completed_requests=1,planned_requests=100))
        require(rt.versions()==rt.before_eval,'EVALUATION_MUTATION')
        after={k:tensor_sha(v) for k,v in rt.selected.items()};require(after==rt.restore_hashes,'FINAL_WEIGHT_HASH')
        result=summary(allrows);require({r['panel']:r['prompts'] for r in result}=={'continuation:R':100,'continuation:P':200,'neighborhood:N':1000},'FINAL_DENOMINATORS')
        raw=save(out/'full-metrics.json',allrows);save(out/'summary.json',result)
        prior={r['row_id']:r for r in read(spec['old_final']['path'])};rp=[r for r in allrows if r['role']=='continuation']
        require(set(prior)=={r['row_id'] for r in rp},'PRIOR_RP_IDENTITY')
        numerical=max(abs(r['nll']-prior[r['row_id']]['nll']) for r in rp)
        save(out/'terminal.json',dict(status='COMPLETED',source=lock['source'],arm=spec['arm'],scope='FINAL_T100_FULL100_R100_P200_N1000',
            requests=100,raw_rows=2600,summary=result,raw=raw,weight_hashes_before=rt.restore_hashes,weight_hashes_after=after,
            nonmutation=True,prior_rp_max_abs_nll_difference=numerical,prior_comparison='recorded, not a new tolerance gate',
            calls=scorer.calls,tokens=scorer.tokens,timers=scorer.seconds,total_seconds=time.monotonic()-start,
            cuda_peak_bytes=rt.torch.cuda.max_memory_allocated(),host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            backward=0,write=0,history_append=0,checkpoint_saved=False))
    except BaseException as exc:
        save(out/'failure.json',dict(status='TECHNICAL_FAILED',exception=repr(exc),traceback=traceback.format_exc(),raw_rows=len(allrows),seconds=time.monotonic()-start))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args();run(a.lock,a.index)
