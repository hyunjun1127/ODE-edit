"""One cold W0/H0 process, fixed B100x100, exact native sequential writer."""
import argparse,importlib,json,os,random,sys,time,traceback
from pathlib import Path
from .io import save,file_sha,signature,content,restore,tensor_sha,digest
from .method import bind,observe,cov_guard
from .metrics import evaluate,merge,strata,FULL_BATCHES
from .reducer import reduce_run

def rng_capture():
    import torch,numpy as np
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())

def rng_restore(x):
    import torch,numpy as np
    random.setstate(x[0]);np.random.set_state(x[1]);torch.set_rng_state(x[2]);torch.cuda.set_rng_state_all(x[3])

def verify_lock(lock):
    for m in lock['members']:
        p=Path(m['path']);assert p.is_file() and p.stat().st_size==m['bytes'],'MEMBER_MISSING:'+str(p)
        assert file_sha(p)==m['sha256'],'MEMBER_CHANGED:'+str(p)
    assert lock['save_checkpoints'] is False and lock['arm']=='MEMIT_HISTORY_NATIVE'
    assert lock['batches']==100 and lock['batch_size']==100 and lock['resource']['cap']==1
    assert lock['full_eval_batches']==FULL_BATCHES

def run(lock_path):
    import numpy as np,torch,transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    lock=json.loads(Path(lock_path).read_text());output=Path(lock['output']);output.mkdir(parents=True,exist_ok=False)
    stage='PREFLIGHT';weights=state=w0=None;committed=[];start=time.monotonic()
    try:
        verify_lock(lock)
        rows=load_prefix(lock['dataset_root'],10000)
        assert isinstance(rows,list) and len(rows)==10000
        assert torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.44.2'
        torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        random.seed(lock['seed']);np.random.seed(lock['seed']);torch.manual_seed(lock['seed'])
        sys.path.insert(0,lock['blue_root']);os.chdir(lock['blue_root'])
        stage='MODEL_LOAD'
        model=AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,low_cpu_mem_usage=True,torch_dtype=torch.float32,attn_implementation='eager').cuda().eval()
        tok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
        evaltok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);evaltok.pad_token_id=evaltok.eos_token_id
        assert tok.padding_side==evaltok.padding_side=='right'
        assert {p.dtype for p in model.parameters()}=={torch.float32} and not torch.is_autocast_enabled()
        module,hp,weights,state=bind(lock['hparams'],lock,model)
        w0={k:v.detach().cpu().clone() for k,v in weights.items()};w0sig=signature(weights,state)
        save(output/'runtime.json',dict(arm=lock['arm'],lock_sha256=file_sha(lock_path),hparams=vars(hp),seed=lock['seed'],
          model_revision=lock['revision'],torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
          attention=model.config._attn_implementation,dtype='float32',solve_dtype='float64',tf32_matmul=False,tf32_cudnn=True,autocast=False,
          writer_tokenizer=dict(padding=tok.padding_side,bos=tok.bos_token_id,pad=tok.pad_token_id,add_bos_property=tok.add_bos_token,actual_probe_ids=tok('MEMIT history')['input_ids']),
          evaluator_tokenizer=dict(padding=evaltok.padding_side,bos=evaltok.bos_token_id,pad=evaltok.pad_token_id,actual_probe_ids=evaltok('MEMIT history')['input_ids'],kernel_padding='explicit_left',microbatch=16),
          W0_H0=w0sig,context_sha256=file_sha(lock['context']),save_checkpoints=False,exact_resume='NOT_AVAILABLE',slurm_job=os.environ.get('SLURM_JOB_ID')))
        # Native target disables gradients itself; immutable original parameter flags do not affect inference.
        prior=content(w0sig);seen=[];previous_cov={}
        for bi in range(1,101):
            current_rows=rows[(bi-1)*100:bi*100];root=output/f'B{bi:03d}';stage=f'B{bi}_ENTRY'
            entry=signature(weights,state);assert content(entry)==prior,'W_H_CHAIN'
            if previous_cov:assert cov_guard(module)==previous_cov
            assert not (root/'commit.json').exists(),'DUPLICATE_COMMIT'
            ew={k:v.detach().cpu().clone() for k,v in weights.items()};eh=state.clone();rng=rng_capture()
            save(root/'entry.json',dict(batch=bi,seen_before=len(seen),signature=entry,request_ids=[r['case_id'] for r in current_rows],request_hashes=[digest(r['requested_rewrite']) for r in current_rows],context_hash=digest(module.CONTEXT_TEMPLATES_CACHE),history_norms=[float(x.norm()) for x in state]))
            try:
                stage=f'B{bi}_NATIVE_WRITE';started=time.monotonic()
                requests=[dict(r['requested_rewrite'],case_id=int(r['case_id'])) for r in current_rows]
                with observe(module,hp,weights,state,requests,model) as counters:
                    returned,returned_history=module.apply_memit_seq_to_model(model,tok,requests,hp,copy=False,return_orig_weights=False,cache_template=None,cache_c=state)
                    assert returned is model and returned_history is state
                torch.cuda.synchronize();edit_seconds=time.monotonic()-started
                assert all(torch.isfinite(v).all() for v in weights.values()) and torch.isfinite(state).all()
                assert all(v.data_ptr()==w0sig['weights'][k]['pointer'] for k,v in weights.items())
                endpoint=signature(weights,state)
                if previous_cov:assert cov_guard(module)==previous_cov
                previous_cov=cov_guard(module)
                if bi==1:save(output/'covariance-identity.json',{str(k):dict(sha256=tensor_sha(v),shape=list(v.shape),dtype=str(v.dtype)) for k,v in module.COV_CACHE.items()})
                save(root/'native-observation.json',counters)
                updates={k:float((v.detach().cpu()-ew[k]).double().norm()) for k,v in weights.items()}
                stage=f'B{bi}_OBSERVER';t=time.monotonic()
                current=evaluate(model,evaltok,current_rows,weights,state);save(root/'current.json',current)
                full=bi in FULL_BATCHES
                past=evaluate(model,evaltok,seen,weights,state,full=full) if seen else None
                cumulative=merge(past,current,full=full)
                assert cumulative['requests']==100*bi
                cumulative['evaluation_type']='ALL_SEEN_RPN' if full else 'ALL_SEEN_REWRITE'
                cumulative['strata']=strata(cumulative,seen+current_rows)
                save(root/('seen-full.json' if full else 'seen-rewrite.json'),cumulative)
                assert signature(weights,state)==endpoint,'OBSERVER_STATE_MUTATION'
                assert cov_guard(module)==previous_cov,'OBSERVER_C0_MUTATION'
                eval_seconds=time.monotonic()-t
                receipt=dict(status='BATCH_COMMITTED',batch=bi,requests=100,seen_requests=bi*100,entry=content(entry),endpoint=content(endpoint),
                  compute_z=counters['compute_z'],solve_calls=counters['solve_calls'],history_append_layers=counters['history_append_layers'],
                  history_key_phase=counters['history_key_phase'],history_norms=[float(x.norm()) for x in state],
                  context_hash=digest(module.CONTEXT_TEMPLATES_CACHE),history_coefficient=1,covariance_guard=previous_cov,
                  layer_update_norms=updates,edit_seconds=edit_seconds,evaluation_seconds=eval_seconds,
                  target_seconds=counters['target_seconds'],key_seconds=counters['key_seconds'],solve_seconds=counters['solve_seconds'],
                  save_checkpoints=False,exact_resume='NOT_AVAILABLE',nonfinite=0,observer_mutation=0,
                  current={k:{a:b for a,b in v.items() if a!='rows'} for k,v in current['metrics'].items()},peak_gpu_bytes=torch.cuda.max_memory_allocated())
                save(root/'commit.json',receipt)
                committed.append(receipt);prior=content(endpoint);seen.extend(current_rows)
                print('MEMIT_HISTORY_BATCH_COMMITTED',bi,len(seen),edit_seconds,flush=True)
            except BaseException:
                restore(weights,ew,state,eh);rng_restore(rng)
                assert content(signature(weights,state))==content(entry)
                save(root/'rollback.json',dict(entry_W_H_exact=True,rng_restored=True,stage=stage,logical_commit=False));raise
            del ew,eh,rng,current,past,cumulative
        assert sum(c['compute_z'] for c in committed)==10000
        assert sum(c['solve_calls'] for c in committed)==500
        assert sum(c['history_append_layers'] for c in committed)==500
        stage='REDUCER';reduce_run(output,rows)
        stage='FINAL_RAM_RESTORE';restore(weights,w0,state,torch.zeros_like(state));assert content(signature(weights,state))==content(w0sig)
        save(output/'terminal.json',dict(status='COMPLETED',batches=100,requests=10000,z_calls=10000,solve_calls=500,history_append_layers=500,
           seconds=time.monotonic()-start,W0_H0_restored=True,save_checkpoints=False,exact_resume='NOT_AVAILABLE',
           manifest=[dict(path=str(p.relative_to(output)),bytes=p.stat().st_size,sha256=file_sha(p)) for p in sorted(output.rglob('*')) if p.is_file()]))
    except BaseException as e:
        cleanup=None
        if weights is not None and w0 is not None and state is not None:
            try:restore(weights,w0,state,torch.zeros_like(state))
            except BaseException as x:cleanup=repr(x)
        save(output/'failure.json',dict(status='TECHNICAL_FAILURE',stage=stage,error=repr(e),traceback=traceback.format_exc(),committed_batches=len(committed),cleanup_error=cleanup,seconds=time.monotonic()-start));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);run(p.parse_args().lock)
