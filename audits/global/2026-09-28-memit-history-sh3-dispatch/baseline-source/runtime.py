"""Native blue=False B100x100; reused sequential transactions/evaluator, no kernel edits."""
import argparse,importlib,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
from project.run_scripts.blue_alphaedit_sequential_comparison.integrity import content,digest,file_sha,restore,save,signature,tensor_artifact,tensor_sha
from .method import bind,covariance_identity
from .observation import observe
from .state import cache_guard,persist
from .evaluation import evaluate,merge
from .data import verify_fixed_dataset

def run(lock_path,output,cell,mode):
    import numpy as np,torch,transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    lock=json.loads(Path(lock_path).read_text()); spec=lock['cells'][cell]
    output=Path(output).absolute();output.mkdir(parents=True,exist_ok=False)
    stage='SOURCE_BINDING';weights=state=w0=None;committed=[];started=time.monotonic()
    try:
        for x in lock['members']:
            assert file_sha(x['path'])==x['sha256'],'SOURCE_ASSET_DRIFT:'+x['path']
        assert transformers.__version__=='4.44.2'
        stage='FROZEN_DATASET_BEFORE_MODEL'
        dataset=verify_fixed_dataset(lock)
        assert subprocess.check_output(['git','-C',lock['blue_root'],'rev-parse','HEAD'],text=True).strip()==lock['blue_head']
        sys.path.insert(0,lock['blue_root']);os.chdir(lock['blue_root'])
        random.seed(lock['seed']);np.random.seed(lock['seed']);torch.manual_seed(lock['seed']);torch.set_num_threads(8)
        stage='MODEL_LOAD'
        model=AutoModelForCausalLM.from_pretrained(lock['snapshot'],local_files_only=True,low_cpu_mem_usage=True,attn_implementation='eager').cuda().eval()
        tok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True)
        tok.add_bos_token=False;tok.pad_token_id=tok.eos_token_id
        evaltok=AutoTokenizer.from_pretrained(lock['snapshot'],local_files_only=True);evaltok.pad_token_id=evaltok.eos_token_id
        assert tok.padding_side==evaltok.padding_side=='right'
        assert {p.dtype for p in model.parameters()}=={torch.float32}
        module,hp,state,binding,apply=bind(spec['method'],spec['config'],lock,model,None)
        assert hp.layers==spec['layers']
        weights={hp.rewrite_module_tmp.format(l)+'.weight':module.nethook.get_parameter(model,hp.rewrite_module_tmp.format(l)+'.weight') for l in hp.layers}
        w0={k:v.detach().cpu().clone() for k,v in weights.items()};w0sig=signature(weights,state)
        sample=json.loads(Path(lock['sample']).read_text())
        assert digest(sample['records'])==sample['ordered_root']
        byid={int(r['case_id']):r for r in dataset}
        for row in sample['records']:assert digest(byid[row['case_id']])==row['raw_record_sha256']
        allrows=[byid[r['case_id']] for r in sample['records']]
        assert mode=='main', 'GPU_SMOKE_SKIPPED_USER_DIRECTED'
        batches=[allrows[i:i+100] for i in range(0,10000,100)]
        del dataset,byid,allrows
        save(output/'runtime.json',dict(mode=mode,cell=cell,spec=spec,lock_sha256=file_sha(lock_path),hparams=vars(hp),blue_head=lock['blue_head'],helper_head=lock['source_head'],local_source_sha256=lock['local_source_sha256'],
             torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),dtype='torch.float32',native_scalar_policy='MEMIT solve FP64; AlphaEdit native FP32; unchanged',attention=model.config._attn_implementation,tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,autocast=torch.is_autocast_enabled(),
             parameter_elements=sum(p.numel() for p in model.parameters()),parameter_tensors=sum(1 for p in model.parameters()),model_revision=lock['revision'],W0=w0sig,binding=binding,slurm_job=os.environ.get('SLURM_JOB_ID'),
             writer_tokenizer=dict(padding=tok.padding_side,bos=tok.bos_token_id,pad=tok.pad_token_id,eos=tok.eos_token_id,add_bos_token=tok.add_bos_token),evaluator_tokenizer=dict(padding=evaltok.padding_side,pad=evaltok.pad_token_id,add_bos_token=getattr(evaltok,'add_bos_token','NOT_EXPOSED')),model_load_seconds=time.monotonic()-started,scientific_promotion=False))
        seen=[];prior=content(w0sig);previous_cov={}
        for bi,rows in enumerate(batches,1):
            root=output/f'B{bi:03d}';stage=f'B{bi}_ENTRY'
            entry=signature(weights,state);assert content(entry)==prior,'W_HISTORY_CHAIN'
            assert not previous_cov or cache_guard(module)==previous_cov,'STATIC_COVARIANCE_CHAIN'
            ew={k:v.detach().cpu().clone() for k,v in weights.items()};em=state.clone()
            save(root/'entry.json',dict(signature=entry,batch=bi,seen_before=len(seen),request_ids=[r['case_id'] for r in rows],request_hashes=[digest(r['requested_rewrite']) for r in rows],context_hash=digest(module.CONTEXT_TEMPLATES_CACHE),history_entries=len(seen) if spec['method']=='AlphaEdit' else None))
            try:
                stage=f'B{bi}_NATIVE_WRITE';edit_start=time.monotonic()
                requests=[dict(r['requested_rewrite'],case_id=int(r['case_id'])) for r in rows]
                with observe(module,hp,weights,state,requests,model,spec['method'],mode=='smoke') as counters:apply(tok,requests)
                edit_seconds=time.monotonic()-edit_start
                assert all(torch.isfinite(v).all() for v in weights.values()) and torch.isfinite(state).all(),'NONFINITE_ENDPOINT'
                assert all(v.data_ptr()==w0sig['weights'][k]['pointer'] for k,v in weights.items())
                endpoint=signature(weights,state)
                if previous_cov:assert cache_guard(module)==previous_cov,'STATIC_COVARIANCE_MUTATION'
                previous_cov=cache_guard(module)
                targets=counters.pop('_target_tensors')
                counters['target_artifact']=tensor_artifact(root/'native-targets.pt',dict(values=targets,identities=counters['z']))
                del targets
                save(root/'native-observation.json',counters);save(root/'contexts.json',module.CONTEXT_TEMPLATES_CACHE)
                update={}
                for k,v in weights.items():
                    d=v.detach().cpu().double()-ew[k].double();n=float(d.norm())
                    update[k]=dict(norm=n,squared_norm=float(d.square().sum()),relative_norm=n/float(ew[k].double().norm()))
                total=sum(v['norm'] for v in update.values())
                for v in update.values():v['magnitude_share']=v['norm']/total if total else None
                del d
                stage=f'B{bi}_EVALUATION';ev=time.monotonic()
                current=evaluate(model,evaltok,rows,weights,state);save(root/'current.json',current)
                heavy=bi in lock['checkpoint_batches'] or mode=='smoke'
                if heavy:
                    past=evaluate(model,evaltok,seen,weights,state) if seen else None
                    full=merge(past,current,content(endpoint));save(root/'seen-full.json',full)
                    assert full['requests']==len(seen)+len(rows)
                    save(root/'seen-rewrite.json',dict(evaluation_type='ALL_SEEN_REWRITE_AT_CURRENT_W',requests=full['requests'],state=content(endpoint),metrics={'RS':full['metrics']['RS']},reused_from='seen-full.json'))
                    del past,full
                else:
                    past_rw=evaluate(model,evaltok,seen,weights,state,full=False) if seen else None
                    rwrows=(past_rw['metrics']['RS']['rows'] if past_rw else [])+current['metrics']['RS']['rows']
                    assert len({x['identity'] for x in rwrows})==len(seen)+len(rows)
                    save(root/'seen-rewrite.json',dict(evaluation_type='ALL_SEEN_REWRITE_AT_CURRENT_W',requests=len(rwrows),state=content(endpoint),metrics={'RS':dict(rows=rwrows,numerator=sum(x['success'] for x in rwrows),denominator=len(rwrows))},current_rows_reused=True))
                    del past_rw,rwrows
                assert signature(weights,state)==endpoint,'EVALUATOR_MUTATION'
                assert cache_guard(module)==previous_cov,'EVALUATOR_COV_MUTATION'
                eval_seconds=time.monotonic()-ev
                cp=None
                if heavy:
                    cp=persist(root/'W-method-state.pt',weights,state,dict(batch=bi,seen_ids=[r['case_id'] for r in seen+rows],state=content(endpoint),lock_sha256=file_sha(lock_path),sample_root=sample['ordered_root'],source=lock['local_source_sha256'],base_model_revision=lock['revision'],stats_members=lock['stats_members']),module,spec['method'])
                receipt=dict(status='BATCH_COMMITTED',batch=bi,requests=len(rows),seen_requests=len(seen)+len(rows),entry=content(entry),endpoint=content(endpoint),W_pointer_exact=True,history_append_passes=counters['history_append_passes'],history_entries_in=len(seen) if spec['method']=='AlphaEdit' else None,history_entries_out=len(seen)+len(rows) if spec['method']=='AlphaEdit' else None,compute_z=counters['compute_z'],solve_calls=counters['solve_calls'],context_hash=digest(module.CONTEXT_TEMPLATES_CACHE),covariance_guard=previous_cov,layer_updates=update,edit_seconds=edit_seconds,target_seconds=counters['target_seconds'],key_seconds=counters['key_seconds'],solve_seconds=counters['solve_seconds'],evaluation_seconds=eval_seconds,checkpoint=cp,nonfinite=0,evaluator_mutation=0,current={k:{a:b for a,b in v.items() if a!='rows'} for k,v in current['metrics'].items()},peak_gpu_bytes=torch.cuda.max_memory_allocated())
                save(root/'commit.json',receipt);committed.append(receipt);prior=content(endpoint);seen.extend(rows)
                if bi==1:save(output/'first-batch-cost.json',dict(edit_seconds=edit_seconds,evaluation_seconds=eval_seconds,checkpoint_bytes=cp['bytes'] if cp else 0,first_batch_no_SH_wait_required=True))
                print('BASE_NATIVE_LIFELONG_BATCH_COMMITTED',cell,bi,len(seen),edit_seconds,flush=True)
            except BaseException:
                restore(weights,ew,state,em);assert content(signature(weights,state))==content(entry)
                save(root/'rollback.json',dict(entry_exact=True,stage=stage,accepted_denominator=0));raise
            del ew,em,current
        stage='FINAL_W0_RESTORE';restore(weights,w0,state,torch.zeros_like(state))
        assert content(signature(weights,state))==content(w0sig)
        assert all(v.data_ptr()==w0sig['weights'][k]['pointer'] for k,v in weights.items())
        save(output/'terminal.json',dict(status='TERMINAL_VALID',mode=mode,cell=cell,batches=len(committed),requests=len(seen),W_chain_links=len(committed)-1,compute_z=sum(r['compute_z'] for r in committed),solve_calls=sum(r['solve_calls'] for r in committed),history_append_passes=sum(r['history_append_passes'] for r in committed),cold_reset_count=1,W0_cache_bytes_pointer_restore=True,version_restore='NOT_CLAIMED: native copy increments counters; evaluator counters exact',nonfinite=0,failure=0,seconds=time.monotonic()-started,manifest_members=[dict(path=str(p.relative_to(output)),sha256=file_sha(p),bytes=p.stat().st_size) for p in sorted(output.rglob('*')) if p.is_file()],scientific_promotion=False))
    except BaseException as e:
        error=None
        if weights is not None and w0 is not None and state is not None:
            try:restore(weights,w0,state,torch.zeros_like(state))
            except BaseException as re:error=repr(re)
        save(output/'failure.json',dict(status='TYPED_HOLD',stage=stage,error=repr(e),traceback=traceback.format_exc(),prior_committed_batches=len(committed),prior_committed_requests=sum(r['requests'] for r in committed),final_denominator=0,W0_restore_error=error,seconds=time.monotonic()-started));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--output',required=True);p.add_argument('--cell',type=int,required=True);p.add_argument('--mode',choices=['smoke','main'],required=True)
    a=p.parse_args();run(a.lock,a.output,a.cell,a.mode)
