"""One immutable parent/layer branch, integrated checks, 100 committed BS1 edits."""
import argparse
import copy
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from .common import *
from .observations import Observer, build_rows, risk


def cache_load(path):
    import torch
    return torch.load(path,map_location='cpu',weights_only=True)


def comparison(now,ref):
    import torch
    result={}
    require(now['row_id']==ref['row_id'] and now['input_sha']==ref['input_sha'] and now['positions']==ref['positions'],'CAPTURE_IDENTITY')
    for l,k in now['keys'].items():
        old=ref['keys'][l];require(k.shape==old.shape,'KEY_SHAPE')
        delta=k.double()-old.double();v=now['values'][l].double()-ref['values'][l].double()
        result[str(l)]=dict(key_delta_norm=float(delta.norm()),key_max_abs=float(delta.abs().max()),
            key_reference_norm=float(old.double().norm()),readout_delta_norm=float(v.norm()))
    return result


def evaluate_set(rt,observer,rows,label,out,*,cache_stage=None,patch_mode=None):
    import torch
    metrics=[];norms=[];w0dir=out/'reference/W0';entrydir=out/'reference/entry'
    for row in rows:
        rid=row['row_id']; teacher=None;w0=None;entry=None
        if (w0dir/f'{rid}.pt').exists():
            w0=cache_load(w0dir/f'{rid}.pt');teacher=w0['teacher_logp']
        if (entrydir/f'{rid}.pt').exists():entry=cache_load(entrydir/f'{rid}.pt')
        patch=None
        if patch_mode:
            require(entry is not None,'PATCH_ENTRY_MISSING')
            patch=dict(mode=patch_mode,entry_key=entry['keys'][8],D8=rt.patch_D8,seed=rt.config['observer']['patch_seed'])
        m,payload=observer.evaluate(row,patch=patch,teacher=teacher)
        if row['role']=='base_sensor' and row['label']=='true':
            rt.latest_sensor_keys[row['row_id']]=payload['keys'][rt.layer]
        if cache_stage:
            if cache_stage=='W0' and row['role'] not in ('base_sensor','base_observer'):
                # W0 scalar reference retained; no unnecessary full-vocab distribution.
                payload['teacher_logp']=None
            tensor_save(out/f'reference/{cache_stage}/{rid}.pt',payload)
        if w0 is not None:m['W0_to_current']=comparison(payload,w0)
        if entry is not None:
            m['entry_to_current']=comparison(payload,entry)
            k=payload['keys'][rt.layer];kref=entry['keys'][rt.layer]
            c=rt.config['observer'];equal=torch.allclose(k,kref,rtol=c['key_rtol'],atol=c['key_atol'])
            m['selected_key_invariant']=bool(equal)
            if rt.layer==8 and label=='S100-post':
                D=(rt.cp['weights']['model.layers.8.mlp.down_proj.weight']-rt.w0[8]).double()
                action=torch.nn.functional.linear((k-kref).double(),D)
                m['L8_negative_control_action_norm']=float(action.norm())
            # Persist per-case excess first, even when fatal.
            if not equal:
                save(out/f'failure-evidence/{label}-{rid}.json',m)
                raise RuntimeError('SELECTED_KEY_INVARIANCE:'+rid)
        metrics.append(m)
    receipt=save(out/f'observations/{label}.json',dict(label=label,rows=metrics,count=len(metrics),
        evaluator='task-v1/MB1/BOS/separate-leading-space-target/FP32/TF-allvalid',selection_feedback=False))
    return metrics,receipt


def snapshot(rt,observer,rows,out,step,stream,commits):
    import torch
    require(step in (50,100),'SNAPSHOT_SCOPE')
    l=rt.layer;before_rng=rt.rng_get();rt.check_nonselected(full=True)
    representative=rows[0];before,_=observer.evaluate(representative)
    full=rt.weights[l].detach().cpu().clone()
    require(full.shape==(4096,14336) and full.dtype==torch.float32,'SNAPSHOT_SCHEMA')
    metadata=dict(instruction_id=NONCE,checkpoint=rt.checkpoint,physical_layer=l,step=step,
        parameter=f'model.layers.{l}.mlp.down_proj.weight',parent_model_revision=REVISION,
        parent_checkpoint=rt.config['checkpoints'][rt.checkpoint]['sha256'],ordered_ids=[int(r['case_id']) for r in stream[:step]],
        last_case_id=int(stream[step-1]['case_id']),source=rt.config['execution_source'],hparams=dict(rt.config['hparams'],layers=[l]),
        singleton_layers=[l],context_sha=digest(rt.contexts),tokenizer_sha=rt.tokenizer_sha,commit=commits[-1],
        tensor_sha=tensor_sha(full),exact_editor_resume='NOT_AVAILABLE',purpose='MODEL_RECONSTRUCTION_ONLY')
    rec=tensor_save(out/f'weights/S{step:03d}.pt',dict(weight=full,metadata=metadata))
    loaded=cache_load(rec['path']);require(torch.equal(loaded['weight'],full),'SNAPSHOT_RELOAD_BITWISE')
    # Fresh parent restoration, not subtract/add; all other original model params untouched.
    rt.set_weights('parent')
    with torch.no_grad():rt.weights[l].copy_(loaded['weight'])
    after,_=observer.evaluate(representative)
    evidence=dict(snapshot=rec,tensor_bitwise=True,nll_before=before['nll'],nll_after=after['nll'],
        nll_abs=abs(before['nll']-after['nll']),tokens_equal=before['token_predictions']==after['token_predictions'],
        answer_logp_byte_equal=before['answer_logp_sha256']==after['answer_logp_sha256'],
        reconstructed_weight_sha=tensor_sha(rt.weights[l]),reconstruction='original frozen nonedited parameters + all parent5 W + selected overlay',
        editor_state_saved=False)
    save(out/f'weights/S{step:03d}-parity.json',evidence)
    require(evidence['reconstructed_weight_sha']==metadata['tensor_sha'] and evidence['tokens_equal'] and evidence['answer_logp_byte_equal']
        and evidence['nll_abs']<=rt.config['observer']['parity_nll_atol'],'RECONSTRUCTION_OUTPUT_PARITY')
    rt.check_nonselected(full=True);rt.rng_set(before_rng)
    return rec


def run(config,index,repo,out):
    import torch
    from .native import Runtime
    require(0<=index<15,'BRANCH_INDEX')
    b=(10,50,90)[index//5];layer=4+index%5;checkpoint=f'B{b:03d}';branch=f'{checkpoint}-L{layer}'
    started=time.monotonic();out.mkdir(parents=True,exist_ok=False)
    save(out/'STARTED.json',dict(branch=branch,index=index,job=os.environ.get('SLURM_JOB_ID'),array_job=os.environ.get('SLURM_ARRAY_JOB_ID'),
        config_sha=config['config_sha'],source=config['execution_source'],instruction_id=NONCE))
    rt=None;commits=[];snapshots=[]
    try:
        rt=Runtime(config,checkpoint,layer,repo)
        rt.latest_sensor_keys={}
        rt.tokenizer_sha=digest(dict(vocab=rt.tok.get_vocab(),special=rt.tok.special_tokens_map,
            backend=rt.tok.backend_tokenizer.to_str(),declared_add_bos=rt.tok.add_bos_token))
        rows=build_rows(config,rt.tok);rows=[r for r in rows if r['checkpoint'] in ('all',checkpoint)]
        require(len(rows)==844,'BRANCH_PANEL_ROW_COUNT')
        save(out/'token-rows.json',dict(rows=rows,tokenizer_sha=rt.tokenizer_sha,rows_sha=digest(rows),writer_and_observer_bos_semantics='native backend unchanged'))
        imports=[]
        for name,module in sorted(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if path and Path(path).is_file() and (str(NATIVE) in path or str(DEPS) in path or 'temporal_routing_diagnostic' in path):
                imports.append(dict(module=name,**record(path)))
        save(out/'runtime-imports.json',dict(imports=imports,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
            autocast=False,attention=rt.model.config._attn_implementation,precision='float32',parent_rng_restored=True))
        observer=Observer(rt,out)
        # Setup is a bounded forward-only reference over the locked same-prefix rows.
        evaluate_set(rt,observer,rows,'W0',out,cache_stage='W0')
        rt.set_weights('parent');rt.set_rng()
        for l in LAYERS:require(tensor_sha(rt.weights[l])==rt.parent_hash[l],'FULL_PARENT_RESTORE')
        require(tensor_sha(rt.M)==config['checkpoints'][checkpoint]['history_sha256'],'FULL_HISTORY_RESTORE')
        entry,entry_rec=evaluate_set(rt,observer,rows,'entry',out,cache_stage='entry')
        sensors=[r for r in rows if r['role'].endswith('sensor')]
        fixed_observers=[r for r in rows if r['role'].endswith('observer')]
        prev_sensor={r['row_id']:r for r in entry if r['role'].endswith('sensor')}
        stream=csv_rows(Path(config['design'])/'continuation-ids.csv')
        data={int(r['case_id']):r for r in read(config['dataset']['path'])}
        save(out/'G0-parent-and-panels.json',dict(status='PASS',parent_restored=True,rows=len(rows),W5=True,M5=True,
            initial_observation=entry_rec,source=config['execution_source'],config_sha=config['config_sha'],
            slot_mapping=dict(physical_layer=layer,parent_slot=layer-4,singleton_slot=0),new_native_fits=0))
        for step,x in enumerate(stream,1):
            current=[r for r in rows if r['role']=='continuation' and r['arrival']==step and r['kind'] in ('R','P')]
            detail=fixed_observers+[r for r in rows if r['role']=='continuation' and
                ((r['arrival']<=step and r['kind'] in ('R','P')) or (r['arrival']==step and r['kind']=='N'))]
            if step in MILESTONES:evaluate_set(rt,observer,detail,f'S{step:03d}-pre',out)
            entry_link=dict(step=step,case_id=int(x['case_id']),previous_commit=commits[-1] if commits else None,
                weight_sha=tensor_sha(rt.weights[layer]),history_sha=tensor_sha(rt.M[layer-4]),source=config['execution_source'])
            if commits:
                prior=read(commits[-1]['path'])['native']
                require(entry_link['weight_sha']==prior['weight_after_sha'] and entry_link['history_sha']==prior['history_after_sha'],'COMMIT_NEXT_ENTRY')
            entry_receipt=save(out/f'entries/S{step:03d}.json',entry_link)
            if step==2:
                save(out/'INITIAL-step1-to-step2.json',dict(status='PASS',first_commit=commits[0],second_entry=entry_receipt,
                    history_appends=1,actual_committed_fits=1,all_15_registration_is_external=True,
                    source=config['execution_source'],config_sha=config['config_sha']))
            record_=data[int(x['case_id'])];request=dict(record_['requested_rewrite'],case_id=int(x['case_id']))
            before,native=rt.fit(request)
            commits.append(save(out/f'commits/S{step:03d}.json',dict(step=step,case_id=int(x['case_id']),entry=entry_receipt,
                native=native,finite_commit=True,functional_success_not_a_commit_gate=True)))
            if step==1:rt.check_nonselected(full=True)
            sensor_metrics,_=evaluate_set(rt,observer,sensors,f'S{step:03d}-sensors',out)
            risks=[]
            for r in sensors:
                if r['role']!='base_sensor' or r['label']!='true':continue
                rid=r['row_id'];k0=cache_load(out/f'reference/W0/{rid}.pt')['keys'][layer]
                kt=rt.latest_sensor_keys[rid]
                risks.append(dict(row_id=rid,case_id=r['case_id'],**risk(before,rt.weights[layer],rt.w0[layer],k0,kt,config['observer']['epsilon'])))
            fs=[]
            for m in sensor_metrics:
                desired='true' if m['role']=='base_sensor' else 'new'
                if m['label']!=desired:continue
                old=prev_sensor[m['row_id']];metric='w0_kl' if m['role']=='base_sensor' else 'nll'
                fs.append(dict(row_id=m['row_id'],case_id=m['case_id'],role=m['role'],metric=metric,
                    before=old[metric],after=m[metric],F=m[metric]-old[metric]))
            save(out/f'sensors/S{step:03d}.json',dict(step=step,questions=risks,functional=fs,
                normalized_means={k:sum(r[k] for r in risks)/len(risks) for k in ('A','B','C')},
                raw_means={k:sum(r[k] for r in risks)/len(risks) for k in ('A_raw','B_raw','C_raw')},
                reference='A/B/C all valid TF input tokens; baseF answer-token-mean KL; historyF new-target mean NLL'))
            del before
            prev_sensor={m['row_id']:m for m in sensor_metrics}
            evaluate_set(rt,observer,current,f'S{step:03d}-at-write',out)
            if step in MILESTONES:
                evaluate_set(rt,observer,detail,f'S{step:03d}-post',out);rt.check_nonselected(full=True)
            if step in (50,100):snapshots.append(snapshot(rt,observer,current,out,step,stream,commits))
        if layer==4:
            rt.patch_D8=(rt.cp['weights']['model.layers.8.mlp.down_proj.weight']-rt.w0[8]).cuda()
            patch_rows=fixed_observers+[r for r in rows if r['role']=='continuation' and r['kind'] in ('R','P')]
            # Native is the identical subset of the already stored S100-post observation.
            native_rows={r['row_id']:r for r in read(out/'observations/S100-post.json')['rows']}
            zero,_=evaluate_set(rt,observer,patch_rows,'patch-zero',out,patch_mode='zero')
            errors=[dict(row_id=r['row_id'],nll_abs=abs(r['nll']-native_rows[r['row_id']]['nll']),
                predictions_equal=r['token_predictions']==native_rows[r['row_id']]['token_predictions'],
                answer_logp_byte_equal=r['answer_logp_sha256']==native_rows[r['row_id']]['answer_logp_sha256']) for r in zero]
            save(out/'patch-zero-parity.json',dict(rows=errors,native='REUSED_S100_POST_EXACT_ROWS',additional_panel_passes=1))
            require(all(r['nll_abs']<=config['observer']['parity_nll_atol'] and r['predictions_equal'] and r['answer_logp_byte_equal'] for r in errors),'ZERO_HOOK_PARITY')
            for mode in ('targeted','random'):evaluate_set(rt,observer,patch_rows,'patch-'+mode,out,patch_mode=mode)
        if layer==8:
            rr=read(out/'observations/S100-post.json')['rows']
            save(out/'L8-zero-negative-control.json',dict(status='MEASURED',evidence='actual S100-post selected key and conditional D8 response, no extra forward',
                formula='D8*(K8,n-K8,entry)',n=100,rows=len(rr),
                max_action_norm=max(r['L8_negative_control_action_norm'] for r in rr)))
        rt.check_nonselected(full=True)
        cost=dict(fits=rt.fits,loss_evaluations=rt.loss_evals,adam_updates=rt.adam,native_seconds=rt.native_seconds,
            observer_calls=observer.calls,observer_seconds=observer.seconds,model_load_seconds=rt.model_load_seconds,
            program_seconds=time.monotonic()-started,peak_gpu_allocated=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved=torch.cuda.max_memory_reserved(),peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            nested_timers_not_additive=True,target_vs_solve='NOT_SEPARATED',checkpoint_saved=True,snapshots=len(snapshots),exact_editor_resume='NOT_AVAILABLE')
        save(out/'cost.json',cost)
        save(out/'COMPLETED.json',dict(status='COMPLETED',branch=branch,commits=100,snapshots=snapshots,
            cost=cost,config_sha=config['config_sha'],source=config['execution_source'],instruction_id=NONCE))
    except BaseException as exc:
        save(out/'TECHNICAL_FAILED.json',dict(status='TECHNICAL_FAILED',branch=branch,exception=repr(exc),traceback=traceback.format_exc(),
            completed_commit_receipts=commits,attempted_fits=rt.fits if rt else 0,seconds=time.monotonic()-started,
            snapshots=snapshots,source=config['execution_source'],exact_editor_resume='NOT_AVAILABLE'))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--index',required=True,type=int)
    args=p.parse_args();lock=read(args.lock);config=read(lock['config']['path'])
    require(sha(lock['config']['path'])==lock['config']['sha256'],'CONFIG_DRIFT')
    for m in lock['source_members']:require(sha(Path(lock['source_root'])/m['path'])==m['sha256'],'SOURCE_DRIFT')
    config['config_sha']=lock['config']['sha256'];config['execution_source']=lock['source_commit']
    run(config,args.index,lock['source_root'],Path(lock['output'])/f'branch-{args.index:02d}')


if __name__=='__main__':main()
