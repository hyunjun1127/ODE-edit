"""One B010 native-weak BS1x100 trajectory; no persistent weights/history."""
import argparse
import copy
import resource
import sys
import time
import traceback
from pathlib import Path
from .binding import WeakRuntime
from project.run_scripts.joint_multilayer_bs10.common import read,save,record,sha,digest,tensor_sha,require,LAYERS,MILESTONES
from project.run_scripts.joint_multilayer_bs10.observations import Scorer,catalog
from project.run_scripts.joint_multilayer_bs10.final_eval import rows_for

def state(rt,anchors,accepted,batch):
    p,n,t,c=rt.rng_get()
    rng=digest(dict(python=p,numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],torch=tensor_sha(t),cuda=[tensor_sha(x) for x in c]))
    return dict(weight=rt.hashes(),history=tensor_sha(rt.M),anchors=digest(anchors),dual=digest({}),
                rng=rng,context=digest(rt.contexts),accepted_ids=list(accepted),offered_batches=batch)

def run(lock_path):
    import torch
    from scripts.fixed_counterfact import verify
    start=time.monotonic();lock=read(lock_path)
    for r in lock['source_members']+[lock['configuration'],lock['patched_compute_z']]:
        require(sha(r['path'])==r['sha256'],'FROZEN_SOURCE_INPUT')
    cfg=read(lock['configuration']['path']);require(cfg['weak']['save_checkpoints'] is False,'NO_CP')
    verify(Path(cfg['dataset']['path']).parent)
    out=Path(lock['root'])/'output';out.mkdir(exist_ok=False)
    batch=0;records=[];rt=None;scorer=None;accepted=[];anchors={}
    try:
        rt=WeakRuntime(cfg,'B010',lock['source_root'],lock['patched_compute_z']);scorer=Scorer(rt)
        data,ids,panels,current,native,neighborhood=catalog(cfg,rt.tok,'B010',rt.contexts)
        require(len(ids)==len(set(ids))==100 and len(panels)==288,'INVENTORY_100_AND_144_PAIRS')
        old_catalog=read(lock['old_catalog']['path'])
        require(sha(lock['old_catalog']['path'])==lock['old_catalog']['sha256'],'OLD_CATALOG')
        tokenrows=panels+sum(current.values(),[])+sum(native.values(),[])+sum(neighborhood.values(),[])
        require(tokenrows==old_catalog,'ORIGINAL_TOKEN_CATALOG')
        save(out/'token-catalog.json',tokenrows)
        imports={}
        for k,v in sys.modules.items():
            p=getattr(v,'__file__',None)
            if p and k.startswith(('AlphaEdit','transformers','rome','util')) and Path(p).is_file():imports[k]=record(p)
        save(out/'runtime.json',dict(source=lock['source'],config_sha=lock['configuration']['sha256'],imports=imports,
            torch=torch.__version__,device=torch.cuda.get_device_name(),save_checkpoints=False,
            exact_resume='NOT_AVAILABLE',compute_z=record(sys.modules['AlphaEdit.native_weak_compute_z'].__file__)))
        teacher={};original=[]
        for row in panels:
            m,lp,_=scorer.metric(row,category='w0_reference');original.append(m)
            if row['role'].startswith('base_') and row['label']=='true':teacher[row['row_id']]=lp
        save(out/'w0-metrics.json',original)
        rt.set_weights('CP');rt.set_rng()
        require(rt.hashes()=={str(l):rt.parent_hash[l] for l in LAYERS},'PARENT_RESTORE')
        require(tensor_sha(rt.M)==cfg['checkpoints']['B010']['history_sha256'],'PARENT_HISTORY')
        fixed=[r for r in panels if r['role'].endswith('observer')];entry=[];entry_caps={}
        for row in panels:
            m,_,caps=scorer.metric(row,teacher.get(row['row_id']),capture=row in fixed,category='entry_reference')
            entry.append(m)
            if row in fixed:entry_caps[row['row_id']]=caps
        save(out/'entry-metrics.json',entry)
        byrow={r['row_id']:r for r in entry}
        anchors={f'old:{r["case_id"]}':byrow[r['row_id']]['nll'] for r in panels if r['role']=='history_control' and r['label']=='new'}
        prev=state(rt,anchors,accepted,0)
        require(prev==read(lock['old_entry_state']['path']),'EXACT_ORIGINAL_B010_ENTRY')
        save(out/'entry-state.json',prev)

        def evaluate(rows,path,capture=False,preweights=None,category='observer'):
            values=[];weights=rt.snapshot() if capture else None
            for row in rows:
                m,_,caps=scorer.metric(row,teacher.get(row['row_id']),capture=capture,category=category)
                if capture:
                    traces=[]
                    for l in LAYERS:
                        k,v=caps[l];k0,v0=entry_caps[row['row_id']][l];dk=k.double()-k0.double()
                        w=weights[l].double();e=w-rt.w0[l].double()
                        tr=dict(layer=l,valid_tokens=k.shape[0],key_norm=float(k.double().norm()),readout_norm=float(v.double().norm()),
                            entry_key_drift=float(dk.norm()),entry_readout_change=float((v.double()-v0.double()).norm()),E_deltaK_norm=float((dk@e.T).norm()))
                        if preweights is not None:tr['actual_delta_norm']=float((w-preweights[l].double()).norm())
                        traces.append(tr)
                    m['layer_traces']=traces
                values.append(m)
            save(path,values);return values

        for batch,cid in enumerate(ids,1):
            bd=out/f'B{batch:03d}';bd.mkdir();before=state(rt,anchors,accepted,batch-1)
            require(before==prev,'ADJACENT_STATE_LINK')
            save(bd/'entry.json',dict(state=before,offered_ids=[cid],parent_commit=record(out/f'B{batch-1:03d}/commit.json') if batch>1 else record(out/'entry-state.json')))
            if batch==2:
                save(out/'initial-link.json',dict(status='INITIAL_VALID_NOT_TERMINAL',arm='NATIVE_WEAK_NLL1',
                    source=lock['source'],first_commit=record(out/'B001/commit.json'),next_entry=record(bd/'entry.json'),
                    target_fits=1,solves=5,history_appends=5,weight_history_rng_context_link=True,save_checkpoints=False))
            pre=rt.snapshot()
            if batch in MILESTONES:evaluate(fixed,bd/'fixed-pre.json',capture=True)
            req=dict(copy.deepcopy(data[cid]['requested_rewrite']),case_id=cid)
            method=rt.fit_batch([req]);method.update(accepted=True,outcome='NATIVE_WEAK_APPLIED')
            save(bd/'selection.json',method)  # No observer drives fitting or selection.
            fit=method['targets'][0]
            for j,row in enumerate(native[cid]):
                length=sum(fit['attention_mask'][j])
                require(row['input_ids']==fit['input_ids'][j][:length] and row['target_ids']==fit['target_ids']
                    and row['positions']==list(range(length-len(fit['target_ids']),length)),'ACTUAL_CONTEXT_FIT_TOKEN_BINDING')
            context_values=evaluate(native[cid],bd/'actual-write-context.json',category='actual_write_context')
            save(bd/'actual-write-context-summary.json',dict(case_id=cid,contexts=len(context_values),
                mean_nll=sum(r['nll'] for r in context_values)/len(context_values),
                latent_fit_final_nll=method['targets'][0]['stop']['final_nll'],scope='actual five-weight forward; no latent injection'))
            scores=evaluate(current[cid],bd/'current.json');accepted.append(cid)
            anchors[f'new:{cid}']=next(r['nll'] for r in scores if r['kind']=='R' and r['label']=='new')
            after=state(rt,anchors,accepted,batch);delta=[]
            for l in LAYERS:
                dw=rt.weights[l].detach().cpu().double()-pre[l].double();norm=float(dw.norm());wn=float(rt.w0[l].double().norm())
                delta.append(dict(layer=l,actual_delta_norm=norm,normalized_energy=.5*(norm/wn)**2))
            save(bd/'commit.json',dict(offered_ids=[cid],accepted=True,before=before,after=after,layer_delta=delta,
                history_appends=5,selection=record(bd/'selection.json'),anchors=anchors,dual={},save_checkpoints=False))
            prev=after;records.append(dict(batch=batch,case_id=cid,seconds=method['seconds'],stop=method['targets'][0]['stop']))
            if batch in MILESTONES:
                evaluate(fixed,bd/'fixed-post.json',capture=True,preweights=pre)
                evaluate(sum([current[c] for c in ids[:batch]],[]),bd/'all-offered.json')
                evaluate(neighborhood[cid],bd/'neighborhood.json')
            rt.check_fixed()
        # Mandatory evaluation while final weights still exist in RAM; before greedy/reducer/exit.
        versions=rt.versions();state_before=state(rt,anchors,accepted,100)
        fullrows=sum([rows_for(rt.tok,data[c]) for c in ids],[])
        full=evaluate(fullrows,out/'final-full-metrics.json',category='final_full_evaluation')
        require(len(full)==2600 and len({r['row_id'] for r in full})==2600,'FULL_2600_ROWS')
        require(rt.versions()==versions and state(rt,anchors,accepted,100)==state_before,'FINAL_EVAL_STATE_UNCHANGED')
        save(out/'final-full-eval-receipt.json',dict(raw=record(out/'final-full-metrics.json'),rows=2600,requests=100,
            same_ram_state=True,nonmutation=True,state=state_before,source=lock['source'],save_checkpoints=False))
        save(out/'final-greedy.json',[scorer.greedy(r) for cid in ids for r in current[cid] if r['label']=='new'])
        from .reduce import reduce
        reduction=reduce(out,lock)
        require(rt.fits==100 and rt.total_solves==500 and rt.loss_evals<=2500 and rt.adam<=2400,'FINAL_WORK_BUDGET')
        rt.check_fixed()
        save(out/'terminal.json',dict(status='COMPLETED',source=lock['source'],config_sha=lock['configuration']['sha256'],
            offered=100,batches=100,scientific_native_fits=rt.fits,native_solves=rt.total_solves,history_appends=500,
            loss_evaluations=rt.loss_evals,adam_updates=rt.adam,save_checkpoints=False,exact_resume='NOT_AVAILABLE',
            final_full_summary=reduction,program_seconds=time.monotonic()-start,model_load_seconds=rt.model_load_seconds,
            calls=scorer.calls,tokens=scorer.tokens,timers=scorer.seconds,scorer_backwards=scorer.backwards,
            cuda_peak_bytes=torch.cuda.max_memory_allocated(),host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            batch_records=records))
    except BaseException as exc:
        save(out/'terminal.json',dict(status='TECHNICAL_FAILED',exception=repr(exc),traceback=traceback.format_exc(),batch=batch,
            completed_batches=len(records),program_seconds=time.monotonic()-start,source=lock['source'],
            save_checkpoints=False,exact_resume='NOT_AVAILABLE',fits=rt.fits if rt else 0))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);run(p.parse_args().lock)
