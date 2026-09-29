"""One single-writer trajectory; normal joint rejection advances offered cursor."""
import argparse
import copy
import os
import pickle
import resource
import sys
import time
import traceback
from .common import *
from .runtime import Runtime
from .observations import Scorer,catalog
from .solver import solve

def rng_sha(rt):
    p,n,t,c=rt.rng_get()
    return digest(dict(python=p,numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],torch=tensor_sha(t),cuda=[tensor_sha(x) for x in c]))

def state(rt,anchors,dual,accepted,batch):
    return dict(weight=rt.hashes(),history=tensor_sha(rt.M),anchors=digest(anchors),dual=digest(dual),
                rng=rng_sha(rt),context=digest(rt.contexts),accepted_ids=list(accepted),offered_batches=batch)

def snapshot(rt,scorer,row,out,batch,metadata,early=False):
    import torch
    weights=rt.snapshot();before,lp,_=scorer.metric(row,category='snapshot_parity')
    payload=dict(weights={f'model.layers.{l}.mlp.down_proj.weight':weights[l] for l in LAYERS},metadata=metadata)
    path=Path(out)/'weights'/metadata['trajectory']/f'T{batch:03d}.pt'
    rec=tensor_save(path,payload);loaded=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    require(set(loaded['weights'])==set(payload['weights']),'SNAPSHOT_KEYS')
    for l in LAYERS:
        require(torch.equal(weights[l],loaded['weights'][f'model.layers.{l}.mlp.down_proj.weight']),'SNAPSHOT_BITWISE')
    # Original immutable nonedited parameters remain in the actual model. Rebuild five edited slots.
    rt.set_weights('W0');rt.restore({l:loaded['weights'][f'model.layers.{l}.mlp.down_proj.weight'] for l in LAYERS})
    after,lp2,_=scorer.metric(row,category='snapshot_parity');rt.check_fixed()
    raw=dict(before=before,after=after,exact_logp=torch.equal(lp,lp2),early_replacement=early,
        tensor_hashes=rt.hashes(),file=rec,exact_editor_resume='NOT_AVAILABLE',offered_batches=batch)
    save(path.with_suffix('.receipt.json'),raw)
    require(torch.equal(lp,lp2),'SNAPSHOT_MODEL_PARITY')
    return raw

def run(lock_path,index):
    import torch
    start=time.monotonic();lock=read(lock_path);config=read(lock['configuration']['path'])
    require(sha(lock['configuration']['path'])==lock['configuration']['sha256'],'CONFIG_CHANGED')
    validate_execution(config)
    require(lock['instruction_id']==NONCE and 0<=index<9,'LOCK_ID')
    for member in lock['source_members']:require(sha(member['path'])==member['sha256'],'SOURCE_CHANGED')
    spec=config['trajectories'][index];out=Path(lock['attempt'])/'output'/spec['name'];out.mkdir(parents=True,exist_ok=False)
    rt=None;last_finite=None;snaps=[];batch=0;accepted=[];anchors={};dual={};records=[];scorer=None
    try:
        rt=Runtime(config,spec['checkpoint'],lock['source_root']);scorer=Scorer(rt)
        data,ids,panels,current,native,neighborhood=catalog(config,rt.tok,spec['checkpoint'],rt.contexts)
        tokenrows=panels+sum(current.values(),[])+sum(native.values(),[])+sum(neighborhood.values(),[])
        save(out/'token-catalog.json',tokenrows)
        save(out/'runtime.json',dict(instruction_id=NONCE,source=lock['source'],config_sha=lock['configuration']['sha256'],
            checkpoint=config['checkpoints'][spec['checkpoint']],imports={k:str(getattr(v,'__file__','')) for k,v in sys.modules.items() if k.startswith(('AlphaEdit','transformers','rome','util'))},
            model_revision=REVISION,torch=torch.__version__,device=torch.cuda.get_device_name(),contexts=rt.contexts,settings=config['settings']))
        teacher={};original=[]
        for row in panels:
            metric,lp,_=scorer.metric(row,category='w0_reference');original.append(metric)
            if row['role'].startswith('base_') and row['label']=='true':teacher[row['row_id']]=lp
        tensor_save(out/'w0-teacher.pt',teacher);save(out/'w0-metrics.json',original)
        rt.set_weights('CP');rt.set_rng();require(rt.hashes()=={str(l):rt.parent_hash[l] for l in LAYERS},'PARENT_RESTORE')
        require(tensor_sha(rt.M)==config['checkpoints'][spec['checkpoint']]['history_sha256'],'PARENT_HISTORY_RESTORE')
        base=[r for r in panels if r['role']=='base_control' and r['label']=='true']
        history={f'old:{r["case_id"]}':r for r in panels if r['role']=='history_control' and r['label']=='new'}
        fixed=[r for r in panels if r['role'].endswith('observer')]
        entry_metrics=[]
        for row in panels:
            metric,lp,caps=scorer.metric(row,teacher.get(row['row_id']),capture=row in fixed,category='entry_reference')
            entry_metrics.append(metric)
            if row in fixed:tensor_save(out/'entry-captures'/f'{row["row_id"]}.pt',caps)
        save(out/'entry-metrics.json',entry_metrics)
        metric_by={r['row_id']:r for r in entry_metrics}
        anchors={h:metric_by[r['row_id']]['nll'] for h,r in history.items()}
        base_anchor=sum(metric_by[r['row_id']]['w0_kl'] for r in base)/len(base)
        prev=state(rt,anchors,dual,accepted,0);last_finite=rt.snapshot()
        save(out/'entry-state.json',prev)

        def evaluate(rows,path,capture=False,preweights=None):
            values=[]
            cpu_weights=rt.snapshot() if capture else None
            for row in rows:
                m,lp,caps=scorer.metric(row,teacher.get(row['row_id']),capture=capture)
                if capture:
                    ref=torch.load(out/'entry-captures'/f'{row["row_id"]}.pt',map_location='cpu',weights_only=True)
                    traces=[]
                    for l in LAYERS:
                        k,v=caps[l];k0,v0=ref[l];dk=k.double()-k0.double()
                        w=cpu_weights[l].double();e=w-rt.w0[l].double()
                        trace=dict(layer=l,valid_tokens=k.shape[0],key_norm=float(k.double().norm()),readout_norm=float(v.double().norm()),
                            entry_key_drift=float(dk.norm()),entry_readout_change=float((v.double()-v0.double()).norm()),E_deltaK_norm=float((dk@e.T).norm()))
                        if preweights is not None:trace['actual_delta_norm']=float((w-preweights[l].double()).norm())
                        traces.append(trace)
                    m['layer_traces']=traces
                values.append(m)
            save(path,values);return values

        for batch in range(1,STEPS+1):
            offered=ids[(batch-1)*BATCH_SIZE:batch*BATCH_SIZE];bd=out/f'B{batch:03d}';bd.mkdir()
            counts_before=dict(calls=dict(scorer.calls),tokens=dict(scorer.tokens),backwards=scorer.backwards,
                backward_tokens=scorer.backward_tokens,backward_seconds=scorer.backward_seconds)
            before=state(rt,anchors,dual,accepted,batch-1)
            require(before==prev,'ADJACENT_STATE_LINK')
            save(bd/'entry.json',dict(state=before,parent_commit=record(out/f'B{batch-1:03d}'/'commit.json') if batch>1 else record(out/'entry-state.json'),offered_ids=offered))
            if batch==2:save(out/'initial-link.json',dict(scope='one trajectory B1 accepted_or_normal_reject to B2 entry',
                 first_commit=record(out/'B001/commit.json'),next_entry=record(out/'B002/entry.json'),arm=spec['arm']))
            last_finite=rt.snapshot()
            if batch in MILESTONES:evaluate(fixed,bd/'fixed-pre.json',capture=True)
            requests=[dict(copy.deepcopy(data[c]['requested_rewrite']),case_id=c) for c in offered]
            if spec['arm']=='NATIVE':
                method=rt.fit_batch(requests);method.update(accepted=True,outcome='NATIVE_APPLIED');preweights=last_finite
            else:
                method,preweights=solve(rt,scorer,requests,{c:current[c] for c in offered},{c:native[c] for c in offered},
                    base,teacher,history,anchors,base_anchor,dual,spec['arm'],bd/'solver',batch)
                dual=method['dual']
            # This seal precedes P/N/observer evaluation. No observer feeds the decision.
            save(bd/'selection.json',method)
            scores=evaluate(sum([current[c] for c in offered],[]),bd/'current.json')
            if method['accepted']:
                accepted.extend(offered)
                for c in offered:
                    row=next(r for r in current[c] if r['kind']=='R' and r['label']=='new');h=f'new:{c}'
                    history[h]=row;anchors[h]=next(r['nll'] for r in scores if r['row_id']==row['row_id'])
            elif spec['arm']!='NATIVE':
                require(rt.hashes()==before['weight'] and tensor_sha(rt.M)==before['history'],'REJECT_ATOMICITY')
            after=state(rt,anchors,dual,accepted,batch)
            delta=[]
            for l in LAYERS:
                dw=rt.weights[l].detach().cpu().double()-preweights[l].double();norm=float(dw.norm());wn=float(rt.w0[l].double().norm())
                delta.append(dict(layer=l,actual_delta_norm=norm,normalized_energy=.5*(norm/wn)**2))
            save(bd/'commit.json',dict(offered_ids=offered,accepted=method['accepted'],before=before,after=after,layer_delta=delta,
                 anchors=anchors,dual=dual,history_appends=len(method['history']),selection=record(bd/'selection.json'),
                 counts_before=counts_before,counts_at_commit=dict(calls=dict(scorer.calls),tokens=dict(scorer.tokens),backwards=scorer.backwards,
                     backward_tokens=scorer.backward_tokens,backward_seconds=scorer.backward_seconds)))
            prev=after;last_finite=rt.snapshot();records.append(dict(batch=batch,offered=BATCH_SIZE,accepted=method['accepted'],seconds=method['seconds']))
            if batch in MILESTONES:
                evaluate(fixed,bd/'fixed-post.json',capture=True,preweights=preweights)
                evaluate(sum([current[c] for c in ids[:batch*BATCH_SIZE]],[]),bd/'all-offered.json')
                evaluate(sum([neighborhood[c] for c in offered],[]),bd/'neighborhood.json')
            if batch in SAVE_STEPS:
                metadata=dict(instruction_id=NONCE,trajectory=spec['name'],offered_ids=ids[:batch*BATCH_SIZE],accepted_ids=accepted,
                   checkpoint=config['checkpoints'][spec['checkpoint']],source=lock['source'],config_sha=lock['configuration']['sha256'],
                   contexts=rt.contexts,anchors=anchors,base_anchor=base_anchor,dual=dual,state=after,exact_editor_resume='NOT_AVAILABLE')
                snaps.append(snapshot(rt,scorer,current[offered[0]][0],out,batch,metadata))
                # Restoring identical bytes changes parameter versions, not experiment state/RNG.
                require(state(rt,anchors,dual,accepted,batch)==prev,'SNAPSHOT_STATE_RESTORE')
        generations=[scorer.greedy(r) for c in ids for r in current[c] if r['label']=='new']
        save(out/'final-greedy.json',generations)
        save(out/'terminal.json',dict(status='COMPLETED',batches=STEPS,offered=STEPS*BATCH_SIZE,accepted=len(accepted),snapshots=len(snaps),
             instruction_id=NONCE,source=lock['source'],config_sha=lock['configuration']['sha256'],scientific_native_fits=rt.fits,
             technical_replays=0,loss_evaluations=rt.loss_evals,adam_updates=rt.adam,program_seconds=time.monotonic()-start,
             calls=scorer.calls,tokens=scorer.tokens,backwards=scorer.backwards,timers=scorer.seconds,
             backward_tokens=scorer.backward_tokens,backward_seconds=scorer.backward_seconds,
             model_load_seconds=rt.model_load_seconds,timer_additivity='program/method/geometry overlap; not additive',
             cuda_peak_bytes=torch.cuda.max_memory_allocated(),host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             exact_editor_resume='NOT_AVAILABLE',batch_records=records))
    except BaseException as exc:
        failure=dict(status='TECHNICAL_FAILED',exception=repr(exc),traceback=traceback.format_exc(),batch=batch,
            program_seconds=time.monotonic()-start,source=lock['source'],config_sha=lock['configuration']['sha256'],completed_batches=len(records),
            scientific_native_fits=rt.fits if rt else 0,calls=scorer.calls if scorer else {},snapshot_recovery='NOT_ATTEMPTED')
        save(out/'first-error.json',failure)
        if rt is not None and last_finite is not None and len(snaps)<len(SAVE_STEPS):
            try:
                rt.restore(last_finite)
                snaps.append(snapshot(rt,scorer,current[ids[0]][0],out,SAVE_STEPS[len(snaps)],
                    dict(trajectory=spec['name'],offered_ids=ids[:len(records)*BATCH_SIZE],accepted_ids=accepted,source=lock['source'],checkpoint=config['checkpoints'][spec['checkpoint']],
                         replacement_for_next_scheduled=True,last_finite_batch=len(records),exact_editor_resume='NOT_AVAILABLE'),early=True))
                failure['snapshot_recovery']='LAST_FINITE_SAVED'
            except BaseException as secondary:failure['snapshot_recovery']=repr(secondary)
        save(out/'terminal.json',failure);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lock',required=True);p.add_argument('--index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','-1')))
    a=p.parse_args();run(a.lock,a.index)
