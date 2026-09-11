"""Independent entry runner: fixed native proposal, OS, and two-memory paths."""
import argparse
import contextlib
import json
import os
import time
import traceback
from pathlib import Path
import torch
from .identity import *
from .geometry import Projector,Metric,static_solution,dot
from .controller import calibration,step
from .banks import protection,select as bank_select,latest
from .observer import View
from . import native,evaluation
from project.run_scripts.single_layer_cumulative_risk import binding
from project.run_scripts.single_layer_cumulative_risk.evaluation import materialized
from scripts.fixed_counterfact import load_prefix

def progress(out,stage,ledger,**more):
    row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),stage=stage,compute=ledger.receipt(),**more)
    save(out/'progress'/f'{time.time_ns()}.json',row)
    print(json.dumps(row,allow_nan=False),flush=True)

def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    return x

def state_snapshot(out,label,state,extra):
    tensor_save(out/f'{label}.pt',dict(weight=state.detach().cpu(),**cpu(extra)))
    save(out/f'{label}-identity.json',dict(selected_weight_sha=tensor_sha(state),path=str(out/f'{label}.pt')))

def load_projector(out,ledger,reference=None):
    if reference:
        x=torch.load(reference,weights_only=True,map_location='cuda')
        obj=Projector(x['vectors'],x['complement'],x['dimension'],x['spectrum'])
        obj.receipt=x['receipt'];ledger.add('full_projector_exact_reuse')
        save(out/'projector-reference.json',dict(path=str(reference),sha256=sha(reference),receipt=obj.receipt))
        return obj
    with ledger.time('full_projector_eigen'):
        raw=torch.load(binding.PROJECTOR,map_location='cpu',weights_only=True,mmap=True)[0].to('cuda')
        obj=Projector.from_raw(raw)
    tensor_save(out/'projector.pt',cpu(dict(vectors=obj.vectors,complement=obj.complement,dimension=obj.dimension,
                                          spectrum=obj.spectrum,receipt=obj.receipt)))
    save(out/'projector-receipt.json',obj.receipt)
    del raw;torch.cuda.empty_cache()
    return obj

def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    ledger=Ledger();phase='START';model=None
    save(out/'run.lock.json',dict(args=vars(args),input_sha=sha(ROOT/'control/input.lock.json'),
        execution_head=os.environ['L4_ROUTING_SOURCE_HEAD'],job=os.environ.get('SLURM_JOB_ID'),
        gpu_count=1,model_forward='FULL_FP32 eager',geometry='FP64 full P*',scientific_promotion=False))
    try:
        torch.set_num_threads(8);torch.manual_seed(20260911)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        lock=json.loads((ROOT/'control/input.lock.json').read_text());cfg=lock['cases'][f'{args.entry}-B{args.batch}']
        for bound in lock['source']:
            if sha(bound['path'])!=bound['sha256']:raise RuntimeError('SOURCE_LOCK_MISMATCH')
        expected=next(m['sha256'] for m in lock['assets'] if m['path']==str(PREPARED[args.entry]))
        if sha(PREPARED[args.entry])!=expected:raise RuntimeError('PREPARED_ASSET_MISMATCH')
        records=load_prefix(binding.DATA,10000);panel=cfg['panel'];inv=cfg['inventory']
        actual=[records[i] for i in inv['current_effective']]
        if not actual:raise ValueError('B0_NOT_A_PLANNED_MODEL_PATH')
        prep=torch.load(PREPARED[args.entry],map_location='cpu',weights_only=True,mmap=True)
        phase='PROJECTOR';p=load_projector(out,ledger,args.projector_reference)
        progress(out,'PROJECTOR_COMPLETE',ledger,receipt=p.receipt)
        phase='MODEL_LOAD';model,tok,evaltok=binding.load_model(ledger)
        weight=dict(model.named_parameters())[WEIGHT];original_w0=weight.detach().cpu().clone()
        if not torch.equal(original_w0,prep['W0']):raise RuntimeError('PRETRAINED_W0_BYTE_MISMATCH')
        we=prep['We'].to('cuda');ptr=weight.data_ptr()
        with torch.no_grad():weight.copy_(we)
        cp,targets,raw_current=binding.load_entry(args.entry,records)
        zmap={r['case_id']:z for r,z in zip(raw_current,targets['values'],strict=True)}
        phase='NATIVE_PREPARATION'
        if cfg['native_reuse_allowed']:
            wn=prep['WN'].to('cuda');native_reused=True
            ledger.add('native_b100_endpoint_reused');ledger.add('native_targets_reused',len(actual))
        else:
            praw=torch.load(binding.PROJECTOR,map_location='cpu',weights_only=True,mmap=True)[0].to('cuda')
            wn_cpu,committed=native.write(model,tok,cp,zmap,actual,praw,ledger)
            wn=wn_cpu.to('cuda');native_reused=False
            del praw,committed
        with torch.no_grad():weight.copy_(we)
        delta=wn.double()-we.double()
        kin,readout,weights=native.representations(model,tok,actual,prep['contexts'],ledger,all_contexts=True)
        b=len(actual);ncontext=len(weights)
        kk=kin.reshape(b,ncontext,-1)
        # Original mean-of-context-types FP32 order, including clean .5.
        splits=[];offset=0
        for kind in prep['contexts']:
            splits.append(kk[:,offset:offset+len(kind)].mean(1));offset+=len(kind)
        k=torch.stack(splits,dim=1).mean(1).T.double()
        z=torch.stack([zmap[r['case_id']] for r in actual]).to('cuda').double().T
        residual=z-readout.double().T
        context_factor=(kk.double()*torch.tensor(weights,device='cuda',dtype=torch.float64).sqrt()[None,:,None]/b**.5).reshape(-1,k.shape[0]).T.contiguous()
        parity=dict(actual_B=b,native_reused=native_reused,canonical_layer_readout=True,context_type_weights=weights,
            readout_shape=list(readout.shape),K_shape=list(k.shape),fixed_z_sha=tensor_sha(z),
            stored_K_max_abs=float((k-prep['K'].to('cuda').double()).abs().max()) if native_reused else None,
            WN_sha=tensor_sha(wn),We_sha=tensor_sha(we),residual_sha=tensor_sha(residual),
            native_r_not_We_times_K=True,old_q_not_reused=True)
        save(out/'native-response-parity.json',parity)
        del kin,kk,readout,z,splits
        phase='BANK_SELECTION'
        for kind,key in [('Past','past_candidates'),('Base','base_candidates')]:
            rows=[records[i] for i in inv[key]]
            if rows:
                subject_keys,_,_=native.representations(model,tok,rows,prep['contexts'],ledger,all_contexts=False)
                scores=(delta@subject_keys.double().T).square().sum(0).cpu().tolist()
                del subject_keys
            else:scores=[]
            inv[kind]=bank_select(inv[key],scores);inv[kind+'_scores']=scores
        inv['BaseAudit']=inv['base_audit'];inv['sample_outcome_influence']=0
        save(out/'bank-manifest.json',inv)
        view=View(model,evaltok.pad_token_id,ledger);teachers={};factor={}
        phase='WE_TEACHERS'
        for kind in ('Past','Base','BaseAudit'):
            rows=protection(records,inv[kind],kind,evaltok)
            teachers[kind],factor[kind]=view.prepare(we,rows,out/f'{kind}-We-teacher.pt',keys=kind!='BaseAudit')
            if kind!='Past':
                teachers[kind+'W0'],_=view.prepare(original_w0.to('cuda'),rows,out/f'{kind}-W0-teacher.pt',keys=False)
        progress(out,'WE_BANKS_COMPLETE',ledger,bank_requests={key:len(inv[key]) for key in ('Past','Base','BaseAudit')})
        # Outcome-free actual functional/materialized gate on one bound input.
        probe=teachers['Base']['rows'][:1] or teachers['Past']['rows'][:1]
        with torch.no_grad():
            a,_=view.forward(wn,probe)
            from .observer import pack
            inputs,pos=pack(probe,evaltok.pad_token_id,'cuda')
            with materialized(weight,wn,ledger):
                result=model(**inputs).logits
                baseline=[torch.log_softmax(result[j,pp].float(),-1) for j,pp in enumerate(pos)]
            if not all(torch.equal(x,y) for x,y in zip(a,baseline)):raise RuntimeError('FUNCTIONAL_MATERIALIZED_MISMATCH')
        view=View(model,evaltok.pad_token_id,ledger)
        phase='STATIC_FULL_SPACE_SOLVE'
        with ledger.time('full_space_factor_static_inverse'):
            zos,metric,static=static_solution(delta,k,residual,context_factor,
                factor['Past'].to('cuda'),factor['Base'].to('cuda'),p)
            wos=(wn.double()+zos).float();anchor=wos.double()-we.double()
        if not torch.isfinite(wos).all():raise FloatingPointError('NONFINITE_OS_WEIGHT')
        static.update(actual_scalar_drift=float(dot(metric.t,wos.double()-wn.double())),
            physical_span_rounding=float((wos.double()-wn.double()-p.right(wos.double()-wn.double())).norm()))
        save(out/'static-solution.json',static)
        state_snapshot(out,'OS',wos,dict(We_ref=str(PREPARED[args.entry]),Zos=zos,DA=anchor))
        tensor_save(out/'geometry.pt',cpu(dict(factors=metric.factors,chol=metric.chol,ridge=metric.ridge,
            t=metric.t,p_vectors=p.vectors,p_complement=p.complement,p_dimension=p.dimension,
            k=k,residual=residual,context_factor=context_factor)))
        del zos,context_factor,delta,residual,factor
        torch.cuda.empty_cache()
        progress(out,'FIRST_VALID_OS',ledger,static=static,temporary_state_parity=True,restore=True,
            peak_gpu_bytes=torch.cuda.max_memory_allocated())
        channels=[kind for kind in ('Past','Base') if teachers[kind]['rows']]

        def observe(state,gradient=False):
            nonlocal view
            view=View(model,evaltok.pad_token_id,ledger)
            values=[];gradients=[];rows={}
            for kind in channels:
                obs,g=view.harm(state,teachers[kind],kind,gradient=gradient)
                values.append(obs['value']);rows[kind]=obs
                if gradient:gradients.append(g.to('cuda'))
            return torch.tensor(values,device='cuda',dtype=torch.float64),gradients,rows

        def endpoint(label,state,new_path=True):
            nonlocal view
            dest=out/label;dest.mkdir()
            if new_path:
                # Canonical history uses averaged rewrite keys; L4 input is fixed
                # by upstream weights. No keys from Base/Past are appended.
                with ledger.time('terminal_history_finalize'):
                    mn=prep['M'].clone();kf=k.float().cpu();mn[0]+=kf@kf.T
                    ledger.add('terminal_history_append');ledger.add('terminal_active_ledger_finalize')
                state_snapshot(dest,'endpoint',state,dict(entry_reference=str(PREPARED[args.entry]),
                    M=mn,active_fact_ordinals=latest(records,sorted(set(inv['seen_active'])|set(inv['current_effective'])))))
                del mn,kf
            else:save(dest/'native-reference.json',dict(path=str(PREPARED[args.entry]),sha256=sha(PREPARED[args.entry])))
            view=View(model,evaltok.pad_token_id,ledger)
            harms={}
            for kind in ('Past','Base','BaseAudit'):
                harms[kind],_=view.harm(state,teachers[kind],kind)
                if kind!='Past':harms[kind+'W0'],_=view.harm(state,teachers[kind+'W0'],kind)
            save(dest/'harms.json',harms)
            if label=='N' and native_reused:
                src=PREPARED[args.entry].parent
                data=json.loads((src/'N-full.json').read_text())
                if data['panel_identity']!=digest(panel):raise ValueError('REFERENCE_PANEL_MISMATCH')
                save(dest/'full.json',dict(**data,reused_source=str(src/'N-full.json'),reused_sha=sha(src/'N-full.json')))
                gen=json.loads((src/'N-generation.json').read_text())
                save(dest/'generation.json',dict(**gen,reused_source=str(src/'N-generation.json'),reused_sha=sha(src/'N-generation.json')))
                ledger.add('evaluation_pairs_reused',len(data['rows']))
            else:
                with materialized(weight,state,ledger):
                    evaluation.measure(model,evaltok,records,panel,ledger,dest/'full.json')
                    evaluation.generation(model,evaltok,records,panel['generation'],ledger,dest/'generation.json')
            if weight.data_ptr()!=ptr or not torch.equal(weight,we):raise RuntimeError('TERMINAL_ENTRY_RESTORE')
            save(dest/'terminal.json',dict(status='TERMINAL_VALID',arm=label,new_path=new_path,
                weight_sha=tensor_sha(state),history_append=1 if new_path else 'REFERENCE',restore=True,
                inner_history_append=0,ledger=ledger.receipt(),scientific_promotion=False))
            progress(out,'ENDPOINT_COMPLETE',ledger,arm=label)

        phase='REFERENCE_AND_OS_EVALUATION'
        endpoint('N',wn,new_path=not native_reused)
        endpoint('OS',wos)
        # Entry and W0 full panels are exact references where the panel matches.
        for label,state in [('ENTRY',we),('W0',original_w0.to('cuda'))]:
            if native_reused:
                src=PREPARED[args.entry].parent/f'{label}-full.json'
                data=json.loads(src.read_text())
                if data['panel_identity']!=digest(panel):raise ValueError('ENTRY_PANEL_REUSE_MISMATCH')
                save(out/f'{label}-full.json',dict(**data,reused_source=str(src),reused_sha=sha(src)))
                ledger.add('evaluation_pairs_reused',len(data['rows']))
            else:
                with materialized(weight,state,ledger):evaluation.measure(model,evaltok,records,panel,ledger,out/f'{label}-full.json')
        # Bank We/W0 audit baselines are NOT substituted from old full panels.
        for label,state in [('ENTRY',we),('W0',original_w0.to('cuda'))]:
            view=View(model,evaltok.pad_token_id,ledger);base={}
            for kind in ('Past','Base','BaseAudit'):
                base[kind],_=view.harm(state,teachers[kind],kind)
                if kind!='Past':base[kind+'W0'],_=view.harm(state,teachers[kind+'W0'],kind)
            save(out/f'{label}-bank-harms.json',base)
        phase='OS_CALIBRATION'
        fa,ga,obs=observe(wos,gradient=True)
        cal=calibration(metric,anchor,ga,fa)
        tensor_save(out/'calibration.pt',cpu(cal));save(out/'OS-calibration-harms.json',obs)
        del ga
        phase='BF_PATHS'
        arms=['BF1','BF8'] if args.batch==100 else ['BF1']
        if args.entry=='Middle' and args.batch==100:arms+=['Frozen-BF8']
        for arm in arms:
            n=1 if arm=='BF1' else 8;h=1/n;z=torch.zeros_like(anchor)
            actual=we;fcurrent=torch.zeros_like(fa)
            path=out/f'{arm}-trajectory';path.mkdir()
            for node in range(n):
                phase=f'{arm}_NODE_{node}'
                if cal['stationary']:
                    term=dict(node=node,stationary=True,h=h,s_next=(node+1)*h)
                    harm=obs
                else:
                    proposal=(we.double()+(node+1)*h*anchor+z).float()
                    # BF1 proposal is the saved OS endpoint; exact reuse avoids
                    # duplicate teacher/gradient work. Frozen only fixes direction.
                    if n==1:
                        fp=fa;gg=cal['gradients'];harm=obs
                    else:
                        fp,graw,harm=observe(proposal,gradient=arm!='Frozen-BF8')
                        gg=cal['gradients'] if arm=='Frozen-BF8' else [g/s for g,s in zip(graw,cal['sigma'])]
                    zn,correction,term=step(metric,z,gg,fcurrent,fp/cal['sigma'],cal['terminal_budget'],node*h,h,cal['epsilon'],cal['aa'])
                    actual=(we.double()+(node+1)*h*anchor+zn).float()
                    if not torch.isfinite(actual).all():raise FloatingPointError('NONFINITE_BF_ENDPOINT')
                    factual,_,actual_harm=observe(actual)
                    fcurrent=factual/cal['sigma'];z=zn
                    term.update(node=node,actual_harm=factual.tolist(),normalized_actual=fcurrent.tolist(),
                        physical_scalar=float(dot(metric.t,actual.double()-we.double())),
                        physical_anchor_scalar=float(dot(metric.t,anchor)),weight_sha=tensor_sha(actual),
                        first_order_error=(fcurrent-(fp/cal['sigma']+torch.tensor([float(dot(g,correction)) for g in gg],device='cuda'))).tolist())
                    harm=dict(proposal=harm,actual=actual_harm)
                    if n!=1 and arm!='Frozen-BF8':del graw,gg
                    del proposal,correction
                tensor_save(path/f'node{node+1:02d}.pt',dict(Z=z.cpu(),We_reference=str(PREPARED[args.entry]),DA_reference=str(out/'OS.pt')))
                save(path/f'node{node+1:02d}.json',dict(**term,harms=harm,compute=ledger.receipt()))
                if n==8 and node+1 in (2,4):
                    with materialized(weight,actual,ledger):evaluation.measure(model,evaltok,records,panel,ledger,path/f'current{node+1:02d}.json',current_only=True)
                progress(out,'BF_NODE_COMPLETE',ledger,arm=arm,node=node+1,n=n)
            endpoint(arm,actual)
        save(out/'terminal.json',dict(status='TERMINAL_VALID',entry=args.entry,batch_raw=args.batch,batch_effective=b,
            new_path_count=len(arms)+1+(not native_reused),N_reuse=native_reused,arms=['N','OS']+arms,
            compute=ledger.receipt(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            reference_input_sha=sha(ROOT/'control/input.lock.json'),scientific_promotion=False))
        progress(out,'RUN_TERMINAL_VALID',ledger)
    except BaseException as exc:
        restored=None
        if model is not None and 'we' in locals():
            with torch.no_grad():dict(model.named_parameters())[WEIGHT].copy_(we)
            restored=bool(torch.equal(dict(model.named_parameters())[WEIGHT],we))
        save(out/'failure-boundary.json',dict(stage=phase,exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
            entry_restore=restored,compute=ledger.receipt(),science_changes=0,failed_attempt_immutable=True,scientific_promotion=False))
        raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',choices=['Early','Middle','Late'],required=True)
    ap.add_argument('--batch',type=int,choices=[1,7,100],default=100);ap.add_argument('--output',required=True)
    ap.add_argument('--projector-reference');main(ap.parse_args())
