"""Bounded paired technical benchmark; never a scientific editing chain.

Reference source is unchanged. All optimizers share explicit oracle budgets;
tensor states live only in this process. Qualification failures exclude routes.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_pilot.prompts import prepare
from project.run_scripts.jlz_pilot.run import MODEL,LAYERS,capture_entry,capture_keys
from project.run_scripts.jlz_sequential.state import Transaction,state_hash,weights,tensor_sha,parameter_guard,hooks
from .core import ROUTES,RouteOracle,CertificationFailed,early_keys,selected_entry,selected_committed
from .reference import Reference
from .budget import PanelBudget,BudgetAccountant
from .solver import solve
from .geometry import CovarianceCache
from .measurement import write,sha,point_record,compare_points,timing,timed
from . import evaluation,native,kernels

ROOT=Path(__file__).resolve().parents[3]
CONTRACT=ROOT/'plans/global/2026-10-01-jlz-efficiency-execution-v1/contract.json'
CONTEXT=Path('/mnt/raid5/janghj/ODE-edit/local/reviews/ep-tw1-completed-2026-09-15/baselines/1/contexts.json')
DATA=Path('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')

def require(test,label):
    if not test:raise RuntimeError(label)

def direction(rho,mask,width):
    generator=torch.Generator(device=rho.device).manual_seed(20261001)
    value=torch.randn((len(rho),width),device=rho.device,generator=generator)
    value*= (rho/value.norm(dim=1))[:,None];value[~mask]=0
    return value

def make_oracle(model,panel,route):
    if route=='REF_MB2':return Reference(model,panel['spec'],panel['teacher'],panel['keys'],panel['adj'],panel['active'],2)
    return RouteOracle(model,panel['spec'],panel['teacher'],panel['adj'],panel['active'],route)

def entry(model,tok,records,contexts,history,cache,out,label):
    spec=prepare(tok,[r['requested_rewrite']|{'case_id':r['case_id']} for r in records],contexts,'cuda')
    (anchors,teacher,nll),seconds=timed(capture_entry,model,spec,2)
    keys,key_seconds=timed(capture_keys,model,spec,contexts,2)
    (ca,ct,cn),candidate_seconds=timed(selected_entry,model,spec,2)
    ck,candidate_key_seconds=timed(early_keys,model,spec,contexts,2)
    comparison=dict(anchor_maxabs=max(float((anchors[l]-ca[l]).abs().max()) for l in LAYERS),
        teacher_maxabs=float((teacher-ct).abs().max()),nll_maxabs=float((nll-cn).abs().max()),
        key_bitwise=all(torch.equal(keys[l],ck[l]) for l in LAYERS),active_exact=torch.equal(nll>=.05,cn>=.05))
    comparison['status']='PASS' if comparison['key_bitwise'] and comparison['active_exact'] and max(comparison[k] for k in ('anchor_maxabs','teacher_maxabs','nll_maxabs'))<=1e-4 else 'UNQUALIFIED'
    adj,geometry=cache.solve(keys,history)
    norms=torch.cat([anchors[l].norm(dim=0) for l in LAYERS]);require(bool((norms>0).all()),'ANCHOR_POSITIVE')
    active=nll>=.05;c,rho,mask=.5/norms.square(),.75*norms,active.repeat(5)
    receipt=dict(label=label,case_ids=[r['case_id'] for r in records],input_identity=spec['identity'],
        teacher_sha=tensor_sha(teacher),key_sha={str(l):tensor_sha(k) for l,k in keys.items()},
        active=active.cpu().tolist(),entry_nll=nll.cpu().tolist(),comparison=comparison,
        reference_entry_seconds=seconds,reference_key_seconds=key_seconds,candidate_entry_seconds=candidate_seconds,
        candidate_key_seconds=candidate_key_seconds,geometry=geometry,common_oracle_teacher_adj='REFERENCE_FIXED',
        extra_whole_batch_oracles=0,entry_forward_not_optimization=True)
    write(out/(label+'-entry.json'),receipt)
    del ca,ct,cn,ck,anchors
    return dict(spec=spec,teacher=teacher,keys=keys,adj=adj,active=active,c=c,rho=rho,mask=mask,entry_receipt=receipt)

def compare_short(reference,refx,actual,x):
    fields=('status','calls','accepted_steps','backtracks','branches')
    exact=all(reference.get(k)==actual.get(k) for k in fields)
    # If all observed objective/gradient/point bytes equal the reference, a
    # near-boundary flag is not an unverified numerical decision for this run.
    calls_exact=reference['point_trace']==actual['point_trace']
    ambiguous=bool(actual.get('boundary_ambiguities')) and not calls_exact
    rel=float((x-refx).norm()/refx.norm().clamp_min(1))
    return dict(status='PASS' if exact and rel<=1e-4 and not ambiguous else 'UNQUALIFIED',
        discrete_exact=exact,all_observed_calls_exact=calls_exact,uncertified_boundary=ambiguous,returned_R_relative=rel)

def small(model,tok,records,contexts,history,cache,budget,out):
    panel=entry(model,tok,records,contexts,history,cache,out,'small4');c,rho,mask=(panel[k] for k in ('c','rho','mask'))
    vector=direction(rho,mask,model.config.hidden_size);references={};qualified={};route_records={};scale=1.
    for route in ROUTES:
        if route.startswith('E123') and panel['entry_receipt']['comparison']['status']!='PASS':
            qualified[route]=dict(status='UNQUALIFIED',reason='E3_ENTRY_OR_KEY');continue
        oracle=make_oracle(model,panel,route);results=[];started=time.monotonic()
        try:
            for fraction in (0.,.01,.5,.999):
                x=vector*fraction;budget.charge('fixed',route);result=oracle(x)
                if route=='REF_MB2' and fraction==0:scale=max(1.,float(result[1][mask].double().norm()))
                record,gradient=point_record(result,x,c,rho,mask,scale)
                require(all(np.isfinite(v) for k in ('nll','kl') for v in record[k]),'NONFINITE_REFERENCE_OR_ROUTE')
                if route=='REF_MB2':references[fraction]=(record,gradient);comparison=dict(status='PASS',reference=True)
                else:comparison=compare_points(*references[fraction],record,gradient)
                results.append(dict(fraction=fraction,point=record,comparison=comparison));del result,gradient,x
            qualified[route]=dict(status='PASS' if all(r['comparison']['status']=='PASS' for r in results) else 'UNQUALIFIED',points=results)
        except (CertificationFailed,FloatingPointError) as exc:
            if route=='REF_MB2':raise
            qualified[route]=dict(status='UNQUALIFIED',reason=str(exc),points=results)
        route_records[route]=oracle.records
        qualified[route]['seconds_with_instrumentation']=time.monotonic()-started
        write(out/('fixed-'+route+'.json'),qualified[route]|{'oracle_records':oracle.records,'budget':budget.report()})
        del oracle;gc.collect();torch.cuda.empty_cache()
    shorts={};refx=None;returned=None
    for route in ROUTES:
        if qualified.get(route,{}).get('status')!='PASS':continue
        oracle=make_oracle(model,panel,route);trace=[];account=BudgetAccountant(12)
        def call(x):
            budget.charge('short',route);r=oracle(x)
            trace.append(dict(x_sha=tensor_sha(x),gradient_sha=tensor_sha(r[1]),smooth=float(r[0])))
            return r
        try:
            result=solve(call,torch.zeros_like(vector),c,rho,mask,cap=12,tol=1e-4,account=account)
            x=result.pop('x');g=result.pop('gradient');payload=result.pop('final_payload')
            result['point_trace']=trace;result['accountant']=account.report()
            if route=='REF_MB2':refx=x.detach().clone();returned={k:payload[k] for k in ('nll','kl')};comparison=dict(status='PASS',reference=True)
            else:comparison=compare_short(shorts['REF_MB2']['solver'],refx,result,x)
            shorts[route]=dict(solver=result,comparison=comparison,oracle_records=oracle.records)
            if result['status']=='NONFINITE':
                if route=='REF_MB2':raise FloatingPointError('REFERENCE_NONFINITE_SHORT')
                shorts[route]['comparison']=dict(status='UNQUALIFIED',reason='NONFINITE_SHORT')
            del payload,x,g
        except (CertificationFailed,FloatingPointError) as exc:
            if route=='REF_MB2':raise
            shorts[route]=dict(comparison=dict(status='UNQUALIFIED',reason=str(exc)),accountant=account.report(),oracle_records=oracle.records)
        write(out/('short-'+route+'.json'),shorts[route]|{'budget':budget.report()})
        del oracle;gc.collect();torch.cuda.empty_cache()
    require(refx is not None,'REFERENCE_SHORT_MISSING')
    return panel,qualified,shorts,refx,returned,route_records

def clean_observation(value):return {k:v for k,v in value.items() if k!='logits'}

def probe(model,tok,records,contexts,history,panel,refx,returned,candidate,out):
    observations=[];receipts=[]
    for route in ('REF_MB2',candidate):
        txn=Transaction(model,history,contexts,[])
        with txn:
            with torch.no_grad():
                B=len(records)
                effective={l:weights(model)[l].detach().clone()+(refx[i*B:(i+1)*B].T.double()@panel['adj'][l].T).float() for i,l in enumerate(LAYERS)}
                for l,w in weights(model).items():w.copy_(effective[l])
                require(all(torch.equal(w,effective[l]) for l,w in weights(model).items()),'RAM_COMMIT_BITWISE')
            # Teacher is the immutable W0 entry; capture_entry would recapture
            # the wrong teacher for KL, so use the pinned committed-loss path.
            if route.startswith('E123'):
                nr,kr=selected_committed(model,panel['spec'],panel['teacher'],panel['active'],2)
            else:
                oracle=Reference(model,panel['spec'],panel['teacher'],panel['keys'],panel['adj'],panel['active'],2)
                nr,kr=oracle.committed_losses();del oracle
            error=max(abs(a-b) for k,a_list in (('nll',nr.cpu().tolist()),('kl',kr.cpu().tolist())) for a,b in zip(a_list,returned[k],strict=True))
            require(error<=1e-3,'SMALL_COMMIT_PARITY')
            post=(early_keys if route.startswith('E123') else capture_keys)(model,panel['spec'],contexts,2)
            require(torch.equal(post[4],panel['keys'][4]),'SMALL_L4_KEY')
            for l in LAYERS:txn.append(l,post[l])
            endpoint=state_hash(model,history)
            observation=evaluation.evaluate(model,tok,records,selected=route!='REF_MB2')
            require(state_hash(model,history)==endpoint,'OBSERVER_MUTATION')
            observations.append(observation)
            row=dict(route=route,entry=txn.before,endpoint=endpoint,appends=len(txn.appended),
                commit_NLL_KL_error=error,L4_key_bitwise=True,evaluated_weight_bitwise=True,technical_not_science=True)
            del effective,post,nr,kr
        row.update(restored=txn.restored,post_restore=state_hash(model,history));receipts.append(row)
        write(out/('small-observer-'+route+'.json'),clean_observation(observation))
        del txn;gc.collect();torch.cuda.empty_cache()
    comparison=evaluation.compare(*observations)
    write(out/'small-probe.json',dict(probes=receipts,observer_comparison=comparison,no_checkpoints=True))
    return comparison

def large(model,tok,records,contexts,history,cache,budget,candidate,out):
    panel=entry(model,tok,records,contexts,history,cache,out,'B100')
    if candidate.startswith('E123') and panel['entry_receipt']['comparison']['status']!='PASS':
        return dict(status='CANDIDATE_EXCLUDED_ENTRY',oracles=0)
    c,rho,mask=(panel[k] for k in ('c','rho','mask'));x=direction(rho,mask,model.config.hidden_size)*.02
    # Only one prefix cache at a time; paired endpoints and input are identical.
    results=[];ref=None;times={'REF_MB2':[],'CAND':[]};scale=1.;oracle=None;last=None
    order=('REF_MB2','CAND','REF_MB2','CAND','CAND','REF_MB2','REF_MB2','CAND')
    for i,label in enumerate(order):
        route='REF_MB2' if label=='REF_MB2' else candidate
        if route!=last:
            del oracle;gc.collect();torch.cuda.empty_cache();oracle=make_oracle(model,panel,route);last=route
        budget.charge('B100',route)
        try:result=oracle(x)
        except (CertificationFailed,FloatingPointError) as exc:
            if label=='REF_MB2':raise
            failure=dict(index=i,route=route,status='CERTIFICATION_FAILED',reason=str(exc),budget=budget.report())
            write(out/f'B100-excluded-{i}.json',failure)
            return dict(status='CANDIDATE_EXCLUDED',completed=len(results),rows=results,failed_attempt=failure)
        # No extra zero-R ninth oracle is authorized. Denominator1 yields a
        # conservative sufficient bound because the original initial scale>=1.
        # Do not label the nonzero candidate's gradient as initial-gradient.
        point,g=point_record(result,x,c,rho,mask,scale)
        if ref is None:ref=(point,g);comparison=dict(status='PASS',reference=True)
        else:comparison=compare_points(*ref,point,g)
        results.append(dict(index=i,route=route,warmup=i<2,point=point,comparison=comparison,timing=oracle.records[-1],prox_bound='RAW_DIFFERENCE_DENOMINATOR1_SUFFICIENT_FOR_FIXED_NORMALIZED_THRESHOLD; original initial scale NOT_MEASURED'))
        write(out/f'B100-oracle-{i}.json',results[-1]|{'budget':budget.report()})
        del result
        if comparison['status']!='PASS':
            if label=='REF_MB2':raise RuntimeError('B100_REFERENCE_REPEAT_FAILURE')
            return dict(status='CANDIDATE_EXCLUDED_PARITY',completed=len(results),rows=results)
        if i==1:
            write(out/'REPAIR_INITIAL_VALID.json',dict(status='KERNEL_SUMMARY_AND_B100_PAIRED_WARMUPS_VALID',
                kernels_sha256=sha(out/'kernels.json'),warmup_reference_sha256=sha(out/'B100-oracle-0.json'),
                warmup_candidate_sha256=sha(out/'B100-oracle-1.json'),candidate=candidate,budget=budget.report(),
                B100_complete=False,not_terminal_PASS=True))
        if i>=2:times[label].append(oracle.records[-1]['seconds'])
    del oracle;gc.collect();torch.cuda.empty_cache()
    txn=Transaction(model,history,contexts,[]);eval_rows=[];evaltimes={'REF':[],'CAND':[]};reference_eval=None
    with txn:
        with torch.no_grad():
            for i,l in enumerate(LAYERS):weights(model)[l].add_((x[i*100:(i+1)*100].T.double()@panel['adj'][l].T).float())
        endpoint=state_hash(model,history)
        require(endpoint['W']==ref[0]['weight_sha'],'B100_ACTUAL_OBSERVER_WEIGHT_IDENTITY')
        for i,label in enumerate(('REF','CAND','REF','CAND','CAND','REF','REF','CAND')):
            observation=evaluation.evaluate(model,tok,records,selected=label=='CAND')
            if reference_eval is None:reference_eval=observation;comparison=dict(status='PASS',reference=True)
            else:comparison=evaluation.compare(reference_eval,observation)
            write(out/f'B100-observer-{i}.json',clean_observation(observation)|{'comparison':comparison,'endpoint':endpoint,'warmup':i<2})
            eval_rows.append(dict(index=i,route=label,seconds=observation['seconds'],comparison=comparison,work=observation['work'],fallback=observation['fallback_full_groups']))
            require(state_hash(model,history)==endpoint,'B100_OBSERVER_MUTATION')
            if comparison['status']!='PASS':
                if label=='REF':raise RuntimeError('B100_REFERENCE_OBSERVER_FAILURE')
                break
            if i>=2:evaltimes[label].append(observation['seconds'])
        del reference_eval,observation
    return dict(status='MEASURED',candidate=candidate,oracle_comparison=timing(times['REF_MB2'],times['CAND']),
        observer_comparison=timing(evaltimes['REF'],evaltimes['CAND']) if all(len(v)==3 for v in evaltimes.values()) else {'status':'UNQUALIFIED'},
        oracle_rows=results,observer_rows=eval_rows,restored=txn.restored,history_appends=0,no_optimizer=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run;out=run/'output';out.mkdir(exist_ok=False);start=time.monotonic();budget=PanelBudget();receipt=dict(status='STARTED',scientific_chains=0,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    try:
        lock=json.loads((run/'execution.lock.json').read_text());contract=json.loads(CONTRACT.read_text())
        for row in lock['members']:
            p=Path(row['path']);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'LOCK_MISMATCH '+str(p))
        torch.set_num_threads(8);random.seed(20261001);np.random.seed(20261001);torch.manual_seed(20261001)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require(torch.cuda.device_count()==1,'ONE_GPU');require(str(torch.__version__)=='2.9.1+cu128' and transformers.__version__=='4.57.1','RUNTIME')
        records=load_prefix(DATA,100);contexts=json.loads(CONTEXT.read_text())
        tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right'
        model,loadseconds=timed(AutoModelForCausalLM.from_pretrained,MODEL,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager',device_map='cuda')
        model.eval();model.requires_grad_(False);model.config.use_cache=False
        history={l:torch.zeros(model.config.intermediate_size,model.config.intermediate_size) for l in LAYERS}
        guard=parameter_guard(model);hookguard=hooks(model);before=state_hash(model,history)
        write(out/'runtime.json',dict(torch=str(torch.__version__),transformers=transformers.__version__,GPU=torch.cuda.get_device_name(),GPU_UUID=str(torch.cuda.get_device_properties(0).uuid),loadseconds=loadseconds,source=lock['source'],job=os.environ.get('SLURM_JOB_ID'),entry=before,save_checkpoints=False))
        cache=CovarianceCache();reuse_root=lock.get('reuse',{}).get('root')
        if reuse_root:
            from .reuse import reconstruct,load as reused_load
            require(before==reused_load(reuse_root,'runtime.json')['entry'],'REUSE_W0_H0_IDENTITY')
            panel,qualified,shorts,refx,returned,route_records=reconstruct(model,tok,records[:4],contexts,history,cache,budget,out,reuse_root,entry,make_oracle)
        else:panel,qualified,shorts,refx,returned,route_records=small(model,tok,records[:4],contexts,history,cache,budget,out)
        # Exactly one declared host cache reuse per layer; the later B100 solve
        # still constructs a new entry-specific system/adj.
        for layer in LAYERS:cache.get(layer)
        if not reuse_root:
            requests=[r['requested_rewrite']|{'case_id':r['case_id']} for r in records[:4]]
            states,native_reference,hp,lookup=native.original(model,tok,requests,contexts)
            write(out/'native-original.json',dict(requests=native_reference,source_sha=sha(native.SOURCE/'AlphaEdit/compute_z.py'),physical_first_candidate_reused=True))
            nc=native.cached(model,tok,states,hp,lookup,contexts);write(out/'native-cached.json',nc)
            nb=native.cached(model,tok,states,hp,lookup,contexts,True) if nc['status']=='PASS' else dict(status='NOT_RUN',reason='CACHED_SINGLETON_UNQUALIFIED')
            write(out/'native-batched.json',nb);del states
        width4=panel['spec']['tokens']['input_ids'].shape[1]
        hundred=prepare(tok,[r['requested_rewrite']|{'case_id':r['case_id']} for r in records],contexts,'cuda')
        kernel=kernels.run(model,width4,hundred['tokens']['input_ids'].shape[1]);write(out/'kernels.json',kernel);del hundred
        options=[r for r in ROUTES[1:] if qualified.get(r,{}).get('status')=='PASS' and shorts.get(r,{}).get('comparison',{}).get('status')=='PASS']
        # Direct route needs the independently measured dense/direct kernel gate.
        options=kernels.filter_direct_routes(options,kernel)
        require(options,'NO_QUALIFIED_CANDIDATE: preserve reference, no B100 route substitution')
        candidate=min(options,key=lambda r:np.median([x['seconds'] for x in route_records[r]]))
        write(out/'selection.json',dict(candidate=candidate,qualified=options,criterion='small fixed-point median timing among numerical/branch-qualified routes; no performance selection',timing_promotion='NOT_YET_B100'))
        small_observer=probe(model,tok,records[:4],contexts,history,panel,refx,returned,candidate,out)
        write(out/'INITIAL_VALID.json',dict(status='SMALL_ACTUAL_QUALIFICATION_KERNEL_AND_RAM_PROBE',candidate=candidate,small_observer=small_observer,
            budget=budget.report(),not_B100_or_terminal_PASS=True,seconds=time.monotonic()-start,existing56684_unchanged=True))
        del panel,refx,returned;gc.collect();torch.cuda.empty_cache()
        result=large(model,tok,records,contexts,history,cache,budget,candidate,out)
        write(out/'B100-summary.json',result)
        require(state_hash(model,history)==before and parameter_guard(model)==guard and hooks(model)==hookguard,'FINAL_STATE_RESTORE')
        receipt.update(status='COMPLETED_BOUNDED_BENCHMARK',candidate=candidate,B100_status=result['status'],small_observer_status=small_observer['status'],state_restored=True)
        write(out/'C0-cache.json',cache.records)
    except Exception:
        receipt.update(status='TECHNICAL_OR_QUALIFICATION_STOP',traceback=traceback.format_exc());raise
    finally:
        receipt.update(seconds=time.monotonic()-start,budget=budget.report(),old56684_mutation=False,server3_actions=0)
        write(out/'terminal.json',receipt)

if __name__=='__main__':main()
