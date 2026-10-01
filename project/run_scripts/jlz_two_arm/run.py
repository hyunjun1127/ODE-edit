"""Shared preparation, two independent pilot jobs, fresh two-lane main chains."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from project.run_scripts.jlz_pilot.prompts import prepare,subject_last
from project.run_scripts.jlz_efficiency.core import selected_entry,early_keys,effective_linear,CertificationFailed
from project.run_scripts.jlz_sequential.state import Transaction,state_hash,weights,tensor_sha,parameter_guard
from .common import LAYERS,write,sha,member,digest,require,numerical
from .oracle import CommonOracle,geometry,prepare_teacher
from .references import select_general,select_replay,filter_general_pool
from .burden import actual_energy_chunked
from .solver import solve
from .observation import observe

def load(config):
    torch.set_num_threads(8);random.seed(20261002);np.random.seed(20261002);torch.manual_seed(20261002)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'ONE_ALLOCATED_GPU')
    require(str(torch.__version__)=='2.9.1+cu128' and transformers.__version__=='4.57.1','PINNED_RUNTIME')
    tok=AutoTokenizer.from_pretrained(config['model'],local_files_only=True);tok.padding_side='right';tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,torch_dtype=torch.float32,
                low_cpu_mem_usage=True,device_map={'':'cuda:0'},attn_implementation='eager').eval()
    model.config.use_cache=False
    for p in model.parameters():require(p.dtype==torch.float32,'FP32_MODEL');p.requires_grad_(False)
    return model,tok

def history_for(model):return {l:torch.zeros(model.config.intermediate_size,model.config.intermediate_size) for l in LAYERS}

def auxiliary(contexts):
    n=np.random.get_state()
    return dict(contexts=digest(contexts),python=digest(random.getstate()),
                numpy=digest([n[0],n[1].tolist(),*n[2:]]),torch_cpu=tensor_sha(torch.get_rng_state()),
                torch_cuda=[tensor_sha(x) for x in torch.cuda.get_rng_state_all()])

def import_receipt(config):
    roots=[Path(config['source']),Path(config['baseline_pilot']['native_root'])]
    result=[]
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if not isinstance(file,str) or not file.endswith('.py'):continue
        path=Path(file).resolve()
        if any(path.is_relative_to(p) for p in roots):result.append(dict(module=name,**member(path)))
    return dict(modules=result,python=sys.version,executable=sys.executable,
                torch=member(torch.__file__),transformers=member(transformers.__file__),
                CUDA=torch.version.cuda,TF32_matmul=torch.backends.cuda.matmul.allow_tf32,
                TF32_cudnn=torch.backends.cudnn.allow_tf32)

def data(config):
    records=json.loads(Path(config['stream']).read_text());full=json.loads(Path(config['general']).read_text())
    pool,_=filter_general_pool(full,records);contexts=json.loads(Path(config['contexts']).read_text())
    return records,pool,contexts

def build(model,tok,records,contexts,history,config,general,replay,teachers,eta):
    t=time.monotonic();B=len(records)
    spec=prepare(tok,[r['requested_rewrite']|{'case_id':r['case_id']} for r in records],contexts,'cuda')
    require(spec['tokens']['input_ids'].shape[0]==B*7,'NATIVE_ROW_COVERAGE')
    require(spec['tokens']['input_ids'].shape[1]<=model.config.max_position_embeddings,'CURRENT_LENGTH')
    anchors,teacher,nll=selected_entry(model,spec,4);keys=early_keys(model,spec,contexts,4)
    active=nll>=.05
    P,M,s,geo=geometry(keys,history,anchors,Path(config['stats']))
    oracle=CommonOracle(model,tok,spec,teacher,P,M,s,general,replay,teachers,eta,4)
    norms=torch.cat([anchors[l].norm(dim=0) for l in LAYERS]);require(bool(torch.isfinite(norms).all()) and bool((norms>0).all()),'ANCHOR_FINITE_POSITIVE')
    x0=torch.zeros(5*B,model.config.hidden_size,device='cuda')
    evidence=dict(input_identity=spec['identity'],teacher_sha256=tensor_sha(teacher),
                  key_sha256={str(l):tensor_sha(k) for l,k in keys.items()},entry_nll=nll.cpu().tolist(),
                  active=active.cpu().tolist(),geometry=geo,seconds=time.monotonic()-t)
    return oracle,x0,.5/norms.square(),.75*norms,active.repeat(5),keys,evidence

@torch.no_grad()
def cross_panel(model,tok,oracle,payload,current,general,replay,out):
    start=time.monotonic();tokens_count=0
    records=current[:4]+general[:4]+replay[:4];rows=[];states=['entry',*map(str,LAYERS),'joint']
    for rec in records:
        rw=rec['requested_rewrite'];prompt=rw['prompt'].format(rw['subject']);tokens=tok(prompt,return_tensors='pt').to('cuda')
        tokens_count+=7*int(tokens['attention_mask'].sum())
        pos=subject_last(tok,rw['prompt'],rw['subject']);values={}
        for state in states:
            effective={l:(payload['weights'][l] if state=='joint' or state==str(l) else oracle.entry[l]) for l in LAYERS}
            capture={}
            class Done(Exception):pass
            def hook(module,args,output):
                h=output[0] if isinstance(output,(tuple,list)) else output
                capture['h']=h[0,pos].detach().clone();raise Done()
            handle=model.model.layers[8].register_forward_hook(hook)
            try:
                with effective_linear(model,effective):
                    try:model.model(**tokens,use_cache=False)
                    except Done:pass
            finally:handle.remove()
            require('h' in capture,'CROSS_CAPTURE_MISSING');values[state]=capture['h']
        joint=values['joint']-values['entry'];singles=[values[str(l)]-values['entry'] for l in LAYERS]
        residual=joint-sum(singles);den=float(joint.double().norm())
        rows.append(dict(case_id=rec['case_id'],absolute=float(residual.double().norm()),joint_norm=den,
                         relative=None if den<=1e-12 else float(residual.double().norm())/den,
                         layer_norms=[float(x.double().norm()) for x in singles],positions='same_subject_last_L8_block_output',states=7))
    write(out,dict(rows=rows,prompt_states=7*len(records),forward_calls=7*len(records),valid_tokens=tokens_count,
                   padded_tokens=tokens_count,seconds=time.monotonic()-start,no_commit=True,record_only=True))

def verify_lock(root,config):
    lock=json.loads((root/'execution.lock.json').read_text())
    require(sha(root/'config.json')==lock['config_sha256'],'CONFIG_LOCK_MISMATCH')
    for row in lock['source_members']:
        require(Path(row['path']).stat().st_size==row['bytes'] and sha(row['path'])==row['sha256'],'SOURCE_IDENTITY '+row['path'])
    # Shared CPU full hashes are reused only while exact filesystem identity holds.
    for row in config['inputs']:
        s=Path(row['path']).stat()
        require(s.st_size==row['bytes'] and s.st_ino==row['inode'] and s.st_mtime_ns==row['mtime_ns'],'INPUT_CHANGED '+row['path'])
    return lock

def prep(root,config,model,tok,records,pool,contexts):
    out=root/'prep';out.mkdir(exist_ok=False);start=time.monotonic();history=history_for(model)
    pristine=state_hash(model,history);w0={l:w.detach().cpu().clone() for l,w in weights(model).items()}
    common=dict(instruction_id=config['instruction_id'],config_sha256=sha(root/'config.json'),source_sha=sha(root/'execution.lock.json'))
    ids=sorted({i for stage in config['schedules'].values() for cell in stage for i in cell['general']})
    lookup={r['case_id']:r for r in pool};general_all=[lookup[i] for i in ids]
    t=time.monotonic();teachers=prepare_teacher(model,tok,general_all,4)
    tmp=out/'w0-teachers.pt.tmp';torch.save(teachers,tmp);os.replace(tmp,out/'w0-teachers.pt')
    write(out/'teacher.json',dict(**common,member=member(out/'w0-teachers.pt'),rows=len(teachers),seconds=time.monotonic()-t,
          model_epoch='W0',purpose='immutable reference input, not edited checkpoint',identities={k:{z:v[z] for z in ('identity','sha256')} for k,v in teachers.items()}))
    general=select_general(pool,records[:100],config['contract']['references'],0)
    oracle,x0,c,rho,mask,keys,evidence=build(model,tok,records[:100],contexts,history,config,general,[],teachers,0)
    generator=torch.Generator(device='cuda').manual_seed(20261002)
    probe=torch.randn(x0.shape,generator=generator,device='cuda');probe*= (.02*rho/probe.norm(dim=1))[:,None];probe[~mask]=0
    comparisons=[];reference=None;times={}
    for route in ('original','dense','direct'):
        t=time.monotonic()
        try:
            value,g,p=oracle(probe,route)
            row=dict(route=route,seconds=time.monotonic()-t,values={'loss':value,'gradient_norm':float(g.norm())},timing=p['timing'])
            if reference is None:reference=(value,g.clone(),{k:p[k] for k in ('nll','kl')})
            else:
                error=dict(loss_abs=abs(value-reference[0]),gradient_relative=float((g-reference[1]).norm()/reference[1].norm().clamp_min(1)),
                           per_request_abs=max(abs(a-b) for k in ('nll','kl') for a,b in zip(p[k],reference[2][k])))
                row['comparison']=numerical(error,{k:1e-3 for k in error})
            times[route]=row['seconds'];del p,g
        except CertificationFailed as e:
            row=dict(route=route,structural_domain_failure=str(e),selected=False)
        comparisons.append(row);write(out/'technical-progress.json',dict(**common,comparisons=comparisons,extra_whole_batch_calls=oracle.calls,cap=6))
    selected='direct' if 'direct' in times and times['direct']<times['dense'] else 'dense'
    require(oracle.calls<=6,'SHARED_TECHNICAL_CAP')
    require(state_hash(model,history)==pristine,'PREP_STATE_MUTATION')
    write(out/'route.json',dict(**common,route=selected,microbatch=4,comparisons=comparisons,extra_whole_batch_calls=oracle.calls,
                              selection='fixed candidate single timing/memory; no quality gate/no stable speed claim',numerical_certification='NOT_ESTABLISHED'))
    del oracle,x0,c,rho,mask,keys,probe,reference;gc.collect();torch.cuda.empty_cache()
    write(out/'W00-observations.json',observe(model,tok,records[:2000],history,0,w0=True))
    from .baseline_pilot import run_baseline_pilots
    def baseline_observer(m,t,rs,family,bi):
        return observe(m,t,rs,{},bi, max(0,len(rs)-4),w0=False)
    pilot=run_baseline_pilots(model,tok,records[:8],contexts,config,out/'baseline-pilots',w0,evaluate=baseline_observer)
    require(state_hash(model,history)==pristine,'BASELINE_PILOT_W0_RESTORE')
    write(out/'receipt.json',dict(**common,status='STRUCTURAL_COMPLETION',route=selected,baseline_pilots=pilot,
          actual_whole_batch_technical_calls=3,seconds=time.monotonic()-start,W0_restored=True,checkpoint_saved=False))

def run_batch(root,phase,arm,config,model,tok,allrecords,pool,contexts,history,ledger,number,teachers,route):
    out=root/f'{phase}-{arm}';B=4 if phase=='pilot' else 100;start=time.monotonic()
    records=allrecords[(number-1)*B:number*B];past=allrecords[:(number-1)*B]
    require(len(ledger)==number-1,'LEDGER_PREFIX')
    general=select_general(pool,records,config['contract']['references'],number-1)
    replay=select_replay(past,records,config['contract']['references'],number-1)
    cell=config['schedules'][phase][number-1]
    require([r['case_id'] for r in records]==cell['ids'] and [r['case_id'] for r in general]==cell['general'] and [r['case_id'] for r in replay]==cell['replay'],'REFERENCE_ORDER_IDENTITY')
    txn=Transaction(model,history,contexts,ledger)
    row=dict(batch=number,phase=phase,arm=arm,entry=txn.before,case_ids=cell['ids'],commit=False,history_appends=0,
             auxiliary_entry=auxiliary(contexts),
             references={'current_ids':cell['ids'],'general_ids':cell['general'],'replay_ids':cell['replay']},
             config_sha256=sha(root/'config.json'),execution_lock_sha256=sha(root/'execution.lock.json'))
    if ledger:
        require(row['entry']==ledger[-1]['post'],'OWN_CHAIN_STATE_LINK')
        require(row['auxiliary_entry']==ledger[-1]['auxiliary_post'],'OWN_CHAIN_RNG_CONTEXT_LINK')
    write(out/f'B{number:03d}-entry.json',row)
    if number==2:
        write(out/'INITIAL_VALID.json',dict(status='B1_COMMIT_5_APPEND_OBSERVER_RESTORE_B2_OWN_ENTRY',phase=phase,arm=arm,
               B1=member(out/'B001-committed.json'),B2_entry=row['entry'],B2_auxiliary=row['auxiliary_entry'],main_representative=phase=='main'))
    oracle=None
    try:
        with txn:
            oracle,x0,c,rho,mask,keys,evidence=build(model,tok,records,contexts,history,config,general,replay,teachers,0 if arm=='A' else 1)
            row['entry_preparation']=evidence
            require(evidence['input_identity']==next(x['identity'] for x in config['packing'] if x['phase']==phase and x['batch']==number),'SEALED_CURRENT_TOKEN_LAYOUT')
            result=solve(lambda x:oracle(x,route),x0,c,rho,mask,cap=32 if phase=='pilot' else 120,
                         final_fun=lambda x:oracle(x,'original'))
            payload=result.pop('final_payload');x=result.pop('x');gradient=result.pop('gradient')
            require(result['calls']==oracle.calls,'SCIENCE_CALL_ACCOUNTING')
            row['solver']=result;row['fit']={k:v for k,v in payload.items() if k not in ('weights','timing')};row['oracle_records']=oracle.records
            write(out/f'B{number:03d}-candidate.json',row)
            if phase=='pilot':
                cross_panel(model,tok,oracle,payload,records,general,replay,out/f'B{number:03d}-cross.json')
            row['layer_allocation']=[]
            for i,l in enumerate(LAYERS):
                t=time.monotonic();R=x[i*B:(i+1)*B].T
                path=Path(config['stats'])/f'model.layers.{l}.mlp.down_proj_float32_mom2_100000.npz'
                with np.load(path,allow_pickle=False) as z:raw=torch.from_numpy(z['mom2.mom2'].copy());count=int(z['mom2.count'])
                A=(raw/count).to(device='cuda',dtype=torch.float64).mul_(15000);del raw
                A.add_(history[l].to(device='cuda',dtype=torch.float64))
                energy=actual_energy_chunked(payload['weights'][l],oracle.entry[l],A,oracle.scales[l],row_chunk=32)
                row['layer_allocation'].append(dict(layer=l,R_norm=float(R.norm()),D_norm=float((R.double()@(oracle.adj[l].T@keys[l].double())).norm()),
                      nonzero_blocks=int((R.norm(dim=0)>0).sum()),clamp_hits=int((R.norm(dim=0)>=rho[i*B:(i+1)*B]*(1-1e-6)).sum()),
                      ideal_omega=payload['layer_omega'][i],**energy,seconds=time.monotonic()-t))
                del A,R
            with torch.no_grad():
                for l,w in weights(model).items():w.copy_(payload['weights'][l])
            require(all(torch.equal(w,payload['weights'][l]) for l,w in weights(model).items()),'INTENDED_COMMIT_BYTES')
            nr,kr=oracle.committed_losses()
            error=max(float((nr-torch.tensor(payload['nll'],device='cuda')).abs().max()),float((kr-torch.tensor(payload['kl'],device='cuda')).abs().max()))
            postkeys=early_keys(model,oracle.spec,contexts,4)
            low_error=float((postkeys[4]-keys[4]).abs().max())
            row['commit_gate']=dict(weight_bitwise=True,loss=numerical({'maxabs':error},{'maxabs':1e-3}),
                  L4_key=numerical({'maxabs':low_error},{'maxabs':0.0}))
            for l in LAYERS:txn.append(l,postkeys[l])
            row['history_appends']=5;row['post']=state_hash(model,history)
            del payload,x,gradient,keys,postkeys,oracle,nr,kr;oracle=None;gc.collect();torch.cuda.empty_cache()
            before_rng=torch.get_rng_state().clone();before_cuda=torch.cuda.get_rng_state_all()
            obs=observe(model,tok,allrecords[:number*B],history,number,(number-1)*B)
            write(out/f'W{number:02d}-observations.json',obs)
            require(torch.equal(torch.get_rng_state(),before_rng) and all(torch.equal(a,b) for a,b in zip(before_cuda,torch.cuda.get_rng_state_all())),'OBSERVER_RNG_MUTATION')
            require(state_hash(model,history)==row['post'],'OBSERVER_STATE_MUTATION')
            row['observation']=dict(path=str(out/f'W{number:02d}-observations.json'),summary=obs['summary'],current=obs['current'],seconds=obs['seconds'])
            row['auxiliary_post']=auxiliary(contexts)
            row.update(commit=True,seconds=time.monotonic()-start,checkpoint_saved=False,peak_gpu_bytes=torch.cuda.max_memory_allocated())
            def publish(new):
                write(out/f'B{number:03d}-committed.json',row);write(out/'ledger.json',new)
            txn.finish(row,publish)
    except Exception as e:
        trace=traceback.format_exc()
        try:write(out/'ledger.json',ledger)
        except Exception:pass
        write(out/f'B{number:03d}-failure.json',dict(status='TECHNICAL_FAILURE',error=repr(e),traceback=trace,
             solver_evidence=getattr(e,'solver_evidence',None),candidate=row,rollback_verified=txn.restored,
             attempted_oracles=oracle.calls if oracle else None,seconds=time.monotonic()-start,exact_resume='NOT_AVAILABLE'))
        raise
    print(json.dumps(dict(event='COMMITTED',phase=phase,arm=arm,batch=number,calls=result['calls'],status=result['status'],seconds=row['seconds'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',choices=['prep','pilot','main'],required=True);p.add_argument('--arm',choices=['A','B']);args=p.parse_args()
    root=args.run;config=json.loads((root/'config.json').read_text());lock=verify_lock(root,config)
    out=root/('prep' if args.phase=='prep' else args.phase+'-'+args.arm);start=time.monotonic();ledger=[]
    terminal=dict(status='STARTED',phase=args.phase,arm=args.arm,job_id=os.getenv('SLURM_JOB_ID'),checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    try:
        if args.phase!='prep':
            shared=json.loads((root/'prep/receipt.json').read_text());require(shared['config_sha256']==sha(root/'config.json') and shared['source_sha']==sha(root/'execution.lock.json') and shared['status']=='STRUCTURAL_COMPLETION','SHARED_PREP_IDENTITY')
            if args.phase=='main':
                for arm in ('A','B'):
                    pilot=json.loads((root/f'pilot-{arm}/terminal.json').read_text())
                    require(pilot['status']=='COMPLETED' and pilot['commits']==2 and pilot['config_sha256']==sha(root/'config.json') and pilot['execution_lock_sha256']==sha(root/'execution.lock.json'),'PILOT_STRUCTURAL_JOIN')
        records,pool,contexts=data(config);model,tok=load(config)
        terminal.update(torch=str(torch.__version__),transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),config_sha256=sha(root/'config.json'),execution_lock_sha256=sha(root/'execution.lock.json'))
        if args.phase=='prep':prep(root,config,model,tok,records,pool,contexts)
        else:
            out.mkdir(exist_ok=False);history=history_for(model)
            write(out/'runtime.json',dict(**terminal,W0_H0=state_hash(model,history),fresh_W0=True))
            tr=json.loads((root/'prep/teacher.json').read_text());require(sha(root/'prep/w0-teachers.pt')==tr['member']['sha256'],'TEACHER_FILE_CHANGED')
            teachers=torch.load(root/'prep/w0-teachers.pt',map_location='cpu',weights_only=True)
            for saved in teachers.values():
                require(saved['logp'].dtype==torch.float32 and bool(torch.isfinite(saved['logp']).all()) and tensor_sha(saved['logp'])==saved['sha256'],'TEACHER_CONSUMER_BUFFER_IDENTITY')
            route=json.loads((root/'prep/route.json').read_text());require(route['config_sha256']==sha(root/'config.json') and route['source_sha']==sha(root/'execution.lock.json'),'ROUTE_IDENTITY')
            shared_w0=json.loads((root/'prep/W00-observations.json').read_text())
            require(shared_w0['state']==state_hash(model,history),'SHARED_W0_IDENTITY')
            write(out/'W00-reuse.json',dict(member=member(root/'prep/W00-observations.json'),same_W0=state_hash(model,history),scope='same endpoint exact model/input/evaluator; first8 pilot subset or first2k main'))
            for b in range(1,3 if args.phase=='pilot' else 21):
                run_batch(root,args.phase,args.arm,config,model,tok,records,pool,contexts,history,ledger,b,teachers,route['route'])
        terminal['status']='COMPLETED'
    except Exception as e:
        terminal.update(status='TECHNICAL_FAILURE',error=repr(e),traceback=traceback.format_exc());print(terminal['traceback'],flush=True)
    finally:
        write(out/'actual-imports.json',import_receipt(config))
        terminal.update(seconds=time.monotonic()-start,commits=len(ledger),history_appends=sum(x['history_appends'] for x in ledger),science_calls=sum(x['solver']['calls'] for x in ledger),peak_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None)
        write(out/'terminal.json',terminal)
    return 0 if terminal['status']=='COMPLETED' else 2

if __name__=='__main__':raise SystemExit(main())
