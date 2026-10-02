"""Sealed prep/pilot/timing/main programs; no scheduler calls or auto-retries."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
import traceback
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from .common import INSTRUCTION,member,sha,write,require,digest
from .adapter import LlamaAdapter
from .inputs import CounterFactAdapter
from .allocation import Geometry
from .oracle import Oracle
from .optimize import fit
from .writer import Transaction,commit,state
from .observe import observe

def seed(config):
    n=config['settings']['seed'];random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

def verify(config):
    require(config['instruction_id']==INSTRUCTION,'WRONG_INSTRUCTION')
    require(torch.__version__==config['runtime']['torch'] and transformers.__version__==config['runtime']['transformers'],'RUNTIME_CHANGED')
    for row in config['assets']:
        p=Path(row['path']);st=p.stat()
        require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_STAT_CHANGED:'+str(p))
    require(not config['settings']['save_checkpoints'],'NO_CP_CONTRACT')

def setup(config,out):
    verify(config);seed(config)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(config['model'],local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='eager',low_cpu_mem_usage=True).to('cuda').eval()
    model.requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    adapter=LlamaAdapter(model,config['profile'])
    history={l:torch.zeros(shape[1],shape[1],dtype=torch.float32) for l,shape in adapter.dims.items()}
    contexts=json.loads(Path(config['contexts']).read_text())
    bench=CounterFactAdapter(tokenizer,contexts)
    data=json.loads(Path(config['stream']).read_text())
    require(len(data)>=2004,'SHORT_STREAM')
    write(out/'runtime.json',dict(instruction=INSTRUCTION,job_id=os.environ.get('SLURM_JOB_ID'),
        runtime=config['runtime'],device=torch.cuda.get_device_name(),capabilities=adapter.capabilities,
        checkpoint_saved=False,source_file=member(Path(__file__)),model=config['model']))
    imports={}
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if file and name.startswith(('project.run_scripts.jlz_native_joint','project.run_scripts.jlz_pilot','transformers.models.llama','transformers.masking_utils','transformers.cache_utils')) and Path(file).is_file():
            imports[name]=member(Path(file))
    write(out/'actual-imports.json',imports)
    return adapter,bench,data,history

def binding(config,phase,arm):
    return dict(instruction=INSTRUCTION,config=digest(config),phase=phase,arm=arm)

def check_receipt(path,config,phase,arm):
    doc=json.loads(Path(path).read_text())
    require(doc['status']=='COMPLETED' and doc['binding']==binding(config,phase,arm),'UPSTREAM_IDENTITY_OR_COMPLETENESS')
    return doc

def reused_w0(config):
    reuse=config.get('w0_reuse')
    require(reuse and reuse['mode']=='REUSE_ONLY_USER_DIRECTED','W0_REUSE_REQUIRED')
    require(sha(reuse['receipt']['path'])==reuse['receipt']['sha256'],'W0_BRIDGE_CHANGED')
    doc=json.loads(Path(reuse['receipt']['path']).read_text())
    require(sha(doc['observations']['path'])==doc['observations']['sha256'],'W0_OBSERVATIONS_CHANGED')
    return doc

def run_batch(a,bench,records,history,config,eta,out,route,qualification=False):
    spec=bench.prepare(records);write(out/'input.json',dict(identity=spec['identity'],ids=spec['record_ids'],rows=len(spec['row_request'])))
    before=state(a,history)
    geometry=Geometry(a,history,config['stats'],a.profile['lambda_C'])
    with Transaction(a,history) as tx:
        D,oracle,fitreceipt=fit(a,spec,geometry,eta,out/'fit',microbatch=config['settings']['fit_microbatch'],
                               route=route,qualification=qualification)
        require(state(a,history)==before,'FIT_WEIGHT_OR_HISTORY_MUTATION')
        anchors={l:x.norm(dim=0).cpu().tolist() for l,x in oracle.anchors.items()}
        actualroute=oracle.route
        # Cache state belongs to fit only; release before actual materialization.
        del oracle;gc.collect();torch.cuda.empty_cache()
        writer=commit(a,spec,D,geometry,tx,out,microbatch=config['settings']['fit_microbatch'],qualification=qualification)
        write(out/'geometry.json',geometry.records)
        receipt=dict(before=before,after=writer['after'],ids=spec['record_ids'],
            input_identity=spec['identity'],candidate_count=fitreceipt['candidates'],Adam_updates=fitreceipt['Adam_updates'],
            history_appends=len(a.sites),layers=list(a.sites),entry_anchor_norm=anchors,route=actualroute,
            checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
        write(out/'commit.json',receipt);tx.finish()
    del D,geometry;gc.collect();torch.cuda.empty_cache()
    return receipt

def timing(a,bench,data,history,config,eta,out,route):
    before=state(a,history);spec=bench.prepare(data[:config['experiment']['batch_size']])
    geometry=Geometry(a,history,config['stats'],a.profile['lambda_C'])
    oracle=Oracle(a,spec,config['settings']['fit_microbatch'],route)
    D={l:torch.zeros(d[0],len(spec['specs']),device=a.device,requires_grad=True) for l,d in a.dims.items()}
    results=[]
    for it in range(4):
        for d in D.values():d.grad=None
        torch.cuda.synchronize();start=time.monotonic()
        res=oracle.evaluate(D,backward=True,initialize=it==0)
        if it==0 and eta:
            for l in a.sites:geometry.solve(l,oracle.keys[l],entry=True)
        norm=sum((a.profile['norm_factor']*d.norm(dim=0)/oracle.anchors[l].square().sum(0)).sum() for l,d in D.items());norm.backward()
        if eta:
            from .allocation import policy
            for l,d in D.items():
                _,g,_=policy(d.detach(),geometry.entry[l],oracle.anchors[l].square().sum(0).mean());d.grad.add_(g.float())
        torch.cuda.synchronize()
        require(all(d.grad is not None and torch.isfinite(d.grad).all() for d in D.values()),'TIMING_GRADIENT')
        results.append(dict(candidate=it+1,role='warmup' if it==0 else 'measured',seconds=time.monotonic()-start,
                            nll=res['nll'],peak_allocated=torch.cuda.max_memory_allocated(),
                            peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,calls=dict(oracle.calls)))
        write(out/f'candidate-{it+1}.json',results[-1])
    require(state(a,history)==before,'TIMING_MUTATION')
    write(out/'receipt.json',dict(candidates=4,warmup=1,measured=3,Adam_updates=0,writes=0,
        no_main_carryover=True,route=route,geometry=geometry.records,results=results))
    del oracle,geometry,D;gc.collect();torch.cuda.empty_cache();seed(config)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
    p.add_argument('--phase',choices=['shared','pilot','main'],required=True)
    p.add_argument('--arm',choices=['JLZ_A','JLZ_B','SHARED'],required=True)
    p.add_argument('--attempt',required=True,type=Path);args=p.parse_args()
    config=json.loads(args.config.read_text());base=args.attempt/'output';out=base/(args.phase+'-'+args.arm)
    out.mkdir(parents=True,exist_ok=False);status='TECHNICAL_FAILED';start=time.monotonic()
    try:
        reuse=reused_w0(config) if config.get('w0_reuse') else None
        require(not (reuse and args.phase=='shared'),'W0_FORWARD_REMOVED_BY_USER')
        if args.phase!='shared' and reuse is None:check_receipt(base/'shared-SHARED/terminal.json',config,'shared','SHARED')
        route=config['settings']['route']
        if args.phase=='main':
            pilots=[check_receipt(base/f'pilot-{arm}/terminal.json',config,'pilot',arm) for arm in ['JLZ_A','JLZ_B']]
            route='full_reference' if any(x['route']=='full_reference' for x in pilots) else route
        a,bench,data,history=setup(config,out)
        initial=state(a,history);write(out/'initial-state.json',initial)
        if reuse:
            require(initial==reuse['state'],'COLD_W0_H0_REUSE_MISMATCH')
            write(out/'W0-reuse.json',dict(bridge=config['w0_reuse']['receipt'],
                status='REUSED_HISTORICAL_NO_NEW_W0_FORWARD',cold_W0_H0_exact=True,
                numerical_bitwise_equivalence='NOT_ESTABLISHED',original_state_preserved=True))
        eta=0 if args.arm=='JLZ_A' else 1
        if args.phase=='shared':
            e=config['evaluation_schedule']['endpoints'][0];lo,hi=e['ordinal_slice']
            obs=observe(a,bench,data[:hi],data[lo:hi],history,0,out/'W000',config['settings']['observer_microbatch'])
            require({k:v['denominator'] for k,v in obs['summary'].items()}=={k:e[k] for k in ('R','P','N')},'W0_DENOMINATORS')
        elif args.phase=='pilot':
            # Bounded exact RAM rollback fixture, no native fit or extra write.
            with Transaction(a,history) as tx:
                with torch.no_grad():
                    a.weights[min(a.sites)].view(-1)[0].add_(1.)
                    history[min(a.sites)].view(-1)[0].add_(1.)
            require(state(a,history)==initial and tx.rollback_verified,'PILOT_ROLLBACK')
            receipt=run_batch(a,bench,data[2000:2002],history,config,eta,out/'B001',route,True);route=receipt['route']
            observe(a,bench,data[2000:2002],data[2000:2002],history,1,out/'W001',config['settings']['observer_microbatch'])
            spec=bench.prepare(data[2002:2004]);oracle=Oracle(a,spec,config['settings']['fit_microbatch'],route)
            D={l:torch.zeros(shape[0],2,device=a.device,requires_grad=True) for l,shape in a.dims.items()}
            with torch.no_grad():oracle.evaluate(D,backward=False,initialize=True)
            require(state(a,history)==receipt['after'],'PILOT_B2_ENTRY')
            write(out/'B002-entry.json',dict(state=state(a,history),ids=spec['record_ids'],teacher_from_own_entry=True,
                 fitting=0,writes=0,entry_preparation_only=True,physical_calls=oracle.calls))
        else:
            if reuse is None:
                check_receipt(base/'shared-SHARED/terminal.json',config,'shared','SHARED')
                sharedstate=json.loads((base/'shared-SHARED/initial-state.json').read_text())
                require(initial==sharedstate,'COLD_W0_H0_MISMATCH')
            timing(a,bench,data,history,config,eta,out/'timing',route)
            require(state(a,history)==initial,'MAIN_NOT_COLD')
            commits=[];size=config['experiment']['batch_size']
            for b in range(1,config['experiment']['sequential_commits']+1):
                lo,hi=(b-1)*size,b*size
                current=data[lo:hi];entry=state(a,history)
                if commits:require(entry==commits[-1]['after'],'PREVIOUS_COMMIT_NEXT_ENTRY')
                write(out/f'B{b:03d}/entry.json',dict(state=entry,ids=[r['case_id'] for r in current],own_entry=True))
                receipt=run_batch(a,bench,current,history,config,eta,out/f'B{b:03d}',route);commits.append(receipt)
                e=config['evaluation_schedule']['endpoints'][b];startrow,endrow=e['ordinal_slice']
                obs=observe(a,bench,data[:hi],data[startrow:endrow],history,b,out/f'W{b:03d}',
                            config['settings']['observer_microbatch'],[r['case_id'] for r in current])
                require({k:v['denominator'] for k,v in obs['summary'].items()}=={k:e[k] for k in ('R','P','N')},'SCHEDULE_DENOMINATORS')
                write(out/f'B{b:03d}/complete.json',dict(commit=member(out/f'B{b:03d}/commit.json'),observer=member(out/f'W{b:03d}/summary.json')))
                if b==1:
                    spec=bench.prepare(data[hi:hi+size])
                    write(out/'initial-main-link.json',dict(main_B1_commit=True,all_layer_H_append=True,observer_restored=True,
                        B2_own_entry_state=state(a,history),B2_ids=spec['record_ids'],B2_input_identity=spec['identity'],
                        observed_level='B2_PREPARED_OWN_ENTRY_BEFORE_FIT',representative_only=True))
        status='COMPLETED'
    except BaseException as exc:
        write(out/'first-error.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                                        checkpoint_saved=False,exact_resume='NOT_AVAILABLE'))
        raise
    finally:
        write(out/'terminal.json',dict(status=status,binding=binding(config,args.phase,args.arm),route=locals().get('route'),
            seconds=time.monotonic()-start,job_id=os.environ.get('SLURM_JOB_ID'),checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
            peak_gpu_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None))

if __name__=='__main__':main()
