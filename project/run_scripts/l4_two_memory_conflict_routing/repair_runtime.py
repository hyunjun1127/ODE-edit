"""Affected BF8/Frozen replay only, exact saved preparations and calibration.

N/OS/BF1, teacher/key/native/P factor generation are not rerun. Original failed
paths remain immutable. This produces replacements, not additional science arms.
"""
import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path
import torch
from .identity import *
from .geometry import Projector,Metric,dot
from .controller_repair import step
from .observer import View
from .runtime import progress,cpu,state_snapshot
from . import evaluation
from project.run_scripts.single_layer_cumulative_risk import binding
from project.run_scripts.single_layer_cumulative_risk.evaluation import materialized
from scripts.fixed_counterfact import load_prefix

def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);ledger=Ledger();model=None;phase='LOAD'
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    save(out/'run.lock.json',dict(source_head=source,job=os.environ.get('SLURM_JOB_ID'),args=vars(args),
        source=[member(Path(__file__)),member(Path(__file__).with_name('controller_repair.py'))],
        science_changes=0,repair='negative dual admitted by dimensionally invalid tolerance',
        input_sha=sha(ROOT/'control/input.lock.json')))
    try:
        torch.set_num_threads(8);torch.manual_seed(20260911)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,tok,evaltok=binding.load_model(ledger);weight=dict(model.named_parameters())[WEIGHT]
        original_w0=weight.detach().cpu().clone();ptr=weight.data_ptr()
        records=load_prefix(binding.DATA,10000);lock=json.loads((ROOT/'control/input.lock.json').read_text())
        for name in args.runs:
            run=Path(name);old=json.loads((run/'terminal.json').read_text());entry=old['entry']
            if old['batch_raw']!=100:raise ValueError('ONLY_AFFECTED_B100_REPAIR')
            dest=out/run.name;dest.mkdir();baseline=ledger.receipt()
            prep=torch.load(PREPARED[entry],weights_only=True,mmap=True,map_location='cpu')
            if not torch.equal(original_w0,prep['W0']):raise ValueError('MODEL_W0_MISMATCH')
            we=prep['We'].to('cuda')
            with torch.no_grad():weight.copy_(we)
            geom=torch.load(run/'geometry.pt',weights_only=True,mmap=True,map_location='cpu')
            metric=Metric.__new__(Metric)
            metric.p=Projector(geom['p_vectors'].to('cuda'),geom['p_complement'],geom['p_dimension'])
            metric.factors=geom['factors'].to('cuda');metric.chol=geom['chol'].to('cuda');metric.ridge=geom['ridge']
            with ledger.time('saved_metric_bind_progress'):metric.bind_progress(geom['t'].to('cuda'))
            k=geom['k'];osdata=torch.load(run/'OS.pt',weights_only=True,mmap=True,map_location='cpu')
            anchor=osdata['DA'].to('cuda');cal=cpu(torch.load(run/'calibration.pt',weights_only=True,mmap=True,map_location='cpu'))
            for field in ('sigma','epsilon','qref','terminal_budget'):cal[field]=cal[field].to('cuda')
            cal['gradients']=[g.to('cuda') for g in cal['gradients']]
            teachers={}
            for kind in ('Past','Base','BaseAudit'):
                teachers[kind]=torch.load(run/f'{kind}-We-teacher.pt',weights_only=True,mmap=True,map_location='cpu')
                if kind!='Past':teachers[kind+'W0']=torch.load(run/f'{kind}-W0-teacher.pt',weights_only=True,mmap=True,map_location='cpu')
            panel=lock['cases'][f'{entry}-B100']['panel'];inv=json.loads((run/'bank-manifest.json').read_text())
            channels=['Past','Base'];arms=['BF8']+(['Frozen-BF8'] if entry=='Middle' else [])
            def observe(state,gradient=False):
                values=[];gradients=[];harms={};view=View(model,evaltok.pad_token_id,ledger)
                for kind in channels:
                    obs,g=view.harm(state,teachers[kind],kind,gradient=gradient)
                    values.append(obs['value']);harms[kind]=obs
                    if gradient:gradients.append(g.to('cuda'))
                return torch.tensor(values,device='cuda',dtype=torch.float64),gradients,harms
            for arm in arms:
                path=dest/f'{arm}-trajectory';path.mkdir();z=torch.zeros_like(anchor);fcurrent=torch.zeros_like(cal['sigma'])
                for node in range(8):
                    phase=f'{entry}/{arm}/node{node+1}';h=.125
                    proposal=(we.double()+(node+1)*h*anchor+z).float()
                    fp,graw,harm=observe(proposal,gradient=arm!='Frozen-BF8')
                    gg=cal['gradients'] if arm=='Frozen-BF8' else [g/s for g,s in zip(graw,cal['sigma'])]
                    zn,correction,term=step(metric,z,gg,fcurrent,fp/cal['sigma'],cal['terminal_budget'],node*h,h,cal['epsilon'],cal['aa'])
                    if any(x<0 for x in term['kkt']['dual']+term['xi']):raise ValueError('NEGATIVE_DUAL_OR_SLACK')
                    actual=(we.double()+(node+1)*h*anchor+zn).float()
                    if not torch.isfinite(actual).all():raise FloatingPointError('NONFINITE_BF_ENDPOINT')
                    factual,_,actual_harm=observe(actual);fcurrent=factual/cal['sigma'];z=zn
                    term.update(node=node,actual_harm=factual.tolist(),normalized_actual=fcurrent.tolist(),
                        physical_scalar=float(dot(metric.t,actual.double()-we.double())),physical_anchor_scalar=float(dot(metric.t,anchor)),
                        weight_sha=tensor_sha(actual),first_order_error=(fcurrent-fp/cal['sigma']-torch.stack([dot(g,correction) for g in gg])).tolist())
                    tensor_save(path/f'node{node+1:02d}.pt',dict(Z=z.cpu(),We_reference=str(PREPARED[entry]),DA_reference=str(run/'OS.pt')))
                    save(path/f'node{node+1:02d}.json',dict(**term,harms=dict(proposal=harm,actual=actual_harm),compute=ledger.receipt()))
                    if node+1 in (2,4):
                        with materialized(weight,actual,ledger):evaluation.measure(model,evaltok,records,panel,ledger,path/f'current{node+1:02d}.json',current_only=True)
                    if arm!='Frozen-BF8':del graw,gg
                    del proposal,correction
                    progress(dest,'BF_NODE_COMPLETE',ledger,arm=arm,node=node+1,n=8)
                endpoint=dest/arm;endpoint.mkdir();phase=f'{entry}/{arm}/endpoint'
                with ledger.time('terminal_history_finalize'):
                    mn=prep['M'].clone();kf=k.float();mn[0]+=kf@kf.T
                    ledger.add('terminal_history_append');ledger.add('terminal_active_ledger_finalize')
                from .banks import latest
                state_snapshot(endpoint,'endpoint',actual,dict(entry_reference=str(PREPARED[entry]),M=mn,
                    active_fact_ordinals=latest(records,sorted(set(inv['seen_active'])|set(inv['current_effective'])))))
                del mn,kf
                view=View(model,evaltok.pad_token_id,ledger);harms={}
                for kind in ('Past','Base','BaseAudit'):
                    harms[kind],_=view.harm(actual,teachers[kind],kind)
                    if kind!='Past':harms[kind+'W0'],_=view.harm(actual,teachers[kind+'W0'],kind)
                save(endpoint/'harms.json',harms)
                with materialized(weight,actual,ledger):
                    evaluation.measure(model,evaltok,records,panel,ledger,endpoint/'full.json')
                    evaluation.generation(model,evaltok,records,panel['generation'],ledger,endpoint/'generation.json')
                if weight.data_ptr()!=ptr or not torch.equal(weight,we):raise ValueError('ENDPOINT_RESTORE')
                save(endpoint/'terminal.json',dict(status='TERMINAL_VALID',arm=arm,new_path=True,weight_sha=tensor_sha(actual),
                    history_append=1,inner_history_append=0,restore=True,ledger=ledger.receipt(),scientific_promotion=False))
                progress(dest,'ENDPOINT_COMPLETE',ledger,arm=arm)
            delta={kind:{k:v-baseline[kind].get(k,0) for k,v in ledger.receipt()[kind].items()} for kind in ('counts','seconds')}
            save(dest/'terminal.json',dict(status='REPAIR_TERMINAL_VALID',entry=entry,arms=arms,original_run=str(run),
                original_receipt_sha=sha(run/'terminal.json'),source_head=source,job=os.environ.get('SLURM_JOB_ID'),
                compute=delta,peak_gpu_bytes=torch.cuda.max_memory_allocated(),science_changes=0,new_scientific_paths=0))
            del metric,geom,cal,teachers,prep,osdata,anchor,z,actual,we
            torch.cuda.empty_cache()
        save(out/'terminal.json',dict(status='REPAIR_TERMINAL_VALID',runs=args.runs,compute=ledger.receipt(),scientific_promotion=False))
    except BaseException as exc:
        restored=None
        if model is not None and 'we' in locals():
            with torch.no_grad():weight.copy_(we)
            restored=bool(torch.equal(weight,we))
        save(out/'failure-boundary.json',dict(stage=phase,exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
            restore=restored,compute=ledger.receipt(),science_changes=0,failed_attempt_immutable=True))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);main(p.parse_args())
