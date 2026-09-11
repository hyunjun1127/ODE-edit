"""Endpoint/snapshot-only mandated diagnostics; never feeds a writer/controller.

This pass reuses saved physical endpoints. It performs no optimization, native
proposal, z computation, history append, or additional trajectory.
"""
import argparse
import json
import math
import os
import subprocess
import traceback
import time
from pathlib import Path
import torch
from .identity import *
from . import native
from .runtime import progress
from project.run_scripts.single_layer_cumulative_risk import binding
from project.run_scripts.single_layer_cumulative_risk.evaluation import evaluator,materialized
from scripts.fixed_counterfact import load_prefix

def base_pairs(model,tok,records,indices,ledger):
    ev=evaluator();rr=[records[i] for i in indices];pp=ev.counterfact_pairs(rr)
    new=[ev.PromptTarget(r['case_id'],'locality_target_new',j,p,r['requested_rewrite']['target_new']['str']) for r in rr for j,p in enumerate(r['neighborhood_prompts'])]
    selected=dict(NS_new=new,NS_true=pp['locality_target_true'],rewrite_true=pp['rewrite_target_true'])
    raw={}
    with ledger.time('base_audit_answer_and_NS'):
        for kind,pairs in selected.items():
            raw[kind]=ev.evaluate_pairs(model,tok,pairs,device=next(model.parameters()).device,microbatch_size=8)
            ledger.add('base_evaluation_candidate_sequences',len(pairs));ledger.add('base_evaluation_forwards',(len(pairs)+7)//8)
    ns=[]
    for a,b in zip(raw['NS_new'],raw['NS_true'],strict=True):
        if not all(math.isfinite(x['nll']) for x in (a,b)):raise FloatingPointError('BASE_NS_NONFINITE')
        if (a['case_id'],a['prompt_index'],a['prompt'])!=(b['case_id'],b['prompt_index'],b['prompt']):raise ValueError('BASE_NS_JOIN')
        ns.append(dict(case_id=a['case_id'],prompt_index=a['prompt_index'],
            identity=digest([a['case_id'],a['prompt_index'],a['prompt'],a['target'],b['target']]),
            new_nll=a['nll'],true_nll=b['nll'],success=b['nll']<a['nll']))
    truth=[dict(case_id=r['case_id'],nll=r['nll'],strict=r['all_tokens_correct'],token_count=len(r['target_token_ids'])) for r in raw['rewrite_true']]
    if not all(math.isfinite(r['nll']) for r in truth):raise FloatingPointError('BASE_TRUE_NLL_NONFINITE')
    return dict(NS=ns,rewrite_target_true=truth,controller_influence=0)

def response_components(delta,keys,goal,contexts,b):
    """Actual FP32 weight differences, applied algebraically to fixed L4 inputs.

    This is the structural down-projection response, NOT a claim of exact
    nonlinear final-logit response. Canonical/rephrase NLLs come from model eval.
    """
    change=delta.double()@keys.double().T
    norm2=goal.double().square().sum(0)
    inner=(change*goal.double()).sum(0)
    projection=torch.zeros_like(change)
    nonzero=norm2>0
    projection[:,nonzero]=goal[:,nonzero].double()*(inner[nonzero]/norm2[nonzero])[None]
    rows=[]
    for j in range(change.shape[1]):
        rows.append(dict(request_index=j//contexts,context_index=j%contexts,
            response_change_sq=float(change[:,j].square().sum()),
            signed_goal_inner=float(inner[j]),goal_sq=float(norm2[j]),
            goal_component_sq=float(projection[:,j].square().sum()),
            perpendicular_sq=float((change[:,j]-projection[:,j]).square().sum()),
            zero_goal=bool(norm2[j]==0)))
    if len(rows)!=b*contexts:raise ValueError('RESPONSE_CONTEXT_COUNT')
    return rows

def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);ledger=Ledger()
    save(out/'run.lock.json',dict(job=os.environ.get('SLURM_JOB_ID'),args=vars(args),
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source=[member(Path(__file__)),member(Path(__file__).with_name('evaluation.py'))],
        input_sha=sha(ROOT/'control/input.lock.json'),new_writer_paths=0,new_native_solves=0,new_z=0,
        purpose='predeclared endpoint-only Base NS/true-answer and per-context structural response'))
    model,tok,evaltok=binding.load_model(ledger);weight=dict(model.named_parameters())[WEIGHT]
    base=weight.detach().cpu().clone();records=load_prefix(binding.DATA,10000)
    lock=json.loads((ROOT/'control/input.lock.json').read_text())
    for runname in args.runs:
        run=Path(runname);terminal=json.loads((run/'terminal.json').read_text())
        if terminal['status']!='TERMINAL_VALID':raise ValueError('UNFINISHED_TRAJECTORY')
        entry=terminal['entry'];braw=terminal['batch_raw'];cfg=lock['cases'][f'{entry}-B{braw}']
        dest=out/f'{entry}-B{braw}';dest.mkdir();prep=torch.load(PREPARED[entry],weights_only=True,mmap=True,map_location='cpu')
        geom=torch.load(run/'geometry.pt',weights_only=True,mmap=True,map_location='cpu')
        inv=json.loads((run/'bank-manifest.json').read_text());we=prep['We'].to('cuda')
        with torch.no_grad():weight.copy_(we)
        cp,targets,raw_current=binding.load_entry(entry,records);del cp
        zmap={r['case_id']:z for r,z in zip(raw_current,targets['values'],strict=True)}
        actual=[records[i] for i in inv['current_effective']];b=len(actual)
        flat=[c for group in prep['contexts'] for c in group];ncontext=len(flat)
        patterns=[c.format(r['requested_rewrite']['prompt']) for r in actual for c in flat]
        words=[r['requested_rewrite']['subject'] for r in actual for c in flat]
        original=native.legacy.kernel();h=native.hp()
        with native.reader(ledger),ledger.time('current_all_context_entry_readout'):
            _,readout=original.get_module_input_output_at_words(model,tok,4,patterns,words,h.layer_module_tmp,h.fact_token)
        goal=torch.stack([zmap[r['case_id']] for r in actual for _ in flat]).to('cuda').double().T-readout.double().T
        q=torch.tensor([1/(len(prep['contexts'])*len(group)) for group in prep['contexts'] for _ in group],device='cuda',dtype=torch.float64)
        keys=geom['context_factor'].to('cuda').T*b**.5/q.repeat(b).sqrt()[:,None]
        wn=prep['WN'].to('cuda') if terminal['N_reuse'] else torch.load(run/'N/endpoint.pt',weights_only=True,map_location='cpu')['weight'].to('cuda')
        os_state=torch.load(run/'OS.pt',weights_only=True,map_location='cpu')['weight'].to('cuda')
        states=[('W0',base),('ENTRY',prep['We']),('N',wn.cpu())]
        for arm in terminal['arms']:
            if arm!='N':states.append((arm,torch.load(run/arm/'endpoint.pt',weights_only=True,map_location='cpu')['weight']))
        for label,statecpu in states:
            state=statecpu.to('cuda')
            response={}
            for refname,reference in [('entry',we),('native',wn),('OS',os_state)]:
                response[refname]=response_components(state.double()-reference.double(),keys,goal,ncontext,b)
            save(dest/f'{label}-response.json',dict(rows=response,case_ids=[r['case_id'] for r in actual],
                meaning='structural physical DeltaW times fixed L4 token input; not final logits',controller_influence=0))
            with materialized(weight,state,ledger):
                vals={kind:base_pairs(model,evaltok,records,inv[kind],ledger) for kind in ('Base','BaseAudit')}
            save(dest/f'{label}-base.json',dict(banks=vals,weight_sha=tensor_sha(state),controller_influence=0))
            progress(dest,'OBSERVATION_ENDPOINT',ledger,arm=label)
        for arm in terminal['arms']:
            if arm not in ('BF8','Frozen-BF8'):continue
            anchor=os_state.double()-we.double()
            for node in (2,4,8):
                z=torch.load(run/f'{arm}-trajectory/node{node:02d}.pt',weights_only=True,map_location='cpu')['Z'].to('cuda')
                state=(we.double()+node/8*anchor+z).float()
                response={name:response_components(state.double()-ref.double(),keys,goal,ncontext,b) for name,ref in [('entry',we),('native',wn),('OS',os_state)]}
                save(dest/f'{arm}-node{node:02d}-response.json',dict(rows=response,weight_sha=tensor_sha(state),controller_influence=0))
        save(dest/'terminal.json',dict(status='OBSERVATIONS_COMPLETE',trajectory_ref=str(run),trajectory_receipt_sha=sha(run/'terminal.json'),
            new_trajectory_count=0,new_native_solves=0,new_z=0,history_append=0,compute=ledger.receipt()))
    save(out/'terminal.json',dict(status='OBSERVATIONS_COMPLETE',runs=args.runs,compute=ledger.receipt(),scientific_promotion=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    try:main(args)
    except BaseException as exc:
        root=Path(args.output)
        if root.is_dir() and not (root/'failure-boundary.json').exists():
            save(root/'failure-boundary.json',dict(exception=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                restore_fact='per-materialization finally restore; outer wrapper does not infer unobserved model state',
                new_writer_paths=0,controller_influence=0,scientific_promotion=False))
        raise
