"""Autonomous E3/E4 orchestrator. No future lifelong/ordering submission path.

State snapshots below are RAM ONLY. Persistent tensors are the explicitly
approved z/K/R/delta/coefficient/history-key diagnostic factors, not W/M bundles.
"""
from pathlib import Path
import json
import time
import copy
import numpy as np
from .common import save,tensor_file,tensor_sha,digest,rng_preserved
from .geometry import geometry_summary
from .observer import evaluate_rpn,evaluate_neighborhood512,evaluate_history512

ENTRIES=(50,70,80,90)
BRANCHES=('NATIVE','SHAM','H5','H6','H56','MASS56')

class ProtectedAction:
    """All-valid-token actual local action and fixed writer request coefficients.

    Observations do not fit directions. Native C is a separate diagnostic solve;
    R@C.T reconstruction error is reported, not assumed exactly native Delta.
    Last-input-token equals prompt prediction position only for one-token targets.
    Input identity permits exact joining to the observer's scoring-position rows.
    """
    def __init__(self,rt,layer,factor):self.rt,self.layer,self.factor=rt,layer,factor;self.rows=[]
    def __enter__(self):
        import torch
        from .component_runner import InputBinding
        self.binding=InputBinding(self.rt.model);self.binding.__enter__()
        dev=self.rt.weights[self.layer].device
        self.C=self.factor['C'].to(dev);self.delta=self.factor['delta'].to(dev)
        def inspect(module,args):
            with torch.no_grad():
                keys=args[0];mask=self.binding.mask();coef=keys@self.C
                action=torch.nn.functional.linear(keys,self.delta)
                for i in range(len(keys)):
                    a=action[i][mask[i]].double();c=coef[i][mask[i]].double()
                    energy=c.square().sum(0);top=torch.argsort(energy,descending=True,stable=True)[:5]
                    self.rows.append(dict(forward_index=len(self.binding.records)-1,batch_row=i,valid_tokens=len(a),
                        local_action_energy=float(a.square().sum()),local_action_mean=float(a.square().sum(1).mean()),
                        last_valid_input_action_energy=float(a[-1].square().sum()),
                        request_coefficient_sum=c.sum(0).cpu().tolist(),request_coefficient_energy=energy.cpu().tolist(),
                        largest_request_columns=top.cpu().tolist(),
                        last_valid_input_coefficients=c[-1].cpu().tolist(),
                        last_input_is_prompt_prediction='ONLY_FOR_ONE_TOKEN_TARGET; join exact observer token positions otherwise'))
        self.handle=self.rt.model.model.layers[self.layer].mlp.down_proj.register_forward_pre_hook(inspect)
        return self
    def __exit__(self,*args):
        self.handle.remove();self.binding.__exit__(*args);del self.C,self.delta

def verify_sham(native,sham,native_observation,sham_observation,path,*,policy='BLOCK_ON_NUMERICAL_MISMATCH'):
    """No historical waiver/numeric tolerance silently imported.

    Record actual differences first. Without a pre-established alternative
    envelope an unequal control is UNRESOLVED, never a quality-based fallback.
    """
    import torch
    import math
    for observation in (native_observation,sham_observation):
        groups=[observation['N']['rows']]
        for section in ('current','history'):
            if section in observation:
                groups.extend(value['rows'] for value in observation[section]['metrics'].values())
        for rows in groups:
            for row in rows:
                for key in ('true_nll','new_nll'):
                    if key in row:assert math.isfinite(float(row[key])),'SHAM_NONFINITE_NLL'
    differences=[]
    for layer in (4,5,6,7,8):
        for field in ('K','R','delta'):
            a=native['factors'][layer][field];b=sham['factors'][layer][field]
            assert a.shape==b.shape and a.dtype==b.dtype,'SHAM_SHAPE_OR_DTYPE_MISMATCH'
            assert torch.isfinite(a).all() and torch.isfinite(b).all(),'SHAM_NONFINITE'
            differences.append(dict(layer=layer,field=field,exact=torch.equal(a,b),
                max_abs=float((a.double()-b.double()).abs().max()),
                relative_frobenius=float(torch.linalg.vector_norm(a.double()-b.double())/torch.linalg.vector_norm(a.double())) if torch.linalg.vector_norm(a.double()) else None))
    for metric in ('RS','PS','NS'):
        a=native_observation['current']['metrics'][metric]['rows'];b=sham_observation['current']['metrics'][metric]['rows']
        assert [r['identity'] for r in a]==[r['identity'] for r in b]
        for field in ('new_nll','true_nll'):
            differences.append(dict(metric=metric,field=field,exact=all(x[field]==y[field] for x,y in zip(a,b)),max_abs=max(abs(x[field]-y[field]) for x,y in zip(a,b))))
    a=native_observation['N']['rows'];b=sham_observation['N']['rows']
    assert [r['identity'] for r in a]==[r['identity'] for r in b]
    differences.append(dict(metric='N512',field='true_nll',exact=all(x['true_nll']==y['true_nll'] for x,y in zip(a,b)),max_abs=max(abs(x['true_nll']-y['true_nll']) for x,y in zip(a,b))))
    if 'history' in native_observation or 'history' in sham_observation:
        for metric in ('RS','PS'):
            a=native_observation['history']['metrics'][metric]['rows'];b=sham_observation['history']['metrics'][metric]['rows']
            assert [r['identity'] for r in a]==[r['identity'] for r in b]
            for field in ('new_nll','true_nll'):
                differences.append(dict(metric='H512/'+metric,field=field,exact=all(x[field]==y[field] for x,y in zip(a,b)),max_abs=max(abs(x[field]-y[field]) for x,y in zip(a,b))))
    okay=all(r['exact'] for r in differences)
    from .numerical_policy import annotate
    receipt=annotate(dict(status='PASS' if okay else 'NUMERICAL_CONTROL_NOT_ESTABLISHED',differences=differences,
        predeclared_tolerance=0,other_backend_repeat_envelope='NOT_ESTABLISHED',
        scientific_failure=False,native_result_preserved=True,SHAM_subtract_readd_roundoff_not_hidden=True),policy)
    save(path,receipt)
    if receipt['blocks_execution']:raise RuntimeError('SHAM_NUMERICAL_CONTROL_NOT_ESTABLISHED; no downstream E4 promotion')
    return receipt

class Observations:
    def __init__(self,rt,entry,panels):
        self.rt=rt;self.entry=entry;self.panels=panels
        self.cal_ids=[i for c in panels['geometry']['cohorts'].values() for i in c['calibration_case_ids']]
        assert len(self.cal_ids)==len(set(self.cal_ids))==512
        self.X=[rt.byid[i] for i in self.cal_ids]
        self.current=rt.rows[entry*100:(entry+1)*100]
        self.received=rt.rows[:entry*100]
        self.seen_tokens=set()
        for row in self.received:
            self.seen_tokens.update(rt.evalt.encode(' '+row['requested_rewrite']['target_new']['str'].lstrip(),add_special_tokens=False))

    def __call__(self,endpoint_id,out,*,full=True):
        rt=self.rt;out=Path(out);out.mkdir(parents=True,exist_ok=True)
        start=time.monotonic();before=rt.signature()
        with rng_preserved():
            capture=rt.capture(self.X)
            xs={str(l):geometry_summary(v.numpy()) for l,v in capture['means'].items()}
            tensor_file(out/'X512-writer-means.pt',{'keys':capture['means'],'case_ids':self.cal_ids})
            n=evaluate_neighborhood512(rt.model,rt.evalt,self.panels['neighborhood'],device='cuda',endpoint_id=endpoint_id,postseal=True,source_root=rt.repo,edit_target_token_ids=self.seen_tokens)
            save(out/'N512.json',n)
            row=dict(endpoint_id=endpoint_id,seal=before,X_geometry=xs,X_tokens=capture['token_receipt'],N=n['compact'])
            if full:
                current=evaluate_rpn(rt.model,rt.evalt,self.current,device='cuda',endpoint_id=endpoint_id,postseal=True,source_root=rt.repo,include_neighborhood=True)
                history=evaluate_history512(rt.model,rt.evalt,self.panels['history'],records_by_case=rt.byid,received_records=self.received,device='cuda',endpoint_id=endpoint_id,postseal=True,source_root=rt.repo)
                save(out/'current.json',current);save(out/'H512.json',history)
                row.update(current=current['compact'],history=history['compact'])
        assert rt.signature()==before,'OBSERVER_W_M_CONTEXT_CURSOR_MUTATION'
        row.update(seconds=time.monotonic()-start,observer_nonmutation=True,history_appends=0)
        save(out/'observation.json',row)
        return row

def read_panels(rt):
    return {k:json.loads((rt.design/name).read_text()) for k,name in [('geometry','geometry-panels.json'),('history','history512.json'),('neighborhood','neighborhood512.json')]}

def _stage_snapshot(rt,entry_M):
    # Immutable shared entry M reference avoids five 4.1GB redundant snapshots.
    from .common import rng_get
    return dict(w={l:w.detach().cpu().clone() for l,w in rt.weights.items()},m=entry_M,
                rng=rng_get(),cursor=rt.batch,context=copy.deepcopy(rt.contexts),cp=rt.cp)

def run_writers(rt,out,gate_dir,*,reuse=None):
    import torch
    from .native_writer import write
    from .component_runner import run_components
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    panels=read_panels(rt);gate_dir=Path(gate_dir)
    assert json.loads((gate_dir/'READY.json').read_text())['status']=='PASS'
    timestamp=torch.load(gate_dir/'timestamp-bank.pt',map_location='cpu',weights_only=True)
    stamp=timestamp['bank'];ids=timestamp['case_ids']
    assert ids==[r['case_id'] for r in panels['history']['records']]
    stamp={int(l):k.T.contiguous() for l,k in stamp.items()}
    ledger=[];total_native_targets=0;diagnostic_cache={}
    new_native_targets=0
    from .writer_diagnostics import summarize
    for entry in ENTRIES:
        rt.set_state(entry);entry_signature=rt.signature();entry_state=rt.snapshot()
        observe=Observations(rt,entry,panels)
        native=None;pre_states={};native_delta={};stage_rows={}
        for branch in BRANCHES:
            rt.restore(entry_state);assert rt.signature()==entry_signature
            bdir=out/f'W{entry:03d}'/branch;bdir.mkdir(parents=True)
            current_banks={};current_bank_states={}
            def current(layer,runtime):
                state=runtime.signature()
                if layer in current_banks:
                    assert current_bank_states[layer]==state,'STALE_HISTORY_BANK'
                    return current_banks[layer]
                capture=runtime.capture([runtime.byid[i] for i in ids])
                bank=capture['means'][layer].T.contiguous()
                tensor_file(bdir/f'current-H512-L{layer}.pt',{'K':bank,'case_ids':ids,'state':runtime.signature()})
                current_banks[layer]=bank;current_bank_states[layer]=state
                return bank
            def callback(stage,runtime,info):
                if branch=='NATIVE' and stage in ('W4','W5'):
                    pre_states[int(stage[1:])+1]=_stage_snapshot(runtime,entry_state['m'])
                if stage=='entry':current(4,runtime)
                elif stage in ('W4','W5','W6','W7'):current(int(stage[1:])+1,runtime)
                if stage in ('W4','W5','W6','W7','W8'):
                    layer=int(stage[1:]);factor=info['factors'][layer]
                    with ProtectedAction(runtime,layer,factor) as actions:
                        row=observe(f'W{entry:03d}/{branch}/{stage}',bdir/'stages'/stage,full=stage in ('W5','W6','W8'))
                    save(bdir/'stages'/stage/'protected-actions.json',dict(layer=layer,rows=actions.rows,
                        input_binding=actions.binding.records,coefficient_sha256=tensor_sha(factor['C']),
                        calibration_or_selection_use=False,C_reconstruction_relative=factor.get('reconstruction_relative_frobenius')))
                else:row=observe(f'W{entry:03d}/{branch}/{stage}',bdir/'stages'/stage,full=stage in ('entry','history'))
                stage_rows[(branch,stage)]=row
            reused=bool(reuse and entry==50 and branch in ('NATIVE','SHAM'))
            if reused:
                from .writer_reuse import restore_completed_branch
                result=restore_completed_branch(rt,entry_state,reuse,branch,bdir,pre_states)
                for layer in (4,5,6,7,8):
                    bank=torch.load(bdir/f'current-H512-L{layer}.pt',map_location='cpu',weights_only=True,mmap=True)
                    assert bank['case_ids']==ids
                    current_banks[layer]=bank['K']
                stage_rows[(branch,'history')]=json.loads((bdir/'stages/history/observation.json').read_text())
            else:
                result=write(rt,rt.current_requests(entry),bdir/'write',branch=branch,
                             shared_z=None if native is None else native['z'],stamp=stamp,current=current,
                             bank_ids=ids,stage_callback=callback)
            if branch=='NATIVE':
                native=result;total_native_targets+=100
                if not reused:new_native_targets+=100
                native_delta={l:f['delta'] for l,f in result['factors'].items()}
            else:
                # Own branch K/R/solve/history are fresh; only same-entry z shared.
                assert result['z']['binding']==native['z']['binding']
            if branch=='SHAM':
                verify_sham(native,result,stage_rows[('NATIVE','history')],stage_rows[('SHAM','history')],bdir/'sham-control.json',
                            policy=getattr(rt,'numerical_comparison_policy','BLOCK_ON_NUMERICAL_MISMATCH'))
            assert set(current_banks)=={4,5,6,7,8},'MISSING_PREWRITE_HISTORY_BANK'
            if not (bdir/'writer-modes.json').exists():
                diagnostics=summarize(rt,result['factors'],stamp,current_banks,entry_state['m'],diagnostic_cache)
                save(bdir/'writer-modes.json',diagnostics)
            ledger.append(dict(entry=entry,branch=branch,status='COMPLETED',receipt=str(bdir/'write/receipt.json'),native_targets=100 if branch=='NATIVE' else 0,
                               reused_completed_branch=reused,new_native_targets=100 if branch=='NATIVE' and not reused else 0))
            if reuse and entry==50 and branch=='H5':
                rt.restore(entry_state);assert rt.signature()==entry_signature
                save(out/'repair-initial.json',dict(status='REPAIRED_EXECUTION_INITIAL_VALID',
                    scope='prior W50 NATIVE/SHAM restored and reused; new H5 completed and entry restored',
                    numerical_equivalence='NOT_ESTABLISHED_OBSERVATION_ONLY_USER_DIRECTED',
                    numerical_policy=getattr(rt,'numerical_comparison_policy',None),
                    completed_new_branch='H5',new_native_targets=0,new_resume_checkpoint_saved=False))
            if branch!='NATIVE':del result
        assert set(pre_states)=={5,6}
        run_components(rt,entry,pre_states,native_delta,native['z']['targets'],rt.current_requests(entry),panels['geometry'],observe,out/f'W{entry:03d}'/'components')
        if entry==50:
            save(out/'G2.json',dict(status='EXECUTION_COMPLETED_NUMERICAL_COMPARISONS_OBSERVER_ONLY' if getattr(rt,'numerical_comparison_policy',None)=='OBSERVATION_ONLY_USER_DIRECTED' else 'PASS',scope='W50 B51 native100 plus SHAM and full-delta controls',native_targets=100,
                branch_receipts=[str(out/'W050'/b/'write/receipt.json') for b in ('NATIVE','SHAM')],
                observer_paths=[str(out/'W050'/b/'stages/history/observation.json') for b in ('NATIVE','SHAM')],
                technical_gates=str(gate_dir/'READY.json'),quality_improvement_required=False,
                note='SHAM arithmetic differences preserved; bitwise equality to NATIVE is not assumed'))
        rt.restore(entry_state);assert rt.signature()==entry_signature
        del native,pre_states,native_delta,entry_state
        save(out/f'W{entry:03d}'/'entry-terminal.json',dict(status='COMPLETED',entry=entry,branches=list(BRANCHES),native_targets=100,restored_entry=entry_signature))
    assert total_native_targets==400
    save(out/'terminal.json',dict(status='COMPLETED',entries=list(ENTRIES),branches=ledger,native_targets=total_native_targets,
        new_native_targets=new_native_targets,reused_native_targets=total_native_targets-new_native_targets,
        followups='FOLLOWUP_NOT_SUBMITTED',new_resume_checkpoint_saved=False))
    return ledger
