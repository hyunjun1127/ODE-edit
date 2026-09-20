"""One arm, one cold process, 100 uninterrupted RAM commits, no checkpoints.

The agent does not observe this program after release. B2 is an internal
identity/connection check, never a quality gate or a separate pilot job.
"""
import argparse
import gc
import hashlib
import json
import resource
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from project.run_scripts.en_adaptive_nullspace.native import requests_from_records
from project.run_scripts.en_adaptive_nullspace.current import capture_current
from project.run_scripts.en_adaptive_nullspace.geometry import build_geometry, request_alias_weights
from project.run_scripts.en_adaptive_nullspace.selector import select_arms
from project.run_scripts.en_adaptive_nullspace.controller import run_controller
from .runtime import Runtime, ARMS
from .objective import Objective
from .history import HistoryLedger
from . import observers
from .io import save, save_gzip
from .block_diagnostics import block_spectrum


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def compact(value):
    return {k:v for k,v in value.items() if k not in ('weight','gradient','actual_delta','history')}


def same_weight(a,b):
    return torch.equal(a.contiguous().view(torch.int32),b.contiguous().view(torch.int32))


def observe(rt, state, ledger, batch, records, entry, native, selected, selected_history, bdir):
    """Official inputs are only requested after the selection seal exists."""
    if not (bdir/'SELECTION_SEALED.json').is_file():
        raise ValueError('OBSERVER_BEFORE_SELECTION_SEAL')
    start=time.monotonic()
    plan=state.plan(records,ledger,batch)
    bycase={r['case_id']:r for r in records}
    evaluated=[]; values={}; timings={}; aliases={}
    try:
        if plan['missing_W0_case_ids']:
            rt.install(rt.W0)
            t=time.monotonic()
            base=observers.evaluate(rt.model,rt.etok,[bycase[c] for c in plan['missing_W0_case_ids']])
            state.remember_baseline(base)
            timings['W0_new_cases']=time.monotonic()-t
            save_gzip(bdir/'W0-new-observer.json.gz',base)
            del base
        for name,weight in (('entry',entry),('native',native),('selected',selected)):
            alias=next((old for old,w in evaluated if same_weight(w,weight)),None)
            if alias is not None:
                values[name]=values[alias];aliases[name]=alias;timings[name]=0.
            else:
                rt.install(weight)
                t=time.monotonic()
                values[name]=observers.evaluate(rt.model,rt.etok,plan['records'])
                timings[name]=time.monotonic()-t
                evaluated.append((name,weight))
        cost=dict(seconds=time.monotonic()-start,endpoint_seconds=timings,
            exact_FP32_endpoint_aliases=aliases,observed_requests=len(plan['request_ids']),
            new_accuracy_extra_forwards=0,method_throughput_excluded=True)
        receipt=state.record(batch,plan,native=values['native'],selected=values['selected'],
            entry=values['entry'],bank_before_current=selected_history,
            bank_after_commit=ledger.bank_ids,cost=cost)
        evidence=dict(**values,**state.evidence(plan))
        save_gzip(bdir/'observer-evidence.json.gz',evidence)
        save_gzip(bdir/'observer.json.gz',receipt)
        return dict(seconds=time.monotonic()-start,cohorts=receipt['cohorts'],cost=cost)
    finally:
        rt.install(selected)
        rt.guard()


def run(config_path):
    config_path=Path(config_path).absolute();config=json.loads(config_path.read_text())
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();rt=None;phase='LOAD';batch=0;last_commit=0
    counts=dict(native_batches=0,native_requests=0,reference_gradient_states=0,candidates=0,
                weighted_SVD=0,logical_commits=0,checkpoint_writes=0)
    try:
        rt=Runtime(output,config)
        basis=np.load(config['basis'],mmap_mode='r',allow_pickle=False)
        if basis.shape!=(14336,14326) or basis.dtype!=np.float64:
            raise ValueError('EXACT_FIXED_PSTAR_BASIS')
        obj=Objective(rt.model,rt.etok,config['generated_root'],config['reference_inputs'],
                      config['history_root'],basis,expected_map_seal=config.get('map_seal'))
        rt.objective=obj
        ledger=HistoryLedger(config['arm'])
        observed=observers.ObserverState(config['arm'])
        state_weight=rt.W0.clone();state_M=torch.zeros_like(rt.P)
        save(output/'execution-start.json',dict(config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            arm=config['arm'],entry='FRESH_W0_ZERO_M4',maximum_batch=100,
            parent_actual_execution='5d452221288f3b924e1737578f11aaa654594422',
            parent_analysis='abd08ee1dde24ca975aace49c925a4ba00211004',
            parent_precision='NOT_ESTABLISHED',new_factor_diagnostics='INTERNAL_ONCE_NO_QUALITY_GATE',
            checkpoints=False,exact_crash_resume='NOT_AVAILABLE',monitoring_agent_required=False))
        for batch in range(1,101):
            bdir=output/f'B{batch:03d}';bdir.mkdir()
            batch_start=time.monotonic();records=rt.records[(batch-1)*100:batch*100]
            if len(records)!=100:
                raise ValueError('FIXED_BATCH_SIZE')
            terminal=batch==100;entry=state_weight
            phase='PREPARE_LEDGER';preparation=ledger.prepare(batch,records,terminal=terminal)
            save_gzip(bdir/'history-preparation.json.gz',preparation)
            phase='NATIVE';rt.install(entry)
            t=time.monotonic();fit=rt.native_runner.fit(requests_from_records(records),state_M)
            WN=fit['weight'];delta=fit['actual_delta'];obj.rebind(WN)
            native_seconds=time.monotonic()-t
            counts['native_batches']+=1;counts['native_requests']+=100
            save(bdir/'native.json',dict(receipt=fit['receipt'],seconds=native_seconds,
                actual_delta_norm=float(delta.norm()),batch=batch,arm=config['arm']))
            phase='CURRENT_GEOMETRY';t=time.monotonic()
            captured=capture_current(rt.model,rt.tok,rt.etok,requests_from_records(records),rt.context)
            captured['manifest']['K_sha256']=hashlib.sha256(captured['K'].contiguous().numpy().tobytes()).hexdigest()
            save_gzip(bdir/'current-inputs.json.gz',captured['manifest'])
            del captured['oracle'];gc.collect()
            geo=build_geometry(captured['K'],captured['weights'],captured['representative_indices'],basis)
            geometry_seconds=time.monotonic()-t;counts['weighted_SVD']+=1
            phase='HISTORY_SELECTION_AND_GRADIENT';t=time.monotonic()
            G,native_objective,bank=obj.prepare(WN,[ledger.versions[i] for i in preparation['pool_ids']],
                                               arm=config['arm'],batch=batch)
            frozen=ledger.select(preparation,bank['selected_ids'])
            save_gzip(bdir/'history-selection.json.gz',dict(bank['receipt'],
                selected_ids=bank['selected_ids'],weights=bank['weights'],bank_identity=obj.bank_identity,
                preparation_id=preparation['preparation_id']))
            counts['reference_gradient_states']+=1
            gradient_seconds=time.monotonic()-t
            save_gzip(bdir/'native-objective.json.gz',native_objective)
            phase='ADAPTIVE_SELECTOR';t=time.monotonic()
            spectrum=geo.gradient_spectrum(G,delta,native_objective['J'])
            blocks=block_spectrum(geo,bank['reference_gradient'],bank['gradient'])
            del bank['reference_gradient'],bank['gradient']
            selection=select_arms(spectrum)
            row=selection['selected']['EN_ADAPT']
            spectrum['G_sha256']=hashlib.sha256(G.contiguous().numpy().tobytes()).hexdigest()
            save_gzip(bdir/'spectrum.json.gz',dict(geometry=geo.diagnostic,spectrum=spectrum,
                selected=row,frontiers=selection['frontiers'],gradient_block_decomposition=blocks,primary_epsilon=.05,
                algebra_only_epsilons=[.01,.1],seconds=time.monotonic()-t))
            D=-row['eta']*geo.direction(G,row)
            input_id=identity(dict(batch=batch,arm=config['arm'],
                current=captured['manifest']['identity_sha256'],K=captured['manifest']['K_sha256'],
                bank=obj.bank_identity,reference_manifest=obj.store.receipt['manifest_sha256']))
            phase='CONTROLLER';t=time.monotonic()
            result=run_controller(WN,G,D,native_objective,obj.evaluate,geo,
                native_norm=spectrum['native_norm'],native_action=spectrum['native_action'],
                input_identity=input_id,arm=config['arm'],
                request_columns=request_alias_weights(captured['manifest']))
            controller_seconds=time.monotonic()-t
            counts['candidates']+=result['evaluations']
            selected=result['weight'];selected_history=list(bank['selected_ids'])
            save_gzip(bdir/'controller.json.gz',compact(result))
            save(bdir/'SELECTION_SEALED.json',dict(batch=batch,arm=config['arm'],status=result['status'],
                selected_trial=result.get('selected_trial'),evaluations=result['evaluations'],
                objective={k:result['objective'][k] for k in ('J','L_R','L_H')},
                bank_identity=obj.bank_identity,official_metrics_used=False))
            del captured,geo,G,D,delta,fit,spectrum,selection,bank,result
            gc.collect()
            phase='STAGE_COMMIT';rt.install(selected)
            # Native M is an owned returned RAM copy. Both ledger and M swap
            # only after all required new teacher artifacts have published.
            t=time.monotonic()
            committed=rt.native_runner.finalize(requests_from_records(records),state_M)
            teachers=obj.capture(preparation['new_versions'],selected) if not terminal else dict(bindings={},receipt=dict(new_versions=0,terminal_unused_teacher_forward=False))
            commit=ledger.commit(batch,records,selected_history,teachers['bindings'],preparation=preparation,terminal=terminal)
            state_weight=selected;state_M=committed['history'];last_commit=batch;counts['logical_commits']+=1
            save_gzip(bdir/'ledger-commit.json.gz',commit)
            save_gzip(bdir/'teacher-bindings.json.gz',teachers)
            save(bdir/'COMMIT.json',dict(batch=batch,arm=config['arm'],native=committed['receipt'],
                history_bank=len(ledger.bank_ids),pending=len(ledger.pending_ids),
                active_versions=len(ledger.active_ids()),exactly_once=True,checkpoint=False))
            commit_seconds=time.monotonic()-t
            phase='POSTSELECTION_OBSERVER'
            obs=observe(rt,observed,ledger,batch,rt.records,entry,WN,selected,selected_history,bdir)
            rt.guard()
            # Finite/identity/connection checks only; scientific fallback and
            # lower official scores are ordinary outcomes and continue.
            if not torch.isfinite(state_M).all() or len(ledger.bank_ids)>512 or len(ledger.pending_ids)>100:
                raise ValueError('NATIVE_HISTORY_LEDGER_CONNECTION')
            if batch==2:
                save(output/'B2_CONNECTION.json',dict(status='CONNECTED_CONTINUING_SAME_RAM',
                    native_batches=rt.native_runner.fit_batches,native_appends=rt.native_runner.history_appends,
                    logical_commits=counts['logical_commits'],weight_matches_selected=same_weight(state_weight,rt.W.detach().cpu()),
                    quality_gate=False,separate_job=False,checkpoint=False,agent_observation_required=False))
            save(bdir/'batch-cost.json',dict(batch=batch,arm=config['arm'],seconds=time.monotonic()-batch_start,
                native=native_seconds,geometry=geometry_seconds,gradient=gradient_seconds,
                controller=controller_seconds,commit_teacher=commit_seconds,observer=obs['seconds'],
                cumulative=counts,history_counts=obj.replay.counts,
                reference_counts=obj.reference.counts,host_maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cuda_peak_bytes=torch.cuda.max_memory_allocated(),save_checkpoints=False))
            del entry,WN,selected,preparation,frozen,committed,teachers,commit,obs,native_objective
            gc.collect()
        phase='FINAL_CPU_REPORT'
        from . import report
        summary=report.build(output,config['arm'])
        report.write(summary,output/'report')
        rt.install(rt.W0);rt.guard()
        save(output/'TERMINAL.json',dict(status='COMPLETED',arm=config['arm'],counts=counts,
            reference_counts=obj.reference.counts,history_counts=obj.replay.counts,
            seconds=time.monotonic()-start,final_weight_restored_to_W0=True,
            save_checkpoints=False,exact_crash_resume='NOT_AVAILABLE',precision='NOT_ESTABLISHED'))
    except BaseException as error:
        rollback=dict(attempted=rt is not None,status='NOT_AVAILABLE')
        if rt is not None:
            try:
                rt.install(rt.W0);rt.guard();rollback['status']='W0_RESTORED_NONSELECTED_GUARD_PASS'
            except BaseException as cleanup:
                rollback.update(status='RESTORE_NOT_VERIFIED',error=repr(cleanup))
        save(output/'technical-failure.json',dict(error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),batch=batch,phase=phase,last_logical_commit=last_commit,
            elapsed_seconds=time.monotonic()-start,counts=counts,rollback=rollback,
            no_scientific_fallback_claim=True,exact_crash_resume='NOT_AVAILABLE',save_checkpoints=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    run(parser.parse_args().config)
