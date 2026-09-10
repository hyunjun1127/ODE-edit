"""Terminal-only exact raw-free reducer and paired request bootstrap."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .transfer import DEST,sha
from project.run_scripts.single_layer_cumulative_risk.records import save
from project.run_scripts.single_layer_cumulative_risk.panels import curve_rows

def read(p):return json.loads(Path(p).read_text())
def table(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def stats(x):
    a=np.asarray(x,dtype=float)
    assert len(a) and np.isfinite(a).all()
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)))
def bootstrap_mean(x):
    a=np.asarray(x,dtype=float);rng=np.random.default_rng(20260911)
    means=a[rng.integers(0,len(a),size=(2000,len(a)))].mean(1)
    lo,hi=np.quantile(means,[.025,.975])
    return dict(ci_mean_low=float(lo),ci_mean_high=float(hi),bootstrap=2000)
def identity(r):return r['panel'],r['metric'],r['case_id'],r['prompt_index'],r['identity']
def metric_check(rows):
    assert len(set(map(identity,rows)))==len(rows)
    for r in rows:
        assert np.isfinite([r['new_nll'],r['true_nll']]).all()
        expected=r['true_nll']<r['new_nll'] if r['metric']=='NS' else r['new_nll']<r['true_nll']
        assert r['success']==expected
        # Source margin always true-new, including NS (negative means NS success).
        expected_margin=r['true_nll']-r['new_nll']
        assert np.isclose(r['margin'],expected_margin,rtol=0,atol=1e-12)
def summaries(rows,entry,arm,step):
    result=[]
    for panel,metric in sorted({(r['panel'],r['metric']) for r in rows}):
        v=[r for r in rows if (r['panel'],r['metric'])==(panel,metric)]
        row=dict(entry=entry,arm=arm,step=step,panel=panel,metric=metric,numerator=sum(r['success'] for r in v),denominator=len(v),
                 evaluation_resolution='Full3900' if len(rows)==3900 else 'Curve1100',evaluation_total_pairs=len(rows))
        row['percent']=100*row['numerator']/len(v)
        for field in ('new_nll','true_nll','margin'):
            row.update({field+'_'+k:z for k,z in stats([r[field] for r in v]).items()})
        for prefix in ('new','true'):
            row[prefix+'_strict_numerator']=sum(r[prefix+'_strict'] for r in v)
            row[prefix+'_token_numerator']=sum(r[prefix+'_token_correct'] for r in v)
            row[prefix+'_token_denominator']=sum(r[prefix+'_token_count'] for r in v)
        result.append(row)
    return result
def paired(left,right,entry,contrast):
    assert set(map(identity,left))==set(map(identity,right))
    rm={identity(r):r for r in right};result=[]
    for panel,metric in sorted({(r['panel'],r['metric']) for r in left}):
        rr=[(l,rm[identity(l)]) for l in left if (l['panel'],l['metric'])==(panel,metric)]
        ids=sorted({l['case_id'] for l,r in rr})
        idx={v:i for i,v in enumerate(ids)}
        rng=np.random.default_rng(20260911)
        draws=rng.integers(0,len(ids),size=(2000,len(ids)))
        for field in ('success','new_nll','true_nll'):
            values=[[] for _ in ids]
            for l,r in rr:values[idx[l['case_id']]].append(float(l[field])-float(r[field]))
            assert len({len(v) for v in values})==1,'UNEQUAL_REQUEST_PROMPT_SUPPORT'
            a=np.asarray([np.mean(v) for v in values]);scale=100 if field=='success' else 1
            draws_mean=a[draws].mean(1)*scale;lo,hi=np.quantile(draws_mean,[.025,.975])
            result.append(dict(entry=entry,contrast=contrast,panel=panel,metric=metric,field=field,
              delta=float(a.mean()*scale),ci_low=float(lo),ci_high=float(hi),bootstrap=2000,requests=len(ids),prompts=len(rr),
              success_to_failure=sum(not l['success'] and r['success'] for l,r in rr),
              failure_to_success=sum(l['success'] and not r['success'] for l,r in rr),
              unit='pp' if field=='success' else 'NLL',paired=True,imputation=0))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,nargs='+',required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False);out=args.output
    lock=read(DEST/'input.lock.json');native_lock=read(DEST/'imports/input.lock.json');raw=[];allrows=[];aggregate=[];curves=[];trajectory=[];runindex=[];probes=[];compute=[];generation=[];endpoints={};training=[];arm_compute=[]
    for entry,spec in lock['entries'].items():
        base=Path(spec['prepared']).parent
        if entry=='Middle':
            gg=read(base/'N-generation.json')['rows'];assert len(gg)==60
            generation.append(dict(entry=entry,arm='N',numerator=sum(r['literal_prefix'] for r in gg),denominator=60,metric='literal_prefix_not_semantic_accuracy'))
            raw.append(dict(path=str(base/'N-generation.json'),sha256=sha(base/'N-generation.json'),bytes=(base/'N-generation.json').stat().st_size,kind='reused_generation'))
        for arm in ('W0','ENTRY','N'):
            p=base/f'{arm}-full.json';assert sha(p)==spec['baseline_full_files'][arm]
            rr=read(p)['rows'];metric_check(rr)
            endpoints[entry,arm]=rr;aggregate+=summaries(rr,entry,arm,0)
            curves+=summaries(curve_rows(rr,native_lock['entries'][entry]['panels']),entry,arm,0)
            raw.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,kind='reused_full'))
            for r in rr:allrows.append(dict(entry=entry,arm=arm,step=0,**r))
    for root in args.runs:
        terminal=read(root/'terminal.json');assert terminal['status']=='TERMINAL_VALID' and terminal['W0_restore']
        assert terminal['other_parameter_mutation']==0 and not (root/'failure.json').exists()
        entry=terminal['entry']
        runtime=read(root/'runtime.json')
        assert runtime['fp32_all'] and runtime['W0_sha']==lock['entries'][entry]['W0_sha']
        assert runtime['source_head']==read(DEST/'execution.lock.json')['source_head']
        assert runtime['input_lock_sha']==sha(DEST/'input.lock.json')
        assert runtime['model_revision']=='8afb486c1db24fe5011ec46dfbe5b5dccdb575c2'
        assert not runtime['autocast'] and not runtime['tf32_matmul'] and not runtime['tf32_cudnn']
        assert read(root/'application-gate.json')['max_logit_difference']==0
        for ledger_name,values in terminal['compute'].items():
            if not isinstance(values,dict):continue
            for name,value in values.items():compute.append(dict(entry=entry,run=str(root),ledger=ledger_name,name=name,value=value))
        compute.append(dict(entry=entry,run=str(root),ledger='memory',name='peak_gpu_bytes',value=terminal['peak_gpu_bytes']))
        compute.append(dict(entry=entry,run=str(root),ledger='memory',name='peak_host_kib',value=terminal['peak_host_kib']))
        compute.append(dict(entry=entry,run=str(root),ledger='storage',name='terminal_output_bytes',value=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())))
        previous_compute={}
        for arm in terminal['arms']:
            dest=root/arm;t=read(dest/'terminal.json');n=t['steps'];assert n==(16 if arm=='EP-N16' else 8)
            assert t['status']=='TERMINAL_VALID' and t['inner_compute_z']==t['inner_native_calls']==t['history_append']==0
            assert t['WN_start_sha']==lock['entries'][entry]['WN_sha']
            steps=[read(dest/f'step-{k:03d}.json') for k in range(1,n+1)]
            start_compute=steps[0]['compute']
            for phase,start,end in [('shared_entry_or_interarm_setup',previous_compute,start_compute),('arm_increment',start_compute,t['compute'])]:
                for ledger in ('counts','seconds'):
                    for name,value in end.get(ledger,{}).items():
                        delta=value-start.get(ledger,{}).get(name,0);assert delta>=0
                        if delta:arm_compute.append(dict(entry=entry,arm=arm,phase=phase,ledger=ledger,name=name,value=delta))
            previous_compute=t['compute']
            runindex.append(dict(entry=entry,arm=arm,status=t['status'],steps=n,source=runtime['source_head'],job=runtime['job'],
                         path=str(dest),terminal_sha=sha(dest/'terminal.json'),selected_weight_sha=t['selected_weight_sha'],
                         WN_start_sha=t['WN_start_sha'],risk=t['risk'],actual_fp32_risk=t['risk_actual_fp32']))
            for k,row in enumerate(steps):
                post=steps[k+1]['pre_terms'] if k+1<n else t['terms']
                ctrl=row['control'];pre=row['pre_terms']
                tr=dict(entry=entry,arm=arm,step=k+1,time=row['time'],pre_objective=pre['objective'],post_objective=post['objective'],
                  pre_edit_nll=pre['edit_nll'],post_edit_nll=post['edit_nll'],essence_kl=post['essence_kl_unweighted'],
                  action=post['normalized_native_action'],**{f:row[f] for f in ('risk_pre','risk_post','risk_actual_fp32','risk_rounding_difference','risk_linear','risk_quadratic','actual_step_norm','actual_step_energy','materialization_off_support_norm')},
                  **{f:ctrl[f] for f in ('q','q_all','q_fraction','slack','e','epsilon','factor','rank','correction_native_norm','nominal_native_norm')},
                  leakage_max=max(map(abs,ctrl['progress_leakage']),default=0),
                  nominal_edit_derivative_mean=float(np.mean(ctrl['jv0'])),
                  nominal_edit_derivative_positive=bool(np.mean(ctrl['jv0'])>0),
                  reduced_cap_exceeded=bool(row['risk_post']>0),actual_fp32_cap_exceeded=bool(row['risk_actual_fp32']>0))
                predicted=np.asarray(row['request_corrected_prediction']);actual=np.asarray(post['request_edit_nll'])-pre['request_edit_nll']
                ids=[r['case_id'] for r in endpoints[entry,'N'] if r['panel']=='Current100' and r['metric']=='RS']
                assert len(ids)==100
                for i,case in enumerate(ids):
                    training.append(dict(entry=entry,arm=arm,step=k+1,case_id=case,pre_edit_nll=pre['request_edit_nll'][i],
                      post_edit_nll=post['request_edit_nll'][i],actual_delta=float(actual[i]),
                      nominal_prediction=row['request_nominal_prediction'][i],corrected_prediction=float(predicted[i]),
                      projected_actual_prediction=row['request_applied_prediction'][i],
                      prediction_error=float(actual[i]-predicted[i]),worsened=bool(actual[i]>0),improved=bool(actual[i]<0),
                      source='six_context_train_request_mean_not_canonical_RS',added_backward=0))
                tr.update(prediction_error_mean=float((actual-predicted).mean()),prediction_error_abs_p90=float(np.quantile(abs(actual-predicted),.9)),request_worsened=int((actual>0).sum()),request_improved=int((actual<0).sum()))
                trajectory.append(tr)
                if row['nominal_shadow_request_nll'] is not None:
                    diff=np.asarray(post['request_edit_nll'])-row['nominal_shadow_request_nll']
                    probes.append(dict(entry=entry,arm=arm,kind='same_state_nominal_shadow',pre_step=k,**stats(diff),**bootstrap_mean(diff),maximum=float(diff.max()),improved=int((diff<0).sum()),worsened=int((diff>0).sum()),denominator=100,
                         first_order_difference=float(np.mean(row['request_corrected_prediction'])-np.mean(row['request_nominal_prediction']))))
            milestones=(2,4,8,16) if n==16 else (1,2,4,8)
            for k in milestones:
                p=dest/f'eval-{k:03d}.json';ev=read(p);rr=ev['rows'];assert len(rr)==(3900 if k==n else 1100);metric_check(rr)
                assert ev['panel_identity']==read(Path(lock['entries'][entry]['prepared']).parent/'N-full.json')['panel_identity']
                aggregate+=summaries(rr,entry,arm,k)
                curves+=summaries(curve_rows(rr,native_lock['entries'][entry]['panels']) if k==n else rr,entry,arm,k)
                for r in rr:allrows.append(dict(entry=entry,arm=arm,step=k,**r))
                if k==n:endpoints[entry,arm]=rr
            if (dest/'generation.json').exists():
                gg=read(dest/'generation.json')['rows'];assert len(gg)==60
                generation.append(dict(entry=entry,arm=arm,numerator=sum(r['literal_prefix'] for r in gg),denominator=60,metric='literal_prefix_not_semantic_accuracy'))
        if (root/'matched-risk.json').exists():
            pp=read(root/'matched-risk.json');ep1=read(root/'EP/step-002.json')['pre_terms']
            diff=np.asarray(ep1['request_edit_nll'])-pp['terms']['request_edit_nll']
            ep_step=read(root/'EP/step-001.json')
            probes.append(dict(entry=entry,arm='EP',kind='matched_risk_at_WN',pre_step=0,**stats(diff),**bootstrap_mean(diff),maximum=float(diff.max()),improved=int((diff<0).sum()),worsened=int((diff>0).sum()),denominator=100,
               first_order_difference=(pp['ep_edit_derivative']-pp['matched_edit_derivative'])/8,
               matched_reduced_risk=pp['risk'],ep_reduced_risk=ep_step['risk_post'],
               ep_actual_fp32_risk=ep_step['risk_actual_fp32'],matched_actual_fp32_risk='NOT_RECORDED',
               matched_velocity_risk_removal=pp['matched_velocity_risk_removal']))
            pr=read(root/'matched-risk-curve.json')['rows'];metric_check(pr);assert len(pr)==1100
            aggregate+=summaries(pr,entry,'MATCHED_RISK_PROBE',1)
            for r in pr:allrows.append(dict(entry=entry,arm='MATCHED_RISK_PROBE',step=1,**r))
            ep_curve=read(root/'EP/eval-001.json')['rows']
            for r in paired(ep_curve,pr,entry,'EP1−MATCHED_RISK_PROBE'):
                probes.append(dict(kind='matched_risk_curve_metric',**r))
        for p in sorted(root.rglob('*')):
            if p.is_file():raw.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,kind='execution'))
    assert len(runindex)==12 and len({(r['entry'],r['arm']) for r in runindex})==12 and sum(r['steps'] for r in runindex)==104
    expected={(e,a) for e in ('Early','Middle','Late') for a in ('H','R','EP')}|{('Middle',a) for a in ('EP-Free','EP-J4','EP-N16')}
    assert {(r['entry'],r['arm']) for r in runindex}==expected
    middle_roots=[r for r in args.runs if read(r/'runtime.json')['job'] in {z['job'] for z in runindex if z['entry']=='Middle'}]
    assert len(middle_roots)==2
    calibrations=[read(r/'calibration.json') for r in middle_roots]
    fixed_fields=['nu','epsilon','q_ref','d_ref_h','teacher_sha','groups','prepared_sha','native_normalizer']
    assert all(calibrations[0][k]==calibrations[1][k] for k in fixed_fields),'MIDDLE_AUXILIARY_REFERENCE_MISMATCH'
    save(out/'middle-reference-reuse.json',dict(status='EXACT_REFERENCE_VALUE_PASS',fields=fixed_fields,
          numerical_calculation_repeated_at_second_process_entry=True,inner_recalibration=0,
          G0_reuse=True,additional_G0_backward=0))
    assert len(trajectory)==104 and sum(r['kind']!='matched_risk_curve_metric' for r in probes)==12
    assert len(allrows)==124800 and len(training)==10400
    assert len({(r['entry'],r['arm'],r['step'],*identity(r)) for r in allrows})==len(allrows)
    probe_metrics=[r for r in probes if r['kind']=='matched_risk_curve_metric']
    probes=[r for r in probes if r['kind']!='matched_risk_curve_metric']
    table(out/'matched-risk-curve-paired.csv',probe_metrics)
    totals={}
    for r in compute:
        if r['ledger']=='counts':totals[r['name']]=totals.get(r['name'],0)+r['value']
    assert totals['gradient_sweeps']==95
    assert totals['edit_backward']+totals['essence_backward']==34390
    assert totals['evaluation_candidate_sequences']==179400
    assert totals['generation_sequences']==180
    assert totals['nominal_shadow_sweeps']==6
    assert totals['request_gradient_observer_bmm']==totals['edit_backward']
    assert totals['training_forward']==41692+len(args.runs) # one extra overlay gate forward per model load
    partitioned={}
    for r in arm_compute:
        if r['ledger']=='counts':partitioned[r['name']]=partitioned.get(r['name'],0)+r['value']
    assert all(partitioned.get(k,0)==v for k,v in totals.items()),'ARM_COMPUTE_PARTITION_MISMATCH'
    save(out/'compute-reconciliation.json',dict(totals=totals,planned_training_forward=41692,
          extra_overlay_gate_forward=len(args.runs),extra_physical_gate_forward=len(args.runs),
          observed_backward=34390,actual_mask_tokens_include_generation_cached_context=True,
          unique_generation_input_tokens='NOT_RECORDED_SEPARATELY',count_imputation=0))
    comparisons=[]
    equivalence=[]
    for i,left in enumerate(runindex):
        for right in runindex[i+1:]:
            if left['entry']==right['entry'] and left['selected_weight_sha']==right['selected_weight_sha']:
                equivalence.append(dict(entry=left['entry'],left=left['arm'],right=right['arm'],
                    selected_weight_sha=left['selected_weight_sha'],
                    full_evaluation_rows_exact_equal=endpoints[left['entry'],left['arm']]==endpoints[right['entry'],right['arm']],
                    stored_actual_risk_scalar_difference=left['actual_fp32_risk']-right['actual_fp32_risk'],
                    independent_unique_state_claim=False))
    table(out/'identical-endpoint-diagnostics.csv',equivalence)
    table(out/'per-arm-compute.csv',arm_compute)
    for entry in ('Early','Middle','Late'):
        for left,right in [('EP','H'),('EP','R'),('H','N'),('R','N'),('EP','N')]+([('EP-Free','EP'),('EP-J4','EP'),('EP-N16','EP')] if entry=='Middle' else []):
            comparisons+=paired(endpoints[entry,left],endpoints[entry,right],entry,left+'−'+right)
    for name,rows in [('run-index',runindex),('trajectory',trajectory),('same-state-probes',probes),('paired-summary',comparisons),('compute-summary',compute),('endpoint-and-curve-summary',aggregate),('curve-common-inventory-summary',curves),('generation-summary',generation),('training-request-metrics',training)]:table(out/(name+'.csv'),rows)
    # Numeric request rows are local-only; publication binds SHA and reducer.
    local=DEST/'analysis'/out.name/'request-metrics.csv';local.parent.mkdir(parents=True,exist_ok=False);table(local,allrows)
    save(out/'request-metrics-reference.json',dict(path=str(local),sha256=sha(local),rows=len(allrows),raw_prompts=0,local_only=True))
    save(out/'raw-input-manifest.json',dict(members=raw,source_policy='sealed native + terminal new runs only',scientific_promotion=False))
    save(out/'analysis-receipt.json',dict(status='EXACT_REDUCER_PASS',trajectories=12,steps=104,request_rows=len(allrows),same_state_probes=12,
          bootstrap=2000,imputation=0,Fixed100_across_entries_independent=False,source_input_lock_sha=sha(DEST/'input.lock.json'),scientific_promotion=False))
    print('EXACT_REDUCER_PASS',out)

if __name__=='__main__':main()
