"""CPU-only independent canonical reducer and B1 factual publication.

Reads finished BPCW output only. Never imports model/runtime/evaluator or submits.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import numpy as np
from .provenance import create_bytes,create_json,sha,now

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def member(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def csvout(path,rows):
    if not rows:return
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    create_bytes(path,stream.getvalue().encode())

def reduce_raw(obj,case_order):
    result={};raw=obj['raw']
    for tag,prefix,multiple in [('RS','rewrite',1),('PS','rephrase',2),('NS','locality',10)]:
        new=raw[prefix+'_target_new'];true=raw[prefix+'_target_true']
        if len(new)!=100*multiple or len(true)!=len(new):raise ValueError('PAIR_CARDINALITY:'+tag)
        seen=set();rows=[]
        for i,(n,t) in enumerate(zip(new,true,strict=True)):
            identity=(n['case_id'],n['prompt_index'],n['prompt'])
            if identity!=(t['case_id'],t['prompt_index'],t['prompt']):raise ValueError('PAIR_IDENTITY')
            if (n['case_id'],n['prompt_index'])!=(case_order[i//multiple],i%multiple):raise ValueError('PAIR_ORDER')
            key=digest([*identity,n['target'],t['target']])
            if key in seen:raise ValueError('DUPLICATE_PAIR')
            seen.add(key);a=float(n['nll']);b=float(t['nll'])
            if not(math.isfinite(a) and math.isfinite(b)):raise ValueError('NONFINITE_NLL')
            success=b<a if tag=='NS' else a<b
            rows.append(dict(case_id=n['case_id'],prompt_index=n['prompt_index'],identity=key,
                new_nll=a,true_nll=b,desired_margin=a-b if tag=='NS' else b-a,success=success,tie=a==b,
                new_strict=bool(n['all_tokens_correct']),true_strict=bool(t['all_tokens_correct'])))
        count=sum(r['success'] for r in rows)
        recorded=obj['metrics'][tag]
        if (recorded['numerator'],recorded['denominator'])!=(count,len(rows)):raise ValueError('STORED_METRIC_MISMATCH')
        for a,b in zip(rows,recorded['rows'],strict=True):
            if any(a[k]!=b[k] for k in ['identity','case_id','prompt_index','new_nll','true_nll','success','new_strict','true_strict']):
                raise ValueError('RECORDED_ROW_MISMATCH')
        result[tag]=rows
    return result

def pair(left,right):
    if len(left)!=len(right) or [r['identity'] for r in left]!=[r['identity'] for r in right]:raise ValueError('CROSS_ENDPOINT_IDENTITY')
    return [dict(identity=a['identity'],case_id=a['case_id'],prompt_index=a['prompt_index'],
        before_success=a['success'],after_success=b['success'],lost=a['success'] and not b['success'],
        gained=not a['success'] and b['success'],new_nll_delta=b['new_nll']-a['new_nll'],
        true_nll_delta=b['true_nll']-a['true_nll'],desired_margin_delta=b['desired_margin']-a['desired_margin'])
        for a,b in zip(left,right,strict=True)]

def refsummary(scan,arm,split,eos_ids=()):
    positions=[p for d in scan['documents'] for p in d['positions']]
    return dict(arm=arm,split=split,documents=len(scan['documents']),positions=len(positions),
        retained_sequences=sum(d['all_choices'] for d in scan['documents']),
        token_flips=sum(not p['preserved'] for p in positions),base_EOS_sequences=sum(d['eos'] for d in scan['documents']),
        base_censored_sequences=sum(d['censored'] for d in scan['documents']),
        candidate_TF_EOS_choices=sum(p['choice'] in eos_ids for p in positions),
        candidate_TF_EOS_sequences=sum(any(p['choice'] in eos_ids for p in d['positions']) for d in scan['documents']),
        minimum_margin=min(p['margin'] for p in positions),mean_d=statistics.mean(p['d'] for p in positions),
        max_d=max(p['d'] for p in positions),outside_base_top8=sum(p['outside_base_top8'] for p in positions))

def audit_scan(scan,documents):
    """Stored full-vocabulary results, not a new full-vocabulary forward."""
    if len(scan['documents'])!=documents:raise ValueError('REFERENCE_DOCUMENT_COUNT')
    ids=[d['source_row_id'] for d in scan['documents']]
    if len(set(ids))!=documents:raise ValueError('REFERENCE_DUPLICATE')
    flips=0;retained=0;positions=0
    for doc in scan['documents']:
        ps=doc['positions'];positions+=len(ps)
        if len({p['position'] for p in ps})!=len(ps):raise ValueError('REFERENCE_POSITION_DUPLICATE')
        for p in ps:
            if not all(math.isfinite(p[k]) for k in ['margin','kappa','mu','logp','d']):raise ValueError('REFERENCE_NONFINITE')
            if abs((p['margin']-p['kappa'])-p['mu'])>1e-12:raise ValueError('REFERENCE_MU')
            if p['preserved']!=(p['choice']==p['target']):raise ValueError('REFERENCE_CHOICE_ID')
            flips+=not p['preserved']
        all_choices=all(p['preserved'] for p in ps)
        if doc['all_choices']!=all_choices:raise ValueError('REFERENCE_SEQUENCE_AGGREGATE')
        retained+=all_choices
    if (scan['positions'],scan['token_flips'],scan['sequence_retained'],scan['all_choices'])!=(positions,flips,retained,flips==0):
        raise ValueError('REFERENCE_AGGREGATE')
    return dict(documents=documents,positions=positions,token_flips=flips,retained_sequences=retained,
                stored_arithmetic='PASS',new_full_vocab_forward=False)

def mechanism_tables(output,report,terminal):
    import torch
    selection=terminal['selection'];roundrows=[];factorrows=[];audit=[];previous=set();gradient_count=0
    for item in selection['rounds']:
        ix=item['round'];directory=output/f'controller/round{ix}'
        rows=read(directory/'factors/rows.json');problem=torch.load(directory/'qp-problem.pt',weights_only=True,mmap=True,map_location='cpu')
        ids=[r['pair']['pair_id'] for r in rows]
        if len(set(ids))!=len(ids) or ids!=problem['ids'] or not previous.issubset(ids):raise ValueError('EXPOSED_PAIR_RETENTION')
        if ix==1 and (len(ids)!=512 or len({r['pair']['source_row_id'] for r in rows})!=512):raise ValueError('FULL_BANK_512')
        if len(ids)>(512 if ix==1 else 1024):raise ValueError('PAIR_CAP')
        b=problem['b'].numpy();G=problem['G'].numpy()
        if G.shape!=(len(ids),len(ids)) or not np.isfinite(G).all() or not np.isfinite(b).all():raise ValueError('QP_STORED_SCHEMA')
        for j,r in enumerate(rows):
            if r['b']!=b[j] or abs(r['b']-(-r['mu']+r['raw_center_inner']))>1e-12:raise ValueError('RAW_CENTER_RHS')
            factorrows.append(dict(round=ix,pair_id=ids[j],b=r['b'],mu=r['mu'],raw_center_inner=r['raw_center_inner'],
                two_head_margin=r['two_head_margin'],full_vocab_margin=r['full_vocab_margin'],seconds=r['seconds']))
        gradient_count+=len(ids);previous=set(ids)
        solpath=directory/'qp-solution.json';sol=read(solpath) if solpath.exists() else None
        if sol is not None:
            a=np.asarray(sol['alpha']);slack=G@a-b
            if not np.isfinite(a).all():raise ValueError('DUAL_NONFINITE')
            audit.append(dict(round=ix,minimum_dual=float(a.min()),minimum_raw_slack=float(slack.min()),
                raw_complementarity_max=float(np.max(np.abs(a*slack))),primal_objective=float(.5*a@G@a),
                dual_objective=float(a@b-.5*a@G@a),source_KKT_pass=sol['diagnostics']['KKT_pass'],
                validation='INDEPENDENT_RAW_FP64_ARITHMETIC; runtime row-scaled certificate separately retained'))
        order=read(directory/'ordering-audit.json')
        roundrows.append(dict(round=ix,rows=len(ids),pair_gradients_cumulative=gradient_count,
            factor_seconds=item['factor_seconds'],gram_seconds=item['gram_seconds'],qp_seconds=item['qp_seconds'],
            reconstruction_seconds=item.get('reconstruction_seconds'),scan_seconds=item['scan_seconds'],
            candidate_scanned=item['candidate_scanned'],all_choices=item.get('all_choices'),token_flips=item.get('token_flips'),
            ideal_norm=item.get('ideal_norm'),actual_norm=item.get('actual_norm'),
            result=item.get('status','CANDIDATE_SCANNED'),CPU_same_problem_order_audit=order['status']))
    if gradient_count!=selection['pair_scalar_gradients'] or gradient_count>1536:raise ValueError('ACTUAL_GRADIENT_COUNT')
    if len(roundrows)>2 or selection['candidate_reference_scans']>2 or selection['current_guard_count']>1:raise ValueError('CONTROLLER_BUDGET')
    csvout(report/'round-ledger.csv',roundrows);csvout(report/'factor-row-ledger.csv',factorrows);csvout(report/'QP-independent-arithmetic.csv',audit)
    checks=[]
    for p in sorted((output/'technical').glob('pair-*-AD.json')):
        v=read(p);checks.append(dict(check=p.stem,status=v['check_status'],value=v['relative_gradient_error'],ceiling=1e-4,scope='one actual scalar pair'))
    v=read(output/'technical/actual-factor-Gram.json')
    checks.append(dict(check='factor-Gram',status=v['check_status'],value=v['relative_error'],ceiling=1e-10,scope='actual reference4; all16 Gram entries'))
    fd=[]
    for p in sorted((output/'technical').glob('FD-pair*/result.json')):
        v=read(p)
        for x in v.get('points',[]):fd.append(dict(direction=p.parent.name,scale=x['k'],h=x['h'],AD=x['AD'],FD=x['FD'],
            relative=x['relative'],signal=x['signal'],noise=x['noise'],match=x['match'],
            actual_linear_FD=x['actual_linear_FD'],selected_adjacent=x['k'] in (v['selected_adjacent'] or [])))
    csvout(report/'technical-actual.csv',checks);csvout(report/'FD-fixed-grid.csv',fd)
    create_json(report/'mechanism-audit.json',dict(status='PASS_STORED_ARITHMETIC',rounds=len(roundrows),
        pair_scalar_gradients=gradient_count,full_first_bank=not roundrows or roundrows[0]['rows']==512,
        prior_pairs_retained=True,actual_center_raw_gradient_RHS=True,no_new_model_execution=True,
        full_factor_D_reconstruction='NOT_RUN; small stored Gram/b/alpha independently reduced'))

def analyze(output,accounting,report):
    report.mkdir(parents=True,exist_ok=False)
    terminal=read(output/'terminal.json');lock=read(output.parent/'execution.lock.json')
    if terminal['batch']!=1 or terminal['requests']!=100 or terminal['sequential_authorized']:raise ValueError('B1_SCOPE')
    objs={arm:read(output/('W0-current.json' if arm=='W0' else f'arms/{arm}/current.json')) for arm in ['W0','N4','BPCW512']}
    panels={a:reduce_raw(v,lock['sample_order']) for a,v in objs.items()}
    final=[];tails=[];paired=[];retention=[]
    for arm,metrics in panels.items():
        for tag,rows in metrics.items():
            count=sum(r['success'] for r in rows);base=sum(r['success'] for r in panels['N4'][tag])
            final.append(dict(arm=arm,metric=tag,numerator=count,denominator=len(rows),percent=100*count/len(rows),
                delta_N4_pp=100*(count-base)/len(rows),ties=sum(r['tie'] for r in rows),
                new_TF_strict=sum(r['new_strict'] for r in rows),true_TF_strict=sum(r['true_strict'] for r in rows)))
            for field in ['new_nll','true_nll','desired_margin']:
                values=[r[field] for r in rows]
                tails.append(dict(arm=arm,metric=tag,field=field,mean=statistics.mean(values),minimum=min(values),
                    p01=float(np.quantile(values,.01)),median=statistics.median(values),p95=float(np.quantile(values,.95)),
                    p99=float(np.quantile(values,.99)),maximum=max(values)))
        w0=panels['W0']['NS'];nowrows=metrics['NS'];pairedN=pair(w0,nowrows)
        denominator=sum(r['success'] for r in w0);lost=sum(r['lost'] for r in pairedN)
        retention.append(dict(arm=arm,W0_correct_N=denominator,retained=denominator-lost,lost=lost,
            total_requested_N=1000,retention_percent=100*(denominator-lost)/denominator if denominator else None))
    for tag in ['RS','PS','NS']:
        transitions=pair(panels['N4'][tag],panels['BPCW512'][tag])
        csvout(report/f'paired-{tag}.csv',transitions)
        paired.append(dict(metric=tag,denominator=len(transitions),lost=sum(r['lost'] for r in transitions),
            gained=sum(r['gained'] for r in transitions),net_gain=sum(r['gained']-r['lost'] for r in transitions),
            mean_new_NLL_delta=statistics.mean(r['new_nll_delta'] for r in transitions),
            maximum_new_NLL_increase=max(r['new_nll_delta'] for r in transitions),
            p95_new_NLL_increase=float(np.quantile([r['new_nll_delta'] for r in transitions],.95))))
    strict=[]
    for arm,panel in panels.items():
        r={r['case_id']:r for r in panel['RS']};p={c:[] for c in lock['sample_order']}
        for row in panel['PS']:p[row['case_id']].append(row)
        values=dict(rewrite_strict=sum(v['new_strict'] for v in r.values()),
            two_P_strict=sum(all(x['new_strict'] for x in v) for v in p.values()),
            R_two_P_strict=sum(r[c]['new_strict'] and all(x['new_strict'] for x in p[c]) for c in r),
            R_two_P_NLL_joint=sum(r[c]['success'] and all(x['success'] for x in p[c]) for c in r))
        if any(objs[arm]['strict'][k]!=v for k,v in values.items()):raise ValueError('STRICT_REDUCER')
        strict.append(dict(arm=arm,denominator=100,**values))
    generation=[]
    for arm,obj in objs.items():
        if arm=='W0':continue
        gs=obj['generation']
        if [g['case_id'] for g in gs]!=lock['sample_order']:raise ValueError('GENERATION_CASE_ORDER')
        for g in gs:
            if g['target_prefix_match']!=(len(g['generated_token_ids'])>=len(g['target_token_ids']) and
                g['generated_token_ids'][:len(g['target_token_ids'])]==g['target_token_ids']):raise ValueError('GREEDY_TARGET_PREFIX')
        generation.append(dict(arm=arm,denominator=len(gs),target_prefix_match=sum(g['target_prefix_match'] for g in gs),
            stopped_on_original_EOS=sum(g['stopped_on_original_eos'] for g in gs),
            reached_max32=sum(g['reached_max_new_tokens'] for g in gs),target_over32_censored=sum(g['target_over_32_censored'] for g in gs)))
        if not all(obj[k] for k in ['selection_seal_verified_before_P_N_access','entry_selected_weight_restored_exact',
                                   'RNG_unchanged_and_restored','physical_endpoint_bytes_installed']):raise ValueError('OBSERVER_RECEIPT')
    csvout(report/'greedy32.csv',generation)
    create_json(report/'runtime-work-counters.json',{k:terminal[k] for k in ['reference_work','pair_work','dev_work','current_work','observer_work']})
    selection=terminal['selection'];nativeRef=read(output/'controller/native-reference.json')
    if selection['status']=='ACCEPTED_REFERENCE_AND_CURRENT':
        selectedRef=read(output/f"controller/round{len(selection['rounds'])}/reference.json")
    else:selectedRef=nativeRef
    eos_ids=read(output/'capsules/R512/000.json')['original_eos_ids']
    references=[refsummary(nativeRef,'N4','R512',eos_ids),refsummary(selectedRef,'BPCW512','R512',eos_ids)]
    references += [refsummary(read(output/f'arms/{a}/Dev128-choice.json'),a,'Dev128',eos_ids) for a in ['N4','BPCW512']]
    scanchecks={name:audit_scan(scan,n) for name,scan,n in [('N4-R512',nativeRef,512),('BPCW-R512',selectedRef,512)]+[
        (a+'-Dev128',read(output/f'arms/{a}/Dev128-choice.json'),128) for a in ['N4','BPCW512']]}
    create_json(report/'reference-stored-arithmetic.json',scanchecks)
    mechanism_tables(output,report,terminal)
    selectedSHA=objs['BPCW512']['selection_seal']['endpoint_weight_sha256']
    if selectedSHA!=selection['selected_sha256'] or selectedRef['weight_sha256']!=selectedSHA:raise ValueError('SELECTED_EVAL_BINDING')
    commits=[]
    import torch
    from project.run_scripts.single_layer_edit_preserving_correction.common import tensor_sha
    for a in ['N4','BPCW512']:
        receipt=read(output/f'arms/{a}/commit.json');cp=output/f'arms/{a}/checkpoint.pt'
        value=torch.load(cp,weights_only=True,mmap=True,map_location='cpu')
        for key,shape,field in [('weight',(4096,14336),'W'),('M4',(1,14336,14336),'M')]:
            t=value[key]
            if tuple(t.shape)!=shape or t.dtype!=torch.float32 or not torch.isfinite(t).all():raise ValueError('CP_SCHEMA')
            if tensor_sha(t)!=receipt['identity'][field]:raise ValueError('CP_TENSOR_SHA')
        if value['batch']!=1 or value['next_batch']!=2 or value['sequential_authorized']:raise ValueError('CP_SCOPE')
        commits.append(dict(arm=a,checkpoint_bytes=cp.stat().st_size,checkpoint_SHA256=sha(cp),
            W_SHA256=receipt['identity']['W'],M_SHA256=receipt['identity']['M'],history_append=receipt['history_appends'],
            CPU_tensor_validation='PASS',runtime_physical_reload=receipt['physical_weight_reload'],
            GPU_B2_continuation='NOT_RUN_NOT_AUTHORIZED'))
    n4cp=torch.load(output/'arms/N4/checkpoint.pt',weights_only=True,mmap=True,map_location='cpu')
    bpcp=torch.load(output/'arms/BPCW512/checkpoint.pt',weights_only=True,mmap=True,map_location='cpu')
    native_cp=torch.load(output/'native/native-capsule.pt',weights_only=True,mmap=True,map_location='cpu')
    selected_ideal=torch.load(output/'selected-ideal.pt',weights_only=True,mmap=True,map_location='cpu')
    if tensor_sha(n4cp['weight'])!=tensor_sha(native_cp['weight']):raise ValueError('N4_NOT_NATIVE_ENDPOINT')
    native_copy=selection['selected_sha256']==selection['native_sha256']
    expected=native_cp['weight'] if native_copy else (native_cp['weight'].double()+selected_ideal['ideal']).float()
    if tensor_sha(expected)!=tensor_sha(bpcp['weight']):raise ValueError('IMMUTABLE_NATIVE_FP32_RECONSTRUCTION')
    create_json(report/'CPU-endpoint-reconstruction.json',dict(status='PASS_CPU_EXACT_STORED_WN_PLUS_IDEAL',
        native_matches_N4=True,BPCW_immutable_native_FP32_equal=True,native_exact_copy_route=native_copy,
        M4_equal_between_arms=torch.equal(n4cp['M4'],bpcp['M4']),
        all_parameters_model_replay=False,GPU_B2_continuation=False,distinct_factors_reconstructed_independently=False))
    geom=read(output/'edit-null-space.json')
    create_json(report/'geometry-summary.json',{k:v for k,v in geom.items() if k!='reduced_rank'} |
        dict(reduced_rank={k:v for k,v in geom['reduced_rank'].items() if k!='singular_values'},
             full_spectrum_local=member(output/'edit-null-space.json'),
             key_identity_binding=member(output/'protected-provenance.json'),
             geometry_helper_kind_not_optimizer='EN-F null-space constructor reused; optimizer is BPCW QP'))
    timers=terminal['timings'];native=timers['native_shared_once']
    n4edit=native+terminal['commits']['N4']['history_seconds']
    bpedit=native+timers['current_keys_geometry']+timers['current_anchor']+timers['controller']+terminal['commits']['BPCW512']['history_seconds']
    ratio=bpedit/n4edit
    compute=[dict(component=k,seconds=v,aggregation='EXCLUSIVE_TOP_LEVEL_PROGRAM_PHASE') for k,v in timers.items()]
    compute += [dict(component='N4_standalone_editing',seconds=n4edit,aggregation='COMPARISON_VIEW_NATIVE_INCLUDED'),
        dict(component='BPCW512_standalone_editing',seconds=bpedit,aggregation='COMPARISON_VIEW_NATIVE_INCLUDED'),
        dict(component='program_wall',seconds=terminal['wall_seconds'],aggregation='TOTAL_NOT_ADD_TO_PHASES')]
    rows={r['metric']:r for r in paired};rs={r['arm']:r for r in retention}
    dv={r['arm']:r for r in references if r['split']=='Dev128'}
    currentok=(selection['selected_sha256']==selection['native_sha256'] or
        read(output/f"controller/round{len(selection['rounds'])}/current-invariant.json")['pass'])
    gates=[dict(gate=1,name='technical/source/state',passed=terminal['technical']['status']=='INTEGRATED_MODEL_CHECKS_PASS' and terminal['nonselected_bytes_equal'],
        evidence=terminal['technical']['status']),
        dict(gate=2,name='all reference choices/current/Past',passed=selectedRef['all_choices'] and currentok,evidence=f"token_flips={selectedRef['token_flips']};Past=N/A0"),
        dict(gate=3,name='RS/PS additional lost IDs zero',passed=rows['RS']['lost']==rows['PS']['lost']==0,evidence=f"RS={rows['RS']['lost']};PS={rows['PS']['lost']}"),
        dict(gate=4,name='NS net/W0-correct N no additional loss',passed=rows['NS']['net_gain']>=0 and rs['BPCW512']['lost']<=rs['N4']['lost'],evidence=f"NSnet={rows['NS']['net_gain']};W0loss={rs['BPCW512']['lost']-rs['N4']['lost']}"),
        dict(gate=5,name='Dev choice flips no increase',passed=dv['BPCW512']['token_flips']<=dv['N4']['token_flips'],evidence=f"flip_delta={dv['BPCW512']['token_flips']-dv['N4']['token_flips']}"),
        dict(gate=6,name='standalone editing <=2x matched N4',passed=ratio<=2,evidence=f'ratio={ratio:.9g}')]
    status='B1_ALL_SIX_PASS' if all(g['passed'] for g in gates) else ('B1_BUDGET_FAIL' if all(g['passed'] for g in gates[:5]) else 'B1_GATE_FAIL')
    for name,data in [('first-final-table',final),('paired-summary',paired),('NLL-distributions',tails),('strict-joint',strict),
        ('W0-N-retention',retention),('reference-Dev',references),('checkpoint-inventory',commits),('compute',compute),('six-gates',gates)]:csvout(report/f'{name}.csv',data)
    account=read(accounting)
    summary=dict(status=status,selection=selection,final=final,paired=paired,strict=strict,generation=generation,retention=retention,references=references,
        gates=gates,editing_ratio=ratio,compute=compute,accounting=account,source=lock['execution']['commit'],
        source_tree=lock['execution']['tree'],lock=member(output.parent/'execution.lock.json'),technical=terminal['technical'],
        native_actual_executions=terminal['native_actual_executions'],native_counts=terminal['native_counts'],
        peak_gpu_allocated=terminal['peak_gpu_allocated'],peak_host_KiB=terminal['peak_host_KiB'],
        sequential_authorized=False,next='WAITING_USER_APPROVAL_FOR_SEQUENTIAL',new_reviewer_GPU=0)
    create_json(report/'summary.json',summary)
    # Full actual output inventory; large model/corpus unchanged assets are prior bindings, not rehashed here.
    raw=[member(p) for p in sorted(output.rglob('*')) if p.is_file()]
    create_json(report/'raw-inventory.json',dict(members=raw,total_bytes=sum(m['bytes'] for m in raw),
        provenance='new B1 output FULL_SHA; no old EN/raw re-audit',source_lock=summary['lock']))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--accounting',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args();print(json.dumps(analyze(a.output,a.accounting,a.report),ensure_ascii=False,indent=2))
