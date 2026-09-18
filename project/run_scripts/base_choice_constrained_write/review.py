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

def refsummary(scan,arm,split):
    positions=[p for d in scan['documents'] for p in d['positions']]
    return dict(arm=arm,split=split,documents=len(scan['documents']),positions=len(positions),
        retained_sequences=sum(d['all_choices'] for d in scan['documents']),
        token_flips=sum(not p['preserved'] for p in positions),EOS_sequences=sum(d['eos'] for d in scan['documents']),
        censored_sequences=sum(d['censored'] for d in scan['documents']),
        minimum_margin=min(p['margin'] for p in positions),mean_d=statistics.mean(p['d'] for p in positions),
        max_d=max(p['d'] for p in positions),outside_base_top8=sum(p['outside_base_top8'] for p in positions))

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
    selection=terminal['selection'];nativeRef=read(output/'controller/native-reference.json')
    if selection['status']=='ACCEPTED_REFERENCE_AND_CURRENT':
        selectedRef=read(output/f"controller/round{len(selection['rounds'])}/reference.json")
    else:selectedRef=nativeRef
    references=[refsummary(nativeRef,'N4','R512'),refsummary(selectedRef,'BPCW512','R512')]
    references += [refsummary(read(output/f'arms/{a}/Dev128-choice.json'),a,'Dev128') for a in ['N4','BPCW512']]
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
    summary=dict(status=status,selection=selection,final=final,paired=paired,strict=strict,retention=retention,references=references,
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
