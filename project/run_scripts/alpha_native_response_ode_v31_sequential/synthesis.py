"""Bounded factual synthesis of immutable completed batches; no model replay.

All proportions retain their actual denominator. Paths/energies are distinct;
cross-batch net weight displacement is not inferred by summing batch norms.
"""
import math
from collections import defaultdict
from .reporting import read,metrics,bits,csvwrite


def ratio(a,b):return a/b if b else None


def physical_rows(writer,meta):
    nodes=writer.get('nodes',[]);sums=defaultdict(lambda:defaultdict(float))
    for n in nodes:
        h=n['h'];active=n['active_layers']
        for i,x in enumerate(n['layer_actions']):
            if active[i]!=x['layer']:raise RuntimeError('NODE_LAYER_MAPPING')
            out=sums[x['layer']]
            for key in ('raw_native_velocity_action','normalized_native_velocity_action',
                        'history_velocity_action','L2_velocity_action','frobenius_velocity_squared'):
                out['integral_'+key]+=h*x[key]
            out['integral_frobenius_velocity_norm']+=h*math.sqrt(x['frobenius_velocity_squared'])
            out['signed_predicted_target_progress']+=h*n['predicted_target_contribution'][i]
        for x in n['actual_physical']:
            sums[x['layer']]['actual_step_norm_sum']+=math.sqrt(x['actual_step_DeltaW_squared'])
            sums[x['layer']]['actual_step_squared_sum']+=x['actual_step_DeltaW_squared']
    endpoint=writer['actual_physical_action'];rows=[]
    for x in endpoint['layers']:
        row=dict(meta,layer=x['layer'],endpoint_native_net_raw=x['native_raw'],
            endpoint_DeltaW_squared=x['frobenius_sq'],endpoint_DeltaW_norm=math.sqrt(x['frobenius_sq']),
            endpoint_energy_share=ratio(x['frobenius_sq'],endpoint['frobenius_net_sq']),
            velocity_trajectory_status='RECORDED' if nodes else 'NOT_RECORDED_OFFICIAL_ONE_PASS',
            cross_batch_net_displacement='NOT_INFERRED_FROM_BATCH_NORMS',**sums[x['layer']])
        for key in ('raw_native_velocity_action','normalized_native_velocity_action'):
            field='integral_'+key;total=sum(s[field] for s in sums.values())
            row[field+'_share']=ratio(sums[x['layer']][field],total) if nodes else None
        rows.append(row)
    return rows


def summarize(root,out,complete_chains,completed_batch_paths=None,load=read):
    # Freeze the input scope at collector entry; an asynchronously completing batch
    # must not silently enter only half of a milestone package.
    allowed=set(completed_batch_paths) if completed_batch_paths is not None else None
    sample=load(root/'sample.lock.json');records={r['case_id']:r for r in sample['records']}
    panels={};cohorts=[];layers=[];nodes=[];pool=[];checkpoint=[];commits=[];sources=set()
    for chain in sorted(root.glob('chain-*')):
        if allowed is not None and not any(p.parent==chain for p in allowed):continue
        runtime=load(chain/'runtime.lock.json');alias=runtime['alias'];arm=runtime['arm'];current=[]
        own={};last={};previous=None;seen_count=0
        for bd in sorted(chain.glob('batch-*')):
            if not (bd/'complete.json').exists() or (allowed is not None and bd not in allowed):continue
            s=load(bd/'complete.json');w=load(bd/'writer.json');e=load(bd/'entry.json');k=s['batch_index']
            c=s['commit'];meta=dict(alias=alias,arm=arm,batch=k,W_sha256=c['committed_weight_sha256'])
            if previous is not None and (e['W_entry']!=previous['committed_weight_sha256'] or
                e['M_entry']!=previous['committed_M_content_sha256']):raise RuntimeError('REPORT_CHAIN_CONTINUITY')
            if c['history_append_count']!=1 or s['semantic_success_filtering_count']!=0:raise RuntimeError('REPORT_HISTORY_POLICY')
            seen_count+=s['requested'];previous=c;current.append(w['endpoint']['evaluation']);sources.add(e['source_head'])
            commits.append(dict(meta,history_append_count=1,cold_initialization_count=e['cold_initialization_count'],
                entry_W=e['W_entry'],entry_M=e['M_entry'],endpoint_M=c['committed_M_content_sha256'],
                target_sha256=w['fixed_z_sha256'],fixed_z_request_count=load(bd/'target-reference.json')['fixed_z_request_count'],
                actual_commit_forward_count=c['model_forward_count'],actual_commit_evaluator_count=c['evaluator_count']))
            if s['checkpoint']:
                cp=s['checkpoint'];checkpoint.append(dict(meta,**cp))
                if cp['reload_tensor_identity']!='PASS' or not cp['actual_selected_weights_saved'] or not cp['actual_history_saved']:
                    raise RuntimeError('REPORT_REAL_CHECKPOINT')
            layers+=physical_rows(w,meta)
            for n in w.get('nodes',[]):
                node=dict(meta,node=n['node'],V_before=n['V_before'],V_after=n['V_after'],V_ratio=n['V_ratio'],
                    qN_ref=n['qN_ref'],model_error_normalized=n['model_error_normalized'],
                    model_error_raw_activation=n['model_error_raw_activation'],
                    finite_step_dissipation_defect=n['finite_step_dissipation_defect'],
                    materialization_discrepancy_normalized=w['materialization_discrepancy_normalized'])
                sh=n['shadows'];single=next((x for x in sh.get('single_layer',[]) if x['layer']==8),None)
                if single:
                    node.update(joint_objective_improvement=sh['joint_objective_improvement'],
                        l8_objective_improvement=single['objective_improvement'],best_single_layer=sh['best_single_layer'],
                        joint_minus_l8_improvement=sh['joint_objective_improvement']-single['objective_improvement'],
                        l8_shadow_native_cosine=single['native_cosine'],l8_shadow_status=single['status'])
                nodes.append(node)
            retention=load(bd/'rewrite-retention.json')
            if len(retention)!=seen_count:raise RuntimeError('REPORT_RETENTION_DENOMINATOR')
            groups=defaultdict(list)
            for r in retention:
                case=r['case_id'];cohort=records[case]['batch_index']
                if cohort==k:own[case]=r['success']
                overwritten=any(x['subject_relation_group']==records[case]['subject_relation_group'] and
                    cohort<x['batch_index']<=k and x['target_new_sha256']!=records[case]['target_new_sha256'] for x in sample['records'])
                groups[cohort].append(dict(case=case,now=r['success'],margin=r['margin'],own=own[case],
                    previous=last.get(case),overwrite_candidate=overwritten))
                last[case]=r['success']
            for cohort,rr in groups.items():
                first=sum(x['own'] for x in rr);conditional=[x for x in rr if x['own'] and not x['overwrite_candidate']]
                cohorts.append(dict(meta,cohort=cohort,age=k-cohort,canonical_denominator=len(rr),
                    current_success=sum(x['now'] for x in rr),at_write_success=first,
                    initially_failed=len(rr)-first,at_write_success_now_failure=sum(x['own'] and not x['now'] for x in rr),
                    forgetting_conditional_denominator=first,
                    prior_failure_now_recovery=sum(x['previous'] is False and x['now'] for x in rr),
                    prior_failure_denominator=sum(x['previous'] is False for x in rr),
                    overwrite_candidate_count=sum(x['overwrite_candidate'] for x in rr),
                    nonoverwrite_forgetting_num=sum(not x['now'] for x in conditional),
                    nonoverwrite_forgetting_den=len(conditional),margin_mean=sum(x['margin'] for x in rr)/len(rr)))
            if (bd/'seen-full.json').exists():panels[(alias,arm,k)]=load(bd/'seen-full.json')
        pooled=dict(alias=alias,arm=arm,scope='online_own_batch_NOT_final_W10',completed_batches=len(current))
        for key,path in [('RS',('preference','rewrite')),('PS',('preference','rephrase')),('NS',('locality','canonical_ns'))]:
            values=[p[path[0]][path[1]] for p in current]
            a=sum(x['prompt_success_count'] for x in values);b=sum(x['prompt_denominator'] for x in values)
            pooled.update({key+'_num':a,key+'_den':b,key+'_rate':ratio(a,b)})
        pool.append(pooled)
    paired=[]
    for alias in {k[0] for k in panels}:
        for left,right in [('O_NATIVE','JV_NATIVE'),('O_NATIVE','L8_ONLY_NATIVE'),('L8_ONLY_NATIVE','JV_NATIVE')]:
            for k in (1,5,10):
                if (alias,left,k) not in panels or (alias,right,k) not in panels:continue
                a,b=panels[(alias,left,k)],panels[(alias,right,k)]
                for kind in ('rewrite','rephrase','locality'):
                    aa,bb=bits(a,kind),bits(b,kind)
                    if aa.keys()!=bb.keys() or any(aa[i]['input_identity']!=bb[i]['input_identity'] for i in aa):
                        raise RuntimeError('PAIRED_ENDPOINT_REQUEST_IDENTITY')
                    pa,pb=sum(x['success'] for x in aa.values()),sum(x['success'] for x in bb.values())
                    paired.append(dict(alias=alias,left=left,right=right,batch=k,kind=kind,denominator=len(aa),
                        left_num=pa,right_num=pb,right_minus_left_count=pb-pa,
                        right_minus_left_rate=(pb-pa)/len(aa),left_success_right_failure=sum(aa[i]['success'] and not bb[i]['success'] for i in aa),
                        left_failure_right_success=sum(not aa[i]['success'] and bb[i]['success'] for i in aa),
                        margin_right_minus_left_mean=sum(bb[i]['margin']-aa[i]['margin'] for i in aa)/len(aa),
                        shared_sample=True,shared_W_M_after_first_batch=False,causal_claim='END_TO_END_CHAIN_COMPARISON'))
    tables=dict(physical_layer_batch_metrics=layers,retention_cohort_metrics=cohorts,
        paired_seen_endpoint_metrics=paired,online_own_batch_metrics=pool,
        node_mechanism_summary=nodes,checkpoint_inventory=checkpoint,state_continuity_summary=commits)
    for name,rows in tables.items():csvwrite(out/f'{name}.csv',rows)
    text=['\n## 수치 기반 구분: 물리적 write·온라인/최종 retention',
        '\n아래 layer 값은 실제 FP32 batch endpoint ΔW이며 velocity-action 적분과 다르다. 여러 batch의 ΔW norm 합을 W10−W0 net norm으로 부르지 않는다.',
        '\n| Model | Arm | Batch | endpoint energy | L8 energy share | L4–7 energy |',
        '|---|---|---:|---:|---:|---:|']
    grouped=defaultdict(list)
    for x in layers:grouped[(x['alias'],x['arm'],x['batch'])].append(x)
    for (alias,arm,k),rr in sorted(grouped.items()):
        if k not in (1,5,10):continue
        total=sum(x['endpoint_DeltaW_squared'] for x in rr);last8=sum(x['endpoint_DeltaW_squared'] for x in rr if x['layer']==8)
        text.append(f'| {alias} | {arm} | {k} | {total:.6g} | {last8/total:.6%} | {total-last8:.6g} |' if total else
                    f'| {alias} | {arm} | {k} | 0 | NA_ZERO | 0 |')
    text+=['\n온라인 own-batch 합계는 online_own_batch_metrics.csv, 동일 final W10의 전체 1,000 문항은 final_metrics.csv에 분리했다. Rewrite forgetting의 모든 canonical 분모와 at-write-success 조건부 분모를 retention_cohort_metrics.csv에서 함께 제공한다. Overwrite 후보는 삭제하지 않고 비-overwrite 조건부 수치도 별도로 제공한다.',
        '\npaired_seen_endpoint_metrics.csv는 같은 prompt/target identity의 양 arm 비교이며, 첫 batch 이후 W/M/z 자체가 같은 실험이라는 뜻이 아니다. Native metric 변화와 physical ΔW 변화를 함께 보며, 분산 자체를 성공조건으로 사용하지 않는다.',
        f'\n실제 완료된 chains={sorted(complete_chains)}. Source identity={sorted(sources)}. 추가 model/evaluator replay=0.']
    return text
