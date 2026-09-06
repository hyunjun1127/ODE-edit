"""Analysis-only joins and rehash; no model/torch/Slurm/evaluator imports."""
import argparse,csv,hashlib,json,math,os,platform,statistics,subprocess,tempfile
from pathlib import Path
from .inputs import Publication,COMMIT,PREFIX,EXECUTION,sha,canonical
from .plots import plot_all,MODELS

NA='NOT_RECORDED'
def num(x):return float(x) if x not in ('',None,NA) else None
def ratio(a,b):return a/b if b else None
def close(a,b):return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
def key(r,node=False,layer=False):
    k=(r['alias'],r['arm'],int(r['batch']))
    if node:k+=(int(r['node']),)
    if layer:k+=(int(r['layer']),)
    return k
def index(rows,**kw):
    result={key(r,**kw):r for r in rows}
    assert len(result)==len(rows),'COMPOSITE_KEY_DUPLICATE'
    return result
def concentration(energies):
    assert all(e>=0 and math.isfinite(e) for e in energies)
    total=sum(energies)
    if not total:return dict(L8_energy_share=None,max_layer=None,entropy=None,effective_layers=None)
    p=[e/total for e in energies];ent=-sum(x*math.log(x) for x in p if x)
    return dict(L8_energy_share=p[-1],max_layer=4+max(range(5),key=lambda i:energies[i]),
        entropy=ent,effective_layers=math.exp(ent))
def write_json(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n')
    path.chmod(0o600)
def write_csv(path,rows):
    fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ['status']
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else NA if v is None else v for k,v in r.items()})
    path.chmod(0o600)
def stats(values):
    import numpy as np
    v=[x for x in values if x is not None]
    return dict(n=len(v),mean=statistics.mean(v) if v else None,median=statistics.median(v) if v else None,
        p90=float(np.quantile(v,.9)) if v else None,max=max(v) if v else None,min=min(v) if v else None)

def derive(tables):
    physical=tables['physical_layer_batch_metrics'];decomp=tables['layer_action_decomposition']
    assert len(physical)==200 and len(decomp)==1000
    current=index(tables['current_batch_metrics']);norm=index(tables['normalization_batch_summary'])
    mechanism=index(tables['node_mechanism_summary'],node=True)
    matrices=index(tables['layer_allocation_nodes'],node=True)
    velocities=index([r for r in decomp if r['kind']=='velocity_action'],node=True,layer=True)
    actual=index([r for r in decomp if r['kind']=='actual_FP32_DeltaW'],node=True,layer=True)
    endpoints=index([r for r in decomp if r['kind']=='actual_endpoint_net_FP32'],layer=True)
    assert len(velocities)==len(actual)==400 and len(matrices)==len(mechanism)==80
    joined=[];layers=[];batches=[];nodes=[];checks=[]
    for k,r in actual.items():
        v=velocities[k];matrix=matrices[k[:-1]];i=k[-1]-4
        g=json.loads(matrix['g']);h=json.loads(matrix['full_H']);c=json.loads(matrix['c'])
        # All included directions have the sealed five-layer inventory here.
        assert len(c)==5 and close(float(v['coefficient']),c[i])
        step=float(r['actual_step_DeltaW_squared']);net=float(r['actual_net_DeltaW_squared'])
        assert step>=0 and net>=0
        row=dict(alias=k[0],arm=k[1],batch=k[2],node=k[3],layer=k[4],
            step_squared=step,step_norm=math.sqrt(step),batch_entry_to_node_squared=net,
            batch_entry_to_node_norm=math.sqrt(net),W0_to_node_squared=NA,
            actual_nonzero=int(r['actual_nonzero']),state_version=matrix['state_version'],
            g=g[i],H_diagonal=h[i][i],signed_g_c=g[i]*c[i],
            qN_ref=float(matrix['qN_ref']),coefficient=c[i],raw_physical_coefficient=float(v['raw_coefficient']),
            h=.5,supported=c[i]>0,
            V_before=float(matrix['V_before']),V_after=float(matrix['V_after']),V_ratio=float(matrix['V_ratio']),
            finite_step_defect=float(matrix['finite_step_dissipation_defect']),
            model_error_raw_activation=float(matrix['model_error_raw_activation']),
            model_error_normalized=float(matrix['model_error_normalized']))
        for name in ('q','direction_raw_action','direction_frobenius_squared','raw_native_velocity_action',
            'normalized_native_velocity_action','history_velocity_action','L2_velocity_action','frobenius_velocity_squared'):
            row[name]=float(v[name])
        for field in ('raw_native','normalized_native','history','L2','frobenius'):
            source_name=field+'_velocity_action' if field!='frobenius' else 'frobenius_velocity_squared'
            row[field+'_work']=.5*row[source_name]
        assert close(row['raw_native_velocity_action'],row['history_velocity_action']+row['L2_velocity_action'])
        assert close(row['raw_physical_coefficient'],c[i]/math.sqrt(row['q']))
        joined.append(row)
    joined.sort(key=lambda r:key(r,node=True,layer=True))
    for k,r in index(physical,layer=True).items():
        energy=float(r['endpoint_DeltaW_squared']);e=endpoints[k]
        assert close(energy,float(e['frobenius_sq']))
        selected=[j for j in joined if key(j,layer=True)==k]
        layer=dict(alias=k[0],arm=k[1],batch=k[2],layer=k[3],batch_net_squared=energy,
            batch_net_norm=math.sqrt(energy),materialized_nonzero=int(e['materialized_nonzero']),
            parameter_elements=int(e['parameter_elements']),W0_net_squared=NA,
            endpoint_native_raw=float(e['native_raw']))
        if selected:
            assert len(selected)==4
            terminal=next(j for j in selected if j['node']==3)
            assert close(terminal['batch_entry_to_node_squared'],energy)
            for field in ('raw_native_work','normalized_native_work','history_work','L2_work','frobenius_work','signed_g_c'):
                layer[field]=sum(j[field] for j in selected)
            layer['signed_predicted_progress']=.5*layer.pop('signed_g_c')
            layer['actual_step_squared_sum']=sum(j['step_squared'] for j in selected)
            layer['actual_step_norm_sum']=sum(j['step_norm'] for j in selected)
            assert close(layer['raw_native_work'],float(r['integral_raw_native_velocity_action']))
            assert close(layer['signed_predicted_progress'],float(r['signed_predicted_target_progress']))
        else:
            layer.update(velocity_status='NOT_RECORDED_OFFICIAL_ONE_PASS',raw_native_work=None,
                normalized_native_work=None,history_work=None,L2_work=None,signed_predicted_progress=None)
        layers.append(layer)
    layers.sort(key=lambda r:key(r,layer=True))
    for k,cur in sorted(current.items()):
        ls=[r for r in layers if key(r)==k];assert len(ls)==5
        es=[r['batch_net_squared'] for r in ls];n=norm[k]
        b=dict(alias=k[0],arm=k[1],batch=k[2],total_batch_net_energy=sum(es),
            total_batch_net_norm=math.sqrt(sum(es)),L4_7_batch_net_energy=sum(es[:4]),
            L8_batch_net_energy=es[-1],**concentration(es))
        for m in ('RS','PS','NS'):
            b[m+'_num']=int(cur[m+'_num']);b[m+'_den']=int(cur[m+'_den']);b[m+'_rate']=b[m+'_num']/b[m+'_den']
        for field in ('raw_native_work','normalized_native_work','history_work','L2_work','signed_predicted_progress'):
            b[field]=sum(r[field] for r in ls) if k[1]=='JV_NATIVE' else None
        b['L4_7_signed_progress']=sum(r['signed_predicted_progress'] for r in ls[:4]) if k[1]=='JV_NATIVE' else None
        b['L8_native_work_share']=ratio(ls[-1]['normalized_native_work'],b['normalized_native_work']) if k[1]=='JV_NATIVE' else None
        b['history_raw_work_ratio']=ratio(b['history_work'],b['raw_native_work']) if k[1]=='JV_NATIVE' else None
        for field in ('minimum_active_scale','maximum_active_scale','max_over_min_active_scale',
            'final_V_ratio','node0_response_Gram_diagonal_max','materialization_discrepancy_normalized'):
            b[field]=num(n[field])
        b['minimum_scale_case_ids']=json.loads(n['minimum_scale_case_ids']);b['W0_net_squared']=NA
        batches.append(b)
    for k,m in sorted(mechanism.items()):
        js=[r for r in joined if key(r,node=True)==k];assert len(js)==5
        n=dict(alias=k[0],arm=k[1],batch=k[2],node=k[3],
            total_step_energy=sum(r['step_squared'] for r in js),
            batch_entry_to_node_energy=sum(r['batch_entry_to_node_squared'] for r in js),
            supported_layers=[r['layer'] for r in js if r['supported']],
            joint_signed_progress=sum(r['signed_g_c'] for r in js),
            L4_7_signed_progress=sum(r['signed_g_c'] for r in js[:-1]),
            **concentration([r['step_squared'] for r in js]))
        for name in ('V_before','V_after','V_ratio','model_error_normalized','model_error_raw_activation',
            'finite_step_dissipation_defect','joint_objective_improvement','joint_minus_l8_improvement',
            'l8_objective_improvement','l8_shadow_native_cosine','materialization_discrepancy_normalized'):
            n[name]=num(m[name])
        n['relative_joint_over_l8_improvement']=ratio(n['joint_minus_l8_improvement'],n['joint_objective_improvement'])
        n['best_single_layer']=int(m['best_single_layer'])
        for j in js:n[f'L{j["layer"]}_step_norm']=j['step_norm']
        nodes.append(n)
    return joined,layers,batches,nodes

def endpoint_partitions(t):
    rows=[]
    for final in t['final_metrics']:
        if final['arm']=='PRE_EDIT_ORIGINAL_W0':continue
        selected=[r for r in t['retention_cohort_metrics'] if r['alias']==final['alias'] and r['arm']==final['arm'] and int(r['batch'])==10]
        assert len(selected)==10
        at=sum(int(r['at_write_success']) for r in selected);lost=sum(int(r['at_write_success_now_failure']) for r in selected)
        initial=sum(int(r['initially_failed']) for r in selected);f=int(final['RS_num']);recovered=f-at+lost
        assert at+initial==1000 and 0<=recovered<=initial
        assert sum(int(r['current_success']) for r in selected)==f
        rows.append(dict(alias=final['alias'],arm=final['arm'],all_denominator=1000,at_write_success=at,
            initially_failed=initial,at_write_success_to_final_failure=lost,conditional_failure_denominator=at,
            initially_failed_to_final_recovery=recovered,initial_failure_denominator=initial,final_RS=f,
            previous_checkpoint_failure_to_recovery=sum(int(r['prior_failure_now_recovery']) for r in selected),
            nonoverwrite_failure=sum(int(r['nonoverwrite_forgetting_num']) for r in selected),
            nonoverwrite_success_den=sum(int(r['nonoverwrite_forgetting_den']) for r in selected),
            overwrite_candidates=sum(int(r['overwrite_candidate_count']) for r in selected)))
    return rows

def table(rows,columns):
    def fmt(v):
        if v is None:return 'N/A'
        if isinstance(v,float):return f'{v:.8g}'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n|'+'|'.join(c[0] for c in columns)+'|\n|'+'|'.join('---' for _ in columns)+'|\n'+'\n'.join('|'+ '|'.join(fmt(r.get(k)) for _,k in columns)+'|' for r in rows)+'\n'

def report(out,t,batches,nodes,partitions,cost,history,gh_comparison,source):
    text=['# AlphaEdit JV sequential 1,000 — layer 집중·물리적 write·retention 상세 사실 보고서',
        '\n분석 기준: 2026-09-07. **완료 main 4 chains만**: Llama/Qwen × Official/JV, 각 10×B100 sequential W/M 누적. '
        'B10 warm pilot 및 미완료 L8-only와 pooling하지 않는다. 신규 GPU/모델/evaluator/Slurm 제출0; sweep HOLD; scientific_promotion=false.',
        '\n## 1. 정의·분모·검증 범위',
        '\nRS=canonical rewrite target-new NLL < target-true NLL, PS=두 rephrase prompt의 같은 strict preference, '
        'NS=neighborhood target-true NLL < target-new NLL; tie는 실패다. Final 분모는 chain별 RS1000/PS2000/NS10000; '
        'current B100는100/200/1000이다. Token denominator는 multi-token 때문에 prompt denominator와 다르다. '
        'Target-new/true NLL 모두 lower가 해당 target의 높은 likelihood를 뜻하나 preference는 양자의 차이이다. '
        'Strict PS는 request의 두 prompt 모두 preference 성공이다. Token accuracy와 자유 생성 정확도는 다른 값이다.',
        f'\nGit publication `{COMMIT}`의48/48 member 및 package receipt 재계산 PASS. Execution `{EXECUTION}`; '
        f'분석 source `{source["head"]}`. Server2 외부 raw/checkpoint는 host 장애로 **독립 재해시하지 못했다**. '
        '현 보고서는 Git의 raw-free tables/receipts를 독립 교차검산한 범위이며 source path의 존재만으로 raw 검증을 주장하지 않는다.',
        '\n## 2. Final W10 전체 1,000 request 성능 (온라인 합계가 아님)']
    final=[]
    for r in t['final_metrics']:
        f=dict(model=r['alias'],arm=r['arm'])
        for m in ('RS','PS','NS'):f[m]=f'{r[m+"_num"]}/{r[m+"_den"]} ({100*float(r[m+"_rate"]):.2f}%)'
        f['strictPS']=f'{r["PS_strict_num"]}/{r["PS_strict_den"]}'
        final.append(f)
    text.append(table(final,[(k,k) for k in ('model','arm','RS','PS','strictPS','NS')]))
    for kind in ('rewrite','rephrase'):
        text.append(f'\n### {kind}: target-new / target-true NLL 별도 분포')
        rs=[]
        for r in t['final_metrics']:
            for target in ('new','true'):
                rs.append(dict(model=r['alias'],arm=r['arm'],target=target,
                    **{q:float(r[f'{kind}_target_{target}_nll_{q}']) for q in ('mean','median','p90','max')}))
        text.append(table(rs,[(k,k) for k in ('model','arm','target','mean','median','p90','max')]))
    text+=['\nQwen의 PS +7/2000과 rephrase target-new 평균·p90 상승은 다른 측정이다. '
        '양쪽 target NLL 및 paired 성공 획득/손실은 별도 CSV로 제공하며 전체 generalization 개선으로 합치지 않는다.',
        '\n## 3. 10개 batch 전체: 집중도와 절대 크기',
        '\nActual FP32 step ΔW는 node직전→직후 차이, batch-net은 batch entry→terminal 차이다. '
        'W0→W10 net은 Git tables에 없음(NOT_RECORDED). Step norm/energy의 합은 net norm/energy가 아니다. '
        'Native action은 S_l metric의 quadratic이고 Frobenius 제곱합과 다르다. Work는 h=.5를 한번 곱한 velocity action 합이다. '
        '각 batch qref/M/N0가 달라 coefficient·normalized share만 cross-batch physical 이동으로 해석하지 않는다.',
        '\n![Layer-wise Update Magnitude](layer-update-magnitude.png)',
        '\n![concentration](concentration-absolute-action-endpoints.png)']
    for model in MODELS:
        for arm in ('JV_NATIVE','O_NATIVE'):
            text.append(f'\n### {model} / {arm}: 모든 B1–B10')
            rows=[r for r in batches if r['alias']==model and r['arm']==arm]
            text.append(table(rows,[('B','batch'),('batch-net energy','total_batch_net_energy'),
                ('batch-net norm','total_batch_net_norm'),('L8 share','L8_energy_share'),('max layer','max_layer'),
                ('L4–7 energy','L4_7_batch_net_energy'),('L4–7 hΣgc','L4_7_signed_progress'),
                ('RS/100','RS_num'),('PS/200','PS_num'),('NS/1000','NS_num'),('V/V0','final_V_ratio')]))
    text+=['\nLlama B10: L8 share와 전체 energy가 동시에 작아진다. B1–B9와 B10을 같은 “다른 layer로의 성공적 이동”으로 '
        '묶을 근거는 없다. Qwen은 높은 L8 share가 지속된다. 두 모델 모두 절대값·분모를 위 표 그대로 보존한다.',
        '\n보조 집중도: p_l=layer energy/total, entropy=-Σp log p, effective layers=exp(entropy). '
        'Total=0은 share/max-layer/entropy/effective 모두 N/A; zero p의 entropy contribution만0. 성공 gate가 아니다. '
        'Signed g_l c_l은 음수 가능하며 음수 layer를 제거하거나 norm share로 바꾸지 않았다.',
        '\n## 4. 내부 4 node progression 및 전체 layer joins',
        '\n![nodes](four-node-progression.png)',
        '\n`node-layer.csv` 400행이 각 B/node/layer의 actual step 및 batch-entry net, coefficient/raw coefficient/q/qref, '
        'g/H diagonal/gc, native/history/L2/Frobenius velocity와 h-work, materialized nonzero를 결속한다. '
        '`nodes.csv`80행과 `batch-layer.csv`200행, `batches.csv`40행은 각각 다른 단위다. '
        'Official은 Euler 값 NOT_RECORDED이며 actual batch ΔW만 비교한다.']
    for model in MODELS:
        text.append(f'\n### {model}: 모든 node의 progression')
        text.append(table([r for r in nodes if r['alias']==model],[('B','batch'),('node','node'),
            ('step energy','total_step_energy'),('batch-entry net energy','batch_entry_to_node_energy'),
            ('L8 step share','L8_energy_share'),('support','supported_layers'),('Σgc','joint_signed_progress'),
            ('V/V0','V_ratio'),('raw model error','model_error_raw_activation'),('finite defect','finite_step_dissipation_defect')]))
    text+=['\n## 5. Native work / history / L2 / normalization',
        '\nHistory/L2 분해는 raw native work와 같은 metric 안에서만 더한다. '
        'History-cost shadow는 actual W/dictionary/response/N0/qref를 고정하고 cost Gram만 초기값으로 바꾼다. '
        'History가 없는 전체 method 또는 actual L8 trajectory가 아니다.']
    text.append(table([r for r in batches if r['arm']=='JV_NATIVE'],[('model','alias'),('B','batch'),
        ('raw work','raw_native_work'),('normalized work','normalized_native_work'),('history work','history_work'),
        ('L2 work','L2_work'),('history/raw','history_raw_work_ratio'),('L8 native share','L8_native_work_share'),
        ('N0 min','minimum_active_scale'),('H diag max node0','node0_response_Gram_diagonal_max')]))
    text.append(table(history,[('model','alias'),('B','batch'),('native cosine','native_cosine'),
        ('Frob cosine','frobenius_cosine'),('shadow/actual native norm','native_norm_ratio'),
        ('actual progress','actual_progress'),('shadow progress','shadow_progress')]))
    text+=['\nLlama B10 작은 양수 N0 row, 큰 H, active set 및 실제 FP32 nonzero count를 각각 보존한다. '
        'g/H는 aggregate이고 per-request raw JVP·Gram decomposition은 이 publication에 없다. '
        'N0 단독 원인, λ 단독 원인, history 단독 원인은 UNRESOLVED. λ=.1/maxdiagH는 scale 진단일 뿐 전체 spectrum의 λ 효과를 배제하지 않는다. '
        'Source N0 floor/clip/행 제외를 추가하지 않았다. 원본 native optimizer delta0 로그와 capture residual의 차이는 '
        '`native_target_log_observations.csv`/`normalization_batch_summary.csv`에 분리했다.',
        '\n## 6. Same-state joint vs best-single/L8']
    shadow=[]
    for model in MODELS:
        r=[n for n in nodes if n['alias']==model]
        shadow.append(dict(model=model,nodes=len(r),best_L8=sum(x['best_single_layer']==8 for x in r),
            **stats([x['relative_joint_over_l8_improvement'] for x in r])))
    text.append(table(shadow,[(k,k) for k in ('model','nodes','best_L8','mean','median','p90','max')]))
    text+=['\n비율=(joint objective improvement−L8 improvement)/joint improvement. Joint improvement0이면 N/A. '
        '40/40과36/40 best-L8 같은 same-state 비교는 actual L8-only chain 완료를 대체하지 않는다. '
        'L8-only는 이 package의 completed denominator에 없으며 SH4 live output을 읽지 않았다.',
        '\n## 7. At-write failure / forgetting / recovery / overwrite']
    text.append(table(partitions,[(k,k) for k in ('alias','arm','at_write_success','initially_failed',
        'at_write_success_to_final_failure','conditional_failure_denominator','initially_failed_to_final_recovery',
        'final_RS','nonoverwrite_failure','nonoverwrite_success_den','overwrite_candidates')]))
    text+=['\nPrevious-checkpoint failure→recovery와 최초 at-write failure→final recovery는 다르다. '
        '220개 cohort/checkpoint 행을 모두 유지하고 overwrite 후보도 제거하지 않는다. '
        'Llama JV 75건 at-write 실패와2건 후속 실패를 모두 forgetting으로 부르지 않는다. '
        'B1–B9 PS 변화와 B10 신규 실패는 서로 다른 범위다. '
        'W0 locality 손실/회복 및 O→JV paired 성공 획득/손실은 각각 분리 CSV다.',
        '\n## 8. 시간·연산량 및 이전 warm pilot 비교']
    text.append(table(cost,[(k,k) for k in ('alias','arm','process_seconds','process_JV_over_O','forward','backward','main_JVP',
        'native_key_captures','solve_instrumented','target_seconds','write_including_endpoint_seconds','endpoint_seconds','peak_gpu_bytes')]))
    text+=['\nProcess ratio는 target·writer·evaluator·state bookkeeping을 포함한다. Terminal compute.wall은 절대 clock이며 '
        'elapsed로 합산하지 않는다. Solve instrumentation에는 NNLS face solves가 들어가므로 Official native solve 수와 같은 종류로 '
        '해석하지 않는다. Model.forward 횟수는 FLOPs가 아니며 FLOP 추정 없음. Memory는 peak이며 합산값 아님.',
        '\n이전 B10 warm pilot은 Official D10A1회로 만든 공통 warm entry에서 D10B/H10 독립 비교였다. '
        '이번1000은 각 arm이 자기 W/M/z를 B1→B10 누적하며 sample도 다르다. '
        '기존 pilot main/old-edit tables를 입력 identity와 함께 별도 보조 표로 제공하고 수치 pooling하지 않는다. '
        'Warm pilot old-edit loss0과 이번 sequential forgetting을 같은 denominator의 결과로 보지 않는다.',
        '\n## 9. Sweep 및 lifelong readiness: 사후·탐색적 판정',
        '\n이번 사용자 instruction의 명시적 해석 요청에 한해 아래 판단을 기재한다. '
        '결과를 본 후 만든 retrospective/exploratory 검토이며 사전 threshold PASS가 아니다.',
        '\n|Model|현재 lifelong readiness|Sweep 검토|한계|\n|---|---|---|---|\n'
        '|Llama|HOLD|λ축 bounded development는 사용자 승인됐으나 현재 제출 HOLD; N0 민감도 분해를 함께 기록할 필요|B10 신규RS25/100, near-no-action; B1부터 PS 약화. λ만 바꾸면 회복된다는 증거 없음|\n'
        '|Qwen|CANDIDATE, lifelong PASS 아님|동일 λ/control/audit matrix를 유지할 비교 후보; 필요성 확정은 UNRESOLVED|RS/PS/NS 증가와 rephrase new NLL tail 악화·L8집중·약1.51× 비용 공존|\n',
        '\nN0 변경은 별도 scientific authority 없이는 하지 않는다. T/N/h는 고정축이고 λ와 동시에 바꾸지 않는다. '
        '기존1000은 retrospective 진단이며 개발 및 독립 confirmation 양쪽에 사용할 수 없다. '
        '새 sample의 development 선택 후 unopened audit가 필요하고 신규10k 자동실행0. 현재 report-first override에 따라 GPU sweep은 HOLD다.',
        '\n## 10. 독립 교차검산·미기록·재현']
    text.append(f'\nGH verification과 독립 CSV 산술 비교: `{gh_comparison["status"]}`, 비교항목{gh_comparison["checks"]}개. '
        'GH memo 문장을 source evidence 대신 복사하지 않았다. detailed 수치는 `gh-crosscheck.json`에 있다.')
    text+=['\nPublication이 담지 않은 W0-net dense displacement, per-request H/JVP 정밀 분해, node-local solve spectrum은 '
        'NOT_RECORDED/UNRESOLVED로 남긴다. Runtime source에서 validator가 실행되었다는 경로 확인과 external raw 재해시는 다른 보증이다. '
        '정밀 원인 검증을 위해 새 모델/evaluator를 실행하지 않았다.',
        '\n재현: `python3 -m project.run_scripts.alpha_native_response_ode_v31_analysis.analyze --repo <worktree> --output <new-create-once-directory> '
        '--gh-verification /mnt/raid5/janghj/ODE-edit/local/state/alpha-jv-gh-review-20260907-v1/verification.json`. '
        'PNG는 같은 derived CSV를 다시 읽어 재실행하고 byte SHA 일치를 검사했다. Input/output/source/environment/row-count identity는 '
        '`analysis-manifest.json`, `rooted-receipt.json`, `plot-reproduction.json`, `input-inventory.json`에 결속한다.']
    # Literal output table inventory binds all accompanying files via manifest.
    text.append('\n### 상세 테이블 목록\n\n'+ '\n'.join(f'- [{p.name}]({p.name})' for p in sorted(out.glob('*.csv'))))
    with (out/'factual-report-ko.md').open('x') as f:f.write('\n'.join(text)+'\n')

def run(repo,out,gh_path):
    repo=Path(repo).absolute();out=Path(out).absolute();out.mkdir(parents=True,exist_ok=False)
    p=Publication(repo);gate=p.verify()
    names=('physical_layer_batch_metrics','layer_action_decomposition','layer_allocation_nodes','node_mechanism_summary',
        'current_batch_metrics','normalization_batch_summary','final_metrics','run_registry','compute_accounting',
        'retention_cohort_metrics','neighborhood_transition_summary','paired_seen_endpoint_metrics','history_cost_shadows',
        'sequential_commit_checks','state_continuity_summary','l8_single_layer_shadows','seen_prefix_metrics',
        'native_target_log_observations','request_normalization_scales','checkpoint_inventory','scheduler_terminal','failure_registry')
    t={n:p.rows(n) for n in names};joined,layers,batches,nodes=derive(t)
    partitions=endpoint_partitions(t);cost=[];history=[]
    for final in t['final_metrics']:
        assert [int(final[m+'_den']) for m in ('RS','PS','NS')]==[1000,2000,10000]
        if final['arm']=='PRE_EDIT_ORIGINAL_W0':continue
        transition=next(r for r in t['neighborhood_transition_summary'] if r['alias']==final['alias']
            and r['arm']==final['arm'] and int(r['batch'])==10)
        assert (int(transition['W0_success_denominator'])-int(transition['W0_success_to_failure'])
            +int(transition['W0_failure_to_recovery']))==int(final['NS_num'])
    for current_row in t['current_batch_metrics']:
        assert [int(current_row[m+'_den']) for m in ('RS','PS','NS')]==[100,200,1000]
    for r in t['run_registry']:
        alias,arm=r['alias'],r['arm'];assert r['status']=='TERMINAL_VALID' and int(r['completed_batches'])==10
        rs=[c for c in t['compute_accounting'] if c['alias']==alias and c['arm']==arm];assert len(rs)==10
        original=next(c for c in t['run_registry'] if c['alias']==alias and c['arm']=='O_NATIVE')
        ledger=json.loads(r['terminal_compute'])
        cost.append(dict(alias=alias,arm=arm,process_seconds=float(r['process_total_seconds']),
            process_JV_over_O=float(r['process_total_seconds'])/float(original['process_total_seconds']),
            forward=ledger['forward'],backward=ledger['backward'],native_key_captures=ledger['native_keys'],
            solve_instrumented=ledger['linalg_solve'],main_JVP=sum(int(x['main_jvp_count']) for x in rs),
            target_seconds=sum(json.loads(x['compute_z'])['wall'] for x in rs),
            write_including_endpoint_seconds=sum(json.loads(x['write_including_endpoint'])['wall'] for x in rs),
            endpoint_seconds=sum(float(x['endpoint_evaluation_seconds']) for x in rs),
            peak_gpu_bytes=max(int(x['peak_gpu_bytes']) for x in rs)))
    for r in t['history_cost_shadows']:
        ca=json.loads(r['c_actual']);cs=json.loads(r['c_initial']);gm=json.loads(r['G_actual'])
        quadratic=lambda c:sum(c[i]*gm[i][j]*c[j] for i in range(len(c)) for j in range(len(c)))
        history.append(dict(alias=r['alias'],batch=int(r['batch']),node=int(r['node']),
            native_cosine=num(r['native_cosine']),frobenius_cosine=json.loads(r['frobenius_angle'])['native_cosine'],
            native_norm_ratio=math.sqrt(quadratic(cs)/quadratic(ca)) if quadratic(ca)>0 else None,
            actual_progress=float(r['actual_predicted_progress']),shadow_progress=float(r['shadow_predicted_progress'])))
    # Verify all commit edges and recorded zero counters, no new runtime claims.
    state=index(t['state_continuity_summary']);commits=index(t['sequential_commit_checks'])
    for k,r in state.items():
        assert int(r['history_append_count'])==1 and int(r['fixed_z_request_count'])==100
        c=commits[k]
        for field in ('evaluator_count','fixed_z_recompute_count','model_forward_count','writer_recompute_count'):assert int(c[field])==0
        if k[-1]>1:
            previous=state[k[:-1]+(k[-1]-1,)];assert r['entry_W']==previous['W_sha256'] and r['entry_M']==previous['endpoint_M']
    term_checks=[]
    for i,(alias,arm) in enumerate((m,a) for a in ('O_NATIVE','JV_NATIVE') for m in MODELS):
        term=p.json(f'chain-{i}-{alias}-{arm}.terminal-receipt.json');runtime=p.json(f'chain-{i}-{alias}-{arm}.runtime.lock.json')
        assert term['W0_restored'] and term['completed_batches']==10 and term['requested']==1000
        assert term['technical_failure_count']==0 and term['imputation_count']==0
        assert runtime['dtype']['loaded_model_dtype']=='torch.float32' and runtime['dtype']['bf16_fp16_cast_count']==0
        assert not runtime['autocast'] and not runtime['tf32']
        term_checks.append(dict(alias=alias,arm=arm,source_head=term['source']['head'],W0=term['W0_sha256'],
            restored=True,requested=1000,history_appends=term['history_appends'],runtime_dtype=runtime['dtype'],
            contexts_sha256=runtime['contexts_sha256'],sample_root=runtime['sample_root']))
        assert term['source']['head']==EXECUTION
    gh_payload=Path(gh_path).read_bytes();gh=json.loads(gh_payload);checks=[]
    for b in batches:
        reference=gh['results'][b['alias']][b['arm']]['physical'][b['batch']-1]
        for local,remote in [('total_batch_net_energy','actual_batch_delta_energy'),('L8_energy_share','l8_share'),('L4_7_batch_net_energy','l4to7_energy')]:
            assert close(b[local],reference[remote]);checks.append(dict(alias=b['alias'],arm=b['arm'],batch=b['batch'],field=local,absolute_difference=abs(b[local]-reference[remote])))
    for r in partitions:
        ref=gh['results'][r['alias']][r['arm']]
        for local,remote in [('at_write_success','at_write_RS'),('initially_failed','initial_failures'),('at_write_success_to_final_failure','later_forgetting'),('initially_failed_to_final_recovery','recovered_initial_failures')]:
            assert r[local]==ref[remote];checks.append(dict(alias=r['alias'],arm=r['arm'],field=local,absolute_difference=0))
    comparison=dict(status='PASS_INDEPENDENT_ARITHMETIC_MATCH',checks=len(checks),items=checks,
        reference_sha256=sha(gh_payload),reference_path=str(gh_path))
    for name,rows in [('node-layer',joined),('batch-layer',layers),('batches',batches),('nodes',nodes),
        ('endpoint-partitions',partitions),('cost',cost),('history-cost-comparison',history)]:write_csv(out/(name+'.csv'),rows)
    for n in ('current_batch_metrics','final_metrics','retention_cohort_metrics','neighborhood_transition_summary',
        'paired_seen_endpoint_metrics','l8_single_layer_shadows','history_cost_shadows','normalization_batch_summary',
        'native_target_log_observations','seen_prefix_metrics','compute_accounting','state_continuity_summary',
        'layer_allocation_nodes','request_normalization_scales'):
        write_csv(out/(n+'.csv'),t[n])
    nll=[]
    for scope,rs in [('CURRENT_B100',t['current_batch_metrics']),('FINAL_W10_OR_W0',t['final_metrics'])]:
        for r in rs:
            for kind in ('rewrite','rephrase','locality'):
                for target in ('new','true'):
                    nll.append(dict(alias=r['alias'],arm=r['arm'],batch=r['batch'],scope=scope,kind=kind,target=target,
                        **{q:float(r[f'{kind}_target_{target}_nll_{q}']) for q in ('mean','median','p90','max')},
                        prompt_den=int(r[f'{kind}_target_{target}_row_count'])))
    write_csv(out/'nll-distributions.csv',nll)
    # Separate immutable pilot tables, never pooled with the sequential stream.
    pilot_prefix='experiment-reports/servers/server1/native-response-v31-b10-warm-pilot-2026-09-06-v1/primary/'
    pilot_inputs=[]
    for name in ('pilot_main_table.csv','old_edit_metrics.csv'):
        data=subprocess.check_output(['git','-C',str(repo),'show',f'{EXECUTION}:{pilot_prefix}{name}'])
        pilot_inputs.append(dict(commit=EXECUTION,path=pilot_prefix+name,sha256=sha(data),bytes=len(data)))
        rows=list(csv.DictReader(__import__('io').StringIO(data.decode())))
        write_csv(out/('previous-warm-'+name),rows)
    def reread(name):
        rs=list(csv.DictReader((out/name).open()))
        for r in rs:
            for k,v in list(r.items()):
                if v==NA:r[k]=None
                else:
                    try:r[k]=float(v)
                    except (ValueError,TypeError):pass
            for k in ('batch','node','layer'):
                if k in r and r[k] is not None:r[k]=int(r[k])
        return rs
    plot_inputs=[reread(n) for n in ('batches.csv','batch-layer.csv','nodes.csv')]
    figures=plot_all(out,*plot_inputs)
    with tempfile.TemporaryDirectory(prefix='alpha-jv-plot-reproduction-') as tmp:
        reproduced=plot_all(tmp,*plot_inputs)
        assert figures==reproduced,'PNG_BYTE_REPRODUCTION'
    import matplotlib,numpy
    write_json(out/'plot-reproduction.json',dict(status='BYTE_IDENTICAL_PASS',runs=2,figures=figures,
        python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=numpy.__version__,
        command='python -m project.run_scripts.alpha_native_response_ode_v31_analysis.analyze --repo REPO --output NEW_DIRECTORY --gh-verification GH_VERIFICATION',
        input_sha256={n:sha((out/n).read_bytes()) for n in ('batches.csv','batch-layer.csv','nodes.csv')}))
    write_json(out/'gh-crosscheck.json',comparison);write_json(out/'terminal-identity-checks.json',term_checks)
    write_json(out/'verification.json',dict(publication=gate,rows=dict(node_layer=len(joined),node=len(nodes),batch_layer=len(layers),batch=len(batches)),
        joins_missing=0,coefficient_identity_failures=0,history_L2_identity_failures=0,terminal_node_batch_net_failures=0,
        recorded_state_edges=36,final_NS_transition_checks=4,current_final_denominator_checks=46,
        imputation=0,model=0,GPU=0,Slurm=0,scientific_promotion=False))
    source=dict(head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        tree=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD^{tree}'],text=True).strip(),
        execution_head=EXECUTION,publication_commit=COMMIT,
        source_files=[dict(path=str(f.relative_to(repo)),sha256=sha(f.read_bytes()),bytes=f.stat().st_size) for f in sorted((repo/'project/run_scripts/alpha_native_response_ode_v31_analysis').glob('*.py'))])
    write_json(out/'source.lock.json',source)
    report(out,t,batches,nodes,partitions,cost,history,comparison,source)
    # Re-read all consumed publication blobs after computation.
    before=dict(p.inventory)
    for name in list(before):p.blob(name)
    assert before==p.inventory,'INPUT_CHANGED'
    write_json(out/'input-inventory.json',dict(publication_members=list(p.inventory.values()),pilot_members=pilot_inputs,
        GH_reference=dict(path=str(gh_path),sha256=sha(gh_payload)),before_after_unchanged=True))
    members=[]
    for f in sorted(out.iterdir()):
        f.chmod(0o600);b=f.read_bytes()
        members.append(dict(path=f.name,bytes=len(b),sha256=sha(b),mode='0600',
            rows=sum(1 for _ in csv.DictReader(f.open())) if f.suffix=='.csv' else None))
    manifest=dict(schema='alpha-jv-sequential-layer-analysis.v1',members=members,members_root=sha(canonical(members)),
        source=source,scientific_promotion=False,new_gpu_model_slurm_actions=0)
    write_json(out/'analysis-manifest.json',manifest)
    body=dict(manifest_sha256=sha((out/'analysis-manifest.json').read_bytes()),members_root=manifest['members_root'],
        report_sha256=sha((out/'factual-report-ko.md').read_bytes()),status='ANALYSIS_ONLY_COMPLETE_SWEEP_HOLD',
        source_head=source['head'],scientific_promotion=False)
    write_json(out/'rooted-receipt.json',dict(body,receipt_identity=sha(canonical(body))))
    return body

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gh-verification',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.repo,a.output,a.gh_verification)))
