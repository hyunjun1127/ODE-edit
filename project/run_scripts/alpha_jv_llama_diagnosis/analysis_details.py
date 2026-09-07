"""Raw-free S interpretation tables; no execution, selection, or imputation."""
import math

from .sweep import MODELS, endpoints, paths

MISSING = "NOT_RECORDED_SCHEMA_GAP"


def augment_tables(tables, accounting=None):
    main = {(r['model'], r['candidate']): r for r in tables['main']}
    tables['paired_headline'] = []
    tables['endpoint_mechanics'] = []
    tables['first_hit_summary'] = []
    tables['kernel_cost'] = []
    tables['budget'] = []
    for (model, candidate), row in main.items():
        if candidate == 'PRE_EDIT':
            continue
        for reference in ('O_NATIVE', 'JV-BASE'):
            if candidate == reference or (model, reference) not in main:
                continue
            ref = main[model, reference]
            out = dict(model=model, candidate=candidate, reference=reference)
            for metric in ('RS', 'PS', 'NS'):
                out[metric + '_delta_pp'] = 100 * (row[metric + '_rate'] - ref[metric + '_rate'])
            for kind in ('rewrite', 'rephrase', 'locality'):
                for target in ('new', 'true'):
                    for stat in ('mean', 'median', 'p90', 'max'):
                        field = f'{kind}_{target}_nll_{stat}'
                        out[field + '_delta'] = row[field] - ref[field]
            out['distribution_tail_delta_is_not_paired_delta_quantile'] = True
            tables['paired_headline'].append(out)
        layer = [r for r in tables['endpoint_layers'] if r['model'] == model and r['candidate'] == candidate]
        energy = row['terminal_frobenius_squared']
        out = dict(model=model, candidate=candidate, actual_endpoint_Frob_sq=energy,
            actual_endpoint_Frob_norm=math.sqrt(energy) if isinstance(energy, (int, float)) else MISSING,
            L8_endpoint_share=next((r['energy_share'] for r in layer if r['layer'] == 8), MISSING),
            V_ratio=row['V_ratio'], integrated_native_work=row.get('E_T', MISSING),
            raw_integrated_work=row.get('raw_integrated_work', MISSING),
            native_path_length=row.get('native_path_length_normalized', MISSING),
            native_net_raw=row.get('native_net_raw', MISSING),
            native_net_normalized=row.get('native_net_normalized', MISSING))
        tables['endpoint_mechanics'].append(out)
    for model in MODELS:
        for path in paths():
            nodes = [r for r in tables['nodes'] if r['model'] == model and r['path_id'] == path.path_id]
            if not nodes:
                continue
            hit = [r for r in tables['first_hit'] if r['model'] == model and r['path_id'] == path.path_id]
            known = [r for r in hit if r['status'] == 'POST_NODE_OBSERVATION_ONLY']
            tables['first_hit_summary'].append(dict(model=model, path_id=path.path_id, requests=len(hit),
                recorded_requests=len(known), ever_observed_hit=sum(r['first_observed_hit_node'] is not None for r in known),
                terminal_strict=sum(bool(r['terminal_strict']) for r in known),
                transient_hit_terminal_miss=sum(bool(r['transient_hit_terminal_miss']) for r in known),
                dynamics_influence=0, entry_hit_status=MISSING))
            sums = {}
            for field in ('seconds_native_dictionary', 'seconds_main_jvp', 'seconds_primary_NNLS',
                          'seconds_shadow_metric_NNLS', 'node_wall_seconds', 'training_semantic_wall_seconds'):
                vals = [r.get(field) for r in nodes]
                sums[field] = sum(vals) if all(isinstance(v, (int, float)) for v in vals) else MISSING
            tables['kernel_cost'].append(dict(model=model, path_id=path.path_id, nodes=len(nodes), **sums,
                scope='SCOPED_NODE_TIMERS_NOT_DISJOINT_FROM_PATH_WALL',
                excludes_from_node_wall='post-node semantic observation, raw sink IO, prefix evaluation, terminal finalization'))
    if accounting:
        for row in accounting['jobs']:
            tables['budget'].append(dict(**row, GPU_hours=row['charged_GPU_seconds']/3600))


def detailed_sections(package, table):
    """All narrative values derive from the displayed frozen tables."""
    t = package['tables']
    sections = ['## 결과 요약과 해석 경계']
    for model in MODELS:
        selected = [r for r in t['main'] if r['model'] == model]
        rows = {r['candidate']: r for r in selected}
        if 'JV-BASE' not in rows or 'O_NATIVE' not in rows:
            continue
        base, official = rows['JV-BASE'], rows['O_NATIVE']
        lambdas = [rows[s.candidate_id] for s in endpoints() if s.candidate_id == 'JV-BASE' or 'LAM-' in s.candidate_id]
        sections += [f'### {model}',
            f"기본 JV(λ=.1,T2,N4)는 Official 대비 RS {100*(base['RS_rate']-official['RS_rate']):+.2f}pp, "
            f"PS {100*(base['PS_rate']-official['PS_rate']):+.2f}pp, NS {100*(base['NS_rate']-official['NS_rate']):+.2f}pp다. "
            f"Rephrase target-new NLL mean은 {official['rephrase_new_nll_mean']:.6f} → {base['rephrase_new_nll_mean']:.6f}, "
            f"p90은 {official['rephrase_new_nll_p90']:.6f} → {base['rephrase_new_nll_p90']:.6f}다. "
            "Preference success와 target-new likelihood/tail은 다른 관찰이므로 하나로 개선이라고 묶지 않는다.",
            f"T2/N4 고정 5λ 실제 sweep의 PS 범위는 {min(r['PS_rate'] for r in lambdas)*100:.2f}–{max(r['PS_rate'] for r in lambdas)*100:.2f}%, "
            f"NS는 {min(r['NS_rate'] for r in lambdas)*100:.2f}–{max(r['NS_rate'] for r in lambdas)*100:.2f}%, "
            f"V/V0는 {min(r['V_ratio'] for r in lambdas):.6g}–{max(r['V_ratio'] for r in lambdas):.6g}다. "
            "이는 정상 cold-entry development 반응이며 역사적 Llama B10 near-stall의 원인 검증이 아니다."]
        physics = {r['candidate']: r for r in t['endpoint_mechanics'] if r['model'] == model}
        if isinstance(physics['JV-BASE']['L8_endpoint_share'], (int, float)) and isinstance(physics['O_NATIVE']['actual_endpoint_Frob_sq'], (int, float)):
            sections.append(f"기본 JV의 L8 actual endpoint energy share는 {100*physics['JV-BASE']['L8_endpoint_share']:.4f}%다. "
                f"전체 실제 squared-Frobenius update는 {physics['JV-BASE']['actual_endpoint_Frob_sq']:.6g}, "
                f"Official은 {physics['O_NATIVE']['actual_endpoint_Frob_sq']:.6g}로, "
                f"비율은 {physics['JV-BASE']['actual_endpoint_Frob_sq']/physics['O_NATIVE']['actual_endpoint_Frob_sq']:.4f}배다. "
                "따라서 이 cold S 표본을 이전 sequential B10의 거의 무동작 endpoint와 같은 현상으로 취급할 수 없다. "
                "L8 집중이 강하지만 이것만으로 집중의 원인이나 유익성을 확정하지 않는다.")
        axis = []
        for label in ('JV-LAM-001', 'JV-LAM-00316', 'JV-BASE', 'JV-LAM-0316', 'JV-LAM-1',
                      'JV-RES-N2', 'JV-RES-N8', 'JV-HOR-T1', 'JV-HOR-T4'):
            r = rows[label]
            axis.append(dict(candidate=label, lambda_=r['lambda_response'], T=r['effective_T'], N=r['effective_N'],
                RS_pct=100*r['RS_rate'], PS_pct=100*r['PS_rate'], NS_pct=100*r['NS_rate'],
                PS_delta_O_pp=100*(r['PS_rate']-official['PS_rate']), NS_delta_O_pp=100*(r['NS_rate']-official['NS_rate']),
                PS_new_mean=r['rephrase_new_nll_mean'], PS_new_p90=r['rephrase_new_nll_p90'], V_ratio=r['V_ratio']))
        sections.append(table(axis, ('candidate','lambda_','T','N','RS_pct','PS_pct','NS_pct','PS_delta_O_pp','NS_delta_O_pp','PS_new_mean','PS_new_p90','V_ratio')))
        sections += ["T2 고정의 N2/N4/N8 차이는 수치 해상도 민감도다. 더 작은 terminal V만으로 더 나은 generalization 또는 수치 수렴 PASS를 선언하지 않는다. "
                     "T1/T2/T4는 h=.5 동일 parent 경로의 endpoint다. 동일 trajectory prefix를 별도 독립 실행으로 부풀리지 않는다."]
        for kind, title in (('rewrite','Rewrite'),('rephrase','Rephrase')):
            nll = []
            for r in selected:
                for target in ('new','true'):
                    nll.append(dict(candidate=r['candidate'], target=target,
                        **{stat:r[f'{kind}_{target}_nll_{stat}'] for stat in ('mean','median','p90','max')}))
            sections += [f'### {model} — {title} NLL (target-new / target-true)',
                table(nll, ('candidate','target','mean','median','p90','max'))]
        accuracy = []
        for r in selected:
            accuracy.append(dict(candidate=r['candidate'],
                rewrite_token_acc=f"{r['rewrite_accuracy_n']}/{r['rewrite_accuracy_d']}",
                rephrase_token_acc=f"{r['rephrase_accuracy_n']}/{r['rephrase_accuracy_d']}",
                rephrase_strict_prompt=f"{r['rephrase_strict_n']}/{r['rephrase_strict_d']}",
                rephrase_strict_request_success=f"{r['rephrase_strict_request_success_n']}/{r['rephrase_strict_request_d']}",
                rephrase_strict_request_accuracy=f"{r['rephrase_strict_request_accuracy_n']}/{r['rephrase_strict_request_d']}"))
        sections += [f'### {model} — secondary accuracy / strict',
            table(accuracy, ('candidate','rewrite_token_acc','rephrase_token_acc','rephrase_strict_prompt','rephrase_strict_request_success','rephrase_strict_request_accuracy'))]
    sections += ['## 실제 weight 변화와 layer 집중',
        table(t['endpoint_mechanics'], ('model','candidate','actual_endpoint_Frob_sq','L8_endpoint_share','V_ratio','integrated_native_work','raw_integrated_work','native_net_raw')),
        "모든 endpoint는 같은 모델의 cold W0 대비 실제 FP32 weight 차이다. L8 share는 해당 endpoint total squared Frobenius를 분모로 쓴다. "
        "이 cold AlphaEdit fixture에서 native history M0=0이므로 source L2 metric과 Frobenius가 일치할 수 있으며, 이를 warm/lifelong history에서도 같다고 일반화하지 않는다. "
        "E=Σh QN(F), native net action, actual rounded endpoint Frobenius는 서로 다른 양이다. Prefix의 미기록 native net은 NA로 유지한다.",
        '![Layer-wise Update Magnitude](layer-update-magnitude.png)',
        '## 내부 node progression / 첫 strict hit',
        table(t['nodes'], ('model','path_id','node','t','V_ratio','actual_step_DeltaW_squared','L8_step_energy_share','support_count','model_error_normalized','finite_step_dissipation_defect')),
        table(t['first_hit_summary'], ('model','path_id','recorded_requests','ever_observed_hit','terminal_strict','transient_hit_terminal_miss')),
        "내부 training strict predicate는 controller observation이다. Canonical rephrase/NS endpoint 결과와 같은 분모로 혼동하지 않는다. "
        "첫 hit 이후 다시 miss가 생겨도 실행을 멈추거나 해당 요청을 제거하지 않았다.",
        '![Path physics](path-physics.png)',
        '## 시간·실제 연산 호출·예산',
        table(t.get('fidelity', []), ('model','status','JVP_calls','FD_forwards','dictionary_build_count','authoritative_write_count','W0_M_RNG_restore')),
        table([r for r in t['compute'] if r['scope']=='ACTUAL_PATH'],
              ('model','component','actual_nodes','main_JVP_calls','dictionary_build_count','native_solve_count','wall_seconds')),
        table(t['kernel_cost'], ('model','path_id','seconds_native_dictionary','seconds_main_jvp','seconds_primary_NNLS','seconds_shadow_metric_NNLS','node_wall_seconds','training_semantic_wall_seconds')),
        table([r for r in t['compute'] if r['scope'] in ('PROCESS_TOTAL','SCOPED_RUNTIME_COMPONENT')],
              ('model','scope','component','wall_seconds','model_forward_invocations','evaluation_seconds','peak_allocated_gpu_bytes','peak_reserved_gpu_bytes')),
        table(t['budget'], ('job_id','state','charged_GPU_seconds','GPU_hours','reserved_remaining_GPU_seconds')),
        "dictionary/solve/JVP는 실제 계수이며 FLOPs로 환산하지 않았다. node timer는 기록 시점상 이후 semantic·raw IO·prefix 평가·terminal을 제외한다. "
        "경로 wall에는 평가·진단·artifact I/O도 포함된다. Official component와 경로 wall의 정의가 달라 pure edit-core overhead ratio는 NOT_COMPARABLE_NOT_SEPARATELY_TIMED다. "
        "특히 T4 parent의 세 endpoint에 임의 시간 몫을 배정하지 않는다. process GPU peak는 누적 high-water이므로 각 arm 독립 peak가 아니다.",
        '![Compute](path-compute.png)',
        '## 후속 판단 — development 한정, audit/lifelong 미확정',
        "이번에는 두 모델 모두 5λ actual 결과와 N/T 축을 빠짐없이 공개했다. Llama의 rephrase/NS tradeoff가 λ만으로 해소됐다고 볼 수 없으며, "
        "Qwen의 preference 또는 NS 이득도 rephrase target-new/tail과 비용을 함께 봐야 한다. "
        "모델별 operating point를 자동 선택하거나 audit 결과를 가정하지 않았다. audit300 미실행, D exact replay/intervention 미실행, sequential confirmation/lifelong 미실행이다. "
        "따라서 이는 개발 표본 결과이며 독립 확인 PASS나 lifelong readiness PASS가 아니다. 추가 run은 이 보고서 생성에 필요하지 않아 제출하지 않았다.",
        '![Rates](endpoint-rates.png)', '![Paired NLL deltas](paired-nll-deltas.png)']
    if package.get('accounting'):
        a=package['accounting']
        sections.append(f"본 S attempt의 실제 charge={a['charged_GPU_seconds']} GPU-sec={a['charged_GPU_seconds']/3600:.6f} GPUh. "
            f"D+S 승인 총48 GPUh 중 등록된 attempt 기준 잔여 {a['remaining_unreserved_GPU_seconds']/3600:.6f} GPUh이며, 이는 실행 확대 승인이 아니다. "
            "새 job 제출0·기존 job 변경0·새 model/evaluation0의 분석 작업이다.")
    return '\n\n'.join(sections)
