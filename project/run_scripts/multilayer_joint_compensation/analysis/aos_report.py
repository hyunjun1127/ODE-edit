"""Korean factual report rendering from independently reduced recorded data."""


def table(headers, rows):
    return '\n'+'| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)+'\n'


def render(summary, paired, functional, physical, tr, attr, manifest):
    def s(state,panel,metric):
        return next(r for r in summary if r['state']==state and r['panel']==panel and r['metric']==metric)
    def f(state,role): return next(r for r in functional if r['state']==state and r['role']==role)
    rows=[]
    for panel in ('Current100','Fixed100','Past100'):
        for state in ('N4','A0','A-OS','B-OS'):
            rr=[s(state,panel,m) for m in ('RS','PS','NS')]
            rows.append([panel,state,*[f"{r['success_n']}/{r['denominator']} ({100*r['success_rate']:.1f}%)" for r in rr],
                         f"{float(rr[0]['new_nll_mean']):.8f}", f"{float(rr[1]['new_nll_mean']):.8f}"])
    report='''# Middle A-OS 완료분 사실 검토 — 전체 A/B campaign 미완료

## 1. 범위와 지표

본 문서는 완료된 job45633 A-OS의 CPU-only recall 리뷰다. 새 모델 load/forward/evaluator/GPU/Slurm 실행은 모두0이다. Scheduler COMPLETED와 수치 수렴은 다르다. 실제 endpoint는 finite이고 저장·평가가 완료됐으나, PCG 두 RHS 모두 사전 한도20회에서 미수렴하여 **APPROXIMATE_PCG_NONCONVERGENCE**다. 초기 full-operator 검사를 수렴 또는 성능 PASS로 대체하지 않는다. scientific_promotion=false.

Current100은 Middle W50 다음 B51의100요청, Fixed100/Past100은 공통 봉인 패널이다. 각 패널 RS100, PS200, NS1000 prompt-pair, 합계3900 pairs다. 세 패널을 독립300요청의 실험 반복으로 간주하지 않는다. RS/PS는 target-new NLL < target-true NLL, NS는 반대의 strict inequality이며 tie는 failure다. NLL은 낮을수록 해당 continuation의 가능도가 높다. `margin=true-new`; 유리한 방향의 desired margin은 RS/PS에서는 margin, NS에서는 -margin이다. Token accuracy/sequence strict는 별도 secondary 지표로 CSV에 보존한다.

N4와 B-OS는 exact common READY/panel이 일치하는 SH2의 봉인 aggregate를 재사용했다. 새 remote raw rehash나 cross-hardware numerical parity를 수행하지 않았다. A0/A-OS 및 실제 removal 관측은 로컬 원시 JSON을 독립 join/검산했다. 다른 BLUE 두-layer/원본5-layer AlphaEdit/MEMIT same-entry baseline은 이 리뷰에서 NOT_MEASURED이며 다른 chain 점수로 채우지 않았다.

## 2. 동일 패널 핵심표
'''
    report+=table(['패널','state','RS n/d','PS n/d','NS n/d','rewrite new NLL mean','rephrase new NLL mean'],rows)
    we=[s('We','Current100',m) for m in ('RS','PS')]
    report+=f"We에서 기록된 Current RS={we[0]['success_n']}/100, PS={we[1]['success_n']}/200이다. 이 removal 기준 관측에는 Current NS 및 Fixed/Past 전체 We 평가가 없으므로0으로 채우지 않는다.\n\n"
    report+='![동일 패널 endpoint](endpoint-rates.png)\n\n## 3. A0 → A-OS paired 변화\n\n'
    rr=[r for r in paired if r['comparison']=='A0_to_A-OS' and r['quantity']=='new_nll_delta']
    report+=table(['패널','metric','분모','Δpp','성공→실패','실패→성공','new NLL Δmean','median','p90','max'],
                  [[r['panel'],r['metric'],r['denominator'],f"{r['delta_pp']:+.2f}",r['loss_n'],r['recovery_n'],*[f"{r[k]:+.8f}" for k in ('mean','median','p90','max')]] for r in rr])
    report+='''`paired-summary.csv`에는 true NLL/desired margin의 동일 통계와 positive/equal/negative count도 있다. `paired-case-deltas.csv`는 exact case/prompt/target identity 해시로 결속한3900쌍 및 We→A-OS Current300쌍이다. N4/B-OS에 대해서는 이번 로컬 aggregate만으로 per-case delta나 loss/recovery를 추정하지 않는다. Request/context별 NLL과 sequence/token strict 전체 요약은 `endpoint-summary.csv`에 있다.

![paired NLL](paired-nll-deltas.png)

## 4. 기능적 bank와 독립 Audit

Base/Past reference는 고정 We다. Base는 KL(We||state), Past는 봉인 psi(.1) NLL 증가다. 아래 숫자는 같은 bank의 raw functional value이며 covariance key energy가 아니다. Context별 가중치를 전부 독립 합산하고 기록된 reduction과 FP32 rounding-bound 내 일치를 확인했다. Count, context NLL, p90/max, 음수 raw value count는 CSV에 있다. BaseAudit/PastAudit는 controller bank와 별도로 기록된 관측이다.
'''
    report+=table(['역할','A0 raw risk','A-OS raw risk','Δ(A-OS−A0)','A0 mean NLL','A-OS mean NLL','contexts'],
                  [[role,f"{f('A0',role)['weighted_value']:.9f}",f"{f('A-OS',role)['weighted_value']:.9f}",
                    f"{f('A-OS',role)['weighted_value']-f('A0',role)['weighted_value']:+.9f}",
                    f"{f('A0',role)['mean_nll']:.9f}",f"{f('A-OS',role)['mean_nll']:.9f}",f('A-OS',role)['contexts']]
                   for role in ('Base','Past','BaseAudit','PastAudit','Current')])
    report+='''주의: Current functional profile은 A0의 기준과 A-OS의 fixed WA teacher가 다르므로 두 profile value 차이를 동일 목적 개선으로 해석하지 않는다. 같은 Current context의 mean NLL은 비교 가능하다. N4 calibration bank risk는 Base .0348951202304, Past .0103827207665다. A-OS의 Base/Past 변화가 Audit 일반화 또는 native 대비 우위를 증명한다고 해석하지 않는다. W0-KL·추가 W0 복구 효과는 이번 raw에서 NOT_RECORDED다.

## 5. PCG·constraint·실제 적용

실행된 K는 native .01H + 2/sigmaB GGN_Base + 1/sigmaP GGN_Past + GGN_Current + A balance다. Native preconditioner는100H^-1이며 K의 정확 역행렬이 아니다. 두 layer를 함께 통과하는 full-sequence functional operator 및 cross block을 유지한다. Current는 joint equality 한 개이며 layer별 equality0이다. Current GGN은 KL/.1 Fisher + NLL-profile outer/.1², Past는 psi'' outer와 psi' Fisher를 함께 사용한다. Full Hessian을 저장한 것은 아니다.
'''
    report+=table(['RHS','iterations','actual relres','actual absres','RHS norm','status'],
                  [[k,d['iterations'],f"{d['relative_residual']:.12g}",f"{d['absolute_residual']:.12g}",f"{d['rhs_norm']:.12g}",d['status']] for k,d in tr['solver']['pcg'].items()])
    sol=tr['solver']
    report+=f"""사전 rtol=1e-4/maxiter20을 바꾸지 않았다. 실제 operator 재적용 잔차와 recursive residual은 분리 기록한다. Stationarity norm={sol['stationarity_norm']:.12g}, reference norm={sol['stationarity_reference_norm']:.12g}, 비율={sol['stationarity_norm']/sol['stationarity_reference_norm']:.12g}. Intended correction의 equality residual={sol['equality_residual']:.12g}; range relative residual={tr['permitted_range_relative_residual']:.12g}. 이 작은 equality 하나가 전체 solve 정확도를 보증하지 않는다. OS에는 BF inequality가 적용되지 않았으므로 raw dual/slack [0,0]은 barrier 만족 PASS가 아니라 N/A다.

Predicted Current change={tr['predicted_current_change']:.12g}와 actual Current train-NLL change={tr['actual_current_change']:.12g}를 구분한다. Predicted normalized Base/Past changes={tr['predicted_risk_change']}, actual={tr['actual_risk_change']}. 실제 변화는 선형 예측과 다르며 이것만으로 특정 원인을 확정하지 않는다. 낮은 성능 또는 finite PCG 미수렴 때문에 endpoint를 제외하지 않았다. No extra h/no inner compute-z/no inner history append/no rank truncation은 source 및 telemetry에서 확인된다.

![solver and risk](solver-and-risk.png)

## 6. 실제 weight와 signed 기여

Intended correction norm={tr['correction_norm']:.12g}, 실제 FP32 endpoint−A0 norm={tr['actual_delta_norm']:.12g}. FP32 addition residual norm={tr['fp32_addition_residual_norm']:.12g}; 전체 We→A-OS batch delta norm={tr['actual_cumulative_norm']:.12g}. Saved endpoint를 CPU에서 읽어 full dense block norm과 SHA를 다시 확인했다. 물리 Frobenius norm과 raw native energy는 다른 단위다.
"""
    report+=table(['layer','실제 AOS−A0 norm','실제 AOS−We norm','raw native energy before','after'],
                  [[r['layer'],*[f"{r[k]:.10g}" for k in ('actual_correction_frobenius','actual_cumulative_frobenius','raw_native_energy_before','raw_native_energy_after')]] for r in physical])
    report+='''![weight magnitude](layer-update-magnitude.png)

Signed attribution은 We, We+D4, We+D8, We+D4+D8의 **실제 기록된 Current RS100/PS200** 관측이다. 여기서 D는 A0부터의 correction만이 아니라 We 기준 이번 batch 누적 delta다. E4/E8/E48은 We 대비 target-new NLL 감소, signed L4=(E4+E48−E8)/2, signed L8=(E8+E48−E4)/2, interaction=E48−E4−E8. 음수도 보존하며 norm-share를 기여로 바꾸지 않는다.
'''
    report+=table(['metric','quantity','n','mean','median','p90','max','positive/zero/negative'],
                  [[r['metric'],r['quantity'],r['denominator'],*[f"{r[k]:+.7f}" for k in ('mean','median','p90','max')],f"{r['positive_n']}/{r['zero_n']}/{r['negative_n']}"]
                   for r in attr if r['quantity'] in ('signed_L4','signed_L8','interaction','standalone_E48')])
    report+='''이 paired interaction 수치는 signed NLL decomposition이다. 편집 부담의 인과적 이전, locality guarantee, lifelong 우위 또는 multi-layer capacity 불가능성으로 확대하지 않는다. 추가 Base/Past removal intervention은 이 리뷰에서 새로 실행하지 않았다.

## 7. History·복원 검증 범위

Endpoint weights L4/L8은 terminal의 tensor SHA와 일치하고 FULL_FP32 shape[4096,14336]이다. Saved final M 및 endpoint에서 capture한 K의 tensor SHA도 일치한다. 두 layer 각각 terminal append1, batch finalization1, inner append0이다. CPU에서 원 식 M_entry+KKᵀ를 row-block으로 검산한 residual은 `history-checks.csv`에 그대로 기록한다. CPU BLAS 연산 순서 차이를 숨겨 byte-exact replay라고 하지 않는다. M4는 원 native history, M8는 common fixture에서 같은 We로 준비한 history이며 독립 새 lifelong provenance가 아니다.

Actual endpoint materialization1/restore1, attribution materialization2/restore2, 각각 version-increment0이 terminal counters에 있다. Virtual/physical forward parity는 실제 관측2contexts에서 byte_equal=true/l2_error0이다. 전체3900 평가를 다시 forward해 parity를 증명한 것은 아니다. Source의 JointView guard는 모든 parameter의 pointer/version/shape/dtype/device 및 selected byte SHA를 검사한다.

**복원 기준은 We다.** common session은 W0 로딩 확인 후 selected weights를 We로 바꾸고 guard를 생성한다. Materialization context는 그 We로 복원한다. Process 종료 직전 W0 전체 bytes 복원 또는 모든 nonselected parameter/buffer bytes의 manifest는 NOT_RECORDED다. 따라서 ‘W0 exact final restore PASS’로 보고하지 않는다. 이 한계 때문에 기록된 endpoint 평가를 삭제하거나 다른 상태로 대체하지 않는다.

## 8. 비용과 카운터

Scheduler45633: COMPLETED0:0, 2026-09-12 12:24:06–18:58:23, allocation23657 GPU-sec=6.57139 GPUh(1GPU). Batch/extern 시간을 중복 합산하지 않는다. AOS protection solve23174.478777s, model load10.338117s, teacher35.055517s, full endpoint147.234326s, removal Current29.158636s, generation90.408074s, native key capture27.075353s. 이 timer들은 계층/중첩 관계가 있으므로 단순 합계를 새로운 GPU 시간으로 만들지 않는다. A0/공통 cold preparation의 과거 비용은 AOS 신규 allocation에 포함되지 않았으며 ‘cold end-to-end 전체’라고 부르지 않는다.

Full K matvec43, native inverse42; 두 RHS 각각 PCG20회+실제 residual 평가1회, stationarity 추가1회다. Functional solve JVP21156/VJP21156, 명시 gradient backward792, logits-forward43296. Actual model invocation47484와 functional-view44283은 포함관계이므로 합산0. Evaluation9000 candidate sequences/1130forward; generation60prompts/1920output tokens(고정 greedy32)이다. 생성 문자열은 local-only이고 semantic accuracy로 재해석하지 않았다.

Host weight transfer counter20,802,003,009,536 bytes는 explicit storage-copy accounting이지 측정한 PCIe bandwidth/전체 tangent traffic이 아니다. Peak allocated40,262,761,984B, reserved42,557,505,536B. CPU PCG vector/actual model은 FP32, native factors FP64, scalar reductions FP64다. 긴 operator 비용과 실제 미수렴을 함께 보고하며 ‘작은 최종 scalar solve라 저렴하다’고 주장하지 않는다. 상세 `compute-ledger.csv`는 단위/포함관계를 보존한다.

## 9. Coverage와 현재 미완료

이번 완료분: Middle A-OS endpoint3900pairs, functional Base/Past/Current 및 Audit, Current removal300pairs×두 layer, 기존 We300pair 재사용, generation60, saved W/M 및 비용이다. Middle A0와 N4/B-OS의 봉인 비교는 위 범위만 재사용했다. E01 두 replacement의 진행/성능 결과는 이 보고서에 혼합하지 않았다.

나머지 Early/Late A 계열, Middle 추가 variant, native Alpha/MEMIT/BLUE 비교, A/N4/BLUE shortchains, SH2의 나머지 B campaign은 이 문서가 완료를 주장하지 않는다. B-BF4 paused task를 조회하거나 깨우지 않았다. Full26+baseline6 endpoints/4shortchains 완료 아님. 미측정 자료는0/추정/다른 chain으로 대체하지 않았다. 이번 표는 사실·수치 package이며 최종 원인 종합과 후속 방법 선택은 GH 소유다.

## 10. 재현·검증·산출물

Raw full terminal24members는 분석 전후 전체 SHA/bytes/mode와 members root를 검증했다. Common184members는 기존 receiver 검증을 재사용하고 이번 분석에 읽은 subset만 새 SHA 검증했다. Dataset SHA/panel/order/target identity,3900row uniqueness, strict tie 방향, functional weighted reduction, endpoint 및 history tensor SHA를 결속했다. Plot은 저장소 Python code로 두 번 실제 렌더링하여 byte-identical 확인했다. Raw prompts/tensors/model/cache/generation text/log는 Git 제외다.
'''
    report+=f"\nExecution source `{manifest['execution_source']}` / tree `{manifest['execution_tree']}`. Analysis source `{manifest['analysis_source']}` (code commit; publication commit과 구분). Pure CPU helper는 `{manifest['helper_commit']}:{manifest['helper_path']}` SHA `{manifest['helper_sha256']}`를 Git object에서 검증하여 재사용한다. 저장소에 해당 object가 없으면 공개된 SH1 source branch를 fetch한 뒤 실행해야 하며, 임의 최신 runtime을 import하지 않는다.\n\n```bash\n{manifest['reproduce_command']}\n```\n\n"
    report+='''주요 파일: `endpoint-summary.csv`, `paired-summary.csv`, `paired-case-deltas.csv`, `functional-risk.csv`, `functional-context-values.csv`, `signed-attribution.csv`, `signed-attribution-summary.csv`, `physical-update.csv`, `history-checks.csv`, `pcg.csv`, `pcg-iterations.csv`, `solver-and-transaction.json`, `compute-ledger.csv`. Input/source/member identities는 `input-manifest.json`, PNG 환경·명령·입력/출력SHA는 `plot-reproduction.json`, 출력 전체 결속은 `analysis-manifest.json` 및 `rooted-receipt.json`이다. 기존 report/raw bytes는 덮어쓰지 않았다.
'''
    return report
