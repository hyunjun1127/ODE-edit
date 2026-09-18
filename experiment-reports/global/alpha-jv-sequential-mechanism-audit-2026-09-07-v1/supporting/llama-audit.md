# GH 독립 보조 분석: Llama JV B10 near-noaction과 λ sweep의 범위

2026-09-07. 분석만 수행했다. 모델 로드, GPU, 새 평가, Slurm, source/normalization 변경, Server2/SH4 live 접근은 모두 0이다. 이 문서는 root GH 종합 검토용이며 기존 보고서를 대체하지 않는다.

## 1. 입력과 검증 범위

실행 source `77358b1546d1baf83b3e251afcce663b08d7bfd7`의 `native_response_ode_v31/algebra.py`, `alpha_native_response_ode_v31_sequential/trajectory.py`와 publication `0d0a0131...`의 main-four-terminal-v1 CSV를 읽었다. SH1 상세 보고서의 새로운 계산을 그대로 채택하지 않고, 원 publication의 80개 full H/g/G/c/q 행에서 독립 계산했다.

주 입력 `layer_allocation_nodes.csv` SHA256: `58c91dd3196a01bcbd5771387ff44c2dc1b1bca7dbee1c2ca74e43323dbc476b`.

- CPU FP64 독립 active-face NNLS: 80/80 기존 coefficient 재현, 최대 상대차 `5.5568e-16`.
- 각 기존 state에 λ=`10^(-3+k/4)`, k=0..12의 fixed-dictionary shadow 1,040개를 계산했다. 실제 trajectory sweep이나 선택된 새 operating point가 아니다.
- 독립 Jacobi eigenvalue 계산은 기존 NumPy 환경의 `eigvalsh`와 80/80 대조했고 최대 상대차 `7.8337e-15`였다.
- Server2 외부 weight/target/raw-log 파일은 현재 재접근하거나 독립 재해시하지 않았다. 그 파일들의 관측은 publication의 raw-free source-binding receipt 수준이며, 재생 가능한 원인 검증과 구분한다.

재현: `python3 llama_sensitivity.py` (새 output directory에서 실행; create-once). 산출물: `llama-spectrum-all80.csv`, `llama-lambda-shadow-all1040.csv`, `llama-sensitivity-summary.json`.

## 2. Llama 최종 실패는 대부분 B10 신규 edit 미실현이다

JV final W10 RS는 923/1000이다. 77건 실패 중 75건은 B10에서 처음 쓴 직후부터 실패한 건이다. B1..B9의 900건은 at-write 900/900이었고 final W10에서도 898/900이다. 나머지 2건만 at-write 성공 후 실패한 rewrite다. 따라서 이를 `전체 1000개가 sequential forgetting으로 붕괴`했다고 표현하면 안 된다.

그러나 B10만 문제인 것도 아니다. Llama JV의 current-B100 PS는 B1..B9 모든 batch에서 Official보다 낮았으며, NS도 B1..B9 전부 낮다. 첫 batch부터 PS 177/200 vs 185/200, NS 858/1000 vs 869/1000이다. 즉 일반화/locality operating-point 차이는 cold B1부터 존재하고, B10에서는 별도의 급격한 under-write가 더해졌다.

B10의 JV는 current RS 25/100, PS 52/200이고 rewrite target-new NLL 평균 8.5842다. Official B10은 RS100/100, PS197/200, rewrite NLL 0.0029466이다. 실제 JV batch-net Frobenius energy는 `2.7167e-9`로, 정상 batch의 약 10^2 energy와 단위 자체가 다를 만큼 작다. L8 비중이 떨어지고 L7이 남는 현상은 유용한 재배분이 아니라 near-noaction이다.

## 3. 수치 선형계는 잘 풀렸지만, normalized response가 거의 유용하지 않았다

source 목적은 `0.5||e-Psi c||² + 0.5 λ cᵀGc`, c>=0, λ=.1이다. 다음 값들은 같은 node0에서 비교했다.

| 관측 | Llama B9 | Llama B10 | Qwen B10 |
|---|---:|---:|---:|
| H 최소 고유값 | 1.37739 | 1,661,413.85 | 3.41963 |
| H 최대 고유값 | 30.9114 | 5,301,647.21 | 59.42775 |
| condition number | 22.442 | 3.191 | 17.378 |
| λ / 최소 고유값 | 0.07260 | 6.019e-8 | 0.02924 |
| 자유부호 λ=0 최적 선형 설명 비율 | 0.995174 | 0.0000446074 | 0.984624 |
| 실제 normalized model error / 예측 step norm | 0.003223 | 8.3151 | 0.001960 |

`자유부호 λ=0 최적 선형 설명 비율 = gᵀ H⁻¹ g / ||e||²`이다. 이는 과거 state의 전체 5-column response span에서 어떠한 부호의 coefficient라도 허용했을 때 설명할 수 있는 normalized error energy 비율이다. 실제 method의 양수 제약을 없애자는 제안이 아니라, 현재 저장된 e/Psi의 정렬을 진단하는 상한이다.

B10 H는 singular나 ill-conditioned 행렬이 아니다. 오히려 condition은 정상 B9보다 작다. 문제는 H의 큰 절대 규모와 거의 직교하는 current error다. B10 node0 `g=[-9.5122,-6.5042,-.8615,4.7500,.4847]`, response/residual cosine은 모든 layer에서 절대값 `.00555` 이하다. 실제 NNLS는 L7 하나만 사용하여 `c7=1.28735e-6`를 선택했다. L8은 사전에 제외된 것이 아니라 full joint solve의 inactive solution이다. 다음 node에서도 계수는 약 1e-6 이하이고 마지막 2개 node의 L8 gain 자체가 음수다.

λ=0이고 양수 제약도 없는 최적 선형 projection조차 B10 error energy의 0.00446%만 설명할 수 있다. 따라서 이 state에 갇힌 문제를 단순히 `history penalty λ가 너무 크다` 또는 `NNLS가 L8에 강제로 몰았다`로 설명할 수 없다.

## 4. Tiny-positive N0가 강한 원인 후보지만, per-request 인과 검증은 아직 없다

N0는 FP32에서 `s_i=||z_i*-Phi_i(entry)||`를 계산하고, `s_i>0`인 request에 `1/(sqrt(B)*s_i)`를 준다. 정확히 0만 inactive이며 작은 양수 floor는 없다. Native writer RHS는 이 가중치를 쓰지 않지만 objective/JVP는 쓴다.

Llama B10:

- case4228: s=`3.5747129e-5`, row weight=`2797.42745`.
- 두 번째 작은 s=`4.3893561`; batch median=`5.2683704`.
- case4228의 `s_min/s_median=6.7852e-6`.
- row-weight 제곱합에서 case4228 비중=`99.999999536%`. 이는 **weight의 비중**이고, 실제 H나 response-energy 비중이라고 바꿔 쓰면 안 된다. 각 request의 raw JVP norm도 필요하기 때문이다.
- sealed native-target observation은 case4228의 Official compute_z가 첫 rounded loss `.043`에서 종료하며 optimizer delta norm=0이라고 기록한다. source 규칙 `.05` 아래 조기 종료는 변경되지 않았다. 그럼에도 별도 terminal capture와의 FP32 차이는 작은 양수이고, N0 active count에 포함됐다.

이 조합은 `거의 이미 만족한 command의 작은 capture 차이가 큰 inverse scale을 받고, normalized Jacobian의 민감도를 지배하면서 다수 신규 command의 realization을 억제했다`는 가설과 일관된다. 그러나 per-request Gram/JVP contribution은 NOT_RECORDED이고 noise 반복 측정 또는 동일 state normalization ablation도 없으므로 이 인과 문장은 아직 확정하면 안 된다.

Qwen B10의 최소 s=48.2253, median117.3486, min/median=.410958이며 최대 row-weight 제곱비중 .0416673이다. Llama B10과 같은 tiny-positive configuration이 관측되지 않았다. 이 차이는 architecture/history 전체 차이와 별도로 sample-specific normalization 조건의 차이다. `Llama는 원래 안 되고 Qwen은 된다`는 보편화도 금지한다.

## 5. λ sweep은 어디에 영향을 주고 어디에는 거의 안 주는가

같은 B10 4개 node의 λ .001..1 shadow에서, 기준 λ=.1 대비 physical field의 native relative distance 최대는 `2.54812e-7`이다. 모든 node에서 active support도 유지된다. B10 node0은 λ=.001에서 `[L7]`, .1에서 `[L7]`, 1에서 `[L7]`이다. 0.1/최소 eigenvalue가 6e-8 수준이어서 이 범위의 정규화는 이미 큰 H에 비해 거의 보이지 않는다.

History-only cost shadow도 일관된다. B10 node0의 frozen actual basis/response에서 M_entry+L2를 M0+L2로 바꾼 기존 shadow는 계수 상대차 약 `1.4945e-9`, native cosine1.0이다. 이는 과거 history가 source에서 빠졌다는 뜻이 아니다. 이 특정 node에서 **cost 직접효과**가 normalized response Hessian에 비해 극히 작다는 뜻이다. History가 이미 바꿔놓은 W/dictionary는 이 shadow에서 그대로다.

반면 정상 batch에서는 λ가 작지 않다. Llama B9 node0에서 .001/.1/1에 따른 native field relative change는 `.04899/0/.28034`이고, L8 native velocity share는 `.9999967/.9994763/.9735486`이다. Qwen B10에서도 상대 변화는 `.02298/0/.16936`이다. 따라서 dense λ sweep은 정상 state의 operating point와 이전 batch trajectory를 바꾸는 연구로는 의미가 있다. 하지만 `현재 B10 tiny-positive state를 직접 구제한다`는 검증으로 간주하면 잘못이다.

실제 λ trajectory가 이전 batch부터 다르게 진행되면 B10 W/z/s/dictionary도 달라질 수 있다. 위 fixed-state 결과만으로 모든 actual λ run이 실패한다고 단정하지 않는다. 반대로 그러한 경로 변화로 우연히 case4228 정규화 상태를 피한 것만으로 원인 교정이 됐다고 단정해서도 안 된다.

## 6. Barrier가 성립한다는 말의 세 수준

1. KKT/instantaneous native dissipation identity: CPU 재현과 stored KKT residual은 실제 구현이 목적식의 해를 계산했다는 근거다. B10은 해 자체가 매우 작아서 identity가 맞아도 유용한 edit가 없다.
2. Fixed Euler의 실제 potential 변화: B10 node0은 V가 증가했고, node1/2는 감소, node3은 증가했다. finite-step defect는 `+4.01e-5, -6.01e-5, -3.92e-5, +3.65e-5`이다. nominal continuous barrier와 실제 finite-step 안전 보장은 다르다.
3. Locality/retention: native action과 current activation target을 사용하는 목적이지 과거 prompt의 output을 직접 감시하는 functional constraint가 아니다. KKT PASS로 NS 또는 old-edit 보호를 증명할 수 없다.

B10 normalized model error는 예측 linear step norm보다 8.3/12.0/25.6/38.3배 크다. virtual/materialized terminal의 normalized discrepancy `.01418` 역시 기록된 작은 V 감소(~`2.64e-5`)보다 훨씬 큰 해석 불확실성을 만든다. 원래 primal FP32 parity를 사후 strict gate로 변경할 이유는 아니지만, 이 상태의 미세한 V/defect 변화가 실제 의미 있는 개선이라고 주장하면 안 된다. 실제 weight 변화가 near-noaction이라는 결론은 endpoint tensor norm으로 별도 지지된다.

## 7. 제한된 후속 권고

- Llama lifelong 진입은 HOLD가 타당하다. 75/77 failure가 신규 B10 under-write이고 PS/NS 약화는 B1부터 있으므로, lifelong을 늘려서는 원인을 분리하기 어렵다.
- λ sweep 필요성은 `초기/정상 operating point 검토`와 `B10 normalization 기작 진단`으로 분리한다. 현재 수행한 1,040개 shadow는 CPU 관측일 뿐 새 sweep 설정/예산을 봉인하거나 제출한 것이 아니다.
- 기존 exact B10 entry state가 복원 가능할 때만 작고 별도 명시된 진단으로 per-request raw/normalized JVP-H contribution, repeated readout noise envelope, target optimizer zero-delta 여부, scale source를 비교하는 것이 높은 우선순위다. Normalization을 바꾸는 actual run은 기술수정이 아니라 새 과학 버전/ablation이어야 한다.
- tiny row를 제거하거나 floor를 넣고 기존 실패를 지우는 것은 금지한다. 기존 1000 denominator를 유지하고, 가능하면 observer-only same-state projection으로 먼저 원인 영향 범위를 검증해야 한다.
- Qwen 결과를 보며 Llama만 좋은 λ를 사후 선택하지 않는다. sample/development/audit 분리와 fixed budget, main 설정이 별도 봉인된 뒤 actual sweep을 검토한다. 현 SWEEP_HOLD는 유지한다.

이 검토의 새로운 핵심은 **B10은 선형계 수치 conditioning failure가 아니라, 저장된 normalized response 공간에서 error가 거의 설명되지 않는 상태이며 .001..1 λ를 촘촘하게 나눠도 그 same-state 문제는 거의 바뀌지 않는다**는 점이다. Tiny-positive N0는 이를 설명하는 강한 관측 가설이나 per-request 원인 분해 없이는 확정하지 않는다.

## 8. 추가 요청: ours hyperparameter 후보의 근거와 우선순위

이 절은 후보 제안이지 실행 승인/제출이 아니다. AlphaEdit의 compute_z optimizer, learning rate, loss/contexts, clipping, target iteration/stop, native L2/P 등 원본 hparams는 그대로 둔다. JV가 추가한 λ와 Euler의 T/N/h만 분리 검토하며, normalization 진단은 별도 science ablation으로 표시한다. 기존 checkpoint/실패 결과를 교체하지 않는다.

`candidate-sensitivity-summary.csv`에는 원래 80개 node 전체를 포함한 4개 층화(전체 Llama40, LlamaB1–B9 36, LlamaB10 4, Qwen40)의 λ13점 요약 52행을 저장했다. B10을 나눈 것은 원인을 살펴보기 위한 표 분리이며 전체 main 분모 제외가 아니다. 모든 비율은 **동일 state의 λ=.1 field 대비**이다. NS/PS를 보고 λ 후보를 골라낸 것이 아니다.

### 8.1 λ actual pilot 후보: .01, .0316228, .1, .316228, 1

기준 .1을 포함하고 위·아래로 half-decade 간격인 5점이다. 기존13점 shadow의 .001 부근은 정상 상태에서도 .01과 거의 같아 먼저 GPU를 더 촘촘히 채울 근거가 약하다. 더 낮은 값이 L8 집중을 낮추지도 않는다. λ1은 native action을 상당히 줄이지만 가장 명확하게 방향/다층 support도 바꾸므로 `분산 증가 vs 단순 감속`을 구분하는 끝점으로 유용하다.

아래 값은 node별 비율/각도의 median이다. native share는 `c8²/sum(c²)`이므로 실제 endpoint Frobenius share와 혼동하지 않는다.

| 모델/진단 층 | λ | gain gᵀc 비율 | native q 비율 | native field 각도 | L8 native velocity share |
|---|---:|---:|---:|---:|---:|
| Llama B1–B9 | .01 | 1.02142 | 1.08100 | 1.231° | 99.9980% |
| Llama B1–B9 | .0316228 | 1.01611 | 1.06043 | .926° | 99.9901% |
| Llama B1–B9 | .1 | 1 | 1 | 0° | 99.9289% |
| Llama B1–B9 | .316228 | .95459 | .84590 | 2.591° | 99.4740% |
| Llama B1–B9 | 1 | .84756 | .56355 | 8.628° | 96.8953% |
| Qwen all40 | .01 | 1.01334 | 1.04119 | .421° | 99.9995% |
| Qwen all40 | .0316228 | 1.01009 | 1.03101 | .319° | 99.9984% |
| Qwen all40 | .1 | 1 | 1 | 0° | 99.9916% |
| Qwen all40 | .316228 | .96982 | .91207 | .963° | 99.9351% |
| Qwen all40 | 1 | .88913 | .70624 | 3.632° | 99.4817% |

Qwen의 λ1 각도는 node별 최대13.95°이고 gain 비율 최소 .39704, native action 비율 최소 .12919이다. Median 하나로 tail 영향이 없는 것으로 해석하면 안 된다. Support 변경도 기록했지만 tiny coefficient의 출입은 큰 physical redistribution을 의미하지 않는다. Llama B10은 위 λ 전체에서 앞 절의 near-identical field이므로 stage0 conditioning 진단과 분리해야 한다.

### 8.2 Euler 후보는 T와 N을 한 번에 바꾸지 않고 두 질문으로 분리

1. **Fixed T=2, λ=.1에서 N={2,4,8,16}**: h={1,.5,.25,.125}. 같은 연속 field/horizon의 discretization error와 실제 materialization 안정성을 비교한다. node 사이 state feedback 수가 달라진다. N16은 높은 비용의 후순위 확인점이지 먼저 전체 1000 chain을 4배 늘릴 이유가 아니다.
2. **Fixed h=.5, λ=.1에서 T={1,2,4}, N={2,4,8}**: 같은 step size로 horizon을 바꾼다. 충분한 progress 부족 vs 장시간 write가 old-edit/locality를 더 손상시키는지를 본다. 이는 convergence 비교가 아니라 method exposure 변화다.

T2,N4는 두 grid에서 공유되는 기준이다. λ grid × N grid × T grid의 전조합을 바로 돌릴 필요는 없다. 먼저 작은 outcome-independent development fixture와 고정 비용에서 한 축씩 비교하고, 선택 규칙/holdout을 새 실행 전에 봉인해야 한다. 이미 실패한 B10을 좋은 결과가 나올 때까지 반복해 고르는 것은 허용되지 않는다.

| T | N | h | main JVP/batch | main layer solves/batch | entry qref용 추가 layer solves |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 1 | 10 | 10 | 5 |
| 2 | 4 | .5 | 20 | 20 | 5 |
| 2 | 8 | .25 | 40 | 40 | 5 |
| 2 | 16 | .125 | 80 | 80 | 5 |
| 1 | 2 | .5 | 10 | 10 | 5 |
| 4 | 8 | .5 | 40 | 40 | 5 |

이는 현재 생산 source의 all-five-direction 경로 비용이다. `runtime.py`가 entry qref를 위해 5개 direction을 따로 build하고, `run_joint`가 각 N node마다 다시 5개 direction을 만들고 JVP한다. 따라서 N4는 **main20 JVP, entry포함25 layer solves**다. 각 node의 current/post activation forward, shadow solve/metric, materialization observer, endpoint 평가와 history finalization은 별도다. Compute_z optimizer 횟수/정책은 N/T/λ 비교에서 늘리거나 바꾸지 않는다. 두 배 JVP가 total runtime 정확히 두 배라는 뜻도 아니다.

### 8.3 Stage0를 분리해야 하는 이유

Llama B10의 current state에서는 λ grid를 조밀하게 해도 거의 같은 field다. T를 늘리거나 N을 높이면 미세한 normalized signal/rounding에 더 자주 반응할 수 있으나, 이를 올바른 realization이라고 예측할 근거는 없다. 먼저 raw/normalized per-request response energy와 capture noise를 분리하는 observer-only 진단으로 `tiny positive N0` 가설을 검증하는 것이 비용 대비 가치가 높다. Source-exact N0를 유지한 λ/T/N operating-point 실험과 normalization을 바꾸는 ablation은 보고서/분모/source namespace를 분리한다.

이 단계의 제안은 성능이 안 나와서 AlphaEdit z를 더 강하게 최적화하거나 native L2를 바꾸자는 뜻이 아니다. 원본 optimizer를 고정한 채 **JV에서 추가된 regularization 강도, integration 정확도, integration horizon**을 서로 다른 원인 축으로 검토하자는 것이다.
