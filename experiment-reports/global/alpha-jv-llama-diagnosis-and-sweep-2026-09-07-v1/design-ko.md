# Llama 원인 진단 + 모델별 hparam sweep 실험 설계

작성: 2026-09-07. **DESIGN_ONLY**. GH용 실행 본문은 [gh-execution-prompt-ko.md](gh-execution-prompt-ko.md).

## 1. 이번 설계의 결론

원인 진단과 hparam sweep은 병렬로 하는 것이 맞다. 단, 서로 다른 질문에 답하도록 입력과 결론을 분리해야 한다.

- **Track D:** 이미 실패한 Llama B10의 같은 W/M/z에서 무엇이 tiny field와 신규 edit 실패를 만들었는가?
- **Track S:** 정상 common-entry에서 λ, 적분 시간 T, resolution N을 바꾸면 실제 edit/locality 품질이 어떻게 달라지는가?

기존 same-state 분석은 B10에서 λ .001–1의 변화가 field에 거의 영향을 주지 않음을 보였다. 반면 정상 Llama state에서는 λ1이 .1 대비 median predicted gain .8476배, native action .5635배, field 회전8.63°를 만들었다. **정상 상태의 hparam 영향은 충분히 클 수 있다.** B10 진단을 기다리며 sweep을 모두 멈추거나, sweep의 좋은 결과를 B10 원인 규명으로 대신하는 두 극단을 피한다. [GH 통합 v2 §8–9·13](https://github.com/hyunjun1127/ODE-edit/blob/e25a5685a4ad7f8606a8c265abf070b0d77a83d1/experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/gh-mechanism-review-ko.md)

상위 계획의 5개 λ 후보를 모두 유지한다. 이전의 3개 coarse actual 우선 배치에서, 이번 사용자 추가 요청에 따라 **5개 λ actual development를 S의 기본 범위**로 올린다. 단, 1k full-chain 5개가 아니라 공통 B100 fixture에서의 비교다. N16은 그대로 조건부다.

두 트랙과 정상 control 준비는 Server4 두 task의 종료에 의존하지 않는다. 별도 GPU가 없으면 CPU 준비를 진행하고 자원 상태를 보고한다. 기존 run을 중단·수정·탑재 공유하지 않는다.

## 2. 어떤 원인들을 구분해야 하는가

### 2.1 확인된 현상을 정확히 정의

완료 Llama JV final RS는923/1000이며 실패77건 중75건은 B10에서 처음부터 실패했다. B1–B9는 at-write900/900, final898/900이다. 그러나 B1–B9의 PS/NS는 모든 batch에서 Official보다 낮았다. B1은 M0이므로 **cold B1의 열세를 과거 cache 보호 부족으로 설명할 수 없다.** [Llama 보조 분석](https://github.com/hyunjun1127/ODE-edit/blob/e25a5685a4ad7f8606a8c265abf070b0d77a83d1/experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/supporting/llama-audit.md)

| 가설 | 기존 근거 / 남은 구분 | 가장 작은 식별 실험 |
|---|---|---|
| D1. Tiny-anchor weighting이 실제 response를 지배 | case4228 scale 약3.57e-5. 역가중치가 크다는 사실만 확인 | Raw requestwise JVP → 각 row의 H/g → N0/NRMS same-state solve |
| D2. Tiny anchor가 readout/rounding 차이 | Compute-z delta0, 다른 capture의 tiny-positive residual | 동일 capture 반복과 source별 context/token/batching 비교; arithmetic-order observer |
| D3. 실제 batch interference를 N0가 과하게 보호 | 작은 target residual이어도 해당 row의 실제 directional response는 클 수 있음 | Raw response 및 실제 해당 request의 semantic 손실을 나머지99개와 함께 평가 |
| D4. Native history/λ가 너무 강해 stall | B10에서 λ와 cost-only M0 shadow는 거의 영향 없음 | 기존 CPU 결과 재사용 + raw unweighted geometry와 비교; λ sweep을 root-cause 대체로 사용하지 않음 |
| D5. NNLS/coordinate/state 구현 오류 | 기존 KKT는 양호하지만 새 adapter는 검증 필요 | Production g/H 재구성, coefficient roundtrip, shared-entry identity |
| D6. Intended write가 FP32 materialization에서 소실 | B10 near-noaction과 큰 relative model error | Intended hF / virtual response / dense materialized delta를 분리 |
| D7. 고정 target 또는 현재 writer의 실현 가능성 한계 | 기존 Official B10은 다른 W/M/z에서 성공 | 같은 JV B10 entry/target에서 Official one-pass control |
| S1. λ가 정상 상태의 quality–action trade-off를 바꿈 | 정상 state의 실제 field 회전·감속 근거 | 5λ, 동일 T/N/normalization/entry의 actual paths |
| S2. Horizon 부족 또는 과도한 anchor 추적 | 현재 T2 한 점만으로 판별 불가 | h.5 고정 T1/T2/T4; first-hit 관측만, stopping은 OFF |
| S3. Euler discretization/수치 오차 | N4 한 점; 더 작은 h가 rounding을 악화시킬 수도 있음 | T2 고정 N2/N4/N8, 실제 endpoint distance·write fidelity |
| S4. Readout에서 좋은 방향이 semantic/generalization에 나쁨 | L8 alignment가 높아도 Llama PS/NS 약화 | Activation progress와 training/held-out NLL·margin의 시간 경로 비교 |

어느 한 가설을 미리 정답으로 고정하지 않는다. 특히 **NRMS 개선 ≠ FP32 bug 확정**이다. N0는 request별 relative error를, NRMS는 공통 단위의 absolute activation error를 중시한다. 같은 entry V0≈.5라도 objective geometry와 λ의 상대효과는 바뀐다. 유효한 작은-residual request를 보호하려는 수학적 trade-off였을 가능성도 남는다.

## 3. Fixture 설계와 복원 한계

### 3.1 D-B10

필수 identity는 원래 JV의 W9, M9, `cache_c_new`, B10 z tensor/hash, request/context order, source/model/tokenizer/P/hparams/backend다.

공개 source는 W1/M1, W5/M5, W10/M10을 저장하고 z는 hash만 기록한다. W9 직접 복원이나 journal 재생 가능성을 가정하지 않는다. Context 원문도 별도 local-only support다. Llama JV의 공개 context identity는 `bef722a3c30990f06084570a55a8e2056f5d38d7b4d5bd303e7d48de70920611`이다. 실제 원문은 봉인 파일/receipt로 검증해야 한다.

복원 순서:

1. 실제 추가 sealed W9/M9/z가 있으면 사용.
2. 검증 가능한 increment/history journal이 실제 있으면 사용.
3. 둘 다 없으면 W5/M5에서 B6–B9를 **한 번 bounded replay**하고 원본 중간 W/M/z hashes를 대조.
4. 다르면 `REPLAY_ANALOGUE`; W0→B9 전체 replay나 tolerance 완화로 자동 확대하지 않음.
5. 필요한 자산이 없으면 `HISTORICAL_EXACT_STATE_UNAVAILABLE`. S는 계속 진행한다.

W10을 W9 대신 쓰면 특히 M10에 B10 history가 이미 들어 있어 잘못된 비교가 된다. W5→W9의 과거 full-batch 시간 약1.75GPUh와 B10 target 약781초는 비용 참고치일 뿐 신규 ETA/승인 상한이 아니다.

### 3.2 정상 control과 새 development/audit

| Fixture | 역할 | 수행 범위 |
|---|---|---|
| D-B10 | Historical anomaly | Raw diagnosis + N0/NRMS actual pair + same-entry Official |
| C-B1 | 원 B1, W0/M0의 정상 cold negative control | Same-state raw response/normalization 중심; retrospective임을 표시 |
| C-B6 | 원 B6, sealed W5/M5의 정상 warm control | W5→W9 replay를 한다면 첫 batch capture를 재사용; B9 추가 복원 불필요 |
| S-DEV | 새 hash-selected B100, 두 모델 동일 IDs, cold W0/M0 | 9 configurations + Official common-entry reference |
| S-AUDIT | 새 미개봉 B100×3 stream, 두 모델 동일 IDs | 후보 lock 이후 별도 실행 단계; 이번 full grid 범위 밖 |

C-B1/C-B6가 source/state 부족으로 exact하지 않으면 analogue로 분리한다. 이 두 retrospective control을 독립 검증으로 세지 않는다. B1과 B6는 state와 request cohort가 함께 달라지므로 history의 단독 causal comparison도 아니다.

S-DEV는 기존1000·live reserved inventory·이전 pilot/dev와 겹치지 않게 immutable manifests로 선정한다. B100 특성을 유지한다. B1/B10의 batch coupling 결과를 single-request로 대체하지 않는다. Dataset 부족 시 성공/실패에 따라 replacement하지 않는다.

## 4. D의 핵심 측정

### 4.1 Source-exact arithmetic를 보존

Source는 target/terminal을 CPU FP32 `[D,B]`로 저장한다. Residual은 **FP32 subtraction**, scale norm도 FP32다. 그 후 FP64 weighting/whitening을 한다. Target과 terminal을 먼저 double로 바꿔 빼는 것은 별도 numerical observer이며 primary residual을 대체하면 안 된다.

Native writer는 전체 B100 raw residual·keys와 divisor1을 받는다. N0 inactive row도 native writer에서 제거하지 않는다. Controller active set만 source와 동일하게 고정한다. qref는 전체5 entry directions의 native action에서 한 번 계산하며 normalization과 무관하다. Node별 q_l은 current direction에 맞춰 다시 계산한다.

Raw response bundle은 `[m,D,B]`다. Production flatten은 `[D,B_active].reshape(-1)`이지 request-major flatten이 아니다. Per-request g/H의 합을 production global g/H와 직접 대조해야 한다.

### 4.2 Per-request geometry

Request i의 raw direction response를 P_i, normalization denominator를 d_i라 하면

$$
\Psi_i=P_i\operatorname{diag}(q^{-1/2})/d_i,\qquad e_i=R_i/d_i,
$$

$$
g_i=\Psi_i^\top e_i,\quad H_i=\Psi_i^\top\Psi_i,\quad
g=\sum_i g_i,\quad H=\sum_i H_i.
$$

Trace share, top1/top5, effective request count, signed g_i, spectrum/projection, native/Frobenius physical angle를 기록한다. 모든 row에 대해 observer-only `H−H_i, g−g_i` influence를 계산할 수 있다. B, target, dictionary, whitening, qref는 고정하고 결과 coefficient는 write에 사용하지 않는다. 특정 case만 제거하는 editor가 아니다. 큰 항 subtraction이 수치적으로 불안정하면 나머지 rows를 다시 합산하며 ridge/음수 고유값 clipping으로 통과시키지 않는다.

### 4.3 N0/NRMS는 무엇만 바꾸는가

N0는 기존 FrozenNormalization에 delegate한다. NRMS는 source-frozen FP32 scales를 FP64로 올린 뒤 active rows의 RMS를 계산한다. 동일 active mask, raw residual, native writer RHS, P/M/L2, z, qref를 유지한다.

V0는 실수 수식상 .5이며 source rounding 아래 약 .5다. Exact .5를 만들려고 N0를 다시 rescale하지 않는다. 기존 inactive-zero row를 이후 node에서 재활성화하지 않는다.

Actual N0와 NRMS는 각자의 changed state에서 dictionary/JVP를 다시 만든다. **Entry만 공통이지 4-node 전체의 D/c sequence를 공유하는 실험이 아니다.**

### 4.4 Actual comparison과 분모

D-B10에서 N0, NRMS 각각 λ.1/T2/N4를 실행한다. Same-entry O_NATIVE는 native one-pass feasibility control이며 원래 Official chain의 B10과 다른 실험이다. 세 branch는 동일 frozen z를 소비한다.

Entry를 한 번, endpoint별로 current RS100/PS200/NS1000과 requestwise NLL/margin/strict를 평가한다. Old B1–B9 rewrite900을 entry와 각 endpoint에서 평가하여 신규 손실·회복을 분리한다. Old PS1800/NS9000은 첫 진단의 필수 비용이 아니다.

기존 runner는 반환 시 entry로 복원한다. 따라서 current/old900/cross-score는 실제 materialized endpoint 활성 scope 또는 검증된 endpoint shadow에서 수행하고 평가 직전 W hash를 확인한다. 반환 후 그대로 평가해 entry를 final로 오인하지 않으며 평가 때문에 history를 다시 append하지 않는다.

N0/NRMS 두 objective 모두에서 모든 saved state를 cross-score하고 raw residual/semantic 결과도 함께 제시한다. 각 arm의 own-V 감소만 비교해 한쪽이 개선됐다고 하지 않는다. Case4228와 나머지99개를 보조 strata로 나누되 main100 분모는 유지한다.

## 5. S: 원인 진단과 독립인 실제 sweep

### 5.1 기본 grid

모든 행은 source N0, unchanged baseline target/P/L2/history, full5 dictionary, first-hit OFF다.

| ID | λ | T | N | h | 질문 |
|---|---:|---:|---:|---:|---|
| JV-BASE | .1 | 2 | 4 | .5 | 기준 |
| JV-LAM-001 | .01 | 2 | 4 | .5 | λ |
| JV-LAM-00316 | .0316227766 | 2 | 4 | .5 | λ |
| JV-LAM-0316 | .3162277660 | 2 | 4 | .5 | λ |
| JV-LAM-1 | 1 | 2 | 4 | .5 | λ |
| JV-RES-N2 | .1 | 2 | 2 | 1 | Fixed-T resolution |
| JV-RES-N8 | .1 | 2 | 8 | .25 | Fixed-T resolution |
| JV-HOR-T1 | .1 | 1 | 2 | .5 | Fixed-h exposure |
| JV-HOR-T4 | .1 | 4 | 8 | .5 | Fixed-h exposure |

원본 정밀값은 [candidate-grid.csv](candidate-grid.csv)에 보존한다. JV-RES-N16도 후보 pool에 보존하되 conditional이다. 두 모델의 baseline compute-z 설정은 원래부터 다르므로 통일하지 않는다.

### 5.2 비용 최소화

λ.1의 h.5/T4 path가 T1/T2 prefix까지 제공한다. 여기에 N2/T2 path2nodes, N8/T2 path8nodes, 나머지4λ의 각4nodes를 더하면 **7 paths,34 nodes,최대170 main target JVP/model/fixture**다. 두 모델 합68nodes/340JVP다. 별도 Official one-pass와 entry reference solves/compute-z/evaluator/FD는 별도 계측한다.

각 모델·fixture에서 compute-z는 한 번 공유한다. Prefix에서는 evaluation-only finalization을 사용하고 history를 append하거나 continuation state/RNG를 바꾸지 않는다. 다른 resolution의 결과를 prefix로 대신하지 않는다. 전체 sequential chain에서 target 공유나 prefix 재활용을 확장 적용하지 않는다.

Prefix는 effective T/N과 parent T/N을 따로 기록한다. Current batch history가 append되지 않은 derived prefix를 sequential-resume W/M checkpoint로 간주하지 않는다.

D에서 NRMS가 좋아져도 진행 중인 S의 normalization을 바꾸지 않는다. 필요하면 S 결과를 닫은 뒤 선택된 λ/T/N에서 N0 대 NRMS의 제한된 교차 확인을 **별도 arm**으로 만든다. 최초부터 normalization×λ×T×N 전체 grid를 만들지 않는다.

### 5.3 해석

- λ의 actual 경로/성능 변화는 정상 operating point의 효과다. B10 root-cause 증명이 아니다.
- Fixed-T에서 endpoint 거리와 실제 dissipation이 안정되는지 본다. 더 작은 h에서 점수가 떨어져도 수치 수렴 실패라고 단정하지 않는다.
- Fixed-h에서 성능이 좋아지면 더 긴 exposure의 효과다. 동일 계산량 개선이 아니다.
- First-hit은 training predicate의 관측 시간만 기록한다. 그 prefix의 NS를 보고 후보를 조기 종료하지 않는다.
- L8 share만 감소하면 개선이 아니다. 절대 action/current progress/RS·PS·NS/NLL tail을 함께 본다.
- 단일 새 cold B100의 sweep은 정상-state calibration이다. Warm/sequential 안정성은 선택 후 별도 audit에서 확인한다.

## 6. 시행착오를 줄이는 구현 경계

새 `alpha_jv_llama_diagnosis/` package에서 fixture binder, normalization view, raw capture, requestwise analyzer, configurable trajectory adapter만 만든다. Native writer, compute-z, projector, core NNLS/JVP를 재작성하지 않는다.

필수 방어점은 다음과 같다.

1. W/M를 restore한 **뒤** Family를 생성해야 family.w0가 올바른 entry다.
2. Saved z를 attach할 때 values만 덮어쓰지 말고 FixedZArtifact와 semantic inventory/hash도 복원한다.
3. Warm state에 cold `prepare_method_state()`를 호출하지 않는다.
4. Config λ를 trajectory·telemetry·history shadow 모두에 전달한다. Source의 hardcoded N0 label도 config-derived ID로 바꾼다.
5. Physical coefficient는 `h*c_l/sqrt(q_l)`다. h를 NNLS objective 안에 넣지 않는다.
6. Primary arithmetic/iteration order를 바꾸지 않는다. FP64 diagnostic subtraction은 별도 컬럼이다.
7. Finite zero-field/near-stall을 technical error로 덮거나 successful endpoint로 대체하지 않는다.
8. 두 branch의 finalization/history policy를 같게 하고, independent experiment reset과 adaptive rollback을 구분한다. 후자는 OFF다.

CPU tests는 위 계약을 확인하는 작은 수학·shape·identity 검사로 제한한다. GPU는 정상 신호 fixture에서 기존 FD/parity 기준을 재사용한다. 진단 대상의 tiny response가 FD envelope 이하라는 이유로 값을 버리거나 false PASS하지 않는다. `NUMERICALLY_UNRESOLVED`로 기록하고 해당 원인 claim을 제한한다.

## 7. 원인을 어느 수준까지 확인했다고 말할 것인가

| 수준 | 필요한 근거 | 허용 결론 |
|---|---|---|
| OBSERVED | 기존 sealed trace | Tiny scale, near-zero physical write, B10 실패가 함께 존재 |
| SAME_STATE_ATTRIBUTION | Exact raw requestwise H/g와 all-row influence | 특정 row/weighting이 현재 controller 해에 얼마나 영향을 주는가 |
| EXACT_STATE_INTERVENTION | Source-exact N0 재현과 동일 W/M/z의 NRMS/Official 비교 | 해당 historical state에서 weighting 개입이 write/endpoint를 바꿨음 |
| ANALOGUE_MECHANISM | 다른 state에서 같은 연결 관측 | 메커니즘의 가능성; 원래 B10 원인 확정은 아님 |
| DEVELOPMENT_BENEFIT | 새 outcome-independent S fixture | 특정 λ/T/N의 유용한 operating point 후보 |
| INDEPENDENT_AUDIT | 미개봉 stream에서 locked candidate | 일반적 안정성/성능에 대한 추가 근거 |

“Normalization이 문제”와 “floating-point bug”는 다른 결론이다. “Sweep로 좋아졌다”와 “root cause를 고쳤다”도 다르다. Barrier algebra가 맞다는 것과 locality/retention을 보호했다는 것은 여전히 구분한다.

## 8. 범위와 전달 원칙

현재 지시문은 D와 S의 소형 GPU phase까지 실행 가능한 명세를 제공한다. 실제 host/GPU/hour cap은 GH/SH의 승인된 자원 배정에 결속한다. 미배정 상한을 임의 숫자로 채우지 않는다. Server4 완료를 자원 승인으로 해석하지 않는다.

추가 historical-output controller, dynamic-z, full lifelong, final method promotion은 이번 진단·sweep에 섞지 않는다. 원인/운영점 근거가 확보된 뒤 상위 연구 계획의 별도 단계로 넘어간다.
