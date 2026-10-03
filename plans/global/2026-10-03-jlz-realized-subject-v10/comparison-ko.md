# Native 기본값으로 수행하는 baseline 비교

2026-10-03, v10 T′ revision 3. **사용자 지시로 B1 강도 보정과 추가6 fit을 철회한다. A/B 모두 λ_n=.5를 고정한다.** 본 문서는 이전 calibration 계약을 대체한다. 문서 갱신 자체로 실험을 시작한 것은 아니다.

## 1. 비교 목적

주 비교 대상은 **BLUE·MEMIT-H·AlphaEdit**다. 각 baseline은 해당 model/benchmark의 확인된 native 설정을 사용하고, v10은 현재 native norm 계수 .5를 유지한다. V9는 내부 method 변화의 참고 결과다. V9의 실현량을 맞추거나 v9 대비 순수 구조 효과를 분리하는 것을 실행 조건으로 두지 않는다.

V10이 목표로 하는 것은 실제로 주입하는 local 변화에 native 형태의 norm을 부과하면서, 모든 eligible layer를 공동 최적화하고 writer 비용으로 동적 배분하는 것이다. V9에서 관측한 S_mean≈.185가 최적 강도라는 근거는 없으므로 새 method의 목표값으로 채택하지 않는다.

V9도 virtual fit에 주입한 D에 norm을 걸었으므로 그 경로에서는 native 형태를 사용했다. V10의 개선은 actual writer의 변화 v^a를 fit과 norm에 함께 사용하는 데 있다. 다층 norm 합, context 가중치, anchor, allocation 비용 및 optimizer 경로가 달라 **동일 .5가 동일한 유효 규제·편집 강도 또는 완전히 동일한 실험 조건을 보장하지 않는다.**

## 2. 현재 실행 profile

| 항목 | 고정값/정책 |
|---|---|
| Main arm | A/B 두 arm |
| 편집 범위 | 기존 순서500 edits, B100×5 |
| Native norm λ_n | **.5, A/B 공통, 모든 batch 고정** |
| Native KL λ_K | .0625, current∥own batch-entry |
| Allocation λ_alloc | .1, 두 arm 결합 방식 유지 |
| Optimizer | Adam η=.1, 25 candidates/24 updates, warm-up0 |
| Clamp | Active clamp 없음; .75 초과는 관측만 수행 |
| B1 | 일반적인 첫 sequential batch |
| 계수 탐색·실현량 matching | 없음 |
| 추가 B1 calibration fit | **0회** |

B·층 수·차원·context 수는 method의 상수가 아니다. 다른 profile은 해당 native adapter의 입력·reduction·계수 출처를 실행 전에 고정한다. 이번 .5를 모든 model/benchmark의 native 기본값이라고 일반화하지 않는다. 관측된 실현량·PS/NS로 계수를 재선택하지 않는다.

필요한 구현 qualification은 유지한다. 두 arm은 독립 W0/H0에서 시작하고, 각 candidate는 자기 batch-entry 상태를 기준으로 구성한다. Final evaluated U를 그대로 commit하고 실제 rewrite mean key history를 정확히 한 번 admission한다. 본선 시작 전에 B1에서 여러 계수를 fit하는 절차나 미선택 trial의 상태 관리는 없다.

## 3. Baseline 비교 조건

1. Model/checkpoint/tokenizer, benchmark split, 편집 요청과 순서, 초기 W0/H0, batch 일정 및 누적/현재 cohort의 정의를 대조한다.
2. Native 문장·lookup·target token·NLL readout 및 KL 방향을 각 method의 source/profile과 연결해 기록한다. Baseline의 계수를 v10에 맞추려고 수정하지 않는다.
3. R/P/N 정의, evaluator, evaluation IDs와 분모를 맞춘다. 현재500-edit 누적 평가의 분모500/1,000/5,000은 해당 CounterFact profile에 한정하며 다른 benchmark에서 복사하지 않는다.
4. BLUE는 실제 구현·writer·write layer 구성을 명시한다. 기존 AlphaEdit-BLUE L4+L8 artifact를 다른 BLUE 구성의 결과로 표기하지 않는다.
5. 기존 완료 baseline 결과를 사용할 때에는 일치한 조건과 다른 runtime/dtype/evaluator를 명시한다. 환경이 다른 결과를 완전한 동일 조건 또는 직접 속도 배율 비교로 해석하지 않는다.

먼저 이미 있는 baseline 산출물을 위 기준으로 비교한다. 이 수정은 baseline 전체 재실행, 새 계수 sweep 또는 PS–NS frontier 실험을 추가하는 지시가 아니다. 현재 새 main 실행 범위는 v10 A/B 두 arm이다.

## 4. 관측은 유지하되 목표로 삼지 않는다

Native 손실, actual NLL/KL, R/P/N과 함께 다음을 기록한다.

\[
S_{mean}=\frac1{Bm}\sum_{l,r}\frac{\|(U_l^{64}K_l^a)_r\|_2}{a_{lr}},\qquad
S_{ctx}=\frac1{Bm}\sum_{l,r,c}w_{rc}\frac{\|v^{a,eff}_{lrc}\|_2}{a_{lr}}.
\]

S_mean은 ideal FP64 mean-key 직접 변화, S_ctx는 context별 effective norm의 가중 평균이다. 두 값은 같지 않으며 전체 hidden 변화도 아니다. Layer/context별 실현 share, Q, v/a 분포와 .75 초과 비율, key/base gap, 성분별 R/q gradient도 유지한다.

이 값들을 λ 선택·후보 채택·layer 제외·중단의 gate로 사용하지 않는다. V9 B1의 [추출 기록](math/v9-b1-strength-reference.json)과 [재현 코드](math/extract_strength_reference.py)는 historical diagnostic으로만 보존한다. 해당 파일이 존재한다는 이유로 matching 단계를 실행하거나 다른 profile의 v9 기준을 추가 생성하지 않는다.

Scalar 동일 실현량에서의1/γ 및1/sqrt(γ) 벌점 비율은 method 해석을 위한 식이다. 이를 계수 보정 식이나 최종 편집량의 증가 상한으로 사용하지 않는다. 실제 편집·일반화·locality와 발생한 실현량을 함께 보고한다.

## 5. 비용 보고

철회한6회 calibration fit은 실행 계획과 예상 비용에서 제외한다. Main fit, 구현 qualification, 성분별 gradient 관측, actual endpoint 평가 비용을 구분한다. 논리적 candidate 수와 실제 forward/backward 호출·token 수·최대 메모리·GPU 시간을 기록한다. 추가 진단 비용을 main-only 시간에서 누락해 baseline보다 빠르다고 주장하지 않는다.
