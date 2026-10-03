# 필수 교정과 반영한 결정

2026-10-03, revision 3. **T′, native 그룹 가중 actual norm, active clamp 없음**을 유지한다. 이번 사용자 지시에 따라 **B1 강도 보정과 추가6 fit을 철회하고 λ_n=.5를 고정**했다. 주 비교는 native baseline이며 v9는 내부 참고 결과다. Production 구현·GPU 실험은 아직 시작하지 않았다.

## 반드시 고칠 항목

| 항목 | v9/제안의 문제 | 교정 |
|---|---|---|
| 변수의 의미 | 직접 주입 D와 writer 실현량을 같은 δ로 부름 | Writer 계수 R와 local 변화 v=Uk를 분리 |
| 편집 이익의 대상 | Fit은 R 전체 효과, 비용은 축소된 writer 효과 | Actual subject key에서 구한 v^a를 fit에 주입; 남는 기저 차이 명시 |
| Key 상태 | Subject-only key를 actual key라고 해석 | K_fit과 K_actual 구분; 실제 key 기준이면 매 후보 actual builder 유지 |
| Commit | Terminal만 재작성하면서 학습한 writer와 같다고 간주 | Terminal actual 평가한 materialized weights를 그대로 commit |
| 배분 비용 | 새 실현 목적에도 R 추종 E를 자동 유지 | G-only로 교정; root A/B 결합 유지 |
| Physical pulse | Detached teacher 추종을 완전한 gap 최소화라고 해석 | 학습에서 제거하고 actual 관측으로 분리 |
| Norm/clamp | R 크기 제한을 실제 subject 변화 제한으로 간주 | Actual v의 rewrite 그룹 가중 norm, active clamp 없음 |
| KL 주입 | 기존 actual builder는 rewrite rows만 처리 | KL rows도 actual 하층 경로로 전달; ridge mean에는 제외 |
| 유효 강도 | 동일 계수면 이전과 동일한 규제라고 간주 | Native 계수 .5 고정, 실현량/Q는 관측; 유효 강도 일치 주장 없음 |
| Native 가중치 | Key 그룹 평균과 NLL 평균을 혼동 | Norm/key는 그룹 평균, NLL은 baseline의 context 균등 평균 |
| 요청 간 결합 | Microbatch별 key solve 또는 P detach | 전체 logical B key와 full K/P gradient 유지 |
| 상태/history | Subject-fit key로 native actual history라고 주장 | 최종 all-token history key를 별도 수집하고 정확히 한 번 admission |
| 주장 | Gap 제거·자동 배분 이동·속도 향상 단정 | 동일입력 operator 일치와 경로 일치 구별; 성능/속도는 미검증 |

사용자의 기존 요구인 전체 eligible layer, native 문장, current∥entry KL, replay 없음, 두 allocation arm, warm-up 없음은 유지한다. 강제 균등·사전 layer 선택·평가 데이터 학습은 추가하지 않는다.

## 반영한 선택

### D1. Key 기준과 학습 경로 — T′로 구체화

| 선택 | 정확히 보존하는 것 | 남는 차이/비용 |
|---|---|---|
| S: subject-only K로 U 생성, 그 U 고정 commit | Fit에서 정의한 writer와 배포 writer 동일 | Native actual-key 상태와 다름; all-token 배포 gap |
| T: actual K로 U 생성, fit에는 Uk^s 주입 | Native key 상태와 ridge 식, terminal 동일-U commit | Subject local 변화 및 기저 차이 |
| **T′: actual K로 U 생성, fit에는 Uk^a 주입** | **위 조건 + 실제 context의 local 변화까지 fit과 공유** | **기저 차이, 두 경로 비용, 변경된 gradient** |

**T′를 채택한다.** 매 후보마다 actual 상층 key를 갱신하고, builder의 actual context별 v^a를 fit subject 위치로 전달한다. V/actual key를 detach하지 않는다. 마지막에 평가한 U를 그대로 commit한다. 작은 k 차이를 이유로 S/T로 자동 전환하지 않는다.

Actual all-token NLL까지 직접 최적화하면 task forward도 배포와 같아지지만, subject-only 학습을 유지하려는 요구와 다른 선택이다. 이번 교정의 기본안으로 자동 추가하지 않는다. 'S fit 후 마지막에만 actual K로 재작성'은 T와 구별한다.

### D2. Norm — actual v의 native 그룹 가중 평균으로 확정

Rewrite context별 ||v^a_lrc||/a_lr²를 key 평균의 w_rc=1/(J n_g)로 가중한다. 현재 두 그룹은 canonical .5, 다섯 prefix 각각.1이다. **NLL은 이 가중치로 바꾸지 않는다.** 현재 NLL은 각 context1/6이며 prefix 다섯 개의 합은5/6이다. 따라서 리뷰의 'prefix들이 NLL 절반'이라는 설명은 맞지 않는다.

Fixed entry canonical a² 분모와 비제곱 norm을 유지한다. R norm, inherited hidden norm, norm-of-mean과 구분한다. KL rows의 v는 별도 관측하며 rewrite norm 평균에 포함하지 않는다.

### D3. Clamp — active clamp 없이 관측으로 확정

R 또는 v에 active cap을 걸지 않는다. 후보별·terminal의 v/a 분포와 .75 초과 비율을 rewrite/KL/layer/context별 분모와 함께 기록한다. 임계 초과를 clip·조기 중단·후보/층 배제의 gate로 쓰지 않는다. 실제 기술 실패는 별도 처리한다.

### D4. 계수 — B1 보정 철회, native .5 고정

[Baseline 비교 계약](comparison-ko.md)에 따라 **λ_n=.5를 A/B와 모든 batch에 고정**한다. 계수 grid·선택 함수·matching 라벨·추가6 B1 fit은 제거한다. PS/NS 또는 실현량을 보고 계수를 고르지 않는다. λ_alloc=.1과 나머지 확정 profile은 유지한다. B1부터 일반 sequential 편집을 진행하도록 설계한다.

V9의 S_mean≈.185는 데이터에서 나온 값이며 새 method의 목표 강도가 아니다. [직접 추출 기록](math/v9-b1-strength-reference.json)은 historical diagnostic으로만 남긴다. 해당 값에 맞추기 위한 재실행이나 calibration 의존성을 두지 않는다. 주 비교 대상은 BLUE·MEMIT-H·AlphaEdit다.

고정 .5는 native 계수 유지의 선택이며 다층 공동 목적의 유효 규제까지 같다는 뜻은 아니다. V9도 virtual에 직접 주입한 D에 norm을 걸었으므로 native 형태 자체가 무관했던 것은 아니다. V10에서 개선하는 것은 실제 writer 변화 v^a와 fit/norm의 대상 일치다. Model·문장·평가기·runtime 차이까지 확인해야 baseline 비교 조건을 설명할 수 있다.

현재 추가로 사용자 결정을 기다리는 method 항목은 없다. 실행 host·job 자원·production source lock은 구현/전달 단계에서 확정할 사항이다. 이 문서 갱신 자체로 실험을 시작하지 않는다.

## 전달받은 리뷰에서 추가로 고친 주장

1. **Ridge와 exact:** 같은 full Y/A/K이면 U까지 동일하다. 'cross-talk만 다르게 남는다'는 설명은 틀리다. 같은 R probe와 같은 Y 비교를 구분한다.
2. **κ 역산:** M 대각의 odds는 batch 조건부 용량이다. Layer 평균에서 얻은 scalar로 full-matrix 배분 비용을 설명할 수 없다.
3. **1.7배 비용:** 해당 값은 old G+E의 실현량당 root 비용이다. 새 G-only root의 scalar 예시는1.43배다.
4. **Adam 인과:** q-gradient norm이 비슷하다는 사실만으로 이후 균등 배분을 전적으로 objective 원인이라고 확정할 수 없다. 성분별 방향·optimizer 상태·초기 경로 영향이 남는다.
5. **상속 gap:** S 내부의 local 주입과 writer 작용은 일치하지만 all-token commit의 상층 상태는 달라질 수 있다. T에서도 masked fit/actual gap은 남는다.
6. **예측:** 비슷한 용량이 균등 배분을 강제하지 않으며 H가 늘면 반드시 L8 share가 감소하는 것도 아니다. Share 모양을 성공 gate로 쓰지 않는다.
7. **비용:** One logical forward는 물리 forward 1회나 wall-clock 절감을 뜻하지 않는다. T는 두 경로를 유지한다.

## 이번 추가 리뷰에서 조건을 바로잡은 부분

| 리뷰 주장 | 반영 판단 |
|---|---|
| S와 T의 gap 크기가 같다 | Local gap 식의 형태가 같을 뿐 U/key/state가 달라 크기는 같다고 할 수 없음 |
| T만 Q가 배포 U의 비용이다 | 같은 U를 commit하는 S도 성립; T의 차이는 actual-key 기준 |
| T′가 gap을 없앤다 | 같은 층 local 변화는 공유하지만 기저 gap·최종 출력 차이는 남음; 전체 norm 감소 보장 없음 |
| Key 상대 오차가 작으면 세 방법이 같다 | U의 작용, 기저 gap, task sensitivity가 필요; key norm만으로 동등성 선언 불가 |
| 편집량이 최대1.67배 커진다 | Scalar 동일 실현량의 벌점 비율이지 optimizer 결과의 배율/상한이 아님 |
| PS/NS를 안 쓰면 튜닝이 아니다 | 해당 절차도 calibration이었으나 revision3에서 사용자 지시로 철회 |
| 추가 계산 거의 없다 | 기존 builder에 KL rows 및 v adjoint가 추가됨; wall-clock 미측정 |
| NLL도 .5/.1 가중이다 | Frozen source는1/6 context 평균; 새 norm과 구분하여 baseline 유지 |

## 구현 검증 및 기록

T′에서는 subject 행을 W_eff로 교체하는 기존 T operator 대신 actual local 변화 추출·이식 연산을 검증한다. Whole-B graph의 v 경계에서 adjoint를 모아 원래 builder에 한 번 전파하며 U/k 경계와 중복 seed하지 않는다. Rewrite와 KL 양쪽 gradient를 포함한다.

[CPU 검산](math/tprime-adjoint-results.json)은 두 층·B3·rewrite6/KL1 context의 작은 비선형 causal 모델에서 수행했다. Dense와 microbatch adjoint의 최대 차이는4.34e−19, K/P를 포함한 성분별/전체 유한차분 최대 차이는3.90e−11 수준이다. Geometry gradient를 끊은 negative control은 오차를 검출했다. 이 결과는 production adapter나 GPU 구현이 검증됐다는 뜻이 아니다.

실험 실행 전에 production의 이식 operator/gradient parity, whole-B 및 microbatch 분할 불변성, K/P gradient, materialization/commit 일치를 확인한다. 성분별 NLL/KL/norm/allocation의 R·q gradient는 매 batch 후보2/9/25에 기록하고 weighted sum/방향도 확인한다. 이는 과학적 배분 결과를 제한하는 gate와 구별한다. 새로운 실제 모델 진단 실험을 시작한 것은 아니다.
