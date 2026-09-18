# Adaptive write의 결정 대상과 누적 capacity 해석

2026-09-16. 사용자의 추가 질문에 대한 결정 문서다. [W0 일곱-arm 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-local-z-adaptive-allocation-design-v1.md)와 [누적 보존 진단](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-16-local-z-adaptive-allocation-budget-diagnostics-v1.md)을 구체화한다. 기존 파일의 동시 변경을 덮어쓰지 않는다. 새 chain, hard budget, 모델 실행을 추가한 문서가 아니다.

## 이번 방법의 identity

주방법은 **L4 native local-z를 기본으로, batch마다 L4 write 강도와 추가 L8 사용 여부·강도를 함께 선택하는 정책**이다. 다층 분산을 성공 조건으로 삼지 않는다. L4-first는 기본 선택을 뜻하며, 먼저 L4를 실제 commit한 뒤 L8을 결정한다는 뜻이 아니다. 약한 L4만으로 품질이 부족해도 L8이 보완하는 조합을 최종 commit 전에 함께 평가한다.

| 결정 축 | 첫 실험의 선택 | 포함하지 않는 주장 |
|---|---|---|
| Target 위치 | write할 layer의 local-z | local-z 자체가 locality를 보장 |
| 동적 단위 | batch | 요청별 최적 layer를 이미 식별 |
| L4 강도 | a4∈{.75,1} | 연속 강도의 전역 최적화 |
| L8 사용 | a8∈{0,.5,1} | 항상 둘 이상의 layer에 분산 |
| 단일층 선택 | 이번 추가 write를 L4에만 수행 가능 | L4/L5/L8 사이의 자유로운 singleton routing |
| Weight 결정 | Current/Past 조건 아래 실제 S64 출력 위험 최소 | raw z 또는 causal score만으로 손상 예측 |
| Learned network | 첫 실험에는 없음 | dynamic과 learned가 동의어 |

L4-only의 높은 RS/PS와 BLUE의 약한 locality 이득은 이 작은 action space를 먼저 볼 근거다. 현재 menu는 a4=0을 포함하지 않으므로 L8-only routing은 검증하지 않는다. L4와 L8의 평균 singleton 순위로 요청별 최적 layer가 항상 L4라고 결론내리지도 않는다. Request별 singleton routing은 다른 후속 가설이다.

여러 요청을 각각 한 layer로 routing하더라도 batch 전체의 수정 layer 수는 여러 개일 수 있다. 공동 solve와 동일 모델의 forward 때문에 요청 간 간섭도 남는다. '요청별 한 layer'와 '모델 전체 lifelong 한 layer'를 구분한다.

## Layer 수와 손상 사이에 단조 관계를 가정하지 않는다

같은 entry에서 보존 출력의 일차 반응을 u4=J4*d4, u8=J8*d8이라 하면

    ||u4+u8||² = ||u4||² + ||u8||² + 2<u4,u8>.

마지막 항이 양수면 손상이 강화되고 음수면 일부 상쇄된다. 이는 작은 변화의 설명이며 유한 write의 실제 결과는 nonlinear forward로 확인한다. 같은 제약·목적에서 d8=0을 허용한 전체 공간은 단층을 포함하지만, 현재 native-direction menu나 C4 proxy가 전체 공간의 최적값을 찾는 것은 아니다.

현재 증거는 '검사한 BLUE 정책보다 L4-only의 NS가 높았다'와 '추가 layer가 순 preservation 이득으로 이어지지 않았다'를 지지한다. 'Layer를 더 건드렸기 때문에 반드시 실패했다'는 원인 확정은 아니다. Norm share와 같은-state stage output을 분리한다.

## 이번 layer support와 lifelong layer support

S_step(t)={l: 이번 actual d_l,t≠0}, S_life(t)={l: W_l,t≠W_l,0}를 각각 기록한다. Gate 값 대신 actual stored-weight 차이도 함께 기록한다. 서로 상쇄되어 net가0인 경우를 위해 ever_written도 따로 둔다.

한번 L8을 수정한 뒤 a8=0을 선택해도 기존 A8=W8,t−W8,0는 남는다. 이는 L8을 원모델로 복구한 것이 아니다. 과거 L8 수정이 있는 상태에서 새로운 L4 write는 L8 input key를 바꿀 수 있으며, 과거 A8과 새 key 변화가 상호작용한다. 따라서 다음 두 비교를 모두 유지한다.

- 같은 LD entry에서 이번 L8 write를 추가한 이득: 현재 action의 조건부 효과.
- W0에서 출발한 LD와 L4D의 최종 차이: 누적 layer 사용 정책의 효과.

기록할 compact 상태는 first_nonzero_L8_batch, L8_write_batch_count, current/cumulative_L8_norm, ever_written_L8, zero_a8_after_L8_used_count다. 기존 weight delta·hash로 산출하며 별도 model forward가 필요 없다.

이번 selector는 one-step risk를 최소화하는 greedy 정책이다. 최초 L8 사용이 향후 editing geometry에 주는 비용까지 최적화한 long-horizon controller라고 부르지 않는다. 과거 N 및 L8 activation 이후의 drift를 관측하여 이 한계를 평가한다. Future requests를 몰래 lookahead하거나 임의의 L8 사용 penalty를 추가하지 않는다.

## Gate를 줄여도 history quota가 비워지지는 않는다

K/P/M이 고정된 한 fitting에서 scalar gate a에 대해 실수 연산상 d(a)=a*d(1)이다. 따라서 고정 key mapping 비용과 tr(d M d^T)는 a²로 변한다. 실제 FP32 endpoint에서는 scaling 오차를 별도로 남긴다.

하지만 native history는

    M_next = M_entry + K_final K_final^T

로 갱신되며 gate 계수를 곱하지 않는다. a=.75는 이번 변화량 축소이지 history capacity를25% 남긴다는 뜻이 아니다. 이 갱신식과 projector의 실용 근사 범위는 [AlphaEdit §3](https://arxiv.org/html/2410.02355v4)을 참조한다.

L4/L8 down-projection만 바꾸고 token/context를 고정하면 K4는 구조적으로 불변이다. 같은 request 순서와 전체1회 append 규칙 아래 M4는 local/terminal/gate 정책과 무관하게 같다. 실제 수치 hash 차이가 있으면 backend/context/cache 불일치를 먼저 조사한다. 따라서 M4 trace/rank로 이 정책들 사이의 preservation 차이를 설명할 수는 없다.

K8는 L4 경로에 따라 달라지므로 M8의 trajectory는 달라질 수 있다. 같은 partial-L4 상태에서는 현재 a8 자체가 K8을 바꾸지 않는다. Gate를 history에 곱하는 변형은 보호 대상을 바꾸는 별도 방법이므로 첫 실험에 섞지 않는다.

## Budget을 측정하는 방식

미지의 고정 'L4 capacity100, L8 capacity100'을 가정하지 않는다. 다음을 분리한다.

| 양 | 계측 | 해석 |
|---|---|---|
| 실제 write 사용량 | step norm, cumulative net norm, path length | parameter 변화; knowledge 개수 아님 |
| 누적 기능적 손상 | 고정 W0-reference KL, held-out NS/strict/margin 변화 | 관측 분포에서의 손상; layer별로 자동 가산 안됨 |
| history 제약 기하 | M/P, key overlap, realization error | 현재 writer의 반응; 잔여 capacity % 아님 |
| 현재 개선 여지 | 동일 품질에서 가능한 후보들의 최소 위험 | 유한 candidate/reference에 조건부 |
| 계산 예산 | target calls, solves, candidate F/B, GPU time | online 실용 비용; intrinsic capacity와 별개 |

허용 손상 epsilon을 명시하면 epsilon−B(W_t)를 해당 제약의 slack으로 계산할 수 있다. 그러나 epsilon 자체는 데이터에서 발견되는 모델 고유 상수가 아니다. 현재는 상한을 임의로 정하지 않고 여러 품질 수준에서의 위험 곡선과 손실 전이를 보고한다.

Layer별 projected-history 유효 rank나 작은 eigenvalue 수는 보조 geometry다. 서로 다른 layer의 key scale·sensitivity·target이 다르므로 rank 비율 또는 norm 비율을 storage share로 읽지 않는다. 1,000개 이후 지표로10k까지의 여력을 외삽하지 않는다.

## 기존 후보로 계산할 두 가지 개선 여지

같은 LD batch entry에서 공통 Current/Past 조건을 만족하는 C4는 a8=0인 후보, C48은 전체6후보다. B는 S64의 고정 W0 KL이다.

    F4(t)  = min feasible C4 B(candidate)
    F48(t) = min feasible C48 B(candidate)
    H_strength(t) = B(N4)-F4(t)
    H_extra8(t)   = F4(t)-F48(t)

둘 다 후보 포함관계상 비음수이며 실제 선택은 기존 numerical tie를 따른다. 이는 '현재 후보에서 줄일 수 있었던 위험'으로 표기하고 미래 edit 수를 뜻하는 remaining capacity로 부르지 않는다. Request 난도가 batch마다 달라지므로 H의 시간 추세만으로 capacity 소진을 식별하지 않는다.

H_strength만 양수면 L4 강도 조절의 여지가 있다. H_extra8도 양수면 같은 entry에서 추가 layer 후보의 여지가 있다. 둘 다0이면 이 유한 menu에서 더 좋은 후보를 찾지 못한 것이다. 해당 proxy에서 최소값이 개선되는 것은 선택 과정의 성질이므로 Dev128/공식 N/과거 N으로 전이돼야 과학적 가치가 있다.

## 층별 signed 출력 비용 ledger

이번 선택된 a4 적용 뒤를 W_mid, 최종을 W_next라 하면

    b4,t = B(W_mid)-B(W_entry)
    b8,t = B(W_next)-B(W_mid)
    b4,t+b8,t = B(W_next)-B(W_entry).

연속된 동일 branch에 대해 sum_t(b4,t+b8,t)=B(W_T)-B(W0)가 성립한다. LD/TD의 partial state는 이미 a8=0 후보로 평가되므로 새로운 후보 생성이 필요 없다. Entry 값은 직전 selected endpoint 값을 재사용할 수 있다. TD가 common N4를 선택하면 W_mid=W_next=N4, b8,t=0이다.

b8,t<0은 그 실제 순서에서 L8 write가 앞선 reference 손상을 줄였다는 뜻이다. 이 분해는 순서와 reference에 의존하며 layer의 고유 손상 비율 또는 순서독립 인과 attribution이 아니다. L4 write가 과거 L8 edit와 상호작용한 효과도 b4에 포함된다.

이 ledger는 기존 누적 mapping/출력 진단을 보완한다. 기정70 editing batches와 후보 menu/selector를 변경하지 않는다. Signed cost를 layer별 hard cap으로 사용하지 않는다. S64 외의 official N stage attribution에는 별도 stage forward가 필요하므로 이미 있는 observer 범위 밖 데이터를 측정된 것으로 보고하지 않는다.

## Learned weights의 구체 후속 형태

Dynamic은 현재 상태에 따라 선택한다는 뜻이며 반드시 neural network 학습을 뜻하지 않는다. 첫 selector는 온라인에서 선언한 후보를 모두 평가한다. L8 write를 선택하지 않아도 해당 batch의 L8 후보 계산 비용은 이미 지불했다. 이를 조건부 target 계산 절감으로 보고하지 않는다.

온라인 selector가 독립 평가에서 유효하면 후속 predictor가 이산 action을 예측하도록 학습할 수 있다. Action은 (.75,0), (1,0), (.75,.5), (.75,1), (1,.5), (1,1)이며 local-z 생성 규칙은 유지한다.

    pi_theta(action | phi_request_batch, phi_history, phi_base_state)

여기서 softmax는 **action 중 하나를 고르는 확률**로 사용하고 모든 layer의 write를 양수로 혼합하지 않는다. 독립 sigmoid+명시적 on/off mask도 가능하나 첫 실험에서 추가하지 않는다. Raw z4 이외에 current loss, normalized residual, key interference, history overlap, 이전까지의 base drift를 후보 입력으로 검토한다. 정확한 predictor와 학습 objective는 현재 미결정 후속이며 최초 adaptive routing이라고 주장하지 않는다. [HiEdit §4](https://arxiv.org/html/2604.11214v1#S4).

예측 비용을 줄이려면 배포 시 모든 L8 candidate의 fresh z와 실제 위험을 먼저 계산하는 feature를 요구해서는 안 된다. 학습 label 생성 비용과 inference feature 비용을 구분한다. Training stream에서 selector 결과를 supervision으로 만들고 별도 stream에서 candidate-screen 대비 성능/비용을 측정한다. Official evaluation N을 online label로 사용하지 않는다.

## 이번 실험에서 내릴 결론

- L4D가 N4보다 좋고 LD의 추가 이득이 없으면: 동적 L4 강도 조절을 지지한다. 다층 분담 claim은 유지하지 않는다.
- LD가 L4D 및 REFIT4보다 같은 품질에서 독립 N 보존을 개선하면: 조건부 추가 layer의 실효성을 지지한다.
- 후보에는 N을 개선하는 다층 write가 있는데 selector가 선택하지 못하면: 후보 공간과 risk proxy 실패를 구분한다.
- S64 개선만 있고 Dev/N 개선이 없으면: preservation proxy 전이를 먼저 다룬다.
- 후반에 위험이 늘어도: current request 난도·proxy coverage·writer realization·과거 key drift를 구분하며 intrinsic capacity 고갈로 바로 결론내리지 않는다.

기존처럼 모든 arm은 동일 W0/M0=0에서 시작한다. 새 실험 결과나 layer별 수치 capacity는 아직 측정되지 않았다.
