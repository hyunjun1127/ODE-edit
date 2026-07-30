# ODE-Edit 관련 연구와 novelty 경계

- 작성일: 2026-07-30
- 문서 성격: Motivation Validation을 위한 proposal-side related-work audit
- claim 상태: `hypothesis only`
- 조사 원칙: 논문 본문, 학회 페이지, 저자 공개 저장소 등 primary source를
  우선하며, 공개 구현의 존재와 이 repo에서의 재현 완료를 구분한다.

## 결론

ODE-Edit의 방어 가능한 연구 질문은 “ODE를 처음 적용한 model editor”,
“최초의 동적 layer selection”, “최초의 history/capacity-aware editor”가 아니다.
검토한 선행연구에서 아직 직접 확인되지 않은 좁은 조합은 다음이다.

> 동일한 현재 model snapshot에서 여러 locate-and-edit layer proposal을
> 비교하고, 현재 요청과 editor-native 통계만으로 proposal 크기를 공동
> 배분한 뒤, 부분 동시 적용 후 모든 proposal을 다시 계산하는
> layer-synchronous state-dependent routing.

이 문장도 최초성 claim이 아니라 검증할 연구 공백이다. 특히
[WilKE](https://arxiv.org/abs/2402.10987),
[NSE](https://arxiv.org/abs/2410.04045),
[AlphaEdit](https://arxiv.org/abs/2410.02355),
[Lifelong Knowledge Editing requires Better Regularization](https://arxiv.org/abs/2502.01636),
[Norm Anchors Make Model Edits Last](https://arxiv.org/abs/2602.02543),
[BetaEdit](https://arxiv.org/abs/2605.09285),
[CrispEdit](https://arxiv.org/abs/2602.15823),
[ODESteer](https://arxiv.org/abs/2602.17560), 그리고
[ODE-M](https://arxiv.org/abs/2605.19409)이 넓은 최초성 주장을 차단한다.

## 네 범주

### Proposal에서 온 내용

- MEMIT의 여러 rewrite layer update를 local actuator로 보고, 동일 snapshot에서
  비교한 뒤 capacity cost에 따라 재배분한다.
- partial write 뒤 key, residual, proposal, utility를 다시 계산하는
  state-dependent trajectory가 one-shot update보다 유리할 수 있다고 가정한다.
- paraphrase, neighborhood, downstream evaluation prompt는 controller가 보지
  않는 information firewall을 둔다.

### Repo/protocol에서 확인한 사실

- 현재 EasyEdit checkout에는 MEMIT, ROME, AlphaEdit, EMMET, PMET 등의 코드가
  있으나, ENCORE, NAS, BetaEdit, CrispEdit, LyapLock, WilKE의 canonical
  구현은 없다.
- 따라서 이 세션에서 실제 실행하지 않은 방법을 “EasyEdit baseline으로
  재현했다”고 쓰지 않는다.
- Session 01의 동일-track baseline은 우선 native MEMIT과 그 proposal을
  공유하는 내부 대조군이다. 외부 구현 baseline은 Motivation이 생존한 뒤
  별도 provenance와 정보 예산을 고정하고 연다.

### GH 추정

- novelty가 남으려면 `same-snapshot`, `joint fractional allocation`,
  `simultaneous partial application`, `relinearization` 네 요소의 결합이
  각각 필요한 이유를 intervention으로 보여야 한다.
- layer utility의 분산이나 rank 변화만으로는 WilKE/NSE보다 강한 동적 routing
  필요성을 보이지 못한다.
- ODE 명칭은 fixed proposal을 여러 번 나누는 대조군보다 refreshed proposal이
  낫고, step refinement와 state dependence가 확인될 때만 유지할 수 있다.

### 사용자 확인 필요

- `ODE-Edit`은 2026년 multimodal editing 논문
  [ODEdit](https://arxiv.org/abs/2601.19700)과 검색·발음이 충돌한다.
  Motivation 판정 전에는 현재 repo 이름을 유지하되, paper method name
  확정 시 명칭 변경 여부를 사용자가 결정해야 한다.

## 정보 예산

| track | edit-time 허용 정보 | 대표 방법 | 비교 원칙 |
| --- | --- | --- | --- |
| B0 editor-native | 현재 request, 표준 rewrite context, 현재 hidden/key, 사전 계산 covariance/projector | ROME, MEMIT, PMET, WilKE, ODE-Edit main | Session 01의 주 비교 |
| B1 history-aware | B0 + 이전 edit key/value/update, queue 또는 streaming statistic | canonical AlphaEdit, NSE, LyapLock, BetaEdit, DeltaEdit | 별도 history panel |
| B2 auxiliary/calibration | pilot edit, capability/replay/calibration data, 미래 batch | NAS 기본 설정, CrispEdit, LocFT-BF | 정보 우위를 표에 명시 |
| B3 inference augmentation | 외부 memory/router/adapter 또는 activation intervention | WISE, GRACE, ODESteer | in-weight editor와 별도 system panel |

정보 예산이 다른 방법을 하나의 숫자로 무차별 순위화하지 않는다. 특히 NAS의
pilot, CrispEdit의 capability data, WISE/GRACE의 inference-time component를
숨긴 비교는 금지한다.

## Layer selection과 allocation

| 지형 | 대표 prior | ODE-Edit에 남는 검증 의무 |
| --- | --- | --- |
| 고정 단일 layer | [ROME](https://arxiv.org/abs/2202.05262), GRACE | 직접 novelty 없음 |
| 고정 layer set의 ordered residual distribution | [MEMIT](https://arxiv.org/abs/2210.07229), AlphaEdit, EMMET | same-state joint comparison이 ordered construction보다 필요한지 |
| 다른 static distribution | [PMET](https://arxiv.org/abs/2308.08742) | best static allocation보다 refresh가 나은지 |
| request-wise single winner | WilKE | fractional multi-layer routing이 winner-take-all보다 나은지 |
| neuron-wise dynamic selection과 반복 multi-layer write | NSE | layer QP와 neuron mask/retry의 차이 및 비용 |
| 고정 multi-layer set의 simultaneous constrained optimization | CrispEdit | auxiliary capability data 없이 editor-native proposal로 얻는 추가 이득 |

MEMIT을 “완전 open-loop”라고 부르지 않는다. EasyEdit 구현은 한 layer update를
임시 반영한 뒤 다음 layer의 key와 residual을 다시 계산한다. 정확한 대비는
ascending-order Gauss–Seidel-style one-pass construction과
same-snapshot Jacobi-style comparison plus all-layer revisit이다.

## History, norm, capacity

| 보호 객체 | 대표 prior | 금지되는 넓은 claim |
| --- | --- | --- |
| base covariance/null geometry | ROME/MEMIT, AlphaEdit, BetaEdit | covariance/projector 최초 |
| 이전 edit constraint | AlphaEdit, NSE, LyapLock, BetaEdit, DeltaEdit | history-aware 최초 |
| solved value/weight/update norm | MPES+norm constraint, NAS, PRUNE | norm-collapse 발견 또는 norm anchor 최초 |
| capability curvature/spectral direction | CrispEdit, SPHERE, REVIVE | capacity subspace 최초 |
| 장기 평균 preservation budget | LyapLock | local QP가 장기 보존을 보증한다는 claim |
| 외부 memory capacity | WISE, GRACE | capacity management 최초 |

따라서 ODE-Edit의 `capacity`는 일반 model capacity가 아니라 후보 rewrite
layer 사이에서 비교하는 **marginal update budget/proxy**로 한정한다.
`C`-weighted displacement가 retention 또는 capability preservation을
예측하는지는 별도 intervention으로 검증하며, 정의상 참이라고 두지 않는다.

## ODE와 barrier 선행연구

- ODESteer는 activation addition을 ODE의 1차 근사로 해석하고 barrier-guided
  multi-step adaptive activation steering을 제안한다.
- ODE-M은 continual model merging에서 parameter-space path와
  loss-increasing step을 막는 barrier constraint를 사용한다.
- 기존 `ODEdit`은 ODE 약칭은 아니지만 multimodal model editing에서 이미
  trajectory라는 표현과 매우 유사한 이름을 사용한다.

따라서 “first ODE editor”는 사용하지 않는다. ODE-Edit이 ODE라는 말을
유지하려면 최소한 다음을 보여야 한다.

1. fixed initial proposal의 `K`-step split보다 refreshed proposal이 낫다.
2. round가 진행되며 proposal direction, allocation 또는 active constraint가
   measurement noise를 넘어 실제로 변한다.
3. state dependence를 제거한 ablation이 악화된다.
4. step size/round 수 변화에서 endpoint 또는 trajectory가 안정화된다.
5. 가능하면 Euler와 Heun/local truncation proxy를 비교한다.

이 중 핵심 항목이 실패하면 `state-dependent iterative controller` 또는
`static capacity-aware routing`으로 이름과 claim을 낮춘다.

## Session 01 baseline tier

### Motivation 내부 필수

아래는 모두 동일 direct-z, request, context, covariance, pre-state에서 갈라지는
paired branch다.

1. native MEMIT ordered construction
2. fixed native delta의 `K`-step split
3. global scalar/first-hitting stop
4. C-normalized uniform layer allocation
5. best preregistered static per-layer allocation
6. best single-layer/WilKE-style proxy
7. same-snapshot simultaneous proposal
8. fixed-direction dynamic coefficient
9. refreshed-direction refreshed-coefficient joint allocation
10. Gauss–Seidel small-step control

`WilKE-style proxy`는 WilKE reproduction이 아니다. 실제 WilKE와 NSE를 비교하려면
각 논문의 target construction, selector, 반복 규칙을 별도 구현·감사해야 한다.

### Motivation 생존 후

- B0/B1 분리: native AlphaEdit, canonical history-aware AlphaEdit, EMMET, PMET
- pinned external baseline: MPES+norm constraint, NAS, WilKE, NSE, BetaEdit
- 다른 정보 예산: CrispEdit, LocFT-BF, WISE, GRACE
- long-horizon claim: SPHERE, REVIVE, DeltaEdit, LyapLock과 비교

## 논문별 재현 위험

| 방법 | 이 repo에서의 현재 상태 | 주요 위험 |
| --- | --- | --- |
| MEMIT | EasyEdit read-only import 가능 | context cache, stats path, layer order, atomic-vs-batch |
| AlphaEdit | EasyEdit 구현과 precomputed projector 존재 | projector threshold, mutable history cache, canonical-history 여부 |
| EMMET/PMET | EasyEdit 코드 존재 | 다른 target/distribution을 routing gain으로 오인 |
| WilKE/NSE | EasyEdit에 없음 | candidate별 target 비용, neuron mask/retry, 구형 backbone 설정 |
| MPES+norm/NAS | EasyEdit에 없음 | early-stop budget, original anchor, pilot 정보 예산 |
| BetaEdit/CrispEdit | EasyEdit에 없음 | 최신 구현/version, history/capability auxiliary data |
| SPHERE/REVIVE/DeltaEdit/LyapLock | canonical local 구현 미확인 | batch 100 결과와 atomic stream 결과 혼동 |

## Claim 문장

현재 허용:

> 기존 연구는 ordered multi-layer distribution, alternative static
> distribution, per-request single-layer selection, dynamic neuron masking,
> simultaneous curvature-constrained optimization을 각각 탐구했다. 본 연구는
> 여러 locate-and-edit proposal을 동일 model state에서 공동 배분하고,
> 동시 부분 적용 뒤 proposal을 재계산하는 더 좁은 문제를 검증한다.

현재 금지:

- “기존 editor는 모두 fixed layer만 사용한다.”
- “cross-layer edit distribution은 연구되지 않았다.”
- “최초의 adaptive/history/capacity/ODE model editor다.”
- “C-weighted cost를 줄이면 장기 capability가 보존된다.”
- “CounterFact 성공으로 practical lifelong editing을 해결했다.”

## Motivation 판정에 미치는 영향

관련 연구는 ODE-Edit을 즉시 kill하지 않지만, 통과 기준을 강화한다.
MV-1의 heterogeneity만으로는 부족하고, MV-2의 refreshed decision advantage,
MV-3의 best static/single-winner 대비 matched-progress frontier, MV-4의
retention intervention이 모두 필요하다. 하나라도 실패하면 해당 mechanism과
claim을 제거하며, descriptive statistic으로 연구 방향을 유지하지 않는다.
