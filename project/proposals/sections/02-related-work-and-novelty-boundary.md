# ODE-Edit 관련 연구와 novelty 경계

- 작성일: 2026-07-30
- 최종 갱신: 2026-07-31
- 문서 성격: Motivation Validation을 위한 proposal-side related-work audit
- claim 상태: `motivation_not_supported_cross_model`; novelty claim 미승격
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

이 문장도 최초성 claim이 아니라 검증할 연구 공백이었다. Session 01에서
same-snapshot allocation signal은 관측됐지만 refreshed-direction 효과가
Llama와 Qwen에서 반대 부호였고 pair gate가 `NO MV3`로 닫혔다. 따라서 현재
이 조합은 **방어 가능한 contribution이 아니라 반증된 cross-model
motivation**이다. 상세 수치와 first-match 판정은
[`Session 01 GH 최종 보고서`](../../../experiment-reports/global/2026-07-31-session-01-motivation-final.md)와
[`MV-2 pair red audit`](../../../audits/global/2026-07-31-mv2refresh-pair-v2.postrun.md)에
분리해 기록했다.

특히
[WilKE](https://arxiv.org/abs/2402.10987),
[NSE](https://arxiv.org/abs/2410.04045),
[AlphaEdit](https://arxiv.org/abs/2410.02355),
[Lifelong Knowledge Editing requires Better Regularization](https://arxiv.org/abs/2502.01636),
[Norm Anchors Make Model Edits Last](https://arxiv.org/abs/2602.02543),
[BetaEdit](https://arxiv.org/abs/2605.09285),
[CrispEdit](https://arxiv.org/abs/2602.15823),
[HiEdit](https://arxiv.org/abs/2604.11214),
[ODESteer](https://arxiv.org/abs/2602.17560), 그리고
[ODE-M](https://arxiv.org/abs/2605.19409)이 넓은 최초성 주장을 차단한다.

## 2026-07-31 primary-source 재검증

| 축 | Primary source에서 확인한 범위 | ODE-Edit에 미치는 경계 |
| --- | --- | --- |
| single/fixed-layer locate-and-edit | [ROME](https://arxiv.org/abs/2202.05262)는 mid-layer FFN의 rank-one factual update를 제안 | fixed-layer editing 최초성 없음 |
| multi-layer/mass editing | [MEMIT](https://arxiv.org/abs/2210.07229)은 다수 memory를 직접 갱신하고, [EMMET](https://arxiv.org/abs/2403.14236)은 ROME/MEMIT을 preservation–memorization 관점에서 통일 | multi-layer/constrained write 최초성 없음 |
| alternative target/distribution | [PMET](https://arxiv.org/abs/2308.08742)은 MHSA/FFN hidden target을 분리해 FFN update를 정밀화 | 다른 layer allocation 자체가 novelty가 아님 |
| preservation geometry | [AlphaEdit](https://arxiv.org/abs/2410.02355)은 preserved knowledge null-space projection을 사용 | projector/covariance/history 보호 최초성 없음 |
| dynamic selection | [WilKE](https://arxiv.org/abs/2402.10987)는 request별 editing layer를 고르고, [NSE](https://arxiv.org/abs/2410.04045)는 neuron-level sequential editing, [HiEdit](https://arxiv.org/abs/2604.11214)는 hierarchical RL로 knowledge-relevant layer를 식별 | adaptive/dynamic layer selection 최초성 금지 |
| lifelong system memory | [WISE](https://arxiv.org/abs/2405.14768)는 side memory/router/sharding, [GRACE](https://arxiv.org/abs/2211.11031)는 discrete latent codebook | in-weight editor와 별도 system/information budget |
| regularization/norm | [Lifelong Knowledge Editing requires Better Regularization](https://arxiv.org/abs/2502.01636)은 MPES와 Frobenius norm constraint, [Norm Anchors Make Model Edits Last](https://arxiv.org/abs/2602.02543)는 original-model norm anchor를 사용 | norm-growth 발견·anchor 최초성 금지 |
| null-space/capability constraint | [BetaEdit](https://arxiv.org/abs/2605.09285)은 approximate null-space leakage와 history-aware update, [CrispEdit](https://arxiv.org/abs/2602.15823)은 capability-loss low-curvature projection을 사용 | capacity/preservation proxy를 정의상 capability로 해석 금지 |
| ODE/trajectory | [ODESteer](https://arxiv.org/abs/2602.17560)는 barrier-guided ODE activation steering, [ODE-M](https://arxiv.org/abs/2605.19409)은 continual model merging의 barrier-aware parameter trajectory, [ODEdit](https://arxiv.org/abs/2601.19700)은 invariant trajectory 기반 multimodal editing | “first ODE/trajectory editor” 및 현재 paper name 금지 |

위 표는 primary source의 제안 범위를 확인한 것이며, 해당 방법을 이 repo에서
재현했다는 뜻이 아니다.

## 네 범주

### Proposal에서 온 내용

- MEMIT의 여러 rewrite layer update를 local actuator로 보고, 동일 snapshot에서
  비교한 뒤 capacity cost에 따라 재배분한다.
- partial write 뒤 key, residual, proposal, utility를 다시 계산하는
  state-dependent trajectory가 one-shot update보다 유리할 수 있다고 가정한다.
- paraphrase, neighborhood, downstream evaluation prompt는 controller가 보지
  않는 information firewall을 둔다.

### Repo/protocol에서 확인한 사실

- 현재 EasyEdit checkout에는 ROME, MEMIT, PMET, AlphaEdit, EMMET, WISE,
  GRACE의 model 코드가 있다.
- pinned backbone 이름의 hparams file은 ROME, MEMIT, AlphaEdit, WISE에
  Llama3-8B와 Qwen2.5-7B가 모두 있고, GRACE는 Qwen2.5-7B가 있다.
  PMET/EMMET에는 이 두 pinned 이름의 pair가 없다.
- WilKE, NSE, MPES+norm, NAS, BetaEdit, CrispEdit, HiEdit, ODESteer, ODE-M,
  ODEdit의 canonical implementation은 해당 EasyEdit method/hparams listing에서
  확인되지 않았다.
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
| B2 auxiliary/calibration | capability/replay/calibration data, 미래 batch | CrispEdit, LocFT-BF | 정보 우위를 표에 명시 |
| B3 inference augmentation | 외부 memory/router/adapter 또는 activation intervention | WISE, GRACE, ODESteer | in-weight editor와 별도 system panel |

NAS는 original-model reference norm을 사용하는 low-overhead anchor이므로 B0의
강한 regularization baseline으로 별도 표시한다. 정보 예산이 다른 방법을
하나의 숫자로 무차별 순위화하지 않는다. 특히 CrispEdit의 capability data와
WISE/GRACE의 inference-time component를 숨긴 비교는 금지한다.

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

## Motivation 최종 판정에 미치는 영향

관련 연구가 Session 01을 kill한 것은 아니다. 사전등록된 MV-2
refreshed-direction primary가 cross-model로 재현되지 않아 empirical gate가
연구를 닫았다. 이 결과와 선행연구를 함께 적용하면 다음만 허용된다.

> 두 고정 모델에서 same-snapshot allocation signal은 관측됐으나,
> all-proposal direction refresh의 효과는 architecture-dependent였고
> cross-model ODE/relinearization motivation은 지지되지 않았다.

따라서 MV-3/MV-4, 외부 baseline 이식, long-horizon benchmark를 진행하지
않는다. fixed-direction dynamic coefficient 또는 static allocation을
연구하려면 ODE contribution의 후속이 아니라 새 proposal로 시작해야 한다.
