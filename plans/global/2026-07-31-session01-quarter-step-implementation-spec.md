# Motivation 세션 — native-distance quarter-step 재계산 진단 사전등록

상태: **실행 전 고정(locked)**
방법론 표기: **ODE-Edit**
대상 실행: `session01-qstep4-pair-v1`

## 1. 출처와 사실 경계

- proposal에서 온 내용:
  model edit을 한 번에 적용하지 않고 경로를 따라 이동하며 방향을 다시
  계산하는 것이 유리할 수 있다는 Motivation을 검증한다.
- repo/protocol에서 확인한 사실:
  이전 MV-2는 `q=1/256`, `h=1/2`였으므로 첫 state treatment가 native
  C-distance의 약 `1/32`였다. Llama direction effect는 음수, Qwen은
  양수였고 두 모델 결론이 갈렸다. 두 실행 모두 technical validity는
  통과했다.
- GH 추정:
  `D/32` state change가 너무 작아 재계산 차이가 numerical/application
  noise에 가까웠을 가능성이 있다. 이는 검증할 가설이지 확인된 원인이
  아니다.
- 사용자 확인 필요:
  없음. 사용자가 native update를 약 1/4씩 나누는 독립 후속 실험과
  Llama/Qwen 동시 제출을 명시했다.

이 진단은 기존 MV-2를 재해석하거나 덮어쓰지 않는다. fresh case와 새
run ID를 쓰는 독립 Motivation evidence다.

## 2. 질문과 최소 신호

주 질문은 다음과 같다.

> native update의 C-distance를 `D`라 할 때, `D/4`씩 네 번 이동하면
> 매 state의 방향 재계산이 고정 방향보다 rewrite progress를 높이는가?

최소 신호는 두 fixed backbone 중 적어도 하나에서 사전등록된 step 4
방향 효과 `A4-B4`가 practical envelope를 넘고 12 case 중 7개 이상에서
양수인 것이다. 중간 step 중 가장 좋은 지점을 사후 선택해 통과시키지
않는다.

## 3. 고정 입력

- model:
  - `llama3-8b-inst`
  - `qwen2.5-7b-inst`
- EasyEdit runtime: repo의 `servers/local/method-runtime.env`와 동일
- layers: `[4, 5, 6, 7, 8]`
- run seed: `17`
- case count: model별 `12`
- case selection: canonical salt rank `[112:124]`
  - canonical first 100 및 이전 MV-2 `[100:112]`와 disjoint
- run IDs:
  - `qstep4_llama_f0_v1`
  - `qstep4_qwen_f0_v1`
- bootstrap: seed `20260802`, resamples `4000`
- practical floor: `1e-4`

## 4. target 및 거리 정의

각 case에서 direct-z는 W0에서 정확히 한 번 계산하고 이후 고정한다.
target token, context, direct-z tensor/artifact identity를 hash로 묶고,
각 descendant는 exact parameter hash chain으로만 이 target을 사용할 수
있다.

ordered MEMIT W0 proposal의 C-energy를 `E_native`라 하고
`D=sqrt(E_native)`로 둔다.

- operational hop 수 `K=4`
- 각 hop 거리 `D/4`
- 각 hop C-energy `E_native/16`
- 총 경로 길이 `D`
- controller central probe 거리 `D/64`

probe를 `D/4`로 같이 키우지 않는다. 이로써 센서 반경과 실제 state
treatment 크기를 분리한다. 모든 covariance는 run-start pinned cache만
읽고 재계산하지 않는다.

## 5. 세 경로와 대조군

첫 hop은 세 경로 모두 W0의 동일한 score-mix action을 사용한다.

- `A` (`refreshed`): step 2--4에서 proposal direction과 controller
  coefficient를 모두 현재 state에서 재계산
- `B` (`coefficient`): W0 proposal basis는 고정하고 coefficient만 현재
  state에서 재계산
- `C` (`fixed`): W0 proposal basis와 W0 coefficient를 모두 고정

정책 차이는 step 2부터 시작한다. 모든 hop은 동일한 `D/4` C-distance를
갖는다.

수치/application control:

- `native_ordered_full`: ordered MEMIT native update one-shot
- `native_ordered_split4`: 같은 ordered update를 정확히 1/4씩 네 번 적용

두 native control의 progress 차이는 반복 적용 잡음 envelope에 포함한다.
`A/B/C`는 score-mix 경로이므로 native ordered endpoint와 방향이 같다고
가정하지 않는다.

## 6. outcome 및 사전등록 효과

outcome은 frozen target의 teacher-forced rewrite utility 변화다. 모든
feature, proposal/action hash, lineage, per-hop budget, branch order를
durable receipt로 먼저 커밋한 뒤 outcome을 평가한다.

branch는 총 13개다.

1. W0 no-op replay
2. common step 1
3. step 2의 A/B/C
4. step 3의 A/B/C
5. step 4의 A/B/C
6. native one-shot
7. native split4

주효과는 step 4로 고정한다.

- direction refresh: `A4-B4`
- coefficient refresh: `B4-C4`
- total refresh: `A4-C4`

보조 효과:

- `A4-native ordered MEMIT`: 현재 단순 ODE-style 경로의 native baseline
  대비 기대효과 proxy
- step별 평균 trajectory: mechanism이 어느 시점부터 분기하는지 진단
- native split4-one-shot: application noise control

중간 best-hop, case별 best-policy, outcome-selected oracle은 pass 근거로
사용하지 않는다.

## 7. 기대효과 예측 경계

이전 MV-2의 실제 방향 재계산 구간은 대략 `D/32`였고 이번 각 hop은
`D/4`이므로 state perturbation은 hop 기준 8배 크다. 따라서
under-manipulation 가설이 맞다면 `A4-B4`의 절대값과 sign consistency가
`1e-4` envelope보다 명확해져야 한다.

이전 aggregate를 거리 비율로 단순 선형 외삽하는 것은 saturation과
direction rotation을 무시하므로 수치 예측으로 채택하지 않는다. 이번
실험에서 측정하는 `A4-C4`가 재계산 자체의 관측 가능한 개선폭이고,
`A4-native`가 현 설계가 native MEMIT보다 개선될 가능성의 직접 proxy다.
둘 다 최종 ODE-Edit 성능 claim이나 upper bound가 아니다.

## 8. 판정과 중단 조건

technical block:

- 12 case exact denominator, target lineage, matched per-hop C, rollback,
  firewall, receipt-before-outcome, native control 중 하나라도 실패
- raw failure case를 제외하거나 성공 case만 분석하는 rescue

scientific kill:

- `A4-C4` mean이 `-envelope`보다 작고 12 case 중 7개 이상이 명확한 음수
  이면 native-distance refresh를 중단
- 두 모델 모두 direction/coefficient/total 신호가 envelope 안이면 현재
  refresh Motivation을 중단하고 static routing 또는 다른 mechanism으로
  pivot

다음 진행 허용:

- 적어도 한 모델에서 `A4-B4`가 mean `> envelope`, positive case `>=7`
- 다른 모델이 명확한 harm이면 cross-model 일반화 claim은 금지하고
  backbone-specific mechanism 진단만 허용
- `B4-C4`만 통과하면 방향 재계산을 버리고 fixed-direction dynamic
  coefficient로 전환

AlphaEdit은 null-space projection으로 update geometry 자체를 바꾸므로
이번 equal-C direction refresh 진단에 섞지 않는다. Motivation 신호가
살아남은 뒤 별도의 projected/unprojected factorial diagnostic으로 다룬다.

## 9. 계산 및 실행 envelope

- model별 controlled NFE: case당 `98`, 총 `1176`
- model별 proposal build: case당 `5`, 총 `60`
- model별 probe panel: case당 `7`, 총 `84`
- server1 parent allocation:
  - job name `odeedit_qstep_pair_v1`
  - GPU `2` (child별 `1`)
  - CPU `16` (child별 `8`)
  - host memory `130000M` (child별 `65000M`)
  - wall cap `12:00:00`
- server1 project GPU cap `3`; overflow이면 제출하지 않고 pending
- Llama/Qwen은 별도 child `srun`으로 동시에 시작하며 한 child가 실패하면
  sibling을 종료하는 fail-fast pair다.

raw tensors, direct-z, JSONL, logs는 `local/`에만 둔다. Git에는 spec,
runner, tests, compact metric summary, audit, reproducibility instruction만
남긴다.
