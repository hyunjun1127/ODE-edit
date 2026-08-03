# Session 02 P1 mechanism diagnostic 및 Motivation closure

- 최종 갱신: **2026-08-03 23:28 KST**
- 방법명: **ODE-Edit**
- 모델: `llama3-8b-inst`, `qwen2.5-7b-inst`
- model dtype: checkpoint-original `torch.bfloat16`
- P1 execution commit: `e285c96d704a04287f65d1c51539ebbb81451d28`
- `S_max=8` continuation head: `1e0440792297930820059212dead9f93af27edc5`
- P1 proposal ID: `83eec5068acec673f7d9d0788e57ce0e9873a55785f7a6932026a4a2420c330b`
- base lock SHA-256: `95376c554f01e00c9c5d71e99dfa3cbb5339a7534db8a62a9a05bc69c0be1bbb`
- 최종 gate: **`MOTIVATION_MECHANISM_CONDITIONAL_PASS`**
- 현재 controller gate: **`NO_GO_FOR_MAIN_TABLE; COMMON_CONTROLLER_REDESIGN_REQUIRED`**
- `S_max` 판정: **공통 `6` 유지; `8` 승격 기각**
- evaluation/generation count: `0`
- scientific superiority claim: **없음**

## 0. Executive decision

ODE-Edit의 핵심 연구 방향은 살아남지만, 현재 Full controller는 main table에 올릴 수 없다.

살아남은 신호는 다음 두 가지다.

1. 두 모델 모두 accepted step 뒤 모든 edit layer의 proposal direction이 다시 형성됐다.
   Full trajectory의 연속 field 사이 direction-ID transition mean은 두 모델 모두 `1.0`이다.
   따라서 joint model state에 종속된 vector field를 반복 적분한다는 mechanism은 실제
   구현에서 비자명하게 작동했다.
2. 동일한 entry state에서 비교 가능한 Qwen case `2022`에서 Native와 Full이 모두
   rewrite event를 만족했고, Full endpoint는 Native보다 total C-distance가 `14.59%`,
   total C-energy가 `27.05%` 낮았다. 이 한 case는 state refresh와 joint allocation이
   더 낮은 capacity endpoint를 찾을 **가능성**을 보인다.

그러나 현재 controller의 공통 모델 성립 조건은 통과하지 못했다.

- Llama Full은 `0/4` first-hit이며 Static은 `2/4` first-hit이다.
- Llama의 affected 3 case를 `S_max=8`까지 이어도 round 7–8 first-hit은 `0/3`이었다.
- Qwen Full은 `2/4` first-hit이지만, 두 모델 공통 성능으로 일반화할 수 없다.
- Full/Native compute는 두 모델에서 대체로 `3x–3.59x`인 yellow 영역이다.
- efficacy, generalization, locality, prior-retention 평가는 이번 P1에서 실행하지 않았다.

따라서 lenient gate는 **연구 방향**만 연다. 현재 controller의 성능 우위, preservation
우위, deployable efficiency, lifelong claim은 열지 않는다. 다음 단계는 더 큰 실험이나
round 수 증가가 아니라, 두 모델에 동일하게 적용되는 allocation/progress/trust controller
재설계다.

## 1. 네 범주의 분리

### 1.1 Proposal에서 온 내용

- 한 edit의 direct-z는 고정하지만, finite multi-layer write가 다음 residual, key,
  downstream activation, layer proposal과 marginal capacity cost를 바꿀 수 있다.
- MEMIT은 앞 layer write를 뒤 layer 계산에 전달하는 ordered one-sweep이고, ODE-Edit은
  joint partial update 뒤 모든 edit layer의 proposal과 allocation을 current state에서
  다시 만든다.
- ODE가 필요한 이유는 write를 단순 분할하기 위해서가 아니라 state-dependent vector
  field를 반복 적분하기 위해서다.
- Proposal은 검증된 claim이나 확정 paper plan이 아니다.

### 1.2 Repo/protocol에서 확인한 사실

- EasyEdit source는 수정하지 않았고, ODE-edit-side hook/runtime만 사용했다.
- 두 모델 모두 original checkpoint BF16, 동일 event backend와 trial backend, 동일 공통
  controller 상수로 실행했다. Model-specific rescue는 없었다.
- P1 R2는 arm outer × canonical edit inner인 4-edit sequential stream이다. Successful
  endpoint는 다음 edit로 이어지고, failed edit만 해당 pre-edit state로 rollback됐다.
- request-scoped state ID와 request-independent target-weight state ID를 분리한 뒤 두 모델
  모두 cross-edit continuity, rollback, arm isolation과 terminal hash gate를 통과했다.
- raw artifact와 logs는 ignored `local/`에만 있고 Git에는 포함하지 않는다.
- server2는 onboarding HOLD이므로 artifact broadcast는 수행하지 않았으며
  `BROADCAST_EXCEPTION_SERVER2_NOT_READY`다.

### 1.3 GH 추정

- Llama에서 direction은 매 round 바뀌지만 allocation cosine이 거의 `1`이고 Full이
  Static보다 나아지지 않는다. 이는 nonstationarity가 없다는 뜻이 아니라, 현재 공통
  controller가 그 변화를 유용한 velocity 재배분으로 전환하지 못했을 가능성이 높다.
- `S_max=8`의 `0/3` 결과 때문에 Llama 병목을 단순 step 부족으로 보는 설명은 약해졌다.
  Progress normalization, capacity marginal cost와 trust acceptance의 상호작용이 우선
  조사 대상이다.
- Qwen case `2022`의 capacity 감소 폭은 다음 controller가 보존할 가치가 있는 local
  upper-bound signal이지, 전체 데이터 평균 기대효과가 아니다.

### 1.4 사용자 확인 필요

- 현재 common-controller redesign을 마친 뒤 sealed small diagnostic을 새로 실행할지,
  또는 ODE-Edit을 static synchronous method로 pivot할지는 다음 GH 설계안 뒤 승인받아야
  한다.
- 모델별 hparam calibration은 미래 별도 단계로 열 수 있지만, 현재 단계에서는 사용자
  지시대로 공통 hparam만 사용한다.
- 10/100-edit, AlphaEdit 확장과 server2 artifact broadcast는 각각 별도 승인/onboarding이
  필요하다.

## 2. Evidence와 무결성

### 2.1 P1 R2 terminal artifacts

| 모델 | Slurm job | 상태 | elapsed | terminal manifest SHA-256 | summary SHA-256 |
|---|---:|---|---:|---|---|
| Llama | `16190` | `COMPLETED 0:0` | `00:21:49` | `19d7619400dd4387f69b05192046297b5331ee5d4e581486ab9544eeafc0da6f` | `1866f8217dadb4007ad42469eda3500e04a78791afcb0ef2761db146f700b71e` |
| Qwen | `16191` | `COMPLETED 0:0` | `00:32:27` | `c830799669a6d7a537cb6b4893f8e2d580f307f5bf445bae68c51885cbb9c381` | `2ba3bca4505822c082e55df4043ab0e73b6ff560f35989f7caf71db71ce65400` |

Local roots:

- `local/results/session02-p1-four-case-mechanism-r2-llama3-8b-inst-83eec506`
- `local/results/session02-p1-four-case-mechanism-r2-qwen2.5-7b-inst-83eec506`

두 terminal manifest의 listed file identity는 전수 rehash PASS다. 각 모델은
controller/compute/mechanism `16/16` records와 direct-z `16`개를 갖고, `evaluation.jsonl`
은 의도적으로 비어 있다. 빈 evaluation 파일은 성능 metric이 아니다.

### 2.2 `S_max=8` affected continuation

- job: `16215`, `COMPLETED 0:0`, elapsed `00:11:34`
- root: `local/results/session02-p1-smax8-affected-cont-v1-llama3-8b-inst-83eec506`
- terminal manifest SHA-256:
  `37e2dda04fbb63ea5f4434c53c32ce93b33553361b2198e66d441aa6105634bc`
- summary SHA-256:
  `146b19da37cf279b8f941a5822e610fb002dca816e958abb3f6b3a14e39d82ff`
- affected cases: `2022`, `20964`, `768`
- R2 rounds 1–6 prefix exact, existing direct-z read-only reuse, recompute `0`
- round 7–8 first-hit: `0/3`; required `>=2/3`
- promotion: `candidate_pass=false`, future common `S_max=6`

### 2.3 Current resource boundary

- server1 project GPU cap은 사용자 지시에 따라 **`3`**이다.
- R2 pair는 aggregate `2/3`, continuation은 `1/3`에서 실행됐고 terminal 뒤 active
  ODE-Edit GPU는 `0`이다.
- `servers/local/gpu-caps.tsv`, `servers/active/server1.md`와
  `servers/connection-inventory.md`가 현재 cap의 source다.
- Executed P1 numerical lock 안의 `project_gpu_cap=4`는 당시 실행 provenance를 보존하기
  위해 변경하지 않는다. 이는 future submission authority가 아니며, 모든 새 제출은
  local helper가 현재 cap `3`으로 다시 검사해야 한다.

## 3. P1 결과

### 3.1 First-hit 상태

| 모델 | Native | Static synchronous | One-refresh | Full ODE-Edit |
|---|---:|---:|---:|---:|
| Llama | `4/4` | `2/4` | `0/4` | `0/4` |
| Qwen | `4/4` | `2/4` | `1/4` | `2/4` |

Llama Full의 세 `resolution_cap_unresolved` case는 `S_max=8` continuation에서도 모두
unresolved였다. 따라서 cap을 더 늘리는 rescue는 중단한다. Qwen의 positive 결과만으로
모델별 controller나 threshold를 도입하지 않는다.

### 3.2 State dependence와 allocation

| 진단 | Llama Full | Qwen Full | 해석 경계 |
|---|---:|---:|---|
| direction-ID transition mean | `1.0` | `1.0` | every refresh에서 actuator direction이 바뀜 |
| ranking drift mean / max | `0.0579 / 0.2` | `0.0545 / 0.2` | 일부 layer 순위 변화 |
| allocation cosine mean / min | `0.999704 / 0.998654` | `0.970226 / 0.858368` | Llama는 거의 고정, Qwen은 일부 실질적 재배분 |
| support turnover | `0` | `0` | active layer set은 고정 |

이는 “proposal refresh가 numerical no-op”이라는 kill test는 통과시킨다. 다만 refresh가
항상 더 좋은 allocation이나 endpoint를 만든다는 주장은 통과시키지 못한다.

### 3.3 동일 entry state에서 확인된 local capacity signal

Qwen order position `0`, case `2022`는 arm history가 갈라지기 전이므로 Native와 Full이
동일한 target-weight entry state와 direct-z tensor에서 출발한다. 둘 다 first-hit했다.

| arm | first-hit | accepted steps | total C-distance | total C-energy | controller GPU sec |
|---|---:|---:|---:|---:|---:|
| Native | yes | `1` | `0.3229217253` | `0.1042784406` | `76.3094` |
| Full | yes | `5` | `0.2758017200` | `0.0760665888` | `266.7341` |

Full은 Native 대비 C-distance `14.59%`, C-energy `27.05%`를 줄였지만 compute는
약 `3.50x`다. 이 결과는 lower-capacity endpoint possibility를 지지한다. 한 case이고
preservation metric이 없으므로 expected benchmark gain이나 일반적인 method improvement로
외삽하지 않는다.

Order position `1+`에서는 arm별 prior successful edit history가 다를 수 있으므로 단순
per-case 수치를 causal same-W0 비교로 사용하지 않는다.

### 3.4 Compute

| 모델 | Full/Native wall range | 최대 ratio | peak allocated / reserved |
|---|---:|---:|---:|
| Llama | `2.894–3.590x` | `3.590x` | `22.411 / 23.631 GiB` |
| Qwen | `0.847–3.501x` | `3.501x` | `25.968 / 28.930 GiB` |

두 모델 모두 locked `>4x` stop line은 넘지 않았지만 `<=3x` green을 안정적으로 만족하지
못했다. Proposal build가 controller time의 대부분을 차지하므로, 다음 구현은 method
semantics를 바꾸지 않는 batching/reuse와 first-hit 효율을 함께 다뤄야 한다.

## 4. Gate 판정

### 4.1 Motivation mechanism gate — conditional pass

다음을 이유로 연구 방향을 kill하지 않는다.

- current joint state에서 모든 layer proposal이 실제로 다시 형성된다.
- allocation도 Qwen 일부 trajectory에서는 numerical noise를 넘게 변한다.
- 동일 entry state의 Qwen case 하나에서 first-hit을 유지하며 더 낮은 C endpoint를 찾았다.
- original BF16, common method, no model rescue 조건에서 기술 pipeline이 양 모델 모두
  terminal integrity를 통과했다.

### 4.2 Current controller gate — fail

다음을 이유로 현재 Full controller를 main table method로 승격하지 않는다.

- Llama에서 Full `0/4`, Static `2/4`로 current acquisition이 붕괴했다.
- `S_max=8`이 affected Llama cases를 하나도 구하지 못했다.
- Qwen positive signal이 Llama에 재현되지 않았다.
- compute가 yellow이고 eff/gen/loc/retention은 측정하지 않았다.

### 4.3 Kill된 설명과 열린 설명

| 설명 | 판정 |
|---|---|
| 고정 update를 잘게 나눈 것뿐이다 | kill: direction IDs가 매 refresh에서 전부 변함 |
| Llama는 단지 round가 2개 부족했다 | kill: `S_max=8` continuation `0/3` |
| 현재 Full controller가 두 모델 공통으로 우수하다 | kill |
| state-dependent field를 유용한 allocation으로 바꿀 여지가 있다 | open; Qwen local signal |
| ODE-Edit이 preservation/eff/gen/loc를 개선한다 | 미검증 |

## 5. 다음 실험 방향

### 5.1 즉시 중단

- `S_max>6` round-count tuning
- Llama/Qwen별 threshold, step, fallback 또는 controller branch
- 10/100-edit expansion, AlphaEdit extension, lifelong scale
- 현재 controller를 사용한 main performance table

### 5.2 Common-controller redesign diagnostic

다음 revision은 두 모델에 동일한 수식과 상수를 사용한다.

1. Llama에서 direction refresh가 allocation 변화로 전달되지 않는 원인을 QP 입력의
   progress scale, capacity marginal cost, trust acceptance별로 분해한다.
2. Progress와 capacity term을 layer/model scale에 덜 민감한 공통 normalization으로
   다시 정의한다. 결과를 본 model alias 분기는 금지한다.
3. `S_max=6`, original BF16, fixed direct-z, two-forward event, full-linear commit emulator와
   first-hit rule은 유지한다.
4. Proposal-build batching/reuse는 exact same-snapshot semantics와 counters를 보존하는
   범위에서만 허용한다.
5. 기존 4 case는 implementation/debug panel로만 쓰고, controller를 고정한 뒤 별도의
   sealed fresh small panel로 confirm한다.

### 5.3 다음 gate

다음 작은 confirmatory panel에서 모두 만족해야 MI/10-edit로 넘어간다.

- 두 모델 모두 Full first-hit collapse가 없고 Static보다 materially 열세가 아님
- 두 모델 각각 최소 하나의 same-entry first-hit comparison에서 capacity frontier가
  non-worse이며 pooled direction은 positive
- refresh-induced allocation/ranking drift가 numerical noise를 넘고 progress와 정렬
- `Full/Native <=4x`; `<=3x`를 목표로 하며 yellow면 명확한 frontier gain 필요
- model-specific rescue/search/fallback `0`
- direct-z once/edit, rollback, first-hit freeze, original BF16과 EasyEdit read-only 유지

재설계 뒤에도 Llama Full이 Static에 명확히 지거나, allocation이 사실상 static이고
frontier gain이 없으면 ODE controller를 kill하고 Static synchronous 또는 One-refresh로
pivot한다.

## 6. 최종 claim boundary

이번 결과로 허용되는 문장은 다음 하나다.

> ODE-Edit의 joint-state refresh는 두 모델에서 layer proposal을 비자명하게 바꾸며,
> 한 Qwen matched-entry case에서는 first-hit을 유지하면서 Native보다 낮은 C-capacity
> endpoint를 찾았다. 그러나 현재 공통 controller는 Llama에서 실패했으므로 method
> superiority가 아니라 controller redesign을 정당화하는 motivation signal이다.

다음은 주장하지 않는다.

- MEMIT/AlphaEdit 대비 efficacy, generalization, locality 또는 retention 우위
- 두 모델 공통 expected gain
- compute reduction 또는 deployable efficiency
- sequential/lifelong collapse 방지
- `S_max=8` 또는 모델별 hparam 필요성
