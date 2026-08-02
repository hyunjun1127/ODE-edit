# Session 01 capacity/history pair c1_v3 — final synthesis

- 날짜: 2026-08-02
- 방법명: **ODE-Edit**
- 모델: `llama3-8b-inst`, `qwen2.5-7b-inst`
- policy: `capacity-qp-history-k4-native-progress-v2`
- technical verdict: **pass**
- scientific verdict: **`SUPERSEDED_IMPLEMENTATION_CONFOUNDED_LOW_UPDATE`**
- Motivation status: **historical/superseded; final resolution in c3 magnitude-only control**
- claim boundary: same-policy 4-edit diagnostic only

> **2026-08-02 GH 정정:** c1은 layer별 상대 share와 global update magnitude를 같은
> coefficient로 묶었고, full requested progress가 infeasible할 때 더 작은 feasible
> target을 scientific endpoint로 허용했다. Llama는 두 family 모두 feasible round가
> `0/16`, overloaded observation이 0인데도 realized path가 native의 약 67--69%였다.
> 따라서 아래의 원래 `HARM_SIGNAL`/closed-negative 해석은 보존된 당시 판정이지만 현재
> method kill evidence로는 무효다. 수치와 raw artifact는 변경하지 않으며, corrected
> 판정은 exact `D/4 × 4` c2에서만 내린다.

> **c3 최종 resolution:** c1 BF share를 유지하고 magnitude만 분리한 c3가 네
> model×family cell 모두 c1보다 회복했다. Canonical final verdict는
> `experiment-reports/global/2026-08-02-session01-caphist-pair-c3-v1-synthesis.md`다.

## 결론 먼저

구현은 수정됐다. c1은 매 round에서 고정 fraction이 아니라 ordered-native endpoint의
남은 rewrite utility 전체를 요청하고, 최대 네 번의 `D/4` trust-cap round를 쓰며,
exact-top1에서 조기 종료하지 않는다. 16개 QP 편집 모두 이 계약을 통과했다.

수정 후 c0보다 두 모델 모두 크게 회복했다. 특히 Qwen Alpha-history는 current utility
`+0.029707`과 공통 cost/KL 감소를 함께 보여 작은 model-level signal을 냈다. 그러나
Llama의 MEMIT/Alpha current utility는 `-4.032392/-2.495123`, Qwen MEMIT도
`-0.397837`이다. 사용자 원칙대로 같은 method가 두 모델에 작동해야 하므로 passing
family는 0개이고 pair verdict는 harm signal이다.

따라서 old implementation bug는 닫혔지만 현재 fixed `K=4` capacity-QP sequential
controller는 Motivation을 통과하지 못했다. Atomic direct-z/relinearization possibility는
별도 mechanism evidence로 남지만, deployable ODE-Edit method superiority는 성립하지
않는다.

## 네 범주

### Proposal에서 온 내용

- finite edit를 current-state-dependent trajectory로 보고 direction을 재계산한다.
- 동일 direct-z의 layer realization 사이에서 cumulative capacity marginal cost가 낮은
  쪽으로 progress를 배분한다.
- MEMIT과 AlphaEdit null-space/history에 모델 공통 controller를 적용한다.
- efficacy를 유지하면서 capacity/preservation을 개선해야 다음 단계로 간다.

### Repo/protocol에서 확인한 사실

- job `15855`가 4 GPU에서 Llama/Qwen × MEMIT/Alpha를 동시에 실행해
  `COMPLETED 0:0`으로 끝났다.
- 8 controller와 8 evaluator가 모두 terminal/pass이고 evaluator checkpoint는 총 32개다.
- EasyEdit는 read-only였고 precomputed covariance/projector/Wikipedia artifact만 썼다.
- evaluation firewall, exact rollback/lineage, direct-z-once, Alpha history append,
  barrier/cap checks가 모두 통과했다.
- Llama와 Qwen은 byte-identical controller policy를 사용했다.

### GH 추정

- c0→c1의 큰 회복은 fixed fractional request와 weak terminal이 실제 confound였음을
  지지한다.
- c1의 남은 trade-off는 QP가 보존/용량 제약 아래 nominal total cap `D`를 실제로
  전부 쓰지 못한 결과와 일치한다. 이는 technical bug가 아니라 현재 objective와
  feasible set의 method limitation이다.
- Qwen Alpha의 양성만으로 모델 공통 method를 만들 수 없고, Llama별 threshold나
  branch를 추가하면 사용자 원칙과 사전등록 gate를 위반한다.

### 사용자 확인 필요

- 없음. 이번 Motivation 결과로 model-specific rescue, 추가 K/threshold/case retune,
  lifelong 확대를 자동 승인하지 않는다.
- Exact realized-length control은 향후 새 Method redesign의 ablation 후보일 수 있지만,
  이 Motivation을 다시 여는 자동 후속 실험은 아니다.

## Pair 결과

모든 수치는 QP minus native다. Reduction 양수는 QP가 해당 drift/cost를 낮췄다는
뜻이다.

| Model | Family | current utility | final all-edit | prior retention | neighborhood KL reduction | generation KL reduction | capacity reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama | MEMIT | `-4.032392` | `-4.041824` | `-3.862633` | `+0.026321` | `+0.972480` | `+0.000087` |
| Llama | Alpha-history | `-2.495123` | `-2.492299` | `-2.708148` | `+0.055909` | `+1.027351` | `+0.000660` |
| Qwen | MEMIT | `-0.397837` | `-0.387928` | `-0.579252` | `+0.025233` | `+0.263864` | `+0.000729` |
| Qwen | Alpha-history | `+0.029707` | `+0.024499` | `-0.431721` | `+0.067991` | `+0.101347` | `+0.001679` |

두 모델 공통 positive cost axes는 capacity, generation/neighborhood KL,
layer-Gini, max-layer-share, target-true NLL drift 감소다. Alpha-history에서는
cumulative Frobenius도 공통 감소한다. 하지만 family gate는 먼저 양 모델 current
non-collapse `>= -0.10`을 요구한다. Llama가 두 family 모두 실패하고 Qwen MEMIT도
실패하므로 common cost axes로 이를 우회할 수 없다.

## 구현 계약과 realized distance

| Model | Family | accepted rounds | mean path/native | native match | budget exhausted |
|---|---|---:|---:|---:|---:|
| Llama | MEMIT | `16` | `0.671819` | `0/4` | `4/4` |
| Llama | Alpha-history | `16` | `0.686245` | `0/4` | `4/4` |
| Qwen | MEMIT | `16` | `0.863935` | `1/4` | `3/4` |
| Qwen | Alpha-history | `15` | `0.742208` | `1/4` | `3/4` |

- 모든 accepted diagnostic에서 `requested_gain == remaining_reference_gain_before`다.
- unmatched endpoint는 모두 4 accepted rounds와 `budget_exhausted=true`다.
- first-hit은 terminal이 아니며, Qwen Alpha 한 편집만 native utility match 뒤 3 rounds로
  종료했다.
- retry/reject는 0회이며 barrier violation은 numerical zero다.

`D/4`는 round별 trust **상한**이다. 네 round의 합을 정확히 `D`로 강제하지 않는다.
따라서 realized path가 작다는 사실을 숨기지 않지만, c0처럼 애초에 75%만 요청하거나
3 rounds에서 끝난 구현 오류와는 다르다. Solver가 full gap을 요청받고도 제약 frontier
안에서 native utility를 못 맞춘 편집은 사전등록대로 scientific negative다.

## c0에서 얼마나 회복했는가

| Model | Family | c0 current delta | c1 current delta | c0 path/native | c1 path/native |
|---|---|---:|---:|---:|---:|
| Llama | MEMIT | `-6.428714` | `-4.032392` | `0.454052` | `0.671819` |
| Llama | Alpha-history | `-5.663860` | `-2.495123` | `0.457809` | `0.686245` |
| Qwen | MEMIT | `-3.710800` | `-0.397837` | `0.399978` | `0.863935` |
| Qwen | Alpha-history | `-3.321125` | `+0.029707` | `0.409667` | `0.742208` |

재구현은 결과를 실질적으로 바꿨다. 그러므로 c0는 계속 superseded 상태다. 동시에
c1에서도 Llama 공통성이 살아나지 않았으므로 단순 under-edit bug 수정만으로
ODE-Edit sequential claim을 구제할 수 없다는 결론도 얻었다.

## Routing과 compute

- Llama overload/suppression/reroute: 두 family 모두 `0/0/0`.
- Qwen overload/suppression/reroute: 두 family 모두 `1/1/1`.
- model-common hard rerouting evidence는 없다.
- QP/native wall ratio: Llama MEMIT `6.34×`, Alpha `4.65×`; Qwen MEMIT `6.09×`,
  Alpha `4.41×`.
- native proposal build는 family/model당 4회, QP는 `19--20`회다.

작은 Qwen Alpha point gain만으로 4--6배 compute를 정당화하지 않는다.

## Gate와 kill decision

| Family | Llama non-collapse | Qwen non-collapse | common proxy | pair gate |
|---|---|---|---|---|
| MEMIT | fail | fail | 있음 | fail |
| Alpha-history | fail | pass | 있음 | fail |

- **Kill:** 현재 `capacity-qp-history-k4-native-progress-v2` fixed-`K=4`
  capacity-QP sequential controller의 model-common Motivation claim.
- **Kill:** 현재 결과에 대한 preservation guarantee, lifelong, deployable superiority,
  overloaded-layer routing claim.
- **보존:** exact QP/capacity algebra, state relinearization hook, canonical Alpha history,
  Qwen Alpha local possibility signal, c0→c1 implementation RCA.
- **금지:** 모델별 rescue, post-hoc threshold/K/case sweep, 더 큰 sequential scale.

## Artifact, agent, Direct-z 경계

- Llama analysis SHA-256:
  `42b63463198c53c39880f7bef1d11d3cd93c9f1fed1a2ea6404b469c7469b78b`.
- Qwen analysis SHA-256:
  `58b25722f69d1e11de3c8360365632a9b5321cd7e805c6b42184934234ee400f`.
- Pair analysis SHA-256:
  `9f0d6cc5d4073aae1c8f103c2c669322be57c6cdd7a40b146a94b4def7bbd0e0`.
- 29 compact files aggregate SHA-256:
  `745637bfb654abdcf2fe8c075d2130b126739854f75eaa51b04b83c3b7b95113`.

Terra Ultra Llama/Qwen/pair agents는 runtime metadata를 검증하지 못해 모두 지정
파일을 읽지 않고 종료했다. 독립 analysis pass는 0건이며 모델별 문서는 GH fallback이다.

Direct-z 임시 session의 atomic evidence는 수치나 gate에 합치지 않았다. 그 report의
“atomic/local mechanism 가능성”과 본 report의 “현재 sequential controller
cross-model 실패”는 서로 다른 claim boundary다.

## 최종 판정

원래 판정은 **closed-negative**였으나, 위 GH 정정에 따라
`SUPERSEDED_IMPLEMENTATION_CONFOUNDED_LOW_UPDATE`로 대체한다. c1은 current
absolute-cap controller가 update를 줄인다는 증거로만 보존하며, BF layer share나
direction refresh의 scientific negative로 사용하지 않는다. Motivation 최종 판정은
exact-distance c2 결과까지 보류한다.
