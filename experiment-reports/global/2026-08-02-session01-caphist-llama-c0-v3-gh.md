# Session 01 capacity/history — Llama c0_v3 GH 분석

- 날짜: 2026-08-02
- 모델: `llama3-8b-inst`
- 방법명: **ODE-Edit**
- 판정: **technical pass; analyzer harm signal; under-edit contract로 superseded**
- claim boundary: same-policy 4-edit Motivation diagnostic

## 연구 질문과 근거

Canonical plan은
`plans/global/2026-08-02-session01-capacity-history-closure-spec.md`다. 동일한
네 case에서 native writer와 cumulative-capacity QP를 비교하고, Alpha track은
native/QP 양쪽에 동일한 canonical historical term을 넣었다. EasyEdit source는
수정하지 않았고 precomputed covariance/projector만 read-only로 사용했다.

## 실행과 artifact

- controller: job `15819`, commit `9900a51`, 네 branch 모두 4/4 terminal/pass.
- evaluator-only recovery: job `15824`, `COMPLETED 0:0`.
- recovery wrapper commit: `2bd1d31`; 실제 evaluator는 controller와 같은 locked
  commit `9900a51`의 unmodified evaluator다.
- raw evaluator root:
  `local/results/raw/session01_motivation/caphist_eval_*_llama_c0_v3/`.
- combined analysis:
  `local/results/raw/session01_motivation/caphist_combined_llama_c0_v3/analysis.json`.
- analysis SHA-256:
  `4931235b2dcb224fb93bef739cc408252ea343ea122a4a67aa72b7f85e33c0c0`.
- seed `41`, edit order `13899, 21196, 15624, 12487`, layers `4--8`.
- raw artifact broadcast: 미실시. 현재 peer SH/clone이 없어 no-peer exception이다.

Terra Ultra Llama analyst를 compact `analysis.json` 하나에만 격리 호출했으나
runtime metadata를 검증할 수 없어 파일을 읽지 않고 종료했다. 따라서 아래는
독립 agent report가 아니라 GH fallback 분석이다.

## Technical validity

- 네 branch × 4 edits = 16 checkpoints, `pass=true`.
- technical boolean failure `0`; evaluation firewall, state lineage, W0 anchor,
  covariance/projector read-only, Alpha history exact, rollback/barrier 모두 통과했다.
- evaluator summary 4/4가 `all_pass=true`; raw text/logit/token은 저장하지 않았다.
- NFE field는 raw compact artifact, analysis, gate에 없다.
- locked staging과 main evaluator directory는 4/4 recursive byte-identical하다.

## QP minus native 효과

양수인 `*_reduction`은 QP가 native보다 해당 cost/proxy를 줄였다는 뜻이다.

| 축 | MEMIT | Alpha-history |
|---|---:|---:|
| current edit utility mean delta | `-6.428714` | `-5.663860` |
| final all-edit utility delta | `-6.443999` | `-5.659343` |
| final prior-retention delta | `-6.586881` | `-5.667065` |
| retention AUC delta | `-4.944429` | `-4.220787` |
| neighborhood KL reduction | `+0.064093` | `+0.155460` |
| generation KL reduction | `+2.033320` | `+1.797439` |
| target-true NLL drift reduction | `+5.072589` | `+5.512439` |
| final capacity reduction | `+0.000127` | `+0.000949` |
| max-layer-share reduction | `+0.272023` | `+0.246791` |
| layer-Gini reduction | `+0.269696` | `+0.230213` |
| cumulative Frobenius reduction | `+0.896398` | `+1.058205` |

사전등록된 non-collapse floor는 `-0.10`이다. 두 family의 current utility delta는
각각 floor보다 약 64배, 57배 더 음수이므로 lenient gate도 명확히 실패한다.
MEMIT current edit delta는 네 edit 모두 음수
`[-4.0848, -4.1367, -11.4780, -6.0153]`이고, Alpha-history도 네 edit 모두 음수
`[-3.2531, -4.1540, -9.6122, -5.6362]`다.

## Routing과 compute

- QP overload observation `0`, suppressed overload `0`, reroute round `0`.
- barrier max violation은 MEMIT `1.69e-21`, Alpha `3.39e-21`로 numerical zero다.
- `all_observed_positive_overloads_suppressed=true`는 관측이 0인 vacuous pass이며,
  “찬 layer를 실제로 우회했다”는 evidence가 아니다.
- first-hit은 MEMIT native `3/4` 대 QP `1/4`, Alpha native `4/4` 대 QP `2/4`다.
- QP proposal build는 family별 `15`회, accepted round는 `11`회이고 reject는 0회다.
  Native는 각각 4회다.
- 누적 evaluator wall은 QP/native가 MEMIT `4.30×`, Alpha `3.20×`다.

## 해석과 판정

QP는 더 작은 path와 더 낮은 capacity/KL/Frobenius를 만들었지만 native와 같은
edit strength를 맞추지 못했다. 각 step의 local gain이 양수라는 acceptance만으로는
native margin을 보존하지 못했고, overload barrier는 한 번도 발동하지 않았다.
따라서 positive cost 축은 capacity rerouting 효과가 아니라 **under-edit**로 설명할
수 있다.

- MEMIT verdict: `CAPACITY_QP_HARM_SIGNAL`.
- Alpha-history verdict: `CAPACITY_QP_HARM_SIGNAL`.
- model verdict: `CAPACITY_HISTORY_MODEL_NO_SIGNAL`.

Llama c0는 Motivation positive evidence로 사용할 수 없고, accepted path가 native의
평균 약 `45%`뿐이므로 ODE-Edit method kill에도 사용할 수 없다. Alpha historical
term도 이 strength mismatch를 제거하지 못했다. c1 전까지 lifelong, preservation
guarantee, deployable method superiority claim은 금지한다.
