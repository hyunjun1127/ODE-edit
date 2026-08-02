# Session 01 capacity/history — Qwen c0_v3 GH 분석

- 날짜: 2026-08-02
- 모델: `qwen2.5-7b-inst`
- 방법명: **ODE-Edit**
- 판정: **technical pass; analyzer harm signal; under-edit contract로 superseded**
- claim boundary: same-policy 4-edit Motivation diagnostic

## 연구 질문과 근거

Canonical plan은
`plans/global/2026-08-02-session01-capacity-history-closure-spec.md`다. Llama와
byte-identical policy/config를 사용해 native writer와 cumulative-capacity QP를
비교했다. Alpha native/QP에는 같은 accepted-edit historical term을 적용했다.

## 실행과 artifact

- controller: job `15819`, commit `9900a51`, 네 branch 모두 4/4 terminal/pass.
- evaluator-only recovery: job `15824`, `COMPLETED 0:0`.
- recovery wrapper commit: `2bd1d31`; actual evaluator는 locked `9900a51`이다.
- raw evaluator root:
  `local/results/raw/session01_motivation/caphist_eval_*_qwen_c0_v3/`.
- combined analysis:
  `local/results/raw/session01_motivation/caphist_combined_qwen_c0_v3/analysis.json`.
- analysis SHA-256:
  `70cb345924da0cfac875c335fa42a4af31766534ef034c48b7302e747f21ea5a`.
- seed `41`, edit order `13899, 21196, 15624, 12487`, layers `4--8`.
- raw artifact broadcast: peer SH/clone 부재로 no-peer exception.

Terra Ultra Qwen analyst는 runtime metadata를 확인할 수 없어 `analysis.json`을
읽지 않고 `BLOCK`으로 종료했다. 아래는 GH fallback 분석이다.

## Technical validity

- 네 branch × 4 edits = 16 checkpoints, technical failure `0`.
- evaluator summary 4/4가 `completed`, `all_pass=true`, checkpoint `4`다.
- evaluation firewall, W0/state lineage, precomputed-only, canonical Alpha history,
  capacity barrier와 finite metric이 모두 통과했다.
- raw evaluation text/logit/token을 저장하지 않았고 NFE field도 없다.
- locked staging과 main evaluator directory는 4/4 byte-identical하다.

## QP minus native 효과

| 축 | MEMIT | Alpha-history |
|---|---:|---:|
| current edit utility mean delta | `-3.710800` | `-3.321125` |
| final all-edit utility delta | `-3.708018` | `-3.316262` |
| final prior-retention delta | `-3.739605` | `-3.210839` |
| retention AUC delta | `-3.680108` | `-3.363195` |
| neighborhood KL reduction | `+0.283246` | `+0.329390` |
| generation KL reduction | `+1.744179` | `+1.525872` |
| target-true NLL drift reduction | `+5.025681` | `+4.814020` |
| final capacity reduction | `+0.000938` | `+0.002020` |
| max-layer-share reduction | `+0.014531` | `+0.213287` |
| layer-Gini reduction | `+0.005713` | `+0.148297` |
| cumulative Frobenius reduction | `+7.249472` | `+6.728575` |

두 family 모두 current utility non-collapse floor `-0.10`을 크게 실패한다.
MEMIT delta는 네 edit 모두 음수
`[-3.2361, -4.9213, -3.0726, -3.6133]`, Alpha-history도 모두 음수
`[-3.0224, -4.7425, -1.8871, -3.6325]`다. Exact first-hit는 native/QP 모두
각 family `4/4`였지만 QP의 margin/utility는 훨씬 낮았다. 즉 top-1 도달만으로
matched efficacy를 보장하지 못한다.

## Routing과 compute

- QP overload observation `0`, suppressed overload `0`, reroute round `0`.
- capacity barrier violation은 두 family 모두 `0`이다.
- cost/concentration 감소는 관측됐지만 hard overload routing의 empirical test는
  발동하지 않았다.
- QP proposal build는 family별 `11`회, accepted round는 `7`회, reject는 0회다.
- 누적 evaluator wall은 QP/native가 MEMIT `2.88×`, Alpha `2.24×`다.

## 해석과 판정

Qwen에서도 QP는 더 적게 쓰고 KL/capacity/norm을 줄였지만, 동일 edit utility를
유지하지 못했다. 특히 first-hit 4/4인데도 utility가 크게 낮아 현재 free-terminal
criterion이 너무 약하다는 사실이 드러났다. 이는 successful rerouting이 아니라
under-edit confound다.

- MEMIT verdict: `CAPACITY_QP_HARM_SIGNAL`.
- Alpha-history verdict: `CAPACITY_QP_HARM_SIGNAL`.
- model verdict: `CAPACITY_HISTORY_MODEL_NO_SIGNAL`.

Llama와 다른 rescue policy를 허용하지 않는다. Qwen c0 accepted path도 native의
평균 약 `40%`뿐이어서 다음 단계 근거나 method kill로 사용할 수 없다. 동일한 c1
matched-quarter policy만 재실행하며, 4-edit 결과를 lifelong 또는 broad preservation
claim으로 확장하지 않는다.
