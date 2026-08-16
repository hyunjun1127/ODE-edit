# P1R19 Strength-Preserving Router — SH2 terminal report

## 결론

FACT — TECH-R1 array job 18434의 8개 Dynamic trajectory가 모두 K=8, tau=1, exit 0으로 끝났다. 모든 cell은 8회 authoritative BF16 write, retry/backtracking/reject 0, first-hit observation-only, action-freeze 후 k0…k8 평가, 마지막 W0 restore PASS를 기록했다.

INFERENCE — 같은 `alpha_apply`를 요구하는 Strength-Preserving Soft는 “보존을 유지하면서 layer share만 바꾼다”는 주가설을 전반적으로 지지하지 않았다. Soft가 공식 지표를 개선한 유일한 endpoint는 Llama/RS의 Gen 18/20→20/20이며, Loc은 모든 모델·allocation에서 Neutral과 같았다. 반면 endpoint structural-P와 cumulative capacity는 네 쌍 모두 Soft가 높았고, functional-P mean은 Llama/RS에서만 명확히 낮았다.

BOUNDARY — 재사용된 outcome-selected R13 seal에 대한 mechanistic/descriptive diagnostic이다. scientific promotion=false. R13 대비 차이는 floor 제거 + physical W-only slope + strength equality를 묶은 full method-package delta이며, 단일 구성요소의 인과효과로 해석하지 않는다.

## 고정 provenance

- instruction: `ODEEDIT-S05-ODE-BF-STRENGTH-PRESERVING-ROUTER-P1R19-V1`, A1/A2
- scientific checkpoint: `657dafd40de50bb396fc9cfaf3862de1d07dd816`
- TECH-R1 launcher checkpoint: `d604953a046b6af4da7386dd6c64c20fc848f625`
- R13 base: `e6facd2d5dfae12d3c094b51981ad99951174109`
- seal root: `3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628`
- request order: `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`
- numerical lock root: `1b0f7c5b...`; source-manifest root: `aabbd8e...`
- array 18434 task order: Llama RS-N/RS-S/BG-N/BG-S, Qwen RS-N/RS-S/BG-N/BG-S
- 실패한 최초 launcher array 18423은 immutable이며 전 task model 전 server/node provenance 오류였다. 결과에 섞지 않았다.

## pre-edit와 endpoint 공식 지표

W0는 alias 내 모든 cell에서 동일했다.

| alias | state | W-only Eff | Gen | Loc |
|---|---|---:|---:|---:|
| Llama | W0 | 2/10 | 3/20 | 84/100 |
| Qwen | W0 | 0/10 | 4/20 | 80/100 |

Native N32 postfreeze efficacy는 두 alias 모두 10/10이다. 이 실행의 local receipt에는 Native Gen/Loc가 기록되지 않아 NOT_RECORDED이다.

| alias | allocation | router | W Eff | W Gen | W Loc | z-oracle Eff | z-oracle Gen | controller new-NLL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Llama | RS | Neutral | 10/10 | 18/20 | 82/100 | 10/10 | 20/20 | 1.002913 |
| Llama | RS | Soft | 10/10 | 20/20 | 82/100 | 10/10 | 20/20 | 0.483299 |
| Llama | BG | Neutral | 10/10 | 20/20 | 81/100 | 10/10 | 20/20 | 0.119672 |
| Llama | BG | Soft | 10/10 | 20/20 | 81/100 | 10/10 | 20/20 | 0.053443 |
| Qwen | RS | Neutral | 10/10 | 20/20 | 79/100 | 10/10 | 20/20 | 0.012528 |
| Qwen | RS | Soft | 10/10 | 20/20 | 79/100 | 10/10 | 20/20 | 0.011883 |
| Qwen | BG | Neutral | 10/10 | 20/20 | 80/100 | 10/10 | 20/20 | 0.061009 |
| Qwen | BG | Soft | 10/10 | 20/20 | 80/100 | 10/10 | 20/20 | 0.104315 |

각 cell의 k0…k8 공식 count와 target-new/old NLL mean, receipt-defined margin vector는 동봉 `cell-*.json`에 있다. Locality margin은 evaluator의 canonical `new-old`, Eff/Gen margin은 `old-new` 정의를 유지했다.

## 같은 alpha에서 Soft 대비

| alias/allocation | Soft의 공식 변화 (Eff/Gen/Loc) | structural-P N→S | functional-P mean N→S | capacity N→S |
|---|---|---:|---:|---:|
| Llama/RS | 0, +2, 0 | .000756→.001110 | .003140→.002637 | .528→.932 |
| Llama/BG | 0, 0, 0 | .003517→.003965 | .007087→.007408 | 2.426→3.114 |
| Qwen/RS | 0, 0, 0 | .141306→.164709 | .057567→.057796 | 15.635→18.339 |
| Qwen/BG | 0, 0, 0 | .152006→.152152 | .075607→.077028 | 16.620→16.716 |

FACT — functional-P는 모두 과거 1e-3 값보다 높지만 P/H는 이 실험에서 observation-only라 endpoint를 veto하지 않았다. H는 empty-history로 inactive였다.

FACT — Llama/BG Soft의 P smooth/raw는 Neutral보다 조금 낮지만 mean P와 structural-P는 높아 “P 개선”으로 단정할 수 없다. Qwen/BG Soft는 target-new NLL도 악화했다.

INFERENCE — Soft allocation의 보존 이득은 일반화되지 않았다. Llama/RS에서 functional-P와 Gen이 좋아진 국소 신호는 있으나, 구조 손상과 capacity 증가 및 다른 세 쌍의 무이득/악화 때문에 주가설은 HOLD다.

## R13 Dynamic counterpart 대비

| alias/cell | R13 last valid W E/G/L | P1R19 K8 W E/G/L | delta |
|---|---|---|---|
| Llama RS-N | 9/17/82 @ k8 | 10/18/82 | +1/+1/0 |
| Llama RS-S | 10/17/82 @ k8 | 10/20/82 | 0/+3/0 |
| Llama BG-N | 8/15/83 @ k6 | 10/20/81 | +2/+5/-2 |
| Llama BG-S | 10/18/83 @ k8 | 10/20/81 | 0/+2/-2 |
| Qwen RS-N | 5/10/80 @ k4 | 10/20/79 | +5/+10/-1 |
| Qwen RS-S | 4/7/81 @ k3 | 10/20/79 | +6/+13/-2 |
| Qwen BG-N | 9/17/81 @ k8 | 10/20/80 | +1/+3/-1 |
| Qwen BG-S | 8/14/80 @ k7 | 10/20/80 | +2/+6/0 |

FACT — R13에서 중도 종료된 cell은 last-valid prefix와 비교했다. 특히 Qwen BG-S k8은 합성·보간하지 않았다. R13 reference packet SHA-256은 `386521002648d7599af530f382a5c1b81486923a052977f20bf78b4cac1d7f91`이다.

INFERENCE — full-strength/matched-strength package는 기존 R13의 under-edit를 크게 줄였다. 다만 locality는 대부분 1–2 point 낮아졌고, 이것은 보존 개선이 아니라 efficacy-strength trade-off다.

## routing·실현

FACT — k0에서 Neutral/Soft pair는 alias×allocation별 field/problem/slope/alpha identity가 exact match였다. 그 뒤 각 trajectory가 독립적으로 다른 W,z state를 만들므로 이후 step의 Soft-vs-Neutral은 같은 상태의 pointwise 비교가 아니라 trajectory-level method contrast다.

FACT — endpoint coefficient Neff는 4.78–5.00이고 top1 share는 .20–.249로 terminal collapse는 없었다. Qwen은 많은 step이 DOF=0/all-cap 상태여서 Soft가 실제 share를 바꿀 여지가 작았다. 네 Qwen trajectory의 endpoint velocity는 모두 다섯 layer cap에 붙었다.

FACT — 최소 coverage는 Llama RS .795, Llama BG .064/.111, Qwen RS .125/.029, Qwen BG .276/.412였다. demand가 alpha_max를 넘은 경우 coverage를 솔직히 낮춰 기록했으며 equality는 alpha_apply에 대해 통과했다.

FACT — Qwen BG-Soft k7에서 physical actual progress가 -0.0733, predicted alpha가 +0.6745였다. locked no-retry contract에 따라 그대로 수용·기록했으며 scientific rescue를 하지 않았다.

FACT — 모든 cell은 pmax=0 recovery 0, retry/reject 0이다. 첫 exact efficacy hit는 Llama BG k4, Llama RS k6, Qwen RS k5, Qwen BG k6였고 모두 k8까지 계속했다.

## 비용

단위는 초. `edit_core`는 z/field refresh, slope reuse, route solve, BF16 write를 포함하며 evaluator와 functional probe는 분리했다.

| alias/cell | edit_core | func probe | stepwise eval | load | accounted wall | scheduler | MaxRSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama RS-N | 2769.2 | 87.2 | 255.0 | 9.1 | 4218.9 | 4245 | 8.44 GB |
| Llama RS-S | 2741.9 | 87.2 | 245.8 | 10.3 | 4132.6 | 4158 | 8.42 GB |
| Llama BG-N | 2609.9 | 81.9 | 229.4 | 10.0 | 3913.4 | 3938 | 9.78 GB |
| Llama BG-S | 2558.0 | 81.2 | 232.9 | 8.9 | 3863.0 | 3889 | 8.51 GB |
| Qwen RS-N | 2864.7 | 88.1 | 268.3 | 14.4 | 4370.1 | 4403 | 21.06 GB |
| Qwen RS-S | 2840.5 | 87.1 | 255.1 | 10.2 | 4276.7 | 4306 | 13.96 GB |
| Qwen BG-N | 3131.5 | 95.0 | 293.5 | 9.2 | 4762.9 | 4795 | 12.06 GB |
| Qwen BG-S | 2741.4 | 84.6 | 279.6 | 9.9 | 4298.0 | 4331 | 11.99 GB |

FACT — rollout count는 cell당 model forward 2194, backward 240, target backward 80, functional-P replay 64였다. stepwise evaluation은 model forward 789/evaluator forward 780이었다. Router 자체의 추가 forward/backward는 0으로 fail-close 검증됐다.

INFERENCE — 비용의 대부분은 K8 state/field refresh와 candidate/write/evaluation이다. equality QP 자체가 주 병목이라는 증거는 없다.

Native reference edit_core는 약 25.7–29.8초, end-to-end 27.0–31.1초였지만 알고리즘·계측 범위가 달라 이를 근거로 exact pure-edit 배수를 주장하지 않는다.

## 독립 감사와 NOT_RECORDED

- FACT — action-freeze/evaluator identity와 final W0 restore는 8개 모두 PASS.
- FACT — `router_added_processed_token_count` serializer가 singular key를 읽는 schema defect가 있다. 실제 ledger의 plural-key delta는 router solve 전후 fail-close가 검사하므로 실행 결정에는 영향이 없지만, terminal의 token-zero 필드 자체는 독립적인 직접 계측으로 보지 않는다.
- FACT — 30/64 transition은 `feasible_allocation_dimension=0`이었지만 status priority 때문에 `WRITER_UNREACHABLE`로 기록됐다. 따라서 `NO_ROUTING_DOF` 문자열만으로 DOF 고갈을 세면 안 된다.
- FACT — initial-contract schema 명칭이 RS/Neutral cell에도 legacy `bg-soft` literal을 쓴다. 내용·hash pair는 맞지만 nomenclature defect다.
- FACT — panel의 accepted snapshot count 8은 write 이후 상태 수이며 evaluator 파일은 entry k0을 포함해 9개다. 72개 전체 W/z evaluator hash link는 검증됐다.
- NOT_PROVEN — positive-but-1e-8 이하 alpha의 zero-write totality, zero-demand certificate, Soft tie의 완전한 deterministic uniqueness는 model run에서 운동되지 않았고 source audit도 완전 증명을 주지 못했다.
- NOT_PROVEN — separate-job Neutral/Soft의 모든 후속 field identity는 성립할 수 없다. exact common-field pair identity는 k0에서만 증명됐다.
- NOT_RECORDED — P1R19 local Native Gen/Loc, standalone restore receipt(hash)는 없다. restore는 terminal boolean과 source-side byte comparison gate로만 증명된다. fresh unseen seal evidence도 없으며, 이 실험은 기존 R13 outcome-selected seal을 의도적으로 재사용했다.

## 최종 판정

FACT — P1R19은 R13 대비 efficacy/generalization under-edit를 제거하거나 크게 완화했다.

INFERENCE — 그러나 “동일 semantic strength에서 Soft redistribution이 W-only Eff/Gen을 보존하면서 Loc/P/H damage를 낮춘다”는 질문에는 **대체로 아니오**다. Llama/RS의 제한적 Gen/functional-P 이득 외에는 official metric 이득이 없고, structural-P/capacity는 일관되게 악화했다.

CLAIM — `MECHANISTIC_DESCRIPTIVE_ONLY / SCIENTIFIC_PROMOTION_FALSE / METHOD_HOLD`.
