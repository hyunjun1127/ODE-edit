# P1R52 PIR-U Sequential 10×B100 Structural-H Post-Energy-Warn R1 — 최종 사실 보고서

> Llama3-8B-Instruct, 동일 봉인 1,000-request stream, B100×10, K8. 이 실행은 Structural-H 최적화의 energy 제약은 유지하되 post-solve energy 재검산만 WARN으로 바꾼 명시적 과학 방법 수정입니다.

## 완료 상태와 핵심 사실

- job `20885`는 `COMPLETED/0:0`, 배치 `10/10`, 요청 `1,000/1,000`, 물리 전환 `80/80`, H 결정 `80/80`로 끝났습니다.
- W0 pointer/parameter bytes가 모두 정확히 복원됐고, retry/backtracking/failure endpoint는 `0/0/0`입니다.
- post-energy WARN은 `60/80`회였습니다. 기존 1e-12 hard 경계를 넘은 결정은 `1/80`회이며 최대값은 `1.9768631176475537e-12`(B4-K2)입니다.
- J0 Structural-H ON 대비 final W10에서 GEN은 `+10/2000`, strict GEN은 `+5/1000`이지만, EFF는 `-3/1000`, LOC는 `-744/10000`입니다.
- 따라서 이 봉인 패키지는 `TECHNICALLY_VALID_NON_PARETO_PACKAGE_RESULT`입니다. PIR-U가 rephrase 쪽 일부 지표를 높였지만 rewrite/locality 저하가 함께 있어 우월성 또는 승격을 주장하지 않습니다.

## 공통 original W0와 final W10 — 5행 핵심 표

|방법|EFF|Rewrite Acc|GEN|GEN strict|Rephrase Acc|Acc strict|LOC|
|---|---:|---:|---:|---:|---:|---:|---:|
|공통 original W0 pre-edit|13/100 (13.00%)|0/100 (0.00%)|28/200 (14.00%)|8/100 (8.00%)|1/200 (0.50%)|0/100 (0.00%)|892/1000 (89.20%)|
|Official EasyEdit MEMIT Sequential|942/1000 (94.20%)|819/1000 (81.90%)|1777/2000 (88.85%)|838/1000 (83.80%)|1287/2000 (64.35%)|512/1000 (51.20%)|7214/10000 (72.14%)|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|998/1000 (99.80%)|996/1000 (99.60%)|1909/2000 (95.45%)|929/1000 (92.90%)|1479/2000 (73.95%)|598/1000 (59.80%)|7763/10000 (77.63%)|
|P1R52 J0 Structural-H ON|999/1000 (99.90%)|992/1000 (99.20%)|1696/2000 (84.80%)|751/1000 (75.10%)|1023/2000 (51.15%)|324/1000 (32.40%)|8394/10000 (83.94%)|
|P1R52 PIR-U Structural-H ON / Post-Energy-Warn R1|996/1000 (99.60%)|970/1000 (97.00%)|1706/2000 (85.30%)|756/1000 (75.60%)|1085/2000 (54.25%)|343/1000 (34.30%)|7650/10000 (76.50%)|

`EFF ≡ rewrite_success`, `GEN ≡ paraphrase_success(rephrase_success)`. Accuracy는 teacher-forced strict suffix-token 정확도이며 success와 별도입니다. method-specific batch-entry W_(b-1) aggregate는 사용자 지시에 따라 측정·보고하지 않습니다.

## Immediate-post와 final W10 절대값

|방법|패널|EFF|Rewrite Acc|GEN|GEN strict|Rephrase Acc|Acc strict|LOC|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|Official EasyEdit MEMIT Sequential|10× immediate-post|996/1000 (99.60%)|991/1000 (99.10%)|1825/2000 (91.25%)|862/1000 (86.20%)|1332/2000 (66.60%)|512/1000 (51.20%)|8056/10000 (80.56%)|
|Official EasyEdit MEMIT Sequential|final W10/B1000|942/1000 (94.20%)|819/1000 (81.90%)|1777/2000 (88.85%)|838/1000 (83.80%)|1287/2000 (64.35%)|512/1000 (51.20%)|7214/10000 (72.14%)|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|10× immediate-post|1000/1000 (100.00%)|1000/1000 (100.00%)|1922/2000 (96.10%)|937/1000 (93.70%)|1495/2000 (74.75%)|606/1000 (60.60%)|8177/10000 (81.77%)|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|final W10/B1000|998/1000 (99.80%)|996/1000 (99.60%)|1909/2000 (95.45%)|929/1000 (92.90%)|1479/2000 (73.95%)|598/1000 (59.80%)|7763/10000 (77.63%)|
|P1R52 J0 Structural-H ON|10× immediate-post|998/1000 (99.80%)|995/1000 (99.50%)|1687/2000 (84.35%)|748/1000 (74.80%)|1019/2000 (50.95%)|321/1000 (32.10%)|8586/10000 (85.86%)|
|P1R52 J0 Structural-H ON|final W10/B1000|999/1000 (99.90%)|992/1000 (99.20%)|1696/2000 (84.80%)|751/1000 (75.10%)|1023/2000 (51.15%)|324/1000 (32.40%)|8394/10000 (83.94%)|
|P1R52 PIR-U Structural-H ON / Post-Energy-Warn R1|10× immediate-post|999/1000 (99.90%)|997/1000 (99.70%)|1620/2000 (81.00%)|686/1000 (68.60%)|965/2000 (48.25%)|280/1000 (28.00%)|8297/10000 (82.97%)|
|P1R52 PIR-U Structural-H ON / Post-Energy-Warn R1|final W10/B1000|996/1000 (99.60%)|970/1000 (97.00%)|1706/2000 (85.30%)|756/1000 (75.60%)|1085/2000 (54.25%)|343/1000 (34.30%)|7650/10000 (76.50%)|

## Final W10 target-new/true NLL와 margin

|방법|Rewrite new / true / margin|Rephrase new / true / margin|
|---|---:|---:|
|Official EasyEdit MEMIT Sequential|0.861355 / 9.603316 / 8.741960|1.753286 / 8.282058 / 6.528772|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|0.027169 / 14.741400 / 14.714231|1.262916 / 10.363093 / 9.100177|
|P1R52 J0 Structural-H ON|0.056457 / 10.752489 / 10.696031|2.430740 / 7.162800 / 4.732060|
|P1R52 PIR-U Structural-H ON / Post-Energy-Warn R1|0.155058 / 11.294212 / 11.139154|2.306271 / 7.738423 / 5.432152|

## Accepted-z 직접 주입과 물리 W writer realization

|방법|z rewrite NLL|z rephrase NLL|W immediate rewrite NLL|W immediate rephrase NLL|W−z rewrite gap|W−z rephrase gap|
|---|---:|---:|---:|---:|---:|---:|
|Official EasyEdit MEMIT Sequential|0.002281|0.773273|0.059120|1.750168|+0.056839|+0.976895|
|Official EasyEdit AlphaEdit Sequential (cache_c ON)|0.001437|0.810024|0.001532|1.230092|+0.000095|+0.420068|
|P1R52 J0 Structural-H ON|0.032130|1.271650|0.046224|2.505594|+0.014093|+1.233944|
|P1R52 PIR-U Structural-H ON / Post-Energy-Warn R1|0.035999|1.275457|0.035257|2.695867|-0.000742|+1.420410|

PIR-U writer realization gap(W−z)은 rewrite 평균/중앙값/p90/max `-0.000742/0.000031/0.002686/0.046875`, rephrase `1.420410/0.472778/4.627661/17.545898`입니다.
물리 W가 z보다 NLL이 높은 prompt는 rewrite `518/1000`, rephrase `1698/2000`입니다. gap이 음수면 W가 더 낮은 NLL이므로 손실로 부르지 않습니다.

## PIR-U B1–B10 절대 지표와 z/W gap

|B|history|W post EFF|W post GEN|GEN strict|LOC|z rewrite NLL|z rephrase NLL|W−z rewrite|W−z rephrase|energy WARN|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|B1|0|99/100 (99.00%)|178/200 (89.00%)|81/100 (81.00%)|888/1000 (88.80%)|0.099202|1.487315|-0.010132|+0.872035|0/8|
|B2|100|100/100 (100.00%)|174/200 (87.00%)|78/100 (78.00%)|901/1000 (90.10%)|0.076817|1.490538|+0.000913|+0.865246|6/8|
|B3|200|100/100 (100.00%)|171/200 (85.50%)|75/100 (75.00%)|882/1000 (88.20%)|0.020785|1.228194|+0.000191|+1.156013|6/8|
|B4|300|100/100 (100.00%)|172/200 (86.00%)|76/100 (76.00%)|832/1000 (83.20%)|0.056093|1.248299|-0.000271|+1.076632|7/8|
|B5|400|100/100 (100.00%)|157/200 (78.50%)|63/100 (63.00%)|832/1000 (83.20%)|0.017843|1.411945|+0.000137|+1.434215|7/8|
|B6|500|100/100 (100.00%)|162/200 (81.00%)|72/100 (72.00%)|754/1000 (75.40%)|0.021543|1.134753|+0.000205|+1.699555|7/8|
|B7|600|100/100 (100.00%)|158/200 (79.00%)|63/100 (63.00%)|824/1000 (82.40%)|0.015825|1.323211|+0.000420|+1.576277|7/8|
|B8|700|100/100 (100.00%)|154/200 (77.00%)|62/100 (62.00%)|764/1000 (76.40%)|0.016233|1.201935|+0.000336|+1.643934|6/8|
|B9|800|100/100 (100.00%)|161/200 (80.50%)|65/100 (65.00%)|803/1000 (80.30%)|0.016731|0.938513|+0.000379|+1.582256|7/8|
|B10|900|100/100 (100.00%)|133/200 (66.50%)|51/100 (51.00%)|817/1000 (81.70%)|0.018917|1.289869|+0.000400|+2.297939|7/8|

## Structural-H hard gate와 Post-Energy WARN

- H 상태: `{'H_EMPTY_EXACT_ATOMIC_EQUIVALENCE': 8, 'H_ACTIVE_CERTIFIED': 72}`; hard gate 실패 `0/80`.
- strength residual 최대 `8.882e-16` (limit 1e-8), P residual 최대 `2.332e-15` (limit 1e-8), min(selected) 최소 `3.072e-16` (limit −1e-8).
- energy residual은 WARN `60/80`; 총 magnitude `4.958e-12`. 기존 1e-12 경계 초과는 B4-K2의 `1.9768631176475537e-12` 한 건입니다.
- optimizer 내부 `E(v) <= E_limit`은 그대로이며, solver/strength/P/nonnegative 실패는 끝까지 hard fail입니다. 새 임계값, polish, shrink, retry, tolerance 변경은 없습니다.
- 이 실행은 old strict H certificate의 기술 수리가 아니라, post-solve energy 재검산을 WARN으로 바꾼 명시적 방법 수정입니다.

## 실제 post-BF16 update NormShare

주지표는 `sqrt(realized_bf16_step_energy) / Σ sqrt(E_l)`입니다. factor energy, beta, coefficient는 보조 설명값입니다.

|layer|NormShare mean|median|p90|max|squared EnergyShare mean|
|---:|---:|---:|---:|---:|---:|
|L4|0.101541|0.103349|0.115437|0.128778|0.046577|
|L5|0.144682|0.141564|0.166749|0.172537|0.092536|
|L6|0.188000|0.188869|0.203437|0.211441|0.153660|
|L7|0.236817|0.237032|0.251486|0.328069|0.242473|
|L8|0.328960|0.325989|0.356366|0.500059|0.464754|

L8 NormShare는 평균 `0.3290`, 중앙값 `0.3260`, p90 `0.3564`, 최대 `0.5001`입니다. 후기 layer 집중은 유지됐으며, 이는 실제 BF16 update norm 기준입니다.

## Target/controller/writer와 순차 상태

- PRIMARY/RESCUE/CURRENT: `7916/40/44` over `8000` active request-steps; clamp-hit `68`, Soft→Neutral fallback `0`.
- history widths `[0, 100, 200, 300, 400, 500, 600, 700, 800, 900]`; terminal active history/lifetime anchors `1000/1000`.
- batch-entry evaluator `0`, inner heldout controller access `0`, accepted-z observation action influence/backward/generation `0/0/0`, duplicate W evaluator forward `0`.
- K당 virtual prefix는 q-only current residual/key/q refresh를 사용하며 current-slope backward `0`; outer당 물리 materialization `1`; physical W는 B1→B10 지속 후 terminal W0로 정확 복원됐습니다.

## Compute와 무결성

- scheduler wall `11189.1s`; accepted-z observation F/B/generation `1000/0/0`, tokens `301441`.
- edit ledger totals: model F `43600`, backward `12500`, target backward `8500`, slope backward `4000`, prefix captures `20500`, materializations `80`.
- terminal evaluator F/tokens `7500/2257101`; W0 pointer/bytes restore `True/True`.
- 실행 소스 HEAD/tree `1118821001008cbbde01af8effae4e0c2d2f921f` / `b2c599d5082f78a11980abd1f56adf011a1f2ce7`. 첫 시도 job20868은 inline accepted-z sealed-reference assertion의 순수 기술 실패로 보존됐고, TECH-R1은 그 assertion만 same-run path에 맞게 수정했습니다.

## Hard cohort와 해석 경계

- z rephrase strict 실패 `57/1000`; z strict 성공→W immediate strict 실패 `261/1000`; final W10 strict 실패 `244/1000`.
- immediate-post strict 성공→final 실패 `18/1000`. 요청별 ID와 수치는 `p1r52-piru-postenergy-hard-cohorts.json`에 있습니다.
- sealed MEMIT/AlphaEdit/J0는 동일 1,000-request/order이지만 다른 source revision에서 생성됐습니다. 따라서 비교는 matched sealed reference이며 exact same-head 단일변수 인과 격리가 아닙니다.
- post-energy WARN 수정과 결과 변화의 고립된 인과를 주장하지 않습니다. `scientific_promotion=false`입니다.

## Machine-readable artifacts

- core comparison: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-piru-seq-postenergy-warn-r1-v1/local/odebf/reports/p1r52-piru-postenergy-warn-r1-10xb100-v1/p1r52-piru-postenergy-core-comparison.json`
- per batch: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-piru-seq-postenergy-warn-r1-v1/local/odebf/reports/p1r52-piru-postenergy-warn-r1-10xb100-v1/p1r52-piru-postenergy-per-batch.json` (10 rows)
- per H decision: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-piru-seq-postenergy-warn-r1-v1/local/odebf/reports/p1r52-piru-postenergy-warn-r1-10xb100-v1/p1r52-piru-postenergy-per-h-decision.json` (80 rows)
- per layer: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-piru-seq-postenergy-warn-r1-v1/local/odebf/reports/p1r52-piru-postenergy-warn-r1-10xb100-v1/p1r52-piru-postenergy-per-layer.json` (400 rows)
- per z/W request: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-piru-seq-postenergy-warn-r1-v1/local/odebf/reports/p1r52-piru-postenergy-warn-r1-10xb100-v1/p1r52-piru-postenergy-per-z-w-request.json` (1,000 rows)
- hard cohorts: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-piru-seq-postenergy-warn-r1-v1/local/odebf/reports/p1r52-piru-postenergy-warn-r1-10xb100-v1/p1r52-piru-postenergy-hard-cohorts.json`
- compute/integrity: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-piru-seq-postenergy-warn-r1-v1/local/odebf/reports/p1r52-piru-postenergy-warn-r1-10xb100-v1/p1r52-piru-postenergy-compute-integrity.json`

scientific_promotion=false

## 부록 A. GEN–LOC 상충과 writer realization 상세 분석

### A.1 핵심 결론

이번 결과는 전체 semantic 성능이 순차적으로 한 번에 붕괴한 형태가 아닙니다. PIR-U는 accepted-z의 rewrite 목표를 물리 W에 매우 강하게 실현했지만, 같은 업데이트가 rephrase 전달과 locality를 균형 있게 보존하지 못했습니다. 최종적으로 J0 Structural-H ON 대비 GEN success는 `+10/2000`(`+0.50%p`), rephrase accuracy는 `+62/2000`(`+3.10%p`)였으나, LOC는 `-744/10000`(`-7.44%p`), rewrite accuracy는 `-22/1000`(`-2.20%p`)였습니다. 따라서 이 결과는 Pareto 개선이 아니라 writer strength와 preservation 사이의 관찰된 상충입니다.

accepted-z 품질은 PIR-U와 J0가 매우 유사했습니다. PIR-U−J0 accepted-z NLL 차이는 rewrite `+0.003869`, rephrase `+0.003807`입니다. 큰 차이는 z 생성보다 z를 물리 W로 옮기는 writer 단계와 이후 sequential 누적에서 발생했습니다.

### A.2 최종 LOC 하락의 직접 손상과 순차 누적 분해

|구분|PIR-U|J0 Structural-H ON|PIR-U−J0|
|---|---:|---:|---:|
|각 B100 immediate-post LOC aggregate|8297/10000 (82.97%)|8586/10000 (85.86%)|-2.89%p|
|immediate-post→final W10 LOC 변화|-6.47%p|-1.92%p|-4.55%p 추가 누적 손상|
|final W10 LOC|7650/10000 (76.50%)|8394/10000 (83.94%)|-7.44%p|

최종 LOC 격차 `-7.44%p` 중 `-2.89%p`는 각 배치를 쓴 직후부터 존재하는 직접 손상이고, 나머지 `-4.55%p`는 후속 배치가 누적되며 J0보다 더 크게 발생한 추가 침식입니다. 산술 비중으로는 약 39%가 immediate-post 차이, 약 61%가 초과 sequential 누적에 해당합니다. 이는 LOC 문제가 단일 시점의 급락만으로 설명되지 않고, 더 큰 즉시 간섭과 더 빠른 장기 침식이 함께 작용했음을 보여줍니다.

### A.3 sequential 진행에 따른 cohort-age 손상

|원래 편집 batch|PIR-U post LOC|PIR-U final W10 cohort LOC|PIR-U 후속 손실|J0 후속 손실|
|---:|---:|---:|---:|---:|
|B1|88.8%|75.5%|-13.3%p|-5.1%p|
|B2|90.1%|76.2%|-13.9%p|-5.1%p|
|B3|88.2%|77.9%|-10.3%p|-2.4%p|
|B4|83.2%|77.5%|-5.7%p|-1.1%p|
|B5|83.2%|77.2%|-6.0%p|-1.5%p|
|B6|75.4%|71.1%|-4.3%p|-2.0%p|
|B7|82.4%|76.8%|-5.6%p|-0.8%p|
|B8|76.4%|72.2%|-4.2%p|-1.2%p|
|B9|80.3%|78.9%|-1.4%p|0.0%p|
|B10|81.7%|81.7%|0.0%p|0.0%p|

B1·B2처럼 이후 800–900개 편집을 더 견뎌야 하는 오래된 cohort의 LOC 손실이 `-13.3/-13.9%p`로 가장 컸습니다. B9·B10은 후속 편집이 적어 손실도 작았습니다. 이는 PIR-U의 locality 손상이 batch age와 후속 편집 수에 따라 누적되는 sequential interference 형태임을 지지합니다. batch/history width와 immediate-post LOC의 순위 상관도 `Spearman=-0.815`로 음의 관계를 보였으나, batch별 값 자체가 완전 단조 감소한 것은 아닙니다.

반면 GEN은 같은 방식으로 붕괴하지 않았습니다. PIR-U의 immediate-post aggregate GEN은 `1620/2000`(81.00%)이고 final W10 GEN은 `1706/2000`(85.30%)입니다. 후속 편집이 일부 이전 rephrase prompt를 돕는 변화도 포함됐습니다. 따라서 관찰된 순차 붕괴는 전체 semantic 기능의 일괄 붕괴가 아니라, locality와 오래된 편집 보존에 집중된 현상입니다.

### A.4 writer 소실은 rewrite와 rephrase에서 서로 다름

|accepted-z→physical W 지표|PIR-U|J0 Structural-H ON|
|---|---:|---:|
|accepted-z rewrite NLL|0.035999|0.032130|
|W immediate rewrite NLL|0.035257|0.046224|
|rewrite W−z gap|-0.000742|+0.014093|
|accepted-z rephrase NLL|1.275457|1.271650|
|W immediate rephrase NLL|2.695867|2.505594|
|rephrase W−z gap|+1.420410|+1.233944|
|W가 z보다 나쁜 PIR-U prompt|rewrite 518/1000|NOT_RECORDED|
|W가 z보다 나쁜 PIR-U prompt|rephrase 1698/2000|NOT_RECORDED|

Rewrite 축에서는 PIR-U writer realization gap이 `-0.000742`로 사실상 0이며 평균적으로 물리 W NLL이 accepted-z보다 근소하게 낮았습니다. EFF도 final W10에서 `996/1000`이므로, 단순히 writer가 약해서 target을 기록하지 못한 문제는 rewrite 축에서 거의 해소됐습니다.

그러나 rephrase 축에서는 평균 W−z gap이 `+1.420410`이고 `1698/2000` prompt에서 물리 W가 accepted-z보다 나빴습니다. J0의 rephrase gap `+1.233944`보다도 `+0.186466` 큽니다. batch별 PIR-U rephrase gap은 B1 `+0.872035`, B2 `+0.865246`, B5 `+1.434215`, B6 `+1.699555`, B8 `+1.643934`, B10 `+2.297939`로 후반에 커졌으며, history width와의 순위 상관은 `Spearman=+0.891`입니다. accepted-z rephrase NLL 자체는 B1–B10에서 약 `0.94–1.49` 범위로 유지됐으므로, 주된 악화 지점은 target z 생성이 아니라 z를 current W에 기록한 뒤 rephrase 방향을 보존하는 writer realization입니다.

따라서 writer 문제는 하나의 scalar strength 부족으로 요약할 수 없습니다. PIR-U는 rewrite 목표를 강하게 실현했지만, 그 물리 업데이트가 rephrase 전이와 locality를 동일하게 보존하지 못했습니다. 관찰된 writer 문제는 `약한 write`보다 `목표별로 비등방적인 realization/conditioning`에 가깝습니다.

### A.5 실제 BF16 업데이트 강도와 후기 layer 집중

80개 accepted transition의 `realized_bf16_step_energy` 합은 PIR-U `158.234993`, J0 `76.766500`으로 PIR-U가 `2.061배`였습니다.

|실제 BF16 step energy|PIR-U|J0|PIR-U/J0|
|---|---:|---:|---:|
|mean|1.977937|0.959581|2.061×|
|median|0.957989|0.769001|1.246×|
|p90|4.589619|1.741641|2.635×|
|max|11.863690|2.937920|4.038×|

실제 update Frobenius norm 기준 NormShare도 전반 layer에서 후기 layer로 이동했습니다.

|layer|PIR-U NormShare mean|J0 NormShare mean|차이|
|---:|---:|---:|---:|
|L4|0.101541|0.16771|-0.06617|
|L5|0.144682|0.19373|-0.04905|
|L6|0.188000|0.20146|-0.01346|
|L7|0.236817|0.20102|+0.03580|
|L8|0.328960|0.23609|+0.09287|

PIR-U L8 NormShare는 평균 `32.90%`, 중앙값 `32.60%`, p90 `35.64%`, 최대 `50.01%`입니다. PIR-U의 모든 batch에서 L8 share가 sealed J0보다 높았습니다. 동시에 PIR-U step-energy p90과 max가 J0보다 각각 약 `2.64배`, `4.04배`였으므로, LOC 하락과 가장 일관된 writer-side 관찰은 `더 큰 물리 업데이트`와 `L7/L8 집중`입니다.

### A.6 source-backed 동작 구조

PIR-U는 positive demand에서 `gamma=1`을 사용하고, layer coefficient를 remaining-pi 비율 `beta_l=pi_l/sum_{j>=l}(pi_j)`로 정합니다. suffix denominator가 줄어들기 때문에 `pi8>0`이면 마지막 L8의 `beta8=1`입니다. L5–L8에서는 앞선 virtual write 뒤의 current residual/key/q를 다시 계산하지만 current semantic slope에 의한 strength 재보정은 하지 않습니다.

따라서 entry 시점의 Structural-H router가 산출한 pi를 순차 prefix의 변경된 residual/q geometry에 적용하게 됩니다. Entry H/P/energy certificate가 최종 5-layer BF16 endpoint에 그대로 보존된다고 주장할 수 없으며, source receipt도 sequential P를 `ENTRY_FIELD_MIXED_GEOMETRY_PROXY_NOT_COMPARABLE`로 구분합니다. 이 구조는 관찰된 `gamma=1` 강한 write, L8 `beta=1`, 후기 layer 집중, 약 2배의 실제 물리 energy와 방향상 일치합니다.

다만 sealed J0 reference는 동일 1,000-request stream/order를 사용하지만 다른 source revision에서 생성됐습니다. 따라서 이 절은 source와 receipt에 의해 지지되는 mechanism association이며, exact same-head 단일변수 인과 증명은 아닙니다.

### A.7 Post-Energy WARN과 LOC 하락의 관계

Post-Energy WARN은 `60/80`이지만 magnitude 합은 `4.958e-12`이고, 기존 `1e-12` 경계를 넘은 것은 B4-K2의 `1.9768631176475537e-12` 한 건입니다. 이 residual은 실제 BF16 update energy가 아니라 optimizer certificate의 original-coordinate 수치 잔차입니다. WARN 변경은 action을 축소하거나 수정하지 않고 이전 strict gate에서 정지했을 trajectory를 계속 실행하도록 했습니다.

따라서 WARN 정책은 B10 terminal까지 도달하게 한 명시적 방법 변경이지만, `-7.44%p` LOC 하락의 크기를 직접 설명하는 물리량은 아닙니다. 관찰된 LOC 결과는 tiny post-solve residual보다 PIR-U trajectory의 실제 BF16 energy, 후기 layer 집중, sequential current-geometry 누적과 더 일관됩니다.

### A.8 종합 판정

- `SEQUENTIAL_LOC_INTERFERENCE_SUPPORTED`: immediate-post 손상과 오래된 cohort의 후속 침식이 모두 J0보다 큽니다.
- `GLOBAL_SEMANTIC_COLLAPSE_NOT_SUPPORTED`: final GEN은 immediate-post aggregate보다 높고 J0보다 `+0.50%p`입니다.
- `REWRITE_WRITER_REALIZATION_RECOVERED`: rewrite 평균 W−z gap은 사실상 0이며 EFF `996/1000`입니다.
- `REPHRASE_WRITER_REALIZATION_LIMITED`: rephrase 평균 W−z gap `+1.420410`, W-worse `1698/2000`이며 history와 함께 증가합니다.
- `LOCALITY_PRESERVATION_LIMITED`: final LOC는 J0 대비 `-7.44%p`입니다.
- `LATE_LAYER_CONCENTRATION_AND_HIGH_UPDATE_ENERGY_ASSOCIATED`: 실제 BF16 energy `2.061배`, L8 NormShare `+9.29%p`가 관찰됐습니다.
- `ISOLATED_CAUSAL_CLAIM_NOT_ESTABLISHED`: J0와 source revision이 다르므로 same-head 단일변수 인과 주장은 하지 않습니다.

이 부록의 결론은 PIR-U가 writer strength 자체를 높이는 데는 성공했지만, writer 소실을 전체적으로 제거한 것이 아니라 rewrite에는 강하고 rephrase/locality에는 불리한 realization으로 이동시켰다는 것입니다. `scientific_promotion=false` 상태는 유지합니다.
