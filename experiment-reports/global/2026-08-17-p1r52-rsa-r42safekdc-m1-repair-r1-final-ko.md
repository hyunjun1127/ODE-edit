# P1R52 Repair-R1 최종 비교 보고서

## 실행 및 계약 무결성

- Repair-R1 HEAD/tree: `34f0b505fe5f32078aa4b8cffd109a7490d8aaa5` / `253a177dcd72d2235d0ece500558b357b578f84b`; parent/pre-repair HEAD `c35ebe5c299c919e68e496cbc2f0d77512c73f7a`.
- attempts/endpoints/typed failures: 40/40/0; step/request-step 320/3200.
- W0/action-freeze/teacher K8 constancy/semantic descent: True/True/True/True.
- max unit-norm residual/raw-energy relerr/safe-preservation inner: 9.548e-15/3.617e-11/2.369e-12 (lock 1e-8).
- model F/B/tokens/materializations: 9220/5000/1508881/320; additional KL F/B 0/0.
- exact pairing: case 40, step 320, request-step 3200, failed-cohort union 340; unmatched 0; imputation 0.

## Repair-R1 vs pre-repair vs P1R51

| model | arm | z NLL repair (Δpre/ΔR51) | W NLL repair (Δpre/ΔR51) | gap repair (Δpre/ΔR51) | z Eff/Gen repair | W Eff/Gen/Loc repair | PRIMARY/RESCUE/CURRENT |
|---|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 0.0467711 (0/-0.0180894) | 0.0983516 (0/-0.00022112) | 0.0515805 (0/0.0178682) | 100/184 | 100/180/872 | 788/11/1 |
| llama3-8b-inst | Soft | 0.0414769 (4.68224e-05/-0.0197677) | 0.0792804 (-9.80223e-06/-0.00506371) | 0.0378035 (-5.66246e-05/0.014704) | 100/183 | 100/181/871 | 791/9/0 |
| qwen2.5-7b-inst | Neutral | 0.455928 (0.0022663/-0.10636) | 0.581572 (0.000956801/-0.410971) | 0.125643 (-0.0013095/-0.304611) | 98/160 | 97/153/842 | 751/22/27 |
| qwen2.5-7b-inst | Soft | 0.474662 (0.00726369/0.00638993) | 0.553507 (0.00259402/0.0398207) | 0.0788456 (-0.00466967/0.0334307) | 97/163 | 97/152/843 | 759/25/16 |

Repair와 pre-repair의 terminal z/W/gap 3항이 byte-derived 숫자로 모두 같은 case는 31/40이고, 하나 이상 다른 case는 9/40이다. 120개 paired scalar 중 최대 절대 차이는 0.0573447이다. 이 차이는 package-level 관측값이며 6.2e-11 수준의 denominator 변경이 개별 결과를 개선·악화시켰다는 인과 주장에 사용하지 않는다.

## Exact active-unit·projection·energy 분해

| model | arm | slope ratio mean/median/p10/min | conflict/removed | KL norm mean | decay norm mean | ref/raw/postclamp/precast/postcast/accepted mean | clamp-only/cast/rescue/current/unused mean | P/cap/BF16 mean | neg progress |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 0.963335/0.998863/0.900509/0.193828 | 636/636 | 2.3716 | 0.0123723 | 462.013/462.013/402.976/402.976/373.606 | 59.0371/-8.00083e-08/29.2611/0.108743/88.407 | 0.00363553/0.0980236/0.259667 | 8 |
| llama3-8b-inst | Soft | 0.963478/0.99886/0.897862/0.208325 | 638/638 | 2.36963 | 0.0123723 | 437.468/437.468/389.42/389.42/366.391 | 48.0478/3.41125e-07/23.0287/0/71.0765 | 0.00347738/0.0838975/0.229233 | 5 |
| qwen2.5-7b-inst | Neutral | 0.970301/0.999268/0.934988/0.212408 | 541/541 | 0.386995 | 1.86735e-07 | 90056/90056/90056/90056/38619.2 | -9.68656e-14/2.23865e-05/6125.53/45311.3/51436.8 | 0.139481/1.48105/1.5258 | 12 |
| qwen2.5-7b-inst | Soft | 0.965556/0.999267/0.918726/0.10563 | 530/530 | 0.369957 | 1.86735e-07 | 89014.4/89014.4/89014.4/89014.4/45058.2 | 7.71715e-13/-1.18519e-05/8003.02/35953.2/43956.2 | 0.0820588/0.476912/0.773614 | 12 |

각 scalar 분포의 mean/median/p90/max와 request-level KL/decay/preservation norm, unit/descent receipt, teacher hash는 machine table에 기록했다. Pre-repair의 clamp+cast 혼합 에너지는 legacy field로만 보존했고, Repair-R1에서 raw→post-clamp/pre-cast→post-cast→accepted와 clamp-only/FP32/rescue/CURRENT/unused를 분리했다.

## R42·Official·Native anchor

- R42 source/report package는 stream/order/evaluator exact identity와 40 case를 재해시했다. `p1r52-repair-r1-paired-case-deltas.json`에 model×arm×case의 z/W/gap, Eff/Gen/Loc, functional-P, runtime 산술 차이를 기록했다.
- Official AlphaEdit는 20 model×case common reference로 Eff/Gen/Loc 및 Eff NLL/margin만 matched다. latent-z, W8 full-six, Gen continuous NLL/margin, P/capacity/energy는 `NOT_RECORDED`다.
- Native exact-matched case/request/step raw-free table은 입력 패키지에 없어 `NOT_RECORDED`; 대체·imputation하지 않았다.

## P1R51 failed cohort 전이

Repair-R1 failed-cohort union 340행은 supplied P1R51 failed cohort와 terminal z-Eff/z-Gen/W-Eff/W-Gen, CURRENT/RESCUE, regression, high-allocation weak progress, z-success/W-failure를 exact request identity로 재평가했다. 각 행은 pre-repair와 Repair-R1의 recovered/persisted/new label, entry/final z·W NLL, allocation, direction/preservation, clamp, selection, writer gap, negative progress를 함께 기록한다. 분류는 관측 연관이며 인과 결론이 아니다.

| model | arm | union | repair recovered/persisted/new | pre recovered/persisted/new | zEff/zGen/WEff/WGen | z-ok/W-fail | hard/new/direction/writer/context/inconclusive |
|---|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | Neutral | 74/100 | 7/38/29 | 7/38/29 | 0/14/0/17 | 3 | 38/29/15/21/14/7 |
| llama3-8b-inst | Soft | 69/100 | 14/35/20 | 14/35/20 | 0/15/0/16 | 1 | 35/20/13/19/15/14 |
| qwen2.5-7b-inst | Neutral | 100/100 | 12/62/26 | 12/62/26 | 2/30/3/35 | 5 | 62/26/23/15/28/12 |
| qwen2.5-7b-inst | Soft | 97/100 | 11/59/27 | 10/60/27 | 3/28/3/38 | 10 | 59/27/21/14/25/11 |

## Qwen Neutral case10 및 leave-one-case-out

Qwen Neutral case10은 1 case/8 step/80 request-step으로 별도 파일에 target direction/descent, reference/raw/clamp/cast/accepted energy, writer predicted/actual/realization, layer entropy/top1/P/capacity를 pre-repair·P1R51과 함께 기록했다. Leave-one-case-out는 4 cell×10 omission=40행이며 9-case z/W/gap repair·pre·P1R51 평균과 paired delta를 기록했다.

## 결론 및 지정

- 기술 상태: `PASS`; Repair-R1 hard gate와 raw-free integrity가 모두 통과했다.
- canonical designation: `CANONICAL_P1R52_REPAIR_R1_METHOD_CONTRACT_ARTIFACT`. 이는 exact active-unit method contract를 충족한 P1R52 산출물 지정이며 scientific promotion은 아니다.
- 비교 결론: 31/40 case는 terminal z/W/gap이 pre-repair와 동일했고 9/40은 수치 차이가 있었다. 따라서 '전부 동일'이라고 기록하지 않으며, 차이를 normalization 한 줄의 독립 인과효과로 해석하지 않는다.
- claim boundary: `P1R51 + R42SafeKDC preservation package`; KL/decay/clamp 각각의 isolated causal claim 없음.
- `scientific_promotion=false`; 후속 설계·실행 추천 없음.
