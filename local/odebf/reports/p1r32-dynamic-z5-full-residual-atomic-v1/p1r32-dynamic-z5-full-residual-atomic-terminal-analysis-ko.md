# P1R32 Dynamic-Z5 + Full-Residual Atomic 최종 분석

## 1. 범위와 결론

- instruction: `ODEEDIT-S05-P1R32-DYNAMIC-Z5-FULL-RESIDUAL-ATOMIC-V1`; source checkpoint `5905f4e217683f363c3318db7889628966c40205`; tree `ce5a87ce2000e5b1d755ec7bc60857a618c857c1`.
- 실행 표본: original outcome-selected B10, seal `3d38b76de81c21e65068b1c1e22466ec55a0c5973ae289081d773391ed92f628`, order `984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b`.
- 4/4 셀은 K8/tau1, 각 outer state의 Adam update 5회, materialization 8회, action-freeze, W0 restore를 완료했다. scientific-invalid endpoint는 0이다.
- **핵심 결론(FACT):** Llama는 target 및 W strength가 일부 회복됐지만 locality가 34/100으로 붕괴했다. Qwen은 dynamic target 자체가 terminal에서 약했고 writer도 추가 손실/불안정을 보였다. Soft는 사실상 c0로 되돌아가 Neutral과 endpoint가 같아 barrier signal이 없었다.
- **SCIENTIFIC_FAIL / comparison boundary:** 실행 order `984fe6…`는 P1R31 10×B10 order `abe62c…`와 다르다. 따라서 요구된 P1R31 same-stream 비교는 NOT_MATCHED이며, P1R31 수치는 방향성 reference일 뿐 paired evidence가 아니다.
- 최종 판정: **DYNAMIC_Z5_FULLRES_NOT_READY**. target-side와 writer-side 문제가 모두 남았고 preservation router의 독립 신호도 확립되지 않았다. `scientific_promotion=false`.

## 2. 입력·무결성

- controlling attachment exact-byte verification: `PASS`.
- P1R31 report local rehash: `PASS`; P1R31 order match: `NOT_MATCHED`.
- evaluator/action trajectory firewall: complete action freeze precedes heldout evaluation; inner heldout access 0; retry/backtracking/history/debt/remaining-step division 0.
- Llama: terminal SHA match `PASS`, manifest identity rehash `PASS`, W0 restore `True`, source `5905f4e217683f363c3318db7889628966c40205`.
- Qwen: terminal SHA match `PASS`, manifest identity rehash `PASS`, W0 restore `True`, source `5905f4e217683f363c3318db7889628966c40205`.

## 3. Endpoint: z-oracle와 BF16 W

| Model | Arm | z E/G | W E/G/L | z Eff NLL | W Eff NLL | z-W E/G gap | negative actual | terminal P | functional-P |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama | Neutral | 9/17 | 9/15/34 | 7.0425 | 8.8255 | 0/2 | 3 | 225.4685 | 12.5236 |
| Llama | Soft | 9/17 | 9/15/34 | 7.0425 | 8.8255 | 0/2 | 3 | 225.4685 | 12.5236 |
| Qwen | Neutral | 5/4 | 4/4/69 | 15.0437 | 16.4000 | 1/0 | 2 | 84808.5898 | 11.2752 |
| Qwen | Soft | 5/4 | 4/4/69 | 15.0437 | 16.4000 | 1/0 | 3 | 84808.5897 | 11.2752 |

### 해석

- Llama: z 9/10·17/20에서 W 9/10·15/20으로 Gen 2개가 추가 손실됐다. 그러나 z 자체도 Official 10/10·20/20 reference보다 약하며, W locality 34/100은 strength 회복 대가가 매우 큼을 뜻한다. 분류는 `BOTH_WEAK_OR_TYPED_INCOMPLETE`이며 target이 writer보다 상대적으로 강한 혼합형이다.
- Qwen: z 5/10·4/20, W 4/10·4/20이다. target-side가 이미 크게 약하고 writer가 Eff 1개를 더 잃는다. k3 이후 residual/capacity 폭증과 0-rho 단계가 나타나므로 target·writer 모두 병목이다.
- Neutral/Soft의 terminal E/G/L 및 연속 NLL은 모델별로 exact-equal이다. 이는 preservation 개선이 아니라 allocation freedom/solver improvement 부재다.

## 4. Target solver 동역학

각 outer k마다 j=0..4의 정확히 5 Adam update와 j=5 final measurement가 기록됐다. 4개 arm 전체 합계는 Adam update 160회, moment reset 32회, current teacher refresh 32회다.

| Model | Arm | k0 j0→j5 NLL | k7 j0→j5 NLL | residual mean k0→k7 | zero-rho steps |
|---|---|---:|---:|---:|---:|
| Llama | Neutral | 8.3655→0.4733 | 9.2294→6.4818 | 4.8788→168.9629 | 0 |
| Llama | Soft | 8.3655→0.4733 | 9.2294→6.4818 | 4.8788→168.9629 | 0 |
| Qwen | Neutral | 9.3092→0.1787 | 15.5495→15.3465 | 86.0379→43351.5624 | 4 |
| Qwen | Soft | 9.3092→0.1787 | 15.5493→15.3496 | 86.0379→43351.5656 | 4 |

- k0에서는 Dynamic-Z5가 두 모델 모두 NLL을 크게 낮춘다. 하지만 Qwen은 writer trajectory가 손상된 뒤 k3 이후 current-state target solve도 회복하지 못하고 terminal z NLL이 15.35까지 상승한다.
- 이는 Adam depth만 늘리면 항상 강한 z가 나온다는 가설을 지지하지 않는다. current W와 intervention coordinate/conditioning의 상호작용이 지배적이다.
- post-freeze snapshot hash 9개/arm은 기록됐지만 k0..k7의 z/W numeric vectors는 terminal JSON에 직렬화되지 않았다. 따라서 단계별 heldout E/G/Loc과 first-hit은 `NOT_RECORDED`; inner target NLL과 terminal z/W만 FACT로 사용했다.

## 5. Step별 성능 변화 추이

아래 `W NLL before/after`는 각 accepted transition의 동일 full-six target-new objective다. `z5 NLL`은 current W에서 5회 Adam 후 intervention target NLL이며 heldout E/G가 아니다.

### Llama Neutral

| k | W NLL before | z5 NLL | predicted | actual | W NLL after | residual norm | P | capacity | fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8.3655 | 0.4733 | 19.2951 | 7.5399 | 0.8256 | 4.8788 | 0.4065 | 146.0249 | false |
| 2 | 0.8256 | 0.0182 | 1.3432 | 0.7772 | 0.0484 | 8.4132 | 0.5607 | 4351.0763 | false |
| 3 | 0.0484 | 0.0033 | 1.1749 | -8.8535 | 8.9019 | 13.4372 | 12.0787 | 499200.7783 | false |
| 4 | 8.9019 | 0.0046 | 39.4291 | -3.5664 | 12.4683 | 38.0835 | 287.8324 | 379544712.4677 | false |
| 5 | 12.4683 | 12.1568 | 0.2543 | 5.6855 | 6.7828 | 563.6433 | 192.4572 | 3274484149.5755 | false |
| 6 | 6.7828 | 3.5717 | 4.6445 | 3.8295 | 2.9532 | 61.6975 | 201.9263 | 233593070.9664 | false |
| 7 | 2.9532 | 1.0651 | 1.2441 | -6.8162 | 9.7694 | 52.6850 | 209.7719 | 447493804.5059 | false |
| 8 | 9.7694 | 6.4818 | 2.5000 | 0.0436 | 9.7258 | 168.9629 | 225.4685 | 2957640844.2412 | false |

### Llama Soft

| k | W NLL before | z5 NLL | predicted | actual | W NLL after | residual norm | P | capacity | fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8.3655 | 0.4733 | 19.2951 | 7.5399 | 0.8256 | 4.8788 | 0.4065 | 146.0249 | true |
| 2 | 0.8256 | 0.0182 | 1.3432 | 0.7772 | 0.0484 | 8.4132 | 0.5607 | 4351.0763 | true |
| 3 | 0.0484 | 0.0033 | 1.1749 | -8.8535 | 8.9019 | 13.4372 | 12.0787 | 499200.7783 | true |
| 4 | 8.9019 | 0.0046 | 39.4291 | -3.5664 | 12.4683 | 38.0835 | 287.8324 | 379544712.4677 | true |
| 5 | 12.4683 | 12.1568 | 0.2543 | 5.6855 | 6.7828 | 563.6433 | 192.4572 | 3274484149.5755 | true |
| 6 | 6.7828 | 3.5717 | 4.6445 | 3.8295 | 2.9532 | 61.6975 | 201.9263 | 233593070.9664 | true |
| 7 | 2.9532 | 1.0651 | 1.2441 | -6.8162 | 9.7694 | 52.6850 | 209.7719 | 447493804.5059 | true |
| 8 | 9.7694 | 6.4818 | 2.5000 | 0.0436 | 9.7258 | 168.9629 | 225.4685 | 2957640844.2412 | true |

### Qwen Neutral

| k | W NLL before | z5 NLL | predicted | actual | W NLL after | residual norm | P | capacity | fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.3092 | 0.1787 | 81.2689 | 7.5252 | 1.7840 | 86.0379 | 62.0021 | 3440.5760 | false |
| 2 | 1.7840 | 0.0036 | 7.6835 | -3.5377 | 5.3217 | 173.7045 | 137.8636 | 10130227.5498 | false |
| 3 | 5.3217 | 0.0010 | 7.5657 | -10.6841 | 16.0058 | 611.2694 | 3388.9750 | 489183235.3016 | false |
| 4 | 16.0058 | 14.8394 | 0.0000 | 0.0000 | 16.0058 | 7630.0970 | 3388.9750 | 0.0000 | false |
| 5 | 16.0058 | 11.3688 | 1.2615 | 0.1907 | 15.8151 | 7625.7655 | 84808.5899 | 597142544054.8698 | false |
| 6 | 15.8151 | 15.7069 | 0.0000 | 0.0000 | 15.8151 | 43364.9034 | 84808.5898 | 0.0000 | false |
| 7 | 15.8151 | 15.5495 | 0.0000 | 0.0000 | 15.8151 | 43357.7074 | 84808.5898 | 0.0000 | false |
| 8 | 15.8151 | 15.3465 | 0.0000 | 0.0000 | 15.8151 | 43351.5624 | 84808.5898 | 0.0000 | false |

### Qwen Soft

| k | W NLL before | z5 NLL | predicted | actual | W NLL after | residual norm | P | capacity | fallback |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.3092 | 0.1787 | 81.2689 | 7.5252 | 1.7840 | 86.0379 | 62.0021 | 3440.5760 | true |
| 2 | 1.7840 | 0.0036 | 7.6835 | -3.5377 | 5.3217 | 173.7045 | 137.8636 | 10130227.5498 | true |
| 3 | 5.3217 | 0.0010 | 7.5657 | -10.6841 | 16.0058 | 611.2694 | 3388.9750 | 489183235.3016 | true |
| 4 | 16.0058 | 14.8394 | 0.0000 | 0.0000 | 16.0058 | 7630.0970 | 3388.9750 | 0.0000 | true |
| 5 | 16.0058 | 11.3688 | 1.2615 | 0.1907 | 15.8151 | 7625.7655 | 84808.5899 | 597142544054.8698 | true |
| 6 | 15.8151 | 15.7069 | 0.0000 | 0.0000 | 15.8151 | 43364.9034 | 84808.5898 | 0.0000 | true |
| 7 | 15.8151 | 15.5495 | -0.0000 | -0.0000 | 15.8152 | 43357.7074 | 84808.5898 | 0.0000 | false |
| 8 | 15.8152 | 15.3496 | -0.0000 | 0.0000 | 15.8152 | 43351.5656 | 84808.5897 | 0.0000 | false |

### 추이 요약

- Llama는 k1~k2에서 W NLL이 빠르게 내려가지만 k3·k4·k7의 negative actual로 되튀며, 이후 부분 회복한다. 즉 target solve의 순간 개선과 BF16 W trajectory의 안정성이 분리된다.
- Qwen은 k1만 큰 개선을 보인 뒤 k2·k3에서 역행한다. k4부터 일부 step의 rho가 0이 되고 residual/P/capacity가 매우 커져 terminal strength가 회복되지 않는다.
- Neutral/Soft 표의 동일성은 endpoint 우연만이 아니라 대부분 step에서 c0 fallback/동일 계수를 사용한 결과다. Qwen Soft k7~k8의 1e-11 수준 변화는 해석 가능한 성능 변화가 아니다.
- heldout z-oracle/W-only E/G/Loc의 k0..k7 숫자 추이는 `NOT_RECORDED`; snapshot receipt hash만 9개/arm 존재한다. 따라서 위 표는 controller full-six objective 추이이며 terminal heldout 표를 대체하지 않는다.

## 6. Writer realization

| Model | Arm | predicted-positive steps | negative actual | max equality residual | P1 step actual/pred | terminal residual mean |
|---|---|---:|---:|---:|---:|---:|
| Llama | Neutral | 8/8 | 3 | 0.000e+00 | 7.5399/19.2951 | 168.9629 |
| Llama | Soft | 8/8 | 3 | 0.000e+00 | 7.5399/19.2951 | 168.9629 |
| Qwen | Neutral | 4/8 | 2 | 0.000e+00 | 7.5252/81.2689 | 43351.5624 |
| Qwen | Soft | 4/8 | 3 | 1.805e-11 | 7.5252/81.2689 | 43351.5656 |

- strength equality certificate는 유지됐으나 local predicted progress가 실제 BF16 progress를 안정적으로 설명하지 못했다. Llama는 양 arm 각 3개, Qwen Neutral은 2개, Soft는 3개 negative-actual transition을 보였다. Qwen Soft의 추가 1개는 rho≈0에서의 4.9e-5 역행으로 numerical-neighborhood 사건이다.
- Qwen residual mean은 k0 약 86에서 k7 약 43,352로 발산했고, Structural-P/capacity도 폭증했다. `full residual` 자체의 magnitude가 feasible/realizable writer scale을 넘어선 것이 주요 writer-side 실패 신호다.
- h 적용 1회, second-h 0, remaining division 0, semantic debt 0은 모든 32 step에서 PASS했다. 따라서 이는 double-h/remaining bookkeeping 오류가 아니라 현 method의 realization 문제다.

## 7. Soft barrier 신호

| Model | Soft changed c (>1e-12) | c0 fallback | terminal ΔP (Soft-Neutral) | endpoint quality delta | verdict |
|---|---:|---:|---:|---:|---|
| Llama | 0/8 | 8/8 | 0 | E/G/L +0/+0/+0 | NO_BARRIER_SIGNAL |
| Qwen | 2/8 | 6/8 | -5.07237e-05 | E/G/L +0/+0/+0 | NO_BARRIER_SIGNAL |

Qwen Soft의 마지막 두 단계는 rho=0 부근에서 약 1e-11 계수 차이만 만들었고 endpoint는 동일하다. 이는 과학적 allocation effect가 아니라 numerical-neighborhood 차이다.

## 8. 비교 경계

### P1R31

P1R31은 stream/order가 다르므로 아래는 **NOT_MATCHED descriptive**이다: Llama RS-N z E/G 100/181, Eff NLL .0824; W 99/178, .1430. Qwen RS-N z 99/162, .3349; W 98/155, .5851. P1R32는 이 reference region에 도달하지 못했다. paired delta나 causal recovery 주장은 금지한다.

### Official AlphaEdit/direct-z final reference

GH가 제공한 same-seal/order P1R18 updated report identity는 `768bfd29a326d503f629513d707a8560c0dbd002b4a26237520b8a27cdfe4ac6`이며 endpoint counts는 Llama 10/10·20/20·79/100, Qwen 10/10·20/20·80/100이다. 그러나 해당 파일은 분석 시 server1에서 부재하여 local rehash와 continuous direct-z NLL/margin은 `NOT_RECORDED/NOT_LOCALLY_VERIFIED`이다. 이 count reference에 비해 P1R32는 Llama z −1/−3, W −1/−5; Qwen z −5/−16, W −6/−16이다.

### P1R30

P1R30 raw-free terminal package는 server1에서 발견되지 않았다. controlling contract가 인용한 Qwen k8 energy 29.17 / actual −1.2367은 predeclared prose reference일 뿐 로컬 artifact rehash가 아니므로 matched spike delta는 `NOT_RECORDED`이다.

## 9. Compute

| Model | Arm | edit-core s | F/B | tokens | materializations | post-freeze panel s | scheduler elapsed | MaxRSS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Llama | Neutral | 122.98 | 615/525 | 70401 | 8 | 416.07 | 1191s | 7142896 KiB |
| Llama | Soft | 137.66 | 615/525 | 70401 | 8 | 416.49 | 1191s | 7142896 KiB |
| Qwen | Neutral | 147.18 | 615/525 | 63811 | 8 | 470.53 | 1362s | 10643804 KiB |
| Qwen | Soft | 160.90 | 615/525 | 63811 | 8 | 469.28 | 1362s | 10643804 KiB |

40 inner Adam updates/trajectory는 Official Native25보다 많다. 본 결과는 compute parity나 efficiency 우위가 아니라 strength-potential 시험이다.

## 10. 실패·수리 provenance

- attempt 19368/19369: ignored session-boundary file 부재, premodel technical failure, scientific/result action 0.
- attempt 19370/19371: P1R24 validator misdispatch, premodel technical failure. TECH-R2에서 P1R32 result dispatch/namespace만 수정했다.
- B1 19373/19374와 production 19375/19376은 source `5905f4e…`로 terminal PASS. scientific/numerical tuning은 없었다.

## 11. FACT / INFERENCE / SCIENTIFIC_FAIL / TECHNICAL_FAIL / NOT_RECORDED

- **FACT:** 4/4 K8 terminal, W0 restore, Adam5×8, teacher8, field8, materialization8, no retry/debt/remaining/H.
- **FACT:** Llama z 9/17, W 9/15/34; Qwen z 5/4, W 4/4/69. Soft와 Neutral endpoint exact-equal.
- **INFERENCE:** Llama는 target보다 writer/Loc 손상이 더 크지만 target도 Official region에 못 미친다. Qwen은 target와 writer가 함께 실패한다.
- **SCIENTIFIC_FAIL:** P1R31 same-stream order 불일치, Qwen strength recovery 부재, Llama locality collapse, barrier signal 부재. 본 패키지는 candidate/promotion gate를 통과하지 않는다.
- **TECHNICAL_FAIL:** 두 premodel attempt는 고립·수리됐다. terminal scientific endpoints에는 unresolved technical failure가 없다.
- **NOT_RECORDED:** k0..k7 post-freeze numeric snapshot vectors, locally rehashed Official continuous direct-z metrics, locally verified P1R30 terminal spike comparison, authoritative GPU peak bytes(only scheduler MaxRSS recorded).

## 12. Claim boundary

재사용된 outcome-selected B10의 mechanistic/descriptive 결과다. P1R31 paired comparison, compute-efficiency, main-table superiority, sequential/Historical generalization을 주장하지 않는다. `scientific_promotion=false`.

## 13. Raw-free artifacts

- `p1r32-per-inner-step.json/csv`: 192 aggregate inner observations; JSON은 request별 NLL/clamp 배열 포함.
- `p1r32-per-step.json/csv`: 32 accepted-step routing/writer/P/capacity/compute rows.
- `p1r32-per-case.json/csv`: 4 endpoint rows.
- `analysis-manifest.json`, `analysis-receipt.json`: content hashes와 rooted inventory.
