# P2R7 P2TARGET-P1DW Shared Writer B10×10 최종 분석

- instruction: `ODEEDIT-S05-P2R7-P2TARGET-P1DW-SHARED-WRITER-V1`
- scheduler: array job `20161`, tasks `0` Llama / `1` Qwen, 두 task 모두 `COMPLETED`, exit `0:0`
- source HEAD/tree: `a7e533a6f65dec7add3f4f6b513f10d51ae81640` / `601af3498b6d1e957c1017b9096cfa11d05c29ee`
- exact P2R2 parent: `c97e8619b42da7954ce0e824c215a8a82d70589a`; execution parent: `581684696b1c921732a5d9cef86bbe0060661836`
- contract SHA256: `d58e5c067d5687559e2bd809d9c5da17c4d546efa6e8feb679f3038f7ebaee5b`
- stream/order: `74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6` / `abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c`
- evaluator/aggregator: `25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145` / `64f009b2fb648627a956b95abd801838d86edf04c8e1888d90ade0ec07b745c0`
- 전체 분모: 40 attempts / 39 complete endpoints / 1 typed incomplete / 400 attempted requests / 390 terminal requests / 316 accepted writer steps / 951 target microstep receipts.
- `scientific_promotion=false`.

## 1. 기술 무결성 게이트

| 항목 | 관측값 | 상태 |
|---|---:|---|
| scheduler task | 2/2 COMPLETED, exit 0:0 | PASS |
| attempt / endpoint / typed incomplete | 40 / 39 / 1 | PASS_WITH_TYPED_SCIENTIFIC_INCOMPLETE |
| valid endpoint K8 / target24 / materialization8 | 39/39 | PASS |
| accepted prefix 포함 writer step / target microstep | 316 / 951 | PASS |
| valid W0 pointer / bytes / action-freeze | 39/39 / 39/39 / 39/39 | PASS |
| incomplete W0 pointer / bytes | 1/1 / 1/1 | PASS |
| routing 변수 / request×layer response matrix | 전 step 5 / 0 | PASS |
| residual `1/h` / physical `h` / second `h` | 전 step 1 / 1 / 0 | PASS |
| omega 합 최대 절대 오차 | `2.220446049250313e-16` | PASS |
| weighted strength 최대 절대 잔차 | `5.186961971048731e-13` | PASS |
| retry / backtracking / candidate materialization | 0 / 0 / 0 | PASS |
| P2R6 QP/shadow, hard-P, functional veto, history-H | 모두 0 | PASS |
| NO_SEMANTIC_DEFICIT | 0 step | 관측 없음 |
| Soft→Neutral full-strength fallback | 3 step | 과학 관측 |
| negative request transition / weighted nonlinear overshoot | 83 / 1 | 과학 관측 |

Llama Soft case04는 outer step4에서 `SCIENTIFIC_SHARED_WRITER_NO_POSITIVE_DIRECTION`으로 종료됐다. 마지막 유효 prefix는 target microstep 15개와 writer/materialization 4개이며 terminal evaluator와 endpoint imputation은 없다. 실패 파일 SHA256은 `fbd24bef9511c5aa05c4abe673049976ed83ffa2ea99b859698ab129cd9f7c37`이고 W0 pointer/bytes 복원은 모두 PASS다. 이 1건은 기술 오류가 아니라 계약에 정의된 typed scientific outcome이다.

첫 W0 outer의 target microstep 0–2는 TECH-R2 파일럿에서 matched P2R2와 모델별 3/3 의미 필드 byte identity PASS가 확인됐다. Production은 동일 final source를 사용했다.

## 2. Model × arm terminal 집계

| model | arm | endpoints/attempts | W E/G/L | z E/G | W Eff/Gen NLL | z Eff/Gen NLL | z/W full6 NLL | W−z full6 gap | actual/pred | P | common cap / energy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | NEUTRAL | 10/10 | 99/100, 186/200, 842/1000 | 98/100, 190/200 | 0.205984 / 1.73038 | 0.189876 / 1.5819 | 0.163405 / 0.185856 | 0.0224516 | 0.793894 | 0.0251068 | 0.0108926 / 13.4841 |
| llama3-8b-inst | SOFT | 9/10 | 90/90, 170/180, 759/900 | 90/90, 174/180 | 0.0890323 / 1.62434 | 0.177956 / 1.45458 | 0.160866 / 0.098748 | -0.0621176 | 0.825473 | 0.0229231 | 0.00895666 / 11.2949 |
| qwen2.5-7b-inst | NEUTRAL | 10/10 | 100/100, 187/200, 841/1000 | 100/100, 189/200 | 0.00324543 / 2.14243 | 0.00133028 / 2.01039 | 0.000823926 / 0.00220317 | 0.00137925 | 0.810303 | 1.17693 | 0.215005 / 45.2987 |
| qwen2.5-7b-inst | SOFT | 10/10 | 100/100, 190/200, 843/1000 | 100/100, 191/200 | 0.00326747 / 2.2239 | 0.00123553 / 2.05518 | 0.000725892 / 0.00229408 | 0.00156819 | 0.823933 | 0.728299 | 0.138711 / 25.0901 |

`z full6`는 terminal W에서 z를 주입해 측정한 full-six target-new NLL이고, `W full6`는 마지막 writer transition 후 W-only full-six NLL이다. 두 값의 차이가 표의 W−z gap이다.

| model | arm | W Eff/Gen/Loc margin | z Eff/Gen margin | W Loc NLL |
|---|---|---:|---:|---:|
| llama3-8b-inst | NEUTRAL | 15.3834 / 8.52999 / 5.07978 | 16.7894 / 9.43157 | 9.92944 |
| llama3-8b-inst | SOFT | 15.8686 / 8.81591 / 5.08382 | 17.0839 / 9.68037 | 9.97438 |
| qwen2.5-7b-inst | NEUTRAL | 16.074 / 8.37234 / 4.70088 | 17.0527 / 9.01123 | 9.98537 |
| qwen2.5-7b-inst | SOFT | 16.2203 / 8.06611 / 4.73719 | 17.1932 / 8.7582 | 10.0154 |

## 3. Deficit weighting, hard tail, routing

| model | arm | mean rho / deficit sum | effective requests mean/min | omega mean-max/max | terminal z p90/worst | terminal gap p90/worst | mean active/q | entropy/top1 | fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | NEUTRAL | 2.88646 / 15.2335 | 3.40381/1.00187 | 0.552365/0.999067 | 0.00111672/8.05088 | 0.0140615/5.55817 | 4.9/23.814 | 1.52523/0.289279 | 0 |
| llama3-8b-inst | SOFT | 3.03842 / 16.5395 | 3.5496/1.00485 | 0.540855/0.997581 | 0.00131986/11.0989 | 0.0111292/0.236624 | 4.96053/25.918 | 1.38637/0.36458 | 3 |
| qwen2.5-7b-inst | NEUTRAL | 2.49391 / 15.7586 | 4.5344/1.17399 | 0.410493/0.921489 | 0.00178951/0.00504624 | 0.00260788/0.0100932 | 4.75/24.9524 | 1.38398/0.371494 | 0 |
| qwen2.5-7b-inst | SOFT | 2.42014 / 15.4077 | 4.78865/1.15288 | 0.399168/0.930508 | 0.00148642/0.00419147 | 0.00367226/0.0145651 | 5/26.544 | 0.99551/0.564315 | 0 |

Llama의 최소 effective request count는 Neutral/Soft `1.001868/1.004852`, 최대 omega는 `0.999067/0.997581`이었다. Qwen은 `1.173986/1.152883`, `0.921489/0.930508`이었다. 따라서 일부 step에서는 weight가 거의 단일 request에 집중됐다. binary concentration threshold는 계약에 없으므로 failure gate로 사용하지 않았다.

`per-request-step.json`에는 3,160개 request-step의 `ellW`, `ellZ`, deficit, omega, actual progress, observation-only chi, W−z gap이 있다. `per-step.json`에는 316개 step의 5 slopes, active mask, q, pi/v/hv, P/common cap/energy, entropy/top1 및 identity가 있다.
Section 3의 writer 통계는 typed incomplete의 유효 4-step prefix를 포함한 316 step 분모이고, terminal z/W tail은 valid endpoint만 사용한다.

## 4. P1AGG → P1DW 파일럿 비교

아래 값은 동일 final source·case01에서 `P1DW - P1AGG`이다. 파일럿 1 case의 직접 산술이며 production 효과 추정값은 아니다.

| model | arm | ΔW E/G/L | ΔW Eff/Gen NLL | Δterminal gap | Δactual/pred | ΔP | Δcommon cap / energy |
|---|---|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | NEUTRAL | +0 / +0 / -1 | +1.7333e-05 / +0.152003 | +0.000240559 | +3.43042 / +4.31228 | +0.00113542 | +0.00220161 / +2.94717 |
| llama3-8b-inst | SOFT | +0 / +0 / +0 | -7.18474e-05 / +0.0385406 | -0.000397472 | +3.48342 / +4.17732 | +0.00191046 | +0.00218682 / +2.68869 |
| qwen2.5-7b-inst | NEUTRAL | +0 / +0 / +1 | +0.000156021 / +0.00289993 | +0.000297674 | +2.48588 / +2.82692 | -0.0615922 | +0.00133998 / +0.173878 |
| qwen2.5-7b-inst | SOFT | +0 / +0 / +0 | -0.000419235 / -0.0929039 | -0.000377865 | +3.05961 / +3.33541 | +0.0294025 | +0.00627561 / +1.17308 |

## 5. DW Neutral → DW Soft production 짝 비교

아래 값은 valid paired case에서 `Soft - Neutral`이다. Llama는 case04 Soft incomplete를 제외한 9쌍, Qwen은 10쌍이다.

| model | pairs | ΣΔW E/G/L | mean ΔEff/Gen NLL | mean Δz/W full6 | mean Δgap / tail detail | mean ΔP | mean Δcommon cap/energy | mean Δentropy/top1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | 9 | +1 / +0 / +2 | -0.101934 / +0.0502244 | +0.000109227 / -0.0614673 | -0.0615765 / `per-case table` | +0.00250302 | +0.00152444 / +1.92872 | -0.135677 / +0.0740843 |
| qwen2.5-7b-inst | 10 | +0 / +3 / +2 | +2.20418e-05 / +0.0814673 | -9.80333e-05 / +9.0912e-05 | +0.000188945 / `per-case table` | -0.448631 | -0.0762948 / -20.2086 | -0.388471 / +0.192822 |

Llama 9쌍에서 Soft는 W Eff `+1/90`, Gen `0/180`, Loc `+2/900`; mean terminal gap `−0.0615765`였다. 같은 쌍에서 P `+0.00250302`, common capacity `+0.00152444`, energy `+1.92872`였다. Qwen 10쌍에서 Soft는 W Eff `0/100`, Gen `+3/200`, Loc `+2/1000`; gap `+0.000188945`, P `−0.448631`, common capacity `−0.0762948`, energy `−20.2086`였다.

Soft weighted-strength residual은 모든 실행 step에서 locked tolerance 안이고 reduced-strength candidate는 0이다. Llama Soft의 fallback 3건은 exact Neutral full-strength였고, Qwen fallback은 0이다.

## 6. P2 residual-transport writer → P1DW shared writer

동일 model/case/stream/order/evaluator의 P2R2를 reference로 사용했다. 아래 값은 `P1DW - P2R2`이며 Llama Soft는 9쌍이다.

| model | arm | pairs | ΣΔz E/G | mean Δz Eff/Gen NLL | ΣΔW E/G/L | mean ΔW Eff/Gen NLL | mean Δz/W full6 | mean Δgap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | NEUTRAL | 10 | -2 / +1 | +0.104392 / -0.0420745 | -1 / -3 / +1 | +0.102727 / +0.0806188 | +0.102736 / +0.103874 | +0.00113795 |
| llama3-8b-inst | SOFT | 9 | +0 / +1 | +0.0761006 / -0.0321196 | +0 / -3 / +7 | -0.0131281 / +0.121908 | +0.0848448 / +0.0261288 | -0.058716 |
| qwen2.5-7b-inst | NEUTRAL | 10 | +0 / +4 | +0.000180166 / -0.197706 | +0 / +2 / +2 | +0.00210471 / -0.0615767 | +0.000138474 / +0.00152417 | +0.00138569 |
| qwen2.5-7b-inst | SOFT | 10 | +0 / +6 | +9.57751e-05 / -0.154295 | +0 / +5 / +4 | +0.0021331 / +0.0168329 | +4.01779e-05 / +0.00161644 | +0.00157626 |

P2R2 legacy raw capacity는 P2R7 actual-BF16 common coordinate와 다르므로 `NOT_COMPARABLE_LEGACY_COORDINATE`; P2R2 common BF16 energy는 `NOT_RECORDED`이다.

## 7. Official AlphaEdit → P1DW

동일 model+case Official reference를 공통 arm-independent baseline으로 join했다. 아래 값은 `P1DW - Official`이다.

| model | arm | pairs | ΣΔE/G/L | mean ΔEff NLL / margin | Official latent-z / Gen·Loc continuous / common cost |
|---|---|---:|---:|---:|---|
| llama3-8b-inst | NEUTRAL | 10 | -1 / +1 / -17 | +0.204815 / +0.631775 | NOT_RECORDED / NOT_RECORDED / NOT_RECORDED |
| llama3-8b-inst | SOFT | 9 | +0 / +0 / -9 | +0.0879393 / +0.878393 | NOT_RECORDED / NOT_RECORDED / NOT_RECORDED |
| qwen2.5-7b-inst | NEUTRAL | 10 | +0 / -5 / +13 | -0.0294006 / +1.79152 | NOT_RECORDED / NOT_RECORDED / NOT_RECORDED |
| qwen2.5-7b-inst | SOFT | 10 | +0 / -2 / +15 | -0.0293785 / +1.93784 | NOT_RECORDED / NOT_RECORDED / NOT_RECORDED |

## 8. 모델별 과학 분석

### Llama3-8B-Instruct

- Target 유지: P2R2 대비 Neutral z Eff/Gen count는 `−2/+1`, Soft 9쌍은 `0/+1`; z Eff NLL은 Neutral `+0.104392`, Soft `+0.0761006`, z Gen NLL은 `−0.0420745/−0.0321196`였다. Eff 연속값과 count는 Neutral에서 유지되지 않았고 Gen은 유지 또는 증가했다.
- Writer hard tail: P2R2 대비 Neutral mean terminal W−z gap은 `+0.00113795`, Soft는 `−0.0587160`였다. Production Soft−Neutral 9쌍 gap은 `−0.0615765`이고 W E/G count는 `+1/0`이다. 따라서 shared writer의 hard-tail 감소 신호는 Soft에서 관측됐지만 Neutral에서는 관측되지 않았다.
- Edit strength/Official: Neutral은 Official 대비 E/G/L `−1/+1/−17`, Soft 9쌍은 `0/0/−9`; Eff NLL delta는 `+0.204815/+0.0879393`이다. Official 수준의 모든 terminal 지표 동시 회복은 기록되지 않았다.
- Barrier: Soft는 same-strength였고 Loc `+2/900`, gap 감소를 기록했지만 P/common capacity/energy가 모두 증가했다. 동일 strength에서 update cost 감소 조건은 충족되지 않았다.
- Typed outcome: Soft case04의 no-positive direction 1건과 Neutral weighted nonlinear overshoot 1 step이 있다. Soft endpoint 분모는 9/10이다.

### Qwen2.5-7B-Instruct

- Target 유지: P2R2 대비 z Eff count는 두 arm 모두 동일, z Gen은 Neutral/Soft `+4/+6`; z Eff NLL은 `+0.000180166/+0.0000957751`, z Gen NLL은 `−0.197706/−0.154295`였다. discrete target strength는 유지됐고 Gen count가 증가했다.
- Writer hard tail: P2R2 대비 mean terminal W−z gap은 Neutral/Soft `+0.00138569/+0.00157626`로 증가했다. W Gen count는 `+2/+5`였으므로 discrete strength 증가는 있었지만 z→W NLL 소실 감소는 기록되지 않았다.
- Edit strength/Official: Official 대비 E/G/L은 Neutral `0/−5/+13`, Soft `0/−2/+15`; Eff NLL delta는 `−0.0294006/−0.0293785`다. Eff는 맞았지만 Gen count는 Official에 미달했다.
- Barrier: Soft는 same-strength에서 P/common capacity/energy를 각각 `−0.448631/−0.0762948/−20.2086` 줄였고 E/G/Loc count는 `0/+3/+2`였다. 반면 Gen NLL과 terminal W−z gap은 `+0.0814673/+0.000188945` 증가했다. cost 감소와 count 비붕괴는 관측됐고 gap 감소는 관측되지 않았다.
- Locality 경계: Soft−Neutral에서 Loc `+2`와 W−z gap 증가가 함께 있어 계약의 `UNDER_EDIT_LOCALITY_ILLUSION` 신호 조건에 해당한다. 동시에 Gen count `+3`을 별도 기록한다.

## 9. Failure taxonomy

| taxonomy | 관측 |
|---|---|
| TARGET_HARD | terminal z full6 p90/worst와 request rows를 기록; 사전 binary threshold가 없어 count는 NOT_BINARIZED |
| WRITER_HARD | terminal W−z gap p90/worst와 request rows를 기록; Llama Soft 쌍에서 감소, Qwen에서 증가 |
| WEIGHT_CONCENTRATION | 최대 omega Llama N/S `0.999067/0.997581`, Qwen `0.921489/0.930508`; 최소 Neff는 각각 `1.001868/1.004852`, `1.173986/1.152883` |
| SHARED_ROUTE_NO_POSITIVE_DIRECTION | Llama Soft case04 outer4에서 1 case; typed incomplete |
| NONLINEAR_OVERSHOOT | Llama Neutral 1 step; 나머지 0 |
| SOFT_NO_DOF | fallback 외 same-state selected pi=Neutral pi인 step 0; fallback 3 step은 별도 |
| BARRIER_PROXY_MISMATCH | aggregate에서 Structural-P 감소와 common capacity/Loc 악화가 함께 나타난 model은 0; per-case 원값은 comparisons.json에 유지 |
| UNDER_EDIT_LOCALITY_ILLUSION | Qwen Soft−Neutral aggregate에서 Loc +2와 gap +0.000188945가 함께 관측; Llama는 gap 감소 |
| TECHNICAL_INVALID | 0 |

## 10. Compute ledger와 runtime

| model | arm | endpoint | target F/B | KL F/B | response F/VJP | weighted F/B/tokens | post-write F | mat | target/writer/eval/total mean sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3-8b-inst | NEUTRAL | 10 | 1650/1200 | 1250/1200 | 400/80 | 400/400/84768 | 450 | 80 | 29.9124/131.882/2.42749/178.751 |
| llama3-8b-inst | SOFT | 9 | 1485/1080 | 1125/1080 | 360/72 | 380/380/80524 | 405 | 72 | 30.8716/134.939/2.41909/182.679 |
| qwen2.5-7b-inst | NEUTRAL | 10 | 1650/1200 | 1250/1200 | 400/80 | 400/400/77864 | 450 | 80 | 21.4361/142.335/2.26702/181.082 |
| qwen2.5-7b-inst | SOFT | 10 | 1650/1200 | 1250/1200 | 400/80 | 400/400/77864 | 450 | 80 | 23.0522/155.638/2.2622/196.32 |

Valid 39 endpoints의 authoritative terminal compute 합계는 target F/B `6435/4680`, KL F/B `4875/4680`, physical capture F `1755`, weighted response F/VJP `1560/312`, post-write F `1755`, writer materialization `312`이다. Incomplete prefix는 target receipt 15, writer/response/materialization receipt 4개이며 실패 endpoint의 top-level terminal compute aggregate는 `NOT_RECORDED_TYPED_INCOMPLETE`다. 316 step의 weighted objective F/B/tokens는 `1580/1580/321020`이다.

Array task wall은 result terminal 기준 Llama `3574.791342s`, Qwen `3847.138456s`다. Endpoint 평균 runtime은 위 표처럼 target/writer/evaluator/total로 분리했다. Peak GPU/host memory와 scheduler MaxRSS는 scientific receipt에 `NOT_RECORDED`다.

## 11. 최종 질문에 대한 분리 답변

- Llama: P2 target의 Gen strength는 유지됐지만 Neutral Eff target 지표와 연속 Eff NLL은 P2R2 수준을 유지하지 못했다. P1DW Soft는 Neutral 대비 W−z hard-tail과 Loc를 개선했으나 update P/common capacity/energy를 줄이지 못했고 1개 typed incomplete가 있었다. 따라서 ‘strong target 유지 + hard-tail 감소 + same-strength cost/locality 개선’의 세 조건이 동시에 성립하지 않았다.
- Qwen: target discrete E/G는 P2R2 수준을 유지하거나 증가했다. P1DW는 P2R2보다 W E/G/L count가 같거나 높았지만 terminal W−z NLL gap은 증가했다. Soft는 same-strength에서 P/common capacity/energy를 줄이고 E/G/Loc count를 비붕괴시켰으나 gap과 Gen NLL은 증가했다. 따라서 cost/count 조건은 성립했지만 z→W 소실 감소 조건은 성립하지 않았다.
- 두 모델을 평균해 단일 결론을 만들지 않았다. `scientific_promotion=false`다.

## 12. FACT / INFERENCE / TECHNICAL_FAIL / SCIENTIFIC_FAIL / NOT_RECORDED

### FACT

- 위 raw-free 원값, 산술 delta, 40 attempts/39 endpoints/1 typed incomplete, 316 step/3,160 request-step, source/stream/evaluator identities.
- 모든 accepted step의 5-variable routing, one-h, full weighted strength, candidate/retry/backtracking0.

### INFERENCE

- §8과 §11의 모델별 결론은 contract §15 질문에 대해 paired arithmetic와 tail 분포로만 도출했다.
- Weight concentration과 hard-tail binary threshold는 사전 수치가 없어 gate로 이산화하지 않았다.

### TECHNICAL_FAIL

- 0. 기술 무결성 위반은 없다.

### SCIENTIFIC_FAIL / typed scientific outcome

- Llama Soft case04 `SCIENTIFIC_SHARED_WRITER_NO_POSITIVE_DIRECTION` 1건. 실패 prefix는 endpoint로 승격하거나 impute하지 않았다.
- `NO_SEMANTIC_DEFICIT`은 0건이다.

### NOT_RECORDED / NOT_COMPARABLE

- Official latent-z, Gen/Loc continuous NLL·margin, Structural-P/common capacity/common BF16 energy/edit-core.
- P2R2 및 P1R43/P1R35 legacy raw capacity와 P2R7 common actual-BF16 capacity의 직접 비교는 `NOT_COMPARABLE_LEGACY_COORDINATE`.
- Stepwise heldout Eff/Gen/Loc는 계약상 0회라 `NOT_RECORDED`.
- Llama Soft case04 terminal E/G/L/z/W는 typed incomplete라 `NOT_RECORDED`, imputation 0.

## 13. 산출물

- `per-attempt.json`: 40 rows.
- `per-step.json`: 316 rows.
- `per-request-step.json`: 3,160 rows.
- `comparisons.json`: P1AGG→P1DW, DW N→S, P2R2→P1DW, Official→P1DW case rows.
- `aggregates.json`: 4 cell aggregates 및 comparison aggregates.
- `independent-review.json`, `analysis-manifest.json`, `analysis-receipt.json`: rehash/gate/rooted receipt.

