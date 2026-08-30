# Fixed-z non-uniqueness engineering screen — Llama/Qwen × AlphaEdit/MEMIT

> 결론: 네 model/method 조합 모두 사전 고정 engineering gate를 통과했다. 동일 direct-z, edit/history/target local equality와 matched covariance action을 유지한 candidate family 안에서도 teacher next-token KL CVaR가 duplicate noise floor보다 분명하게 달라졌다. 이는 **engineering premise screen PASS**이며 confirmatory claim이나 promotion은 아니다.

## 범위와 분모

- CounterFact 고정 8 case/order를 model/method별로 각각 평가했다. case 교체·outcome 기반 재생성은 0이다.
- case마다 direct-z 1회, reference duplicate 2회, random tangent 4축×±=8 candidates이다.
- 총 candidate 분모는 4 groups × 8 cases × 8 = 256, duplicate 분모는 64, teacher prompt 평가는 candidate당 32개이다.
- FULL-FP32, final-audit bank open/use 0, ODE/Euler/barrier/sequential/promotion 0이다.

## 최종 gate 요약

| model / method | valid cases | spread > 3×noise | median spread | median noise | gamma=0 | direct-z | W0 restore | 판정 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Llama / AlphaEdit | 8/8 | 8/8 | 9.814370e-05 | 0 | 8/8 | 8 / recompute 0 | 8/8 | PASS |
| Qwen / AlphaEdit | 8/8 | 8/8 | 8.274172e-05 | 0 | 8/8 | 8 / recompute 0 | 8/8 | PASS |
| Llama / MEMIT | 8/8 | 8/8 | 7.294424e-06 | 0 | 8/8 | 8 / recompute 0 | 8/8 | PASS |
| Qwen / MEMIT | 8/8 | 8/8 | 1.036940e-04 | 0 | 8/8 | 8 / recompute 0 | 8/8 | PASS |

## Left-padding 및 Official 호출 결속

| model / method | 최대 batch/singleton/reorder/pad-length 오차 | FP32 tolerance | Official semantic calls | input/attention/position identity | pad 결속 |
|---|---:|---:|---:|---|---|
| Llama / AlphaEdit | 3.429484e-06 | 3.051758e-05 | 129 | PASS | explicit pad=eos, model/generation config |
| Qwen / AlphaEdit | 3.532504e-06 | 3.051758e-05 | 129 | PASS | explicit pad=eos, model/generation config |
| Llama / MEMIT | 3.429484e-06 | 3.051758e-05 | 89 | PASS | explicit pad=eos, model/generation config |
| Qwen / MEMIT | 3.532504e-06 | 3.051758e-05 | 89 | PASS | explicit pad=eos, model/generation config |

attention_mask nonzero columns와 cumsum position_ids를 semantic 위치의 유일한 근거로 사용했다. selected hidden, target key, edit/history key, next-token logits/NLL을 left-padded batch↔singleton, reorder, padding-length 변화로 비교했다. Official right-padding 호출은 같은 text를 left-padding hook으로 replay하여 semantic input_ids/attention/position identity를 call별로 봉인했다.

## Official baseline 및 gamma=0 재현

| model / method | endpoint exact | max target-logit rel | metric object identity | rewrite-new strict | rephrase-new strict | rewrite-new NLL mean/median/p90/max | rephrase-new NLL mean/median/p90/max |
|---|---:|---:|---|---:|---:|---|---|
| Llama / AlphaEdit | 8/8 | 0 | PASS | 8/8 | 10/16 | 8.159130e-04/6.167773e-04/0.001649/0.001906 | 1.780901/0.962107/4.150270/5.406350 |
| Qwen / AlphaEdit | 8/8 | 0 | PASS | 8/8 | 11/16 | 0.023707/0.013954/0.047804/0.073350 | 1.388447/0.644140/4.185456/5.086974 |
| Llama / MEMIT | 8/8 | 0 | PASS | 8/8 | 7/16 | 0.025606/0.003245/0.061346/0.178081 | 2.787090/3.265903/6.617598/6.988693 |
| Qwen / MEMIT | 8/8 | 0 | PASS | 8/8 | 12/16 | 0.019467/0.019416/0.037261/0.037759 | 1.265069/0.674836/3.811155/5.357556 |

## Candidate functional 및 task metric 분포

| model / method | valid candidates | CVaR mean/median/p90/max | case spread mean/median/p90/max | rewrite-new NLL mean/median/p90/max | rephrase-new NLL mean/median/p90/max |
|---|---:|---|---|---|---|
| Llama / AlphaEdit | 64/64 | 0.005745/0.005376/0.014268/0.014351 | 1.124815e-04/8.975971e-05/2.047457e-04/2.082661e-04 | 8.158813e-04/6.166581e-04/0.001906/0.001907 | 1.780904/0.961775/4.623563/5.408824 |
| Qwen / AlphaEdit | 64/64 | 0.001591/0.001342/0.003115/0.003494 | 1.245584e-04/7.174851e-05/2.268046e-04/5.047955e-04 | 0.023707/0.013953/0.073286/0.073445 | 1.388448/0.643914/4.544229/5.089259 |
| Llama / MEMIT | 64/64 | 2.071609e-04/2.223116e-04/4.001398e-04/4.107862e-04 | 7.070394e-06/6.337694e-06/1.284301e-05/1.544380e-05 | 0.025606/0.003245/0.178033/0.178155 | 2.787089/3.265756/6.937929/6.989575 |
| Qwen / MEMIT | 64/64 | 0.001119/0.001257/0.001889/0.002142 | 9.305880e-05/9.039097e-05/1.654673e-04/2.540420e-04 | 0.019467/0.019406/0.037703/0.037839 | 1.265070/0.674992/4.913247/5.362977 |

## Case별 functional spread

| case | Llama AlphaEdit | Qwen AlphaEdit | Llama MEMIT | Qwen MEMIT |
|---:|---:|---:|---:|---:|
| 21100 | 2.488913e-05 | 1.076657e-04 | 1.041888e-06 | 7.708790e-05 |
| 1477 | 7.906160e-05 | 5.047955e-04 | 5.380964e-06 | 2.540420e-04 |
| 20838 | 2.082661e-04 | 3.721466e-05 | 7.294424e-06 | 1.173564e-04 |
| 18707 | 1.562927e-04 | 4.134898e-05 | 9.348209e-06 | 2.165383e-05 |
| 14288 | 4.858640e-05 | 8.274172e-05 | 3.145949e-06 | 1.275067e-04 |
| 16426 | 2.032369e-04 | 1.053687e-04 | 1.544380e-05 | 3.179139e-05 |
| 19041 | 8.137571e-05 | 6.075529e-05 | 1.172838e-05 | 1.036940e-04 |
| 17609 | 9.814370e-05 | 5.657692e-05 | 3.179535e-06 | 1.133818e-05 |

## Equality, action, rank

| model / method | max NK_E/H/T | max target activation/logit rel | max ± action mismatch | rank median | null dimension median |
|---|---|---|---:|---:|---:|
| Llama / AlphaEdit | 1.535839e-07/6.770035e-08/9.051798e-08 | 1.743212e-07/6.041719e-04 | 0 | 34.000000 | 1.430200e+04 |
| Qwen / AlphaEdit | 1.787022e-07/4.422638e-08/5.417686e-08 | 1.412678e-07/0.001179 | 0 | 34.000000 | 1.891000e+04 |
| Llama / MEMIT | 8.602554e-08/6.997114e-08/8.301063e-08 | 1.721536e-07/1.367342e-04 | 0 | 34.000000 | 1.430200e+04 |
| Qwen / MEMIT | 1.060501e-07/6.405631e-08/5.124297e-08 | 1.406028e-07/0.001015 | 0 | 34.000000 | 1.891000e+04 |

- [matched ± candidate CVaR scatter](matched-pair-cvar-scatter.svg): 동일 축의 −/+ candidate CVaR. 점선은 y=x이다.
- [covariance-action overhead 대 CVaR drift](action-drift-scatter.svg): action-matched tangent의 상대 action 증가와 Official reference 대비 functional drift.
- [null dimension 대 case spread](rank-spread-scatter.svg): constraint rank 이후 남은 input tangent dimension과 case functional spread.
- 정확한 점 데이터는 [per-candidate.csv](per-candidate.csv), case 데이터는 [per-case.csv](per-case.csv)에 있다.

## 계산 및 완전성

| model / method | wall seconds | peak GPU allocated | FULL-FP32 | final audit used |
|---|---:|---:|---|---:|
| Llama / AlphaEdit | 122.576 | 39537181184 bytes | PASS | 0 |
| Qwen / AlphaEdit | 152.075 | 42430530048 bytes | PASS | 0 |
| Llama / MEMIT | 191.818 | 40007033344 bytes | PASS | 0 |
| Qwen / MEMIT | 277.356 | 42973796864 bytes | PASS | 0 |

## 실패 및 제외 lineage

최종 science denominator의 invalid/failed case는 0이다. 아래 technical/superseded attempts는 결과 선택에 사용하지 않았다.

| job/lineage | 분류 | denominator | 근거 |
|---|---|---:|---|
| 29066 | PURE_TECHNICAL_RESOURCE_EXCLUSION | 0 | 180GB request could not be scheduled; model/science endpoint 0 |
| 29069 | PURE_TECHNICAL_SOURCE_EXCLUSION | 0 | target-key tensor orientation failed before valid candidate panel |
| 29072 | PURE_TECHNICAL_NUMERICAL_LOCK_EXCLUSION | 0 | existing FP32 endpoint/rank lock was not yet bound; superseded without tolerance tuning |
| 29083,29087 | SUPERSEDED_TELEMETRY_INCOMPLETE | 0 | valid AlphaEdit smoke/screen preceded complete Official-to-hook and key padding receipt |
| 29098 | PURE_TECHNICAL_ARTIFACT_PATH_EXCLUSION | 0 | Official MEMIT local covariance alias missed pinned cache and failed offline before candidate panel |
| 29111,29122 | SUPERSEDED_TELEMETRY_INCOMPLETE | 0 | valid MEMIT smoke/screen preceded complete Official-to-hook and key padding receipt |
| 29144_1 | PURE_TECHNICAL_SCHEDULER_EXCLUSION | 0 | task-owned Qwen AlphaEdit smoke cell canceled after 15s during walltime backfill repair; terminal result 0 |
| 29145_[0-1] | PURE_TECHNICAL_SCHEDULER_EXCLUSION | 0 | pending-only MEMIT smoke array replaced by one-hour walltime submission; model endpoint 0 |

## Factual interpretation

1. 네 조합 모두 exact target identity와 matched action을 유지한 8-candidate family를 8/8 case에서 생성했다.
2. duplicate 평가가 이 실행에서 bitwise-deterministic해 noise가 0이었고, 모든 case의 positive spread가 사전 3×noise 조건을 넘었다. 따라서 수치의 절대 크기와 함께 raw per-candidate 값도 보존한다.
3. 결과는 fixed-z local equality만이 아니라 full-model target activation/logit 및 32-prompt teacher-KL CVaR로 검증됐다.
4. 이는 동일 direct-z/action 아래 기능 보존 결과의 non-uniqueness가 engineering screen에서 관찰됐다는 근거다. 8-case screen이므로 population-level confirmatory claim, architecture pooling, sequential/continual claim, 자동 promotion은 하지 않는다.
5. 계약상 다음 단계는 unused CounterFact IDs의 별도 128 request/model × 3 seed confirmatory gate 설계이며, 이번 task에서는 실행하지 않았다.

scientific_promotion=false
