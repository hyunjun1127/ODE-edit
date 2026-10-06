# PRICE: layer cap·기본 예산·극소 block·게시 누락 검토

작성일: 2026-10-06 KST. 상태: **검토와 후속 설계 완료, method 수정·신규 실험 제출 미실시**.

핵심 판단은 기본 예산 1.0과 cap 제거를 독립 arm으로 비교하되, 그 전에 모든 비교 arm에 동일한 breakpoint zero 교정을 적용해야 한다는 것이다. 극소 block은 실현 비율의 분모를 오염시키며 norm regularizer의 gradient에도 영향을 줄 수 있다. 현재 PRICE에서 우선 개선할 관측 대상은 paraphrase의 초기 획득이고, 함께 보존해야 할 성질은 누적 유지 성능이다.

## 1. 검토 범위와 증거

- 게시 기준: `677dad89`, 실제 PRICE 실행 source: `0415aba3c160170d306be8196792f198dad4d122`.
- 게시된 W15 manifest에 봉인된 B1–B15, 1,500개 요청만 검토했다. server4의 scalar 원자료 90개, 68,784,876 bytes를 SHA-256 및 크기로 대조했다. B16 이후 결과를 포함하지 않는다.
- 입력은 각 batch의 `entry-price.json`, `fit.json`, `events.jsonl`, `commit.json`, `writer/realization.json`, `writer/subject-alltoken-gap.json`이다. 모델 forward·backward·solve·toy 실험 없이 stdlib로 집계했다.
- 별도의 1회 실행 상태 조회에서 PRICE job 59768은 RUNNING, collector 59769는 dependency PENDING이었다. 완료 여부에 대한 주장은 하지 않는다. 실행 source, 설정, 원자료, job은 변경하지 않았다.
- FLAT/REVERSE는 사용자가 취소한 대조군이다. 이 설계는 그 작업을 재개하지 않는다.

재현 입력과 SHA는 [evidence-summary.json](evidence-summary.json), 집계기는 [recover_diagnostics.py](recover_diagnostics.py)에 있다. 이전 실험의 성능은 [게시 W15 보고서](../../../experiment-reports/servers/server4/jlz-interference-priced-l1-2k/w15/report-ko.md)에 따른다.

## 2. cap 제거와 기본 예산 증가는 서로 다른 구간을 바꾼다

현재 제약을 `x_lr = ||R_lr|| / a_lr`로 쓰면 다음과 같다.

```text
sum_l pi_lr * x_lr <= beta_r,   pi_lr >= 1
x_lr <= 0.75
beta_base = 0.75
beta_max,r = 0.75 * max_l pi_lr
```

기본 예산에서 `x_lr <= beta_r / pi_lr <= 0.75`이므로 **local cap은 이미 공유 제약에 포함된다**. 동일 proposal·가격·anchor에서 CAP075와 FREE075의 정확한 사영은 같다. stage 0에서 cap에 닿았다는 기록을 독립적인 local cap 병목으로 해석하면 안 된다.

특정 층에서 cap 제거가 feasible set을 넓히는 조건은 `beta_r > 0.75*pi_lr`이다. 따라서 beta=1.0에서는 `pi_lr < 4/3`인 층에서 차이가 가능하며, 최저가 층은 항상 여기에 포함된다. CAP100은 층별 0.75를 유지하면서 추가 budget을 다른 층에 배분할 수 있고, FREE100은 최저가 한 층에도 상대 norm 1.0까지 허용한다. 어느 층에 실제 배분되는지는 loss gradient와 optimizer 결과에 달려 있다.

복구한 controller 기록은 다음과 같다.

| 최종 상태 | 요청 수 | 비율 |
|---|---:|---:|
| SATISFIED_BASE | 1,378 | 91.87% |
| SATISFIED_EXPANDED | 94 | 6.27% |
| UNSATISFIED_MAX | 28 | 1.87% |
| UNSATISFIED / ZERO_STEP | 0 / 0 | 0% |

확장 stage 0/1/2/3/4의 요청 수는 **1,378/32/12/8/70**이다. 확장된 요청은 122개(8.13%)이며, 확장 후 cap 접촉을 경험한 요청은 92개(전체의 6.13%)이다. 확장된 1,452 request-update 중 1,046개(72.04%)가 한 층 이상 cap에 닿았다. 전체 36,000 request-update의 2.91%이다. 이 비율은 기존 궤적의 직접 접촉 범위다. cap 변경에 따른 다른 owner의 반응과 이후 batch 변화까지 6.13%로 제한된다는 뜻은 아니다.

모든 request-update에서 shared budget이 active였고, 최종 `weighted_spend / beta`도 약 1이었다(차이는 약 1e-9). 현재 부족함을 “공유 예산을 남기고 있다”로 설명할 근거는 없다. cap 제거는 같은 가격 예산 안에서 배분을 바꾸고, beta 증가는 허용 예산 자체를 늘린다.

cap-free도 무제한 write는 아니다.

```text
||R_lr|| <= beta_r * a_lr / pi_lr
max_R sum_l ||R_lr||^2 = beta_r^2 * max_l (a_lr/pi_lr)^2
```

다만 이것은 유한성이지 locality 보장이 아니다. 관측된 `pi_max` 최대 3.62194를 기존 최대 예산식에 넣으면, cap-free에서 최저가 층의 상대 norm을 최대 2.71646까지 허용할 수 있다. 이는 feasible upper bound이며 실제 수행된 write가 아니다. 따라서 local cap을 없애면서 최대 예산까지 무심코 키우면 비교할 정책이 달라진다.

근거: [controller-summary.csv](controller-summary.csv), [cap-activity.csv](cap-activity.csv).

## 3. beta_base=1.0을 볼 이유와 해석 한계

PRICE의 B1–B15 paraphrase preference는 편집 직후 birth cohort 합계 **2,743/3,000 = 91.43%**, W15 **2,744/3,000 = 91.47%**이다. 그 사이 lost 35, gained 36이므로 개별 망각이 없다는 뜻은 아니지만, aggregate로 보면 획득 수준이 유지된다. 동일 first500은 W5→W15 PS 92.8→91.6%, NS 87.32→84.76%이다.

따라서 beta_base=1.0 arm은 초기 paraphrase 획득을 높이면서 이 유지 성능을 보존하는지 확인할 의미가 있다. 최저가 한 층의 cap-free 최대 상대 norm은 0.75→1.0, 제곱 norm 상한은 약 1.78배가 된다. 실제 PS 개선은 보장되지 않는다. Paraphrase는 학습·가격·확장 controller 입력에 넣지 않는다.

또한 `F = mean(rewrite NLL) + 0.0625 * KL`이다. UNSATISFIED_MAX 28건의 rewrite NLL 중앙값은 0.000654이고, 비가중 KL 중앙값은 1.0968이다. `F` 미달을 모두 rewrite 부족으로 해석해 더 큰 write로 해결하려 하면 잘못된 방향일 수 있다. 두 중앙값은 각각의 분포 통계이며 동일 요청의 분해값을 더한 것이 아니다. 후속 보고에는 request별 rewrite NLL, 가중 KL, F를 함께 연결해야 한다.

“F 만족”, held-out PS preference, PS strict는 별개다. 현재 W15 PS strict는 62.5%이며, 큰 budget의 판단은 preference만으로 하지 않는다. current acquisition, 같은 cohort retention, NS, lost/gained를 함께 비교한다.

## 4. 극소 block은 단순 표시 문제가 아니다

진단 구간은 `0 < ||R_lr||/a_lr <= 1e-12`로 정의했다. 이 임계값은 기존 기록을 분류하는 용도이며, model block을 삭제하거나 “의미 없는 write”를 정의하는 기준이 아니다.

| 범위 | 전체 block | 정확한 0 | 극소 양수 | 임계값 초과 |
|---|---:|---:|---:|---:|
| 모든 postprojection update | 180,000 | 117,081 | 6,495 (3.61%) | 56,424 |
| 최종 layer × owner | 7,500 | 4,410 | 311 (4.15%) | 2,779 |

update 극소 양수의 상대 norm 중앙값은 6.36e-17이다. 저장된 zero mask와 numeric exact zero의 불일치는 0건이었다. 즉, 집계기가 zero mask를 잘못 읽는 문제가 아니라 **projector가 만든 미세 양수를 그대로 nonzero로 취급하는 문제**다. 최종 nonzero 층 수 평균도 3,090/1,500=2.060이고, 진단 임계값 초과만 세면 2,779/1,500=1.853이다. 두 수치는 서로 다른 정의이며 후자를 기존 실제 support로 덮어쓰면 안 된다.

`projection.py`는 breakpoint 값을 set으로 저장해 owner와 ZERO/CAP 종류를 잃는다. 선택한 breakpoint에서 `max(n - tau*w, 0)`를 다시 계산하면서 미세 양수가 남을 수 있고, `length == 0` 비교를 통과하지 못한다. FP32는 이런 작은 값도 표현할 수 있다. 기존 59721 수리는 KKT 허용 범위에서 breakpoint를 채택하도록 했으나 endpoint zero의 정확한 저장은 추가하지 않았다.

실현 지표는 target norm이 양수면 비율을 정의한다. 게시된 B3 L5 rewrite norm ratio 최대값은 **1.301e15**다. 복구한 전체 terminal canonical 비율 평균도 2.289e13이다. 극소 target을 분리한 조건부 분포에서도 중앙값 0.6388, p95 9.586, 평균 11.657, 최대 23,310이 남는다. 큰 값에는 작은 분모와 다른 owner의 response가 함께 기여할 수 있으므로, 비율 이상치를 제거했다고 실현 mismatch가 해결된 것으로 해석할 수 없다.

더 중요한 경로는 norm gradient다.

```text
regularizer_l,r = (0.5 / a_star,r^2) * ||R_lr||
nonzero gradient = (0.5 / a_star,r^2) * R_lr / ||R_lr||
exact-zero gradient = 0  (현 구현의 선택)
```

즉 1e-15 block에도 norm 자체와 무관한 크기의 gradient가 생긴다. 다음 update의 Adam moment와 층별 gamma에 영향을 줄 수 있다. 바로 그 작은 write가 모델 출력을 거의 바꾸지 않더라도, 이후 최적화 궤적까지 같다고 주장할 수 없다. 영향의 실제 성능 크기는 아직 측정하지 않았다.

교정은 다음 계약으로 제한한다.

1. breakpoint를 `(value, block, ZERO 또는 CAP)`로 유지한다. 선택된 endpoint의 해당 길이를 정확한 0/cap으로 저장하고 tied endpoint를 일관되게 처리한다. interior 해의 양수 block을 임의 임계값으로 삭제하지 않는다.
2. FP64 길이·primal·complementarity와 FP32 저장 feasibility를 기존 허용기준으로 검사한다. endpoint 변경 mask·크기를 기록한다. 허용오차를 확대하거나 저장 후 일괄 축소하지 않는다.
3. exact-zero, endpoint 교정, 진단용 tiny-target을 구분한다. raw target/action norm, ratio의 defined 여부·사유·표본 수·단위를 저장한다. norm regularizer의 0 처리와 실제 block은 일치해야 한다.
4. R=0인 owner도 다른 owner write로 반응할 수 있다. 그 action을 0으로 만들지 않고 anchor로 정규화한 leakage로 남긴다. 모멘트와 task-gradient에 의한 재진입을 유지한다.
5. 기존 RS/PS/NS는 별도 평가 결과로 보존한다. 텔레메트리 재집계와 optimizer에 영향을 주는 zero 교정은 구분한다. 새 수치 교정이 적용된 CAP075가 후속 비교의 기준이어야 한다.

근거: `project/run_scripts/jlz_interference_l1/projection.py:16–45,81–98`, `telemetry.py:15–22,95–104`, `project/run_scripts/jlz_v12r/optimizer.py:17–21`. 구체 계약은 [numerical-repair-contract.json](numerical-repair-contract.json), 관측은 [tiny-block-audit.csv](tiny-block-audit.csv).

## 5. 게시 누락 복구와 배분 claim

원자료는 존재한다. `collect.py:223–225`가 expanded/statuses를 반환하지만, `:261,300–301`의 요약은 제한된 counter만 누적한다. `review_w15.py`도 가격·stage·support 진단을 게시하지 않는다. 이번 검토에서는 아래를 복구했다.

| 층 | 가격 pi 중앙값 | p95 | 최저가 요청 수 | 최저가 비율 | 최종 priced-spend share 평균 |
|---|---:|---:|---:|---:|---:|
| L4 | 1.000 | 1.252 | 872 | 58.13% | 80.66% |
| L5 | 1.083 | 1.368 | 390 | 26.00% | 11.08% |
| L6 | 1.224 | 1.683 | 142 | 9.47% | 5.40% |
| L7 | 1.437 | 2.156 | 41 | 2.73% | 1.93% |
| L8 | 1.488 | 2.363 | 55 | 3.67% | 0.93% |

tie 0/1,500, flooring 0/7,500이다. 요청별 pi_max 중앙값 1.538, p95 2.386, 최대 3.622다. 가격은 각 batch own-entry에서 계산되어 batch 내 고정되며, 이 표는 서로 다른 batch를 모은 분포다. share는 요청별 share의 평균이지 pooled energy 비율이 아니다.

L4가 항상 최저가는 아니지만 배분의 큰 부분을 차지한다. 이것은 L4를 강제하는 규칙이 아니라는 근거와 일치한다. 그러나 최저가 비율 58.13%보다 배분 비중 80.66%가 크므로, 가격만으로 배분이 결정된다는 증거는 아니다. loss gradient·anchor·optimizer도 작동한다. 취소된 가격 대조군 없이 가격의 순수 인과 효과를 주장하지 않는다.

추가 게시의 최소 계약은 다음과 같다.

- 가격: batch/layer별 raw kappa·적용 pi 분포, 최저가 count와 tie/floor, source/entry price SHA, anchor와 cap mode.
- controller: 종료 상태, 확장 stage, own update 수, base/max/current beta, 실제 weighted spend와 slack, F·rewrite NLL·가중 KL. cap 접촉은 stage 0과 확장 후를 분리한다.
- 배분: requested norm/energy/priced-spend, 실제 response energy, 정확한 support·tiny diagnostic support·재진입. 가격 최저가와 실제 최대배분 층의 교차표도 같은 owner 기준으로 게시한다.
- 실현: canonical/mean/rewrite/KL 역할과 owner/context 집계 단위, 비율 정의 가능 수·0/tiny 수, median/p90/p95와 raw 평균, anchor-normalized action/leakage. `count`를 rewrite row 수로 오표기하지 않는다.
- 안정적 집계: 가능한 역할은 `sqrt(sum ||action||^2 / sum ||target||^2)`와 energy-weighted directional/error 지표를 병기한다. 향후 rewrite context별 제곱 norm·dot 합을 축약 저장한다. 기존 owner-mean norm의 제곱을 context energy 합으로 취급하지 않는다. 없는 충분통계는 NOT_RECORDED로 표시한다.
- 성능·비용: acquisition와 같은 cohort retention, PS/NS preference 및 strict, lost/gained, fit·price·projection·telemetry·evaluation 비용. 소규모 분모에서 ratio만으로 mismatch를 단정하지 않는다.

## 6. 후속 method 실험 설계

모든 arm은 PRICE를 사용한다. 같은 수치 교정 source에서 독립 cold W0/H0로 시작하고, 이후 own committed state를 사용한다.

| Arm | beta_base | local cap | 비교 목적 |
|---|---:|---|---|
| CAP075 | 0.75 | 0.75 a | 교정 source 기준 |
| FREE075 | 0.75 | 없음 | 원래 base에서 cap 제거 |
| CAP100 | 1.00 | 0.75 a | cap 유지 시 base 증가 |
| FREE100 | 1.00 | 없음 | base 증가와 cap 제거의 결합 |

```text
native_c = 0.75                     # 바꾸지 않음
beta_max_native_scale = 0.75        # 바꾸지 않음
beta_max,r = max(beta_base, 0.75 * max_l pi_lr)
beta_r(e) = beta_base * exp((e/4) * log(beta_max,r/beta_base))
```

base=1.0에서 `pi_max < 4/3`인 요청은 최대 budget도 1.0으로 오른다. 그 수를 별도 보고하고 “상한은 같고 base만 증가했다”고 서술하지 않는다. 다른 두 arm의 최대 budget까지 1.0으로 올려 원래 기준을 바꾸지 않는다.

환원 성질도 구분해야 한다. 단일층에서 pi=1이므로 FREE075는 공유 예산 자체가 native 0.75 clamp를 재현한다. FREE100은 상대 norm 1.0을 허용하므로 그 환원을 유지하지 않는다. CAP100은 단일층에서도 local 0.75 cap을 유지하며, 이 경우 beta=1.0을 모두 사용하지 못할 수 있다. 따라서 native clamp 환원을 모든 arm의 공통 claim으로 쓰지 않는다.

이 동일성은 제약집합·사영 수준이다. soft writer의 실현 응답을 쓰는 목적함수까지 native baseline과 같다는 뜻은 아니다. 또한 `UNSATISFIED_MAX`는 controller가 최대 beta 단계에 있다는 뜻이며, cap-on의 다른 설정에서도 공유 예산을 전부 썼다는 의미로 확대하지 않는다. 현재 B1–B15에서 spend/beta가 약 1이라는 것은 별도로 측정한 사실이다.

기존 grace 12 own updates, 확장 4단계, threshold .05, 25 evaluations/24 updates, terminal payload commit, soft native writer, realized-response objective와 same-layer pullback, history 갱신은 유지한다. cap-free는 `t=max(n-tau*w,0)`의 별도 분기로 구현하며 cap을 infinity로 채우지 않는다. JSON은 `cap_mode=none`, cap null을 명시하고 검증기도 이에 맞춘다.

현재 `controller.py`는 c 하나로 cap/base/max를 묶고, `fit.py:31–32`는 profile의 c를 전달하지 않는다. `price.py` 및 `collect.py`에도 .75가 반복되어 있다. **profile 숫자 하나만 1로 바꾸는 것은 올바른 arm 구현이 아니다.** profile→prepare/preflight→controller→fit→price receipt→collect→source/config lock까지 독립 knob가 전달되어야 한다.

toy와 별도 pilot 없이 실제 BS100×20 method trajectory로 비교한다. 첫 실제 B1에서 이미 존재하는 geometry/proposal로 가격·사영·endpoint·KKT·feasibility·payload 계약을 확인하고 이어서 실행하도록 설계한다. base .75의 cap-on/off 사영 동일성은 같은 실제 proposal로 추가 LM forward 없이 점검 가능하다. 낮은 성능을 이유로 조용히 설정을 변경하거나 B1을 재시작하지 않는다.

우선 질문은 CAP100이 초기 PS를 높이고 유지 성능을 보존하는지, FREE100이 그 조건에서 더 나은 배분을 제공하는지다. FREE075는 확장 구간의 cap 효과를 분리한다. 깨끗한 네 arm 비교에는 같은 교정을 적용한 새 CAP075가 필요하다. 이전 source의 기존 PRICE를 그대로 기준으로 쓰면 cap/base 효과와 zero 교정 효과가 섞인다.

method claim은 “측정한 간섭 가격으로 공유 요청 예산을 배분하고, 실제 writer response를 목적함수에 반영한다”로 유지한다. soft writer에서 R과 response가 수치적으로 같아진다거나, 가장 싼 층이 semantic locality를 보장한다고 주장하지 않는다.

구현 계약과 전체 실행 설계는 [followup-design.json](followup-design.json)에 있다. 이번 검토에서 생성한 것은 설계·진단 파일이며, running source 수정이나 새 arm 제출은 하지 않았다.
