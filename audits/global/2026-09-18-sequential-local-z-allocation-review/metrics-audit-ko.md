# Sequential local-z v2: 원시 성능·품질 제약·비용 독립 검산

2026-09-18. 기존 보고서 및 분석 코드를 읽은 뒤, 해당 분석 모듈을 import하지 않는 별도 Python reducer로 server4의 원시 JSON을 read-only 검산했다. 새 GPU 실행·모델 재평가·실험 수정·원시 자료 전송은 없다.

재현 코드: [verify_metrics.py](verify_metrics.py). 이번 실행의 aggregate 결과: [metrics-checks.json](metrics-checks.json). 실행 결과 `PASS`, final 평가행78,000개, online score446개, candidate guard446개를 대조했다. 파일에는 개별 raw 평가행 대신 집계·차이·선택 진단만 저장했다.

## 결론

보고된 W10 count와 분모, strict, 조건부 NS retention, E/H/B 집계 및 feasible 분류는 검산 범위에서 일치한다. 핵심 문제는 수치 오류보다 **보장하는 품질과 해석하려는 품질의 차이**다. Native-context E/현재 canonical 성공 ID/Past64 보호는 P 일반화나 공식 N 보존을 보장하지 않는다. 같은 진입 상태 비교에서도 S64를 개선하면서 P와 N을 함께 악화한 실제 선택이 있다.

C45678의 W10 preference R/P/N은 N4 대비 +0.10/+0.10/+0.86pp로 개발 stream의 사전 지정된 방향을 만족한다. 그러나 P strict는 −1.80pp, 두 paraphrase가 모두 strict인 요청은 −3.20pp이며 P target-new NLL도 평균 악화한다. 그러므로 “편집 품질 손실 없는 다층 개선”으로 일반화할 수 없다. C48은 P −0.90pp로 사전 지정된 PS 손실 0pp 조건을 만족하지 않는다. G48 대비 연속 탐색의 우월성도 이 결과에서 입증되지 않는다.

## 검산 입력과 방법

원격 공통 root: `/data/janghj/ODE-edit/local/sequential-local-z-allocation/20260917-v2`.

- `arms/{N4,F48,G48,C4,C48,C45678}/attempt-v1/output/B010/seen-full.json`: 총 78,000 R/P/N 원문항 평가행. 원 CounterFact first1000의 case ID, prompt index, prompt/target identity SHA를 별도 재생성하여 대조했다. R/P는 new NLL < true NLL, N은 true NLL < new NLL로 성공을 다시 계산했다. 모든 arm 동점 0.
- 각 `B*/episode/scores/*.json`: 446 score 파일 전체. 100개 요청의 native 6-context 평균 E, 64 canonical Past H, 64문서 KL B, canonical strict/pair ID를 rows에서 재계산하고 저장 scalar와 맞췄다. 각 후보의 E/H 허용폭 1e−4와 성공 ID 부분집합 조건으로 feasible을 다시 판정했다. 저장된 446 completed candidate 판정과 일치한다.
- `B{001,005,010}/candidate-observer/*-binding.json` 및 `*-current.json`: 봉인된 state token으로 own-entry N4와 selected를 연결한 후, 동일 prompt identity별 P/N lost/gained와 NLL 차이를 재계산했다.
- 60개 `selected-current.json`, W0 observation: at-write→W10 전이와 W0-success-conditioned NS를 재계산했다.
- 552개 `episode/fits/*/receipt.json` 및 6개 `terminal.json`: target/Adam/loss/solve/online-score와 program/controller 시간 합계를 별도 집계했다.

E의 request별 context 평균은 실행 source가 FP32 `loss.mean()`을 사용한다. 저장 context scalar를 Python double로 평균하면 요청별 최대 7.9473e−7 차이가 있다. 이 차이는 FP32 reduction에서 생기며, 위 검사에서는 해당 단계에 1e−6의 산술 비교 허용폭을 사용했다. **실험 자체의 E/H guard 1e−4는 변경하지 않았고, feasible 재검산은 저장 원 scalar로 exact 비교했다.** Request 평균 이후 E/H/B와 성공 ID는 저장 reducer 의미와 일치한다. 모델 forward의 정확성을 새로 검증한 것은 아니다.

참고 source: 실행 `source-v1/project/run_scripts/sequential_local_z_allocation/metrics.py`의 `training_E`, `canonical`, `score`; 사후 `completed-review-20260918-v1/worktree/project/run_scripts/sequential_local_z_allocation/analysis/completed_review_20260918/{reducer,performance}.py`. E는 token→context→request 평균이고 canonical 평균은 별도 값이며 제약이 아니다. P/N evaluator는 post-seal observer다.

## W10 재계산 결과와 지표별 해석

| Arm | R /1000 | P /2000 | N /10000 | P TF-strict /2000 | 두 P 모두 strict /1000 |
|---|---:|---:|---:|---:|---:|
| N4 | 999 | 1934 | 8026 | 1354 | 513 |
| F48 | 1000 | 1909 | 8324 | 1280 | 457 |
| G48 | 1000 | 1938 | 8088 | 1337 | 500 |
| C4 | 998 | 1932 | 8046 | 1349 | 511 |
| C48 | 999 | 1916 | 8170 | 1313 | 488 |
| C45678 | 1000 | 1936 | 8112 | 1318 | 481 |

P preference는 두 target 중 new를 더 선호하는지이며, TF-strict는 teacher-forced target 전체 token이 argmax인지다. 어느 것도 자유 생성 평가와 동치가 아니다.

| N4 대비 | P preference lost/gained | P strict lost/gained | P new-NLL Δ평균 | ΔNLL p99 | ΔNLL 최대 |
|---|---:|---:|---:|---:|---:|
| F48 | 50/25 | 149/75 | +0.200480 | +4.632597 | +10.939518 |
| G48 | 22/26 | 94/77 | +0.066126 | +4.569659 | +9.960560 |
| C4 | 6/4 | 23/18 | +0.007410 | +1.344785 | +4.634196 |
| C48 | 40/22 | 116/75 | +0.163303 | +3.992435 | +8.312148 |
| C45678 | 30/32 | 120/84 | +0.074610 | +4.152845 | +9.183013 |

모든 신규 arm의 평균 P new-NLL이 N4보다 높다. G48/C45678의 작은 preference gain은 이 사실과 동시에 성립한다. N4에서 성공한 P를 모두 유지한 것도 아니다. C45678은 P 30개를 잃고 32개를 얻었으며 strict는 120개를 잃고 84개를 얻었다.

원시 행에서 case 단위 cluster bootstrap 2,000회, seed20260918로 다시 계산한 기술적 95% CI:

| 비교 | P preference Δpp [CI] | P strict Δpp [CI] | N preference Δpp [CI] |
|---|---:|---:|---:|
| G48−N4 | +0.20 [−0.55,+0.95] | −0.85 [−2.10,+0.45] | +0.62 [+0.23,+1.02] |
| C45678−N4 | +0.10 [−0.75,+0.95] | −1.80 [−3.35,−0.30] | +0.86 [+0.37,+1.37] |
| C45678−G48 | −0.10 [−0.90,+0.70] | −0.95 [−2.35,+0.45] | +0.24 [−0.22,+0.68] |
| C48−G48 | −1.10 [−1.85,−0.35] | −1.20 [−2.40,+0.10] | +0.82 [+0.39,+1.23] |

한 고정 개발 순서의 행 재표집이며, 다른 순서·seed·unseen 구간의 재현성을 나타내지 않는다. 특히 C45678의 PS +0.1pp를 통계적 비열화 증명으로 읽을 수 없다.

## Guard를 통과한 실제 선택에서 드러나는 품질 간극

아래는 각 arm의 **같은 batch 진입 상태에서 만든 own N4 대비 selected**이며 독립 N4 lifelong chain과의 차이가 아니다. Canonical 악화 수는 요청별 new NLL 증가가 1e−4보다 큰 개수다. 이 기준은 진단용이며 새 guard로 사용하지 않았다.

| 선택 | Δnative E | ΔS64 B | current canonical 악화 /100 | canonical Δ평균 | canonical Δ최대 |
|---|---:|---:|---:|---:|---:|
| C48 B2 | −0.01054053 | −0.00056919 | 82 | +0.00137325 | +0.06413206 |
| C48 B4 | +0.00009957 | −0.00035184 | 89 | +0.00067670 | +0.00620494 |
| C48 B5 | −0.00514683 | −0.000375998 | 91 | +0.00439839 | +0.09563675 |
| C48 B9 | +0.00009203 | −0.00069526 | 93 | +0.00460724 | +0.10024821 |
| C48 B10 | −0.00060285 | −0.00096355 | 85 | −0.01236215 | +0.14550633 |
| C45678 B2 | −0.01692625 | −0.00064066 | 80 | −0.00291984 | +0.02798429 |
| C45678 B5 | −0.00102587 | −0.00063353 | 82 | +0.00820596 | +0.07933029 |
| C45678 B10 | −0.00169627 | −0.00087442 | 88 | −0.00798065 | +0.13675487 |

이는 native context 평균뿐 아니라 canonical 평균이 개선되어도 개별 요청의 다수가 약해질 수 있음을 보여준다. C48 B10은 canonical 평균이 개선되지만 85/100 요청이 악화했다. Worst current ID는 C48 B5 case9401(+0.09563675), C48 B10 case630(+0.14550633), C45678 B5 case15446(+0.07933029), C45678 B10 case10711(+0.13675487)이다.

Past64도 평균/성공 ID 보호이며 요청별 NLL 보호가 아니다. 예를 들어 C48 B10의 H는 허용범위 내 +0.000089919지만 13/64 요청이 >1e−4 악화하고 case2798은 +0.00430094 악화한다. C45678 B5는 H가 −0.00005301 개선하면서 14/64가 악화한다. 이러한 사례들은 구현 위반이 아니라 설계에서 명시적으로 허용한 결과다.

공식 observer의 same-entry 변화는 더 직접적이다. 아래 P 분모200, N 분모1000이며 모두 guard를 통과한 선택이다.

| 선택 | P lost/gained (Δpp) | P strict lost/gained (Δpp) | P new NLL Δ평균 | N lost/gained (Δpp) | N true NLL Δ평균 |
|---|---:|---:|---:|---:|---:|
| G48 B10 | 2/2 (0) | 5/2 (−1.5) | +0.135243 | 5/6 (+0.1) | −0.013942 |
| C48 B5 | 3/1 (−1.0) | 8/0 (−4.0) | +0.197325 | 9/1 (−0.8) | +0.034648 |
| C48 B10 | 10/2 (−4.0) | 18/6 (−6.0) | +0.408016 | 2/13 (+1.1) | −0.034260 |
| C45678 B5 | 7/4 (−1.5) | 18/8 (−5.0) | +0.198296 | 11/10 (−0.1) | +0.021662 |
| C45678 B10 | 5/4 (−0.5) | 13/11 (−1.0) | +0.135486 | 4/7 (+0.3) | −0.022568 |

C48/C45678 B5는 S64가 개선되지만 같은 endpoint의 공식 N preference 및 true likelihood가 악화한다. 따라서 W0 KL 축소→모든 locality 개선이라는 해석은 실제 반례로 배제된다. C48 B10의 PS −4pp는 새 층의 의미 있는 편집 방향을 더했다는 주장과 함께 반드시 읽어야 한다. Official P/N은 사후 관측이며 이 결과에 맞추어 온라인 선택을 바꿨다는 증거는 없다.

## Lifelong 성능과 own-entry 기준은 다른 질문이다

OwnN4 fallback은 이전 batch의 suffix 편집을 유지한 채 현재 batch L4를 full-write한다. 따라서 ownN4를 여러 번 선택했다고 독립 N4 chain으로 돌아간 것이 아니다. C4의 최종 S64 0.02138810은 N4 chain 0.02128090보다 오히려 약간 높다. 각 batch에서 ownN4 대비 S64 비증가를 보장해도 최종 독립 N4 chain 대비 우위를 보장하지 않는 구체적 예다.

W0 성공 N 분모는 모든 arm 공통 8,820이다. 최종 유지 수 N4/F48/G48/C4/C48/C45678는 각각 7,852/8,168/7,921/7,868/8,004/7,942로 보고서와 일치한다. W0-conditioned N 보존 이득은 존재하지만 모든 기존 지식 보존을 뜻하지 않는다.

| Arm | At-write N 성공 /10000 | At-write→W10 N lost/gained | 순감소 pp |
|---|---:|---:|---:|
| N4 | 8379 | 470/117 | −3.53 |
| F48 | 8529 | 295/90 | −2.05 |
| G48 | 8387 | 412/113 | −2.99 |
| C4 | 8382 | 453/117 | −3.36 |
| C48 | 8421 | 349/98 | −2.51 |
| C45678 | 8396 | 383/99 | −2.84 |

C48의 N4 대비 최종 N +1.44pp 중 at-write 격차는 +0.42pp이고 이후 순손실 감소분은 +1.02pp다. C45678은 각각 +0.86/+0.17/+0.69pp다. 이 산술 분해는 서로 다른 trajectory의 관측이며 인과효과 분해가 아니다. At-write는 10개 서로 다른 시점 모델의 관측을 합친 값이다.

전체 first1000은 ACTIVE999/SUPERSEDED1을 포함한다. 원 설계대로 요청1000을 주분모로 유지한 것은 타당하며 overwrite를 조용히 삭제하지 않았다. 하나의 overwrite가 높은 rewrite NLL tail에 미칠 수 있으므로 R 최대 NLL 하나로 전체 품질을 평가해서도 안 된다.

## 작성 시점의 약화와 이후 누적 손상의 구분·긴 horizon 전망

사용자가 요구한 efficacy/generalization/locality 관계를 시간축에서 다시 나누었다. 아래 수치는 추가 학습 없이 원시 `selected-current.json`, `B005/seen-full.json`, `B010/seen-full.json`, `past64.json`을 독립 집계했다.

| Arm | At-write P % | W10 P % | 변화 pp | W5 first500의 P % → W10 같은500 P % | W5 first500의 N % → W10 같은500 N % |
|---|---:|---:|---:|---:|---:|
| N4 | 96.40 | 96.70 | +0.30 | 96.10 → 95.80 | 84.06 → 79.62 (−4.44) |
| F48 | 95.35 | 95.45 | +0.10 | 93.90 → 94.00 | 85.62 → 82.82 (−2.80) |
| G48 | 96.60 | 96.90 | +0.30 | 95.60 → 95.90 | 84.04 → 80.30 (−3.74) |
| C4 | 96.40 | 96.60 | +0.20 | 96.10 → 95.90 | 84.06 → 79.74 (−4.32) |
| C48 | 95.80 | 95.80 | 0.00 | 95.00 → 95.10 | 84.34 → 81.02 (−3.32) |
| C45678 | 96.55 | 96.80 | +0.25 | 95.90 → 96.30 | 84.04 → 80.48 (−3.56) |

**Generalization의 주된 약화는 작성 시점부터 존재한다.** C48은 at-write P가 이미 N4보다 −0.60pp, G48보다 −0.80pp다. 전체 final P 차이 −0.90/−1.10pp를 모두 “나중 batch가 과거 paraphrase를 잊게 했다”로 설명할 수 없다. C48에서 at-write P 95.80%가 W10까지 그대로이고, first500만 보면 P는 오히려 +0.10pp다. C45678은 at-write P 96.55%, W10 96.80%로 올라간다. 선택 순간의 local-z 방향과 gate가 target canonical에 집중하면서 paraphrase를 얼마나 일반화하는지가 주요한 축이다. B10 current P도 N4 97.5%, G48 96.0%, C48 91.5%, C45678 95.5%로 C48의 즉시 약화가 분명하다. 단, B10은 특정100요청이므로 단일 cohort의 난이도를 전체 추세로 일반화하지 않는다.

**Locality의 손상은 작성 시점 차이와 누적 손상이 함께 있으며, 누적 항이 더 크다.** C48/C45678은 N4보다 at-write N이 +0.42/+0.17pp 좋고 최종 +1.44/+0.86pp 좋다. 같은 first500을 W5→W10에서 추적해도 N이 각각 −3.32/−3.56pp 떨어진다. N4의 −4.44pp보다 손실 속도가 작지만 손상 자체가 멈추지 않았다. First500 canonical R은 N4/F48/G48/C48/C45678에서 500/500을 계속 유지하고 C4만500→499다. 이 구간에서는 canonical efficacy 유지와 neighborhood 보존이 분리되어 움직인다.

S64는 모든 arm에서 B1부터 B10까지 매 batch 증가했다. 선택은 자기 ownN4 endpoint보다 KL을 줄이는 문제이며 W0로 되돌리는 문제가 아니다.

| Arm | S64 B1 | S64 B5 | S64 B10 | B5→B10 증가 | Dev128 B5→B10 |
|---|---:|---:|---:|---:|---:|
| N4 | 0.00172091 | 0.01031443 | 0.02128090 | +0.01096647 | 0.01087085 → 0.02265189 |
| G48 | 0.00172091 | 0.00909653 | 0.01911654 | +0.01002002 | 0.00961883 → 0.01933609 |
| C48 | 0.00172091 | 0.00851268 | 0.01600751 | +0.00749482 | 0.00889983 → 0.01758247 |
| C45678 | 0.00171474 | 0.00830352 | 0.01764250 | +0.00933899 | 0.00921919 → 0.01938770 |

C45678은 B5 S64에서 C48보다 낮지만 B6에 순서가 바뀌고 B10에는 더 높다. “5층을 허용하니 더 긴 horizon에서 capacity가 보존될 것”이라는 외삽을 지지하지 않는다. 적어도 현 실행 후반에서는 C48의 KL 증가가 더 느리다. 반대로 C48의 낮은 KL은 P 약화와 함께 나타나므로 이 단일 축만으로 장기 우열을 결정할 수도 없다.

Past64의 실제 보호 범위도 horizon 해석에 중요하다. 모든 arm은 동일 received-event sample을 사용했고 B2 eligible100 중64(64%), B5 400 중64(16%), B10 899 중64(7.119%)를 검사했다. B7에는 현재 overwrite로 eligible가599, B8/B9/B10에는 이미 superseded된 사실이 빠져699/799/899다. B2–B10의 576 sample 슬롯은 **서로 다른 case185개**뿐이며 60개는 한 번, 5개는 아홉 번 모두 관측됐다. B10 Past64의 도착 cohort 분포는 B1..B9 순서로5/10/6/9/4/6/7/9/8개다. 이는 표본 선정의 오류가 아니라 고정 SHA priority라는 사전 설계 결과다.

따라서 더 긴 stream에서도 Past64가64로 유지되면 매 시점 직접 검사하는 과거 비율은 더 낮아진다. History M도 과거 보호에 작동하므로 샘플 외 요청이 전혀 보호되지 않는다고 말할 수는 없다. 다만 H/strict/pair의 명시적 수치 보장은 그때 표본64개에만 적용되고, 기준 자체도 entry가 아닌 ownN4다. 전체 past의 무손실 retention이나 그 누적 상한은 없다.

**제한적으로 예상할 수 있는 것:** 이 설정을 그대로 연장한다면 canonical efficacy가 높은 상태에서도 locality/KL 손상이 누적될 가능성을 우선 점검해야 한다. 이미 모든 arm의 KL이 단조 증가하고 동일 old cohort N이 감소한다. C48의 generalization 손실은 장기 forgetting 방지만으로 해결되지 않을 가능성이 높다. C45678이 남은 용량을 더 확보했다거나 L7/L8이 후반에 유용해질 것이라는 근거는 현재 artifact에 없다. 정확한 B20/B50 성능, 임계 시점, arm 순위 유지, 곡선의 선형성을 이10batch에서 추정하지 않는다.

계산량도 같은 상한을 그대로 적용한다는 조건에서만 확장할 수 있다. Batch100당 필수 N4 fit을 포함한 사전 최대4,100 target request-call/12,000 Adam/16,100 loss/41 solve/29 online score는 horizon에 따라 누적된다. 실제 시간은 early-stop, cache, feasible 후보와 pruning, 관측 범위에 따라 바뀐다. C45678은 이미 10/10batch suffix-fit 탐색 상한에서 멈췄으므로, 더 긴 horizon에서 어려운 feasible region이 생겨도 자동으로 더 깊이 탐색하는 정책은 아니다. 성능 악화 시 같은 예산에서 ownN4 선택이나 불완전 pruning이 늘 가능성은 있으나 증가를 보장하거나 현재 데이터로 확률을 계산할 수 없다.

## 실측 비용과 다층 비교의 의미

| Arm | Target calls | Adam updates | Loss evals | Solve | Online score | Program wall 초 | N4 대비 wall | N4 대비 controller 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N4 | 1000 | 24000 | 25000 | 10 | 10 | 4534.98 | 1.000× | 1.000× |
| F48 | 2000 | 27912 | 29912 | 20 | 20 | 5554.16 | 1.225× | 1.279× |
| G48 | 3000 | 28152 | 31152 | 30 | 60 | 6815.74 | 1.503× | 1.609× |
| C4 | 1000 | 24000 | 25000 | 10 | 40 | 5274.59 | 1.163× | 1.168× |
| C48 | 15100 | 65759 | 80859 | 151 | 174 | 16701.54 | 3.683× | 4.593× |
| C45678 | 33100 | 45791 | 78891 | 331 | 142 | 19711.48 | 4.347× | 5.467× |

Target call은 request×prefix 반복을 포함하고 unique edit 요청 수가 아니다. C45678은 C48보다 target calls가 2.19배이나 Adam은 0.696배, loss eval은 0.976배다. 더 많은 zero-Adam fit, key 추출·solve·상태/후보 관리가 있기 때문에 target 수 또는 Adam 한 가지만 runtime 대용으로 쓰기 어렵다. Program wall은 관측/history 등 실험 overhead를 포함하고 controller 시간도 순수 writer FLOP는 아니다. Native inclusive와 세부 target/key/solve timer는 중첩이므로 합산하지 않았다.

C45678은 G48 대비 program wall 약2.89배이며 R 동일, P −0.1pp, N +0.24pp, Dev128은 0.01938770 대0.01933609로 조금 높다. 이 조합에서 다층 연속 탐색이 실용적으로 우월하다고 판정할 근거는 약하다. 반면 C45678−C48은 P +1.0pp와 N −0.58pp의 교환이므로 “층을 더 허용하니 모두 좋아졌다”는 결과도 아니다.

F48이 locality와 Dev에서 가장 좋지만 PS와 strict 손실도 가장 크고 10개 중7개 selected endpoint가 guard를 실패한 것은 승인된 fixed baseline 예외다. 다른 arm의 기술 실패와 합치지 않아야 한다.

## Layer 기여 해석에 연결할 주의점

1. C45678 B5의 support3 선택과 S64 개선은 확인되지만 같은 선택에서 P/N 손실이 있다. 품질에 도움을 주는 세 층이라는 결론에는 지표별 조건을 붙여야 한다.
2. Parent의 pruning ledger 검산에 따르면 C45678 B5의 L6 제거는 실제 완료됐고 E=0.01553756이 ownN4 E=0.01467522+0.0001을 넘었다. 따라서 **그 a4/a5 조합에서는 L6가 E를 유지하는 데 필요했다는 국소 증거**가 있다. 미완료된 검사는 L5 제거다. 3층 전체의 필수성이나 다른 gate의 2층 후보로는 불가능하다는 주장은 입증되지 않았다.
3. C4가 거의 N4를 선택한 결과는 강도조절의 원리적 무효를 뜻하지 않는다. Parent의 별도 후보 검산에서 동일 W0 진입 B1의 C45678 pruning이 C4 search가 놓친 feasible L4-only a4=0.9982652821을 찾았다. 따라서 arm간 차이에는 탐색 경로 효과도 있다.
4. 각 arm은 B2 이후 다른 chain이므로 최종 C45678−C48로 L5/L6와 L8의 순수 인과 기여를 분리할 수 없다. B1/B5/B10 observer는 같은-entry endpoint 비교에 유용하지만 가상 lifelong chain을 제공하지 않는다.

## 판정 범위

이 분석은 raw artifact 내부 산술과 평가 정의를 독립 확인했다. 저장 W/M 전체가 없어 exact final model 복원·새 forward parity를 수행한 것은 아니다. 보고서의 핵심 성능표에서 수치 오기는 발견하지 못했다. 결과를 가장 정확히 표현하면 **특정 개발 stream에서 S64와 neighborhood 보존 이득을 찾았지만, 특히 P strict/NLL 및 비용을 고려하면 품질 무손실·연속성·다층 기여의 강한 주장은 성립하지 않는다**는 것이다.
