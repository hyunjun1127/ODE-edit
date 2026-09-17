# Cold local-z 7-arm: 독립 산출물 점검과 연구 해석

2026-09-17 · global-head 독립 검토. 대상 정본 commit `c6b9937a29e663da305464cd7a04448b24ea6361`; 실행 source `32a92ad6f3fff2f258d8778f3936d152e975ac1b`.

대상: [SH4 상세 사실 리뷰](../servers/server4/local-z-adaptive-allocation-seq1000-2026-09-16-v1/completed-review-20260917-v1/diagnostic-report-ko.md). 원 보고서·실험 산출물은 수정하지 않았다. 이번에 수행한 것은 source/CSV/manifest/figure 검토와 원격 저장 JSON·selected delta의 CPU 계산이다. 새 model forward, teacher 생성, GPU 실험, scheduler 변경은 없다.

## 1. 판단

집계·선택·산출물 연결에서 주요 불일치는 발견하지 못했다. 이 결과는 **local-z 기반 다층 정책에 locality 개선 여지가 있지만, 현재 LD에는 paraphrase 비용이 있고 adaptive 선택 자체의 추가 이득은 아직 분리되지 않았다**는 근거다.

따라서 “L4-only가 locality에서 항상 최선”, “다층이면 항상 더 많은 손상이 생긴다”, “dynamic allocation이 static allocation보다 우수하다”, “layer capacity를 측정했다” 중 어느 것도 이번 실험의 결론으로 확정할 수 없다.

실행은 새 W0, zero M4/M8에서 시작한 7개의 100×10 chain이다. 동일 first1000을 반복 사용했으므로 unique request는1000, arm-request 관측은7000이다. 한 order·한 method seed이며 과거 5000-edit 상태를 재사용한 비교가 아니다. 공통 출발점을 맞춘 것은 warm-state 교란을 제거한 것이며 모든 통계·수치 불확실성을 제거한 것은 아니다.

## 2. 실제 비교한 정책

| Arm | target/write 및 강도 | 역할 |
|---|---|---|
| N4 | L4 local-z, full native write | 강한 단층 기준 |
| REFIT4 | L4 local-z .75 write 후 같은 L4에서 fresh target/fit | 추가 최적화를 같은 층에 쓰는 대조 |
| L75 | L4 local-z .75 후 해당 partial state에서 L8 local-z full write | 고정 local (.75,1) |
| T75 | entry에서 구한 Z8 고정; L4 .75, L8 full residual write | 고정 terminal (.75,1) |
| L4D | own-entry N4와 local L4 .75 중 선택 | L4 강도만 적응 |
| LD | a4∈{.75,1}, a8∈{0,.5,1}; local-z 후보6개 | 두 층의 강도 및 L8 사용 여부 선택 |
| TD | terminal 후보6개와 own-entry local N4 중 선택 | terminal 제안을 포함한 선택기 |

LD는 request별 학습 router나 softmax(Wz)가 아니다. 실제 후보 endpoint를 batch마다 만들고 E/H/strict 조건을 통과한 후보 중 S64 KL을 최소화하는 이산 선택기다. Layer5–7 및 L8-only routing은 이 실험에서 조사하지 않았다.

LOCAL은 L4를 부분 적용한 뒤 그 상태의 L8에서 z를 fresh 계산한다. TERMINAL은 entry Z8을 고정하며 L4에서도 residual은 Z8−h8(entry), L8에서는 Z8−h8(partial)이다. Z8−h4로 다른 층의 좌표를 단순 차감하지 않는다. 원 native solve를 사용하며 gate는 actual FP32 endpoint 차이에 적용한다.

TD는 순수 terminal 정책이 아니다. 또 T75는 원래 5-layer AlphaEdit/MEMIT 또는 CAKE의 그대로인 재현 arm이 아니다. 이 결과를 해당 선행 방법 전체의 성능으로 일반화하지 않는다.

## 3. 최종 성능과 실제 선택

RS/PS는 target-new NLL이 target-true보다 작은 비율, NS는 반대다. 이는 free-generation 정답률이나 teacher-forced 전토큰 argmax strict와 다르다.

| Arm | RS /1000 (%) | PS /2000 (%) | NS /10000 (%) | PS−N4 pp | NS−N4 pp |
|---|---:|---:|---:|---:|---:|
| N4 | 999 (99.90) | 1934 (96.70) | 8026 (80.26) | 0 | 0 |
| REFIT4 | 999 (99.90) | 1929 (96.45) | 8187 (81.87) | −0.25 | +1.61 |
| L75 | 1000 (100.00) | 1919 (95.95) | 8227 (82.27) | −0.75 | +2.01 |
| T75 | 997 (99.70) | 1942 (97.10) | 7161 (71.61) | +0.40 | −8.65 |
| L4D | 999 (99.90) | 1922 (96.10) | 8063 (80.63) | −0.60 | +0.37 |
| LD | 999 (99.90) | 1909 (95.45) | 8310 (83.10) | −1.25 | +2.84 |
| TD | 999 (99.90) | 1934 (96.70) | 8026 (80.26) | 0 | 0 |

LD−N4의 NS는 lost120/gained404, PS는 lost48/gained23이다. RS 총999는 같지만 lost1/gained1로 문항이 다르다. 최종 상대점수만으로 완전한 보존을 선언하지 않는다.

- L4D: B8에만 .75 L4, 나머지9회 N4.
- LD: B4에만 N4, 나머지9회 (.75,.5).
- TD: 10회 모두 N4. 전체 batch W4/W8/M4와 최종 raw NLL이 N4와 같다. M8은 TD에서만 계속 append되어 다르다.

LD−L4D는 NS+2.47pp, PS−.65pp다. 이는 검사한 두 정책 사이에서 추가 L8를 허용하는 정책의 locality 이득을 보여주지만 같은-state functional L8 기여량은 아니다.

### Dynamic 효과를 분리하지 못한 핵심 대조군

**고정 (.75,.5) chain이 없다.** LD의 9회 선택과 같은 강도를 쓰는 static 대조 대신 L75=(.75,1)이 있다. LD−L75에는 L8 강도 감소와 상황별 선택의 효과가 함께 들어 있다. B4 fallback의 장기적 가치도 분리되지 않았다. 따라서 현재는 “좋은 강도 조합을 찾아 사용했다”는 관측이 있고, “적응성이 있어야 그 이득이 생긴다”는 검증은 없다.

보고서의 no-plateau/strict-only shadow는 실제 LD trajectory의 저장 점수에서 다시 고른 것이다. 그 다른 정책을 cold에서10batch 실행한 결과로 읽으면 안 된다.

## 4. Neighborhood 손상이 발생한 시점

동일 N10000을 공통 W0, 각 요청의 도착 batch entry, 그 batch의 at-write, 최종 W10에서 비교했다. Entry/at-write는 여러 모델 시점의 pooled 관측이며 하나의 모델 상태가 아니다.

| 시점 | N4 성공수 | LD 성공수 | LD−N4 |
|---|---:|---:|---:|
| 공통 W0 | 8820 | 8820 | 0 |
| 해당 batch entry | 8606 | 8686 | +80 |
| 해당 batch at-write | 8379 | 8496 | +117 |
| W10 | 8026 | 8310 | +284 |

따라서 최종 +284는 도착 전 누적 순손상 차이80 + 해당 batch 순손상 차이37 + 이후 batch 순손상 차이167로 산술 분해된다. 인과 기여율이 아니다.

At-write에서 성공했던 N의 이후 loss는 N4 470/8379, LD292/8496이다. 회복도 각각117,106개 있어서 순손실은353,186개다. LD에서도 forgetting은 남아 있다. W0에서 N이 모두 정답이었던 것도 아니므로 최종 NS를 곧바로 원래 지식의 보존율이라고 부르지 않는다. W0 성공 문항에 조건부로 계산한 추가 표는 아래 CPU 보완에 둔다.

PS는 N4가 at-write1928→final1934, LD1910→1909다. 최종25개 손실 차이 중18개는 이미 쓰기 직후 격차,7개는 그 뒤의 순변화 차이다. LD의 PS 문제를 주로 긴 horizon의 forgetting으로 설명하는 것은 현재 숫자에 맞지 않는다.

## 5. 선택기가 보호한 것과 보호하지 못한 것

E는 native rewrite training context의 target-new NLL 평균이다. 조건은 `E(c) ≤ max(E(ownN4), .05)+1e-4`다. 여기에 ownN4의 canonical rewrite strict 성공 ID 포함조건을 붙였다. Past64도 mean H와 strict 성공 ID를 ownN4와 비교한다.

이 조건은 다음을 보장하지 않는다.

- 각 요청의 NLL 또는 confidence 무악화.
- canonical rewrite의 두-target 선호 RS 자체의 전 문항 보존.
- paraphrase PS 또는 Neighborhood NS 보존.
- batch entry 대비 Past 무손실 또는 모든 과거 요청 보존.

LD B1에서 ownN4 E≈.0034046, selected E≈.0124145지만 ceiling .0501 아래여서 허용됐다. 원 후보 CSV에서 LD selected canonical rewrite NLL은 pooled788/1000요청에서 ownN4보다 높았다. strict ID를 지키는 것과 likelihood를 지키는 것은 다르다. 다만 이 관측만으로 PS 손실의 원인을 plateau 하나로 확정할 수는 없다.

H도 ownN4 대비 조건이므로 ownN4가 이미 잃은 과거 요청을 원천적으로 방지하는 보장은 없다. 또한 B2–10에서 확인한 Past는 각64개이며 전체 과거 요청이 아니다.

S64는 선택용 C4 패널이며 Dev128은 독립 observer다. LD의 W10 S64 KL은 N4보다36.37%, Dev128은38.80% 낮고 official NS도 높았다. 선택 패널 밖에서 같은 방향의 관측은 있지만, candidate별 official P/N이 없어 같은 상태의 최선 S64 후보가 NS/PS에도 최선인지 알 수 없다.

지표 순위도 일률적이지 않다. LD는 L75보다 NS가 .83pp 높지만 N true strict는1918 대1927로 낮다. 두-target 선호, target-true 평균 NLL, top-1 strict, C4 분포 보존을 같은 지표처럼 사용하지 않는다.

## 6. 산출물 감사 범위

| 확인 | 결과 | 해석 경계 |
|---|---|---|
| 공개 report/source/manifest | 41개 산출물·11개 분석 source의 명시 SHA/size 일치 | 공개 요약의 연결 확인 |
| CSV 산술·교차표 | 독립4,636개 조건 불일치0 | 원 strict ID는 별도 raw 검사 필요 |
| 원격 JSON 재집계 | final·entry·at-write NLL,190후보,30동적선택,63인접 state 연결 직접 검사 | 저장 forward 결과의 산술 검산 |
| 실행 source | core 및 imported fitter/evaluator가 frozen source와 일치 | 신규 forward·solve 재현 아님 |
| 기존 tensor review | 21CP/84tensor, B2–10 후보171개 materialization hash 일치 기록 | 이번 독립 감사에서 전체118.6GB를 다시 hash하지 않음 |
| B1 모든 후보 | 독립 W0 tensor 기반 materialization 미검증 | B1 selected CP 검증과 구분 |
| history | 110append source·receipt·hash 연결 | finalizer K 미저장으로 Gram 재연산 불가 |
| 원모델 비편집 parameter | runtime guard 및 자산 identity | 전체 backbone 독립 byte 감사 아님 |
| 그림4개 | CSV 값·축·label·패널 범위 확인 | current curves 각 점의 cohort가 다름 |

입력 manifest의2136항목은 CURRENT_FULL_SHA179개와 과거 fullSHA+현재 stat 재사용1957개로 구성된다. 새 full hash가 모든 입력에 대해 수행됐다고 읽으면 안 된다.

Terminal 기술 연결검사는 원 non-blue 경로와 차이를 기록했지만 허용폭 gate를 assert하지 않는다. 실제 max-abs는 L4 `2.9802322387695312e-08`, L8 `5.8710575103759766e-06`; L2는 각각 `2.986646085770043e-06`, `.00012781928757292288`이다. 구현 오류를 입증하는 관측은 아니지만 “terminal bit-exact parity PASS”도 아니다. 보고서는 이를 source/update connection으로 구분한다.

추가 budget 진단 문서의 candidate 전체 P/N observer, DiagKeys128, key drift는 frozen runtime에 없다. 이것을 수행한 것으로 간주하지 않는다. 저장된 endpoint/delta/candidate 점수에서 CPU로 계산할 수 있는 부분만 아래에 보완했다.

## 7. 비용과 운영 이탈

| Arm | target calls | actual Adam | solve | 후보수 | GPU allocation h |
|---|---:|---:|---:|---:|---:|
| N4 | 1000 | 24000 | 10 | 10 | 1.668 |
| REFIT4 | 2000 | 27864 | 20 | 10 | 2.050 |
| L75 | 2000 | 27756 | 20 | 10 | 1.875 |
| T75 | 1000 | 23459 | 20 | 10 | 1.971 |
| L4D | 1000 | 24000 | 10 | 20 | 1.859 |
| LD | 3000 | 28584 | 30 | 60 | 4.226 |
| TD | 2000 | 48000 | 40 | 70 | 5.139 |

LD는 N4보다 target calls가3배지만 실제 Adam은1.191배다. 추가 local target의 많은 zero-step/early-stop 때문에 단순 target call 배수와 optimizer 비용 배수가 다르다. 반면 LD candidate scoring 포함시간은8419.78초로 전체 program15210.3초의 약55.4%다. 이 시간에는 restore/hash/I/O도 포함된다. 아직 이 선택기는 저비용 학습 router가 아니다.

준비 포함 allocation합19.5922GPUh. cap1 제출 기록과 달리 실제 최대2GPU, 초과19428초가 기록되어 있다. 원인·행위자·변경시점은 미확인이다. 이 관측은 성능 결과를 자동 무효화하지 않지만 통제된 wall-time 속도 비교를 허용하지 않는다. Source-visible target/solve/후보수와 실제 시간을 구분한다.

## 8. 연구 방향에서 다음에 분리할 것

1. **Local-z를 다음 비교의 주 경로로 유지할 근거는 있다.** 이 실험에서 matched-gate T75−L75의 NS차이는−10.66pp이며 TD terminal제안은 최종 선택되지 않았다. 다만 보편적인 terminal-z 열등성이나 PS까지 포함한 우월성을 증명한 것은 아니다.
2. **최우선 대조는 cold static(.75,.5) 대 LD다.** 동일한 horizon·order·평가에서 강도 조합과 적응성의 이득을 분리해야 한다. 기존 LD의 shadow 점수로 대체할 수 없다.
3. **PS 비용을 분리해 다룬다.** 평가용 official P를 controller로 가져오는 대신 별도 rewrite/paraphrase calibration split 또는 train-derived guard를 설계하고 official P는 observer로 유지해야 한다. 어떤 guard가 비용을 줄이는지는 새 실험 문제다.
4. **후보 공간과 선택 proxy를 분리한다.** 향후 사전 지정한 batch에서 모든 후보의 official P/N을 선택 봉인 후 observer로 기록하면, 좋은 후보가 없었던 것인지 S64가 고르지 못한 것인지 구분할 수 있다.
5. **Budget은 이동량·기능 변화·미래 edit 여유를 구분한다.** Norm 및 M append만으로 layer의 남은 capacity를 숫자로 선언하지 않는다. 장기 capacity는 더 긴 cold chain과 사전 고정한 품질/보존 허용폭에서의 실패 시점으로 검증할 문제다.

위 항목은 이번 리뷰의 연구 해석이며 새 실험을 제출하거나 정책을 변경한 기록이 아니다.

## 9. 저장 산출물에서 추가한 CPU 진단

독립 계산 파일: [집계 JSON](../../audits/global/2026-09-17-local-z-independent-review/independent-reduction.json), [CPU 계산기](../../audits/global/2026-09-17-local-z-independent-review/inspect_saved.py). 원격 raw는 server4의 기존 `local/local-z-adaptive-allocation/20260916-v1/`에서 읽기만 했다.

### 9.1 W0에서 성공한 Neighborhood의 조건부 유지

모든 arm에서 W0 NS 성공은8820/10000이다. 다음 유지율의 분모는8820이며 최종 NS의 분모10000과 다르다. 의미는 두-target NLL 선호의 유지이며 자유 생성 지식 정확도는 아니다.

| Arm | W0 성공→최종 성공 | W0 성공→최종 실패 | W0 실패→최종 성공 | W0 성공 조건부 유지율 |
|---|---:|---:|---:|---:|
| N4 | 7852 | 968 | 174 | 89.02% |
| REFIT4 | 8011 | 809 | 176 | 90.83% |
| L75 | 8028 | 792 | 199 | 91.02% |
| T75 | 6932 | 1888 | 229 | 78.59% |
| L4D | 7888 | 932 | 175 | 89.43% |
| LD | 8145 | 675 | 165 | 92.35% |
| TD | 7852 | 968 | 174 | 89.02% |

LD−N4의 최종 NS+284는 W0 성공 문항의 추가 유지293개와 W0 실패 문항의 추가 회복−9개로 나뉜다. 이 표는 locality 이득이 원래 맞혔던 문항의 보존에도 나타났음을 보여준다.

### 9.2 저장된 selected delta의 layer별 누적 이동량

70개의 selected-delta.pt를 CPU에서 읽었다. δ는 저장된 FP32 endpoint 차분이다. `path`는 Σt||δ_l,t||F, `net`은 ||Σtδ_l,t||F이며 합산은 FP64로 했다. `energy`는 Σt||δ_l,t||F²다. 기록된 차분으로부터 계산한 것이며 FP32 차분 합이 실제 Wt−W0 tensor와 bit-exact하다고 주장하지 않는다. 이번 재검산에서는 delta size와 기존 SHA receipt를 연결했고33GB가량의 delta payload SHA를 새로 모두 계산하지 않았다.

| Arm | L4 path | L8 path | L4 net | L8 net | L4 energy | L8 energy |
|---|---:|---:|---:|---:|---:|---:|
| N4 | 80.5772 | 0 | 25.5264 | 0 | 650.4317 | 0 |
| REFIT4 | 73.7686 | 0 | 23.3720 | 0 | 545.7485 | 0 |
| L75 | 59.3145 | 51.9147 | 18.7827 | 16.6445 | 352.2196 | 276.7756 |
| T75 | 134.3084 | 170.2589 | 42.6449 | 53.8817 | 1815.1932 | 2908.3815 |
| L4D | 78.5017 | 0 | 24.9352 | 0 | 620.7591 | 0 |
| LD | 61.3670 | 24.3912 | 19.5041 | 8.2447 | 379.7469 | 67.9627 |
| TD | 80.5772 | 0 | 25.5264 | 0 | 650.4317 | 0 |

LD는 N4 대비 L4 path가23.84%, L4 energy가41.62% 작다. 그러나 gate 차이뿐 아니라 각 branch의 entry/target이 다르므로 이를 고정 총 edit 부담의 물리적 이전율로 해석하지 않는다.

Layer path끼리 단순 합한 값과 concat path는 다르다. Σt sqrt(||δ4,t||²+||δ8,t||²)는 N4 80.5772, REFIT4 73.7686, L75 79.0961, T75 216.8719, L4D 78.5017, LD 66.7938, TD 80.5772다. LD는 두 층을 사용해도 이 전체 parameter 이동량이 N4보다 작다. “접촉한 층 수가 많으면 전체 변화량도 반드시 크다”는 가정은 이번 저장 update와 맞지 않는다.

모든 arm의 M4 hash는 각 batch마다 동일하다. TD는 L8 update가0인10batch에도 M8가 계속 append된다. 따라서 더 작은 gate/norm을 사용했다고 history 증가량이나 capacity 소비가 그 비율만큼 줄었다고 말할 수 없다.

### 9.3 같은 상태의 S64 headroom 및 L4→L8 순서별 변화

LD 후보 중 quality-feasible L4-only 최솟값을 D4*, 전체 최솟값을 D48*라 하면 H_strength=D(ownN4)−D4*, H_extra8=D4*−D48*다. 검사한 이산 후보·제약·현재 state에 대한 값이다.

LD의10batch에서 H_strength는 전부0이다. H_extra8은 B4에서0, 나머지9회 양수이며 합은 .0050159984다. 그 합은10개의 서로 다른 state에서 얻은 비교값의 합이며, 실제 final N4−LD S64 차이 .00774006과 같은 누적 인과효과가 아니다. 후보공간 포함관계상 비음수라는 성질만으로 방법의 우월성이 증명되는 것도 아니다.

각 선택 경로의 b4=D(partial4)−D(entry), b8=D(selected)−D(partial4)를 저장 점수로 계산했다. B1의 원 teacher-effective-check에 실제 D=0이 저장되어 있음을 별도 확인했다. Σb4=.0118965434, Σb8=.0016443436으로 합 .0135408871은 LD 최종 S64 KL과 일치한다. B2 이후 entry D는 직전 selected D를 재사용했으며 별도 entry forward를 추가한 값은 아니다.

**L8를 쓴9회 모두 b8>0이다.** 즉 그 순서에서 L8 write가 partial-L4 대비 C4 KL을 감소시킨 것은 아니다. 대신 .75 L4만 쓰면 quality 조건을 만족하지 못하는 상태에서 L8가 edit 품질을 보완했고, 조합 전체는 ownN4보다 낮은 KL을 얻었다.

B1 예시는 특히 분명하다.

| 후보 | E | S64 KL | 품질 조건 |
|---|---:|---:|---|
| .75 L4만 | .182800 | .000951336 | 실패 |
| .75 L4 + .5 L8 | .0124145 | .001252665 | 통과·선택 |
| full L4 N4 | .00340461 | .001720908 | 통과 |

따라서 “L8가 기존 locality 손상을 직접 복구했다”보다 **“L4 write를 줄였을 때 부족해진 edit를 L8로 보완하여, quality-feasible 조합의 보존 비용을 낮췄다”**가 이 same-state 관측에 더 잘 맞는다. 이 해석은 선택된 이산 후보와 S64에서의 관측 범위이며 official N의 layer별 기여를 측정한 것은 아니다.

Parameter path/net/energy, 순서 조건부 S64 변화, history Gram은 서로 다른 진단이다. 이 중 어느 하나도 남은 edit 개수나 intrinsic layer capacity의 절대 budget은 아니다.
