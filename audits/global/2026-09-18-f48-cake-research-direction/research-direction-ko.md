# F48의 locality 이득, CAKE 분배의 의미, 다음 실험의 우선순위

2026-09-18. 완료된 sequential-local-z v2의 코드·원자료 독립 검산을 바탕으로 과거 실험, CAKE 원논문·공식 구현·실행 산출물, 최신 correction 설계를 함께 확인했다. 이번 작업은 분석이다. 신규 GPU 실행·모델 forward·실험 정책 변경은 없다.

## 1. 연구 판단

**F48은 이번 6개 정책 중 가장 큰 locality 이득을 N4 대비 약22.5%의 추가 시간으로 얻은 강한 기준이다. 핵심 신호는 “고정 배분이 적응 배분보다 우월하다”가 아니라, full native L4보다 약한 write와 조건부 보완을 조합하면 훨씬 작은 보존 손상으로 상당한 편집 성능을 유지할 수 있다는 것이다.** 다만 paraphrase 일반화 손실은 분명히 있다. 다음 목표는 F48의 NS/KL 이득을 유지하면서 그 손실을 줄이는 것이어야 한다.

CAKE에 대한 static 비판은 **causal prior/weight가 고정**이라는 범위에서 타당하다. 논문도 초기 score 유지가 비용 절충임을 명시한다. 그러나 CAKE는 매층 현재 residual/key를 다시 계산한다. 우리 F48 역시 gate는 고정이지만 L8 target과 실제 write는 partial-L4 상태에 의존한다. 따라서 연구 기여를 단순히 static→dynamic으로 잡으면 양쪽 방법의 실제 구조와 현재 증거를 놓친다.

더 직접적인 질문은 **“어느 층이 기존 사실을 강하게 복원하는가”가 “현재 상태에서 같은 편집 품질을 가장 작은 보존 비용으로 달성하는 방향은 무엇인가”를 얼마나 잘 예측하는가**다. 총 강도, target 재계산, 보완 위치, 적응 선택, 평가 proxy를 분리해서 답해야 한다.

## 2. F48의 이득과 실제로 지불한 대가

동일 v2 capsule, W0에서 시작한 B100×10의 각자 독립 chain이다. RS/PS/NS는 각각 rewrite/paraphrase/neighborhood의 원하는 target 대 competing target **NLL 선호 성공률**이다. Strict 정답률과 동일하지 않다. Dev128은 선택에 쓰지 않은 W0 full-vocabulary KL observer다. 비용은 이 실험 runner의 전체 program 시간이며 순수 writer 시간은 아니다.

| 정책 | RS % | PS % | NS % | P strict % | Dev128 KL ↓ | 시간/N4 |
|---|---:|---:|---:|---:|---:|---:|
| N4 | 99.90 | 96.70 | 80.26 | 67.70 | .022652 | 1.000 |
| F48 | 100.00 | 95.45 | 83.24 | 64.00 | .013647 | 1.225 |
| G48 | 100.00 | 96.90 | 80.88 | 66.85 | .019336 | 1.503 |
| C4 | 99.80 | 96.60 | 80.46 | 67.45 | .022559 | 1.163 |
| C48 | 99.90 | 95.80 | 81.70 | 65.65 | .017582 | 3.683 |
| C45678 | 100.00 | 96.80 | 81.12 | 65.90 | .019388 | 4.347 |

F48−N4는 NS **+2.98pp**, PS **−1.25pp**, P strict **−3.70pp**다. 두 paraphrase가 모두 strict 성공인 요청은 513→457/1000으로 **−5.60pp**다. PS의 paired case-bootstrap 95% 구간은 [−2.20,−.30]pp, NS는 [+2.43,+3.52]pp다. 이는 이 하나의 완료 chain에서 요청을 재표집한 불확실성이며 seed/order 간 일반화 구간은 아니다.

NS 분모10000에서 N4 성공을 잃은 문항126개, 새로 성공한 문항424개로 net+298이다. PS 분모2000에서는 lost50/gained25로 net−25다. 보존 이득이 모든 문항의 무손상을 뜻하지 않는다.

F48의 S64 KL은 N4 대비 **41.27%**, Dev128 KL은 **39.75%** 낮다. W0에서 맞혔던 neighborhood만 보면 NS는89.02→92.61%로 +3.58pp다. 따라서 단일 S64 proxy의 우연한 감소만으로 보기는 어렵다. 동시에 F48의 P target NLL은 평균+.20048, p99 변화+4.63260이고, 1333/2000 문항에서 악화됐다. Rewrite 성공률의 포화가 일반화 품질까지 보장하지 않는다는 증거다.

“PS 1pp 손실당 NS gain”은 설명용 비율일 뿐 보편적인 효용이 아니다. 원래 목표가 PS 무손실이라면 F48은 그 목표의 해가 아니다. 현재 관측 frontier의 유용한 tradeoff 점이며 전역 Pareto 최적성도 검증하지 않았다.

![실제 6개 chain의 frontier와 동일 첫 배치의 단계별 반응](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-f48-cake-research-direction/f48-frontier-and-mechanism.png)

왼쪽은 전체 1k chain, 오른쪽은 같은 W0에서 평가한 첫100 요청의 실제 후보다. 오른쪽 녹색 선은 N4 native E+1e−4 경계다. 축과 실험 단위가 다르므로 두 panel의 점을 직접 대응시키지 않는다. [재집계 표](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-f48-cake-research-direction/v2-final-frontier.csv), [동일-entry 후보표](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-f48-cake-research-direction/same-entry-b1-endpoints.csv).

## 3. Layer allocation에서 실제로 식별된 기전

### 3.1 .75와 .5는 정규화된 예산 비율이 아니다

현재 F48은 다음 순서다. D4는 entry에서 얻은 native L4 weight 제안이고 D8은 적용된 prefix에서 새로 만든 native L8 제안이다.

\[
W'_4=W_4+.75D_4(W,M,Q),\qquad
W'_8=W_8+.5D_8(W'_4,M,Q).
\]

Gate 합은1.25지만 그것을 총 edit budget1.25라고 해석할 수 없다. 다른 층의 다른 target/key/solve가 만든 방향에 서로 다른 배율을 곱한 것이다. CAKE의 sum-to-one residual weight와도 의미가 다르다. F48은 **계수는 고정, 방향은 상태 의존**인 정책이다.

전체10batch 실제 concat step norm의 합은 N4 80.5772, F48 64.8908로 약19.47% 작다. Step norm 제곱합은 약35.20% 작다. 이 수치는 변경 강도 감소와 보존 개선의 연관성을 뒷받침하지만, 최종 net weight displacement·층별 norm·Fisher energy·remaining capacity 측정은 아니다. 서로 다른 chain의 집계라 norm 감소만으로 인과 설명도 끝나지 않는다.

### 3.2 L8은 약한 L4 이후 편집을 보완한다

G48 B1의 동일 W0/동일 요청에서 실제 완료된 후보를 재집계했다.

| 후보 | Native E ↓ | S64 KL ↓ | PS % | P strict % | NS % |
|---|---:|---:|---:|---:|---:|
| L4 1, L8 0 | .003405 | .0017209 | 97.0 | 62.5 | 86.5 |
| L4 .75, L8 0 | .182800 | .0009513 | 91.0 | 49.5 | 87.6 |
| L4 .75, L8 .5 | .012414 | .0012527 | 95.0 | 60.5 | 86.9 |
| L4 .75, L8 1 | .010281 | .0020327 | 94.5 | 63.5 | 86.3 |

L4를 .75로 줄이면 KL은44.7% 줄지만 PS가6pp 내려간다. 거기에 L8 .5를 추가하면 PS4pp를 회복하고 native E도 크게 낮아진다. 대신 partial-L4보다 KL은31.7% 증가한다. 그 결과가 full-L4보다 KL27.2% 낮은 F48 점이다. 과거 cold7에서도 L8을 선택한9batch 전부에서 partial-L4→L8의 S64 KL이 증가했다.

따라서 이 L8은 **그 직전 상태의 generic damage를 지우는 repair가 아니라, 약해진 편집을 비교적 작은 추가 손상으로 보완하는 방향**이다. Full-L4 뒤 preservation gradient로 KL을 낮추는 repair와 목적·대조군을 분리해야 한다.

L8 gate를 .5→1로 높이면 이 B1에서는 KL이 더 커지고 PS preference가95→94.5로 내려가지만 P strict는60.5→63.5로 올라간다. “강할수록 gen이 좋다/나쁘다”라는 단조 관계조차 지표에 따라 달라진다. Full-L4 뒤 native L8은 여기서 사실상 수치적 미소 write였으므로, 다른 층을 덧붙이는 것만으로 F48 기전이 생기지 않는다.

### 3.3 Adaptive arm이 F48을 못 이긴 이유를 분리해야 한다

F48은 v2 guard 예외로 고정 commit한다. 같은 entry의 own-N4 대비 native E guard를10batch 중7회 위반하고, 그중2회는 Past H도 위반한다. 따라서 constrained C48/C45678이 F48의 locality 점을 선택하지 못한 일부 이유는 **허용 품질 영역의 차이**다. 그 결과로 곧바로 static allocator가 우월하다고 할 수 없다.

그렇다고 native E를 느슨하게 만들기만 하면 gen이 해결되는 것도 아니다. F48 B5는 native E를 통과해도 same-entry PS가 own-N4보다4pp 낮다. C48 B5도 E/KL을 개선하면서 canonical 요청91/100의 NLL이 악화되고 PS/NS가 하락했다. 기존 guard는 native-context 평균과 canonical 성공 ID를 지키며, 공식 paraphrase 일반화를 직접 제한하지 않는다.

C45678은 최종 L4를10회, L5를6회, L6를2회 사용하고 L7/L8은0회다. 이는 **앞층까지 적용한 prefix에서** 추가 native 편집이 거의 필요 없었다는 관측이다. L8을 독립적으로 쓸 가치가 없다는 결론은 아니다. F48의 L8과 C45678의 L8은 진입 상태가 다르다.

검색도 완전한 optimum oracle이 아니다. C45678은 매batch suffix fit cap에 도달했고, 완료 search 점105개는 모두5층 support였으며 최종 sparsity는 pruning에서 나왔다. L7/L8이 전체321개 추가fit 중175개를 차지했다. L8 target9300개는 전부 zero-Adam이었지만 forward/key/solve와 상태 작업은 계속 발생했다. C4도 B1에서 feasible한 .998265 근처 점을 찾지 못했다. “더 넓게 탐색했으니 가능한 배분의 상한을 확인했다”고 말할 수 없다.

## 4. CAKE의 static·heuristic 비판을 어디까지 할 수 있는가

[CAKE 원논문](https://aclanthology.org/2026.acl-long.918.pdf#page=20)은 초기 attribution profile을 유지한다고 명시한다. Softmax는 주어진 score에 entropy 목적을 더한 문제의 해이고, residual allocation은 주어진 weight와 선형 sensitivity 아래의 quadratic 문제다. 실제 구현은 sensitivity를 근사하고 매층 residual을 갱신한다. 이 조건부 최적성이 실제 PS/NS/KL의 최적 배분을 의미하지는 않는다. 자세한 수식·가정·코드 위치는 [방법 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-f48-cake-research-direction/cake-method-audit-ko.md)에 있다.

실행된 official CAKE config에서 계산한 값은 다음과 같다. 마지막 열은 weight가 아니라 현재 남은 residual에 실제 곱하는 비율이다.

| 층 | score | softmax weight w | remaining 비율 w/sum(remaining w) |
|---|---:|---:|---:|
| L4 | .481244 | .243951 | .243951 |
| L5 | .474382 | .227773 | .301268 |
| L6 | .465637 | .208701 | .395060 |
| L7 | .444066 | .168206 | .526343 |
| L8 | .433519 | .151369 | 1.000000 |

L8의 score/weight가 가장 작아도 마지막 남은 residual은 전량 실현하려고 한다. 실제 CAKE 10k의 층별 step norm 합은 L8 비중32.76%로 가장 크다. “CAKE가 L4에24%, L8에15%의 실제 parameter 변경을 쓴다”는 해석은 틀린다. [공식 apply 코드](https://github.com/zjh-vinky/CAKE/blob/c8243e1d7e43ca9cf64d552f96221fcb9561aac2/Cake/Cake_main.py#L116), [실행 산출물 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-f48-cake-research-direction/cake-and-repair-results-ko.md).

우리 관점에서 검증할 간극은 세 가지다.

1. **측정하는 개입의 차이.** 기존 사실의 activation을 복원하는 tracing score가 새로운 사실을 weight에 쓰는 효율이나 주변 사실의 손상을 직접 측정하지 않는다.
2. **현재 상태의 차이.** W0에서 얻은 평균 prior는 어떤 요청이 왔는지, 앞층에 얼마나 썼는지, 누적 M과 보존 제약이 어떻게 달라졌는지 반영하지 않는다. residual 갱신은 이 중 일부만 보완한다.
3. **총 강도의 차이.** 상대 weight를 바꿔도 최종 layer는 남은 global target을 다 쓰려고 한다. F48은 conditional local native endpoint의 절반에서 멈출 수 있다. 상대배분과 target completion을 별도 변수로 다뤄야 한다.

즉 score는 측정 기반 prior이며, 그 score를 실제 write utility로 변환하는 부분에 모델링 가정이 있다. Softmax가 있다는 사실 자체가 잘못은 아니다. 동적 tracing을 반복해도 측정량과 필요한 utility가 다르면 문제가 남는다. Causal localization이 최적 editing layer를 예측하지 못한 선행 실험도 있으나, 그 결과를 현재 Llama stream에 자동 적용하지 않고 여기서 직접 검증해야 한다. [Hase et al., NeurIPS 2023](https://arxiv.org/abs/2301.04213).

우리 저장소의 실제 CAKE는 first1k W10에서98.90/85.15/81.18%, 10k W100에서98.40/88.775/62.935%다. V2 F48과 비교하면 PS/NS가 낮지만 actual context 문자열, L2(10 대1), decay(.4 대.5), clamp(.5 대.75), target family, seed/일부 수치설정이 다르다. 이는 whole-policy 참고 결과이며 causal weighting만의 대조가 아니다. 실행 source에는 hardcoded score가 있고 그 score까지 연결된 raw tracing receipt는 확인하지 못했다.

## 5. 과거 실험이 추가로 식별한 것

각 row 내부에서만 비교한다. Warm suffix, cold first1k, 다른 capsule의 과거 baseline을 하나의 paired 표로 합치지 않았다. 상세 출처와 분모는 [과거 근거 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-f48-cake-research-direction/historical-evidence-ko.md)에 있다.

| 완료 비교 | 관측 | 이번 방향에 주는 의미 |
|---|---|---|
| Warm W50 suffix, L4 .75만 적용 | N4 대비 PS−2.10/NS+.80pp | 강도 감소만으로도 tradeoff가 생김 |
| Warm REFIT4 | PS+.60/NS+.67pp, 단 P strict−.90pp | 다른 층 없이 fresh refit도 효과; gen 지표를 분리해야 함 |
| Cold7 REFIT4 | PS−.25/NS+1.61pp | cold에서도 NS 방향은 유지되나 warm PS 이득은 재현되지 않음 |
| Cold7 같은(.75,1), local 대 terminal | local95.95/82.27, terminal97.10/71.61의 PS/NS | allocation보다 target family 효과가 클 수 있음 |
| Cold7 LD | (.75,.5)9회, PS−1.25/NS+2.84pp | v2 fixed F48과 유사; 강한 fixed baseline의 필요성 |
| Warm frozen/carried target 반복 | REFIT4보다 PS 강화, NS 악화 | target reset과 단순 추가 최적화는 다름 |
| EP-TW-1 | 국소 KL 감소, correction norm은 native의 .00675–.02355%; final NS−.16pp | 미세 proxy 개선이 functional 이득을 뜻하지 않음 |
| SL-ZFlow | target NLL 개선과 동시에 PS−2.95/NS−8.42pp | competing target과 margin까지 봐야 함 |
| B-OS 부분 repair | 미수렴 PCG와 finite step에서 Current/Base 악화 | 특정 큰 correction의 실패; repair 전반 불가능성은 아님 |

### 새로 확인한 repair 기술 결과

9/17 repair는 오래된 PENDING 메모 이후 **실제 technical READY**가 존재한다. V2 N4 B1과 같은 full-L4 anchor에서 L8 preservation 방향을 적용했을 때 S64 KL이 .001720908→.001310121로23.87% 감소했고 canonical strict/pair100/100을 유지했다. 단순 R-GD도 .001310110로 사실상 같은 첫 probe 결과였다. 이 점에서는 QP의 추가 이점이 확인되지 않았다.

이 결과는 native L8 target을 다시 만드는 F48 보완과 다른 방향이다. Full-L4 이후 보존 개선의 가능성을 보여주지만 B1 한 번의 technical endpoint이며 Past 없음, PS/NS/Dev 미측정, scientific commits0이다. Layer별 repair 순위나 lifelong 효과를 주장할 수 없다. 특히 canonical E 보호가 v2 native-context E 보호와 같지 않다.

### 최신 same-layer correction 설계와의 접점

[9/18 EN-F 설계](/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-18-single-layer-edit-preserving-correction-design-v1.md)는 full native 이후 **같은 L4**에서 full-token edit response를 유지하는 `D K_E=0` 방향으로 W0 KL을 낮추려 한다. 현재는 상세 설계/CPU toy 검증이고 model runner/GPU 결과는 없다. 위 L8 pilot이 이 L4 공간의 유효 rank/gradient를 증명하지 않는다.

EN-F의 H1–H3와 SCALE 대조는 “편집을 약하게 한 결과인가, 편집을 유지하는 별도 보존 자유도인가”를 직접 묻는다. 이번 F48 분석과 연구 질문이 잘 연결된다. 이미 이 설계가 진행 방향이라면 그 검증을 우선하고, CAKE/다층 allocation을 같은 실행에 섞어 가설을 늘리지 않는 것이 적절하다. 아래 allocation 후속은 별도의 조건부 연구 경로다.

## 6. 후속 실험: 무엇부터 식별할 것인가

모두 **미실행 제안**이다. 기존 데이터는 개발 자료다. 아래 단계는 한 번에 전체 matrix를 제출하라는 뜻이 아니며, 각 결과가 다음 단계의 필요성을 결정한다.

### P0. 기존 자료로 얻을 수 있는 분석을 먼저 고정

이번 분석에서 동일-entry B1 frontier, final PS/NS/strict, at-write와 retention, 비용을 다시 연결했다. 남은 보고 규칙은 다음처럼 고정하는 것이 좋다.

- original strict-quality track과 품질 손실을 허용하는 tradeoff track을 분리한다. F48의 guard 위반을 삭제하거나 threshold를 사후 변경하지 않는다.
- official P/N과 Dev는 현재처럼 선택 봉인 후 observer로 유지한다. 이 결과로 다음 방법을 고르는 것은 개발 의사결정이며 independent final test가 아니다.
- 관측된 후보를 재선택해 이어 붙인 hypothetical chain을 실제 정책 성능으로 보고하지 않는다. 배분이 달라지면 다음 W/M/target도 달라진다.

### P1. 가장 싼 기전 식별: scalar, 같은 층 refit, L8 보완

먼저 동일 W0/같은 B100에서 native L4 fit 하나를 공유한다. Native L4 방향의 배율만 바꾸는5점(.75,.875,.95,.99,1)은 추가 target fitting 없이 평가할 수 있다. Partial-L4 .75 상태에서 conditional L8 fit 하나와 same-L4 fresh fit 하나를 만든다. 두 보완 방향 각각에 .5/1의 gate를 적용한다. 총 **9 endpoint, native fit3회=target request-call300회**의 진단이다. 수치적 동일 endpoint가 생기면 중복 평가를 제거하고 기록한다. Forward/scoring·restore 비용은 별도다.

| 비교 | 식별할 질문 | 추가 native fit |
|---|---|---:|
| L4 scalar5점 | locality가 단순 감쇠로 얼마나 설명되는가 | 0; 공통 L4 fit1회 |
| Partial-L4→fresh L4, gate .5/1 | 같은 층의 추가 fresh fit으로 보완 가능한가 | 1 |
| Partial-L4→fresh L8, gate .5/1 | 다른 층의 조건부 방향이 추가 이득을 주는가 | 1 |

핵심 판정은 **동일 E/strict/pair 및 관측 PS 품질 수준에서 scalar curve보다 낮은 KL/높은 NS를 얻는가**다. 후보별 공식 PS를 보고 online 정책의 step을 고르는 실험이 아니라, 보완 방향의 가능성을 진단하는 사후 response surface다. 여기서 고른 설정은 calibration 선택으로 기록하고 다음 고정 평가에서 검증한다.

Fresh L4는 target 갱신과 추가 solve를 함께 바꾼다. Target reset 자체를 식별하려면 동일 partial-L4에서 처음의 frozen L4 target과 fresh L4 target을 동일 solve 규약으로 실현하는 대조가 추가로 필요하다. L8의 조건부 target 기여를 보려면 entry-L8 target을 한 번 계산해 같은 partial-L4에서 frozen target을 실현하는 대조를 추가한다(+1fit). 이것은 CAKE의 정확한 재현이 아닌 target 재계산 ablation이다. Prefix가 다르면 native fit을 재사용하지 않는다.

B1만으로 일반화하지 않는다. 필요성이 확인되면 공통 native chain의 B5/B10 entry에서 반복한다. 현재 v2는 W/M checkpoint를 저장하지 않았으므로 이 상태를 무료로 다시 평가할 수 없다. 원래 capsule의 재실행 또는 완전한 과거 checkpoint의 parity 확인 비용을 명시해야 한다. 같은 entry의 shadow는 기전 진단이며 독립 lifelong chain이 아니다.

**확대 판단:** scalar가 F48/refit의 품질–보존 곡선을 사실상 설명하면 allocation complexity보다 좋은 strength control이 우선이다. Same-layer refit이 L8과 같으면 다른 층 없이 추가 fresh fit으로 설명할 수 있고, target reset의 고유 역할은 위 대조로 확인한다. L8이 matched-quality에서 명확히 좋을 때 층 선택 연구의 필요성이 생긴다.

### P2. 보완 층과 causal prior의 예측력을 검증

다층 방향이 필요하다는 결과가 나올 때 진행한다. C45678에서 실제 선택된 L5, F48의 L8, 동일층 L4를 우선 비교하고 L6는 별도 필요성이 있을 때 추가한다. L7/L8 zero-Adam이 많았다는 이유로 모든 prefix에서 해당 층을 제거하지는 않는다.

주요 측정량은 각 prefix의 실제 finite 후보에 대한 (a) rewrite 회복량, (b) 추가 S64/Dev 손상, (c) past 손상, (d) actual layer ΔW, (e) fit/score 비용이다. 예를 들어

\[
U_l(a)=E(W_{prefix})-E(W_{prefix}+aD_l),\quad
C_l(a)=B(W_{prefix}+aD_l)-B(W_{prefix}).
\]

U/C라는 불안정한 단일 비율로 모두 정렬하기보다 같은 품질을 만족하는 후보의 보존–비용 frontier를 비교한다. 사전 edit guard를 유지하면서 C<0이면 preservation repair 성격, C>0와 U>0이면 edit 보완 성격이다. Guard 없이 C<0만 요구하면 편집을 취소한 후보도 repair로 오인하게 된다. 이 수치는 definition이며 이미 그러한 분리가 최적인 controller라는 검증 결과는 아니다.

CAKE causal prior는 이 **실제 조건부 효율 순위**를 예측하는지 먼저 검사한다. Current residual, native norm, history 조건수 및 단순 depth 추세와 비교하되, 고정5층 score는 layer identity의 결정함수라 unrestricted layer fixed effects와 독립된 효과를 식별할 수 없다는 점을 명시한다. 선형 depth보다 예측이 좋다는 사실만으로 causal semantics가 입증되지는 않는다. 고정 score의 반복을 독립 표본 수천 개로 세지 않고 요청·batch·order 의존성을 반영하며, 동일 family의 prior permutation 개입이나 여러 state/model에서의 score 변화로 추가 가치를 검증한다.

CAKE weighting 자체의 인과 대조는 CAKE의 target/solve/strength를 고정한 채 uniform, 원래 causal, reversed/permuted prior를 비교한다. 우리 local-z에 그 prior를 가져오면 별도 **CAKE-prior local-z variant**로 이름 붙인다. Gate와 remaining residual 비율을 혼동해 원 CAKE baseline이라 부르지 않는다.

총 강도 효과는 상대 prior와 별도 축으로 통제한다. CAKE 마지막 residual completion을 바꾸는 실험은 명시적인 strength ablation이며 원 방법과 구별한다. Prior별 temperature/strength를 official test에 맞춰 각각 튜닝하면 공정한 인과 비교가 사라진다.

Online tracing 재계산은 기본 후속이 아니다. Frozen score가 실제 utility를 어느 정도 예측하되 horizon에 따라 그 관계가 변한다는 증거가 있을 때 일부 anchor에서 refresh 효과와 비용을 측정한다. 예측력이 애초에 없으면 더 자주 측정하는 것이 해결책이라는 근거가 없다.

### P3. Adaptive의 가치와 비용을 따로 증명

같은 target family·candidate family·quality track에서 다음을 비교한다.

1. F48 원점과 calibration에서 고정한 best static vector.
2. 소수 completed endpoint로 고르는 online adaptive 정책.
3. 진단용으로만 쓰는 더 넓은 finite grid; 전역 oracle이라고 부르지 않음.

먼저 같은 entry/state에서 (3)의 후보가 고정 정책의 action보다 더 좋은 품질–보존 점을 제공하는지 진단한다. 이것은 조건부 기회 차이이며 더 좋은 lifelong trajectory를 증명하지 않는다. 정책 우월성은 각자 W0에서 시작해 자신의 선택으로 W/M을 이어 가는 실제 chain끼리 비교한다. 같은-entry 차이가 없다면 현재 후보 family에서 adaptive 필요성이 입증되지 않은 것이다. 차이가 있다면 native fit2–3회와 소수 endpoint score로 그 이득을 얼마나 재현하는지 평가한다. 이 fit 수는 목표 비용이며 검증된 성능 보장이 아니다.

학습 정책은 현재 v2 후보를 학습 데이터로 활용할 수 있으나, 후보 관측이 guard·budget·pruning에 편향되어 있고 같은 chain에 종속되어 있음을 반영해야 한다. Official N의 사후 최적 action을 label로 만들었다면 이를 개발용 privileged supervision으로 명시하고 독립 평가 stream에서 검증한다. 현재 official P/N observer 규약을 조용히 online optimizer로 변경해서는 안 된다.

Best static을 이긴다는 주장은 (i) 같은 gen에서 locality 개선, (ii) 같은 tradeoff에서 비용 감소, (iii) 새 order/horizon에서 안정성 향상 중 무엇인지 사전에 정한다. “계수가 batch마다 달라졌다”만으로는 기여가 아니다.

### 별도 경로: full edit를 유지하는 preservation correction

PS 손실0이 중요한 목표라면 F48 강도 탐색과 함께 생각할 가치가 높은 경로다. Full native 이후의 편집 반응을 보호하면서 보존 방향을 찾으므로 약한 edit만으로 설명될 위험을 직접 줄인다. 9/17 L8 pilot은 L4 weight를 고정하고 다른 층을 바꾸지만, EN-F는 같은 L4 weight를 바꾸면서 지정된 full-token 반응을 유지한다. 어느 경우도 S64 감소만으로 성공을 선언할 수 없다.

현재 9/18 EN-F의 H1(rank와 projected functional gradient), H2(실제 Dev/N 개선), H3(SCALE 대비·gen 보호)를 먼저 검증하는 편이 현재 설계와 일치한다. L8 GD/QP pilot은 가능성의 참고이며 EN-F 대체 증거가 아니다. 단순 GD가 QP와 사실상 같은 첫 결과를 낸 만큼 복잡한 QP나 대규모 repair-layer sweep을 먼저 확대할 근거는 아직 약하다.

## 7. Horizon이 늘 때 무엇을 예상하고 측정할 것인가

### 현재 관측은 즉시 손상과 후속 손상 감소를 모두 시사한다

| 정책 | pooled at-write NS | W10 NS | 이후 net 변화 |
|---|---:|---:|---:|
| N4 | 83.79 | 80.26 | −3.53pp |
| F48 | 85.29 | 83.24 | −2.05pp |

F48의 최종 NS+2.98pp는 산술적으로 at-write 차이+1.50과 이후 net 변화 차이+1.48로 나뉜다. 서로 다른 chain의 기술적 분해이며 별개 인과효과 추정은 아니다. 같은 first500을 W5→W10에서 추적해도 N4 NS84.06→79.62(−4.44), F48 85.62→82.82(−2.80)다. 약한 write의 누적 간섭 완화 가설과 일치한다.

반면 F48의 pooled at-write PS95.35는 N4 96.40보다 이미1.05pp 낮다. Final 차이는−1.25pp다. 오래 기다리면 gen 문제가 해결될 것이라는 근거는 없다. 우선 작성 시점 일반화 품질을 개선해야 한다.

1k에서 유리한 F48이10k에서도 같은 우위를 유지할지는 미측정이다. M 누적, 예전 key의 현재 모델 부적합, remaining edit 자유도, 요청 간 충돌 때문에 약한 write가 후반에 부족해질 수도 있다. 모든 arm의 S64 KL은10batch 내에서 계속 증가했고 damage plateau 증거도 없다.

Past64의 마지막 batch 실제 보호 비율은1k에서64/899≈7.12%다(과거900건 중 현재 overwrite 제외). 고정64개를 유지하면5k에서 대략64/4900≈1.31%,10k에서64/9900≈.65%로 줄어든다. 미래 분모는 overwrite 수에 따라 달라진다. V2 전체에서 Past64의 직접 출력 품질 guard에 등장한 고유 과거 요청은185개뿐이었다. Native M에는 전체 요청의 key가 누적되므로 이185개를 history 전체로 해석하면 안 된다. 또한 고정64개의 출력 guard 통과가 전체 과거 편집 보존을 보장하지 않는다. 더 긴 horizon에는 cohort별 at-write→후속 변화, 성공 lost/gained, P strict/joint, margin tail을 함께 추적해야 한다.

확장에서는 잠근 후보 소수와 N4/F48/G48을 우선한다. 하나의 order에서 모든 method를10k로 늘리는 것보다 short horizon에서 서로 다른 order에 대한 일관성을 확인하고, 생존한 후보를1k→5k→10k로 늘리는 편이 질문을 더 잘 분리한다. 새로운 order는 order robustness이며 이미 본 요청의 unseen 일반화가 아니다. Scope가 고정되면 calibration/observer/최종 holdout 역할도 함께 봉인해야 한다.

## 8. 계산량과 반드시 남길 산출물

| 정책 | target request-call | native solve | online score | native Adam step | program 초 |
|---|---:|---:|---:|---:|---:|
| N4 | 1,000 | 10 | 10 | 24,000 | 4,535 |
| F48 | 2,000 | 20 | 20 | 27,912 | 5,554 |
| G48 | 3,000 | 30 | 60 | 28,152 | 6,816 |
| C48 | 15,100 | 151 | 174 | 65,759 | 16,702 |
| C45678 | 33,100 | 331 | 142 | 45,791 | 19,711 |

F48은 target 횟수가2배지만 Adam은 약1.16배, 전체시간은1.225배다. 둘째 fit의 많은 요청이 이미 충분히 편집되어 있기 때문이다. 반대로 C45678은 C48보다 Adam이 적어도 전체시간은 길다. Zero-Adam에도 초기 forward, key extraction, solve, 상태복구·복사·hash·teacher read가 발생한다. Adam 수만 비용으로 쓰면 층 탐색의 실제 부담을 놓친다.

C45678은 G48보다 전체시간2.89배지만 최종 PS−.10/NS+.24pp이고 Dev KL은 조금 더 나쁘다. 현재 관측은 더 큰 탐색을 실용 기본값으로 채택할 근거가 약하다. 다섯 층 모두의 history append는 모든 v2 arm에50회씩 수행했으므로 F48의 결과를 구현 최적화된 두 층 전용 production memory 비용이라고 해석해서도 안 된다.

고정 batch size와 후보 cap에서 editor+고정 bank 평가 작업량은 대체로 horizon에 선형이다. 다만 요청 난도와 early-stop 변화로 상수는 변한다. 현재 매5batch full-seen 평가를 그대로 연장하면 평가 요청 누적량은1k에서1,500,5k에서27,500,10k에서105,000으로 각각18.3배/70배다. Whole program 시간을 단순5배/10배로 예측할 수 없다. Cohort panel+사전 고정 endpoint full evaluation으로 평가 일정을 설계하고 필요 시 전체 결과를 보고한다.

Native history M shape는 horizon과 무관하게 `[1,14336,14336]`다. FP32 다섯 M은3.828GiB, 다섯 P까지7.656GiB로 기본 shape는 고정이다. History 값의 누적과 capacity/conditioning 변화는 별개의 문제다. 선택층 수가 적다는 사실만으로 남은 capacity를 수치화할 수 없다.

CAKE tracing을 매batch 갱신하면 이전 fact/noise/restore forward 비용이 추가된다. 원래 frozen prior 생성 비용, online 재계산 비용, amortized 비용을 각각 기록해야 한다. “Softmax 몇 번”의 계산량으로 tracing 정책 전체 비용을 비교하면 안 된다.

다음 실행의 필수 산출물은 아래와 같다.

- 배치 entry와 각 prefix의 state/history identity, target anchor/Δz, 실제 적용 후 layer별 ΔW norm 및 relative norm. 전역 concat norm만으로 layer 분배를 설명하지 않는다.
- Layer별 누적 path length와 최종 net displacement를 따로 보관한다. 필요하면 방향 내적/각도, activation response norm, preservation covariance energy도 명시적인 정의와 함께 기록한다.
- Same-entry partial-L4→보완과 full-L4→선택점의 E/KL/P/N 차이, request별 new/true NLL·margin·strict/joint 성공.
- Accepted/rejected/zero-effective endpoint 전부, projected gradient/rank 등 사용한 방법의 진단, tolerance와 실제 FP32 materialization 변화.
- Native fit/target/Adam·forward/backward token·solve/score·teacher read·state I/O·history·observer·총 할당시간을 분리한 비용. Cache 공유 비용과 독립 arm 비용을 구별한다.
- Replay할 anchor는 모델 변경 weight뿐 아니라 해당 history·context·RNG capsule까지 완전하게 저장한다. 부족하면 exact resume/replay를 주장하지 않는다.

## 9. 지금 가능한 주장과 앞으로 필요한 증거

지금 가능한 주장은 **“고정 두 단계 local-z 정책 F48은 native L4에 비해 작은 추가 비용으로 큰 보존 이득을 보였으나 paraphrase 품질을 희생했다. 실측 단계 비교는 초기 강도 감소와 후속 edit 보완의 결합을 지지한다”**다.

다음으로 가장 가치 있는 증거는 **같은 편집·일반화 품질에서 scalar-only를 넘어서는 보완/보존 방향이 존재하는가**다. 이것이 확인되면 그 방향의 위치를 고정 prior가 잘 예측하는지, 현재 상태에 따라 선택해야 하는지, 그 선택 비용이 이득에 비해 작은지를 순서대로 검증할 수 있다. 현재 증거로 CAKE의 causal prior를 무효라고 선언하거나 dynamic layer allocation의 우월성을 선언할 수는 없다.

검토 기반: [v2 전체 독립 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-sequential-local-z-allocation-review/independent-review-ko.md), [CAKE 방법 감사](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-f48-cake-research-direction/cake-method-audit-ko.md), [과거 실험 비교](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-f48-cake-research-direction/historical-evidence-ko.md), [CAKE 실제 결과·repair 상태](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-18-f48-cake-research-direction/cake-and-repair-results-ko.md). 이 문서는 실행 contract가 아니라 근거와 조건부 우선순위를 정리한 연구 방향이다.
