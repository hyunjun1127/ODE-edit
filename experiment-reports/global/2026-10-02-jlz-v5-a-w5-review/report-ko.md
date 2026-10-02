# JLZ v5 Arm A: 중단된 500-edit 산출물 CPU 상세 리뷰

2026-10-02. 대상은 server4 v5 Arm A의 완료된 B1–B5와 W5 누적 평가다. 기존 실험을 재개하거나 새 GPU 실험을 실행하지 않았다. Arm B에 대한 결론은 포함하지 않는다.

**현재 가장 강한 결론은 “얻은 편집을 잃었다”보다 “처음부터 paraphrase로 충분히 전달되는 편집을 얻지 못했다”이다.** 편집 직후의 P는 합산 71.5%, W5는 73.2%다. W5 P 실패 268개 중 266개가 최초 commit 직후부터 실패했다. v5가 측정된 neighborhood 선호율을 pre-edit 가까이 유지한 것은 사실이지만, 이 결과만으로 성공적인 편집과 보존의 균형을 달성했다고 말할 수 없다.

## 1. 원자료와 비교 범위

- Arm A 실행 원본: `/data/janghj/ODE-edit/local/jlz-writer-coupled-v5/20261002-v1/attempt-r1/arm-A/main`.
- 실제 실행 source commit: `fd2082e4720aa971dc64b00a133fce1d8e5e7d93`. 보관된 source.tar의 `jlz_writer_coupled` 파일은 검토 worktree 파일과 byte 단위로 일치한다.
- 읽기 전용 snapshot은 251파일, 8,384,562바이트다. archive SHA256은 `b0cffde5ad826f70f2fb90da3531e8e5c5314d89210b953f4ad7bab7cbe06924`. 파일별 SHA/크기를 검사했다.
- W5의 6,500행: R 500, P 1,000, N 5,000. 중복·누락 ID가 없으며 원문항 재집계와 저장된 summary가 일치한다.
- W1–W4는 그 시점에 편집한 100개만 평가했고 W5는 누적 500개를 평가했다. “편집 직후 합산”은 서로 다른 다섯 모델 시점의 결과를 문항별로 모은 값이며, 단일 시점 누적 평가로 표현하지 않는다.
- 같은 첫 500개 W0 raw를 재사용했다. W0 SHA256은 `4436ce3de7889164398c189d4cb8be56716245499e1623bd924962b06243d4d8`. 전체 2k W0의 NS를 500개 기준과 혼용하지 않았다.
- v4↔v5는 prompt identity와 target token hash를 대조했다. W0는 prompt identity·case·kind·index·token count를 대조했지만 W0 파일 자체에는 target token hash가 없다.

baseline은 같은 고정 stream, 모델 revision, native context와 같은 논리적 NLL 평가 정의를 사용한다. 다만 baseline은 Transformers 4.44.2/full head/주로 MB16, v5는 4.57.1/selected-position head/MB2 및 new/true 교차배치다. 동일 state에서 평가기 수치 parity를 새로 증명한 비교는 아니다. BLUE baseline은 L4+L8, L2=1 조건이다. native AlphaEdit L4–L8, L2=10과 조건을 구분한다.

추가로 server3 MEMIT-H와 server4 BLUE의 B005 `seen-full.json`만 읽기 전용으로 가져와 remote/local SHA를 대조했다. 서로 다른 raw schema를 고정 dataset의 첫 500개 실제 문장과 target에 연결해 각 6,500행의 case·kind·index·문장·target identity 및 token count를 검증했다. baseline raw에는 token-ID hash가 없어 전체 token-ID의 직접 일치까지 확인한 것은 아니다.

## 2. NS를 pre-edit와 함께 읽어야 한다

| 방법 | R /500 | P /1,000 | N /5,000 |
|---|---:|---:|---:|
| v5 A | 95.20% | 73.20% | 87.46% |
| v4 A | 99.60% | 98.70% | 62.24% |
| MEMIT-H | 99.00% | 87.30% | 86.28% |
| AlphaEdit | 98.60% | 89.10% | 82.20% |
| AlphaEdit-BLUE | 100.00% | 96.20% | 83.80% |
| CAKE | 97.80% | 80.40% | 85.02% |
| Pre-edit, 같은 첫 500개 | — | — | 87.84% |

v5 A의 N은 pre-edit 대비 −0.38%p다. 문항 단위로는 기존 성공 4,392개 중 4,320개 유지, **72개 손실**, 기존 실패 중 **53개 개선**으로 최종 4,373개다. 순손실 19개가 모든 출력의 불변을 뜻하지 않는다. N의 기존 정답 NLL은 평균 5.299→4.894로 낮아졌고, 편집 target NLL도 11.075→10.357로 낮아졌다.

MEMIT-H 대비 N 이득은 +1.18%p이고 P 차이는 −14.10%p다. BLUE 대비 N +3.66%p, P −23.00%p다. 서로 분모와 의미가 다른 지표를 단순 합산해 상쇄시키면 안 된다. 현재는 N을 잘 유지한 관측점이지만 baseline보다 적은 편집 일반화를 얻었다.

## 3. P 부족은 주로 초기 획득 문제다

| 편집 cohort | 편집 직후 P /200 | W5에서 같은 문항 P /200 |
|---|---:|---:|
| B1 | 83.0% | 86.0% |
| B2 | 74.0% | 77.0% |
| B3 | 64.0% | 66.0% |
| B4 | 73.0% | 73.5% |
| B5 | 63.5% | 63.5% |
| 합산 /1,000 | **71.5%** | **73.2%** |

편집 직후→W5의 P 전이는 성공 유지 713개, 성공→실패 2개, 실패→성공 19개, 실패 유지 266개다. R은 성공 476개를 모두 유지하며 처음부터 실패한 24개가 그대로다. 평가 사이에 일시적으로 잊었다 회복했는지는 관측되지 않으므로 “중간 모든 시점의 망각 0”이라는 뜻은 아니다.

baseline의 편집 직후 P 합산→W5도 MEMIT-H 86.6→87.3%, AlphaEdit 88.7→89.1%, BLUE 95.8→96.2%, CAKE 78.4→80.4%다. 이 값들은 baseline 공개 집계의 전후 순변화이며 paired lost/gained로 해석하지 않는다.

v4 A는 편집 직후 합산 P 99.0→W5 98.7%였다. **과거 history가 없는 B1부터 v4 P99% 대 v5 P83%**다. 따라서 v5의 낮은 P를 누적 history의 억제나 장기 망각만으로 설명할 수 없고, v4의 높은 P를 후속 편집의 부수적 drift만으로 설명할 수도 없다. B1에서 v4 N85%, v5 N88.4%로 이미 tradeoff가 발생했다.

cohort마다 다른 사실을 편집하므로 B1보다 B5가 낮다는 것만으로 시간·history 크기에 따른 인과적 악화를 주장하지 않는다.

## 4. R만 올리면 해결되는 정도가 아니다

- R 성공 476개 사실의 P는 **727/952=76.37%**다.
- 전체 P 실패 268개 중 **225개(84%)가 R 성공 사실**에서 발생한다.
- R 성공 사실 중 58개는 두 paraphrase 모두 실패, 109개는 하나만 성공, 309개는 두 개 모두 성공이다.
- R 실패 24개 사실에 딸린 P 48개를 가정상 모두 성공으로 바꿔도 P는 77.5%다. 이는 가상의 산술 상한이며 실제 개선 예상치가 아니다.

직접 문장에서는 새 답을 선호하지만 문장을 바꾸면 일반화가 약한 실패가 큰 비중이다. R의 남은 4.8% 실패와 별도로 이 문제를 다뤄야 한다.

baseline raw를 같은 P 문항끼리 짝지은 결과도 이를 뒷받침한다.

| P 1,000문항 비교 | 둘 다 성공 | baseline만 성공 | v5 A만 성공 | 둘 다 실패 |
|---|---:|---:|---:|---:|
| MEMIT-H ↔ v5 A | 719 | 154 | 13 | 114 |
| BLUE ↔ v5 A | 728 | 234 | 4 | 34 |

v5 A의 실패 268개 중 MEMIT-H와 BLUE가 모두 성공한 문항은 147개, BLUE만 성공 87개, MEMIT-H만 성공 7개, 둘 다 실패 27개다. 보편적으로 어려운 문항만 남은 결과가 아니다.

예를 들어 counterfactual case8781의 새 target은 `marble`이다. A의 rewrite new NLL은 0.0635인데 paraphrase P1에서는 13.406이고, 같은 P1에서 BLUE는 0.0782다. case3205의 새 target `Scots`도 A rewrite 0.00632 대 P0 14.490, BLUE P0 1.7435다. 두 경우 모두 A는 R 성공/P 실패이고 MEMIT-H와 BLUE는 해당 P 선호율 평가에 성공했다. 진단 예시이며 전체 사례를 대표하는 무작위 표본은 아니다.

| 같은 W5 P 1,000문항 | P 선호율 | Teacher-forced strict | 새 답 평균 NLL ↓ |
|---|---:|---:|---:|
| v5 A | 73.2% | 40.5% | 3.408 |
| MEMIT-H | 87.3% | 58.5% | 2.174 |
| AlphaEdit | 89.1% | 62.0% | 1.918 |
| BLUE | 96.2% | 66.9% | 1.547 |
| CAKE | 80.4% | 51.3% | 2.738 |
| v4 A | 98.7% | 73.8% | 1.257 |

strict는 teacher forcing 하에서 target의 모든 토큰 argmax가 맞은 비율이며 자유 생성 정확도가 아니다. v5 R의 strict는 90.0%, new NLL은 0.636이다. P 실패 268개 중 209개는 `true NLL − new NLL < −1`, 161개는 < −2다. 실패 margin 중앙값은 −2.469다. 선호율 판정 경계 주변의 작은 수치 흔들림만으로 관측 패턴을 설명하기 어렵다. 다만 이것이 평가기 차이가 전혀 없음을 증명하지는 않는다.

P 1,000개 중 986개는 단일 토큰 target이고 이들에서도 P는 73.12%다. multi-token teacher forcing을 주원인으로 삼을 근거는 없다.

v4→v5 P 평균 margin은 11.211→2.929다. 새 답 NLL이 1.257→3.408로 높아지고 기존 답 NLL은 12.468→6.337로 낮아진 결과다. 즉 v4의 높은 preference에는 새 답 강화와 기존 답 억제가 함께 기여했다. 둘 중 하나만 원인으로 설명하지 않는다.

## 5. 실제 구현은 무엇을 최적화했는가

검토한 코드에서 즉각적인 KL 방향 오류, request/context 정규화 오류, anchor chain rule 오류, prox 수식 오류, 최종 후보와 commit 가중치 불일치는 발견하지 못했다. 이것은 검토한 항목의 결론이며 모든 잠재 버그의 부재 증명은 아니다.

Arm A는 η=0이다. native current‖entry KL, norm, C0/history geometry는 있지만 **추가적인 과거 문장 보존 손실은 계산하지 않는다**. metadata의 reference count가 있다는 이유로 A도 replay penalty를 받았다고 해석하면 안 된다.

현재 v5는 entry에서 고정한 P를 이용해 모든 eligible layer L4–L8에서

\[
W_\ell^{\rm eff}=W_\ell^{\rm entry}+\operatorname{cast}_{32}(D_\ell P_\ell^\top)
\]

를 만들고, 이 물리적 가중치를 **모든 토큰에 적용한 forward**의 native 문장 손실을 공동 최적화한다. 받아들인 실제 가중치를 그대로 commit한다. 이는 v4의 virtual subject intervention과 실제 writer 사이 평가 불일치를 줄이는 장점이다.

그러나 D의 열을 곧바로 “해당 subject에서 실제 실현되는 local δ”로 읽으면 안 된다. subject key k에서 직접 write 작용은

\[
\Delta h_{\ell,r,c}^{\rm direct}=D_\ell P_\ell^\top k_{\ell,r,c}(D_{<\ell})
\]

다. 다른 요청의 D 열, context별 key, 아래층 수정이 함께 영향을 준다. 전체 activation 변화에는 기존 W가 변경된 key에 작용하는 항도 포함된다. \(D_{\ell,r}\) 자체와는 일반적으로 다르다.

따라서 native와 같은 문장·KL 방향·norm 형태를 유지해도 **norm을 걸고 최적화하는 변수의 기능적 의미까지 동일하지는 않다**. 현재 D는 고정된 write basis의 계수이자 nominal payload다. norm/clamp는 이 D에 적용되며 실현된 subject 변화에 직접 적용되는 것이 아니다. 이 구분은 “각 층 local-z를 공동 최적화해 배분한다”는 claim을 점검할 때 중요하다.

native subject δ 최적화는 subject 위치를 통해 손실을 개선하도록 제한한다. 반면 v5의 물리적 forward는 subject뿐 아니라 관계를 설명하는 prompt 토큰의 변화를 통해서도 NLL을 개선할 수 있다. **학습 문장에서는 효과적이나 paraphrase에서는 약한 경로를 이용했을 가능성**이 있다. 현재 데이터는 R–P 격차와 이 경로의 존재를 보여주지만 실제로 이 경로가 원인인지는 확인하지 못했다. 단일 토큰 target이 대부분이므로 preceding answer token 경로보다 prompt/subject 경로의 구분이 핵심이다.

근거 코드: `solver.py:112`(norm/ball prox), `solver.py:213`(global τ), `physical.py:9`(모든 토큰 linear/VJP), `physical.py:21`(materialization), `oracle.py:41`(NLL), `oracle.py:53`(KL), `oracle.py:76`(relative coordinate gradient), `entry.py:50`(fixed P), `writer.py:35`(accepted weight commit). 파일은 `project/run_scripts/jlz_writer_coupled/` 아래에 있다.

## 6. solver가 충분히 진행되었다는 근거가 없다

5개 batch 모두 25 candidate/24 backward를 쓰고 `candidate_budget`으로 종료했다. accepted update 수는 **22,22,20,22,21**이다. 초기 zero와 거절된 후보도 예산에 포함된다.

| Batch | accepted updates | 최종 τ / 최초 trial τ | 후보24→25 native NLL /request |
|---|---:|---:|---:|
| B1 | 22 | 1/4 | 0.418→0.372 |
| B2 | 22 | 1/4 | 0.711→0.676 |
| B3 | 20 | 1/16 | 0.616→0.607 |
| B4 | 22 | 1/4 | 0.449→0.400 |
| B5 | 21 | 1/8 | 0.436→0.406 |

이는 학습 readout에서 native 6개 rewrite context를 같은 비중으로 합친 NLL이며, 최종 모델의 공식 R/P 평가 NLL과 동일한 측정치가 아니다.

τ는 모든 layer/request에 공통이고 최대 group gradient로 초기화한다. 거절 시 절반으로 줄고, 수락 후에도 커지지 않는다. 한 요청·층의 큰 gradient나 곡률이 다른 group의 진행까지 제약할 수 있다. 25라는 숫자가 native Adam의 25회와 같아도 진행량은 동일하지 않다. 서로 다른 좌표계의 τ와 Adam learning rate 숫자를 직접 배율 비교하지 않는다.

**거절된 13개 후보는 모두 직전 수락 후보보다 composite 목적값 자체는 낮았다.** 거절 이유는 smooth-majorization 부등식 불충족이다. 이는 diagnostic 성능 gate가 아닌 solver 수락 조건이며 구현은 설계대로다. 감소 후보를 버리고 τ를 더 줄이는 보수성이 실제로 나타났다는 증거지만, 모든 감소 후보를 무조건 수락하자는 결론은 아니다. 개별 group gradient가 저장되지 않아 특정 outlier 요청이 τ를 지배했다는 것까지 확인할 수는 없다.

최종 2,500개 layer/request group 중 zero group도 clamp 도달 group도 없고, 최대 relative norm은 약 0.16145로 cap 0.75보다 훨씬 작다. cap 자체가 최종 강도를 잘라냈다는 설명은 지지되지 않는다. 반면 작다는 사실만으로 norm penalty 영향이 없다고 할 수는 없다.

마지막 후보에도 손실 감소와 nonzero 이동이 있으며, 마지막 backward가 있는 후보24의 gradient는 크다. **후보25는 forward-only이므로 기록된 zero gradient는 초기화 placeholder이며 수렴 증거가 아니다.** 최종 proximal/KKT residual은 저장되지 않았다. 후보24의 진짜 gradient와 norm subgradient bound를 사용한 별도 CPU 계산은 solver 산출물에 둔다.

기록된 gradient 기준 후보24의 smooth gradient 전체 norm은 batch별 110.81–141.37이고, 가능한 norm subgradient norm의 상한은 2.67–2.78이다. `후보25 최대 group norm + 마지막 전체 step 길이 < 0.75`이므로 후보24의 모든 ball도 내부다. 따라서 후보24의 composite first-order residual 하한은 108.14–138.62다. SUM 손실과 상대 v 좌표 단위이며, **최종 후보25의 residual을 측정한 값은 아니다**. 직전 상태가 norm/clamp와 균형을 이뤄 멈춘 해라는 설명은 맞지 않는다.

최종 목적의 요청당 평균 구성은 native NLL 0.49216, weighted KL 0.02162, norm 0.04069다. 손실 값의 크기 자체를 gradient 영향력 비율로 해석하지 않는다.

현재 증거는 제한된 예산에서 최적화가 덜 진행되었다는 설명을 강하게 지지한다. 다만 더 진행했을 때 공식 P가 얼마 개선되고 N을 얼마나 잃을지는 이 로그만으로 알 수 없다. 학습 손실 감소는 P 개선의 보장이 아니다.

## 7. 물리적인 write 크기도 크게 달라졌다

서로 다른 다섯 weight matrix의 Frobenius norm 제곱합 제곱근으로 batch별 실제 increment를 비교했다.

| Batch | v4 실제 ΔW norm | v5 실제 ΔW norm | v5/v4 |
|---|---:|---:|---:|
| B1 | 17.668 | 2.187 | 12.38% |
| B2 | 18.123 | 1.641 | 9.05% |
| B3 | 19.111 | 1.496 | 7.83% |
| B4 | 19.653 | 1.378 | 7.01% |
| B5 | 22.562 | 1.261 | 5.59% |

v5는 v4보다 훨씬 작은 수정으로 종료된다. 높은 N과 낮은 P라는 관측과 양립하지만, 방향과 작용 위치가 중요하므로 norm만으로 성능의 원인을 확정하지 않는다. B1 이후 두 방법의 entry state도 다르다. 이것은 batch별 increment이며, 배치 간 내적이 없으므로 최종 누적 ΔW norm으로 합산하지 않는다.

현재 v5에는 요청별 실제 subject displacement와 context별 writer realization을 저장한 자료가 없다. nominal D norm, 상대 norm, layer별 ΔW norm만 있다. 따라서 “어느 층이 실제 의미 변화의 몇 %를 담당했고 어느 층이 과부하인가”는 이 값들만으로 결론내릴 수 없다. 계층별 작은 몫을 최적 배분의 증거로 보기도 어렵다. 아직 수렴하지 않은 solver의 속도·conditioning이 섞여 있기 때문이다.

층 독점은 관측되지 않는다. 5개 batch의 상대 v 제곱 norm 비중은 L4–L8 순서로 26.22/20.92/20.15/18.94/13.77%, 실제 weight increment 제곱 norm 비중은 19.19/16.72/21.07/25.50/17.52%다. 모든 group이 활성화되었다. 두 비중은 계산된 크기 분포이지 인과적 편집 기여율이 아니다. 현재 P 부족을 소수 층에 몰린 배분 탓으로 돌릴 근거도 없다.

## 8. 설계 개선에 주는 결론

**우선순위는 초기 편집 일반화의 회복이다.** 현재 Arm A의 낮은 P를 해결하기 위해 과거 보존 loss를 더 강화하는 방향은 이 관측의 주원인과 맞지 않는다. 이미 성공한 편집의 W5 유지보다 처음 commit할 때의 P가 부족하다.

1. **고정 계산 예산에서 joint solver의 진행량을 개선한다.** 공통 scalar τ의 지속적 축소를 그대로 두고 cap만 키우는 접근은 계산 병목을 반복한다. 양의 metric/preconditioner와 그 metric에 맞는 prox·수락 규칙 등으로 conditioning을 다루는 것이 우선이다. objective와 native 문장·계수·clamp의 의미를 보존해야 하며, 각 group을 강제로 같은 크기로 정규화하거나 배분 균등화를 도입하는 것과 구별한다. 정확한 solver 설계는 별도 문서화가 필요하다.
2. **local-z의 의미와 실제 write의 연결을 분명히 한다.** subject를 통한 변경으로 native 손실을 개선하는 의도와 실제 물리적 writer 검증을 함께 유지해야 한다. D라는 이름만 local-z로 붙이는 것으로는 부족하다. 실제 subject 작용, native context 사이의 일관성, 다른 요청으로의 작용을 기준으로 변수를 해석하거나 설계해야 한다. 단순히 v4의 virtual-only 최적화로 돌아가면 기존 locality 문제를 다시 떠안을 수 있다.
3. **배분 판단을 실제 작용과 연결한다.** nominal norm 몫만으로 안전한 층/효율적인 층을 고르지 않는다. 모든 eligible layer를 열어 두고, 결과적으로 한 층에 집중되는 것도 허용한다. 요청·층별 realized subject change와 native context별 realization은 배분 해석에 필요한 정보이며, 엄격한 임의 diagnostic gate로 layer를 제거할 이유는 아니다.
4. **훈련 문장은 그대로 유지한다.** baseline과 같은 native 문장만 쓰며 official paraphrase/neighborhood를 새 목적함수나 온라인 후보 선택에 넣지 않는다. 이번 결과로 P를 직접 학습해 지표를 맞추는 방향을 정당화할 수 없다.

이 리뷰는 구조적 후보들의 우선순위를 정리한 것이다. full-context geometry, fixed P, 물리적 all-token graph, optimizer, 모델 상태가 함께 달라졌으므로 각 변경의 기여를 분리한 인과적 결론은 아니다. 현재 산출물만으로 “joint local-z라는 발상 자체가 틀렸다”거나 “misalignment 해결만 하면 P가 회복된다”고 결론내릴 수 없다.

## 9. 산출물과 재현

검토 worktree: `/mnt/raid5/janghj/.codex/worktrees/odeedit-jlz-v5-a-w5-review-20261002-v1`.

그 아래 `local/jlz-v5-a-w5-review/`에 읽기 전용 snapshot, acquisition receipt, 다음 CPU 분석을 보관한다.

- `metrics/analyze.py`, `metrics/summary.json`, `metrics/facts.csv`, `metrics/paired-prompts.csv`: identity 대조, W0·at-write·W5 paired 전이, R 조건부 P, target 길이·margin 분석.
- `baselines/reduce.py`, `baselines/summary.json`, `baselines/normalized.csv`, `baselines/comparability.json`: 기존 baseline 게시 집계와 비교 조건.
- `baselines/paired.py`, `baselines/paired-summary.json`, `baselines/paired-paraphrases.csv`, `baselines/raw-acquisition-receipt.json`: MEMIT-H·BLUE raw와 문항 단위 비교 및 수집 증거.
- `geometry/compare.py`, `geometry/summary.json`, `geometry/batches.csv`, `geometry/layers.csv`: v4/v5 실제 weight increment 비교.
- `solver/`: 후보별 진행량, 실제 gradient, nominal group norm 분석.

요약 JSON/CSV는 이 report 디렉터리에도 복사해 검토 자료로 제공한다. 분석 script는 원본 snapshot 경로를 사용하며 Python 표준 라이브러리만으로 재실행한다. 이번 검토에서 새 모델 forward/backward, GPU job 제출, production source 변경, 실험 재개는 없다.
