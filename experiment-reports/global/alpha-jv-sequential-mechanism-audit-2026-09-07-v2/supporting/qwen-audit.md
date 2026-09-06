# Qwen AlphaEdit JV sequential 1,000 — locality 독립 점검

작성: GH 독립 Qwen 감사, 2026-09-07. Analysis-only; source/실험/GPU/evaluator/서버2 통신/SH4 live artifact 접근 0.

## 1. 가장 중요한 결론

Qwen final W10에서 JV의 NS는 Official보다 **399/10,000 prompt, +3.99 percentage points** 높다. 이 개선은 집계 오류나 예전 token-preservation metric을 NS로 부른 결과가 아니라, 같은 prompt/target identity에서 `NLL(target_true) < NLL(target_new)`를 비교한 canonical preference 결과다. 평균 neighborhood NLL margin도 +0.516994 nat 개선되며, 그 평균 이동의 약 97.45%는 target-true NLL 감소에서 나온다. 잘못된 target-new 확률만 더 낮춰 상대 순위를 조작한 양상으로 설명되지는 않는다.

그러나 **“L8 집중 때문에 좋아졌다” 또는 “barrier가 historical locality를 보장했다”는 인과 결론은 현재 자료로 성립하지 않는다.** JV는 모든 batch에서 거의 L8만 쓰면서도 Official보다 총 batch-net Frobenius energy가 더 크다. 따라서 “단순히 모든 layer의 write를 작게 했기 때문에 좋아졌다”는 물리적 설명은 지지되지 않는다. 대신 **L4–7의 직접 weight disturbance를 매우 크게 줄이고 terminal-target layer에 write를 집중한 경로가 유리했을 가능성**은 실측과 일관된다. 이 설명과 current-response/native-history cost 자체의 이득을 분리하려면 이미 승인된 actual L8-only 결과가 핵심이다.

## 2. 검토 범위와 재현 가능한 입력

P = `/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-alpha-jv-review-20260907/experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1/`

S = `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-alpha-jv-sequential-detailed-analysis-v1/experiment-reports/servers/server1/alpha-jv-sequential1000-layer-review-2026-09-07-v1/`

확인한 publication: P의 factual report, final/current/seen metrics, paired endpoints, neighborhood transitions, online aggregates, physical layer batch metrics, history-cost shadows, cohort metrics, sample lock, external-table identities; S의 상세 report 및 동일 입력의 해석 범위. CSV에서 아래 수치는 별도 재계산했다. 실행 source는 `77358b1546d1baf83b3e251afcce663b08d7bfd7`, publication snapshot은 `0d0a0131e4a6a2a645dfa6530377d420a084d136`이다.

원본 prompt-level CSV/원본 weight·cache에 대한 독립 raw rehash는 이 감사에서 하지 않았다. 아래 external 두 표는 server2 local-only이며 같은 절대경로의 server1 파일은 존재하지 않는다. 파일 존재 확인만 수행했으며 server2 통신은 하지 않았다.

- `.../local/alpha-native-response-v31-sequential-routing/main-four-terminal-20260907-v1/prompt_transition_metrics.csv`, SHA `b9a38bd13e3812e0c9639a8ab19b0e4d790e70a4279be0c98979573b6132dca5`.
- 같은 root의 `rewrite_retention_matrix.csv`, SHA `81421d92ceee419fed2fc70d641cf52f6f28750aca6bb3ebb8dd958745d2453b`.

## 3. NS 구현과 denominator

실제 runtime call chain을 확인했다.

- `ordered_response_barrier_ode/runtime.py:777`의 endpoint가 `evaluate_counterfact_with_canonical_ns`를 호출한다.
- `ordered_response_barrier_ode/counterfact_locality_evaluator.py:70`은 stock rewrite/rephrase/locality-true evaluator 결과에 동일 neighborhood prompt의 target-new 평가를 추가한다. Shared tokenizer는 writer-compatible right-padding 설정을 유지하고 evaluator가 입력을 수동 left-pad한다.
- `alphaedit_strength_neutral_barrier/evaluator.py:75`는 target continuation의 teacher-forced token 평균 `-log_softmax`를 NLL로 반환한다. 자유생성 정확도가 아니다.
- `ordered_response_barrier_ode/artifacts.py:893`의 `_canonical_ns_payload`는 정확히 `true_nll < new_nll`, tie failure, prompt denominator, request의 모든 neighborhood가 성공한 strict count를 별도로 산출한다.
- `alpha_native_response_ode_v31_sequential/reporting.py:40`의 `bits`는 같은 case/prompt의 new/true NLL과 input identity를 결속한다. `synthesis.py`의 paired 비교는 arm 간 prompt/target identity가 다르면 거부한다.

Final W10 분모는 request 1,000, RS prompt 1,000, PS prompt 2,000, NS prompt 10,000이다. NS token denominator 10,110/10,150과 섞지 않는다. 모든 arm이 같은 sample/order이지만 B1 이후 W/M/z는 자기 chain의 state이므로 end-to-end 비교이지 same-state causal contrast는 아니다.

## 4. 최종 비교: 점수뿐 아니라 변화 문항과 NLL

| Final W10 Qwen | Official | JV | JV−Official |
|---|---:|---:|---:|
| RS | 992/1000 | 997/1000 | +0.50 pp |
| PS | 1887/2000 | 1894/2000 | +0.35 pp |
| strict PS, request의 두 prompt 모두 preference 성공 | 900/1000 | 910/1000 | +1.00 pp |
| NS | 6978/10000 | 7377/10000 | +3.99 pp |
| strict NS, request의 열 neighborhood 모두 preference 성공 | 230/1000 | 284/1000 | +5.40 pp |

P/final_metrics.csv 및 paired_seen_endpoint_metrics.csv:

- NS: Official 실패→JV 성공 1,289, Official 성공→JV 실패 890, 순증 399. 모두 보존 또는 모두 개선된 것이 아니다.
- RS: 개선 7, 악화 2, 순증 5. Rewrite mean margin은 오히려 −0.169303 nat 이동했다. Success의 작은 증가를 모든 request의 confidence 증가와 동일시하면 안 된다.
- PS: 개선 91, 악화 84, 순증 7. Mean margin은 +0.004625 nat로 거의 동일하다. PS의 0.35 pp만으로 큰 generalization 우월성을 주장하기 어렵다.

Neighborhood likelihood 분해:

| NLL, token mean을 prompt 평균한 값 | W0 | Official W10 | JV W10 |
|---|---:|---:|---:|
| target-true mean | 5.640120 | 6.835014 | 6.331188 |
| target-true median | 5.074674 | 6.504759 | 5.909853 |
| target-true p90 | 10.863803 | 12.156931 | 11.543141 |
| target-new mean | 10.498654 | 9.033091 | 9.046259 |
| target-new − target-true mean margin | 4.858534 | 2.198076 | 2.715070 |

정확한 평균 분해: Δnew = +0.0131677293, Δtrue = −0.5038263843, Δmargin = +0.5169941137. 따라서 `−Δtrue/Δmargin = 0.9745302142`. 이는 집계 평균 이동의 분해이지 97.45% 문항 또는 causal contribution을 뜻하지 않는다. True NLL 평균·median·p90가 모두 개선되어 binary threshold crossing만으로 설명되지 않는 likelihood evidence가 있다. 그럼에도 W0 true NLL보다 높고 NS도 W0 84.63% 대비 73.77%로 낮다.

Neighborhood target-true의 teacher-forced all-token-correct prompt도 946→1036/10000으로 증가했다. 이는 secondary evidence이며 canonical NS와 다른 지표다.

## 5. W0 보존과 새 회복은 분리해야 한다

P/neighborhood_transition_summary.csv의 final 동일 10,000 prompt:

| 항목 | Official | JV |
|---|---:|---:|
| W0 성공 prompt | 8463 | 8463 |
| W0 성공→최종 실패 | 1838 | 1390 |
| W0 실패 prompt | 1537 | 1537 |
| W0 실패→최종 성공 | 353 | 304 |

JV는 W0 성공 prompt의 새 실패를 **448개 줄였고**, W0 실패 prompt의 회복은 **49개 적었다**. 순증 399는 주로 원래 성공하던 neighborhood를 덜 잃는 데서 나온다. 새 실패 수의 상대 감소는 448/1838=24.37%다. 단, 두 방법 모두 상당한 W0 손실을 낳으며, 이 한 CounterFact neighborhood inventory가 전체 pretrained knowledge/capability를 대표한다는 보장은 없다.

## 6. 시간과 cohort: 초반에는 locality가 더 나빴다

| Checkpoint의 동일 seen-prefix | Official NS | JV NS | 차이 |
|---|---:|---:|---:|
| W1, first100 requests | 849/1000 | 833/1000 | −1.60 pp |
| W5, first500 requests | 3716/5000 | 3907/5000 | +3.82 pp |
| W10, all1000 requests | 6978/10000 | 7377/10000 | +3.99 pp |

이 행들은 각 checkpoint 내 paired 비교다. 평가 request inventory 자체가 100→500→1000으로 늘어나므로 세 점을 고정 cohort의 시간 추이처럼 읽으면 안 된다. Full seen PS/NS는 1/5/10에만 존재한다.

Current-B100 NS의 JV−Official count는 B1..B10에서 `[-16, −8, +10, +46, +58, +17, +19, +32, +62, +45] /1000`이다. B3 이후 여덟 current batches에서 이득이 있지만 각기 다른 cohort/state이며 반복 독립 trial은 아니다. L8 집중은 B1부터 이미 강했고 B1/B2 NS는 더 낮았다. 따라서 “집중한다면 즉시/항상 locality가 높다”는 단순 설명은 반례를 가진다. History 축적 이후 차이가 생기는 것과 일관되지만 sample/cohort 및 actual W/key 변화도 함께 진행되어 history-cost의 단독 원인으로 귀속할 수 없다.

온라인 own-batch pooling과 최종 retention:

- NS: Official 7457→6978 (−479), JV 7722→7377 (−345), online 대비 final decline 차이 134 prompt.
- PS: Official 1942→1887 (−55), JV 1928→1894 (−34). JV는 online PS가 −14 낮았으나 final PS는 +7 높다.
- RS: Official at-write 998→final992 (후속 실패6), JV 1000→997 (후속 실패3).

이 차이는 같은 fixed prompt inventory의 at-write state에서 W10로 이동한 집계 score 변화다. 개별 문항의 신규 실패·회복 수를 순감소량과 동일시하지 않는다. Rewrite의 실제 후속 실패 분모는 Official6/998, JV3/1000이다. Final first-batch RS는 Official98/100, JV99/100으로 차이가 작다. 따라서 retained edit의 우월성은 NS에서 더 뚜렷하고 RS/PS에서는 소수 사례 수준이다.

## 7. 집중했는데 locality가 왜 좋아질 수 있는가

P/physical_layer_batch_metrics.csv의 실제 FP32 batch-entry→terminal ΔW를 재계산했다. 아래 energy는 Frobenius norm 제곱이며 native work와 다르다.

| B | JV/Official total batch energy | JV L8 energy share | JV/Official L4–7 energy |
|---|---:|---:|---:|
| 1 | 2.662598 | 99.963612% | 0.00114954 |
| 2 | 3.097302 | 99.948325% | 0.00200860 |
| 3 | 1.687264 | 99.977105% | 0.00050649 |
| 4 | 1.256011 | 99.992644% | 0.00010690 |
| 5 | 1.523185 | 99.993082% | 0.00012589 |
| 6 | 1.823019 | 99.995698% | 0.00008951 |
| 7 | 2.878561 | 99.960684% | 0.00145544 |
| 8 | 2.532884 | 99.988275% | 0.00038333 |
| 9 | 2.055835 | 99.997454% | 0.00006034 |
| 10 | 3.135275 | 99.875923% | 0.00540225 |

JV의 L4–7 actual energy는 모든 batch에서 Official의 약 0.0060%–0.5402%다. 반면 전체 energy는 모든 batch에서 1.256–3.135배다. 열 batch의 **batch energy 합**은 Official38,235.449, JV83,843.291이고 L4–7 합은30,704.229 vs36.073이다. 이는 W10−W0 net energy 또는 native-action work가 아니므로 그러한 이름을 붙이지 않는다.

가능한 경쟁 설명:

1. **Early-layer disturbance 회피:** L4–7를 거의 고정하여 downstream representation/key 변화를 줄이고 L8에서 직접 target activation을 맞추는 경로가 이 architecture/sample에서 유리했을 수 있다. 직접 early weight action은 실제로 크게 작다. Historical key/activation drift와 NS의 request-level 결속은 현재 publication에 없으므로 causal claim은 pending이다.
2. **L8-native geometry/regularization의 선택 효과:** 높은 current response 효율과 해당 native cost 아래 L8이 유리했을 수 있다. 이는 다층 분산이 아니라 적절한 concentrated actuator 선택의 가능성이다. 현재 joint-vs-best-single shadow의 Qwen40/40 best single layer는 L8이며, joint 대비 L8-only objective 차이는 작다. Actual L8-only chain에서 거의 같은 결과라면 “adaptive mixing이 만들어낸 이득”은 약해진다.
3. **History 기반 metric이 scale을 보정한 효과:** M이 controller/writer에 들어가는지와 실제 endpoint 이득의 인과는 별개다. History-cost-only shadow는 같은 dictionary/current response를 유지한 비교라 direct cost 효과만 분리한다. Qwen B10 node0 actual-vs-initial-cost native cosine≈0.99998853이므로 강한 history 유도 rotation을 이미 입증한 것은 아니다. M이 direction 생성과 과거 trajectory에 미친 효과는 이 shadow가 제거하지 않는다.
4. **Barrier와 locality의 간접 연결:** native action dissipation은 과거 prompt의 output preservation constraint가 아니다. Native geometry가 locality와 상관될 수는 있으나 NS +3.99 pp를 barrier inequality의 보증처럼 해석하면 안 된다. L8-only 대조가 유용하지만 history 없는 whole-method 대조가 아니므로 그 결과만으로 history의 전체 인과를 확정할 수도 없다.

NS가 좋아졌다는 사실과 모든 metric이 좋아졌다는 결론도 다르다. Rephrase target-new mean NLL은 2.180005→2.283015, p90는6.388785→7.372859, max는17.531527→20.512180으로 나빠졌다. Median은0.604330→0.585057로 조금 좋아졌다. PS +0.35 pp와 악화된 tail이 공존한다. 평균/꼬리 및 response quality를 같이 보고해야 한다.

## 8. “유의미”의 통계적 범위

이 연구의 효과 크기는 명확하다: deterministic한 한 sample/order에서 same-prompt final NS +3.99 pp, mean margin +0.517 nat, W0-success 새 실패 감소448, request-strict NS +5.4 pp. 하지만 **통계적 유의성 또는 여러 order/seed/model로 일반화되는 효과는 아직 확인하지 않았다.**

10,000 neighborhood prompt를 독립 Bernoulli trial로 취급하는 단순 paired McNemar/Wilson p-value는 같은 request당 열 prompt, relation/target 공유, 하나의 sequential trajectory 의존성을 무시한다. 1289/890이라는 paired discordance만으로 request-cluster의 분산을 복원할 수 없다. 이 감사는 독립 prompt p-value를 제시하지 않는다.

Publication sample.lock에는1,000 requests,999 distinct subject-relation group(1개 pair중복),313 distinct target-new hash가 있다. 중복 subject-relation이 적다는 사실이 neighborhood prompt의 중복 또는 relation-level dependence가 없음을 뜻하지 않는다. 원본 prompt-level public CSV가 복구되면 O/JV를 `(case_id,prompt_index,input identity)`로 join하여 case 또는 사전 정의한 관계 cluster 단위 paired bootstrap, effect distribution 및 개선 case 비율을 **analysis-only**로 산출할 수 있다. 추가 모델/evaluator 실행부터 할 필요는 없다. 동일10개 chronological batch를 독립10개 repetition처럼 처리해서도 안 된다.

## 9. 이 자료로 우선 답할 수 있는 질문과 다음 판정

- Qwen locality 이득이 실제로 있는가: **이 한 experiment의 canonical inventory에서는 YES**, 기존 teacher-forcing accuracy와 혼동한 수치가 아니다.
- 단순 전체 감속인가: **총 actual batch Frobenius energy 기준 NO**. Native total cost/trajectory speed에 대한 별도 분석은 필요하다.
- L8 집중이 실패인가: **NO**. 이 setting에서는 high concentration과 더 좋은 NS/유지된 RS가 공존한다.
- L8 집중 또는 barrier 때문에 좋아졌는가: **UNRESOLVED**. Current-state basis/history scale/early-layer write 회피/actual path가 동시에 바뀌었다.
- History가 없는가: evaluator/publication만으로 그런 결론을 내릴 수 없다. Runtime의 M binding/metric/history finalization source audit와 직접 cost shadow를 함께 읽어야 한다.
- Lifelong으로 바로 승격 가능한가: **이 NS 한 수치만으로 불가**. Actual L8-only, rephrase tail, history/normalization mechanism, 비용을 결합한 모델별 판단이 필요하다.
- 분석 후 새 실험을 추가했는가: **0**. 이미 승인된 SH4 실행은 이 감사에서 polling/수정/추가하지 않았다.
