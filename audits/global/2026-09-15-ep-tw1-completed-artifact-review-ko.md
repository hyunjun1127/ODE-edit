EP-TW-1 완료 실험의 독립 산출물 리뷰

작성: 2026-09-15. 대상: Server4 job 47962, `gate-skip-r1`, W0/M0에서 B100×10으로 진행한 신규 1,000개 편집. 실행 source는 `6d317bdb2660d7e9919bc3a9fb878564e9729e37`이다. 검토한 완료 보고서와 부록 52개 파일은 `c18f961f0ee1074f5e89bfc9f19750cb19901a52`의 내용과 모두 동일하다. 실제 로컬 snapshot manifest의 HEAD는 이후 `9ae2ecc4…`이며, 이 차이 때문에 EP 산출물이 바뀌지 않았음을 별도로 확인했다.

**이번 설정에서 EP-TW-1의 최종 보존 우위는 관측되지 않았다.** 가장 가까운 L4-only baseline과 비교하면 RS는 같고, PS는 1문항, NS는 16문항 적다. 추가 계산도 발생했다. 다만 이것만으로 executable target space에서의 보존 보정 자체가 무효라고 결론 내리기에는, 실제 실행된 보정이 native write의 **0.00675–0.02355%**로 매우 작았다. 저장 tensor를 재계산하면 그 직접적인 이유는 모든 batch에서 적용된 `alpha_cap=1`이다.

동시에 국소적으로 확인된 양성 결과도 있다. 실제 평가한 corrected 후보 30개 모두 자기 batch의 native preview보다 C4 평균 KL이 낮았으며, 그 감소량은 저장 gradient의 1차 예측과 매우 가깝다. 따라서 이번 결과는 **작은 실행 보정이 현재 편집 품질 조건 아래에서 generic KL을 줄일 수 있다는 작동 증거**다. 이를 lifelong 보존 개선, ODE 경로의 필요성, barrier 우회 증거로 확대할 수는 없다.

아래 수치는 완료 보고서를 옮겨 적은 것만이 아니다. [원 완료 보고서](/mnt/raid5/janghj/ODE-edit/local/reviews/ep-tw1-completed-2026-09-15/source/experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/gate-skip-r1/completed-review-v1/diagnostic-report-ko.md), 실제 실행 source, Server4에 남은 과학 산출물 JSON 86개와 tensor 파일 20개, baseline 관련 파일 17개를 대조했다. 원시 NLL/ID를 이용한 집계·선택 확인 262개와 파일·tensor·연결 확인 238개가 통과했다. 이 수는 **저장 자료의 감사 항목 수**이며, gradient 검증이나 모델 실험의 PASS 수가 아니다. 새로운 모델 forward, GPU 작업 제출, 모델 편집은 수행하지 않았다. [검산 결과](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/review-checks.json)

**실행 범위와 구현을 먼저 구분해야 한다.** 이번 run은 W50에서 시작한 이전 REFIT4 suffix와 달리 편집 전 W0와 M0에서 시작했다. 고정 first-1000 순서의 B001–B010을 하나의 모델·history 경로로 처리했다. 각 batch에서 fresh native target 100개를 계산하고 L4 native preview를 만든 다음, 그 preview 주변에서 한 번의 보정 방향을 계산했다. 이후 `RAW, C1, C05, C025`를 비교해 하나를 commit했다. 이는 2-stage REFIT4도, target refresh를 포함한 여러 번의 editing episode도 아니다.

실제 controller의 generic reference는 고정 original teacher의 C4 S64다. 문서당 128개 고정 scoring position의 full-vocabulary KL `KL(p0 || pW)`를 사용한다. 입력 길이는 BOS를 포함해 257 tokens다. Current E는 현재 100개 요청의 canonical desired-target NLL이며, 요청별 token 평균을 다시 요청 평균으로 합친다. 공식 paraphrase/neighborhood 정답, accepted-old 평가, Dev128은 이번 selector의 목적·제약에 들어가지 않았다. 과거 accepted-edit ledger는 기록·평가용으로 존재하지만, **이번 EP-TW-1에 old-edit retention gradient나 개별 ceiling은 없다.** 이전 BG-TW 전체 사양의 모든 기능을 시험한 것으로 해석하면 안 된다.

후보의 수학적 형태는 다음과 같다.

\[
V(C)=V_p+CA_t,\qquad
g_E=\nabla_C E(V(C))\big|_{C=0},\quad
g_D=\nabla_C D(V(C))\big|_{C=0}.
\]

선택 조건은 실제 materialized 후보에서 `E(candidate) ≤ E(RAW)`이고, RAW가 달성한 현재 strict 성공 ID를 모두 유지하는 것이다. 조건을 만족하는 유한 후보 중 D가 가장 작은 후보를 고른다. RAW가 선택된 B005/B010도 native edit를 정상 commit했다. Parent 무편집 상태로 되돌려 locality를 얻거나 요청을 분모에서 제외한 경우가 아니다. Inner history append는 0회, 최종 L4 history finalization은 batch당 1회, 총 10회다. Native target은 총 1,000회, Adam update는 24,000회, loss 평가는 25,000회였다.

이 규약에는 N4의 C4 drift를 먼저 측정하는 단계, 고정 KL budget b, log barrier, old-loss penalty가 없다. 사용자 지시에 따라 수치 진단을 생략한 실행이다. 원시 상태도 `INITIAL_EXECUTION_OBSERVED_WITH_VALIDATION_SKIPPED`이며, 이를 G0 PASS로 바꾸지 않았다. 생략을 무단 변경으로 취급하거나 관측한 모든 성능을 무효화할 이유는 없다. 다만 생략한 검증까지 완료된 것으로 말하지 않는다. 실행된 [policy](/mnt/raid5/janghj/ODE-edit/local/reviews/ep-tw1-completed-2026-09-15/execution-source/project/run_scripts/bg_tw_reference/ep_tw/policy.py:191)와 [runner](/mnt/raid5/janghj/ODE-edit/local/reviews/ep-tw1-completed-2026-09-15/execution-source/project/run_scripts/bg_tw_reference/ep_tw/runner.py:178)를 직접 확인했다.

**최종 성능은 요청 원분모로 읽어야 한다.** 아래 baseline도 각각 first-1000의 W10 결과이며 full10k endpoint가 아니다. 모든 행을 실제 per-item NLL에서 다시 집계했다. R/P/N의 분모는 각각 1,000/2,000/10,000이다.

| 정책 | RS | PS | NS | 성공 수 R / P / N |
|---|---:|---:|---:|---:|
| EP-TW-1 | 99.80% | 97.10% | 80.56% | 998 / 1,942 / 8,056 |
| AlphaEdit-BLUE L4-only, 가장 가까운 N4 | 99.80% | 97.15% | 80.72% | 998 / 1,943 / 8,072 |
| AlphaEdit-BLUE, L4+L8 | 99.70% | 96.95% | 80.57% | 997 / 1,939 / 8,057 |
| MEMIT-BLUE | 99.50% | 94.95% | 85.07% | 995 / 1,899 / 8,507 |
| AlphaEdit | 98.90% | 92.70% | 75.10% | 989 / 1,854 / 7,510 |
| MEMIT | 94.90% | 89.15% | 70.18% | 949 / 1,783 / 7,018 |

따라서 plain AlphaEdit/MEMIT보다 수치가 높다는 사실을 이번 보정의 효과로 바로 귀속하면 안 된다. 출발 writer 설정과 layer 구성이 다르다. 가까운 N4와 비교한 추가 효과가 우선이며, 이 비교에서 관측된 변화는 **RS 0.00%p, PS −0.05%p, NS −0.16%p**다. MEMIT-BLUE는 더 낮은 PS와 더 높은 NS를 보인다. 이번 실험으로 모든 baseline에 대한 품질–보존 우위를 얻었다고 할 수 없다. [baseline 재집계](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/baseline-recomputed.csv)

N4와의 대응 문항 전이는 R lost1/gained1, P lost13/gained12, N lost109/gained93이다. 평균이 비슷해도 동일한 요청들이 성공한 것은 아니다. PS desired NLL의 paired 악화량 p95는 0.87436 nats, NS true NLL의 paired 악화량 p95는 0.66595 nats다. 반대로 개선된 문항도 있다. 순차 trajectory를 통째로 비교한 값이며 이 tail 전체를 한 번의 작은 보정의 직접 손상으로 읽지 않는다. [문항 전이와 NLL tail](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/baseline-paired-recomputed.csv)

평가 지표의 의미도 분리해야 한다. EP-TW-1의 최종 canonical TF-strict는 996/1,000이고, paraphrase TF-strict는 1,341/2,000=67.05%다. 두 paraphrase 모두 strict인 요청은 503/1,000이다. 따라서 PS 97.10%는 두 후보 NLL ranking 성공률이지 자유 생성 정확도 97.10%가 아니다. N4의 paraphrase strict는 1,347개이므로 EP가 6개 적다. NS의 true NLL 평균도 EP 5.302484, N4 5.302156으로 사실상 비슷하며 EP의 명확한 true-likelihood 개선이 관측된 것은 아니다.

**baseline의 신뢰 범위는 보고서보다 한 단계 더 확인했다.** 두 실행의 seed 설정은 20260907/20260915로 다르고 wrapper/source version도 완전히 같지는 않다. 반면 모델 revision, sample 순서, parsed contexts, 핵심 native hparams, FP32/eager와 주요 runtime 설정은 맞는다. 여기에 실제 N4 B001의 `native-targets.pt`를 추가로 읽어 비교했다.

| 첫 batch의 직접 비교 | 독립 확인 결과 |
|---|---|
| native target 100개, 각각 4,096차원 | 100개 전부 tensor 원소가 정확히 같음 |
| L4 입력 key, 100×14,336 | tensor hash 같음 |
| 보정 전 R 100 / P 200 / N 1,000의 new·true NLL | 총 1,300문항 모두 정확히 같음 |

그러므로 seed 문자열 차이만으로 N4 비교 전체를 부적절하다고 배제하는 것은 지나치다. 최초 native proposal의 평가까지는 실제 일치가 확인됐다. 다만 B002 이후에는 EP가 자기 수정된 모델에서 target을 다시 계산하므로 같은 상태의 직접 대조가 아니다. 전 구간의 모든 tensor/kernel/state parity 또는 같은 seed의 독립 반복을 입증한 것도 아니다. **가용 baseline으로 최종 추가 이득이 보이지 않는다는 판단은 가능하되, −16 NS를 일반적인 인과 효과나 유의한 열세로 확정하지 않는 것**이 맞다. [B001 tensor·문항 일치 근거](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/n4-b001-initial-comparison.json)

**이번 결과를 해석하는 핵심은 보정 크기다.** 실제 코드의 보정 방향은

\[
d=-g_D+\frac{\min(\langle g_E,g_D\rangle,0)}{\|g_E\|^2}g_E,
\qquad
\alpha=\min\!\left(1,\frac{0.25\|V_p-W_{entry}\|_F}{\|dA_t\|_F+10^{-12}}\right).
\]

이후 native target ball에 투영하고, 필요하면 write trust bound를 다시 적용한다. 저장된 gE/gD/A/target/anchor/radius에서 이 과정을 독립적으로 재계산했을 때 **최종 C tensor가 10개 batch 모두 정확히 재현됐다.** 즉 관측된 작은 C는 단순 보고서 표기 실수로 설명되지 않는다.

| Batch | 선택 | cap이 없을 때 α, ball 적용 전 | 실제 α | 선택 보정 norm / native norm | 자기 RAW 대비 S64 KL 감소 |
|---|---|---:|---:|---:|---:|
| B001 | C1 | 2,904.19 | 1 | 0.008608% | 0.25639% |
| B002 | C05 | 1,851.60 | 1 | 0.006751% | 0.13246% |
| B003 | C1 | 1,524.85 | 1 | 0.016395% | 0.23740% |
| B004 | C05 | 1,140.11 | 1 | 0.010964% | 0.15146% |
| B005 | RAW | 1,010.48 | 1 | 0 | 0 |
| B006 | C1 | 1,164.59 | 1 | 0.021466% | 0.21560% |
| B007 | C05 | 1,099.03 | 1 | 0.011374% | 0.10612% |
| B008 | C1 | 1,061.59 | 1 | 0.023549% | 0.19557% |
| B009 | C1 | 1,084.54 | 1 | 0.023051% | 0.17882% |
| B010 | RAW | 943.76 | 1 | 0 | 0 |

**설정의 0.25는 native norm의 25%라는 상한이다. 실제 선택된 보정이 25%였다는 뜻이 아니다.** Nonzero 보정은 그 상한의 0.0270–0.0942%만 사용했다. 전체 10개의 selected correction norm 합은 0.0099292, 같은 경로에서 native batch delta norm 합은 80.6563이다. Norm 합은 경로 길이의 요약이며 cumulative net norm이나 출력 손상의 인과 기여율이 아니다. [batch별 tensor 재계산](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/tensor-mechanism.csv)

`alpha_cap=1`은 실행 전에 선언된 기술 config이므로 숨겨진 변경이라고 부를 근거는 없다. 문제는 실제 결과에서 그것이 예외적인 overflow 방어가 아니라 **10/10 batch의 주요 step 크기 결정 규칙**으로 작동했다는 점이다. 최소화 목표 D에 양의 상수 c를 곱해도 원래 제약 최적화 문제의 해는 같다. 그러나 cap이 활성화된 구현에서는 gD와 d가 c배가 되어 실제 step이 달라질 수 있다. 따라서 loss의 단위와 reduction에 민감한 실질적 learning rate가 된다. 이를 중립적인 수치 보호 장치로만 설명하면 안 된다.

반대로 cap을 곧바로 없애면 성능이 좋아진다는 결론도 없다. 저장 tensor에서 **loss forward 없이 기하만** 계산해 보니, uncapped α 후 동일 ball/trust를 적용한 보정은 native norm의 24.31–24.63%이고 87–100개 요청이 target ball에 걸린다. 이는 이번 실제 보정보다 약 세 자릿수 이상 큰 전혀 다른 크기의 후보다. 이 후보들의 E, strict, KL, 최종 성능은 평가하지 않았다. 따라서 이 계산은 효과 예측이 아니라, 무조건적인 cap 제거가 얼마나 큰 변경인지를 보여준다.

**Projection이 9회 작동했다는 횟수만으로 강한 edit–preservation 충돌을 주장할 수 없다.** 실제 gE와 gD의 cosine 범위는 −0.009752에서 +0.002665로 거의 직교한다. Projection이 활성화된 9개 batch에서 방향을 바꾼 크기 `||d+gD||/||gD||`는 0.2419–0.9752%다. 비활성 B006은 0이다.

Projection 후 남는 D의 1차 감소량은 unconstrained −gD의 99.99049–99.99941%이며, B006은 100%다. 현재 설정에서는 보존 gradient의 방향을 크게 바꾼 것이 아니라 거의 유지했다. 작은 부호 차이가 엄격한 E screen에서는 중요할 수 있으므로 projection을 불필요하다고 단정할 수는 없다. 다만 이 결과로 복잡한 route control이나 큰 충돌 회피가 입증됐다고 말하기는 어렵다.

선택된 보정과 native write의 cosine은 −0.06030에서 −0.02716이다. 즉 보정은 native update의 단순 scalar 축소와도 다르다. 작은 크기의 거의 비평행 성분이다. **새 방향이 있다는 사실과 그 방향이 endpoint preservation에 유용하다는 사실은 별개**다.

Target ball projection은 총 111 request-events, 추가 write trust retraction은 0회였다. Ball·FP32 연산 이후 C는 αd에서 norm 기준 0.0479–0.6878% 달라졌다. `<gE,C>`는 8/10 batch에서 양수였지만 절대량은 작다. B001은 ball에 걸린 요청이 0개인데도 양수였다. 따라서 양수 8회를 모두 target ball과의 충돌 또는 loss-landscape barrier의 증거로 해석할 수 없다.

**국소 C4 개선은 실제 저장 loss에서 확인되지만, 개선 범위를 좁게 표현해야 한다.** Corrected 후보 30개 전부에서 `D(candidate) < D(RAW)`였다. 그 실제 ΔD를 후보별 `β〈gD,C〉`로 나눈 비율은 0.99433–1.00245다. 관측한 후보 메뉴에서는 감소량이 gradient의 1차 예측과 매우 잘 맞는다. 이는 chosen direction에서의 국소 일관성이지, 생략된 모든 finite difference·full-VJP 검증을 대신하는 보증은 아니다.

채택된 corrected endpoint는 8개이며, 자기 RAW 대비 평균 KL 감소율은 0.1061–0.2564%다. 서로 다른 8개 parent에서의 KL 차이를 더한 0.00013944는 terminal N4 대비 KL 개선량이 아니다. 다른 branch의 미래 상태가 없으므로 그런 비교로 바꿀 수 없다.

채택된 nonzero 보정의 문서별 결과 8×64=512 observations 중 KL이 낮아진 경우는 510개, 높아진 경우는 2개다. 이는 대부분 문서에서 작동한 신호다. 그러나 같은 관측에서 natural continuation NLL이 올라간 경우는 175개이고, 평균 NLL 변화는 −0.00000867 nats다. Original distribution을 가까이 보존하는 것과 자연문장의 실제 다음 token NLL을 낮추는 것은 같은 목적이 아니다. 같은 고정 64개 문서를 여러 batch에서 센 것이므로 512개 독립 문서처럼 해석하지 않는다. [문서별 원시 차이](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/generic-document-deltas.csv)

누적 generic drift 자체도 계속됐다. S64 selected KL은 B001 0.00157413에서 B010 0.02013259로 커졌다. Controller에 사용하지 않은 Dev128의 KL은 W5 0.01210122에서 W10 0.02631042로 증가했다. 이 값은 손상이 모두 막혔음을 보여주지 않는다. 동시에 N4의 동일 C4 평가가 없으므로 N4보다 drift가 빠른지 느린지도 알 수 없다. S64와 Dev128은 서로 다른 문서 집합이므로 값 차이만으로 과적합을 확정하지 않는다.

**Current 품질 screen은 구현대로 작동했지만, 경계 근처의 판별은 매우 작다.** 30개의 corrected 후보 중 13개가 E 조건 때문에 탈락했고, strict ID 손실 때문에 탈락한 후보는 없었다. 선택은 C1 5회, C05 3회, RAW 2회, C025 0회다. Nonfinite 후보나 중복 materialization 때문에 생긴 탈락도 없었다.

E 초과로 탈락한 후보의 평균 E 증가량은 **4.59×10⁻¹¹에서 1.76×10⁻⁷ nats**였다. 작은 scale이 항상 더 잘 통과한 것도 아니다. B001은 C1과 C025가 통과하고 C05는 탈락한다. B003은 C1/C05가 통과하고 C025가 탈락하며, B004는 C05가 통과하고 C025가 탈락한다. 곡률, 항목 간 상쇄, FP32 materialization·loss 계산의 작은 차이가 모두 가능한 해석이다. 동일 후보 반복 forward의 jitter/ULP 측정은 이번 실행에서 생략됐으므로 어느 하나로 원인을 확정하지 않는다. 이 비단조성 자체도 barrier bypass의 증거가 아니다.

관측된 method와 reporting evaluator의 현재 NLL row는 RAW/selected 20개 패널에서 모두 같았다. 따라서 이번 저장 결과에는 두 evaluator가 서로 다른 E를 보고 선택했다는 근거가 없다. 하지만 이 일치는 동일 모델을 반복 실행했을 때의 jitter 측정은 아니다.

평균 품질 보존과 요청별 품질 보존도 다르다. 채택된 8개 nonzero 후보에서 현재 canonical NLL이 상승한 요청은 batch당 33–53개다. 최대 개별 상승은 4.69014×10⁻⁵ nats였다. B002에서는 48개 요청의 NLL이 상승하고 16개가 하락했는데, 평균 E는 1.10619×10⁻⁸ 낮아져 C05가 통과했다. 개별 NLL 절대 변화의 평균에 대한 순평균 변화 비율은 0.70%에 불과했다. 조건을 속인 것이 아니라 **선언한 평균 제약이 허용하는 상쇄**다. [40개 후보 재집계](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/candidate-recomputed.csv)

더 직접적인 R/P/N 평가에서는 각 batch RAW→selected의 ranking·strict 성공 bit 변화가 current 30개 패널과 accepted-old 27개 패널 모두 0이었다. NLL은 변했지만 성공 여부는 바뀌지 않았다. 따라서 이번 작은 보정이 즉시 현재 paraphrase나 과거 edit의 성공 수를 복구했다는 근거는 없다. 그렇다고 훗날의 trajectory가 완전히 같다는 뜻도 아니다. [current 비교](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/current-raw-selected.csv), [accepted-old 비교](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/old-raw-selected.csv)

**1,000개 순차 실행에서 확인된 retention은 별도의 결과다.** 각 요청의 자기 batch 직후와 W10에서 같은 문항을 연결하면 다음과 같다.

| 지표 | 자기 at-write 성공 | W10 성공 | 성공→실패 | 실패→성공 | 순변화 |
|---|---:|---:|---:|---:|---:|
| R, 1,000문항 | 1,000 | 998 | 2 | 0 | −2 |
| P, 2,000문항 | 1,936 | 1,942 | 8 | 14 | +6 |
| N, 10,000문항 | 8,351 | 8,056 | 420 | 125 | −295 |

같은 N 문항들에서 at-write→W10 true NLL 악화량 p95는 2.14993 nats, p99는 5.95542 nats다. P new NLL 악화량 p95는 0.34699, p99는 0.89779다. 평균 P 성공이 조금 늘었다고 모든 과거 편집이 유지된 것은 아니다.

첫 500개 요청만 W5→W10으로 고정해도 N은 4,202/5,000에서 4,011/5,000으로 감소한다. Lost256/gained65, 순손실191개다. 이 집합은 기준점과 평가 시점이 공통이어서 서로 다른 at-write 모델을 묶는 문제를 피한다. 이때 N true NLL 악화량 p95는 2.48491, p99는 6.96704 nats다. **후속 edit 동안의 보존 손실은 이번 방법에서도 남는다.** 다만 동일 cohort의 N4 retention 비교 없이 이것을 N4보다 개선 또는 악화했다고 단정하지 않는다. [cohort 전이·tail](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/cohort-recomputed.csv)

각 batch entry의 current N 성공을 묶으면 8,615개이고, 자기 write 직후 8,351개다. 해당 write 전후 lost294/gained30으로 순264개 감소한 다음 후속 batch에서 순295개가 더 줄었다. Entry pool은 서로 다른 W에서 측정했으므로 이를 W0의 동일 모델 평가로 부르면 안 된다.

Acceptance coverage는 at-write 기준 1,000/1,000이었다. 최종 ledger에서는 exact subject/relation 규약상 999개 ACTIVE, 1개 SUPERSEDED다. 최종 canonical strict 실패 4개 중 1개가 SUPERSEDED, **3개가 ACTIVE**다. R ranking 실패 2개는 모두 ACTIVE다. 따라서 모든 최종 실패를 의도된 overwrite로 설명할 수 없다. Active-only 점수도 부록에 남겼지만 주표의 분모는 요청 1,000개를 유지했다. [상태별 집계와 실패 ID](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/metrics-recomputed.json)

**계산량은 결과 대비 상당하며, 주 비용은 후보 검사다.** 아래는 현재 run 안에서 겹치지 않게 묶은 주요 계측 component다. Native 내부 z/key/readout/RHS solve 시간은 native fitting에 포함되므로 다시 더하지 않는다. Generic F/B 및 teacher read도 generic gradient 전체 시간 안에 포함된다.

| Component | 10 batches 합 | B100당 평균 |
|---|---:|---:|
| Native fitting | 2,956.77초 | 295.68초 |
| 최종 history 처리 | 141.90초 | 14.19초 |
| 별도 writer map A 구성 | 3.31초 | 0.33초 |
| Current gradient | 10.17초 | 1.02초 |
| S64 gradient, teacher read 포함 | 169.44초 | 16.94초 |
| 후보 screen과 관련 처리 | 743.84초 | 74.38초 |
| 위 online component 소계 | 4,025.42초 | 402.54초 |

추가로 분리 가능한 map+gradient+screen은 총 **926.76초, B100당 92.68초**다. 이번 run 내부 native fitting+history 3,098.67초에 대한 비율은 **29.91%**다. 이것은 내부 component 장부 비교이며 별도의 동일 조건 N4 run과 직접 측정한 전체 wall overhead는 아니다. 과거 W50 suffix N4의 304.89초/B100이나 REFIT4의 372.75초/B100과 나누어 공정한 속도비를 만들지 않는다.

추가 component 중 screen이 **80.26%**다. Generic gradient의 순수 F/B는 123.46초, teacher read는 45.43초로 계측됐고 둘은 generic 전체에 포함된다. 따라서 첫 비용 개선 대상으로 반복 후보 검사를 보는 근거가 있다. 원 native RHS solve 10회 외에 A map 구성용 solve도 10회 있다. 보고서의 native solve counter가 10이라고 해서 run 전체 선형 solve가 10회였다고 해석하면 안 된다.

실제 allocation은 **7,694 GPU초=2.13722 GPUh**, program elapsed는 7,688.08초다. Selected evaluation만 1,262.06초이며, entry/raw observer·저장·복구·기타 overhead가 전부 별도 항목으로 분리돼 있지는 않다. 그러므로 4,025.42초 소계를 cold end-to-end 시간이나 완전히 계측한 순수 배포 시간이라고 부르지 않는다. 기존 실패 두 건 473+69초와 재사용 teacher 준비 98초를 연구 장부에 포함하면 8,334 GPU초=2.315 GPUh지만, 이를 이번 새 allocation에 중복 가산하지 않는다.

S64 backward는 640 document-passes, 81,920 scored tokens였다. Corrected 후보 검사만 S64 forward 1,920 document-passes, 245,760 scored tokens를 썼다. Current gradient·probe와 full prefix 연산은 추가다. 이 비용을 native target optimizer의 Adam step 한두 개와 같은 것으로 세지 않는다.

저장된 후보 loss만으로 제한적인 비용 반사실도 확인했다. 이번 메뉴에서는 C1→C05→C025 순으로 검사하고 최초 E-feasible 후보에서 멈춰도 동일한 10개 선택이 나왔다. 이 경우 corrected probe는 30회에서 **17회**가 된다. 이는 13회, 43.33% 감소한 probe 개수이며 실측 wall 절감률은 아니다. 앞으로 더 큰 보정에서 D의 단조성이 유지된다는 보장도 없으므로, 새 정책으로 쓰면 finite-menu min-D와는 별도로 선언해야 한다. [저장 메뉴의 선택 재생](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/review-checks.json)

**산출물의 현재 보존 상태는 완료 당시 기록과 다르다.** 보고서 inventory에는 과학 파일 116개, 13,407,776,774 bytes가 있다. 현재 지정 경로에서 확인되는 것은 JSON 86개와 tensor 20개, 합계 2,836,903,572 bytes다. 보고서에 있는 B001–B010의 `checkpoint.pt` 10개는 현재 경로에 없다. 삭제 주체·시각·이유는 이번 조사에서 확인하지 않았으며 추정하지 않는다.

남은 tensor 20개의 파일 hash/size는 보고서 inventory와 일치했다. 각 batch의 `native-targets-map.pt`에는 native proposal, target, anchor/radius, A가 있고 `route.pt`에는 gE/gD/C/A가 있다. 이들로 실제 FP32 `RAW, C1, C05, C025` weight를 재구성해 **40/40 candidate hash**, **10/10 selected commit hash**를 확인했다. W0의 L4 weight 하나도 기존 safetensors에서 CPU로 읽어 시작 hash를 확인했으므로 B001부터 연속적인 L4 weight 연결을 독립 확인했다. 보고서에서 비어 있던 B001의 native/보정 norm도 이 방식으로 보완했다.

하지만 이것은 전체 checkpoint 복구와 같지 않다. M4/history tensor, 전체 RNG, ledger와 optimizer/cache를 포함한 resume 상태를 현재 checkpoint에서 직접 읽은 것은 아니다. History와 상태 연결은 저장 JSON 및 실행 source의 검사 기록까지 확인했다. 원본 checkpoint 전체를 이번에 독립 재검증했다고 쓰면 안 된다. Teacher cache 전체 값과 모든 다른 layer tensor도 재독출하지 않았다. [원격 파일 상태와 SHA 근거](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/artifact-provenance.json), [CPU tensor 확인 원기록](/mnt/raid5/janghj/ODE-edit/audits/global/2026-09-15-ep-tw1-completed-artifact-review/tensor-inspection.jsonl)

**보고서의 주요 성능·선택·비용 산술에서 결론을 뒤집는 집계 오류는 찾지 못했다.** 보완해야 할 것은 주로 의미와 현재 증거 범위다. `alpha_cap_active=10`에서 더 나아가 원래 α가 944–2,904였다는 점, projection 활성 횟수에 비해 실제 방향 변화가 1% 미만이라는 점, B001 native target·NLL parity, 평균 E의 항목별 상쇄, 현재 checkpoint 부재를 함께 읽어야 한다. 작은 ΔE를 수치 잡음이라고 확정하거나 111회의 ball hit를 강한 barrier 충돌이라고 부르는 것도 근거가 부족하다.

**후속 연구의 우선순위는 이 operating point를 확장해도 되는지를 판단하는 것이다.** 이번 설정의 그대로인 EP-TW-1을 full10k나 많은 downstream panel로 확대하는 것은 현재의 작은 변화·추가 비용을 반복 측정할 가능성이 높다. Audit128/MMLU68 완료를 다음 구현의 선행조건으로 둘 필요는 없다. 그렇다고 그 평가가 같을 것이라고 이미 확인된 사실로 쓰지는 않는다. 최종 일반능력 주장은 독립 평가를 열었을 때 판단한다.

첫 수정은 **loss 단위에 종속된 α cap을 실제 executable write 크기로 관리하는 것**이 적절하다. 예를 들면 `||CA||/||Δ_native||`로 시작 크기와 수치 상한을 분리하고, 작은 정규화 시작값에서 기존 실제 E·strict screen을 유지한다. 현재 관측된 0.007–0.024%보다 큰 첫 시험값으로 0.1%를 둘 수 있으나, 이것은 검증된 최적값이 아니라 공개된 optimizer 설정이다. 단번에 25%로 올릴 근거는 없다. 고정 KL 허용한도를 없애는 것과 최적화 step 규칙·연산 상한까지 없애는 것은 다른 문제다. N4 calibration이나 새로운 damage budget을 다시 들일 필요는 없다.

보다 큰 보정에서 실제 E 증가 때문에 계속 RAW로 돌아간다면, 그것은 다음 구체적인 설계 질문이 된다. Native endpoint의 작은 E를 유지하는 실제 feasible correction이 있는지, 그리고 하나의 tangent 방향만으로는 부족해 current-quality restoration이 필요한지다. 그 경우 보존 방향 이동 후 같은 writer family 안에서 current E를 회복하는 제한된 두 번째 보정을 검토할 수 있다. 이는 이번 결과가 입증한 사실이 아니라 다음 가설이다. 원래 조건을 몰래 완화하거나 모든 rejection을 noise로 취급해서는 안 된다.

기전 비교의 최소 단위는 **W0부터 1,000개, B100×10의 독립 sequential chain**을 유지하는 편이 맞다. 한 batch의 수치·연산 점검은 구현 확인용이며 method 성공 판정은 여기서 하지 않는다. 제안 후보의 크기 규약을 고정한 다음 동일 순서의 native 대조와 비교하고, 실제 보존 변화가 관측될 때만 projection 유무, quality restoration 유무를 좁혀 비교한다. 다섯 baseline은 최종 비교 표에 유지하되 모든 후보 수정 때 전부 재실행할 필요는 없다. 기존 REFIT4의 W50 suffix 수치를 W0 comparator로 재사용하지 않는다.

현재 데이터로는 −gD와 projected d의 차이가 매우 작으므로, projection ablation을 넓게 늘리는 것보다 크기 문제를 먼저 해결하는 편이 정보가 많다. 같은 시기에 후보 검사 조기 종료 같은 비용 정책은 소수 메뉴로 비교할 수 있다. 다만 두 변경을 합친 결과만 보고 어느 쪽이 효과를 냈는지 주장하지 않도록 크기와 검사 비용을 각각 기록한다. 새로운 보정 크기의 선형 예측과 actual loss도 실행 중에 기록할 수 있으며, 이번에 사용자가 생략한 대규모 진단을 다시 실행 선행조건으로 복원할 필요는 없다.

이후에도 평가 우선순위는 current R/P, at-write→terminal retention, active/superseded, N의 true/new NLL과 tail, 고정 original KL, 비용이다. S64에서 local ΔD가 낮은 것만으로 보존 method를 채택하지 않는다. 더 큰 실행 보정이 PS나 current E를 해치지 않으면서 terminal N/old retention을 개선할 때 다음 범위를 넓힐 근거가 생긴다. Current E가 계속 실패한다면 그 실패를 드러내는 것도 이 실험의 유효한 결과다.

**논문에서 지금 허용되는 문장은 제한적이다.** “Native writer의 실행 가능한 residual 공간에서 고정 original-output KL gradient를 계산하고, native current-quality 조건 아래 작은 보정 후보를 선택할 수 있었다”는 문장은 근거가 있다. “이 1,000개 순차 실행에서 native 대비 locality와 비용을 개선했다”, “projection의 필요성을 입증했다”, “barrier를 우회했다”, “multistep/ODE가 필요하다”는 문장은 아직 근거가 없다. 이번 EP-TW-1은 큰 경로 변화의 시험이 아니라 작은 국소 보정의 실행 결과로 위치시키는 것이 정확하다.

이 리뷰에서는 새 감사 문서와 재계산 부록만 작성했다. 기존 실행 결과·방법 계약·원격 상태를 수정하거나 후속 실험을 제출하지 않았다.
