첨부 분석문은 **BLUE가 L4의 요청·history를 분담하는 대조군이 아니라는 점, L8 추가가 순 locality 개선으로 이어지지 않았다는 점, 유사한 총점이 동일한 손상 문항을 뜻하지 않는다는 점**을 정확히 짚는다. 주요 수치도 보존된 집계와 일치한다. 다만 마지막의 “핵심 병목은 수정 layer 수의 부족이 아니라”는 결론은 수정해야 한다. 현재 자료는 단일층의 공동 보존 capacity가 충분한지 부족한지 구분하지 못한다.

작성: 2026-09-13. 대상: [첨부 분석문](/mnt/raid5/janghj/.codex/attachments/6f7041c0-8f76-4314-b845-4f9cd3bbb75c/pasted-text.txt). 연구 맥락: [baseline mechanism 설계](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-12-baseline-mechanism-first-lifelong-editing-design.md).

이번 검토에서는 로컬 보존 CSV의 수치·전이·산술, 실행 source, 기존 감사의 근거를 대조했다. GPU/model forward, 새 editing, 원격 실행은 하지 않았다. 이전 감사의 원시 8,978개 파일 해시 검증과 12,949,200 prompt-state 재집계를 이번에 다시 수행한 것으로 보고하지 않는다. 원시 파일 전체와 checkpoint tensor는 로컬 보존 package에 들어 있지 않다. [보존 범위](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/README.md:5), [기존 감사 범위](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md:18).

**명칭부터 구분해야 비교가 정확하다.** 아래 native는 BLUE repository의 blue=False, L4–L8 다섯 층을 수정하는 기존 arm이다. BLUE는 blue=True, L4+L8 두 층이다. L4-only부터 L8-only까지는 BLUE 방식의 layer-local target을 사용하는 singleton이다. 다른 진단의 “native proposal N”은 BLUE-L4 endpoint를 가리키므로 다섯 층 native와 다르다.

| AlphaEdit 구성 | 최종 RS % | 최종 PS % | 최종 NS % |
|---|---:|---:|---:|
| native, L4–L8 | 73.430 | 62.885 | 55.285 |
| BLUE, L4+L8 | 98.880 | 95.775 | 63.726 |
| L4-only | 99.390 | 95.680 | 65.348 |
| L5-only | 99.340 | 94.195 | 62.822 |
| L6-only | 96.830 | 85.740 | 58.396 |
| L7-only | 92.400 | 80.970 | 56.277 |
| L8-only | 93.960 | 77.780 | 54.703 |

분모는 RS 10,000, PS 20,000, NS 100,000이며 모두 동일 fixed10k의 최종 실제 W100이다. W100은 100 batches, 10,000 edits 이후다. W0 NS는 89.212%다. [원 통합 보고서](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/diagnostic-report-ko.md:24).

L4-only는 현재 AlphaEdit 설정에서 합리적인 주 개발 baseline이다. 다만 BLUE가 PS에서 0.095%p 앞서므로 모든 지표의 지배 관계는 아니다. L5의 RS도 L4와 0.050%p, 5개 차이에 불과하다. At-write 성공 후 lost는 L4 55개, L5 57개다. 따라서 **L4/L5의 높은 rewrite retention**과 **비슷한 retention에서 L4의 paraphrase/locality 이점**을 다른 질문으로 다뤄야 한다. [전이와 비교](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md:55).

L4−L5 NS 차이는 이미 1k에서 3.150%p이며 최종에는 2.526%p다. Seen-prefix는 평가 구성이 변하므로 손상 속도의 인과 비교는 아니지만, 최종 L4 우위 전체를 “장기 붕괴가 더 느리다”로 설명할 근거도 없다. [L4 곡선](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/diagnostic-report-ko.md:396), [L5 곡선](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/diagnostic-report-ko.md:417).

MEMIT에서는 RS 최고가 L7 84.440%, PS 최고가 L5 75.040%, NS 최고가 L4 57.848%다. AlphaEdit L4의 결과를 편집 방법 전반의 보편적 L4 우위로 확장하지 않는다. [양 계열 주표](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md:43).

**첨부의 수치에는 결론을 바꾸는 오류가 없다.** 아래 BLUE/L4-only는 AlphaEdit 계열이다.

| 첨부의 주장 | 대조 결과와 해석 |
|---|---|
| 100/1k/2k/5k/8k/10k seen-prefix NS | 성공률 일치. 8k 차이의 정확한 값은 1.1975%p로 세 자리 통상 반올림은 1.198%p이며 원문 1.197은 작은 표시 오차 |
| 첫 batch N: 86.7→64.2 / 86.7→66.2% | 일치. 같은 1,000 N이므로 후기 batch 난도만으로 저하를 설명할 수 없음 |
| W0 성공 중 lost: 28,306 / 26,583 | 일치. 분모 89,212, 조건부 손실률 31.729% / 29.798% |
| 최종 NS 차이 1.622%p, 성공 여부 불일치 17.558% | 일치. 7,968 대 9,590의 양방향 차이. 100,000 N 평가 row 기준이며 고유 fact 수는 아님 |
| 평균 L4 batch norm 10.3723 / 10.3814 | 일치. Actual stored-weight batch-net Frobenius norm |
| L4 norm 합 1,037.2295 / 1,038.1361 | 일치. Batch norm 합이며 최종 누적 weight 차이 norm은 아님 |
| BLUE L8 평균 norm share 약 26.2% | 일치. Batch별 비율 평균 26.197%. Norm 합끼리의 비율 27.5865%와 구분 |
| true/new NLL 분해와 평균 margin 손실 | 일치. 전체 N의 paired 평균이며 손실 문항만의 평균이나 인과 기여율은 아님 |

근거: [누적 지표](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/AlphaEdit-cumulative-metrics.csv:31), [고정 cohort](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/cohort-retention.csv:3238), [문항 전이](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/paired-final-transitions.csv:67), [norm](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/layer-summary.csv:4), [W0 pairing](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/w0-all14-paired-transitions.csv:7).

**쓰기 전 손상은 L4-only의 세 entry에서 확인된 관측이다.** 첨부에 출처 없이 언급된 Early/Middle/Late 표를 복원하면 다음과 같다.

| 다음 B100의 Current neighborhood | W0 NS % | entry NS % | native proposal N 후 NS % | entry→N %p |
|---|---:|---:|---:|---:|
| Early: 1,000 edits 후 | 90.0 | 83.2 | 81.3 | −1.9 |
| Middle: 5,000 edits 후 | 88.8 | 72.4 | 71.1 | −1.3 |
| Late: 9,000 edits 후 | 87.5 | 63.4 | 62.6 | −0.8 |

각 행은 서로 다른 다음 100 requests의 1,000 N을 그 행의 W0/entry/endpoint에서 비교한다. 동일 Current cohort를 세 시점에 반복한 표가 아니다. N은 기존 BLUE-L4 proposal이다. “후기 N이 도착 전에 상당히 손상돼 있다”는 해석은 맞지만, 세 batch의 차이를 전체 100 batches의 평균 손상·나이에 따른 손상률 감소·BLUE 두 층의 시간적 손상 분해로 확대할 수 없다. [원 진단 표와 정의](/mnt/raid5/janghj/ODE-edit/local/reviews/l4-two-memory-v2-independent-audit-2026-09-11-v1/source-package/diagnostic-report-ko.md:20).

**BLUE의 코드 설명은 ‘전체 residual을 목표로 쓴다’로 정교화해야 한다.** BLUE는 현재 모델에서 L4 local z를 계산해 L4를 수정한 다음 변경된 모델에서 L8 local z를 계산한다. resid=targets는 남은 layer 수로 residual을 나누지 않는다는 뜻이다. 요청을 5,000개씩 분담하거나 동일 residual을 절반씩 나누는 구조가 아니다. [봉인 실행 source](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/runtime-source/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source/AlphaEdit/AlphaEdit_main.py:58).

그러나 “순차적 완성”, “전체 residual 실현”은 목표와 실제 결과를 혼동할 수 있다. 실제 solve는
\[
A=P(KK^\top+M)+\lambda I,\qquad A\Delta W^\top=PKR^\top
\]
이며 BLUE/singleton의 \(\lambda=1\)이다. Projector·history·regularization을 포함한 fitting으로 \(\Delta WK=R\)이라는 equality를 보장하지 않는다. z는 해당 layer의 block output에서 정의되며, L4와 L8의 z도 서로 다른 target이다. “각 layer의 local residual 전체를 RHS에 넣는다”가 정확하다. [residual과 solve](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/runtime-source/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source/AlphaEdit/AlphaEdit_main.py:115).

보존 요소가 전혀 없는 구현도 아니다. Compute-z에는 generic KL prompt, delta regularizer, norm clamp가 있다. L8의 KL reference는 해당 z 최적화 시작점, 즉 L4 write 이후 상태다. 따라서 “보존 정규화는 있지만 W0/batch-entry neighborhood를 복구하는 직접 목표는 없다”가 정확하다. 별도 worktree의 해당 source는 Server4 import manifest와 SHA를 대조했다. [KL reference](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/blue-source/AlphaEdit/compute_z.py:141), [전체 loss](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/blue-source/AlphaEdit/compute_z.py:159), [import 근거](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/manifest.json:21).

모든 선택 layer write가 끝난 뒤 각 layer의 전체 batch key를 다시 계산해 M에 한 번 더한다는 첨부 설명도 맞다. 다만 M은 원래 지식 covariance C0나 projector P와 다르며, 과거 edit의 기능적 equality도 아니다. 해당 key는 context별 subject-last 입력을 평균한 표현이므로 모든 context/token 방향의 보호와 같지 않다. 모든 요청이 들어간다는 사실만으로 유효 rank, 지식 보호량, 제약 충돌 정도가 정해지지 않는다. [history append](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/runtime-source/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source/AlphaEdit/AlphaEdit_main.py:237), [key 평균](/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/blue-source/AlphaEdit/compute_ks.py:19).

**L4 key/history 공통성은 강한 구조적 근거지만 기능적 손상의 동일성을 뜻하지 않는다.** 고정 token IDs·positions·mask·contexts와 수정 parameter 범위를 유지하면 L4 down-projection 입력은 그 weight나 뒤의 L8 down-projection 수정에 영향을 받지 않는다. 같은 초기 상태와 계산 경로에서는 BLUE/L4-only의 L4 K/M이 같아야 한다. 실제 교차-arm tensor 동일성은 별도 확인 대상이다. Residual R은 현재 L4와 downstream L8 상태를 통해 달라지므로 같은 solve geometry에서도 다른 update가 가능하다.

이번 검토에서 기존 checkpoint inventory를 교차 비교하니 **B1의 L4 weight SHA는 BLUE/L4-only가 같고, B5부터 이후 모든 저장 checkpoint에서는 다르다.** 첨부의 “trajectory가 달라질 수 있다”보다 강한 기록상 근거다. 최초 차이가 B2–B5 중 언제 생겼는지, 차이 크기·cosine·기능적 영향은 해시로 알 수 없다. 이번에 tensor를 다시 로드한 결과는 아니다. [BLUE B1/B5](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/checkpoint-tensors.csv:26), [L4-only B1/B5](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/checkpoint-tensors.csv:62).

**Singleton key 불변성은 L4만의 특성이 아니다.** L5-only부터 L8-only까지 자기 down-projection 하나만 수정한다면 같은 논리가 성립한다. BLUE L8의 key는 앞선 L4 수정으로 변할 수 있지만 L8-only의 key는 그렇지 않다. 따라서 key 안정성만으로 singleton 성능 순위를 설명할 수 없다. [정확한 적용 범위](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-12-baseline-mechanism-first-lifelong-editing-design.md:104).

**Native 비교에서는 여러 정책이 동시에 달라진다.** Native AlphaEdit은 batch-entry에서 L8 z를 요청당 한 번 계산하고, 각 layer에서 현재 L8 output과의 residual을 남은 layer 수 5/4/3/2/1로 나눈다. BLUE는 각 layer의 현재 모델에서 local z를 계산하며 divisor=1이다. L2도 native 10, BLUE/singleton 1이다. Native→BLUE 개선을 layer 수·local z·key drift 어느 하나의 효과로 귀속할 수 없다. [실제 설정표](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/diagnostic-report-ko.md:93).

Native AlphaEdit의 at-write RS는 99.90%였지만 최종 73.43%, at-write 성공 중 lost는 2,649개다. 낮은 최종 성능은 최초 write 실패만으로 설명되지 않는다. 다층의 상위 historical key가 후속 하위 write 때문에 달라진다는 가설은 가능하지만, L2·residual·target 효과와 분리해 검증한 결과는 아니다. [전이 근거](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md:63).

**문항 전이와 NLL 분해는 첨부의 가장 좋은 부분이다.**

| 최종 N 성공 여부 | L4 성공 | L4 실패 |
|---|---:|---:|
| BLUE 성공 | 55,758 | 7,968 |
| BLUE 실패 | 9,590 | 26,684 |

이 표는 두 전체 lifelong 정책의 결과다. 7,968개를 L8의 직접 복구로 부를 수 없고, 26,684개를 공통으로 삭제된 원지식으로 부를 수도 없다. 전자는 L4 trajectory 차이를 포함하고 후자는 W0 실패 문항을 포함할 수 있다. 첨부는 이 경계를 이미 적절히 명시했다.

| W0→W100, 전체 100,000 N 평균 | true NLL 변화 | new NLL 변화 | 안전 margin 감소 |
|---|---:|---:|---:|
| BLUE | +1.8995 | −2.5378 | 4.4372 |
| L4-only | +1.4558 | −2.9031 | 4.3589 |

단위는 nats/token이다. 출력 손상의 두 성분을 구분하지만 각 문항의 동일 비율 변화나 L4/L8 인과 기여율을 뜻하지 않는다. 첨부의 안전 margin \(m=\mathrm{NLL}_{new}-\mathrm{NLL}_{true}\)은 기존 CSV의 margin=true−new와 부호가 반대다. 파생 열에 이를 명시해야 한다. [원 평가식](/mnt/raid5/janghj/ODE-edit/local/reviews/blue-native-lifelong-independent-audit-2026-09-11-v1/source-package/diagnostic-report-ko.md:54).

L4의 W0-success→failure 26,583개 중 4,242개, 15.958%는 true NLL이 증가하지 않았는데 competing-new가 강해져 실패했다. True NLL 보호만으로 NS 보존을 대체할 수 없다. 반대로 두 후보의 선호인 NS만으로 기존 정답의 절대 출력 능력 전체를 대표할 수도 없다. [손실 문항 분해](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md:103).

**‘높은 edit 성능’과 ‘기존 지식 손상’의 범위를 명시해야 한다.** L4 RS 99.390%는 두 target의 평균-token NLL 비교다. Rewrite TF strict는 95.290%, paraphrase TF strict는 66.810%다. NS 65.348%도 전체 pretrained knowledge의 보존율은 아니다. Neighborhood true TF strict는 W0 21.567%에서 8.210%로 하락해 절대 지표의 악화도 있지만, TF strict는 자유 생성 정확도와 다르다. [보조 지표](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md:159).

정확히 겹치는 N/edit prompt 2,432개를 제외해도 L4 NS는 89.521→65.731%, −23.791%p다. 명시적 중복만으로 큰 손상을 설명할 수 없다. 의미적 alias·paraphrase 충돌까지 제거한 결과는 아니며, 과거 target의 정당한 overwrite도 active/superseded로 분리해야 한다. [중복 민감도](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md:117).

**Capacity의 수학은 충분조건과 존재성을 분리해야 한다.** 같은 외부 제약·평가 목적 아래 L8=0이 허용되면 \(\mathcal F_4\subseteq\mathcal F_{4,8}\)은 맞다. 그러나 native/BLUE의 서로 다른 L2·목표 정책을 동일 최적화 문제로 취급할 수는 없다. 이 포함관계는 수정 공간의 가능성에 대한 논증이며 BLUE의 탐색 능력을 보장하지 않는다.

고정된 true/new sequence의 모든 관련 token key를 \(K_N\)으로 두고 매 update가 \(\Delta W_tK_N=0\)이면 \((\sum_t\Delta W_t)K_N=0\)이라는 첨부의 정리도 맞다. L4-only에서 다른 parameter까지 고정돼 있으면 해당 sequence의 후속 계산도 보존된다.

그러나 **이 조건을 만족하면서 필요한 edit도 수행하는 비영 update가 존재한다는 뜻은 아니다.** \(K_N\)이 입력 공간을 모두 span하면 \(\Delta WK_N=0\)은 \(\Delta W=0\)을 요구한다. Edit residual이 비영이면 정확한 동시 실현은 불가능하다. 반대로 exact hidden 보존이 불가능해도 NS margin을 양수로 유지할 수 있으므로, 이 강한 제약의 불가능성이 NS 보존의 불가능성을 곧바로 뜻하지도 않는다.

AlphaEdit 논문도 실무상 정확히 0인 covariance eigenvalue 공간 대신 작은 eigenvalue 공간을 선택한다고 설명한다. 이 근사·표본 범위와 실제 평가 sequence 전체 key는 구분해야 한다. 이 부분의 첨부 인용은 적절하다. [AlphaEdit §3.2](https://arxiv.org/html/2410.02355v3#S3.SS2).

기존 two-memory v2는 선택 key의 구조적 보호를 출력 보존과 혼동하지 말아야 한다는 사례다. Middle의 Base mapping energy는 약 \(6.63\times10^{-10}\)까지 줄었지만 Base KL은 native 0.059035→0.198794로 증가했다. 보호 위치는 Base 전체 input token의 약 11.5–11.6%였다. 선택 위치에서 작은 \(\Delta WK\)가 전체 출력 보존을 보장하지 않았다. 이를 native lifelong 손상의 확정 원인·prefix 기여율·모든 대안 writer의 실패로 일반화하지 않는다. [구조 proxy와 실제 반응](/mnt/raid5/janghj/ODE-edit/experiment-reports/global/l4-two-memory-v2-independent-audit-2026-09-11-v1/independent-review-ko.md:64).

**첨부의 결론에서 바꿀 표현은 다음과 같다.**

| 원문 표현 | 권장 표현 |
|---|---|
| “순차적 완성”, “전체 residual write” | “각 layer의 local residual 전체를 RHS로 사용하는 순차적 fitting” |
| “L4에서 이미 발생하는 간섭을 거의 그대로 둔 채” | “L4 요청·history 범위와 평균 update norm을 거의 유지한 채. 기능적 간섭의 동일성은 미측정” |
| “L4의 z와 weight trajectory도 달라질 수 있다” | “L4 weight는 B1에서 같고 B5 이후 저장 지점에서 다르다. 방향·기능 차이 크기는 미측정” |
| “두 번째 layer가 … 누적 간섭을 줄이는 방식으로 사용되지 않았다” | “명시적 분담·W0/entry N 복구 목표가 없으며 최종 NS의 순 개선은 관측되지 않았다” |
| “핵심 병목은 수정 layer 수의 부족이 아니라” | “현재 정책의 preservation 병목이 확인됐으며 writer 선택과 가용 자유도의 기여는 아직 미분리” |
| “L8이 … 손상 보상에 활용하지 못한 결과” | “L8을 추가한 전체 정책이 순 preservation 개선을 만들지 못한 결과. 개별 회복·추가 손상은 미분해” |
| “4-cell이 필요 없다” | “현상 진단과 연구 동기를 위해 대안 writer를 포함한 4-cell을 선행조건으로 요구할 필요가 없다” |

대안 writer 없이도 현재 문제의 크기와 BLUE의 명시적 분담 부재는 충분히 설명할 수 있다. 이후 특정 writer의 유효성, 단일층의 충분한 공동 보존 여력, target/writer 변경의 개별 효과를 주장한다면 해당 질문에 맞는 개입 근거가 필요하다. 그것이 반드시 현재 시점의 특정 4-cell이어야 하는 것은 아니다.

**실험 방향은 L4를 출발점으로 삼되 첫 산출물을 원인 보고서로 잡는 것이 타당하다.** 다음 순서는 신규 writer 개발을 요구하지 않는다.

| 우선순위 | 기존 자산으로 할 확인 | 구분할 질문 |
|---|---|---|
| 1 | L4/L5/L6–L8의 같은 case at-write→final, true/new NLL, cohort 연결 | 높은 R retention의 공통성, P/N 차이, 처음부터 margin이 큰 효과 |
| 1 | 저장 L4의 누적 \(S_t=W_t-W_0\) cosine/차이 norm; 가능한 K/M slice 비교 | 같은 크기지만 다른 변화인가, 공통 geometry가 실제로 일치하는가 |
| 2 | 소수 BLUE batch의 entry→after-L4→after-L8 동일 panel 평가 | L8이 L4가 손상시킨 같은 문항을 회복하는가, 다른 문항을 손상시키는가 |
| 3 | 필요할 때 같은 entry의 native update 크기 조절 및 기존 설정의 한 요소 변경 | 강도·목표·정규화·history 중 최소 변경으로 개선되는 부분이 있는가 |

문항 연결은 기존 raw가 있는 환경에서 수행해야 한다. 집계 CSV만으로 결합분포를 복원할 수 없다. At-write margin matching은 관찰 분석이며 layer 선택 이후의 변수를 조건화하므로 인과 기여율을 산출하는 절차로 쓰지 않는다.

세 상태 진단은 N 안전 margin \(m=\mathrm{NLL}_{new}-\mathrm{NLL}_{true}\)으로
\[
d_4=m_{\rm entry}-m_{\rm after\,L4},\qquad
d_8=m_{\rm after\,L4}-m_{\rm after\,L8}
\]
를 기록한다. \(d_4+d_8=m_{\rm entry}-m_{\rm final}\)은 정확한 분해다. 다만 **그 BLUE entry와 L4→L8 순서에 조건부인 이번 batch 효과**다. 과거 L8 update가 현재 L4 residual에 미친 영향이나 전체 lifelong L8 정책의 효과를 모두 분리한 것은 아니다.

평균과 함께 같은 문항의 성공→실패→성공, 성공→성공→실패 등 전이를 집계하고 margin 악화와 threshold crossing을 구분한다. Current N, 고정 historical N, 아직 자기 batch가 도착하지 않은 N 또는 독립 general panel을 동일 identity로 평가한다. True/new NLL과 R/P도 함께 기록해 보존 개선과 edit 감소의 교환을 확인한다. 진단 N을 runtime update 선택에 이용하면 정보 조건이 달라지므로 공식 평가 N을 방법의 선택·학습 신호로 쓰지 않는다.

12개 저장 checkpoint만으로 임의 batch 중간 상태를 복원할 수 있다고 가정하지 않는다. 실제 인접 batch의 전후 selected weights가 있으면 after-L4를 구성할 수 있지만, 1k와 2k처럼 여러 batch 간격의 weight를 섞은 상태는 한 batch의 after-L4가 아니다. Resume에는 selected weights뿐 아니라 M/P, contexts, RNG, 다음 request index, target-cache identity가 필요하다. [기존 복원 설계](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-12-baseline-mechanism-first-lifelong-editing-design.md:176).

새 writer의 방향은 진단 결과로 정한다. 작은 update로 현재·과거 R/P를 유지하면서 N 손상을 줄이면 강도·정규화부터 검토할 이유가 있다. 크기 조절이 즉시 under-edit를 만들면 방향·관측 위치·출력 민감도·목표의 조합을 조사한다. L8이 같은 N을 회복하지만 다른 N에서 손상을 만들면 그 교환 구조를 확인한다. 소형 개입 실패 하나로 단일층 capacity 부족을 선언하지 않는다.

모든 현재 결과는 하나의 fixed10k order에 대한 기술통계다. Request당 10개 N과 중복 fact의 의존성을 유지하며 100,000 row를 독립 반복으로 취급하지 않는다. 이미 fixed10k로 layer·후보를 선택했으므로 동일 문항의 order 변경은 순서 강건성 검사이며 blind held-out 검증이 아니다. [정보·평가 계약](/mnt/raid5/janghj/ODE-edit/project/proposals/2026-09-12-baseline-mechanism-first-lifelong-editing-design.md:126).

첨부의 최종 연구 주장은 다음 수준이 근거와 일치한다.

> 이 fixed10k/B100 설정에서 AlphaEdit의 L4-only와 L5-only는 높은 rewrite retention을 보였지만 기존 neighborhood 응답은 크게 손상됐다. BLUE의 L8 추가는 L4의 전체 요청·history 처리와 평균 update norm을 줄이지 않았으며 최종 neighborhood 선호의 순 개선도 만들지 못했다. 두 정책은 서로 다른 L4 weight trajectory와 문항별 손상 양상을 보인다. 따라서 현재 결과는 단순 layer 추가만으로 해소되지 않은 preservation 병목을 보여준다. 다음 연구는 L4를 주 baseline으로 삼아 현재·과거 edit 성능을 유지하면서 누적 off-target 영향을 줄일 수 있는지 확인하고, 그 결과에 따라 필요한 writer 변경과 추가 layer의 역할을 정하는 것이다.
