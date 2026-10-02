# JLZ v7 main A — 500-edit 간단 완료 리뷰

2026-10-03 KST, 사용자 요청 `odeedit_jlz_v7_s4_main_A`만 검토했다. **57513 COMPLETED/0:0**, cold W0에서 BS100×5와 W5 누적 평가 완료. 5 commits / 125 candidates / 120 Adam / 층별 history append 총25회를 저장 기록과 대조했다. B·collector는 이번 리뷰 범위 밖이며 두 arm 전체 완료로 확대하지 않는다.

## W5: 동일 고정 첫 500개 baseline 비교

| 방법 | 기존 job | RS /500 | PS /1000 | NS /5000 |
|---|---|---:|---:|---:|
| JLZ v7 A | 57513 | 500 (100.00%) | 992 (99.20%) | 3598 (71.96%) |
| AlphaEdit | 42657 | 493 (98.60%) | 891 (89.10%) | 4110 (82.20%) |
| AlphaEdit-BLUE | 39283_1 | 500 (100.00%) | 962 (96.20%) | 4190 (83.80%) |
| CAKE | 48101 | 489 (97.80%) | 804 (80.40%) | 4251 (85.02%) |
| MEMIT-H | 54007 | 495 (99.00%) | 873 (87.30%) | 4314 (86.28%) |

v7 A에서 baseline을 뺀 PS 차이는 각각 +10.10/+3.00/+18.80/+11.90%p, NS 차이는 −10.24/−11.84/−13.06/−14.32%p다. 인과적 원인이나 방법 우열 판정은 하지 않는다.

RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL인 prompt 비율이다. 동률은 실패. 위 baseline은 **500개 편집 직후 W5**이며 W100에서 첫500만 추출한 수치가 아니다. 기존 [baseline W5 집계](../../cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2/baseline-cumulative.csv), [CAKE W5 집계](../../cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2/seen-prefix.csv), [MEMIT-H W5 집계](../../../server3/memit-history-fixed10k-20260928-v1/completion-review-r1/all-seen-metrics.csv)를 SHA 결속해 재사용했다. baseline raw를 이번에 다시 전수 검산하지 않았으며 새 평가·전송·baseline 실행은 없다.

모델은 Llama3-8B-Instruct 동일 revision/고정 stream이지만 동일 실행조건 비교는 아니다. AlphaEdit는 blue=false/L4–L8/L2=10, BLUE는 blue=true/**L4+L8/L2=1**이다. CAKE는 L4–L8, clamp .5/decay .4 등 별도 설정이다. v7와 MEMIT-H도 writer/최적화가 다르다. 기존 baseline은 Transformers 4.44.2/full head/주로 MB16/cuDNN TF32 on, v7는 4.57.1/selected-position head/MB2/cuDNN TF32 off다. 같은 상태의 평가기 parity를 새로 증명하지 않았다. 비교별 per-case lost/gained는 이번 baseline 집계에서 계산하지 않았다.

## v7 A 검산 및 짧은 수치 요약

W5 원시 6500행의 case/kind/prompt 순서, 중복·누락, finite NLL, true/new token 분모, TF strict, state hash를 검산하고 독립 CPU reducer가 기존 summary와 정확히 일치했다. W1–4는 각각 current100, W5만 누적500이다. W5 active500/500이며 current100은 동일 W5 raw의 부분집합이다.

| W5 지표 | TF token-micro | TF prompt-macro | TF strict | true NLL 평균 | new NLL 평균 |
|---|---:|---:|---:|---:|---:|
| R | 505/507 (99.606%) | 99.600% | 498/500 (99.600%) | 16.166430 | 0.029912 |
| P | 831/1014 (81.953%) | 81.950% | 817/1000 (81.700%) | 13.171630 | 0.746679 |
| N | 724/5070 (14.280%) | 13.780% | 654/5000 (13.080%) | 6.136429 | 8.824745 |

TF desired target는 R/P=new, N=true. TF strict는 teacher forcing의 모든 target token argmax 일치이며 자유 생성 정확도가 아니다. 높은 NLL preference와 TF 정확도를 혼동하지 않는다.

- 편집 직후 문항별 결과 합산→W5: R 500→500, lost0/gained0; P 991→992, lost1/gained2; N 3869→3598, lost397/gained126. 편집 직후 합산은 다섯 서로 다른 시점이며 단일 모델 결과가 아니다.
- 재사용 W0→W5 N: 4392→3598/5000, lost902/gained108, 순−794(−15.88%p). W0 raw의 prompt identity/target token count 및 기존 입력 결속 receipt를 재사용했다. W0 원행에는 token-ID hash가 없어 새 동일-runtime token-ID/수치 parity 검증으로 표현하지 않는다.
- 전 batch LR .1/warmup0; candidate25 gradient=null/미측정, 추가 Adam0. Physical pulse5/10/15/20, current/past partition 전체성 확인. 메모리 참조수 0/16/16/16/16, 최종 seen500/resident128.
- 5개 W/H 연속성·exact commit receipt·observer W/H/memory 비변이 결속. 저장된 SPD 상대 residual 최대 6.08945e−15, builder→terminal key 차 최대 8.58307e−6, 기존 허용 범위 안이다. terminal은 forward-only이므로 solve 내부 grad=false가 정상이며 top-level wiring flag를 측정 gradient로 읽지 않는다.
- noB6/noCP, 원 runtime·raw 불변. W/H 텐서가 영속 저장되지 않았으므로 이 CPU 리뷰는 해시/receipt 대조이며 실제 행렬 재실행·새 모델 검증이 아니다. 별도 독립 red reviewer는 사용하지 않았고 owner의 독립 reducer 검산이다.

## 비용·출처·재현

57513 단일 accounting snapshot: 2026-10-02 23:07:13→2026-10-03 01:38:15 KST, 9062초 × 1GPU = **2.51722 allocated GPUh**. Allocation은 utilization이 아니다. runner wall9056.61초, peak GPU48,636,370,944B(45.30GiB), 자체 peak RSS34,718,448KiB(33.11GiB). batch step와 observer 시간은 [batch-costs.csv](batch-costs.csv); 내부 timer는 포함관계가 있어 단순 추가합산하지 않는다. 이 비용은 prep·B·collector·과거 baseline 비용을 포함하지 않는다. 기존 양 prep 총821 allocated GPU초는 이전 준비 receipt 재사용이며 A main 비용에 중복 배분하지 않았다.

- 실제 실행 source: `fe6758512eba37595e99244d150d115268553f01`, tree `d35b488ca158fe04aef15ac7b4ddebeb69d0a88b`.
- lock SHA `97a03ae24674f0e7fbe115e96f0b0d930166aaa4fa80a5af29c30206b1e9f09e`.
- config 파일 SHA `5138907f84cd00cc0397e8e67b326b130460eb76296a3e255bf29333de562588`; commit 내부 config `b71a9b7e850e6e0baacf7b35501f1cef69ec5d4150c57ca31ef189b690b65b76`는 동일 JSON의 canonical digest로 서로 다른 hash convention이다.
- 실행 archive SHA `882ea0a6399b053b4cf7ef86b24894648fc2b4be2a7959ee09677f6b54164df7`. 이번 분석 source/publication은 실행source와 별도다.
- 원본: `/data/janghj/ODE-edit/local/jlz-causal-writer-v7/20261002-v1/attempt-r1/main-A/`.
- [집계 및 체크](summary.json), [baseline/TF/NLL 비교](baselines-W5.csv), [시점별 지표](metrics.csv), [paired 전이](paired.csv), [cohort](cohort-retention.csv), [마지막 후보 층별 통계](terminal-layer-stats.csv), [원자료 경로·크기·SHA](artifact-index.json), [게시 manifest](package-manifest.json).

```bash
python3 -B -m unittest project.run_scripts.jlz_causal_writer.test_review_main_a
python3 -B project/run_scripts/jlz_causal_writer/review_main_a.py \
  --attempt /data/janghj/ODE-edit/local/jlz-causal-writer-v7/20261002-v1/attempt-r1 \
  --repo "$PWD" --output /EXACT/NEW/REVIEW/OUTPUT
```

분석은 Python 표준 라이브러리만 사용하며 scheduler/model/torch를 호출하지 않는다. 소형 CPU 검산코드와 보고/CSV/manifest만 게시한다. Raw/tensor/prompt/fullstdout은 local KEEP/Git0. `NO_BROADCAST_NOT_REQUIRED`, `monitoring_active=false`, `automatic_resume=false`. B 상태·종료 대기 및 타 task 변경0.
