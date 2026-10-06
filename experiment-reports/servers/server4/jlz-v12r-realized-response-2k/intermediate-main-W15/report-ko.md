# v12-R MAIN 중간 결과: W15 / 1,500 edits

MAIN job `59262`의 확정 완료 범위는 B1–B15, 총 1,500 edits다. B16 후보 로그 쓰기에서 `OSError: [Errno 28] No space left on device`가 발생하여 종료했다. W20 결과는 없다.

## 누적 preference 결과

| 시점 | 요청 수 | RS | PS | NS | 조화평균 |
|---|---:|---:|---:|---:|---:|
| W0_FIRST500 | 500 | 7.000% (35/500) | 11.200% (112/1000) | 87.840% (4392/5000) | 12.319% |
| W5_ALL_SEEN | 500 | 100.000% (500/500) | 96.200% (962/1000) | 85.080% (4254/5000) | 93.317% |
| W0_FIRST1000 | 1000 | 7.100% (71/1000) | 11.350% (227/2000) | 88.200% (8820/10000) | 12.485% |
| W10_ALL_SEEN | 1000 | 99.700% (997/1000) | 94.550% (1891/2000) | 77.150% (7715/10000) | 89.370% |
| W0_FIRST1500 | 1500 | 7.867% (118/1500) | 11.300% (339/3000) | 88.507% (13276/15000) | 13.221% |
| W15_ALL_SEEN | 1500 | 99.333% (1490/1500) | 90.033% (2701/3000) | 69.607% (10441/15000) | 84.411% |

RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL이며 동률은 실패다. W0는 각 endpoint와 같은 first-N 부분집합으로 표시했다. 전체 W0 2,000개와 current pre/post 및 first100/500 집계는 `metrics.csv`와 `metric-details.json`에 있다.

## W15 teacher-forced 정확도

| 종류 | strict | token micro | prompt macro | desired NLL |
|---|---:|---:|---:|---:|
| R | 97.333% (1460/1500) | 97.372% | 97.367% | 0.145580 |
| P | 64.233% (1927/3000) | 64.619% | 64.533% | 2.211072 |
| N | 15.753% (2363/15000) | 17.019% | 16.470% | 6.028454 |

R/P의 desired target은 new, N은 true다. 자유 생성 정확도가 아니다. True-minus-new와 new-minus-true margin을 각각 이름으로 구분해 보존했다.

## 검산·보존·비용

- 원 NLL/TF 행의 case/prompt/token identity·순서·유한값·분모를 독립 CPU reducer로 재집계했으며 저장 aggregate와 일치했다.
- 15 fit/15 commit/14 W-H-RNG-context-ledger join/75 history append. 후보 375회, Adam update 360회, request update 35975회. Terminal 추가 backward 0.
- 각 writer의 동일 평가 weight copy, no-resolve/no-double-add, rewrite-only history 1회와 다음 entry 연결을 검산했다. B16 entry는 B15 상태와 일치하지만 B16 성공 commit은 없다.
- 원 qualification job59261은 MAIN4요청·3고정후보 TECHNICAL_READY. 본 검산은 CPU raw 재집계이며 새 GPU/모델 forward/평가 0이다.
- MAIN parent 할당은 13,369 GPU초(3.7136 GPUh)이며 실패한 B16 시간도 포함한다. 원 qualification 121 GPU초는 별도다. Batch/extern step 중복 합산은 하지 않았다.
- Fit/build/subject/pullback/optimizer/observer 비용은 `stage-cost.csv`와 `summary.json`에 있다. Inclusive fit/writer/batch 시간과 내부 항목을 다시 합산하지 않는다.
- Mean/canonical/rewrite/KL 실현률·방향·cosine·오차의 행 평균은 `realization.csv`, 층별 Q/update norm은 `layer-cost.csv`에 있다. Undefined ratio는 null/count로 보존했다.
- Current pre→post 및 At-write/W0→W15 paired lost/gained/retention은 `retention.json`에 있다. 대조군·새 baseline 비교는 이번 게시 범위에 없다.
- ENOSPC가 오류 receipt 쓰기도 막았다. 0byte rollback/terminal `.tmp`는 완료 또는 복원 증거로 쓰지 않았다. 공간 소모 주체는 NOT_IDENTIFIED다.
- noCP, exact resume NOT_AVAILABLE. 원 source/raw/log를 보존했으며 Git에는 MAIN 중간 보고와 소형 집계만 게시한다. 재실행 코드는 이번 게시에서 제외했다.

## 실행과 산출물 결속

- Task/nonce: `jlz-v12r-realized-response-2k` / `USER-GH-SH4-JLZ-V12R-20261006`.
- 실행 source: `635798ba276957312ec1aceda686ad563c906957`; config SHA `718a6b1428000e26d726e7094b170d715b1c3596e8821e4a30b6ba751a0b8d1a`.
- 원 artifact: `/data/janghj/ODE-edit/local/jlz-v12r-realized-response-2k/attempt`. CPU 검산 원본과 row-file SHA inventory: `/data/janghj/ODE-edit/local/jlz-v12r-realized-response-2k/review-main-prefix-20261006-verified`.
- 모델: Meta-Llama-3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, L4–L8, seed20261002, FP32 model/FP64 geometry, BS100.
- 설계: `plans/global/2026-10-06-jlz-v12r-realized-response/`. 실행·뒤의 중간결과 게시 commit은 구분한다.
- NO_BROADCAST_NOT_REQUIRED: 원 raw는 동일 S4 local KEEP, Git 소형 보고/집계 공유.
- 별도 red 검토 결과는 `audits/servers/server4/jlz-v12r-realized-response-2k/intermediate-main-review.json`에 기록한다.
