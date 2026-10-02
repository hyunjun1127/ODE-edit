# JLZ A/B 사용자 중단·기존 부분 산출물 게시

최신 권한 `ODEEDIT-GH-ALL-SH-STOP-EXPERIMENTS-S4-PUBLISH-20261002-R1`.
상태 **STOPPED_USER_PARTIAL**. 이 문서는 새 상세 리뷰나 추가 실험이 아닌 기존 scalar 산출물의 최소 검산·게시다.

## 중단 사실

GH가 2026-10-02 03:01:58 UTC에 owner janghj(1025), server4 및 exact source/argv를 확인하여
collector 56962를 먼저, main 56960/56961을 다음에 취소했다(rc0). SH4는 중복 취소하지 않았다.
세 job의 CANCELLED와 빈 사용자 queue를 직접 확인했다. 다른 서버/job을 변경하지 않았다.
등록된 downstream collector도 취소되어 자연 진행하는 후속 job이 없다.

| 실행 | job | 최종 scheduler | 저장 완료 | 미완료 |
|---|---:|---|---|---|
| prep | 56957 | COMPLETED | 공통 준비·baseline 소형 pilot | 없음 |
| pilot A / B | 56958 / 56959 | COMPLETED | 각각 BS4×2 | 없음 |
| main A / B | 56960 / 56961 | CANCELLED by 1025 | 각각 BS100×11, 1,100요청 | B12 진입 후 중단, B12–20 commit 없음 |
| collector | 56962 | CANCELLED by 1025 | 실행0 | 원 full20 보고 미생성 |

각 main의 atomic commit11/ledger11, history55, 기록된 oracle1320회가 일치한다.
총 main22/계획40 commits, history110/계획200. 두 B12의 중단 시점 oracle 횟수는 NOT_RECORDED이며
1320×2를 전체 실행 oracle 횟수로 쓰지 않는다. B12 rollback은 NOT_VERIFIED.
각 완료 batch의 finite `BUDGET_STOP`은 원 정책대로 유지했으며 수렴 PASS가 아니다.
원 runtime terminal.json은 두 main 모두 미생성이다. 외부 중단 receipt가 scientific 완료를 대신하지 않는다.

## 마지막 저장 endpoint W11

R/P는 처음1,100요청 전체, N은 **B11 현재100요청의1,000prompt만**이다.
W11 전체1,100요청 N은 NOT_MEASURED. R/P는 new<true, N은 true<new, tie는 실패다.
TF는 teacher forcing이며 free generation 정확도가 아니다.

| arm | 패널 | preference 성공/분모 | TF strict/분모 | TF token-micro | true NLL | new NLL |
|---|---|---:|---:|---:|---:|---:|
| A | R | 1100/1100 | 1097/1100 | 0.997312 | 12.293320 | 0.018969 |
| B | R | 1100/1100 | 1098/1100 | 0.998208 | 12.198016 | 0.016181 |
| A | P | 1493/2200 | 843/2200 | 0.391129 | 7.123351 | 3.713043 |
| B | P | 1514/2200 | 884/2200 | 0.410394 | 7.154309 | 3.677165 |
| A | N(current) | 882/1000 | 247/1000 | 0.268932 | 4.785483 | 10.235955 |
| B | N(current) | 883/1000 | 250/1000 | 0.271845 | 4.771504 | 10.246833 |

직전 full-N milestone W10의 N은 A8783/10000, B8790/10000이다.
[기존 scalar 집계 CSV](stored-endpoint-metrics.csv)에 W10/W11 및 pilot 마지막 endpoint,
원 true/new NLL·TF token 분모·prompt macro·tail을 보존했다.
기존 reducer 함수로 case/prompt/target/order/finite/분모·state/no-mutation을 확인했으며 새 forward는0이다.
새 paired/bootstrap/기전 분석 캠페인은 수행하지 않았다.
**W20와 정확2k baseline 비교는 NOT_MEASURED**. 과거 baseline2k와 W11을 같은 endpoint/분모로 합치지 않는다.

## 비용·보존·출처

Parent 할당 기준 main A/B 각각24,874 GPU초, 합49,748 GPU초(13.818889 GPUh).
현재 attempt 전체50,702 GPU초, 기존 prep 실패1,116 GPU초를 한 번 더해 연구 누적51,818 GPU초(14.393889 GPUh).
batch/extern 중복 합산0. Allocation은 GPU utilization이 아니다. 프로그램 구간 timer는 중첩 가능하며
[commit CSV](commit-summary.csv)의 seconds를 allocation에 더하지 않는다.
[원 parent accounting](parent-accounting.csv)은 이번 중단의 한정 snapshot이다.

실행 source `2af2af3ba7a4e2e6b5d0ac5c8e53841fcb275b1b`, lock
`920dc4630814126e9300d8dfa7098629e928996b39c536cf736f418fa9a625aa`.
원 attempt: `/data/janghj/ODE-edit/local/jlz-twoarm/20261002-bs100x20-v1/attempt-r2-source-inventory/`.
[raw inventory](raw-artifact-inventory.json)는 저장된 pilot/main scalar 파일의 size/SHA이며 내용은 Git0.
모델·teacher·CP를 재해시하거나 이전 raw 전체를 재분석하지 않았다.
원 source/config/lock/로그/실패/관측은 KEEP. 새 checkpoint0, exact_resume=NOT_AVAILABLE.
기존 [초기 인계](../report-ko.md)의 초기 gate·이전 비용·baseline 표는 역사 그대로 보존한다.
설계와 실행 배경은 [사용자 지시](../../../../../plans/global/2026-10-02-jlz-two-arm-v2/user-dispatch-bs100x20-ko.md)를 따른다.

분석 source는 `audits/servers/server4/2026-10-02-user-stop-all/publish_stored.py`이며 기존
`project/run_scripts/jlz_two_arm/collect.py` 산술만 호출했다. 첫 게시 스크립트에서 observation.sha256
선택필드를 필수로 읽은 KeyError를 CPU에서 수정했다. 원 runtime/raw 변경0, 모델 호출0.
이번 검산은 owner 수행이며 새 독립 red/model 감사가 아니다. 수치 인증 NOT_ESTABLISHED 유지.
Markdown 실제 browser 렌더는 NOT_VERIFIED; 표/CSV와 상대링크는 게시 검사한다.

NO_BROADCAST_NOT_REQUIRED. monitoring_active=false, automatic_resume=false.
추가 사용자 실행 승인 전 모든 신규 submit/retry/resume/과학 작업은 중지한다.
