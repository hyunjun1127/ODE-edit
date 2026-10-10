# 완료 원자료 / CF FLU-CON eval-only / cap3

승인 nonce: `USER-GH-S1-S2-COMPLETED-TABLE-S1-FLUCON-CAP3-20261010-R1`.
accepted turn: `01a1236d-9293-73b0-aff5-7e2789e84602` (GH dispatch receipt와 대조).
SH1 session `01a04939-f93a-7b50-bca0-65438eab2062`, devbox, 원 root `/mnt/raid5/janghj/ODE-edit`, origin `hyunjun1127/ODE-edit` 확인.
별도 `codex/server1-completed-flucon-cap3-20261010` WT에서 수행. README는 GH sole writer이며 직접 수정하지 않았다.

## cap / 보존

root의 ignored `servers/local/gpu-caps.tsv` server1 행만4→3 수정. node/memory/패턴/다른 서버 행은 불변.
fresh snapshot 2026-10-10 01:30:05 UTC: allocation1, DAG폭1, author CF62529 RUNNING → zsRE62530 PENDING.
기존 source/archive와 job을 변경하지 않았고 pending resource edge 조정도 불필요했다. 새 DAG폭3 검산.
새로운 CP 삭제/전송/fit/W0/별도 GPU qualification은 없다. 취소된62262/62263은 그대로 보존.

## 실제 평가 등록

실행 source `37094d92d70d0499304da9f7b82ab59340c57232`, official tree `aa1dc1a64e563ef46ad3d3c11e9c9a0c8b514d91`.
구현 commit `b3db3249`. 이 보고서 게시 commit은 실행 source와 별도다.

| 실제 job | 대상 CP 원 job | resource dependency | release 직후 상태 |
|---|---|---|---|
| official-s1-flucon-eval-gptj-memit_fe_history-cap3 (62581) | 61927 | 없음 | PENDING |
| official-s1-flucon-eval-llama3-memit_fe_history-cap3 (62582) | 61928 | 없음 | PENDING |
| official-s1-flucon-eval-qwen25-memit_fe_history-cap3 (62583) | corrected 62061 | afterany62581 | PENDING |
| official-s1-flucon-eval-collector (62584) | 위3개, GPU0 | afterany62581/62582/62583 | PENDING |

정확 source/config/CP/input/reference/owner/fullargv/resources/dependency/held script 검산 후 전량 release했다.
GPU별1GPU/8CPU/65536MiB/48h ceiling, collector GPU0/8CPU/24576MiB/4h. cap3 내 빈 두 lane을 사용하고 세 번째 평가는 같은 평가 task 뒤에 직렬 연결했다.
author 전체 chain을 기다리는 추가 group barrier는 없다. 기존 세 native 평가62259/60/61은 완료 raw를 재사용하고 중복 제출하지 않는다.
author CF62529는 W20 없음, author zsRE62530은 generation 비대상. 역사적 remote MEMIT/Alpha/BLUE CP는 local/current-cohort 평가 후보로 승격하지 않았다.
과거 history 평가 취소를 취소취소/requeue한 것이 아니다. 최신 지시가 명시한 FE_HISTORY 범위와 유효한 최종 endpoint를 검산하여 별도 immutable eval-only attempt를 등록했다.
잘못된 Qwen context61975 및 실패61929는 제외했다. corrected62061의 원20commit/순서/최종 CP full SHA를 결속했다.

기존 CAKE-compatible generation/reference/seed/definition을 유지한다. 한 생성 결과로 Flu/Con을 함께 계산하며 원 bits/cosine은 보존한다.
기존 approved CF eval-only tracking protocol instruction을 그대로 사용하고, 최신 실행 nonce는 config의 registration_authority 및 해당 hash/receipt로 결속했다. 공통 logger/schema를 복제하지 않았다.
실제 W&B/온라인 readback/생성 완료는 아직 관측하지 않았으며 제출을 완료 점수로 표시하지 않는다.

## CPU 검산 / 표

`table-rows.json`과 CSV에 정확 raw/terminal/config SHA 및 기존 재사용 audit를 결속한다.
zsRE 6개는 frozen public-query/token proof와 전체2K의 저장 predicted/target 정오값으로 request-macro E/G/loc_ans를 독립 재집계한다. W0agreement 및 token micro로 대체하지 않는다.
기존 CF factual raw는 최근 검산의 SHA/terminal 불변을 재확인하여 재사용한다. native baseline/history/author는 별도 행이다.
종료된 native FluCon3개는 이미 완료된 frozen CPU collector의 reference-bound 전량 재채점 proof를 원 endpoint SHA/terminal/config/source에 결속해 재사용한다. 이번 owner reducer는 원 raw 해시/완료 범위/작업량/분모/집계 및 표시를 재검산한다. 표시만 원 미반올림 평균×100 뒤 half-up2자리. 결측/DEFERRED는0으로 대체하지 않는다.
처음 CPU raw reducer의 BLAS 과다 스레드 사용을 확인하여 로컬 CPU 프로세스만 중단하고 OPENBLAS1/OMP2로 재실행했다. 이후 기존 collector의 같은 raw 전량 재채점 완료receipt를 발견하여 중복 재채점을 중단하고 그 검산을 재사용했다. 중단된 로컬 CPU rescore를 완료로 주장하지 않는다. 과학 job/score 정의/허용오차 변경 및 GPU 재실행은 없다.
최소 source/control/transport CPU44 PASS, source166 SHA/import0 PASS. 실제 모델/GPU qualification PASS가 아니다.

완료12행 + 미완료 author2행을 회수했다. zsRE6개 E/G/Loc 재집계는 기존 public-query 결과와 동일하다.
각2000 requests, Llama token분모 E6035/G6035/Loc12465, missing0이다.

| Llama CF method | generation job | Flu 원평균 bits | Con 원평균 cosine | 표시 Flu×100 | 표시 Con×100 |
|---|---|---|---|---|---|
| FT | 62259 | 4.49504268731324 | 0.029030793469636437 | 449.50 | 2.90 |
| SPHERE | 62260 | 6.192038526176254 | 0.3364011388508934 | 619.20 | 33.64 |
| MEMIT-FE | 62261 | 4.200683524775806 | 0.08397526788376594 | 420.07 | 8.40 |

세 generation 모두 planned/Flu-valid/Con-valid2000, prompts20000, generated tokens1796707, missing0.
Flu는 퍼센트가 아니며 Con도 정확도가 아니다. 원 raw와 W&B 숫자는 변경하지 않았다.

exact CP/config/output/source/lock/초기 상태는 `audits/servers/server1/completed-table-flucon-cap3-20261010/submission.json`에 있다.
실제 local receipt는 `local/official-baselines/server1/completed-table-flucon-cap3-20261010/registration-r1/`이다.
최초 GH direct 중간 제출 안내는 transport timeout으로 ACK 미관측: 전달 완료로 표기하지 않았다. 최종 compact 게시 경로로 인계한다.
raw/CP/model 대형전송 없음: NO_BROADCAST_NOT_REQUIRED. GPU 완료 대기/recurring monitor/자동 재제출 없음.
