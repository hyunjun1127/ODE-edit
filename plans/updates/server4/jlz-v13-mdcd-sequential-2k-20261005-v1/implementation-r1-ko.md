# V13 MD/CD sequential 2k 구현 결속

권한 nonce: `ODEEDIT-USER-GH-SH4-JLZ-V13-MDCD-2K-CAP2-20261005-R1`.
새 실행 namespace는 `project/run_scripts/jlz_realized_writer_sequential/`이다.
이 업데이트는 orchestration 설명이며 정본 수식·계수·precision·허용오차 변경이 아니다.

기존 실행 `082300955e21a2c29218d66d98c5d2c37bc53a20`의 실제 import closure 34개와 현물 runtime/native/assets를 재결속했다.
Planner entry/optimize/optimizer/telemetry, V13 capture/geometry/writer/telemetry 및 원 evaluator는 byte 그대로 재사용한다.
기존 actual B1 native/planner/cast/local-action/restore receipt 재사용이지 새 20배치 GPU 검증 완료 주장이 아니다.
별도 BS2 GPU pilot, full-B 추가 fit, 기존 B1 재실행은 없다.

## 연속 상태

- MD/CD 각각 독립 프로세스에서 cold W0/H0·같은 first2000을 시작한다. batch별 계획/teacher/anchor/cache는 자기 현재 모델에서 새로 capture한다.
- 기존 five-branch `run.py`를 실행하지 않는다. 새 controller가 20개의 fresh fit과 선택된 writer 하나를 호출한다.
- `BatchTransaction`은 원 RAM transaction을 확장한다. 성공 시 `finish()`를 호출해 W/H를 유지한다. 실패는 해당 batch entry W/H/RNG/cache/context로 복원하며 이전 성공 prefix는 유지한다.
- `last_virtual`은 매 fit 전에 비우고 fit 종료의 row coverage를 다시 검사한다. writer factor는 각 layer/batch의 현재 H에서 새로 계산되며 이후 재사용하지 않는다.
- 원 `apply_sequential`이 final rewrite-only nested mean CPUFP32 H를 이미 append한다. controller에서 별도 `commit()`/H append를 호출하지 않는다.
- CD KL 행도 owner D를 사용하며 owner/context 수는 원 adapter에서 도출한다. H에는 KL을 넣지 않는다.
- 마지막 B20 이후 loop가 끝난다. B21, checkpoint 또는 resume 경로는 없다.

## 관측 및 검산

20개 native pack/전 2000 CSV row·field·lookup·token, observer 26,000 identity/token 행을 CPU에서 봉인했다.
매 batch pre/post current, W5/10/15/20 allseen; allseen current는 같은 raw에서 산출한다.
Active/superseded의 all_records는 그 batch seen prefix이다. W0 all은 원 first2000 모집단이다.
MD/CD의 event/output/reducer namespace는 분리했다. 정확 identity set도 비교하고 count만으로 PASS하지 않는다.
CPU collector는 원 scalar NLL/TF와 commit/H/history/state join/후보 budget/paired를 다시 계산한다.
누락·rollback·IO/identity mismatch는 PARTIAL/TECHNICAL_BLOCKED로 남기며 0점으로 대체하지 않는다.

새 owner CPU 검사 10개 통과 후 end-to-end controller/collector와 실패주입 fixture를 추가해 총 12개가 통과했다.
JSON commit IO 검산은 RAM의 integer case-ID object key가 JSON string key로 바뀌는 표현 차이를 roundtrip 정규화한다. 목적/수식 변경은 없다.
첫 10개 및 확장 12개 receipt는 각각 local에 immutable 보존했다. 별도 independent reviewer는 사용하지 않았다.

## 자원 및 보존

MD/CD 두 GPU1 lane을 기본 병렬 등록한다. 자기 arm끼리 afterok/직렬 dependency를 만들지 않는다.
Project capacity가 차면 둘 다 같은 외부 resource afterany barrier를 사용하며 무관 job은 변경하지 않는다.
각 8CPU/59392MiB/24h, exportNONE/Requeue0; collector GPU0/8CPU/24576MiB/4h, afterany 두 arm.
24h는 요청 wall 상한이지 2k 실측 ETA가 아니다.
두 arm 합 noCP output·atomic/scratch·report·margin reserve 12GiB를 먼저 요구하고, 각 startup에는 6GiB 이상을 검사한다.
기존 B1 peak RSS/VRAM은 근거로 기록하되 신규 20배치 시간 PASS라고 하지 않는다.
Raw/metric 전체행/fullstdout/tensor는 local에만 남는다. 코드·compact 보고·SHA/size receipt만 Git에 게시한다.
NoCP, exact_resume=NOT_AVAILABLE. 원 B1/V14/source/raw 및 다른 task를 수정하지 않는다.

등록 held 검산 및 release 후 정식 resource pending 또는 bounded 첫 commit→다음 entry 사실만 인계한다.
그 뒤 monitoring_active=false/automatic_resume=false; 신규 poll/heartbeat/자동 retry 없이 봉인 20배치/collector만 자연 진행한다.
