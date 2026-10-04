# JLZ V13 B1 등록·초기 인계

상태: **SUBMITTED_RESOURCE_PENDING**. 실험 완료 보고가 아니다.

- 권한 nonce: `ODEEDIT-USER-GH-SH4-JLZ-V13-B1-REALIZATION-20261005-R1`
- Authority: `37019ce3ea3b4ae24f8fed29a236b31d102911a8`
- 실행 source: `08230095` (게시 main commit과 구분)
- 실행 lock SHA256: `06b9fe80e31cd104f0d1045c8f4e2475f0aabc55bef9374e3ed7f6784ae977f5`
- SH4/session: `server4` / `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`

## 범위 및 근거

정본 `plans/global/2026-10-05-jlz-v13-realized-writer/`와 B1 override 전체 정독/SHA 검산.
Llama3-8B-Instruct revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, fixed10k first100, cold W0/H0.
fit 1회 후 같은 계획을 RT/RD/MT/MD/CD에서 관측한다. 각 branch W/H/RNG/context/cache/ledger를 RAM 복원한다.
5 branch는 100 unique requests의 5 endpoint이며 500 unique edits가 아니다.
B2·추가 fit·BS2 순차 pilot·새 baseline·계수 sweep·checkpoint 저장 없음.

## 검증 상태

| 항목 | 상태 |
|---|---|
| 정본/실행 JSON SHA 및 source 경계 | PASS |
| first100 native token/lookup | 700행, 원 native와 일치 |
| 평가 token identity | R100/P200/N1000, 1,300행 |
| 정본 CPU reference | 16 PASS; 원본 receipt 불변 |
| production CPU + planner 회귀 | 21 PASS; tiny random CPU Llama 포함 |
| 독립 reviewer | 미사용, owner audit/red 체크리스트 |
| actual 8B GPU 검산 | NOT_OBSERVED |
| B1 fit·5 writer 완료 | NOT_OBSERVED |
| 성능·실현 수치 | NOT_MEASURED |

CPU 결과를 actual Llama parity로 표기하지 않았다.
Qualification은 B1 입력 subset의 고정 후보/연산 비교이며 추가 fit은 없다.
확정 기술 불일치는 고정 tolerance로 차단하고 낮은 성능·range mismatch는 기록만 한다.

## 제출 및 초기 snapshot

| job | 역할 | 자원 | 초기 상태 |
|---|---|---|---|
| 58381 | 단일 B1 fit + 5 writer 평가 | GPU1/CPU8/59392MiB/24h | PENDING (Resources) |
| 58382 | 독립 CPU raw reducer/report | GPU0/CPU8/24576MiB/4h | PENDING (Dependency), afterany:58381 |

두 job 모두 owner/source/fullargv/node/partition/memory/wall/exportNONE/Requeue0/dependency를 held 상태에서 검사한 후 release.
등록 전 SH4 project active GPU0, tracked/local cap3, 본 task cap1. 기존 job 변경 없음.
24h는 요청 wall 상한이지 측정 ETA가 아니다. 초기 snapshot에서 GPU 할당 없음.

## 산출물·중단 경계

원자료 root: `/data/janghj/ODE-edit/local/jlz-v13-realized-writer-b1/20261005-v1/attempt-r1/`.
실행 중/종료 raw는 `B1/`, CPU 결과는 `cpu-report/`에 봉인 프로그램이 기록한다.
CPU collector는 metrics/paired/realization/writer Q/상태·history·비용 검산과 report/inventory 뒤 terminal을 쓴다.
부분 결과는 PARTIAL_OR_TECHNICAL_BLOCKED, 없는 값은 NOT_MEASURED.

정식 resource-pending 초기 인계 후 `monitoring_active=false`, `automatic_resume=false`.
새 polling/heartbeat/자동 retry 없음. 이미 등록된 프로그램만 B1까지 자연 진행한다.
Raw/model/tensor/prompt/fullstdout는 Git 미게시·원본 KEEP.
NO_BROADCAST_NOT_REQUIRED: 동일 서버 task로 compact source/report/manifest만 Git 공유.

구현/감사: `project/run_scripts/jlz_realized_writer/`, `audits/servers/server4/jlz-v13-realized-writer-b1/`.
등록 receipt: `runs/server4/jlz-v13-realized-writer-b1/submission-r1.json`.
