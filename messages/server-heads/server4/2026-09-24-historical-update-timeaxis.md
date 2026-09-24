# Historical timeaxis — 제출 인계

Instruction: GH-SH4-HISTORICAL-UPDATE-TIMEAXIS-20260924-V1.

T0 shared CPU PASS, CP24/tensor120/model4shard/토큰 결속. 별도 모델 GPU T1은 아래 등록 시점에 NOT_OBSERVED였다. CPU12 PASS는 GPU parity의 대체가 아니다.

| 실행 | 실제 job | 연결 | 자원 |
|---|---:|---|---|
| BASE_ALPHAEDIT | 53176 | T0 후 내부 T1→T2P→T2F→T3B | 1GPU/8CPU/60416MiB |
| BASE_MEMIT | 53177 | 같은 단계마다 두 family atomic PASS join | 1GPU/8CPU/60416MiB |
| CPU reducer/collector | 53178 | afterany:53176:53177; 둘 모두 완료해야 T3A/T4 | CPU8/24576MiB/GPU0 |

Held owner/command/argv/resource/dependency 검사 후 release했다. 최초 snapshot은 PENDING(None); 이 값만으로 GPU resource shortage를 주장하지 않는다. 원 plan의 상태 수173/score task383/156 main/16 pair를 job 개수와 구분한다. 본 task는 PENDING/초기 gate에서 끝내지 않는다.

실행 source commit `6ef71ed2`, lock SHA `bc8ae75e983c68a1eb4489e9c3be98e0f77f8a567cf32e9eb88d6b9e1c994d9f`.
실제 lock/submission/release는 `/data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1/attempt-v1/`에 있다. Runtime source는 그 아래 `source/`; 원 shared checkout와 input CP는 변경하지 않았다.

등록 wall7일은 partition 범위 안의 보수적 상한, 실측 GPUh나 예상완료 주장 아님. 실제 T2P paddedtoken/초로 예측을 기록한다. noCP·same-host NO_BROADCAST_NOT_REQUIRED. 별도 독립 reviewer 미사용, owner source/CPU 검산 수행.

## CPU collector 저장 순서 수리 / 새 mapping

원 위 표는 최초 submission 기록이다. GPU53176/53177은 변경하지 않았다. CPU53178은 보고/manifest I/O 완료 전 COMPLETED를 기록하는 source 순서 문제를 발견하여 실행 전에 취소했다. 실제 accounting은 CANCELLED, elapsed0, allocation없음이다. 원파일/source/receipt를 보존한다.

새 CPU collector **53179**는 `afterany:53176:53177`, CPU8/24576MiB/exportNONE/Requeue0이며 held inspection→release 완료. 실제 계산·GPU수치 변경0. 새 analysis source `78a633a9`, 실행 source6ef71ed2와 구별한다. Source·검사·새ID receipt는 local `attempt-v1/collector-repair-r1/receipt.json`; RCA는 audits의 collector-preexecution-order-rca.md다. 완료 보고·artifact inventory를 먼저 저장한 뒤 마지막 atomic COMPLETED를 쓰도록 수정했다. 좁은 CPU4 tests PASS, actual GPU T1은 아직 NOT_RUN이다.

## 최신 사용자 중지 — CPU 점검만 / monitoring=false

사용자 “리소스가 없으니 코드 파이프라인 점검만 하고 모니터링은 중단하자”를 적용했다. 위 T4까지 계속 관찰 지시는 이 범위에서 대체된다. 53176/53177/53179는 그대로 보존하며 이후 scheduler/log/result 조회·수리·재제출0, automatic_resume=false다.

기존 CPU16 tests PASS이나 파이프라인 전체 PASS는 아니다. frozen backend.py:52의 ACTUAL+force_removal 경로가 없는 construction_endpoint를 먼저 읽는 KeyError를 별도 CPU 반례로 재현했다. 실제 job 실패를 새로 조회한 것이 아니다. 원 source와 job을 수정하지 않고 사용자 recall을 기다린다. 근거: `audits/servers/server4/historical-update-timeaxis-20260924-v1/user-pause-r1/pipeline-review-ko.md`.

## 최신 정정 적용 / 53182·53183·53184 교체 완료

사용자는 오류 수리·재제출은 수행하되 job 모니터링만 중단하라고 정정했다. 위 점검-only 상태는 역사이며 현재 상태가 아니다. T1 force_removal 우선분기 최소수리, CPU22/22 PASS(WT와 frozen), actual GPU gate는 미관측이다. source730a4a9768e5650e01fd9afdc4e0f7895c86ea92 / lock54abf9a7572eceda008db9c001ed8d1fc9de8f5a7397990a4bed26f313bc1699.

이전53176/53177/53179는 정확 owner/source/미시작을 확인하고 취소(elapsed0/allocation0)했다. 새 Alpha53182, MEMIT53183, CPU53184(afterany:53182:53183)는 검사·release 완료. 2026-09-24T06:30:37Z 제출 snapshot PENDING(None)을 마지막으로 진행조회하지 않는다. 원source/raw/CP KEEP, 다른job변경0, cap2/각59GiB/noCP불변. `monitoring_active=false`, `automatic_resume=false`. 상세: `experiment-reports/servers/server4/historical-update-timeaxis-20260924-v1/routing-repair-r2/report-ko.md`.

## 2026-09-25 명시 rerun recall — 수치계약 차단

최신 nonce `ODEEDIT-GH-SH4-HISTORICAL-TIMEAXIS-RERUN-20260925-R1`의 한정 상태확인/RCA를 수행했다. 53182/53183 FAILED, 53184 scheduler COMPLETED지만 collector science TECHNICAL_BLOCKED. 최초 MEMIT W100 MB16↔MB1 NLL0.0005242824554443359>0.00025이며 Alpha는 자체T1 PASS 뒤 peer failure로 중단됐다. Source 정의/기준을 유지하는 수리 원인이 아직 확인되지 않아 임의반복·완화 없이 `BLOCKED_NUMERICAL_CONTRACT`로 신규제출0을 보고한다. 이전 위 PENDING은 과거 관측이다.

보존9endpoint·2592답변행 비교 CPU검산, Alpha12diagonal receipt·T0/CP24/model4 결속. 실패W100 raw는 NOT_RECORDED. 원부모GPU1569초(0.4358333h), step중복0. 원source/raw/실패/CP보존, GPU/Slurm write/수치변경0. 상세 `experiment-reports/servers/server4/historical-update-timeaxis-20260924-v1/rerun-20260925-r1/report-ko.md`. Monitoring/automatic_resume=false로 STOP한다.

## 2026-09-25 사용자 기록 전용 전환 — 실제 release 인계

최신 nonce `ODEEDIT-GH-SH4-HISTORICAL-TIMEAXIS-RECORD-ONLY-20260925-R1`을 적용했다.
이전 BLOCKED_NUMERICAL_CONTRACT는 역사기록이며 최신 권한은 수치 비교 기록 전용 진행이다.
새 Alpha53283/MEMIT53284/collector53285(afterany:53283:53284)를 held 검사·release했다.
원 failed53182/53183/53184와1569 GPU초는 보존·재취소0. 저장9endpoint/Alpha12diagonal을 explicit bridge로 재사용한다.
실행 `b856babbca101c096d72a38a3ec9c936a85a4cf3`, lock `f99939e6edda921c6320110c44a4a0b03e5c8911e2937c8955ea5e98d9ba7ebc`.
CPU32 PASS는 실제 GPU 수치 PASS가 아니다. `numerical_certification=NOT_ESTABLISHED` 유지.
Release 후 진행조회0/INITIAL_NOT_OBSERVED, monitoring_active=false/automatic_resume=false.
상세: `experiment-reports/servers/server4/historical-update-timeaxis-20260924-v1/record-only-20260925-r1/report-ko.md`.
