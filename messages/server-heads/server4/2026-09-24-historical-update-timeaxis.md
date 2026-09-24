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
