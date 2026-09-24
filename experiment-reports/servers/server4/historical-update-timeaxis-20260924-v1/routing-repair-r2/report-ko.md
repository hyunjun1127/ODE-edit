# T1 코드 오류 수리·재제출 인계 — 모니터링 중단

최신 사용자 정정에 따라 **확인된 오류는 수리하고 재제출하되 job 진행 모니터링은 하지 않는 것**으로 처리했다. 이전 점검-only/수리 미수행 기록은 당시의 역사로 보존한다. 이번 인계는 제출 완료이며 실제 모델 gate 또는 과학 실험 완료 보고가 아니다.

## 수리와 CPU 확인

원 source `6ef71ed2`의 `Backend.state`가 ACTUAL recipe에 force_removal이 들어올 때 존재하지 않는 construction_endpoint를 먼저 조회하던 순서 오류를 고쳤다. 해당 override를 먼저 해석한다. 수학·FP64 제거/FP32 한 번 변환·5개 전체 weight·복원·tolerance·평가패널·MB16은 동일하다. 모델/편집/GPU 검증을 CPU 테스트로 대체하지 않는다.

추가6개는 실제 production 메서드의 ACTUAL/forced diagonal/single/pair/exception restore/malformed input을 작은 CPU5tensor로 검사한다. 기존16+신규6 총22 PASS: worktree 5.624초, frozen source 5.869초. 기존 CP/model/input T0는 SHA receipt+현재 CP24 stat/토큰SHA로 재사용했고 대형 fullSHA·전송을 반복하지 않았다. T0 이후 실제 GPU 수치검증은 NOT_OBSERVED다.

## 정확한 교체

| 역할 | 이전 job | 이전 실제 상태 | 새 job | 연결·자원 |
|---|---:|---|---:|---|
| BASE_ALPHAEDIT | 53176 | PENDING→CANCELLED, elapsed0, allocation없음 | 53182 | 내부 T1→T2P→T2F→T3B, 1GPU/8CPU/60416MiB |
| BASE_MEMIT | 53177 | PENDING→CANCELLED, elapsed0, allocation없음 | 53183 | 각 단계 두 family matching PASS join, 1GPU/8CPU/60416MiB |
| CPU T3A/T4 collector | 53179 | PENDING→CANCELLED, elapsed0, allocation없음 | 53184 | afterany:53182:53183, CPU8/24576MiB/GPU0 |

미시작 exact owner/command/argv/dependency를 확인한 뒤 collector를 먼저 취소하고 GPU 두 개를 취소했다. 다른 job 변경0. 취소 accounting에서 세 job 모두 elapsed0·allocation없음이므로 이전 실행의 GPU 할당비용은0이다. 기존53178의 앞선 취소 이력도 보존한다. 완료 scientific row는 없으므로 재사용 주장은 원 입력/T0로 한정한다.

새3개는 upfront held inspection→release 완료. 등록 직후 `2026-09-24T06:30:37.125949+00:00` snapshot은 모두 PENDING/(None). 이 reason만으로 GPU 부족을 확정하지 않는다. 이후 scheduler/log/result를 조회하지 않는다. 신규 실제 allocation·peak·성능·gate는 미관측이다.

Cap 검사: active_project_gpu0+requested2≤cap2, 두 GPU host요청 합120832MiB/각60416MiB. CPU/GPU script 명시 memory audit2/2 PASS. 7일 wall은 기존 계획의 보수적 scheduler 상한이며 실측시간/GPUh budget이 아니다. freeze 시 free328445677568B, reserve20GiB. 원 CP/source/receipt/로그를 삭제·덮어쓰지 않았다. save_checkpoints=false.

## provenance

- 새 execution source `730a4a9768e5650e01fd9afdc4e0f7895c86ea92`, tree `48b61f229bedb75ec025a8371c160b6a1aad3dab`.
- Archive SHA `c14df932c221c503302035efc7a4825fb2b4ab74614cbf25462aa1ce60a07948`, 122880B.
- Lock SHA `54abf9a7572eceda008db9c001ed8d1fc9de8f5a7397990a4bed26f313bc1699`.
- Submission SHA `47e48606fd136469150fd2764117d7d5158e26cb1ef22cf3019a5e37a759689a`.
- Release SHA `b441ea2fa8649711204e9f59a5e1161066d6b9607bf4e9fe850065f02134230f`.
- Cancel receipt SHA `51b3723e1809347c2ea1f40f47ad7382061abebdf88877566fd19ff9001da5bf`.
- Local immutable root `/data/janghj/ODE-edit/local/historical-update-timeaxis/20260924-v1/attempt-r2-routing/`.

위 실행 source와 별도로 이번 보고·상태 기록 commit을 남긴다. 보고 갱신을 실제 실행 source 변경과 혼동하지 않는다.

Owner source review/CPU 검사이며 독립 red agent는 사용하지 않았다. NO_BROADCAST_NOT_REQUIRED. 자율 등록된 프로그램만 자연 진행하고 agent는 `monitoring_active=false` / `automatic_resume=false`로 중지한다. 이후 사용자 호출 전 주기 확인·자동 추가제출·자동수리를 하지 않는다.
