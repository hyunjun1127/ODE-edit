# JLZ v4 compute-r1 A/B 2k — 제출 인계

## 최신 USER override: W0 신규 평가 제거

기존 W0 전체 결과(R2000/P4000/N20000)를 CPU identity/token/행 검산 후 재사용한다.
57282 W0와 미실행 downstream 57283–57287은 SH4가 취소했다. 원자료는 보존하며 prior 비용332 GPU초를 신규 실행과 분리한다.

| 새 단계 | Job | Dependency |
|---|---:|---|
| Pilot A | 57290 | 없음 |
| Pilot B | 57291 | 없음 |
| Timing/Main A | 57292 | afterok:57290:57291 |
| Timing/Main B | 57293 | afterok:57290:57291 |
| CPU collector | 57294 | afterany:57290:57291:57292:57293 |

새 immutable source: `7c1d421c1bac95e3b00cce7262c0b5dd4fa164d0`.
Lock SHA: `ecdd88e9d01516a0d51cd3fed10380139c59425f8e3188f22b5d555baddf585b`.
W0 재사용 bridge SHA: `e4b75f19786291c14f6baaa4d6e1948eec127f4b530b261725dc6b2472719008`.
CPU 회귀7 PASS. A/B는 각각 cold W0/H0이며 pilot 두 개 실제 RUNNING을 확인했다.
Main 초기 연결은 NOT_OBSERVED. 신규 W0 job/forward0, baseline0, noCP/cap2 유지.
W0의 이전/현재 평가 배치 배치 차이와 비트 동일성 미확립을 공개한다.
[override 감사](../../../../audits/servers/server4/jlz-native-joint-v4-bs100x20-20261002-v1/user-remove-w0-r1.md).

아래는 취소 전 원 attempt-r1 제출 이력이며 현재 job mapping이 아니다.

상태: SUBMITTED_INITIAL_NOT_OBSERVED. 완료·실제 main gate PASS를 뜻하지 않는다.

Instruction: `ODEEDIT-USER-GH-SH4-JLZ-V4-COMPUTE-R1-2K-20261002-R1`.
Execution source: `de2cd4197132eb994717e3a6074fcad47487253c`.
Lock SHA256: `ad11b0a53db622af57e2f6f0922df49a9c31a312c0d7c6f5dca86545a9777a19`.

| 단계 | Job | Dependency |
|---|---:|---|
| collector | 57287 | afterany:57285:57286:57283:57284:57282 |
| main-JLZ_A | 57285 | afterok:57283:57284 |
| main-JLZ_B | 57286 | afterok:57283:57284 |
| pilot-JLZ_A | 57283 | afterok:57282 |
| pilot-JLZ_B | 57284 | afterok:57282 |
| shared-SHARED | 57282 | 없음 |

전체 held owner/fullargv/resource/source/dependency 검사 후 release했다.
pilot은 arm당 BS2 25후보/24 Adam commit 및 다음 BS2 entry만, timing은 arm당4후보이며 write0.
main은 각 cold W0/H0 BS100×20, 총40 commit/1000후보/960 Adam이다. 신규 baseline0.

## 검산과 자원

정본 11파일 FULL_READ/SHA, 승인 archive33 regular member SHA/size, case2000행 및 평가 일정 검산.
CPU tokenizer22 pack과 회귀6 tests PASS. 실제 GPU 정합은 아직 미관측이며 별도 독립 red는 사용하지 않았다.
cap2; GPU job 각1GPU/8CPU/60416MiB, collector0GPU/8CPU/24576MiB.
host 예상44GiB/GPU 예상65GiB, disk reserve30GiB. GPU wall7일은 ETA가 아니다.

## 보존·한계

Local: `/data/janghj/ODE-edit/local/jlz-native-joint-v4/20261002-compute-r1/attempt-r1`.
save_checkpoints=false; exact_resume=NOT_AVAILABLE. Raw/teacher/tensor/prompt/fullstdout Git0.
기존 v2 및 타 task의 STOP 유지. NO_BROADCAST_NOT_REQUIRED: 같은 서버의 소형 source/receipt만 게시.
최종 수치/완료단계/실측비용은 아직 NOT_MEASURED. 등록된 runner/collector가 저장하며 사용자 recall 때 상세회수한다.
