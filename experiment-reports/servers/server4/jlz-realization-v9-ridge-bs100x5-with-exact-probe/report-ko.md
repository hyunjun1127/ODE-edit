# JLZ v9 ridge 구현과 제출 현황

현재 상태는 MAIN_RUNNING_INITIAL_NOT_OBSERVED이다. 계측 저장 누락을 수정한 새 source로 main A 57572와 main B 57573을 release했고 RUNNING을 관측했다. Main B1 commit과 B2 연결은 아직 미관측이다.

권한 nonce는 `ODEEDIT-USER-GH-SH4-JLZ-V9-RIDGE-500-EXACT-PILOT-20261003-R1`이며 원본 26 content와 bundle inventory를 포함한 27 regular archive member의 SHA 및 size를 검증했다. 기존 v7 source와 결과는 보존했다.

| 단계 | 고정 범위 | 현재 증거 |
|---|---|---|
| CPU owner 검산 | 13 tests | PASS |
| 실제 Q1 | A와 B 각각 BS2 두 batch | 각 50후보 48업데이트와 두 commit 저장 |
| Main A와 B | 각각 cold W0와 H0에서 BS100 다섯 batch | 57572와 57573 RUNNING 관측 |
| Q2 exact 비교 | Main A B1의 동일 terminal D | NOT_RUN |
| W5 누적 평가 | arm마다 R500 P1000 N5000 | NOT_MEASURED |

현재 CPU collector는 57574다. 기존 미실행 main 57569와 57570 및 collector 57571은 해당 task의 계측 구현 오류 수리로 PENDING에서 취소했고 elapsed는 모두 0이다. 다른 task는 변경하지 않았다. 새 main은 기존 Q1 job afterany와 프로그램 내부의 명시 reuse bridge 및 READY 검사를 사용한다. 모든 신규 job은 owner와 full argv, source와 lock, resource와 dependency를 held 상태에서 검사한 뒤 release했다. Project cap2, GPU job별 1GPU와 8CPU, host 60416MiB, export NONE, Requeue0이다.

현재 실행 source는 `b8c4c96fef79d3ad6ba37b1f6f4055bd58343e66`, lock SHA는 `f8a6c2a4cf06b1284f35940c703b8037336bbc5103f206f5dd014e65ad76a9df`다. [새 제출 receipt](../../../../runs/odeedit_jlz_v9_s4_20261003/submission-r2.server4-server-head.json)에 archive/config와 Q1 재사용 근거를 기록했다. Q1 원 source `ee1d8bd7`, 원 제출 및 raw는 보존했고 수정하지 않았다. 보고 게시 commit과 실행 source는 별개다.

## 실제 Q1과 수리 증거

Q1 A의 동일 입력 dense/direct 평균 loss 차이는 0, 최대 층별 gradient RMS 차이는 1.22313e-11이었다. Whole B 순서와 microbatch 비교의 평균 loss 차이는 5.68025e-9, 최대 층별 gradient RMS 차이는 4.68908e-9였다. 원 계약 기준을 통과했다. 이는 BS2 개발 입력의 검증이며 모든 B100과 전체 trajectory의 동등성을 의미하지 않는다.

Q1 A는 Slurm COMPLETED 0:0, allocated GPU 385초다. Q1 B는 두 commit 및 READY와 프로그램 terminal COMPLETED가 저장된 후 Slurm FAILED 0:11이 기록됐고 allocated GPU 371초다. 저수준 원인은 NOT_ESTABLISHED다. B의 Slurm 실패를 PASS로 바꾸지 않았다. 새 source는 원 Q1 commit/candidate/history 연속성과 source/input, unchanged core source, CPU 수리 전후 fixture의 최종 W/H hash 및 NLL/KL 일치를 확인한 명시 bridge를 사용한다. 추가 GPU pilot fit은 0이다.

문맥별 방향 지표, terminal virtual-to-actual KL, proposal radial 및 frozen exact operator 비교 저장을 보강했다. 계산 식과 계수, q Adam 및 candidate 수는 변경하지 않았다. CPU 회귀 13개와 수리 전후 tiny fixture 일치는 actual GPU 재검증과 구분한다. 원 Q1에서 저장하지 않은 해당 추가 telemetry는 소급 생성하지 않는다.

신규 baseline, full W0 평가, checkpoint 저장은 0이다. 독립 reviewer는 사용하지 않았다. 대표 main B1→B2 또는 정식 resource pending까지만 초기 확인을 계속한다.

상세 구현 결속은 [구현 기록](../../../../plans/updates/server4/jlz-realization-v9-ridge-bs100x5-with-exact-probe/implementation-r1.md)에 있다. Raw와 tensor는 local에 보존하고 작은 source 및 보고만 게시한다.
