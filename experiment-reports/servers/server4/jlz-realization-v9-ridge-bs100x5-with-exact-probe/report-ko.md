# JLZ v9 ridge 구현과 제출 현황

현재 상태는 SUBMITTED_RELEASED이다. SH4가 새 v9 production source를 구현하고 CPU 13 tests를 통과했다. Q1 A와 B의 RUNNING은 관측했으나 아직 기술 qualification 결과와 main 초기 연결은 관측하지 않았다.

권한 nonce는 `ODEEDIT-USER-GH-SH4-JLZ-V9-RIDGE-500-EXACT-PILOT-20261003-R1`이며 원본 26 content와 bundle inventory를 포함한 27 regular archive member의 SHA 및 size를 검증했다. 기존 v7 source와 결과는 보존했다.

| 단계 | 고정 범위 | 현재 증거 |
|---|---|---|
| CPU owner 검산 | 13 tests | PASS |
| 실제 Q1 | A와 B 각각 BS2 두 batch | 57567과 57568 RUNNING 관측 |
| Main A와 B | 각각 cold W0와 H0에서 BS100 다섯 batch | 57569와 57570 afterok Q1 |
| Q2 exact 비교 | Main A B1의 동일 terminal D | NOT_RUN |
| W5 누적 평가 | arm마다 R500 P1000 N5000 | NOT_MEASURED |

CPU collector 57571은 4개 GPU job의 afterany다. 모든 job은 owner와 full argv, source와 lock, resource와 dependency를 held 상태에서 검사한 뒤 release했다. Project cap2, GPU job별 1GPU와 8CPU, host 60416MiB, export NONE, Requeue0이다. Dependency 대기를 resource shortage로 보고하지 않는다.

실행 source는 `ee1d8bd7adb9d6c34021ba0b9f36f435565c13f7`, lock SHA는 `a6a5ac27db2532f00b3c16bf911b863b70de1a8c8db62dd9abb040c0de6f77cc`다. [제출 receipt](../../../../runs/odeedit_jlz_v9_s4_20261003/submission.server4-server-head.json)에 archive와 config SHA, 정확한 dependency를 기록했다. 보고 게시 commit과 실행 source는 별개다.

신규 baseline, full W0 평가, checkpoint 저장은 0이다. 독립 reviewer 및 실제 GPU numerical PASS는 아직 없다. 대표 main B1→B2 또는 정식 resource pending까지만 초기 확인을 계속한다.

상세 구현 결속은 [구현 기록](../../../../plans/updates/server4/jlz-realization-v9-ridge-bs100x5-with-exact-probe/implementation-r1.md)에 있다. Raw와 tensor는 local에 보존하고 작은 source 및 보고만 게시한다.
