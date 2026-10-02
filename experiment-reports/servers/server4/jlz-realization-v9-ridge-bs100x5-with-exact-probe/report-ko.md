# JLZ v9 ridge 구현과 제출 현황

현재 상태는 IMPLEMENTING_NOT_SUBMITTED이다. SH4가 새 v9 production source를 구현하고 CPU 13 tests를 통과했다. 실제 GPU Q1, Q2, main 결과는 아직 관측하지 않았다.

권한 nonce는 `ODEEDIT-USER-GH-SH4-JLZ-V9-RIDGE-500-EXACT-PILOT-20261003-R1`이며 원본 26 content와 bundle inventory를 포함한 27 regular archive member의 SHA 및 size를 검증했다. 기존 v7 source와 결과는 보존했다.

| 단계 | 고정 범위 | 현재 증거 |
|---|---|---|
| CPU owner 검산 | 13 tests | PASS |
| 실제 Q1 | A와 B 각각 BS2 두 batch | NOT_RUN |
| Main A와 B | 각각 cold W0와 H0에서 BS100 다섯 batch | NOT_SUBMITTED |
| Q2 exact 비교 | Main A B1의 동일 terminal D | NOT_RUN |
| W5 누적 평가 | arm마다 R500 P1000 N5000 | NOT_MEASURED |

신규 baseline, full W0 평가, checkpoint 저장은 0이다. 독립 reviewer 및 실제 GPU numerical PASS는 아직 없다. 등록 후 job/source/lock과 대표 초기 상태를 별도 기록한다.

상세 구현 결속은 [구현 기록](../../../../plans/updates/server4/jlz-realization-v9-ridge-bs100x5-with-exact-probe/implementation-r1.md)에 있다. Raw와 tensor는 local에 보존하고 작은 source 및 보고만 게시한다.
