# JLZ v9 ridge 구현과 제출 현황

현재 상태는 MAIN_INITIAL_PAUSED_USER_BOUNDARY다. 계측 저장 누락을 수정한 main A 57572와 main B 57573을 등록했으며, **대표 B의 B1 commit과 B2 entry 연결을 확인한 뒤 agent 모니터링을 중단했다.** 두 arm의 500-edit 완료와 A의 Q2는 인계 시점 미관측이다.

권한 nonce는 `ODEEDIT-USER-GH-SH4-JLZ-V9-RIDGE-500-EXACT-PILOT-20261003-R1`이며 원본 26 content와 bundle inventory를 포함한 27 regular archive member의 SHA 및 size를 검증했다. 기존 v7 source와 결과는 보존했다.

| 단계 | 고정 범위 | 현재 증거 |
|---|---|---|
| CPU owner 검산 | 13 tests | PASS |
| 실제 Q1 | A와 B 각각 BS2 두 batch | 각 50후보 48업데이트와 두 commit 저장 |
| Main A와 B | 각각 cold W0와 H0에서 BS100 다섯 batch | 등록 완료, 대표 B 초기 연결 확인 |
| Q2 exact 비교 | Main A B1의 동일 terminal D | NOT_OBSERVED_AT_HANDOFF |
| W5 누적 평가 | arm마다 R500 P1000 N5000 | NOT_MEASURED |

현재 CPU collector는 57574다. 기존 미실행 main 57569와 57570 및 collector 57571은 해당 task의 계측 구현 오류 수리로 PENDING에서 취소했고 elapsed는 모두 0이다. 다른 task는 변경하지 않았다. 새 main은 기존 Q1 job afterany와 프로그램 내부의 명시 reuse bridge 및 READY 검사를 사용한다. 모든 신규 job은 owner와 full argv, source와 lock, resource와 dependency를 held 상태에서 검사한 뒤 release했다. Project cap2, GPU job별 1GPU와 8CPU, host 60416MiB, export NONE, Requeue0이다.

현재 실행 source는 `b8c4c96fef79d3ad6ba37b1f6f4055bd58343e66`, lock SHA는 `f8a6c2a4cf06b1284f35940c703b8037336bbc5103f206f5dd014e65ad76a9df`다. [새 제출 receipt](../../../../runs/odeedit_jlz_v9_s4_20261003/submission-r2.server4-server-head.json)에 archive/config와 Q1 재사용 근거를 기록했다. Q1 원 source `ee1d8bd7`, 원 제출 및 raw는 보존했고 수정하지 않았다. 보고 게시 commit과 실행 source는 별개다.

## 실제 Q1과 수리 증거

Q1 A의 동일 입력 dense/direct 평균 loss 차이는 0, 최대 층별 gradient RMS 차이는 1.22313e-11이었다. Whole B 순서와 microbatch 비교의 평균 loss 차이는 5.68025e-9, 최대 층별 gradient RMS 차이는 4.68908e-9였다. 원 계약 기준을 통과했다. 이는 BS2 개발 입력의 검증이며 모든 B100과 전체 trajectory의 동등성을 의미하지 않는다.

Q1 A는 Slurm COMPLETED 0:0, allocated GPU 385초다. Q1 B는 두 commit 및 READY와 프로그램 terminal COMPLETED가 저장된 후 Slurm FAILED 0:11이 기록됐고 allocated GPU 371초다. 저수준 원인은 NOT_ESTABLISHED다. B의 Slurm 실패를 PASS로 바꾸지 않았다. 새 source는 원 Q1 commit/candidate/history 연속성과 source/input, unchanged core source, CPU 수리 전후 fixture의 최종 W/H hash 및 NLL/KL 일치를 확인한 명시 bridge를 사용한다. 추가 GPU pilot fit은 0이다.

문맥별 방향 지표, terminal virtual-to-actual KL, proposal radial 및 frozen exact operator 비교 저장을 보강했다. 계산 식과 계수, q Adam 및 candidate 수는 변경하지 않았다. CPU 회귀 13개와 수리 전후 tiny fixture 일치는 actual GPU 재검증과 구분한다. 원 Q1에서 저장하지 않은 해당 추가 telemetry는 소급 생성하지 않는다.

## 대표 초기 확인과 종료 경계

2026년 10월 3일 05시 46분 05초 KST에 B의 B1 25후보와 24 Adam 업데이트, exact evaluated weight commit, H 5회 갱신을 확인했다. W1 R100/P200/N1000의 저장 raw를 CPU에서 다시 집계했고 기존 summary와 일치했다. B1 after state, W1 observer state, B2 entry state가 일치하며 observer 비변이도 확인했다.

B1 기록 시간은 1593.9624초, W1 observer는 51.0706초다. 전자는 entry/fit/commit을 포함한 process timer이며 parent allocated GPU 시간이나 전체 ETA로 바꾸지 않는다. [초기 증거](../../../../audits/servers/server4/jlz-realization-v9-ridge-bs100x5-with-exact-probe/initial-handoff.json)에 원 receipt의 path/size/SHA와 state 연결을 보존했다.

신규 baseline, full W0 평가, checkpoint 저장은 0이다. 독립 reviewer는 사용하지 않았다. monitoring_active=false와 automatic_resume=false로 종료하며 scheduler/log/result를 더 조회하지 않는다. 이미 봉인된 두 runner와 collector만 예정된 W5 평가까지 진행하도록 등록돼 있다. 상세 완료 리뷰는 사용자 recall 때 수행한다.

편집 baseline들의 완료 비교표는 이번 초기 인계에서 NOT_REVIEWED다. 기존 first500 W0만 identity를 결속해 재사용했다. Markdown renderer가 설치되지 않아 실제 렌더는 미검증이며, 보고 링크와 JSON 형식은 CPU에서 검사했다.

상세 구현 결속은 [구현 기록](../../../../plans/updates/server4/jlz-realization-v9-ridge-bs100x5-with-exact-probe/implementation-r1.md)에 있다. Raw와 tensor는 local에 보존하고 작은 source 및 보고만 게시한다.
