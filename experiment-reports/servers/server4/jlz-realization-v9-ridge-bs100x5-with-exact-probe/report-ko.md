# JLZ v9 ridge 500 edit 완료 요약

2026년 10월 3일 사용자 완료 recall에서 **A와 B 각각 BS100 × 5 및 W5 누적평가 완료**를 확인했다. 두 GPU job과 CPU collector 모두 Slurm COMPLETED 0:0이다. 공통 500문항을 두 독립 cold W0/H0 경로에서 편집했으며 총 10 commit, 250후보, 240 Adam 업데이트, 층별 H 갱신 50회다. B6와 신규 checkpoint는 없다.

## W5 최종 수치

| Arm | RS rewrite | PS paraphrase | NS neighborhood |
|---|---:|---:|---:|
| A | 498/500 = 99.60% | 949/1000 = 94.90% | 4143/5000 = 82.86% |
| B | 499/500 = 99.80% | 947/1000 = 94.70% | 4136/5000 = 82.72% |

RS와 PS는 new NLL < true NLL, NS는 true NLL < new NLL이며 동률은 실패다. 별도의 teacher-forced strict 정확도는 A의 R/P/N 각각 498/500, 664/1000, 951/5000, B는 499/500, 663/1000, 949/5000이다. Free generation 정확도가 아니다. NLL과 TF token micro 및 prompt macro를 포함한 W1–W5 전체 표는 [metrics CSV](completed-recall-r1/collector-metrics.csv)에 있다.

각 문항의 편집 직후에서 W5까지 R 성공 손실/획득은 A와 B 모두 0/0, P는 A 19/7, B 21/8이다. 기존 first500 W0 NS 4392/5000 대비 W5 손실/획득은 A 357/108, B 364/108이다. 이 W0 자료의 과거 microbatch layout은 bitwise 동등성 검증 대상이 아니었다. [짝비교와 cohort 집계](completed-recall-r1/collector-paired-cohorts.json)는 원 collector 산출물을 그대로 보존했다.

## Q2 exact 비교

A B1의 동일한 terminal D에서 frozen ridge K operator 비교와 별도 causal exact shadow를 수행했다. Causal exact L4–L8 모두 rank100이며 정해진 qualification을 통과했다. 최대 FP64 상대 잔차는 9.88275e-15, 최대 FP32 RMS 오차는 4.52095e-8이다. 층별 기준과 결과는 [Q2 요약](completed-recall-r1/q2-summary.json)에 있다.

동일 첫100의 ridge endpoint RS/PS/NS는 100/100, 192/200, 868/1000이고, exact shadow는 100/100, 196/200, 855/1000이다. 이는 A B1 한 시점 비교이며 exact 500-edit trajectory 결과가 아니다. 추가 fit/update/backward는 0, shadow H append는 0이며 W/H/RNG 복원과 원 ridge payload 불변을 확인한 receipt 이후 ridge를 commit했다.

## 종료와 비용

| Job | 역할 | Slurm 상태 | 종료 시각 KST | 할당 GPU초 |
|---|---|---|---|---:|
| 57572 | Main A | COMPLETED 0:0 | 10월 3일 07:54:09 | 9516 |
| 57573 | Main B | COMPLETED 0:0 | 10월 3일 07:36:13 | 8440 |
| 57574 | CPU collector | COMPLETED 0:0 | 10월 3일 07:54:13 | 0 |

Main 합계는 17,956 GPU초(4.9878 GPU시간), 기존 Q1 756초를 포함하면 18,712초(5.1978 GPU시간)다. CPU collector 경과시간은 2초다. Parent allocation만 합산했으며 batch/extern 및 중첩 process timer를 더하지 않았다. A에는 Q2 비용이 포함된다. [단발 accounting](../../../../audits/servers/server4/jlz-realization-v9-ridge-bs100x5-with-exact-probe/completed-recall-r1/accounting.json), [원 process timer](completed-recall-r1/collector-cost.json), [종료 및 peak 기록](completed-recall-r1/collector-summary.json)을 구분해 보존했다.

## 완료 검산과 자료 보존

기존 collector의 COMPLETED_500_BOTH와 누락 0을 확인했다. 이번 recall에서는 모델 없이 저장 scalar raw 24,700행(두 arm W1–W5 및 Q2)을 별도 소형 집계 코드로 재검산해 NLL/TF/분모/CSV 일치를 확인했다. 10 commit, 250후보, state 연속성, observer 비변이, exact weight hash 및 no B6를 확인했다. 같은 owner의 CPU 검산이며 독립 reviewer 또는 새 GPU 검증이 아니다. [검산과 원본 path/size/SHA](completed-recall-r1/verification.json), [재현 코드](../../../../audits/servers/server4/jlz-realization-v9-ridge-bs100x5-with-exact-probe/completed-recall-r1/verify_summary.py)를 게시한다.

원 collector inventory는 623항목, 646,421,537B를 기록한다. 여기에는 local 진단 tensor 100개가 포함되며 checkpoint가 아니다. 이번 recall에서 tensor/model/CP 전체 재해시는 하지 않았고 inventory 자체 SHA를 terminal receipt와 대조했다. Raw/tensor/prompt/full stdout은 local에 보존하고 Git에 넣지 않았다. 신규 평가·baseline·submit은 모두 0이다. AlphaEdit, AlphaEdit-BLUE, CAKE, MEMIT-H의 새 비교 검산은 이번 간단 완료 정리 범위에서 NOT_REVIEWED다.

기존 보고 위치와 형식을 유지하고 현재 완료 수치와 과거 기록을 분리했다. Markdown의 HTML 렌더 구조·표·링크·JSON/CSV를 CPU 검사하며 브라우저 시각 검산은 미수행이다. `NO_BROADCAST_NOT_REQUIRED`: 같은 서버의 원자료는 유지하고 소형 코드·보고·manifest만 Git으로 게시한다. monitoring_active=false, automatic_resume=false이며 새 실행이나 주기 감시는 하지 않는다.

## 실행 결속과 과거 기록

권한 nonce는 `ODEEDIT-USER-GH-SH4-JLZ-V9-RIDGE-500-EXACT-PILOT-20261003-R1`이며 원본 26 content와 bundle inventory를 포함한 27 regular archive member의 SHA 및 size를 검증했다. 기존 v7 source와 결과는 보존했다.

| 단계 | 고정 범위 | 현재 증거 |
|---|---|---|
| CPU owner 검산 | 13 tests | PASS |
| 실제 Q1 | A와 B 각각 BS2 두 batch | 각 50후보 48업데이트와 두 commit 저장 |
| Main A와 B | 각각 cold W0와 H0에서 BS100 다섯 batch | 두 arm 모두 완료 |
| Q2 exact 비교 | Main A B1의 동일 terminal D | causal exact 5층 QUALIFIED 및 restore 기록 |
| W5 누적 평가 | arm마다 R500 P1000 N5000 | 두 arm 모두 완료, 누락 0 |

기존 미실행 main 57569와 57570 및 collector 57571은 계측 구현 오류 수리로 PENDING에서 취소했고 elapsed는 모두 0이었다. 다른 task는 변경하지 않았다. R2 main은 기존 Q1 job afterany와 프로그램 내부의 명시 reuse bridge 및 READY 검사를 사용했다. 모든 job은 owner와 full argv, source와 lock, resource와 dependency를 held 상태에서 검사한 뒤 release했다. Project cap2, GPU job별 1GPU와 8CPU, host 60416MiB, export NONE, Requeue0이었다.

현재 실행 source는 `b8c4c96fef79d3ad6ba37b1f6f4055bd58343e66`, lock SHA는 `f8a6c2a4cf06b1284f35940c703b8037336bbc5103f206f5dd014e65ad76a9df`다. [새 제출 receipt](../../../../runs/odeedit_jlz_v9_s4_20261003/submission-r2.server4-server-head.json)에 archive/config와 Q1 재사용 근거를 기록했다. Q1 원 source `ee1d8bd7`, 원 제출 및 raw는 보존했고 수정하지 않았다. 보고 게시 commit과 실행 source는 별개다.

## 실제 Q1과 수리 증거

Q1 A의 동일 입력 dense/direct 평균 loss 차이는 0, 최대 층별 gradient RMS 차이는 1.22313e-11이었다. Whole B 순서와 microbatch 비교의 평균 loss 차이는 5.68025e-9, 최대 층별 gradient RMS 차이는 4.68908e-9였다. 원 계약 기준을 통과했다. 이는 BS2 개발 입력의 검증이며 모든 B100과 전체 trajectory의 동등성을 의미하지 않는다.

Q1 A는 Slurm COMPLETED 0:0, allocated GPU 385초다. Q1 B는 두 commit 및 READY와 프로그램 terminal COMPLETED가 저장된 후 Slurm FAILED 0:11이 기록됐고 allocated GPU 371초다. 저수준 원인은 NOT_ESTABLISHED다. B의 Slurm 실패를 PASS로 바꾸지 않았다. 새 source는 원 Q1 commit/candidate/history 연속성과 source/input, unchanged core source, CPU 수리 전후 fixture의 최종 W/H hash 및 NLL/KL 일치를 확인한 명시 bridge를 사용한다. 추가 GPU pilot fit은 0이다.

문맥별 방향 지표, terminal virtual-to-actual KL, proposal radial 및 frozen exact operator 비교 저장을 보강했다. 계산 식과 계수, q Adam 및 candidate 수는 변경하지 않았다. CPU 회귀 13개와 수리 전후 tiny fixture 일치는 actual GPU 재검증과 구분한다. 원 Q1에서 저장하지 않은 해당 추가 telemetry는 소급 생성하지 않는다.

## 대표 초기 확인과 종료 경계

2026년 10월 3일 05시 46분 05초 KST에 B의 B1 25후보와 24 Adam 업데이트, exact evaluated weight commit, H 5회 갱신을 확인했다. W1 R100/P200/N1000의 저장 raw를 CPU에서 다시 집계했고 기존 summary와 일치했다. B1 after state, W1 observer state, B2 entry state가 일치하며 observer 비변이도 확인했다.

B1 기록 시간은 1593.9624초, W1 observer는 51.0706초다. 전자는 entry/fit/commit을 포함한 process timer이며 parent allocated GPU 시간이나 전체 ETA로 바꾸지 않는다. [초기 증거](../../../../audits/servers/server4/jlz-realization-v9-ridge-bs100x5-with-exact-probe/initial-handoff.json)에 원 receipt의 path/size/SHA와 state 연결을 보존했다.

이 초기 인계에서는 A의 Q2와 두 W5를 NOT_OBSERVED로 남기고 모니터링을 중단했다. 이번 사용자 recall의 완료 확인은 그 뒤 별도 관측이며, 초기 기록을 완료 관측으로 소급하지 않는다.

상세 구현 결속은 [구현 기록](../../../../plans/updates/server4/jlz-realization-v9-ridge-bs100x5-with-exact-probe/implementation-r1.md)에 있다. Raw와 tensor는 local에 보존하고 작은 source 및 보고만 게시한다.
