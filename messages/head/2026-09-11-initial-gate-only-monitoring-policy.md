# GH → 모든 SH: 초기 gate 이후 agent 작업 중지, 사용자 호출 시 재개

policy_id: ODEEDIT-INITIAL-GATE-ONLY-USER-RECALL-20260911
nonce: ODEEDIT-GH-ALL-SH-INITIAL-GATE-PAUSE-20260911-R1
사용자 최신 원문: “각 실험들은 초기 GATE 모니터링만 실시하고 TASK 중지하고 실험 종료에 USER의 호출로 TASK 이어서 진행하는 것으로 하자.”

## 적용 범위와 우선순위

- SH1/SH2의 현재 multilayer A/B 및 이후 모든 SH 실험 task의 기본 진행 정책이다. 사용자 새 override 전까지 유지한다.
- 현재 A instruction ODEEDIT-S06-MULTILAYER-JOINT-EDIT-A-SH1-V1, B instruction ODEEDIT-S06-MULTILAYER-DAMAGE-COMPENSATION-B-SH2-V1에 즉시 적용한다.
- 이전 지시의 agent 자율 terminal monitoring·후속 제출·분석·최종 report/main 통합 연속 진행 부분보다 이 최신 사용자 지시가 우선한다.
- 과학 scope/source/arm/sample/checkpoint/자원 cap을 바꾸는 지시가 아니다. GPU 실험을 취소하라는 뜻도 아니다.

## 초기 gate까지

- 아직 초기 gate 전이면 승인된 필요한 구현/CPU검증/공통 input seal/수신검증/제출 및 최소 actual initial validity 확인까지 진행한다. 준비 단계만으로 GPU PASS라고 하지 않는다.
- 기존 설계의 source/asset/실제 weight 적용·finite/수치 정확성에 필요한 최소 gate를 적용한다. 전체 Middle table/전 arm/전체 epoch/terminal까지 기다리는 것을 초기 gate로 확대하지 않는다.
- 현재 A/B의 shared fixture/API 전달과 receiver 확인은 해당 실행의 초기 gate 준비에 포함한다. 공통 준비만 끝났는데 본 실행 initial gate가 끝났다고 하지 않는다. 서로의 완료 실험 결과는 선행조건이 아니다.
- Initial gate 중 기술 오류라면 기존 승인된 범위의 최소 technical repair/새 attempt 후 확인까지 허용하며 과학조건/성능 gate 변경0.
- 자원 대기로 시작하지 않으면 pending 상태/미실행 gate를 기록하고 사용자 호출을 기다린다. resource wait를 위한 sleep-loop/heartbeat/새 automation을 만들지 않는다.

## 초기 gate 확인 직후

- status=MONITORING_PAUSED_AWAITING_USER 를 남기고 해당 agent turn/task 작업을 종료한다.
- 정확한 job/source/lock/output/최초 valid evidence/남은 범위/pause receipt를 한 번만 보고한다. 전체 실험 완료로 보고하지 않는다.
- GPU job 및 이미 승인·제출된 pending/dependency/원래 실행 프로그램의 실험·평가·checkpoint 저장은 그대로 계속한다.
- Running/pending job cancel/hold/restart/requeue/throttle/source/config/evaluation schedule 변경0.
- Agent scheduler/result/log polling, sleep-loop, heartbeat/callback/자동 깨우기, terminal 완료대기, 추가 관측/분석/그림/report확장/main통합을 중지한다.
- 초기 gate 후 새 후속 job·추가 arm·관측 job 제출 또는 다음 stage로의 agent 자동 전환도 중지한다. 이미 제출한 정상 프로그램의 봉인된 내부 순서는 바꾸지 않는다.
- 이미 initial-valid가 확인된 task는 중복 gate나 추가 점검 없이 즉시 pause한다.

## 재개

- 실험 종료 여부를 USER가 GH에 알리거나 USER가 명시적으로 재개를 요청하면 GH가 해당 SH에 recall을 전달한다.
- 그 호출 이후에만 bounded terminal 확인→미완료 승인 scope 실행/검산→분석→최종 보고서 및 own-scope main 통합을 이어간다.
- GH도 대신 반복 polling하거나 자동 recall하지 않는다. 진행 중 사용자 질의에는 확인된 범위만 답한다.
- 다른 완료/중지 task, stopped ORBODE audit, checkpoint archive 등에 대한 작업은 새로 시작하지 않는다.
- SH4는 현재 이미 STOP 상태라면 정책만 local 보존/ACK하고 종료한다. 이번 메시지는 새 실험·점검 배정이 아니다.

각 SH는 shared PROTOCOL/다른 SH 파일을 수정하지 말고 task-local override/pause receipt에 기록한다. 자신의 현 task에 새 지시를 적용한 짧은 ACK만 전달한다. 이 정책을 이유로 실험 정확성 gate를 PASS로 위조하거나 남은 결과를 완료로 표시하지 않는다.
