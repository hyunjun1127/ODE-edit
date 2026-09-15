# GH → SH4: EP-TW-1 진단 gate 생략 및 본실험 직접 제출

Instruction ID: ODEEDIT-S06-EP-TW1-DIAGNOSTIC-GATES-SKIP-RUN-SH4-V1
Nonce: ODEEDIT-GH-SH4-EP-TW1-GATES-SKIP-RUN-20260915-R1
Parent: ODEEDIT-S06-EP-TW1-C4-V3-OURS-FIRST-SH4-V1
수신: server4 / session 01a04939-b5c7-7a03-ba2d-ef3343d62cfd / CWD /data/janghj/ODE-edit / repository hyunjun1127/ODE-edit.
발신: GH / session 01a04939-8873-7673-8dca-4c7fc5e31af0.
기준 origin/main: 82cfb0ac424451306d32bff3969e3bedd4ce4b21. 실제 수정·실행 source는 새 attempt에서 따로 기록한다.

## 최신 사용자 override — 즉시 실행 권한

사용자 원문: “지금 점검 gate에서 계속 막히는데 너무 strict한 gate에서 계속 막히는데 그런거 다 제거바라고 run올리라고 해.”

이 지시는 이전 DIAGNOSIS_ONLY/STOP 및 saved-episode 기술 PASS 후에만 본실험을 허용한 조건을 이번 재제출에 한해 대체한다. SH4가 필요한 최소 코드 변경과 본실험 제출을 직접 완료한다. GH 재승인이나 별도 수치검사 PASS를 기다리지 않는다. 신규 진단 job, smoke job, FD 수렴 수리 job은 제출하지 않는다. 이미 증명된 ULP 인덱스 문제를 다시 진단하거나 해당 검사를 수리·실행하는 일도 본실험의 선행조건으로 만들지 않는다.

## 1. 실행 경로에서 제거할 검사

별도의 test-only 실행 경로를 실제로 호출하지 않는 skip mode를 만든다. 검사를 그대로 실행한 뒤 exception만 무시하거나 tolerance를 무한대로 키우는 방식은 금지한다. 특히 CUDA assert 후 계속 실행해서는 안 된다.

- saved-episode repair prerequisite와 repair_pass receipt 요구, E→D→science conditional gate를 이번 실행에서 제거한다.
- E/D의 FD grid·수렴·resolution·jitter 검사, 직접 weight-gradient 대조, 독립방향 probe, ULP 통계 및 그로 인한 종료 조건을 호출하지 않는다.
- W0 self-KL threshold gate, model-level functional/materialized 수치 parity와 추가 synthetic zero/nonzero 검증, 초기 validate_episode/scientific_checks 등 진단 전용 forward/backward를 생략한다.
- 진단 목적의 tensor exact-equality/normalization/roundoff/assertion을 science 진입 조건으로 두지 않는다. canonical evaluator와 method observer 간 부동소수점 NLL byte-equality처럼 진단용 중복 비교는 가능한 기존 계산의 warning/기록으로만 남긴다. 추가 forward를 그 기록 때문에 돌리지 않는다.
- 본래 method에서 계산한 E/D·gE/gD·선택/실제 update 등의 값과 유효성은 원래 로깅으로 저장할 수 있다. 진단 observer를 새로 붙여 실행을 지연시키지 않는다.
- 기존 수치 허용량을 사후 완화하여 PASS를 만들지 않는다. 새 lock/receipt/report에는 모든 생략 항목을 SKIPPED_USER_DIRECTED, numerical_validation=NOT_ESTABLISHED로 기록한다. 기존47942 E direct 비교 PASS나 다른 관측은 과거 해당 범위로만 보존한다.

## 2. 유지할 실행 무결성과 과학적 method

이 지시는 진단 검증 gate의 제거이지 EP-TW-1 방법의 변경이 아니다.
- EP 후보의 E 무악화/strict ID-set 보존, min D64 선택/RAW fallback, native anchor·trust·projection, gE/gD 정의, native fitter·P·M·history1·ledger·evaluator·sample order는 변경하지 않는다. quality screen은 방법 자체이므로 검사 gate와 혼동해 제거하지 않는다.
- 잘못된 모델/데이터/레이어/shape/device, 실제 NaN/Inf, source/input identity 불일치, 비선택 weight나 history의 의도하지 않은 변경, 실제 저장 실패·공간 부족 등 실행 자체를 무효화하는 오류만 기존 최소 hard stop으로 유지한다. 추가 parity gate를 이 이름으로 다시 도입하지 않는다.
- 불가능한 인덱스/실제 CUDA assert/OOM/import 오류는 진단 skip과 별개인 실제 실행 오류다. 첫 오류를 보존하고 가짜 정상 종료나 임의 데이터를 만들지 않는다.
- 평가·checkpoint·commit 기록, 정상 B1→B10 state 전달, 실패한 열린 batch의 기존 rollback은 유지한다. 이미 유효하게 실행 중인 다른 job/source는 변경하지 않는다.

## 3. 제출 범위와 경로

fresh pretrained W0 / cold M0 / 기존 fixed CounterFact first1000 / B100×10 / Llama L4-only 단일 EP-TW-1 scientific replacement를 직접 제출한다. commit0인47884/47942에서 B2로 이어 실행하지 않는다. 새 scientific B1의 정상 native target/solve 계산은 필요하지만 saved-episode 진단용 재계산은 없다.

기존192 W0 teacher, C4 reference768, model/tokenizer/P/context/fixed sample 자산은 재사용한다. teacher 재생성, N4 calibration, baseline editing, 다른 method/다른 후보/추가 10k run은 없다.
기존 v3 및 repair 계약 중 위 gate·진입조건만 이번 사용자 override가 우선하며 나머지 method/자료/자원 범위는 유지한다.

새 clean codex/server4-ep-tw1-gate-skip-run-v1 branch/worktree와
local/ep-tw1-c4/20260915-v1/gate-skip-r1/를 사용한다. 기존 attempt-v1, repair-r1, 실패 보고·source·473/69 GPU-sec와 teacher98초를 덮어쓰거나 지우지 않는다.

허용 source write:
project/run_scripts/bg_tw_reference/ep_tw/ 내 runner.py/technical.py/model_adapter.py/control.py/preparation.py/operations.py/handoff.py/run.sbatch/repair_control.py/repair_runtime.py/repair.sbatch/repair_handoff.py/README.md/REPAIR.md의 skip mode·연결·기록 및 직접 관련 테스트/새 launcher.
policy.py/ledger.py의 과학적 의미, native vendor/fitter/native_map/teacher 구현, 공유 환경/원본 모델·자산/타task는 불변. 새 lock은 이전 검증 required 필드를 새 skip status로 정직하게 바꾸고 과거 immutable lock을 수정하지 않는다.

허용 기록:
- audits/servers/server4/2026-09-15-ep-tw1-gate-skip/
- experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/gate-skip-r1/
- messages/acks/server4/2026-09-15-ep-tw1-gate-skip.md
- messages/server-heads/server4/2026-09-15-ep-tw1-gate-skip.md
- runs/odeedit_ep_tw1_gate_skip_s4_v1/
- tasks/status/odeedit_ep_tw1_gate_skip_s4_v1/server4.json
- 위 새 local root와 자신의 새 분석 worktree 안 raw-free source/archive/receipts.

## 4. 자원·최소 준비·인계

Slurm submission ALLOWED. 기본 job명 odeedit_ep_tw1_nogate_s4, 단일1GPU/8CPU/명시mem60416M/exportNONE/Requeue0/12h. GPU hour cap=null. server4 전체 프로젝트 active+admitted pending을 포함해 cap2를 지킨다. cap2는 상한이며 GPU를 채우려고 같은 chain을 복제하지 않는다. 자리 없으면 cap 내에서 안전하게 PENDING 등록한다.

최소 준비는 source 실행/skip routing의 CPU import·syntax 또는 작은 mock, 정확한 input/모델/샘플/출력 경계와 cap/memory/disk 확인이다. 이전 수치검사·86 fixtures 재완주·별도 red worker의 numerical PASS는 제출 선행조건이 아니다. 필요한 작은 체크는 SH4가 직접 처리한다. red 수치검증 block은 본 지시 및 아래 GH waiver로 user-directed skip 처리한다. 실제 resource/source/data 무결성 오류만 중단한다. GH의 중복 raw/GPU 감사는 기다리지 않는다.

Source/새 validation mode·job args freeze → held owner/source/resource inspection → release 후 compact job/lock/output/skip 목록을 인계한다. Pending이면 즉시 대기한다. 즉시 실행되면 최초 실제 정상 batch commit/state 전달을 짧게 확인할 수 있으나 새로운 모델 수치 gate를 추가하지 않는다. 그 관측은 INITIAL_EXECUTION_OBSERVED_WITH_VALIDATION_SKIPPED이며 예전 numerical G0_PASS가 아니다. 시작이나 최초 batch를 장기 기다릴 필요 없이 제출 인계 후 WAITING_USER_RESUME 가능하다.

제출된 단일 프로그램은 B1–B10/평가/저장을 자연 수행한다. agent polling/heartbeat/terminal 대기/자동추가제출/자동상세분석은 인계 후 하지 않는다. 사용자 완료 recall 때 상세 보고를 진행한다. scoped code+compact override/제출 보고의 non-force main push는 허용하되 제출 지연의 새 과학 gate로 만들지 않는다. 기존 source/실패 bytes와 다른 main 작업을 보존한다. Raw/tensor/prompt/teacher/gradient/full stdout Git0. NO_BROADCAST_NOT_REQUIRED.

GH는 “실행 승인”과 “수치 검증 완료”를 구분한다. FD 수렴은 아직 미확인이라는 사실을 결과에 남기고 과학적 효능/비교 결론을 미리 확정하지 않는다.
