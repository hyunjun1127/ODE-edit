# EP-TW-1 — 사용자 지시 진단 생략·본실험 제출

Instruction `ODEEDIT-S06-EP-TW1-DIAGNOSTIC-GATES-SKIP-RUN-SH4-V1`. **WAITING_USER_RESUME**, validation **SKIPPED_USER_DIRECTED**,
numerical_validation **NOT_ESTABLISHED**. G0_PASS 또는 1000요청 완료 보고가 아니다.

## 제출 사실

단일 **47962 / odeedit_ep_tw1_nogate_s4** held→owner/source/args/resource 검사→release.
마지막 관측 `2026-09-15T05:32:06.139575+00:00`: **PENDING, Reason=None, RunTime0**.
그 뒤 scheduler/log/result/첫batch/terminal 조회를 하지 않았다. Pending 이유를 자원부족으로 추정하지 않는다.
프로그램은 B1–B10을 자연 실행하고 agent는 사용자 명시 호출을 기다린다.

Fresh pretrained W0/coldM0, Llama revision `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, FP32/eager,
L4 down_proj / P physical4→asset0→local0 / native BLUE singleton L2=1.
Fixed10k 공식 loader 전체 asset 검증 후 first1000/B100×10 동일 order를 사용한다.
전체 root5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729,
prefix40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd.
실패 commit0에서 B2 resume하지 않는다. 기존 teacher192/C4 reference768/model/P/context 재사용.
새 teacher, N4 calibration, baseline 편집, saved-episode 진단, 다른 policy, 추가 scientific chain은 모두0.

## 생략과 유지

Saved-episode/repair_pass/E→D 조건부진입, FD/grid/convergence/resolution/jitter,
direct-gW/독립방향/ULP, W0 self-KL gate, functional/materialized/synthetic parity,
validate_episode/scientific_checks 및 redundant native-map RHS 비교는 **실제로 호출하지 않는다**.
Exception 무시·tolerance 확대·CUDA assert 이후 계속실행이 아니다.
Canonical↔method NLL/strict 비교는 이미 계산한 rows의 warning이며 추가 forward0.
원 .15 FD 규칙을 PASS로 바꾸지 않고 FD 수렴과 전체 model 수치 검증은 미확립으로 남긴다.

Method E≤Ep(양의 허용량0), 정확 strict ID-set 보존, minD64 선택/RAW fallback,
별도 gE/gD, native anchor/ball/trust, target quota, P/M/history1/accepted-ledger는 그대로다.
policy.py/ledger.py 및 non-EP helper/native bytes 불변을 source identity로 확인했다.
Shape/device/dtype/finite/source/data/order, 비선택 weight/history·observer mutation,
실제 저장/selected restore/chronological commit→next-entry hard stop은 유지한다.
CP/evaluation/rollback 및 B1–B10 내부 순서는 변경하지 않는다.

CPU routing mock5/AST/shell syntax PASS는 **수치검증이 아니다**. 기존86 fixture/GPU진단 재실행0.
새 source/input 2085 소형 fullSHA와 기존34 heavyasset fullSHA+현재 stable stat을 결속했다.
새 teacher/model 전체 재해시·재생성0. 과거47942 E direct PASS는 그 단일 범위의 관측으로만 보존한다.
본실험이 첫B100/B2를 완료하면 marker는 INITIAL_EXECUTION_OBSERVED_WITH_VALIDATION_SKIPPED이며
기존 numerical G0_PASS가 아니다. 현재 marker는 미관측이다.

## 자원·비용·보존

기존 project active/admitted GPU0 + 신규1 =1≤cap2. 1GPU/8CPU/**60416M**/exportNONE,
Requeue0/12h, GPU-hour hardcap=null. 다른 job/source/cap/throttle 변경0.
제출 직전 free **118212673536 B**, free inodes 226305287.
예상30GiB + 안전여유20GiB 확인; 독점 예약은 아니다. 10개 W4/M4 tensor 예상10,569,646,080B.
과거2–8 GPUh 계획값은 신규 실측이 아니며, 진단 생략 후 속도 향상 근거로 사용하지 않는다.
원47884 **473 GPU-sec**, repair47942 **69 GPU-sec**, teacher47592 **98 GPU-sec**는 별도 보존한다.
nativefit286.5445초는 원473초의 중첩 component라 더하지 않는다. 신규 할당시간은 아직 미관측이다.
기존 attempt-v1/repair-r1/실패·부분표·locks 불변, 원본 삭제/overwrite/타task재개0.
Raw prompt/teacher/gradient/tensor/checkpoint/fullstdout local-only. PNG 생성0.

## 재현·인계

실행 source `6d317bdb2660d7e9919bc3a9fb878564e9729e37`, tree `6db14423a207eaccad195ce291c40bb8ce93cc8d`.
Archive `2dec619e12f72e1cf6fce843c61adc68726ddc025a6b09cff326bb4b6229124a`, lock `5a19c2be919362d08b2ea80e69a406d7b6de569aade8de639036f69db5d5d8f9`.
Source `/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/gate-skip-r1/source-v1`; output `/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/gate-skip-r1/scientific-v1`.
Resume `/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/gate-skip-r1/resume-manifest.json` SHA `d66d4acf77bdc45f36945308f4d691ad2bff894e60d0f439f9f64e3802fbe637`.
허용된 구현은 ep_tw 내부 skip routing만이며 actual frozen source와 이후 publication commit은 구분한다.
CLI/작은 routing 검사/launcher는 ep_tw/README.md. 새 실행명령은 sealed held-submission receipt에 있다.
`monitoring_active=false`, `automatic_resume=false`, `resume_trigger=explicit_user_call`.
실행승인과 수치검증완료/효능 주장을 구분한다. scientific_promotion=false.
NO_BROADCAST_NOT_REQUIRED, 사용자 recall 전 결과해석·후속제출0.
