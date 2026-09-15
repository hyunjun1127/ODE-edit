# EP-TW-1 v3 — 초기 준비·제출 인계

Instruction `ODEEDIT-S06-EP-TW1-C4-V3-OURS-FIRST-SH4-V1`. 상태 **G0_PENDING / WAITING_USER_RESUME**. 실제 G0·1000요청 완료 보고가 아니다.

## 마지막 확인과 실행 범위

단일 persistent job **47884 / odeedit_ep_tw1_s4**를 held검사→release했다.
마지막 관측 2026-09-15T03:13:54.989677+00:00은 **PENDING, Reason=None, RunTime0**이다.
새로 release된 시점의 scheduler 이유만 기록하며 자원 소진 또는 실제 시작을 추정하지 않는다.
그 후 scheduler/result/log polling을 하지 않았다. 프로그램은 그대로 두며 agent만 사용자 호출을 기다린다.

EP-TW-1 하나, pre-edit W0/coldM0, Llama3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2,
L4 down_proj/Pstack index0→singleton0, native BLUE-style L2=1, fixed10k first1000/B100×10이다.
공식 loader가 전체 fixed10k bytes/order를 검증한 뒤 prefix1000을 사용한다. W50/W90/history 반입·shuffle·baseline editing0.
whole root5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729,
prefix root40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd.
원 BG-1 CALIBRATION_MISSING/V1source/teacherlock은 불변이다. 새 EP에는 calibration 입력·old selector가 없다.

## 준비와 검증 수준

|항목|현재 증거|
|---|---|
|FULL_READ|정본6문서/envelope/dispatch/PROTOCOL 및 native/fitter/map/teacher/평가·transaction source|
|C4|기존768문서/token identity 재사용, 재선정·재다운로드0|
|Teacher47592|COMPLETED0:0, 24/24 fullSHA·size·FP32[8,128,128256]·finite 및192 source-ID 검산|
|Teacher 실물|12,608,080,896B, 기존98 allocated GPU-sec; 신규teacher0|
|CPU|51 deterministic fixture PASS, shell 및 explicit60416M audit PASS|
|P mapping|CPU weights_only selectedFP32[1,14336,14336], physical4/source0/local0 SHA 일치|
|Red|known source BLOCK0; 실제 Llama correctness PASS 아님|
|Actual selfKL/gradient/materialization|NOT_OBSERVED; 첫B100 프로그램 내 기술검사 예정|
|B1 commit/history1/B2 link|NOT_OBSERVED; G0 marker는 해당증거 뒤에만 생성|

기존 source의 max24Adam/25loss/early-stop/clamp와 native RHS solve를 유지한다.
Raw Vp는 실제 저장 FP32 native endpoint이고 factorized RA로 대체하지 않는다.
별도 gE/gD를 평균 누적하고 residual projection/anchorball/executabletrust 이후 RAW/C1/C05/C025를 독립 materialize한다.
E≤Ep **양의 허용량0**, 정확한 strict ID subset을 만족하는 finite 후보 중 D64 최소이며 numerical ambiguity는 RAW 우선이다.
후보 보정이 invalid이면 RAW native를 commit하고 parent rejection하지 않는다. Native 자체 비유한 것은 technical failure다.
첫B1 FD가 불충분하면 G0 기술 미검증/실패로 남기며 성능 gate로 대체하지 않는다.
FD15% 허용은 사전 고정한 기술 근사 검사의 해상도이며 current quality 허용량이 아니다.

수정 후 검사한 edge는 비유한 gradient→RAW_ONLY, 비유한 corrected observation 제외/Vp복원,
동일-byte excluded root의 aliases 전파다. Actual native kernel·기존 helper bytes는 수정0.
Accepted ledger는 최종 strict 성공 요청만 받고 실패 intent가 과거 accepted target을 supersede하지 않는다.
P/N·accepted-old·Dev는 observer 전용이다. Innerhistory0/finalizer1, 매 batch W4/M4/context/RNG/ledger CP와
CPUreload/실제 selected-state 복원 및 다음 entry identity를 프로그램에 넣었다. GPU continuation replay는 별도 미검증이다.

## 자원·비용·보존

제출 직전 기존 project active/admitted0 + 신규1≤cap2. 다른 사용자 작업·기존 job 변경0.
1GPU/8CPU/mem60416M/exportNONE/no-requeue, wall12h, GPU-hour hardcap=null.
2–8 GPUh/신규30GiB는 사전 estimate이며 실측이 아니다. 12h는 상한 estimate8h에 reserve1.5를 둔 scheduler wall이다.
예상10 selectedW4/M4 CP tensor10,569,646,080B. 실제 제출 가용123,141,615,616B; 공간 독점예약은 아니다.
새 실행 actual wall/GPU/hostpeak/성능은 아직 미관측이며 기존 teacher98s를 새 editing 비용에 가산하지 않는다.
소스·원자료·다른 paused task는 불변. Native proposal/candidate/gradient/teacher/prompt/checkpoint/fullstdout은 local-only.
NO_BROADCAST_NOT_REQUIRED; 새 모델/서버간 raw전송/삭제0. PNG 생성0, 이후 필요시 직접코드만 사용한다.

## 재현·호출 시 재개 경로

실행 source `3fb0bfb27773ec9d016e76fcdc978dc996f010a7`, tree `46dfaff451159cefef1230372e54a93e01c58ef3`.
Archive `69194e1804d8e2f31ecd802c7426770f747f2ba588a0684019bf944c1eee403b`, execution lock `306ad09b56d470ce47562bd3f36230d7f96fe258769dc0fb5b3dc192c33d954b`.
Frozen root `/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/attempt-v1/source-v1`; output `/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/attempt-v1/scientific-v1`.
Resume `/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/attempt-v1/resume-manifest.json`, SHA `f890b8955729caa27480961813f8ae4638fa51f8200adbc746185966f5640b4c`.
실행 명령/검증 CLI는 `project/run_scripts/bg_tw_reference/ep_tw/README.md`에 있다.
실제 결과파일 존재 여부는 pending 이후 추가 관측하지 않았다. 다음 사용자 recall에서 정확 job/terminal과 G0 증거를 구분해 확인한다.
`monitoring_active=false`, `automatic_resume=false`, `resume_trigger=explicit_user_call`.
Report/Audit/MMLU/다른 policy/전체10k 및 baseline C4 forward는 연기, scientific_promotion=false.
