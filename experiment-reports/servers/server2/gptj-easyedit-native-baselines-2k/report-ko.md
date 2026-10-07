# GPT-J stock EasyEdit MEMIT / AlphaEdit 2k — 제출 인계

Nonce: `USER-GH-SH2-GPTJ-EASYEDIT-NATIVE-BASELINES-2K-20261007-R1`.
Implementation note `USER-GH-SH2-GPTJ-EASYEDIT-NATIVE-BASELINES-2K-20261007-R1-IMPLEMENTATION-NOTE` 수신. W0 독립 등록을 먼저 완료했다. 실제 server2 / SH2 session `01a0493a-074c-7f91-9a13-769116326fef`.

## 실제 job 및 자원

|역할|Job|의존성|release 직후 단발 상태|
|---|---:|---|---|
|BASE_MEMIT|60656|afterany:60652|PENDING|
|BASE_ALPHAEDIT|60657|afterany:60652|PENDING|
|자기 CPU collector|60658|afterany:60656:60657|PENDING|

Source/config/owner/full argv/실제 script/node/QoS/resource/dependency held 검사를 완료하고 전부 release했다. 최신 admission의 own GPU frontier는 60652였다. 두 native lane은 이 frontier 후 각각1GPU로 실행되므로 project cap2를 넘지 않는다. 기존 FE/W0/다른 job을 취소·변경하지 않았다. 각 GPU job 6CPU/59392MiB/48h, collector GPU0/6CPU/24576MiB/4h. Wall은 요청 상한이지 ETA가 아니다. 실제 장치 종류/VRAM/node/QoS/admission은 local held-inspection.json에 봉인했다. 실측 peak/비용은 runner/collector가 기록하며 아직 미관측이다.

## 실행 identity와 source

- 실행 source `56d3a445553b60bf1a5e33f0e820e0699364ba0b`
- Config SHA256 `a9ff4078e0839adcee07ce3ece0c26ed1d6f0d9ff71fadfd69dc203f1866decf`
- Lock SHA256 `9c5bc040deb42f0e6dc66136edf49c5ab37e576cc25ca4feb71e2a36da7d8839`
- Local root `/mnt/raid5/janghj/ODE-edit/local/gptj-easyedit-native-baselines-2k/attempt-r1/`
- Model revision `47e169305d2e8376be1d31e765533382721b2cc1`, 로컬 원 cache 재사용. Runtime Torch2.9.1+cu128 / Transformers4.57.1, FP32/eager/eval/TF32 off/autocast off.
- 기존 EasyEdit 실제17-file import closure를 task-local private source로 byte-exact 복사·hash pin했다. 공유 native/EasyEdit/env 수정0. 현재 source는 이미 GPT-J P/cache 초기화를 지원하므로 추가 compatibility math patch0. 이는 pristine upstream 주장과 다르다; 원 HEAD와 각 imported SHA는 config에 남겼다.

Scaffold는 gpt2xl_native_baselines이나 모델 구조·native 값은 GPT-J로 새 결속했다. 28block/hidden4096/intermediate16384/vocab50400/RoPE/parallel MLP/untied head/readout27, fc_out weight[4096,16384]이다. GPT2 Conv1D/5layer/Llama PRICE geometry를 사용하지 않는다.

## 고정 방법 및 입력

두 independent cold W0 arm, 동일 ordered first2000, seed20261002, BS100×20. Native YAML L3–L8 SIX, anchor8/readout27, lr.5, 최대25 evaluation/24 Adam, clamp.75/norm.5/KL.0625, 원 total-loss early stop을 유지한다. MEMIT native coefficient15000/remaining-layer divisor/원 FP64 solve→delta.float add, caller lifetime H 없음. Alpha native FP32 projector/operator/L2=10/threshold.02/자기 H를 유지하고 post-all-layer native keys로 H를 갱신한다. Alpha YAML mom2_update_weight는 native Alpha solve에서 사용되지 않는다. BLUE kwargs/PRICE planner/controller/추가 fit·pilot·baseline 없음.

Native source의 원 context generation/cache를 첫 실제 apply에서 수행한다. Wrapper는 complete z를 미리 fitting하지 않는다. Scalar telemetry는 original compute_z return의 실제 iteration/기존 loss만 읽으며 수식·fit 횟수·Adam은 변경하지 않는다. Native z/key/solve/execute counters 및 Alpha tensor version을 관측한다. 정상20batch 기준 arm당2000fit/120solve, Alpha120history append, MEMIT0history. Failure/uncommitted work도 비용/계수에 남긴다.

Six C0 files는 각[16384,16384] FP32 stored mom2, count54924275를 확인했고 전체 finite를 CPU chunk 검사했다. Native moment는 sum/count다. 기존 GPT-J P는 [6,16384,16384] FP32/finite, slot0…5를 원 YAML L3…8에 결속했다. 원 파일 hash와 현재 stat를 봉인했으며 ours five-slot P 재사용0, 새 P/C0 계산·dump0이다. 새 spectral decomposition으로 projector 생성 과정을 재인증했다는 주장은 하지 않는다. Cold six FP32 projection hash는 모델 mmap read로 구했다; CPU 모델 forward0.

## 관측·W0 bridge·저장

Current100 pre/post, milestones W5/10/15/20 all-seen, 최종 R2000/P4000/N20000을 원 GPT-J PRICE scorer/packing으로 관측한다. Milestone에서도 current는 해당100개다. Scalar schema 9fields/pct/nats/N desired=true/strict/harmonic 및 실제 edit 축을 유지한다. W0 reference-only 축 예외는 native edited-run에 적용하지 않는다.

New W060652의 own fresh raw가 완료되어 exact model/revision/assets/runtime/scorer/token/row/config/결과 hash와 결속될 때만 scalar bridge로 재사용한다. 해당 chunk는 rows/optimizer_feedback만 있고 state가 없으므로 state-bearing BLUE raw로 취급하지 않는다. Native six-layer cold H ledger는 별개이며 H/editor resume는 하지 않는다. W0 source가 미완료/실패라면 polling하지 않고 허용 W0 관측을 자체 수행한다. 동일 raw를 모델 편집 state로 복원하지 않는다.

W&B entity wkdguswns2256 / project `layer allocation`, role scientific, model/family gptj, writer memit/alphaedit, actual job ID와 run.name을 new UUID에 결속한다. Scalar-only, no console/code/artifacts/tensors upload. W&B startup/run URL 및 actual fit/write는 아직 `NOT_OBSERVED`이다. SDK accepted와 remote delivery는 구분한다.

NoCP: W/H/RNG/delta/optimizer 영속 저장0, exact_resume=NOT_AVAILABLE. Error rollback은 RAM only다. 원자료 local KEEP/Git0/NO_BROADCAST_NOT_REQUIRED.

## CPU 검산과 인계 경계

Owner CPU8 tests PASS: 실제17file import/parser, request 변환, MEMIT original solve/no-H, Alpha6층 once-only append·두 entry 연결·reset1회, actual z counter/telemetry, logger failure isolation, state 없는 W0 raw bridge 및 launcher. Source compile/diff check 수행. 두 test 개발 오류는 예상 exception 형식/fixture chunk 크기였고 science source/허용오차 변경은 없었다. 독립 reviewer는 없으며 CPU fixture는 GPU correctness PASS가 아니다.

Collector는 독립 stored-row reducer로 분모/20commit/19state links/native counters/paired lost-gained/first500/cohort/실패prefix 및 자체 job allocation을 정리하고 report/manifest 이후 terminal을 쓴다. Scheduler 성공과 scientific completion을 구분한다. 현재는 제출 인계이며 결과 완료가 아니다.

재현 명령(자동 재제출 아님): `python -m project.run_scripts.gptj_native_baselines.preflight`; 봉인 실행은 `...gptj_native_baselines.run --attempt <root> --arm BASE_MEMIT|BASE_ALPHAEDIT`, collector는 `...gptj_native_baselines.collect --attempt <root>`이다. Own audit submission/cpu-tests/asset-checks 및 local source.tar/execution.lock.json/held-inspection.json 참조.

Release 후 단발 snapshot으로 인계했다. 이후 agent monitoring/heartbeat/automatic retry0, sealed runner/collector만 자연 진행한다. 완료 리뷰는 사용자 recall 때 수행한다.
