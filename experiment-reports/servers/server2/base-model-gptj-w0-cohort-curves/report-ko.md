# GPT-J cold W0 cohort curves — 제출 인계

Nonce: `USER-GH-SH1-SH2-W0-COHORT-CURVES-RERUN-20261007-R1-SERVER2`.
2026-10-07, 실제 server2 / SH2 session `01a0493a-074c-7f91-9a13-769116326fef`.

## 실제 등록 상태

|역할|Job|의존성|release 직후 단발 상태|
|---|---:|---|---|
|신규 cold W0 GPU 평가|60652|없음|PENDING|
|CPU collector|60653|afterany:60652|PENDING|

두 job의 held owner/실제 전체 argv/source/config/node/CPU/memory/GPU/dependency/exportNONE/Requeue0 검사와 release를 완료했다. 이 표는 release 직후 관측이며 실험 완료나 이후 상태를 주장하지 않는다. GPU job은 1GPU/6CPU/59392MiB/4h, collector는 GPU0/6CPU/24576MiB/4h이다. 4h는 요청 wall이며 ETA가 아니다. 기존 60147/60148은 정확 owner/source receipt와 단발 sacct로 COMPLETED 0:0을 확인했다. 기존 source/raw/job 변경·취소는 없다.

## 봉인 identity

- 실행 source: `b717cc0daabfc44fa9714cbb88ab86bfd9aaea23`
- Config SHA256: `1723f3eea5c28f59c516a021bafb6faf68fbe46a55296a6cf4ee12c405220b48`
- Lock SHA256: `85a725d8284a8330ba8abca597de4ab236259df837684a1339adf9ecaa1d83dc`
- Local root: `/mnt/raid5/janghj/ODE-edit/local/base-model-gptj-w0-cohort-curves/attempt-r1/`
- Model: EleutherAI/gpt-j-6B, revision `47e169305d2e8376be1d31e765533382721b2cc1`.
- 기존 full model/input SHA와 현재 size/inode/mtime를 결속해 재사용했다. 기존 관측 raw를 새 평가로 대체하지 않는다.
- FP32/eager/TF32 off/autocast off, native GPT-J scorer/packing, physical MB2, seed20261002. C0/P/H/fit/solve/edit/CP=0.

## 새 측정과 곡선 의미

Runner는 cold W0 first2000을 한 번 평가한다(R2000/P4000/N20000, 26000 pair/52000 candidate). 자신의 신규 per-case raw를 CPU로만 재집계한다. Ordered occurrence 100개씩 current/post 20점, prefix500/1000/1500/2000 all_seen/post 4점이다. Full2k summary를 current로 복사하지 않는다. 전체 W0 x0을 먼저 기록하고 cohort x100…2000 순서를 검산한다.

`edits=reference_cohort_edits`는 비교 cohort 축이다. 실제 model/applied/pre/post state는 항상0이다. immutable config에 reference_only=true, evaluation_model_state=W0, edits_axis_semantics=reference_cohort_progress, w0_reference_schema=w0-cohort-v1을 결속한다. R/P desired=new, N desired=true; pct/nats/분모/strict/harmonic을 보존한다. Token micro는 정수 numerator/denominator 재합산이다.

원 shared validator는 변경하지 않았다. 이 승인에 한한 전용 `gptj_server2_cohort_tracking`은 기존 transport/privacy/identity 구현을 복사하고, 정확 W0-only config·state0·cohort schedule·필수 scalar를 검증한다. 일반 edited-run validator의 POST_STATE_AXIS는 그대로다. 새 UUID와 실제 Slurm job/name/config를 startup에 결속한다. Finish 시 단발 bounded history scan으로 1 full W0+20 current+4 all_seen를 확인하도록 구현했다. SDK 접수와 remote verification은 별개이며 transport 거절을 local receipt에 기록한다.

## 검산 및 한계

CPU7 tests PASS: ordinal/서로 다른 cohort, token micro, N desired, state0 예외 한정, axis 순서, job array0/signed step, fake-SDK 전체 coverage/readback, no-write와 launcher. 별도 독립 reviewer는 없고 owner 검산이다. Fixture의 fake job ID는 온라인 업로드하지 않았다. CPU tests는 actual model/online PASS가 아니다.

W&B startup/run URL, 실제 GPU 평가, 실제 remote curve delivery는 `NOT_OBSERVED`이다. Sealed runner와 collector가 자연 진행한다. 장기 polling/heartbeat/automatic retry는 없으며 사용자 recall 전 능동 모니터링은 중단한다. NoCP, raw local KEEP, NO_BROADCAST_NOT_REQUIRED. 원 dirty root와 다른 task는 보존했다.

재현 명령(새 submission 재실행 권한 아님): `python -m project.run_scripts.base_model_eval.gptj_server2_cohort_tests`; 준비·등록 기록은 own audit의 submission.json/cpu-tests.json 및 local held-inspection.json/execution.lock.json에 있다. 등록 프로그램은 `gptj_server2_cohort run/collect --attempt <위 root>`를 사용한다.
