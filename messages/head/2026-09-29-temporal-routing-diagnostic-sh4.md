# GH → SH4: BASE checkpoint temporal-routing diagnostic

Instruction / ACK nonce: `ODEEDIT-GH-SH4-TEMPORAL-ROUTING-DIAGNOSTIC-20260929-R1`.
사용자: “/mnt/raid5/janghj/ODE-edit/plans/global/2026-09-29-temporal-routing-diagnostic-v1/design-ko.md 이거 SERVER4에서 TASK 진행시키자.”
수신 server4 / session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, CWD `/data/janghj/ODE-edit`, repo hyunjun1127/ODE-edit.
발신 GH `01a04939-8873-7673-8dca-4c7fc5e31af0`.

## 1. 실제 실행 권한과 정본

설계 정독 → 자산 결속 → 새 runner/계측/저장 구현 → 최소 기술 확인 → 승인 15개 branch 정상 제출을 수행한다. 계획 작성만으로 종료하거나 GH의 재승인을 기다리지 않는다. CPU/design PASS를 실제 Llama PASS로 확대하지 않는다.

정본 root `plans/global/2026-09-29-temporal-routing-diagnostic-v1/`의 11파일을 전체 읽고 SHA/size 검증한다. design SHA `9cfc6dd9fe0d580950d96c92ce21fbeafbe046cad44d8b718fed5b2ae333141f`, contract SHA `f55e8ba2028f53d52b7dfa9e76c5ebfa15c0303ec6a0eb19b2314c490c27da0b`. CSV의 sealed CRLF를 임의 정규화하지 않는다. 설계의 DESIGN_ONLY/NOT_EXECUTED는 작성 당시 상태이며 본 실행 승인을 막지 않는다.

연결 native 본체, 기존 singleton adapter, damage-signals 문헌 메모까지 `audits/global/2026-09-29-temporal-routing-sh4-dispatch/input-manifest.json`에 결속한다. 문헌 메모의 추가 방법 제안은 참고이며 이번 실험 범위를 늘리지 않는다. PROTOCOL, 실제 registry/경계, 자원 정책 및 원 BASE_ALPHAEDIT source/config/context/P/tokenizer도 확인한다.

## 2. 고정 과학 범위

BASE_ALPHAEDIT B010/B050/B090 각각의 **다섯 weight와 다섯 history Gram 전체**에서 출발한다. CP마다 L4/L5/L6/L7/L8 고정 singleton 5개, 총15개 독립 branch. 각 branch는 같은 100개 요청을 BS1으로 실제 순차 누적하며 scientific fit1500회다. 기존 W0→1k/5k/9k 편집 재실행0. 다음 branch는 부모 CP W/M와 RNG/context를 다시 복원하고 앞 branch endpoint에서 시작하지 않는다.

`continuation-ids.csv`가 실제 순서다. 일반 fixed10k의 first100이나 CP 직후 원래 batch로 대체하지 않는다. 동일 dataset/sample lock에서 ordinal9000 이후 metadata-only eligible fresh subject 100개를 선정한 상대순서이며 모든 branch에 동일하다. 모델 성능 기반 재선별/실패 요청 제외/다른 층 재시도0.

Native AlphaEdit layers=[physical L], **blue=false/L2=10/scale1**, fresh selected-layer z, target cache off, residual divisor1. FP32/eager/matmulTF32false/cuDNNtrue, 원 context, v_steps25/lr.1/decay.5/clamp.75/KL.0625를 유지한다. P와 history의 physical slot L−4를 singleton slot0에 정확히 연결한다. 매 write 후 선택층 native post-write key Gram을 한 번만 append하고 나머지4 history는 불변이다. Finite 기능 실패도 적용된 W/history와 평가 분모에 남긴다.

기존 BLUE/L2=1 강제 adapter를 그대로 쓰지 않는다. 원 native 본체 SHA `a4d24f793b2a711c7c415894b98f2bbe1c51661a862f61add4570d76a38cdb89` 및 실제 dependency/import closure와 새 adapter를 분리 봉인한다. 이번 승인에 별도 z batching/BF16/clipping/hook 최적화를 끼워 넣지 않는다. 다른 완료 task의 numerical record-only waiver도 자동 상속하지 않는다.

Step1만 같은 CP/request의 층간 비교다. Step2부터는 다른 누적 trajectory 비교이며 live optimal routing 인증이 아니다. 정책 학습/random routing/추가 EN보정/R512/strength sweep/fullSVD/Hessian/KFAC/추가 long chain은 승인하지 않는다.

## 3. 자산 재사용과 제한 전송

S4 historical-timeaxis/E3/alpha-key 입력에 이미 있는 같은 BASE CP를 우선 조사한다. checkpoint-bindings.csv의 exact family/B010/B050/B090/bytes/SHA 및 W/M·원 source·P·context·dataset을 결속한다. 기존 검증 receipt와 현재 실물 증거 수준을 구분하고 서로 다른 hash convention을 직접 비교하지 않는다. 기존 자산이 일치하면 대형 재전송0.

필요한 CP가 S4에 없을 때에만 동봉 transfer approval의 server2 **정확3파일 중 누락분**을 S4 task 전용 input 경로로 sole-writer pull/SHA 검증한다. SOURCE_KEEP/no overwrite/no delete; 다른 CP나 live task 출력의 재귀 복사0. 이번 실행 사용자 권한 안의 필수 입력 확보이며 동일 승인을 다시 요청하지 않는다. 모델/P/context/data는 기존 S4 승인 자산을 우선 재사용, 재생성0. 원 parent model/세 CP payload는 저장된 결과 복원에 필수이므로 보존한다.

## 4. 필수 계측과 해석

정본 panel-ids 120행: base sensor16/observer32 공통, history sensor8/observer16은 CP별. Sensor/observer 및 subject 분리, history 최신 유효 relation/oldest-newest와 entry-correct 분모를 유지한다. Paraphrase/observer를 writer 학습·선택에 넣지 않는다.

매 edit: target/anchor/value norm, actual dW norm, base sensor A/B/C 및 base/history F, canonical+2P NLL/margin/target-prefix strict. Raw와 질문별 정규화 평균을 함께 보존한다. A/B/C는 정본 그대로, B/C signed 유지. Base F는 KL 차이, history F는 유효 목표 NLL 차이로 따로 보고한다. KL은 answer-prediction token mean→question mean. 수치 안정 epsilon 및 random-patch seed 등 미명시 구현 상수는 결과를 보기 전 결정·lock하고 근거를 남긴다. 새 threshold나 판정 규칙을 과학 결과에 맞추지 않는다.

상세 n=0/1/10/25/50/100: fixed observers, 누적 continuation retention, 현재 fixed10N, selected/downstream key/readout/MLP/residual norm. milestone write **직전** observer도 평가하여 marginal과 interval을 구분한다. 동일 teacher-forced prefix/valid token 위치를 결속하며 TF를 free generation accuracy라 부르지 않는다. 같은 forward에서 가능한 true/new NLL·preference와 TF token-micro/prompt-macro/strict 및 분모를 분리 저장한다.

W0→CP, CP→n, n−1→n을 구분한다. 기존 history는 CP-entry 성공 집합의 손실/회복, 신규100은 실제 at-write 성공의 retention과 acquisition failure를 별도로 계산한다. 선택층 key는 branch-entry와 불변이며 Kt−K0는 상속된 BASE shift다. continuation이 만든 새로운 key drift는 downstream에서 본다. Sensor와 observer 상관 및 품질 조건부 보조 비교는 같은 step·층쌍을 독립 반복으로 세지 않는다.

Patch는 각 CP L4 branch n100의 L8에만 정본 `−D8(K8,n100−K8,entry)`; native/zero/targeted/norm-matched fixed-random, 최대3사건·사건당 추가3패널pass/refit0. Base/history 회복과 신규100 retention을 함께 측정한다. L8-only key-invariance zero negative control을 기존 capture와 결속하고 별도 patch sweep으로 확대하지 않는다. Zero-hook parity 실패 시 인과 해석하지 않는다.

## 5. 명시 저장 예외 / 기술 경계

**이번 설계는 기본 no-checkpoint의 명시적 task 예외다.** 15branch 각각 n50/n100에 실제 적용 selected-layer full FP32 [4096,14336] tensor를 저장한다. 총30개, payload7,046,430,720B(6.5625GiB)+header/metadata. factor/norm/hash/delta만으로 대체0. n0은 부모CP 참조; 다른 시점 fullweight/fullmodel/editor bundle 자동 저장0.

독립 CPU copy → temp file → atomic rename → reload bitwise 및 reconstructed-model output parity, nonselected parent weights 불변 확인. 원모델→부모 five weights→선택층 overlay 재구성, 부모 revision/CP/fileSHA/ordered IDs/last case/source/config/tokenizer/context metadata 필수. Patch는 native snapshot을 수정하지 않는다. **model reconstruction용이지 editor-state resume이 아니므로 exact_editor_resume=NOT_AVAILABLE**(중간 H/RNG 미저장).

최소 CPU/실제 integrated 검산: full-parent restore와 slot mapping, L2=10 singleton/native update/history1, selected key invariance, nonselected W/H 불변, observer 비변이, input/order/source/finite, 실제 snapshot IO/reload. 기술 duplicate native fit은 최대2회, 전체 fit≤1502/loss≤37550/Adam≤36048. 기존 설계검사 PASS를 GPU PASS로 바꾸지 않고 별도 대형 반복/FD 캠페인이나 성능 개선 gate를 만들지 않는다. 수치 비교는 기준·실측·초과·문항을 먼저 저장, 설계의 실제 불변성/restore/parity 요구를 허위 PASS로 우회하지 않는다.

낮은 품질/망각/음성 patch는 정상 과학 결과다. NaN·identity/복원·IO/구조 오류는 해당 branch 기술 실패로 보존하고 collector가 누락을 드러내야 한다. 원오류를 보고 후 범위 내 명백한 코드/serialization 오류의 최소수리 가능하나 수식·허용치 변경이나 무제한 branch 반복-to-PASS는 금지한다. 모델-only snapshot으로 editor 재개를 가장하지 않는다. fit 예산이 추가로 필요하면 근거를 보고하고 확대하지 않는다.

## 6. 자원·등록·모니터링 경계

현재 server4 **project cap2**, 본 task 최대2GPU. 기술/prep/다른 admitted capacity까지 합산한다. 각job1GPU/8CPU/host≤60416MiB, server4/gpu/exportNONE/Requeue0. 사용자 타job 취소/이동/선점0. 15branch는 단일 writer별 분할 및 array%2/동등2lane으로 모두 등록하고 GPU 자리가 없으면 cap-safe pending으로 유지한다. CPU collector는 afterany로 성공/실패/누락을 수집하되 scheduler COMPLETED와 scientific completeness를 구분한다.

S4 free disk/RAM·CPU·source/dependency/assets 현재 상태를 확인한다. CP 재사용 여부, FP64 solve peak, model+fivehistory/P·capture scratch, 30snapshot+raw/atomic temp+여유를 포함한 실제 resource plan을 봉인한다. 선형 ETA/계획/wall request/실측 GPU초를 구분한다. 임의 삭제·공간 waiver/평가 축소0.

기존 사용자 초기 gate 후 pause 운영을 유지한다. **15개 branch 전량 정상 등록·held inspection·release + 대표 실제 branch step1 commit/history1→step2 entry 연결 확인** 후 SH4/worker의 능동 polling·sleepwait·heartbeat·자동 recall을 중지한다. 이를 B100→B2 gate와 혼동하지 않는다. 다른 branch가 pending이라는 이유만으로 대표 actual을 PASS로 만들지 않는다. 전량 정상 release 후 실제 자원 부족으로 본실험이 실행되지 못하면 정확 pending 근거와 INITIAL_NOT_OBSERVED를 남기고 중지할 수 있다. 기술/PENDING 준비만 하고 미등록 branch를 자동 진행되는 것처럼 보고하지 않는다.

등록 프로그램은 추가 GH/user 승인 없이 각100edit/정해진 snapshot/patch/compact reducer를 자연 진행한다. 중간 성능으로 후속 branch를 선별하지 않는다. 초기/pending 인계 후 완료 상세 CPU리뷰는 사용자 recall 때 수행한다. GH는 중복 GPU/raw 감사나 장시간 실험 모니터를 만들지 않는다.

## 7. 구현·게시 허용 경로와 첫 회신

전용 clean branch `codex/server4-temporal-routing-diagnostic-20260929-v1`.
새 구현 `project/run_scripts/temporal_routing_diagnostic/`; 읽기 전용 native를 task-local vendor/adapter로 연결 가능. 공유 원source/helper/env는 수정하지 않는다.
Local `/data/janghj/ODE-edit/local/temporal-routing-diagnostic/20260929-v1/`.
자기 scope만 nonforce branch/main 게시를 승인한다:

- `experiment-reports/servers/server4/temporal-routing-diagnostic-20260929-v1/`
- `audits/servers/server4/temporal-routing-diagnostic-20260929-v1/`
- `plans/updates/server4/temporal-routing-diagnostic-20260929-v1/`
- `tasks/status/temporal-routing-diagnostic-20260929-v1/server4.json`
- `messages/acks/server4/2026-09-29-temporal-routing-diagnostic.md`
- `messages/server-heads/server4/2026-09-29-temporal-routing-diagnostic.md`
- `runs/odeedit_temporal_routing_diagnostic_s4_20260929/` (exact namespace 명시 허용)
- `transfers/verifications/2026-09-29-temporal-routing-sh4/`
- ignored local boundary/cap의 실제 task 값, 다른 task 설정 불변.

Source/config/sample/실제import/부모CP/실행job mapping 및 raw/artifact inventory를 남긴다. 원 raw/tensor/prompt/fullstdout Git0. 과거 dirty/main 변경 보존, source와 분석 publication 구별. NO_BROADCAST_NOT_REQUIRED.

첫 ACK/M0에 이 nonce, 정본 FULL_READ/hash, source/부모CP 재사용·누락, singleton L2=10 구현계획, 15branch/1500fits/30snapshot 예외, cap2 및 RAM/disk/time 계획, 남은 actual검증을 보고하고 진행한다. 단계별 GH ACK를 선행조건으로 기다리지 않는다. 접수/구현/등록/actual gate/terminal을 구분한다.
