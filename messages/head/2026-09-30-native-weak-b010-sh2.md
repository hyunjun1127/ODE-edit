# B010 native weak 대조 실험 실행 지시

Instruction / ACK nonce: `ODEEDIT-GH-SH2-NATIVE-WEAK-B010-NLL1-20260930-R1`.
GH session `01a04939-8873-7673-8dca-4c7fc5e31af0` → SH2 session `01a0493a-074c-7f91-9a13-769116326fef`.
Server2, CWD `/mnt/raid5/janghj/ODE-edit`, repository hyunjun1127/ODE-edit.
사용자: “native-weak: target fit을 NLL ≈ 1.0에서 멈춤 실험을 돌려서 강도를 joint와 맞춰볼 필요성이 있다. sh2에게 task 전달하라.”

## 목적과 범위

완료 B010 다층 BS1 비교에서 target fitting을 덜 진행한 native가 joint와 비슷한 획득 강도 및 locality를 보이는지 조사한다. 목표는 종료 기준을 조정한 대조이며, 실제 강도 동등성은 결과로 확인할 사항이다. 기존 NATIVE/JOINT_STEP/JOINT_CUM의 성능 차이를 강도 하나로 인과적으로 단정하지 않는다.

현재 비교 문맥에 따라 **B010 부모에서 새 NATIVE_WEAK_NLL1 한 경로, BS1×100 offered edits**를 실행한다. B050/B090, 새 seed/threshold sweep/추가 joint 경로는 승인하지 않는다. 다른 진행 job/collector나 취소된 경로는 변경·재개하지 않는다. 기존 3arm 결과를 exact 재사용하고 중복 실험하지 않는다.

필독은 원 joint METHOD/design/contract, 최신 BS1 override, SH2 migration envelope, actual S2 runner/native/compute_z/scorer 및 아래 B010 완료리뷰다. 이전 전체읽기와 동일 bytes는 hash로 재사용할 수 있다.
- `messages/head/2026-09-29-joint-multilayer-bs1-sh2-migration.md`
- `plans/global/2026-09-29-joint-multilayer-bs10-diagnostic-v1/`
- `plans/global/2026-09-29-joint-multilayer-preservation-method-ko.md`
- `plans/updates/server2/joint-multilayer-bs1-20260929-v1/user-cap3.md`
- `experiment-reports/servers/server2/joint-multilayer-bs1-20260929-v1/b010-review-20260930-v1/report-ko.md` 및 final-eval-update-r1 원표/verification.
- `project/run_scripts/joint_multilayer_bs10/`의 원 runtime/runner/observations/server2 bindings/final_eval.
- PROTOCOL/실제 session·자원·noCP 정책.

기존 실행2a7762a4368de0dbe540619ceacefd7760d25698와 finaleval62005204cf2b0a5abf5ef2830849bc9103199515, 완료게시6dcba810dfd8e331c62b7352c0813f9b82b9e7a6를 구분한다. 보고상 기존 최종 R/P/N은 NATIVE99/190/744, STEP100/142/785, CUM100/143/786(분모100/200/1000)이며 새 결과를 이에 맞추어 튜닝하지 않는다.

## 고정 native weak 구현

SH2 실제 수신 native-imports/AlphaEdit/compute_z.py에서 기존 stopping은 `loss < 5e-2`이고, loss는 nll_loss+kl_loss+weight_decay이다. **새 arm은 그 stopping predicate만 `nll_loss <= 1.0`으로 바꾼다.**
nll_loss는 현재 원 compute_z의 target token 평균→rewriting context 평균인 값 그대로다. canonical 단일문항 NLL, total loss, log probability, rounded print값, preference margin으로 대신하지 않는다. Context 수/길이/가중치/target tokenization은 기존과 동일하다.

- 매 iteration에서 원 forward/loss를 계산한 직후, backward/Adam 이전에 finite nll_loss를 원정밀도 scalar로 검사한다.
- 최초 NLL≤1.0인 현재 delta를 반환한다. 초기 delta0에서 이미 만족하면 Adam0으로 정상종료한다.
- 최초 crossing의 undershoot를 허용하고 실제값을 기록한다. 1.0에 정확히 맞추는 interpolation/bisection/extra optimization/step 재선택0.
- 최대25 loss evaluations /24 Adam updates, lr.1/decay.5/clamp.75/KL.0625 및 원 objective·Adam·clamp 그대로다.
- 끝까지 NLL>1.0이면 BUDGET_EXHAUSTED_ABOVE_TARGET로 기록하고 마지막 finite target을 원 native에 전달한다. 추가iteration/retry/강한nativefallback/요청제외0.
- threshold crossing과 budget exhaustion의 경계(마지막 forward에서 도달 포함), initial-below-threshold, nonfinite, 원 Trace telemetry가 predicate 변경을 추적하는지 CPU 검사한다.
- 원 source/shared EasyEdit/native를 편집하지 않는다. task-local compute_z 사본 또는 동등 최소 override를 사용하고 원 module에 process-local로 바인딩한다. exact diff/AST로 종료 predicate와 필요한 telemetry 연결 외 objective/writer 변경이 없음을 검산한다. 기존 Trace는 `loss` 비교를 찾으므로 `nll_loss` predicate에서 trace 누락/잘못된계측이 나지 않게 최소 보완한다.

기존 z를 축소하거나 강한 target fit 결과에서 중간값을 사후 선택하지 않는다. 원 NATIVE 이후 상태를 이어붙이지 않는다. B010 원본 부모 전체 W/M/context/RNG에서 시작하며 이후100step은 **자기 직전 state**로 매번 새 target/키/update를 계산한다.

L4–L8 전체 native writer/blue=false/L2=10/순차5layer residual divisor5/4/3/2/1/원 raw-P solve/post-all-layer history append를 그대로 유지한다. NATIVE_WEAK는 joint의 preservation guard·dual·optimizer를 이식하는 방법이 아니다. 수용 여부에 관계없이 원 native의 finite update/history와 offered 분모를 보존한다. native 한 요청 fit 및5solve/history가 wrapper에서 이중 호출되지 않게 한다.
FP32/eager/TF32 matmul false/cuDNN true/원 Transformers4.44.2 및 같은 모델 revision을 유지한다. 새 batching/prefix-cache/hooking 효율화/BF16/gradient clipping을 섞지 않는다.

## 입력과 관측

기존 B010의 exact 부모 CP, 동일 seed/RNG/context/P/history/모델 및 고정500 continuation의 **앞100**을 같은 순서로 사용한다. fixed10k의 일반 first100이나 다른 B010 다음100으로 바꾸지 않는다. existing SHA/수신receipt를 적합 범위에서 재사용, 자산 재생성/대형전송0.

기존 B010 panel144pair(Base control16/observer64, History control16/observer48)를 유지한다. 전체 272 inventory와 B010 분모144를 구분한다. 공식 P/N 및 observer는 최적화·stop·요청선택에 쓰지 않는다.

필수 로그와 결과:
1. 요청별 초기/매iteration/최종 fit NLL, KL/decay/total loss, 최초crossing/종료이유/Adam·forward수, target/delta norm.
2. 실제 five-weight write 후 같은 native rewriting contexts의 target NLL, canonical R true/new NLL 및 margin, RS/strict, 층별 actual update norm/총에너지,5층 history append와 다음entry 연결. 기존 joint에 저장된 context loss와 비교할 때 native latent injection과 actual weight loss를 분리한다.
3. 같은 기존 관측 시점의 current/at-write/all-offered R/P, Base/history observer 및 최종 greedy 규칙을 재사용한다. TF token-micro/prompt-macro/strict와 preference/free generation을 구분한다.
4. **현재 RAM의 최종 T100 state에서 R100/P200/N1000 전체 평가를 종료 전에 반드시 수행한다.** 마지막 요청 NS10을 finalNS1000으로 대체0. canonical source/row/token/target/order·tie=failure 정의는 기존 final_eval와 동일하게 결속한다.
5. 기존3arm 대 새weak의 최종 NLL/RSPSNS/TF/paired lost-gained, at-write→final, Base/history 및 비용을 보고한다. 요청을 cluster로 묶은 CI는 기존 reducer의 봉인 방식이 있으면 그대로 사용하고 새결과로 변경0. W0 NS1000 미측정이면 절대보존 주장0.
6. strength matching은 fitted-latent NLL뿐 아니라 실제 write후/최종 canonical 및 context NLL·margin·strict·update norm/획득량 분포로 평가한다. threshold같음=실현강도같음으로 쓰지 않는다. 맞지 않더라도 이번 결과를 보존하고 자동추가튜닝0.

새 run100 native target fits/500solves/500layerhistoryappend 상한(정상완료 기준), 최대2500fitloss/2400Adam, 필요한 observer/cost 별도. 기존3arm/raw를 CPU 비교용 재사용하고 재평가0.
모든 actual wall/allocation/peak/native/observer/I-O 계측 가능 범위를 분리한다. 기존29.2575GPUh/507초 평가비용을 새비용에 중복가산0.

## 저장과 기술 경계

이번은 **새 단일 대조 run**이므로 별도 새 checkpoint 저장 지시 없는 기본정책 `save_checkpoints=false`를 적용한다. 앞 9경로의 명시36snapshot 저장 예외를 새경로에 자동확대하지 않는다.
Weight/M/full-resume bundle·동등 full-state delta 저장0, exact_resume=NOT_AVAILABLE.
대신 raw NLL/TF·scalar fitting/controller·case/order/config/hash/history receipts는 보존한다.
기존 원본 CP와 과거25/50/75/100 snapshots는 읽기전용 보존, 삭제0. 최종 full RPN은 RAM 살아있을 때 수행하여 후속 weight 재생성 필요를 만들지 않는다.

CPU 회귀와 실제 첫step write/history5→step2 연결은 구현 correctness 확인용이지 성능선별 gate가 아니다.
새 장시간 FD/ULP/교차host재현 campaign0. 기존 입력identity/finite/IO/비편집parameter·state비변이/정확history 및 실제 metric completeness 검사는 유지한다.
기술오류는 원source/raw/cost 보존·원인 선보고 후 narrow repair/newimmutableattempt 가능하나 threshold/예산/과학방법은 바꾸지 않는다. 단순 NLL crossing undershoot나 quality저하는 기술FAIL이 아니다.

## 자원과 제출 권한

Server2 **project GPU cap3**, 이 새task는 동시1GPU/1job. 기존 B050 두 경로를 포함 실제 active+admitted capacity를 제출 직전 세고 타job변경0. cap3 사용자override와 과거 trackedcap2의 차이는 별도authority로 기록한다.
각1GPU/8CPU/host60416MiB/exportNONE/Requeue0, server2. Wall은 같은host native B010 실측 및 inline 최종평가/IO 여유로 제출 전 봉인하며 임의GPU시간hardcap은 추가하지 않는다.
부족하면 held검사 후 cap-safe pending 등록, 기존작업취소/중복등록0. 모델/부모CP/CPU P-M/RAM 및 noCP output의 실제 메모리·diskreserve 계획을 남긴다. A6000 한계에 맞추어 과학정밀도/패널을 줄이지 않는다.

새 source/config/threshold/actual imports/input order를 freeze→held owner/fullargv/source/resource/dependency 검사→release까지 진행하라.
대표 actual 첫step write/history→step2 연결 확인 또는 정상제출 후 실제resourcepending 인계까지 bounded관찰한다.
그 뒤 monitoring_active=false/automatic_resume=false/polling·callback·terminalwait0; 등록프로그램은100step+fullfinalRPN+compactreducer를 자연완료한다. 사용자recall 때 상세완료리뷰/main갱신.
추가GH승인/원3arm재검증을 실행선행조건으로 삼지 않는다.

## 산출물과 허용 경로

Dedicated clean branch `codex/server2-native-weak-b010-nll1-20260930-v1`.
Local `/mnt/raid5/janghj/ODE-edit/local/native-weak-b010/20260930-v1/`.
읽기: 기존 joint task의 봉인source/inputs/CP/config/B0103arm raw/최종평가, native-imports, 공용모델·P/context 및 정본. 다른task 과학raw 조회0.
쓰기와 own-scope nonforce branch/main 게시:
- `project/run_scripts/native_weak_b010/` (공유원runtime read-only import/전용adapter)
- `experiment-reports/servers/server2/native-weak-b010-20260930-v1/`
- `audits/servers/server2/native-weak-b010-20260930-v1/`
- `plans/updates/server2/native-weak-b010-20260930-v1/`
- `tasks/status/native-weak-b010-20260930-v1/server2.json`
- `messages/acks/server2/2026-09-30-native-weak-b010.md`
- `messages/server-heads/server2/2026-09-30-native-weak-b010.md`
- `runs/odeedit_native_weak_b010_s2_20260930/` exact namespace 허용.
기존 B010 report는 덮어쓰지 않고 새비교보고에서 연결한다. Global設計/공유env/원native/source/다른task 수정0. Raw/tensor/prompt/fullstdout Git0; NO_BROADCAST_NOT_REQUIRED.

최초 nonce ACK와 FULL_READ/원 stopping식/새stop정의/정확B010 IDs·원본재사용/source diff/CPU·GPU단계/자원·현재미제출을 보고하고 실행을 진행한다. jobID/actual초기/최종완료를 각각 구분한다. SH2의 실험수신 및 정상등록을 GH가 확인하되 장기별도모니터를 만들지 않는다.

