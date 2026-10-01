# 사용자 실행 명령 전달: JLZ 두 arm pilot → BS100×20, GH → SH4

- nonce: `ODEEDIT-USER-GH-SH4-JLZ-TWOARM-BS100X20-20261002-R1`
- 발신: 사용자 지시를 전달하는 설계 세션 `01a0f6b9-74e5-7683-a798-029e477c29b1`
- 수신 GH: `01a04939-8873-7673-8dca-4c7fc5e31af0`, `/mnt/raid5/janghj/ODE-edit`
- 실행 SH4: `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, `/data/janghj/ODE-edit`
- repository: `hyunjun1127/ODE-edit`

## 사용자 원문과 실행 범위

> 이거 간단한 pilot test 돌려보고 바로 bs100, 20 step sequential로 진행해서 baseline과 비교하는 것으로 진행하고 싶다. gh에게 task 전달시키고 sh4에게 진행시키도록 하는 것 user 명령으로 전달한다고 하자

이는 설계 검토만의 요청이 아니라 **GH가 SH4에 구현·작은 GPU pilot·본실험·baseline 비교를 배정하라는 직접 사용자 명령**이다. 20 step은 optimizer iteration이 아니라 BS100으로 20회 순차 write하는 **총2,000개 요청**이다. 대상은 직전 대화의 JLZ 두 arm 전체다.

기존 설계의 `NOT_SUBMITTED`, `DISCUSSION_DRAFT`, `auto_launch=false`, 향후 BS100×10은 작성 당시 상태다. 이번 사용자 명령이 실행 범위와 자동 연결 여부를 갱신한다. Pilot의 계산·commit·H 연결을 확인하면 **추가 사용자 승인 없이 BS100×20으로 이어간다**. Pilot 성적, 수렴 여부, 한 층 집중, low cosine, NS 저하는 본실험 진입의 hard gate가 아니다. 확인된 구현 오류·비유한 최종값·상태 손상은 고치거나 정확한 blocker로 보고한다.

## GH가 읽고 SH4에 제공할 설계

동일 host의 다음 worktree에 원 설계·CPU 검산 산출물이 있다. 아직 이 세션이 main에 게시하지 않았다. GH는 자신이 검토한 전용 worktree에서 이 정확한 파일들을 source-controlled task로 승격하고, SH4가 접근할 수 있는 commit·path·hash를 전달한다. Git은 source/task 저장이며 실제 지시는 app-server direct로 전달한다.

`/mnt/raid5/janghj/.codex/worktrees/odeedit-jlz-two-arm-design-20261002-v1/plans/global/2026-10-02-jlz-two-arm-v2/`

- `design-ko.md`: 계산 그래프, 두 arm, reference, 비용, gate, cross 진단
- `contract-draft.json`: 공통 방법 정의; 아래 실행 overlay가 과거 단계·scope를 갱신
- `inspect_design.py`, `evidence.json`: CPU 수식·gradient·reference 분리·단층 해 허용 검산; **모델 pilot는 미실행**
- `execution-bs100x20.json`: 이번 사용자 명령의 실행 overlay
- `handoff-manifest.json`: 위 파일과 본 전달문의 SHA/size

기존 코드의 재사용 근거는 원 main `0b57e9e4e125f083477faf85ec77375d573bcb55`, 원 JLZ 실행 source `c91962dd4952eb58a16b6b00dcaa3018be71693f`이다. 설계의 FP32 actual-write 정의와 FP64 geometry를 지킨다. 별도 효율화의 E123_MB4 측정은 direct-R=false의 고정 후보 기술 측정이며 새 두-arm 속도·성능 근거가 아니다.

## 공통 방법과 두 arm

1. L4–L8 전체 down_proj를 처음부터 공동 변수로 둔다. 사전 subset/top-k/최소 사용층/최대 점유율 제약 없음. 두 arm 모두 단층 전담 및 나머지 층0을 허용한다.
2. `W_entry+(R.double()@P.T).float()`의 actual-write를 모든 token 위치에 적용한 실제 모델로 NLL/KL를 계산한다. 상위 activation과 요청 간 cross effect가 실제 forward/backward에 포함된다. Native z 선최적화나 activation-only proxy를 넣지 않는다.
3. Native KL current||own-entry, 기존 group norm=.5, clamp=.75, zero-column entry NLL<.05를 유지한다. 기존 inactive current 요청의 NLL/KL도 공통 보호항에 포함한다.
4. 일반 reference16, 과거 replay최대16은 양쪽 공통이다. 일반 KL current||W0, βG=.0625; 최신 유효 past desired-target NLL, βE=1.0. 평가 NS prompt 제외, dataset/claim/subject/prompt disjointness 및 supersession 규칙은 설계 그대로다.
5. A는 η=0, B는 η=1.0의 추가 quadratic ideal-write 비용만 다르다. 이 세 계수는 미교정 초기안임을 남기되 **이번 첫 실행에서는 고정**한다. Pilot 점수로 자동 조정하거나 sweep를 붙이지 않는다. B의 total objective 감소나 고른 배분을 승리 기준으로 쓰지 않는다.
6. 양쪽 동일 runtime·route·microbatch 선택 규칙·SPG/BB/prox/line search·호출 예산. 각 arm은 W0/zero H에서 독립 시작하며 batch 사이 reset하지 않는다. Own entry teacher/K/P/H가 이후 달라지는 것은 정상이다.
7. Prefix 재사용, selected-position full-vocabulary head, per-microbatch right crop, 단일 down_proj, L8 key 조기 종료 등을 공통 적용한다. Direct R VJP는 dX와 실제 materialization을 보존하여 공통 기술 확인 후 쓴다. 경로 선택을 장기 가속 연구로 늘리지 않고 검증된 공통 경로로 본실험을 진행한다.

## 작은 pilot 뒤 바로 본실험

- 작은 pilot: 고정 순서 첫8개, BS4×2, 두 arm 각각 배치당 whole-batch oracle cap32. Science 상한 총128회. 두 번째 배치에서 past replay·H 연결을 확인한다. Baseline도 같은8개의 작은 작동 확인을 하되 native 종료/Adam 규칙을 유지한다.
- Cross 진단: 작은 pilot에 한해 같은 entry·최종 후보·같은 prompt에서 entry/단층 가상5/joint의 효과를 비교한다. 최대336 prompt-state no-grad 관측, 별도 비용 장부. 층 제외·배분 hard gate로 쓰지 않는다.
- 이전에 제안한 독립 BS100×1 cap32 pilot chain은 **생략**한다. 실제 B100 same-candidate 기술 점검은 공통 최대6회 oracle 이내로 짧게 수행하며 science와 별도 계수한다. 그 뒤 본실험 B1을 바로 시작한다.
- 본실험: fixed10k 원 순서 첫2,000개, BS100×20, 두 arm 각각 batch cap120. 두 arm science 총상한4,800회. 초기+accepted/rejected trial+원본 재확인+fresh final 모두 cap에 포함하고 final1회 예약. Caller/solver 공통 budget accountant 사용.
- Pilot의 편집 상태는 본실험으로 넘기지 않는다. 두 arm과 baseline 모두 main W0에서 새 chain을 시작한다. Pilot과 main 각각 입력 ID/order/hash를 봉인한다.
- 유한한 미수렴 후보는 기존 사용자 승인대로 last accepted/초기점을 fresh 최종 평가해 commit한다. 학습·평가 품질로 요청/층/배치를 빼지 않는다.
- Trial 비유한값은 두 arm 공통으로 non-mutating trial만 거절하고 step을 절반으로 줄여 남은 cap에서 계속하는 정책을 구현한다. 초기/최종 비유한, CUDA/state 오류, 의도한 weight와 다른 commit은 중단한다. 원 production의 any-trial fatal 규칙을 소급 변경하지 않는다.
- 유한한 route 반올림 오차는 원 수치/FAIL을 보존하고 warning으로 계속한다. dX 누락·잘못된 VJP/prox/Ω 미분 등 확인된 구현 결함은 경고로 넘기지 않는다.

## Baseline 비교의 실행 기본값

사용자는 이번 메시지에서 baseline 이름을 다시 열거하지 않았다. 기존 논의에 따라 **MEMIT-H를 주 비교**, 기존 BASE_MEMIT와 BASE_ALPHAEDIT을 보조 비교로 결속하는 실행 기본값을 전달한다. GH가 정확한 기존 source/arm 이름을 확인해 고정하며, 단순한 명칭 확인 때문에 사용자 재승인을 기다리지 않는다.

- 모두 같은 pinned Llama 모델, fixed first2k/order, BS100×20, L4–L8, 같은 평가 정의로 비교한다. Baseline 고유 native compute_z/Adam·정지·clamp·writer·history 의미는 바꾸지 않는다. JLZ의 새 reference나 quadratic을 baseline에 섞지 않는다.
- 기존 결과를 재사용할 수 있으면 우선 재사용한다. 단, **2k write 직후 모델에서 평가한 동일 cohort**여야 한다. 10k endpoint의 first2k 평가나 다른 BS/순서/층 결과를 동등 baseline으로 쓰지 않는다.
- 정확한 source/config/input/model/evaluator identity·raw 분모가 결속되지 않거나 같은2k endpoint가 없으면 SH4에서 필요한 baseline만 BS100×20 새로 실행한다. 사용자 baseline 비교 명령에 포함된 작업이다. 재사용과 신규 실행을 결과표에서 구분한다.
- BLUE(L4+L8)·server3 HJ BS10은 역사 참고로만 분리한다. 이 task를 새로운 HJ/BLUE factorial이나 additional layer-subset 실험으로 확대하지 않는다.
- 속도 비교는 같은 하드웨어·precision·동등 작업량으로 측정된 것만 배율로 쓴다. H200/A6000/Blackwell 시간은 별도 열로 두며 직접 속도배로 쓰지 않는다. Native 요청별 optimizer calls와 JLZ whole-batch oracle를 같은 단위로 세지 않는다.

## 평가·보고·자원·소유권

- 기존 current-batch R100/P200/N1000, 매 batch all-seen R/P, B5/B10/B20 all-seen R/P/N을 기록한다. B10 first1k는 R1000/P2000/N10000, B20 first2k는 R2000/P4000/N20000이다. 입력자료 cardinality도 검산한다.
- W0 같은 first2k 기준, preference RS/PS/NS, TF token/prompt/strict, true/new NLL, at-write→endpoint·first1k→2k retention, active/superseded 및 lost/gained를 보고한다. Reference 성적은 독립 평가와 분리한다.
- 층별 ideal/actual energy·R/D norm·clamp hit·잔차·사용층 수와 집중도는 기록한다. 총 objective 대신 동일 calls/실제 시간의 품질과 비슷한 edit strength에서 locality를 비교한다.
- GH는 이 사용자 명령을 근거로 SH4에 구현·Slurm 제출·pilot 후 main 자동 연결을 명시적으로 허용하는 envelope를 발행한다. 별도 사용자 재승인 gate는 필요 없다.
- Server4 project cap2를 지킨다. 각 job1GPU, 기존 점유와 admitted pending을 합산해 dependency/admission한다. Host request≤60416MiB/GPU. 기존 다른 job 취소·변경 권한은 없다. 큰 sweep·모델/데이터 신규 수집은 범위 밖이다.
- SH4 허용 source: 새 `project/run_scripts/jlz_two_arm/` 및 그 전용 tests/launchers. 기존 frozen JLZ·server3 runtime/raw를 hotpatch하지 않는다. 필요한 baseline wrapper는 전용 namespace에서 실행한다.
- SH4 task ID: `jlz-twoarm-bs100x20-20261002-v1`. 전용 non-main worktree/branch, `local/jlz-twoarm/20261002-bs100x20-v1/`, `audits/servers/server4/jlz-twoarm-bs100x20-20261002-v1/`, `experiment-reports/servers/server4/jlz-twoarm-bs100x20-20261002-v1/`, `messages/server-heads/server4/`, `tasks/status/jlz-twoarm-bs100x20-20261002-v1/server4.json`, `runs/odeedit_jlz_twoarm_s4_20261002/` 사용.
- GH는 own task/source/compact report의 non-force 게시·main 통합 범위를 envelope로 정한다. 원 source/config/input/runtime SHA, job ID·argv·dependency·output path·실제 whole-batch/physical F/B/token/time/memory를 남긴다.
- 영구 R/W/H/optimizer checkpoint는 기본 저장하지 않는다. Raw/log는 local, Git에는 작은 코드·manifest·요약만 둔다. 이 task의 raw 대량 broadcast는 불필요로 예외 기록하고 compact 결과는 GH에 전달한다. 기존 자산·다른 실험의 보존 정책은 변경하지 않는다.
- Pilot 후 main 연결은 이번 사용자 명령으로 허용된 DAG에 미리 묶는다. SH4는 실제 main B1 commit·H append·B2 entry 연결 또는 cap을 지킨 정식 GPU-resource pending을 초기 인계로 보고한다. 이후 장기 agent polling/heartbeat는 만들지 않으며 봉인된 runner와 CPU reducer가 실행된다. 최종 수치 수집·baseline 비교표 생성은 이 task의 허용 산출물이다.

## GH 직접 응답 요청

현재 사용자 명령과 nonce를 ACK하고 SH4에 직접 전달하라. SH4가 다른 unrelated task로 active이면 그 turn을 steer하거나 중단하지 말고 다음 idle turn에 이 task를 전달한다. GH 응답에는 durable task/commit, SH4 delivery accepted turn 또는 정확한 전달 대기 사유를 담는다. **전달 수락·실험 제출·실행 완료를 구분**한다. 이 설계 세션이 실험 GPU job을 직접 제출한 것으로 기록하지 않는다.
