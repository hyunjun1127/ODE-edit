# GH → SH2: BLUE L4-only progress-preserving barrier 전 과정 실행

- instruction_id: ODEEDIT-S06-BLUE-L4-PROGRESS-PRESERVING-BARRIER-SH2-V1
- nonce: ODEEDIT-GH-SH2-L4-EP-20260911-R1
- from: GH / 01a04939-8873-7673-8dca-4c7fc5e31af0
- to: SH2 / 01a0493a-074c-7f91-9a13-769116326fef
- server/repository/CWD: server2 / hyunjun1127/ODE-edit / /mnt/raid5/janghj/ODE-edit
- 상태: 사용자 승인 구현·GPU 실행·분석·최종 보고서까지 위임. 실행 완료 보고가 아님.
- authoritative design: plans/global/2026-09-11-blue-l4-only-progress-preserving-barrier-experiment-design.md
- design SHA256: 30be08eccc4b74b34acfeb9e3deb5cbffc9e6c4a02768e6f69fddf5593d99a8e
- design bytes/lines: 37031 / 425
- 배정 시 remote main 및 ABC 재사용 source: d2c808015d8e1b34a039c125139d6d66bbca6c73
- GH shared root HEAD: ddc178584ef14efd5d4e1271b3c324e3ebd3e443 (dirty/untracked 사용자 자료 보존)

## 1. 사용자 요청과 책임

사용자는 위 문서를 전체 읽은 뒤 참조 코드·자산을 확인하고, 구현에서 멈추지 말고 지정 실험·분석·최종 한국어 보고서까지 완료하라고 명시했다. 이번 기준은 9월 11일 문서이며 이전 여러-layer 배분이나 ABC의 optimizer/soft-filter 설정을 혼합하지 않는다. 문서의 작성 시점 'GPU 미실행' 문구는 현재 승인 후 실행을 금지하는 지시가 아니다.

SH2에게 구현, 필요한 자산 수신, 최소 correctness 확인, 자원 내 제출, 기술 오류 수정, 단계 전환, 결과 평가·분석·그림·보고서·main 통합을 일괄 위임한다. GH가 같은 코드/raw/tests를 중복 점검하거나 단계마다 재승인하지 않는다. SH2는 혼자만 작업하는 저장소가 아니므로 타 agent/사용자의 변경을 되돌리지 않고 새 전용 경로에서 작업한다.

기존 PROTOCOL의 SH factual-only 기본 규칙에 대한 **이번 instruction 한정 사용자 명시 예외**: 사용자가 가능한 설명·미분리 요인·실용성·기전 질문에 대한 분석을 요청했으므로 SH2 최종 보고서에 이를 포함한다. 관측 사실/가능한 설명/확인하지 못한 가설을 분리한다. scientific_promotion=false. 후속 실험을 자동 추가하지 않는다.

## 2. 승인된 write/Git 범위

새 branch 권장: codex/server2-blue-l4-progress-barrier-v1. 새 clean worktree에서 실제 local/remote HEAD와 관련 diff를 기록하고 일관된 source를 pin한다. 아래 신규 경로가 있으면 create-once run/version suffix를 붙인다.

- 구현·tests: project/run_scripts/blue_l4_progress_barrier/
- raw/imports/locks/snapshots/logs: local/blue-l4-progress-barrier/<attempt-id>/
- 보고서: experiment-reports/servers/server2/blue-l4-progress-barrier-2026-09-11-v1/
- audit: audits/servers/server2/2026-09-11-blue-l4-progress-barrier/
- 진행 보고: messages/server-heads/server2/2026-09-11-blue-l4-progress-barrier.md
- 상태: tasks/status/blue-l4-progress-barrier-sh2-v1/server2.json
- 작은 run metadata: runs/blue-l4-progress-barrier-sh2-v1/
- 필요 서버 계획: plans/updates/server2/2026-09-11-blue-l4-progress-barrier.md
- 수신 검증: transfers/verifications/2026-09-11-blue-l4-progress-barrier-sh2-*.json

기존 single_layer_cumulative_risk/, BLUE/EasyEdit 및 기존 raw는 read-only 재사용한다. 필요한 adapter는 새 package에 구현하며 기존 kernels의 일괄 refactor를 하지 않는다. Shared dirty root reset/stash/cleanup 금지. Source·tests·설계 참조·raw-free CSV·직접 코드로 생성한 PNG·manifest·보고서를 전용 branch에 non-force push할 수 있다. 완료한 본인 scope는 최신 main과 충돌 없는 clean integration worktree에서 non-force main 통합/push까지 승인한다. Git conflict는 중단·보고한다. Raw weights/tensors/prompts/generation text/cache/full logs는 Git에 넣지 않는다. GH 소유 설계/지시/transfer 승인 파일은 임의 수정하지 않는다.

## 3. 자원·실행 승인

Slurm submission=ALLOWED. 서버2 프로젝트 GPU cap=2이며 이미 실행 중/예약된 다른 project 작업을 포함한다. 기본 1 GPU/process, 8 CPU, explicit --mem=60416M, --export=NONE. 메모리 ceiling 초과·GPU 동시 탑재·다른 작업 cancel/preempt/throttle 수정은 금지한다. 실제 admission 직전 다시 확인하고 부족하면 적법한 dependency/throttle로 pending하거나 WAITING_FOR_ISOLATED_RESOURCE로 보고하되 CPU 구현은 계속한다.

별도 유료 자원·자동 host 확대 금지. 이번 작업에 숫자로 지정된 GPU-hour 상한은 없으며 이전 D/S 48시간, pilot 8시간을 상속하지 않는다. 설계의 2.5–4 GPUh는 추정이지 실행 cap/보장이 아니다. GPU-hour cap null 자체를 새 승인 gate로 삼지 말고, 승인된 고정 12-trajectory 범위 안에서 Middle 실제 비용·메모리·저장 공간을 측정해 전체 추정과 실제 ledger를 보고한다. 실제 불가능한 자원 문제나 과학 계약 변경이 필요한 모순만 구체적으로 상신한다.

## 4. 읽기·자산 전송

설계 425행 전체와 해당 참조 코드의 실제 구현·asset identity를 먼저 확인한다. 과거 설계 문서는 계보이며 최신 식을 대체하지 않는다. source/report SHA와 execution SHA를 구분한다. GH는 참조 objective/runtime/binding 경로와 세 prepared 파일의 존재를 확인했으나 이번 tensor 재검증/모델 parity를 수행한 것은 아니다. 수신 및 실제 복원 검증은 SH2 소유다.

Server1 봉인 source:
- /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/project/run_scripts/single_layer_cumulative_risk/
- artifact root: /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1
- A/Early/native-r4/prepared.pt: 3424107939 bytes
- A/Middle/native-r1/prepared.pt: 3424119843 bytes
- A/Late/native-r4/prepared.pt: 3424131811 bytes
- input.lock.json, imports/entries/B010/B050/B090 W-method-state.pt, B011/B051/B091 native-targets.pt 및 binding metadata
- 설계가 지정한 기존 W0/ENTRY/N full metrics, N generation, panels, native config/contexts/BLUE helper 및 identity receipts
- C0: /mnt/raid5/janghj/EasyEdit/examples/data/stats/Meta-Llama-3-8B-Instruct/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz
- P: /mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt

동봉 transfers/approvals/2026-09-11-blue-l4-progress-barrier-inputs-sh2.md의 scoped rsync pull을 SH2 단독 transfer owner에게 승인한다. 이미 server2에 정확한 동일 bytes가 있으면 중복 복사하지 않는다. 필요 파일 allowlist만 source SHA/size·destination SHA로 봉인하고 기존 파일에 overwrite하지 않는다. 타 task 진행 중 output을 탐색하지 않는다. SH1에 exact 봉인 경로/identity가 필요한 경우 peer direct 문의는 가능하나 SH1의 재실험/중복 감사 대기는 선행조건이 아니다. 원본 native 재실행이나 cold compute_z를 자산 대체 수단으로 자동 사용하지 않는다.

## 5. 절대 유지할 과학·구현 경계

- Llama revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FULL-FP32 actual model/write, 수정 weight는 L4 down_proj [4096,14336] 하나.
- 공통 시작점 **정확한 WN**, W(X)=WN+X Ubᵀ, X0=0. We+projected-native approximation으로 대체하지 않는다. Saved Ub 그대로; scalar amplitude/full Q/orth(K)/orth(PK)로 대체 금지.
- Inner z/K/U/P/M 고정, 새 compute_z/native wrapper/history append 0. Actual-output edit gradient는 매 node 갱신. M은 pre-native M이며 MN 또는 current KKᵀ를 이중 반영하지 않는다.
- We essence teacher; KL(student||We), .0625. 공통 loss의 .1 action penalty는 ΔN을 포함한 We 기준, cross term·constant를 유지한다.
- U의 실제 Gram을 반영한 H_R/C_N/H_N, native effective metric B=H_N/ν, ν entry 한 번. Initial nominal h=.125 displacement만 native norm1%; correction 후 재정규화 금지.
- EP 보호는 edit-only 평균 NLL derivative, total objective가 아니다. jv0>0도 그대로 기록. J4는 case-ID hash 고정 네 group25씩.
- W0 중심 covariance risk / WN cap, κ=2, ε=.1 q_ref 고정. q_ref curvature항의 d_ref=(1/8)v0는 N16에도 동일. h는 actual Euler update에만.
- FP64 native-whitened closed form, row-normalized rank tolerance1e-10. Existing Euclidean soft_filter/ridge로 대체 금지. q=0/zero observer/stationary/flat risk 구분.
- 같은 group/microbatch/reduction/order/nominal path를 모든 arm에 사용. Group edit gradient 한번 계산해 mean + essence + analytic action을 만들고 evaluate(backward)+group gradient 중복 호출 금지.
- 새 norm cap, line search, rollback, adaptive gain, request ceiling, outcome gate, winner-prefix 선택 금지. 유효 finite poor/risk-up/slack/near-zero 결과를 보존한다.

## 6. 고정 실행·평가

Middle 공통 asset/weight/teacher/history/small-matrix 확인 → Middle H/R/EP N8 실제 실행·비용 → Early/Late H/R/EP → Middle EP-Free, EP-J4 N8, EP-N16을 모두 완료한다. 주9/72 + 보조3/32 = **12 trajectories/104 logical steps**. Exact N3은 재사용 baseline이다. 5-batch sequential, 여러-layer, 다른 모델, 전체 parameter sweep은 범위 밖이다.

설계 §12–13 그대로:
- EP 각 entry pre0/3/7 same-state nominal shadow. pre0 H1 재사용, 추가 edit-only forward6회.
- 공통 WN matched-risk one-step probe entry별1개, 총3; 추가 backward 없이 objective+Curve. 실제 path에 carry하지 않는다.
- N8 step1/2/4 Curve1100, step8 Full3900+terminal objective/risk. N16 step2/4/8/16 대응 t.
- W0/We/WN exact reused Full, duplicate denominator 없음. Model/backend/packing identity와 미검증 numerical parity 경계 기록.
- Middle N/H/R/EP 기존 동일 generation panel/decoding, N재사용, 새180 sequences.
- 모든 요청별 NLL/new·true/PS/NS/TF/일차예측·실제차이/악화·회복 기록; 2000회 request-paired bootstrap.
- 실제 saved snapshots만 trade-off curve에 사용; 보간·best-NS endpoint selection 금지.
- 총 계획 95 gradient sweeps/34390 model B/41692 training F/89700 새 prompt pairs/179400 candidate sequences는 **계획**이다. Teacher/load/geometry/evaluator/generation/diagnostics/실제 재사용/오류 비용은 분리하고 실제 ledger를 보고한다.

최소 correctness 두 묶음(상태·실제 적용 / 작은 수식)을 SH2가 수행한다. 새 대형 사전 audit/성능 gate를 만들지 않는다. 오류는 영향받은 attempt만 새 ID로 수리·재실행, 기존 실패 결과 immutable. 과학적 설정 변경은 상신한다.

## 7. 분석·보고와 진행 방식

SH2가 설계 §14·17 및 사용자가 요청한 모든 질문에 답한다: H 대 N, EP 대 H/R, same-state matched-risk의 edit 보호, 평균 보호 아래 request/Fixed/Past 손실, risk와 실제 locality의 관계, Free/J4/N16 설명, 추가 비용을 고려한 실용성. Cached-native 이후 비용과 cold-z 포함 온라인 총비용을 혼동하지 않는다. 평균 일차 equality를 실제 정답 보장이나 locality 보장으로 표현하지 않는다.

필수 산출물: run-index.csv, trajectory.csv, same-state-probes.csv, request-metrics.csv, paired-summary.csv, compute-summary.csv, reconstructable X snapshots+U/WN references, diagnostic-report-ko.md. Source/runtime/input/science/resource identities와 오류·미실행 목록 및 변경 이유를 결속한다. PNG는 저장한 직접 코드 실행으로 생성하고 재현 명령을 포함한다; imagegen/Codex 이미지 생성 도구 사용 금지.

보고 시점: FULL_READ/자산·source pin, Middle 첫 실제 correctness/비용, 주9개 완료, 12개 전체 완료+최종 report/main SHA. 이번 요청은 최종 실험·분석까지 완주이므로 SH2가 필요한 범위의 진행 확인·기술 repair·후속 제출·terminal 수집을 담당하고 중간마다 GH 승인 대기하지 않는다. 불필요한 빈번 polling/별도 범용 heartbeat infrastructure를 만들지 않는다. 다른 task에 이미 설정된 monitoring pause는 해제하지 않는다. GH는 handoff 후 중복 점검하지 않는다.

완료 후 raw는 local-only 보존하고 source/raw-free report만 공유한다. 불필요한 전체 raw broadcast 대신 같은 프로젝트 보존 경로·manifest와 NO_BROADCAST_NOT_REQUIRED 예외를 기록할 수 있다; 별도 복제 필요시 승인된 project helper/비파괴 allowlist 방식만 쓴다. 최종보고와 본인 scope의 main 통합이 끝나면 task STOP, 새 run은 제출하지 않는다.
