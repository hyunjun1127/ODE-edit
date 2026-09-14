# GH → SH4: C4 reference / BG-1 ours-first 실행 envelope

- Instruction ID: `ODEEDIT-S06-BG1-C4-OURS-FIRST-SH4-V1`
- Parent: `GH-BG1-C4-OURS-FIRST-20260915-V1`
- Nonce: `ODEEDIT-GH-SH4-BG1-C4-OURS-FIRST-20260915-R1`
- 사용자 요청: 2026-09-15 method 지시문 정독 후 SH4 배정. GPU cap=2.
- 수신: server4 / SH4 / session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`.
- 경계: 실제 hostname server4, CWD `/data/janghj/ODE-edit`, repository `hyunjun1127/ODE-edit`.
- 기준 main: `c2710f3e8fddf2a8df230f076e4519cc6d0d77e5`, tree `f89adf1551c9c770b42836f1f6a570cf70e7e65c`. 실행 source는 구현 완료 후 별도 pin한다.

## 1. 우선권과 완료 경계

사용자 원문 `project/proposals/2026-09-15-bg1-c4-ours-first-gh-instruction.md` 및 `plans/global/2026-09-15-bg1-c4-ours-first-dispatch-contract.json` 전체를 먼저 읽는다. 이어 reference survey, C4 data contract, from-base staged-design-v2, method contract, checks를 전체 결속한다. 원본 SHA/bytes는 같은 날짜 GH source-inventory에 보존한다. 최신 PROTOCOL 전체와 실제 source도 읽고 FULL_READ receipt를 남긴다. 요약만으로 실행하지 않는다.

이번 dispatch는 과거 7-policy/70-batch 신규 실행, REFIT4 재실행 및 완료까지 모니터링 지시보다 우선한다. 신규 scientific 정책은 **BG-1 하나 / W0 B100×10 / first1000**다. Runner가 dispatch JSON의 `new_scientific_policies=["BG-1"]`와 금지 목록을 실제 assert하도록 한다. 기존 cells/first_policies 전체 순회 금지. 문서의 후속 실험·해석 절은 이번 제출 승인이 아니다.

SH4가 준비·구현·기술 수리·source 검증·승인된 제출·G0 확인을 소유한다. 이미 문서 안에서 결정 가능한 사항을 매 단계 GH에 재승인 요청하지 않는다. 과학 조건 변경이 필요한 모순은 typed HOLD. GH는 중복 GPU/raw 감사를 하지 않는다. SH 보고는 실행 사실·분모·비용·검증·재현된 RCA에 한정한다.

**G0_PASS 또는 확정 terminal G0_FAIL/G0_BLOCKED 후 즉시 WAITING_USER_RESUME**. 이 task의 모든 worker/red/blue에도 같은 경계를 전파한다. 과거 REFIT4의 지속모니터링 override는 이번에 적용하지 않는다.

## 2. 정확한 write/실행 권한

별도 `codex/server4-bg1-c4-ours-first-v1` 또는 create-once 동등 branch/worktree를 최신 main에서 만든다. shared dirty·기존 raw·설계 원본은 보존한다.

허용 구현:
- `project/run_scripts/bg_tw_reference/`: builder, fixed teacher, fixture/restore, native map adapter, BG-1, calibration-forward, runner/Slurm wrapper, tests, compact receipt.
- `project/run_scripts/low_cost_write_donor_pilot/`: **read-only 재사용**. 수정할 파일을 이번에는 승인하지 않는다. .5/.25 materializer 등 필요한 wrapper는 신규 package에 구현한다. 기존 helper의 warm/정책/arm/hardcoded numerator를 통째로 상속하지 않는다.
- Native vendor/compute_z/기존 hparam·환경·shared P/cache를 변경하지 않는다.

허용 기록:
- `local/bg1-c4-ours-first/20260915-v1/<attempt-id>/` (S4 absolute prefix `/data/janghj/ODE-edit/`), 준비/teacher/calibration/technical/scientific namespace 분리.
- `experiment-reports/servers/server4/bg1-c4-ours-first-2026-09-15-v1/`.
- `audits/servers/server4/2026-09-15-bg1-c4-ours-first/`.
- `messages/acks/server4/2026-09-15-bg1-c4-ours-first.md`.
- `messages/server-heads/server4/2026-09-15-bg1-c4-ours-first.md`.
- `tasks/status/odeedit_bg1_c4_ours_first_s4_v1/server4.json`.
- `runs/odeedit_bg1_c4_ours_first_s4_v1/`의 compact run별 receipt.
- 필요한 `transfers/verifications/2026-09-15-bg1-c4-ours-first/`.
- global 설계/정책/PROTOCOL/다른SH 경로는 수정0.

Slurm **allowed**: pinned C4 builder/W0 teacher192/좁은 BG 기술 검사/저장 N4 endpoint C4 calibration forward/BG-1 B1–B10 단일 scientific chain. Gate 의존성을 만족한 뒤 승인된 source를 freeze→held inspection→release한다. Source branch non-force push와 검증된 source·compact preparation/G0 factual publication의 own-scope main integration은 승인한다. G0 뒤 전체 chain 결과 분석·보고서 확장/main 통합은 사용자 recall까지 금지한다. 자동 sync/callback으로 task를 깨우지 않는다.

## 3. 자원 / 다운로드 / 선택적 자산 복원

Server4 프로젝트 GPU cap=2. 기존 active와 admitted pending을 함께 계수하며 각 job 기본1GPU/1process, 명시 mem≤60416M/GPU, CPU·wall·disk는 실제 preflight에 pin한다. 이전 GPU-hour 예산 상속0, 이번 hour cap=null. cap2는 두 scientific BG 복제나 baseline 추가 허가가 아니다. 독립 준비·teacher shard·calibration/검사 작업은 상태와 출력 소유를 나누어 cap2 내 병렬화할 수 있다. 순차 의존 BG 본 chain은 한 개뿐이다. 준비와 본실험 합산, pending dependencies/throttle로 초과 admission 방지. 타user/기존 job 선점·취소·자원변경0.

허용 외부 획득은 data contract의 pinned C4 train/validation 첫 gzip 2개(합359779975 bytes), 해당 pinned README, composition용 고정 PSL metadata다. 전체 C4/Pile/새 model을 중복 다운로드하지 않는다. 기존 exact model/tokenizer/P/stats는 read-only.

S4에 없는 저장 N4 W0→B1–B10 state/delta가 필요하면 기존 S2 migration catalog와 owner 협의로 정확한 파일 allowlist를 만들고 read-only 선택 복사한다. 이미 승인·검증된 프로젝트 checkpoint archive/initial6 경로의 ordinary project artifact만 대상으로 하며 SH4를 단일 수신/복사 owner로 둔다. 크기/owner/hash/공간을 먼저 확인, create-once staging, no-delete/no-overwrite, 수신 fullSHA와 필요 source/next-index/context를 결속한다. S2 기존 보존본은 이동·삭제하지 않는다. 무관 raw 전체나 모델/P/stats 재전송0. 다른 source/destination 권한 예외가 필요하면 해당 경계만 보고한다.

신규 raw text/teacher/checkpoint는 local-only, 이번 단계 다른 서버 broadcast 불필요(`NO_BROADCAST_NOT_REQUIRED`: S4 단일 실행/고비용 private raw 보존, compact Git control plane). 자동 afterany broadcast/새 agent completion callback을 생성하지 않는다.

## 4. Baseline 재사용과 실제 calibration 의존성

AlphaEdit / MEMIT / AlphaEdit-BLUE / MEMIT-BLUE / AlphaEdit-L4_only는 기존 결과·checkpoint만 재사용, REFIT4는 기존 보조 근거만 연결. W0/order/model/config/layer/evaluator/분모/backend 및 증거 범위를 reuse manifest에 기록한다. 다른 조건의 warm REFIT4를 paired W0 baseline으로 표기하지 않는다. 미가용 자료는 REUSE_SCOPE_LIMITED/NOT_AVAILABLE; 표 전체를 기다리며 독립 구현을 중단하지 않는다.

N4 calibration은 예외적인 실제 의존성:
1. 동일 W0/order/native L4/L2=1의 B1–B10 **전 endpoint** 또는 검증된 당시 저장 delta materialization 가능성을 조사한다.
2. 새 고정 C4 S64/p0에서 forward만 하여 D64 10개를 구한다. 원 source target/writer를 호출해 missing N4 trajectory를 재생성하지 않는다.
3. `b=max(b_num,.9*max(D64_N4_B1...B10))` 고정. b_num/epsilon/screen tolerance/dtype 등 numerical choices는 기술자료로 사전 lock, 성능 tuning0. 적용 tolerance가 있으면 실제 screen ceiling b+tolerance도 공개.
4. 일부 checkpoint의 max를 전체 max로 쓰지 않고, 다른 corpus/warm W50→60 KL로 대체하지 않는다. 전부 없으면 CALIBRATION_MISSING. 가능한 source/data/teacher/기술 준비 후 G0_BLOCKED로 인계하며 scientific submit0.
5. N4 새 forward 비용과 과거 editing 비용 분리. 모든 baseline C4/downstream 재평가를 선행조건으로 추가0.

## 5. C4 / teacher 데이터 경계

C4-WebRef-v2 실제 builder를 구현한다. schema `text/url/timestamp`, revision1588ec454efa1a09f29cd18ddd04fe05fc8653a2. Prefix probe를 full download/hash라고 부르지 않는다. full bytes/SHA/gzip CRC/JSONL row count/README hash를 남긴다.

전체 고정 shard hash sampling seed20260915, source row ID와 우선순위·충원·near-duplicate·URL normalization은 JSON 그대로. Train512=S64+Dev128+Reserve320, validation Report256, 총768을 첫 model loss 전에 seal. 도메인 겹침/Wikipedia 허용 및 tagging. 평가 오염 checker는 입력 fingerprint만 사용하고 답/score/future subject를 sampler에 주지 않는다. 미접근 overlap 범위는 미검증으로 기록한다.

Exact W0 tokenizer로 자연 text256+BOS1, input257, target [129,257), logits [128,256) 128 positions. Chat/EOS/문서연결/decode-reencode0. W0 Llama3-8B-Instruct revision8afb486c1db24fe5011ec46dfbe5b5dccdb575c2, FP32/eval, V128256 assert.
Full-vocab FP32 logp0 teacher는 S64+Dev128=192만 먼저 생성(12608077824 tensor bytes, 약11.7422GiB). CPU mmap 8docs/shard, 초기GPU microbatch1. Document mass/position reduction과 kernel/tokenizer/cache hashes pin. KL(p0||pW), vocab sum→128position mean→document mean. Reserve/Report teacher 생성과 Report loss는 연기. Dev는 W5/W10 관찰 전용. Native Wikipedia P/C0 변경0.

## 6. BG-1 수학과 native 연결

W0 cold L4 only/L2=1, fixed10k 공식 loader로 **앞1000**을 verify/load_prefix. S4 asset `local/datasets/counterfact-fixed-10k-v1/counterfact.json` SHA3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1; whole ordered root5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729. 첫1000 order는 기존 JVP prefix와도 결속; shuffle/새sample/성능기반교체0. W50/M50 warm 반입0.

한 batch의 native Z/H/K/P/M, context-repeat와 native 연산 예외를 freeze하고 S(R)=RA를 실제 solver에서 유도한다. Canonical H와 context K를 동일 반응으로 가정하지 않는다. Phi=Wparent+eta*RA는 모든 token의 실제 L4 down_proj update이다. Source-native RHS solve의 raw endpoint는 보존하며 gradient map/FP32 RA와 direct native 차이를 수치 기록한다. 임의 orthogonal projection/Cholesky/inverse/SPD 대체0. 필요한 고정 A 구성은 원 map과 검증한다.

Native target 요청당1회/max24Adam/25loss, early stop·local teacher·anchor·clamp source 그대로. Inner z 재학습0, derivative 시 solver/parent/P/K/M fixed. Canonical Ecur=request별 desired-token평균 후100request평균; native multi-context loss와 구분한다.

F=Ecur+mu*b_tau((b-D64)/b), zeta=.25,tau=.1,mu=.01. Native post-write preview에서 residual gradient1/B100. Microbatch별 scalar barrier를 별도로 적용해 평균하지 말고 **전체 D64에서 정해진 slope**로 정확히 누적하여 원 objective gradient를 만든다. 추가 forward/teacher I/O가 필요하면 비용에 포함한다. W0 self-KL gradient0은 실패 아님.

alpha=zeta*||S(Rprop)||/(||S(G)||+epsilon), target anchor/radius ball projection 후 executable correction trust<=zeta*native action. Zprop ball 만족 확인. Numerical zero/alpha cap/epsilon/rounding 등 문서가 맡긴 선택만 사전 lock, 성능 따라변경0.
후보 RAW1/CORR1/CORR.5/CORR.25 최대4; actual materialized screen D64≤fixed b(+declared numerical tolerance), feasible 중 Ecur 최소, 동률 actual delta norm→고정order. 모두실패면parent. 별도nativefallback/추가candidate/line search0.
100target 후 batchwrite, innerappend0/processedB100당 finalizer1(정상zero-write도1). 최종TF-strict fulfilled accepted-label/acceptance-time loss와 write acceptance를 구분. BG-1 old feedback0, P/N/Historical/Audit/MMLU/FutureN/Dev/Report를controller입력으로 주지 않는다.

## 7. 검증 / persistent 실행 / 비용

Reference/teacher/입력 scoring shift/selfKL, raw-native와 route-disabled 연결, all-token actual materialization, residual FD/VJP, full-D64 gradient와 microbatch mass, C2 join/negative slack, target-ball/trust, candidate restore, history exactlyonce/zero-write, selected/nonselected guards와 resume-state를 CPU 및 좁은 실제 GPU 기술 검사로 확인한다.
기술 single-B100 native comparator는 BG 구현 검사용이며 별도 baseline sequential rerun 허가가 아니다. Native target quota를 중복 소비하면 technical ledger로 공개한다. 수치차이를 은폐하는 허용오차 완화/결과선별0; finite efficacy와 technical validity 구분.

Red preflight는 data/eval leakage, one-policy scope, resource/memory/session, calibration10 completeness, actual source and native relation, safe writes, raw-free Git를 점검한다. Red warn은 근거와 함께 진행 가능, unresolved block은 해당 실행 HOLD. 독립성이 있는 복잡한 builder/optimizer/감사에만 bounded subagent 활용하고 단순 확인은 직접 수행한다. Parent SH4가 통합/commit하며 서로 변경을 되돌리지 않는다.

준비 완료 후 BG-1 **단일 persistent B1–B10 job** 제출. 수동 B2/후속 agent 메시지 의존0. Batch별 actual W4 또는 검증 가능한 delta + M/history + P binding/context/RNG/ledger/next ordinal·terminal/eval/typed failure를 durable 저장. Full-model-per-candidate 영구복사0. Stage별 compute-z/route current/control F/B/token/microbatch/candidates/rejection/materialization/finalization/evaluation/I/O/setup/allocated GPUh를 분리한다. .9 calibration 개발의 미래정보 사용과 비용을 숨기지 않는다.

사전 programmed평가: each batch current canonical R/P/N·strict·NLL, atwrite→W10/first500 W5→W10/all1000, active/superseded/coverage, D64each/Dev128W5W10. 원분모 R1000/P2000/N10000은 실제 canonical inventory로 확인. W0부터 시작하므로 old5000/full6000 분모반입0. 미측정은 NOT_RECORDED. Audit/MMLU/Report 독립평가는이번연기. 실행 중/후 성능보고를 agent 자동해석으로 확장하지 않는다.

## 8. G0 / 정지 / 인계

G0_PASS는 reference768+teacher192실물, source/config/order+기술검사+고정calibration, 실제persistent job실행, **firstB100 처리완료** finite/candidate또는정상zero-write/원분모100/finalization1/nextordinal100/resume state, agent없이B2–B10진행 구조가 전부 있을 때만.
성능상승/NS비악화/전요청성공/rejection0/1.5×는gate아님. JobID나model load만으로PASS0. scheduler대기는G0_PENDING, 긴blocking대신상태보고. G0전 승인된 재현기술오류만 새attempt수리 가능; 동시에중복scientificchain0. 확정FAIL/BLOCKED인계뒤자동repair금지.

G0_PASS/terminalFAIL/BLOCKED 확인 즉시 task-local `resume-manifest.json`을 작성하고 G0 factual report·compact handoff만 마친 뒤 **WAITING_USER_RESUME**로 종료. 모든agent polling/logtail/sleep-loop/heartbeat/timer/automation/callback turn/terminal대기/분석·후속submit중단. 제출된 정상BGjob은 취소/hold하지 않고 예정log/checkpoint/eval과 fatal integrity fail-stop만 유지. 다른task공용서비스 일괄중단0.

Resume manifest에는 instruction/task/run/host/owner, exec source/branch/config, reference/teacher/calibration hashes, reuse범위, 준비/평가/scientific job IDs/dependencies, 마지막관찰time/state/batch/nextordinal, checkpoint/history/ledger/RNG, terminal/eval 예상path 및현재존재여부, G0각항목증거를 넣는다.
`monitoring_active=false, automatic_resume=false, resume_trigger=explicit_user_call`; agent상태와job마지막상태분리. Localmanifest SHA와 compact요약을 GH에전달. GH는 `plans/global/2026-09-15-bg1-c4-ours-first-handoff.md`에 사용자인계를 결속한다.
최종 source/report/G0 사실과 전체1000 완료를 구분하며 모든 deferred정책·baseline editing0 명시. 사용자recall전 결과해석/claim/새policy/후속실험0.
