# Session 02 — Compute-aware Main Table Structural Lock v3

- 작성: **2026-08-03 KST**
- GH: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- 상태: **`PRIMARY_STRUCTURAL_LOCK; NUMERICAL_LOCK_REVISION_REQUIRED; GPU_SLURM_HOLD`**
- method: **ODE-Edit**
- actuator: **MEMIT first**
- models: `Llama3-8B-Instruct`, `Qwen2.5-7B-Instruct`
- compute-review input SHA-256:
  `028f712d9e69cc5a5fe8ee1fea620562931b8afa0b618b7527e11fa0b1913eba`

이 문서는 첫 main table의 primary execution spec이다. 이전
`2026-08-03-session02-fast-main-table-spec.md`와 충돌하면 이 문서를 우선한다.

2026-08-03 SH1 head `8275f653` 검토 뒤 추가된 v3가 v2의 counter, timer,
trust-radius 조항보다 우선한다. 해당 head는 reusable invariant skeleton과 35개 CPU test를
제공했지만 실제 model-scale backend와 executable runner는 아직 제공하지 않았으므로 numerical
lock과 GPU 실행 권한을 부여하지 않는다.

## 1. GH 판정

첨부 계산량 리뷰의 핵심은 수용한다.

- accepted macro-round의 평균을 `1--2`로 맞추지 않는다.
- trajectory length는 semantic first-hit, trust accuracy와 state nonlinearity가 만든
  결과 변수다.
- 비싼 단위는 단순 step 수가 아니라 current-state field relinearization이다.
- reject 뒤 state가 같다면 field를 재구축하지 않는다.
- full-weight 또는 layer별 backward 대신 한 rewrite backward에서 모든 low-rank actuator의
  directional derivative를 얻는다.
- compute는 performance와 함께 co-primary frontier로 판정한다.

다음은 그대로 수용하지 않고 보정한다.

1. `S_max=6`은 외부 논문이 보장한 최적값이 아니라 공통 safety/resolution cap이다.
2. 별도 6-arm 4-edit screen은 main table을 늦추므로 profiler와 resolution screen을
   하나로 합치고 네 arm만 사용한다.
3. ODESteer의 7--9% throughput 감소는 activation-space 결과이므로 weight-editing
   overhead 예측값으로 사용하지 않는다.
4. entry에서 event를 이미 만족하는 edit가 있을 수 있으므로 `K_acc=0`도 명시적으로
   기록한다.
5. 900--1000 edit age 분석은 1K stage 전에는 열지 않는다.

## 2. Co-primary objective

ODE-Edit은 다음 두 축을 동시에 만족해야 한다.

1. 같은 semantic terminal에서 Static/Scalar보다 retention frontier가 좋아야 한다.
2. 추가 성능이 비싼 field evaluation과 GPU time을 정당화해야 한다.

Full이 기능적으로 좋아도 compute가 과도하면 deployable efficiency claim은 열지 않는다.
Static 또는 One-refresh가 같은 기능을 더 싸게 만들면 ODE를 단순화한다.

## 3. Canonical compute accounting

Counter는 실제 model call과 논리적 controller action을 섞지 않는다.

- `N_z`: direct-z optimization count
- `N_model_fwd`: evaluation을 제외한 실제 full-model forward invocation count
- `N_event_fwd`: `N_model_fwd` 중 rewrite event 측정에 사용한 invocation count
- `N_field_state_fwd`: `N_model_fwd` 중 current-state field 구축에 사용한 invocation count
- `N_field`: accepted-state joint field build count
- `N_proposal_build`: native, scalar, synchronous, coordinate를 포함한 proposal-build count
- `N_native_sweep`: ordered native terminal sweep count
- `N_bw`: rewrite backward count
- `K_acc`: accepted parameter transition count
- `N_trial`: accepted와 rejected functional trial의 논리적 시도 수
- `N_reject`: rejected trial count
- `N_write`: committed write transaction count
- `N_eval`: controller freeze 뒤 evaluation invocation count

`N_trial`은 forward count가 아니다. Trial에서 수행한 event forward는 `N_trial`과
`N_event_fwd`에 각각 기록하되 비용식에서 두 번 더하지 않는다. 과거 `N_state_fwd`는 이 정의와
중복되므로 폐기하거나 위 counter들의 합으로만 파생한다.

공식 비용은 counter에 평균 단가를 곱해 추정하지 않고 동기화된 component timer의 합으로
계산한다.

\[
C_{editor}=C_z+C_{checkpoint}+C_{event}+C_{proposal}+C_{bw\_hook}
+C_{QP}+C_{trial}+C_{commit}+C_{restore}+C_{terminal\_geometry}.
\]

`controller_gpu_seconds`와 `controller_wall_seconds`를 모두 보고한다. Native와 Scalar의
ordered terminal proposal, entry checkpoint/restore, terminal net geometry도 이 합에서 빠질 수
없다. Context-template 생성이나 model warm-up 같은 run-level setup은 별도 계측하고

\[
C_{amortized/edit}=C_{editor}+C_{setup}/N_{edits}
\]

를 함께 보고한다. `N_eval`과 downstream evaluator 비용은 controller cost와 별도다.
`Full/Native` ratio는 direct-z를 포함한 동일 범위의 editor time으로 계산하며 GPU time과 wall
time을 둘 다 제시한다.

필수 불변식은 다음이다.

- `N_z=1` per outer edit
- per-layer backward `=0`
- field rebuild on reject `=0`
- dense weight copy for trial `=0`
- accepted step마다 full target weight CPU backup `=0`
- first-hit 뒤 z/proposal/field/trial/write `=0` (Native 포함)
- QP coefficient와 actual functional write coefficient 일치

## 4. Efficient field/trial implementation

### Field build

1. 허용 rewrite context를 batch forward해 current residual/key와 필요한 activation을 모은다.
2. 모든 edit layer proposal을 같은 state ID에서 구성한다.
3. 한 rewrite backward의 module input/grad-output hook으로 모든
   `a_l=-D Phi[B_l]`를 계산한다.
4. target weight gradient tensor를 materialize하지 않고 low-rank contraction만 수행한다.

초기 GPU profiler에서 hook 값은 scalar-gate autograd 또는 finite directional check와
locked tolerance 안에서 일치해야 한다. Target weights를 직접 `torch.autograd.grad`의
inputs로 넘기는 dense-gradient 구현은 technical reference oracle로만 허용하며 main
backend와 cost 측정에는 사용하지 않는다.

### Functional trial

Dense weight를 복사하지 않고 linear output에 low-rank branch를 더한다.

\[
W_lx \mapsto W_lx+y_lU_l(V_l^\top x).
\]

Reject이면 branch를 폐기하고 같은 field에서 `h`만 줄여 QP와 trial을 다시 실행한다.

Accepted write의 failure safety는 edit-entry checkpoint 하나와 outer transaction이 담당한다.
매 accepted round마다 모든 target weight를 CPU로 복제하는 구현은 계산량 계약을 위반한다.
Injected mid-commit failure에서 weight, RNG와 controller state가 entry와 exact identity여야 하며
checkpoint/restore 시간은 숨기지 않는다.

### Event batching

Old/new teacher-forced score는 가능한 경우 하나의 padded combined batch forward에서 계산하고
target별 length normalization은 유지한다. Combined 결과는 two-forward reference와 locked
tolerance 안에서 같아야 한다. Concrete backend 제약으로 불가능하면 두 invocation을
`N_model_fwd`와 `N_event_fwd`에 그대로 기록하고 P0 technical HOLD로 보고한다.

### Terminal geometry

Capacity는 micro-step energy 합이 아니라 entry-to-terminal **net write**의 C-energy다. 이
계산도 controller timer에 포함한다. P0 exact dense reference가 controller cost의 10%를 넘으면
P1 전에 accumulated low-rank cross-term evaluator를 구현하고 dense reference identity를
통과시킨다.

### Accepted-trial graph reuse

P0에서는 outcome을 보지 않고 두 backend를 profile한다.

- `no_grad_trial`: 낮은 peak memory, accept 뒤 next field를 위한 forward 재실행
- `cached_trial_graph`: trial graph를 보존하고 nonterminal accept 뒤 in-place weight commit
  **전에** 그 graph에서 backward와 next-field cache를 완성한 후 동일 delta를 commit

MEMIT proposal builder가 committed dense weight를 다시 읽어야 해 이 순서를 지킬 수 없으면
cached mode는 technical fail로 닫는다. 두 backend가 event/progress/applied write에서
수치적으로 같아야 한다. 두 모델 모두 memory
cap 안에 있고 pooled controller GPU time이 더 낮은 하나를 **공통 backend**로 lock한다.
모델별 backend 선택은 금지한다.

## 5. Trajectory와 common resolution cap

Accepted round 수에 평균 목표를 두지 않는다.

\[
0\le K_{acc}\le S_{max},\qquad S_{max}=6\ \text{(initial common cap)}.
\]

- entry-hit은 `K_acc=0`으로 별도 기록한다.
- first-hit이면 즉시 종료한다.
- reject는 `K_acc`에 포함하지 않고 current field를 재사용한다.
- cap 도달 시 positive slope, acceptable trust, event deficit이 남으면
  `resolution_cap_unresolved`다.

P1의 pooled 8 Full trajectories 중 unresolved cap이 2개 이상이면 같은 affected cases에
`S_max=8` continuation check를 연다. 1개면 P2까지 6을 유지한다. P2에서 pooled 40 Full
trajectories의 10% 이상이 unresolved cap이면 affected cases를 8까지 연다. 7--8 round에서
절반 이상 first-hit하면 P3의 common cap을 두 모델 모두 8로 올리고, 아니면 6을 유지하며
cap failure를 숨기지 않는다. 이 조건은 retention outcome을 보지 않고 resolution evidence만
사용한다.

## 6. Fast execution progression

### P0/P1 — fused component profiler and resolution identity

| Model | Edits | Order | Arms |
|---|---:|---:|---|
| Llama | 4 | 1 | Native, Static, One-refresh, Full |
| Qwen | 4 | 1 | Native, Static, One-refresh, Full |

- 첫 1 case/model은 backend profiler와 numerical identity 전용으로 중복 실행하고 그
  profiler output은 scientific 우열에 사용하지 않는다. Common backend를 lock한 뒤 네
  case 모두를 selected backend로 실행한 canonical result만 P1에 포함한다.
- `One-refresh`는 최대 2 accepted transitions를 허용한다.
- Full은 initial common `S_max=6`을 사용한다.
- Scalar와 Ordered는 이 단계에서 unit/schema contract만 통과한다.
- Llama/Qwen job은 같은 batch로 제출한다.

P0/P1 산출물은 component time, peak memory, hook identity, field/retry invariant,
`K_acc`와 proposal/allocation drift다.

P0를 열기 전에 toy backend가 아닌 ODE-edit-side concrete EasyEdit/MEMIT backend,
differentiable smooth event, local executable runner와 fail-closed sbatch template이 있어야 한다.
Dry-plan renderer만으로는 P0 실행 준비를 통과하지 않는다.

### P2 — 10-edit ODE/compute identity

| Model | Edits | Orders | Arms |
|---|---:|---:|---|
| Llama | 10 | 2 | Native, Scalar, Static, Ordered, Full |
| Qwen | 10 | 2 | Native, Scalar, Static, Ordered, Full |

여기서 처음으로 retention--`N_field`와 retention--GPU-sec frontier를 판정한다.
One-refresh는 P1 ablation 결과를 사용하며 P2 전체 matrix에 추가하지 않는다.

### P3 — first 100-edit main table

| Model | Edits | Orders | Main arms |
|---|---:|---:|---|
| Llama | 100 | 2 | Native, Scalar, Static, Full |
| Qwen | 100 | 2 | Native, Scalar, Static, Full |

Ordered는 joint-synchronous mechanism ablation으로 P2 표에 남기고 100-edit main matrix에서
제외한다. 이것으로 이전 5-arm main 설계보다 100-edit arm compute를 20% 줄인다.

AlphaEdit, 1K+, RK4/Heun, hard signed barrier는 P3 뒤에만 연다.

## 7. Co-primary tables

### Performance table

| Method | Current rewrite ↑ | Retention AUC ↑ | Final retention ↑ | Paraphrase ↑ | Neighborhood ↑ |
|---|---:|---:|---:|---:|---:|

### Compute/trajectory table

| Method | GPU sec/edit ↓ | Wall sec/edit ↓ | Total/Native ↓ | Model F/edit ↓ | Proposal/edit ↓ | N_field/edit ↓ | Trial/edit ↓ | Median K | P90 K | Pr(K>=3) | Reject/edit ↓ | Peak GiB ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Component profiler에는 setup, `C_z`, checkpoint, event forward, state forward, native/synchronous
proposal solve, hook backward, QP, trial, retry, commit, restore와 terminal geometry를 따로 둔다.
Evaluation time은 섞지 않는다.

Mechanism panel에는 field C-cosine, efficiency ranking turnover, active support turnover,
allocation cosine, first-hit round, Omega/max-load/Gini를 둔다. 100-edit에서는 edit age별
`K_acc`와 cost도 보고한다.

## 8. P2 gate

다음 조건을 모두 사용한다.

1. Full current acquisition이 두 모델 모두 Native 대비 non-collapse
2. Full-vs-Scalar와 Full-vs-Static retention 방향이 어느 모델에서도 material negative가
   아니고 pooled paired direction은 positive
3. proposal/ranking/allocation drift가 numerical noise보다 큼
4. Full이 Static 또는 One-refresh에 retention과 compute 양쪽에서 strictly dominated되지 않음
5. reject field rebuild, per-layer backward, dense trial copy, post-first-hit work가 모두 0
6. 두 모델에 같은 backend, cap, event, trust와 fallback 적용

Compute expansion 기준은 다음처럼 prelock한다.

- `Full GPU-sec / Native <=3.0`: green
- `(3.0,4.0]`: yellow; Full-vs-Static frontier gain이 있어야 P3를 열며 efficiency claim은 보류
- `>4.0`: P3 HOLD; backend/field cost를 먼저 줄이거나 method를 단순화

이 수치는 기존 fixed four-hop C3의 4.80--6.93배 overhead를 실질적으로 낮추기 위한 GH
engineering gate이며 이론적 상수가 아니다.

## 9. Kill/pivot rules

- 대부분 `K_acc<=2`이고 Full과 Static/One-refresh가 같음: static 또는 one-refresh로 pivot
- `K_acc>=3`이 자주 나오지만 frontier gain 없음: ODE necessity kill
- Full이 Static보다 좋아도 `>4x` compute: full main expansion HOLD, efficient backend 우선
- cap unresolved가 많음: common 8-round resolution check; 모델별 cap rescue 금지
- rejection이 비용의 대부분: field가 아니라 trust initialization 문제로 판정
- Omega 감소와 functional retention 악화: capacity objective 제거 또는 재정의
- 한 모델만 살리는 step/event/backend/fallback 필요: common method gate fail

## 10. Outcome-free numerical lock

P0 GPU 전에 SH1은 다음을 outcome 0건 상태에서 제안하고 GH가 승인한다.

- exact 4 case IDs, canonical order와 hash
- tokenizer/leading-space/context manifest와 event tolerance
- smooth surrogate, `h_0`, progress scale, trust accept/reject/expand constants
- QP/equality/support tolerance와 `S_max=6`
- scalar alpha grid/bisection tolerance와 seed
- hook-vs-reference identity tolerance
- cached/no-grad backend equivalence 및 common selection rule
- timer synchronization/warm-up/repetition과 GPU-sec definition
- artifact schema, P0/P1 command, expected memory/time

P0 profiler 뒤에는 prelocked technical selection rule로 backend만 고른다. Event, step,
trust, cap, case 또는 order를 performance outcome에 맞춰 바꾸지 않는다. Canonical P1은
selected common backend로 네 case를 다시 실행한다.

Trust radius는 절대 `h_0=h_max=0.25`로 두지 않는다. Entry의 raw synchronous joint proposal을
unit-C normalization하기 전에 측정한 `D_sync_entry`를 공통 reference로 하여

\[
h_0=\tfrac14D_{sync,entry},\qquad h_{max}=\tfrac12D_{sync,entry}
\]

로 정한다. 이는 native distance를 terminal budget으로 강제하는 규칙이 아니라 초기
trust-radius의 단위 보정이다. `D_sync_entry<=denominator_epsilon`이면 fail-close한다. P0에서는
같은 entry의 Native arm으로 `h_0/D_native`를 진단하고, 한 모델이라도 `[1/8,1/2]` 밖이면 P1
lock을 HOLD한다. 모델별 radius rescue는 금지한다.

## 11. Execution boundary

- EasyEdit source는 read-only; ODE-edit-side hook만 사용
- precomputed covariance/null/projector/Wikipedia artifact 재계산 금지
- raw output/log은 ignored `local/`
- SH1이 implementation/P0/P1/P2/P3를 실행하고 SH2는 locked-source reproduction 담당
- server1 cap 4 GPU, `198117 MiB/GPU`; 각 제출은 별도 envelope 필요
- Llama와 Qwen은 같은 batch로 제출
- 현재 revision envelope의 GPU cap은 0이며 Slurm 제출은 금지다. Concrete backend, fair
  accounting, revised lock과 CPU/dry gates를 GH가 다시 승인한 뒤에만 별도 P0 submit envelope를
  발행한다.
- 결과별 별도 Terra Ultra analysis report 요구 유지; runtime mismatch면 claim promotion HOLD
- 현재 Slurm은 numerical lock과 GH submit envelope 전까지 HOLD
