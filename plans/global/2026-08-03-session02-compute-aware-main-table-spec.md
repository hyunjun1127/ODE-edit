# Session 02 — Compute-aware Main Table Structural Lock v2

- 작성: **2026-08-03 KST**
- GH: `019fb1ea-03cb-7c20-bb3b-eba5f8d6f5f2`
- 상태: **`PRIMARY_STRUCTURAL_LOCK; INVARIANT_IMPLEMENTATION_ALLOWED; NUMERICAL_LOCK_AND_SLURM_HOLD`**
- method: **ODE-Edit**
- actuator: **MEMIT first**
- models: `Llama3-8B-Instruct`, `Qwen2.5-7B-Instruct`
- compute-review input SHA-256:
  `028f712d9e69cc5a5fe8ee1fea620562931b8afa0b618b7527e11fa0b1913eba`

이 문서는 첫 main table의 primary execution spec이다. 이전
`2026-08-03-session02-fast-main-table-spec.md`와 충돌하면 이 문서를 우선한다.

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

- `N_z`: direct-z optimization count
- `N_state_fwd`: trial과 별도로 실행한 current-state forward count
- `N_field`: accepted-state actuator field build count
- `N_bw`: rewrite backward count
- `K_acc`: accepted parameter transition count
- `N_trial`: accepted와 rejected trial forward count
- `N_reject`: rejected trial count
- `N_eval`: controller freeze 뒤 evaluation forward count

Controller 비용은 다음처럼 분리한다.

\[
C_{edit}=C_z
+N_{state\_fwd}C_{state\_fwd}
+N_{field}(C_{proposal}+C_{bw\_hook})
+N_{trial}C_{trial}
+C_{QP}+C_{commit}.
\]

`N_eval`과 downstream evaluator 비용은 controller cost와 별도 보고한다.
`GPU sec/edit`와 `Full/Native` ratio는 direct-z를 포함하고 `N_eval`을 제외한 end-to-end
editor time으로 계산한다.

필수 불변식은 다음이다.

- `N_z=1` per outer edit
- per-layer backward `=0`
- field rebuild on reject `=0`
- dense weight copy for trial `=0`
- first-hit 뒤 field/trial/write `=0`
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

| Method | GPU sec/edit ↓ | Total/Native ↓ | State F/edit ↓ | N_field/edit ↓ | Trial F/edit ↓ | Median K | P90 K | Pr(K>=3) | Reject/edit ↓ | Peak GiB ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Component profiler에는 `C_z`, state forward, proposal solve, hook backward, QP, trial,
retry, commit 시간을 따로 둔다. Evaluation time은 섞지 않는다.

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

## 11. Execution boundary

- EasyEdit source는 read-only; ODE-edit-side hook만 사용
- precomputed covariance/null/projector/Wikipedia artifact 재계산 금지
- raw output/log은 ignored `local/`
- SH1이 implementation/P0/P1/P2/P3를 실행하고 SH2는 locked-source reproduction 담당
- server1 cap 4 GPU, `198117 MiB/GPU`; 각 제출은 별도 envelope 필요
- Llama와 Qwen은 같은 batch로 제출
- 결과별 별도 Terra Ultra analysis report 요구 유지; runtime mismatch면 claim promotion HOLD
- 현재 Slurm은 numerical lock과 GH submit envelope 전까지 HOLD
