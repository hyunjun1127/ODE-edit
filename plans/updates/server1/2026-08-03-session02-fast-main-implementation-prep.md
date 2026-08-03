# Session 02 compute-aware implementation prep — server1/SH1

## 판정

`ODEEDIT-S02-FAST-MAIN-IMPL-PREP-V1` 및
`ODEEDIT-S02-COMPUTE-AWARE-IMPL-PREP-V2`의 CPU implementation-prep 범위는
완료했다. 구현 commit은
`9792597f2f43d6482468ebf2d79a82fbcb253b9b`이며, scientific outcome은 0건이다.
GPU는 사용하지 않았고 Slurm submit, push, merge도 수행하지 않았다. Numerical lock은
GH 승인 전 제안 상태이며 P0/P1 실행 권한을 부여하지 않는다.

작성 시각은 `2026-08-03 16:24:32 KST`다. Session은
`019fc63e-5217-7250-9c22-c5b2ec4248f0`, server/role은
`server1`/`head-server1-sh1`, worktree는
`/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit`, branch는
`codex/odeeditsh1`이다.

## Canonical lineage와 경계

- protocol SHA-256:
  `5c2221f221886d34f670935274d1d0ceb34436ee813a97d6c5a909325171ad37`
- GH management object `7a84cd2`의 local cherry-pick: `e2a247a`
- corrected invariant checkpoint: `563da28`
- canonical main object `67bcf9b8cb89e9e8c544ad1c94c92d639dcd2229`의 local
  cherry-pick: `e40dc9b`
- compute-aware spec SHA-256:
  `35c7a0ef8c25d3a339f9cb41a816e4b6b9702341a989822812ebaed0fe7902a0`
- implementation/lock commit:
  `9792597f2f43d6482468ebf2d79a82fbcb253b9b`
- EasyEdit는 read-only로만 조회했다. 기존 foreign dirty state를 수정, stage, commit,
  stash 또는 reset하지 않았다. Artifact 재계산·다운로드도 0건이다.
- 별도 direct-z 임시 track의 code/artifact/report는 열거나 결합하지 않았다.

## 구현 결과

`project/run_scripts/ode_edit_method/`에 다음 공통 contract를 구현했다.

- Native, Scalar first-hit, Static synchronous, Ordered adaptive, Full ODE-Edit와
  P0/P1용 One-refresh가 동일 `ControllerConfig`와 event/trust policy를 사용한다.
- Full/One-refresh는 accept 뒤 모든 layer를 같은 current snapshot에서 다시 구성한다.
  Static은 entry proposal/slope/share를 고정하며 current event와 common trust가 global
  magnitude만 바꾼다. Ordered는 ascending cyclic coordinate를 current state에서 만들고
  layer revisit를 허용한다.
- Direct-z는 arm/edit당 정확히 한 번 lazy compute한다. Entry first-hit 뒤에는
  field/trial/write가 0이다.
- Rejected trial은 read-only functional hook과 CPU/CUDA RNG 복구를 사용한다. 동일
  state/field/proposal에서 trust radius만 바꾸며 field/QP input을 재사용한다.
- Trial과 commit을 분리했다. Primary trial은 target weight copy/mutation 없이 low-rank
  functional hook으로 평가하고, accept 뒤 동일 coefficient를 별도 transaction으로
  적용한다. Adaptive accepted write는 full dense update 대신 row-bounded low-rank update를
  사용하며 failure 시 CPU cleanup backup으로 모든 target weight를 exact restore한다.
- `ActuatorDirectionalHook`은 target weight `.grad`를 만들지 않고 한 rewrite backward에서
  module input `x`와 grad-output `g`를 이용해 모든 `g^T U(V^T x)` contraction을 계산한다.
  Target-weight `torch.autograd.grad`는 CPU reference oracle로만 남겼다.
- Current MEMIT `cached_trial_graph`는 `UNSUPPORTED_FAIL_CLOSED`다. 공통 scientific
  backend는 `no_grad trial -> accepted commit -> dedicated state rebuild`이며 cached mode는
  arm 또는 성능 선택 후보가 아니다.
- `N_z`, `N_state_fwd`, `N_field`, `N_bw`, `K_acc`, `N_trial`, `N_reject`,
  `N_eval`, `N_write`, component CPU/GPU timer, controller GPU seconds와 peak memory
  recorder를 연결했다. Evaluation time은 controller GPU seconds에서 제외된다.
- Runner는 rewrite-only `ControllerRequest`만 받는다. Paraphrase/neighborhood/held-out
  payload를 포함한 일반 mapping은 action 전에 표현할 수 없다.
- Omega는 completed outer edit의 terminal net write를 정확히 한 번 append하며, 실패 시
  entry restore가 끝나기 전에는 append하지 않는다.

## CPU gate

다음 명령이 35 tests를 모두 통과했다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=error \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B -m unittest discover \
  -s project/run_scripts/ode_edit_method/tests -v
```

검증 범위는 hook/reference identity(`atol=1e-10` CPU), target `.grad is None`, storage
pointer/version 보존, functional-trial/accepted-write identity, accepted-write failure cleanup,
CPU RNG checkpoint behavior, CUDA RNG capture/restore code-path inspection, reject field reuse,
QP coefficient exact apply, Omega
subdivision invariance, length-normalized event, information firewall, first-hit freeze,
Static/Ordered/Full/One-refresh identity, scalar alpha cap/bracket, runtime counter accounting,
cached fail-close와 lock/launcher 변조 거부를 포함한다.

추가 gate도 통과했다.

- `bash -n scripts/check-agent-access.sh`
- `scripts/check-agent-access.sh --staged`
- `git diff --cached --check`
- P0/P1 JSON dry-plan render 및 strict JSON parse

## Outcome-free numerical-lock 제안

제안 파일은
`project/run_scripts/ode_edit_method/numerical_lock_proposal.json`이다. File SHA-256은
`3cfd8f98e6f80c0a9cc614712f4712855256bb9da83bcded0dc09d66c43207e4`, canonical
proposal ID는
`12ac2b1051cb9fbd1326bc453e94124e7e26756481975fcbc492705641c46b54`다.

### Case, order, model/context pin

- CounterFact SHA-256:
  `d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f`
  (`45,108,470` bytes, `21,919` rows)
- selection seed: `ode-edit-session02-fast-main-cases-v1`
- exact cases/order: `2022, 12498, 20964, 768`
- order SHA-256:
  `e7746f38ba36f8d36815b4e58084ad408be6cb64f77d04661c6bc4ca688e08f7`
- request-order SHA-256:
  `bffe1b3655ae73223f9901f8e5d31af86779740a2682db142a0bf1415a66c57c`
- P0 one-case prefix hash:
  `03f87dc82db1e1eb51c679506cad9e918f1776034006ac0aba44901246113adb`
- common execution seed: `17`
- Llama revision:
  `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`; context manifest:
  `3020b3f5cea62e6cfbd173f0c99a4348cecf7e84425bb087720ced7f395482e5`
- Qwen revision:
  `a09a35458c702b33eeacc393d103063234e8bc28`; context manifest:
  `e0c5f61d874334a2cab26f82fb3594d9ab88c9e5816be390274a0bb5e98c93bd`
- Context group sizes는 `[1,5]`이며 seed 17로 fresh regenerate한 뒤 action 전에 manifest
  ID가 일치해야 한다. Raw template/prompt/target은 Git에 기록하지 않는다.
- Target은 없을 때만 leading space 하나를 붙이고 special token 없이 tokenize한다.
  Full context는 special token을 사용하며 right padding, object-token mean log-likelihood,
  suffix identity와 empty-object fail-close를 잠근다.

Observed MEMIT hparam pins는 Llama
`2b81838b49b1a5e0e4ef41f229216fda473200093fcbed6bd040f07a70d43818`
(`631` bytes), Qwen
`fccad05cf749c710ba0ce58ae24203f0bb90d9bf488966a5184dacd489a0311b`
(`639` bytes)다. 향후 실행 전 existing fixed-artifact manifest의 covariance/projector/
Wikipedia-stat hash와 size를 read-only preflight해야 한다.

### Event, controller, solver

| 항목 | 제안값 |
| --- | --- |
| hard event | `max_context(mean_ll_old - mean_ll_new)` |
| first hit / hard worsening tolerance | `1e-6` / `1e-6` |
| smooth surrogate / `tau` | mean-log-sum-exp deficit / `0.1` |
| `h0`, common radius cap | `0.25`, `0.25` |
| `kappa`, `beta` | `1.0`, `0.8` |
| reject/expand ratio | `0.1`, `0.75` |
| radius contraction/expansion | `0.5`, `1.5` |
| `S_max`, One-refresh cap | `6`, `2` accepted transitions |
| max rejects per state | `4` |
| slope/progress support epsilon | `1e-10`, `1e-10` |
| QP equality/trust tolerance | `1e-8`, `1e-8` |
| trust/unit-C/load denominator epsilon | `1e-12` each |
| QP dtype / dual iterations | `float64`, `128` |
| scalar grid | `0, 1/16, ..., 1` |
| scalar bisection | width `2^-14`, max `14` iterations |
| scalar nonmonotonic tolerance | `1e-6`; `alpha>1` forbidden |
| hook CPU identity | absolute `1e-10` |
| P0 hook scalar gate | `atol=5e-5`, `rtol=5e-3` |
| functional-vs-committed gate | `atol=5e-5`, `rtol=5e-3` |

QP output coefficient가 actual applied coefficient이며 post-QP rescale은 없다. Signed
Psi/cross term은 diagnostic only이고 hard signed barrier와 model-specific policy/rescue는
금지한다.

## P0/P1 dry plan과 resource forecast

실행하지 않고 paired plan만 검증하는 명령은 다음과 같다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=. /mnt/raid5/janghj/EasyEdit/.venv/bin/python -B \
project/run_scripts/session02_compute_aware_p01.py --stage both --dry-run
```

출력은 두 모델을 동일 batch로 묶고 각 job을 `1 GPU`, `8 CPU`, host memory
`65,000 MiB`로 계획한다. Aggregate `2 GPU <= server1 cap 4`, host memory는
`198,117 MiB/GPU` cap 이하이다. 이 값은 forecast이며 submit 권한이 아니다.

| stage | model당 구조 | 반복 | wall / GPU-hour forecast | peak reserved forecast |
| --- | --- | --- | --- | --- |
| P0 | 1 case, 4 arms | warmup 1 + recorded 3, median | `04:00:00` / `0.5–2.0` | `42–48 GiB` |
| P1 | 4 cases, 4 arms | canonical 1 | `16:00:00` / `3–10` | `42–48 GiB` |

P0 model당 전체 profiler 상한은 `N_z=16`, `N_state_fwd=724`, `N_field=N_bw=36`,
`K_acc=56`, `N_trial=284`, `N_reject=224`, `N_eval=16`, `N_write=60`이다.
P1 model당 상한은 우연히 동일한 숫자다(4 cases × canonical 1 repetition). 이는
accept/reject cap을 서로 독립적으로 더한 보수적 구조 상한이며 실제 outcome/NFE 관측값이
아니다. Static field freeze는 `N_field`를 줄이지만 trial retry 수를 줄인다고 가정하지 않았다.

Full/Native controller GPU-sec ratio의 prelocked gate는 `<=3x` green,
`(3x,4x]` yellow, `>4x`이면 P3 HOLD다. Evaluation GPU time은 이 ratio에서 제외한다.

## 남은 위험과 다음 gate

### 사실

- Model GPU execution, scientific metric, generation, raw artifact는 0건이다.
- Current MEMIT cached graph는 committed dense weight와 no-grad proposal path 때문에
  technical infeasible이며 fail-close한다.
- EasyEdit working tree는 본 작업 전부터 foreign dirty였고 SH1은 변경하지 않았다.
- `TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH`가 남아 있다. 추가 probe는 생성하지
  않았다.

### 추정

- GPU time/peak-memory 범위는 기존 resource record와 구조 cap에 기반한 forecast다.
  Session 02 outcome으로 해석할 수 없다.
- P0 model dtype에서 functional trial과 committed write가 제안 tolerance 안에 들 것으로
  예상하지만 CPU fixture 외에는 아직 관측하지 않았다.

### 실행 전 확인 필요

- GH의 numerical-lock 승인과 별도 GPU/Slurm envelope
- 두 모델 offline cache revision, hparam, covariance/projector/Wikipedia-stat hash/size
  preflight
- fresh context manifest exact match와 actual tokenizer suffix identity
- P0에서 hook/reference scalar gate, functional/committed identity, counter completeness,
  peak memory와 common no-grad backend component table
- 실제 결과가 생성될 때 Terra Ultra runtime을 1회 재확인; mismatch면 interpretation/claim
  promotion HOLD

따라서 현재 상태는 `IMPLEMENTATION_PREP_COMPLETE; NUMERICAL_LOCK_PENDING_GH;
GPU_SLURM_HOLD`다.
