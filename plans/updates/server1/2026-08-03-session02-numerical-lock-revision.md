# Session 02 numerical-lock revision — server1/SH1

## 판정

`ODEEDIT-S02-NUMLOCK-REVISION-V1`의 implementation-prep 범위를 완료했다. Concrete
EasyEdit-backed backend, 공정 compute accounting, executable P0 runner, submit 기능이 없는
sbatch template과 outcome-free numerical-lock v3를 local commit
`e6fa39548224e92244e7d02118a1375e47b2431e`에 보존했다. Scientific outcome은 0건이며
GPU/model load, Slurm submit, push, merge는 수행하지 않았다.

현재 판정은
`REVISION_PREP_COMPLETE; NUMERICAL_LOCK_PENDING_GH; GPU_SLURM_PUSH_HOLD`다. 이 문서는
P0 실행 권한을 부여하지 않는다.

작성 기준 시각은 `2026-08-03 17:37:38 KST`다. Canonical SH1 session은
`019fc63e-5217-7250-9c22-c5b2ec4248f0`, server/role은 `server1`/`server-head`, worktree는
`/mnt/raid5/janghj/.codex/worktrees/29e4/ODE-edit`, branch는
`codex/odeeditsh1`이다.

## Canonical lineage와 작업 경계

- protocol SHA-256:
  `5c2221f221886d34f670935274d1d0ceb34436ee813a97d6c5a909325171ad37`
- fair-accounting checkpoint:
  `5e4886747f630595f10ced166d3dc7b86234c282`
- GH canonical object
  `f8a613bd2e8b739dc6972576e298733fbbe13151`의 conflict-free local cherry-pick:
  `345d1e467c417a2ace6fc7282ec42a7a9171f544`
- implementation revision:
  `e6fa39548224e92244e7d02118a1375e47b2431e`
- canonical spec SHA-256:
  `8ee7f5557f298bb67c1e85ed6de1aaa6a285b7b8d28ecd4dcbe1973bdf85531d`
- GH가 canonical SH1 registry를 main `6d1a2da`에서 정정했다는 direct broadcast를
  수신했다. Registry/`servers/active/server1.md`는 GH 소유이므로 이 branch에서 수정하거나
  cherry-pick하지 않았다.
- EasyEdit checkout은 기존 foreign dirty state를 유지한다. SH1은 EasyEdit 파일을 수정,
  stage, commit, stash, reset하지 않았고 artifact download/recompute도 수행하지 않았다.
- 별도 direct-z temporary track의 code, artifact, report를 열거나 결합하지 않았다.

## 구현 결과

### Concrete common backend

`EasyEditMemitBackend` 하나가 Llama3-8B-Instruct와 Qwen2.5-7B-Instruct를 동일 코드
경로로 처리한다.

- EasyEdit public bridge와 pinned MEMIT primitive를 read-only로 호출한다.
- Entry event가 miss일 때 direct-z를 정확히 한 번 계산하고 해당 outer edit 동안 tensor와
  artifact identity를 고정한다. Entry hit이면 first-hit contract에 따라 `N_z=0`이다.
- Native/Scalar는 canonical ordered terminal proposal을 공유한다.
- Static/One-refresh/Full/Ordered는 accepted current snapshot의 모든 layer proposal을
  synchronous하게 만들며, Full은 accept 뒤 전 layer를 다시 선형화한다.
- Proposal snapshot은 exact target-weight hash chain에 결박되고, trial 사이의 unchanged-state
  검사는 storage pointer/version/shape/dtype/device로 수행한다. Scalar probe도 dense entry
  hash를 반복하지 않는다.
- Precomputed covariance는 setup에서 full hash/size 검증 후 한 번 로드한다. Adaptive refresh는
  backend-owned read-only tensor cache를 사용하고 pointer/version guard를 검사하므로 동일한
  multi-GiB file을 field마다 다시 decompress하지 않는다. Run 중에는 file stat과 bounded
  first/middle/last byte sample을 검사하고 terminal cleanup에서 전체 fixed-artifact hash를
  다시 검증한다.
- Native public bridge의 내부 covariance preflight는 global lock 안에서 이미 검증된 contract로
  좁게 대체하고 `finally`에서 원래 ODE-side bridge function을 복원한다. EasyEdit source 자체는
  변경하지 않는다.

### Event, derivative와 information firewall

- Rewrite old/new teacher-forced score를 하나의 right-padded combined batch forward로 계산하며
  각 object token length로 별도 normalize한다.
- Controller input은 rewrite-only typed `ControllerRequest`다. Paraphrase, neighborhood,
  held-out evaluation payload는 action freeze 전에 표현할 수 없다.
- Primary derivative backend는 `ActuatorDirectionalHook`이다. 한 rewrite backward에서 module
  input `x`와 grad-output `g`를 capture하고 각 low-rank direction의
  `g^T U(V^T x)` contraction을 계산한다.
- Target parameter별 backward와 target-weight dense gradient materialization은 0이다.
  Target-weight `torch.autograd.grad`는 CPU reference oracle로만 남는다.
- P0 warmup의 Full arm은 hook 결과를 scalar finite difference와 잠긴 tolerance로 비교한다.

### Trial, transaction, rollback과 Omega

- Common scientific backend는
  `no-grad functional trial -> exact accepted commit -> dedicated committed-state event/rebuild`다.
- Current MEMIT `cached_trial_graph`는 `UNSUPPORTED_FAIL_CLOSED`이며 비교 arm이나
  model-specific 선택 후보가 아니다.
- Functional trial은 dense target-weight copy 없이 low-rank activation hook으로 실행한다.
  QP coefficient를 그대로 적용하며 post-QP rescale은 없다.
- Edit-entry checkpoint는 outer transaction당 한 번만 만든다. Accepted step별 full CPU weight
  backup은 0이며, injected mid-commit failure는 entry weight, Torch CPU/CUDA RNG, Python RNG,
  NumPy RNG와 backend state를 exact restore한다.
- Reject는 unchanged field/QP input을 재사용하고 trust radius만 줄인다. Reject field rebuild는
  0이다.
- Omega는 completed outer edit의 terminal net C-energy를 edit당 한 번 append한다. Micro-step
  energy sum은 사용하지 않으며 failure restore 전에 append하지 않는다.

## 공정 compute accounting

다음 counter를 서로 중복 없이 기록한다.

`N_z`, `N_model_fwd`, `N_event_fwd`, `N_field_state_fwd`, `N_field`,
`N_proposal_build`, `N_native_sweep`, `N_bw`, `K_acc`, `N_trial`, `N_reject`,
`N_eval`, `N_write`.

`N_trial`은 logical attempt이고 forward proxy가 아니다. Top-level model pre-hook가 evaluation을
제외한 실제 `N_model_fwd`를 세며 event/field scope만 해당 subset counter로 분류한다.

Component wall/GPU timer는 setup, entry checkpoint, restore, terminal geometry, direct-z,
Native proposal, synchronous proposal, trust scale, event, field state forward, hook backward,
QP, trial, commit과 evaluation을 분리한다. `controller_wall_seconds`와 synchronized
`controller_gpu_seconds`는 entry checkpoint 직전부터 terminal commit 또는 failure cleanup까지
동일 범위다. Context generation, artifact preflight, arm-isolation baseline과 backend initial exact
snapshot capture는 setup으로 별도 기록하고 amortized sec/edit에만 더한다. Model load는 제외해
별도 wall time으로 기록한다.

P0 구조 상한은 model당 warmup 1 + recorded 3 전체에 대해 다음과 같다. 이는 outcome/NFE
관측값이 아니다.

| Counter | P0 profile 상한 | P1 상한 |
| --- | ---: | ---: |
| `N_z` | 16 | 16 |
| `N_event_fwd` | 300 | 300 |
| `N_field_state_fwd` | 257 | 252 |
| `N_field` | 36 | 36 |
| `N_proposal_build` | 40 | 40 |
| `N_native_sweep` | 4 | 4 |
| `N_bw` | 36 | 36 |
| `K_acc` | 60 | 60 |
| `N_trial` | 224 | 224 |
| `N_reject` | 224 | 224 |
| `N_eval` | 0 | 16 |
| `N_write` | 60 | 60 |

`N_model_fwd`는 위 logical counter에서 환산하지 않고 P0에서 actual top-level calls를 직접
측정한다.

## Read-only feasibility preflight

GPU/model load 없이 fixed source/artifact와 selected row를 read-only로 확인했다.

| Model | verified files | fixed-artifact manifest ID | elapsed |
| --- | ---: | --- | ---: |
| Llama3-8B-Instruct | 25 | `273d06af367904aac61931331e6aa2ea09204b0db1f6c7557bca16a9e2c84d2b` | 18.99 s |
| Qwen2.5-7B-Instruct | 25 | `d247b27edf638b7a9625954e8bf2aeac745b61247230f8bcf733e991a3ed9d16` | 33.13 s |

Case `2022,12498,20964,768`와 locked request IDs가 일치했고 rewrite-only projection을
확인했다. 이 preflight는 tokenizer/context와 model-scale numerical identity를 증명하지
않으며, 실제 P0 action 전에 loaded-runtime preflight가 다시 fail-close해야 한다.

## Revised outcome-free numerical lock

제안 파일은
`project/run_scripts/ode_edit_method/numerical_lock_proposal.json`이다.

- file SHA-256:
  `c0d74f6d614ea0b644488f4eaa9cbfacc1551def3b4f2516c6877fa0f3f8c710`
- canonical proposal ID:
  `7d15789d5952f281166db8579db1cd7397d3dbb419b447178eeb0c7976d06a95`
- exact case order: `2022, 12498, 20964, 768`
- order SHA-256:
  `e7746f38ba36f8d36815b4e58084ad408be6cb64f77d04661c6bc4ca688e08f7`
- request-order SHA-256:
  `bffe1b3655ae73223f9901f8e5d31af86779740a2682db142a0bf1415a66c57c`
- common seed: `17`

| 항목 | 제안값 |
| --- | --- |
| hard event | `max_context(mean_ll_old - mean_ll_new)` |
| first-hit / hard-worsening tolerance | `1e-6` / `1e-6` |
| smooth surrogate / `tau` | mean-log-sum-exp deficit / `0.1` |
| trust scale | raw entry synchronous joint `D_sync_entry` before unit-C normalization |
| `h0`, `hmax` | `0.25*D_sync_entry`, `0.50*D_sync_entry` |
| reject/expand ratio | `0.1`, `0.75` |
| radius contraction/expansion | `0.5`, `1.5` |
| `S_max`, One-refresh accepted cap | `6`, `2` |
| max rejects/state | `4` |
| slope/progress epsilon | `1e-10`, `1e-10` |
| QP equality/trust tolerance | `1e-8`, `1e-8` |
| denominator epsilon | `1e-12` |
| QP dtype / dual iterations | `float64`, `128` |
| Scalar grid | `0, 1/16, ..., 1`; `alpha>1` forbidden |
| Scalar bisection | width `2^-14`, max 14 iterations |
| hook CPU oracle tolerance | absolute `1e-10` |
| P0 hook finite-difference gate | `atol=5e-5`, `rtol=5e-3`, epsilon `1e-3` |
| functional/committed identity | `atol=5e-5`, `rtol=5e-3` |
| terminal geometry trigger | wall 또는 GPU controller time의 `>10%`이면 P1 전 low-rank cross-term evaluator 필요 |
| Full/Native gate | `<=3x` green, `(3x,4x]` yellow, `>4x` P3 HOLD |

P0에서 `h0/D_native`가 두 모델 중 하나라도 `[1/8,1/2]` 밖이면 P1 lock을 HOLD한다.
Model-specific radius, threshold, sign 또는 fallback rescue는 없다.

## Executable P0 path와 dry plan

Tracked runner에는 Slurm submit 기능이 없으며 `--execute`를 명시해야 한다. 향후 별도 GH
GPU envelope가 승인한 뒤 사용할 exact commands는 다음 형태다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/mnt/raid5/janghj/EasyEdit \
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B \
  project/run_scripts/session02_compute_aware_p0.py \
  --model-alias llama3-8b-inst \
  --output-root local/results/session02-p0-llama3-8b-inst --execute

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:/mnt/raid5/janghj/EasyEdit \
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B \
  project/run_scripts/session02_compute_aware_p0.py \
  --model-alias qwen2.5-7b-inst \
  --output-root local/results/session02-p0-qwen2.5-7b-inst --execute
```

현재 허용된 dry renderer 검증 명령은 다음이다.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B \
  -m project.run_scripts.session02_compute_aware_p01 \
  --stage p0 --dry-run
```

Dry output은 paired two-model batch, model당 `1 GPU`, `8 CPU`, `65,000 MiB`, `04:00:00`,
aggregate `2 GPU <= project cap 4`, `submission_authorized=false`를 확인했다. P0 forecast는
model당 `0.5–2 GPUh`, peak reserved `42–48 GiB`이며 아직 관측된 resource fact가 아니다.

Output은 `manifest.json`, `controller_steps.jsonl`, `compute.jsonl`, 빈
`evaluation.jsonl`, `summary.json`, `terminal_manifest.json`을 사용한다. Model/case/arm/order/
seed/commit/hash/status, 모든 counter/timer/memory, event, terminal Omega와 direct-z artifact
hash를 기록하고 raw prompt/target/context는 저장하지 않는다.

## CPU gate

다음 suite가 46/46 PASS했다.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B \
  -m unittest discover -s project/run_scripts/ode_edit_method/tests -v
```

추가 PASS:

- Python compileall
- `bash -n project/run_scripts/session02_compute_aware_p0.sbatch`
- strict JSON parse와 lock/spec hash validation
- paired P0 dry-plan schema, aggregate GPU와 submit=false assertion
- `bash scripts/check-agent-access.sh --staged`
- `git diff --cached --check`
- tracked payload/submit/download/recompute static scan

## 남은 위험과 다음 gate

### 사실

- Actual model GPU execution, generation, scientific evaluation/output은 0건이다.
- EasyEdit checkout에는 completion 시점 179개의 pre-existing foreign dirty status entry가
  있다. 본 작업은 그 파일을 변경하지 않았다.
- `TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH`가 남아 있다. GH 지시에 따라 추가 probe나
  Sol runtime 대체 agent를 만들지 않았다.
- Canonical spec의 `N_z=1 per outer edit` 문구와 Native 포함 post-entry-first-hit work 0
  문구 사이에 긴장이 있다. Revised envelope의 Native entry-hit regression 요구를 우선해
  proposal은 entry hit `N_z=0`, 그 외 `N_z=1`로 fail-close한다. GH lock 승인 시 명시적
  확인이 필요하다.

### P0에서 확인할 기술 gate

- fixed offline model/tokenizer revision과 fresh context manifest/suffix identity
- combined event와 two-forward reference identity
- actual model에서 one-backward hook/finite-difference identity와 target `.grad is None`
- functional trial/committed event numerical identity
- scoped Native covariance-preflight replacement의 복원과 precomputed tensor cache integrity
- counter completeness, synchronized component time와 peak memory
- `h0/D_native`, Full/Native wall/GPU ratio와 terminal geometry 10% trigger
- model-scale injected failure cleanup 및 terminal manifest completeness

하나라도 실패하면 P0 technical HOLD이며 model-specific rescue 없이 GH에 보고한다.
