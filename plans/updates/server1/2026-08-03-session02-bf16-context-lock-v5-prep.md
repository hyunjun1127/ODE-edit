# Session 02 original-BF16 context lock v5 prep

상태: `OUTCOME_FREE_V5_PREP_PASS / PENDING_GH_APPROVAL / P0_RETRY_HOLD`.

## Identity

- instruction: `ODEEDIT-S02-BF16-CONTEXT-LOCK-V5-PREP-V1`
- parent: `ODEEDIT-S02-BF16-CONTEXT-LOCK-PROBE-PAIR-V1`
- base main: `26a8ffe9b7077e9b9707c5b92f5704c5fb6386e7`
- schema: `ode-edit-compute-aware-numerical-lock-proposal/v5`
- status: `OUTCOME_FREE_PROPOSAL_PENDING_GH_APPROVAL`
- outcome count: 0
- proposal ID: `c8ce4b31b76731b0d9875d8fa5a1e5abb5775a5822fb578ca989317ea7f11f01`
- lock SHA-256: `b2eeb74ca21c47a864b48b8e80b76de7778d5f6098a1cd5c5e86b9e28ba0f8b5`

## Changed lock fields

V4 대비 변경은 schema/instruction/parent/revision/base provenance와 두 모델의
`context_manifest`에만 한정했다. Deep-diff gate에서 method, controller,
tolerance, case/order, resource, trial, solver 및 execution-root template가 모두
변경되지 않았음을 확인했다.

| Model | V5 manifest ID | Template SHA-256 | Legacy provenance ID |
| --- | --- | --- | --- |
| Llama | `22c26dc11fb13acd51d5bdc483e4b9dd46fa40029fe10b167bff1a1642f7e686` | `0a2069beafc60e170251103028fde716a160a60a8048bb000a649cf26c233bb0` | `3020b3f5cea62e6cfbd173f0c99a4348cecf7e84425bb087720ced7f395482e5` |
| Qwen | `5b7144416638fb3deec1f12f204a401e06edd1293aa2fe4a22f9080f3e8bd41b` | `ff84e360b7a4a413275da818862c7c4431ac21c7fbeaf26e4d2296278f9a4bf9` | `e0c5f61d874334a2cab26f82fb3594d9ab88c9e5816be390274a0bb5e98c93bd` |

각 context에는 source, `[1,5]`, `checkpoint-original`, repeat 2,
`exact_match=true`, probe execution head, terminal/raw-manifest SHA-256와 legacy
provenance `method_evidence=false`를 기록했다. Raw templates, raw path,
credential은 lock에 넣지 않았다.

## Validator and tests

Validator는 alias별 observed identity를 data mapping으로 보유하고 하나의 common
loop에서 전체 context mapping을 exact 비교한다. 따라서 extra raw field도
fail-close한다.

- focused lock tests: `10/10 PASS`.
- full method regression: `72/72 PASS`, warnings as errors.
- old v4, 두 legacy ID, repeat 1, exact false, wrong dtype policy/hash,
  legacy method evidence true: 모두 fail-close.
- `py_compile`, JSON parse, dry plan, diff check: PASS.

## Dry plan

P0/P1 proposal ID는 모두 `c8ce4b31...1f01`, aggregate GPUs 2,
`submission_authorized=false`다. P0 new roots는 현재 absent다.

- `local/results/session02-p0-original-dtype-simple-t-v1-llama3-8b-inst-c8ce4b31`
- `local/results/session02-p0-original-dtype-simple-t-v1-qwen2.5-7b-inst-c8ce4b31`

Tracked boundary는 `gpu_now=0`, `slurm_now=false`, submission false를 유지한다.
이번 prep에서는 GPU/model load/Slurm/retry/push를 수행하지 않았다. V5 승인과
P0 retry는 별도 GH envelope가 필요하다.
