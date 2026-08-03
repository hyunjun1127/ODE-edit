# Session 02 P0 original-dtype and simple-T preparation

Instruction: `ODEEDIT-S02-P0-ORIGINAL-DTYPE-AND-SIMPLE-T-V1`

Parent checkpoint: `8f3fb48d9db14a364a777917e4229aa911277610`

Status: `CPU_DRY_GATES_PASS / GPU_SLURM_RETRY_HOLD`.

This is an outcome-free implementation and dry-plan record. No model was
loaded, no GPU was used, and no scientific evaluation or generation occurred.

## Legacy versus method dtype boundary

The existing `load_fixed_model` API remains the Motivation/MV-0 reproduction
entry point. It still passes `torch_dtype=torch.float32` and now emits
`dtype_policy=legacy-motivation-float32` from observed parameter dtype rather
than hardcoding the metadata value.

The separate method-only entry point is
`load_fixed_model_checkpoint_original`. It passes Transformers `dtype="auto"`
with the same revision, offline, device-map, tokenizer, `use_cache`, and cache
identity contracts. After load it requires:

- the set of all floating parameter dtypes to be exactly
  `{torch.bfloat16}`;
- `model.config.torch_dtype is torch.bfloat16`;
- `dtype_policy=checkpoint-original`;
- observed parameter dtype and checkpoint-original dtype both recorded as
  `torch.bfloat16`.

Forced FP32, mixed parameter dtype, and config/parameter mismatch fail closed
through one alias-independent validator. The method runner imports and calls
only the checkpoint-original entry point; an AST source guard rejects a
regression to the legacy API.

The v4 lock records the GH-confirmed pinned provenance: Llama config BF16 with
291/291 BF16 safetensor tensors and Qwen config BF16 with 339/339 BF16 tensors.
These facts select the dtype independently of memory or method outcomes. The
failed P0 manifests remain legacy-float32 technical provenance and are not
renamed original-dtype evidence.

## Simple-T runtime boundary

Adaptive proposal semantics now resolve through
`QuantizedRowBlockFunctionalTrial` with common `row_block=64`.
`LowRankFunctionalTrial` is documented and retained only as a continuous
reference/diagnostic. The simple-T backend:

- evaluates every adaptive candidate, including rejected candidates;
- remains inside the existing `trial` component timer and increments the
  existing logical `N_trial` once;
- uses the exact QP coefficients without rescaling;
- preserves the existing post-commit event identity hard gate;
- takes no full dense trial weight/delta, target mutation/subtraction, or
  per-trial checkpoint;
- has no alias-specific branch or two-tier pretrial.

Native P0 commits remain on their canonical native path and do not invoke a
finite adaptive trial. Native-terminal scalar probing is outside this P0
simple-T integration and fails closed rather than falling back to the legacy
continuous overlay. No controller, trust, QP, direct-z, accepted-write, or
Qwen dense-solver formula changed.

## Fixed numerical regression

- A/B FP32 continuous tangent fixed stress: `24/24 PASS`.
- T/C BF16 fixed finite grid output: exact `36/36 PASS`.
- T/C BF16 fixed finite grid combined event: exact `36/36 PASS`.
- Coefficient-zero, multi-row-block, pointer/version/`.grad`/requires-grad,
  RNG, exception cleanup, and exact Linear fail-close gates: PASS.
- Method tests: `62/62 PASS`, warnings as errors.
- Loader tests: `6/6 PASS`, warnings as errors.
- Compileall, shell syntax, JSON parse, and `git diff --check`: PASS.

No fixed seed, scale, coefficient, or tolerance was changed.

## Revised outcome-free lock and dry plan

- Lock schema: `ode-edit-compute-aware-numerical-lock-proposal/v4`.
- Proposal ID:
  `c4176176fe54132466d0d75397d647564be8f152396e4c6c1fde63ed09326023`.
- Lock file SHA-256:
  `6fe38820652cecd2bde8bbe2fbf47ac65943d776e915af56a65dc07840381be8`.
- Trial backend: `quantized-rowblock-commit-emulator`.
- Dtype policy: `checkpoint-original`, required observed dtype
  `torch.bfloat16`.
- Submission authority: false; GPU now 0; Slurm now false.

Dry renderer command:

```bash
/mnt/raid5/janghj/EasyEdit/.venv/bin/python -B -m project.run_scripts.session02_compute_aware_p01 --stage p0 --dry-run
```

It emits paired, same-batch jobs with the following absent roots:

- `local/results/session02-p0-original-dtype-simple-t-v1-llama3-8b-inst-c4176176`
- `local/results/session02-p0-original-dtype-simple-t-v1-qwen2.5-7b-inst-c4176176`

The tracked sbatch template remains no-submit preparation and now fails closed
unless a future GH envelope supplies the explicit
`ODEEDIT_P0_SUBMISSION_AUTHORIZED=checkpoint-original-simple-t` token.

## Resource and compute forecast

The dry plan retains the approved envelope shape only as a future request:
one GPU, eight CPUs, 65,000 MiB host memory, and four hours per model; paired
aggregate two GPUs under project cap four. It grants no submission.

Checkpoint-original BF16 GPU time, allocated/reserved memory, and
Full/Native ratio are `UNMEASURED_CHECKPOINT_ORIGINAL_BF16_P0`. The lock keeps
P1 GPU-time forecasting on HOLD until that measurement and keeps the
`Full/Native >4x` HOLD gate. The prior synthetic CPU microfixture is
recorded as simple-T/continuous median `3.855567094593102x` and extra-MAC
estimate `96x`, explicitly not as a model-scale forecast.

The Qwen transient dense-solver fix is preserved. This preparation makes no
causal allocation between forced FP32 and dense temporary assembly for the old
OOM.

## File identity

- `gpu_runtime.py`:
  `157b7ce3d30ce46fe6d238674504aa24402c4234aea8d99a8f97493b493a7745`
- loader tests:
  `93e8a8285e76645bdd6887003618ee270b1c8390a3abd7991814111f31be1b15`
- `functional_trial.py`:
  `b450bdb417fabd8c4476acd813fe1008b0b5c9ad225befc0b932772be579238b`
- `memit_adapter.py`:
  `62306c66166d9715aa99653ab48fb7394f9007b496d9dfe990659633a94195e4`
- `easyedit_backend.py`:
  `b7decd7f27a4de6037c257e753bdc89f9b1416d344b389c4bc7f8702b8310ad7`
- `lock.py`:
  `89dd0e823904e1ab8f87a6783202ffca212f8cfd890a6d6812bfc5f3c617f90a`
- executable runner:
  `7d8ec9506280f9d8bdbb2cf112a6109de56fe1884bee6d59aa255c37b138426d`
- no-submit sbatch template:
  `4db4b073c3e9ddc836b19f20d9b58723b31c1f90e44603ea485a0dc546417083`

These hashes are pre-commit file identities. The local commit is reported by
direct inbox after the final staged access gate.

## Remaining confirmation

FACT: all CPU/mock/dry gates above pass.

INFERENCE: original BF16 plus simple-T should remove the known legacy-FP32
provenance error and make adaptive finite trials emulate BF16 commits, but CPU
gates cannot establish model-scale GPU feasibility.

CONFIRMATION_NEEDED: a separate GH execution envelope must authorize paired
P0, then observe both loaded dtype manifests, model-scale A/B and T/C gates,
time, memory, and Full/Native ratios. Terra runtime mismatch still holds
scientific interpretation and claim promotion.

The old roots/logs and GH-owned local terminal metadata remain unmodified.
`BROADCAST_EXCEPTION_SERVER2_NOT_READY` remains recorded.
