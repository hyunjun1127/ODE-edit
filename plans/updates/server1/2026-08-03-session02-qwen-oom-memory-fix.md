# Session 02 Qwen OOM memory-only fix

Instruction: `ODEEDIT-S02-P0-QWEN-OOM-MEMFIX-V1`

Checkpoint: `02da467680de31f0196f7d90cad634599e662c1b`

Status: `CPU_GATES_PASS / GPU_RETRY_HOLD`.

## RCA

FACT: job 16026 failed in Static synchronous proposal construction at
`covariance_weight * covariance + keys @ keys.T`; the next 2.67 GiB allocation
failed with only 1.93 GiB free.

FACT: the execution implementation cached a float32 GPU covariance for every
visited layer, then converted the current matrix to float64 and used an
expression that could simultaneously materialize multiple float64 square
matrices.

INFERENCE: the Qwen down-projection dimension is 18,944. This is consistent
with each verified covariance file size and the exact 2.67 GiB allocator
request. One square matrix is 1.3369140625 GiB in float32 or 2.673828125 GiB in
float64. Five retained float32 layer copies total 6.6845703125 GiB. Removing
four already-completed layer copies alone removes 5.34765625 GiB at layer 8;
the in-place assembly also removes explicit float64 product/add temporaries.

The exact factorization/workspace peak inside `torch.linalg.solve` remains
implementation-dependent and was not measured on GPU.

## Memory-only change

`TransientDenseMemitSolver` now receives the guarded CPU covariance source,
creates one owned tensor directly at the keys' device/dtype, then executes:

```text
system.mul_(lambda)
system.addmm_(keys, keys.T)
torch.linalg.solve(system, keys)
```

The equation, float64 production solve dtype, coefficients, factor orientation,
and native solver are unchanged. No Woodbury, iterative solve, dtype downcast,
layer change, allocator flush, or model-specific branch was introduced. The
per-layer GPU covariance dictionary was removed; no dense system survives the
solver call. The verified source pointer/version guard and read-only file
provenance remain active, with no file reopen/decompress path added.

Changed implementation hashes:

- `dense_memit.py`: `a7b10a49ce6072b8eb8716019ee9834418054cf989a1b7976b185ea68421ef11`
- `easyedit_backend.py`: `ee3cfadfa1ce315752f001104e0770bfa9925c31d8b1dd9af849e2c8ccbf0896`
- focused test: `test_dense_memit.py` SHA-256 `76f015d934964e45f409f8c667d3bdac90dc073c1f0baa7bcfed442e27ed14e4`

## CPU evidence

- float64 reference versus transient solver: max absolute/relative error `0/0`
- float32 reference versus transient solver: max absolute/relative error `0/0`
- source pointer/version unchanged: PASS
- one explicit dense construction and no retained solver cache: PASS
- same `torch.linalg.solve(system, keys)`: static/runtime mock gate PASS
- all method tests: 55/55 PASS, warnings as errors
- staged access, Bash syntax, compile, and diff checks: PASS

These are synthetic CPU implementation checks, not a model result.

## Retry proposal only

The existing output root is immutable and collides, so the old command must
not be reused. A future GH envelope must supply a new absent output root before
authorizing a command of the following form:

```bash
sbatch --job-name=odeedit_s02_p0_qwen --export=ALL,MODEL_ALIAS=qwen2.5-7b-inst,OUTPUT_ROOT=<GH_AUTHORIZED_NEW_OUTPUT_ROOT> project/run_scripts/session02_compute_aware_p0.sbatch
```

No retry was submitted. GPU/model load, Slurm, push, P1, and scientific claims
remain HOLD. A GPU retry must still measure factorization workspace, peak
allocated/reserved memory, A/B hook identity, B/C commit identity, and a
terminal manifest.

Raw failure root/hash is preserved in
`runs/session02-p0-tech-v1/terminal-metadata.json`. Broadcast status is
`BROADCAST_EXCEPTION_SERVER2_NOT_READY`.
