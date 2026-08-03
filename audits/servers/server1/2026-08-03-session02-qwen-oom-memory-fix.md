# Session 02 Qwen OOM memory-fix audit

Audit verdict: `PASS_CPU_STRUCTURE / GPU_CONFIRMATION_NEEDED`.

## Required invariants

| Invariant | Evidence | Verdict |
|---|---|---|
| Same dense system | CPU reference identity at float64 and float32; mocked solve receives exact `lambda*C + K*K^T` | PASS |
| Same solver | production call remains `torch.linalg.solve(system, keys)` | PASS |
| At most one explicit square construction | one source-to-device `copy=True`; `mul_` and `addmm_`; source static test rejects `keys @ keys` expression | PASS |
| Source immutability | pointer/version/dtype/device/shape guard; focused mutation test | PASS |
| No layer GPU cache | `_solve_covariance_by_layer` removed; solver has slots and retains no system | PASS |
| No file reopen/recompute | verified in-memory source is returned directly; file path logic unchanged | PASS |
| Common models/policy | no alias/controller branch added | PASS |
| EasyEdit immutability | no EasyEdit path changed or staged | PASS |

The structural allocation estimate does not include CUDA allocator
fragmentation or `torch.linalg.solve` factorization workspace. Therefore this
audit does not claim the A6000 retry will fit. The exact model-scale peak is a
`CONFIRMATION_NEEDED` item under a future one-shot GPU envelope.

Tests: 55/55 method regressions PASS with `-W error`; access checker and
`git diff --check` PASS. Local checkpoint:
`02da467680de31f0196f7d90cad634599e662c1b`. Push and retry were not performed.

Artifact broadcast is `BROADCAST_EXCEPTION_SERVER2_NOT_READY`; the incomplete
Qwen raw root remains ignored, preserved, and non-metric.
