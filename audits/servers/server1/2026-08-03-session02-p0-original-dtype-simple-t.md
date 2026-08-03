# Session 02 P0 original-dtype and simple-T audit

Instruction: `ODEEDIT-S02-P0-ORIGINAL-DTYPE-AND-SIMPLE-T-V1`

Verdict: `CPU_DRY_PASS / GPU_RETRY_HOLD`.

| Gate | Result |
|---|---|
| Legacy Motivation loader remains forced FP32 | PASS |
| Legacy metadata policy explicit and observed | PASS |
| Method loader passes `dtype="auto"` only | PASS |
| Original floating parameter set exactly BF16 | mock PASS; actual GPU pending |
| Config BF16 identity | mock PASS; actual GPU pending |
| Forced FP32, mixed dtype, config mismatch | fail-close PASS |
| Runner legacy-loader source regression | AST guard PASS |
| Adaptive finite trial backend | simple quantized T, row block 64 |
| Rejected candidates use T | PASS by single backend path |
| Two-tier pretrial | disabled and lock-rejected |
| Existing post-commit event identity gate | retained |
| LowRank continuous overlay in runtime primary | absent |
| Full dense trial weight/delta or mutation | absent |
| Alias-specific dtype/controller/trial rescue | absent |
| Fixed A/B | PASS, 24/24 |
| Fixed T/C output/event | exact PASS, 36/36 each |
| Method regression | PASS, 62/62 warnings as errors |
| Loader regression | PASS, 6/6 warnings as errors |
| New output roots | both absent |
| Dry plan submission authority | false |
| GPU/model load/Slurm/retry/push | absent |

The lock records pinned BF16 config/header evidence supplied by GH, not a
memory-outcome-dependent dtype choice. Old float32 P0 artifacts remain labeled
legacy technical provenance. No EasyEdit source was modified and no existing
failure artifact was edited, deleted, or reused.

The only unresolved technical promotion gate is actual paired GPU validation:
loaded dtype, A/B, T/C, memory, timing, and terminal manifests. The CPU
simple-T microfixture is preserved as a sizing warning, not extrapolated to a
model-scale ratio. `TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH` and
`BROADCAST_EXCEPTION_SERVER2_NOT_READY` remain open.
