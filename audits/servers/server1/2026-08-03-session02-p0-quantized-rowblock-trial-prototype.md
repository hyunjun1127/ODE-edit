# Session 02 quantized row-block finite-trial audit

Instruction: `ODEEDIT-S02-P0-QUANTIZED-ROWBLOCK-TRIAL-PROTOTYPE-V1`

Verdict: `FIXED_BF16_PROTOTYPE_PASS / PRODUCTION_INTEGRATION_HOLD`.

| Gate | Result |
|---|---|
| Prelocked T/C output grid | PASS, exact 36/36 |
| Prelocked T/C combined-event grid | PASS, exact 36/36 |
| Fixed A/B regression | PASS, 24/24 |
| Coefficient-zero identity on required BF16 grid | PASS, exact 36/36 |
| Target pointer/version/`.grad`/requires-grad | PASS |
| CPU RNG and injected exception cleanup | PASS |
| Full dense trial weight/delta | absent; row-block source and shape guard PASS |
| Temporary target-weight mutation/subtraction | absent |
| Exact Linear/weight/bias contract | fail-close |
| Model alias branch | absent |
| Qwen solver/controller/QP/direct-z/commit writer | unchanged |
| Tolerance/grid adaptation | absent |
| Full CPU regression | PASS, 60/60 with warnings as errors |
| GPU/model load/Slurm/retry/push | absent |

The prototype is not a tautological T/C comparison: T evaluates detached,
read-only effective row blocks through `F.linear`; C invokes the separate
existing in-place accepted writer and then the normal module forward. Parameter
storage and versions are checked before C is applied.

The required BF16 fixture passes, but retained P0 manifests state
`torch.float32`. A separate 130-output CPU coefficient-zero diagnostic showed
that row-block GEMM tiling can differ bitwise from a full FP32 Linear. This does
not invalidate the fixed BF16 prototype result; it prevents production/retry
promotion until GH resolves the execution dtype and model-scale zero-identity
gate.

No `memit_adapter.py` or runtime wiring was made. GH still owns the simple-T
versus two-tier integration decision. `TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH`
and `BROADCAST_EXCEPTION_SERVER2_NOT_READY` remain open.
