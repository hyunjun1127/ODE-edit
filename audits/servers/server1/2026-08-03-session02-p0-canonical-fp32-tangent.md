# Session 02 canonical FP32 tangent audit

Instruction: `ODEEDIT-S02-P0-CANONICAL-FP32-TANGENT-V1`

Verdict: `PASS_A_B / B_C_DIAGNOSTIC_MISMATCH / RETRY_HOLD`.

| Gate | Result |
|---|---|
| Fixed A/B grid and tolerance unchanged | PASS, 24/24 |
| Alpha-zero module and combined-event identity | PASS, exact 12/12 |
| Dense float target-weight oracle on CPU | PASS |
| Independent scalar-forward versus captured-gradient paths | PASS |
| Sign/orientation/captured-input negative controls | PASS |
| Target dense `.grad` materialization | zero |
| Pointer/version/requires-grad/RNG cleanup | PASS |
| Model-alias branch | absent |
| Controller/QP/direct-z/write-scale changes | absent |
| Fixed B/C finite diagnostic | MISMATCH, output 0/36 and event 34/36 |
| Tolerance, seed, scale, or coefficient adaptation | absent |
| GPU/model load/Slurm/retry/push | absent |

The A/B redesign closes the continuous-tangent identity but does not make a
continuous output overlay equal to a finite parameter-dtype commit. That
distinction is explicit and is not hidden by tolerance relaxation. The
follow-on T prototype owns finite T/C identity; this audit does not promote a
GPU retry or a scientific result.

The local P0 failure metadata is excluded from the SH index and commit. Its
path, schema/required field inventory, and SHA-256 are recorded in the paired
plan update without reproducing its raw payload.

`TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH` and
`BROADCAST_EXCEPTION_SERVER2_NOT_READY` remain open.
