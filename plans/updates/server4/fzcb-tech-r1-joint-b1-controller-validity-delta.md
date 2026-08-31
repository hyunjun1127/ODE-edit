# FzCB TECH-R1 joint B1 controller-validity delta mapping

Authority:

- proposal SHA256 `47666dada9a95be2a4d4414dcb826990b94150c8dacea08a763ed063049a3853`
- TECH-R1 contract SHA256 `4356a2cfea04687f25a2972e3a3f4636d57d4140d76dd2db2913b5d74d0467ac`
- preserved implementation lineage `fd6ac88f8278ea4135769965b0d8cde9435cf151`
- preserved hold/report lineage `e8fccfb77fe825aae0f5fa5d66769956077d8e66`

| Contract delta | Existing equation-identical asset | TECH-R1 responsibility |
|---|---|---|
| six tolerance units | matrix-free CG/KKT and FP32 lock | `tolerances.py` seals unit-specific policy and resolves action-unit tolerances before arm decisions |
| full equality-null authority | `FullModelControlOperator.apply/adjoint`, equality-null projection | `sensitivity.py` provides exact projected-gradient primitive and, because stock detached key capture blocks exact production HVP, the mandatory nested 2/8/32/128 FD sketch ladder with dimension correction/CI |
| strict CBF/budget validity | analytic scalar rectifier, suffix KKT | `verifier.py` enforces proposal `psi_min`, every-waypoint `h_cc`, closure, action, current/next viability and rollback |
| failure-safe partial results | create-once JSON and exact-copy snapshots | `journal.py` atomically seals each arm or typed failure without replacing prior arms |
| true frozen negative control | initial MEMIT `T/H` capture | `comparators.py` freezes entry `T0,H0,J_Phi0,C0` and checks `E+V=A0`, `h=0`; refreshed current-model C is forbidden |
| strong static comparator | same native basis/equality/corrector | `comparators.py` uses one endpoint direct-transcription SQP problem, not Euler/backtracking |
| predictive vs realized suffix | refreshed local suffix solve | `rollout.py` separates `predicted_suffix_after` from an actual remaining equality-only rollout and computes concordance/regret observation-only |
| Llama/Qwen joint campaign | existing model/artifact/stream/padding guards | one array launcher maps task 0/1 to Llama/Qwen under one sealed campaign manifest; one cell cannot cancel or erase the other |

The prior F2 q-KL/reference-fact/functional-anchor controller and K0 promotion logic remain forbidden.  No target, stock MEMIT hparam, layer set, request, context, `kappa`, evaluator, stream, or FULL-FP32 rule changes.

