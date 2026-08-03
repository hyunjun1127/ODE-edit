# Session 02 Llama hook RCA audit

Audit verdict: `FAIL_BF16_A_B_STRESS / REDESIGN_REQUIRED`.

Supersession note: this is the pre-redesign audit. The subsequent canonical
FP32 tangent closes fixed A/B at 24/24; the separate finite T/C prototype is
audited in `2026-08-03-session02-p0-quantized-rowblock-trial-prototype.md`.
GPU retry remains HOLD.

## Red-team checks

| Check | Result |
|---|---|
| Missing mismatch values not inferred | PASS; retained logs contain only the exception string |
| Independent oracle, not tautology | PASS; B differentiates float32 alphas through a functional output graph, while A contracts separately captured `x/g` |
| Dense target gradient materialization | zero in primary/scalar paths; `.grad is None` assertions PASS |
| Parameter/RNG/state mutation | pointers, versions, requires-grad, and exception cleanup PASS |
| Negative control | injected primary sign error is rejected by B reference |
| Fixed FD weakened/skipped | removed as hard oracle rather than tolerance-relaxed; metadata marks it diagnostic-disabled |
| A/B and B/C separated | scalar gate checks A/B; existing accepted nontrivial functional/commit gate checks B/C under its own lock keys |
| Model-specific rescue | absent |
| Controller/QP/direct-z/evaluation changes | absent |
| Reference cost accounting | dedicated forward/backward counters and component timer added |
| Non-dyadic BF16 nonlinear/rank-2 stress | FAIL: 12/24 A/B comparisons exceed unchanged lock; combined event identity remains exact |

The actual run used float32. The BF16 stress confirms that A and B are not a
generally equivalent lower-precision arithmetic path; it does not prove that
this was job 16025's cause. No model-scale rerun or scientific outcome was
produced.

The pre-review method suite was 55/55 PASS with warnings as errors. The added
fixed stress test fails by design evidence (12/24 comparisons); it was not
committed and no tolerance was changed. Local implementation commit remains
`02da467680de31f0196f7d90cad634599e662c1b`; no amend and no push.

`TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH` remains because the required
runtime could not be verified and no Sol substitution was accepted. Artifact
broadcast is `BROADCAST_EXCEPTION_SERVER2_NOT_READY`.
