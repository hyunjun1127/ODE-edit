# BF16 context-lock paired calibration synthesis

Pair verdict: `PASS_PASS / REPRODUCIBLE_ORIGINAL_BF16_CONTEXTS / ZERO_SCIENTIFIC_OUTCOME`.

Llama job `16063` and Qwen job `16064` were submitted and started together from
`37a713233b95614d2743a12abc4d1036daed95f0`. Both used the same
checkpoint-original loader, seed 17, `fresh=True`, two-repeat schema and pinned
EasyEdit source. Both repeats were byte-exact within each model.

- Llama candidate ID: `22c26dc11fb13acd51d5bdc483e4b9dd46fa40029fe10b167bff1a1642f7e686`
- Qwen candidate ID: `5b7144416638fb3deec1f12f204a401e06edd1293aa2fe4a22f9080f3e8bd41b`

Both differ from the respective legacy-FP32 lock IDs while preserving the same
source labels, seed and `[1,5]` group sizes. This establishes reproducible
original-BF16 context provenance for the current pinned environment; it does
not constitute edit, retention, compute-frontier or scientific superiority
evidence.

No edit request, direct-z, covariance, dataset or evaluation was accessed.
Raw templates remain only in ignored local manifests, and raw-boundary plus
terminal-hash checks passed. Broadcast is
`BROADCAST_EXCEPTION_SERVER2_NOT_READY`. Numerical lock update and P0 retry
remain subject to a separate GH envelope.
