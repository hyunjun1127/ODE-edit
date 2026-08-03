# Original-dtype simple-T P0 pair synthesis

Pair verdict: `P0_TECHNICAL_BLOCK / ZERO_SCIENTIFIC_OUTCOME / P1_HOLD`.

The exact approved pair was submitted together from
`dfafe486cb5307c3afe5cd8c1ce8b5fe6ed33def`. Both jobs received one GPU and
started simultaneously within the four-GPU project cap. Llama job `16061` and
Qwen job `16062` independently fail-closed before action with the same locked
fresh-context manifest mismatch.

Original-dtype loader guards ran before the failure, so the code path reached
context preparation only after common BF16 policy/config/parameter validation.
No persisted manifest exists, however, and this attempt therefore does not
provide a terminal loaded-dtype manifest, A/B or T/C result, direct-z,
controller row, GPU peak memory, component timing, or Full/Native ratio.

Both roots contain only empty controller/compute/evaluation JSONL files. Empty
files are not outcomes. No evaluation, generation, write or scientific claim
was produced. The identical gate boundary across models is consistent with a
common context-lock provenance issue, but exact causality requires a separate
outcome-free RCA; no repair or retry is authorized here.

Raw roots and logs remain local and unchanged. Broadcast is
`BROADCAST_EXCEPTION_SERVER2_NOT_READY`. Analysis is
`ANALYSIS_PENDING/TERRA_RUNTIME_MISMATCH`. Retry and P1 remain HOLD pending a
new GH envelope.
