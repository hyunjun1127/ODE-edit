# Session 02 P0 technical pair synthesis

Pair verdict: `P0_TECHNICAL_BLOCK / ZERO_SCIENTIFIC_OUTCOME / P1_MAIN_TABLE_HOLD`.

Both exact jobs were submitted together at 17:53:28 KST from execution commit
`f0db6743bed05be4c7073dbfe4d2d0ec11ea0961`, within the four-GPU project cap.
Both passed offline model/artifact/tokenization and combined-event preflight,
then failed in warm-up before terminal accounting:

- Llama job 16025: Full rep-0 one-sided finite-difference hook validator mismatch.
- Qwen job 16026: Static rep-0 dense-system CUDA OOM.

The failures are separate technical blockers. No evaluation/generation ran,
no terminal compute rows exist, and no arm completed a recorded repetition.
Therefore Full/Native compute ratios, retention, `h0/D_native`, terminal
geometry, and peak-memory comparisons are all unavailable—not zero.

Remediation commit `02da467680de31f0196f7d90cad634599e662c1b` contains
outcome-free CPU fixes and passed the pre-review 55-test suite. A later fixed
BF16 nonlinear/rank-2 red stress failed A/B derivative identity in 12 of 24
comparisons, so no amend commit was created. Qwen memory code is structurally
CPU-ready but GPU-unconfirmed; Llama requires arithmetic redesign. Neither
retry nor P1 is authorized.

Compact raw identities are in
`runs/session02-p0-tech-v1/terminal-metadata.json`. Raw outputs/logs remain
local and ignored. Broadcast is `BROADCAST_EXCEPTION_SERVER2_NOT_READY`.
Terra analysis is `TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH`; no
scientific claim can be promoted.
