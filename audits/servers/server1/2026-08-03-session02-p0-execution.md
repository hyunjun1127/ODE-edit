# Session 02 P0 execution audit

## Boundary and submission

- Canonical SH1 worktree/branch/session and effective server-head identity: PASS.
- Clean exact execution head `f0db6743bed05be4c7073dbfe4d2d0ec11ea0961`: PASS.
- Runner/sbatch/lock hashes: PASS before submission.
- Offline cache, existing covariance artifacts, output-root absence, and parent writability: PASS.
- Aggregate request: two GPUs with zero existing ODE-Edit jobs, `0 + 2 <= 4`: PASS.
- Exact pair submitted once at 17:53:28 KST: jobs 16025 and 16026.
- No source mutation, retry, requeue, parameter change, output cleanup, evaluation, or generation occurred during execution.

Both jobs reached model load, manifest creation, event-batching identity, and
first-arm entry. Qwen then hit a CUDA OOM; Llama continued independently and
then hit the hook-validator contract failure. Each failure was reported to GH
immediately and neither job was resubmitted.

## Artifact and repository controls

- Raw roots and logs are preserved at their authorized ignored paths.
- Empty `compute.jsonl`, `controller_steps.jsonl`, and `evaluation.jsonl` are explicitly non-metrics.
- Tracked Git contains only compact hashes/status, never raw tensor/log payloads.
- EasyEdit was read-only; its pre-existing foreign dirty state remains 179 entries and was not staged or edited by SH1.
- No covariance/null/Wikipedia recompute, redownload, or network fallback occurred.
- No direct-z temporary track or unrelated repository/session was accessed.

Ordinary raw broadcast could not be verified because server2 onboarding and
rsync verification remain HOLD. Status:
`BROADCAST_EXCEPTION_SERVER2_NOT_READY`.

Post-terminal Terra runtime was not verifiably `gpt-5.6-terra/ultra`; bounded
agents produced no repository analysis. Status:
`TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH`.

Final audit verdict: execution procedure PASS, technical pair FAIL, scientific
outcome count zero, P1 HOLD.
