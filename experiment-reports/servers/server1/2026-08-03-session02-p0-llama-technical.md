# Session 02 P0 Llama technical report

Verdict: `P0_TECHNICAL_BLOCK`, `ZERO_SCIENTIFIC_OUTCOME`, `P1_HOLD`.

## Execution identity

- Instruction: `ODEEDIT-S02-P0-TECH-PAIR-V1`
- Execution commit: `f0db6743bed05be4c7073dbfe4d2d0ec11ea0961`
- Job: `16025`, `odeedit_s02_p0_llama`, one RTX A6000
- Start/end: 2026-08-03 17:53:28–17:59:07 KST; `FAILED 1:0`, 339 seconds
- Model/tokenizer revision: `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`
- Observed model dtype: `torch.float32`; the failed run was not BF16/FP16

## Technical gates

| Gate | Status | Compact evidence |
|---|---|---|
| Offline fixed model/artifact preflight | PASS | fixed artifact manifest `273d06af...`, source manifest `48b07333...` |
| Tokenization/context manifest | PASS | context manifest `3020b3f5...`; new/old object token count 1 |
| Combined event batching | PASS | max abs new `1.239776611328125e-05`, old `9.834766387939453e-06`; within `5e-5/0.005` |
| First arm entry | PASS | Native warm-up direct-z created; `N_z` terminal accounting unavailable |
| Hook derivative validator | FAIL | Full warm-up rep-0 raised `MethodContractError: actuator hook differs from P0 scalar finite difference` |
| Actual hook/FD mismatch values | UNAVAILABLE | stderr retained only the exception text; no per-layer numbers were persisted |
| Functional/commit, terminal geometry, counters/timers | UNAVAILABLE | failure occurred before a terminal record |

The warm-up created Native, Static, One-refresh, and Full direct-z files, but
`compute.jsonl`, `controller_steps.jsonl`, and `evaluation.jsonl` are empty and
the manifest remains `RUNNING_TECHNICAL_ONLY`. These partial files are not
metrics and are not interpreted as arm outcomes.

## Preserved artifact identity

- Root: `local/results/session02-p0-tech-v1-llama3-8b-inst-7d15789d`
- Manifest SHA-256: `e8270184b6bd0cc3f9630b4c8b22498fefc27ffb7f376092b27c5d627bac89e4`
- stdout/stderr SHA-256: `f26b4a56...` / `cf5a9a91...`
- Direct-z hashes are recorded in `runs/session02-p0-tech-v1/terminal-metadata.json`; payloads remain ignored and unmodified.

No Full/Native wall or GPU ratio, `h0/D_native`, terminal-geometry fraction,
peak GPU GiB, or terminal counters exist. Host `MaxRSS` was 9,781,488 KiB and
must not be confused with GPU peak memory.

## Follow-up state

CPU-only remediation commit `02da467680de31f0196f7d90cad634599e662c1b`
replaces the one-sided FD hard oracle with an independent scalar-gate graph
validator. A subsequent fixed non-dyadic BF16 red stress found A/B derivative
arithmetic mismatch: 12 of 24 layer comparisons failed the unchanged
`5e-5/5e-3` gate, despite exact alpha-zero combined-event identity. Therefore
the checkpoint is not retry-ready; canonical continuous-field arithmetic
requires GH redesign and Llama retry remains unauthorized.

Analysis status is `TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH`; no Sol
substitution was used. Raw broadcast is
`BROADCAST_EXCEPTION_SERVER2_NOT_READY` because peer onboarding/rsync
verification is not ready.
