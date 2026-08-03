# Session 02 P0 Qwen technical report

Verdict: `P0_TECHNICAL_BLOCK`, `ZERO_SCIENTIFIC_OUTCOME`, `P1_HOLD`.

## Execution identity

- Instruction: `ODEEDIT-S02-P0-TECH-PAIR-V1`
- Execution commit: `f0db6743bed05be4c7073dbfe4d2d0ec11ea0961`
- Job: `16026`, `odeedit_s02_p0_qwen`, one RTX A6000
- Start/end: 2026-08-03 17:53:28–17:58:17 KST; `FAILED 1:0`, 289 seconds
- Model/tokenizer revision: `a09a35458c702b33eeacc393d103063234e8bc28`
- Observed model dtype: `torch.float32`

## Technical gates

| Gate | Status | Compact evidence |
|---|---|---|
| Offline fixed model/artifact preflight | PASS | fixed artifact manifest `d247b27e...`, source manifest `48b07333...` |
| Tokenization/context manifest | PASS | context manifest `e0c5f61d...`; new/old object token count 1 |
| Combined event batching | PASS | max abs new `2.6702880859375e-05`, old `2.5153160095214844e-05`; within `5e-5/0.005` |
| First arm entry | PASS | Native warm-up direct-z created; `N_z` terminal accounting unavailable |
| Static synchronous field | FAIL | CUDA OOM while assembling `lambda*C + K*K^T` at layer 8 |
| Remaining Full/commit/terminal gates | UNAVAILABLE | process exited before terminal records |

The allocator requested another 2.67 GiB with 1.93 GiB free on a 47.40 GiB
GPU; PyTorch reported 43.04 GiB allocated and 2.07 GiB reserved but
unallocated. Native and Static warm-up direct-z files exist, but all three
JSONL files are empty and the manifest remains `RUNNING_TECHNICAL_ONLY`.
These partial files are not metrics.

## Preserved artifact identity

- Root: `local/results/session02-p0-tech-v1-qwen2.5-7b-inst-7d15789d`
- Manifest SHA-256: `3a6e763ec51222819c690b117c18e350d5e92d1d7b9f204f7bca3db500e1abbd`
- stdout/stderr SHA-256: `dac84d4f...` / `424af466...`
- Direct-z hashes are recorded in `runs/session02-p0-tech-v1/terminal-metadata.json`; payloads remain ignored and unmodified.

No Full/Native ratio, `h0/D_native`, terminal geometry, complete counters, or
terminal GPU peak exists. Host `MaxRSS` was 13,919,060 KiB and is not GPU
memory.

## Follow-up state

CPU-only remediation commit `02da467680de31f0196f7d90cad634599e662c1b`
removes the layer-accumulating GPU covariance cache and constructs the same
dense native system through one transient allocation plus in-place
`mul_`/`addmm_`, followed by the same `torch.linalg.solve`. This has not been
validated with a model/GPU; Qwen retry remains unauthorized.

Analysis status is `TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH`. Raw
broadcast is `BROADCAST_EXCEPTION_SERVER2_NOT_READY` because peer
onboarding/rsync verification is not ready.
