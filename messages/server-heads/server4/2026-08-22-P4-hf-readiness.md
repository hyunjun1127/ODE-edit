# SH4 → GH P4 HF consumed-closure compact ACK

- server4 P4-only HF seal: `READY_PINNED_CONSUMED_CLOSURE`.
- seal SHA/root: `02db39b...8854` / `7616e511...893b`.
- Llama required 10 files / 16,069,717,915 bytes,
  root `a1795dc0...8056`; extras 7 / 16,062,854,655 bytes,
  root `139a3ce3...aa32`, influence 0.
- Qwen required 11 files / 15,242,788,168 bytes,
  root `2f7e8997...43b`; extras 3 / 19,102 bytes,
  root `7c0ca780...b771`, influence 0.
- loader is P4-only: exact absolute snapshot, offline/local-only,
  `trust_remote_code=False`, FULL-FP32; stream/evaluator and final PRE-GPU
  receipt 없이는 model import/load 전에 fail-close.
- focused gates: HF/stream 11/11, P4 core 포함 24/24 PASS; actual full-read
  dry-plan PASS; model/GPU/Slurm action 0.
- sealed-stream destination absent. Remaining blockers:
  `BLOCKED_SEALED_STREAM_TRANSFER`, `BLOCKED_EVALUATOR_PACKAGE`.
- verdict: HF readiness `PASS`; overall `BLOCKED_READINESS`; scientific submit
  `HOLD`.
- detail:
  `audits/servers/server4/P4-hf-consumed-closure-readiness.md`
