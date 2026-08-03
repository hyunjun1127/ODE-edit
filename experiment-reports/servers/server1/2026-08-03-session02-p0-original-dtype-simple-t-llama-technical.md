# Llama original-dtype simple-T P0 technical report

Verdict: `FAIL_CLOSED_BEFORE_ACTION / ZERO_SCIENTIFIC_OUTCOME`.

- job: `16061` (`odeedit_s02_p0_orig_llama`)
- execution head: `dfafe486cb5307c3afe5cd8c1ce8b5fe6ed33def`
- submitted/started: 2026-08-03 19:07:55 KST
- terminal: `FAILED`, exit `1:0`, 41 seconds
- resources: 1 GPU, 8 CPU, 65000 MiB host-memory request
- Slurm batch MaxRSS: 279308 KiB (host RSS, not GPU peak)
- output root:
  `local/results/session02-p0-original-dtype-simple-t-v1-llama3-8b-inst-c4176176`
- logs:
  `local/logs/session02-p0-odeedit_s02_p0_orig_llama-16061.{out,err}`

The checkpoint-original loader and its BF16 runtime guards returned before
context preparation. The run then stopped at the fresh-context manifest gate
with `fresh context manifest differs before action`. Because the manifest is
written after this gate, loaded dtype fields were not persisted; their passage
is control-flow evidence, not a terminal manifest.

No arm or direct-z started. The three JSONL files are empty, each with SHA-256
`e3b0c442...b855`; root file-list digest is `4d8f5520...483d`. `manifest.json`,
`summary.json` and `terminal_manifest.json` are absent. A/B, T/C, GPU memory,
component time and Full/Native ratio are unavailable, not zero.

stderr SHA-256 is
`59e52ed7656b303ce33c6973926528a8afab0d186c5489334558edbd0788ea4f`.
Retry is not authorized. Analysis is `ANALYSIS_PENDING/TERRA_RUNTIME_MISMATCH`.
Raw broadcast is `BROADCAST_EXCEPTION_SERVER2_NOT_READY`.
