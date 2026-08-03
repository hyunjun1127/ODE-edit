# Session 02 original-dtype simple-T P0 실행 감사

판정: `FAIL_CLOSED_BEFORE_ACTION / RAW_PRESERVED / SCIENTIFIC_OUTCOME_0`.

## Pre-submit gate

| Gate | Verdict | Evidence |
| --- | --- | --- |
| session/repo/role | PASS | canonical SH1, Sol Ultra, dedicated worktree |
| execution identity | PASS | `dfafe486cb5307c3afe5cd8c1ce8b5fe6ed33def` |
| proposal/lock | PASS | `c4176176...6023`; lock `6fe38820...be8` |
| runner/sbatch | PASS | SHA-256 `7d8ec950...426d`; `4db4b073...7083` |
| fixed artifacts | PASS | Llama manifest `273d06af...4d2b`; Qwen `d247b27e...9d16` |
| offline cache | PASS | pinned revision, config/tokenizer, four weight shards/model |
| output collision | PASS | both new roots absent before submit |
| Slurm cap | PASS | active/pending 0, requested 2, cap 4 |
| memory cap | PASS | 130000 MiB requested, 396234 MiB pair cap |
| EasyEdit mutation | PASS | before/after status digest `b36feeea...15d`; SH diff 0 |

## Terminal evidence

Both jobs failed with the exact exception
`MethodContractError("fresh context manifest differs before action")` at
`project/run_scripts/ode_edit_method/preflight.py:299`.

| Model | Job | State | Host MaxRSS | stderr SHA-256 |
| --- | --- | --- | --- | --- |
| Llama | `16061` | `FAILED 1:0`, 41 s | 279308 KiB | `59e52ed7656b303ce33c6973926528a8afab0d186c5489334558edbd0788ea4f` |
| Qwen | `16062` | `FAILED 1:0`, 63 s | 1318948 KiB | `a1976c4f7101241937a9f64a497b89ce668c29b5e42497181d28763d08ebab06` |

Each output root contains exactly the three expected JSONL names, all empty
(SHA-256 `e3b0c442...b855`). The compact sorted file-list digest for each root is
`4d8f5520eac5b456049daef9fd1d578254a4314e8f9d36257ae588d0ef29483d`.
This digest is local provenance, not a terminal-manifest substitute.

The code path establishes that checkpoint-original loader validation completed
before context preparation: policy `checkpoint-original`, checkpoint dtype
`torch.bfloat16`, observed floating-parameter dtype `torch.bfloat16`, and BF16
config validation did not fail. Because `manifest.json` is written only after
context preparation, no persisted loaded-runtime manifest exists. Exact GPU
peak memory, controller time, A/B, T/C, direct-z and Full/Native ratio are
therefore unavailable.

## Red-team closure

- No action, arm, write, evaluation, generation or scientific result occurred.
- Empty JSONL files are not metrics.
- No retry, cancellation, cleanup, source change, tolerance change or model
  rescue occurred.
- Previous P0 artifacts and GH-owned untracked terminal metadata remain
  unchanged; the latter SHA remains `dd3890b0...fd89`.
- Raw rsync was not attempted because server2 remains onboarding HOLD:
  `BROADCAST_EXCEPTION_SERVER2_NOT_READY`.
- No terminal scientific artifact was eligible for analysis; Terra remains
  `ANALYSIS_PENDING/TERRA_RUNTIME_MISMATCH`.
