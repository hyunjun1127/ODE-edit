# BG-1 / C4 preparation and mathematical core

This is a partial implementation for `ODEEDIT-S06-BG1-C4-OURS-FIRST-SH4-V1`,
not a validated end-to-end scientific runner. Only BG-1 W0 first1000 is approved.
No seven-policy dispatch, baseline editing, warm-entry fallback, or automatic
post-gate monitoring is implemented.

- `acquire` retrieves the two pinned C4 gzip files, pinned README and PSL snapshot.
- `builder` creates the exact text/token reference768 without loading a model.
- `preparation overlap` emits input-only overlap material. Future stream and
  uninspected evaluator inventories are explicitly not claimed overlap-free.
- `preparation freeze` seals a W0 teacher192 source/asset lock. It requires the
  pinned Transformers 4.44.2 environment used by `preparation.sbatch`.
- `teacher` generates S64+Dev128 full-vocabulary FP32 teacher shards only. It has
  no editor, optimizer, history, scientific submission, or continuation callback.
- `native_map` and `correction` implement/test the frozen native RHS map,
  full-D64 barrier gradient weighting, anchor/trust projection and four-candidate
  materialization/selection. Toy CPU FD/VJP is not Llama model parity.
- `gate` rejects incomplete B1..B10 saved-N4 calibration. Preparation readiness
  cannot become G0_PASS without actual persistent scientific firstB100 evidence.

As of this preparation, both retained N4-compatible L4 trajectories have only
B1/B5/B10 checkpoints and no reconstructible missing increments. Calibration is
`CALIBRATION_MISSING`; a partial maximum, zero threshold or native editing rerun
cannot fill it. BG model adapter/persistent controller, real-model gradient
checks, candidate transactions/history/resume validation remain incomplete.

CPU fixtures (no model load):

```bash
PYTHONPATH=/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/deps-transformers-4.44.2:. \
 /data/janghj/EasyEdit/.venv/bin/python -m unittest discover \
 -s project/run_scripts/bg_tw_reference -t . -v
```

All acquired text, token arrays, teacher tensors, model assets and logs are
local-only. Compact source/checksum/preparation/G0 records are publishable.
`G0_BLOCKED` or `G0_PASS` ends agent work until an explicit user recall; a queued
preparation job is not a teacher-valid or scientific-valid result.
