# Saved-episode FD repair (technical, not a new policy)

Authority: `ODEEDIT-S06-EP-TW1-SAVED-EPISODE-FD-REPAIR-SH4-V1`.
`policy.py`, `ledger.py`, native fitter/map/target and teacher bytes are unchanged.

`repair_runtime` loads the exact saved failed B001 Vp/A/target evidence and pinned
W0 for all other parameters. It never imports a fitter or solver. It saves each
complete E/D gradient before the next stage, compares direct selected-weight leaf
gradients with residual VJPs, and runs both fixed directions at all ten scales.
E FD must pass before D FD. `.15` derivative/convergence and `1e-7` absolute bounds
are unchanged. Two adjacent resolved smaller scales are required, with all coarse
failures retained. No extrapolation or score-selected independent direction.

Old post-fit RNG/original gradient tensors are missing: this is not continuation
or old-gradient byte replay. The new diagnostic RNG is sealed, seed2026091503.
An exception inside an unfinished gradient sweep preserves prior completed
stages and failure location, not an invented complete partial gradient.

CPU reproduction (no model load):

```sh
PYTHONDONTWRITEBYTECODE=1 /data/janghj/EasyEdit/.venv/bin/python -m unittest discover -s project/run_scripts/bg_tw_reference/ep_tw -t . -p 'test_*.py'
```

Preparation commands are create-once; do not rerun against sealed output:

```sh
python -m project.run_scripts.bg_tw_reference.ep_tw.repair_control preflight --worktree <clean-worktree> --attempt <repair-root>
python -m project.run_scripts.bg_tw_reference.ep_tw.repair_control freeze --worktree <committed-worktree> --attempt <repair-root>
python -m project.run_scripts.bg_tw_reference.ep_tw.operations --worktree <worktree> --attempt <repair-root>
```

The one-GPU conditional `repair.sbatch` starts science only after successful
technical process exit AND an exact source/science-lock-bound technical PASS.
Science is one fresh-W0/coldM0 EP-TW-1 B100x10 chain, not B2 resume. New B1
Vp/A/gE/gD must be exact for endpoint-check reuse; otherwise the same bounded
endpoint checks run once before first commit. Diagnostic cost is separate from
method gradient cost. Old473GPU-sec (including nativefit286.5445s) and teacher98s
are reused historical costs, not charged twice.

Full tensor/prompt/probe output stays local. After actual scientific G0, PENDING
handoff, or terminal technical HOLD/FAIL, agents stop and await explicit user
recall. A technical PASS alone is not scientific G0. Existing output is never
overwritten; no automatic retries, callbacks, baseline reruns or teacher rebuild.
