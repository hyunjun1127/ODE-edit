# Job47962 CPU-only review

Authority: ODEEDIT-S06-EP-TW1-47962-COMPLETED-DETAILED-REVIEW-SH4-V1,
including the design-conformance addendum. These tools never instantiate a model
or invoke a native writer/evaluator. `state_audit.py` and `mechanism.py` use CPU
`torch.load(weights_only=True, mmap=True)` sequentially with four CPU threads.
Runtime, policy, native source, old results and teacher remain read-only.

The actual runtime is commit `6d317bdb2660d7e9919bc3a9fb878564e9729e37`.
The review's source commit is recorded in the generated comparison capsule;
it is not the execution commit. Diagnostic validation remains user-directed
SKIPPED and numerical validation NOT_ESTABLISHED, including after CPU checks.

Run from a clean analysis worktree. Every `--output` must be a new path; tools
refuse overwrite. Replace the following task-specific variables explicitly:

```sh
review_worktree=/data/janghj/ODE-edit/local/worktrees/server4-ep-tw1-47962-completed-review-v1
review_python=/data/janghj/EasyEdit/.venv/bin/python
review_scratch=/data/janghj/ODE-edit/local/ep-tw1-c4/20260915-v1/gate-skip-r1/completed-review-v1
review_code=project/run_scripts/bg_tw_reference/ep_tw
cd "$review_worktree"
PYTHONDONTWRITEBYTECODE=1 "$review_python" -m unittest project.run_scripts.bg_tw_reference.ep_tw.test_review_nogate -v
PYTHONDONTWRITEBYTECODE=1 "$review_python" "$review_code/review_nogate.py" --worktree "$review_worktree" --output "$review_scratch/metrics-new"
PYTHONDONTWRITEBYTECODE=1 "$review_python" "$review_code/review_nogate/state_audit.py" --output "$review_scratch/state-new"
PYTHONDONTWRITEBYTECODE=1 "$review_python" "$review_code/review_nogate/mechanism.py" --worktree "$review_worktree" --output "$review_scratch/mechanism-new"
PYTHONDONTWRITEBYTECODE=1 "$review_python" "$review_code/review_nogate/baselines.py" --worktree "$review_worktree" --output "$review_scratch/baselines-new"
PYTHONDONTWRITEBYTECODE=1 "$review_python" "$review_code/review_nogate/evidence.py" --worktree "$review_worktree" --output "$review_scratch/evidence-new"
```

The publication builder intentionally binds the reviewed local directories
`metrics-v3`, `state-v1`, `baselines-v1`, `mechanism-v1`, `evidence-v1` under
`review_scratch`. It copies aggregate/checksum products, not raw data. The first
metric attempt stopped on an analysis-only timer scalar/dict mismatch; that
partial output remains local. The final reducer handles the recorded float
without inventing a separately timed FB component. `metrics-v2` was provisional;
`metrics-v3` excludes peak memory from additive cost totals. Later independent
reproduction must use new paths and compare rather than replace sealed files.

```sh
review_package=experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/gate-skip-r1/completed-review-v1
PYTHONDONTWRITEBYTECODE=1 "$review_python" "$review_code/review_nogate/build_publication.py" --worktree "$review_worktree" --output "$review_package"
PYTHONDONTWRITEBYTECODE=1 "$review_python" "$review_code/plot_nogate_review.py" --package "$review_package" --output "$review_package/figures"
PYTHONDONTWRITEBYTECODE=1 "$review_python" "$review_code/plot_nogate_review.py" --package "$review_package" --output "$review_scratch/figures-reproduction-new"
```

PNG generation uses CSV inputs only and fixed Matplotlib/Agg metadata. Record
input/code/PNG SHA in `plot-manifest.json`; compare output SHA from the two runs.
No imagegen, image-editing tool or manual numerical adjustment is used.

## Aggregation contract

- RS/PS: new sequence-mean NLL < true; NS: true < new; ties fail.
- Identity: case ID + prompt index + sealed prompt/target identity hash.
- TF strict and token correctness are independent secondary fields; two-P
  strict is request-level, never replaced by the NLL-based pair success count.
- Current own-batch pooling is not actual-W10 retention. Full/prefix/current
  reuse rows are not summed into a larger denominator.
- Paired loss and recovery retain original and conditional denominators.
- Quantiles use linear interpolation at `(n-1)*q`. NLL distributions are
  descriptive prompt-level summaries, not independent PS/NS request replicates.
- Candidate selection is independently reduced from recorded E, strict IDs,
  D and frozen numerical tie constants. No model policy is re-executed.
- CPU reconstruction from saved Vp/C/A/W is tensor algebra, not model/GPU parity.
- Baseline W10 JSONs are sealed local references with identity pairing but
  different seed/config/trajectory. W0 is published aggregate-only reuse.
- Cost counters and nested timers are not blindly summed; map-A solve is
  source-counted separately from native RHS solve.

No broadcasting is required: all private raw remains on server4. Only this code,
raw-free CSV/report/PNG and hashes are published. Package manifest excludes itself
and rooted receipt to avoid self-hash cycles. Git publication HEAD/tree live in
the final handoff, not falsely embedded as their own commit hash.
