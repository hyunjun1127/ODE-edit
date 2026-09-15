# Server4 completed CAKE/cap review

Review-only instruction ODEEDIT-S06-SERVER4-COMPLETED-CAKE-CAP-DETAILED-REVIEW-SH4-V1.
No model load, GPU forward/evaluator/replay, Slurm mutation or science-file edits.
`begin` performs the single explicitly authorized exact-four-job scheduler snapshot;
do not rerun it during reproduction. Reuse its receipt. Existing pause is preserved.

CPU reproduction (each output must be a new path):

```bash
python -B -m project.run_scripts.server4_completed_review.cake first --output NEW_FIRST
python -B -m project.run_scripts.server4_completed_review.cake full --output NEW_CAKE
python -B -m project.run_scripts.bg_tw_reference.ep_tw.sweep_review arm --arm CAP10 --allocated 7559 --worktree "$PWD" --output NEW_CAP10
python -B -m project.run_scripts.bg_tw_reference.ep_tw.sweep_review arm --arm CAP100 --allocated 7735 --worktree "$PWD" --output NEW_CAP100
python -B -m project.run_scripts.bg_tw_reference.ep_tw.sweep_review arm --arm NORM_ONLY --allocated 7639 --worktree "$PWD" --output NEW_NORM_ONLY
python -B -m project.run_scripts.bg_tw_reference.ep_tw.sweep_review combine --worktree "$PWD" --analysis-root THREE_ARMS --output NEW_SWEEP_PACKAGE
```

These exact allocated seconds come from scheduler receipts, not estimates. Tensor
reloads are `weights_only`, `map_location=cpu`. CAKE has no W/M tensor checkpoint by
user choice; its hashes and chronology do not establish recoverability. Cap sweep
has actual 30 CPs, separately verified. FD/direct/selfKL checks remain user-skipped.

Analysis repairs only: CAKE current schema uses weight_state/cache_sha256 rather
than state; observer influence is integer0. Runtime lost-ID arrays sort by string,
while semantics require the exact set, not numeric sorting. Failed CPU analysis
namespaces are preserved; no scientific runtime or numerical tolerance was repaired.

Previously unpublished sweep runtime files are copied exactly from owned commit
a7bfd2f (execution subset8c64366), for publication, never executed by this review.
Only analysis modules receive new modifications. Historical reports and GH table
fixes are unchanged. Source/cost inventories distinguish prior reuse from this audit.

Final CSV/PNG/Markdown/seal validation (no scheduler or scientific raw reload):

```bash
python -B -m project.run_scripts.server4_completed_review.validate_publication --output NEW_VALIDATION
```

This runs focused CPU tests and regenerates nine PNGs into new output directories,
then compares exact bytes against the publication. Markdown tables are checked for
column consistency, heading spacing and local link existence. Source/extension and
aggregate-content review is distinct from model-level numerical validation.
