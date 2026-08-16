# ODE-edit experiment reports

This directory preserves the raw-free reports, aggregate tables, analysis scripts, manifests, and receipts available from the experiment worktrees.

- Raw result roots, model weights, caches, prompts, generations, and private target tensors are intentionally excluded.
- `source-handoff/` contains raw-free reports received from the other execution host when the original report worktree was not locally accessible.
- `BRANCH_INDEX.tsv` records the branch names and exact tips before branch consolidation into `main`.
- Exact historical source is preserved through the merge parents reachable from `main`; use the recorded commit SHA to inspect a specific experiment revision.

