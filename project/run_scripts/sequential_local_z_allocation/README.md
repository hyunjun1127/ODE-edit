# Sequential Local-z Allocation v2

Fresh W0, seed20260916, fixed CounterFact first1000/B100×10. Six arms only:
N4, F48, G48, C4, C48, C45678. The Slurm array prioritizes C45678 at index0.
Native BLUE target/solve/finalizer and the canonical evaluator are read-only imports.
The old repair runtime is not called.

`controller.py` preserves the published CPU reference equations and adds durable
events and a selection seal. `runtime.py` applies each physical L4–L8 gate to
the whole weight, then fits later layers at that actual prefix. Each arm appends
all five histories once per committed B100, including zero-gated layers. Official
P/N and Dev are observed only after the next selected state is sealed.

SciPy1.15.3 legacy COBYLA is required. `maxiter=64` limits function evaluations;
`tol=.01` is a trust-region lower bound, not a loss tolerance or convergence
certificate. [Official versioned documentation](https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-cobyla.html).

## Preparation and operation

1. `preparation --worktree PATH`: create-once authority/identity receipt.
2. `control check --worktree PATH`: CPU tests/import/source checks; no model.
3. Commit scoped source, then `control freeze --worktree PATH`.
4. `operations technical`: cap2 resource admission, held inspection and release.
5. Only after actual `TECHNICAL_READY`, `operations science`: all six jobs upfront.

Module prefix: `python -m project.run_scripts.sequential_local_z_allocation.`
The launcher binds exact source/lock/dependencies with exportNONE, one GPU,
8CPUs,60416MiB and Requeue0. There are no callbacks or automatic submissions.
The original PENDING handoff interpretation was superseded by the explicit
2026-09-17 main-gate recall. Technical PENDING or PASS is not a stopping point.
Use `resume_control` for that recall: finish required actual technical checks,
register/inspect/release all six mains, then observe scientific B1
selection/history5/B2 entry. Only verified GPU-resource shortage of properly
released mains, with no main running, allows a PENDING handoff. Generic
Resources/None/dependency/hold is not sufficient evidence. Detailed completion
review still needs a later recall. The original frozen source and old pause
receipt remain unchanged.

## Validation and storage limits

CPU fixtures are not actual Llama validation. The bounded common technical job
checks native connections, teacher/repeated metrics, state/zero/cache/history
and C45678 completed non-N4 gate coverage≥7, distinct a4≥2. Insufficient coverage
does not generate READY. No FD/repair/QP or quality-improvement gate is added.

Snapshots share immutable CPU weight buffers and precommit history buffers in
RAM. Finalization clones histories before writing, so entry rollback remains
available. No full-model or W/M checkpoint is written to disk. Native target/key
evidence, exact loss/Adam/clamp counters, scalar/hash/selection/commit ledgers and
all required evaluation rows remain local. Exact crash-resume or retrospective
selected-weight reconstruction is **NOT_AVAILABLE**. Previously deleted cold7
checkpoints are not recreated. Raw tensors/prompts/logs never enter Git.
