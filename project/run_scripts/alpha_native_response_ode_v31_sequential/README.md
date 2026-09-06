# AlphaEdit JV sequential routing

Instruction: ODEEDIT-S06-ALPHA-JV-SEQUENTIAL-ROUTING-SH2-V1.
Source parent: 979b4606b82d5095f89503ea052d586e8c097f38.

This module reuses the stock Official entrypoint, v3.1 NativeDictionary,
FrozenNormalization, NNLS and grouped FP32 overlay. The new Euler adapter
records the required sequential observers and has an explicit L8-only support;
it does not dispatch L8 through the old pilot's ray branch. N0/.1/T2/N4/h.5,
full layer/P mapping and existing numerical tolerances remain unchanged.

Each chain initializes cold Alpha state once. Each batch creates a new family,
computes its own fixed-z, binds the current entry M, and creates a new dictionary.
The existing endpoint finalizer appends full-inventory post keys once, captures
the actual W/M, then restores entry. The existing commit copies those captured
bytes without compute-z, writer, forward, evaluator or history recomputation.
The next batch checks exact W and M content identities. Finite poor outcomes
continue without semantic filtering. Checkpoints contain real selected weights
and dense M; immediate reload is verified, not claimed as incremental replay.

Smoke: two development requests in two separate batches per model, distinct from
the sealed main1000. Four O/JV chains take priority over two L8-only chains.
All output is create-once and local. W0 full-neighborhood references are measured
once per model after smoke W/M restoration. Seen rewrite is evaluated every
batch; full seen RS/PS/NS only at 1/5/10, reusing current-batch rows at the exact
same W state. Online diagonal and final-W10 retention are separate artifacts.

Only CPU support, coordinate/metric, evaluation binding, checkpoint and commit
checks are newly required. Model smoke verifies virtual/materialized parity and
actual two-batch state propagation. All-node/history shadows are observer-only;
no lambda or normalization sweeps. Initial-history Gram uses the actual L2
metric at the actual fixed basis/qref, not identity substituted for cold M.

Run `python -m unittest -q project.run_scripts.alpha_native_response_ode_v31_sequential.test_sequential`.
Seal a clean source using `python -m ...provenance --repo REPO --root ROOT --official SOURCE`.
The launcher requires held inspection and a fresh resource recount. Server2 cap3,
one GPU/process, memory60416MiB, eight CPUs. No main merge authorization.
