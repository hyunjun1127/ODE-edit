# GH → SH1: Alpha JV D/S independent implementation and execution envelope

instruction_id: ODEEDIT-S06-ALPHA-JV-LLAMA-DIAGNOSIS-SWEEP-SH1-V1
nonce: ODEEDIT-SH1-DIAG-SWEEP-20260907-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-f93a-7b50-bca0-65438eab2062
expected_cwd: /mnt/raid5/janghj/ODE-edit
repository: hyunjun1127/ODE-edit
assigned_server: server1 / devbox

## Authority and exact contract

User explicitly assigns this new implementation/experiment to SH1. Prior
analysis-first / SWEEP_HOLD does not prohibit this new D/S task. Follow the full
471-line execution contract below, not this dispatch summary alone:

`experiment-reports/global/alpha-jv-llama-diagnosis-and-sweep-2026-09-07-v1/gh-execution-prompt-ko.md`
SHA256: 98ee832e597b1f79c79ca48c62838af8f49f5306c08b3480b990d494f42f15f5
bytes: 34790; lines: 471.

This clean dispatch worktree is /mnt/raid5/janghj/.codex/worktrees/odeedit-gh-alpha-jv-diag-dispatch-20260907-v1.
The plan/design directories were supplied by the user in the dirty GH root and
copied byte-exact, without changing their design content. Main-at-read is
e25a5685a4ad7f8606a8c265abf070b0d77a83d1; local dirty GH HEAD is
ddc178584ef14efd5d4e1271b3c324e3ebd3e443, not the runtime base.
Read every named source/reference fully as needed before acting; record actual
hashes/HEAD-tree. Source runtime 77358b1546d1baf83b3e251afcce663b08d7bfd7 and
publication 0d0a0131e4a6a2a645dfa6530377d420a084d136 are different identities.

Precedence: latest user task and this prospective resource override > detailed
execution contract > detailed design > broad research plan > old candidate
priority proposals. Broad plan functional-history H0/H1 is NOT authorized here.
Five lambda values are ACTUAL for both models, including .03162277660168379 and
.31622776601683794; do not downgrade them to CPU-only. S does not await D.

## Ownership and allowed writes

SH1 is accountable integrator. Assign independent bounded D and S owners, plus a
single shared fixture/config/trajectory integration owner (SH1 may own this).
D owns inventory/replay/capture/requestwise/interventions; S owns sample,
immutable grid, candidate execution and sweep tables. Workers are not alone and
must not revert one another; immutable data may be shared, mutable Family/global
cache/overlay may not. Workers report to SH1 and do not push or self-approve.

Allowed new source: `project/run_scripts/alpha_jv_llama_diagnosis/**`.
Place new runner/Slurm templates/tests/plot builder inside that package; do not
inherit the old sequential 1000 launcher or change its hardcoded globals.
Reuse old native/JVP/NNLS/state primitives by import. If a primitive really
needs edits outside this path, send an exact scope/RCA request first.
Do not modify live/source-locked runtime files to enable this task.

Allowed durable reporting/control:
- `experiment-reports/servers/server1/alpha-jv-llama-diagnosis-sweep-2026-09-07-v1/**`
- `audits/servers/server1/2026-09-07-alpha-jv-llama-diagnosis-sweep/**`
- `messages/server-heads/server1/2026-09-07-alpha-jv-llama-diagnosis-sweep*.md`
- `messages/acks/2026-09-07-alpha-jv-llama-diagnosis-sweep*.md`
- `tasks/status/2026-09-07-alpha-jv-llama-diagnosis-sweep*.json`
- `runs/server1/2026-09-07-alpha-jv-llama-diagnosis-sweep/**`
- ignored `local/alpha-jv-llama-diagnosis-sweep/<unique-attempt-id>/**`
- ignored `local/state/alpha-jv-llama-diagnosis-sweep-20260907/**`
- ignored local cap registries in this task/root (only prospective cap update).
GH owns global synthesis/canonical plan and main integration. Dedicated branch
commit and non-force push are approved for scoped source/tests/raw-free reports;
main merge/push awaits GH final source/artifact review. Existing roots and
failed attempts stay immutable. Use create-once paths; no user-worktree cleanup.

## Resource assignment and the one unresolved value

Server1 cap is 2 for NEW runs, one process and one GPU per job. D/S may use one
isolated GPU each when actual admission checks permit; allocation is not a claim
that GPUs are currently free. Count all project allocations, including existing
grandfathered runs and non-exited COMPLETING allocations. No preemption/co-load.
Existing runs on all servers remain entirely unchanged.

Explicit memory request: preserve server1 task convention 182272M per GPU,
below repository maximum 183296M; never omit --mem. Use source-pinned export
environment and existing resource/session checks. No paid/new infrastructure.

GPU-hour cap is UNASSIGNED (null) in both user design manifests. Do not invent
8h from the old pilot or 48h from another run. GH has asked user for the cap.
Implementation, CPU correctness, sealed-asset inventory/sample and offline
preflight are authorized immediately. Before model/GPU/Slurm submission report
separate D/S estimates (including model residency, fidelity, evaluations,
technical attempts, and one possible W5→W9 replay reservation).
Until GH conveys the user budget, GPU submission alone remains
WAITING_FOR_ISOLATED_RESOURCE / GPU_HOUR_BUDGET_UNASSIGNED.
This is NOT a hold on CPU work or an S dependency on D.
Once a budget addendum is received, minimum gates → authorized scoped GPU runs
without per-candidate approval. No automatic budget/scope expansion.

## Implementation review notes that must not become new science gates

- D exact W9/z is unproven. Inventory first; one bounded W5→B6–B9 replay only
  after reserving cost. First mismatch recorded; no W10-as-W9, no W0→B9 rescue.
  Sealed historical server2 assets may be unavailable: one bounded check and
  explicit unavailable/analogue status; do not stall S or poll failed server2.
- Restore W/M before new Family; attach complete FixedZ/context/semantic identity.
  FP32 subtraction and source N0 rounding precede FP64 weights; raw JVP first,
  then whitening. Preserve [D,B_active] flatten order and every native RHS row.
- qref fixed at five-layer entry, current q_l renewed; pass immutable lambda/T/N/h
  into actual trajectory and all telemetry/shadows. No partial monkeypatch.
- All-row influence is observer-only; NRMS only in the separate D branch.
  An improved S operating point is not proof of the original B10 cause.
- run_joint/run_official restore entry on return. Evaluate current/old900 and
  cross-objectives with the verified actual endpoint active, not the returned
  entry. Do not rerun writer/compute-z or append history to evaluate.
- S: new common cold B100, both models, 9 JV endpoints + Official + entry.
  Seven paths/model, 34 nodes/model nominal; five-lambda actual comparisons.
  Prefix T1/T2 from T4/h.5 is observation-only, append/capture 0, and maintains
  W/M/RNG/overlay integrity. Child effective clock differs from parent label.
- Keep model-specific stock z hparams, native P/L2, FULL-FP32 policy unchanged.
  Do not turn existing scientific failures into technical exclusions.
- Minimal correctness only: relevant CPU contract tests and normal-signal first
  GPU entry/nonzero-node FD/overlay fidelity; no per-candidate FD bank. Keep
  typed numerical/unavailable boundaries and evaluate finite poor endpoints.
- All plots/heatmaps are reproducible repository code producing PNG; no image
  generation/Codex visual tool. Raw tensors/prompts/cache are local-only.

## Reporting, transfer and review

First send FULL_READ/M0 with exact document identities and D/S ownership, then
resource estimate + availability/sample + CPU readiness. Give a compact first
S table as soon as available, independently of D; D attribution/intervention
report separately. Report technical failure, scientific outcome, unavailable and
budget statuses separately, preserve denominators and all nine candidates.

Pre-submit red-team: only source/input binding, cap/memory/budget, isolation,
the named minimal math/lifecycle/normalization/prefix gates; block stops affected
fixture, not other independent work. No outcome/tolerance gate inflation.
Post-run: verify counts, restore, raw/package hashes, truthful endpoints and cost.
Final REVIEW_READY must include exact runtime/analysis source, all tables,
reproducible PNGs, input availability, missing/NOT_RUN statuses and charged ledger.
GH independently reviews code and artifacts, not ACK-as-scientific-promotion.

Communication is registered app-server peer-direct both directions, not inbox
fallback. Long reports via path/SHA; explicit final completion is mandatory.
No unnecessary monitoring loop. Baseline/sample inventories are sealed-only;
Server4 processes/source/outputs/checkpoints/env/cache remain untouched.
Raw broadcast to running Server4 or unavailable Server2 is not required and
would harm isolation: record artifact-broadcast exception, GH has same-host raw
access; publish checksums/raw-free package on approved dedicated branch.
scientific_promotion=false; full lifelong/audit execution beyond this scope=0.
