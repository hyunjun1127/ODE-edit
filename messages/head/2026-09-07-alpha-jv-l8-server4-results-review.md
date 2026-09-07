# GH → SH4: L8-only rerun terminal validation and comparative report

instruction_id: ODEEDIT-S06-ALPHA-JV-L8-SERVER4-RESULTS-REVIEW-V1
parent_instruction: ODEEDIT-S06-ALPHA-JV-SEQUENTIAL-ROUTING-SH2-V1
nonce: ODEEDIT-SH4-L8-REPORT-20260907-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
expected_cwd: /data/janghj/ODE-edit
repository: hyunjun1127/ODE-edit
authority: user requests SH4 summarize L8-only rerun results and produce report.

## Scope and authority

This is analysis/report work, not a new experiment. Perform one bounded scheduler
and task-owned artifact availability check for job38433_[4-5], then validate and
analyze terminal/sealed outputs. Mapping4=Llama/5=Qwen L8_ONLY_NATIVE, fresh
W0/M0 sequential B100x10. Do not assume COMPLETED from the user's wording.
If a cell is still running, preserve it and report that fact; analyze only
already sealed eligible outputs, label partials, and do not create a polling loop.
A partial chain is not final W10. A missing terminal/metric is not a zero score.

No model/GPU/evaluator/edit replay/new Slurm submission, cancel, restart,
post-hoc repair of results or running-source modification is authorized by this
report instruction. Current grandfathered runs remain unchanged; prospective
Server4 new-run cap2/mem60416M remains in force. SH1's48GPUh D/S budget is not
a grant for this task. Leave ORBODE37649 and SH1 D/S source/process/data alone.

Fetch Git in a separate clean analysis worktree, not a source-locked execution
worktree. Do not clean or revert user changes. Prefer existing analysis/state
parsers and plotting code; only modular report-specific changes. Code-generated
PNG only, no Codex visualization/imagegen/manual plot editing.

Allowed new/modifiable paths:
- project/run_scripts/alpha_native_response_ode_v31_sequential/l8_takeover_analysis/**
- experiment-reports/servers/server4/alpha-jv-l8-only-sequential1000-review-2026-09-07-v1/**
- audits/servers/server4/2026-09-07-alpha-jv-l8-only-results-review/**
- messages/server-heads/server4/2026-09-07-alpha-jv-l8-results*.md
- messages/acks/server4/2026-09-07-alpha-jv-l8-results*.md
- tasks/status/alpha-jv-l8-results-review-20260907/server4.json
- ignored local/alpha-jv-l8-only-results-review/<unique-attempt>/**
If output exists, use a new version/attempt. Original raw and reports immutable.
Dedicated branch commit/non-force push approved; main integration remains GH
reviewed. Workers may assist bounded checks but do not push or approve themselves.

## Bind evidence before conclusions

Submitted source44602a1a80554c67da0ef9646b43d843104785f2/
treecdb089a139f180c822d1e6bf44fe3d27b858c1ce,
branch codex/server4-alpha-jv-l8-takeover-v1.
Task root /data/janghj/ODE-edit/local/state/alpha-jv-migration-server4-20260907/tech-r1,
submission receipt /data/janghj/ODE-edit/local/state/alpha-jv-migration-server4-20260907/submission-38433.json.
Resolve actual source/job/paths from these receipts; distinguish submission
reference, executed source, analysis source and report commit. Do not silently
mix versions. Verify actual terminal validity, exit state, request/order/arm
inventory, duplicates/nonfinite/failures and source/science differences.

Comparison: completed O_NATIVE/JV_NATIVE4 chains on Server2, job37980,
scientific source77358b1546d1baf83b3e251afcce663b08d7bfd7,
publication0d0a0131e4a6a2a645dfa6530377d420a084d136:
experiment-reports/servers/server2/alpha-native-response-v31-sequential-routing-2026-09-06-v1/main-four-terminal-v1/factual-report-ko.md
report SHA b9b7fd9b7f37f782ee5fe81608d1b80e942e701ccf01b9896d4d79401d2617ad.
Reuse that sealed publication/CSV; do not rerun O/JV. Exact common sample1000
root40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd.
Cross-host numerical/backend/asset differences must be disclosed, not assumed
bitwise identical solely because source settings match.

Read GH analysis (main-published):
experiment-reports/global/alpha-jv-sequential-mechanism-audit-2026-09-07-v2/gh-mechanism-review-ko.md
and SH1 detailed layer report:
experiment-reports/servers/server1/alpha-jv-sequential1000-layer-review-2026-09-07-v1/factual-report-ko.md.
Use completed sealed evidence only; do not read SH1 current D/S or ORBODE partials.

Server2 old38306_4/5 was user-directed external termination for takeover, not
scientific failure. SH2 now reports only B1=100 each, at B2_TARGET cancellation;
W1/M1 exists, finalW10 absent. Keep old partial consumption/exclusion distinct
and never join its prefix into the fresh38433 chain or count it as a second
completed replication. Its frozen failure receipt must not be rewritten.
If Server2 external raw is unavailable, use published evidence and disclose
verification limits; don't block the whole Server4 report on recovery.

## Minimal essential integrity

Validate L8 terminal receipts/raw member hashes once (stream large files),
sample/order, B1→B10 W/M continuity, own arm/batch fixed-z once and recompute0,
history append1/batch, new batch dictionary/reference, finite values, final
restore and actual W/M checkpoint1/5/10 where recorded. Source-preserved
FULL-FP32/exceptions/padding/context/target policy. Confirm current and
all-seen metrics are evaluated at the stated materialized W, not restored W0.
These are evidence checks, not new outcome rejection gates or extra GPU probes.

L8_ONLY_NATIVE is support-restricted JV, NOT Official AlphaEdit with hparams
layers=[8]: full five-layer inventory/P indexing, entry qref all5, full residual
directions, N0/lambda.1/T2/N4/h.5, current-state L8 JVP, and baseline all-layer
history finalization remain. Check actual source and receipts for this contract.
If missing information, record NOT_RECORDED rather than infer a PASS.

## Required report: O vs JV vs L8-only, both models

Put a compact two-model three-arm main table first, at the same finalW10:
all1000 canonical RS/PS/NS exact numerator/denominator and percentage.
Canonical metrics are NLL preference, ties fail; not free-generation accuracy.
Alongside main metrics include rewrite/rephrase strict or token accuracy with
the correct label, target-new/target-true NLL and margin mean/median/p90/max.
Do not call historical teacher-forcing true-token accuracy canonical NS.

Then include:
1. All10 current-B100 rows and full seen-prefix rewrite retention matrix,
   separately from all-seen RS/PS/NS at recorded W1/W5/W10.
   No interpolation of missing intermediate PS/NS. Online own-batch pooling
   must not replace finalW10 full1000 retention.
2. Cohort/age and at-write→final transitions: initially failed vs success→loss
   vs recovery vs declared overwrite, with whole and conditional denominators.
   First-B100 acquisition and later forgetting are distinct.
3. Prompt-level W0 and Official-relative NS new loss/recovery and true/new NLL
   shifts where existing evidence permits. Aggregate gain alone does not prove
   same prompts preserved. If paired raw missing, label unavailable.
4. Batch/node/layer physical write: absolute velocity and net DeltaW,
   total magnitude, raw/normalized native work vs net action, history/L2
   decomposition, signed predicted progress and L8 share. No claim that share
   movement alone is useful redistribution. Compare JV's small early-layer
   contribution to zero-support L8-only without forcing dispersion.
5. V/V0, N0 scale distributions, NNLS/KKT, actual barrier increment vs
   finite-step defect, predicted/actual response and materialization mismatch.
   In Llama B10, check recorded near-stall/at-write failure/tiny scales and
   difference from JV; never infer same B10 W/M/z across divergent chains.
6. Actual wall/GPU time, target/keys/solves/mainJVP/evaluator/finalization/
   snapshot cost. L8 reduces JVP count but do not assume1/5 whole-runtime:
   inspect whether all5 native directions were still built.
7. Provenance, exclusions, absent metrics, tests, exact reproduction commands
   and paths for CSV/PNG/raw-local inputs, hashes, manifest/rooted receipt.

Figures should include cumulative/current canonical metrics, rewrite retention
heatmaps, per-batch absolute update/progress, and norm/barrier diagnostic panels
only where recorded. Weight-update title: "Layer-wise Update Magnitude";
no uniform-allocation reference line or "bars:" footer on weight figures.
Activation ideal schedule, if reused, belongs only to its own distinct panel.

## Interpretation and handoff

Answer separately for Llama and Qwen:
- Does L8-only reproduce JV's locality/retention relative to Official?
- Does full JV show benefit that support-restricted L8 cannot explain?
- Is Llama's B10 anomaly shared, absent, or unobserved? Do not turn a different
  chain result into a same-state causal explanation of N0 or history.
- Is the effect consistent with early-layer write avoidance, smaller total
  update, exposure/feedback, or still unresolved? Compare absolute action.
- KKT/native dissipation vs finite Euler behavior vs functional retention are
  different claims. Layer concentration alone is neither success nor failure.
- Single-stream/model-specific results are descriptive; do not claim statistical
  significance without appropriate paired/request-cluster evidence. Host
  differences and missing raw verification limit the comparison.
- No new hparam choice, production normalization patch or lifelong release.

Report terminal/input validation first. As soon as the canonical six-arm table
is available, send its path/SHA without waiting for all plots. Then complete
the factual Korean report+CSV+reproducible PNG+manifest/rooted receipt, essential
CPU checks/rehash, and push the dedicated branch.
Mandatory terminal REVIEW_READY to GH through registered peer-direct: report
absolute/repo-relative path, branch HEAD/tree, package identity, denominators,
failures/exclusions/missing status, jobs state and new model/GPU actions0.
GH will independently inspect code/artifacts before main integration.
Raw remains local; broadcast exception is to protect ongoing isolated runs.
Git raw-free publication suffices for GH initial review; any later raw access
uses exact sealed paths, not broad live transfer.
scientific_promotion=false.
