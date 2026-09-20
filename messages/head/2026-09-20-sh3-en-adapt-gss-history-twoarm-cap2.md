# GH → SH3: RES / GSS_REC only, server3 cap2

Instruction: ODEEDIT-S06-EN-ADAPT-GSS-HISTORY-10K-SH3-V1 (same task).
Nonce: ODEEDIT-GH-SH3-GSS-TWOARMS-CAP2-20260920-R1.
Target: server3 / ubuntu / session 01a0b9c8-d12c-7133-b336-cc1b5fc2a6b3.

## Latest user authority

“RES / GSS_REC 두 arm만 사용하는 것으로 해봐. 그리고 cap 2개로 하자.”

This overrides the prior three-arm, cap1 and shared-prefix execution plan.
Keep original design/contract/cells/six authority files byte-identical as history;
bind this addendum and derived two-arm execution contract into new locks.
EN_ADAPT_H_RES and EN_ADAPT_H_GSS_REC are the only approved arms.
Do not implement or submit a separate uniform EN_ADAPT_H_GSS run.

## Execution

- Register two independent cold W0/zero M4 chains upfront, one GPU per arm,
  each B100×100 on the identical fixed10k order. Project/task cap2 includes
  other active allocations and admitted concurrent pending capacity.
- Each job performs its own B1–B2 once and continues from that same RAM state
  through B3–B100. No separate pilot, performance-based continuation gate,
  duplicate prefix restart or cross-job edited-state transfer.
- Two chains total 200 native/reference-gradient batch states, at most 400
  correction candidates. These are planned counts, not measured costs.
  Shared reference/model inputs are read-only; own history/teachers/state differ.
- Original three-arm 292-state/584-candidate planning is superseded, not evidence
  of reused work. Under the original no-overwrite pool upper-count convention,
  GSS_REC has at most 94 overflow sweeps and 57,516 selection NLL fact VJPs
  (600 + 93×612); RES has zero selection NLL VJPs. Actual occupancy/counts rule.
- Uniform RES versus recency-weighted GSS_REC changes selection AND weighting.
  This comparison cannot isolate a recency-only or selection-only causal effect.

## Unchanged scientific and reporting contract

Retain fixed R512/full-vocabulary W0 teacher, history bank512/pending100,
fact-version overwrite/repeat rules, signed-cosine selection, fixed sketch map,
GSS_REC age weights, fixed Pstar, native/optimized-z hook, epsilon/controller
and candidate rules. Do not retune thresholds, introduce new arms, substitute
reference subsets, or condition execution on official P/N.
Original evaluation schedule and RS/PS/NS plus TF rewrite/rephrase/neighborhood
accuracy (token micro, prompt macro, strict), NLL and paired analyses remain.
Past precision limitations remain NOT_ESTABLISHED where applicable.

## Resource and stop rules

Each job: 1GPU/8CPU/host≤121856MiB, ubuntu/gpu, exportNONE, Requeue0.
Check actual aggregate CPU/host memory, disk peak including both cold teacher
stores, and realistic wall-time before source/resource lock and held inspection.
Cap2 is permission, not a promise of simultaneous allocation. No duplicates to
fill slots, no changes to unrelated jobs, no deleting existing artifacts.
Update only server3 task-local cap config to match tracked cap2.
If a former uniform GSS job is already registered, report exact identity and
cancel only that newly excluded same-task job while preserving outputs/cost.
Do not claim cancellation or registration without receipts.

save_checkpoints=false, including equivalent edited W/M/delta/optimizer resume
bundles. Keep within-job RAM continuation; exact crash-resume NOT_AVAILABLE.
Cold history teachers/keys, metrics and cost ledgers are not edited checkpoints.
No previous storage waiver is inherited.

After held owner/source/args/resource inspection and release, perform NO
scheduler/log/result/initial/B2/B7 polling, terminal wait, heartbeat or callback.
Publish compact submission/source/locks/job mapping and stop
MONITORING_PAUSED_AWAITING_USER. Programs naturally execute; subsequent user
recall is required for completion review. Implementation ACK is not GPU PASS.
Do not request a new science approval merely because the arm count changed.
