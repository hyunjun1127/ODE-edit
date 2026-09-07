# GH → SH1: D/S total 48 GPU-hour budget and conditional execution release

instruction_id: ODEEDIT-S06-ALPHA-JV-LLAMA-DIAGNOSIS-SWEEP-SH1-V1
nonce: ODEEDIT-SH1-DS-BUDGET48-20260907-R1
source_session: 01a04939-8873-7673-8dca-4c7fc5e31af0
target_session: 01a04939-f93a-7b50-bca0-65438eab2062
expected_cwd: /mnt/raid5/janghj/ODE-edit
repository: hyunjun1127/ODE-edit
authority: explicit user "gpu hour 상한은 48hour이다"

## Budget assignment

- Assigned host: server1/devbox. Concurrent new project GPU cap: 2/server.
- Total D + S campaign GPU-hour ceiling: **48 GPUh = 172800 GPU-seconds**.
- This is a combined campaign budget, NOT 48 hours per cell, job, track or model,
  and not 48 hours of two-GPU wall time. At full two-GPU occupancy, 24 hours
  consumes the entire budget. Shorter occupation leaves more remaining budget.
- Include GPU-resident model load, fixture/fidelity, raw captures, controls,
  allowed bounded replay, primary trajectories, diagnostic GPU work, endpoint/
  old-edit evaluation, materialization and every technical attempt. Count actual
  allocation/residency conservatively, not only kernel time. CPU-only work and
  unallocated queue wait are not GPU consumption.
- Freeze the assigned 48 in a new resource lock/addendum; keep prior null-budget
  design/CPU receipts immutable. Report charged, reserved-running and remaining
  GPU-seconds. Bound all concurrent worst-case reservations by the remaining
  campaign ceiling; do not grant each job 48h. No automatic budget enlargement.
- Keep explicit --mem=182272M per GPU (repo ceiling183296M), one process/GPU.
  Recount actual isolated capacity immediately before admission. Existing
  grandfathered runs on all servers remain unchanged. No Server4 sharing,
  preemption, source modification, live-artifact reads or resource recovery.

## Execution authority

The budget-unassigned HOLD is lifted. Implementation and scientific GPU/Slurm
submission within the original D/S scope are approved after the remaining
essential production launcher/asset/resource/fidelity locks. SH1's CPU67/67
report is not retroactively PRE-GPU PASS. Complete only these necessary
checks, publish the source/runtime/resource/science/sample binding, then submit
without another GH per-gate or per-candidate approval.

The CPU handoff reference is71987bcb3022631f298a1caa111443fa44a472f6,
tree84375fe636a4e0ef0a9001b2e723a92efcaa1092; immutable primitives77358 remain
unchanged unless separately scoped repair is approved. Seal actual executable
source rather than claiming a CPU-only source is already a production launcher.

D and S stay independent: S may enter normal-fixture fidelity and the two-model
9-endpoint sweep while D closes asset availability. Missing historical W5/context
does not justify W10-as-W9 or an automatic W0→B9 replay. D remains typed exact
unavailable/analogue as the original contract specifies; if the authorized
W5→B6–B9 one-time route becomes possible, reserve its cost first. Extra budget
does not authorize remote recovery, an invented analogue fixture, repeated
hash-matching replay, a new full-chain sweep or extra science arms.

Keep all five lambda values actual in both models, the seven-path/34-node
shared-prefix plan, source-exact N0 for S, stock model-specific compute-z,
fixed sample/audit reservation and existing numerical policy. N16 is only the
original conditional paired follow-up under its separate manifest after N8
review and within this same budget. No full Cartesian/lifelong/audit expansion.
NRMS remains D-only ablation; no result-dependent rescue/replacement.

## Reports and handoff

Return compact budget ACK (resource.lock path/SHA,48GPUh,cap2), followed by
actual PRE-GPU/submission receipt when achieved. Publish first S candidate table
without waiting for D, then separate D diagnosis/intervention findings.
Report all finite poor endpoints and unavailable/budget statuses; do not convert
the user budget into a scientific PASS. Preserve completed prefixes on technical
failure and count failed attempts. Mandatory final REVIEW_READY binds source,
raw/aggregate/PNG/code/locks and full cost ledger; GH reviews independently.
Dedicated branch push remains approved; main integration still GH-reviewed.
No new proposal or broad gate expansion. Existing peer-direct/ownership/report
paths and artifact-broadcast exception remain in force.
scientific_promotion=false.
