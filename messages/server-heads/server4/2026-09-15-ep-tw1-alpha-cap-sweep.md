# SH4 cap sweep submission — completion in progress

Instruction ODEEDIT-S06-EP-TW1-ALPHA-CAP-SWEEP-SH4-V1. CAP1/job47962 reused; new CAP10/CAP100/NORM_ONLY registered upfront.

Execution8c64366c2314f188e034e5f4403fe89f0eaad873/tree76ef5072b70134bf4bc1f9d3659a056041922bde. Archive304be4f3de58ce2f9457f63324646525000191609ba64e1bb1e290253e59ea21 (1751040B). Source branch pushed non-force; not a completion/main report.

| Arm | Job | Dependency | Execution lock SHA |
|---|---|---|---|
| CAP10 | 48148 | none | e1ecd43a3b26162dc9cbecfa83ca9ac6f79fd76a29d5600c7273f1db7cab70b4 |
| CAP100 | 48149 | afterany48148 | 6a1c34a722849f73b2dd8203d2190cfbbb147374802a1f33e89118eb9c742eeb |
| NORM_ONLY | 48150 | afterany48149 | fda2657e39ba00af78b0ff554c1a2af9d75204601fcaa3984c7a6c4e0bee8b1b |

All held owner/source/1GPU8CPU60416M/12h/exportNONE/Requeue0/dependency inspections passed before release. Admission observed CAKE48101 reservation1; sweep lane1 gives project aggregate cap2. No CAKE job/source/output changes or science monitoring. Dependency successors do not run concurrently with their predecessor, and independent arms continue after a predecessor technical failure without quality-based selection.

Release2026-09-15T10:53:09.668547UTC, all PENDING. Subsequent own-job bounded observation: CAP10 ReqNodeNotAvail, CAP100/NORM_ONLY Dependency. GPU initial evidence NOT_RUN. This task remains active under the explicit completion override; no callback/daemon or other-task recall.

Root `/data/janghj/ODE-edit/local/ep-tw1-alpha-cap-sweep/20260915-v1/`; arm outputs `<arm>/attempt-v1/scientific-v1`. Release receipt `submission-v1/release-receipt.json` SHA2ec7bb751b9fdf50bbad89215961d144c3a4f361608a712cdfbb0e7a8b9d8311. Resource free245889884160B, 60GiB sweep reserve+20GiB safety (not exclusive filesystem reservation). Existing CP/weight/teacher preserved; new sweep 10CP/arm and W10 weights retained.

SKIPPED_USER_DIRECTED / numerical_validation=NOT_ESTABLISHED. No FD/ULP/direct/selfKL GPU diagnostics. No baseline/teacher generation. CPU11 focused config/legacy arithmetic/routing tests PASS only. Estimated new3 cost6.4117GPUh, measurement pending. GH M0 direct ACK received, no duplicate audit requested. Completion/report/main integration remain pending.
