# USER-disabled GPU qualification cold rerun

Authority: USER-GH-SH1-SH2-OFFICIAL-NO-GPU-QUAL-RERUN-20261009-R1.
Direct owner nonce: GH-SH1-OFFICIAL-NO-GPU-QUAL-RERUN-20261009-R1.

## Exact transition

Actual old receipts/launcher SHA/owner janghj/Command/WorkDir/devbox were checked
before mutation and again before each operation. Pending targets were held before
reverse-topological cancellation. Cancelled: 61658–61664, 61675–61679,
61681–61694 (26 jobs). Completed qualification 61657 was preserved, not cancelled.
No main was completed at reconciliation. No old checkpoint or raw was removed.
Accounting elapsed/allocation and before/after scheduler evidence are retained at
`local/official-baselines/server1/no-gpu-qualification-r1/cancellation/`.
Other servers, PRICE/OURS, GPT2 and unrelated tasks were unchanged.

## Source changes

- New task-local noqual runner/prepare profile; no qualification/smoke CLI mode.
- Explicit immutable policy: qualification and checkpoint-resume GPU equivalence
  `NOT_RUN_USER_DISABLED`, not PASS. Old qualification outputs/plan/smoke gates
  are forbidden in new configs. Existing old profiles/validators are retained.
- Exact registered subset: CF FT/MEMIT/MEMIT_FE/ALPHAEDIT/SPHERE; zsRE six methods.
  No unregistered CF BLUE chain.
- Same old native algorithms/hparams/seed/order/BS100x20, checkpoints, factual
  schedules and runtime guards. Cold main starts; no old partial resume.
- CF W20 generation stays DEFERRED; original pipeline CF W0 factual/generation
  and zsRE baseline reference observations remain. Projected CF own cold W0
  factual behavior remains; zsRE independent W0 reference and no generation remain.
- W0 readiness is issued only after actual required observations. Portable
  qualification-backed W0 publication is explicitly NOT_PUBLISHED, not faked.
  New own consumers validate original source/assets/observation identity without
  waiting for removed qualification receipts. Other servers/readers unchanged.
- GPU0 collector verifies actual factual/commit/checkpoint records and reports
  qualification USER_DISABLED, independently of scientific completion.
- GPU cap4 from latest direct USER and actual local server1 cap row; no increase.
  Four lanes, only required W0 input edges and resource afterany edges; no old IDs.

## CPU evidence and stage

Focused tests: noqual routing, existing submit/collector/tracking/zsRE regressions,
63 PASS, CUDA_VISIBLE_DEVICES empty. Source verify:157 source SHA,269 Python,
external task imports0. Initial test iteration exposed an import name shadow and
incomplete CF fixture; fixed and rechecked before source freeze. No GPU test run.
Owner source review performed; no separate reviewer used.

Prepared11 configs from original sealed actual registrations, preserving asset
manifests and raw source lineage. Storage plan512GiB; fresh node/QoS/inode/free disk
checks and held source/argv/resources/dependency verification are performed by
the existing submit control path. Registration source and actual jobs will be
recorded separately after submission. README is GH-owned and unchanged here.

## Actual held submission / release

Execution source `94304dc93db928300a71d8dcd0f87199fdb93f1f`, official tree
`4b09cc81252f8e190621941bdd9b9234dd3013bd` was main-published before archiving.
All14 exact jobs held-inspected and released in one deliberate registration pass.
Current snapshot: 2026-10-09 04:28:08 KST. Qualification and checkpoint-resume
GPU equivalence are NOT_RUN_USER_DISABLED throughout, not failed or PASS.

| dataset/role | method | job | state | dependency |
| --- | --- | ---: | --- | --- |
| CF W0 input | none | 61709 | RUNNING | none |
| zsRE W0 input | none | 61710 | RUNNING | none |
| CF | AlphaEdit | 61711 | RUNNING | none |
| CF | SPHERE | 61712 | RUNNING | none |
| CF | FT | 61713 | PENDING | afterok61709 |
| CF | MEMIT | 61714 | PENDING | afterok61709; afterany61710 |
| CF | MEMIT_FE | 61715 | PENDING | afterok61709; afterany61711 |
| zsRE | FT | 61716 | PENDING | afterok61710; afterany61712 |
| zsRE | MEMIT | 61717 | PENDING | afterok61710; afterany61713 |
| zsRE | AlphaEdit | 61718 | PENDING | afterok61710; afterany61714 |
| zsRE | AlphaEdit-BLUE | 61719 | PENDING | afterok61710; afterany61715 |
| zsRE | MEMIT_FE | 61720 | PENDING | afterok61710; afterany61716 |
| zsRE | SPHERE | 61721 | PENDING | afterok61710; afterany61717 |
| CPU collector | none | 61722 | PENDING | afterany61709–61721 |

GPU1/CPU8/65536MiB/48h per GPU job; collectorGPU0/CPU8/24576MiB/4h.
Current stricter local cap4, admitted DAG width4; pre-admission owner devbox queue0.
Actual disk available1,247,462,383,616B/inodes334,947,363; planning reserve512GiB.
Old cancelled IDs and qualification proof are absent from the new DAG.
Job names/config SHA/source/dependencies/snapshot are in the compact
`audits/servers/server1/official-baselines-20261008/no-gpu-qualification-rerun-20261009/submission.json`.
Exact raw control receipt directory:
`/mnt/raid5/janghj/ODE-edit/local/official-baselines/server1/no-gpu-qualification-r1/registration-r1/`.
W&B identity/remote startup readback and scientific completion have not been
observed in this control handoff. No long GPU wait or recurring monitor.
The GH cancellation relay was transport-accepted; owner ACK was not observed.
