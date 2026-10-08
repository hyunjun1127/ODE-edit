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
