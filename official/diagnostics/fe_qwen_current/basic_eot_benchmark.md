# Conditional BASIC EOT main CF / zsRE

Source only; no submission or recovery assertion. Root alone executes after
actual heldout W20 passes independent E/G/Loc gates 95/85/70. Missing or stopped
continuation fails closed. Native L2=10 is NOT author-default L2=0. Clamp stays1.
No masks, sink list, forced nonzero-position EOT, new targets, or key replacement.
Existing c36/8f6/36fc files are imported read-only, never patched on disk.

## Bindings and CLI

Start with the original b255 bindings (config/lock/runtime/pointer/checkpoint/
commit/writer), unchanged absolute member paths. Add:

- `heldout_result`: completed36fc `continuation-result.json` member.
- `heldout_commits`: ordered15 members B06..B20 in the continuation output.
- `heldout_pointer`: final continuation `checkpoints/latest.json` member.
- `dataset`: `cf` or `zsre`.
- `main_stream`: immutable original main first2000 common-record member.
- `original_main_config`: existing original main provenance config member with
  exactly matching `dataset` and `stream`. Not a newly relabeled heldout stream.
- `execution`: `source_commit`, actual scheduler `job_id`, `server`, `attempt`,
  `source_run_id` (heldout continuation W&B run). Root writes once at startup;
  `ODEEDIT_RENT_JOB_ID` must match.
- `storage`: existing host `lock_path`, `reserve_bytes` >=32GiB,
  `payload_bound`, `raw_margin_bytes`. Root performs whole-host admission too.

All members are `{path,bytes,sha256}`. Parent5 commits/pointer/paired proof are
followed through the immutable continuation bindings, not guessed by filename.
All raw/query files referenced by receipts must remain accessible. Latest W20
and parent B5 payloads are fullSHA/content-verified using mmap, never restored
into benchmark models. No full H clone. An unavailable proof is a technical
input dependency, not source readiness or execution authorization.

```
python -m official.diagnostics.fe_qwen_current.basic_eot_benchmark run BINDINGS NEW_OUTPUT
python -m official.diagnostics.fe_qwen_current.basic_eot_benchmark collect CF_RESULT ZSRE_RESULT NEW_REPORT
```

One fresh process/model per dataset. Actual edit requests come directly from
`main_stream` via unchanged `benchmark.native_records`, not a native data loader;
the archived native config's unused legacy data-path label is retained for exact
numeric/config comparison, and actual dataset/path/SHA are explicit in identity.
Fresh author contexts, H0 and seed; current-entry targets and c36 input adapter.
No heldout weights, H, contexts or targets are restored. C0 paths remain readonly.

W0 full2000, current100 B1..20, allseen500/1000/1500/2000. CF original queries
and strict NLL reducer; zsRE public queries with full CPU native parity and
independent saved predicted-target request-macro reduction. No EOT in evaluation.
CF quality stop80/70/65 only, after current raw. zsRE has **no such Loc gate**.
Finite/runtime guards remain. Milestone raw is saved before host-locked atomic
latest checkpoint replacement; only latest W20 survives normal completion.

W&B task/group `fe-qwen-basic-eot-final-main-20261012`, profile
`FE_PLUS_OURS_BASIC_POSITION0_EOT_FINAL_MAIN`, separate dataset arms and immutable
job identity. CF generation DEFERRED; zsRE has no generation config. Raw local.
`benchmark-result.json` is per-dataset completion, not paired success. Collector
re-audits both datasets,20 commits,25 endpoints,original query token plans and
CP/native-config identities; reports W0/W20/Loc delta, `recovery_claim=false`.
No generation in this adapter. Root assesses actual trajectory/performance.

## Source CPU receipt

2026-10-12: `test_basic_eot_benchmark`, `test_basic_eot_continue`,
`test_failure_eot_probe`, `test_failure_eot_prefix`, `test_benchmark`:
27 tests PASS, no skips (pinned author root provided). Includes actual tiny
native fit/supplier/writer resume and adapter parity, public-query prediction
reduction, CF NLL reduction, tracking schemas, preserved evaluation RNG, missing
W20/endpoint/commit and wrong-policy rejection. The native resume test uses a
tiny deterministic representation fixture, not pretrained resume certification.
`official.tools.verify`: source171/Python464/integrityPASS/externalimports0.
GPU/model-pretrained/online/Slurm/rent access: NOT_RUN by this source owner.
Actual heldout W20 and final CF/zsRE results were unavailable at publication;
no conditional proof was fabricated or weakened.
