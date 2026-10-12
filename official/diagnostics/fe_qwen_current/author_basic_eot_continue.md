# Author zsRE B15 → W20 continuation

Source-only implementation of the explicit user finish instruction. Root alone
stages and executes. No metric-quality early stop; technical failures remain
failures. No diagnostic prefix run, tuning, generation, or scheduler calls.

```
python -m official.diagnostics.fe_qwen_current.author_basic_eot_continue run BINDINGS NEW_OUTPUT
python -m official.diagnostics.fe_qwen_current.author_basic_eot_continue collect NEW_OUTPUT/continuation-result.json NEW_AUDIT.json
```

Bindings schema `author-basic-eot-zsre-B15-resume-v1`, dataset `zsre`:

- `execution`: new actual `job_id` matching `ODEEDIT_RENT_JOB_ID`, `source_commit`,
  `server`, `attempt`, `source_run_id=a2615463bd5d4b75`.
- File members (path/bytes/full SHA): `resume_source`, `author_source` (frozen
  2144 author module), `author_bindings` (original 121243 bindings), `runtime`,
  `author_settings`, `commit` (B15), `pointer`, `checkpoint`, `parent_query_parity`.
- `parent_commits`: ordered B01..B15 members, last equals `commit`.
- `parent_evaluations`: original 19 event records, each batch/endpoint/requests/
  raw member/queries member/summary; W0 is original full2000, not W0first500.
- `storage`: same prefix admission contract (payload_bound, reserve_bytes,
  raw_margin_bytes, lock_path); reserve at least32GiB, shared host lock.

Parent CP must be original B15 (7,856,481,561 bytes,
`1235764707e1128122c94d4da5c3926a5aaa8d02bfbf219f0e981d3f8e8eee43`).
Old partial B16 fits are not commits. New output begins B16; native cfg remains
2000/100 and author L2=0, clamp1. Cold nonselected weights are retained, selected
BF16 W and native FP32 H use private mmap restore; context/RNG hashes must match
before first fit. Parent checkpoint is rehashed after execution, never replaced.

Original public-query evaluation and c36 input adapter are inherited unchanged.
Parent W0 is independently CPU-audited then logged as a parent observation,
without a new model forward. New current100 B16..20 and allseen2000 give six new
endpoints. Collector independently links/reduces all 20 commits/25 endpoints.
Latest W20 checkpoint is saved in the new output under the shared atomic lock.
Status `AUTHOR_BASIC_EOT_ZSRE_RESUMED_W20_COMPLETE` states completion, not recovery.
Paired generation gating is unchanged and is not a prerequisite to this resume.

CPU tests use actual pinned native compute_z/supplier/writer with tiny BF16 Qwen;
representation extraction is a deterministic small fixture, not pretrained
parity. They check uninterrupted20 versus B15-private-resume tail W/H/RNG and
unchanged parent bytes. Actual author YAML and public-query regression tests
must also be run. No GPU evidence is claimed by source tests.
