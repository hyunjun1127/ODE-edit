# Conditional rescue-selected final CF Flu/Con

This is a **new eval-only** adapter, not an edit, target-fit, history replay,
W0 generation or job submission. It does not modify the rescue or old final
adapters/attempts. Root owns rent execution; SH1/SH2 remain HOLD/submit0.

## Required actual evidence — no missing-result fallback

1. `rescue collect` must have written a real SELECTED L2=100 proof. It is
   independently re-audited from held-out raw/queries/commits/checkpoint.
2. `rescue final-collect FINAL_ROOT` must have written actual
   `final-result.json` with `both_W20_complete=true`, CF and zsRE W20/2000,
   same selected L2. Preparation re-reduces both runs, all25 endpoints each,
   W0 +20 commits, source/config/stream/query/selection/CP bindings read-only.
3. Only the final CF `checkpoints/latest.pt` and `latest.json` are accepted.
   Held-out, partial, old schema, changed cohort, W0 targets, another L2 or
   another rescue contract are rejected. Current policy is rescue-L2-100-v1;
   no later hparam is adopted automatically.

There is currently **no actual final CP or success claim** in this source release.
Future failure/absence remains a typed preparation failure, never a synthetic
READY/PASS or reason to submit an unrelated run.

## Root host bindings and API

Bind an immutable JSON on rent, with standard `{path,bytes,sha256}` members:

```json
{
  "server": "rent",
  "final_result": {"path": "/ABS/FINAL_ROOT/final-result.json", "bytes": 0, "sha256": "ACTUAL"},
  "stream": {"path": "/ABS/original-cf-stream.json", "bytes": 4145654, "sha256": "66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37"},
  "reference": {"path": "/ABS/reference-binding.json", "bytes": 0, "sha256": "ACTUAL"},
  "tracking_env_file": "/ABS/APPROVED_PRIVATE_ENV",
  "reserve_bytes": 34359738368,
  "raw_margin_bytes": 4294967296
}
```

`ACTUAL`/zero member sizes above are **documentation placeholders**, rejected by
verification. Use real full SHA/bytes. Original ordered CF2000 case-ID hash:
`0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4`.
Final CF's existing model-assets member is reused unchanged; every regular
model/tokenizer member is verified. Snapshot alias is the one already bound by
that final run, not a newly substituted base model.

Reference binding uses the common asset loader wrapper:
`{"manifest": <original manifest member>, "asset_paths": {"attribute_snippets.json": "/ABS/...", "idf.npy": "/ABS/...", "tfidf_vocab.json": "/ABS/..."}}`.
Original manifest identity must remain
`75e595c7f26ec334830e9bb9ca6028098c19ea84a9509a5713985847683f8ea6`.
Original manifest6769B SHA
`6d9a713ab7eaa10871f277e10a3974e0bd5b140c265be80052c279f258876ca8`.
No refit/download/transfer is implemented. Manifest also requires exact NLTK
resource hashes and scoring versions: numpy2.2.6/scipy1.15.3/sklearn1.7.2/nltk3.6.5.
Thus the native editing environment's numpy1.26.4 is **not** silently accepted
for scoring. Root must bind an appropriate separate evaluation runtime; editing
runtime/frozen jobs remain unchanged. Preparation checks these assets without
loading a model and records exact evaluation runtime for execution.

```sh
python -m official.diagnostics.fe_qwen_current.flucon_prepare NEW_ROOT BINDINGS.json
PYTHONPATH=NEW_ROOT/source python -m official.diagnostics.fe_qwen_current.flucon run NEW_ROOT/config.json NEW_ROOT/lock.json
python -m official.diagnostics.fe_qwen_current.flucon collect NEW_ROOT/config.json NEW_ROOT/lock.json --output NEW_REPORT.json
```

Prepare requires a clean, published source checkout and creates an immutable
official source archive/lock/config. Root separately performs actual admission
and launch (no scheduler code here). Execution requires one visible GPU and an
actual job identity (rent `ODEEDIT_RENT_JOB_ID` through the shared transport).
No overwrite/retry/resume of an existing generation output is attempted.
Reserve>=32GiB+raw margin>=4GiB and free inodes are checked at prepare/start;
no new checkpoint, export, history clone or source deletion is created.

## Precision, restoration and metrics

- Authenticate full CP SHA before CPU mmap. Check current payload schema,
  native config, identity/cursor/2000 case order, B20 raw references and latest
  pointer; hash only selected tensors in bounded chunks. Full file SHA binds
  saved integrity/history/RNG bytes, **not** a fresh full-H digest computation.
- Selected Qwen down_proj weights layers4–8 must be BF16 shape3584×18944.
  Reuse `fe_cake_flucon.bind_weights`: validate all tensors before writing,
  exact BF16→FP32 copy and BF16 roundtrip; every unselected FP32-loader parameter
  keeps its object/storage/version. CP history, contexts and RNG are not applied.
- Load base model FP32/eager, TF32 off. This is common **evaluation precision**,
  not a change to author BF16 editing and not a claim that the entire FP32-loaded
  base equals all BF16-rounded editing weights.
- Reuse unchanged `NativeGenerationObserver`, profile
  `cf-cake-native-casebatch-kv-total100-globalrng-v1`, seed20261007 and original
  reference assets. Generate once for both metrics at W20/first2000 only.
  All generation prompts/occurrences/source/runtime/CP/state are bound, including
  non-ASCII record identities using the common generation digest convention.
- `read_observed(..., assets=...)` independently verifies token/count/work/raw
  identities and re-scores from the same generation text/fixed references.
  Missing reasons/valid denominators stay explicit. Flu bits/Con cosine raw are
  unchanged; `paper_generation` adds separately named x100 half-up2 strings.

Tracking is the existing official single transport, group/task
`fe-qwen-current-z-final-cf-flucon-20261012`, role eval_only, dataset cf,
profile `cf-native-generation-W20-only-v1`, source_run_id+checkpoint SHA.
Standard `all_seen/post/{fluency,consistency,generation}` and W20 progress only,
edits2000. No factual/fit/current/W0 publication, no partial final score.
Standing approved CF-generation schema instruction remains the canonical
transport authority; the current USER task is separately frozen as
`registration_authority` in local immutable config (no logger/schema fork).

CPU fixtures cover real tiny mmap restoration and untouched/RNG guards,
re-signed wrong-schema/identity/order/hparams/dtype/shape rejection, missing
both-W20 proof, actual shared fake-model generation/query/Unicode binding,
independent raw re-scoring/missingness and shared eval-only schema guards.
These are not pretrained/GPU/performance/real-checkpoint tests.
