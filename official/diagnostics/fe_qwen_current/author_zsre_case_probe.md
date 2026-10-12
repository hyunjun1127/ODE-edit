# Author zsRE POST-B15 three-case probe

Source-only implementation for `USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1`.
Root alone stages/runs on rent. No SH1/SH2 submission. No benchmark, W20,
recovery, or historical B13/pre-B15 reproduction claim.

```bash
python -m official.diagnostics.fe_qwen_current.author_zsre_case_probe \
  NEW_BINDINGS.json NEW_OUTPUT_DIRECTORY
```

The immutable bindings JSON has:

- `schema`: `author-zsre-post-B15-case-probe-v1`
- `case_ids`: `[1417,1425,1457]`, in this fixed order.
- `execution`: `{source_commit: NEW_40HEX, job_id: ACTUAL_JOB_ID, attempt: NEW_NAME}`.
  Job ID must equal worker `ODEEDIT_RENT_JOB_ID`, not parent `121243`.
- Standard regular-file `{path,bytes,sha256}` members: `probe_source` (this new
  Python module), `author_source` (unchanged `author_basic_eot_benchmark.py`),
  `author_bindings` (actual original zsRE job's bindings), `runtime`,
  `author_settings`, `commit` (`B15.json`), `pointer` (`latest.json`), `checkpoint`.
  Original parent bindings' referenced paths must remain readable and unchanged;
  this probe does not rewrite or relabel their provenance.

Parent source is `2144a6feffb51f66ecc24de2f818f632d5eeb8a7`, job `121243`,
zsRE stream `f42ee4bc6e98b1133e48dc81102201ccd85ccc917a97160ee3c6a0f3a38f378c`.
Exact B15 checkpoint: 7,856,481,561 bytes,
SHA `1235764707e1128122c94d4da5c3926a5aaa8d02bfbf219f0e981d3f8e8eee43`.
Author YAML, actual runtime/settings/identity, B15 commit/raw/pointer, selected
weights/cursor/order and unchanged c36 adapter are checked. Historical b255
references inside the parent binding are verified as provenance only: their
L2=10 configuration is never executed, and their old checkpoint is never loaded.
The actual probe uses parent **L2=0, clamp1, all author model/algs fields**, BF16,
SDPA, seed0, current-entry targets. No CLI hparam overrides exist.
The execution environment retains native NumPy1.26.4/tokenizers0.21.4 plus
the parent lock's exact torch/transformers versions; no scoring-runtime switch.

## Execution and interpretation

Load original cold HF locally; preserve only W0 L5 down-proj on CPU. Read-only
mmap of B15 restores five selected BF16 weights, contexts and RNG. Native seed
helper runs before saved RNG restore (including native cuDNN flags). Nonselected
parameters stay untouched. `cache_c` is never indexed, cloned, hashed or restored;
the authenticated whole-file SHA binds it without tensor traversal. No contexts
are regenerated. Source checkpoint is hashed again after the probe.

Exactly three native fits precede three supplier replays. c36 input overlay is
unchanged, and all six rewrite plus KL rows must be unaffected by basic EOT here.
Passive hooks retain only subject vectors at the initial delta-zero forward and
the final fit iteration: L5 down-proj key/output and block output. Outputs include
actual IDs/masks/labels/lookups, per-context NLL/probability, L4 init/delta/returned
target, final-fit vs replay/cast errors and L4–L8 replay targets. Six extra bare
forwards maximum (plain/injected for each fixed case) are performed, no evaluation.

Key group geometry uses the native BF16 reduction order:
`mean([Kbare, mean(Kprefix1..5)])`, excluding KL. The first-fit delta-zero keys
have teacher-forced suffixes but causal subject prefixes; the explicit bare
forward measures batching/representation differences. Final-fit averaged keys
are **injected-fit** keys, not actual writer keys. All quantities are post-B15
entry-state observations, not historical lower-layer-written residual/key states.

Existing tested row-chunked FP64 linear attribution computes the same observed
key under W0 and B15 L5 weights and their difference, with signed direction
fractions, top-eight channels, native BF16 rounding and decomposition error. This
is read-only linear arithmetic, not a model intervention or a solve. No full
matrices are emitted. Weights/versions, RNG, hooks/modes, requires-grad settings,
configuration and contexts are checked/preserved. The sole new output is local
`case-probe.json`, not a checkpoint or W&B measurement.

## CPU evidence

Tests use the actual pinned author `compute_z`, `compute_ks`, nethook, and actual
current-entry supplier with a random tiny BF16 Qwen (28 layers, fixture-sized
widths and three optimization iterations). Observer on/off targets, all model
weights/RNG/hooks are exact; native group mapping/KL exclusion and linear
decomposition are checked. Real author YAML acceptance, old tuned-input rejection,
fixed-case/entry guards, and a poisoned H payload prove the no-H restore route.
These CPU fixtures are not pretrained/GPU or historical-trajectory parity.
Validation before publication: **5 new tests + 9 reused case-probe/author-config
tests = 14 PASS**, CUDA hidden. No pretrained/model asset load in those tests.

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest \
  official.diagnostics.fe_qwen_current.test_author_zsre_case_probe -v
```

Both-author-W20 FluCon gating remains unchanged and unmet by this B15 diagnostic.
