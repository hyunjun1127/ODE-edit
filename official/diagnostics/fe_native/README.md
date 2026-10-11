# Native FE sequential diagnostic

This package prepares a pinned checkout of [the author FE repository](https://github.com/jugechengzi/FE),
commit `478134dfb24b43f4e18b47e8500893ce3f9cc50f`. It is **not an editing runner**.
Execution is `python main.py` in that checkout, followed by the author's `main.py test_only=True`.

The author's README requires replacing `algs/memit/memit_main.py` with `FE-memit_main.py`.
`patch.py` performs that installation and changes the target-supply section of `apply_memit_to_model`:
each 100-request batch computes firstforward targets on its current entry model before any weights
are written. Future targets are neither loaded nor calculated. Batch-local indices address only
those 100 target columns. Original request order and the accumulated native history remain intact.

`main.py`, model loading, author YAML settings, BF16, context generation, `compute_z`, `batch_edit`,
covariance loading, LU solve, key precision, history accumulation, and weight assignment stay native.
The `algs/memit/compute_z.py` used here is the author's existing function; its only difference from
`z_methods/compute_z.py` is two informational print statements. Forward replay follows the author's
`compute_z_batch_first_forward` on bare prompts, with support for tensor/tuple block outputs.
The editing process has no ODE runner or EasyEdit algorithm imports. The default control
uses the original prior and native contexts and precision.

There is a separate observation-only addition: `checkpoint_observer.py` saves native weights,
history, contexts, RNG and case IDs at W0 and every 500 edits. It does not evaluate or update the
model. Snapshot tests verify unchanged weights/history/RNG. The final native `save_model` also
runs normally. Milestone payloads contain resume state but automatic native resume is not implemented.

Paths/cache aliases and data-format conversion adapt local assets without changing the 2,000
ordered CF requests. The original-prior control uses the existing covariance NPZs, including
GPT-J L3. The diagnostic is kept separate from the current baseline main table. If alternative samples
are needed later, declare their seed/order before seeing results and retain the original failure.

CPU checks:

```bash
FE_NATIVE_AUTHOR_ROOT=/path/to/pinned/FE python -m unittest official.tests.test_fe_native -v
```

They exercise the actual native apply loop and unchanged writer on small tensors, batch-entry
target scope, persistent history, target replay parity, model mutation rejection, and snapshot
roundtrips. GPU performance must be established by the submitted native runs.

`evaluate.py` is an independent, read-only checkpoint evaluator. It runs after native
training, checks checkpoint/config/stream identities, restores only the selected weights
into a fresh BF16 model, and records common CF per-case raw observations at W0 and each
available 500-edit checkpoint. W0 uses all 2,000 requests; W5/W10/W15/W20 use all seen
requests at that point. The checkpoint file SHA is verified before and after evaluation.
It never invokes target fitting or modifies the native training program.
A native configuration with `dataset: zsre` is evaluated with the public-query zsRE evaluator
(`official.evaluation.zsre_paper`, request macro, Loc = `loc_ans`, `model_family` = the configured
model) on the immutable official zsRE stream rows (`original_stream`); a missing field means CF.
The native editing input for zsRE is the same first 2,000 official rows converted to the author's
zsRE record fields (`prompt`/`subject`/`target_new`, plus the author's paraphrase/neighborhood fields
for its own `test_only` metrics); see `official/runners/server4/fe_zsre_lane.py`.

`+diagnostic_history_batches=[...]` is an optional storage-only observer setting. Without it every
snapshot stores the native history (the original behavior). With it, only the listed completed-batch
snapshots keep the history tensor; the others store weights, contexts, RNG and case IDs with
`history=None` and `history_saved=False`, so each one is still evaluable. Editing is unaffected.

## Explicit causal controls

The following controls separate the target reference from request arrival and covariance
validity. All retain the native writer, author hparams, persistent history, BF16 and the same
ordered requests. A control is recorded in its own immutable configuration and output folder.

| Control | Target reference | Requests fitted at a time | Covariance |
| --- | --- | --- | --- |
| Current sequential diagnostic | Current batch-entry model | Current 100 only | Original NPZ |
| GPT-J numerical repair | Current batch-entry model | Current 100 only | Derived L3 NPZ; other layers original |
| Qwen initial-model control | Initial model W0 | Current 100 on arrival only | Original NPZ |
| GPT-J matching initial-model control | Initial model W0 | Current 100 on arrival only | Same derived L3 NPZ; other layers original |

`+diagnostic_z_reference=initial_model_on_arrival` selects the initial-model reference for either model. `TargetSupplier`
saves the initial selected weights on CPU, temporarily restores them for the arriving batch's
target computation, and restores all editing weights before the native writer runs. The same
restoration happens if fitting raises. There is no future target cache. The unchanged native data loader still reads the full dataset JSON;
the restriction applies to target fitting and use, not streaming I/O. This option changes the
reference model relative to the default and must be labeled separately; it does not silently
replace the current baseline. The default is `batch_entry`.

The native configuration still prints legacy fields such as `z_method: all` and
`num_z_samples: 2000`. The replaced target-supply path does not consult these fields:
the apply loop passes only its current 100 requests to `TargetSupplier`. Batch markers
record their actual case IDs; fitting outside those markers is rejected by the log audit.

`covariance.py` creates a separate native-format NPZ from an explicitly identified defective
FP32 moment cache. It preserves the count and other metadata, applies the documented
storage-resolution repair, and writes an FP64 summed moment. It verifies the reloaded matrix
is positive definite and records the original/derived SHA and correction size. It rejects
healthy assets and never modifies the source NPZ. Only the repair control points the author's
native cache path to this derived file; the author YAML and solver remain unchanged.

The checkpoint records the target reference, and the evaluator rejects a configuration with
a different reference. CPU checks also exercise initial-reference restoration on success and
failure, derived covariance roundtrips, metadata preservation and rejection of wrong sources.
Run registrations, scientific differences and current evidence are in the
[FE investigation report](../../../experiment-reports/global/fe-collapse-repair-20261011/report-ko.md).
