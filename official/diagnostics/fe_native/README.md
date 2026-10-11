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
No ODE runner, EasyEdit algorithm, prior repair, shared context override, or FP32 override is used.

There is a separate observation-only addition: `checkpoint_observer.py` saves native weights,
history, contexts, RNG and case IDs at W0 and every 500 edits. It does not evaluate or update the
model. Snapshot tests verify unchanged weights/history/RNG. The final native `save_model` also
runs normally. Milestone payloads contain resume state but automatic native resume is not implemented.

Paths/cache aliases and data-format conversion adapt local assets without changing the 2,000
ordered CF requests. Original covariance NPZs are used without repair, including GPT-J L3.
The diagnostic is kept separate from the current baseline main table. If alternative samples
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
