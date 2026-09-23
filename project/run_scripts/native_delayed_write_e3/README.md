# BASE delayed-write endpoint diagnostics

Authority: GH-SH4-NATIVE-DELAYED-WRITE-E3-20260924-V1.

`receive_design` validates fifteen authorized small inputs. `receive_memit` is a
separate necessary ordinary repo-local input pull, justified under envelope §2
and PROTOCOL; it does not extend the small-file approval. Both are create-once.

`panels` fixes all selections without model outcomes. `prepare` binds CPU tests,
original BASE checkpoint/source/data provenance, input panels and resource plans.
`control` registers a held single-GPU persistent science runner plus CPU afterany
collector, inspects source/args/owner/dependency/resources, then releases both.

Internal DAG: G00 → G10 → G20 → G21 → G30 → G31 → G40 → G50 → G51 → G60.
Each transition requires the prior atomic PASS with exact instruction/attempt/
source/data/panel identity. `reduce` is G70 and the failure collector; it has no
model or submission imports. A failure cannot turn into scientific exit0.

No native fitting, new weight update/history append or checkpoint writes.
Original writer/evaluator/source bytes are read-only lineage. Forward tokenization
and FP32 eager policy match the frozen canonical evaluator. New module/patch code
is independent and validated on original rows in G10, not claimed validated by CPU
toy tests. The two families' saved W4–W8 are not confused with BLUE/L4-only states.

General128 W0 full-vocabulary reference is retained only in CPU RAM for the single
persistent process; it is not a disk resume mechanism. Full token keys are bounded
to input chunks and recomputed for patching. Diagnostic scalar/row output is
local-only; compact reports/CSV contain no prompts or tensors.

CPU tests:

```
/data/janghj/EasyEdit/.venv/bin/python -m unittest project.run_scripts.native_delayed_write_e3.test_contract -v
```

Fidelity thresholds are sealed pre-outcome: original NLL absolute32eps32 +
relative256eps32 plus10×actual repeat noise; strict IDs and own-key/L4 invariance
exact; module mapping relative ceiling1e-4. Historical lenient gates are not
inherited. Algebraic module identity and final nonlinear NLL interaction are
separate quantities. Query-specific patching is not deployable repair.
