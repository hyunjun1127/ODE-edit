# CAKE native fixed10k lifelong

Upstream `zjh-vinky/CAKE` c8243e1d7e43ca9cf64d552f96221fcb9561aac2,
MIT. This wrapper calls its original `apply_Cake_to_model` once per B100.
It does not reimplement its solve, final-L8 targets, causal allocation or history.
The execution copy removes the unused missing `notebooks.util` import and adds
a final LF. The remaining native AST is unchanged; original bytes are preserved.

Latest direct user override: “CAKE 부분은 weight 저장 하지 말고 그냥 올려라”.
Persistent W/M tensor checkpoints are disabled. Five-layer W/M continue in memory,
with batch-entry in-memory rollback. State hashes and RNG/context metadata are NOT
a reconstructable checkpoint. Termination requires a new explicit execution decision;
there is no automatic replay/requeue. Existing checkpoints elsewhere are untouched.

Evaluation is the original server4 canonical MB16 evaluator, not CAKE's final-only
CLI or its GLUE evaluation. Each batch has current R/P/N and all-seen rewrite;
full R/P/N at B1,5,10,20,...,100. Same-state current rows are reused in full-seen.
No W0 GPU evaluation; reuse sealed42673. Dataset is the official immutable fixed10k.

Source preparation:

```bash
python3 -B -m project.run_scripts.cake_native_lifelong.preflight --repo "$PWD" --output /data/janghj/ODE-edit/local/cake-native-lifelong/20260915-v1/attempt-v1/preparation-v1
```

After source commit, `prepare_execution --repo ... --preparation .../preparation.json`
creates a new local source closure/lock. `run.sbatch TASK_ROOT` is held/inspected/released
only after fresh cap2, explicit60416M, paths, identities and disk checks. One1GPU chain;
48h walltime is not a GPU-hour hardcap. No separate GPU smoke or FD/ULP gate.
CPU fixtures check config/storage/route; they do not establish Llama execution validity.

Plan budget: 16GiB raw/source/log reserve, no saved W/M/target tensors. 10000 native
targets, at most250000 loss evaluations/240000 Adam updates, 500 solves, 100 whole
history passes=500 layer appends. Actual counters/timing remain output facts, not
forecasts. Native target/keys/solve timings are nested in write wall and not additive.
Publications contain only source, counts, hashes and paths; raw NLL/logs remain local.

After PENDING or first actual native batch observation, agent monitoring pauses.
The persistent100batch program continues without agent callbacks. No postterminal
analysis or new run until explicit user recall.
