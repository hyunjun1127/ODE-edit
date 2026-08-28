# AlphaEdit strength-neutral barrier hook

This experiment is implemented in ODE-edit and imports an unmodified stock
EasyEdit checkout at runtime.  It does not patch, vendor, or replace any
EasyEdit source file.

The explicit `N=1` arm calls upstream `apply_AlphaEdit_to_model`.  The `N>=2`
split and barrier arms reuse upstream `compute_z`, `compute_ks`, activation
capture, projector/cache state and shape matching through a thin hook.  Native
target/key/solve state is computed once per layer; each Euler node uses a
native predictor, observes the target-excluded barrier at that predictor,
restores the node entry, and applies the guided predictor-corrector step.

Runtime source identity is sealed by `official-source-lock.json` and checked by
`firewall.py`.  A tracked modification or byte mismatch in the external
EasyEdit checkout fails before model execution.

## Cross-server launch inputs

Both Slurm launchers take the ODE checkout, stock EasyEdit checkout and exact
HEADs through `ODE_SNB_SOURCE_ROOT`, `ODE_SNB_EXPECTED_HEAD`,
`ODE_SNB_EASYEDIT_ROOT`, and `ODE_SNB_EXPECTED_EASYEDIT_HEAD`.  A different
server may additionally bind its local paths with `ODE_SNB_ENV_PROJECT`,
`ODE_SNB_MODEL_PATH`, `ODE_SNB_DATASET`, and `ODE_SNB_PROJECTOR`.  Set
`ODE_SNB_EXPECTED_HOST` when a host lock is required; leaving it unset permits
the scheduler-selected host.  Result roots remain create-once.

Run order is strict: preflight G0 first, then `atomic-b1`, then `atomic-b10`.
The sequential stage stays unsubmitted until those gates are accepted.
