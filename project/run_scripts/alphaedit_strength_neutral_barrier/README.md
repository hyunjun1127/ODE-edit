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
