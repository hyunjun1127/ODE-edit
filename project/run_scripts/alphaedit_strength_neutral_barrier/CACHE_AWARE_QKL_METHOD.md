# Cache-aware one-sided predictive q-KL AlphaEdit

This package is an ODE-edit hook over an immutable stock EasyEdit checkout.
It keeps stock `compute_z`, layer key/residual refresh, projector `P`, history
cache `C`, L2 coefficient, cache append, and the exact N=1 upstream bypass.

## Formula-to-symbol map

| Mathematical object | Code symbol |
|---|---|
| `R`, `K`, `J=A^{-1}PK`, `M`, `F=RJ^T` | `geometry.factorize_alphaedit_velocity` / `NativeFactorization` |
| `M_energy`, `M_fast`, PSD/pinv receipt | `geometry.build_writer_metric` / `MetricReceipt` |
| `U=GJ=L(X^T J)` | `geometry.low_rank_pullback` |
| `R_tilde=R-[c]_+/eta U M^dagger` | `geometry.project_one_sided_velocity` |
| `W_(n+1)=W_n+h F_tilde` | `geometry.apply_velocity_` |
| native lookahead `W_n+hF` and exact restore | `writer._execute_guided` node loop |
| request-macro/token-mean target-excluded q-KL | `target_path.evaluate_values` and `TargetEventBatch.event_weights` |
| token-mean target NLL | `target_path.sequence_nll_by_request` |
| stock N=1 bypass | `writer.apply_strength_neutral_barrier_to_model` |

The predictive point is a gradient evaluation point, not an accepted native
candidate. There is no endpoint accept/fallback, KL budget, line search,
target-gradient equality, hard `DK=0`, or raw projector-Q correction.

## Arms

- `OFFICIAL_ALPHAEDIT`, N=1: calls stock EasyEdit in full.
- `STATIC_SPLIT_OFF`, N=2/4: factorized native velocity, no projection.
- `QKL_PROJECTED_ODE`, N=2/4: one-sided predictive projection.

For N>=2, `T=1`, `h=1/N`; `h` is applied exactly once at the write boundary.
The only controller inputs are canonical rewrite prompts, target sequences, and
their teacher-forced prefixes. Rephrase, locality, heldout, and target-true
observations have zero controller influence.

## Claim boundary

The code enforces the first-order lookahead-point condition
`<grad B(W_n+hF), F_tilde> <= 0` up to recorded floating-point residual. It does
not claim finite-step q-KL monotonicity, formal CBF invariance, guaranteed
locality, or equality to ODE-M.

## Runtime

`run_preflight.py` performs the focused synthetic and actual-model G0.
`run_experiment.py` runs five cells for `atomic-b1`, `atomic-b10`,
`atomic-b100`, or `sequential-b10x10`. Runtime identity is sealed by
`official-source-lock.json` and `firewall.py`; result roots are create-once.
