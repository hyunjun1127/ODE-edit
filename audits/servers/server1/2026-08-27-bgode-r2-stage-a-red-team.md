# BGODE-R2 Stage-A adversarial audit

Verdict: `STAGE_A_CPU_MATH_PASS / PRE_GPU_NOT_YET_RELEASED`.
Scientific promotion is false. Model/GPU/Slurm actions are zero.

## Locked boundary

The audit covers only the termination-free prefix-event reducer, FP64 event
moments, two-equality Plain/Fisher/Full numerical core, generic explicit Euler
utilities, telemetry schema, and the immediate ordered-AlphaEdit validation
guard. It does not claim a production runtime or a scientific endpoint.

| Gate | Adversarial question | Evidence | Result |
|---|---|---|---|
| Event topology | Are later divergence, first-token divergence, source-prefix, and target-prefix disjoint/exhaustive? | parameterized tiny-vocab streaming/reference tests | PASS |
| Termination | Can EOT/EOS/delimiter influence the event identity? | token-free schema and source test | PASS, count0 |
| Rare event | Does an approximately `exp(-160)` branch remain finite? | FP64 log-space test | PASS |
| Vocab mismatch | Is tokenizer vocabulary incorrectly assumed equal to output head? | sealed distinct sizes and invalid-ID rejection | PASS |
| Directional score | Do supplied forward tangents match independent JVP and central FD? | one forward-JVP and one FD fixture | PASS |
| Score moment | Does `sum pi S` center and is `G` symmetric PSD? | moment fixture | PASS |
| Two equalities | Do signs and pair-mass derivative follow from `A^T u=[1,0]`? | analytical/numerical fixture | PASS |
| Moving/anchor | Is optimizer equivalence used only under exact progress equality? | parallel-gradient and KKT solution identity | PASS |
| KKT | Does the eigenspace/Schur solution match a direct augmented solve? | full-rank fixture | PASS |
| Singular feasible | Is a null direction tolerated when `g,A` lie in the retained range? | rank-2-in-dimension-3 fixture | PASS |
| Equality infeasible | Is rank-one `A` rejected without fallback? | typed boundary and spectrum receipt | PASS |
| Range boundary | Is a direction outside `range(G)` a numerical implementation boundary? | typed boundary with full eigenvalues/residuals | PASS |
| Precision | Can an FP32 cutoff discard a valid direction retained by FP64? | fixed `diag(1,1e-8,0)` fixture | PASS |
| Finite step | Is coefficient exactly `h*u` with one FP32 physical cast? | controller API and dtype assertion | PASS |
| Partial step | Are only final schedule steps shortened? | `1.0/0.3 -> .3,.3,.3,.1` fixture | PASS |
| Frozen affine | Do one-step and split Euler agree for constant velocity? | 16-way split fixture | PASS within FP64 accumulation bound |
| Nonlinear flow | Does error decrease for `N=4,8,16,32`? | scalar nonlinear fixture | PASS |
| Telemetry | Are local and cumulative progress errors separate? | per-node records | PASS |
| AlphaEdit build | Is the approved validator called immediately after proposal? | injected call counter and factor hash receipt | PASS, call1 |
| Predictor mismatch | Are unit and effective prefixes distinct fields? | layerwise fixture | PASS |
| Forbidden mechanisms | Do dynamic modules contain scalar localizer/root solvers or endpoint forcing? | source scan | PASS, count0 |
| Prohibited numerical aids | Are floor/damping/ridge/fallback counters nonzero? | immutable boundary receipt | PASS, all0 |

## Red-team limitations retained

1. Tiny-vocabulary equivalence does not prove production tokenizer/model
   plumbing; that is a Stage-B/C technical gate.
2. The CPU fixture proves FP64 can preserve one deliberately small direction;
   it does not predeclare that a real Qwen spectrum will pass.
3. A production `range(G)` residual still requires full spectrum telemetry and
   remains `NUMERICAL_IMPLEMENTATION_BOUNDARY`; tolerances may not be tuned.
4. An equality-rank or Schur-rank failure remains
   `ACTUATOR_EQUALITY_INFEASIBLE`; no one-equality result may be substituted.
5. Euler convergence in a scalar toy does not establish model convergence or
   barrier efficacy.
6. Ordered predictor mismatch is measured, not eliminated by smaller outer
   steps.
7. No R1 report or result byte was changed.
