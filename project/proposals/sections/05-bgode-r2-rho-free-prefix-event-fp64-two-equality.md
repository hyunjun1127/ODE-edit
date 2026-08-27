# BGODE-R2: rho-free prefix-event FP64 two-equality flow

Status: Stage-A CPU mathematical closure. No model, GPU, Slurm, or scientific
endpoint is produced by this document. `scientific_promotion=false`.

This document supersedes the interpretation of BGODE-R1 without changing any
R1 source, result, or report byte. The R2 method intentionally removes the
scalar finite-step endpoint localizer. It also replaces termination-dependent
events with a termination-free prefix partition and promotes normalized pair
mass preservation to a second equality.

## 1. Immutable provenance and interpretation

- Fixed native target: Official AlphaEdit `compute_z(W0, R)` exactly once;
  recomputation count is zero.
- State: the physical model weights `W(t)`.
- Actuators: the existing ordered, genuine `P`-inside AlphaEdit factors.
  Each dictionary build is immediately passed to
  `validate_ordered_alphaedit_build`; no post-hoc projection is accepted.
- History: entry snapshot at every dynamic node, append zero inside the flow,
  and append once only after an accepted terminal transaction.
- R1 Llama immutable interpretation:
  `TECHNICAL_PASS_COARSE_RHO_FREE_EULER / SCIENTIFIC_HOLD_NO_NUMERICAL_CONVERGENCE_AND_INVALID_EOT_EVENT_SEMANTICS`.
- R1 Qwen immutable interpretation:
  `EXACT_ALPHAEDIT_ADAPTER_PASS / NUMERICAL_CONTROLLER_HOLD_FP32_RANGE_G`.

The old packages remain factual historical artifacts. R2 does not rewrite
them or inherit the old Qwen range result as scientific infeasibility.

## 2. Termination-free event partition

Let `S=(s1,...,sp)` and `Y=(y1,...,yq)` be distinct non-empty token paths.
There is no EOT, EOS, delimiter, length normalization, or performance-driven
termination selection.

At every internal prefix in the two-path trie, one collateral event aggregates
all next tokens outside the set of trie children. For a distinguished path
that is not a prefix of the other, its event is its finite cylinder. If `S` is
a prefix of `Y`, the source event is the `S` cylinder with the next target
continuation excluded; the target event is the complete `Y` cylinder. The
opposite prefix relation is symmetric. These events are disjoint and exhaust
the infinite continuation space.

The implementation stores one aggregate event per trie prefix rather than one
Python object per vocabulary token. Excluded-token mass is evaluated with
FP64 log-sum-exp. Tokenizer and output-head vocabulary sizes are recorded
separately and may differ; all distinguished token IDs must be valid in both.

For each event `e` and actuator `j`, the supplied serial forward-mode tangent
defines

`S[e,j] = D log pi_W(e)[B_j]`.

The reducer consumes FP32 logits/tangents, casts them immediately to FP64, and
returns FP64 log probabilities and event scores. It verifies normalization
with a dtype/dimension/scale backward-error bound and verifies score centering
`sum_e pi(e) S_e ~= 0`.

## 3. Reference and two equalities

Let `pY`, `pS`, and `c=pY+pS`. The reference changes only the conditional
odds inside the distinguished pair:

`r(t)=r0+t`, where `r=log pY-log pS`, while the initial pair mass and all
collateral event probabilities remain fixed.

At a node:

- progress direction `a = S_Y-S_S`;
- normalized pair-mass direction
  `b = (pY/c) S_Y + (pS/c) S_S`;
- equality matrix `A=[a,b]` and rates `d=[1,0]`.

Therefore `A^T u=d` gives the local statements

`d r/dt = 1`, `d log c/dt = 0`,

and hence

`dpY/dt = pY*pS/c > 0`, `dpS/dt = -pY*pS/c < 0`.

These are local differential identities. R2 makes no exact finite-step
progress or exact finite-step pair-mass claim.

## 4. FP64 controllers

With event probabilities and scores,

`G = sum_e pi(e) S_e S_e^T`

is symmetrized and tested as an FP64 PSD Fisher pullback. The moving-barrier
gradient is

`g_t = -sum_e pi*_t(e) S_e`.

The main controller is

`u = argmin_v g_t^T v + 1/2 v^T G v  subject to A^T v=d`.

Fisher-only uses `g=0`. Plain is not an ambiguous equal direction: it is the
Euclidean minimum-norm solution

`u_plain = A (A^T A)^dagger d`.

All three apply exactly the same two-equality rank and residual boundary.
With an eigendecomposition of `G`, the controller:

1. derives its cutoff only from FP64 epsilon, dimension, and matrix norm;
2. checks `g` and both columns of `A` lie in `range(G)`;
3. checks the equality matrix and Schur matrix have rank two;
4. solves the singular-aware KKT system;
5. validates equality and stationarity residuals against backward-error
   bounds and records the complete `G` and Schur spectra.

An FP64 range failure is `NUMERICAL_IMPLEMENTATION_BOUNDARY`. Genuine equality
or Schur rank deficiency is `ACTUATOR_EQUALITY_INFEASIBLE`. Ridge, damping,
probability floors, a one-equality fallback, or tuned cutoff are forbidden.

## 5. Finite-step flow and accounting

At node `n`, `beta_n = h_n u_n`. Only the last step may be shorter than the
configured maximum. The physical write boundary casts `beta_n` from FP64 to
FP32 exactly once; all model forwards/JVPs and physical weights remain FP32.

The flow records local progress error

`(r_{n+1}-r_n)-h_n`

and cumulative progress error

`(r_{n+1}-r_0)-sum_{m<=n} h_m`

separately. It also records entry/exit moving and anchored barrier terms,
pair-mass drift, outside conditional KL, live event entry versus controller
reference event, terminal displacement, path length, integrated kinetic
energy `sum ||Delta W_n||^2/h_n`, and legacy `sum ||Delta W_n||^2`.

No endpoint is forced onto the reference path. Numerical convergence is an
empirical `N={4,8,16,32}` question for later stages.

## 6. Ordered-dictionary predictor mismatch

Every proposed ordered dictionary is validated immediately. For each layer,
R2 distinguishes the unit-prefix state used by the native ordered factor build
from the effective prefix implied by current coefficients. Both mismatch
sequences are observation-only. Step subdivision does not erase this internal
predictor approximation, and predictor-corrector is out of scope.

## 7. Stage-A closure and nonclaims

The focused CPU suite covers four topology cases, rare events, vocabulary
mismatch, forward-mode JVP and finite-difference score identities, score
centering, Fisher/KKT identities, pair-mass signs, singular boundaries, a
FP32-fail/FP64-pass fixture, affine split identity, nonlinear Euler
convergence, ordered-build validation, and forbidden dynamic symbols.

R2 does not claim classical CBF invariance, monotone barrier decrease, exact
finite-step progress, exact native ordered writer equivalence, standard
locality safety, scientific efficacy, or promotion. Model runtime and all GPU
results require the subsequent staged gates.
