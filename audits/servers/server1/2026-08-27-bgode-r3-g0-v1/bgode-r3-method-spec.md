# BGODE-R3: fine-event target-excluded factor-space flow

Status: `R3_G0_MATHEMATICAL_PASS_G1_NOT_RUN` candidate. This specification
defines the CPU mathematical core only. Model load, GPU, Slurm, natural-sample
selection, live writing, endpoint efficacy, and scientific promotion are not
performed by G0.

BGODE-R3 is a new namespace. It does not alter any BGODE-R1 or BGODE-R2 source,
result, audit, or raw byte. In particular, R2 Stage C remains a factual 4/6
topology result: the two prefix cells per model passed a degenerate
three-event/rank-two geometry, while both unequal non-prefix cells stopped at
the old explicit `range(G)` controller boundary. Those facts are motivation,
not evidence for R3.

## 1. Fixed scientific boundary

- Atomic request batch size is exactly one.
- The ODE state is physical `W(t)`.
- Official AlphaEdit computes the target `z*` once at entry `W0`; recomputation
  count is zero.
- The state-refreshed, native-derived ordered AlphaEdit dictionary uses the
  genuine `P`-inside solve, pinned projector/statistics, and entry history.
- History appends zero times inside nodes and at most once after an accepted
  terminal transaction.
- Controller inputs contain only the edit request, fixed target, model state,
  and the canonical event partition. Held-out/locality/evaluator outcomes have
  decision influence zero.
- The flow has no scalar endpoint localizer, endpoint correction, accepted-step
  search, alternate solver, or hidden retry.

The allowed formal claim is an instantaneous target-excluded KL tracking
potential in normalized actuator coordinates. Classical CBF forward
invariance, finite-step monotonicity, exact native AlphaEdit trajectory,
global locality, and `ODE-BF Full` are not claimed.

## 2. One canonical prefix-resolving event identity

For distinct non-empty token paths `S` and `Y`, build their two-path trie. At
each internal prefix `a`, each vocabulary token outside the trie children is a
first-departure atomic event `(a,v)` with

`pi_(a,v)=P_W(a) P_W(v|a)`.

There is no EOS, EOT, delimiter, or length normalization. Tokenizer and output
head vocabulary sizes are sealed independently; every distinguished path token
must exist in both, while collateral event rows may span the output vocabulary.

The topology convention is fixed per request/model and is consumed unchanged
by normalization, score centering, Fisher, the W0 conditional, KL, and the
equalities:

1. Unequal non-prefix or common-prefix divergence: source and target leaf
   cylinders are one distinguished macro each; every other departure is
   atomic.
2. Source-prefix-target: target leaf is one target event. At the source prefix,
   every token except the target continuation remains an atomic event in the
   barrier/Fisher/W0 conditional. Those atomic rows are probability-weighted
   together only when constructing the source equality score.
3. Target-prefix-source: the tokens at the target prefix except the source
   continuation collapse to one target-exclusive macro; the source leaf is one
   source event; other departures remain atomic.

The shorter prefix object is called a `prefix-resolved exclusive source event`
or `prefix-resolved exclusive target event`, never an exact short completion.

Production-shaped layouts store numeric prefix/token/mode tensors. They do not
allocate a Python object per vocabulary token. Logits and directional logits
arrive in FP32 and are reduced in FP64 log space. Underflow, overflow, or a
nonfinite reduction is a typed numerical boundary; probability floors are not
used.

For event score `s_e in R^L`, the reducer verifies

`sum_e pi_e = 1`, `sum_e pi_e s_e = 0`, and

`G_fine = sum_e pi_e s_e s_e^T`.

Collapsing a collateral set `C` loses exactly

`G_fine - G_aggregate = p_C Cov(s|C) >= 0`.

## 3. Immutable W0 conditional and target-excluded reference

Let the single collapsed target event be `Y`, `p=pi(Y)`, and

`q_e=pi_e/(1-p)` for `e != Y`.

`q0` is captured once from the same fine event identity at W0. It is never
recomputed from a live node and never replaced by a source aggregate. Its
tensor bytes, shape, and event identity have a create-once artifact interface;
only hashes and shapes are intended for Git.

The reference is the target-only exponential tilt

`pi*_t(e) = pi_0(e) exp(t 1[e=Y]) / (1-p0+p0 exp(t))`.

Thus

`p*(t)=sigmoid(logit(p0)+t)` and

`pi*_t(e)=(1-p*(t)) q0_e` for `e != Y`.

The potential is

`B_t(W)=KL(pi*_t || pi_W)`

with exact decomposition

`B_t = KL(Ber(p*) || Ber(pY(W))) + (1-p*) KL(q0 || qW)`.

The anchored excess identity is

`KL(pi0||piW)-KL(Ber(p0)||Ber(pY(W)))=(1-p0)KL(q0||qW)`.

The factor-space objective uses a log-space construction:

`X = diag(sqrt(pi)) S`,

`y_e = -pi*_e/sqrt(pi_e) = -exp(log pi*_e - .5 log pi_e)`.

Then `X^T y=g_t=-sum_e pi*_e s_e`. Conversion of a mathematically positive
probability to zero is a typed numerical boundary rather than a floored value.

## 4. Target-excluded equalities

At a live node define

`mu_N=sum_(e!=Y) q_e s_e`.

`sY` is the target macro score. `sS` is the source leaf score, or in the
source-prefix topology the current-probability weighted score of the atomic
source-exclusive rows. Define

`c_eta=sY-mu_N`, `c_S=sS-mu_N`, `C=[c_eta,c_S]`, `d=[1,0]`.

The controller enforces `C^T u=d`. With score centering this implies

`sY^T u=1-pY`, `mu_N^T u=-pY`, `sS^T u=-pY`,

and therefore

`dot pY=pY(1-pY)>0`, `dot pS=-pY pS<0`.

The corresponding continuous-coordinate solution is

`pY(t)=sigmoid(logit(pY0)+t)` and

`pS(t)=((1-pY(t))/(1-pY0)) pS0`.

Rank two is not mandatory. A zero/redundant second equality is valid when the
complete augmented system is consistent. This is the same two-equality
definition, not a one-equality mode switch.

## 5. Normalized AlphaEdit actuators

For each layer block, compute the exact factor-Gram norm in FP64:

`alpha_j=||B_j||_F`, `Bhat_j=B_j/alpha_j`.

Zero `alpha_j` is a typed actuator boundary. The same normalized basis is used
for JVPs, event scores, equalities, controllers, and physical writes. With
`beta=h u`, writing raw factors uses coefficient `beta_j/alpha_j`. Because the
layers are disjoint parameter blocks, `||beta||_2` equals the block-Frobenius
norm of the physical normalized write (up to the audited FP32 representation).

The adapter retains its legacy public behavior. R3 adds only a lightweight
per-node build ledger and release policy. Proposal validation occurs once
immediately after the ordered build; node factors/JVP buffers have trajectory
retention count zero after the node.

Ordered factors are a state-refreshed native-derived dictionary, not the exact
coefficient-conditioned writer. If native construction temporarily used unit
raw coefficients while R3 applied normalized `beta`, the layer-j prefix
mismatch is computed from the actual `beta`:

- unit prefix norm `||alpha_<j||_2`,
- applied prefix norm `||beta_<j||_2`,
- mismatch `||beta_<j-alpha_<j||_2`.

Subdivision does not remove this internal unit-prefix approximation.

## 6. Factor-space controller

The three future arms share the same normalized `C,d`:

- Plain: minimum normalized-coordinate Euclidean norm.
- Fisher: minimize `.5 ||Xu||^2`.
- Full: minimize `.5 ||Xu+y||^2`.

Production does not eigendecompose or invert `G=X^T X`. It first checks the
complete equality consistency with a column-equilibrated SVD of `C^T`. The
outcome-independent rank rule is

`rcond=max(m,n) eps_FP32`.

Raw and equilibrated singular spectra, thresholds, retained ranks, and
consistency residuals are recorded. A consistent rank-one system is allowed;
an inconsistent complete system is `EqualityInfeasible`.

Using a minimum-norm particular point `up` and orthonormal null basis `Z`, R3
solves the objective directly from the SVD of `XZ`. It records the reduced
spectrum and verifies equality and null-space stationarity with deterministic
backward-error bounds. Retained actuator directions are exposed for symmetric
FD/JVP validation. The small Gram is observation-only diagnostics.

At `t=0`, `pi*=pi`, score centering gives `g=0`, so Full and Fisher must agree.
At any node, for `delta=uB-uF`, the implementation records

`g^T uB = g^T uF - delta^T G delta <= g^T uF`

and its numerical residual. This is a local attribution identity, not a
finite-step monotonicity theorem.

## 7. Rho-free Euler and staging

For horizon `T` and node count `N`, `h_n=min(T/N,T-t_n)`,

`beta_n=h_n u_n`, and

`W_(n+1)=W_n + sum_j beta_(n,j) Bhat_(n,j)`.

Dictionary, prefix logits/JVPs, fine events, reference moments, equalities, and
controller are refreshed at every node. The method asserts differential laws
only: local error is `O(h^2)` and global error is `O(h)`. Dense/block physical
weight differences, rather than sums of changing dictionary coefficients, are
the displacement authority.

G0 is CPU-only. G1 will select the earliest natural request per topology using
only sealed canonical order and exact rewrite-prompt tokenization. G2 is held
until each model's natural unequal-nonprefix G1 gate passes and a separate
pre-GPU Cauchy lock seals successive physical endpoint, target-logit, and q-KL
criteria. The future fixed-strength curve is predeclared as
`T={0.5,1,2,3,5}, N=32`; it is not run or used for selection before G2
convergence.

`scientific_promotion=false` throughout these stages.
