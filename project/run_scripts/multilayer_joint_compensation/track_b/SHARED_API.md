# Shared API v1 — SH2 → SH1

Authority: design SHA a8f32937ddd2fc4f87cb3a2cfb9ef95de67d2fd39ce877142fa7b80a1884bf05.
Owners: SH2 functional/linear_solve/elastic_qp; SH1 full-model callback, fixed
native geometry, contracts, fixture/history/banks/evaluation and package init.

## Coordinates and nonmutation

`WeightTree = tuple[Tensor, ...]`, ordered support `(4,8)`, `(8,)` or `(4,)`.
Each block retains full physical output×input shape. `FunctionalPanel` receives
absolute selected weights; linear operator / solver receives displacements in
those same blocks. Caller provides the complete current fixed model state for
the other parameters. No scalar-direction/rank compression is performed.
All selected weights affect the full token sequence in `logits_fn(weights)`;
the callback returns only flattened prediction positions `[T,V]`. A must retain
the true L4→L8 input dependence. Pure callbacks must not assign weights, mutate
cache/history, call optimizers, or create `requires_grad_` inside a transform.

Scalar vector inner products / 2D dual algebra are FP64. Model/vector storage is
unchanged. The full-space permitted projector P*, native metric/preconditioner
are caller-provided fixed-We operators. Apply projection consistently to the
RHS, K input/output and preconditioner: kernel does not silently project an
unrestricted answer after solving. No per-layer Current equality.

## Functional API

`OutputBatch(logits_fn, target_ids, context_index, token_mean_weights,
context_weights, teacher_logp, reference_nll, identity='', input_tokens=0)`.

- Target IDs / context index: int64 `[T]`; index local contiguous `[0,C)`.
- Token weights: `[T]`, exactly 1/target-length within context.
- Context weights: `[C]`, GLOBAL `1/(B * contexts_in_request)`, never normalized
  independently per physical chunk. Sum across panel is one. `[]` is B=0.
- Teacher logp `[T,V]` / reference NLL `[C]` are detached, fixed and local-only.
- Base/Past teacher/reference is We; B Current reference is WN; A Current
  reference is its prescribed fixed nominal path, not the previous iterate.
- Each tensor must be on the device compatible with the callback's logits.

`FunctionalPanel(batches, role, tau=.1)` with role `base`, `past`, `current`:

- `.linearize(weights, need_nll_gradient=False) -> PanelLinearization`
  fields `value`, `mean_nll`, `gradient`, `nll_gradient`, `context_rows`, `ledger`.
  Request `need_nll_gradient=True` only for Current. It performs a second
  backward on the same forward graph and records that cost. No duplicate
  evaluator forward is hidden. `context_rows` is local-only raw data.
- `.value(weights) -> float` actual scalar risk/profile, no-grad, counts forward.
- `.observe(weights) -> {value, mean_nll, context_rows}` shares the same single
  no-grad forward for risk/profile and NLL; used for Frozen RHS/post-node facts.
- `.ggn(weights, direction) -> WeightTree`: full-model JVP and VJP. Two actual
  model forwards per microbatch (one reverse graph, one forward AD), explicitly
  counted; no explicit Jacobian/Hessian or functional cross-block removal.
- `.counts` cumulative logits-forward/input-token, gradient-backward,
  JVP/VJP call/vector and GGN-matvec counts. Does not count teacher capture or
  evaluation performed outside the panel; caller records those separately.

Base uses teacher KL and CE Fisher. Past uses psi'' NLL-Jacobian outer PLUS
psi' CE Fisher. Current uses KL Fisher/tauE + NLL-profile outer/tauE²;
residual×NLL-Hessian is not part of its PSD profile GGN. At psi's two C1
breakpoints the outside-region second derivative is used and must be recorded.
Finite difference of the full nonlinear weight Hessian is NOT a GGN oracle.

## Assembly contract

Caller forms normalized gB=Base.gradient/sigmaB and gP=Past.gradient/sigmaP,
sigma=max(native-WN-risk,1e-3), fixed for the batch. Risks/budgets passed to the
elastic solver are likewise normalized. K = .01/h*native_metric +
2/sigmaB*BaseGGN + 1/sigmaP*PastGGN + 1/h*CurrentGGN + A-balance curvature.
u = 2*gB+gP+Current.profile_gradient/h + A-balance cumulative gradient.
a = Current.mean_NLL_gradient. B has no balance or nominal increment.
Frozen freezes first-predictor derivative weights and linear terms, not actual
risks, tE or budgets. Absolute weight snapshots used for frozen callbacks must
remain immutable; the kernel does not make a hidden full-model clone.

## PCG and elastic API

`pcg(operator, rhs, precondition=None, rtol=1e-4, maxiter=20) -> PCGResult`.
`solution`, `status`, `iterations`, `relative_residual`, `absolute_residual`,
`rhs_norm`, `matvec_calls`, `precondition_calls`, `recursive_residuals`;
`.receipt()` excludes tensor solution. Zero RHS costs zero operator calls.
Final TRUE residual costs one extra matvec; finite nonconvergence is
`APPROXIMATE_PCG_NONCONVERGENCE`, not a capacity failure or solver PASS.
Nonfinite/nonpositive curvature/preconditioner throws a typed technical error.

`solve_constrained(operator, u, a, t_e, *, precondition=None,
gradients=None, risks=None, budgets=None, h=1., rho=(2.,1.),
rtol=1e-4, maxiter=20) -> ElasticResult`.

OS: `gradients=None`, h=1, tE=0. BF: gradients=(gB,gP), h=.25,
risks=(fB,fP), budgets=(betaB,betaP), scalar tE per fixed reference.
Result fields: `correction`, `equality_correction`, CPU FP64 `dual`, `slack`,
`diagnostics`. Apply returned correction **once**, without multiplying by h
again; it is the finite correction C in the design's quadratic. A's predictor
already includes h*D0 before this call.

Diagnostics includes per-RHS PCG receipts; Current equality/dual; actual KKT
stationarity; primal, slack-stationarity and complementarity residuals; exact
nonnegative dual domain; raw risk Gram and its antisymmetry from independent
approximate PCG solves. The scalar quadratic uses that Gram's symmetric part;
antisymmetry and actual original-operator KKT residuals are not hidden.
Tiny 2×2 SPD solve enumerates all four active sets, no negative dual admission
or clipping. CPU fixture's dense oracle is test-only, not production code.
Exactly zero a with nonzero tE yields
`CURRENT_DEFICIT_FIRST_ORDER_UNRECOVERABLE` while retaining Current curvature.
No near-zero clipping, adaptive retry, endpoint replacement or performance gate.

## Verification scope

Run: `/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m unittest discover -s
project/run_scripts/multilayer_joint_compensation/tests -v`.
Initial 14 CPU tests cover all four dual active sets against an independent
primal oracle, no negative dual, PCG true residuals, zero/approximate paths,
full cross-block GGN, Past two-term action, global mean physical1/2/4 and FD
of the actual objective gradient. Actual model/packing/FP32 forward and
checkpoint restoration are separate required gates; not claimed by CPU PASS.

## B path adapter v1

`track_b.protocol.BProblem` owns immutable WN selected weights, support, fixed
Base/Past/Current panels, fixed native operator callbacks, native risks,
Current-reference mean NLL, validate_state callback and fixture identity.
`run(problem, arm, on_node=None)` returns final FP32 blocks and local-only rows.
Arms are B-OS/B-BF4/B-Frozen-BF4 and explicitly named N4-L4-FUNCTIONAL-OS.
The inner live model remains frozen under JointView. All inner states are
full-sequence functional FP32 selected weights. Caller materializes the final
blocks and validates actual byte/forward parity before claiming an endpoint.
No hidden writer/history append in inner callbacks.

`track_b.native_adapter.operators(geometries)` adapts SH1 full-P* FP64 native
geometry actions back to each input PCG vector's dtype; normalization remains
trace(S)/din and factor/contraction internals remain FP64. This explicit vector
storage boundary prevents FP64 native actions from silently producing FP64
model weights. It is neither low-rank approximation nor a BF16 conversion.

Operational override policy ODEEDIT-INITIAL-GATE-ONLY-USER-RECALL-20260911 applies:
after minimum actual initial-valid, agent stops; no autonomous terminal polling,
follow-up submissions, analysis/main integration before explicit USER recall.
