# AlphaEdit direct-z paired possibility diagnostic — Qwen p0 v1

## Decision

`dzf_alpha_qwen_p0_v1` on `qwen2.5-7b-inst` is a technically valid,
eight-case atomic mechanism diagnostic.  Its locked analyzer verdict is
`PAIRED_ALPHA_SIGNALS_REPORTED_NO_SUPERIORITY_DECISION`: all 8/8 cases passed,
all 64 planned outcomes are present, and the analysis ITD is 8.

The main positive result is narrow but useful.  At the same endpoint C-energy,
the four-hop genuine-Alpha BF path improves over the static genuine-Alpha
C-matched write on direct-z residual by `+0.08233046` in 6/8 cases and on
output-NLL reduction by `+0.05692602` in 7/8 cases.  It also has a smaller
low-rank endpoint Frobenius norm by `0.10153338` in 7/8 cases.  These three
axes pass the precommitted lenient gate of positive paired mean plus at least
5/8 positive cases.  The representation signal is accompanied by better
generated-context error, rewrite progress, exact margin, and paraphrase NLL.

It is not a preservation result.  BF worsens the higher-is-better preservation
score by `-0.20097198` (only 1/8 favorable), increases off-token spill by
`0.01056363` (only 2/8 favorable), and costs 24 NFE/case rather than 5 for the
static native write.  Therefore the defensible Qwen conclusion is **local
state-refresh signal at fixed C-budget, with an explicit preservation/spill/
compute trade-off**.  It is not Alpha-wide or ODE-Edit method superiority,
and it says nothing about lifelong retention or sequential-collapse
prevention.

All effect tables below use the analyzer's sign-normalized convention:
positive means that the left-hand arm is better on that individual axis.
Counts are positive cases out of the fixed ITD denominator 8.  CIs are
descriptive; they are not part of the lenient gate.

## Fixed execution and technical contract

- The manifest fixes the eight arms in the required order and the same eight
  cases used by paired MEMIT-v2: `21100, 1477, 20838, 18707, 14288, 16426,
  19041, 17609`.  Configuration SHA-256 is
  `ac3630a1fbb47f4b141d39f130b8a7d1d80722f16f6271c2615afa5fb57008b7`;
  selection SHA-256 is
  `12393c041fdcea6b06b15ccb4b83b9bc23a7f611fa317e4a5f3f213687bb1d5c`.
- Frozen direct-z replay is anchored to `dzf_qwen_p0_v2`, source-manifest
  SHA-256
  `1df5dbb1cf69679cf302938f270d2bb94f2cdd324903b8f930b986b477e88a7b`,
  and replay lock
  `8ee67660c167233e95f77cc170011eccf1526d08bd1337f5fd6da612db641f6e`.
  Each target was loaded exactly once and recomputed zero times.
- The pinned Qwen Alpha configuration and precomputed read-only projector were
  used.  The run records zero isolated `cache_c` for every case, no covariance
  or projector recomputation, and exact projector byte integrity.
- Per case, the event ledger records one genuine ordered Alpha build, four
  genuine synchronous BF builds, one post-hoc ordered build, four BF hops,
  eight outcomes, and one durable commitment.  All eight events pass frozen
  target/cross-solver anchor checks, genuine/post-hoc provenance checks,
  firewall, receipt-before-outcome, and exact W0 rollback.
- Eight exclusive receipts bind the same arm order.  Every receipt records
  feature/action flush and `fsync` before receipt creation and binds the actual
  BF evaluation endpoint, including raw path energy, normalization scale,
  direction hash, and final C-energy.
- Genuine C-matched, post-hoc C-matched, and BF C-energies match their paired
  MEMIT reference within contract tolerance; the cone stays within the same
  cap.  The maximum observed low-rank Alpha-system residual is
  `2.97499374e-7`, well below `5e-4`.  All eight cone solves converged and none
  was the special zero-proposal case.
- Projector violation is an outcome rather than a validity gate.  The largest
  arm-level maximum is `5.37369466e-6`, so the result demonstrates numerical
  containment only to that measured tolerance, not downstream preservation.
- Slurm job `15755`, Qwen child step `15755.1`, completed with exit `0:0` in
  `02:14:18`; `MaxRSS` was `8,904,392K`.  This is execution provenance, not an
  effect.

The raw manifest, summary, and outcome SHA-256 values independently reproduce
the analyzer's source hashes:
`ecc9a7bd3b5bd2b45e21ab4a230dceac54dc5cdbd5a506dee1e87fb894df078a`,
`ffc0e593f0301c4f78a525d975860288f685eda3da3a36dfa149b3025e94a01d`,
and `7e26b878b497ba319b859f18da1984ea6f98d21b6d79be6395ea0e279a43cba6`.
The deterministic analysis SHA-256 is
`5c4676adec7de5e0fa5495842702fd991b8f93251b4d95c312490d1340756c1e`.

## What “genuine Alpha” tests

For each layer, the isolated first-edit genuine Alpha update is

\[
U=PK,\qquad M=I+K^TU,\qquad
\Delta W_{\mathrm{genuine}}=R M^{-T}U^T.
\]

The post-hoc ablation instead first solves with identity projection and only
then right-projects the chosen update,

\[
\Delta W_{\mathrm{post}}=
R(I+K^TK)^{-T}K^T P.
\]

These expressions need not commute: genuine Alpha places `P` in both the
coefficient system and right-hand side, whereas post-hoc projection does not.
The raw feature records confirm that this is a non-degenerate distinction.
Across the five edited weights, the mean genuine/post-hoc update cosine is
`0.92172815`, while their mean Frobenius difference relative to the genuine
reference is `0.40522219`.  Thus any empirical difference below is genuinely
about solve geometry, not two names for the same endpoint.  It still does not
make either construction universally preferable.

## Absolute direct-z write: native Alpha and BF

Both deployable C-matched writes materially realize the frozen direct-z target
relative to no-op and satisfy the requested exact target in all eight cases.
The table includes all representation, edit, preservation, spill, update-cost,
containment, and compute axes needed to see the trade-off.

| Axis (left arm versus no-op) | Genuine Alpha C-match: mean; +/8 | Genuine-Alpha BF K=4: mean; +/8 |
|---|---:|---:|
| Direct-z residual improvement | `+0.46528274`; 8/8 | `+0.54761319`; 8/8 |
| Canonical delta gain | `+0.49422100`; 8/8 | `+0.63198208`; 8/8 |
| Canonical delta cosine | `+0.94135040`; 8/8 | `+0.92723035`; 8/8 |
| Generated mean-error improvement | `+0.41738468`; 8/8 | `+0.52271576`; 8/8 |
| Generated worst-error improvement | `+0.38605860`; 8/8 | `+0.49305430`; 8/8 |
| Output-NLL reduction | `+10.45983613`; 8/8 | `+10.51676214`; 8/8 |
| Rewrite output progress | `+13.92994156`; 8/8 | `+14.64727440`; 8/8 |
| Exact target margin | `+14.18361241`; 8/8 | `+14.90717953`; 8/8 |
| Exact target satisfied | `+1.00000000`; 8/8 | `+1.00000000`; 8/8 |
| Paraphrase-NLL reduction | `+8.55274400`; 8/8 | `+9.43366311`; 8/8 |
| Preservation score | `-0.92632730`; 0/8 | `-1.12729928`; 0/8 |
| Off-token spill (lower is better) | `-0.04357059`; 0/8 | `-0.05413422`; 0/8 |
| Endpoint C-energy (lower is better) | `-0.06375028`; 0/8 | `-0.06375028`; 0/8 |
| Endpoint Frobenius norm (lower is better) | `-2.57325908`; 0/8 | `-2.47172570`; 0/8 |
| Projector violation (lower is better) | `-5.1366e-6`; 0/8 | `-5.1390e-6`; 0/8 |
| NFE relative to no-op (lower is better) | `-4`; 0/8 | `-23`; 0/8 |

The absolute-write finding is therefore real but not self-justifying.  Native
Alpha removes `46.5%` of the no-op normalized residual on average, and BF
removes `54.8%`; both produce strong edit proxies.  Yet both move held-out
behavior away from W0 and incur nonzero sequence spill.  Writing direct-z more
faithfully is not, by itself, evidence of better preservation.

## Genuine solve versus post-hoc projection at matched C

At the same C-energy and NFE, post-hoc projection is better on Qwen target and
edit axes, while genuine Alpha is better on preservation and spill.  The left
arm in this table is genuine Alpha C-match and the right arm is post-hoc
C-match.

| Axis | Sign-normalized mean; +/8 | Reading |
|---|---:|---|
| Direct-z residual | `-0.09317304`; 0/8 | Post-hoc is closer to direct-z in every case. |
| Canonical delta gain | `-0.09336335`; 0/8 | Post-hoc has the larger target-directed gain. |
| Canonical delta cosine | `-0.02618068`; 0/8 | Post-hoc has the better alignment. |
| Generated mean error | `-0.08737324`; 0/8 | Post-hoc is better in every case. |
| Generated worst error | `-0.08386899`; 0/8 | Post-hoc is better in every case. |
| Output-NLL reduction | `-0.04695102`; 0/8 | Post-hoc has the larger reduction. |
| Rewrite output progress | `-0.79576240`; 0/8 | Post-hoc is better in every case. |
| Exact target margin | `-0.64181018`; 1/8 | Post-hoc generally has more margin. |
| Exact target satisfied | `0`; 0/8 | Tie: both satisfy all eight cases. |
| Paraphrase-NLL reduction | `-0.57513455`; 1/8 | Post-hoc is better in 7/8. |
| Preservation score | `+0.09821588`; 7/8 | Genuine Alpha preserves held-out behavior better. |
| Off-token spill | `+0.00322912`; 6/8 | Genuine Alpha usually spills less. |
| Endpoint C-energy | `+6.26e-9`; 5/8 | Numerical matching noise, not a budget advantage. |
| Endpoint Frobenius norm | `-0.01858916`; 3/8 | Genuine is slightly larger on average. |
| Projector violation | `+1.18e-8`; 4/8 | No directional containment separation. |
| NFE | `0`; 0/8 | Both use 5 NFE/case. |

This rejects a simplistic claim that putting `P` inside the Alpha solve must
increase direct-z fidelity.  On this panel it does the opposite, but it shifts
the trade-off toward lower held-out KL and lower spill.  The projector
equation changes the feasible direction; it does not impose a scalar ordering
over fidelity, preservation, and norm simultaneously.

## BF refresh versus native genuine Alpha at matched C

This is the primary ODE-local comparison.  BF refreshes a genuine Alpha
direction at four verified descendants inside one atomic edit and then
normalizes the combined endpoint to exactly the native reference C-budget.

| Axis (BF minus native, sign-normalized) | Mean; +/8 | Lenient gate |
|---|---:|---|
| Direct-z residual | `+0.08233046`; 6/8 | pass |
| Canonical delta gain | `+0.13776107`; 8/8 | pass |
| Canonical delta cosine | `-0.01412005`; 2/8 | fail |
| Generated mean error | `+0.10533109`; 7/8 | pass |
| Generated worst error | `+0.10699570`; 7/8 | pass |
| Output-NLL reduction | `+0.05692602`; 7/8 | pass |
| Rewrite output progress | `+0.71733285`; 7/8 | pass |
| Exact target margin | `+0.72356713`; 6/8 | pass |
| Exact target satisfied | `0`; 0/8 | tie |
| Paraphrase-NLL reduction | `+0.88091911`; 7/8 | pass |
| Preservation score | `-0.20097198`; 1/8 | fail |
| Off-token spill | `-0.01056363`; 2/8 | fail |
| Endpoint C-energy | `-2.42e-9`; 4/8 | matched |
| Endpoint Frobenius norm | `+0.10153338`; 7/8 | pass |
| Projector violation | `-2.48e-9`; 4/8 | no separation |
| NFE | `-19`; 0/8 | fail; BF is more expensive |

The descriptive bootstrap intervals reinforce the main directional pattern:
direct-z residual `[0.02827215, 0.14197551]`, output-NLL reduction
`[0.00541535, 0.14113468]`, and preservation score
`[-0.34313069, -0.07168475]`.  The smaller-Frobenius mean has interval
`[-0.06918730, 0.24499900]`, so it passes the precommitted lenient count/mean
gate but is not a strict interval-exclusion result.

This pattern is consistent with a state-dependence mechanism: refreshed Alpha
directions reach the same C-budget along an endpoint with better local target
and edit behavior, and slightly smaller Euclidean low-rank norm.  It does not
show that numerical splitting alone is beneficial; the signal can only come
from direction changes induced by relinearization.  The simultaneous
preservation and spill regressions are evidence against turning this local
mechanism into a broad model-preservation claim.

## Cone reachability and oracle ceiling

The synchronous genuine-Alpha cone is a non-deployable W0 controllability
diagnostic under the same reference C cap.  `oracle_do_z` is a weight-free
activation intervention.  Both establish possibility ceilings, not method
quality.

| Axis versus no-op | Genuine Alpha cone: mean; +/8 | Oracle do-z: mean; +/8 |
|---|---:|---:|
| Direct-z residual improvement | `+0.37876187`; 8/8 | `+0.99999999`; 8/8 |
| Canonical delta gain | `+0.42010786`; 8/8 | `+1.00000000`; 8/8 |
| Canonical delta cosine | `+0.88393210`; 8/8 | `+1.00000000`; 8/8 |
| Generated mean-error improvement | `+0.34982255`; 8/8 | `+0.99999997`; 8/8 |
| Generated worst-error improvement | `+0.32856971`; 8/8 | `+0.99999997`; 8/8 |
| Output-NLL reduction | `+10.43335807`; 8/8 | `+10.54005588`; 8/8 |
| Rewrite output progress | `+13.20972961`; 8/8 | `+15.88906384`; 8/8 |
| Exact target margin | `+13.69907719`; 8/8 | `+15.67628711`; 8/8 |
| Exact target satisfied | `+1`; 8/8 | `+1`; 8/8 |
| Paraphrase-NLL reduction | `+7.87081769`; 8/8 | `+9.25800990`; 8/8 |
| Preservation score | `-0.94572745`; 0/8 | `-1.01225985`; 0/8 |
| Off-token spill | `-0.02567594`; 0/8 | `0`; 0/8 |
| Endpoint C-energy | `-0.06375028`; 0/8 | `0`; 0/8 |
| Endpoint Frobenius norm | `-2.31386637`; 0/8 | `0`; 0/8 |
| Projector violation | `-5.1436e-6`; 0/8 | `0`; 0/8 |
| NFE relative to no-op | `-14`; 0/8 | `0`; 0/8 |

The cone's `37.9%` mean residual reduction proves that the bounded Alpha
response set can partially realize the replayed target; the oracle's nearly
unit reduction proves that the target itself is active.  The cone is neither a
deployable controller nor an upper bound on refreshed nonlinear paths, and
its preservation loss shows again that reachability does not imply benign
writing.

## Natural full budget versus C-matched budget

Removing the MEMIT-BF C cap gives both static solvers a strong fidelity gain,
but spends more update energy and norm.  The left arm is each full-budget arm.

| Axis | Genuine full vs C-match: mean; +/8 | Post-hoc full vs C-match: mean; +/8 |
|---|---:|---:|
| Direct-z residual | `+0.52851892`; 8/8 | `+0.19969868`; 8/8 |
| Canonical delta gain | `+0.51005850`; 8/8 | `+0.20688120`; 8/8 |
| Canonical delta cosine | `+0.05863778`; 8/8 | `+0.01968284`; 8/8 |
| Generated mean error | `+0.39674296`; 8/8 | `+0.18036949`; 8/8 |
| Generated worst error | `+0.39077810`; 8/8 | `+0.17851448`; 8/8 |
| Output-NLL reduction | `+0.07670711`; 8/8 | `+0.02345101`; 8/8 |
| Rewrite output progress | `+1.90634818`; 8/8 | `+0.83162782`; 8/8 |
| Exact target margin | `+1.68134761`; 8/8 | `+0.75166011`; 7/8 |
| Exact target satisfied | `0`; 0/8 | `0`; 0/8 |
| Paraphrase-NLL reduction | `+0.86916506`; 7/8 | `+0.32203104`; 7/8 |
| Preservation score | `-0.06808518`; 2/8 | `-0.02882546`; 2/8 |
| Off-token spill | `-0.02819981`; 0/8 | `-0.01107111`; 0/8 |
| Endpoint C-energy | `-0.11662662`; 0/8 | `-0.03966615`; 0/8 |
| Endpoint Frobenius norm | `-1.71993665`; 0/8 | `-0.68790336`; 0/8 |
| Projector violation | `0`; 0/8 | `0`; 0/8 |
| NFE | `0`; 0/8 | `0`; 0/8 |

In absolute terms, genuine full reduces mean residual to `0.00619835` versus
`0.53471726` for genuine C-match, but its mean C-energy rises from
`0.06375028` to `0.18037690` and its Frobenius norm from `2.57325908` to
`4.29319574`.  Post-hoc full reduces mean residual from `0.44154423` to
`0.24184555`, while C-energy rises from `0.06375029` to `0.10341643` and norm
from `2.55466992` to `3.24257328`.  Thus target fidelity is strongly
budget-sensitive, and natural-budget results cannot be used as evidence for a
more efficient write.

## Compute accounting

Raw NFE totals exactly reproduce the analyzer total `488`:

| Arm | NFE/case | NFE over 8 cases |
|---|---:|---:|
| `no_op_replay` | 1 | 8 |
| `oracle_do_z` | 1 | 8 |
| `alpha_genuine_ordered_full` | 5 | 40 |
| `alpha_genuine_ordered_c_matched` | 5 | 40 |
| `alpha_posthoc_bp_full` | 5 | 40 |
| `alpha_posthoc_bp_c_matched` | 5 | 40 |
| `alpha_bf_genuine_refreshed_k4` | 24 | 192 |
| `alpha_sync_z_cone_genuine` | 15 | 120 |

The BF local signal therefore costs `4.8x` the native C-matched NFE.  This
experiment measures a mechanism, not a favorable accuracy/compute frontier.

## Relation to the paired Qwen MEMIT diagnostic

The earlier Qwen MEMIT-v2 report found cone feasibility but no current-BF
directional gap: at matched C its BF-versus-native direct-z gain was
`-0.05456161` with only 2/8 positive cases, and output-NLL gain was effectively
zero (`+0.00002117`, 2/8).  Under the genuine Alpha geometry here, the
corresponding BF refresh has direct-z gain `+0.08233046` (6/8) and output-NLL
gain `+0.05692602` (7/8).  Both tracks retain unfavorable preservation:
`-0.08699755` for MEMIT BF versus its native control and `-0.20097198` here.

This cross-report sign change is a useful architecture-conditional clue that
the benefit is tied to proposal/solve geometry and state-dependent refresh,
not to the bare act of dividing one fixed update into four pieces.  It is not
yet a same-case difference-in-differences result, and the two native proposal
families must not be treated as interchangeable.  The final cross-track
synthesis must retain that distinction and the common preservation failure.

## Final Qwen interpretation

This run answers the motivating possibility questions as follows:

1. **Does native Alpha actually write the frozen direct-z target?**  Yes,
   partially and consistently: `+0.46528274` residual improvement, 8/8, with
   strong edit proxies and exact satisfaction in all eight cases.
2. **Can BF reflect direct-z better than the native Alpha write at the same
   C-budget?**  Yes on this Qwen panel under the lenient gate:
   `+0.08233046`, 6/8, together with output/edit and generated-context gains.
3. **Does BF obtain the signal with a smaller update?**  Its low-rank
   Frobenius endpoint is smaller by `0.10153338` in 7/8 at matched C, a lenient
   signal whose descriptive CI crosses zero.  It is not smaller in C-energy by
   construction and is much larger in NFE.
4. **Does better direct-z writing imply model preservation?**  No.  BF has
   worse held-out KL and spill than native Alpha; both writes are worse than
   no-op on preservation.
5. **Is genuine Alpha uniformly better than post-hoc projection?**  No.
   Genuine Alpha trades worse target/edit fidelity for better preservation and
   spill at the matched budget.

Accordingly, Qwen closes only a **local geometric/mathematical motivation**:
relinearized genuine-Alpha BF can select a better target-directed endpoint
than its static genuine baseline under an identical C cap, and can do so with
a modestly smaller low-rank Frobenius norm.  The run does not close a method
claim.  It contains no sequential edits, no long-horizon retention test, no
collapse trajectory, and no evidence that ODE-Edit broadly preserves the
model or dominates AlphaEdit/MEMIT.
