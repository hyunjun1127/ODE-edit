# Session 01 AlphaEdit direct-z paired possibility specification

## Decision boundary

This is an atomic mechanism diagnostic.  It asks whether the exact frozen
direct-z targets from MEMIT p0 v2 can be realized inside AlphaEdit's pinned
null-space geometry, and whether a four-hop ODE/BF numerical refresh exposes a
useful local signal at the same update budget.  It is not a lifelong run: K=4
is an optimizer path inside one fresh edit.  The experiment cannot establish
sequential-collapse prevention, long-horizon retention, broad downstream
preservation, or method superiority.

Scientific interpretation is lenient, but the execution contract is strict.
A directional signal requires positive paired mean and at least 5/8 positive
cases per model.  Bootstrap intervals are descriptive.  Llama and Qwen are
never pooled.

## Fixed target replay and models

- Models are exactly `llama3-8b-inst` and `qwen2.5-7b-inst` at the revisions in
  `manifests.py`; editable layers are 4--8 `mlp.down_proj`.
- Cases and order are exactly the MEMIT p0 v2 salted ranks `[132:140]`:
  `21100, 1477, 20838, 18707, 14288, 16426, 19041, 17609`.
- `direct_z_alpha_replay_lock.json` pins each source manifest, source summary,
  direct-z artifact bytes, tensor hash, W0 lineage hash, target-token hash, and
  paired MEMIT BF C-energy.  Alpha loads every target once and computes it zero
  times.  A miss or identity drift aborts; it never falls back to compute-z.
- The target anchor requires exact model revision, context manifest, request
  ID, z layer, W0 hashes for all five rewrite weights, MEMIT target hparams,
  target tokens, and direct-z artifact/tensor identities.  Alpha's solver YAML
  has a separate identity; a differing hparams hash is never silently ignored.
- EasyEdit source and artifacts are read-only.  The ODE repo owns every hook,
  proposal, receipt, and output.  Existing Wikipedia covariance and AlphaEdit
  projector files are reused; no statistics, null space, SVD, download, cache
  accumulation, or EasyEdit write is allowed.

## Genuine isolated-first-edit Alpha algebra

For layer (i), let (K_i\in\mathbb R^{d_{in}\times m}), distributed target
residual (R_i\in\mathbb R^{d_{out}\times m}), pinned projector (P_i), and
(\lambda=\texttt{L2}=1).  AlphaEdit's first independent edit has
`cache_c=0`, so its native system is

\[
 A_i X_i=(P_iK_i)R_i^\top,
 \qquad A_i=\lambda I+(P_iK_i)K_i^\top .
\]

With (U_i=P_iK_i) and (M_i=\lambda I+K_i^\top U_i),

\[
 X_i=U_iM_i^{-1}R_i^\top,
 \qquad
 \Delta W_i=X_i^\top=R_iM_i^{-\top}U_i^\top .
\]

This exact Woodbury/associativity identity avoids the multi-gigabyte
(d_{in}\times d_{in}) solve.  CPU float64 tests compare it with the direct
dense equation.  Runtime factors must be finite and their normalized linear
system residual must not exceed `5e-4`.  The implementation is algebraically
equivalent to the pinned upstream isolated first-edit system, not claimed to
be bitwise identical to its dense float32 trajectory.

Every case and arm starts with zero `cache_c`; no post-case key outer product
is accumulated or saved.  Ordered proposals measure current (K_i) and
current z after temporarily applying preceding layer updates, divide residual
by the number of remaining layers, and restore W0 from exact backups.

## Genuine versus post-hoc geometry

The post-hoc ablation does not use a MEMIT proposal.  It first solves the same
isolated Alpha equation with (P=I),

\[
 \Delta W_{base}=R(\lambda I+K^\top K)^{-\top}K^\top,
\]

then constructs (\Delta W_{post}=\Delta W_{base}P).  Genuine Alpha instead
places P inside both its coefficient system and right-hand side.  This cleanly
separates solve geometry from merely projecting an already chosen update.

The containment diagnostic is

\[
 \ell_P(\Delta W)=
 \frac{\lVert\Delta W(I-P)\rVert_F}
      {\lVert\Delta W\rVert_F},
\]

computed from low-rank Gram products without materializing a dense update.
Because the stored P is approximate, finite leak is an outcome, not a
technical failure or proof of global downstream preservation.

## Locked arms and budget

The exact operational order is:

1. `no_op_replay`
2. `oracle_do_z`
3. `alpha_genuine_ordered_full`
4. `alpha_genuine_ordered_c_matched`
5. `alpha_posthoc_bp_full`
6. `alpha_posthoc_bp_c_matched`
7. `alpha_bf_genuine_refreshed_k4`
8. `alpha_sync_z_cone_genuine`

For case (c), (E_{ref}(c)) is the already committed endpoint C-energy of
the paired MEMIT-v2 BF arm.  Genuine matched, post-hoc matched, and evaluated
Alpha BF endpoints are scaled to exactly (E_{ref}) within relative `5e-5`
and absolute `1e-10`; the cone is capped by (E_{ref}).  C is the pinned
Wikipedia moment metric used only to compare update size and is distinct from
Alpha's zero `cache_c` first-edit system.

Alpha BF uses genuine synchronous Alpha directions at W0 and at each verified
quarter-step descendant.  Four probe-selected directions each have path
energy (E_{ref}/16); the combined endpoint direction is normalized once to
(E_{ref}).  The target bytes never refresh.  This is a local numerical path,
not four sequential facts.  The synchronous cone solves a non-negative ridge
problem over five W0 unit-C layer responses under the same cap and is a
non-deployable controllability ceiling.

## Outcomes and precommitted comparisons

Every arm records direct-z residual ratio, canonical gain/cosine, generated
mean/worst error, sequence off-token spill, rewrite progress/NLL/margin/exact
satisfaction, paraphrase NLL, held-out next-token KL and `-KL`, endpoint
C-energy, low-rank Frobenius norm, projector leak, and NFE.  Serialization is
scalar/hash-only; prompts, targets, hidden vectors, full logits, generations,
weights, P, and raw keys are forbidden.

Primary model-local comparisons are:

- genuine C-matched versus post-hoc C-matched: whether P-inside-solve changes
  target fidelity, containment, update norm, edit signal, spill, or KL;
- Alpha BF versus genuine C-matched: whether relinearization supplies a useful
  direction at identical target and C-energy;
- cone versus no-op and oracle versus no-op: bounded actual-write reachability
  and the weight-free ceiling;
- full versus C-matched: natural-budget sensitivity.

The final same-case cross-track table contains `memit_ordered_c_matched`,
`memit_bf_current_refreshed_k4`, `alpha_genuine_ordered_c_matched`, and
`alpha_bf_genuine_refreshed_k4`.  Join keys require model, case, request,
selection, W0 lineage, target tensor/token, held-out payload, and matched
C-energy.  For each scalar (M), report the model-local lift

\[
 L_t=M(BF_t)-M(native_t)
\]

with sign normalized only for presentation, and the difference in differences
(L_{Alpha}-L_{MEMIT}).  Fidelity, edit, preservation, norm, and spill remain
separate; conflicting signs are retained.  No pooled score may turn one
positive proxy into an ODE-Edit superiority claim.

## Technical acceptance

- exact `8 x 8 = 64` outcomes per model and exact arm order;
- target loaded once/case, recompute count zero, replay/target anchor exact;
- pinned projector mmap/read-only and start/end byte identity exact;
- pinned covariance only, no recompute/download;
- genuine ordered solve once/case, all BF refresh calls genuine, post-hoc arms
  only from the unprojected Alpha base;
- receipt before held-out evaluation, scalar firewall, and byte-exact W0/RNG
  rollback for every case;
- fixed two-GPU simultaneous pair on `devbox`, one visible A6000 per child,
  under the dedicated direct-z session boundary and server1 cap four.
