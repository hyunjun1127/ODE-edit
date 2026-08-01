# Session 01 direct-z possibility diagnostic

## Status and claim boundary

This is a user-authorized possibility diagnostic, not a method-comparison or
benchmark experiment.  It asks whether the frozen MEMIT direct-z intervention
is behaviorally useful, whether the existing synchronous actuator cone can
realize it under the current BF endpoint budget, and whether direct-z fidelity
co-varies with rewrite and preservation proxies.  A closer direct-z write is
not assumed to improve editing or preservation.

Allowed claims stop at the fixed two-snapshot, fresh-case panel.  Accuracy,
general downstream capability, retention, sequential-collapse prevention, and
ODE-Edit superiority are out of scope.

## Fixed execution envelope

- Models: `llama3-8b-inst`, `qwen2.5-7b-inst` only.
- Cases: canonical salted ranks `[132:140]`, eight atomic CounterFact cases,
  disjoint from the first 132 ranks used by earlier Session 01 diagnostics.
- Layers: MEMIT layers `[4,5,6,7,8]`; direct-z layer `8`.
- Direct-z: computed once per case at W0 and reused byte-identically by every
  branch.
- BF path: existing output-utility score-mix, four `D/4` hops with direction
  and coefficient refresh at each descendant state; `D` is native ordered
  MEMIT C-distance.
- Controller probe: `D/64`, authorized six rewrite contexts only.
- Offline inputs: fixed EasyEdit source, fixed model revisions, pinned existing
  Wikipedia moments.  Null-space projector identity is preflight-only and is
  never deserialized in this MEMIT track.  Download and recomputation are
  forbidden.
- Resources: one two-GPU Slurm parent, one visible A6000 per model child,
  `16 CPU / 130000M / 12h`.  The user raised server1 project cap from 3 to 4,
  permitting the pair alongside the existing two-GPU adaptive job.

## Locked arm order

1. `no_op_replay`
2. `oracle_do_z`
3. `native_ordered_full`
4. `native_alpha_c_matched`
5. `bf_current_refreshed_k4`
6. `sync_z_cone_oracle`

`oracle_do_z` adds the frozen delta only at the layer-8 subject position and
does not write weights. `native_alpha_c_matched` scales the complete ordered
proposal to the BF endpoint C-energy. `sync_z_cone_oracle` is a non-deployable
diagnostic ceiling: it solves a non-negative L2-ball ridge problem over the
five W0 unit-C synchronous layer responses, with the BF endpoint C-distance as
its cap. It may not be presented as ODE-Edit.

All path/proposal hashes, the z-cone coefficients, arm order, and budgets are
written and fsynced to an exclusive receipt before endpoint outcomes or
held-out prompts are evaluated.

## Scalar outcomes

Primary representation scalar:

`z_residual_ratio = ||h8_canonical(Wa)-z*|| / ||z*-h8_canonical(W0)||`.

Generated contexts use the shared-delta target
`h8_c(W0)+(z*-h8_canonical(W0))`, summarized by mean and worst relative error.
Also record target-axis gain/cosine and full-sequence off-token spill at layer
8. Raw z, hidden states, keys, logits, prompts, or token IDs may not enter JSON.

Output scalars reuse the exact suffix-verified teacher-forced evaluator:
smooth progress, NLL reduction, minimum target margin, and all-context/all-token
top-1. Evaluation-only CounterFact paraphrases report NLL reduction.

Preservation is an explicitly limited proxy: mean pre-to-post next-token KL on
the first four neighborhood and first four generation prompts, plus endpoint
C-energy and low-rank Frobenius norm. `preservation_score` is negative held-out
KL so that larger is better. These prompts are loaded only after the arm
commitment and never influence path construction, scaling, stopping, or the
z-cone solver.

## Lenient possibility classification

Technical validity, eight-case ITD, fixed arm order, direct-z once-per-case,
receipt-before-outcome, exact rollback, finite scalars, and pinned-cache reuse
remain mandatory. Scientific classification is descriptive and deliberately
lenient; bootstrap intervals are reported but need not exclude zero.

- `oracle_ceiling_open`: oracle-do improves mean output and a majority of
  cases, without requiring significance.
- `cone_feasible`: z-cone reduces mean direct-z residual versus no-op and does
  so in a majority of cases.
- `bf_tracks_z`: current BF reduces residual versus no-op in mean and majority.
- `bf_output_without_z`: BF improves output in mean/majority while its z
  residual is not better than native; interpreted as possible task-relevant
  projection, not failure.
- `fidelity_preservation_alignment`: lower residual and higher preservation
  score have the direction expected in mean paired contrasts; correlation is
  descriptive only.
- Architecture disagreement is retained, never pooled across raw utility
  scales, and reported as conditional evidence.

No case, arm, or model may be removed after outcomes. Thresholds, cases,
budgets, and arm definitions are immutable for this run. A technical failure
blocks pair-level scientific closure; a scientifically null or unfavorable
result is still a completed diagnostic.

## Reports and minimal audit

The runner writes ignored raw/scalar artifacts under
`local/results/raw/session01_motivation/dzf_*_p0_v1`. A deterministic CPU
analyzer writes one compact projection per model. After each model is terminal,
a separate agent sees only this spec, that model's manifest/summary/analyzer
projection, and its own Slurm metadata, then writes its model report. A final
pair report applies the fixed possibility classification without retuning.

The only required audits are: source/cache/model identity, resource cap and
Slurm envelope, exact case/arm/count schema, direct-z lineage, scalar firewall,
receipt order, rollback, and deterministic analysis. Broad re-audits of prior
Session 01 evidence are intentionally omitted.
