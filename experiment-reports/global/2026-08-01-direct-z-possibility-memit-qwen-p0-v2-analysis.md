# MEMIT direct-z possibility diagnostic — Qwen p0 v2

## Decision

`dzf_qwen_p0_v2` on `qwen2.5-7b-inst` is a technically valid, fixed-panel
possibility diagnostic with the analyzer verdict
`CONE_FEASIBLE_NO_DIRECTIONAL_BF_GAP_OBSERVED`.  The evidence opens the
*synchronous direct-z cone feasibility* branch, but does **not** open a
current-BF directional advantage branch: the exact flags are
`oracle_ceiling_clear: true`, `sync_cone_feasibility_clear: true`,
`current_bf_directional_gap_clear: false`, and
`method_superiority_decision: false`.  It therefore does not establish an
ODE-Edit advantage, method superiority, benchmark improvement, causal
mediation, lifelong retention, or sequential robustness.

Here, **BF** means the locked `bf_current_refreshed_k4` arm.  **DF cone** is
the locked `sync_z_cone_oracle` arm: a non-deployable, synchronous direct-z
diagnostic ceiling, not ODE-Edit.  The sign convention is kept in every scalar
name below; a negative value is not silently converted into a positive result.

## Fixed contract and completion evidence

The manifest fixes six arms in the specified order, eight cases, 4,000 paired
bootstrap resamples (seed `20260801`), configuration SHA-256
`b9b5f6850a0544cea799961c7e0da731ee99ba9a782196feaeb4014222d91ab4`, and
selection SHA-256
`12393c041fdcea6b06b15ccb4b83b9bc23a7f611fa317e4a5f3f213687bb1d5c`.
The summary records all 8/8 planned cases as passed, zero failed cases, 48
outcomes, direct-z computed once per case, receipt before outcome evaluation,
firewall pass, and exact rollback for every arm.  The deterministic analysis
also marks the artifact valid (ITD denominator 8; 8 successful cases), with
analysis SHA-256
`123e13f0beb42f5f6dad01e06b68e3d114ebdd661422b1c9c08781f288c01188`.

The designated Slurm accounting records are likewise terminal: job `15740`
completed with exit code `0:0` in `02:47:34` (16 CPU, 2 GPUs, 126.95G), and
step `15740.1` completed with exit code `0:0` in `02:47:32` (8 CPU, 1 GPU,
63.48G).  This is execution provenance, not evidence of scientific effect.

## What is feasible

The primary representation quantity is

\[
r_z(W_a)=\frac{\lVert h_8^{\mathrm{canonical}}(W_a)-z^*\rVert}
{\lVert z^*-h_8^{\mathrm{canonical}}(W_0)\rVert}.
\]

Thus `z_residual_gain_noop_minus_cone > 0` means that the bounded synchronous
cone gets closer to the frozen target than no-op.  The cone feasibility signal
is unambiguous on this fixed panel: its mean gain is `0.44754599` (median
`0.44335462`, 8/8 positive, paired bootstrap 95% CI
`[0.40004270, 0.49234505]`).  Its output-NLL gain over no-op is `10.48807717`
(median `10.48360740`, 8/8 positive, CI `[8.38877952, 12.71538155]`).  The
analyzer consequently sets `sync_cone_feasibility_clear: true`.

The frozen activation itself is also behaviorally active: the non-weight
`oracle_do_z` arm has residual gain `0.99999999` and output-NLL gain
`10.54005588`, both positive in 8/8 cases.  These two facts are a geometric
and behavioral **possibility** signal: at the matched endpoint-C budget, the
available synchronous response cone can move materially toward an effective
frozen target.  They are not a deployable-controller result.

The favorable cone signal has an explicit cost that must remain in the record:
`preservation_gain_cone_minus_noop = -1.03687006` (8/8 negative, CI
`[-1.49582045, -0.57993919]`; higher `preservation_score = -heldout_kl` is
better).  So target attainability and output improvement do not imply the
preservation property needed for a broad editing claim.

## BF versus the direct-z cone ceiling

All rows are paired eight-case analyzer scalars.  “Positive” follows the
stored scalar orientation, and the field name states the comparator order.

| Axis | Exact paired scalar | Mean (median; positive/8; 95% CI) | Reading |
|---|---|---:|---|
| Direct-z fidelity | `z_objective_gap_bf_minus_cone` | `-0.25748369` (`-0.24182018`; 0/8; `[-0.28858216, -0.22897919]`) | BF is below the DF-cone target objective in every case. |
| Output/edit objective | `output_objective_gap_cone_minus_bf` | `-0.04504471` (`-0.02966494`; 0/8; `[-0.08493038, -0.01854487]`) | By this field's cone-minus-BF direction, BF has the higher output objective; it does not repair the fidelity deficit. |
| Exact target margin | `exact_margin_gap_cone_minus_bf` | `-1.40007865` (`-1.32100964`; 0/8; `[-2.03822494, -0.76511657]`) | The margin scalar similarly favors BF under cone-minus-BF orientation. |
| Generated mean error | `generated_error_mean_gap_bf_minus_cone` | `-0.23935915` (`-0.22811917`; 0/8; `[-0.27334938, -0.20743676]`) | BF has lower generated-context mean error, a mixed counterpoint to its worse direct-z objective. |
| Generated worst error | `generated_error_worst_gap_bf_minus_cone` | `-0.23116977` (`-0.20329825`; 0/8; `[-0.26744421, -0.19933276]`) | BF has lower worst generated-context error. |
| Preservation | `preservation_difference_bf_minus_cone` | `-0.09311463` (`-0.11201239`; 1/8; `[-0.26756258, 0.10171833]`) | BF is lower on the higher-is-better preservation score in 7/8 cases; this interval is descriptive and crosses zero. |
| Off-token behavior | `off_token_spill_gap_bf_minus_cone` | `+0.02745909` (`+0.02239313`; 8/8; `[0.01892304, 0.03794668]`) | BF has more layer-8 full-sequence off-token spill in every case. |

The analyzer's required current-BF classification is therefore
`current_bf_directional_gap_clear: false`.  The output-side counterpoints are
real paired findings, but they cannot be relabeled as a direct-z-fidelity or
ODE advantage when BF's direct-z objective is lower in all eight cases and
spill is higher in all eight.

## C-matched native control: fidelity, edit, preservation, and norm

BF and `native_alpha_c_matched` have matched endpoint-C energy
(`endpoint_c_native_alpha_bf_matched: true`; paired difference
`-3.49245965e-09`).  At that matched C budget, BF has the following mixed or
unfavorable comparisons:

| Axis | Exact paired scalar | Mean (median; positive/8; 95% CI) |
|---|---|---:|
| Direct-z gain | `z_gain_bf_vs_native_alpha` | `-0.05456161` (`-0.05820610`; 2/8; `[-0.10120709, -0.00519308]`) |
| Output-NLL gain | `output_nll_gain_bf_vs_native_alpha` | `+0.00002117` (`-0.00087111`; 2/8; `[-0.00203097, 0.00305830]`) |
| Output-progress gain | `output_progress_gain_bf_vs_native_alpha` | `-0.11081997` (`-0.16710901`; 2/8; `[-0.27713016, 0.07243150]`) |
| Exact-margin gain | `exact_margin_gain_bf_vs_native_alpha` | `-0.12983727` (`-0.08659172`; 1/8; `[-0.27556679, 0.01785444]`) |
| Paraphrase gain | `paraphrase_gain_bf_vs_native_alpha` | `+0.20916645` (`+0.10083959`; 7/8; `[0.06502618, 0.40044363]`) |
| Preservation gain | `preservation_gain_bf_vs_native_alpha` | `-0.08699755` (`-0.01527831`; 2/8; `[-0.19236904, -0.00940425]`) |
| Low-rank Frobenius norm | `endpoint_frobenius_difference_bf_minus_native_alpha` | `-0.83019237` (`-1.32648998`; 1/8; `[-1.80287749, 0.35026771]`) |

The smaller BF low-rank Frobenius norm is a budget-shape observation, not a
quality proof: its interval crosses zero, while BF's direct-z and preservation
gains against the C-matched control are negative.  The one favorable,
paraphrase-specific contrast does not override the primary fidelity,
preservation, or mixed output results.

## Signal for the final ODE-Edit judgment

This Qwen panel supplies a narrow mathematical signal: a frozen target can be
partially realized by the W0 synchronous response cone under the BF endpoint-C
cap.  It does **not** show that the refreshed four-hop BF trajectory exploits
that controllability more effectively than the C-matched native proposal:
BF's paired direct-z gain is negative versus native alpha, and its direct-z
objective is lower than the non-deployable cone ceiling in all cases.  The
reported z/output/preservation relations are also descriptive and noncausal:
Pearson `z_gain_vs_output_gain = -0.00913324` (bootstrap CI
`[-0.16901113, 0.10592495]`) and `z_gain_vs_preservation_gain = 0.06501878`
(CI `[-0.15314351, 0.40953189]`).

Accordingly, this result may contribute one architecture-conditional
controllability datum to a later cross-method comparison, but it must retain
its unfavorable and mixed evidence.  It cannot support an ODE-Edit
superiority, lifelong-editing, retention, or general downstream-capability
conclusion.
