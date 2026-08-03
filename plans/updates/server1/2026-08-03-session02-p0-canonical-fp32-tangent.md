# Session 02 canonical FP32 tangent milestone

Instruction: `ODEEDIT-S02-P0-CANONICAL-FP32-TANGENT-V1`

Parent checkpoint: `02da467680de31f0196f7d90cad634599e662c1b`

Status: `A_B_PASS / CONTINUOUS_B_C_DIAGNOSTIC_MISMATCH / GPU_RETRY_HOLD`.

This is an outcome-free CPU numerical-boundary report. It contains no P0
scientific metric, evaluation, generation, or method-comparison claim.

## Canonical continuous tangent

For each factor direction `U @ V.T`, the implemented BF16/FP16 tangent is

`delta_y32 = (x.float() @ V.float()) @ U.float().T`.

Path A contracts captured output gradients directly with this output-space
delta. Path B creates the same delta independently in a scalar-alpha forward
graph, multiplies alpha in tangent dtype, and performs one final cast/add to
the module output. Float64 CPU oracle fixtures retain float64 arithmetic.

The primary path still materializes neither a dense target-weight gradient nor
a dense target-weight direction. A and B remain independent execution paths:
A consumes separately captured `x/g`; B differentiates a forward scalar gate.

## Fixed A/B gate

The locked nonlinear stress uses BF16, two Linear layers with SiLU, rank 2,
seeds `(13, 29, 47, 71)`, scales `(0.25, 1.0, 4.0)`, and
`atol=5e-5, rtol=5e-3`.

- A/B: `24/24 PASS`.
- Alpha-zero combined event identity: exact `12/12`.
- Maximum absolute derivative error: `1.1920928955078125e-07`
  (seed 29, scale 4, layer 1).
- Maximum relative derivative error: `5.7828123713324247e-08`
  (seed 13, scale 0.25, layer 0).
- Sign, factor-orientation, and captured-input corruptions are rejected by the
  independent scalar reference.

The obsolete fixed-epsilon one-sided finite difference remains
`diagnostic-disabled`; neither its epsilon nor its tolerance was retuned.

## Continuous B versus committed C diagnostic

The separately prelocked BF16 finite grid uses seeds `(17, 37, 59, 83)`,
direction scales `(0.25, 1.0, 4.0)`, coefficients `(0.125, 0.5, 1.0)`, and
the unchanged functional-commit tolerance.

- Output: `0/36 PASS`.
- Combined event: `34/36 PASS`.
- Maximum output absolute error: `0.5`.
- Maximum symmetric output relative error: `0.17307692766189575`.
- Maximum event absolute error: `0.060573577880859375`.
- Maximum symmetric event relative error: `0.0084869269443629`.

This triggered the original `B_C_MISMATCH` stop condition without changing a
case, coefficient, or tolerance. GH subsequently separated continuous tangent
B from finite-trial T under
`ODEEDIT-S02-P0-QUANTIZED-ROWBLOCK-TRIAL-PROTOTYPE-V1`; B/C is therefore kept
as a precision-boundary diagnostic rather than reclassified as a hard gate.

## Compute and memory shape

The canonical direct contraction adds one output panel with shape
`[flattened_rows, output_width]`. The former reassociated contraction used two
rank panels, each `[flattened_rows, rank]`. Both avoid the
`[output_width, input_width]` dense direction. The direct path adds the
`rank -> output_width` low-rank matmul and the output-panel inner product; its
FP32 panel storage is `4 * flattened_rows * output_width` bytes. This is a
model-scale memory item for the next GPU technical gate, not a CPU scientific
result.

## Source and verification

- `derivatives.py` SHA-256:
  `0636176eb557e02fe6fdf685f7b97ee23643bcd5ad0cc06bd92dd5aa4bbe7645`
- `functional_trial.py` SHA-256 after the subsequent T prototype:
  `8639c5385fc0be0abb84c8bcb0713070ec003e3c79841dd6478a55aaabc835a9`
- Focused derivative/functional tests: `18/18 PASS`, warnings as errors.
- Full method tests after the T prototype: `60/60 PASS`, warnings as errors.
- Compileall and `git diff --check`: PASS.
- Target `.grad`, pointer, version, requires-grad, and RNG cleanup: PASS.
- Qwen dense solver, controller, QP, direct-z, and commit writer: unchanged by
  this milestone.

## Provenance boundary

`runs/session02-p0-tech-v1/terminal-metadata.json` remains local, unmodified,
untracked/unstaged provenance per
`ODEEDIT-S02-RUNS-METADATA-GH-OWNED-V1`.

- SHA-256:
  `dd3890b0dad73bc391788b8ade8cb34f0d8c7a944afee1f52d471915aafffd89`
- Schema: `ode-edit-session02-p0-failure-metadata/v1`.
- Required top-level fields: `analysis_status`, `broadcast_status`,
  `empty_jsonl_interpretation`, `empty_jsonl_sha256`, `execution_head`,
  `generated_at_kst`, `instruction_id`, `jobs`, `lock_sha256`, `p1_status`,
  `pair_status`, `proposal_id`, `remediation_head`,
  `remediation_red_stress`, `schema_version`, `scientific_outcome_count`.
- Required per-job fields: `direct_z_sha256`, `elapsed_seconds`, `end_kst`,
  `exit_code`, `job_id`, `job_name`, `manifest_sha256`, `max_rss_kib`,
  `model_alias`, `start_kst`, `state`, `stderr_sha256`, `stdout_sha256`,
  `stop_condition`, `terminal_manifest_present`.

GPU/model load, Slurm, retry, push, P1, and claims remain HOLD.
`TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH` and
`BROADCAST_EXCEPTION_SERVER2_NOT_READY` remain recorded.
