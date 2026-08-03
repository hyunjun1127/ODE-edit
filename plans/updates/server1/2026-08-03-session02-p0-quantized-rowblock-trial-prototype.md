# Session 02 quantized row-block finite-trial prototype

Instruction: `ODEEDIT-S02-P0-QUANTIZED-ROWBLOCK-TRIAL-PROTOTYPE-V1`

Parent checkpoint: `02da467680de31f0196f7d90cad634599e662c1b`

Status: `CPU_PROTOTYPE_PASS / RUNTIME_NOT_WIRED / GPU_RETRY_HOLD`.

This report is outcome-free. It validates arithmetic and resource structure
only; it contains no evaluation, generation, or scientific method result.

## Four numerical boundaries

- A: captured `x/g`, FP32 continuous output-space tangent contraction.
- B: independent alpha-zero scalar-forward FP32 tangent reference.
- T: finite-coefficient, read-only quantized row-block Linear emulator.
- C: existing accepted commit, FP32 row-block outer product followed by
  parameter-dtype cast and in-place add.

A/B remains the hard derivative gate. T/C becomes the hard finite-step gate.
The observed B/C difference remains a diagnostic: output `0/36`, event
`34/36` on its fixed BF16 grid.

## T arithmetic and safety contract

`QuantizedRowBlockFunctionalTrial` uses `row_block=64`. For every exact
`torch.nn.Linear` target and every output-row block it computes:

1. `(U_block32 @ V32.T) * coefficient`;
2. one cast to the parameter dtype;
3. `parameter_block + quantized_update_block` without mutation;
4. `F.linear(hidden, effective_weight_block, bias_block)`;
5. concatenation of activation blocks along the output dimension.

The primitive detaches target weight, bias, and factors from target-gradient
materialization. It retains no effective block after the forward, performs no
full-weight clone/checkpoint/delta, and never calls the accepted writer. It
fails closed for non-exact `torch.nn.Linear`, geometry mismatch, nonzero target
`.grad`, coefficient-zero output mismatch, or parameter/RNG mutation. It is
exported as a reusable primitive but is not connected to `memit_adapter.py`,
the controller, or the current runtime.

## Fixed CPU gates

Prelocked T/C grid: BF16 two-layer SiLU, rank 2, seeds
`(17, 37, 59, 83)`, direction scales `(0.25, 1.0, 4.0)`, coefficients
`(0.125, 0.5, 1.0)`, row block 64, and unchanged
`atol=5e-5, rtol=5e-3`.

- T/C output: `36/36 PASS`; maximum absolute and relative error `0.0`.
- T/C combined event: `36/36 PASS`; maximum absolute and relative error `0.0`.
- Coefficient-zero output identity: exact `36/36`.
- Fixed A/B regression: `24/24 PASS`; maximum absolute error
  `1.1920928955078125e-07`, maximum relative error
  `5.7828123713324247e-08`.
- Target pointer/version/`.grad`/requires-grad and CPU RNG exactness: PASS.
- Injected exception cleanup and commit-attempt rejection: PASS.
- Exact Linear fail-close and alias-independent source/schema: PASS.
- Focused tests: `18/18 PASS`, warnings as errors.
- Full method regression: `60/60 PASS`, warnings as errors.

## Synthetic cost and temporary bound

Timer fixture was fixed before measurement: seed 101, one CPU thread, BF16,
batch rows 16, two contexts, layers `256->512->256`, rank 2, coefficient 0.5,
row block 64, 20 warm-ups per path, 200 recorded samples per path, alternating
`B,T` then `T,B`. Each sample includes context setup/cleanup and both forwards.

| Path | Median | p95 | Min | Max |
|---|---:|---:|---:|---:|
| Continuous B | 0.7919975 ms | 0.826945 ms | 0.737255 ms | 0.961 ms |
| Quantized T | 3.0535995 ms | 3.088398 ms | 2.991468 ms | 6.163425 ms |

Median T/B ratio: `3.855567094593102x`.

The extra arithmetic estimate for the same two-context fixture is 98,304 MACs
for B versus 9,437,184 MACs for T (`96x`, excluding the common original module
forward and elementwise/bias work). T's observed largest effective block was
32,768 elements (`64 * 512`); its largest output block was 1,024 BF16 elements.
For BF16, the explicit cast peak is bounded by one FP32 update block plus one
BF16 quantized block, at most 196,608 bytes for that largest block, plus
activation-block storage and backend GEMM workspace. No
`output_width * input_width` full trial weight/delta exists.

## Integration alternatives for GH decision

**Simple T for every finite candidate.** Every accept/reject event is measured
on C-equivalent quantized arithmetic. It has the smallest semantic and rollback
surface, but pays the measured T cost for every rejected candidate.

**Two-tier B then T.** B cheaply filters a candidate and T validates only a
provisional acceptance. A T rejection leaves model state unchanged and its
event must be the authoritative trust verdict. This can save rejected-candidate
cost, but requires a canonical rule for B false rejects, two separately counted
event evaluations, and trust-state handling; otherwise it changes method
semantics.

SH recommendation: use simple T for the next technical warm-up if GH prioritizes
semantic closure; consider two-tier only after GH locks conservative B-filter
semantics and counter accounting. No runtime integration was made in this
prototype envelope.

## Source identity and remaining risks

- `functional_trial.py` SHA-256:
  `8639c5385fc0be0abb84c8bcb0713070ec003e3c79841dd6478a55aaabc835a9`
- `derivatives.py` SHA-256:
  `0636176eb557e02fe6fdf685f7b97ee23643bcd5ad0cc06bd92dd5aa4bbe7645`
- package export SHA-256:
  `871da81a0bd8e0b4941f3982262143994d99261d13c00ff850d4e4486d80ee85`
- focused test SHA-256:
  `04c790875619e48eb0b3084c04e55a7405681d39d01a0f6bb9116410019d1866`

FACT: both retained P0 manifests record model dtype `torch.float32`, while the
mandatory prototype grid is BF16. An auxiliary, outcome-free 130-output CPU
probe was coefficient-zero exact for BF16 and float64 but fail-closed for
float32 because row-block `F.linear` tiling changed bits. This was not used to
alter the required grid or tolerance.

CONFIRMATION_NEEDED: GH must reconcile the intended BF16/FP16 execution
contract with the retained float32 manifests before retry. Model-scale
coefficient-zero identity, T/C event identity, GPU time, and memory remain
unvalidated. This is why retry readiness is `HOLD`, despite the fixed BF16 CPU
prototype passing.

The local terminal metadata remains unmodified and unstaged at
`runs/session02-p0-tech-v1/terminal-metadata.json`, SHA-256
`dd3890b0dad73bc391788b8ade8cb34f0d8c7a944afee1f52d471915aafffd89`,
schema `ode-edit-session02-p0-failure-metadata/v1`. Its required-field inventory
is recorded in the canonical-tangent plan update; raw contents are not copied.

GPU/model load, Slurm, retry, push, and claims remain HOLD.
`TERRA_SUBAGENT_AUDIT_PENDING_RUNTIME_MISMATCH` and
`BROADCAST_EXCEPTION_SERVER2_NOT_READY` remain open.
