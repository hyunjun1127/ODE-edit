# Session 02 Llama hook-validator RCA

Instruction: `ODEEDIT-S02-P0-LLAMA-HOOK-RCA-V1` plus amendment A1

Checkpoint: `02da467680de31f0196f7d90cad634599e662c1b`

Status: `CPU_STRESS_FAIL / A_B_ARITHMETIC_REDESIGN_REQUIRED / RETRY_HOLD`.

Supersession note: this file records the pre-redesign checkpoint. GH later
defined the canonical FP32 tangent; its fixed A/B gate passed 24/24, while the
finite B/C mismatch was separated into the T/C prototype. See
`2026-08-03-session02-p0-canonical-fp32-tangent.md` and
`2026-08-03-session02-p0-quantized-rowblock-trial-prototype.md`. Retry remains
HOLD.

## Retained evidence

FACT: job 16025 failed in Full warm-up rep-0 with
`actuator hook differs from P0 scalar finite difference`.

FACT: exact searches of retained stdout/stderr found no per-layer hook value,
finite-difference value, absolute error, or relative error. The raw mismatch
cannot be reconstructed and is not estimated.

FACT: the manifest records `torch.float32`, not BF16/FP16. Consequently BF16
quantization is not established as the cause of this run. Fixed-epsilon
cancellation/truncation and A/B arithmetic order remain possible, unseparated
causes.

## Three numerical paths

- A — primary hook: captures module input `x` and output gradient `g`, promotes
  `x/g/U/V` to float32 unless float64 is present, and evaluates
  `sum((g @ U) * (x @ V))`.
- B — functional overlay: casts `U/V` to hidden dtype, evaluates
  `(hidden @ V) @ U.T`, casts the delta and coefficient to output dtype, and
  adds it to the module output.
- C — accepted write: forms row-blocked `U @ V.T` in float32 for non-float64
  parameters, multiplies by the accepted coefficient, casts to parameter
  dtype, and mutates the parameter inside the outer transaction.

A float64 CPU identity does not prove A=B or B=C at lower precision or model
scale. The two comparisons remain separate gates even when their current lock
tolerance numbers coincide.

## Hard-reference correction

The one-sided epsilon `1e-3` comparison is no longer a hard oracle. It is
retained only as metadata with role `diagnostic-disabled`.

`ScalarGateDirectionalReference` assigns one float32 scalar alpha to every
layer and performs one combined rewrite event forward/backward. At alpha zero
its hooks reproduce path B's hidden/output dtype and factorized matmul graph,
while primary path A independently uses captured `x/g` contraction. The hard
comparison uses the existing scalar-gate lock (`atol=5e-5`, `rtol=0.005`) and
now reports layer, both derivatives, and absolute error on failure.

The validator adds separately classified `N_reference_gate_fwd`,
`N_reference_gate_bw`, and `reference_gate` wall/GPU component time. It does
not increment primary `N_bw`, touch controller slopes/QP/direct-z, or expose
evaluation fields.

Changed hashes:

- `derivatives.py`: `6c78a30ec4958a430b263600f5b5a407b6a99df41276260bce354462f7b5d211`
- `easyedit_backend.py`: `ee3cfadfa1ce315752f001104e0770bfa9925c31d8b1dd9af849e2c8ccbf0896`
- `instrumentation.py`: `07c4451efa85d993cce889871743e9d30b2c366e5acbb70f30bc985594cf61ad`
- hook tests: `33c72be5fe5f02be89abfbe85ce126f0d5ab81add01ba3db693605ed416002e0`

## CPU evidence

- float64 fixture A versus B max absolute error: `1.553381590024827e-10`
- float64 dense-weight oracle versus B max absolute error: `1.5533815726775924e-10`
- BF16 synthetic fixture: A=`0.125`, B=`0.125`, fixed epsilon `1e-3` one-sided FD=`0.0`
- BF16 nontrivial coefficient B/C event output: exact, max absolute error `0.0`
- injected A-path sign error: scalar hard oracle rejects it
- target `.grad is None`, data pointer/version/requires-grad restoration, and exception cleanup: PASS
- model-independent schema/no alias branch: PASS
- full method regression: 55/55 PASS with warnings as errors

The BF16 fixture proves that fixed epsilon can false-fail in principle; it does
not diagnose the actual float32 Llama failure.

## Red-review nonlinear stress

Before execution, the stress grid was fixed to seeds `(13, 29, 47, 71)`,
scales `(0.25, 1.0, 4.0)`, rank 2, and a two-layer BF16 SiLU model with a
two-context combined event. Alpha-zero combined-event identity was exact for
all 12 cases. The unchanged A/B gate produced 24 layer comparisons: 12 passed
and 12 failed.

- Maximum absolute error: `0.010189056396484375` (seed 47, scale 4, layer 1;
  relative error `0.0030029625`, so this row passes by relative tolerance).
- Maximum relative error: `0.11128361934823808` (seed 13, scale 0.25, layer 0;
  absolute error `0.00011201365850865841`, FAIL).
- All layers/scales for seeds 13 and 71 failed. Their relative discrepancies
  were scale-invariant: 11.128%/1.571% and 3.584%/0.744%, respectively.

No case or tolerance was changed. The focused failing test remains uncommitted
as stop-condition evidence; no checkpoint amend was created. This establishes
that exact alpha-zero output/event identity does not imply derivative identity
between A's float32 reassociation and B's BF16 graph.

## Root-cause confidence and readiness

- FACT: the obsolete one-sided FD hard gate failed; values were not retained.
- FACT: the fixed BF16 stress confirms an A/B arithmetic mismatch class,
  independent of one-sided FD.
- INFERENCE: job 16025's float32 failure may still be FD cancellation or an
  A/B order effect; retained values are insufficient to choose.
- CONFIRMATION_NEEDED: GH must define canonical continuous-field arithmetic,
  then reclose A/B, accepted-scale B/C, and model-scale gates.

Production field arithmetic was not changed. GH must decide whether the
canonical derivative follows continuous float32 A, functional path B, or
quantized accepted path C; this implementation does not silently align or
relax them.

Llama retry, GPU/model load, Slurm, push, P1, and claims remain HOLD. Raw root
identity is preserved in `runs/session02-p0-tech-v1/terminal-metadata.json`.
Broadcast status: `BROADCAST_EXCEPTION_SERVER2_NOT_READY`.
