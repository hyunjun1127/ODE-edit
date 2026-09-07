# Alpha JV D/S resource and cost review — CPU preparation only

Status: GPU_HOUR_BUDGET_UNASSIGNED. This is not a hold on D/S CPU work.
Scientific promotion: false. Scheduler/model/GPU/new experimental-result actions: 0.

## Read scope and ownership

The independent resource owner fully read the execution envelope (142 lines),
operational addendum (26 lines), and complete execution contract (471 lines).
Envelope SHA256: `9b9ab0b4c7345d5e5a6793457852ece08022cdd100301e5e13213b04e55d75a1`.
Addendum SHA256: `cef298e6c0b4e4613727483ca20b42d2bf787c09d712cc1099e12387a87b5603`.
Contract SHA256: `98ee832e597b1f79c79ca48c62838af8f49f5306c08b3480b990d494f42f15f5`.
Only `resources.py`, `test_resources.py`, and this audit draft are owned here.
Other workers' science/binding implementations and historical source are untouched.

## Admission interface

`resources.py` is pure CPU: no scheduler query, mutation, model import, or GPU call.
`ResourcePolicy` fixes prospective cap≤2, one GPU per process, and task memory
≤182272 MiB/GPU. `validate_sbatch_memory()` requires exactly one explicit
job-total `#SBATCH --mem=182272M`, rejecting extra `--mem-per-*` alternatives.

`assess_admission()` requires a fresh, complete scheduler snapshot supplied by
the caller. Failed/incomplete/stale evidence is not zero usage. RUNNING,
CONFIGURING, COMPLETING and terminal-but-not-confirmed-exited allocations count,
including grandfathered jobs; unknown allocation quantities fail closed.
Admission is not a reservation. Immediately before real submission the repository
`scripts/check-slurm-resource-cap.sh` remains mandatory, with live registry and
tracked `control/gpu-concurrency-policy.tsv` policy.

`BudgetAuthority()` defaults to null, not an inherited 8h/48h cap. Only a future
user authority identity and SHA may accompany a finite positive cap. The API
does not itself authenticate a user message: the integration owner must bind the
actual immutable user/GH authorization before constructing it.

`UsageLedger` is immutable and rejects overlapping intervals of the same GPU
allocation. Failed technical attempts, fidelity, load, evaluation, replay and
allocated residency all charge elapsed GPU seconds. It does not exclude costs
because an attempt's scientific denominator is zero. The caller must supply
charged intervals through admission time plus all outstanding uncharged task
reservations; the returned receipt binds both. New reservation+charged+outstanding
must fit the authorized total. Existing jobs are never altered by this API.

## Reproducible source-backed estimate, not a budget

Command:

```bash
python3 -m project.run_scripts.alpha_jv_llama_diagnosis.resources --repo .
```

The command reads immutable Git bytes at publication
`0d0a0131e4a6a2a645dfa6530377d420a084d136`; historical scientific runtime is
`77358b1546d1baf83b3e251afcce663b08d7bfd7`. It does not read Server2 live roots.

| Input | Bytes | SHA256 |
|---|---:|---|
| `main-four-terminal-v1/compute_accounting.csv` | 78625 | `b3281974c89db7c425b8a18760b6f8a102b118c3fab97a54c8771c4ac112b7bf` |
| `main-four-terminal-v1/run_registry.csv` | 2479 | `19496c0fb9d843c6d180af0394afc293185a3a2e634428162cd545da7b57baf3` |

The source has 40 unique arm/model/batch rows. It measures load/context setup,
B100 target generation, writer including endpoint, and endpoint evaluation.
Stored `terminal_compute.wall` is a monotonic clock reading, not elapsed time;
this estimator explicitly does not use it.

S proxy formula per model:

`load + z(B100 once) + 34/4 × (JV N4 writer − endpoint evaluator) + (Official writer − endpoint evaluator) + 11 × current-B100 evaluation`.

The 11 evaluations are nine JV endpoints, one Official endpoint, and entry.
Seven actual paths, 34 nodes and 170 main target JVP calls per model remain the
nominal schedule; saved common-entry reuse is accounted separately at runtime.
The mixed four-node writer timer includes fixed lifecycle costs, so multiplying
by 34/4 is a transparent proxy, not an isolated per-node measurement.

| Scope | Recorded-component mean proxy GPUh | Component-min / component-max proxy GPUh |
|---|---:|---:|
| S Llama | 1.80327449 | 1.71878790 / 2.00008485 |
| S Qwen | 1.80771525 | 1.73085095 / 1.98485797 |
| S both models | 3.61098974 | 3.44963885 / 3.98494283 |
| D Llama main: target + N0/NRMS/O + entry/current100 evaluations | 0.66256761 | Not a complete D estimate |
| D optional W5→B6–B9 exact replay | 1.74972615 | Separate one-time historical replay cost |

These min/max combinations are not confidence intervals or reliable upper bounds.
No cross-host speed multiplier is known: the existing run was on Server2 and
new execution is Server1. Cold state, sample/context, contention and repeated
restoration also differ. The following remain explicitly NOT_SEPARATELY_RECORDED
or new-execution overhead, not zeros:

- initial normal-signal FD/overlay fidelity;
- D repeated-capture/raw-bundle observations and optional controls;
- D four old900 rewrite and endpoint cross-objective observations;
- repeated Family/endpoint restore overhead beyond the mixed timer proxy;
- technical-attempt reservation and actual cross-host speed variation.

Thus no exact complete D/S ceiling is established. A planning allowance may be
presented to the user separately, with assumptions, but cannot become authorized
GPU hours. In particular the optional replay cannot silently consume the base
estimate. Asset unavailability removes neither S scope nor the nine candidates.

## Focused CPU evidence

`python3 -m unittest -q project.run_scripts.alpha_jv_llama_diagnosis.test_resources`
completed **13/13 PASS**. Negatives cover null/unauthorized budget, cap violation,
grandfathered/COMPLETING allocations, failed scheduler evidence, unresolved GPU
counts, stale/future snapshots, duplicate allocations, immutable technical-cost
ledger, interval double charging, reservations exceeding budget, duplicate/wrong
memory directives, and nonfinite/invalid input counts.

## Early independent lifecycle review

The first published shared modules were read-only inspected. The grid has all
nine JV endpoints and seven actual paths/model. Source N0 delegation, fixed
qref checks, raw-response-before-whitening and the explicit immutable lambda
passed to new shadows are present in the inspected implementation.

One minimal CPU reproduction initially exposed a callback-boundary defect: an
`endpoint_sink` consuming RNG after `observe_prefix()` had already finished its
guard returned `TERMINAL_VALID` with changed RNG while the prefix receipt claimed
`W_M_RNG_overlay_restore_pass=true`. This was sent to the integration owner for
correction, without editing shared files. The reproduction used only the existing
nonlinear CPU fixture, an observation callback calling `torch.rand(1)`, and no
model load/GPU/scheduler action. This is an intrinsic lifecycle check, not an
outcome/parity performance gate.

The integration owner subsequently added `observation_callback_guard()` around
the complete raw, prefix and terminal sinks and moved the prefix callback inside
`observe_prefix()`. Independent read-only re-review and CPU reproduction now give
**PASS_CPU** for this bounded correction:

| Independent check | Result |
|---|---|
| Raw sink consumes RNG | `OBSERVATION_CALLBACK_INTERVENTION`; RNG and W0 restored |
| Prefix sink consumes RNG | Same typed boundary; RNG and W0 restored |
| Terminal sink consumes RNG | Same typed boundary; RNG and W0 restored |
| Historical N0/.1/T2/N4 vs configurable loop, identical separate nonlinear CPU fixtures | Four nodes exact for c, q, raw coefficients, V before/after, E, terminal activation SHA, actual physical telemetry |
| Same legacy/new terminal selected-weight SHA | Exact match |
| Same legacy/new main target JVP count | 20/20 |

The source-equivalence comparison used the original
`alpha_native_response_ode_v31_sequential.trajectory.run_joint()` at `batch_index=2`
and new `alpha_jv_llama_diagnosis.trajectory.run_joint()` with
`TrajectoryConfig(.1,2,4)`, using separately constructed `test_shared.fixture()`
instances. Only the existing CUDA-synchronization wrapper was mocked for this
CPU fixture. The original algorithm source was not modified. The new focused
`test_all_sinks_rng_mutation_rejected` additionally preserves the regression in
the integration owner's tests. No remaining blocking discrepancy was found in
this bounded callback and source-exact CPU-path review.

Re-reviewed working-source SHA256:

- `fixture.py`: `e48bdd27eebb83eb671958cc6aa3f2699eafa3c815136a20a31bf24a0194f8b2`
- `trajectory.py`: `65694a0264b097f95a3d19eca5b6824049e0a8cd53cdece62c9177b806eb1b07`

Full actual runner/resource integration review remains pending. This document
does not claim actual GPU fidelity or approve GPU entry; the user GPU-hour
authorization remains unassigned.
