# ODE-BF SH1/SH2 Experiment Review

Date: 2026-08-08 (KST)
Scope: Session 04/05 ODE-BF technical and causal experiments through SH1 fixed-E8 R7
Review mode: read-only reconstruction; no model, GPU, Slurm submission, or result mutation

## 1. Executive conclusion

The experiments establish a useful technical core, but they do **not** yet establish a completed scientific result for the current cold-start fixed-E8 method.

What is established:

1. The W64/BF16 virtual-write, atomic commit/rollback, and joint-B10 plumbing work on both Llama-3-8B-Instruct and Qwen2.5-7B-Instruct.
2. The early warm-start/pre-shared-residual controller under-edited badly. Its hard functional-P gate and its progress contract were major causes of non-advancement.
3. Fixed outer-entry P accounting and full residual arms materially improved edit progress. Turning preservation vetoes off allowed both models to reach `10/10` efficacy and `20/20` generalization at full pseudo-time, but functional-P drift increased substantially, especially for Llama.
4. Target-new-NLL routing is a plausible replacement for the old margin routing, but the existing causal runs do not show a decisive final-endpoint advantage once all preservation vetoes are removed.
5. Functional-P used as a soft routing signal remains scientifically untested in the intended cold fixed-E8 method because the real-model P1R7 runs never completed even the neutral arm.
6. Historical-H remains untested: every relevant B10-1 run had empty history, so H was vacuous.

The immediate blocker is numerical/harness behavior, not evidence that the scientific method failed:

- Llama R7 reaches six neutral fixed steps (`tau=0.75`, online efficacy `5/10`) and then aborts in an **unused soft-shadow stage-2 QP** while running the `E8-NEUTRAL` arm.
- Qwen R7 aborts in the authoritative neutral minimum-capacity QP after SLSQP returns status 8, even though the independently recomputed primal, stationarity, and complementarity residuals all satisfy the locked tolerances.

The next implementation should therefore (a) remove the neutral baseline's hard dependency on a soft-shadow solution, (b) make independently recomputed KKT/feasibility the acceptance authority rather than SciPy's boolean `success`, and (c) replace repeated identical SLSQP continuation with a deterministic second backend for genuinely uncertified soft QPs. The KKT tolerance must not be relaxed.

## 2. Evidence policy and limitations

### 2.1 Evidence grades

- **LOCAL-VERIFIED FACT**: recomputed from files currently mounted in the SH1 worktree, including JSON receipts, failure hashes, Git source, and scheduler accounting.
- **SEALED-HANDOFF FACT**: taken from an earlier terminal packet whose hashes and source boundary were checked at the time, but whose server2 result directory is no longer mounted on this host.
- **INFERENCE**: a causal interpretation supported by facts and source control flow, but not directly serialized as an outcome.
- **NOT_RECORDED**: not present in the immutable receipt and not reconstructed from hashes.

### 2.2 Current locally verified source and results

- SH1 branch: `codex/odeeditsh1-s05-fixed-e8-soft-routing-p1r7-v1`
- Current SH1 HEAD: `a90756b89e32edaa225aa4eb13c9f6e23b865339`
- Fixed-E8 scientific implementation parent: `7faa432b3264cae83355775a3fde2d7bbfc3bab2`
- R6 receipt checkpoint: `f23472963ac2dbb46282afd9c3a514bb75d59d5b`
- R7 numerical-continuation checkpoint: `a90756b89e32edaa225aa4eb13c9f6e23b865339`
- Locally mounted result base: `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-target-new-nll-v1/ODE-edit/local/odebf/results`

The server2 SH2 roots were create-once and hashed when produced, but they are not mounted in the current server1 workspace. Consequently, SH2 numerical rows below are explicitly marked as sealed-handoff evidence. No hash was reverse-engineered into a missing value.

## 3. Method evolution and what each stage answered

| Stage | Main design | What it actually tested | Terminal interpretation |
|---|---|---|---|
| SH2 P0 R4 | Joint B10, W64 solve, BF16 virtual/commit | dtype, solver certificate, transaction, rollback, W0 restore | Technical candidate passed both models; no scientific claim |
| SH2 P1 R1 | Warm Native direct-z, pre-shared residual, K8 | Initial sequential B10 feasibility vs Native | Severe no-op/under-edit; non-Native arms far below Native |
| SH2 P1 R2/R3 | Trust-ratio progress, mean-P, fixed outer-entry P, arm-local infeasibility | Whether moving P baseline and all-or-nothing progress caused rejection | Accepted trajectories appeared; terminal cumulative P remained the dominant final false component |
| SH2 P1R4 full residual | Same full residual offered to all layers | Whether pre-sharing suppressed efficacy | Large efficacy recovery, especially Qwen; terminal P still failed |
| SH2 adaptive-tau | Same-state retry, explicit tau, full residual variants | ODE-like adaptive stepping and routing distribution | Llama under-edited before tau=1; Qwen hit a solver boundary after one valid transition |
| SH1 target-new-NLL | Change routing progress loss only | Margin objective vs target-new-only objective | Early constrained prefix favored new-NLL for Qwen; evidence incomplete for final endpoint |
| SH1 functional-P off / all off | Disable P veto, then all preservation decisions | Whether hard preservation veto caused under-edit | Full-tau all-off reached Native efficacy/gen; P drift exposed the trade-off |
| SH1 P-soft/hard | Add functional-P to routing but retain hard gate | Whether local P model can redirect routing | Hard P still stopped trajectories; Qwen soft routing collapsed to one layer in the observed endpoint |
| SH1 cold fixed-E8 P1R7 | Cold z-base, target-new-NLL, 8 fixed Euler steps, structural+functional soft routing, no preservation veto | Intended low-overhead current method | No complete scientific arm; repeated technical fixes ended at solver/harness boundaries |

## 4. SH2 results that determine the current design

### 4.1 Technical foundation: W64 candidate

**SEALED-HANDOFF FACT.** SH2 P0 R4 completed both models with genuine joint B10 rank 10, five edited layers, no W32 fallback, exact W64 virtual-versus-committed parameter bytes/logits/event, exact rollback at writes `[0,2,4]`, and exact final W0 restoration. W64 certificate residuals were below `1e-5` for all layers. Cross-solver N32/W64 bytes were not exact, so the only valid claim was `W64_TECHNICAL_CANDIDATE_NO_NATIVE_EQUIVALENCE_CLAIM`.

This is the correct foundation for later experiments: the transaction path is credible, but Native equivalence and scientific preservation were not established.

### 4.2 Warm-start K8 small-batch failure

**SEALED-HANDOFF FACT.** In SH2 P1R1, Native succeeded while all three non-Native arms selected no accepted state.

| Model | Arm | Efficacy | Generalization | Locality | Accepted/rejected | Main boundary |
|---|---:|---:|---:|---:|---:|---|
| Llama | N32 Native | 10/10 | 20/20 | 89/100 | committed | reference |
| Llama | F_G / F_BF / R_BF | 1/10 | 1/20 | 92/100 | 0/8 | every trial underwrote requested progress |
| Qwen | N32 Native | 10/10 | 20/20 | 88/100 | committed | reference |
| Qwen | F_G / F_BF / R_BF | 2/10 | 3/20 | 89/100 | 0/8 | every trial underwrote requested progress |

The read-only RCA showed positive actual progress in all 144 trials, but only `6.31%` to `40.51%` of the old requested progress. Functional-P raw-max also failed often, but 32 P-passing trials still failed progress; therefore the universal cause was the progress contract, not P alone.

**INFERENCE.** This run did not show that dynamic routing is ineffective. It showed that a linear predicted-progress target was being treated as an exact BF16-realized requirement and that rejected slots repeated the same state.

### 4.3 Fixed-entry P and arm-local continuation

The fixed outer-entry correction removed the moving-baseline mismatch. It allowed accepted trajectories, but terminal cumulative P remained the dominant final false component.

| Model | Arm | Accepted/rejected | Terminal efficacy | Terminal P mean | Final status |
|---|---:|---:|---:|---:|---|
| Llama | F_G | 7/1 | 3/10 | 0.0013347 | terminal-P infeasible |
| Llama | F_BF | 6/2 | 4/10 | 0.0012028 | terminal-P infeasible |
| Llama | R_BF | 6/2 | 4/10 | 0.0030129 | terminal-P infeasible |
| Qwen | F_G | 8/0 | 6/10 | 0.0012966 | terminal-P infeasible |
| Qwen | F_BF | 7/1 | 5/10 | 0.0008233 | feasible, but no exact hit |
| Qwen | R_BF | 7/1 | 10/10 | 0.0012913 | exact efficacy hit, terminal-P infeasible |

**FACT.** At B10-1, H had zero items, so structural and functional H were decision-vacuous.

### 4.4 Full residual instead of Native pre-sharing

Giving every layer the same full residual materially improved efficacy. It also increased cumulative P drift.

| Model | Arm | Accepted/rejected | Efficacy at fixed K8 | Terminal P mean | Status |
|---|---:|---:|---:|---:|---|
| Llama | F_G | 4/4 | 5/10 | 0.0029696 | terminal-P infeasible |
| Llama | F_BF | 4/4 | 5/10 | 0.0033571 | terminal-P infeasible |
| Llama | R_BF | 4/4 | 9/10 | 0.0099319 | terminal-P infeasible |
| Qwen | F_G | 5/3 | 10/10 | 0.0012095 | terminal-P infeasible |
| Qwen | F_BF | 5/3 | 10/10 | 0.0012116 | terminal-P infeasible |
| Qwen | R_BF | 5/3 | 10/10 | 0.0014091 | terminal-P infeasible |

**INFERENCE.** Full residual fixed a real under-edit mechanism. It did not solve the preservation trade-off. This supports keeping full-residual arms while changing preservation from a hard heuristic budget to a soft allocation signal.

### 4.5 Adaptive warm-start result

**SEALED-HANDOFF FACT.** Llama completed the adaptive causal diagnostic; Qwen stopped after one valid accepted transition at a solver-certificate boundary.

| Model/variant | tau | K accepted | trials/rejects | Eff | Gen | Loc | Final P | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Llama PS-S8 | 0.625 | 5 | 24/12 | 4/10 | 8/20 | 82/100 | 0.003292 | fixed-S8 complete, no hit |
| Llama PS-A8 | 0.375 | 3 | 8/5 | 3/10 | 6/20 | 83/100 | 0.001355 | retry/min-dt exhaustion |
| Llama FR-A8 | 0.125 | 1 | 6/5 | 3/10 | 6/20 | 83/100 | 0.001445 | retry/min-dt exhaustion |
| Llama FR-A16 | 0.125 | 3 | 7/4 | 3/10 | 6/20 | 83/100 | 0.001172 | retry/min-dt exhaustion |
| Qwen PS-S8 prefix | 0.0625 | 1 | prefix only | 1/10 | NOT_RECORDED | NOT_RECORDED | 0.0008958 | next field solver failed in original run |

Later observability reproduced Qwen's candidate prefix but did **not** reproduce the next-field failure, so the boundary was classified `NONDETERMINISTIC_OR_INSTRUMENTATION_SENSITIVE`, not a stable scientific effect.

Routing did not initially collapse in Llama FR-A8: shares were `[0.1103, 0.1653, 0.2303, 0.2276, 0.2666]`, top-1 `0.2666`, effective layers `4.64`. Llama's final adaptive summaries remained distributed (top-1 about `0.238` to `0.264`, effective layers `4.62` to `4.92`). Qwen's initial field was more concentrated (top-1 about `0.360`, effective layers `3.50`) but not a one-layer collapse.

## 5. SH1 causal results

The following table is **LOCAL-VERIFIED FACT**, extracted from mounted terminal and stepwise receipts. Metrics are the final accepted endpoint available for each arm. These are reused-sample causal diagnostics, not promotion results.

### 5.1 Baselines

| Model | W0 eff/gen/loc | N32 Native eff/gen/loc |
|---|---:|---:|
| Llama | 2/10, 3/20, 84/100 | 10/10, 20/20, 79/100 |
| Qwen | 0/10, 4/20, 80/100 | 10/10, 20/20, 80/100 |

### 5.2 Routing-loss and preservation interventions

| Model | Design/arm | tau | Eff | Gen | Loc | Functional-P mean | Top-1 coefficient share |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama | Margin, hard P | 0.125 | 3/10 | 6/20 | 83/100 | 0.001445 | 0.267 |
| Llama | New-NLL, hard P | 0.125 | 3/10 | 6/20 | 83/100 | 0.001752 | 0.281 |
| Qwen | Margin, hard P | 0.03125 | 1/10 | 6/20 | 81/100 | 0.001034 | 0.424 |
| Qwen | New-NLL, hard P | 0.0625 | 4/10 | 7/20 | 80/100 | 0.001259 | 0.411 |
| Llama | New-NLL, functional-P off, structural P on | 0.5 | 9/10 | 18/20 | 80/100 | 0.018019 | 0.331 |
| Llama | New-NLL, all preservation off | 1.0 | 10/10 | 20/20 | 80/100 | 0.082328 | 0.369 |
| Qwen | New-NLL, functional-P off, structural P on | 1.0 | 10/10 | 20/20 | 80/100 | 0.002675 | 0.524 |
| Qwen | New-NLL, all preservation off | 1.0 | 10/10 | 20/20 | 80/100 | 0.002500 | 0.402 |
| Llama | New-NLL, P-soft routing + hard P | 0.125 | 3/10 | 6/20 | 83/100 | 0.001149 | 0.365 |
| Qwen | New-NLL, P-soft routing + hard P | 0.015625 | 1/10 | 6/20 | 81/100 | 0.001450 | ~1.000 |

Key deductions:

1. **Hard P is a major progress limiter.** Removing only functional-P veto moved Llama from `3/10` at `tau=0.125` to `9/10` at `tau=0.5`; removing all preservation vetoes reached `10/10` at `tau=1`.
2. **The P metric is not meaningless.** At full tau, Llama's mean positive P damage rose to `0.0823`, far above the old `0.001` budget. The correct conclusion is not “ignore P,” but “do not turn an uncalibrated heuristic budget into an all-or-nothing step veto.”
3. **Target-new NLL helps some prefixes but is not yet proven superior at the final endpoint.** Qwen improved from `1/10` to `4/10` under the constrained prefix. Under all-off full tau, both margin and target-new-NLL reached `10/10, 20/20, 80/100`.
4. **Soft routing with a retained hard gate did not test the desired no-veto method.** It still stopped at the hard P boundary. Qwen also showed a near-single-layer coefficient allocation, so stepwise concentration telemetry is mandatory.
5. **Locality did not collapse in these B10 causal runs.** Full-tau all-off locality was `80/100`, equal to Qwen Native and one point above Llama Native. This does not establish lifelong preservation; it only shows that this one B10 did not suffer catastrophic locality loss under the official local panel.

## 6. Latest cold fixed-E8 P1R7 review

### 6.1 Intended method

The frozen design at `7faa432` is:

- cold `z_base` start; no Native direct-z in the controller;
- target-new suffix NLL objective;
- layer-local full residual on layers `[4,5,6,7,8]`;
- fixed Euler grid `K=8`, `h=1/8`, target `tau=1`;
- no adaptive retry, backtracking, or first-hit stopping;
- structural and functional H/P as normalized **soft routing scores**, not scientific vetoes;
- progress, BF16 transaction, and write trust retained as technical constraints;
- two arms: `E8-NEUTRAL` and `E8-SOFT`.

Source anchors in current HEAD:

- constants and tolerances: `project/run_scripts/ode_bf/fixed_e8_soft_routing.py:31-45`;
- independently recomputed certificate: `fixed_e8_soft_routing.py:563-687`;
- SLSQP invocation/continuation: `fixed_e8_soft_routing.py:690-804`;
- progress target `requested = 0.25 * p_max`: `fixed_e8_soft_routing.py:988-1009`;
- neutral minimum-capacity solve: `fixed_e8_soft_routing.py:1012-1031`;
- soft stage-1/stage-2 solves: `fixed_e8_soft_routing.py:1032-1161`;
- velocity selection after all solves: `fixed_e8_soft_routing.py:1162-1170`;
- fixed-grid first-hit observation-only receipts: `project/run_scripts/ode_bf/fixed_e8_runtime.py:1613-1635`.

The planned functional-basis cost is 48 endpoints per arm because each of 8 fields evaluates one baseline plus five one-layer probes: `8 * (1 + 5) = 48`. Empty history adds no H probes. This is still substantial, but it is bounded and far below the old 128-trial adaptive ceiling.

### 6.2 Technical repair sequence

The first P1R7 pair failed before model load because a rooted object used `schema` while the generic loader expected `schema_version`. Subsequent children repaired provenance, launcher token parity, numerical-lock schema, GPU identity, zero-capacity entry handling, and fail-before-receipt observability. These were plumbing repairs and supplied no scientific evidence.

### 6.3 R6 numerical boundary

R6 ran at source `f23472963ac2dbb46282afd9c3a514bb75d59d5b`.

| Model | Job | Reached | Failing phase | SciPy success | Primal | Stationarity | Complementarity | Locked result |
|---|---:|---|---|---:|---:|---:|---:|---|
| Llama | 17117 | E8-NEUTRAL step 6 | soft min-capacity within xi tie | true | 1.94e-13 | **1.1091e-5** | 2.35e-10 | fail, KKT tol 1e-5 |
| Qwen | 17118 | E8-NEUTRAL step 1 | neutral min-capacity | false | 5.02e-9 | 2.43e-11 | 5.48e-7 | fail only on optimizer success flag |

Locally rehashed R6 artifacts:

- Llama failure: `f2c2afd42a7fb124ea94015e93e83cce39bf36f1fc3696e7bed3c8af02b0e1da`
- Llama failing solver receipt: `6a12310c914d365cf3e73b97932a58d721bdab86e8434416da9af825042ae0cf`
- Qwen failure: `d750f84fc7c0b617abe717284201cad717329c8d28a947759dae74fda7d5dae7`
- Qwen failing solver receipt: `98089243a9a5a12271ced5e58565e6bf64b611d42d8a756007a2702e3c48f09f`

### 6.4 R7 repeated-continuation result

R7 at `a90756b` added up to two same-QP SLSQP continuations without changing objective, constraints, or tolerance. It did not fix either boundary.

| Model | Job | Total optimizer passes | Status history | Primal | Stationarity | Complementarity | Outcome |
|---|---:|---:|---|---:|---:|---:|---|
| Llama | 17131 | 4 | `[0,0,0,0]` | 1.76e-13 | **1.7101e-4** | 2.13e-10 | still fails stationarity; worse than R6 |
| Qwen | 17132 | 3 | `[8,8,8]` | 5.02e-9 | 2.43e-11 | 5.48e-7 | unchanged status-8 failure |

Scheduler facts: Llama failed after `00:29:25` with batch MaxRSS `7085.50M`; Qwen failed after `00:02:15` with MaxRSS `10344776K`. Both stopped before a terminal panel.

Locally rehashed R7 artifacts:

- Llama failure: `961ef579cd57a9c451ae33968316b2393b126465d661883eb76f7f26a22e009e`
- Llama failing solver receipt: `8568aa6963b4c53f84bb3cfd43aeb8da5fc7e399a19335667a1e586db5c1d1f6`
- Qwen failure: `4adb517da590db015d03b14e108a4733351be889389d27eb17f3eb35ea8760df`
- Qwen failing solver receipt: `fce7cdf35131fbfa25e4e4c7653b12a5690be2833a1709d99958738754efbd18`

SciPy status 8 is “Positive directional derivative for linesearch” in the pinned SciPy source. Repeating SLSQP from its own output reproduced the same status and point.

### 6.5 Available R7 scientific prefix

Only the Llama neutral prefix contains meaningful non-terminal scientific telemetry. Gen/loc are intentionally held until action freeze and are therefore **NOT_RECORDED** for this incomplete run.

| Accepted index | tau | Online efficacy | Functional-P mean (observation) | Structural-P soft value | Trust value | Neutral top-1 share | Effective layers |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.125 | 0/10 | 0.000377 | 0 | 0 | zero write | undefined |
| 2 | 0.250 | 0/10 | 0.000488 | 0.000013 | 0.00511 | 0.879 | 1.28 |
| 3 | 0.375 | 0/10 | 0.000806 | 0.000108 | 0.02146 | 0.679 | 2.02 |
| 4 | 0.500 | 0/10 | 0.000746 | 0.000319 | 0.02846 | 0.594 | 2.45 |
| 5 | 0.625 | 3/10 | 0.001115 | 0.000656 | 0.03422 | 0.521 | 2.88 |
| 6 | 0.750 | 5/10 | 0.002310 | 0.001134 | 0.03956 | 0.451 | 3.32 |

This prefix is informative but not a method result:

- edit efficacy starts improving only after `tau=0.5`;
- the early neutral routing is strongly concentrated and gradually spreads across layers;
- P damage rises with progress but is observation-only as intended;
- neither `tau=1` nor the E8-SOFT arm is reached;
- terminal gen/loc, cold-vs-Native floor, and soft-vs-neutral endpoint contrast are **NOT_RECORDED**.

Qwen accepted one zero-write recovery grid point at `tau=0.125`, with efficacy `0/10`, then failed at the next authoritative neutral solve. No Qwen cold edit outcome exists.

## 7. Critical harness boundary

`solve_fixed_e8_routing` computes the neutral solution, then unconditionally computes soft stage 1 and soft stage 2, and only afterward chooses which velocity to return. Therefore `E8-NEUTRAL` has a hard dependency on the numerical solvability of an observation-only soft shadow.

This is exactly what kills Llama R7:

- the authoritative neutral minimum-capacity certificate at step 6 is not the reported failure;
- the failure is `soft-minimum-capacity-within-xi-tie`;
- the returned velocity would have been `pre_soft_velocity` for `E8-NEUTRAL`;
- the soft-shadow decision influence count is zero, but its certificate failure aborts the arm.

This dependency is not required by the stated matched-compute contract. The contract requires identical field, functional probes, evaluator schedule, and candidate opportunities; it does not require a baseline to fail because a non-selected comparison solver failed. If identical solver work is desired for accounting, the soft shadow can still be attempted and recorded, but it must be arm-local diagnostic state for `E8-NEUTRAL`.

Qwen is different: its failure is in `neutral-minimum-capacity`, which is authoritative for both arms. It cannot be bypassed as a shadow failure.

## 8. Recommended numerical policy

### Priority 0: preserve scientific semantics

Do not change `K=8`, `h=1/8`, `kappa=0.25`, objectives, constraints, score normalizations, cases, P population, or the locked tolerances (`1e-8` primal and `1e-5` KKT). Do not add trajectory retries or model-dependent tuning.

### Priority 1: split authoritative and shadow solves

Refactor the current solver into two typed paths:

1. `solve_neutral_authoritative`: maximum progress + neutral minimum capacity; once certified, it can produce the neutral candidate.
2. `solve_soft_authoritative`: reuse the same field/probe inventory, then solve soft stage 1 and stage 2.

For the neutral arm, a soft-shadow failure must serialize `SOFT_SHADOW_NUMERIC_UNAVAILABLE` and have zero candidate/clock/endpoint influence. It must not suppress the already certified neutral candidate. For the soft arm, the same failure remains fail-closed.

### Priority 2: make the independent certificate authoritative

Change optimizer status from a hard predicate to diagnostic metadata. A candidate is numerically accepted iff:

- all values are finite;
- independently recomputed maximum primal violation is `<=1e-8`;
- independently recomputed stationarity and complementarity are each `<=1e-5`;
- the signed-progress and xi-tie conditions are included in those constraints.

Under this policy, the Qwen R7 point passes. Its signed progress is below the requested value by `5.0188e-9`, which is exactly the recorded primal violation and is within the existing `1e-8` tolerance. SciPy status 8 remains visible but does not overrule the mathematical certificate.

This is not a tolerance relaxation. It removes a redundant backend-specific boolean from a separately certified mathematical result.

The next receipt schema must also make that certificate independently auditable. Persist the small coefficient vector, every ordered constraint slack, and the nonnegative least-squares KKT multipliers (or an equivalent canonical residual payload) in raw-free form. The immutable R7 receipts contain only summary residuals, so their certificate cannot be reconstructed independently after the run; the Qwen conclusion above is a source-and-receipt review of the recorded certificate, not a retroactive endpoint promotion.

### Priority 3: replace repeated SLSQP continuation for genuine KKT misses

R7 proves that “run SLSQP again from the same point” is not a reliable recovery policy. For a finite point that fails the independent certificate, run one deterministic secondary solve of the **same** tiny convex program:

- recommended immediate backend: SciPy `trust-constr`, with analytic objective Jacobian/Hessian and analytic constraint Jacobians/Hessians;
- seed: the SLSQP candidate;
- same bounds, objective, progress target, trust constraint, xi tie, and tolerance;
- no field rebuild, model call, candidate evaluation, tau advance, or scientific retry;
- accept only if the existing independent certificate passes;
- serialize backend, iterations, objective value/gap, constraint residuals, and KKT residuals;
- if both backends fail, fail-close as `NUMERIC_QP_UNCERTIFIED`.

Lock the SciPy version, constraint ordering, derivative definitions, options, and seed, and require a dry-repeat identity test. Calling the backend “deterministic” is justified only after that repeat gate passes. Implement the constraints with fixed `LinearConstraint`/`NonlinearConstraint` ordering and analytic Hessians so the fallback is genuinely the same program rather than a numerical reformulation with altered semantics.

An enumerated active-set solver is a stronger long-term option because the dimension is only five (six with xi), but it is more implementation work. `trust-constr` is the minimal independent diagnostic backend available in the pinned environment.

### Priority 4: validate certificate geometry on CPU before another pair

Add frozen tiny fixtures covering:

- status-8 but independently certified acceptance;
- soft stage-2 with progress and structural-P xi-tie both active;
- neutral result byte/number identity with soft shadow enabled vs disabled;
- trust-constr recovery passing the unchanged certificate;
- canonical coefficient/slack/multiplier receipt reconstruction and dry-repeat identity;
- both backends failing without tolerance relaxation;
- shadow failure never changing the neutral state, target, samples, RNG, or clock.

Then submit one distinct pair only. Do not spend another GPU pair on additional identical SLSQP continuations.

## 9. Claim boundaries

### Supported now

- Joint B10 W64/BF16 transaction mechanics are technically sound on both models.
- Full residual is better than Native-style pre-sharing for edit progress in these pilots.
- Hard functional-P at an arbitrary `0.001` threshold causes severe under-edit.
- Full-tau routing can recover Native-level efficacy/gen on the tested B10 when preservation vetoes are disabled.
- Functional-P damage grows substantially along those trajectories, so preservation cannot simply be removed from the final method.
- Routing can become concentrated; stepwise layer telemetry is necessary.

### Not supported now

- cold fixed-E8 scientific superiority;
- any E8-SOFT benefit over E8-NEUTRAL;
- Native-level efficacy/gen/loc for the cold method;
- lifelong or large-scale sequential preservation;
- historical-H benefit;
- calibrated structural or functional preservation budgets;
- formal continuous-time CBF guarantees;
- final first-hit stopping policy.

## 10. Plain-language summary

The experiments started with a controller that was too cautious: it often refused edits before they had a chance to work. Giving every layer the full edit residual and letting the router divide it helped a lot. When the preservation veto was removed, both models could complete all eight steps and match Native edit success on the tested batch, but the pretrained-distribution drift—especially for Llama—became much larger.

The new method tries to keep the useful parts: start without Native direct-z, take exactly eight Euler steps, and use preservation as a soft routing preference instead of a hard stop. That is a reasonable direction. However, its real-model run has not finished yet because the small routing optimizer fails at numerical boundaries. One failure is partly a harness mistake: the neutral baseline is being killed by an unused soft-routing solve. The other is SciPy reporting failure even though our own numerical certificate passes.

So the next step is not to change the method or loosen safety thresholds. It is to separate the neutral and soft solver paths, trust the explicit KKT certificate, and use an independent deterministic solver only when the certificate truly fails. After that, rerun one Llama/Qwen pair and finally measure the intended cold fixed-eight-step method.
