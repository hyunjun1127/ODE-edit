# Local-z adaptive allocation (cold W0, seven fixed policies)

Instruction: `ODEEDIT-S06-LOCAL-Z-ADAPTIVE-ALLOCATION-SEQ1000-SH4-V1`.
This package launches neither the warm donor runner nor an EP controller.

## State and science

All arms start from the same pretrained FP32/eager W0 and zero M4/M8,
seed 20260916, both TF32 switches off. A preparation process runs the native
W0 context generator once, seals text plus tokenizer IDs and post-context RNG,
checks the existing teacher against this W0, and saves a shared cold capsule.
Teacher replacement is conditional on the fixed 5e-7 reproduction threshold;
only the same S64/Dev128 IDs/tokens can be regenerated.

`NativeSingletonFitter` and native `compute_z` are read-only. LOCAL uses the
writer's local target/readout. `terminal_writer` extracts the original native
K/readout/repeat/direct-solve/add statements, explicitly setting readout L8
for both writers. It does not inject Z8 into a local4 residual. Terminal targets
are computed once at entry. Gates use CPU FP32 endpoint differences; 0/1 copy
endpoints exactly. Original native remaining-layer RHS scaling is a technical
reference only, with its rounding difference retained.

Each proposal fit's input state, layer, contexts, hparams and source identify
its episode-local cache scope. Only a8 variants at one a4 share a fit. No cache
survives the batch. Every fit/score starts at the sealed batch-entry RNG;
commit returns RNG to that entry value. Native model is eval, no dropout is
introduced. Thus evaluation/order/cache hits do not advance the next RNG.
Candidate endpoint W is applied to actual parameters at every token.

E is native training target-new NLL, token then all rewrite-context then
request mean. H and strict sets are canonical rewrite observers. Past64 uses
all *received* events, latest event per raw subject/relation, excludes current
overwrite facts, and uses the fixed SHA priority. No accepted-only ledger.
P/N, Dev, and future requests are not controller arguments. Nonfinite candidate
state is a technical failure, never quality-infeasible fallback.

History is untouched during all fits. Final native finalizer executes once
per eligible physical layer at selected W. LD/TD append M8 even for zero L8
write or common N4. Fixed arms receive no dynamic quality gate.

## Technical and execution boundary

CPU tests cover controller, Past64, FP32 gates, negative nonfinite cases,
seven proposal inventories and restore. They are not model PASS.
The preparation job then checks actual same-entry local/native connections,
terminal/nonblue source and update connection, repeated candidate E/H/D and
strict sets in reverse evaluation order, final history and same-process
save/restore. It performs no FD/ULP/KKT/gradient experiment.
Technical comparators explicitly replay captured targets into the unchanged
native apply function; they are not new baseline sequential chains.
Different terminal endpoint-vs-RHS FP32 rounding is reported, not required
to equal historical bytes. GPU off/on continuation remains NOT_TESTED.

Science is a single array `%1`, afterok the common preparation and fail-closed
on its exact source/lock READY. Mapping: 0 LD, 1 N4, 2 REFIT4, 3 L75,
4 T75, 5 L4D, 6 TD. All seven are pre-registered; no performance filtering.
At B2 entry a durable initial marker binds the prior B1 commit. All B1–B10
work runs autonomously without agent callbacks.

## Storage and accounting

Save every native fitted endpoint and target capture, candidate score/strict
sets, selection/shadows, entry/current and selected/current rows, B5 full500,
B10 full1000 with identity subsets, Dev128 B5/B10, actual delta journals and
B1/B5/B10 selected W4/W8/M4/M8/context/RNG/received-ledger snapshots.
No full pretrained model copies. Candidate weight pairs reconstruct from
saved fits, gates and entry; FP32 delta journals alone do not establish exact
replay. CPU checkpoint reload is separate from GPU continuation.
The reserved disk allowance is 212GiB including conditional teacher12GiB;
12h per process is an estimate with reserve, not an efficacy or GPUh gate.
Native target/loss/actual Adam return-frame counters are captured without
changing optimizer code. Per-iteration clamp-hit counts are NOT_RECORDED;
final anchors/deltas/radii are retained. Nested fit/total timers must not sum.

## Commands

```bash
python -B -m unittest project.run_scripts.local_z_adaptive_allocation.test_local_z -v
python -B -m project.run_scripts.local_z_adaptive_allocation.control check --worktree "$TASK_WORKTREE"
python -B -m project.run_scripts.local_z_adaptive_allocation.control freeze --worktree "$TASK_WORKTREE"
python -B -m project.run_scripts.local_z_adaptive_allocation.control submit --worktree "$TASK_WORKTREE"
```

Freeze and submit are create-once, not retry loops. The fixed Python/dependency
environment is in `run.sbatch`. Resource admission includes existing project
reservations, technical/preparation and the array in **one** cap1 lane.
Pause after actual initial dynamic gate or PENDING/HOLD handoff; no automatic
completion monitoring, follow-up submit, or terminal report.
