# D/S 48 GPUh release — execution resource preparation

Status: PASS_CPU_FOR_LAUNCH_BINDING; no actual submission by this reviewer.

## Authority full read

The resource reviewer independently read all73lines of
`/mnt/raid5/janghj/.codex/worktrees/odeedit-gh-alpha-jv-diag-dispatch-20260907-v1/messages/head/2026-09-07-alpha-jv-ds-budget48h-release.md`.
Observed regular non-symlink, mode0664,4400bytes, ownerjanghj:janghj;
SHA256`54789342ea78151442a09ec07696db9ce80caae6f39a665c2ca4568b881a3c15`.
The authority is **48GPUh=172800GPU-seconds combined D+S**, not per cell/track/model.
Prior null-budget receipts remain immutable; this addendum changes admission,
not science or prior evidence. Stock z, original λ/T/N grid and D/S independence
are unchanged.

## Source and ownership

Only the new task package receives `launch.py`, `server1_sweep.sbatch`,
`test_launch.py`. Existing scientific runtime and shared source were not edited.
The preparation helper never calls a scheduler. SH1 alone performs real live
resource checks, held submission, inspection, registration and release.

`preparelaunch()` takes an exact committed clean repository, fresh create-once
attempt root, user budget lock, sample/assets/CPU check locks, campaign registry,
and any prior accounting snapshot. It seals copied locks and source HEAD/tree,
new-package member hashes, input bindings, source script and launch identity.
An uncommitted production `gpu_runtime.py` or dirty tracked source is refused.
It emits a suggested held command only, never executes it.

`register_submission()` is called after SH1 obtains actual held job IDs and
inspection/live-admission receipt hashes. It writes a create-once exact mapping
for both array tasks, task owner/UID/name, source path/HEAD and namespace. The
registration is not inferred from the job name alone.

`validate_launch()` runs before model loading and verifies clean queued source
HEAD/tree, new source member hashes, all lock bytes/root, current48h authority,
runtime array task mapping, ownerUID, job name and exact source/run paths.
It returns PREMODEL_MECHANICAL_PASS, explicitly not GPU fidelity PASS.
The runtime must still perform asset/fidelity checks mandated by the contract.

## First S wave reservation

| Item | Locked value |
|---|---:|
| Mapping | cell0Llama,cell1Qwen |
| Array | 0-1%2 |
| GPU/process | 1 |
| Node | devbox |
| CPUs/task | 8 |
| Explicit job-total memory | 182272M |
| Wall limit/cell | 8h=28800seconds |
| Maximum first-wave reservation | 16GPUh=57600GPU-seconds |
| Initial charged consumption | 0GPU-seconds |
| Initially unreserved remainder | 32GPUh=115200GPU-seconds |
| Environment/requeue | --export=NONE / --no-requeue |

Actual isolated availability is not reserved by preparation. Immediately before
submission SH1 runs the repository cap/memory helper; active, configuring and
uncleared completing allocations include grandfathered runs. No existing run
is changed or preempted. Logs are directed to the unique ignored attempt root.

## Charged ledger, queued reservations and malformed evidence

`reconcile_accounting()` only consumes exact registered task IDs and supplied
read-only scheduler snapshots. Required sacct columns are
`JobID|User|JobName|State|ElapsedRaw|AllocTRES|TimelimitRaw`; use `JobID`, not
potentially differently formatted `JobIDRaw`, for array task identifiers.
The parent can request `-X -P -n` with wide JobID/JobName columns. The live snapshot
is `%i|%T|%M` for these same exact IDs. This reviewer executed neither command.

Every attempt's elapsed allocation time charges the same campaign, including
OOM, failure, cancellation and diagnostics. A cancelled accounting record still
present as live COMPLETING is charged at the greater accounting/live elapsed
time, not assumed released. Technical denominator exclusion never excludes cost.
Unallocated PENDING elapsed0 consumes0 but reserves its full requested wall time.
Running reservations retain only their remaining worst-case wall time, avoiding
double counting already charged residency. Step rows cannot duplicate charges.

Missing registered jobs, incomplete/failed queries, unexpected ownership or
namespace, GPU count mismatch, malformed elapsed/time limits, unresolved live
states, and inconsistent totals fail closed. Failed-before-allocation records
with no sufficient allocation evidence are unresolved, not guessed zero. New
two-cell worst-case reservation plus charged and outstanding reservations must
fit172800seconds. There is no automatic retry or budget enlargement.

## Focused CPU evidence

`python3 -m unittest -q project.run_scripts.alpha_jv_llama_diagnosis.test_launch`
passes **11/11tests**, covering48hcombined/16hfirst reservation, technical charge,
queued-zero consumption, uncleared COMPLETING, malformed/missing/owner accounting,
full ledger budget arithmetic, duplicated steps, create-once registration,
runtime task/owner/source checks, queued source drift and uncommitted source.
`bash -n server1_sweep.sbatch` and module py_compile PASS.

The tests use synthetic temporary files, mocked Git source metadata and supplied
scheduler text, not actual Git commits or scheduler commands. Model load/GPU/
Slurm submit or cancel actions by the reviewer remain0. Parent performs actual
full CPU integration and source/asset checks before any launch.
