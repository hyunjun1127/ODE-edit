# R512/G256 matched B1 — source readiness

Scope: Instruction ODEEDIT-S06-EN-EXECUTION-REUSE-R512-G256-B1-SH4-V1. Preparation and the two primary schedules are separate frozen executions. This is source/CPU readiness, not actual Llama parity or a B1 result.

The new matched runner calls the unchanged EN-F optimizer with independent per-arm gradient and trial observations. Both use the same generated-reference adapter and unchunked full-T vocabulary head, CPU FP64 document-order accumulation, exact current target-union heads and full-token16-position invariant heads. No KV, GPU-accumulator, head-chunking or geometry-reduction substitution was added. Reference preparation source a297039dc756a0e8953e4bab4695e66361a161ac remains immutable.

New code covers W0 generated-teacher/cache binding, native-B1 read-only reuse, immutable shared current keys/geometry, legacy/reuse schedules, bounded actual physical AD/Current checks, checkpoint/history, postseal canonical/Dev observers and CPU raw-NLL reduction. Two timed schedules do not reuse gradients/candidate decisions. Prior same-host W0/N4 canonical raw is bridged only after exact model bytes, inputs/tokens, native endpoint, evaluator source/layout and runtime conditions match. Its prior cost is retained separately. New generated256 Dev is not an old Dev metric alias.

`max_batches=1`, `sequential_authorized=false`, `auto_continue=false`; no B2 dispatcher, dependency or callback. Native reuse means new fit0, not a newly measured native timing. Each N4/legacy/reuse endpoint gets its own history1 checkpoint. The checkpoint contains W4/M4, context, serialized RNG, received ledger/registry, order, source and teacher identity. CPU reload and selected physical copy do not certify GPU continuation.

## Independent bounded review and fixes

Workers owned endpoint/observation, generated teacher/reference and generated-oracle components. Their CPU fixture results are not actual model evidence. Subsequent separate read-only/API audit found: scalar-only loss comparison, normal empty-space false mismatch, and an overbroad Dev access0 statement. These were corrected: every full-bank sweep includes document-row digest/coverage; legal empty/unresolved-space outcomes explicitly have no gradient coverage; immutable Dev metadata/cache loading is distinguished from online evaluation.

Independent transaction tests exposed insufficient restored metadata verification, output-directory confinement, cold-entry timing and exception rollback. Parent corrected all four. Tests exercise actual in-memory torch serialization/weights_only and native-finalizer API with a CPU stub, not filesystem durability or GPU execution. Actual atomic fsync/mmap, complete model binding and bounded physical checks remain in the B1 program.

Current focused/integrated CPU suite:142 tests passed in39.298s (`local/en-execution-reuse/20260919-v1/receipts/cpu-matched-r8.json`); source-bound receipt is checked again at the matched freeze. Earlier140/r5 and142/r7 remain immutable evidence. The suite includes real unchanged optimizer + toy suffix/session/guard/invariant wiring, full512 synthetic document gradient paths, source/state corruption negatives, strict/finite/cardinality reducer checks, evaluator-reuse compatibility negatives, full synthetic publication/accounting/GFM fixtures and B2 denial. No new numerical waiver. Actual generation/TF parity is preparation evidence only; the matched gradient/candidate/selection comparison is not yet run.

Finite candidate model overflow retains the original optimizer backtracking category only after the fixed teacher bank is independently checked in full. The generated oracle preserves partial coverage/cost and validates endpoint/prefix exit state before raising a dedicated typed exception. Parent validation errors, corrupt teacher, nonfinite gradient, OOM or wrong state remain technical failures. Interrupted trial coverage is never marked as a completed512 sweep. These are CPU negative checks, not actual overflow observations.

## Timing and storage boundaries

Reference setup/prefix loading/teacher checks/native-reuse binding/shared geometry/bounded checks are outside each timed controller. Schedule entry reset/hash overhead is separately recorded. Timed controller includes native anchor, endpoint session, one independent gradient sweep, all attempted trials, evidence writes and session close; checkpoint/history/official observers are separate. Counters retained from before a reset are explicitly marked DO_NOT_ADD. All nested component timers are nonadditive.

Final scientific comparison requires exact candidate/decision/endpoint parity, not merely protection tolerances. Per-document loss hashes, G/H, chi/eta, all trials and selected endpoints are compared. B1-only measurements are not stable speedup distributions, general locality claims or sequential authorization.

The fixed order is legacy then reuse; filesystem/page caches are not flushed. Raw wall differences therefore include order/cache-warmness effects and are not all assigned causally to dedup. Current setup/native/geometry and bounded AD gradients are separately accounted. The updated matched-stage reserve explicitly adds0.875GiB for the two bounded physical/cached AD gradient tensors; the original preparation lock/estimate is preserved. This is a storage-accounting clarification within the existing operational margin, not a new scientific probe or model run.

Python Markdown/Markdown-it/CommonMark renderers and pandoc were not installed at readiness. The new publication checker reports GFM/CSV/link checks separately and actual HTML rendering as NOT_RUN_NOT_INSTALLED; it never assigns renderer PASS from table parsing alone.

No prior task jobs, raw, checkpoints or shared runtime were changed; no large remote transfer. NO_BROADCAST_NOT_REQUIRED: same-host inputs and local output.
