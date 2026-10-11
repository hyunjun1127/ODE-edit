# Qwen FE current-entry-z held-out stabilization

Authority: `USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1`.
Latest execution ownership: **rent / requester root**. SH1 owns this shared source only;
server1/server2 new submissions are held by the latest user placement correction.
No jobs are submitted by this package. README main-results table is not
modified. This package has no scheduler submitter, retry or monitoring daemon.

## Scientific lock

`contract.SPEC` is locked before results. Four cold Qwen CF chains use fixed10k
`[4000:6000]`, BS100 × 20, L2 `0, .1, 1, 10`. Main `[0:2000]`, historical
alternative `[2000:4000]`, and selection IDs/order are checked independently.
`L2=0` is a **held-out control only**, never a benchmark promotion candidate.
Only `.1, 1, 10` satisfying finite W20 E≥95/G≥85/Loc≥70 may be selected by raw
harmonic Score, tie smaller L2. If all tuned arms score below the valid control,
or none meets the gate, no promotion occurs. A strictly identified numerical
failure counts as −infinity. Incomplete/resource/schema/transport errors block
selection; they are not quietly discarded. A failed control needs its factual
failure receipt, not invented W20 data. This is a practical selection criterion,
not a performance or paper-reproduction guarantee.

Author FE commit `478134dfb24b43f4e18b47e8500893ce3f9cc50f` is cloned to a new
path. Original `main.py` model loading and `batch_edit`/C0/key/history/LU/update
functions are preserved. The prior reviewed `TargetSupplier` computes the current
100 targets at L4 and firstforward-replays later layers on **the same current
batch-entry model**, before writing. No W0 fallback or future-request targets.
Author BF16/default attention, L4–8/clamp1/steps35/lr.5/decay.001/KL.0625/C0=15000
remain unchanged. Only native `cfg.algs.L2` differs among arms; CPU tests execute
the actual native writer and compare its solved update with the L2 system.

Storage-only change: native final full-model `save_model` is replaced with the
observer finish receipt. W0 factual is observed once without a checkpoint;
current100 every batch, all-seen at 500/1000/1500/2000. Raw/query evidence is saved
before latest checkpoint replacement. One selected-W/native-H/RNG/context/cursor
`latest.pt` per run, host-shared lock acquired before allocation/serialization of
the single atomic temporary. No old checkpoint/source/output is modified.
Latest-only resume **state** is retained; this package does not auto-resume or
claim pretrained GPU resume parity.

`storage.py` preserves the existing checkpoint payload/digest format but hashes
contiguous CPU bytes via 8 MiB memoryview windows. Synchronous serialization borrows
the already-CPU native H (no extra full-H clone); selected GPU weights alone are
copied to CPU. Chunked finite checks avoid a full-H boolean temporary. Small BF16,
FP32, NumPy/RNG fixtures verify exact digest equality and reader compatibility.
Checkpoint receipts record actual process max RSS; fitting/solver memory is not
changed or claimed to be GPU-qualified. The SH2 request is bounded by its actual
60416 MiB policy, not the historical SH1 96 GiB request.

## Host binding / API

Use the exact published source commit on a clean non-main checkout. SH2 supplies
one local JSON `bindings` (not credentials), now bound by the rent execution owner:

* `server`: `rent` (server1/server2 remain supported for compatibility, not current submission authority);
  `workspace`, `python`, `author_clone`, `tracking_env_file`,
  `checkpoint_lock`, `hf_cache`: absolute own-host paths. `author_clone` must be
  a Git repository containing the pinned commit, **not a frozen archive**.
* `fixed_stream`, `main_stream`, `alt_stream`, `model_assets`: exact
  `{path, bytes, sha256}` members. Main and alternative are existing original
  streams, not silently rewritten copies. `model_assets` follows the existing
  `model.snapshot/model.identity/model.members` manifest contract.
* `C0`: ordered five `{layer, original:{path,bytes,sha256}}` entries L4–8,
  existing original uncentered moments. Fresh full SHA checks precede use.
* `runtime`: `{"torch":"2.9.1+cu128","transformers":"4.51.3"}`; `cpus`: explicit
  1–12, matching Slurm. Host-local HF cache must resolve Qwen main to revision
  `a09a35458c702b33eeacc393d103063234e8bc28` and the verified snapshot.
* `reserve_bytes`: existing approved reserve (minimum 32 GiB rent/server2 / 64 GiB
  server1); `planned_latest_count`: ≥4 (use 6 to budget the two possible later
  benchmark runs); `raw_margin_bytes`: ≥2 GiB, choose an adequate admission
  budget (16 GiB used in SH1 planning). No reserve reduction is implied.

Stage admission credits only already allocated, unique, manifest-bound latest
files under the same host lock; it does not double-reserve those bytes when a
later queued arm starts. Future latest slots, one temporary, raw margin and the
full host reserve remain budgeted. Exact resource grants/Slurm dependencies are
SH2 adapter responsibilities and are not inferred from the task cap exception.

```bash
python -m official.diagnostics.fe_qwen_current.prepare NEW_ATTEMPT BINDINGS.json
# Per GPU job, from frozen source directory, after owner held inspection:
python -m official.diagnostics.fe_qwen_current.launch NEW_ATTEMPT/configs/L2-0p1.json NEW_ATTEMPT/lock.json
# Once all four arms are terminal: CPU-only, no automatic benchmark submission
python -m official.diagnostics.fe_qwen_current.collect NEW_ATTEMPT
```

Config names: `L2-0`, `L2-0p1`, `L2-1`, `L2-10`. `prepare` emits the selection
lock, native patch diff, immutable four configs, source archive/lock, model/C0
bindings and storage receipt. `launch` verifies those bytes before exec of the
author main.py. The execution owner owns resource admission/registration/job receipts.
The task-only cap exception does not authorize unrelated job/cap mutations.
Rent uses the existing official transport's `ODEEDIT_RENT_JOB_ID` and Kubernetes
identity binding; it does not fabricate Slurm IDs or change the logger.

`collect` validates W20 raw token/NLL/correctness/work using separately retained
token query plans, independently recomputes request-macro E/G/Loc/Score, verifies
20 commits and final CP SHA, then writes immutable `selection-result.json`.
Non-numeric errors/missing files fail closed. CF/zsRE benchmark execution remains
a separate **conditional next stage** after a valid selection, not four extra
benchmark replications. No benchmark job is created by the collector.

## W&B separation and observability

Reuse `official.tracking` unchanged. `task_id` (SDK group) is
`fe-qwen-current-z-heldout-selection-20261012`; `evaluation_profile` is
`heldout-selection-not-main`; baseline is `FE-heldout-control` or
`FE-heldout-tuned-L2`; role `scientific` means actual measurement, **not main-table
eligibility**. Arm encodes L2/clamp/steps/lr/decay, writer encodes batch-entry-z,
config/source/stream/selection hashes bind full hparams and split. These distinct
identities must not be relabeled as final main FE. Standard current/all-seen
metric names and edits axes are unchanged; raw IDs/text/tensors remain local.

CPU evidence only: selector failure/tie cases, exact native writer/L2 consumption,
RNG/state-neutral observations, BF16 latest overwrite/roundtrip, actual CF current
and all-seen scalar payloads, independent raw tamper rejection, and existing
native/transport regression tests. No pretrained GPU or online PASS is inferred.
# Conditional final CF / zsRE benchmarks

Execution remains requester/root-owned on rent. This API does not submit jobs.
Do not change the frozen `selection-r3` source, configs or running processes.

After all four held-out arms resolve as valid W20 or explicit numerical failure,
run the existing `collect` command. `selection-result.json` must be `SELECTED`.
The final preparer **independently re-runs that CPU raw selection audit**, checks
the saved receipt byte member and exact reconstructed result, and refuses
NO_PROMOTION, missing arms, technical/resource failures, or L2=0 promotion.

Use the same bindings schema with these additional members:

```json
{
  "stage": "final_benchmark",
  "selected_result": {"path": "/workspace/checkpoints/fe-qwen-working-20261012/selection-r3/selection-result.json", "bytes": 0, "sha256": "ACTUAL_RECEIPT_SHA_REQUIRED"},
  "zsre_stream": {"path": "ABSOLUTE_ORIGINAL_ZSRE_STREAM", "bytes": 1490821, "sha256": "f42ee4bc6e98b1133e48dc81102201ccd85ccc917a97160ee3c6a0f3a38f378c"},
  "planned_latest_count": 6
}
```

The example receipt bytes/SHA are placeholders, not an admissible receipt.
Keep original `fixed_stream`, `main_stream`, `alt_stream`, model/C0 members,
host lock and >=32GiB rent reserve. Model members must be regular resolved blob
paths; the snapshot alias is separately checked. No model download occurs.

```sh
python -m official.diagnostics.fe_qwen_current.prepare NEW_FINAL_ROOT final-bindings.json
PYTHONPATH=NEW_FINAL_ROOT/source python -m official.diagnostics.fe_qwen_current.launch NEW_FINAL_ROOT/configs/final-cf-L2-SELECTED.json NEW_FINAL_ROOT/lock.json
PYTHONPATH=NEW_FINAL_ROOT/source python -m official.diagnostics.fe_qwen_current.launch NEW_FINAL_ROOT/configs/final-zsre-L2-SELECTED.json NEW_FINAL_ROOT/lock.json
python -m official.diagnostics.fe_qwen_current.benchmark NEW_FINAL_ROOT
```

Read the actual returned config member names (decimal L2 uses `p`, e.g. `0p1`).
Preparation requires clean published HEAD and creates a new frozen source and
native clone, not an in-place edit. It returns exactly two configs with the same
selected L2 and independent cold HF loads. It does not restore held-out weights.
No main-benchmark metric influences selection or creates another configuration.

CF uses the exact original main first2000 SHA already locked by selection.
zsRE retains each original edit request once; only its evaluator expands native
public decode/re-tokenize queries. The preparer uses the local native tokenizer
and executes the pinned public query AST parity check over all 2000 requests.
This is CPU input parity, not pretrained output parity.

Both runs retain W0 full2000, each committed current100 and all-seen
500/1000/1500/2000 raw. CF retains strict NLL / R/P/N diagnostics and official
request-macro scores. zsRE records public-query E/G/loc_ans request-macro metrics
under `zsre/*`, not CF NLL or W0 agreement relabels. The independent reducer
checks ordered case/query/target mapping, predicted IDs, correctness bits, token
denominators, finite work and request-macro arithmetic. Final collection audits
all 25 endpoints per run, 20 commits plus W0, final checkpoint SHA, and reports
zsRE W20 Loc minus W0 Loc. It requires both final W20 endpoints before emitting
`both_W20_complete`; it makes no automatic recovery claim.

Final group is `fe-qwen-current-z-final-benchmark-20261012`, separate from
held-out selection. Existing current/all-seen metric names/axes remain standard.
CF generation stays DEFERRED; zsRE has no generation config. Runtime selected
L2, native current-entry z/history/writer/BF16/default backend are unchanged.
Milestone raw is persisted before latest-only checkpoint replacement under the
same host serialization lock. Storage admission credits only identity-bound
materialized held-out/final latest payloads; reserve, one atomic tmp and raw
margin remain budgeted. No checkpoint removal or automatic job submission.
