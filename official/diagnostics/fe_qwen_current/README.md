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
