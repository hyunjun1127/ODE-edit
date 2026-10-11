# FE original W0-fixed z

> 2026-10-11 사용자 정정: 이 W0 전체 target 선계산 경로는 참고·이력 실험이다. 현재 FE baseline은
> `fe_author_history.py`의 수정 EasyEdit 구현 + author hparams + 누적 history + 현재 배치 진입 z다.
> 이 문서의 실행 계약으로 현재 FE main table을 대체하지 않는다.

## Covariance repair (2026-10-11)

New preparations set `covariance_policy="project_roundoff"`. A missing field
means `strict`: an indefinite covariance is rejected. The GPT-J L3 FP32 cache
used by job63153 has three negative eigenvalues; promoting it to FP64 and
successfully solving with LU does not make it a valid ridge prior. The CF run
first jumped to a 64,549.98 update at B3/L3. See the
[investigation](../../experiment-reports/global/fe-collapse-repair-20261011/report-ko.md).

`official.baselines.fe_covariance.prepare_covariance` returns healthy matrices
unchanged. For a failed Cholesky check, the explicit repair policy raises only
eigenvalues below `max(eps_storage, n*eps64) * ||C||_F` to that resolution.
Negativity exceeding `eps_storage * ||C||_F` is rejected. This is a recorded
numerical regularization of the prior, not a claim to recover the author's
unavailable exact covariance. Clipping negative eigenvalues to zero alone is
insufficient: the remaining nearly singular inverse can produce larger edits.
No edit targets, author hparams, history weighting, or healthy covariance bytes
are changed. The corrected matrix remains FP64.

The original NPZ is read-only. A repaired prior is saved separately under the
run's `covariance/layer-<n>.pt`, with input/derived SHA and the eigenspectrum
correction recorded in `covariance-assets.json`. Checkpoint identity includes
the receipt digest. Resume reads the sealed derived matrix instead of repeating
eigendecomposition. Each author ridge system must also pass a positive-definite
check before its original LU solve; a finite but indefinite result is rejected
before that layer's history/weight update. The EasyEdit FE history solve uses
the same check. Existing frozen experiments are unchanged.

For Qwen, the completed author W0-fixed runs63217/63218 do not exhibit the old
collapse. The `FE_HISTORY_AUTHOR` profile is a distinct batch-entry-target,
FP32 variant; matching clamp/steps does not make its target reference policy
identical to this runner. Keep its results and identity separate.

Execution/storage repair: tensor hashes use contiguous dtype-preserving uint8 bytes (BF16 supported). Model loading does not force an attention backend; the author/default resolved backend is recorded in each observation runtime.json. Atomic latest.pt carries its own complete identity/cursor/content-integrity digest and is authoritative; latest.json is advisory, so a metadata-write crash cannot invalidate a committed payload. Metadata tmp names are unique and cleaned on exceptions. Resume keeps original observations intact and writes new metrics/commits/runtime/tracking under resumes/<unique-attempt>; resume-parent.json binds the exact saved milestone. CPU tests cover stale/missing metadata, corrupted payload rejection, BF16 hashing and B6 observation collisions. These are storage/CPU checks, not GPU resume parity evidence.

Authority USER-FE-ORIGINAL-W0-RESET-20261011-R1. Native source jugechengzi/FE commit478134dfb24b43f4e18b47e8500893ce3f9cc50f; apply `fe_original_author.patch` to a dedicated clone with `git apply`, then `python -m official.runners.fe_original_compat CLONE` verifies the exact delta. Only missing-unused import, lazy unrelated WISE/data import, and replay output Tensor/tuple/list handling change. Native fitting/key/solve/history remains the author source. FE-memit_main.batch_edit is called, not EasyEdit/FE_HISTORY.

`python -m official.runners.fe_original --config CONFIG --lock LOCK [--resume]`.
Config fields are illustrated by `fe_original_prepare.py` (SH1 only preparation). SH2 supplies its server/model/asset/output/runtime paths, no duplicate writer. `config_sha256` is canonical `official.experiments.prepare.digest(config without config_sha256)`.
Model is llama3/gptj/qwen25; dataset cf/zsre; requests2000, batch_size100, seed0, milestones[5,10,15,20], dtypebfloat16. `llms` is the exact original model YAML mapping; keep GPTJ's six layers3..8. `assets` points to the existing fullSHA model/tokenizer manifest; `C0` maps each physical layer to existing FP32 uncentered NPZ sum/count. `stream` is an immutable original official full2000 JSON member. `author_members` seals every consumed author source, `author_path` is the separately pinned clone. `query_proof` binds zsRE public-query CPU parity. `output` is a fresh ignored run path. `storage_min_free_bytes` is explicit per server (S1 64GiB, S2 32GiB), not a hardcoded 256GiB gate.

Host-common `checkpoint_lock`: `/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/checkpoint-serialization.lock` on both hosts (different hosts/filesystems, no crosshost mutex). Every run on one host uses the SAME path. `save_latest(...,lock_path=...,reserve_bytes=...)` acquires flock BEFORE cloning/creating tmp; checks estimated payload + reserve, actual serialized bytes and remaining reserve; holds through file fsync/replace/directory fsync/metadata fsync/replace/directory fsync. Failure removes its own uncommitted payload tmp and unlocks; no other run's latest removed. Actual run has one latest.pt at500/1000/1500/2000, no W0 CP. Tests exercise two simultaneous writers, failure cleanup, overwrite and CPU restore. Runtime resume is supported but no extra GPU qualification is run.

W0 native context is generated once by author precompute; all2000 targets use author compute_z_batch_first_forward BEFORE any edit. The resulting read-only z cache is bound to source/config/stream/model/contexts identity; same contexts passed to author edit. No batch-entry refit or final-model key recapture. Latest contains selected BF16 weights, native FP32 cache_c, RNG/context/cursor and z-reference. Author FP64 solve/write order is unchanged.

Every batch current/post100 R/P/N; milestones all_seen/post actualprefix. CF common factual strict/NLL diagnostics; zsRE common public-query requestmacro loc_ans. Evaluation isn't native author's metric, and no paper numeric parity is claimed. Generation DEFERRED for CF, absent zsRE. W&B single official transport, new writer `fe_author_w0_fixed`, baseline `FE-author-repo-W0-fixed-z-sequential`, independent immutable job/source/config, retained original official tracking parent authority; execution config carries this reset nonce. No old FE run reused.

SH1 isolated Python3.12 runtime reuses existing torch2.9.1+cu128 read-only via pth, installs author transformers4.51.3/tokenizers0.21.4/OmegaConf2.3.0. Python compatibility deviations: Hydra1.3.2 (author1.2.0), NumPy1.26.4 (author1.25.2 unavailable on3.12). No claim exact environment reproduction. SH2 may retain its documented compatible NumPy; bind actual versions in runtime manifest. GPU/pretrained observation only inside actual lanes, no separate pilot.
