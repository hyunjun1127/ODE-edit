# FE author clamp/steps history profile (2026-10-10)

Authority `USER-SH-FE-AUTHOR-HPARAMS-2K-CF-ZSRE-20261010-R1`.
Only Llama3/Qwen2.5 × CF/zsRE. Separate table rows; not upstream DOW-KE reproduction.
`official/hparams/MEMIT_FE_HISTORY_AUTHOR/profiles.json` preserves author Git/blob/SHA provenance.
Legacy MEMIT_FE JSON and registry defaults stay unchanged. `official.baselines.fe_author_profile.resolve(model,device=0)`
passes the whole locked profile to the actual `registry.hparams(... overrides=...)`, validates the parsed dataclass,
and permits only author clamp/step differences plus operational visible-device binding.

Shared CLI: `python -m official.runners.fe_author_history --config CONFIG --lock EXECUTION_LOCK`.
Default is independent coldW0/H0. `--resume` remains explicit same-identity checkpoint recovery only,
not used for the new cold chains. Shared config validator: `validate_config(config)`.
Schema `official-fe-author-history-v1`, instruction_id above, task_id `fe-author-hparams-2k-20261010`,
method `MEMIT_FE_HISTORY`, model `llama3|qwen25`, server `server1|server2`, dataset `cf|zsre`.
Owner supplies distinct absolute output, arm, tracking_env_file, runtime, assets member, ordered stream member/digest,
author_profile_sha256, hparams, storage_min_free_bytes and canonical config_sha256 (digest excluding itself).
See SH1 prepare module for exact normalized asset structure: existing full-hashed model members/snapshot/identity,
tokenizer_sha256, C0 per layer member/module/masked-token-count, assets_sha256.
SH2 normalizes its own existing assets into that small manifest; no payload copy/rebuild.

Lock fields: source_directory/source_commit/official_tree/source_members/configs. Frozen source includes entire official/.
Caller exports OFFICIAL_CODE_COMMIT and uses matching existing environment/runtime. CPU/mem/admission/Slurm are owner-specific.
Storage minimum is explicit per-owner bytes (no hardcoded 256GiB inherited gate). Checkpoint save checks that minimum;
absence/space failure preserves source/previous CP. No automatic archive/delete; consumers pending keep W20.

Science: seed0, first2000, BS100×20, existing FP32/eager/TF32off, native FP64 history solve/CPU rollback unchanged.
Native apply receives each current batch at the batch-entry model, z disk-cache None, no whole2K W0 target freeze.
H is added to native covariance system and final-model native mean keys appended once/layer after all writes;
failed apply/evaluation/save does not advance durable H checkpoint. W0 and each committed batch save selected FP32 W,
H, contexts, RNG, cursor, identity; latest1 and finalW20 preserved. No separate GPU qualification/smoke/refit.

Factual schedule preserves history baseline W0 + W5/10/15/20 prefix evaluations (no additional current generation).
CF calls official factual, generation_schedule DEFERRED_CHECKPOINT_EVALUATION.
zsRE calls only official.evaluation.zsre_paper.evaluate(model_family=...), public-query request-macro loc_ans, no W0 agreement.
Before submission owner CPU compare_queries on own 2K stream/tokenizer; no pretrained forward parity claim.
zsRE config has no generation_schedule/FluCon fields. W&B uses existing official transport, actual job identity,
author-specific arm/task/config; official_zsre_metrics for W0 and milestones. Raw stays local.

Ownership: SH1 shared profile/runner; SH2 own adapter/prepare/submit. No shared math edits or logger fork.
