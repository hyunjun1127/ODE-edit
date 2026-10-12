# Author numeric settings + current-entry z + BASIC EOT

New explicit USER instruction under USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1.
No old L2=10 W20 completion/selection prerequisite. No best-failed promotion.
Cold original-main CF and zsRE, independently. Not unmodified-author-bitwise FE:
BS100x20/seed0/current-entry targets/basic position0 EOT are the user protocol.

## Bindings / execution

Use prior b255 config/lock/runtime/pointer/checkpoint/commit/writer reference
members and the 8a relocation API: `dataset`, `main_stream`,
`original_main_config`, `execution`, `storage` unchanged. No sampler/rebuild.
Original main configs keep fullSHA and original path/stat provenance; relocated
stream must freshly verify identical bytes/fullSHA, unique/order2000, CF
heldout-disjoint. Root reports original CF edit/evaluation fields and order
match OURS and zsRE original fullSHA/cursor match; caller still verifies members.

Add **`author_yaml: {algs: MEMBER, llms: MEMBER}`**, exact raw files from pinned
author478134df `configs/algs/memit.yaml` and `configs/llms/qwen2.5-7b.yaml`.
Module pins fullSHA4799ac1d… and0e1481d0… (confirmed from git objects).
These may be copied unchanged to rent regular paths. No `heldout_result`,
`heldout_commits`, or `heldout_pointer` needed. Historical checkpoint is only
reference metadata: never loaded/restored into the new run. Optional old keys
have no execution/selection effect.

`execution.source_run_id` links the working diagnostic W&B run, not a selection.
Actual job_id must equal ODEEDIT_RENT_JOB_ID. New group
`fe-qwen-author-basic-eot-main-20261012`; profile
`FE_AUTHOR_HPARAMS_CURRENT_Z_BASIC_EOT`; arms label L2-0/clamp1 explicitly.

```
python -m official.diagnostics.fe_qwen_current.author_basic_eot_benchmark run BINDINGS NEW_OUTPUT
python -m official.diagnostics.fe_qwen_current.author_basic_eot_benchmark collect CF_RESULT ZSRE_RESULT NEW_REPORT
```

No Hydra/extra override CLI. Before allocating model, read/hash YAMLs and replace
entire historical llms/algs mappings with exact pinned mappings. Record all
historical differences and whole resolved config SHA. Audit **every** model and
algs field, plus BF16/seed0/BS100/2K/default covariance/no hidden method modes,
then repeat immediately before actual native apply and after completion.
Original path config is retained; actual data is explicit main_stream converted
by native_records (legacy cfg.data label is unused by direct native apply).

L2=0/add_old_keys=true; Qwen clamp1/35steps/lr.5/decay.001/KL.0625/C015000/L4–8.
Same native BF16/default backend, averaged K/writer/history, c36 EOT in all
editing input sites, no sink-list/no masks/no W0-z. Fresh HF/context/H0.
Old frozen files remain unchanged. Observer inherits audited evaluation,
native targets/write and hostlocked atomic latest-checkpoint callbacks directly.

Original unmodified evaluation: W0full2K/current100/allseen500/1000/1500/2000,
CF NLL and zsRE public-query request-macro Loc. CF80/70/65 quality stop remains;
no such Loc gate for zsRE. Raw before checkpoint; latest only each500, reserve
>=32GiB and host shared lock. CFgeneration deferred, zsRE no generation.
Separate collector requires actual20 commits/25 endpoints/runtime/CP L2=0,
full native config and YAML binding, independent query/raw reductions on both
datasets, and reports Loc delta. No L2=10 CP or W20 proof can satisfy it.
No automatic recovery or pure-author-reproduction claim. Root alone stages,
admits, submits and monitors; SH1 performs no GPU/remote/submission operations.

## CPU source receipt

2026-10-12: six suites (`test_author_basic_eot_benchmark`,
`test_basic_eot_benchmark`, `test_basic_eot_continue`, `test_failure_eot_probe`,
`test_failure_eot_prefix`, `test_benchmark`) 34 PASS/no skips. Includes real pinned
Qwen YAML equality and L2-10 rejection; actual resolved L2-0; same native
writer/adapter callback objects; tiny BF16 native regression and independent
CF/zsRE raw/query/schema/state-preservation tests. No pretrained forward/GPU,
online W&B or execution-performance claim. Old source/run bytes unchanged.
