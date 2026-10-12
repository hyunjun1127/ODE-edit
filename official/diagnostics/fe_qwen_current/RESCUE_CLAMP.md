# Single prospective rescue: native L2=100, clamp=0.5

Same USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1 latest explicit improvement.
Previous clamp1 run stopped QUALITY_EARLY_STOP at B9 (57/62/56.1), not numerical
failure or W20. Its B5 CP/source/raw remain. Hydra process exit1 does not rename
the typed quality receipt. This source does not submit, stop or move any job.

Only changed scientific knob is author `llms.clamp_norm_factor=0.5`. Native
L2=100/BF16/default backend/solver/history/current-entry firstforward and all
other hparams remain. No old CP restore, W0-z, solver repair, target normalization
or extra arm. New immutable SPEC is `rescue-L2-100-clamp-0p5-v1`.

## Immediate held-out execution API (root only)

Use the SAME host bindings as rescue, server=rent, fresh output root. Reserved
storage/host checkpoint lock and original immutable heldout[4000:6000] unchanged.

```sh
python -m official.diagnostics.fe_qwen_current.rescue_clamp prepare NEW_ROOT BINDINGS.json
PYTHONPATH=NEW_ROOT/source python -m official.diagnostics.fe_qwen_current.rescue_clamp launch NEW_ROOT/configs/L2-100-clamp-0p5.json NEW_ROOT/lock.json
python -m official.diagnostics.fe_qwen_current.rescue_clamp collect NEW_ROOT
```

Exactly ONE config/run name `L2-100-clamp-0p5`. Config explicitly records L2=100,
clamp_norm_factor=.5, author_hparam_profile and distinct selection SPEC SHA.
Author YAML remains pinned original; launch appends explicit Hydra override
`llms.clamp_norm_factor=0.5`, tested against original YAML (only that field
changes). Native runtime guard verifies resolved actual clamp before W0;
runtime/CP/raw identity and diagnostic writer output contain both knobs/profile.
Tracking new group `fe-qwen-current-z-rescue-L2-100-clamp-0p5-20261012`; clearly
author-hparam-tuned, not default or existing clamp1 result.

Current100 per batch, all-seen/checkpoint every500, latest-only/host lock remain.
Any current R<80/P<70/N<65 yields committed QUALITY_EARLY_STOP, raw and latest
scheduled CP KEEP. Large norms alone do not stop. No failed-result promotion.
Collector reuses independent all25 endpoint/query/raw/commit/fullCP audit and
checks runtime, actual mmap CP identity/native config L2/clamp. Old clamp1 proof
or relabelled/mismatched new config fails. Actual full W20 gates remain95/85/70.
Success selection records both selected_L2=100 and selected_clamp_norm_factor=.5.

## Conditional subsequent benchmark

The same `rescue_clamp prepare/launch/final-collect` supports stage=final_benchmark
only after an actual independently re-audited new SELECTED proof, same cold
CF/zsRE setting; no benchmark retuning. Final names
`final-cf-L2-100-clamp-0p5` / `final-zsre-L2-100-clamp-0p5`.
The earlier e87 Flu/Con adapter intentionally still rejects the new SPEC;
its explicit new-profile binding will follow separately, not block held-out.

SH1/SH2 HOLD; root rent total ownGPU<=2 and actual admission/monitoring. No
pretrained/GPU/success claim from CPU tests. Existing frozen attempts unaffected.
