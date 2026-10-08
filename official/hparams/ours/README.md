# PRICE configuration

One resolver: `official.ours.config.resolve(model, arm=None, preset=None)`.
It returns an immutable, recursively frozen mapping and canonical JSON SHA256.
Use `profile(resolved, runtime)` to bind architecture/input settings; contradictory
knobs fail. Engine, controller, optimizer, norm and subject loss consume that same
resolved profile. `plain(resolved)` is the serialization boundary.

| Model | Native clamp / lr / norm | PRICE default base / cap / lr / norm |
| --- | --- | --- |
| llama3 | .75 / .1 / .5 | .75 / .75 / .1 / .5 |
| qwen25 | 4 / .5 / .001 | .75 / .75 / .1 / .5 |
| gptj | .75 / .5 / .5 | .75 / .75 / .5 / .5 |

Writer files copy the unchanged baseline MEMIT JSON with original path/SHA.
An upstream change fails closed until explicitly adopted; it never changes ours
silently. Layers, anchor=max(layers), readout and ridge coefficient derive solely
from that copy. Arm overrides contain PRICE keys only. Native preset explicitly
maps clamp to base/cap/maximum scale, v_lr to lr, v_weight_decay to lambda_N;
it is not the default and is equivalent in numeric settings to Q7 for Qwen.

Native clamp bounds a single-layer delta. PRICE base bounds a price-weighted
multi-layer group-L1 sum; equal numbers do not imply equal editing strength.
`beta_max_scale` is independent of `c`; max budget is
max(beta_base, beta_max_scale * max_layer_price). Uncapped requires c=null.
The historical receipt key `beta_max_native_scale` remains a derived alias;
it is never an independent source of values.

All 13 knobs are explicit in price.json. Unknown/nonfinite/invalid values fail
closed. K_eval=max_updates+1; terminal has no backward/update. Norm is added
analytically once; zero layers remain eligible. The active mask identity guard
is retained. M1 changes only lookup-zero anchors using five existing prefix norms.

## Constants deliberately retained

- Price RMS off-owner reduction, relative floor 1e-6, absolute floor 1e-12,
  price range 1e6, denominator >1e-8: scientific price definition, not sweep knobs.
- Original LOO residual/coefficient, geometry, payload and FP32 projection
  tolerances: unchanged technical guards, not outcome-tuned.
- FP64 projection breakpoint ZERO/CAP tags, exact endpoint handling, tiny
  reporting threshold 1e-12: no pruning or postcast rescue added.
- Legacy unused energy optimizer defaults and historical full-builder diagnostic
  allocation coefficient .1 are retained for provenance, not production PRICE.
- Unit NLL coefficient and request/context reduction are fixed definitions.

Old `official/hparams/PRICE/*.json` are deprecated inputs (bytes retained).
New runs must resolve the ours tree; do not merge both config formats.

Current task permits Q0–Q7 held-out Tier1–2 after default GPU reproduction.
These JSON files by themselves do not authorize another experiment.
