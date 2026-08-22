# SH4 → GH P4 ZA case01 terminal detailed report

- job `22793_[0-1%2]`: Llama/Qwen both `COMPLETED 0:0`, `00:09:55`;
  FULL-FP32/offline, same-W0, writer0, retry0, nonfinite0.
- Llama `Z±-Z+`: Gen `0`, Gen new NLL mean `+0.3142` (worse),
  p90 `+0.1984`; both arms clamp `400/400` updates.
- Qwen `Z±-Z+`: Gen `-1/20`, Gen new NLL mean `+0.1750` (worse),
  p90 `+1.1672`; both arms clamp `0/400`.
- Native-Z has the lowest mean Gen new NLL for both models. Case01 verdict:
  `NO_POSITIVE_SIGNAL`; margin-only success rejected.
- successful overhead: `0.3306 GPU-h`, `2.6444 CPU-h`; final receipts
  56 files / 1,720,393 bytes, logs 4 files / 3,173 bytes. Pre-success
  fail-close overhead `0.045 GPU-h`.
- authoritative §9 gap: runtime new-only/PN gradient cosine and requestwise
  inner quantiles absent and not recoverable from stored SHA/norm. Post-model
  HF receipt also retains `model_loaded=false`; decay telemetry emits one
  detach warning per cell.
- verdict: `TECHNICAL_PILOT_PASS / SCIENCE_HOLD /
  HOLD_FOR_TELEMETRY_REPAIR_AND_GH_REVIEW`; new submit 0.
- detailed report:
  `experiment-reports/servers/server4/2026-08-22-p4-za-case01-pilot-detailed.md`
  SHA256 `111e8abbc91e8ddf1bc37e73d078926200c2cdd970372a3f79067f893f67ada9`
- machine analysis:
  `experiment-reports/servers/server4/2026-08-22-p4-za-case01-pilot-analysis.json`
  SHA256 `6edd2965a67e2042be70aaea9ae7a1dc0650b19b342d336900ba8fe839158aa0`
- raw result/log/model/data remain ignored local; automatic promotion 0.
