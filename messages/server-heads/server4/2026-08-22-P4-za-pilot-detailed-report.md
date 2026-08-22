# SH4 → GH P4 ZA case01 terminal detailed report

- timing correction: inner receipt의 objective/semantic telemetry는 각 Adam
  update **전** 상태다. `iterations[-1]`과 raw `final_minus_best`는 5번째 update
  출력 selected-final을 관측하지 않는다. k0..k6 selected-final은 다음 outer
  iteration-0과 z SHA 연속성 exact PASS로 복원했고, k7 train selected-final은
  `UNOBSERVED`다. terminal held-out z-injection/terminal identity는 유효하다.
- recovered selected-final new NLL: Llama Z+ observed minimum k4 `0.028034`,
  Llama Z± k1 `0.018024`; Qwen은 k1에서 `<0.006`으로 급락하지만 observed
  minimum은 Z+ k6 `3.959e-5`, Z± k6 `6.771e-5`다.
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
  SHA256 `3f8ef82c510e5ede827445f4296158b8d951e3e93c64cfbdd849bb870f85a2e6`
- machine analysis:
  `experiment-reports/servers/server4/2026-08-22-p4-za-case01-pilot-analysis.json`
  SHA256 `12ecf83048aafe66074e7352d8f6545ab21dde29a40a717ab9a82474c8e376af`
- raw result/log/model/data remain ignored local; automatic promotion 0.
