# SH4 → GH P4 ZA M1 vs M5 case01 terminal ACK

- user-authorized scientific delta: target inner Adam update `M=5→1` only;
  K8/objective/KL/decay/clamp/lr/stream/context/evaluator/FP32/W0-frozen/writer0
  unchanged. M1 source HEAD/tree `26dc6007...4bcb6` / `8ecedf0c...65c8`.
- focused gates 23 PASS; transfer/final PRE-GPU PASS, final identity
  `9e24d098...9cebe`. Selected post-update z observation and gradient cosine are
  recorded with decision influence 0.
- job `22818_[0-1%2]`: Llama/Qwen `COMPLETED 0:0`, `4:16 / 4:06`, each
  1GPU/8CPU/65000MiB; nonfinite0, writer0, W0 unchanged/restored. Job `22816`
  was a 1s pre-model ignored-session-boundary fail-close, scientific result0.
- M1 absolute: Llama Z+ Gen `20/20`, NLL `2.8498`; Z± `19/20`, `1.8609`.
  Qwen Z+ `20/20`, `3.0978`; Z± `20/20`, `2.2903`; Loc all `93/100`.
- barrier `Z±−Z+` Gen NLL: Llama M5 `+0.3142` → M1 `-0.9889` but Gen
  hit `-1`; Qwen M5 `+0.1750` → M1 `-0.8075`, hit delta `0`.
- selected train z trajectory: Llama Z+ M1 minimum outer4 `0.9720`; Qwen
  minima move to outer6 Z+ `0.0614` / outer7 Z± `0.0296`. M1 reduces M5
  early saturation but remains non-monotone; Llama clamp 10/10 every outer,
  Qwen moment-reset first-step norm remains about 29.9.
- overhead: successful `0.1394 GPU-h`, `1.1156 allocated CPU-h`, 57.8% below
  M5; objective evaluations/arm `40→16` (-60%).
- verdict: `M1_BUDGET_ABLATION_SUPPORTED_FOR_FURTHER_VALIDATION`;
  `PROGRESSIVE_EDIT_NOT_ESTABLISHED` because ZA is writer0 and target
  trajectory oscillates. automatic promotion 0; next submit HOLD.
- report: `experiment-reports/servers/server4/2026-08-22-p4-za-m1-vs-m5-case01.md`
  SHA `d5538a4b0a57a7873137cee220723d62339ec96831c9383187a9a658a5f4a870`.
- analysis JSON SHA
  `45c6012882015cf6ddf338ab05009fd1c86d868e451420687afabcac0c51246f`.
- raw logs/results remain ignored local; model/data/checkpoint Git add 0.
