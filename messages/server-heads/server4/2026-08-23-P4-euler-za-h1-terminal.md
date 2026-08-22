# P4-Euler ZA h=1 Llama B2–B10 terminal

- instruction: `ODEEDIT-S05-P4-EULER-PROJECTED-SEMANTIC-ODE-V1`
- status: `TERMINAL_TECHNICAL_PASS_POST_ZA_PAUSED`
- policy: `USER_DIRECTED_POST_CALIBRATION_HYPERPARAMETER_REVISION`
- classification: `PROJECTION_DOMINATED_EXPLORATORY_RUN`
- source: `f4fbfb5de132de9b986e4b291c5b41784c182743` / tree
  `8c3f444014907f69b1bb793674f7a5a9e2b9ee0e`
- Slurm: `22854_[2-10]%2`, B2–B10 `9/9 COMPLETED 0:0`
- scope: Llama only, `h=1`, `M=5`, `T_z=5`, W0-frozen target-only;
  B1 calibration-only excluded; Qwen0; ZB0
- receipts: case 9, arm 27, action-freeze 18, microstep 90; rooted mismatch,
  nonfinite, W mutation, duplicate, optimizer/Adam/backward/parameter-grad,
  writer/cache 모두 0
- primary limitation:
  - Z+: clamp `286/450=.635556`, all-5-step saturation `19/90`
  - Z±: clamp `381/450=.846667`, all-5-step saturation `54/90`
  - clamp–NLL/margin 관계는 factual association only; causal claim 0
- train endpoint mean: Z+ new NLL `0.977994`, margin `9.815328`;
  Z± new NLL `1.382357`, margin `9.365802`
- rewrite strict: Z+ `89/90`, Z± `84/90`; rephrase strict: Z+ `84/90`,
  Z± `81/90`
- paired Z±−Z+ mean: train new NLL `+0.404362`, train margin `-0.449526`;
  rewrite new NLL `+0.327021`; rephrase new NLL `+0.338653`
- report:
  `experiment-reports/servers/server4/p4-euler-za-llama-h1-b2b10-2026-08-23/report-ko.md`
- manifest root:
  `c3498eee45a2ec1a05153f1dbb2d929f37e300f0cedac10e21a191e398d23fb4`
- receipt identity:
  `d0af2a3d722742ecc86a951f787e075b5b89f04f76a908e6d21e7c5ad7fa37be`
- boundary: h=.25 B1 calibration and Stage2 latent geometry diagnostic are not
  pooled with this exploratory B2–B10 run; continuous-ODE convergence,
  step-size stability, writer/weight edit, promotion claims 0
- next: `POST_ZA_PAUSE_ACTIVE / IDLE_AWAITING_GH_CALL`; new GPU/Slurm/model,
  tuning, rerun, ZB 0
