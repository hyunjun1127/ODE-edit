# Realization Debt Phase A terminal audit

- instruction: `ODEEDIT-S06-REALIZATION-DEBT-PHASE-A-LIFELONG-V6-V1`
- status: `ANALYSIS_ONLY_TERMINAL_PASS`
- contract: `b46f8620dda25f44836dab7f8facbb1c8212270ee83744fcc310938045f191a2` (6775 bytes, 211 newline-terminated lines, regular non-symlink, observed mode 0664)
- sealed base: `21da9f0512cde81fa09ccd37b8306051872e3a89` / `54cebf2916eb0aa4b59850df566fff2f4c7cb8ec`
- analysis source commit: `c9d85cf5abf47be495f53334d637a152832b1005`
- exact denominators: 4 arms, 100 batches/arm, 100 requests/batch, 5 layers/request; 200000 layer rows, 40000 request endpoints, 2000 batch-layer action rows
- composite-key duplicates, missing endpoint/action joins, core nonfinite, both debt identity failures, imputation, interpolation, reconstruction, request replication of batch actions: all `0`
- `debt_parallel_native` identity maximum absolute residual: `0.0`
- `debt_native` identity maximum absolute residual: `1.1368683772161603e-13` (within the recorded FP64 backward-error bound)
- plots: 5 repository-Python PNGs; actual independent rerun byte mismatch `0`
- schema boundaries: `absolute_allocation_energy_weighted_debt=NOT_COMPUTABLE_FROM_SCHEMA`; `exact_g_metric_action=NOT_COMPUTABLE_FROM_SCHEMA`
- interpretation: contemporaneous descriptive association only; future-forgetting/causal claim `0`; `scientific_promotion=false`
- editing/checkpoint evaluation/model/GPU/Slurm actions: all `0`

Canonical package:

- report SHA256: `91013cb776859eaaa80638c3f1780dbf82a5bebeb926caab68c6fafe983ba395`
- manifest SHA256: `abdc519ba949c3d37de638250766f8a0c1f2189f9a83682bb7ebc7733b7d805e`
- manifest identity: `5c6f4a0ca309f197309f12a44f2c314e819f9d814f165cacda0d82b2b436afb2`
- member root: `d909364ce91b89327cecf2d998c629590ae6c9678ddf6e94b49e8ff4412589b5`
- rooted receipt SHA256: `39ad4b494458cdc032c44c84cbf70da683e2f9bf3768f7a537261bf3e19a95b0`
- rooted receipt identity: `6ed535b0eeb0480b3c49a9bbd1a14a79b77b805cbd438dad633f2af274965043`

The canonical report records the large L4 outliers separately from median/p90 behavior and preserves exact top-row request provenance. No outcome was removed or imputed.
