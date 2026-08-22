# P4-Euler Stage1 Llama NLL metric gate revision

- instruction: `ODEEDIT-S05-P4-EULER-PROJECTED-SEMANTIC-ODE-V1`
- authority: `USER_DIRECTED_GATE_POLICY_REVISION`
- revised verdict: `TASK_METRIC_GATE_PASS`
- decisive evidence: selected `h=.25` Stage1 NLL strength curve and same-`T_z=1.25` M5/M10 train-target metric stability
- selected h/T_z: `.25 / 1.25` unchanged
- same-T request evidence: Z+ and Z± each M5/M10 positive margin `10/10`; descriptive new-NLL<1 `10/10`
- old geometry gate: `FAILED_OBSERVED_NONDECISIONAL`; d_i values and old terminal report remain immutable
- claim boundary: empirical train-target metric stability only; latent endpoint convergence, continuous-ODE proof, heldout/writer/sequential claims are not established
- report: `experiment-reports/servers/server4/p4-euler-stage1-llama-nll-metric-gate-2026-08-23/report-ko.md`
- tables: 12 CSVs covering all 32 h×M×arm cells, M/h effects, request-level NLL, same-T stability, correlations, compute/overhead
- manifest root: `623aa80daf1bb178d244b93b516d27feb9c784b09eae6f39a6720ced1768ffea`
- receipt identity: `fbe3a7a383560863527ddc598985bc6cec65b0d1fa4487f066103f174d14db90`
- actions: recompute0, result modification0, GPU/Slurm0, ZA/ZB0, scientific promotion=false
- submit state: HOLD pending GH instruction
