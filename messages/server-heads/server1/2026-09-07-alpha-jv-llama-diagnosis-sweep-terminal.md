# SH1 S sweep — REVIEW_READY

S job38475_0/1 both COMPLETED: two models × nine JV endpoints, plus two Official and two pre-edit observations. Fourteen actual paths, 68 nodes; each observation RS100/PS200/NS1000. D exact replay/intervention and independent audit300 are not run; this is not the final D+S or lifelong verdict.

Canonical report: `experiment-reports/servers/server1/alpha-jv-llama-diagnosis-sweep-2026-09-07-v1/sweep-terminal-v1/factual-report-ko.md`.

- Report SHA256: `e5bbc8c9598cb4c3ec816b54ae52dba1ae576f93e6eb6fe77a90cd8ac7a8f91f`
- Manifest SHA256: `e8020a4cb1465952a17f0a412a67e096893bf1807da3184566336cefe44b8777`
- Members root: `f5b4147481a3e6b5d050c76ba213100dabd65f142b1657500af25245a1277f29`
- Receipt SHA256: `d8eb8fc4ca5fceb72c3c9d81c4d01f71ca565c6920d00745abde4de4ac636cf8`
- Receipt identity: `c3ea915094d22bdd694c20db28d9542069d6c7f34793ec6c012f43d970caaa55`

Llama BASE vs Official: RS100/100 vs100/100, PS178/200 vs189/200, NS881/1000 vs891/1000. Five-lambda PS87.5–89.5% does not close Official94.5%; N2/T4 improve PS to92.5% with NS87.5%. Qwen BASE vs Official: RS100/100 both, PS194/200 both, NS824/1000 vs813/1000, but rephrase new-NLL mean2.062990 vs1.872372 and p90 6.269457 vs5.476517. BASE L8 actual endpoint energy share Llama99.9081%, Qwen99.9665%. No causal/retention/promotion claim.

Execution HEAD/tree `ee6594882e03b57cd6e8934dc758868bb1defdd8` / `4df08fb8523afedb9babf87e21bf8c527980be71`; execution checkout `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-alpha-jv-ds-exec-r1`. Analysis HEAD/tree `59263e89aee31995f1f53a1b725c42af43a244f4` / `5de4669d8633887da0c90798260b7e2721e63b2f`; analysis checkout `/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s06-alpha-jv-diagnosis-sweep-v1`.

Actual cost24899GPU-sec=6.916389GPUh; outstanding reservation0, combined48GPUh ceiling unchanged. Process totals are not nested-component sums. Main actual JVP340 plus fidelity JVP20; FD validation forwards120. Actual node/solver/forward/wall/peak tables distinguish scoped diagnostics, path IO/evaluation, and process totals. No inferred FLOPs or arbitrary T4 prefix cost allocation.

Checks:19 focused tests PASS,594 independent count/NLL checks PASS,360 raw JSON members rehashed,5 Python PNGs byte-reproduced in-process and via separate CLI, package/access/markdown/diff PASS. Tensor bytes and production journal reconstruction were not independently replayed; no claim otherwise. Raw tensors/prompts/logs/cache/model/dataset are excluded from Git. Analysis actions newmodel/GPU/Slurm0. Source+raw-free package only on dedicated branch; main integration remains GH-review-only.

Review inputs: immutable raw root `<execution checkout>/local/alpha-jv-llama-diagnosis-sweep/run-20260907-sweep-r1`; root source/science/sample/assets/resource locks and per-cell runtime/fidelity/terminal receipts. Package `input-manifest.json` pins consumed and inventoried JSON; `source-manifest.json` binds execution/analysis separately. Code/test paths: `project/run_scripts/alpha_jv_llama_diagnosis/{analysis_s,analysis_details,test_analysis_s}.py`. Independent release checks: `audits/servers/server1/2026-09-07-alpha-jv-llama-diagnosis-sweep/sweep-terminal-analysis-v1/release-gates.json`.
