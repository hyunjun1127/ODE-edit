# ORBODE round0 B100 exhaustive review terminal audit

- Instruction: `ODEEDIT-ORRBODE-ROUND0-DETAILED-REVIEW-20260904-R1`
- Status: `ANALYSIS_ONLY_TERMINAL_PASS`
- Canonical execution: Slurm array `35694`; children `35737`, `35752`, `35889`, `35694`; all `COMPLETED/0:0`
- Raw source HEAD/tree: `84e6cde74db9b85bb9ffa16685ff80f50f2e6e06` / `32d996fd3c7e124c6068d713e6b0abb80dfb9105`
- Denominator: 4 cells, 100 requests/cell, 5 primary arms; 2,000 request-arm endpoints. `ORBHit` has zero primary-denominator influence.
- Derived rows: 38,400 prompt observations; 2,400 request-stage rows; 32,000 request-step rows; 320 step rows; 100 layer-update rows.
- Runtime integrity: official fidelity PASS, stream/order/W0/cache/evaluator binding PASS, model storage and ORBODE dynamic path FP32 PASS, nonfinite/dynamic-z/retry/forbidden-access/inner cache-history mutation all zero, W0 and method-state restore PASS.
- Stock MEMIT transient native solve FP64 is explicitly scoped and is not mislabeled as an all-algorithm FP32 path.
- Dynamic endpoints: 16/16 technically valid and `HORIZON_SEMANTIC_MISS`; global first hit 0/16. This is preserved as scientific observation, not technical failure.
- Exclusions: attempts `35520`, `35567`, and `35582` have denominator zero; B1 pilot `35615` is provenance-only; `round0-tech-r3` was dry-plan-only.
- Missing fields are labeled `NOT_RECORDED_SCHEMA_GAP`; imputation, reconstruction, rerun, editing/model/GPU/Slurm action counts are zero.
- Deterministic figures: 5 repository-Python PNGs, independent byte reproduction failure count zero.
- Focused unit tests: 10/10 PASS; Python compile, Markdown table structure, Git diff, package full-rehash and access checks PASS.
- Remaining-nine execution remains `HOLD`; `scientific_promotion=false`.

Canonical identities:

- Report SHA256: `54ef00e3944413a9908dd0ccabd33a487831c878b5a43bfbd55955ba03fa0a82`
- Manifest SHA256 / identity: `78b6217e952f03d446089d773bac365ace29ede806653ed72436d60ebd279a55` / `b6f65a56c7582d16c66ec353498e8c8b16fb4aa041b9422d992dd4ccf746c04d`
- Member root: `6a7e2fa7c697d4d5cfce87054ce5727585ca9c99b171ed574f66047cb8fb8b31`
- Rooted receipt SHA256 / identity: `d0fdeb5a1c4fd4e055265a1833576ff81ae3bce5093207bad92e531e4a059342` / `026b3022f07d02f5b34177b56743047127531c15c0f5c3256d1a803d0c2dc134`
