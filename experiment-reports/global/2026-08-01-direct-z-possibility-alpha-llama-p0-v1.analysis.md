# Paired Alpha direct-z analysis — llama3-8b-inst

- run: `dzf_alpha_llama_p0_v1`
- verdict: `PAIRED_ALPHA_SIGNALS_REPORTED_NO_SUPERIORITY_DECISION`
- technical validity: `True`
- ITD: `8/8` successful
- analysis SHA-256: `bf3eb0e774e571a102659600f089d982f40083c016c664531ee19c2735e54eb0`
- every displayed effect is sign-normalized: positive means the listed lhs is better on that individual axis.
- scope: a single-model paired signal report, not a method-superiority or lifelong-editing analysis.

| Paired contrast / metric | Normalized mean | Positive | Bootstrap mean CI95 |
|---|---:|---:|---:|
| genuine C vs posthoc C: `z_residual_ratio` | -0.01133774 | 0/8 | [-0.01364612, -0.00923959] |
| genuine C vs posthoc C: `output_nll_reduction` | -0.24165694 | 0/8 | [-0.37668386, -0.12572157] |
| genuine C vs posthoc C: `preservation_score` | 0.00407069 | 8/8 | [0.00161195, 0.00680932] |
| genuine C vs posthoc C: `endpoint_frobenius_norm` | -0.00282356 | 0/8 | [-0.00372944, -0.00185908] |
| genuine C vs posthoc C: `endpoint_right_projector_violation_ratio` | -0.00000001 | 4/8 | [-0.00000002, 0.00000001] |
| BF vs genuine C: `z_residual_ratio` | 0.02056762 | 7/8 | [0.00266616, 0.03252131] |
| BF vs genuine C: `output_nll_reduction` | 3.78883688 | 8/8 | [2.76115873, 4.49475663] |
| BF vs genuine C: `preservation_score` | -0.14487684 | 0/8 | [-0.23870629, -0.06912718] |
| BF vs genuine C: `endpoint_frobenius_norm` | -0.02535538 | 0/8 | [-0.03485340, -0.01554822] |
| BF vs genuine C: `endpoint_right_projector_violation_ratio` | -0.00000003 | 3/8 | [-0.00000009, 0.00000003] |
| genuine C vs no-op (absolute write): `z_residual_ratio` | 0.28445854 | 8/8 | [0.25463349, 0.31273054] |
| genuine C vs no-op (absolute write): `output_nll_reduction` | 4.16866735 | 8/8 | [2.82923913, 5.67533763] |
| genuine C vs no-op (absolute write): `preservation_score` | -0.12773744 | 0/8 | [-0.33825387, -0.01419412] |
| genuine C vs no-op (absolute write): `endpoint_frobenius_norm` | -0.33386235 | 0/8 | [-0.34454232, -0.32313186] |
| genuine C vs no-op (absolute write): `endpoint_right_projector_violation_ratio` | -0.00000436 | 0/8 | [-0.00000443, -0.00000430] |
| BF vs no-op (absolute write): `z_residual_ratio` | 0.30502616 | 8/8 | [0.26308114, 0.34123596] |
| BF vs no-op (absolute write): `output_nll_reduction` | 7.95750422 | 8/8 | [6.64869990, 9.64526162] |
| BF vs no-op (absolute write): `preservation_score` | -0.27261428 | 0/8 | [-0.52447093, -0.08540515] |
| BF vs no-op (absolute write): `endpoint_frobenius_norm` | -0.35921773 | 0/8 | [-0.36769922, -0.35200759] |
| BF vs no-op (absolute write): `endpoint_right_projector_violation_ratio` | -0.00000439 | 0/8 | [-0.00000443, -0.00000435] |
| genuine cone vs no-op: `z_residual_ratio` | 0.35262485 | 8/8 | [0.32701608, 0.37275394] |
| genuine cone vs no-op: `output_nll_reduction` | 6.78356875 | 8/8 | [4.97457652, 8.82835995] |
| genuine cone vs no-op: `preservation_score` | -0.20876647 | 0/8 | [-0.45112444, -0.04952373] |
| genuine cone vs no-op: `endpoint_frobenius_norm` | -0.29241373 | 0/8 | [-0.30769743, -0.27532387] |
| genuine cone vs no-op: `endpoint_right_projector_violation_ratio` | -0.00000419 | 0/8 | [-0.00000426, -0.00000411] |
| oracle vs no-op: `z_residual_ratio` | 0.99999999 | 8/8 | [0.99999999, 1.00000000] |
| oracle vs no-op: `output_nll_reduction` | 10.42197685 | 8/8 | [8.90617024, 11.98474647] |
| oracle vs no-op: `preservation_score` | -1.18917908 | 0/8 | [-2.02330742, -0.51948224] |
| oracle vs no-op: `endpoint_frobenius_norm` | 0.00000000 | 0/8 | [0.00000000, 0.00000000] |
| oracle vs no-op: `endpoint_right_projector_violation_ratio` | 0.00000000 | 0/8 | [0.00000000, 0.00000000] |
| genuine full vs C-match: `z_residual_ratio` | 0.69010628 | 8/8 | [0.66214676, 0.71900138] |
| genuine full vs C-match: `output_nll_reduction` | 6.25140007 | 8/8 | [3.98542080, 8.41219197] |
| genuine full vs C-match: `preservation_score` | -0.89236980 | 0/8 | [-1.53345875, -0.37393281] |
| genuine full vs C-match: `endpoint_frobenius_norm` | -0.61223190 | 0/8 | [-0.64750881, -0.57175998] |
| genuine full vs C-match: `endpoint_right_projector_violation_ratio` | 0.00000000 | 0/8 | [0.00000000, 0.00000000] |
| posthoc full vs C-match: `z_residual_ratio` | 0.64494383 | 8/8 | [0.62510578, 0.66721151] |
| posthoc full vs C-match: `output_nll_reduction` | 6.00916520 | 8/8 | [3.69927012, 8.24221152] |
| posthoc full vs C-match: `preservation_score` | -0.85309715 | 0/8 | [-1.47725519, -0.35862316] |
| posthoc full vs C-match: `endpoint_frobenius_norm` | -0.56869850 | 0/8 | [-0.60075272, -0.53682404] |
| posthoc full vs C-match: `endpoint_right_projector_violation_ratio` | 0.00000000 | 0/8 | [0.00000000, 0.00000000] |

## Technical contract

- frozen target replay exact: `True`
- frozen target loaded once per case: `True`
- direct-z recomputes: `0` (must be `0`)
- precomputed projector only / integrity exact: `True` / `True`
- genuine ordered once/case, BF all hops genuine, post-hoc projection-only: `True` / `True` / `True`
- paired MEMIT run / replay lock: `dzf_llama_p0_v2` / `8ee67660c167233e95f77cc170011eccf1526d08bd1337f5fd6da612db641f6e`
- exact rollback / firewall / receipt: `True` / `True` / `True`

## Interpretation boundary

- The JSON reports every scalar axis, including direct-z fidelity, output/edit proxies, preservation, spill, endpoint C/Frobenius cost, NFE, and projector leak.  Projector leak is a reported axis, not a validity gate.
- The 4,000-resample paired-bootstrap intervals are descriptive; the lenient signal gate is positive mean plus at least 5/8 positive cases.
- Do not pool Llama and Qwen, relabel a post-hoc BP update as genuine AlphaEdit, infer method superiority, or infer lifelong/sequential-collapse behavior.
