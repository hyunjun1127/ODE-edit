# Paired Alpha direct-z analysis — qwen2.5-7b-inst

- run: `dzf_alpha_qwen_p0_v1`
- verdict: `PAIRED_ALPHA_SIGNALS_REPORTED_NO_SUPERIORITY_DECISION`
- technical validity: `True`
- ITD: `8/8` successful
- analysis SHA-256: `5c4676adec7de5e0fa5495842702fd991b8f93251b4d95c312490d1340756c1e`
- every displayed effect is sign-normalized: positive means the listed lhs is better on that individual axis.
- scope: a single-model paired signal report, not a method-superiority or lifelong-editing analysis.

| Paired contrast / metric | Normalized mean | Positive | Bootstrap mean CI95 |
|---|---:|---:|---:|
| genuine C vs posthoc C: `z_residual_ratio` | -0.09317304 | 0/8 | [-0.10996609, -0.07808952] |
| genuine C vs posthoc C: `output_nll_reduction` | -0.04695102 | 0/8 | [-0.10351014, -0.01087411] |
| genuine C vs posthoc C: `preservation_score` | 0.09821588 | 7/8 | [0.02922394, 0.18778090] |
| genuine C vs posthoc C: `endpoint_frobenius_norm` | -0.01858916 | 3/8 | [-0.05477999, 0.01077424] |
| genuine C vs posthoc C: `endpoint_right_projector_violation_ratio` | 0.00000001 | 4/8 | [-0.00000002, 0.00000005] |
| BF vs genuine C: `z_residual_ratio` | 0.08233046 | 6/8 | [0.02827215, 0.14197551] |
| BF vs genuine C: `output_nll_reduction` | 0.05692602 | 7/8 | [0.00541535, 0.14113468] |
| BF vs genuine C: `preservation_score` | -0.20097198 | 1/8 | [-0.34313069, -0.07168475] |
| BF vs genuine C: `endpoint_frobenius_norm` | 0.10153338 | 7/8 | [-0.06918730, 0.24499900] |
| BF vs genuine C: `endpoint_right_projector_violation_ratio` | -0.00000000 | 4/8 | [-0.00000009, 0.00000009] |
| genuine C vs no-op (absolute write): `z_residual_ratio` | 0.46528274 | 8/8 | [0.41520081, 0.51039150] |
| genuine C vs no-op (absolute write): `output_nll_reduction` | 10.45983613 | 8/8 | [8.29599508, 12.69542507] |
| genuine C vs no-op (absolute write): `preservation_score` | -0.92632730 | 0/8 | [-1.30861291, -0.55644866] |
| genuine C vs no-op (absolute write): `endpoint_frobenius_norm` | -2.57325908 | 0/8 | [-2.89849906, -2.23395744] |
| genuine C vs no-op (absolute write): `endpoint_right_projector_violation_ratio` | -0.00000514 | 0/8 | [-0.00000519, -0.00000508] |
| BF vs no-op (absolute write): `z_residual_ratio` | 0.54761319 | 8/8 | [0.49956531, 0.60143944] |
| BF vs no-op (absolute write): `output_nll_reduction` | 10.51676214 | 8/8 | [8.36442509, 12.76747712] |
| BF vs no-op (absolute write): `preservation_score` | -1.12729928 | 0/8 | [-1.57721116, -0.65605320] |
| BF vs no-op (absolute write): `endpoint_frobenius_norm` | -2.47172570 | 0/8 | [-2.74644422, -2.17521670] |
| BF vs no-op (absolute write): `endpoint_right_projector_violation_ratio` | -0.00000514 | 0/8 | [-0.00000522, -0.00000507] |
| genuine cone vs no-op: `z_residual_ratio` | 0.37876187 | 8/8 | [0.34078023, 0.41504979] |
| genuine cone vs no-op: `output_nll_reduction` | 10.43335807 | 8/8 | [8.32283403, 12.73833804] |
| genuine cone vs no-op: `preservation_score` | -0.94572745 | 0/8 | [-1.35659492, -0.53884612] |
| genuine cone vs no-op: `endpoint_frobenius_norm` | -2.31386637 | 0/8 | [-2.55851813, -2.05513735] |
| genuine cone vs no-op: `endpoint_right_projector_violation_ratio` | -0.00000514 | 0/8 | [-0.00000518, -0.00000511] |
| oracle vs no-op: `z_residual_ratio` | 0.99999999 | 8/8 | [0.99999998, 1.00000000] |
| oracle vs no-op: `output_nll_reduction` | 10.54005588 | 8/8 | [8.44502321, 12.73345688] |
| oracle vs no-op: `preservation_score` | -1.01225985 | 0/8 | [-1.61277860, -0.49970018] |
| oracle vs no-op: `endpoint_frobenius_norm` | 0.00000000 | 0/8 | [0.00000000, 0.00000000] |
| oracle vs no-op: `endpoint_right_projector_violation_ratio` | 0.00000000 | 0/8 | [0.00000000, 0.00000000] |
| genuine full vs C-match: `z_residual_ratio` | 0.52851892 | 8/8 | [0.48312644, 0.57795977] |
| genuine full vs C-match: `output_nll_reduction` | 0.07670711 | 8/8 | [0.02112627, 0.16310548] |
| genuine full vs C-match: `preservation_score` | -0.06808518 | 2/8 | [-0.32197700, 0.18692162] |
| genuine full vs C-match: `endpoint_frobenius_norm` | -1.71993665 | 0/8 | [-2.12301288, -1.37478192] |
| genuine full vs C-match: `endpoint_right_projector_violation_ratio` | 0.00000000 | 0/8 | [0.00000000, 0.00000000] |
| posthoc full vs C-match: `z_residual_ratio` | 0.19969868 | 8/8 | [0.15652122, 0.24475326] |
| posthoc full vs C-match: `output_nll_reduction` | 0.02345101 | 8/8 | [0.00700154, 0.04732460] |
| posthoc full vs C-match: `preservation_score` | -0.02882546 | 2/8 | [-0.18013028, 0.12292515] |
| posthoc full vs C-match: `endpoint_frobenius_norm` | -0.68790336 | 0/8 | [-0.85572541, -0.52158761] |
| posthoc full vs C-match: `endpoint_right_projector_violation_ratio` | 0.00000000 | 0/8 | [0.00000000, 0.00000000] |

## Technical contract

- frozen target replay exact: `True`
- frozen target loaded once per case: `True`
- direct-z recomputes: `0` (must be `0`)
- precomputed projector only / integrity exact: `True` / `True`
- genuine ordered once/case, BF all hops genuine, post-hoc projection-only: `True` / `True` / `True`
- paired MEMIT run / replay lock: `dzf_qwen_p0_v2` / `8ee67660c167233e95f77cc170011eccf1526d08bd1337f5fd6da612db641f6e`
- exact rollback / firewall / receipt: `True` / `True` / `True`

## Interpretation boundary

- The JSON reports every scalar axis, including direct-z fidelity, output/edit proxies, preservation, spill, endpoint C/Frobenius cost, NFE, and projector leak.  Projector leak is a reported axis, not a validity gate.
- The 4,000-resample paired-bootstrap intervals are descriptive; the lenient signal gate is positive mean plus at least 5/8 positive cases.
- Do not pool Llama and Qwen, relabel a post-hoc BP update as genuine AlphaEdit, infer method superiority, or infer lifelong/sequential-collapse behavior.
