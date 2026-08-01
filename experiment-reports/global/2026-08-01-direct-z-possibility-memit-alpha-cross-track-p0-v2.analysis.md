# Direct-z MEMIT × AlphaEdit cross-track analysis

- verdict: `CROSS_TRACK_AXIS_SIGNALS_REPORTED_NO_MODEL_POOLING`
- cross-bound join: `PASS_CROSS_BOUND_JOIN_ALPHA_SELF_BINDING_AND_C_BUDGET`
- replay lock: `8ee67660c167233e95f77cc170011eccf1526d08bd1337f5fd6da612db641f6e`
- analysis SHA-256: `32b7bf86d28a3ee5feafb540ca7d352c3c70a78d79e87dc5f8274d91feeb89c0`
- scope: fixed atomic geometry signal; no model pooling or method-superiority decision

## llama3-8b-inst

- technical validity: `True`
- cross-bound join certificates: `8/8`
- Alpha self-bound provenance certificates: `8/8`
- verdict: `CROSS_TRACK_AXIS_SIGNALS_REPORTED_NO_SUPERIORITY_DECISION`

| Metric | MEMIT BF lift | Alpha BF lift | Alpha−MEMIT 2×2 interaction | AlphaBF−MEMITBF endpoint |
|---|---:|---:|---:|---:|
| z_residual_ratio | -0.13500684 (0/8) | 0.020567618 (7/8)✓ | 0.15557446 (8/8)✓ | -0.21303884 (0/8) |
| generated_delta_error_mean | -0.11268154 (0/8) | 0.026230122 (7/8)✓ | 0.13891166 (8/8)✓ | -0.17573496 (0/8) |
| generated_delta_error_worst | -0.058206066 (1/8) | 0.042201087 (7/8)✓ | 0.10040715 (8/8)✓ | -0.13321338 (0/8) |
| delta_gain | 0.015117309 (7/8)✓ | 0.14513059 (8/8)✓ | 0.13001328 (8/8)✓ | -0.27241866 (0/8) |
| delta_cosine | -0.094609184 (0/8) | -0.18616539 (0/8) | -0.091556204 (0/8) | -0.14765366 (0/8) |
| off_token_spill_ratio | -0.026656869 (0/8) | -0.036090123 (0/8) | -0.0094332542 (1/8) | 0.028884281 (8/8)✓ |
| output_progress | 0.76534057 (8/8)✓ | 4.9169953 (8/8)✓ | 4.1516548 (8/8)✓ | -5.3578786 (0/8) |
| output_nll_reduction | 0.16067749 (8/8)✓ | 3.7888369 (8/8)✓ | 3.6281594 (8/8)✓ | -2.3268044 (0/8) |
| exact_margin_min | 1.188584 (8/8)✓ | 4.7084776 (8/8)✓ | 3.5198936 (7/8)✓ | -5.3698869 (0/8) |
| exact_satisfied | 0 (0/8) | 0.5 (4/8) | 0.5 (4/8) | -0.375 (0/8) |
| paraphrase_nll_reduction | 0.80229617 (6/8)✓ | 3.2602912 (8/8)✓ | 2.4579951 (6/8)✓ | -2.5489456 (1/8) |
| heldout_kl | -0.096319793 (0/8) | -0.14487684 (0/8) | -0.048557048 (4/8) | 0.3362728 (8/8)✓ |
| endpoint_frobenius_norm | -0.040890991 (0/8) | -0.02535538 (0/8) | 0.015535611 (6/8)✓ | 0.46195763 (8/8)✓ |
| nfe | 0 (0/8) | -19 (0/8) | -19 (0/8) | -21 (0/8) |

Positive values are sign-normalized improvements. A check mark requires positive mean and at least 5/8 positive cases.
The joint-case bootstrap CI is descriptive and is available in JSON.

## qwen2.5-7b-inst

- technical validity: `True`
- cross-bound join certificates: `8/8`
- Alpha self-bound provenance certificates: `8/8`
- verdict: `CROSS_TRACK_AXIS_SIGNALS_REPORTED_NO_SUPERIORITY_DECISION`

| Metric | MEMIT BF lift | Alpha BF lift | Alpha−MEMIT 2×2 interaction | AlphaBF−MEMITBF endpoint |
|---|---:|---:|---:|---:|
| z_residual_ratio | -0.054561611 (2/8) | 0.082330457 (6/8)✓ | 0.13689207 (8/8)✓ | -0.15741648 (0/8) |
| generated_delta_error_mean | -0.023686734 (3/8) | 0.10533109 (7/8)✓ | 0.12901782 (8/8)✓ | -0.12978544 (0/8) |
| generated_delta_error_worst | -0.013361113 (3/8) | 0.1069957 (7/8)✓ | 0.12035681 (8/8)✓ | -0.1277579 (0/8) |
| delta_gain | -0.0069985178 (4/8) | 0.13776107 (8/8)✓ | 0.14475959 (8/8)✓ | -0.13824734 (0/8) |
| delta_cosine | -0.020693929 (0/8) | -0.014120045 (2/8) | 0.0065738838 (4/8) | -0.045165238 (0/8) |
| off_token_spill_ratio | -0.0036023907 (3/8) | -0.010563629 (2/8) | -0.0069612381 (2/8) | 9.2025779e-05 (6/8)✓ |
| output_progress | -0.11081997 (2/8) | 0.71733285 (7/8)✓ | 0.82815281 (6/8)✓ | -0.89626947 (0/8) |
| output_nll_reduction | 2.1168074e-05 (2/8) | 0.056926016 (7/8)✓ | 0.056904848 (7/8)✓ | -0.01635973 (0/8) |
| exact_margin_min | -0.12983727 (1/8) | 0.72356713 (6/8)✓ | 0.8534044 (7/8)✓ | -0.67598736 (0/8) |
| exact_satisfied | 0 (0/8) | 0 (0/8) | 0 (0/8) | 0 (0/8) |
| paraphrase_nll_reduction | 0.20916645 (7/8)✓ | 0.88091911 (7/8)✓ | 0.67175266 (6/8)✓ | -0.10662905 (2/8) |
| heldout_kl | -0.086997554 (2/8) | -0.20097198 (1/8) | -0.11397443 (3/8) | 0.0026854109 (4/8) |
| endpoint_frobenius_norm | 0.83019237 (7/8)✓ | 0.10153338 (7/8)✓ | -0.72865899 (1/8) | 1.6656857 (8/8)✓ |
| nfe | 0 (0/8) | -19 (0/8) | -19 (0/8) | -21 (0/8) |

Positive values are sign-normalized improvements. A check mark requires positive mean and at least 5/8 positive cases.
The joint-case bootstrap CI is descriptive and is available in JSON.

## Interpretation boundary

- The 2×2 interaction asks whether BF gains more over its native writer under AlphaEdit than under MEMIT.
- The endpoint contrast separately asks whether the AlphaEdit BF endpoint itself beats the MEMIT BF endpoint.
- A positive interaction does not imply a positive endpoint contrast.
- The replay lock pins the cross-bound target/W0 identities; Alpha-only context/state/snapshot/parameter/solver digests are separately self-bound, not externally pinned by that lock.
- Heldout identity is cross-track and cross-model equal, but is not replay-lock pinned.
- These are fixed-panel atomic signals, not lifelong retention or collapse-prevention evidence.
