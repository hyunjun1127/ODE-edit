# Token accounting distinction

R1/R2 `actual_model_forward_input_tokens` is a sum of the attention mask. For
uncached forwards it counts nonpadding input tokens. During cached generation
it counts attention-visible history, not only the newly processed query token.
The original raw counter is preserved and must be labeled accordingly in the
report, not presented as a universal physical token/FLOP count.

New source additionally records input tensor elements (including padding),
nonpadding query tokens (the last input-length columns of the mask), and full
attention-visible nonpadding tokens. Actual forward invocation counts were
already scoped correctly. No model outcome or evaluation formula changes; no
editing rerun for this observation-only accounting distinction.
