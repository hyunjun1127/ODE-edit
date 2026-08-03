# Llama BF16 context-lock technical calibration

Verdict: `PASS / EXACT_REPEAT / OUTCOME_FREE`.

- job: `16063` (`odeedit_s02_ctx_llama`), `COMPLETED 0:0`, 14 seconds
- execution head: `37a713233b95614d2743a12abc4d1036daed95f0`
- revision/model/tokenizer: `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`
- dtype policy: `checkpoint-original`
- dtype/config/checkpoint/parameters: `torch.bfloat16`
- seed/repeats/fresh: `17` / `2` / `true`
- observed repeat IDs: both
  `22c26dc11fb13acd51d5bdc483e4b9dd46fa40029fe10b167bff1a1642f7e686`
- template-byte hashes: both
  `0a2069beafc60e170251103028fde716a160a60a8048bb000a649cf26c233bb0`
- group sizes: both `[1,5]`
- exact match: `true`
- legacy lock ID: `3020b3f5cea62e6cfbd173f0c99a4348cecf7e84425bb087720ced7f395482e5`

Output root:
`local/results/session02-bf16-context-lock-probe-v1-llama3-8b-inst-e26d5e9`.
Raw context SHA is `f1f428a0...3991`; terminal manifest SHA is
`d37bc471...c94e`. Raw strings remain local-only.

Edit/direct-z/covariance/dataset/evaluation counters are all 0. This ID is an
outcome-free new-lock candidate only; P0 retry is not authorized.
