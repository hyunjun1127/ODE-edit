# Qwen BF16 context-lock technical calibration

Verdict: `PASS / EXACT_REPEAT / OUTCOME_FREE`.

- job: `16064` (`odeedit_s02_ctx_qwen`), `COMPLETED 0:0`, 13 seconds
- execution head: `37a713233b95614d2743a12abc4d1036daed95f0`
- revision/model/tokenizer: `a09a35458c702b33eeacc393d103063234e8bc28`
- dtype policy: `checkpoint-original`
- dtype/config/checkpoint/parameters: `torch.bfloat16`
- seed/repeats/fresh: `17` / `2` / `true`
- observed repeat IDs: both
  `5b7144416638fb3deec1f12f204a401e06edd1293aa2fe4a22f9080f3e8bd41b`
- template-byte hashes: both
  `ff84e360b7a4a413275da818862c7c4431ac21c7fbeaf26e4d2296278f9a4bf9`
- group sizes: both `[1,5]`
- exact match: `true`
- legacy lock ID: `e0c5f61d874334a2cab26f82fb3594d9ab88c9e5816be390274a0bb5e98c93bd`

Output root:
`local/results/session02-bf16-context-lock-probe-v1-qwen2.5-7b-inst-e26d5e9`.
Raw context SHA is `6bd9ba26...2084`; terminal manifest SHA is
`bd7e169c...d79d`. Raw strings remain local-only.

Edit/direct-z/covariance/dataset/evaluation counters are all 0. This ID is an
outcome-free new-lock candidate only; P0 retry is not authorized.
