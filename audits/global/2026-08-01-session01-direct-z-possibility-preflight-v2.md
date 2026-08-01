# Session 01 direct-z possibility zero-outcome repair v2 preflight

## Failure provenance and repair scope

- v1 Slurm job `15739` ended `FAILED (1:0)` after `00:01:46`; its child
  states are one `FAILED (1:0)` and one fail-fast `CANCELLED (0:9)`.
- Both `dzf_llama_p0_v1/outcomes.jsonl` and
  `dzf_qwen_p0_v1/outcomes.jsonl` contain exactly zero rows. Features and
  actions are also zero rows. v1 is technical provenance, not scientific
  evidence, and remains read-only.
- The source repair removes one duplicated positional `contexts` argument from
  `EasyEditBridge.propose_ordered_memit_factors`. Operational identities move
  to `odeedit_dzf_pair_v2`, `dzf_{llama,qwen}_p0_v2`, and
  `dzf_pair_v2.submitted`.
- Models, fresh ranks `[132:140]`, six arm definitions/order, direct-z target,
  BF K=4 controller, z-cone solver, C budgets, metrics, bootstrap, firewall,
  and lenient possibility classification are unchanged.

## Minimal v2 gates

- Independent AST/API audit bound the runner's 40 bridge, quarter-step, MV-1,
  and fidelity calls to the actual callable signatures with zero mismatches.
  The new `inspect.signature.bind` regression rejects extra positional,
  unknown keyword, starred positional, and `**kwargs` drift.
- Focused helper/runner/API/analyzer suite: `31/31 PASS`.
- Full motivation suite: `223/223 PASS`.
- Runner/API-test compile, three launcher `bash -n`, and `git diff --check`:
  `PASS`.
- Repeated fixed artifact identities are unchanged:
  - Llama: `273d06af367904aac61931331e6aa2ea09204b0db1f6c7557bca16a9e2c84d2b`
    over 25 pinned files.
  - Qwen: `d247b27edf638b7a9625954e8bf2aeac745b61247230f8bcf733e991a3ed9d16`
    over 25 pinned files.
  - EasyEdit bridge:
    `48b073331dd46293941f96d5f800aa358d966498d6b0336252ee37c0d91975bf`
    over 13 pinned source files.
- EasyEdit's pre-existing porcelain state hash remains exactly
  `f0570aea6efb0b989bff971b5af0dbd20737b808f0127e133e1169df48176f27`;
  no EasyEdit file was changed by v1 or the repair.
- Dedicated ignored direct-z boundary remains separate from the public GH
  boundary. Existing adaptive/Alpha ownership and job are untouched.
- v2 run directories, marker, and active job name are all absent at preflight.
- Resource gate repeats:
  `active_project_gpus=2 + requested_gpus=2 <= cap=4`, memory
  `130000M <= 396234M`, decision `submit_now`.
- Existing Wikipedia moments remain pinned/read-only; recomputation/download
  and projector loading are forbidden in this track.
- Exact `8 x 6 = 48` model outcome schema, receipt-before-outcome, scalar-only
  serialization, byte-exact rollback, and direct-z-once remain mandatory.

- 최종 판정: `PASS` — zero-outcome API repair v2 pair 1회에만 유효
