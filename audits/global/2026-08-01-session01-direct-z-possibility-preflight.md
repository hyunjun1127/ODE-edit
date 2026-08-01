# Session 01 direct-z possibility preflight

## Scope

- User-authorized possibility diagnostic only; no method-superiority claim.
- Fixed models: `llama3-8b-inst`, `qwen2.5-7b-inst`.
- Fixed fresh ranks `[132:140]`, eight cases per model, six committed arms.
- One two-GPU parent with one visible A6000 per model child.
- Existing adaptive/Alpha job and GH task remain independently owned and
  untouched.

## Minimal checks

- Role boundary: direct-z task
  `019fbb9a-e810-7330-ac55-5b72b7c24337`; dedicated ignored boundary file
  SHA-256 `796134ea862eead0ea85e0dea2d46994c663b7d2a5cd2e00b11afea7552cbb0f`.
  The public GH boundary file was not changed.
- Git/role at preflight: `main`, `HEAD == origin/main`, agent
  `head-server1-gh/global-head/server1`. Submission remains blocked until the
  new tracked files are committed, pushed, and the worktree is clean.
- Fixed artifact preflight:
  - Llama manifest `273d06af367904aac61931331e6aa2ea09204b0db1f6c7557bca16a9e2c84d2b`
    over 25 pinned files.
  - Qwen manifest `d247b27edf638b7a9625954e8bf2aeac745b61247230f8bcf733e991a3ed9d16`
    over 25 pinned files.
  - EasyEdit bridge manifest
    `48b073331dd46293941f96d5f800aa358d966498d6b0336252ee37c0d91975bf`
    over 13 pinned source files.
- EasyEdit is treated as an external read-only working tree. Its pre-existing
  porcelain state hash at preflight is
  `f0570aea6efb0b989bff971b5af0dbd20737b808f0127e133e1169df48176f27`;
  the experiment may not alter that state.
- Existing Wikipedia moments are identity-checked and loaded read-only by the
  runner. Dataset download/recomputation and projector deserialization are not
  part of this MEMIT possibility track.
- Focused CPU checks: helper/runner/analyzer unit tests `30/30 PASS`; full
  motivation suite `222/222 PASS`; runner/helper/analyzer compile `PASS`;
  three launcher `bash -n PASS`; `git diff --check PASS`.
- Resource gate:
  `active_project_gpus=2 + requested_gpus=2 <= cap=4`, requested memory
  `130000M <= 396234M`, decision `submit_now`.
- Scalar firewall: raw z, hidden states, logits, prompts, token IDs, and dense
  updates are not serialized. Outcomes are written only after an exclusive
  feature/action receipt.
- Analysis contract: exact `8 x 6 = 48` rows, deterministic 4000-resample
  bootstrap, ITD denominator retained, technical failures block, scientific
  direction uses the locked lenient possibility classification.

- 최종 판정: `PASS` — user-authorized direct-z possibility pair 1회에만 유효
