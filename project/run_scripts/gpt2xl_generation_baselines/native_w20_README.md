# GPT2-XL MEMIT / AlphaEdit / CAKE: W20-only generation

This explicit new task/profile authorizes only `BASE_MEMIT`, `BASE_ALPHAEDIT`
and `CAKE`. Each uses an independent cold model with the unchanged ordered
first2000, BS100 × 20 native trajectory. RPN W0, current pre/post and
W5/10/15/20 observations remain unchanged. Fluency/consistency generation runs
only on the actual W20 ordered first2000 state: no generation W0 READY,
cross-arm generation prerequisite or intermediate generation endpoints.

The new authority nonce is
`USER-GH-SH1-GPT2XL-MEMIT-ALPHAEDIT-CAKE-W20-GENERATION-20261008-R1`.
The canonical generation policy may retain the prior BLUE/PRUNE/RECT instruction
ID; this profile explicitly verifies and inherits its `W20_ONLY_FIRST2000`
schedule while binding the distinct new envelope/task/nonce. Existing
`w20_*` profile guards, arm defaults and frozen sources are not modified.

Reuse the exact prior model/token/native assets, generation reference and
prelocked ≤8-prompt qualification PLAN. Each new arm records its actual route
receipt inside its approved job. CPU/static verification is not GPU
qualification, online delivery PASS or scientific completion. Native hparams,
solver, dtype and history semantics are retained: MEMIT has no H; stock
AlphaEdit begins with an empty module history and initializes the five-layer H
on the first native call; CAKE carries its native five-layer history.

Owner CLI, using the existing native runtime Python:

1. `-m project.run_scripts.gpt2xl_generation_baselines.native_w20_prepare --out <new CPU directory> --attempt <new LOCAL attempt> --source-attempt <prior attempt-cache-repair-r2> --transition-receipt <new exact reconciliation JSON>`
2. `-m project.run_scripts.gpt2xl_generation_baselines.native_w20_preflight --config <new CPU directory/config.json>`
3. Commit the reviewed scope, then `-m project.run_scripts.gpt2xl_generation_baselines.native_w20_submit --config <new CPU directory/config-sealed.json> --attempt <new LOCAL attempt>`.

The attempt and CPU directory are create-once. The new nonce duplicate guard
blocks a started or partially registered pass as well as a completed submission;
an unknown-ID failure is not blindly retried. Old failed/cancelled source,
raw observations, partial cost and receipts remain immutable.

Fresh resource admission uses all current own server1 project GPU allocations
and the released/admitted GPU dependency graph regardless of job name. If the
protected pending BLUE/PRUNE/RECT graph is still present, both new heads follow
its GPU leaves `61437` **and** `61438`, not its root `61436` or CPU collector
`61439`. Cap2 gives MEMIT + AlphaEdit lanes, with CAKE following the new MEMIT;
a stricter cap1 serializes all three. The own GPU0 collector follows only these
three new IDs. Dependencies are resource-only `afterany`; no W0 generation or
scientific score gate is invented. Existing jobs and other owners are not
modified, and normal scheduler resource PENDING is a valid initial handoff.

All three GPU jobs and the target-only GPU0 collector are held, exact
owner/full-argv/source/input/resources/dependency inspected, then released.
Source/config/PLAN/reference/native/launcher/collector and App-versus-WT session
identity are sealed. No model/C0/P download or recomputation, durable edited
weights/H/optimizer checkpoint, historical run backfill, generated text/token
upload, periodic agent monitor or automatic retry occurs. W&B uses actual job
identity and approved scalar progress/final metrics; raw text/tokens stay local.
