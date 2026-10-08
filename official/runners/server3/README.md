# server3 Qwen2.5 official baseline runner

`run.py` binds the six native implementations through `official.baselines.registry`.
`assets.py` verifies the server3 EasyEdit **assets** (not EasyEdit algorithm code),
physical L4–L8 C0/projector mapping, source streams, model revision and runtime.
`native_state.py` owns each cold chain's W, native history, context and RNG.
`submit.py` freezes the exact published `origin/main` source and registers held
Slurm jobs only after asset, W&B, storage and cap checks.

The Qwen plan has 12 main logical rows plus five CF auxiliary rows. The CF
BLUE main row aliases the selected L2 grid result, leaving 16 physical 2K
editing chains. Stages are `qualify` (shared CF W0 and six B3 resume-parity
jobs), `cf` (ten physical chains), and `zsre` (shared W0, one independent FT
B1 smoke, six physical chains). Each stage is serialized on one server3 GPU.
The CF BLUE winner is selected from exact W20 receipts with the published
Score/Specificity/Efficacy/Generalization/L2 tie order; zsRE uses that L2.

Run `python -m official.runners.server3.submit plan --matrix-root MATRIX
--output-root OUTPUT --stage qualify` for a read-only mapping. The mutable
commands are `submit-held`, `release`, `select-blue`, and `resume-held`;
`--help` lists their exact arguments. Submission requires the implementation
to be merged into `origin/main`, a clean official tree, the exact local
manifest, the Slurm policy and an authenticated online W&B project. The
Slurm launcher uses `--export=NONE` and a verified HOME credential source.
Do not run from an unpublished worktree or substitute missing model/reference
assets by recomputation or download.

Each edit chain starts at cold W0. The checkpoint stores only editable FP32 W,
native `cache_c` where applicable, context, RNG, evaluation cursor and
identity. An immutable pending receipt is bound into its cursor before the
checkpoint pointer advances; resume verifies it and completes a missing small
commit/evaluation receipt without re-running that batch. W20 is retained.
`qualify` compares actual continuous B3 with B2→B3 reload using identical
W/history/context/RNG/cursor hashes and factual raw. CPU tests alone do not
establish native GPU parity.

The common factual forward evaluator is owned by server1 at
`official.evaluation.factual`. Until its actual `evaluate_cases` API and raw
schema are published and checked against this caller, server3 preflight blocks
science. The current shared W&B scalar schema also fixes CounterFact W0 counts
and cannot truthfully accept zsRE's distinct W0 counts; the zsRE stage blocks
until a shared schema extension is published. The server3 folder does not
fork either shared evaluator or logger.

Failed physical edit jobs have an identity-checked, held checkpoint-resume
path. A failed shared W0, qualification or zsRE smoke prerequisite requires a
separately reviewed immutable retry and dependency repair; this submitter
does not silently recreate those jobs. Scientific receipts and W&B transport
receipts are separate. A queued scalar is not remote delivery proof.
