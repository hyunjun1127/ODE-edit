# A implementation / execution start

Instruction: ODEEDIT-S06-SINGLE-LAYER-CUMULATIVE-RISK-ABC-SH1-V1.
Control override: ODEEDIT-GH-SH1-CUMRISK-AUTONOMY-20260910-R1.
SH1 owns A/B/C completion and stage transitions; no additional GH release required.
GPU order remains A then B then C. Scientific promotion is false.

Base HEAD 627139777a347f543f82749ef58cf4da82116031;
base tree 0835b5c781242f82fcd3c7a69fc78d3373a6ec71.
Execution is frozen in a separate worktree; original BLUE/EasyEdit files are read-only.

Full-read input SHA256:

- User goal: c54122b0cb96ffae4320aeff99ad938e63b7ec937f314a93c7e7cf1f7da8a117.
- 514-line design: 1b9c0e35b0115fe79c0e376775070b81abff8f4420f32625bf1332752032ee5c.
- 403-line discussion: 15f62756dfec69a1d34fdbed16d1bab2b2af1b9abf5c1eadf3251627400f35e7.

Local input.lock.json SHA256:
33cb332962444b4aad0d244bf70028b9e865b318626d6f74dbe60adda9648821.
The separate immutable control-override.json supersedes the old B/C release flag
in the input lock without rewriting those input bytes.

The first executable scope is A native preparation plus independent direct
candidates. B/C runtime and analysis remain to be completed. No claim of a
completed campaign is made by this source commit.

Native readout delegates to the original BLUE representation reader in physical
microbatches of two, preserving all contexts and ordered rows. The native solve,
history finalization, cached target identity, and stored projector are unchanged.
Direct objective uses full-weight functional execution, logical 100 requests and
all six contexts, request microbatch two/context microbatch one. Native essence
is KL(student || frozen entry teacher), coefficient 0.0625 once. Learning-rate
selection uses post-update W29..W32 common training objectives and requires an
actually stored finite W32 endpoint. PS/NS influence on selection is zero.

Resource policy: server1/devbox cap2; one GPU/process; 8 CPUs;
explicit job memory 182272 MiB; initial wall request 48 hours.
No inherited GPU-hour budget or scope expansion. Measure first execution before
remaining cost estimates. Existing unrelated allocations are untouched.

Reproduction: run the package's test_focused unittest module, compileall and
bash -n run.sbatch. Focused algebra/contract checks are not GPU numerical PASS.
First-valid requires actual input/readout/update/evaluation progress.
