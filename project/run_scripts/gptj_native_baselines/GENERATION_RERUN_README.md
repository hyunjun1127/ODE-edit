# GPT-J six-native generation rerun: SH1 source/reference bound

## Manual USER recall: cache-repair-r2 (2026-10-08)

Six r1 jobs61160..61165 failed at ONLINE_STARTUP before model load. Their
sidecars actually reached READY_ONLINE, but the parent immutable identity
validator rejected the private qualification_plan_sha256 config key. New
telemetry uses the already-supported generation_qualification_plan_sha256 key;
the internal scientific PLAN key is unchanged. A real parent pipe-reader CPU
test now covers READY_ONLINE -> immutable identity, not only mocked SDK READY.
Old runs/source/logs are preserved, never renamed or backfilled.

The explicit --profile r2 writes only local/cache-repair-r2 (under the parent
local root), binds direct USER recall metadata and new run UUIDs, and preserves
the r1 default. Native budgets/methods/inputs/noCP are unchanged. The published
SH1 rich metadata compatibility fix1413ac0f/treee6935b9a is read-only reused.
No prior actual qualification receipt exists; the first new GPU job writes it.

One deliberate registration pass, after source commit and fresh admission:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.gptj_native_baselines.generation_cache_bind --profile r2 --shared-source 1413ac0ff550f2132feda355e44519506fc0eb84 --package-tree e6935b9a099b14d7ad8e5e39e5455da08b2159f8
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.gptj_native_baselines.generation_cache_cpu_checks --profile r2 --out /mnt/raid5/janghj/ODE-edit/local/gptj-baselines-fluency-consistency-2k/cache-repair-r2/cpu-integration-r2.json
PYTHONDONTWRITEBYTECODE=1 /mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.gptj_native_baselines.generation_cache_submit --profile r2
```

These are create-once commands, not an automatic retry or monitoring loop.
Scalar realtime logging retains actual job name/config IDs, separate progress,
fit and performance axes, immutable run identity and bounded remote readback.
CPU99 PASS is not actual GPU qualification or W20 completion.

## Historical original registration (unchanged)

Task: `gptj-baselines-fluency-consistency-2k`.
Nonce: `USER-GH-ALL-SH-BASELINE-FLUENCY-CONSISTENCY-RERUN-20261007-R1`.

SH1 shared source83535c6a / tree6b9ed049 and reference identity75e595c7 are
bound through actual load_assets/observe/subset/read_observed/generation_payload.
Exact received manifest/member hashes and native environment versions match.
This readiness is not GPU or submission PASS. Production config is immutable
preparation-r2/config.json; historical provisional preparation-r1 remains.
generation_bridge preserves raw receipts/work/identity and atomic cold-W0
READY. generation_submit registers seven held jobs before release; collector
independently reduces stored rows, without loading a model or creating a run.

`generation_prepare` reuses the exact existing stock/four-arm model, native
closures, hparams, C0/P, schedule, runtime and R/P/N scorer using small-source
hashes and previous heavy-asset SHA plus current stat. It writes create-once
provisional metadata; no tensor/model load, checkpoint or shared-source edit.

`generation_run` dispatches stock MEMIT/AlphaEdit and CAKE/BLUE/PRUNE/RECT to
their existing factories. It carries native H, integrates true pre and post
observers, and derives milestone current metrics from one measured prefix.
The SH2 bridge must adapt SH1 code, not reimplement the generator or score
formulas. It must validate exact ordered occurrence IDs, metric sums/counts,
typed missing reasons and state/profile identity; restore all RNG in `finally`;
and reuse cold W0 observations, including B1 pre, without duplicate generation.
`locked()` rejects unbound source/reference before online/model startup.

Production telemetry now reuses SH1 shared logger/schema and generation_payload
from raw summary. Private copies and CPU26 are preserved historical fixtures,
not production transport. Their old assertion that shared generation keys are
absent predates SH1 adoption and is not current integration evidence.
Shared files are read-only; means keep bits/cosine, missing means are omitted.

`generation_plan` is a pure future DAG/count plan. Within effective cap2,
BASE_MEMIT is the sole cold W0 generation publisher. After it terminates,
two independent lanes are BASE_ALPHAEDIT → ALPHAEDIT_BLUE → RECT and
CAKE → PRUNE. With stricter cap1, all six are serialized. CPU collector is
afterany all six, not a performance gate. Real submission must freshly resolve
all own ODE GPU allocations/admitted frontier, actual assets/resources and
inspect seven held jobs before release. No job IDs are guessed here.

CPU fixtures (no SDK/network/model/GPU/Slurm):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.gptj_native_baselines.generation_cpu_checks --receipt-name cpu-checks-preparation-r2.json
```

Source/status/report publication is separate from scientific completion.
Focused integration receipt is cpu-integration-r1.json; historical CPU44 is
not repeated as actual GPU qualification. generation_bind and generation_submit
are create-once, not commands to repeat existing preparation or job registration.
Receipts are immutable; choose a new `rN` receipt name for a deliberate later
CPU reproduction, rather than overwriting an earlier receipt.
NoCP, no automatic retry/monitor; old outputs and protected ours/W0/FE jobs
are preserved.
