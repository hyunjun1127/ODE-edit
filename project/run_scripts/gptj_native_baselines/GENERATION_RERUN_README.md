# GPT-J six-native generation rerun: provisional CPU preparation

Task: `gptj-baselines-fluency-consistency-2k`.
Nonce: `USER-GH-ALL-SH-BASELINE-FLUENCY-CONSISTENCY-RERUN-20261007-R1`.

This is **not an executable source freeze or a Slurm submission**. The new
caller is CPU-fixture tested, but SH1's shared generation source/API and exact
reference READY bundle have not been bound. The `generation_bridge` adapter,
actual reference receive/final config, submission and independent generation
reducer integration remain pending that authoritative interface. Do not infer
GPU/generation qualification from fixtures or fabricate a READY config.

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

The task-private telemetry transport differs from the immutable common
client/worker only in import routing, sidecar module and declared generation
metric axes. Original SDK/auth/privacy/job identity/run UUID/readback remain.
Shared `experiment_tracking` files are not edited. Scalar means retain bits
and cosine units; unavailable means are omitted, not zero-filled.

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
Receipts are immutable; choose a new `rN` receipt name for a deliberate later
CPU reproduction, rather than overwriting an earlier receipt.
NoCP, no automatic retry/monitor; old outputs and protected ours/W0/FE jobs
are preserved.
