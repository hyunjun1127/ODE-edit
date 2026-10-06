# FE-MEMIT sequential baseline adapter

Authority: `plans/global/fe-sequential-2k/contract.json` and current W&B policy.
Default profile is fixed-prefix2000 BS100×20. The current explicit USER override
(`plans/updates/server2/fe-sequential-2k/user-10k.md`) selects the same method on
fixed10k BS100×100, job `fe-sequential-10k`. Neither is the paper's bulk2000 table.
The historical `fe-sequential-2k` branch/local/report namespace is retained.

Implementation mapping:

- `adapter.py`: pinned upstream subject-last token function via exact AST-only loader; native35 loss/34 Adam, seven contexts with physicalMB1 activation recompute; canonical absolute-z4 replay; FP64 dense FE solve, prewrite Gram/history and FP64 W+delta.
- `prepare.py` / `profile.py`: readonly upstream SHA, local asset stat linked to prior fullSHA, explicit horizon occurrence/token identities, source/runtime/resource plan. 10k:130k observer identities,819200000-byte RAM target table,500 solves/history appends.
- `run.py`: persistent W0 fit-all then replay-all RAM table; cold H0 and own-state chain, W0/current pre/post/every-five-batch all-seen evaluation and RAM rollback. W100 final R10000/P20000/N100000;99 own-state links.
- `collect.py`: independent NLL preference/TF/harmonic/tail and exact paired/cohort CPU arithmetic, failure-aware completion report.
- `submit.py`: one GPU runner + afterany CPU collector, cap-safe dependency, exact held inspection/release; requires online tracking readiness before registration.

Source qualification is CPU-only, not an actual Llama PASS. Prior CPU7 math evidence is preserved; `resume_preflight.py` checks only horizon/telemetry/resources and proves the adapter differs solely by a scalar callback. Current registration status is recorded separately under the server2 report/status.

`telemetry.py` maps only already-computed finite scalar values to SH1's read-only
`experiment_tracking.init/log/finish`; it does not implement another logger.
Startup online authentication precedes model loading. SDK0.30.0 is isolated from
the scientific Python. One synthetic3-point smoke/readback passed; that is not
GPU model validation. No raw prompts, tensors, credentials or code are uploaded.
Phase IDs:0 startup,1 W0,2 fit,3 replay,4 write,5 evaluation,6 terminal;
TF cohort IDs1/2/3 correspond to R/P/N. Evaluation denominators travel with scores.

No model/H/targets/delta/optimizer/RNG tensor is saved. Activation checkpointing is recompute in RAM, not a durable model checkpoint. Failure stops the affected persistent run; no automatic retry or exact crash-resume.

CPU reproduction (create-once receipt name):

```bash
python -B -m project.run_scripts.fe_baseline.preflight --receipt-name cpu-new
```

The current narrow resume checks are `python -B -m project.run_scripts.fe_baseline.resume_preflight` (create-once local receipt).
Use `prepare --requests 10000 --out-name preparation-online-10k-r1 --cpu <receipt> --online-receipt <smoke/result.json>` then `submit --config <configuration.json> --attempt-name attempt-online-10k-r1`.
48h is the inherited request ceiling, not a measured completion ETA. Interrupted
noCP runs cannot exact-resume; no automatic extra runs are registered.

Scientific environment remains pinned separately from the isolated W&B SDK. No shared EasyEdit/native/upstream files are changed. External upstream checkout stays ignored/local and is not vendored into Git.
