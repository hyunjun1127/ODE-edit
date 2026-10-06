# FE-MEMIT sequential2k adapter

Authority: `plans/global/fe-sequential-2k/contract.json` and current W&B policy.
This is one fixed-prefix2000 BS100×20 profile, not the paper's bulk2000 table.

Implementation mapping:

- `adapter.py`: pinned upstream subject-last token function via exact AST-only loader; native35 loss/34 Adam, seven contexts with physicalMB1 activation recompute; canonical absolute-z4 replay; FP64 dense FE solve, prewrite Gram/history and FP64 W+delta.
- `prepare.py`: readonly upstream SHA, current local asset stat linked to prior fullSHA, all2000 occurrence/token identities,26k observer rows, source/runtime/resource plan.
- `run.py`: persistent W0 fit2000 then replay2000 RAM table; cold H0,20 own-state commits,100 history appends; W0/pre/post/milestone evaluation and RAM rollback.
- `collect.py`: independent NLL preference/TF/harmonic/tail and exact paired/cohort CPU arithmetic, failure-aware completion report.
- `submit.py`: one GPU runner + afterany CPU collector, cap-safe dependency, exact held inspection/release; requires online tracking readiness before registration.

Status: CPU tests passed; **NOT_GPU_QUALIFIED, NOT_SUBMITTED**. Current policy requires SH1 shared logger binding and successful online setup. The shared logger is not implemented here. Existing preparation config is historical CPU evidence, not a final executable run lock.

No model/H/targets/delta/optimizer/RNG tensor is saved. Activation checkpointing is recompute in RAM, not a durable model checkpoint. Failure stops the affected persistent run; no automatic retry or exact crash-resume.

CPU reproduction (create-once receipt name):

```bash
python -B -m project.run_scripts.fe_baseline.preflight --receipt-name cpu-new
```

Scientific environment remains pinned separately from the isolated W&B SDK. No shared EasyEdit/native/upstream files are changed. External upstream checkout stays ignored/local and is not vendored into Git.
