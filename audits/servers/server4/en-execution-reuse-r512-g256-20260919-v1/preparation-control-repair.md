# Preparation control-only inspection repair

Execution source a297039dc756a0e8953e4bab4695e66361a161ac and preparation lock bytes remain unchanged. Job50410 was submitted held; first inspection rejected the scheduler `Command` value because Slurm records only the script there. `SubmitLine` contains the full submitted source/lock/stage argv.

New control parses exact Command and SubmitLine independently, checks exact owner, state/reason, node, CPU/memory/GPU and no-array/no-dependency fields, and releases the same allowlisted job. Prefix-adversarial server40/CPU80/GPU10 tests are rejected. No duplicate job, job-configuration change or scientific-runtime change; no model had run while held.

Original `held-inspection.json`, frozen source and lock were not overwritten. New `held-inspection-control-r1.json` and `submission.json` are local receipts. This is a control defect, not a scientific numerical failure. All actual model validation remains pending.

Independent bounded read-only reviewer found the READY-property call defect, large-asset entry revalidation gap, incremental storage double-count, CPU/dependency/lock binding gaps and substring resource issue before actual model execution. Parent fixed them and ran97 CPU tests, then10 focused boundary tests for the observed Command/SubmitLine distinction. CPU tests and source review are not real-model validation.
