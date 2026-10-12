# Frozen FP32 post-run audit: exact scheduler step reconstruction

The frozen2632 collector supplied only SLURM_JOB_ID when reconstructing config.
A recorded `SLURM_STEP_ID=-5` therefore made the otherwise identical tracking
config fail. This separate CPU auditor does **not** edit/patch that source or
its result, rerun science, or ignore the step field.

Use the run's **original frozen official package** on PYTHONPATH (2632 module
SHA3b0dc0cf355b2ae21e2d57dec12264ed782ab7e979bd19b8e729aec291541180),
and put only this new auditor file in a separate tool directory:

```
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/path/to/frozen/source python /new/tools/author_basic_eot_fp32_audit.py AUDIT_BINDINGS NEW_REPORT.json
```

Audit bindings JSON: schema `author-basic-eot-fp32-postrun-audit-v1`, and three
fresh path/bytes/SHA256 members: `result` (actual benchmark-result.json),
`tracking_identity` (tracking/identity.json), `tracking_receipt`
(tracking/receipt.json). Bind terminal receipt bytes after the worker finishes.
Missing/partial scientific W20 is rejected. No historical result fallback.

Crosschecks runtime tracking against immutable tracker config/complete job
identity and the receipt's independent startup readback, run ID/name and final
identity when present. Scientific runtime/result/CP identity and execution
source/job are checked too. The frozen scientific identity has job_id but not a
step field; that is not fabricated. Any optional nested job_identity must agree.
Only then reconstruct actual SLURM_JOB_ID plus recorded optional step/array
fields. Full regenerated tracking config must match, including step. Rent uses
ODEEDIT_RENT_JOB_ID with no manufactured step. No scheduler/network query.

All old author YAML, frozen source, full assets, ordered original2000, FP32
runtime/backend/TF32, 20 commits/25 endpoints, public query tokenization and
independent raw reducers, W20 W/H checkpoint SHA/integrity guards remain.
Only tokenizer CPU loading and checkpoint mmap/hash occur; no model or forward.
Receipt transport status/drops are reported separately, not called scientific
success or online history parity. The old in-job collector failure is not
rewritten/relabelled; this creates a new independent audit report only.
