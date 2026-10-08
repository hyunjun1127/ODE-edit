# SH4 Qwen 12 prospective checkpoint receiver

Request: USER-SH4-SH1-QWEN12-CHECKPOINT-DESTINATION-20261009-R1.
Observed UTC: 2026-10-08T19:07:44.350605+00:00.
Status: TASK_ROOT_PREPARED; SH4_CUTOVER_REGISTRATION_PENDING. No payload admission yet.

Created canonical task root:
`/mnt/raid5/janghj/ODE-edit/local/checkpoint-archive/server4/qwen-baselines-server4-20261009/`

Final member layout remains `<run>/<attempt>/job-<actual_job_id>/<full_checkpoint_sha256>/`.
Do not rsync into a guessed final directory: transfer only the sealed manifest's
exact files and incoming names into the staging path returned by `Receiver.admit`.

## Actual readiness

- Helper source: `0a536420bc47870980ff7c1449ab5d721f762534` (unchanged).
- Checked checkout HEAD: `fcca2007eefe485e290d3360f0f356fd3bab4fe0`.
- Policy file SHA256: `6da0d1fcb576a51610daf3a733b896cf48a373dc60a3d8aa3c2ec1405ecfd9da`.
- Available bytes: 1,251,784,515,584; available inodes: 334,973,931.
- Planning capacity tested: 94,103,404,544 bytes (twice reported total plus 12 metadata budgets), plus 68,719,476,736-byte operational reserve.
- Receiver nonblocking lock acquired; outstanding reservations 0; max concurrent receiver 1.
- This is a snapshot, not a durable reservation for all twelve runs. Per-payload fresh admission remains required.
- Root/task directory traversal uses the existing no-symlink dirfd helper.
- CPU storage fixtures: `python3 -m unittest project.run_scripts.checkpoint_archive.test_archive -q`: 46 PASS.
- Initial local invocation mistakes (unittest slash syntax, relative policy path, overly broad metadata member) were rejected; corrected calls passed. No helper changes or weakened validation.
- Transfer bytes 0; deletion bytes 0; actual job IDs supplied 0; GPU/Slurm/job changes 0.

## Caller contract / missing input

Use the existing `project/run_scripts/checkpoint_archive/README.md` API, not a duplicate helper.
SH4 must provide its actual prospective cutover receipt as exact `{path, bytes, sha256}`
and small original JSON bytes for independent receiver pinning. Required schema is
`final-checkpoint-archive-server-cutover-v1`, server `server4`, common archive
instruction, scope `NEW_SUBMISSIONS_ONLY`, policy file SHA, real ACK nonce and
actual timezone-qualified received_at_utc. Do not invent historical times.
Seal adoption/submission lock before new sbatch; record actual registration proof after returned ID.

API: `Receiver.from_policy_file(absolute_policy_path, cutover_members=trusted_map)`;
`seal_manifest(...)`; receiver `admit(manifest)`; exact one-way allowlist transfer;
`verify(admission, manifest)`; independent `recheck(receipt)`;
source `source_delete_gate(...)`. The latter never unlinks files and eligibility
is not deletion. CLI provides admit/verify; recheck is a Python API.

Receiver CLI (all argument paths absolute; OUT must be new):
```
python3 -m project.run_scripts.checkpoint_archive admit --policy POLICY --cutovers TRUSTED_MAP --manifest MANIFEST --receipt-output OUT
python3 -m project.run_scripts.checkpoint_archive verify --policy POLICY --cutovers TRUSTED_MAP --manifest MANIFEST --admission ADMISSION --receipt-output OUT
```

Seven original companion roles and frozen source evidence-adapter replay are
mandatory, as documented in the helper README. No per-run destination admission
can be fabricated without actual job/source/final payload identity.

Important: the current receiver requires all planned consumers clear already at
admission. CF deferred evaluation still needing its checkpoint does not satisfy
this contract. Keep source; do not claim job termination alone permits transfer
and deletion, or silently mark future consumers complete. A copy-only pending-
consumer workflow/central consumer handoff is not implemented by this preparation.
Thus sequential cleanup of all twelve files is not yet guaranteed by this root.

No existing checkpoint/source/job was changed. No payload broadcast is required
for preparation (NO_BROADCAST_NOT_REQUIRED); only this compact receipt is shared.
