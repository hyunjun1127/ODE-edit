# Prospective final checkpoint archive

This is storage/source-policy code, not a scientific runner. It uses only the
Python standard library and never deserializes a checkpoint, runs a model,
submits Slurm, opens a network connection, copies a payload, or unlinks a source.
The actual source owner supplies the separately approved one-shot transport and,
if later eligible, the final individual source unlink.

Latest authority is `USER-GH-ALL-SH-FINAL-CHECKPOINT-ARCHIVE-SERVER1-20261009-R1`.
The prospective correction in main `f9a6084ab050c1283ff3910e75c2032a81bc5997`
has policy **file** SHA256
`6da0d1fcb576a51610daf3a733b896cf48a373dc60a3d8aa3c2ec1405ecfd9da`.
Every already registered PENDING/RUNNING/terminal job and its checkpoints stays
KEEP. Changing a Boolean, timestamp, filename or job label cannot opt it in.
There is no new checkpoint permission for an existing noCP experiment.

## API

```python
from project.run_scripts.checkpoint_archive import Receiver, seal_manifest, source_delete_gate

# Metadata is recorded by the actual SH at its policy cutover, then independently
# approved/pinned at the receiver. Do not invent a cutover for a historical job.
receiver = Receiver.from_policy_file(
    canonical_policy_path,
    cutover_members={"server1": {"path": cutover_path, "bytes": size, "sha256": full_sha}},
)
manifest = seal_manifest(
    run=full_actual_run_identity,
    identity=full_checkpoint_identity,
    checkpoint=exact_final_payload_path,
    companions=seven_companion_paths,
    provenance_references=source_and_asset_references,
    policy_sha256=actual_policy_file_sha,
    cutover_member=original_cutover_member,
    evidence_adapter=source_owned_frozen_adapter,
)
admission = receiver.admit(manifest)
# The caller transports only admission.incoming_names from manifest.files to
# admission.staging. No recursion, glob, delete flag or directory synchronization.
receipt = receiver.verify(admission, manifest)
# This re-hashes every preserved received member, including the small originals.
receiver.recheck(receipt)

# A source-side trusted receiver client may invoke the independent recheck over
# the already authenticated channel. An rsync/Slurm exit status is not an ACK.
gate = source_delete_gate(
    receipt, manifest,
    fresh_companions=fresh_writer_and_all_consumer_evidence_paths,
    destination_verify=independent_authenticated_receiver_recheck,
    evidence_adapter=same_source_owned_frozen_adapter,
)
# ELIGIBLE_SOURCE_OWNER_FINAL_RECHECK_REQUIRED is NOT SOURCE_REMOVED.
```

`root=` overrides are refused in production; tests explicitly use `test_only`.
Production construction binds a current full-SHA policy file, not a normalized
JSON digest. The receiver checks independently pinned cutover metadata again on
every admission/verification. Only server1 deploys the receiver.

## Identity and original proof contract

`run` has exactly `origin_server`, `task_id`, `run_id`, `attempt`, actual positive
string `actual_job_id`, and `registration_attempt_id`. `identity` has exactly
`config_sha256`, `stream_sha256`, `code_commit`, `official_tree_sha256`,
`model_revision`, `tokenizer_sha256`, and `assets_sha256`. `code_commit`,
`official_tree_sha256` (the historical field name) and `model_revision` are exact
40-hex Git identifiers; config/stream/tokenizer/assets are exact64-hex SHA256
content hashes, except that `official_tree_sha256` also accepts the existing
server3/server4 runner's 64-hex sorted-file content SHA256. Its original value
and the checkpoint identity digest are never converted or relabeled. New archive
manifests bind `checkpoint_identity_types`: this field is `git-tree-sha1` for
40 hex or `content-sha256` for 64 hex. Commit/revision remain exactly 40 hex;
the other content fields remain exactly 64 hex. A missing or conflicting type
binding on a 64-hex tree manifest is rejected even after manifest re-signing.
Legacy untyped 40-hex manifests remain supported without rewriting receipts.
This is typed provenance, not proof that a source tree was recomputed: the
unchanged original source lock/evidence replay must still bind the exact value.
Values are full hashes, never dummy IDs. Paths/tree/hashes retain provenance.

Cutover schema is `final-checkpoint-archive-server-cutover-v1`, with `server`,
`instruction_id`, `scope=NEW_SUBMISSIONS_ONLY`, actual policy-file `policy_sha256`,
`ack_nonce`, and timezone-qualified `received_at_utc`. The source and receiver
pin its exact `{path, bytes, sha256}` member. The receiver needs a trusted actual
cutover receipt, not a source-created per-run substitute.

The seven small companion roles are:

| Role | Required source-owned evidence |
| --- | --- |
| `latest_pointer` | Actual official `latest.json`: integer batch20, final_W20 true, exact payload file/fullSHA and checkpoint identity digest; same payload directory. |
| `adoption` | `final-checkpoint-archive-adoption-v1`; scope/instruction/policy, candidate run identity without job ID, cutover receipt SHA, actual adoption UTC. |
| `submission_lock` | `final-checkpoint-archive-submission-lock-v1`; exact candidate/checkpoint identity, adoption/policy SHA, pre-submit seal UTC, checkpoint directory and already-approved creation authority, required final-calculation inventory, all planned consumer IDs/kinds, exact evidence-adapter contract. |
| `submission_receipt` | `final-checkpoint-archive-actual-submission-v1`; actual returned job ID/response, full run/checkpoint identity, original adoption/lock SHA, actual scheduler submission UTC and later actual receipt UTC, every consumer's actual job-ID binding. |
| `scientific_terminal` | `final-checkpoint-scientific-terminal-v1`; exact run/checkpoint/payload/pointer, actual completed W20/2000 and commits1..20, every preplanned final calculation COMPLETE at W20, and each original evidence member. |
| `writer_termination` | `final-checkpoint-writer-stopped-v1`; same identity/payload/pointer, exact writer job, STOPPED_NO_FUTURE_MUTATION, closed lease, stop UTC and original writer evidence. |
| `consumer_clearance` | `final-checkpoint-all-consumers-clear-v1`; same identity/payload/pointer; complete inventory/no unlisted dependents; every planned consumer's kind/actual job/state/original evidence; explicit not-in-use and check UTC after writer stop. |

Chronology is receiver-approved cutover <= source adoption <= lock seal <
actual scheduler submission <= actual receipt seal < actual completion <= writer
stop <= all-consumer check <= current time. Actual registration proof is written
AFTER a real returned ID; the adoption/source/config lock is sealed BEFORE sbatch.
No COMPLETE/exit0/afterany/filename shortcut is accepted. Missing planned final
calculations or even one registered resume/evaluation/collector consumer HOLD.

### Frozen source-owned original-evidence adapter

`submission_lock.evidence_adapter` and
`provenance_references.evidence_adapter` must be the same exact contract:

```json
{"module": "approved.module", "function": "replay_archive_evidence",
 "source_member": {"path": "/absolute/frozen/module.py", "bytes": 1234,
                   "sha256": "FULL_SHA256"}}
```

The module file must match the actual callback's `__code__.co_filename`, module
and function, fullSHA and bytes. The source verifier hashes/reads all original
authority/calculation/writer/consumer JSON members before calling:

```python
adapter(kind, subject, original_document, expected, original_member, adapter_contract)
```

`kind` is `checkpoint_authority`, `final_calculation`, `writer_termination` or
`consumer_clearance`. `expected` binds original run/config/source/input and
payload/pointer/W20, planned calculation or consumer identity, actual job/state;
consumer replay also receives the precise fresh check UTC. The adapter must
recognize its actual original receipt schema, extract/compare the original
values, and reject missing/unknown/wrong-source/in-use/incomplete evidence.
It must not manufacture its result by copying expectations without replay.

Return exactly `final-checkpoint-original-evidence-replay-v1` with `kind`,
`subject`, `original_member`, `original_content_sha256` (canonical JSON content
digest), `adapter_source_sha256`, `expected`,
`status=SOURCE_OWNED_ORIGINAL_EVIDENCE_REPLAY_PASS`, and exact typed-true checks
`original_schema_recognized`, `identity_matched`, `endpoint_matched`,
`actual_status_matched`, `planned_scope_matched`. `_replay_result` describes the
exact resulting shape; using it does NOT replace actual original-data checks.
Generic Boolean/PASS-only callbacks are refused. No default scientific adapter
is shipped: unknown source schemas are `ARCHIVE_PENDING_KEEP_SOURCE` until the
source owner implements and reviews its bounded read-only replay.

The manifest retains original paths/fullSHA/stat plus this source-owned replay.
The complete sealed source manifest, including pinned restore provenance,
original identity/cutover and replay metadata, is embedded in the create-once
`VERIFIED_DESTINATION.json`. The receiver revalidates it on current proof recheck;
it does not rely on an unspecified external manifest file remaining available.
Receipts are bounded to the reserved1MiB metadata budget.
Each unique small original JSON member is added to the exact allowlist as
`original_0000`, etc. The receiver independently streams fullSHA/bytes, reads all
transferred originals and checks their content digests/replay bindings. Storage
verification is explicitly **not independent scientific/GPU certification**.
Large raw observations/text/token rows and credentials are not suitable small
proof members and must never be packaged/uploaded as archive metadata.

## Filesystem and states

Layout is exactly
`<origin_server>/<task_id>/<run_id>/<attempt>/job-<actual_job_id>/<fullSHA>/`.
The destination and receipts are create-once; Linux `renameat2(RENAME_NOREPLACE)`
also refuses an existing empty destination. Files use O_NOFOLLOW regular-file
descriptors, nlink1, ancestor no-symlink traversal, and before/after SHA/stat/path
checks. A source and destination path/object/hardlink alias is HOLD. Server1's
own checkpoint is already central: register/protect it and KEEP, with unlink0.
Even a separately preserved and independently verified copy never makes an
origin-server1 payload deletion-eligible; the deletion gate raises
`SERVER1_SOURCE_REGISTER_AND_PROTECT_KEEP_NO_UNLINK`.

One receiver slot is reserved under nonblocking flock. Capacity includes twice
the whole exact allowlist (payload plus staging/partial contingency), bounded
receipt overhead and at least64GiB operating reserve, with inode headroom.
Outstanding reservations—including a crash between reservation and active
pointer—block the next admission. They are not silently discarded or retried.
Files and directories are fsynced before atomic sealing. The exact staging
allowlist is rechecked before and after writing the receipt. Concurrent incoming
files, changed members or reservation corruption never produce a valid seal.

The source-delete helper replays fresh originals via the same frozen adapter,
rechecks independent destination preservation and the current source's fullSHA,
inode/dev/nlink/bytes/mtime. It only returns
`ELIGIBLE_SOURCE_OWNER_FINAL_RECHECK_REQUIRED`, with logical bytes and allocated
bytes separate. The source owner still must immediately pin the parent dirfd,
recheck hash/stat and all consumers and individually unlink the exact approved
payload. This module implements **no unlink**, recursive/glob deletion, network
transport, administrator repair, historical scan or automatic retry. Metadata,
latest pointer, source/raw/log and source-owned receipts remain KEEP.

## CLI and CPU verification

```text
python3 -m project.run_scripts.checkpoint_archive admit \
  --policy /exact/current/control/final-checkpoint-archive-policy.json \
  --cutovers /exact/trusted/server-cutover-members.json \
  --manifest /exact/new-source-manifest.json --receipt-output /exact/new-admission.json

python3 -m project.run_scripts.checkpoint_archive verify \
  --policy /exact/current/control/final-checkpoint-archive-policy.json \
  --cutovers /exact/trusted/server-cutover-members.json \
  --manifest /exact/new-source-manifest.json --admission /exact/new-admission.json \
  --receipt-output /exact/new-verification-receipt.json

python3 -m unittest project.run_scripts.checkpoint_archive.test_archive -q
```

The tests create explicitly labeled tiny CPU fixtures in temporary directories.
They exercise prospective chronology, original proof replay, full hashes, aliases,
source/destination mutation, reserve/orphan/concurrency, atomic collision handling
and metadata-only eligibility. They are not real-job, transfer, deletion, model,
GPU qualification or scientific-completion evidence. Actual caller adoption and
receiver READY/deployment remain separate parent-owned steps.
