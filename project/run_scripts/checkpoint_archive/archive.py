"""Exact final checkpoint admission and independent immutable receiver proof.

No tensor deserialization, transport, model/GPU/Slurm operation or source unlink
is implemented. Source owners supply the seven immutable small proof members.
The latest USER scope is prospective NEW_SUBMISSIONS_ONLY, even when an older
policy document did not yet spell out that supersession.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import datetime as dt
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid


INSTRUCTION = "USER-GH-ALL-SH-FINAL-CHECKPOINT-ARCHIVE-SERVER1-20261009-R1"
SCOPE = "NEW_SUBMISSIONS_ONLY"
MANIFEST_SCHEMA = "final-checkpoint-source-manifest-v1"
RECEIPT_SCHEMA = "final-checkpoint-VERIFIED_DESTINATION-v1"
RESERVE_BYTES = 64 * 1024**3
METADATA_BUDGET = 1024**2
MAX_COMPANION = 4 * 1024**2
ROLES = ("latest_pointer", "adoption", "submission_lock", "submission_receipt",
         "scientific_terminal", "writer_termination", "consumer_clearance")
IDENTITY_FIELDS = ("config_sha256", "stream_sha256", "code_commit", "official_tree_sha256",
                   "model_revision", "tokenizer_sha256", "assets_sha256")
GIT_IDENTITY_FIELDS = frozenset(("code_commit", "official_tree_sha256", "model_revision"))
RUN_FIELDS = ("origin_server", "task_id", "run_id", "attempt", "actual_job_id", "registration_attempt_id")


class ArchiveError(ValueError):
    def __init__(self, code, detail=""):
        self.code, self.detail = code, str(detail)
        super().__init__(code + (": " + self.detail if detail else ""))


def require(value, code, detail=""):
    if not value:
        raise ArchiveError(code, detail)


def digest(value):
    try:
        data = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ArchiveError("INVALID_FINITE_JSON") from error
    return hashlib.sha256(data).hexdigest()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def timestamp(value):
    try:
        instant = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(instant.tzinfo is not None, "TIMESTAMP_TIMEZONE_REQUIRED")
        return instant.astimezone(dt.timezone.utc)
    except (ValueError, AttributeError, TypeError) as error:
        raise ArchiveError("INVALID_UTC_TIMESTAMP") from error


def safe_name(value):
    require(type(value) is str and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value))
            and value not in (".", ".."), "UNSAFE_IDENTITY_PATH_COMPONENT")
    return value


def sha(value):
    require(type(value) is str and bool(re.fullmatch(r"[a-f0-9]{64}", value)), "FULL_SHA256_REQUIRED")


@contextmanager
def parent_fd(path, *, create=False):
    """Resolve every absolute directory with O_NOFOLLOW, not a racy resolve()."""
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts and path.name not in ("", ".", ".."), "EXACT_ABSOLUTE_PATH_REQUIRED")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for name in path.parts[1:-1]:
            if create:
                try:
                    os.mkdir(name, 0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
            try:
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=descriptor)
            except FileNotFoundError:
                raise
            except OSError as error:
                raise ArchiveError("SYMLINK_OR_UNSAFE_PATH_ANCESTOR", str(path)) from error
            os.close(descriptor); descriptor = child
        yield descriptor, path.name
    finally:
        os.close(descriptor)


def _stat_row(value):
    return dict(dev=value.st_dev, inode=value.st_ino, nlink=value.st_nlink, bytes=value.st_size,
                mtime_ns=value.st_mtime_ns, allocated_bytes=value.st_blocks * 512)


def inspect_file(path):
    """Streaming hash through one pinned fd; reject hardlinks and path races."""
    path = Path(path)
    with parent_fd(path) as (directory, name):
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        except OSError as error:
            raise ArchiveError("SOURCE_OR_DESTINATION_NOT_REGULAR", str(path)) from error
        try:
            before = os.fstat(descriptor)
            require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "SHARED_OBJECT_OR_NONREGULAR_HOLD", str(path))
            checksum = hashlib.sha256()
            while True:
                block = os.read(descriptor, 8 * 1024**2)
                if not block:
                    break
                checksum.update(block)
            after, named = os.fstat(descriptor), os.stat(name, dir_fd=directory, follow_symlinks=False)
            require(_stat_row(before) == _stat_row(after) == _stat_row(named), "FILE_CHANGED_DURING_HASH", str(path))
            return dict(path=str(path), realpath=str(path), sha256=checksum.hexdigest(), **_stat_row(after))
        finally:
            os.close(descriptor)


def read_json(path, expected=None):
    before = inspect_file(path)
    if expected:
        require(before == expected, "SOURCE_CHANGED_OR_MEMBER_MISMATCH", str(path))
    require(before["bytes"] <= MAX_COMPANION, "COMPANION_TOO_LARGE", str(path))
    with parent_fd(path) as (directory, name):
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        try:
            with os.fdopen(descriptor, "r", closefd=False) as stream:
                value = json.load(stream)
        finally:
            os.close(descriptor)
    require(inspect_file(path) == before, "FILE_CHANGED_DURING_READ", str(path))
    digest(value)  # Reject NaN/Infinity, not just decoder acceptance.
    return value


def _proof_text(path, expected):
    require(expected["bytes"] <= MAX_COMPANION, "COMPANION_TOO_LARGE")
    with parent_fd(path) as (directory,name):
        descriptor = os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=directory)
        try:
            with os.fdopen(descriptor,"rb",closefd=False) as stream:
                data = stream.read(MAX_COMPANION+1)
        finally: os.close(descriptor)
    require(len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"]
            and inspect_file(path) == expected, "SOURCE_PROOF_CHANGED_DURING_SEAL")
    try: return data.decode("utf-8")
    except UnicodeDecodeError as error: raise ArchiveError("PROOF_UTF8_JSON_REQUIRED") from error


def _json_bytes(value):
    digest(value)
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode()


def write_once(path, value):
    """fsync a private temporary file, then atomically create the final name."""
    data = _json_bytes(value)
    with parent_fd(path, create=True) as (directory, name):
        temporary = ".pending-" + uuid.uuid4().hex
        try:
            os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ArchiveError("IMMUTABLE_PATH_ALREADY_EXISTS", str(path))
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
        except FileExistsError as error:
            raise ArchiveError("IMMUTABLE_PATH_ALREADY_EXISTS", str(path)) from error
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(data); stream.flush(); os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _renameat_noreplace(directory, temporary, directory, name)
        os.fsync(directory)


def _replace_control(path, value):
    # Mutable receiver control pointer only; immutable admission/proofs retained.
    path = Path(path)
    temporary = path.with_name(".control-" + uuid.uuid4().hex)
    write_once(temporary, value)
    with parent_fd(path) as (directory, name):
        try:
            old = os.stat(name, dir_fd=directory, follow_symlinks=False)
            require(stat.S_ISREG(old.st_mode) and old.st_nlink == 1, "UNSAFE_CONTROL_POINTER")
        except FileNotFoundError:
            pass
        os.replace(temporary.name, name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)


def run_identity(value):
    require(type(value) is dict and set(value) == set(RUN_FIELDS), "INCOMPLETE_RUN_IDENTITY")
    require(value["origin_server"] in ("server1", "server2", "server3", "server4"), "UNKNOWN_ORIGIN_SERVER")
    for key in RUN_FIELDS:
        safe_name(value[key])
    require(bool(re.fullmatch(r"[1-9][0-9]*", value["actual_job_id"])), "ACTUAL_SLURM_JOB_ID_REQUIRED")


def checkpoint_identity(value):
    require(type(value) is dict and set(value) == set(IDENTITY_FIELDS), "INCOMPLETE_CHECKPOINT_IDENTITY")
    for key, item in value.items():
        if key == "official_tree_sha256":
            require(type(item) is str and bool(re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", item)),
                    "CHECKPOINT_IDENTITY_FULL_SHA_REQUIRED", key)
            continue
        length = 40 if key in GIT_IDENTITY_FIELDS else 64
        require(type(item) is str and bool(re.fullmatch(r"[a-f0-9]{"+str(length)+r"}", item)),
                "CHECKPOINT_IDENTITY_FULL_SHA_REQUIRED", key)


def checkpoint_identity_types(value):
    """Archive provenance only; never rewrite the original CP identity/digest."""
    checkpoint_identity(value)
    return {key: ("git-tree-sha1" if len(item) == 40 else "content-sha256")
            if key == "official_tree_sha256" else
            ("git-commit-sha1" if key in GIT_IDENTITY_FIELDS else "content-sha256")
            for key, item in value.items()}


def _unsigned(value, key):
    return {name:item for name,item in value.items() if name != key}


def validate_manifest(value):
    require(type(value) is dict and value.get("schema") == MANIFEST_SCHEMA and value.get("scope") == SCOPE
            and value.get("instruction_id") == INSTRUCTION
            and value.get("manifest_sha256") == digest(_unsigned(value, "manifest_sha256")), "MANIFEST_IDENTITY_OR_PROSPECTIVE_SCOPE")
    run_identity(value["run_identity"]); checkpoint_identity(value["checkpoint_identity"])
    types = value.get("checkpoint_identity_types")
    # Old 40-hex manifests remain verifiable without changing their hashes.
    require(("checkpoint_identity_types" not in value and
             len(value["checkpoint_identity"]["official_tree_sha256"]) == 40) or
            types == checkpoint_identity_types(value["checkpoint_identity"]),
            "CHECKPOINT_IDENTITY_TYPE_BINDING_MISMATCH")
    sha(value["policy_sha256"])
    require(type(value.get("proof_packet")) is dict and set(value["proof_packet"]) == set(ROLES),
            "COMPLETE_ADMISSION_PROOF_PACKET_REQUIRED")
    require(type(value.get("proof_contents")) is dict and set(value["proof_contents"]) == set(ROLES),
            "EXACT_ADMISSION_PROOF_BYTES_REQUIRED")
    metadata_member(value.get("cutover_member"))
    require(type(value.get("cutover")) is dict, "RECORDED_SERVER_CUTOVER_REQUIRED")
    files = value["files"]
    replay = value.get("source_evidence_replay",{})
    original_roles = {record.get("file_role") for record in replay.get("records",[]) if type(record) is dict}
    require(type(files) is dict and set(files) == {"checkpoint", *ROLES, *original_roles}
            and all(type(role) is str and bool(re.fullmatch(r"original_[0-9]{4}",role)) for role in original_roles),
            "EXACT_FINAL_FILE_ALLOWLIST_REQUIRED")
    objects, paths = set(), set()
    for role, row in files.items():
        require(type(row) is dict and set(row) == {"path", "realpath", "sha256", "dev", "inode", "nlink", "bytes", "mtime_ns", "allocated_bytes"}, "FILE_MEMBER_FIELDS")
        sha(row["sha256"])
        require(type(row["path"]) is str and Path(row["path"]).is_absolute() and row["path"] == row["realpath"]
                and ".." not in Path(row["path"]).parts and row["nlink"] == 1
                and all(type(row[key]) is int and row[key] >= 0 for key in ("dev", "inode", "bytes", "mtime_ns", "allocated_bytes")), "UNSAFE_SOURCE_MEMBER")
        require(row["bytes"] > 0 and (role == "checkpoint" or row["bytes"] <= MAX_COMPANION), "EMPTY_OR_LARGE_COMPANION")
        require(row["path"] not in paths and (row["dev"],row["inode"]) not in objects, "SOURCE_ALIAS_HOLD")
        paths.add(row["path"]); objects.add((row["dev"],row["inode"]))
        if role in ROLES:
            text = value["proof_contents"][role]
            require(type(text) is str, "PROOF_UTF8_JSON_REQUIRED")
            contents = text.encode("utf-8")
            require(len(contents) == row["bytes"] and hashlib.sha256(contents).hexdigest() == row["sha256"]
                    and json.loads(text) == value["proof_packet"][role], "ADMISSION_PROOF_FULL_SHA_OR_CONTENT_MISMATCH")
    require(sum(len(item.encode("utf-8")) for item in value["proof_contents"].values()) <= METADATA_BUDGET//2,
            "COMPACT_PROOF_PACKET_BUDGET_EXCEEDED")
    require(type(value["provenance_references"]) is dict and bool(value["provenance_references"]), "RESTORE_PROVENANCE_REQUIRED")
    for key in ("pinned_source", "native_hparams", "base_model", "tokenizer", "input"):
        require(key in value["provenance_references"], "RESTORE_PROVENANCE_MISSING", key)
        reference = value["provenance_references"][key]
        require(type(reference) is dict and bool(reference), "RESTORE_PROVENANCE_MEMBER_REQUIRED", key)
        sha(reference.get("sha256"))
    verify_proofs(value, value["proof_packet"])
    _validate_replays(value)
    return value


def metadata_member(value):
    require(type(value) is dict and set(value) == {"path", "bytes", "sha256"}
            and type(value["path"]) is str and Path(value["path"]).is_absolute()
            and ".." not in Path(value["path"]).parts
            and type(value["bytes"]) is int and 0 < value["bytes"] <= MAX_COMPANION,
            "ORIGINAL_METADATA_MEMBER_REQUIRED")
    sha(value["sha256"])
    return value


def _read_metadata_member(value):
    metadata_member(value)
    row = inspect_file(value["path"])
    require(row["bytes"] == value["bytes"] and row["sha256"] == value["sha256"], "METADATA_MEMBER_FULL_SHA_OR_BYTES_MISMATCH")
    return read_json(value["path"], row)


def _cutover(value, server, policy_sha256):
    require(value.get("schema") == "final-checkpoint-archive-server-cutover-v1"
            and value.get("server") == server and value.get("instruction_id") == INSTRUCTION
            and value.get("policy_sha256") == policy_sha256 and value.get("scope") == SCOPE
            and type(value.get("ack_nonce")) is str and bool(value["ack_nonce"]), "RECORDED_SERVER_CUTOVER_REQUIRED")
    return timestamp(value["received_at_utc"])


def verify_proofs(manifest, values):
    """Normalize source-owned evidence; never infer completion from exit0/name."""
    run, identity, files = manifest["run_identity"], manifest["checkpoint_identity"], manifest["files"]
    candidate = {key:value for key,value in run.items() if key != "actual_job_id"}
    pointer = values["latest_pointer"]
    require(pointer.get("batch") == 20 and type(pointer.get("batch")) is int and pointer.get("final_W20") is True
            and pointer.get("sha256") == files["checkpoint"]["sha256"]
            and pointer.get("identity_sha256") == digest(identity)
            and pointer.get("file") == Path(files["checkpoint"]["path"]).name,
            "LATEST_POINTER_NOT_EXACT_FINAL_W20")
    require(Path(files["latest_pointer"]["path"]).parent == Path(files["checkpoint"]["path"]).parent,
            "POINTER_PAYLOAD_DIRECTORY_MISMATCH")
    adoption, lock, submission = (values[name] for name in ("adoption", "submission_lock", "submission_receipt"))
    cutover_at = _cutover(manifest["cutover"], run["origin_server"], manifest["policy_sha256"])
    require(adoption.get("schema") == "final-checkpoint-archive-adoption-v1" and adoption.get("scope") == SCOPE
            and adoption.get("instruction_id") == INSTRUCTION and adoption.get("policy_sha256") == manifest["policy_sha256"]
            and adoption.get("candidate_identity") == candidate
            and adoption.get("cutover_receipt_sha256") == manifest["cutover_member"]["sha256"], "PRE_SUBMISSION_ADOPTION_REQUIRED")
    require(lock.get("schema") == "final-checkpoint-archive-submission-lock-v1" and lock.get("candidate_identity") == candidate
            and lock.get("checkpoint_identity") == identity and lock.get("adoption_sha256") == files["adoption"]["sha256"]
            and lock.get("policy_sha256") == manifest["policy_sha256"] and lock.get("scope") == SCOPE,
            "IMMUTABLE_NEW_SUBMISSION_LOCK_REQUIRED")
    require(lock.get("checkpoint_directory") == str(Path(files["checkpoint"]["path"]).parent)
            and lock.get("checkpoint_creation_authorized") is True
            and lock.get("artifact_role") == "APPROVED_FINAL_EXPERIMENT_CHECKPOINT",
            "PRESEALED_CHECKPOINT_PATH_AND_AUTHORITY_REQUIRED")
    metadata_member(lock.get("checkpoint_authority_member"))
    require(submission.get("schema") == "final-checkpoint-archive-actual-submission-v1"
            and submission.get("run_identity") == run and submission.get("checkpoint_identity") == identity
            and submission.get("submission_lock_sha256") == files["submission_lock"]["sha256"]
            and submission.get("adoption_sha256") == files["adoption"]["sha256"]
            and submission.get("actual_registered_job_id") == run["actual_job_id"]
            and submission.get("registration_response") == "Submitted batch job " + run["actual_job_id"],
            "ACTUAL_NEW_REGISTRATION_RECEIPT_REQUIRED")
    require(timestamp(adoption["adopted_at_utc"]) <= timestamp(lock["sealed_at_utc"]) < timestamp(submission["scheduler_submitted_at_utc"])
            <= timestamp(submission["receipt_sealed_at_utc"]), "LEGACY_OR_POST_SUBMISSION_ADOPTION_HOLD")
    require(cutover_at <= timestamp(adoption["adopted_at_utc"])
            and cutover_at < timestamp(submission["scheduler_submitted_at_utc"]), "LEGACY_SUBMISSION_PROTECTED")
    terminal, writer, consumers = (values[name] for name in ("scientific_terminal", "writer_termination", "consumer_clearance"))
    for name, proof in (("scientific_terminal",terminal),("writer_termination",writer),("consumer_clearance",consumers)):
        require(proof.get("run_identity") == run and proof.get("checkpoint_identity") == identity
                and proof.get("checkpoint_sha256") == files["checkpoint"]["sha256"]
                and proof.get("latest_pointer_sha256") == files["latest_pointer"]["sha256"], "FINAL_PROOF_IDENTITY_MISMATCH", name)
    require(terminal.get("schema") == "final-checkpoint-scientific-terminal-v1" and terminal.get("actual_complete") is True
            and terminal.get("final_W20") is True and terminal.get("completed_batch") == 20
            and terminal.get("completed_edits") == 2000 and terminal.get("commit_batches") == list(range(1,21))
            and terminal.get("actual_final_calculations_complete") is True, "ACTUAL_FINAL_COMPUTATION_PROOF_REQUIRED")
    require(type(terminal.get("completed_batch")) is int and type(terminal.get("completed_edits")) is int
            and type(terminal.get("commit_batches")) is list
            and all(type(batch) is int for batch in terminal["commit_batches"]), "EXACT_FINAL_COUNTER_TYPES_REQUIRED")
    calculations = terminal.get("required_calculations")
    require(type(calculations) is dict and bool(calculations), "FINAL_CALCULATION_INVENTORY_REQUIRED")
    require(type(lock.get("required_final_calculations")) is list and bool(lock["required_final_calculations"])
            and len(set(lock["required_final_calculations"])) == len(lock["required_final_calculations"])
            and set(calculations) == set(lock["required_final_calculations"]), "ALL_PLANNED_FINAL_CALCULATIONS_REQUIRED")
    for name, proof in calculations.items():
        safe_name(name)
        require(proof.get("state") == "COMPLETE" and proof.get("endpoint") == "W20" and proof.get("actual_measurement") is True
                and proof.get("checkpoint_identity_sha256") == digest(identity), "FINAL_CALCULATION_NOT_COMPLETE")
        metadata_member(proof.get("original_receipt_member"))
    require(timestamp(submission["scheduler_submitted_at_utc"]) < timestamp(terminal["completed_at_utc"]), "FINAL_COMPLETION_CHRONOLOGY")
    require(writer.get("schema") == "final-checkpoint-writer-stopped-v1" and writer.get("state") == "STOPPED_NO_FUTURE_MUTATION"
            and writer.get("actual_writer_job_id") == run["actual_job_id"] and writer.get("writer_lease_closed") is True
            and timestamp(writer["stopped_at_utc"]) >= timestamp(terminal["completed_at_utc"]), "WRITER_STILL_ACTIVE_HOLD")
    metadata_member(writer.get("original_writer_evidence_member"))
    require(consumers.get("schema") == "final-checkpoint-all-consumers-clear-v1"
            and consumers.get("inventory_complete") is True and consumers.get("no_unlisted_dependents") is True
            and consumers.get("state") == "ALL_CONSUMERS_VERIFIED_CLEAR"
            and timestamp(consumers["checked_at_utc"]) >= timestamp(writer["stopped_at_utc"]), "CONSUMER_INVENTORY_OR_ACTIVITY_HOLD")
    entries = consumers.get("consumers")
    require(type(entries) is list and len(entries) >= 1, "ALL_CONSUMER_EVIDENCE_REQUIRED")
    planned = lock.get("planned_consumers")
    require(type(planned) is dict and bool(planned) and planned.get("writer") == "writer"
            and all(kind in ("writer", "evaluation", "collector", "resume", "downstream") for kind in planned.values()),
            "PRESEALED_ALL_CONSUMER_INVENTORY_REQUIRED")
    bindings = submission.get("consumer_bindings")
    require(type(bindings) is dict and set(bindings) == set(planned)
            and all(type(job) is str and bool(re.fullmatch(r"[1-9][0-9]*", job)) for job in bindings.values())
            and bindings["writer"] == run["actual_job_id"], "ACTUAL_CONSUMER_REGISTRATION_BINDINGS_REQUIRED")
    names = set()
    for row in entries:
        identifier = safe_name(row.get("consumer_id"))
        require(identifier not in names and row.get("state") in ("TERMINAL", "DOES_NOT_DEPEND_ON_CHECKPOINT")
                and row.get("not_using_checkpoint") is True and row.get("kind") == planned.get(identifier)
                and row.get("actual_job_id") == bindings.get(identifier),
                "CONSUMER_IN_USE_OR_UNKNOWN_HOLD")
        names.add(identifier)
        metadata_member(row.get("original_evidence_member"))
    require(names == set(planned), "ALL_REGISTERED_CONSUMERS_REQUIRED")
    require(timestamp(consumers["checked_at_utc"]) <= timestamp(utc_now()), "FUTURE_COMPLETION_PROOF_HOLD")
    return dict(final_W20_verified=True, actual_final_calculations_verified=True,
                writer_stopped=True, all_declared_consumers_clear=True, prospective_submission_verified=True,
                scientific_completion_independently_certified=False)


def _evidence_specs(manifest, values):
    base = dict(run_identity=manifest["run_identity"],checkpoint_identity=manifest["checkpoint_identity"],
                checkpoint_sha256=manifest["files"]["checkpoint"]["sha256"],
                latest_pointer_sha256=manifest["files"]["latest_pointer"]["sha256"],endpoint="W20")
    lock = values["submission_lock"]
    yield dict(kind="checkpoint_authority",subject="checkpoint",member=lock["checkpoint_authority_member"],
               expected=dict(base,actual_job_id=manifest["run_identity"]["actual_job_id"],required_state="CHECKPOINT_CREATION_AUTHORIZED"))
    for name,proof in sorted(values["scientific_terminal"]["required_calculations"].items()):
        yield dict(kind="final_calculation",subject=name,member=proof["original_receipt_member"],
                   expected=dict(base,actual_job_id=manifest["run_identity"]["actual_job_id"],required_state="ACTUAL_W20_CALCULATION_COMPLETE"))
    yield dict(kind="writer_termination",subject="writer",member=values["writer_termination"]["original_writer_evidence_member"],
               expected=dict(base,actual_job_id=manifest["run_identity"]["actual_job_id"],required_state="STOPPED_NO_FUTURE_MUTATION"))
    for proof in sorted(values["consumer_clearance"]["consumers"],key=lambda row:row["consumer_id"]):
        yield dict(kind="consumer_clearance",subject=proof["consumer_id"],member=proof["original_evidence_member"],
                   expected=dict(base,actual_job_id=proof["actual_job_id"],required_state=proof["state"],
                                 checked_at_utc=values["consumer_clearance"]["checked_at_utc"]))


def _adapter_contract(adapter, contract):
    require(callable(adapter) and type(contract) is dict and set(contract)=={"module","function","source_member"},
            "ARCHIVE_PENDING_KEEP_SOURCE","recognized source evidence adapter required")
    metadata_member(contract["source_member"])
    require(getattr(adapter,"__module__",None)==contract["module"]
            and getattr(adapter,"__name__",None)==contract["function"]
            and hasattr(adapter,"__code__")
            and str(Path(adapter.__code__.co_filename).absolute()) == contract["source_member"]["path"],
            "EVIDENCE_ADAPTER_SOURCE_OR_MODULE_MISMATCH")
    source = inspect_file(contract["source_member"]["path"])
    require(source["sha256"]==contract["source_member"]["sha256"] and source["bytes"]==contract["source_member"]["bytes"],
            "EVIDENCE_ADAPTER_SOURCE_OR_MODULE_MISMATCH")


def _replay_result(spec, document, contract):
    return dict(schema="final-checkpoint-original-evidence-replay-v1",kind=spec["kind"],subject=spec["subject"],
                original_member=spec["member"],original_content_sha256=digest(document),
                adapter_source_sha256=contract["source_member"]["sha256"],expected=spec["expected"],
                status="SOURCE_OWNED_ORIGINAL_EVIDENCE_REPLAY_PASS",
                checks=dict(original_schema_recognized=True,identity_matched=True,endpoint_matched=True,
                            actual_status_matched=True,planned_scope_matched=True))


def _replay_sources(manifest, values, adapter):
    contract = values["submission_lock"].get("evidence_adapter")
    require(manifest["provenance_references"].get("evidence_adapter") == contract,
            "PRESEALED_EVIDENCE_ADAPTER_PROVENANCE_REQUIRED")
    _adapter_contract(adapter,contract)
    records, documents = [], {}
    for spec in _evidence_specs(manifest,values):
        document = _read_metadata_member(spec["member"])
        try: result = adapter(spec["kind"],spec["subject"],document,deepcopy_json(spec["expected"]),dict(spec["member"]),deepcopy_json(contract))
        except Exception as error:
            raise ArchiveError("ARCHIVE_PENDING_KEEP_SOURCE","original evidence schema/replay rejected") from error
        require(type(result) is dict and result == _replay_result(spec,document,contract),
                "ARCHIVE_PENDING_KEEP_SOURCE","strict original evidence replay result required, not bool")
        records.append(dict(kind=spec["kind"],subject=spec["subject"],result=result))
        documents[spec["member"]["path"]] = document
    require(inspect_file(contract["source_member"]["path"])["sha256"] == contract["source_member"]["sha256"],
            "EVIDENCE_ADAPTER_CHANGED_DURING_REPLAY")
    return contract,records,documents


def deepcopy_json(value):
    return json.loads(json.dumps(value,allow_nan=False))


def _validate_replays(manifest, original_documents=None):
    values = manifest["proof_packet"]
    contract = values["submission_lock"].get("evidence_adapter")
    require(type(contract) is dict and set(contract)=={"module","function","source_member"}
            and manifest["provenance_references"].get("evidence_adapter")==contract,
            "PRESEALED_EVIDENCE_ADAPTER_PROVENANCE_REQUIRED")
    metadata_member(contract["source_member"])
    replay = manifest.get("source_evidence_replay",{})
    require(replay.get("schema")=="final-checkpoint-source-evidence-replay-v1"
            and replay.get("adapter")==contract
            and replay.get("replay_sha256")==digest(_unsigned(replay,"replay_sha256")), "SOURCE_EVIDENCE_REPLAY_REQUIRED")
    specs = list(_evidence_specs(manifest,values))
    require(type(replay.get("records")) is list and len(replay["records"])==len(specs), "ALL_ORIGINAL_EVIDENCE_REPLAY_REQUIRED")
    for spec,record in zip(specs,replay["records"]):
        require(record.get("kind")==spec["kind"] and record.get("subject")==spec["subject"], "ORIGINAL_EVIDENCE_SCOPE_MISMATCH")
        row = manifest["files"].get(record.get("file_role"),{})
        require(all(row.get(key)==spec["member"][key] for key in ("path","bytes","sha256")), "ORIGINAL_EVIDENCE_MEMBER_BINDING_MISMATCH")
        result = record.get("result",{})
        require(type(result) is dict, "STRICT_ORIGINAL_EVIDENCE_REPLAY_REQUIRED")
        sha(result.get("original_content_sha256"))
        # The receiver hashes/reads actual small originals independently after
        # transfer; admission uses the source-owned frozen adapter replay.
        document = (original_documents or {}).get(record["file_role"])
        reference_result = dict(schema="final-checkpoint-original-evidence-replay-v1",kind=spec["kind"],subject=spec["subject"],
            original_member=spec["member"],original_content_sha256=result["original_content_sha256"],
            adapter_source_sha256=contract["source_member"]["sha256"],expected=spec["expected"],
            status="SOURCE_OWNED_ORIGINAL_EVIDENCE_REPLAY_PASS",
            checks=dict(original_schema_recognized=True,identity_matched=True,endpoint_matched=True,actual_status_matched=True,planned_scope_matched=True))
        require(result==reference_result and all(type(item) is bool for item in result["checks"].values()), "STRICT_ORIGINAL_EVIDENCE_REPLAY_REQUIRED")
        if original_documents is not None:
            require(document is not None and digest(document)==result["original_content_sha256"], "RECEIVED_ORIGINAL_EVIDENCE_CONTENT_MISMATCH")


def seal_manifest(*, run, identity, checkpoint, companions, provenance_references, policy_sha256, cutover_member, evidence_adapter=None):
    run_identity(run); checkpoint_identity(identity); sha(policy_sha256)
    require(type(companions) is dict and set(companions) == set(ROLES), "EXACT_SMALL_PROOF_ALLOWLIST_REQUIRED")
    files = {"checkpoint":inspect_file(checkpoint)}
    files.update({role:inspect_file(path) for role,path in companions.items()})
    values = {role:read_json(row["path"],row) for role,row in files.items() if role != "checkpoint"}
    proof_contents = {role:_proof_text(row["path"],row) for role,row in files.items() if role != "checkpoint"}
    cutover_value = _read_metadata_member(cutover_member)
    result = dict(schema=MANIFEST_SCHEMA, instruction_id=INSTRUCTION, scope=SCOPE,
                  run_identity=run, checkpoint_identity=identity, files=files,
                  checkpoint_identity_types=checkpoint_identity_types(identity),
                  provenance_references=provenance_references, policy_sha256=policy_sha256,
                  proof_packet=values, proof_contents=proof_contents,
                  cutover_member=cutover_member, cutover=cutover_value, sealed_at_utc=utc_now())
    verify_proofs(result,values)
    contract,records,documents = _replay_sources(result,values,evidence_adapter)
    original_roles = {}
    for index,path in enumerate(sorted(documents)):
        role = "original_"+str(index).zfill(4)
        files[role] = inspect_file(path)
        original_roles[path] = role
    for record in records:
        record["file_role"] = original_roles[record["result"]["original_member"]["path"]]
    replay = dict(schema="final-checkpoint-source-evidence-replay-v1",adapter=contract,records=records)
    replay["replay_sha256"] = digest(replay)
    result["source_evidence_replay"] = replay
    result["manifest_sha256"] = digest(result)
    validate_manifest(result)
    return result


def _layout(manifest):
    run = manifest["run_identity"]
    return Path(run["origin_server"],run["task_id"],run["run_id"],run["attempt"],
                "job-" + run["actual_job_id"],manifest["files"]["checkpoint"]["sha256"])


def _renameat_noreplace(src_dir, src_name, dst_dir, dst_name):
    library = ctypes.CDLL(None, use_errno=True)
    require(hasattr(library,"renameat2"), "ATOMIC_NOREPLACE_UNAVAILABLE")
    result = library.renameat2(src_dir,src_name.encode(),dst_dir,dst_name.encode(),1)
    if result:
        code = ctypes.get_errno()
        raise ArchiveError("DESTINATION_COLLISION" if code == errno.EEXIST else "ATOMIC_SEAL_FAILED",os.strerror(code))


def _rename_noreplace(source, target, expected_inode=None):
    """Linux atomic directory seal; unlike rename(), even empty targets reject."""
    with parent_fd(source) as (src_dir,src_name), parent_fd(target,create=True) as (dst_dir,dst_name):
        current = os.stat(src_name,dir_fd=src_dir,follow_symlinks=False)
        require(stat.S_ISDIR(current.st_mode) and (expected_inode is None or (current.st_dev,current.st_ino) == expected_inode),
                "STAGING_DIRECTORY_CHANGED_HOLD")
        _renameat_noreplace(src_dir,src_name,dst_dir,dst_name)
        os.fsync(src_dir); os.fsync(dst_dir)


def _directory_names(path, expected_inode):
    with parent_fd(path) as (directory,name):
        fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=directory)
        try:
            actual=os.fstat(fd)
            require((actual.st_dev,actual.st_ino)==expected_inode,"STAGING_DIRECTORY_CHANGED_HOLD")
            return set(os.listdir(fd))
        finally:os.close(fd)


class Receiver:
    def __init__(self, policy, *, policy_sha256, cutover_members, root=None, test_only=False, _policy_member=None):
        require(policy.get("schema") == "final-checkpoint-central-archive-v1" and policy.get("instruction_id") == INSTRUCTION,
                "CANONICAL_POLICY_REQUIRED")
        require(policy.get("scope",{}).get("prospective_only") is True, "CANONICAL_PROSPECTIVE_POLICY_REQUIRED")
        sha(policy_sha256)
        require(test_only or (_policy_member is not None and _policy_member["sha256"] == policy_sha256
                and read_json(_policy_member["path"],_policy_member) == policy), "EXACT_POLICY_FILE_BINDING_REQUIRED")
        if _policy_member is not None: self.policy_member = _policy_member
        require(type(cutover_members) is dict and bool(cutover_members), "TRUSTED_SERVER_CUTOVER_MEMBERS_REQUIRED")
        self.cutovers = {server:dict(member=member,value=_read_metadata_member(member)) for server,member in cutover_members.items()}
        for server, proof in self.cutovers.items():
            _cutover(proof["value"],server,policy_sha256)
        configured = Path(policy["destination"]["root"])
        require(root is None or test_only, "PRODUCTION_ROOT_OVERRIDE_FORBIDDEN")
        self.root, self.policy_sha256 = Path(root) if root is not None else configured, policy_sha256
        require(self.root.is_absolute() and self.root not in (Path("/"),Path.home()), "DEDICATED_ARCHIVE_ROOT_REQUIRED")

    @classmethod
    def from_policy_file(cls, path, *, cutover_members, root=None, test_only=False):
        member = inspect_file(path)
        policy = read_json(path,member)
        receiver = cls(policy,policy_sha256=member["sha256"],cutover_members=cutover_members,root=root,test_only=test_only,
                       _policy_member=member)
        return receiver

    def _trusted_manifest(self, manifest):
        validate_manifest(manifest)
        require(manifest["policy_sha256"] == self.policy_sha256,"POLICY_FULL_SHA_MISMATCH")
        trusted = self.cutovers.get(manifest["run_identity"]["origin_server"])
        require(trusted is not None and trusted["member"]["sha256"] == manifest["cutover_member"]["sha256"]
                and trusted["member"]["bytes"] == manifest["cutover_member"]["bytes"]
                and trusted["value"] == manifest["cutover"], "UNTRUSTED_OR_CHANGED_SERVER_CUTOVER")
        require(_read_metadata_member(trusted["member"]) == trusted["value"], "TRUSTED_SERVER_CUTOVER_CHANGED")
        if hasattr(self,"policy_member"):
            require(inspect_file(self.policy_member["path"]) == self.policy_member,"CANONICAL_POLICY_CHANGED")

    @contextmanager
    def _locked(self):
        lock = self.root/".receiver"/"lock"
        with parent_fd(lock,create=True) as (directory,name):
            descriptor = os.open(name,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=directory)
            try:
                meta = os.fstat(descriptor)
                require(stat.S_ISREG(meta.st_mode) and meta.st_nlink == 1,"UNSAFE_RECEIVER_LOCK")
                try:
                    fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ArchiveError("ARCHIVE_PENDING_KEEP_SOURCE","receiver concurrency1") from error
                yield
            finally:
                os.close(descriptor)

    def _capacity(self, needed_bytes, needed_inodes):
        usage = os.statvfs(self.root)
        available = usage.f_bavail * usage.f_frsize
        require(available >= needed_bytes + RESERVE_BYTES and usage.f_favail >= needed_inodes+256,
                "ARCHIVE_PENDING_KEEP_SOURCE","disk/inode reserve")
        return dict(available_bytes=available,available_inodes=usage.f_favail,
                    minimum_operational_reserve_bytes=RESERVE_BYTES,requested_reservation_bytes=needed_bytes,
                    outstanding_reserved_bytes=0,receiver_max_concurrent=1)

    def _outstanding(self):
        """Receiver-control metadata only, including crash-orphan reservations."""
        folder = self.root/".receiver"/"reservations"
        with parent_fd(folder,create=True) as (directory,name):
            try: os.mkdir(name,0o700,dir_fd=directory)
            except FileExistsError: pass
            fd = os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=directory)
            try: names = os.listdir(fd)
            finally: os.close(fd)
        result = []
        for name in names:
            require(bool(re.fullmatch(r"[a-f0-9]{32}\.json",name)), "UNKNOWN_RECEIVER_RESERVATION_METADATA_HOLD")
            admission = read_json(folder/name)
            require(admission.get("schema") == "final-checkpoint-destination-admission-v1"
                    and admission.get("admission_sha256") == digest(_unsigned(admission,"admission_sha256"))
                    and name == admission.get("reservation_id","")+".json", "RESERVATION_METADATA_CHANGED_HOLD")
            completed = self.root/".receiver"/"completed"/name
            try: proof = read_json(completed)
            except FileNotFoundError:
                result.append(admission);continue
            require(proof.get("schema") == "final-checkpoint-reservation-completed-v1"
                    and proof.get("admission_sha256") == admission["admission_sha256"]
                    and proof.get("destination") == admission["destination"], "RESERVATION_COMPLETION_CHANGED_HOLD")
            receipt = read_json(Path(proof["destination"])/"VERIFIED_DESTINATION.json")
            require(receipt.get("receipt_sha256") == proof.get("receipt_sha256")
                    and receipt["receipt_sha256"] == digest(_unsigned(receipt,"receipt_sha256"))
                    and receipt.get("admission_sha256") == admission["admission_sha256"], "RESERVATION_COMPLETION_RECEIPT_CHANGED_HOLD")
        return result

    def admit(self, manifest):
        self._trusted_manifest(manifest)
        with self._locked():
            require(not self._outstanding(),"ARCHIVE_PENDING_KEEP_SOURCE","outstanding immutable reservation concurrency1")
            active_path = self.root/".receiver"/"active.json"
            if active_path.exists():
                active = read_json(active_path)
                require(active.get("reservation_id") is None,"ARCHIVE_PENDING_KEEP_SOURCE","outstanding reservation concurrency1")
            total = sum(row["bytes"] for row in manifest["files"].values())
            capacity = self._capacity(2*total+METADATA_BUDGET,len(manifest["files"])*2+32)
            destination = self.root/_layout(manifest)
            with parent_fd(destination,create=True) as (directory,name):
                try:
                    os.stat(name,dir_fd=directory,follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise ArchiveError("DESTINATION_COLLISION",str(destination))
            identifier = uuid.uuid4().hex
            staging = self.root/".staging"/identifier
            with parent_fd(staging,create=True) as (directory,name):
                os.mkdir(name,0o700,dir_fd=directory);os.fsync(directory)
            value = dict(schema="final-checkpoint-destination-admission-v1",stage="DESTINATION_ADMITTED",
                reservation_id=identifier,manifest_sha256=manifest["manifest_sha256"],policy_sha256=self.policy_sha256,
                scope=SCOPE,instruction_id=INSTRUCTION,run_identity=manifest["run_identity"],
                staging=str(staging),destination=str(destination),admitted_at_utc=utc_now(),capacity=capacity,
                incoming_names={role:role+".pt" if role=="checkpoint" else role+".json" for role in manifest["files"]},
                source_deleted=False,actual_transferred=False)
            value["admission_sha256"] = digest(value)
            write_once(self.root/".receiver"/"reservations"/(identifier+".json"),value)
            _replace_control(active_path,dict(reservation_id=identifier,manifest_sha256=manifest["manifest_sha256"],
                                           reserved_bytes=capacity["requested_reservation_bytes"]))
            return value

    def verify(self, admission, manifest):
        self._trusted_manifest(manifest)
        require(admission.get("admission_sha256") == digest(_unsigned(admission,"admission_sha256"))
                and admission.get("manifest_sha256") == manifest["manifest_sha256"]
                and admission.get("policy_sha256") == manifest["policy_sha256"] == self.policy_sha256,
                "ADMISSION_MANIFEST_POLICY_MISMATCH")
        identifier = safe_name(admission["reservation_id"])
        with self._locked():
            stored = read_json(self.root/".receiver"/"reservations"/(identifier+".json"))
            require(stored == admission and read_json(self.root/".receiver"/"active.json").get("reservation_id") == identifier,
                    "EXACT_ACTIVE_RESERVATION_REQUIRED")
            staging, destination = self.root/".staging"/identifier, self.root/_layout(manifest)
            require(admission["staging"] == str(staging) and admission["destination"] == str(destination),"UNIQUE_LAYOUT_MISMATCH")
            expected = admission["incoming_names"]
            with parent_fd(staging) as (stage_parent,stage_name):
                stage_fd = os.open(stage_name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=stage_parent)
                try:
                    stage_stat = os.fstat(stage_fd)
                    stage_inode = (stage_stat.st_dev,stage_stat.st_ino)
                    require(set(expected) == set(manifest["files"]) and set(os.listdir(stage_fd)) == set(expected.values()),"RECEIVED_ALLOWLIST_MISMATCH")
                finally:
                    os.close(stage_fd)
            received, values, originals = {}, {}, {}
            for role, name in expected.items():
                require(Path(name).name == name and name == (role+".pt" if role=="checkpoint" else role+".json"),"RECEIVED_UNSAFE_MEMBER_NAME")
                original = manifest["files"][role]
                actual = inspect_file(staging/name)
                require(actual["sha256"] == original["sha256"] and actual["bytes"] == original["bytes"],"RECEIVED_FULL_SHA_OR_BYTES_MISMATCH",role)
                require(actual["path"] != original["path"],"SAME_OBJECT_HOLD")
                # Source path may also be visible on shared storage. Never infer
                # cross-host object equality solely from numeric dev/inode.
                try:
                    with parent_fd(original["path"]) as (directory, basename):
                        local_source = os.stat(basename,dir_fd=directory,follow_symlinks=False)
                        require(not stat.S_ISLNK(local_source.st_mode) and
                                (local_source.st_dev,local_source.st_ino) != (actual["dev"],actual["inode"]),"SAME_OBJECT_HOLD")
                except FileNotFoundError:
                    require(manifest["run_identity"]["origin_server"] != "server1","SOURCE1_PATH_MISSING_HOLD")
                received[role] = actual
                if role in ROLES:
                    values[role] = read_json(actual["path"],actual)
                elif role != "checkpoint":
                    originals[role] = read_json(actual["path"],actual)
                with parent_fd(actual["path"]) as (directory,basename):
                    descriptor = os.open(basename,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=directory)
                    try: os.fsync(descriptor)
                    finally: os.close(descriptor)
            checks = verify_proofs(manifest,values)
            require(values == manifest["proof_packet"], "RECEIVED_PROOF_PACKET_CHANGED")
            _validate_replays(manifest,originals)
            self._capacity(METADATA_BUDGET,32)
            # Recheck received paths immediately before the atomic directory seal.
            for role, row in received.items():
                require(inspect_file(row["path"]) == row,"RECEIVED_MEMBER_CHANGED_BEFORE_SEAL",role)
            require(_directory_names(staging,stage_inode)==set(expected.values()),"RECEIVED_ALLOWLIST_CHANGED_BEFORE_SEAL")
            final_members = {role:dict(row,path=str(destination/expected[role]),realpath=str(destination/expected[role]))
                             for role,row in received.items()}
            receipt = dict(schema=RECEIPT_SCHEMA,stage="VERIFIED_DESTINATION",instruction_id=INSTRUCTION,scope=SCOPE,
                run_identity=manifest["run_identity"],checkpoint_identity=manifest["checkpoint_identity"],
                manifest_sha256=manifest["manifest_sha256"],admission_sha256=admission["admission_sha256"],
                sealed_source_manifest=manifest,
                source_members=manifest["files"],destination=str(destination),members=final_members,
                validation=checks,verified_at_utc=utc_now(),source_deleted=False,
                independent_receiver_full_sha_verified=True,distinct_preserved_copy=True)
            receipt["receipt_sha256"] = digest(receipt)
            require(len(_json_bytes(receipt)) <= METADATA_BUDGET,"VERIFIED_RECEIPT_METADATA_BUDGET_EXCEEDED")
            write_once(staging/"VERIFIED_DESTINATION.json",receipt)
            with parent_fd(staging/"VERIFIED_DESTINATION.json") as (directory,_):os.fsync(directory)
            require(_directory_names(staging,stage_inode)==set(expected.values())|{"VERIFIED_DESTINATION.json"},
                    "RECEIVED_ALLOWLIST_CHANGED_BEFORE_SEAL")
            _rename_noreplace(staging,destination,stage_inode)
            write_once(self.root/".receiver"/"completed"/(identifier+".json"),dict(
                schema="final-checkpoint-reservation-completed-v1",admission_sha256=admission["admission_sha256"],
                destination=str(destination),receipt_sha256=receipt["receipt_sha256"]))
            _replace_control(self.root/".receiver"/"active.json",dict(reservation_id=None,last_verified_receipt_sha256=receipt["receipt_sha256"]))
            return receipt

    def recheck(self, receipt):
        """Independent current full-SHA recheck; no reliance on a stale PASS."""
        require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("stage") == "VERIFIED_DESTINATION"
                and receipt.get("receipt_sha256") == digest(_unsigned(receipt,"receipt_sha256"))
                and receipt.get("scope") == SCOPE and receipt.get("instruction_id") == INSTRUCTION,
                "VERIFIED_RECEIPT_IDENTITY_REQUIRED")
        run_identity(receipt["run_identity"])
        sealed_manifest=validate_manifest(receipt.get("sealed_source_manifest"))
        require(sealed_manifest["manifest_sha256"]==receipt["manifest_sha256"]
                and sealed_manifest["run_identity"]==receipt["run_identity"]
                and sealed_manifest["checkpoint_identity"]==receipt["checkpoint_identity"]
                and sealed_manifest["files"]==receipt["source_members"], "IMMUTABLE_RESTORE_PROVENANCE_REQUIRED")
        destination = Path(receipt["destination"])
        expected_destination = self.root/Path(receipt["run_identity"]["origin_server"],receipt["run_identity"]["task_id"],
                receipt["run_identity"]["run_id"],receipt["run_identity"]["attempt"],"job-"+receipt["run_identity"]["actual_job_id"],
                receipt["source_members"]["checkpoint"]["sha256"])
        require(destination == expected_destination, "RECEIPT_LAYOUT_MISMATCH")
        with self._locked():
            require(read_json(destination/"VERIFIED_DESTINATION.json") == receipt, "IMMUTABLE_DESTINATION_RECEIPT_CHANGED")
            require(set(receipt["members"]) == set(receipt["source_members"]), "EXACT_RECEIPT_MEMBER_ALLOWLIST_REQUIRED")
            with parent_fd(destination) as (directory,name):
                fd = os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=directory)
                try:
                    require(set(os.listdir(fd)) == {Path(row["path"]).name for row in receipt["members"].values()}|{"VERIFIED_DESTINATION.json"},
                            "SEALED_DESTINATION_ALLOWLIST_CHANGED")
                finally: os.close(fd)
            for role,row in receipt["members"].items():
                require(Path(row["path"]).parent == destination and inspect_file(row["path"]) == row,
                        "ARCHIVE_MEMBER_CHANGED", role)
            _validate_replays(sealed_manifest,{role:read_json(row["path"],row) for role,row in receipt["members"].items()
                                               if role.startswith("original_")})
            return True


def source_delete_gate(receipt, manifest, *, fresh_companions, destination_verify, evidence_adapter=None):
    """Metadata-only ELIGIBLE gate; source owner must do its own safe unlink.

    destination_verify must be an independent receiver proof recheck/transport
    identity implementation, not an rsync exit0 or a scheduler return value.
    This function deliberately has no deletion or transfer implementation.
    """
    validate_manifest(manifest)
    require(manifest["run_identity"]["origin_server"] != "server1",
            "SERVER1_SOURCE_REGISTER_AND_PROTECT_KEEP_NO_UNLINK")
    require(receipt.get("schema") == RECEIPT_SCHEMA and receipt.get("stage") == "VERIFIED_DESTINATION"
            and receipt.get("receipt_sha256") == digest(_unsigned(receipt,"receipt_sha256"))
            and receipt.get("manifest_sha256") == manifest["manifest_sha256"]
            and receipt.get("run_identity") == manifest["run_identity"]
            and receipt.get("source_members") == manifest["files"]
            and receipt.get("independent_receiver_full_sha_verified") is True
            and receipt.get("distinct_preserved_copy") is True,"INDEPENDENT_VERIFIED_DESTINATION_REQUIRED")
    require(callable(destination_verify) and destination_verify(receipt) is True,"DESTINATION_RECHECK_REQUIRED")
    require(set(fresh_companions) == set(ROLES),"FRESH_ALL_CONSUMER_WRITER_PROOFS_REQUIRED")
    # New source-owned clearance can strengthen chronology without rewriting old
    # archived companions. All other proof inputs must remain byte-identical.
    values = {role:read_json(path) for role,path in fresh_companions.items()}
    for role in ROLES:
        if role not in ("consumer_clearance","writer_termination"):
            require(inspect_file(fresh_companions[role]) == manifest["files"][role],"SOURCE_PROOF_CHANGED_HOLD",role)
    verify_proofs(manifest,values)
    _replay_sources(manifest,values,evidence_adapter)
    require(timestamp(values["consumer_clearance"]["checked_at_utc"]) >= timestamp(receipt["verified_at_utc"]),"FRESH_POST_RECEIPT_CLEARANCE_REQUIRED")
    actual = inspect_file(manifest["files"]["checkpoint"]["path"])
    require(actual == manifest["files"]["checkpoint"],"HOLD_SOURCE_CHANGED_OR_IN_USE")
    for row in receipt["members"].values():
        require(row["path"] != actual["path"],"SAME_OBJECT_HOLD")
    return dict(stage="ELIGIBLE_SOURCE_OWNER_FINAL_RECHECK_REQUIRED",source_removed=False,
                eligible_payload_paths=[actual["path"]],receipt_sha256=receipt["receipt_sha256"],
                source_bytes=actual["bytes"],source_allocated_bytes=actual["allocated_bytes"],
                remaining_gate="SOURCE_OWNER_DIRFD_HASH_STAT_CONSUMER_RECHECK_THEN_EXACT_UNLINK_ONLY")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("admit","verify"))
    parser.add_argument("--policy",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--cutovers",type=Path,required=True,help="Trusted server -> immutable cutover member map")
    parser.add_argument("--admission",type=Path)
    parser.add_argument("--receipt-output",type=Path,required=True)
    args=parser.parse_args()
    receiver=Receiver.from_policy_file(args.policy,cutover_members=read_json(args.cutovers))
    manifest=read_json(args.manifest)
    if args.command=="admit":value=receiver.admit(manifest)
    else:
        require(args.admission is not None,"ADMISSION_REQUIRED")
        value=receiver.verify(read_json(args.admission),manifest)
    write_once(args.receipt_output,value)
    print(json.dumps(dict(stage=value["stage"],source_deleted=False,
                         receipt_output=str(args.receipt_output)),sort_keys=True))


if __name__=="__main__":main()
