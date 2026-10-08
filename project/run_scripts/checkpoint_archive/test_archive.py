"""CPU temporary-file tests only; these fixtures are not scientific receipts."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest.mock import patch

from . import archive as a


def member(path):
    row = a.inspect_file(path)
    return {key:row[key] for key in ("path","bytes","sha256")}


def fixture_evidence_adapter(kind, subject, document, expected, original_member, contract):
    """Explicit CPU fixture adapter; never an actual scientific proof producer."""
    a.require(document.get("schema")=="CPU_FIXTURE_ORIGINAL_EVIDENCE_NOT_SCIENCE", "FIXTURE_SCHEMA_UNKNOWN")
    for key in ("run_identity","checkpoint_identity","checkpoint_sha256","latest_pointer_sha256","endpoint"):
        a.require(document.get(key)==expected[key],"FIXTURE_ORIGINAL_IDENTITY_MISMATCH")
    if kind=="checkpoint_authority":
        a.require(document.get("checkpoint_creation_authorized") is True,"FIXTURE_NO_AUTHORITY")
    elif kind=="final_calculation":
        a.require(document.get("actual_final_calculations",{}).get(subject) is True,"FIXTURE_CALCULATION_NOT_COMPLETE")
    elif kind=="writer_termination":
        a.require(document.get("writer_state")==expected["required_state"],"FIXTURE_WRITER_NOT_STOPPED")
    elif kind=="consumer_clearance":
        a.require(document.get("consumer_jobs",{}).get(subject)==expected["actual_job_id"]
                  and document.get("consumer_states",{}).get(subject)==expected["required_state"]
                  and a.timestamp(document["observed_at_utc"])>=a.timestamp(expected["checked_at_utc"]),"FIXTURE_CONSUMER_ACTIVE_OR_STALE")
    else: raise a.ArchiveError("FIXTURE_KIND_UNKNOWN")
    a.require(expected["actual_job_id"] == (document["consumer_jobs"][subject] if kind=="consumer_clearance"
              else document["run_identity"]["actual_job_id"]),"FIXTURE_ACTUAL_JOB_MISMATCH")
    return a._replay_result(dict(kind=kind,subject=subject,member=original_member,expected=expected),document,contract)


def fixture_boolean_adapter(*args):
    return True


class Fixture:
    def __init__(self, root, origin="server1", original_transform=None, adapter=fixture_evidence_adapter, content_tree=False):
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.source = self.root/"source"
        self.source.mkdir()
        self.policy_path = self.root/"policy.json"
        a.write_once(self.policy_path,dict(schema="final-checkpoint-central-archive-v1",instruction_id=a.INSTRUCTION,
            scope=dict(prospective_only=True),destination=dict(root="/unused/production/archive")))
        self.policy_sha = member(self.policy_path)["sha256"]
        self.cutover = self.root/"cutover.json"
        a.write_once(self.cutover,dict(schema="final-checkpoint-archive-server-cutover-v1",server=origin,
            instruction_id=a.INSTRUCTION,scope=a.SCOPE,policy_sha256=self.policy_sha,
            ack_nonce="CPU_FIXTURE_NOT_REAL_OWNER_ACK",received_at_utc="2026-10-09T00:00:00+00:00"))
        self.cutover_member = member(self.cutover)
        self.run = dict(origin_server=origin,task_id="cpu-fixture",run_id="fixture-run",attempt="fixture-attempt",
                        actual_job_id="12345",registration_attempt_id="fixture-registration")
        self.identity = {key:(hashlib.sha1 if key in a.GIT_IDENTITY_FIELDS else hashlib.sha256)
                         (("CPU_FIXTURE_"+key).encode()).hexdigest() for key in a.IDENTITY_FIELDS}
        if content_tree:
            self.identity["official_tree_sha256"] = hashlib.sha256(b"CPU_FIXTURE_SORTED_FILES").hexdigest()
        self.payload = self.source/"final-W20.pt"
        self.payload.write_bytes(b"CPU fixture bytes only: not a model or checkpoint tensor.\x00"*7)
        payload_sha = member(self.payload)["sha256"]
        self.paths = {role:self.source/("latest.json" if role=="latest_pointer" else role+".json") for role in a.ROLES}
        a.write_once(self.paths["latest_pointer"],dict(batch=20,final_W20=True,file=self.payload.name,
                    sha256=payload_sha,identity_sha256=a.digest(self.identity)))
        self.evidence = self.root/"original-evidence.json"
        self.original = deepcopy(dict(schema="CPU_FIXTURE_ORIGINAL_EVIDENCE_NOT_SCIENCE",actual_gpu_measurements=0,
            run_identity=self.run,checkpoint_identity=self.identity,checkpoint_sha256=payload_sha,
            latest_pointer_sha256=member(self.paths["latest_pointer"])["sha256"],endpoint="W20",
            checkpoint_creation_authorized=True,actual_final_calculations=dict(factual=True,generation=True),
            writer_state="STOPPED_NO_FUTURE_MUTATION",consumer_jobs=dict(writer="12345",evaluation="12345",collector="12346",resume="12347"),
            consumer_states={key:"TERMINAL" for key in ("writer","evaluation","collector","resume")},
            observed_at_utc="2026-10-09T01:02:00+00:00"))
        if original_transform:original_transform(self.original)
        a.write_once(self.evidence,self.original)
        ref = member(self.evidence)
        self.adapter_impl=adapter
        self.adapter_contract=dict(module=__name__,function=adapter.__name__,source_member=member(Path(__file__).absolute()))
        candidate = {key:value for key,value in self.run.items() if key!="actual_job_id"}
        a.write_once(self.paths["adoption"],dict(schema="final-checkpoint-archive-adoption-v1",scope=a.SCOPE,
            instruction_id=a.INSTRUCTION,policy_sha256=self.policy_sha,candidate_identity=candidate,
            cutover_receipt_sha256=self.cutover_member["sha256"],adopted_at_utc="2026-10-09T00:01:00+00:00"))
        a.write_once(self.paths["submission_lock"],dict(schema="final-checkpoint-archive-submission-lock-v1",scope=a.SCOPE,
            policy_sha256=self.policy_sha,candidate_identity=candidate,checkpoint_identity=self.identity,
            adoption_sha256=member(self.paths["adoption"])["sha256"],sealed_at_utc="2026-10-09T00:02:00+00:00",
            checkpoint_directory=str(self.source),artifact_role="APPROVED_FINAL_EXPERIMENT_CHECKPOINT",
            checkpoint_creation_authorized=True,checkpoint_authority_member=ref,
            evidence_adapter=self.adapter_contract,
            required_final_calculations=["factual","generation"],
            planned_consumers=dict(writer="writer",evaluation="evaluation",collector="collector",resume="resume")))
        a.write_once(self.paths["submission_receipt"],dict(schema="final-checkpoint-archive-actual-submission-v1",
            run_identity=self.run,checkpoint_identity=self.identity,actual_registered_job_id="12345",
            registration_response="Submitted batch job 12345",adoption_sha256=member(self.paths["adoption"])["sha256"],
            submission_lock_sha256=member(self.paths["submission_lock"])["sha256"],
            scheduler_submitted_at_utc="2026-10-09T00:03:00+00:00",receipt_sealed_at_utc="2026-10-09T00:04:00+00:00",
            consumer_bindings=dict(writer="12345",evaluation="12345",collector="12346",resume="12347")))
        common = dict(run_identity=self.run,checkpoint_identity=self.identity,checkpoint_sha256=payload_sha,
                      latest_pointer_sha256=member(self.paths["latest_pointer"])["sha256"])
        calculations = {name:dict(state="COMPLETE",endpoint="W20",actual_measurement=True,
            checkpoint_identity_sha256=a.digest(self.identity),original_receipt_member=ref) for name in ("factual","generation")}
        a.write_once(self.paths["scientific_terminal"],dict(**common,schema="final-checkpoint-scientific-terminal-v1",
            actual_complete=True,final_W20=True,completed_batch=20,completed_edits=2000,commit_batches=list(range(1,21)),
            actual_final_calculations_complete=True,required_calculations=calculations,completed_at_utc="2026-10-09T01:00:00+00:00"))
        a.write_once(self.paths["writer_termination"],dict(**common,schema="final-checkpoint-writer-stopped-v1",
            state="STOPPED_NO_FUTURE_MUTATION",actual_writer_job_id="12345",writer_lease_closed=True,
            original_writer_evidence_member=ref,
            stopped_at_utc="2026-10-09T01:01:00+00:00"))
        entries = [dict(consumer_id=name,kind=name,state="TERMINAL",actual_job_id=job,
                        not_using_checkpoint=True,original_evidence_member=ref)
                   for name,job in dict(writer="12345",evaluation="12345",collector="12346",resume="12347").items()]
        a.write_once(self.paths["consumer_clearance"],dict(**common,schema="final-checkpoint-all-consumers-clear-v1",
            inventory_complete=True,no_unlisted_dependents=True,state="ALL_CONSUMERS_VERIFIED_CLEAR",
            checked_at_utc="2026-10-09T01:02:00+00:00",consumers=entries))
        self.provenance={key:dict(sha256=hashlib.sha256(key.encode()).hexdigest(),fixture=True)
                for key in ("pinned_source","native_hparams","base_model","tokenizer","input")}
        self.provenance["evidence_adapter"]=self.adapter_contract
        self.manifest=self.seal()
        self.receiver = a.Receiver.from_policy_file(self.policy_path,cutover_members={origin:self.cutover_member},
            root=self.root/"receiver",test_only=True)

    def seal(self, adapter="DEFAULT"):
        if adapter=="DEFAULT":adapter=self.adapter_impl
        return a.seal_manifest(run=self.run,identity=self.identity,checkpoint=self.payload,companions=self.paths,
            policy_sha256=self.policy_sha,cutover_member=self.cutover_member,provenance_references=self.provenance,evidence_adapter=adapter)

    def receive(self, admission):
        for role,row in self.manifest["files"].items():
            shutil.copyfile(row["path"],Path(admission["staging"])/admission["incoming_names"][role])

    def mutated(self, role, transform):
        value = deepcopy(self.manifest)
        transform(value["proof_packet"][role])
        self._packet_member(value,role)
        if role=="adoption":
            for child in ("submission_lock","submission_receipt"):
                value["proof_packet"][child]["adoption_sha256"] = value["files"][role]["sha256"]
                self._packet_member(value,child)
        if role in ("adoption","submission_lock"):
            value["proof_packet"]["submission_receipt"]["submission_lock_sha256"] = value["files"]["submission_lock"]["sha256"]
            self._packet_member(value,"submission_receipt")
        value["manifest_sha256"] = a.digest(a._unsigned(value,"manifest_sha256"))
        return value

    @staticmethod
    def _packet_member(value,role):
        data = a._json_bytes(value["proof_packet"][role])
        value["proof_contents"][role] = data.decode()
        value["files"][role].update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

    def fresh_clearance(self, verified_at):
        paths = dict(self.paths)
        document=deepcopy(self.original);document["observed_at_utc"]=verified_at
        fresh_original=self.root/"fresh-original-evidence.json";a.write_once(fresh_original,document)
        fresh_member=member(fresh_original)
        for role in ("writer_termination","consumer_clearance"):
            proof = deepcopy(self.manifest["proof_packet"][role])
            if role=="consumer_clearance":
                proof["checked_at_utc"] = verified_at
                for row in proof["consumers"]:row["original_evidence_member"]=fresh_member
            else:proof["original_writer_evidence_member"]=fresh_member
            path = self.root/("fresh-"+role+".json")
            a.write_once(path,proof);paths[role] = path
        return paths


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="checkpoint-archive-cpu-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.clock = patch.object(a,"utc_now",return_value="2026-10-09T02:00:00+00:00")
        self.clock.start();self.addCleanup(self.clock.stop)
        self.f = Fixture(self.temp.name)
        self.space = patch.object(a.os,"statvfs",return_value=types.SimpleNamespace(f_bavail=2**40,f_frsize=1,f_favail=100000))
        self.space.start();self.addCleanup(self.space.stop)

    def assert_code(self,code,fn):
        with self.assertRaises(a.ArchiveError) as result: fn()
        self.assertEqual(result.exception.code,code)

    def verified(self):
        admission = self.f.receiver.admit(self.f.manifest)
        self.f.receive(admission)
        return self.f.receiver.verify(admission,self.f.manifest)

    def test_full_receiver_recheck_is_independent_create_once(self):
        receipt = self.verified()
        self.assertEqual(receipt["stage"],"VERIFIED_DESTINATION")
        self.assertTrue(self.f.receiver.recheck(receipt))
        self.assertFalse(receipt["source_deleted"])
        self.assertFalse(receipt["validation"]["scientific_completion_independently_certified"])
        self.assertEqual(receipt["sealed_source_manifest"],self.f.manifest)
        self.assertEqual(receipt["sealed_source_manifest"]["provenance_references"],self.f.provenance)
        self.assert_code("DESTINATION_COLLISION",lambda:self.f.receiver.admit(self.f.manifest))
        self.assertTrue(self.f.payload.exists())

    def test_presubmit_chronology_actual_receipt_is_after_submission(self):
        self.f.receiver.admit(self.f.manifest)
        packet = self.f.manifest["proof_packet"]
        self.assertLess(a.timestamp(packet["submission_lock"]["sealed_at_utc"]),
                        a.timestamp(packet["submission_receipt"]["scheduler_submitted_at_utc"]))
        self.assertGreater(a.timestamp(packet["submission_receipt"]["receipt_sealed_at_utc"]),
                           a.timestamp(packet["submission_receipt"]["scheduler_submitted_at_utc"]))

    def test_registered_before_adoption_holds(self):
        value=self.f.mutated("submission_receipt",lambda p:p.update(scheduler_submitted_at_utc="2026-10-08T23:59:00+00:00"))
        self.assert_code("LEGACY_OR_POST_SUBMISSION_ADOPTION_HOLD",lambda:self.f.receiver.admit(value))

    def test_adoption_before_recorded_cutover_holds(self):
        value=self.f.mutated("adoption",lambda p:p.update(adopted_at_utc="2026-10-08T23:59:00+00:00"))
        self.assert_code("LEGACY_SUBMISSION_PROTECTED",lambda:self.f.receiver.admit(value))

    def test_empty_legacy_adoption_rejected(self):
        value=self.f.mutated("adoption",lambda p:p.clear())
        self.assert_code("PRE_SUBMISSION_ADOPTION_REQUIRED",lambda:self.f.receiver.admit(value))

    def test_actual_submission_source_config_mismatch_rejected(self):
        value=self.f.mutated("submission_receipt",lambda p:p["checkpoint_identity"].update(config_sha256="f"*64))
        self.assert_code("ACTUAL_NEW_REGISTRATION_RECEIPT_REQUIRED",lambda:self.f.receiver.admit(value))

    def test_guessed_job_number_rejected(self):
        value=self.f.mutated("submission_receipt",lambda p:p.update(registration_response="sbatch exit0"))
        self.assert_code("ACTUAL_NEW_REGISTRATION_RECEIPT_REQUIRED",lambda:self.f.receiver.admit(value))

    def test_final_pointer_false_and_partial_rejected(self):
        for change in (dict(batch=19),dict(final_W20=False)):
            with self.subTest(change=change):
                value=self.f.mutated("latest_pointer",lambda p:p.update(change))
                self.assert_code("LATEST_POINTER_NOT_EXACT_FINAL_W20",lambda:self.f.receiver.admit(value))

    def test_missing_planned_final_calculation_rejected(self):
        value=self.f.mutated("scientific_terminal",lambda p:p["required_calculations"].pop("generation"))
        self.assert_code("ALL_PLANNED_FINAL_CALCULATIONS_REQUIRED",lambda:self.f.receiver.admit(value))

    def test_computation_exit0_does_not_substitute_actual_receipt(self):
        value=self.f.mutated("scientific_terminal",lambda p:p.update(actual_final_calculations_complete=False))
        self.assert_code("ACTUAL_FINAL_COMPUTATION_PROOF_REQUIRED",lambda:self.f.receiver.admit(value))

    def test_counter_boolean_spoof_rejected(self):
        value=self.f.mutated("scientific_terminal",lambda p:p["commit_batches"].__setitem__(0,True))
        self.assert_code("EXACT_FINAL_COUNTER_TYPES_REQUIRED",lambda:self.f.receiver.admit(value))

    def test_active_writer_rejected(self):
        value=self.f.mutated("writer_termination",lambda p:p.update(state="RUNNING"))
        self.assert_code("WRITER_STILL_ACTIVE_HOLD",lambda:self.f.receiver.admit(value))

    def test_missing_registered_consumer_rejected(self):
        value=self.f.mutated("consumer_clearance",lambda p:p["consumers"].pop())
        self.assert_code("ALL_REGISTERED_CONSUMERS_REQUIRED",lambda:self.f.receiver.admit(value))

    def test_consumer_in_use_and_wrong_job_rejected(self):
        for update in (dict(not_using_checkpoint=False),dict(actual_job_id="99999")):
            with self.subTest(update=update):
                value=self.f.mutated("consumer_clearance",lambda p:p["consumers"][-1].update(update))
                self.assert_code("CONSUMER_IN_USE_OR_UNKNOWN_HOLD",lambda:self.f.receiver.admit(value))

    def test_proof_packet_bound_to_original_file_bytes(self):
        value=deepcopy(self.f.manifest)
        value["proof_packet"]["writer_termination"]["state"]="RUNNING"
        value["manifest_sha256"]=a.digest(a._unsigned(value,"manifest_sha256"))
        self.assert_code("ADMISSION_PROOF_FULL_SHA_OR_CONTENT_MISMATCH",lambda:self.f.receiver.admit(value))

    def test_exact_policy_file_sha_not_normalized_digest(self):
        self.assertNotEqual(self.f.policy_sha,a.digest(a.read_json(self.f.policy_path)))
        value=deepcopy(self.f.manifest)
        value["policy_sha256"]=a.digest(a.read_json(self.f.policy_path))
        value["manifest_sha256"]=a.digest(a._unsigned(value,"manifest_sha256"))
        self.assert_code("RECORDED_SERVER_CUTOVER_REQUIRED",lambda:self.f.receiver.admit(value))

    def test_trusted_cutover_mutation_rejected(self):
        self.f.cutover.write_text("{}")
        self.assert_code("METADATA_MEMBER_FULL_SHA_OR_BYTES_MISMATCH",lambda:self.f.receiver.admit(self.f.manifest))

    def test_policy_file_mutation_rejected(self):
        self.f.policy_path.write_text("{}")
        self.assert_code("CANONICAL_POLICY_CHANGED",lambda:self.f.receiver.admit(self.f.manifest))

    def test_operating_reserve_and_inode_guard(self):
        for usage in (types.SimpleNamespace(f_bavail=a.RESERVE_BYTES,f_frsize=1,f_favail=100000),
                      types.SimpleNamespace(f_bavail=2**40,f_frsize=1,f_favail=1)):
            with self.subTest(usage=usage),patch.object(a.os,"statvfs",return_value=usage):
                self.assert_code("ARCHIVE_PENDING_KEEP_SOURCE",lambda:self.f.receiver.admit(self.f.manifest))

    def test_outstanding_reservation_concurrency_one(self):
        admission=self.f.receiver.admit(self.f.manifest)
        self.assertGreaterEqual(admission["capacity"]["requested_reservation_bytes"],
                                2*sum(row["bytes"] for row in self.f.manifest["files"].values()))
        self.assert_code("ARCHIVE_PENDING_KEEP_SOURCE",lambda:self.f.receiver.admit(self.f.manifest))

    def test_crash_orphan_reservation_never_bypasses_concurrency(self):
        with patch.object(a,"_replace_control",side_effect=RuntimeError("fixture crash after immutable reservation")):
            with self.assertRaises(RuntimeError): self.f.receiver.admit(self.f.manifest)
        self.assertFalse((self.f.receiver.root/".receiver"/"active.json").exists())
        self.assert_code("ARCHIVE_PENDING_KEEP_SOURCE",lambda:self.f.receiver.admit(self.f.manifest))

    def test_source_and_future_artifact_authority_bound_before_submit(self):
        value=self.f.mutated("submission_lock",lambda p:p.update(checkpoint_directory="/unrelated/model"))
        self.assert_code("PRESEALED_CHECKPOINT_PATH_AND_AUTHORITY_REQUIRED",lambda:self.f.receiver.admit(value))
        value=self.f.mutated("submission_lock",lambda p:p.update(checkpoint_creation_authorized=False))
        self.assert_code("PRESEALED_CHECKPOINT_PATH_AND_AUTHORITY_REQUIRED",lambda:self.f.receiver.admit(value))

    def test_receiver_future_completion_proof_rejected(self):
        value=self.f.mutated("consumer_clearance",lambda p:p.update(checked_at_utc="2026-10-10T00:00:00+00:00"))
        self.assert_code("FUTURE_COMPLETION_PROOF_HOLD",lambda:self.f.receiver.admit(value))

    def test_existing_empty_destination_rejected(self):
        destination=self.f.receiver.root/a._layout(self.f.manifest)
        destination.mkdir(parents=True)
        self.assert_code("DESTINATION_COLLISION",lambda:self.f.receiver.admit(self.f.manifest))

    def test_atomic_seal_collision_never_overwrites(self):
        admission=self.f.receiver.admit(self.f.manifest);self.f.receive(admission)
        destination=Path(admission["destination"]);destination.mkdir(parents=True)
        self.assert_code("DESTINATION_COLLISION",lambda:self.f.receiver.verify(admission,self.f.manifest))
        self.assertEqual(list(destination.iterdir()),[])
        self.assertTrue(self.f.payload.exists())

    def test_received_payload_corrupt_or_wrong_size_rejected(self):
        admission=self.f.receiver.admit(self.f.manifest);self.f.receive(admission)
        (Path(admission["staging"])/"checkpoint.pt").write_bytes(b"corrupt")
        self.assert_code("RECEIVED_FULL_SHA_OR_BYTES_MISMATCH",lambda:self.f.receiver.verify(admission,self.f.manifest))

    def test_extra_or_missing_transfer_file_rejected(self):
        admission=self.f.receiver.admit(self.f.manifest);self.f.receive(admission)
        (Path(admission["staging"])/"not-allowlisted.txt").write_text("extra")
        self.assert_code("RECEIVED_ALLOWLIST_MISMATCH",lambda:self.f.receiver.verify(admission,self.f.manifest))

    def test_new_extra_transfer_member_during_hash_never_seals(self):
        admission=self.f.receiver.admit(self.f.manifest);self.f.receive(admission)
        original=a.inspect_file; inserted=False
        def injected(path):
            nonlocal inserted
            row=original(path)
            if not inserted and str(path)==str(Path(admission["staging"])/"checkpoint.pt"):
                inserted=True
                (Path(admission["staging"])/"late-extra.txt").write_text("fixture concurrent uploader")
            return row
        with patch.object(a,"inspect_file",side_effect=injected):
            self.assert_code("RECEIVED_ALLOWLIST_CHANGED_BEFORE_SEAL",lambda:self.f.receiver.verify(admission,self.f.manifest))
        self.assertFalse(Path(admission["destination"]).exists())

    def test_received_symlink_and_hardlink_rejected(self):
        admission=self.f.receiver.admit(self.f.manifest);self.f.receive(admission)
        path=Path(admission["staging"])/"checkpoint.pt";path.unlink();path.symlink_to(self.f.payload)
        self.assert_code("SOURCE_OR_DESTINATION_NOT_REGULAR",lambda:self.f.receiver.verify(admission,self.f.manifest))
        path.unlink();os.link(self.f.payload,path)
        self.assert_code("SHARED_OBJECT_OR_NONREGULAR_HOLD",lambda:self.f.receiver.verify(admission,self.f.manifest))

    def test_source_symlink_ancestor_rejected(self):
        link=self.f.root/"source-link";link.symlink_to(self.f.source,target_is_directory=True)
        self.assert_code("SYMLINK_OR_UNSAFE_PATH_ANCESTOR",lambda:a.inspect_file(link/self.f.payload.name))

    def test_source_regular_hardlink_rejected(self):
        os.link(self.f.payload,self.f.root/"shared-object.pt")
        self.assert_code("SHARED_OBJECT_OR_NONREGULAR_HOLD",lambda:a.inspect_file(self.f.payload))

    def test_immutable_metadata_write_never_overwrites(self):
        path=self.f.root/"once.json";a.write_once(path,dict(value=1))
        self.assert_code("IMMUTABLE_PATH_ALREADY_EXISTS",lambda:a.write_once(path,dict(value=2)))
        self.assertEqual(a.read_json(path),dict(value=1))

    def test_receipt_recheck_rejects_later_archive_change(self):
        receipt=self.verified()
        Path(receipt["members"]["checkpoint"]["path"]).write_bytes(b"changed")
        self.assert_code("ARCHIVE_MEMBER_CHANGED",lambda:self.f.receiver.recheck(receipt))

    def test_source_gate_is_eligible_only_and_never_unlinks(self):
        self.f=Fixture(self.f.root/"remote-source",origin="server2")
        receipt=self.verified()
        gate=a.source_delete_gate(receipt,self.f.manifest,
            fresh_companions=self.f.fresh_clearance(receipt["verified_at_utc"]),destination_verify=self.f.receiver.recheck,
            evidence_adapter=fixture_evidence_adapter)
        self.assertEqual(gate["stage"],"ELIGIBLE_SOURCE_OWNER_FINAL_RECHECK_REQUIRED")
        self.assertFalse(gate["source_removed"])
        self.assertEqual(gate["eligible_payload_paths"],[str(self.f.payload)])
        self.assertTrue(self.f.payload.exists())

    def test_source_gate_requires_fresh_post_receiver_clearance(self):
        self.f=Fixture(self.f.root/"remote-source",origin="server2")
        receipt=self.verified()
        self.assert_code("FRESH_POST_RECEIPT_CLEARANCE_REQUIRED",lambda:a.source_delete_gate(receipt,self.f.manifest,
            fresh_companions=self.f.paths,destination_verify=self.f.receiver.recheck,evidence_adapter=fixture_evidence_adapter))

    def test_source_gate_detects_mutation_and_missing_receiver_ack(self):
        self.f=Fixture(self.f.root/"remote-source",origin="server2")
        receipt=self.verified();fresh=self.f.fresh_clearance(receipt["verified_at_utc"])
        self.assert_code("DESTINATION_RECHECK_REQUIRED",lambda:a.source_delete_gate(receipt,self.f.manifest,
            fresh_companions=fresh,destination_verify=lambda receipt:False,evidence_adapter=fixture_evidence_adapter))
        self.f.payload.write_bytes(b"source changed")
        self.assert_code("HOLD_SOURCE_CHANGED_OR_IN_USE",lambda:a.source_delete_gate(receipt,self.f.manifest,
            fresh_companions=fresh,destination_verify=self.f.receiver.recheck,evidence_adapter=fixture_evidence_adapter))

    def test_server1_own_source_always_keep_even_independently_verified_separate_copy(self):
        receipt=self.verified()
        self.assertTrue(self.f.receiver.recheck(receipt))
        source=a.inspect_file(self.f.payload)
        copied=receipt["members"]["checkpoint"]
        self.assertNotEqual((source["dev"],source["inode"]),(copied["dev"],copied["inode"]))
        self.assert_code("SERVER1_SOURCE_REGISTER_AND_PROTECT_KEEP_NO_UNLINK",lambda:a.source_delete_gate(
            receipt,self.f.manifest,fresh_companions=self.f.fresh_clearance(receipt["verified_at_utc"]),
            destination_verify=self.f.receiver.recheck,evidence_adapter=fixture_evidence_adapter))
        self.assertTrue(self.f.payload.exists())

    def test_git40_and_content64_hash_fields_are_not_interchangeable(self):
        for key in a.IDENTITY_FIELDS:
            if key == "official_tree_sha256":
                continue
            with self.subTest(field=key):
                identity=dict(self.f.identity)
                identity[key]="a"*(64 if key in a.GIT_IDENTITY_FIELDS else 40)
                self.assert_code("CHECKPOINT_IDENTITY_FULL_SHA_REQUIRED",lambda:a.checkpoint_identity(identity))
        a.checkpoint_identity(self.f.identity)

    def test_content_tree_roundtrip_and_type_tampering(self):
        f = Fixture(self.f.root/"content-tree", origin="server4", content_tree=True)
        original = dict(f.identity)
        manifest = f.manifest
        self.assertEqual(manifest["checkpoint_identity"], original)
        self.assertEqual(manifest["checkpoint_identity_types"]["official_tree_sha256"], "content-sha256")
        admission = f.receiver.admit(manifest)
        f.receive(admission)
        receipt = f.receiver.verify(admission, manifest)
        self.assertTrue(f.receiver.recheck(receipt))
        self.assertEqual(receipt["checkpoint_identity"], original)
        for change in ("missing", "wrong"):
            bad = deepcopy(manifest)
            if change == "missing":
                del bad["checkpoint_identity_types"]
            else:
                bad["checkpoint_identity_types"]["official_tree_sha256"] = "git-tree-sha1"
            bad["manifest_sha256"] = a.digest(a._unsigned(bad,"manifest_sha256"))
            self.assert_code("CHECKPOINT_IDENTITY_TYPE_BINDING_MISMATCH",lambda:a.validate_manifest(bad))

    def test_legacy_untyped_git_tree_manifest_accepted(self):
        old = deepcopy(self.f.manifest)
        del old["checkpoint_identity_types"]
        old["manifest_sha256"] = a.digest(a._unsigned(old,"manifest_sha256"))
        a.validate_manifest(old)

    def test_tree_hash_malformed_rejected(self):
        for tree in ("a"*39, "a"*41, "a"*63, "a"*65, "g"*64, "A"*40, None, 123):
            identity = dict(self.f.identity, official_tree_sha256=tree)
            self.assert_code("CHECKPOINT_IDENTITY_FULL_SHA_REQUIRED",lambda:a.checkpoint_identity(identity))

    def test_no_evidence_adapter_cannot_claim_verified_completion(self):
        self.assert_code("ARCHIVE_PENDING_KEEP_SOURCE",lambda:self.f.seal(adapter=None))

    def test_original_member_must_exist_and_match_current_full_sha(self):
        self.f.evidence.unlink()
        self.assert_code("SOURCE_OR_DESTINATION_NOT_REGULAR",lambda:self.f.seal())

    def test_original_evidence_mutation_or_wrong_identity_rejected(self):
        self.f.evidence.write_text("{}")
        self.assert_code("METADATA_MEMBER_FULL_SHA_OR_BYTES_MISMATCH",lambda:self.f.seal())

    def test_received_original_evidence_is_exact_allowlist_and_sha_checked(self):
        admission=self.f.receiver.admit(self.f.manifest);self.f.receive(admission)
        role=next(name for name in admission["incoming_names"] if name.startswith("original_"))
        (Path(admission["staging"])/admission["incoming_names"][role]).write_text("{}")
        self.assert_code("RECEIVED_FULL_SHA_OR_BYTES_MISMATCH",lambda:self.f.receiver.verify(admission,self.f.manifest))

    def test_original_replay_boolean_or_missing_binding_is_not_accepted(self):
        value=deepcopy(self.f.manifest)
        value["source_evidence_replay"]["records"][0]["result"]=True
        value["source_evidence_replay"]["replay_sha256"]=a.digest(a._unsigned(value["source_evidence_replay"],"replay_sha256"))
        value["manifest_sha256"]=a.digest(a._unsigned(value,"manifest_sha256"))
        with self.assertRaises(a.ArchiveError):self.f.receiver.admit(value)

    def test_original_wrong_model_or_unknown_schema_cannot_be_attested_complete(self):
        self.assert_code("ARCHIVE_PENDING_KEEP_SOURCE",lambda:Fixture(self.f.root/"other-model",
            original_transform=lambda document:document["checkpoint_identity"].update(model_revision="e"*64)))
        self.assert_code("ARCHIVE_PENDING_KEEP_SOURCE",lambda:Fixture(self.f.root/"unknown-schema",
            original_transform=lambda document:document.update(schema="EXIT0_IS_NOT_COMPLETION")))

    def test_original_live_consumer_or_incomplete_calc_rejected_by_replay(self):
        self.assert_code("ARCHIVE_PENDING_KEEP_SOURCE",lambda:Fixture(self.f.root/"live-consumer",
            original_transform=lambda document:document["consumer_states"].update(resume="RUNNING")))
        self.assert_code("ARCHIVE_PENDING_KEEP_SOURCE",lambda:Fixture(self.f.root/"missing-calculation",
            original_transform=lambda document:document["actual_final_calculations"].update(generation=False)))

    def test_arbitrary_boolean_adapter_never_creates_scientific_proof(self):
        self.assert_code("ARCHIVE_PENDING_KEEP_SOURCE",lambda:Fixture(self.f.root/"boolean-adapter",adapter=fixture_boolean_adapter))


if __name__=="__main__":unittest.main()
