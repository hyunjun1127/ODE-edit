"""CPU-only integration fixtures, not real-model or scientific gate evidence.

The actual program and RAM transaction functions run with tiny CPU tensors.
Native fitting, technical/model checks, observers and numerical stage decisions
are explicit mocks. No result/input artifacts, GPU, scheduler or disk state
are accessed; every attempted durable write is inspected in memory.
"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import program, transaction
from .config import B1_ARMS, CHAIN_ARMS, require_stage
from project.run_scripts.single_layer_edit_preserving_correction.common import digest, tensor_sha


def lock_fixture():
    return dict(phase="GATED_PROGRAM", maximum_batch=10, scientific_gates_required=True,
        scheduler_writes_in_program=False, agent_monitoring_after_release=False,
        disk_state_checkpoints=False, save_checkpoints=False, editor_sha256="CPU_FIXTURE",
        model_revision="CPU_FIXTURE", sample_order=list(range(1000)),
        execution={"commit": "CPU_FIXTURE_NOT_EXECUTION_EVIDENCE"})


def records_fixture():
    return [dict(case_id=i, requested_rewrite=dict(subject=f"subject-{i}", relation_id="r",
        prompt="{}", target_new={"str": "new"}, target_true={"str": "old"}))
        for i in range(1000)]


class MemoryEvidence:
    """Create-once JSON sink: tensor/RNG/full-state persistence is rejected."""
    def __init__(self):
        self.rows = {}

    @staticmethod
    def _check(value, path="root"):
        if isinstance(value, torch.Tensor):
            raise AssertionError("TENSOR_PERSISTENCE_FORBIDDEN:" + path)
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ("weight", "M4", "_state", "delta", "ideal_delta", "actual_delta"):
                    raise AssertionError("STATE_PAYLOAD_PERSISTENCE_FORBIDDEN:" + path + "." + key)
                # The identity contains only an RNG digest. A raw RNG mapping
                # or array is not permitted under this task's user override.
                if key == "rng" and not isinstance(child, str):
                    raise AssertionError("RNG_PAYLOAD_PERSISTENCE_FORBIDDEN:" + path)
                MemoryEvidence._check(child, path + "." + key)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                MemoryEvidence._check(child, f"{path}[{index}]")

    def write(self, path, value):
        key = str(path)
        if key in self.rows:
            raise FileExistsError(key)
        self._check(value)
        # Snapshot at call time, just as JSON serialization does. The real
        # transaction adds _state to its returned object only after this call.
        json.dumps(value, allow_nan=False)
        self.rows[key] = deepcopy(value)
        return dict(path=key, sha256=digest(value), bytes=len(json.dumps(value)))

    def ending(self, suffix):
        return [value for path, value in self.rows.items() if path.endswith(suffix)]


class FakeRuntime:
    def __init__(self):
        self.lock = lock_fixture()
        self.output = Path("/CPU_MOCK_ONLY_NO_FILES")
        self.records = records_fixture()
        self.W0 = torch.zeros((2, 3), dtype=torch.float32)
        self.W = self.W0.clone()
        self.M = torch.zeros((1, 3, 3), dtype=torch.float32)
        self.P = torch.eye(3, dtype=torch.float32)[None]
        self.hp = SimpleNamespace()
        self.module = SimpleNamespace()
        self.model, self.tok, self.etok = object(), object(), object()
        self.context = [["CPU fixture context"]]
        self.rng = {"fixture_rng_state": [1, 2, 3]}
        self.identity = {"CPU_FIXTURE": True}
        self.pmap = {"physical_layer": 4, "source_index": 0}
        self.generated_store = SimpleNamespace(receipt={"manifest_sha256": "CPU_FIXTURE"})
        self.oracles = []
        self.finalize_calls = []
        self.reference_object = object()
        self.fitter = SimpleNamespace(finalize=self.finalize)

    def copy_weight(self, weight):
        self.W.copy_(weight)

    def sync_oracles(self):
        pass

    def guard(self):
        if self.W.dtype != torch.float32 or self.M.dtype != torch.float32:
            raise AssertionError("CPU_STATE_DTYPE")

    def requests(self, records):
        return records

    def reference(self):
        return self.reference_object

    def finalize(self, model, tok, records, bindings):
        if len(bindings) != 1:
            raise AssertionError("HISTORY_LAYER_CARDINALITY")
        layer, _, memory, _ = bindings[0]
        before = tensor_sha(memory)
        memory.add_(1.)
        self.finalize_calls.append([r["case_id"] for r in records])
        return [dict(layer=layer, history_append=1, compute_ks=1, compute_z=0, solve=0,
            before_sha256=before, after_sha256=tensor_sha(memory), weight_sha256=tensor_sha(self.W))]


class ProgramHarness:
    """Run real stage routing and real transaction state in a sealed mock world."""
    def __init__(self, *, b1=True, s3=True, technical=True, resource_hold=None):
        self.rt = FakeRuntime()
        self.evidence = MemoryEvidence()
        self.batch_calls, self.observe_calls, self.mechanism_calls, self.resource_calls = [], [], [], []
        self.b1, self.s3, self.technical = b1, s3, technical
        self.resource_hold = resource_hold
        self.output = Path("/CPU_MOCK_ONLY_NO_FILES")
        self.ram_states = []
        self.scheduler_guards = []

    def batch(self, rt, reference, records, *, stage, arm_names, batch_number,
              ledger, directory, gate=None, shadow=False, history_cache_directory=None):
        if reference is not rt.reference_object or len(records) != 100:
            raise AssertionError("BATCH_REFERENCE_OR_CARDINALITY")
        expected = list(range((batch_number - 1) * 100, batch_number * 100))
        if [r["case_id"] for r in records] != expected or len(ledger) != expected[0]:
            raise AssertionError("OWN_CHAIN_PREFIX_OR_LEDGER")
        for arm in arm_names:
            require_stage(stage, arm, batch_number, gate)
        entry, entry_m = rt.W.clone(), rt.M.clone()
        if not torch.equal(entry_m, torch.full_like(entry_m, float(batch_number - 1))):
            raise AssertionError("OWN_CHAIN_MEMORY_BATCH_COUNT")
        self.batch_calls.append(dict(stage=stage, arms=tuple(arm_names), batch=batch_number,
            shadow=shadow, entry_sha=tensor_sha(entry), memory_sha=tensor_sha(entry_m),
            history_cache_directory=str(history_cache_directory) if history_cache_directory else None))
        commits, weights, selections, results = {}, {}, {}, {}
        for index, arm in enumerate(arm_names):
            # Distinct mock branch deltas expose accidental cross-arm sharing.
            code = {"N4": 1, "EN_KL_Q": 2, "DEC_LINE": 3, "DEC_MODES_STEP": 4, "DEC_MODES_CUM": 5}[arm]
            weight = entry + float(code + batch_number) / 100
            selections[arm] = {"sha256": digest(dict(arm=arm, batch=batch_number))}
            commits[arm] = transaction.commit(rt, arm, batch_number, records, weight,
                entry_m, ledger, selections[arm], Path(directory) / "commits" / arm,
                stage=stage, gate=gate)
            self.ram_states.append(commits[arm]["_state"])
            weights[arm] = weight
            results[arm] = SimpleNamespace(receipt=lambda arm=arm: dict(arm=arm, MOCK_NUMERICAL_RESULT=True))
        return dict(records=records, commits=commits, weights=weights, results=results,
            selections=selections, selection_seal={"sha256": "CPU_FIXTURE_SELECTION"},
            native={"weight": next(iter(weights.values()))}, entry=entry, entryM=entry_m,
            directory=Path(directory))

    def observe(self, rt, reference, result, w0, *, batch_number, stage, extra_reference=True):
        self.observe_calls.append((stage, batch_number, tuple(result["commits"])))
        value = {arm: dict(MOCK_OBSERVER=True, W0_correct_NS={"MOCK_RETENTION": True})
                 for arm in result["commits"]}
        return value, deepcopy(value)

    def space_check(self, root, required, phase):
        self.resource_calls.append((phase, required))
        if phase == self.resource_hold:
            raise program.StageResourceHold("CPU_FIXTURE_RESOURCE_HOLD:" + phase)

    def run(self):
        mock_observer = SimpleNamespace(observe=lambda *a, **k: {"MOCK_W0_OBSERVER": True})
        with ExitStack() as stack:
            for name in ("run", "Popen", "check_output", "check_call", "call"):
                self.scheduler_guards.append(stack.enter_context(patch("subprocess." + name,
                    side_effect=AssertionError("NO_SUBPROCESS_OR_SCHEDULER_IN_PROGRAM"))))
            stack.enter_context(patch("torch.save", side_effect=AssertionError("NO_DISK_TENSOR_SAVE")))
            stack.enter_context(patch("torch.load", side_effect=AssertionError("NO_EXISTING_RESULT_READ")))
            stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=0))
            stack.enter_context(patch("torch.cuda.max_memory_reserved", return_value=0))
            stack.enter_context(patch("builtins.print"))
            stack.enter_context(patch.object(program, "space_check", side_effect=self.space_check))
            stack.enter_context(patch.object(program, "inherit_hook", return_value=(
                {"weight": self.rt.W0.clone(), "MOCK_RETAINED_INPUT": True}, self.rt.records[:4])))
            stack.enter_context(patch("project.run_scripts.single_layer_mechanism_first.technical_decision.run_checks",
                return_value={"pass_": self.technical, "MOCK_NOT_ACTUAL_VALIDATION": True}))
            stack.enter_context(patch("project.run_scripts.single_layer_mechanism_first.native.CapturedHookedFitter",
                return_value=SimpleNamespace(finalize=self.rt.finalize)))
            stack.enter_context(patch.object(program, "batch", side_effect=self.batch))
            stack.enter_context(patch.object(program, "observe_batch", side_effect=self.observe))
            stack.enter_context(patch.object(program, "CanonicalObserver", return_value=mock_observer))
            stack.enter_context(patch("project.run_scripts.single_layer_mechanism_first.postselection.mechanism_report",
                side_effect=lambda *a, **k: self.mechanism_calls.append(k)))
            stack.enter_context(patch.object(program, "b1_gate", return_value={
                "name": "B1_TO_S3", "pass": self.b1, "MOCK_GATE_DECISION": True}))
            stack.enter_context(patch.object(program, "s3_gate", return_value={
                "name": "S3_TO_S10", "pass": self.s3, "MOCK_GATE_DECISION": True}))
            stack.enter_context(patch.object(program, "write", side_effect=self.evidence.write))
            stack.enter_context(patch.object(transaction, "write", side_effect=self.evidence.write))
            stack.enter_context(patch.object(program, "restore_rng"))
            stack.enter_context(patch.object(transaction, "capture_rng", return_value=deepcopy(self.rt.rng)))
            stack.enter_context(patch.object(transaction, "restore_rng"))
            # commit's duplicate-file guard remains fail-closed in its separate
            # test; this fixture has no real filesystem or historical result.
            stack.enter_context(patch("pathlib.Path.exists", return_value=False))
            result = program.run(self.rt, self.output)
        for guard in self.scheduler_guards:
            guard.assert_not_called()
        return result


class ProgramIntegrationTests(unittest.TestCase):
    def test_b1_failure_blocks_all_s3_and_s10(self):
        h = ProgramHarness(b1=False)
        h.run()
        self.assertEqual([(x["stage"], x["batch"]) for x in h.batch_calls], [("B1", 1)])
        self.assertEqual(h.batch_calls[0]["arms"], B1_ARMS)
        self.assertEqual(len(h.rt.finalize_calls), 4)
        self.assertEqual(h.evidence.ending("terminal.json")[0]["status"], "STOPPED_B1_GATE")
        self.assertFalse(h.evidence.ending("S3-to-S10.json"))
        self.assertEqual([x[0] for x in h.resource_calls], ["T0_B1"])

    def test_s3_failure_blocks_s10(self):
        h = ProgramHarness(s3=False)
        h.run()
        self.assertEqual([(x["stage"], x["batch"], x["arms"]) for x in h.batch_calls[1:]],
            [("S3", b, (a,)) for b in (2, 3) for a in CHAIN_ARMS])
        self.assertEqual(len(h.rt.finalize_calls), 10)
        self.assertEqual(h.evidence.ending("terminal.json")[0]["status"], "STOPPED_S3_GATE")
        self.assertEqual([x[0] for x in h.resource_calls], ["T0_B1", "S3"])

    def test_complete_conditional_serial_flow_and_ram_chain_isolation(self):
        h = ProgramHarness()
        h.run()
        expected = [("B1", 1, B1_ARMS)] + [
            ("S3" if b <= 3 else "S10", b, (a,)) for b in range(2, 11) for a in CHAIN_ARMS]
        self.assertEqual([(x["stage"], x["batch"], x["arms"]) for x in h.batch_calls], expected)
        self.assertEqual(len(h.batch_calls), 28)  # One shared B1 + 27 own-entry batches.
        self.assertEqual(len(h.rt.finalize_calls), 31)  # Four B1 commits + three times nine.
        self.assertEqual([x for x in h.batch_calls if x["shadow"]], [h.batch_calls[1]])
        b2 = {x["arms"][0]: x for x in h.batch_calls if x["batch"] == 2}
        self.assertEqual(b2["DEC_MODES_STEP"]["entry_sha"], b2["DEC_MODES_CUM"]["entry_sha"])
        self.assertNotEqual(b2["N4"]["entry_sha"], b2["DEC_MODES_CUM"]["entry_sha"])
        b3 = {x["arms"][0]: x for x in h.batch_calls if x["batch"] == 3}
        self.assertNotEqual(b3["DEC_MODES_STEP"]["entry_sha"], b3["DEC_MODES_CUM"]["entry_sha"])
        terminal = h.evidence.ending("terminal.json")[0]
        self.assertEqual(terminal["status"], "S10_COMPLETE")
        self.assertEqual(terminal["automatic_slurm_submissions"], 0)
        self.assertFalse(terminal["agent_monitoring_required"])
        for arm in CHAIN_ARMS:
            self.assertEqual(terminal["completed"][arm], list(range(2, 11)))
            cache_paths = {x["history_cache_directory"] for x in h.batch_calls
                           if x["batch"] > 1 and x["arms"] == (arm,)}
            self.assertEqual(len(cache_paths), 1)
            self.assertNotIn(None, cache_paths)
        self.assertTrue(all(len(ids) == 100 for ids in h.rt.finalize_calls))

    def test_no_checkpoint_or_equivalent_payload_persisted(self):
        h = ProgramHarness()
        h.run()
        commits = h.evidence.ending("commit.json")
        self.assertEqual(len(commits), 31)
        self.assertEqual(len(h.ram_states), 31)
        self.assertTrue(all("weight" in x and "M4" in x and "rng" in x for x in h.ram_states))
        for result in commits:
            self.assertIsNone(result["checkpoint"])
            self.assertEqual(result["checkpoint_IO_seconds"], 0)
            self.assertEqual(result["exact_crash_resume"], "NOT_AVAILABLE")
            self.assertNotIn("_state", result)
        self.assertTrue(all(path.endswith(".json") for path in h.evidence.rows))

    def test_technical_not_established_does_not_become_ready_or_batch(self):
        h = ProgramHarness(technical=False)
        with self.assertRaisesRegex(RuntimeError, "T0_FULL_CHECKS_NOT_ESTABLISHED"):
            h.run()
        self.assertFalse(h.batch_calls)
        self.assertFalse(h.evidence.ending("T0_READY.json"))
        self.assertFalse(h.evidence.ending("terminal.json"))

    def test_storage_hold_is_not_scientific_pass_or_next_batch(self):
        h = ProgramHarness(resource_hold="S3")
        h.run()
        self.assertEqual(len(h.batch_calls), 1)
        terminal = h.evidence.ending("terminal.json")[0]
        self.assertEqual(terminal["status"], "RESOURCE_HOLD")
        self.assertIsNone(terminal["last_gate"])

    def test_authority_and_no_checkpoint_fail_closed(self):
        program.require_program(lock_fixture())
        for key, invalid in (("phase", "S10"), ("maximum_batch", 11),
            ("scientific_gates_required", False), ("scheduler_writes_in_program", True),
            ("agent_monitoring_after_release", True), ("disk_state_checkpoints", True),
            ("save_checkpoints", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                program.require_program(dict(lock_fixture(), **{key: invalid}))

    def test_evidence_sink_rejects_state_delta_rng_and_tensors(self):
        for payload in (dict(weight=[[1.]]), dict(M4=[[1.]]), dict(delta=[[1.]]),
                        dict(rng={"raw": [1]}), dict(nested=[torch.ones(2)])):
            with self.subTest(payload=list(payload)), self.assertRaises(AssertionError):
                MemoryEvidence().write("/CPU_MOCK_ONLY.json", payload)


class RAMTransactionTests(unittest.TestCase):
    def setUp(self):
        self.rt = FakeRuntime()
        self.evidence = MemoryEvidence()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(transaction, "capture_rng", return_value=deepcopy(self.rt.rng)))
        self.stack.enter_context(patch.object(transaction, "restore_rng"))
        self.stack.enter_context(patch.object(transaction, "write", side_effect=self.evidence.write))
        self.stack.enter_context(patch("torch.save", side_effect=AssertionError("NO_CHECKPOINT_SAVE")))
        self.exists = self.stack.enter_context(patch("pathlib.Path.exists", return_value=False))

    def commit(self):
        return transaction.commit(self.rt, "DEC_MODES_CUM", 1, self.rt.records[:100],
            torch.ones_like(self.rt.W), self.rt.M.clone(), [], {"sha256": "CPU_SELECTION"},
            Path("/CPU_MOCK_ONLY/commit"), stage="B1")

    def test_history_once_commit_restores_entry_then_exact_ram_restore(self):
        before_w, before_m = self.rt.W.clone(), self.rt.M.clone()
        result = self.commit()
        self.assertEqual(len(self.rt.finalize_calls), 1)
        self.assertTrue(torch.equal(self.rt.W, before_w))
        self.assertTrue(torch.equal(self.rt.M, before_m))
        state = result["_state"]
        ledger, identity = transaction.restore(self.rt, state, arm="DEC_MODES_CUM", next_batch=2)
        self.assertEqual(len(ledger), 100)
        self.assertEqual(identity, state["identity"])
        self.assertTrue(torch.equal(self.rt.W, torch.ones_like(self.rt.W)))
        self.assertTrue(torch.equal(self.rt.M, torch.ones_like(self.rt.M)))
        self.assertEqual(result["candidate_observer_appends"], 0)

    def test_ram_clone_is_independent_b1_alias_not_new_history(self):
        state = self.commit()["_state"]
        clone = transaction.clone_b1_cum_as_step(state)
        clone["weight"].zero_(); clone["M4"].zero_()
        clone["received_ledger"][0]["requested_rewrite"]["subject"] = "changed"
        self.assertEqual(clone["arm"], "DEC_MODES_STEP")
        self.assertEqual(clone["new_fit_calls"], 0)
        self.assertEqual(len(self.rt.finalize_calls), 1)
        self.assertEqual(int(torch.count_nonzero(state["weight"])), 6)
        self.assertEqual(int(torch.count_nonzero(state["M4"])), 9)
        self.assertEqual(state["received_ledger"][0]["requested_rewrite"]["subject"], "subject-0")

    def test_wrong_arm_next_index_teacher_or_mutated_state_rejected(self):
        state = self.commit()["_state"]
        for changes, arm, next_batch in (({}, "N4", 2), ({}, "DEC_MODES_CUM", 3),
                ({"teacher_manifest": "other"}, "DEC_MODES_CUM", 2),
                ({"weight": torch.zeros_like(self.rt.W)}, "DEC_MODES_CUM", 2)):
            candidate = dict(state, **changes)
            with self.subTest(changes=list(changes), arm=arm, next_batch=next_batch), self.assertRaises(ValueError):
                transaction.restore(self.rt, candidate, arm=arm, next_batch=next_batch)

    def test_duplicate_commit_guard_precedes_history(self):
        self.exists.return_value = True
        with self.assertRaisesRegex(ValueError, "DUPLICATE_COMMIT"):
            self.commit()
        self.assertFalse(self.rt.finalize_calls)

    def test_commit_write_error_preserves_actual_entry_and_is_not_fallback(self):
        self.stack.enter_context(patch.object(transaction, "write", side_effect=OSError("CPU_IO_ERROR")))
        before_w, before_m = self.rt.W.clone(), self.rt.M.clone()
        with self.assertRaisesRegex(OSError, "CPU_IO_ERROR"):
            self.commit()
        self.assertTrue(torch.equal(self.rt.W, before_w))
        self.assertTrue(torch.equal(self.rt.M, before_m))
        self.assertFalse(self.evidence.rows)


class NarrowRedRegressionTests(unittest.TestCase):
    def test_actual_held_argv_mismatch_rejected_without_scheduler(self):
        from .submit_program import inspection
        source, lock = Path("/frozen"), Path("/own.lock")
        script = source / "project/run_scripts/single_layer_mechanism_first/run.sbatch"
        command = ["sbatch", str(script), str(source), str(lock)]
        common = ("UserId=janghj(123) JobName=odeedit_slmf_S10_s4 JobState=PENDING "
            "Reason=JobHeldUser gres/gpu=1 NumCPUs=8 mem=60416M ReqNodeList=server4 Requeue=0 ")
        good = common + f"Command={script} {source} {lock} WorkDir={source} Dependency=afterok:50974"
        bad = common + f"Command={script} /WRONG_SOURCE /WRONG_LOCK WorkDir={source} Dependency=afterok:50974"
        self.assertTrue(all(inspection(good, command, source, lock).values()))
        self.assertFalse(inspection(bad, command, source, lock)["actual_full_argv"])

    def test_compressed_array_capacity_fails_closed_without_queue_read(self):
        from .submit_program import other_capacity
        self.assertEqual(other_capacity("50974|parent|RUNNING|gpu:1"), [])
        rows = other_capacity("60000_0|other|RUNNING|gpu:1\n60000_1|other|PENDING|gpu:1")
        self.assertEqual(sum(x["GPU"] for x in rows), 2)
        for value in ("60000_[0-9%2]|other|PENDING|gpu:1", "60000|other|PENDING|UNKNOWN"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                other_capacity(value)

    def test_science_history_receipt_property_api_no_model(self):
        from . import science
        from .history import HistoryObservation
        h = HistoryObservation("w", "h", None, [], 0., True, None,
            dict(complete=True, active_requests=0), {})
        rt = FakeRuntime()
        rt.native = lambda *a: {"weight": torch.ones_like(rt.W)}
        factory = SimpleNamespace(receipt={"CPU_PROPERTY_FIXTURE": True})
        reference_result = SimpleNamespace(compact=lambda: {})
        sink = MemoryEvidence()
        with ExitStack() as stack:
            stack.enter_context(patch("pathlib.Path.mkdir"))
            stack.enter_context(patch.object(science, "write", side_effect=sink.write))
            stack.enter_context(patch.object(science, "capture_rng", return_value={}))
            stack.enter_context(patch("project.run_scripts.single_layer_mechanism_first.history_runtime.HistoryFactory",
                                      return_value=factory))
            stack.enter_context(patch.object(science, "history_evaluate", return_value=h))
            stack.enter_context(patch.object(science, "scan", return_value=reference_result))
            stack.enter_context(patch.object(science, "DecisionOracle"))
            stack.enter_context(patch.object(science, "commit", return_value={"MOCK_COMMIT": True}))
            result = science.batch(rt, None, rt.records[:100], stage="S3", arm_names=("N4",),
                batch_number=2, ledger=[], directory=Path("/CPU_MOCK_ONLY_RECEIPT"),
                gate={"name": "B1_TO_S3", "pass": True})
        self.assertEqual(sink.ending("history-prefix-receipt.json"), [factory.receipt])
        self.assertEqual(set(result["commits"]), {"N4"})

    def test_t0_pass_cannot_certify_wrong_current_native_replay(self):
        from . import postselection
        from .mechanism import MechanismTechnicalError
        rt = FakeRuntime()
        diagnostics = SimpleNamespace(receipt={"source_order": {"replay_endpoint_equal": False}},
            mode_rows=[], mode_factors=[], algebra_map=torch.zeros(3, 1), ideal_map=None)
        result = dict(selection_seal={"sha256": "CPU_SELECTION"}, entry=rt.W0,
            entryM=rt.M, native={"weight": rt.W0, "captures": {"solve_update": [torch.zeros_like(rt.W0)]}},
            reference_native=SimpleNamespace(factors=None))
        reference = SimpleNamespace(_capsules=[{"source_row_id": i} for i in range(512)])
        sink = MemoryEvidence()
        with ExitStack() as stack:
            stack.enter_context(patch.object(postselection, "extract_native_capture", return_value={
                "K": torch.zeros(3, 1), "R": torch.zeros(2, 1), "receipt": {}}))
            stack.enter_context(patch.object(postselection, "allowed_space", return_value=SimpleNamespace(
                basis=torch.eye(3).numpy())))
            stack.enter_context(patch.object(postselection, "analyze_writer", return_value=diagnostics))
            stack.enter_context(patch.object(postselection, "stream_reference_action", return_value={"rows": []}))
            stack.enter_context(patch.object(postselection, "write", side_effect=sink.write))
            stack.enter_context(patch.object(postselection, "save_tensor", return_value={"MOCK_SAVE": True}))
            with self.assertRaisesRegex(MechanismTechnicalError, "established native parity required"):
                postselection.mechanism_report(rt, reference, result, {}, {},
                    Path("/CPU_MOCK_ONLY_PARITY"), technical={"pass_": True})


if __name__ == "__main__":
    unittest.main()
