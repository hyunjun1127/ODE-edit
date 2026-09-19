"""Independent tiny CPU audit: serialized RNG, finalizer and oracle contracts.

No GPU, model construction/load, scheduler, production asset/raw access, or
production checkpoint writes. Commit I/O is mocked with an in-memory actual
torch.save/weights_only load round trip; this does not validate fsync/mmap.
"""
import copy
from dataclasses import asdict
import io
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import transaction
from .config import ARMS, Scope, runtime_policy
from .current_oracle import MeasuredCurrentOracle
from .model import Runtime
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng
from project.run_scripts.low_cost_write_donor_pilot.fitting import NativeSingletonFitter
from project.run_scripts.single_layer_edit_preserving_correction.common import digest, tensor_sha
from project.run_scripts.single_layer_edit_preserving_correction.runtime import Runtime as LegacyRuntime
from project.run_scripts.single_layer_edit_preserving_correction.sequential_state import registry, require_finalizer


def fixture_lock():
    lock = dict(scope=asdict(Scope()), runtime_policy=runtime_policy(), stage="MATCHED_B1",
        resources=dict(GPU=1, CPU=8, mem_MiB=60416, wall_hours=24, node="server4",
                       export="NONE", requeue=0, GPUhour_hardcap=None),
        sequential_authorized=False, auto_continue=False,
        output="/data/janghj/ODE-edit/local/en-execution-reuse/20260919-v1/B1/cpu-audit/output",
        execution={"commit": "CPU_FIXTURE_NOT_EXECUTION_SOURCE"},
        model_revision=runtime_policy()["model_revision"], sample_order=list(range(100)))
    lock["lock_identity"] = digest(lock)
    return lock


class FakeRuntime:
    def __init__(self):
        self.lock = fixture_lock()
        self.W = torch.zeros(3, 4)
        self.M = torch.zeros(1, 4, 4)
        self.P = torch.eye(4)[None]
        self.hp = SimpleNamespace(layers=[4])
        self.model, self.tok = object(), object()
        self.context = [["CPU fixture"]]
        self.pmap = {"physical_layer": 4, "local_index": 0}
        self.records = [dict(case_id=i, requested_rewrite=dict(subject=f"fixture-{i}", relation_id="r",
                         target_new={"str": "new"})) for i in range(100)]
        self.history_calls = self.copy_calls = self.guard_calls = 0
        self.finalizer_fault = None
        self.fitter = SimpleNamespace(finalize=self.finalize)

    def requests(self, records):
        return [dict(copy.deepcopy(r["requested_rewrite"]), case_id=r["case_id"]) for r in records]

    def guard(self):
        self.guard_calls += 1
        if bool(torch.count_nonzero(self.M)):
            raise RuntimeError("COLD_M_HISTORY_MUTATED")

    def copy_weight(self, weight):
        if weight.dtype != torch.float32 or weight.shape != self.W.shape or not torch.isfinite(weight).all():
            raise ValueError("INVALID_WEIGHT_WRITE")
        self.W.copy_(weight)
        self.copy_calls += 1

    def finalize(self, model, tok, requests, bindings):
        if model is not self.model or tok is not self.tok or requests != self.requests(self.records):
            raise AssertionError("actual finalizer argument contract changed")
        if len(bindings) != 1:
            raise AssertionError("one finalizer binding required")
        layer, hp, memory, projector = bindings[0]
        if layer != 4 or hp is not self.hp or projector is not self.P or memory is self.M:
            raise AssertionError("B1 isolated final history binding changed")
        self.history_calls += 1
        before = tensor_sha(memory)
        memory.add_(torch.eye(4)[None])
        if self.finalizer_fault == "rng":
            random.random()
        if self.finalizer_fault == "weight":
            self.W.add_(1)
        receipt = dict(layer=4, history_append=1, compute_ks=1,
                       before_sha256=before, after_sha256=tensor_sha(memory), weight_sha256=tensor_sha(self.W))
        if self.finalizer_fault == "count":
            receipt["history_append"] = 2
        return [receipt]


class CommitHarness:
    """In-memory serialization, preserving actual RNG/payload Python types."""
    def __init__(self, tamper=None):
        self.tamper = tamper
        self.payload = None
        self.jsons = []
        self.saved = None
        self.loads = 0
        self._torch_load = torch.load

    def atomic(self, path, payload):
        self.payload = copy.deepcopy(payload)
        stream = io.BytesIO()
        torch.save(payload, stream)
        self.saved = stream.getvalue()
        return dict(path=str(path), bytes=len(self.saved), sha256="fixture-file-sha")

    def load(self, path, *, weights_only, mmap, map_location):
        if not weights_only or not mmap or map_location != "cpu":
            raise AssertionError("production safe mmap reload arguments changed")
        self.loads += 1
        payload = self._torch_load(io.BytesIO(self.saved), weights_only=True, map_location="cpu")
        if self.tamper:
            self.tamper(payload)
        return payload

    def write_json(self, path, value):
        self.jsons.append((str(path), copy.deepcopy(value)))


class TransactionTests(unittest.TestCase):
    def test_skip_reload_still_commits_one_history_and_checkpoint(self):
        harness = CommitHarness()
        with patch.object(transaction, 'atomic_without_reload', harness.atomic), \
             patch.object(transaction.torch, 'load', side_effect=AssertionError('reload must not run')), \
             patch.object(transaction, 'create_json', harness.write_json), \
             patch.object(transaction.Path, 'exists', return_value=False):
            result = transaction.commit(self.rt, ARMS[0], self.weight, {'decision': 'NOOP'},
                {'sha256': 'teacher-fixture'}, self.directory, verify_reload=False)
        self.assertEqual(self.rt.history_calls, 1)
        self.assertEqual(self.rt.copy_calls, 1)
        self.assertEqual(result['CPU_reload'], 'SKIPPED_USER_DIRECTED')
        self.assertEqual(result['physical_weight_reload'], 'NOT_RUN')
        self.assertIsNotNone(harness.saved)

    def setUp(self):
        self.rng = capture_rng()
        self.addCleanup(restore_rng, self.rng)
        self.rt = FakeRuntime()
        self.weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)/16
        self.directory = self.rt.lock["output"] + "/checkpoints/" + ARMS[0]

    def call(self, harness, arm=ARMS[0], directory=None):
        with patch.object(transaction, "atomic_tensor", harness.atomic), \
             patch.object(transaction.torch, "load", harness.load), \
             patch.object(transaction, "create_json", harness.write_json), \
             patch.object(transaction.Path, "exists", return_value=False):
            return transaction.commit(self.rt, arm, self.weight, {"decision": "NOOP"},
                                      {"sha256": "teacher-fixture"}, directory or self.directory)

    def test_actual_serialized_rng_and_finalizer_schema_round_trip(self):
        harness = CommitHarness()
        result = self.call(harness)
        self.assertEqual(result["status"], "B1_COMMITTED")
        self.assertEqual(self.rt.history_calls, 1)
        self.assertEqual(self.rt.copy_calls, 2)
        self.assertEqual(harness.loads, 1)
        self.assertEqual(set(harness.payload["rng"]), {"python", "numpy", "torch", "cuda"})
        self.assertEqual(harness.payload["rng"]["cuda"], [])
        self.assertEqual(digest(capture_rng()), digest(self.rng))
        self.assertEqual(tensor_sha(self.rt.W), tensor_sha(self.weight))
        self.assertEqual(int(torch.count_nonzero(self.rt.M)), 0)
        self.assertEqual(int(torch.count_nonzero(harness.payload["M4"])), 4)
        self.assertEqual(result["history_appends"], 1)
        self.assertEqual(result["candidate_history_appends"], 0)
        self.assertEqual(harness.payload["registry"], registry(harness.payload["received_ledger"]))
        self.assertEqual(len(harness.payload["received_ledger"]), 100)
        self.assertEqual(harness.payload["identity"]["order"], digest(self.rt.lock["sample_order"]))
        self.assertFalse(harness.payload["sequential_authorized"])
        self.assertFalse(harness.payload["auto_continue"])
        self.assertEqual(len(harness.jsons), 1)

    def test_finalizer_count_weight_and_rng_fail_before_checkpoint(self):
        for fault, message in (("count", "HISTORY_ONCE_BINDING"),
                               ("weight", "HISTORY_ONCE_BINDING"),
                               ("rng", "HISTORY_RNG_MUTATED")):
            with self.subTest(fault=fault):
                self.rt = FakeRuntime()
                self.rt.finalizer_fault = fault
                harness = CommitHarness()
                with self.assertRaisesRegex(ValueError, message):
                    self.call(harness)
                self.assertIsNone(harness.saved)
                self.assertEqual(harness.jsons, [])
                restore_rng(self.rng)

    def test_invalid_arm_preparation_phase_or_duplicate_does_not_finalize(self):
        with self.assertRaisesRegex(ValueError, "DUPLICATE_OR_UNAUTHORIZED"):
            self.call(CommitHarness(), arm="EN-F")
        self.assertEqual(self.rt.history_calls, 0)
        self.rt.lock["stage"] = "GENERATED_REFERENCE_PREPARATION"
        self.rt.lock["output"] = self.rt.lock["output"].replace("/B1/", "/PREP/")
        self.rt.lock["lock_identity"] = digest({k:v for k,v in self.rt.lock.items() if k != "lock_identity"})
        self.directory = self.rt.lock["output"] + "/checkpoints/" + ARMS[0]
        with self.assertRaisesRegex(ValueError, "DUPLICATE_OR_UNAUTHORIZED"):
            self.call(CommitHarness())
        self.assertEqual(self.rt.history_calls, 0)

    def test_duplicate_checkpoint_does_not_install_weight_or_finalize(self):
        with patch.object(transaction.Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "DUPLICATE_OR_UNAUTHORIZED"):
                transaction.commit(self.rt, ARMS[0], self.weight, {}, {}, self.directory)
        self.assertEqual(self.rt.copy_calls, 0)
        self.assertEqual(self.rt.history_calls, 0)

    def test_commit_directory_outside_locked_output_is_rejected_before_mutation(self):
        harness = CommitHarness()
        with self.assertRaises(ValueError):
            self.call(harness, directory="/tmp/outside-locked-B1-output")
        self.assertEqual(self.rt.copy_calls, 0)
        self.assertEqual(self.rt.history_calls, 0)
        self.assertIsNone(harness.saved)

    def test_nonzero_cold_history_is_rejected_before_checkpoint(self):
        self.rt.M.fill_(1)
        harness = CommitHarness()
        with self.assertRaisesRegex(RuntimeError, "COLD_M_HISTORY_MUTATED"):
            self.call(harness)
        self.assertEqual(self.rt.history_calls, 0)
        self.assertIsNone(harness.saved)

    def test_finalizer_failure_rolls_back_physical_weight_and_rng(self):
        for fault in ("weight", "rng"):
            with self.subTest(fault=fault):
                self.rt = FakeRuntime()
                entry = self.rt.W.clone()
                self.rt.finalizer_fault = fault
                with self.assertRaises(ValueError):
                    self.call(CommitHarness())
                self.assertEqual(tensor_sha(self.rt.W), tensor_sha(entry))
                self.assertEqual(digest(capture_rng()), digest(self.rng))
                restore_rng(self.rng)

    def test_checkpoint_tensor_or_identity_corruption_rejected(self):
        for mutate in (lambda p: p["weight"].add_(1),
                       lambda p: p["M4"].add_(1),
                       lambda p: p.update(weight=p["weight"].reshape(2, 6)),
                       lambda p: p.update(weight=p["weight"].view(torch.int32)),
                       lambda p: p["identity"].update(W="different"),
                       lambda p: p.update(sequential_authorized=True)):
            with self.subTest(mutate=mutate):
                self.rt = FakeRuntime()
                harness = CommitHarness(mutate)
                with self.assertRaisesRegex(ValueError, "CHECKPOINT_RELOAD_IDENTITY"):
                    self.call(harness)
                self.assertEqual(harness.jsons, [])

    def test_checkpoint_context_ledger_registry_and_scope_corruption_rejected(self):
        corruptions = {
            "context": lambda p: p.update(context=[["different"]]),
            "ledger": lambda p: p["received_ledger"].pop(),
            "registry": lambda p: p["registry"][0].update(status="SUPERSEDED"),
            "schema": lambda p: p.update(schema="wrong"),
            "batch": lambda p: p.update(batch=2),
            "auto_continue": lambda p: p.update(auto_continue=True),
            "source": lambda p: p.update(source={"commit": "other"}),
            "teacher": lambda p: p.update(teacher_manifest={"sha256": "other"}),
            "order": lambda p: p["sample_order"].reverse(),
            "next_batch": lambda p: p.update(next_batch=3),
            "history": lambda p: p["history"][0].update(history_append=2),
            "selection": lambda p: p.update(selection={"decision": "other"}),
        }
        for field, mutate in corruptions.items():
            with self.subTest(field=field):
                self.rt = FakeRuntime()
                harness = CommitHarness(mutate)
                with self.assertRaises(ValueError):
                    self.call(harness)
                self.assertEqual(harness.jsons, [])


class OracleContractTests(unittest.TestCase):
    def test_actual_native_finalizer_receipt_matches_transaction_contract(self):
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        memory, projector = torch.zeros(1, 4, 4), torch.eye(4)[None]
        model, tokenizer, hp = object(), object(), object()
        fitter = NativeSingletonFitter.__new__(NativeSingletonFitter)
        fitter.torch = torch
        fitter._check = lambda *args: ("selected", weight)
        def functions(counts, capture):
            self.assertIsNone(capture)
            def finalizer(actual_model, actual_tok, requests, actual_hp, *, cache_template, cache_c, P):
                self.assertIs(actual_model, model)
                self.assertIs(actual_tok, tokenizer)
                self.assertIs(actual_hp, hp)
                self.assertIs(cache_c, memory)
                self.assertIs(P, projector)
                counts["compute_ks"] = counts.get("compute_ks", 0) + 1
                cache_c.add_(torch.eye(4)[None])
                return actual_model, cache_c
            return None, finalizer
        fitter._functions = functions
        before = tensor_sha(memory)
        receipt = fitter.finalize(model, tokenizer, [{"case_id": 1}], [(4, hp, memory, projector)])
        require_finalizer(receipt, before, tensor_sha(weight), tensor_sha(memory))
        self.assertEqual(receipt[0]["history_append"], 1)
        self.assertEqual(receipt[0]["compute_ks"], 1)

    @staticmethod
    def bare_measured():
        oracle = MeasuredCurrentOracle.__new__(MeasuredCurrentOracle)
        oracle.device, oracle.shape = torch.device("cpu"), (3, 4)
        oracle.weight_work = dict(calls=0, finite_scanned_bytes=0, H2D_calls=0, H2D_bytes=0,
                                  wall_seconds=0., cuda_validation_upload_ms=0.)
        oracle._weight_events = []
        return oracle

    def test_measured_weight_preserves_cpu_validation_and_has_no_false_h2d(self):
        oracle = self.bare_measured()
        weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.assertIs(oracle._weight(weight), weight)
        self.assertEqual(oracle.weight_work["calls"], 1)
        self.assertEqual(oracle.weight_work["finite_scanned_bytes"], 48)
        self.assertEqual(oracle.weight_work["H2D_calls"], 0)
        self.assertEqual(oracle.collect_weight_timing(), oracle.weight_work)
        with self.assertRaises(ValueError):
            oracle._weight(weight.double())
        with self.assertRaises(FloatingPointError):
            oracle._weight(torch.full_like(weight, float("nan")))

    def test_protected_override_preserves_exact_dedup_and_metadata(self):
        # Same prefix+same key dedups; same prefix+different key does not;
        # different prefix+same key does not. No model is instantiated.
        keys = torch.tensor([[[1., 2.], [1., 2.], [2., 3.], [1., 2.]]])
        cache = SimpleNamespace(keys=keys)
        fake_oracle = SimpleNamespace(caches=[cache])
        aliases = [dict(cache=0, position=i, prefix_sha=prefix, key_column=i)
                   for i, prefix in enumerate(("a", "a", "a", "b"))]
        def sequences(*args):
            return ["packed"], ["row"], [], {"key_aliases": copy.deepcopy(aliases)}
        def runtime():
            return SimpleNamespace(tok=object(), etok=object(), model=object(), context=[["fixture"]],
                                   requests=lambda records: records, oracles=[])
        measured, legacy = runtime(), runtime()
        with patch("project.run_scripts.en_execution_reuse.model.protected_sequences", sequences), \
             patch("project.run_scripts.en_execution_reuse.model.MeasuredCurrentOracle", return_value=fake_oracle), \
             patch("project.run_scripts.single_layer_edit_preserving_correction.runtime.protected_sequences", sequences), \
             patch("project.run_scripts.single_layer_edit_preserving_correction.runtime.FullWeightLlamaOracle", return_value=fake_oracle):
            result = Runtime.protected_oracle(measured, [])
            expected = LegacyRuntime.protected_oracle(legacy, [])
        self.assertEqual(result[1], expected[1])
        torch.testing.assert_close(result[2], expected[2], atol=0, rtol=0)
        self.assertEqual(result[3], expected[3])
        self.assertEqual([x["actual_key_column"] for x in result[3]["actual_key_aliases"]], [0, 0, 1, 2])
        self.assertEqual(result[2].shape, (2, 3))
        self.assertEqual(measured.oracles, [fake_oracle])

    def test_physical_scope_restores_exact_weight_on_exception(self):
        # Exercise the inherited real copy/restore method with tiny CPU
        # Parameters and a stub guard; no model instantiation or forward.
        oracle = self.bare_measured()
        oracle.parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4), requires_grad=False)
        nonselected = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        oracle.model = SimpleNamespace(named_parameters=lambda: [("selected", oracle.parameter), ("other", nonselected)])
        oracle.work = dict(physical_weight_installs=0, physical_exact_restores=0,
                          physical_copy_restore_seconds=0.)
        # Match physical_weight's nonselected exclusion with its actual name.
        from project.run_scripts.single_layer_edit_preserving_correction.alltoken import WEIGHT
        oracle.model.named_parameters = lambda: [(WEIGHT, oracle.parameter), ("other", nonselected)]
        def guard(model):
            return tuple(("parameter", name, p.data_ptr(), p._version) for name,p in model.named_parameters()), ()
        oracle.guard = guard(oracle.model)
        oracle._guard = lambda: None
        oracle._sync = lambda: None
        entry = oracle.parameter.detach().clone()
        with patch("project.run_scripts.single_layer_edit_preserving_correction.alltoken.model_guard", guard):
            with self.assertRaisesRegex(RuntimeError, "fixture physical failure"):
                with oracle.physical_weight(entry+1, gradient=False):
                    self.assertTrue(torch.equal(oracle.parameter, entry+1))
                    raise RuntimeError("fixture physical failure")
        self.assertTrue(torch.equal(oracle.parameter, entry))
        self.assertFalse(oracle.parameter.requires_grad)
        self.assertEqual(oracle.work["physical_exact_restores"], 1)
        self.assertEqual(oracle.weight_work["calls"], 1)


if __name__ == "__main__":
    unittest.main()
