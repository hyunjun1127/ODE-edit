"""CPU seam tests for the sealed loop, not neural/sketch fidelity evidence.

Use the production HistoryLedger, parent two-candidate controller, and strict
atomic JSON writers. Mock only model/native/geometry/observer/report work.
Neither CUDA allocation nor scheduler commands are needed or permitted here.
"""
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from project.run_scripts.en_adapt_gss_history import runner, report
from project.run_scripts.en_adapt_gss_history.history import HistoryLedger
from project.run_scripts.en_adapt_gss_history.io import read, save_gzip
from project.run_scripts.en_adaptive_nullspace.controller import run_controller


ARMS = ("EN_ADAPT_H_RES", "EN_ADAPT_H_GSS_REC")


def fixture_records():
    return [dict(case_id=i, requested_rewrite=dict(subject=f"subject-{i}",
                relation_id="P-fixture", prompt="{} is", target_new=dict(id="Q1", str="new"),
                target_true=dict(id="Q0", str="old"))) for i in range(10000)]


class Harness:
    """Small CPU stand-ins share one mutable physical weight and owned M chain."""

    def __init__(self, failure_batch=None, failure_phase="fit", cleanup_failure=False,
                 invalid_json=False):
        self.runtime_instances = []
        self.ledgers = []
        self.objectives = []
        self.native = None
        self.records = fixture_records()
        self.failure_batch, self.failure_phase = failure_batch, failure_phase
        self.cleanup_failure, self.failed = cleanup_failure, False
        self.invalid_json = invalid_json
        self.observer_calls = 0
        self.report_calls = []

    def runtime(self, output, config):
        owner = self
        if config["arm"] not in ARMS:
            raise ValueError("EXACT_TWO_ARM_FIXED10K_SCOPE")
        if config["batches"] != 100 or config["batch_size"] != 100:
            raise ValueError("EXACT_TWO_ARM_FIXED10K_SCOPE")
        if config["save_checkpoints"] or config["cross_job_prefix_sharing"]:
            raise ValueError("NO_CP_NO_CROSS_JOB_STATE")

        class FakeRuntime:
            def __init__(self):
                self.records = owner.records
                self.W0 = torch.zeros((2, 2), dtype=torch.float32)
                self.W = self.W0.clone()
                self.P = torch.eye(2, dtype=torch.float32).unsqueeze(0)
                self.model = self
                self.tok = self.etok = None
                self.context = ["fixture context"]
                self.guard_calls = 0
                self.native_runner = owner.make_native(self)

            def install(self, weight):
                if owner.failed and owner.cleanup_failure and torch.equal(weight, self.W0):
                    raise RuntimeError("fixture rollback failure")
                self.W = weight.detach().clone()

            def guard(self):
                self.guard_calls += 1
                if not torch.isfinite(self.W).all():
                    raise ValueError("fixture nonfinite physical weight")

        rt = FakeRuntime()
        self.runtime_instances.append(rt)
        return rt

    def make_native(self, rt):
        owner = self

        class FakeNative:
            def __init__(self):
                self.fit_batches = self.fit_requests = self.history_appends = 0
                self.fit_entries, self.finalized_weights, self.M_entries = [], [], []

            def fit(self, requests, history):
                batch = self.fit_batches + 1
                if owner.failure_batch == batch and owner.failure_phase == "fit":
                    owner.failed = True
                    raise RuntimeError("fixture native failure")
                if len(requests) != 100 or int(history[0, 0, 0]) != self.history_appends:
                    raise AssertionError("native M did not follow its own uninterrupted trajectory")
                if self.finalized_weights and not torch.equal(rt.W, self.finalized_weights[-1]):
                    raise AssertionError("next native fit did not start from previous selected W")
                self.fit_entries.append(rt.W.clone())
                self.M_entries.append(history.clone())
                self.fit_batches += 1
                self.fit_requests += len(requests)
                delta = torch.full_like(rt.W, .5)
                weight = rt.W + delta
                rt.install(weight)
                return dict(weight=weight, actual_delta=delta,
                            receipt=dict(batch=batch, requests=len(requests), history_append=False))

            def finalize(self, requests, entry_history):
                if int(entry_history[0, 0, 0]) != self.history_appends:
                    raise AssertionError("native finalize duplicated or lost an M append")
                self.history_appends += 1
                self.finalized_weights.append(rt.W.clone())
                return dict(history=entry_history + 1,
                            receipt=dict(history_appends=self.history_appends, requests=len(requests)))

        self.native = FakeNative()
        return self.native

    def ledger(self, arm):
        obj = HistoryLedger(arm)
        self.ledgers.append(obj)
        return obj

    def objective(self, model, tok, generated_root, reference_inputs, history_root, basis,
                  expected_map_seal=None):
        owner = self

        class FakeObjective:
            def __init__(self):
                self.store = SimpleNamespace(receipt=dict(manifest_sha256="fixture-W0-reference"))
                self.reference = SimpleNamespace(counts=dict(native=0, candidate=0))
                self.replay = SimpleNamespace(counts=dict(capture_batches=0, captured_versions=0))
                self.bank_identity = None
                self.batch = 0
                self.captures, self.pool_sizes = [], []
                self.value = None

            def rebind(self, weight):
                if not torch.equal(model.W, weight):
                    raise AssertionError("oracle did not bind to the actual native endpoint")

            def prepare(self, weight, version_rows, *, arm, batch):
                self.batch = batch
                self.reference.counts["native"] += 1
                self.pool_sizes.append(len(version_rows))
                ids = [row["version_id"] for row in version_rows[:512]]
                for row in version_rows:
                    if row["teacher_binding"] is None:
                        raise AssertionError("replay consumed a missing original teacher")
                self.bank_identity = f"fixture-bank-{arm}-{batch}"
                self.value = dict(J=1., L_R=.75 if ids else 1., L_H=.25 if ids else 0.)
                if owner.invalid_json:
                    self.value["forbidden_tensor"] = np.ones((1, 2))
                bank = dict(selected_ids=ids, weights=([1 / len(ids)] * len(ids) if ids else []),
                            gradient=torch.full_like(weight, .25 if ids else 0., dtype=torch.float64),
                            reference_gradient=torch.full_like(weight, .75 if ids else 1., dtype=torch.float64),
                            receipt=dict(pool=len(version_rows), selected=len(ids),
                                         selected_ids=ids, weights=([1 / len(ids)] * len(ids) if ids else [])))
                return torch.ones_like(weight, dtype=torch.float64), self.value, bank

            def evaluate(self, weight):
                self.reference.counts["candidate"] += 1
                multiplier = 1.1 if self.batch % 3 == 2 else .7
                return dict(J=self.value["J"] * multiplier,
                            L_R=self.value["L_R"] * multiplier,
                            L_H=self.value["L_H"] * multiplier)

            def capture(self, version_rows, weight):
                if owner.failure_batch == self.batch and owner.failure_phase == "teacher":
                    owner.failed = True
                    raise RuntimeError("fixture teacher capture failure")
                self.captures.append((self.batch, len(version_rows)))
                self.replay.counts["capture_batches"] += 1
                self.replay.counts["captured_versions"] += len(version_rows)
                return dict(bindings={row["version_id"]: dict(path=f"fixture/{row['version_id']}",
                                 sha256="0" * 64, created_batch=self.batch)
                             for row in version_rows},
                            receipt=dict(new_versions=len(version_rows), batch=self.batch))

        obj = FakeObjective()
        self.objectives.append(obj)
        return obj

    def capture_current(self, model, tok, etok, requests, context):
        return dict(K=torch.eye(2, dtype=torch.float32), weights=np.array([.5, .5]),
                    representative_indices=[0, 1], oracle=object(),
                    manifest=dict(identity_sha256=f"current-{requests[0]['case_id']}",
                                  case_ids=[r["case_id"] for r in requests],
                                  aliases=[dict(case_id=r["case_id"], actual_key_column=0, weight=.01)
                                           for r in requests]))

    def geometry(self, keys, weights, representative_indices, basis):
        owner = self

        class FakeGeometry:
            diagnostic = dict(fixture_only=True)
            basis = np.eye(2, dtype=np.float64)
            vectors = np.array([[1.], [0.]], dtype=np.float64)
            rank = 1

            def gradient_spectrum(self, gradient, delta, loss):
                return dict(native_norm=float(delta.norm()), native_action=20.,
                            batch=owner.native.fit_batches)

            def direction(self, gradient, row):
                return gradient.clone()

            def diagnostics(self, ideal, actual, request_columns=None):
                ideal_norm = float(np.linalg.norm(ideal))
                actual_norm = float(np.linalg.norm(actual))
                rounding = float(np.linalg.norm(actual - ideal))
                return dict(finite=True, ideal_norm=ideal_norm, actual_norm=actual_norm,
                            ideal_response=ideal_norm, actual_response=actual_norm,
                            rounding_norm=rounding, rounding_response=rounding,
                            ideal_P_leakage=0., actual_P_leakage=0., rounding_P_leakage=0.)

        return FakeGeometry()

    def selector(self, spectrum):
        return dict(selected=dict(EN_ADAPT=dict(eta=0. if spectrum["batch"] % 3 == 1 else .25)),
                    frontiers=[dict(fixture_only=True)])

    def observer_state(self, arm):
        class FakeObserverState:
            def plan(self, records, ledger, batch):
                current = records[(batch - 1) * 100:batch * 100]
                return dict(missing_W0_case_ids=[r["case_id"] for r in current], records=current,
                            request_ids=[r["case_id"] for r in current])

            def remember_baseline(self, baseline):
                pass

            def record(self, batch, plan, **kwargs):
                return dict(cohorts=dict(current=plan["request_ids"]), batch=batch)

            def evidence(self, plan):
                return dict(fixture_request_ids=plan["request_ids"])

        return FakeObserverState()

    def observer_evaluate(self, model, tok, records):
        self.observer_calls += 1
        return dict(request_ids=[r["case_id"] for r in records], endpoint=float(model.W.sum()),
                    fixture_only=True)

    def report_build(self, output, arm):
        self.report_calls.append((Path(output), arm))
        return dict(arm=arm, fixture_only=True)

    def report_write(self, summary, destination):
        destination.mkdir(exist_ok=False)
        save_gzip(destination / "fixture-summary.json.gz", summary)

    def patches(self):
        stack = ExitStack()
        mappings = {
            "Runtime": self.runtime, "Objective": self.objective, "HistoryLedger": self.ledger,
            "capture_current": self.capture_current, "build_geometry": self.geometry,
            "select_arms": self.selector,
        }
        for name, replacement in mappings.items():
            stack.enter_context(patch.object(runner, name, replacement))
        stack.enter_context(patch.object(runner.np, "load", return_value=SimpleNamespace(
            shape=(14336, 14326), dtype=np.dtype("float64"))))
        stack.enter_context(patch.object(runner.gc, "collect", return_value=0))
        stack.enter_context(patch.object(runner.torch.cuda, "max_memory_allocated", return_value=0))
        stack.enter_context(patch.object(runner.observers, "ObserverState", self.observer_state))
        stack.enter_context(patch.object(runner.observers, "evaluate", self.observer_evaluate))
        stack.enter_context(patch.object(report, "build", self.report_build))
        stack.enter_context(patch.object(report, "write", self.report_write))
        return stack


class RunnerSeamTests(unittest.TestCase):
    def config(self, root, arm):
        config = dict(output=str(root / "output"), arm=arm, basis="fixture-basis",
                      generated_root="fixture-reference", reference_inputs="fixture-inputs",
                      history_root=str(root / "history"), batches=100, batch_size=100,
                      save_checkpoints=False, cross_job_prefix_sharing=False)
        path = root / "config.json"
        path.write_text(json.dumps(config))
        return path

    def test_both_arms_run_100_batches_one_ram_chain_through_normal_fallback(self):
        self.assertIs(runner.run_controller, run_controller)
        for arm in ARMS:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                harness = Harness()
                with harness.patches():
                    runner.run(self.config(root, arm))
                output = root / "output"
                self.assertEqual(len(harness.runtime_instances), 1)
                self.assertEqual(len(harness.ledgers), 1)
                self.assertEqual(len(harness.objectives), 1)
                ledger, objective, rt = harness.ledgers[0], harness.objectives[0], harness.runtime_instances[0]
                self.assertEqual((harness.native.fit_batches, harness.native.fit_requests,
                                  harness.native.history_appends), (100, 10000, 100))
                self.assertTrue(torch.equal(harness.native.fit_entries[0], rt.W0))
                self.assertTrue(torch.equal(harness.native.M_entries[0], torch.zeros_like(rt.P)))
                for batch, M in enumerate(harness.native.M_entries):
                    self.assertTrue(torch.equal(M, torch.full_like(rt.P, batch)))
                self.assertEqual(ledger.last_batch, 100)
                self.assertEqual(len(ledger.eventledger), 10000)
                self.assertEqual(len(set(ledger.commit_ids)), 100)
                self.assertEqual(ledger.pending_ids, [])
                self.assertEqual(objective.captures, [(batch, 100) for batch in range(1, 100)])
                self.assertEqual(objective.replay.counts["captured_versions"], 9900)
                terminal = read(output / "TERMINAL.json")
                self.assertEqual(terminal["counts"]["logical_commits"], 100)
                self.assertEqual(terminal["counts"]["checkpoint_writes"], 0)
                self.assertEqual(terminal["status"], "COMPLETED")
                self.assertTrue(terminal["final_weight_restored_to_W0"])
                self.assertTrue(torch.equal(rt.W, rt.W0))
                b2 = read(output / "B2_CONNECTION.json")
                self.assertEqual(b2["status"], "CONNECTED_CONTINUING_SAME_RAM")
                self.assertEqual(b2["logical_commits"], 2)
                self.assertFalse(b2["quality_gate"])
                self.assertTrue((output / "B003" / "COMMIT.json").is_file())
                blocks = read(output / "B003" / "spectrum.json.gz")["gradient_block_decomposition"]
                self.assertEqual(blocks["extra_model_passes"], 0)
                self.assertEqual(blocks["extra_SVDs"], 0)
                self.assertGreater(blocks["mode_cross_term"][0], 0)
                statuses = [read(output / f"B{b:03d}" / "controller.json.gz")["status"]
                            for b in (1, 2, 3)]
                self.assertEqual(statuses, ["NO_SIGNAL_OR_NO_STEP",
                                           "SEARCH_LIMIT_NO_ACCEPTED_CANDIDATE", "ACCEPTED"])
                controller = read(output / "B003" / "controller.json.gz")
                checks = controller["ledger"][0]["geometry_checks"]
                self.assertTrue(all(type(value) is bool for value in checks.values()))
                self.assertFalse((output / "technical-failure.json").exists())
                self.assertEqual(len(harness.report_calls), 1)
                self.assertGreater(harness.observer_calls, 100)
                self.assertEqual(read(output / "B100" / "teacher-bindings.json.gz")["bindings"], {})
                tensor_suffixes = {".pt", ".pth", ".npy", ".npz", ".safetensors"}
                self.assertFalse(any(path.suffix in tensor_suffixes for path in output.rglob("*")))

    def test_excluded_uniform_gss_arm_fails_before_runtime_or_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = Harness()
            with harness.patches(), self.assertRaisesRegex(ValueError, "EXACT_TWO_ARM"):
                runner.run(self.config(root, "EN_ADAPT_H_GSS"))
            self.assertEqual(harness.runtime_instances, [])
            receipt = read(root / "output" / "technical-failure.json")
            self.assertEqual(receipt["last_logical_commit"], 0)
            self.assertEqual(receipt["rollback"]["status"], "NOT_AVAILABLE")

    def test_runtime_failure_preserves_prior_commit_and_records_w0_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = Harness(failure_batch=2)
            with harness.patches(), self.assertRaisesRegex(RuntimeError, "fixture native failure"):
                runner.run(self.config(root, ARMS[0]))
            output = root / "output"
            receipt = read(output / "technical-failure.json")
            self.assertEqual(receipt["batch"], 2)
            self.assertEqual(receipt["last_logical_commit"], 1)
            self.assertEqual(receipt["rollback"]["status"], "W0_RESTORED_NONSELECTED_GUARD_PASS")
            self.assertTrue((output / "B001" / "COMMIT.json").is_file())
            self.assertFalse((output / "B002" / "COMMIT.json").exists())
            self.assertEqual(harness.ledgers[0].last_batch, 1)
            self.assertTrue(receipt["no_scientific_fallback_claim"])
            self.assertEqual(receipt["exact_crash_resume"], "NOT_AVAILABLE")

    def test_cleanup_error_is_separate_and_original_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = Harness(failure_batch=1, cleanup_failure=True)
            with harness.patches(), self.assertRaisesRegex(RuntimeError, "fixture native failure"):
                runner.run(self.config(root, ARMS[1]))
            receipt = read(root / "output" / "technical-failure.json")
            self.assertEqual(receipt["error"], "fixture native failure")
            self.assertEqual(receipt["rollback"]["status"], "RESTORE_NOT_VERIFIED")
            self.assertIn("fixture rollback failure", receipt["rollback"]["error"])
            self.assertEqual(receipt["last_logical_commit"], 0)

    def test_teacher_error_leaves_ledger_uncommitted_and_original_receipts_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = Harness(failure_batch=2, failure_phase="teacher")
            with harness.patches(), self.assertRaisesRegex(RuntimeError, "fixture teacher capture failure"):
                runner.run(self.config(root, ARMS[1]))
            output = root / "output"
            receipt = read(output / "technical-failure.json")
            self.assertEqual(receipt["phase"], "STAGE_COMMIT")
            self.assertEqual(receipt["last_logical_commit"], 1)
            self.assertEqual(harness.ledgers[0].last_batch, 1)
            self.assertEqual(len(harness.ledgers[0].eventledger), 100)
            self.assertTrue((output / "B002" / "SELECTION_SEALED.json").is_file())
            self.assertFalse((output / "B002" / "COMMIT.json").exists())
            self.assertEqual(receipt["counts"]["logical_commits"], 1)

    def test_strict_receipt_rejects_array_before_publish_and_no_logical_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = Harness(invalid_json=True)
            with harness.patches(), self.assertRaises(TypeError):
                runner.run(self.config(root, ARMS[0]))
            output = root / "output"
            self.assertFalse((output / "B001" / "native-objective.json.gz").exists())
            self.assertFalse((output / "B001" / "COMMIT.json").exists())
            receipt = read(output / "technical-failure.json")
            self.assertEqual(receipt["last_logical_commit"], 0)
            self.assertEqual(receipt["error_type"], "TypeError")
            self.assertEqual(receipt["rollback"]["status"], "W0_RESTORED_NONSELECTED_GUARD_PASS")


if __name__ == "__main__":
    unittest.main()
