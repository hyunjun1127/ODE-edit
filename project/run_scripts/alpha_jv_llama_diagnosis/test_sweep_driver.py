"""CPU injected lifecycle fixtures; no model load and no numerical GPU claim."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256, tensor_sha256
from .contracts import BindingBoundary
from .fixture import EntrySnapshot
from .sweep_driver import SweepCallbacks, run_model_sweep


class FakeBundle:
    def __init__(self, family, ledger):
        self.artifact = deepcopy(family.fixed_z)
        self.semantic = deepcopy(family.semantic_inventory)
        self.ledger = ledger
        self.ledger["bundle_capture"] += 1

    def attach(self, family):
        if family.fixed_z is not None:
            raise BindingBoundary("FIXTURE_REPEATED_TARGET_ATTACH")
        family.fixed_z = deepcopy(self.artifact)
        family.semantic_inventory = deepcopy(self.semantic)
        self.ledger["target_attach"] += 1


class FakeFamily:
    def __init__(self, *, model, module, ledger):
        # The shared EntrySnapshot.make_family must restore before this call.
        if float(model.weight.detach().sum()) != 0 or bool(module.cache_c.any()):
            raise BindingBoundary("CPU_STALE_FAMILY_CONSTRUCTION")
        self.model, self.module, self.ledger = model, module, ledger
        self.parameters = {"weight": model.weight}
        self.w0 = {"weight": model.weight.detach().clone()}
        self.w0_sha256 = tensor_set_sha256(self.w0)
        self.fixed_z = None
        self.semantic_inventory = None
        self.metric_observer = None
        self.bound = False
        ledger["factories"] += 1

    def bind_existing_method_state(self):
        if self.bound:
            raise BindingBoundary("CPU_REPEATED_STATE_BIND")
        self.bound = True
        self.ledger["bind_existing"] += 1

    def prepare_method_state(self):
        raise AssertionError("cold preparation is forbidden inside restored fixture driver")

    def compute_fixed_z(self):
        if self.fixed_z is not None:
            raise BindingBoundary("CPU_RECOMPUTE_Z")
        self.ledger["compute_z"] += 1
        self.fixed_z = SimpleNamespace(values=torch.ones((2, 1), dtype=torch.float32),
                                       identity_sha256="complete-fixed-target-fixture")
        self.semantic_inventory = {"context": "bound", "request_order": "same-case-order"}

    def terminal(self):
        return self.model.weight.detach().reshape(2, 1).clone()

    def evaluate_endpoint(self):
        self.ledger["evaluate"] += 1
        return {"actual_weight_sum_at_evaluation": float(self.model.weight.detach().sum())}

    def reset_entry(self):
        with torch.no_grad():
            self.model.weight.copy_(self.w0["weight"])
            self.module.cache_c.zero_()
            self.module.cache_c_new = True

    def run_official(self, *, fixed_z):
        if fixed_z is not self.fixed_z or self.semantic_inventory["context"] != "bound":
            raise BindingBoundary("CPU_FIXED_TARGET_AUTHORITY")
        self.ledger["official"] += 1
        try:
            with torch.no_grad():
                self.model.weight.add_(2)
                self.module.cache_c.add_(1)
                self.module.cache_c_new = False
            self._captured_endpoint_weights = {"weight": self.model.weight.detach().clone()}
            self._captured_endpoint_method_state = self.module.cache_c.clone()
            self._captured_endpoint_cache_c_new = bool(self.module.cache_c_new)
            self.last_terminal = self.terminal()
            return {"status": "TERMINAL_VALID", "evaluation": self.evaluate_endpoint()}
        finally:
            self.reset_entry()


class FakeDictionary:
    def __init__(self, family):
        self.family = family
        self.qref = self.qfref = None
        family.ledger["dictionaries"] += 1

    def build(self, terminal, version):
        self.family.ledger["reference_build"] += 1
        return list(range(4, 9))

    def capture_reference(self, builds):
        if self.qref is not None or builds != list(range(4, 9)):
            raise BindingBoundary("CPU_REFERENCE_RECAPTURE")
        self.family.ledger["reference_capture"] += 1
        self.qref, self.qfref = 5., 7.
        return {"qN_ref": 5., "qF_ref": 7., "capture_count": 1}


def fake_trajectory(family, dictionary, normalization, config, *, output, arm, prefixes,
                    raw_sink=None, endpoint_sink=None):
    ledger = family.ledger
    ledger["joint_calls"] += 1
    ledger["path_configs"].append(config)
    ledger["normalization_objects"].append(id(normalization))
    if dictionary.family is not family or (dictionary.qref, dictionary.qfref) != (5., 7.):
        raise BindingBoundary("CPU_REFERENCE_BINDING")
    if family.semantic_inventory != {"context": "bound", "request_order": "same-case-order"}:
        raise BindingBoundary("CPU_COMPLETE_TARGET_BUNDLE")
    observations = {}
    try:
        # Lifecycle fixture only: not a native/controller approximation.
        for complete in range(1, config.N + 1):
            if raw_sink is not None:
                raw_sink(complete - 1, {"stage": "PRE_NODE", "increments_before": ()})
            with torch.no_grad():
                family.model.weight.add_(1)
            if raw_sink is not None:
                raw_sink(complete - 1, {"stage": "POST_NODE", "increments_after": ()})
            if complete in prefixes:
                label = prefixes[complete]
                ledger["prefix_evaluate"] += 1
                observations[label] = dict(config.endpoint_clock(complete), candidate_id=label,
                    history_append_count=0, persistent_endpoint_capture_count=0,
                    evaluation=family.evaluate_endpoint())
                if endpoint_sink is not None:
                    endpoint_sink(label, observations[label], {"weight": family.model.weight.detach().clone()}, family.terminal(), ())
        family._captured_endpoint_weights = {"weight": family.model.weight.detach().clone()}
        family._captured_endpoint_method_state = family.module.cache_c + 1
        family._captured_endpoint_cache_c_new = True
        observations[arm] = dict(config.endpoint_clock(config.N), candidate_id=arm,
            history_append_count=1, evaluation=family.evaluate_endpoint(), finite_poor_endpoint_included=True)
        if endpoint_sink is not None:
            endpoint_sink(arm, observations[arm], family._captured_endpoint_weights, family.terminal(), ())
        return {"status": "TERMINAL_VALID", "endpoints": observations, "nodes": [None] * config.N}
    finally:
        family.reset_entry()


class SweepDriverTests(unittest.TestCase):
    @contextmanager
    def fixture(self, **overrides):
        ledger = {key: 0 for key in ("bundle_capture", "target_attach", "factories", "bind_existing",
                  "compute_z", "evaluate", "official", "dictionaries", "reference_build",
                  "reference_capture", "joint_calls", "prefix_evaluate")}
        ledger.update(path_configs=[], normalization_objects=[])
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        module = SimpleNamespace(cache_c=torch.zeros((5, 2, 2), dtype=torch.float32), cache_c_new=True)
        entry = EntrySnapshot.capture({"weight": model.weight}, module)
        callbacks = SweepCallbacks(dictionary_factory=FakeDictionary,
                                  target_bundle_capture=lambda f: FakeBundle(f, ledger),
                                  trajectory=fake_trajectory)
        callbacks = replace(callbacks, **overrides)
        with TemporaryDirectory() as tmp:
            kwargs = dict(model_alias="llama3-8b-inst", model=model, module=module,
                          entry_snapshot=entry, family_factory=FakeFamily,
                          family_kwargs={"ledger": ledger}, output=Path(tmp) / "new-S-cell",
                          callbacks=callbacks)
            yield kwargs, ledger, entry

    def test_seven_actual_paths_nine_endpoints_complete_target_once(self):
        with self.fixture() as (kwargs, ledger, entry):
            result = run_model_sweep(**kwargs)
            self.assertEqual(result["status"], "S_NINE_ENDPOINTS_COMPLETE")
            self.assertEqual((ledger["joint_calls"], len(result["endpoints"])), (7, 9))
            self.assertEqual((ledger["compute_z"], ledger["bundle_capture"], ledger["target_attach"]), (1, 1, 8))
            self.assertEqual((ledger["reference_build"], ledger["reference_capture"]), (1, 1))
            self.assertEqual((ledger["factories"], ledger["bind_existing"]), (9, 9))
            self.assertEqual(sum(c.N for c in ledger["path_configs"]), 34)
            self.assertEqual(len(set(ledger["normalization_objects"])), 1)
            self.assertEqual(tensor_set_sha256({"weight": kwargs["model"].weight}), entry.W_sha256)
            self.assertFalse(result["D_restore_dependency"])

    def test_restoration_precedes_each_Family_construction(self):
        with self.fixture() as (kwargs, ledger, _entry):
            with torch.no_grad():
                kwargs["model"].weight.fill_(99)
                kwargs["module"].cache_c.fill_(88)
            # A model accidentally left changed is restored before any factory.
            run_model_sweep(**kwargs)
            self.assertEqual(ledger["factories"], 9)

    def test_endpoint_evaluation_is_actual_endpoint_not_restored_entry(self):
        with self.fixture() as (kwargs, ledger, _entry):
            result = run_model_sweep(**kwargs)
            self.assertEqual(result["entry_evaluation"]["actual_weight_sum_at_evaluation"], 0)
            self.assertEqual(result["Official_endpoint"]["evaluation"]["actual_weight_sum_at_evaluation"], 4)
            self.assertEqual(result["endpoints"]["JV-HOR-T1"]["observation"]["evaluation"]
                             ["actual_weight_sum_at_evaluation"], 4)
            self.assertEqual(result["endpoints"]["JV-BASE"]["observation"]["evaluation"]
                             ["actual_weight_sum_at_evaluation"], 8)
            self.assertEqual(ledger["evaluate"], 11)  # entry+Official+9; no reconstruction eval
            self.assertEqual(ledger["prefix_evaluate"], 2)

    def test_missing_endpoint_typed_failure_and_entry_cleanup(self):
        def missing(*args, **kwargs):
            result = fake_trajectory(*args, **kwargs)
            result["endpoints"].pop(next(iter(result["endpoints"])))
            return result
        with self.fixture(trajectory=missing) as (kwargs, ledger, entry):
            with self.assertRaisesRegex(BindingBoundary, "S_PATH_ENDPOINT_COMPLETENESS"):
                run_model_sweep(**kwargs)
            self.assertEqual(ledger["joint_calls"], 1)
            self.assertEqual(tensor_set_sha256({"weight": kwargs["model"].weight}), entry.W_sha256)

    def test_missing_restore_cannot_be_hidden_by_cleanup(self):
        def broken_restore(*args, **kwargs):
            result = fake_trajectory(*args, **kwargs)
            with torch.no_grad():
                args[0].model.weight.add_(.25)
            return result
        with self.fixture(trajectory=broken_restore) as (kwargs, _ledger, entry):
            with self.assertRaisesRegex(BindingBoundary, "S_RETURNED_WITHOUT_ENTRY_RESTORE"):
                run_model_sweep(**kwargs)
            self.assertEqual(tensor_set_sha256({"weight": kwargs["model"].weight}), entry.W_sha256)

    def test_wrong_prefix_clock_or_history_fails(self):
        for field, value, message in (("effective_T", 4, "S_ENDPOINT_CLOCK_MISMATCH"),
                                       ("history_append_count", 1, "S_PREFIX_LIFECYCLE_MISMATCH")):
            def bad_prefix(*args, **kwargs):
                result = fake_trajectory(*args, **kwargs)
                result["endpoints"]["JV-BASE"][field] = value
                return result
            with self.fixture(trajectory=bad_prefix) as (kwargs, _ledger, _entry):
                with self.assertRaisesRegex(BindingBoundary, message):
                    run_model_sweep(**kwargs)

    def test_qref_mutation_fails(self):
        def bad_reference(*args, **kwargs):
            result = fake_trajectory(*args, **kwargs)
            args[1].qref = 9
            return result
        with self.fixture(trajectory=bad_reference) as (kwargs, _ledger, _entry):
            with self.assertRaisesRegex(BindingBoundary, "S_ENTRY_REFERENCE_CHANGED"):
                run_model_sweep(**kwargs)

    def test_result_output_is_create_once(self):
        with self.fixture() as (kwargs, ledger, _entry):
            run_model_sweep(**kwargs)
            before = ledger["compute_z"]
            with self.assertRaises(FileExistsError):
                run_model_sweep(**kwargs)
            self.assertEqual(ledger["compute_z"], before)

    def test_first_fidelity_and_fixed_target_before_Official(self):
        events = []
        def target_sink(bundle, entry, normalization, reference):
            events.append("fixed_target_saved")
            self.assertEqual(reference["capture_count"], 1)
            self.assertEqual(bundle.semantic["context"], "bound")
        def fidelity(family, dictionary, normalization):
            self.assertEqual(family.ledger["official"], 0)
            self.assertEqual(family.ledger["compute_z"], 1)
            self.assertEqual(dictionary.qref, 5)
            events.append("fidelity")
            return {"status": "PASS_CPU_CALLBACK_FIXTURE"}
        def component(identity, payload):
            events.append((identity["component"], identity["phase"]))
        with self.fixture(first_fidelity=fidelity, fixed_target_sink=target_sink, component_sink=component) as (kwargs, _, _):
            result = run_model_sweep(**kwargs)
            self.assertLess(events.index("fixed_target_saved"), events.index("fidelity"))
            self.assertLess(events.index("fidelity"), events.index(("O_NATIVE", "begin")))
            self.assertEqual(result["first_fidelity"]["status"], "PASS_CPU_CALLBACK_FIXTURE")

    def test_fixed_target_sink_mutation_fail_close(self):
        def bad_sink(bundle, entry, normalization, reference):
            bundle.artifact.values.add_(1)
        with self.fixture(fixed_target_sink=bad_sink) as (kwargs, _, entry):
            with self.assertRaisesRegex(BindingBoundary, "S_OBSERVER_BINDING_MUTATION"):
                run_model_sweep(**kwargs)
            self.assertEqual(tensor_set_sha256({"weight": kwargs["model"].weight}), entry.W_sha256)

    def test_raw_and_endpoint_callbacks_have_path_identity_and_M_semantics(self):
        raw_rows, final_rows = [], []
        def raw(path_id, node, payload):
            raw_rows.append((path_id, node, payload["stage"]))
        def ep(path_id, label, result, weights, activation, entry, method_state, increments):
            final_rows.append((path_id, label, result["history_append_count"] if "history_append_count" in result else 1))
            if label in ("JV-HOR-T1", "JV-BASE"):
                self.assertTrue(torch.equal(method_state["alpha_cache"], entry.alpha_cache))
                self.assertEqual(method_state["cache_c_new"], entry.cache_c_new)
            else:
                self.assertTrue(torch.equal(method_state["alpha_cache"], entry.alpha_cache + 1))
                self.assertEqual(method_state["cache_c_new"], label != "O_NATIVE")
        with self.fixture() as (kwargs, _, _):
            run_model_sweep(**kwargs, raw_sink=raw, endpoint_sink=ep)
            self.assertEqual(len(set(row[0] for row in raw_rows)), 7)
            self.assertEqual(len(raw_rows), 68)
            self.assertEqual(len(set(raw_rows)), 68)
            self.assertEqual(len(final_rows), 10)  # Official+9, prefix does not duplicate path


if __name__ == "__main__":
    unittest.main()
