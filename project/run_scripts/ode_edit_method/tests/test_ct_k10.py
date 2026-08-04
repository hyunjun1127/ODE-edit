from __future__ import annotations

import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_edit_method.contracts import (
    EventReading,
    MethodContractError,
)
from project.run_scripts.ode_edit_method.ct_k4 import CTArm, run_ct_arm
from project.run_scripts.ode_edit_method.ct_k10 import (
    CT_K10,
    CT_K10_ARM_ORDER,
    CT_K10_CUMULATIVE,
    CT_K10_LAMBDAS,
    NAIVE_TENTH_RESIDUAL_FRACTION,
    CTK10Arm,
    _run_k10_arm,
    assert_shared_direct_z_identities,
    run_ct_k10_arm,
)
from project.run_scripts.ode_edit_method.ct_k10_lock import (
    CT_K10_LOCK_PATH,
    load_ct_k10_lock,
)
from project.run_scripts.ode_edit_method.events import ControllerRequest
from project.run_scripts.ode_edit_method.hooks import (
    FactorDirection,
    TorchCheckpoint,
    apply_accepted_factors,
    set_cumulative_factors_from_checkpoint,
)
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.lock import controller_config, load_lock
from project.run_scripts.ode_edit_method.tests.test_ct_k4 import (
    _ToyTransportBackend,
    _TwoLinear,
)
from project.run_scripts.session03_ct_k10_common import dry_plan
from project.run_scripts.session03_ct_k4_common import (
    _assert_case_direct_z_accounting,
    _assert_frozen_endpoint_identity,
)
from project.run_scripts.session03_ct_k10_common import CT_K10_SPEC


class _K10ToyTransportBackend(_ToyTransportBackend):
    def __init__(self, instrumentation: EditInstrumentation | None = None) -> None:
        super().__init__()
        self.instrumentation = instrumentation
        self.build_count = 0

    def build_transport_field(self, frozen_target):
        self.build_count += 1
        if self.instrumentation is not None:
            self.instrumentation.increment("N_bw")
        return super().build_transport_field(frozen_target)

    def commit_frozen_cumulative(self, entry, batch, coefficients, fraction):
        self.state = int(round(float(fraction) * CT_K10))
        return tuple(float(fraction) * value for value in coefficients)


class _NoChangeK10Backend(_K10ToyTransportBackend):
    class _NoChangeTrial:
        def __init__(self, backend, coefficients) -> None:
            self.backend = backend
            self.applied_coefficients = tuple(coefficients)

        def __enter__(self):
            self.backend.trial_state = self.backend.state
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.backend.trial_state = None
            return False

    def trial(self, batch, coefficients):
        return self._NoChangeTrial(self, coefficients)

    def commit(self, batch, coefficients):
        return tuple(coefficients)


class _FailingK10Backend(_K10ToyTransportBackend):
    def commit(self, batch, coefficients):
        if self.state == 1:
            raise RuntimeError("injected K10 commit failure")
        return super().commit(batch, coefficients)


class CTK10Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = controller_config(load_lock())
        self.request = ControllerRequest(
            "1", "{} is", "Ada", "Paris", "London"
        )

    def test_equal_macro_time_schedule_and_naive_negative_control(self) -> None:
        remaining = 1.0
        endpoints = []
        for value in CT_K10_LAMBDAS:
            remaining *= 1.0 - value
            endpoints.append(1.0 - remaining)
        self.assertEqual(len(endpoints), CT_K10)
        for observed, expected in zip(
            endpoints, CT_K10_CUMULATIVE, strict=True
        ):
            self.assertAlmostEqual(observed, expected, places=15)
        self.assertAlmostEqual(
            NAIVE_TENTH_RESIDUAL_FRACTION, 0.3486784401, places=15
        )

    def test_frozen_k10_final_bytes_and_output_equal_one_shot_bf16(self) -> None:
        torch.manual_seed(37)
        model = _TwoLinear(torch.bfloat16).eval()
        direction = FactorDirection(
            0,
            "layers.0.weight",
            torch.tensor([[0.37], [-0.19], [0.11]], dtype=torch.float32),
            torch.tensor([[0.23], [-0.41]], dtype=torch.float32),
        )
        coefficients = (0.73,)
        entry = TorchCheckpoint.capture(
            model, (direction.weight_name,), backup_device="cpu"
        )
        entry_weight = model.layers[0].weight.detach().clone()
        probe = torch.tensor([[0.31, -0.27]], dtype=torch.bfloat16)
        pointer = model.layers[0].weight.data_ptr()
        requires_grad = model.layers[0].weight.requires_grad
        rng = torch.get_rng_state().clone()

        apply_accepted_factors(model, (direction,), coefficients)
        one_shot_weight = model.layers[0].weight.detach().clone()
        one_shot_output = model(probe).detach().clone()
        entry.restore(model)

        first_step_changed = False
        for position, fraction in enumerate(CT_K10_CUMULATIVE):
            set_cumulative_factors_from_checkpoint(
                model, entry, (direction,), coefficients, fraction
            )
            if position == 0:
                first_step_changed = not torch.equal(
                    model.layers[0].weight, entry_weight
                )
        self.assertTrue(first_step_changed)
        self.assertTrue(torch.equal(model.layers[0].weight, one_shot_weight))
        self.assertTrue(torch.equal(model(probe), one_shot_output))
        self.assertEqual(model.layers[0].weight.data_ptr(), pointer)
        self.assertEqual(model.layers[0].weight.requires_grad, requires_grad)
        self.assertIsNone(model.layers[0].weight.grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_frozen_endpoint_gate_includes_energy_and_evaluation(self) -> None:
        event = EventReading(
            hard_phi=0.0,
            smooth_phi=0.0,
            context_margins=(1.0,),
            nfe=2,
            target_new_log_likelihoods=(1.0,),
            target_old_log_likelihoods=(0.0,),
            event_mode="toy",
            decision_deficits=(0.0,),
        )
        arguments = {
            "one_shot_hashes": {"layers.0.weight": "a" * 64},
            "frozen_hashes": {"layers.0.weight": "a" * 64},
            "one_shot_event": event,
            "frozen_event": event,
            "one_shot_energy": {0: 1.25},
            "frozen_energy": {0: 1.25},
            "one_shot_evaluation": {"efficacy_token_accuracy": 1.0},
            "frozen_evaluation": {"efficacy_token_accuracy": 1.0},
            "atol": 5e-5,
            "rtol": 5e-3,
        }
        _assert_frozen_endpoint_identity(**arguments)
        changed = dict(arguments)
        changed["frozen_energy"] = {0: 1.2501}
        with self.assertRaisesRegex(RuntimeError, "C-energy"):
            _assert_frozen_endpoint_identity(**changed)
        changed = dict(arguments)
        changed["frozen_evaluation"] = {"efficacy_token_accuracy": 0.0}
        with self.assertRaisesRegex(RuntimeError, "evaluation"):
            _assert_frozen_endpoint_identity(**changed)

    def test_frozen_k10_float64_displacement_is_constant(self) -> None:
        model = _TwoLinear(torch.float64).eval()
        with torch.no_grad():
            model.layers[0].weight.zero_()
        direction = FactorDirection(
            0,
            "layers.0.weight",
            torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64),
            torch.tensor([[0.5], [0.25]], dtype=torch.float64),
        )
        entry = TorchCheckpoint.capture(model, (direction.weight_name,))
        previous = model.layers[0].weight.detach().clone()
        increments = []
        for fraction in CT_K10_CUMULATIVE:
            set_cumulative_factors_from_checkpoint(
                model, entry, (direction,), (0.5,), fraction
            )
            current = model.layers[0].weight.detach().clone()
            increments.append(current - previous)
            previous = current
        for current in increments[1:]:
            torch.testing.assert_close(
                current, increments[0], atol=1e-15, rtol=0.0
            )

    def test_refresh_k10_runs_ten_fields_and_observes_without_freezing(self) -> None:
        metrics = EditInstrumentation("toy-k10-refresh")
        backend = _K10ToyTransportBackend(metrics)
        result = run_ct_k10_arm(
            CTK10Arm.ODE_REFRESH_CT_K10,
            request=self.request,
            backend=backend,
            frozen_target=object(),
            denominators={0: 1.0, 1: 1.0},
            config=self.config,
            instrumentation=metrics,
        )
        snapshot = metrics.finalize().to_dict()
        self.assertEqual(len(result.steps), CT_K10)
        self.assertEqual(result.field_build_count, CT_K10)
        self.assertEqual(backend.build_count, CT_K10)
        self.assertEqual(result.first_hit_step, 1)
        self.assertFalse(snapshot["first_hit"])
        self.assertEqual(snapshot["counters"]["N_field"], CT_K10)
        self.assertEqual(snapshot["counters"]["N_bw"], CT_K10)
        self.assertEqual(snapshot["counters"]["N_trial"], CT_K10)
        self.assertEqual(snapshot["counters"]["N_write"], CT_K10)
        self.assertNotEqual(
            result.steps[0].source_state_id,
            result.steps[0].terminal_state_id,
        )

    def test_frozen_k10_builds_once_but_commits_ten_nonzero_steps(self) -> None:
        metrics = EditInstrumentation("toy-k10-frozen")
        backend = _K10ToyTransportBackend(metrics)
        result = run_ct_k10_arm(
            CTK10Arm.BF_FROZEN_CT_K10,
            request=self.request,
            backend=backend,
            frozen_target=object(),
            denominators={0: 1.0, 1: 1.0},
            config=self.config,
            instrumentation=metrics,
        )
        snapshot = metrics.finalize().to_dict()
        self.assertEqual(len(result.steps), CT_K10)
        self.assertEqual(result.field_build_count, 1)
        self.assertEqual(backend.build_count, 1)
        self.assertEqual(snapshot["counters"]["N_field"], 1)
        self.assertEqual(snapshot["counters"]["N_bw"], 1)
        self.assertEqual(snapshot["counters"]["N_write"], CT_K10)
        self.assertEqual(backend.state, CT_K10)

    def test_k4_dispatch_is_behavior_identical_to_frozen_k4_implementation(self) -> None:
        direct_metrics = EditInstrumentation("toy-k4-direct")
        direct_backend = _ToyTransportBackend()
        direct = run_ct_arm(
            CTArm.ODE_REFRESH_CT_K4,
            request=self.request,
            backend=direct_backend,
            frozen_target=object(),
            denominators={0: 1.0, 1: 1.0},
            config=self.config,
            instrumentation=direct_metrics,
        )
        wrapped_metrics = EditInstrumentation("toy-k4-wrapped")
        wrapped_backend = _ToyTransportBackend()
        wrapped = run_ct_k10_arm(
            CTK10Arm.ODE_REFRESH_CT_K4,
            request=self.request,
            backend=wrapped_backend,
            frozen_target=object(),
            denominators={0: 1.0, 1: 1.0},
            config=self.config,
            instrumentation=wrapped_metrics,
        )
        self.assertEqual(direct.to_dict(), wrapped.to_dict())
        self.assertEqual(
            direct_metrics.finalize().to_dict()["counters"],
            wrapped_metrics.finalize().to_dict()["counters"],
        )

    def test_k10_first_step_no_change_and_exception_fail_closed(self) -> None:
        unchanged = _NoChangeK10Backend()
        with self.assertRaisesRegex(MethodContractError, "first interval"):
            run_ct_k10_arm(
                CTK10Arm.ODE_REFRESH_CT_K10,
                request=self.request,
                backend=unchanged,
                frozen_target=object(),
                denominators={0: 1.0, 1: 1.0},
                config=self.config,
                instrumentation=EditInstrumentation("toy-k10-no-change"),
            )
        self.assertEqual(unchanged.state, 0)

        failing = _FailingK10Backend()
        with self.assertRaisesRegex(RuntimeError, "injected K10"):
            run_ct_k10_arm(
                CTK10Arm.ODE_REFRESH_CT_K10,
                request=self.request,
                backend=failing,
                frozen_target=object(),
                denominators={0: 1.0, 1: 1.0},
                config=self.config,
                instrumentation=EditInstrumentation("toy-k10-fail"),
            )
        self.assertEqual(failing.state, 0)
        self.assertIsNone(failing.trial_state)

    def test_shared_direct_z_requires_one_compute_and_all_five_identities(self) -> None:
        identity = {
            "tensor_sha256": "a" * 64,
            "artifact_sha256": "b" * 64,
            "artifact_size": 123,
            "source_state_id": "c" * 64,
        }
        rows = {arm: dict(identity) for arm in CT_K10_ARM_ORDER}
        assert_shared_direct_z_identities(identity, rows, global_n_z=1)
        counts = {arm: int(index == 0) for index, arm in enumerate(CT_K10_ARM_ORDER)}
        _assert_case_direct_z_accounting(CT_K10_SPEC, identity, rows, counts)
        with self.assertRaisesRegex(MethodContractError, "global N_z"):
            assert_shared_direct_z_identities(identity, rows, global_n_z=0)
        wrong_counts = dict(counts)
        wrong_counts[CTK10Arm.BF_ONESHOT_FULL] = 1
        with self.assertRaisesRegex(RuntimeError, "counter distribution"):
            _assert_case_direct_z_accounting(
                CT_K10_SPEC, identity, rows, wrong_counts
            )
        rows[CTK10Arm.ODE_REFRESH_CT_K10]["tensor_sha256"] = "d" * 64
        with self.assertRaisesRegex(MethodContractError, "ode-refresh-ct-k10"):
            assert_shared_direct_z_identities(identity, rows, global_n_z=1)

    def test_k10_source_has_no_early_stop_or_model_alias_branch(self) -> None:
        source = inspect.getsource(_run_k10_arm)
        self.assertNotIn("mark_first_hit", source)
        self.assertNotIn("early_stop_enabled=True", source)
        self.assertNotIn("model_alias", source)
        self.assertNotIn("open_evaluation", source)

    def test_lock_and_dry_plan_seal_fresh_panel_and_common_arms(self) -> None:
        lock = load_ct_k10_lock()
        self.assertEqual(
            lock["selection"]["p1_case_ids"],
            ["21154", "11042", "16884", "21126"],
        )
        self.assertEqual(lock["policy"]["K10"], 10)
        self.assertEqual(lock["policy"]["K4"], 4)
        self.assertFalse(lock["policy"]["early_stop_arm"])
        self.assertFalse(lock["resources"]["submission_authorized"])
        for stage in ("p0", "p1"):
            plan = dry_plan(lock, stage)
            self.assertFalse(plan["submission_authorized"])
            self.assertEqual(plan["arms"], [arm.value for arm in CT_K10_ARM_ORDER])
            self.assertEqual(plan["pair_gpu"], 2)
            self.assertEqual(plan["server1_project_gpu_cap"], 3)
            self.assertTrue(
                all(
                    f"session03-ct-k10-{stage}-" in job["output_root"]
                    for job in plan["jobs"]
                )
            )

        raw = json.loads(CT_K10_LOCK_PATH.read_text(encoding="utf-8"))
        for mutate in (
            lambda row: row["selection"]["p1_case_ids"].__setitem__(0, "2022"),
            lambda row: row["selection"]["rank_sha256"].__setitem__(0, "f" * 64),
            lambda row: row["selection"]["tokenizer_revisions"].__setitem__(
                "llama3-8b-inst", "f" * 40
            ),
            lambda row: row["policy"].__setitem__("K10", 9),
            lambda row: row["policy"].__setitem__("early_stop_arm", True),
        ):
            changed = json.loads(json.dumps(raw))
            mutate(changed)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "lock.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(MethodContractError):
                    load_ct_k10_lock(path)


if __name__ == "__main__":
    unittest.main()
