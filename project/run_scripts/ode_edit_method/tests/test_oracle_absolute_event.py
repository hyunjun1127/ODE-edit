from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import torch

from project.run_scripts.ode_edit_method.contracts import MethodContractError
from project.run_scripts.ode_edit_method.event_strength import absolute_event_metrics
from project.run_scripts.ode_edit_method.events import ControllerRequest
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.oracle_absolute_event import (
    ORACLE_ABSOLUTE_MEAN_MARGIN_EVENT_MODE,
    OracleAbsoluteMeanMarginTarget,
    calibrate_oracle_absolute_mean_margin_target,
    differentiable_oracle_absolute_mean_margin_event_from_log_likelihoods,
    measure_differentiable_oracle_absolute_mean_margin_event,
    measure_oracle_absolute_mean_margin_event,
    oracle_absolute_mean_margin_event_from_log_likelihoods,
)
from project.run_scripts.ode_edit_method.tests.test_oracle_event import (
    _OracleToyModel,
    _WordTokenizer,
)
from project.run_scripts.ode_edit_motivation.contracts import FileRecord
from project.run_scripts.ode_edit_motivation.direct_z import FrozenDirectZ
from project.run_scripts.ode_edit_motivation.hooks import tensor_sha256


TAU = 0.1
EPSILON = 1e-6


def _target() -> OracleAbsoluteMeanMarginTarget:
    # M_z=16 and L_new,z=-1.  V3 uses M_z only for validity/diagnostics;
    # its relative requirement is exactly zero and L_req=-2.5.
    return OracleAbsoluteMeanMarginTarget.calibrate(
        entry_new=(-4.0, -4.0),
        entry_old=(-3.0, -3.0),
        oracle_new=(-1.0, -1.0),
        oracle_old=(-17.0, -17.0),
        epsilon=EPSILON,
    )


class OracleAbsoluteMeanMarginDecisionTests(unittest.TestCase):
    def test_positive_mean_and_absolute_floor_hit_despite_low_q_margin(self) -> None:
        target = _target()
        reading = oracle_absolute_mean_margin_event_from_log_likelihoods(
            (-1.83, -1.83),
            (-2.83, -2.83),
            target=target,
            tau=TAU,
            nfe=2,
        )
        entry = oracle_absolute_mean_margin_event_from_log_likelihoods(
            (-4.0, -4.0),
            (-3.0, -3.0),
            target=target,
            tau=TAU,
            nfe=2,
        )
        metrics = absolute_event_metrics(
            reading,
            entry,
            tau=TAU,
            denominator_epsilon=1e-12,
            oracle_target=target,
        )
        self.assertTrue(reading.is_hit(EPSILON))
        self.assertGreater(metrics["q_new"], 0.5)
        self.assertLess(metrics["q_margin"], 0.5)
        self.assertEqual(target.required_mean_margin, 0.0)

    def test_absolute_pass_with_negative_mean_margin_fails(self) -> None:
        reading = oracle_absolute_mean_margin_event_from_log_likelihoods(
            (-1.5, -1.5),
            (-1.0, -1.0),
            target=_target(),
            tau=TAU,
            nfe=2,
        )
        self.assertGreater(reading.decision_deficits[0], 0.0)
        self.assertFalse(reading.is_hit(EPSILON))

    def test_positive_mean_with_insufficient_new_likelihood_fails(self) -> None:
        reading = oracle_absolute_mean_margin_event_from_log_likelihoods(
            (-3.0, -3.0),
            (-4.0, -4.0),
            target=_target(),
            tau=TAU,
            nfe=2,
        )
        self.assertLessEqual(reading.decision_deficits[0], 0.0)
        self.assertGreater(reading.decision_deficits[1], 0.0)
        self.assertFalse(reading.is_hit(EPSILON))

    def test_extreme_negative_context_cannot_veto_passing_uniform_means(self) -> None:
        target = _target()
        left = oracle_absolute_mean_margin_event_from_log_likelihoods(
            (-0.5, -2.5),
            (-4.0, 0.0),
            target=target,
            tau=TAU,
            nfe=2,
        )
        right = oracle_absolute_mean_margin_event_from_log_likelihoods(
            (-2.5, -0.5),
            (0.0, -4.0),
            target=target,
            tau=TAU,
            nfe=2,
        )
        self.assertTrue(left.is_hit(EPSILON))
        self.assertLess(min(left.context_margins), 0.0)
        self.assertEqual(left.decision_deficits, right.decision_deficits)
        self.assertEqual(left.hard_phi, right.hard_phi)
        self.assertEqual(left.smooth_phi, right.smooth_phi)

    def test_lowering_old_only_cannot_satisfy_absolute_floor(self) -> None:
        reading = oracle_absolute_mean_margin_event_from_log_likelihoods(
            (-4.0, -4.0),
            (-10.0, -10.0),
            target=_target(),
            tau=TAU,
            nfe=2,
        )
        self.assertLessEqual(reading.decision_deficits[0], 0.0)
        self.assertGreater(reading.decision_deficits[1], 0.0)
        self.assertFalse(reading.is_hit(EPSILON))

    def test_differentiable_and_nondifferentiable_readings_are_identical(self) -> None:
        target = _target()
        new = torch.tensor([-0.5, -2.5], dtype=torch.float64, requires_grad=True)
        old = torch.tensor([-4.0, 0.0], dtype=torch.float64, requires_grad=True)
        event = differentiable_oracle_absolute_mean_margin_event_from_log_likelihoods(
            new,
            old,
            target=target,
            tau=TAU,
        )
        expected = oracle_absolute_mean_margin_event_from_log_likelihoods(
            tuple(float(value) for value in new.detach()),
            tuple(float(value) for value in old.detach()),
            target=target,
            tau=TAU,
            nfe=2,
        )
        self.assertEqual(event.reading, expected)
        event.smooth_phi.backward()
        self.assertTrue(torch.isfinite(new.grad).all())
        self.assertTrue(torch.isfinite(old.grad).all())

    def test_primary_decision_source_has_no_shadow_threshold(self) -> None:
        source = inspect.getsource(
            oracle_absolute_mean_margin_event_from_log_likelihoods
        )
        for forbidden in (
            "q_margin",
            "oracle_mean_margin",
            "min(context",
            "min(margin",
            "legacy",
            "model_alias",
            "fallback",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("deficits = (-mean_margin", source)


class OracleAbsoluteCalibrationAccountingTests(unittest.TestCase):
    def test_ordinary_event_paths_each_make_two_actual_model_calls(self) -> None:
        request = ControllerRequest(
            case_id="v3-ordinary",
            prompt="ctx {}",
            subject="S",
            target_new="N",
            target_old="O",
        )
        target = OracleAbsoluteMeanMarginTarget.calibrate(
            entry_new=(-4.0,),
            entry_old=(-3.0,),
            oracle_new=(-1.0,),
            oracle_old=(-17.0,),
            epsilon=EPSILON,
        )
        for differentiable in (False, True):
            with self.subTest(differentiable=differentiable):
                model = _OracleToyModel().eval()
                instrumentation = EditInstrumentation("v3-ordinary-event")
                instrumentation.attach_model(model)
                with instrumentation.model_forward_scope("event"):
                    if differentiable:
                        event = measure_differentiable_oracle_absolute_mean_margin_event(
                            model,
                            _WordTokenizer(),
                            request,
                            (("{}",),),
                            target=target,
                            tau=TAU,
                        )
                        reading = event.reading
                    else:
                        reading = measure_oracle_absolute_mean_margin_event(
                            model,
                            _WordTokenizer(),
                            request,
                            (("{}",),),
                            target=target,
                            tau=TAU,
                        )
                instrumentation.detach_model()
                counters = dict(instrumentation.finalize().counters)
                self.assertEqual(reading.nfe, 2)
                self.assertEqual(counters["N_model_fwd"], 2)
                self.assertEqual(counters["N_event_fwd"], 2)
                self.assertEqual(counters["N_bw"], 0)

    def test_total_four_forwards_oracle_increment_two_and_zero_backward(self) -> None:
        model = _OracleToyModel().eval()
        tokenizer = _WordTokenizer()
        request = ControllerRequest(
            case_id="v3-oracle-toy",
            prompt="ctx {}",
            subject="S",
            target_new="N",
            target_old="O",
        )
        values = torch.tensor([[4.0], [0.0]], dtype=torch.float32)
        direct_z = FrozenDirectZ(
            values=values,
            source_snapshot_id="snapshot",
            source_state_id="state",
            model_id="toy",
            context_id="context",
            request_ids=(request.case_id,),
            z_layer=0,
            tensor_sha256=tensor_sha256(values),
            artifact=FileRecord(
                path="/local/toy-v3-direct-z", sha256="0" * 64, size=0
            ),
        )
        bindings = SimpleNamespace(
            compute_z=SimpleNamespace(
                find_fact_lookup_idx=lambda *_args, **_kwargs: -1
            )
        )
        hparams = SimpleNamespace(
            layers=(0,), layer_module_tmp="layers.{}", fact_token="last"
        )
        instrumentation = EditInstrumentation("v3-oracle-calibration")
        instrumentation.attach_model(model)
        calibration = calibrate_oracle_absolute_mean_margin_target(
            model,
            tokenizer,
            request,
            (("{}",),),
            direct_z=direct_z,
            bindings=bindings,
            hparams=hparams,
            tau=TAU,
            epsilon=EPSILON,
            entry_forward_scope=instrumentation.model_forward_scope("event"),
            oracle_forward_scope=instrumentation.model_forward_scope(None),
        )
        instrumentation.detach_model()
        counters = dict(instrumentation.finalize().counters)
        self.assertEqual(calibration.entry_forward_count, 2)
        self.assertEqual(calibration.oracle_forward_count, 2)
        self.assertEqual(
            calibration.entry_forward_count + calibration.oracle_forward_count,
            4,
        )
        self.assertEqual(counters["N_model_fwd"], 4)
        self.assertEqual(counters["N_event_fwd"], 2)
        self.assertEqual(counters["N_bw"], 0)
        self.assertEqual(calibration.target.required_mean_margin, 0.0)
        self.assertEqual(
            calibration.target.to_dict()["event_mode"],
            ORACLE_ABSOLUTE_MEAN_MARGIN_EVENT_MODE,
        )
        source = inspect.getsource(calibrate_oracle_absolute_mean_margin_target)
        self.assertNotIn("calibrate_oracle_mean_target", source)

    def test_oracle_validity_still_requires_positive_margin_and_new_gain(self) -> None:
        with self.assertRaisesRegex(MethodContractError, "mean margin"):
            OracleAbsoluteMeanMarginTarget.calibrate(
                entry_new=(-4.0,),
                entry_old=(-3.0,),
                oracle_new=(-1.0,),
                oracle_old=(-0.5,),
                epsilon=EPSILON,
            )
        with self.assertRaisesRegex(MethodContractError, "insufficient"):
            OracleAbsoluteMeanMarginTarget.calibrate(
                entry_new=(-1.0,),
                entry_old=(-2.0,),
                oracle_new=(-1.0,),
                oracle_old=(-2.0,),
                epsilon=EPSILON,
            )


if __name__ == "__main__":
    unittest.main()
