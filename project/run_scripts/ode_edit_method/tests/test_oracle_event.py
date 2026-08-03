from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import torch

from project.run_scripts.ode_edit_method.contracts import (
    Arm,
    MethodContractError,
)
from project.run_scripts.ode_edit_method.event_strength import absolute_event_metrics
from project.run_scripts.ode_edit_method.oracle_event import (
    ORACLE_MEAN_EVENT_MODE,
    ORACLE_REALIZATION_FRACTION,
    OracleActivationHook,
    OracleMeanEventTarget,
    differentiable_oracle_mean_event_from_log_likelihoods,
    oracle_mean_event_from_log_likelihoods,
)
from project.run_scripts.ode_edit_method.runtime import FiveArmRunner
from project.run_scripts.ode_edit_method.tests.test_runtime_invariants import (
    _ToyBackend,
    _request,
    _test_config,
)
from project.run_scripts.ode_edit_method.controller import OmegaLedger
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.events import ControllerRequest
from project.run_scripts.ode_edit_method.oracle_event import calibrate_oracle_mean_target
from project.run_scripts.ode_edit_motivation.contracts import FileRecord
from project.run_scripts.ode_edit_motivation.direct_z import FrozenDirectZ
from project.run_scripts.ode_edit_motivation.hooks import tensor_sha256


TAU = 0.1
EPSILON = 1e-6


def _target() -> OracleMeanEventTarget:
    # Oracle means: L_new,z=-1.0 and M_z=0.4.  Therefore rho=.5 locks
    # L_req=-2.5 and M_req=.2 from an entry L_new=-4.0.
    return OracleMeanEventTarget.calibrate(
        entry_new=(-4.0, -4.0),
        entry_old=(-3.0, -3.0),
        oracle_new=(-1.0, -1.0),
        oracle_old=(-1.2, -1.6),
        epsilon=EPSILON,
    )


class OracleMeanDecisionTests(unittest.TestCase):
    def test_uniform_aggregate_is_permutation_invariant(self) -> None:
        target = _target()
        left = oracle_mean_event_from_log_likelihoods(
            (-0.5, -3.0),
            (-2.5, -2.0),
            target=target,
            tau=TAU,
            nfe=2,
        )
        right = oracle_mean_event_from_log_likelihoods(
            (-3.0, -0.5),
            (-2.0, -2.5),
            target=target,
            tau=TAU,
            nfe=2,
        )
        self.assertEqual(left.event_mode, ORACLE_MEAN_EVENT_MODE)
        self.assertEqual(left.decision_deficits, right.decision_deficits)
        self.assertEqual(left.hard_phi, right.hard_phi)
        self.assertEqual(left.smooth_phi, right.smooth_phi)
        self.assertTrue(left.is_hit(EPSILON))
        # The second context is negative, but cannot veto two passing means.
        self.assertLess(min(left.context_margins), 0.0)

    def test_lowering_old_alone_cannot_satisfy_absolute_strength(self) -> None:
        target = _target()
        reading = oracle_mean_event_from_log_likelihoods(
            (-4.0, -4.0),
            (-10.0, -10.0),
            target=target,
            tau=TAU,
            nfe=2,
        )
        self.assertLessEqual(reading.decision_deficits[0], 0.0)
        self.assertGreater(reading.decision_deficits[1], 0.0)
        self.assertFalse(reading.is_hit(EPSILON))

    def test_oracle_validity_is_fail_closed_and_common(self) -> None:
        self.assertEqual(ORACLE_REALIZATION_FRACTION, 0.5)
        with self.assertRaisesRegex(MethodContractError, "mean margin"):
            OracleMeanEventTarget.calibrate(
                entry_new=(-4.0, -4.0),
                entry_old=(-3.0, -3.0),
                oracle_new=(-2.0, -2.0),
                oracle_old=(-1.0, -1.0),
                epsilon=EPSILON,
            )
        with self.assertRaisesRegex(MethodContractError, "insufficient"):
            OracleMeanEventTarget.calibrate(
                entry_new=(-1.0, -1.0),
                entry_old=(-2.0, -2.0),
                oracle_new=(-1.0, -1.0),
                oracle_old=(-2.0, -2.0),
                epsilon=EPSILON,
            )

    def test_differentiable_event_uses_same_two_aggregate_deficits(self) -> None:
        target = _target()
        new = torch.tensor([-0.5, -3.0], dtype=torch.float64, requires_grad=True)
        old = torch.tensor([-2.5, -2.0], dtype=torch.float64, requires_grad=True)
        event = differentiable_oracle_mean_event_from_log_likelihoods(
            new,
            old,
            target=target,
            tau=TAU,
        )
        event.smooth_phi.backward()
        self.assertEqual(event.reading.nfe, 2)
        self.assertIsNotNone(new.grad)
        self.assertIsNotNone(old.grad)
        self.assertTrue(torch.isfinite(new.grad).all())
        self.assertTrue(torch.isfinite(old.grad).all())

    def test_observability_recomputes_q_and_legacy_shadow(self) -> None:
        target = _target()
        entry = oracle_mean_event_from_log_likelihoods(
            (-4.0, -4.0),
            (-3.0, -3.0),
            target=target,
            tau=TAU,
            nfe=2,
        )
        endpoint = oracle_mean_event_from_log_likelihoods(
            (-0.5, -3.0),
            (-2.5, -2.0),
            target=target,
            tau=TAU,
            nfe=2,
        )
        metrics = absolute_event_metrics(
            endpoint,
            entry,
            tau=TAU,
            denominator_epsilon=1e-12,
            oracle_target=target,
        )
        self.assertAlmostEqual(metrics["q_margin"], 1.25)
        self.assertAlmostEqual(metrics["q_new"], 0.75)
        self.assertGreater(metrics["legacy_hard_phi"], 0.0)
        self.assertLessEqual(metrics["decision_hard_phi"], EPSILON)

    def test_primary_source_has_no_per_context_decision_or_alias_branch(self) -> None:
        source = inspect.getsource(oracle_mean_event_from_log_likelihoods)
        for forbidden in (
            "min(context",
            "min(margin",
            "model_alias",
            "llama3",
            "qwen2",
            "fallback",
            "return event_from_log_likelihoods(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("max(deficits)", source)


class _LayerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.guard = torch.nn.Parameter(torch.tensor([0.375], dtype=torch.float32))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers[0](hidden)


class OracleActivationHookTests(unittest.TestCase):
    def test_bf16_clone_add_matches_locked_output_layout_and_cleans_up(self) -> None:
        model = _LayerModel()
        hidden = torch.tensor(
            [
                [[0.1, -0.3], [0.2, 0.4], [0.5, -0.7]],
                [[-0.2, 0.6], [0.7, -0.4], [0.8, 0.9]],
            ],
            dtype=torch.bfloat16,
        )
        delta = torch.tensor([0.37, -0.19], dtype=torch.float32)
        parameter = model.guard
        guard = (
            parameter.data_ptr(),
            parameter._version,
            parameter.grad,
            parameter.requires_grad,
            torch.get_rng_state().clone(),
        )
        expected = hidden.clone()
        rows = torch.arange(2)
        columns = torch.tensor([1, 2])
        expected[rows, columns, :] = (
            expected[rows, columns, :] + delta.to(torch.bfloat16)
        )
        with OracleActivationHook(model, "layers.0", (1, -1), delta) as hook:
            actual = model(hidden)
        self.assertEqual(hook.calls, 1)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(parameter.data_ptr(), guard[0])
        self.assertEqual(parameter._version, guard[1])
        self.assertIs(parameter.grad, guard[2])
        self.assertEqual(parameter.requires_grad, guard[3])
        self.assertTrue(torch.equal(torch.get_rng_state(), guard[4]))
        self.assertTrue(torch.equal(model(hidden), hidden))


class _WordTokenizer:
    padding_side = "right"
    pad_token_id = 0
    bos_token_id = 1
    unk_token_id = 99

    def __init__(self) -> None:
        self.ids = {"ctx": 2, "S": 3, "N": 4, "O": 5}

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        values = [self.ids[word] for word in text.strip().split()]
        return ([self.bos_token_id] if add_special_tokens else []) + values


class _OracleToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(100, 2)
        self.layers = torch.nn.ModuleList([torch.nn.Identity()])
        self.lm_head = torch.nn.Linear(2, 100, bias=False)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.lm_head.weight.zero_()
            self.lm_head.weight[4, 0] = 1.0
            self.lm_head.weight[5, 0] = -1.0

    def forward(
        self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        del attention_mask
        hidden = self.layers[0](self.embedding(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


class OracleCalibrationCounterTests(unittest.TestCase):
    def test_entry_two_plus_oracle_two_forwards_and_no_backward(self) -> None:
        model = _OracleToyModel().eval()
        tokenizer = _WordTokenizer()
        request = ControllerRequest(
            case_id="oracle-toy",
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
            artifact=FileRecord(path="/local/toy-direct-z", sha256="0" * 64, size=0),
        )
        bindings = SimpleNamespace(
            compute_z=SimpleNamespace(
                find_fact_lookup_idx=lambda *_args, **_kwargs: -1
            )
        )
        hparams = SimpleNamespace(
            layers=(0,),
            layer_module_tmp="layers.{}",
            fact_token="last",
        )
        instrumentation = EditInstrumentation("oracle-calibration")
        instrumentation.attach_model(model)
        calibration = calibrate_oracle_mean_target(
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
        self.assertEqual(calibration.hook_call_count, 2)
        self.assertEqual(counters["N_model_fwd"], 4)
        self.assertEqual(counters["N_event_fwd"], 2)
        self.assertEqual(counters["N_bw"], 0)
        self.assertGreater(calibration.target.oracle_mean_margin, 0.0)
        self.assertGreater(
            calibration.target.oracle_mean_new,
            calibration.target.entry_mean_new + EPSILON,
        )

    def test_materialized_parameter_grad_fails_before_oracle_forward(self) -> None:
        model = _OracleToyModel().eval()
        model.embedding.weight.grad = torch.zeros_like(model.embedding.weight)
        tokenizer = _WordTokenizer()
        request = ControllerRequest(
            case_id="oracle-grad",
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
            artifact=FileRecord(path="/local/toy-direct-z", sha256="0" * 64, size=0),
        )
        bindings = SimpleNamespace(
            compute_z=SimpleNamespace(
                find_fact_lookup_idx=lambda *_args, **_kwargs: -1
            )
        )
        hparams = SimpleNamespace(
            layers=(0,), layer_module_tmp="layers.{}", fact_token="last"
        )
        with self.assertRaisesRegex(MethodContractError, "grad"):
            calibrate_oracle_mean_target(
                model,
                tokenizer,
                request,
                (("{}",),),
                direct_z=direct_z,
                bindings=bindings,
                hparams=hparams,
                tau=TAU,
                epsilon=EPSILON,
            )

    def test_exception_removes_hook_without_mutation(self) -> None:
        model = _LayerModel()
        hidden = torch.zeros((1, 2, 2), dtype=torch.bfloat16)
        with self.assertRaisesRegex(MethodContractError, "layout differs"):
            with OracleActivationHook(
                model,
                "layers.0",
                (0, 1),
                torch.ones(2),
            ):
                model(hidden)
        self.assertTrue(torch.equal(model(hidden), hidden))


class EagerDirectZRuntimeTests(unittest.TestCase):
    class _EagerBackend(_ToyBackend):
        def __init__(self) -> None:
            super().__init__(goal=0.0)
            self.prepared = False

        def prepare_event_target(self, frozen_target: object) -> None:
            if self.direct_z_calls != 1 or frozen_target != ("frozen-z", _request()):
                raise MethodContractError("eager target was not direct-z once")
            self.prepared = True

        def event(self, request):  # type: ignore[no-untyped-def]
            if not self.prepared:
                raise MethodContractError("entry event preceded oracle target")
            return super().event(request)

    def test_entry_hit_still_computes_direct_z_once_for_oracle_backend(self) -> None:
        for arm in (Arm.NATIVE_MEMIT, Arm.STATIC_SYNCHRONOUS, Arm.FULL_ODE_EDIT):
            with self.subTest(arm=arm.value):
                backend = self._EagerBackend()
                result = FiveArmRunner(
                    _test_config(),
                    OmegaLedger({0: 1.0, 1: 1.0}),
                ).run(
                    arm,
                    edit_id=f"oracle-{arm.value}",
                    request=_request(),
                    backend=backend,
                )
                self.assertEqual(result.direct_z_compute_count, 1)
                self.assertEqual(backend.direct_z_calls, 1)
                self.assertTrue(backend.prepared)


if __name__ == "__main__":
    unittest.main()
