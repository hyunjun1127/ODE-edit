from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from project.run_scripts.ode_edit_method.contracts import (
    ControllerConfig,
    MethodContractError,
)
from project.run_scripts.ode_edit_method import events as event_module
from project.run_scripts.ode_edit_method.derivatives import (
    ActuatorDirectionalHook,
    ScalarGateDirectionalReference,
    all_layer_directional_derivatives,
    assert_scalar_gate_matches_hook,
    directional_gradient_scope,
)
from project.run_scripts.ode_edit_method.events import (
    ControllerRequest,
    EVENT_BACKEND_MODE,
    EVENT_MODEL_FORWARD_CALLS,
    InformationFirewall,
    event_from_log_likelihoods,
    build_allowed_contexts,
    build_teacher_batch,
    measure_differentiable_event,
    measure_event,
    normalize_object_text,
    score_combined_teacher_batches,
    score_teacher_batch,
    context_manifest_payload,
)
from project.run_scripts.ode_edit_method.hooks import (
    FactorDirection,
    TorchCheckpoint,
    TorchFactorTrial,
    terminal_net_c_energy,
)
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation


class InformationFirewallTests(unittest.TestCase):
    def test_counterfact_projection_drops_evaluation_fields(self) -> None:
        row = {
            "case_id": 17,
            "requested_rewrite": {
                "prompt": "{} lives in",
                "subject": "Ada",
                "target_new": {"str": "Paris"},
                "target_true": {"str": "London"},
            },
            "paraphrase_prompts": ["evaluation secret"],
            "neighborhood_prompts": ["held-out secret"],
            "generation_prompts": ["generation secret"],
        }
        request = ControllerRequest.from_counterfact_row(row)
        self.assertEqual(request.case_id, "17")
        self.assertFalse(hasattr(request, "paraphrase_prompts"))
        self.assertNotIn("secret", repr(request))

        firewall = InformationFirewall(
            request,
            evaluation_payload={"paraphrases": row["paraphrase_prompts"]},
        )
        with self.assertRaises(MethodContractError):
            firewall.open_evaluation()
        action_hash = firewall.freeze_action({"alpha": 0.5})
        self.assertEqual(len(action_hash), 64)
        self.assertEqual(
            firewall.open_evaluation(),
            {"paraphrases": ["evaluation secret"]},
        )

    def test_event_uses_length_normalized_log_likelihoods(self) -> None:
        reading = event_from_log_likelihoods(
            target_new=(-1.0, -3.0),
            target_old=(-2.0, -2.5),
            tau=0.2,
        )
        self.assertEqual(reading.context_margins, (1.0, -0.5))
        self.assertEqual(reading.target_new_log_likelihoods, (-1.0, -3.0))
        self.assertEqual(reading.target_old_log_likelihoods, (-2.0, -2.5))
        self.assertAlmostEqual(reading.hard_phi, 0.5)
        self.assertEqual(normalize_object_text("Paris"), " Paris")
        self.assertEqual(normalize_object_text(" Paris"), " Paris")

    def test_controller_schema_has_no_model_specific_branch(self) -> None:
        fields = set(ControllerConfig.__dataclass_fields__)
        for forbidden in (
            "model",
            "model_alias",
            "llama",
            "qwen",
            "fallback",
            "sign_rule",
        ):
            self.assertNotIn(forbidden, fields)

    def test_two_forward_event_is_exact_and_combined_primitive_is_not_primary(self) -> None:
        class _Tokenizer:
            padding_side = "right"
            pad_token_id = 0
            bos_token_id = 1
            unk_token_id = 2
            name_or_path = "cpu-fixture"

            def __init__(self) -> None:
                self._ids: dict[str, int] = {}

            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                values = []
                if add_special_tokens:
                    values.append(self.bos_token_id)
                for token in text.split():
                    if token not in self._ids:
                        self._ids[token] = len(self._ids) + 3
                    values.append(self._ids[token])
                return values

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(64, 7)
                self.output = torch.nn.Linear(7, 64, bias=False)
                self.calls = 0

            def forward(self, *, input_ids, attention_mask):
                del attention_mask
                self.calls += 1
                return SimpleNamespace(logits=self.output(self.embedding(input_ids)))

        torch.manual_seed(23)
        tokenizer = _Tokenizer()
        model = _Model().double()
        request = ControllerRequest(
            case_id="batch-fixture",
            prompt="{} works at",
            subject="Ada",
            target_new="New York City",
            target_old="London",
        )
        templates = (("{}",), ("Because {}", "Today {}"))
        contexts = build_allowed_contexts(request, templates)
        new_batch = build_teacher_batch(tokenizer, contexts, request.target_new)
        old_batch = build_teacher_batch(tokenizer, contexts, request.target_old)
        new_reference = score_teacher_batch(model, new_batch)
        old_reference = score_teacher_batch(model, old_batch)
        self.assertEqual(model.calls, 2)
        combined_new, combined_old = score_combined_teacher_batches(
            model,
            (new_batch, old_batch),
            differentiable=False,
        )
        self.assertEqual(model.calls, 3)
        self.assertTrue(
            torch.allclose(
                combined_new,
                torch.tensor(new_reference, dtype=combined_new.dtype),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        self.assertTrue(
            torch.allclose(
                combined_old,
                torch.tensor(old_reference, dtype=combined_old.dtype),
                atol=1e-12,
                rtol=1e-12,
            )
        )
        expected = event_from_log_likelihoods(
            new_reference,
            old_reference,
            tau=0.1,
            nfe=EVENT_MODEL_FORWARD_CALLS,
        )
        with mock.patch.object(
            event_module,
            "score_combined_teacher_batches",
            side_effect=AssertionError("combined scorer entered selected runtime"),
        ):
            model.calls = 0
            metrics = EditInstrumentation("two-forward-event")
            metrics.attach_model(model)
            with metrics.model_forward_scope("event"):
                measured = measure_event(
                    model, tokenizer, request, templates, tau=0.1
                )
            metrics.detach_model()
            self.assertEqual(measured, expected)
            self.assertEqual(
                measured.target_new_log_likelihoods,
                tuple(float(value) for value in new_reference),
            )
            self.assertEqual(
                measured.target_old_log_likelihoods,
                tuple(float(value) for value in old_reference),
            )
            self.assertEqual(measured.nfe, EVENT_MODEL_FORWARD_CALLS)
            self.assertEqual(model.calls, 2)
            counters = metrics.finalize().to_dict()["counters"]
            self.assertEqual(counters["N_model_fwd"], 2)
            self.assertEqual(counters["N_event_fwd"], 2)

            model.calls = 0
            differentiable_metrics = EditInstrumentation("two-forward-field")
            differentiable_metrics.attach_model(model)
            with differentiable_metrics.model_forward_scope("field"):
                differentiable = measure_differentiable_event(
                    model, tokenizer, request, templates, tau=0.1
                )
            differentiable_metrics.detach_model()
        self.assertEqual(
            differentiable.reading.nfe,
            EVENT_MODEL_FORWARD_CALLS,
        )
        self.assertTrue(differentiable.smooth_phi.requires_grad)
        self.assertEqual(differentiable.reading, expected)
        self.assertEqual(model.calls, 2)
        differentiable_counters = differentiable_metrics.finalize().to_dict()[
            "counters"
        ]
        self.assertEqual(differentiable_counters["N_model_fwd"], 2)
        self.assertEqual(differentiable_counters["N_field_state_fwd"], 2)
        self.assertEqual(EVENT_BACKEND_MODE, "two-separate-teacher-forced-forwards")
        selected_source = inspect.getsource(measure_event) + inspect.getsource(
            measure_differentiable_event
        )
        self.assertNotIn("score_combined_teacher_batches", selected_source)
        self.assertNotIn("model_alias", selected_source)
        self.assertNotIn("llama3-8b-inst", selected_source)
        self.assertNotIn("qwen2.5-7b-inst", selected_source)
        manifest = context_manifest_payload(request, templates, tokenizer)
        self.assertEqual(
            manifest["target_suffix_identity"]["new"]["token_count"], 3
        )
        self.assertNotIn(request.target_new, repr(manifest))

    def test_two_forward_event_records_aggregate_into_one_all_layer_backward(self) -> None:
        class _Tokenizer:
            padding_side = "right"
            pad_token_id = 0
            bos_token_id = 1
            unk_token_id = 2
            name_or_path = "two-forward-hook-fixture"

            def __init__(self) -> None:
                self._ids: dict[str, int] = {}

            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                values = [self.bos_token_id] if add_special_tokens else []
                for token in text.split():
                    if token not in self._ids:
                        self._ids[token] = len(self._ids) + 3
                    values.append(self._ids[token])
                return values

        class _Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 4)
                self.first = torch.nn.Linear(4, 5, bias=True)
                self.second = torch.nn.Linear(5, 32, bias=False)

            def forward(self, *, input_ids, attention_mask):
                del attention_mask
                hidden = torch.tanh(self.first(self.embedding(input_ids)))
                return SimpleNamespace(logits=self.second(hidden))

        torch.manual_seed(41)
        tokenizer = _Tokenizer()
        model = _Model().double()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        generator = torch.Generator().manual_seed(43)
        directions = (
            FactorDirection(
                layer=0,
                weight_name="first.weight",
                left=torch.randn(5, 2, generator=generator, dtype=torch.float64),
                right=torch.randn(4, 2, generator=generator, dtype=torch.float64),
            ),
            FactorDirection(
                layer=1,
                weight_name="second.weight",
                left=torch.randn(32, 2, generator=generator, dtype=torch.float64),
                right=torch.randn(5, 2, generator=generator, dtype=torch.float64),
            ),
        )
        request = ControllerRequest(
            case_id="two-forward-hook",
            prompt="{} works at",
            subject="Ada",
            target_new="New York City",
            target_old="London",
        )
        templates = (("{}",), ("Because {}", "Today {}"))
        targets = tuple(
            dict(model.named_parameters())[direction.weight_name]
            for direction in directions
        )
        pointers = tuple(parameter.data_ptr() for parameter in targets)
        versions = tuple(parameter._version for parameter in targets)
        rng_state = torch.get_rng_state().clone()

        with directional_gradient_scope(model, directions):
            dense_event = measure_differentiable_event(
                model, tokenizer, request, templates, tau=0.1
            )
            dense = all_layer_directional_derivatives(
                dense_event.smooth_phi,
                model,
                directions,
            )

        original_grad = torch.autograd.grad
        primary_metrics = EditInstrumentation("two-forward-hook-primary")
        primary_metrics.attach_model(model)
        with ActuatorDirectionalHook(
            model,
            directions,
            instrumentation=primary_metrics,
        ) as hook:
            with primary_metrics.model_forward_scope("field"):
                primary_event = measure_differentiable_event(
                    model, tokenizer, request, templates, tau=0.1
                )
            self.assertTrue(
                all(len(records) == 2 for records in hook._records.values())
            )
            with mock.patch("torch.autograd.grad", wraps=original_grad) as grad_mock:
                primary = hook.compute(primary_event.smooth_phi)
                self.assertEqual(grad_mock.call_count, 1)
        primary_metrics.detach_model()

        scalar_metrics = EditInstrumentation("two-forward-hook-scalar")
        scalar_metrics.attach_model(model)
        with ScalarGateDirectionalReference(
            model,
            directions,
            instrumentation=scalar_metrics,
        ) as reference:
            with scalar_metrics.model_forward_scope("reference_gate"):
                scalar_event = measure_differentiable_event(
                    model, tokenizer, request, templates, tau=0.1
                )
            with mock.patch("torch.autograd.grad", wraps=original_grad) as grad_mock:
                scalar = reference.compute(scalar_event.smooth_phi)
                self.assertEqual(grad_mock.call_count, 1)
        scalar_metrics.detach_model()

        assert_scalar_gate_matches_hook(
            primary,
            scalar,
            abs_tol=5e-5,
            rel_tol=5e-3,
        )
        for dense_value, primary_value in zip(
            dense.values, primary.values, strict=True
        ):
            self.assertAlmostEqual(
                dense_value.event_derivative,
                primary_value.event_derivative,
                delta=1e-10,
            )
        primary_counters = primary_metrics.finalize().to_dict()["counters"]
        self.assertEqual(primary_counters["N_model_fwd"], 2)
        self.assertEqual(primary_counters["N_field_state_fwd"], 2)
        self.assertEqual(primary_counters["N_bw"], 1)
        scalar_counters = scalar_metrics.finalize().to_dict()["counters"]
        self.assertEqual(scalar_counters["N_model_fwd"], 2)
        self.assertEqual(scalar_counters["N_reference_gate_fwd"], 2)
        self.assertEqual(scalar_counters["N_reference_gate_bw"], 1)
        self.assertEqual(scalar_counters["N_bw"], 0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_state))
        for parameter, pointer, version in zip(
            targets, pointers, versions, strict=True
        ):
            self.assertIsNone(parameter.grad)
            self.assertEqual(parameter.data_ptr(), pointer)
            self.assertEqual(parameter._version, version)
            self.assertFalse(parameter.requires_grad)


class TrialRollbackTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)).double()
        self.direction = FactorDirection(
            layer=0,
            weight_name="0.weight",
            left=torch.tensor([[0.2], [-0.4]], dtype=torch.float64),
            right=torch.tensor([[0.5], [0.3], [-0.1]], dtype=torch.float64),
        )

    def test_reject_and_exception_restore_weight_and_rng_exactly(self) -> None:
        checkpoint = TorchCheckpoint.capture(self.model, ("0.weight",))
        with TorchFactorTrial(self.model, (self.direction,), (0.7,)):
            _ = torch.rand(4)
        checkpoint.assert_exact(self.model, include_rng=True)

        with self.assertRaisesRegex(RuntimeError, "failure"):
            with TorchFactorTrial(self.model, (self.direction,), (0.7,)):
                _ = torch.rand(4)
                raise RuntimeError("failure")
        checkpoint.assert_exact(self.model, include_rng=True)

    def test_terminal_net_energy_is_subdivision_invariant(self) -> None:
        entry = TorchCheckpoint.capture(self.model, ("0.weight",))
        covariance = {0: torch.eye(3, dtype=torch.float64)}
        names = {0: "0.weight"}
        with TorchFactorTrial(self.model, (self.direction,), (0.8,)) as trial:
            trial.commit()
        one_step = terminal_net_c_energy(self.model, entry, covariance, names)
        entry.restore(self.model)

        for _ in range(4):
            with TorchFactorTrial(self.model, (self.direction,), (0.2,)) as trial:
                trial.commit()
        four_steps = terminal_net_c_energy(self.model, entry, covariance, names)
        self.assertAlmostEqual(one_step[0], four_steps[0], places=13)


if __name__ == "__main__":
    unittest.main()
