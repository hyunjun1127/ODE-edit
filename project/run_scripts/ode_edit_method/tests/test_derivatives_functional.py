from __future__ import annotations

import copy
import inspect
import random
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_edit_method.contracts import MethodContractError
from project.run_scripts.ode_edit_method.derivatives import (
    ActuatorDirectionalHook,
    ScalarGateDirectionalReference,
    all_layer_directional_derivatives,
    assert_scalar_gate_matches_hook,
    directional_gradient_scope,
)
from project.run_scripts.ode_edit_method.functional_trial import (
    LowRankFunctionalTrial,
    QuantizedRowBlockFunctionalTrial,
)
from project.run_scripts.ode_edit_method.hooks import FactorDirection
from project.run_scripts.ode_edit_method.hooks import apply_accepted_factors
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation


class TinyNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(3, 4, bias=True)
        self.second = torch.nn.Linear(4, 2, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.tanh(self.first(value)))


class TinyBfloatLinear(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


class BfloatNonlinearStressNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(5, 7, bias=True)
        self.second = torch.nn.Linear(7, 3, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.nn.functional.silu(self.first(value)))


def _directions() -> tuple[FactorDirection, ...]:
    return (
        FactorDirection(
            layer=0,
            weight_name="first.weight",
            left=torch.tensor(
                [[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5], [0.1, 0.2]],
                dtype=torch.float64,
            ),
            right=torch.tensor(
                [[0.4, 0.3], [-0.5, 0.2], [0.1, -0.2]], dtype=torch.float64
            ),
        ),
        FactorDirection(
            layer=1,
            weight_name="second.weight",
            left=torch.tensor([[0.2], [-0.3]], dtype=torch.float64),
            right=torch.tensor([[0.5], [0.1], [-0.2], [0.4]], dtype=torch.float64),
        ),
    )


def _loss(model: TinyNetwork, inputs: torch.Tensor) -> torch.Tensor:
    output = model(inputs)
    return (output.square().mean() + 0.3 * output.mean())


def _combined_stress_event(
    model: BfloatNonlinearStressNetwork,
    inputs: torch.Tensor,
) -> torch.Tensor:
    logits = _combined_stress_output(model, inputs).float()
    targets = torch.tensor(
        [0, 2, 1, 2, 0, 1],
        device=logits.device,
        dtype=torch.long,
    )
    selected = torch.log_softmax(logits, dim=-1)[
        torch.arange(logits.shape[0], device=logits.device), targets
    ]
    return -selected.mean() + 0.137 * torch.logsumexp(logits, dim=-1).mean()


def _combined_stress_output(
    model: BfloatNonlinearStressNetwork,
    inputs: torch.Tensor,
) -> torch.Tensor:
    first = model(inputs)
    second = model(inputs * 0.73 + 0.11)
    return torch.cat((first, second), dim=0)


def _stress_directions(
    generator: torch.Generator,
    scale: float,
) -> tuple[FactorDirection, ...]:
    factor_scale = float(scale) ** 0.5
    return (
        FactorDirection(
            layer=0,
            weight_name="first.weight",
            left=torch.randn(7, 2, generator=generator, dtype=torch.float64)
            * factor_scale,
            right=torch.randn(5, 2, generator=generator, dtype=torch.float64)
            * factor_scale,
        ),
        FactorDirection(
            layer=1,
            weight_name="second.weight",
            left=torch.randn(3, 2, generator=generator, dtype=torch.float64)
            * factor_scale,
            right=torch.randn(7, 2, generator=generator, dtype=torch.float64)
            * factor_scale,
        ),
    )


class DirectionalDerivativeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TinyNetwork().double()
        self.inputs = torch.tensor(
            [[0.2, -0.7, 1.1], [0.5, 0.4, -0.3]], dtype=torch.float64
        )
        self.directions = _directions()

    def _scalar_identity(self, direction: FactorDirection) -> float:
        alpha = torch.zeros((), dtype=torch.float64, requires_grad=True)
        parameters = dict(self.model.named_parameters())
        parameters[direction.weight_name] = (
            parameters[direction.weight_name]
            + alpha * (direction.left @ direction.right.transpose(0, 1))
        )
        output = torch.func.functional_call(self.model, parameters, (self.inputs,))
        loss = output.square().mean() + 0.3 * output.mean()
        return float(torch.autograd.grad(loss, alpha)[0])

    def test_one_backward_all_layers_matches_scalar_autograd(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        metrics = EditInstrumentation("derivative-edit")
        original_grad = torch.autograd.grad
        with directional_gradient_scope(self.model, self.directions):
            smooth = _loss(self.model, self.inputs)
            with mock.patch("torch.autograd.grad", wraps=original_grad) as grad_mock:
                field = all_layer_directional_derivatives(
                    smooth,
                    self.model,
                    self.directions,
                    instrumentation=metrics,
                )
                self.assertEqual(grad_mock.call_count, 1)
        self.assertTrue(all(not p.requires_grad for p in self.model.parameters()))
        self.assertEqual(field.backward_calls, 1)
        for value, direction in zip(field.values, self.directions, strict=True):
            self.assertAlmostEqual(
                value.event_derivative,
                self._scalar_identity(direction),
                places=11,
            )
            self.assertEqual(value.progress_slope, max(0.0, -value.event_derivative))
        snapshot = metrics.finalize().to_dict()
        self.assertEqual(snapshot["counters"]["N_model_fwd"], 0)
        self.assertEqual(snapshot["counters"]["N_bw"], 1)

    def test_actuator_hook_matches_dense_oracle_without_target_grad(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        with directional_gradient_scope(self.model, self.directions):
            dense_smooth = _loss(self.model, self.inputs) + _loss(
                self.model, self.inputs * 0.5
            )
            reference = all_layer_directional_derivatives(
                dense_smooth,
                self.model,
                self.directions,
            )

        target_parameters = tuple(
            dict(self.model.named_parameters())[direction.weight_name]
            for direction in self.directions
        )
        pointers = tuple(parameter.data_ptr() for parameter in target_parameters)
        versions = tuple(parameter._version for parameter in target_parameters)
        metrics = EditInstrumentation("actuator-hook-edit")
        original_grad = torch.autograd.grad
        with ActuatorDirectionalHook(
            self.model,
            self.directions,
            instrumentation=metrics,
        ) as hook:
            with metrics.state_forward("field"):
                first = _loss(self.model, self.inputs)
            with metrics.state_forward("field"):
                second = _loss(self.model, self.inputs * 0.5)
            with mock.patch("torch.autograd.grad", wraps=original_grad) as grad_mock:
                observed = hook.compute(first + second)
                self.assertEqual(grad_mock.call_count, 1)
                grad_inputs = grad_mock.call_args.args[1]
                self.assertTrue(
                    all(not isinstance(value, torch.nn.Parameter) for value in grad_inputs)
                )

        self.assertEqual(observed.backend, "activation-actuator-hook")
        self.assertEqual(reference.backend, "dense-target-gradient-reference-only")
        for expected, actual in zip(reference.values, observed.values, strict=True):
            self.assertAlmostEqual(
                expected.event_derivative,
                actual.event_derivative,
                delta=1e-10,
            )
        for parameter, pointer, version in zip(
            target_parameters, pointers, versions, strict=True
        ):
            self.assertIsNone(parameter.grad)
            self.assertEqual(parameter.data_ptr(), pointer)
            self.assertEqual(parameter._version, version)
            self.assertFalse(parameter.requires_grad)
        counters = metrics.finalize().to_dict()["counters"]
        self.assertEqual(counters["N_model_fwd"], 2)
        self.assertEqual(counters["N_field_state_fwd"], 2)
        self.assertEqual(counters["N_bw"], 1)

    def test_scalar_gate_matches_dense_oracle_and_primary_hook(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        with directional_gradient_scope(self.model, self.directions):
            dense_event = _loss(self.model, self.inputs)
            dense = all_layer_directional_derivatives(
                dense_event,
                self.model,
                self.directions,
            )
        with ActuatorDirectionalHook(self.model, self.directions) as hook:
            hook_event = _loss(self.model, self.inputs)
            primary = hook.compute(hook_event)
        metrics = EditInstrumentation("scalar-gate-reference")
        with ScalarGateDirectionalReference(
            self.model,
            self.directions,
            instrumentation=metrics,
        ) as reference:
            with metrics.state_forward("reference_gate"):
                scalar_event = _loss(self.model, self.inputs)
            scalar = reference.compute(scalar_event)
        rows = assert_scalar_gate_matches_hook(
            primary,
            scalar,
            abs_tol=5e-5,
            rel_tol=5e-3,
        )
        self.assertEqual(len(rows), len(self.directions))
        for dense_value, scalar_value in zip(
            dense.values, scalar.values, strict=True
        ):
            self.assertAlmostEqual(
                dense_value.event_derivative,
                scalar_value.event_derivative,
                delta=5e-5,
            )
        counters = metrics.finalize().to_dict()["counters"]
        self.assertEqual(counters["N_model_fwd"], 1)
        self.assertEqual(counters["N_reference_gate_fwd"], 1)
        self.assertEqual(counters["N_reference_gate_bw"], 1)
        self.assertEqual(counters["N_bw"], 0)

    def test_scalar_gate_negative_controls_catch_primary_corruption(self) -> None:
        from project.run_scripts.ode_edit_method import derivatives as module

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        with ScalarGateDirectionalReference(self.model, self.directions) as reference:
            scalar_event = _loss(self.model, self.inputs)
            scalar = reference.compute(scalar_event)
        original = module._activation_low_rank_contraction

        def wrong_sign(*args, **kwargs):
            return -original(*args, **kwargs)

        def wrong_orientation(module_input, grad_output, direction):
            corrupted = FactorDirection(
                layer=direction.layer,
                weight_name=direction.weight_name,
                left=direction.left.flip(0),
                right=direction.right,
            )
            return original(module_input, grad_output, corrupted)

        def wrong_input(module_input, grad_output, direction):
            return original(module_input.roll(1, dims=-1), grad_output, direction)

        for label, corruption in (
            ("sign", wrong_sign),
            ("orientation", wrong_orientation),
            ("captured-input", wrong_input),
        ):
            with self.subTest(corruption=label):
                with mock.patch.object(
                    module,
                    "_activation_low_rank_contraction",
                    side_effect=corruption,
                ):
                    with ActuatorDirectionalHook(
                        self.model, self.directions
                    ) as hook:
                        hook_event = _loss(self.model, self.inputs)
                        corrupted = hook.compute(hook_event)
                with self.assertRaisesRegex(
                    MethodContractError, "scalar-gate hard reference"
                ):
                    assert_scalar_gate_matches_hook(
                        corrupted,
                        scalar,
                        abs_tol=5e-5,
                        rel_tol=5e-3,
                    )

    def test_scalar_gate_exception_cleanup_and_model_independent_schema(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        parameters = tuple(self.model.parameters())
        pointers = tuple(parameter.data_ptr() for parameter in parameters)
        versions = tuple(parameter._version for parameter in parameters)
        baseline = self.model(self.inputs).detach().clone()
        with self.assertRaisesRegex(RuntimeError, "injected scalar-gate failure"):
            with ScalarGateDirectionalReference(self.model, self.directions):
                _ = self.model(self.inputs)
                raise RuntimeError("injected scalar-gate failure")
        self.assertTrue(torch.equal(self.model(self.inputs), baseline))
        for parameter, pointer, version in zip(
            parameters, pointers, versions, strict=True
        ):
            self.assertIsNone(parameter.grad)
            self.assertEqual(parameter.data_ptr(), pointer)
            self.assertEqual(parameter._version, version)
            self.assertFalse(parameter.requires_grad)
        source = inspect.getsource(ScalarGateDirectionalReference)
        self.assertNotIn("llama3-8b-inst", source)
        self.assertNotIn("qwen2.5-7b-inst", source)

    def test_bfloat16_fixed_small_fd_can_false_fail_while_a_b_agree(self) -> None:
        model = TinyBfloatLinear(1, 1).to(dtype=torch.bfloat16)
        with torch.no_grad():
            model.linear.weight.fill_(1.0)
        model.linear.weight.requires_grad_(False)
        inputs = torch.ones((1, 1), dtype=torch.bfloat16)
        direction = FactorDirection(
            layer=0,
            weight_name="linear.weight",
            left=torch.tensor([[0.125]], dtype=torch.float64),
            right=torch.tensor([[1.0]], dtype=torch.float64),
        )
        with ActuatorDirectionalHook(model, (direction,)) as hook:
            primary_event = model(inputs).sum()
            primary = hook.compute(primary_event)
        with ScalarGateDirectionalReference(model, (direction,)) as reference:
            scalar_event = model(inputs).sum()
            scalar = reference.compute(scalar_event)
        assert_scalar_gate_matches_hook(
            primary,
            scalar,
            abs_tol=5e-5,
            rel_tol=5e-3,
        )

        baseline = float(model(inputs).float().sum())
        epsilon = 1e-3
        with LowRankFunctionalTrial(model, (direction,), (epsilon,)):
            shifted = float(model(inputs).float().sum())
        finite_difference = (shifted - baseline) / epsilon
        derivative = primary.values[0].event_derivative
        self.assertEqual(finite_difference, 0.0)
        self.assertEqual(derivative, 0.125)
        self.assertGreater(abs(derivative - finite_difference), 5e-5)

    def test_seeded_bfloat16_nonlinear_rank2_a_b_stress(self) -> None:
        seeds = (13, 29, 47, 71)
        scales = (0.25, 1.0, 4.0)
        observed_cases = []
        for seed in seeds:
            for scale in scales:
                with self.subTest(seed=seed, scale=scale):
                    generator = torch.Generator().manual_seed(seed)
                    model = BfloatNonlinearStressNetwork().to(dtype=torch.bfloat16)
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.copy_(
                                torch.randn(
                                    parameter.shape,
                                    generator=generator,
                                    dtype=torch.float32,
                                ).to(torch.bfloat16)
                                * 0.37
                            )
                            parameter.requires_grad_(False)
                    inputs = (
                        torch.randn(
                            3,
                            5,
                            generator=generator,
                            dtype=torch.float32,
                        )
                        * 0.41
                        + 0.07
                    ).to(torch.bfloat16)
                    directions = _stress_directions(generator, scale)
                    baseline_event = _combined_stress_event(model, inputs).detach()

                    with ActuatorDirectionalHook(model, directions) as hook:
                        primary_event = _combined_stress_event(model, inputs)
                        primary = hook.compute(primary_event)
                    with ScalarGateDirectionalReference(
                        model, directions
                    ) as reference:
                        scalar_event = _combined_stress_event(model, inputs)
                        self.assertTrue(
                            torch.equal(scalar_event.detach(), baseline_event)
                        )
                        scalar = reference.compute(scalar_event)
                    rows = assert_scalar_gate_matches_hook(
                        primary,
                        scalar,
                        abs_tol=5e-5,
                        rel_tol=5e-3,
                    )
                    observed_cases.extend(rows)
        self.assertEqual(len(observed_cases), len(seeds) * len(scales) * 2)


class FunctionalTrialTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = TinyNetwork().double()
        self.inputs = torch.randn(5, 3, dtype=torch.float64)
        self.directions = _directions()

    def test_matches_dense_reference_without_weight_copy_or_mutation(self) -> None:
        coefficients = (0.35, 0.6)
        reference = copy.deepcopy(self.model)
        with torch.no_grad():
            for direction, coefficient in zip(
                self.directions, coefficients, strict=True
            ):
                parameter = dict(reference.named_parameters())[direction.weight_name]
                parameter.add_(
                    coefficient * direction.left @ direction.right.transpose(0, 1)
                )
        expected = reference(self.inputs)
        parameters = dict(self.model.named_parameters())
        pointers = {name: value.data_ptr() for name, value in parameters.items()}
        versions = {name: value._version for name, value in parameters.items()}
        baseline = self.model(self.inputs)

        with LowRankFunctionalTrial(
            self.model, self.directions, coefficients
        ) as trial:
            observed = self.model(self.inputs)
            self.assertEqual(trial.applied_coefficients, coefficients)
        self.assertTrue(torch.allclose(observed, expected, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.equal(self.model(self.inputs), baseline))
        for name, value in self.model.named_parameters():
            self.assertEqual(value.data_ptr(), pointers[name])
            self.assertEqual(value._version, versions[name])

    def test_exception_and_commit_attempt_leave_no_hook_or_write(self) -> None:
        baseline = self.model(self.inputs)
        with self.assertRaisesRegex(RuntimeError, "probe failure"):
            with LowRankFunctionalTrial(self.model, self.directions, (0.2, 0.3)):
                _ = self.model(self.inputs)
                raise RuntimeError("probe failure")
        self.assertTrue(torch.equal(self.model(self.inputs), baseline))

        with self.assertRaises(MethodContractError):
            with LowRankFunctionalTrial(
                self.model, self.directions, (0.2, 0.3)
            ) as trial:
                trial.commit()
        self.assertTrue(torch.equal(self.model(self.inputs), baseline))

    def test_read_only_trial_then_exact_accepted_write_matches(self) -> None:
        coefficients = (0.35, 0.6)
        pointers = {
            name: parameter.data_ptr()
            for name, parameter in self.model.named_parameters()
        }
        with LowRankFunctionalTrial(
            self.model, self.directions, coefficients
        ):
            trial_output = self.model(self.inputs).detach()
        applied = apply_accepted_factors(
            self.model, self.directions, coefficients, row_block=2
        )
        committed_output = self.model(self.inputs).detach()
        self.assertEqual(applied, coefficients)
        self.assertTrue(
            torch.allclose(trial_output, committed_output, atol=1e-12, rtol=1e-12)
        )
        for name, parameter in self.model.named_parameters():
            self.assertEqual(parameter.data_ptr(), pointers[name])

    def test_outer_checkpoint_restores_mid_commit_failure_exactly(self) -> None:
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        from project.run_scripts.ode_edit_method import hooks as hook_module
        from project.run_scripts.ode_edit_method.hooks import TorchCheckpoint

        checkpoint = TorchCheckpoint.capture(
            self.model,
            tuple(direction.weight_name for direction in self.directions),
        )
        before_rng = torch.get_rng_state().clone()
        before_python_rng = random.getstate()
        import numpy as np
        before_numpy_rng = np.random.get_state()

        real_apply = hook_module._apply_factor_update_
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected accepted-write failure")
            return real_apply(*args, **kwargs)

        with mock.patch.object(
            hook_module, "_apply_factor_update_", side_effect=fail_second
        ):
            with self.assertRaisesRegex(RuntimeError, "injected accepted-write failure"):
                apply_accepted_factors(self.model, self.directions, (0.2, 0.3))
        # No per-accepted-step backup: the first layer was partially committed.
        self.assertFalse(torch.equal(self.model.first.weight, before["first.weight"]))
        _ = random.random()
        _ = np.random.random()
        checkpoint.restore(self.model)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[name]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        self.assertEqual(random.getstate(), before_python_rng)
        observed_numpy_rng = np.random.get_state()
        self.assertEqual(observed_numpy_rng[0], before_numpy_rng[0])
        self.assertTrue(np.array_equal(observed_numpy_rng[1], before_numpy_rng[1]))
        self.assertEqual(observed_numpy_rng[2:], before_numpy_rng[2:])

    def test_functional_trial_restores_rng_and_detects_weight_mutation(self) -> None:
        before_rng = torch.get_rng_state().clone()
        with LowRankFunctionalTrial(self.model, self.directions, (0.2, 0.3)):
            _ = torch.rand(4)
            _ = self.model(self.inputs)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))

        with self.assertRaisesRegex(MethodContractError, "parameter version changed"):
            with LowRankFunctionalTrial(self.model, self.directions, (0.2, 0.3)):
                with torch.no_grad():
                    self.model.first.weight.add_(1.0)

    def test_trial_and_accepted_write_reject_materialized_target_grad(self) -> None:
        self.model.first.weight.grad = torch.zeros_like(self.model.first.weight)
        with self.assertRaisesRegex(MethodContractError, r"\.grad must be None"):
            with LowRankFunctionalTrial(self.model, self.directions, (0.2, 0.3)):
                pass
        with self.assertRaisesRegex(MethodContractError, r"\.grad must be None"):
            apply_accepted_factors(self.model, self.directions, (0.2, 0.3))

    def test_bfloat16_nontrivial_functional_and_commit_paths_match_separately(self) -> None:
        model = TinyBfloatLinear(2, 2).to(dtype=torch.bfloat16)
        with torch.no_grad():
            model.linear.weight.copy_(
                torch.tensor(
                    [[1.0, 0.5], [-0.25, 0.75]], dtype=torch.bfloat16
                )
            )
        model.linear.weight.requires_grad_(False)
        inputs = torch.tensor(
            [[1.0, -0.5], [0.25, 0.75]], dtype=torch.bfloat16
        )
        direction = FactorDirection(
            layer=0,
            weight_name="linear.weight",
            left=torch.tensor([[0.5], [-0.25]], dtype=torch.float64),
            right=torch.tensor([[0.25], [0.5]], dtype=torch.float64),
        )
        coefficient = 0.5
        pointer = model.linear.weight.data_ptr()
        with LowRankFunctionalTrial(model, (direction,), (coefficient,)):
            functional = model(inputs).detach().clone()
        applied = apply_accepted_factors(
            model,
            (direction,),
            (coefficient,),
            row_block=1,
        )
        committed = model(inputs).detach()
        self.assertEqual(applied, (coefficient,))
        self.assertTrue(torch.equal(functional, committed))
        self.assertEqual(model.linear.weight.data_ptr(), pointer)
        self.assertIsNone(model.linear.weight.grad)
        self.assertFalse(model.linear.weight.requires_grad)

    def test_quantized_trial_zero_identity_cleanup_and_shape_contract(self) -> None:
        parameters = tuple(self.model.parameters())
        pointers = tuple(parameter.data_ptr() for parameter in parameters)
        versions = tuple(parameter._version for parameter in parameters)
        requires_grad = tuple(parameter.requires_grad for parameter in parameters)
        baseline = self.model(self.inputs).detach()
        before_rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "injected quantized trial failure"):
            with QuantizedRowBlockFunctionalTrial(
                self.model,
                self.directions,
                (0.0, 0.0),
                row_block=64,
            ) as trial:
                observed = self.model(self.inputs).detach()
                self.assertTrue(torch.equal(observed, baseline))
                _ = torch.rand(4)
                contract = trial.temporary_shape_contract
                self.assertLessEqual(
                    contract["max_effective_weight_elements"],
                    64 * max(direction.right.shape[0] for direction in self.directions),
                )
                raise RuntimeError("injected quantized trial failure")
        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
        self.assertTrue(torch.equal(self.model(self.inputs).detach(), baseline))
        for parameter, pointer, version, flag in zip(
            parameters,
            pointers,
            versions,
            requires_grad,
            strict=True,
        ):
            self.assertEqual(parameter.data_ptr(), pointer)
            self.assertEqual(parameter._version, version)
            self.assertIsNone(parameter.grad)
            self.assertEqual(parameter.requires_grad, flag)

        with self.assertRaisesRegex(MethodContractError, "read-only"):
            with QuantizedRowBlockFunctionalTrial(
                self.model,
                self.directions,
                (0.2, 0.3),
            ) as trial:
                trial.commit()

        source = inspect.getsource(QuantizedRowBlockFunctionalTrial)
        self.assertIn("left[start:end] @ right_t", source)
        self.assertNotIn("weight.detach().clone", source)
        self.assertNotIn("llama3-8b-inst", source)
        self.assertNotIn("qwen2.5-7b-inst", source)

    def test_quantized_trial_rejects_noncanonical_linear_and_target_grad(self) -> None:
        class LinearSubclass(torch.nn.Linear):
            pass

        model = TinyBfloatLinear(2, 2)
        model.linear = LinearSubclass(2, 2, bias=False)
        direction = FactorDirection(
            layer=0,
            weight_name="linear.weight",
            left=torch.ones(2, 1),
            right=torch.ones(2, 1),
        )
        with self.assertRaisesRegex(MethodContractError, "exact torch.nn.Linear"):
            with QuantizedRowBlockFunctionalTrial(model, (direction,), (0.5,)):
                pass

        self.model.first.weight.grad = torch.zeros_like(self.model.first.weight)
        with self.assertRaisesRegex(MethodContractError, r"\.grad must be None"):
            with QuantizedRowBlockFunctionalTrial(
                self.model,
                self.directions,
                (0.2, 0.3),
            ):
                pass

    def test_quantized_trial_exercises_multiple_output_row_blocks(self) -> None:
        generator = torch.Generator().manual_seed(109)
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 130, bias=True)
        ).to(torch.bfloat16)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(
                    torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=torch.float32,
                    ).to(torch.bfloat16)
                    * 0.17
                )
                parameter.requires_grad_(False)
        inputs = torch.randn(4, 5, generator=generator).to(torch.bfloat16)
        direction = FactorDirection(
            layer=0,
            weight_name="0.weight",
            left=torch.randn(130, 2, generator=generator),
            right=torch.randn(5, 2, generator=generator),
        )
        baseline = model(inputs).detach()
        with QuantizedRowBlockFunctionalTrial(
            model,
            (direction,),
            (0.0,),
            row_block=64,
        ):
            self.assertTrue(torch.equal(model(inputs).detach(), baseline))

        with QuantizedRowBlockFunctionalTrial(
            model,
            (direction,),
            (0.375,),
            row_block=64,
        ) as trial:
            trial_output = model(inputs).detach()
            contract = trial.temporary_shape_contract
        apply_accepted_factors(
            model,
            (direction,),
            (0.375,),
            row_block=64,
        )
        self.assertTrue(torch.equal(trial_output, model(inputs).detach()))
        self.assertEqual(contract["max_effective_weight_elements"], 64 * 5)
        self.assertEqual(contract["max_output_block_elements"], 4 * 64)

    def test_seeded_bfloat16_nonlinear_rank2_t_c_stress(self) -> None:
        seeds = (17, 37, 59, 83)
        scales = (0.25, 1.0, 4.0)
        coefficients = (0.125, 0.5, 1.0)
        comparisons = 0
        output_passes = 0
        event_passes = 0
        zero_identity_passes = 0
        for seed in seeds:
            for scale in scales:
                for coefficient in coefficients:
                    with self.subTest(
                        seed=seed,
                        scale=scale,
                        coefficient=coefficient,
                    ):
                        generator = torch.Generator().manual_seed(seed)
                        model = BfloatNonlinearStressNetwork().to(
                            dtype=torch.bfloat16
                        )
                        with torch.no_grad():
                            for parameter in model.parameters():
                                parameter.copy_(
                                    torch.randn(
                                        parameter.shape,
                                        generator=generator,
                                        dtype=torch.float32,
                                    ).to(torch.bfloat16)
                                    * 0.37
                                )
                                parameter.requires_grad_(False)
                        inputs = (
                            torch.randn(
                                3,
                                5,
                                generator=generator,
                                dtype=torch.float32,
                            )
                            * 0.41
                            + 0.07
                        ).to(torch.bfloat16)
                        directions = _stress_directions(generator, scale)
                        parameters = tuple(model.parameters())
                        pointers = tuple(
                            parameter.data_ptr() for parameter in parameters
                        )
                        versions = tuple(parameter._version for parameter in parameters)
                        requires_grad = tuple(
                            parameter.requires_grad for parameter in parameters
                        )
                        before_rng = torch.get_rng_state().clone()
                        baseline_output = _combined_stress_output(model, inputs).detach()
                        with QuantizedRowBlockFunctionalTrial(
                            model,
                            directions,
                            (0.0, 0.0),
                            row_block=64,
                        ):
                            zero_output = _combined_stress_output(model, inputs).detach()
                        zero_identity_passes += int(
                            torch.equal(zero_output, baseline_output)
                        )
                        with QuantizedRowBlockFunctionalTrial(
                            model,
                            directions,
                            (coefficient, coefficient),
                            row_block=64,
                        ):
                            functional_output = _combined_stress_output(
                                model, inputs
                            ).detach()
                            functional_event = _combined_stress_event(
                                model, inputs
                            ).detach()
                        self.assertTrue(torch.equal(torch.get_rng_state(), before_rng))
                        for parameter, pointer, version, flag in zip(
                            parameters,
                            pointers,
                            versions,
                            requires_grad,
                            strict=True,
                        ):
                            self.assertEqual(parameter.data_ptr(), pointer)
                            self.assertEqual(parameter._version, version)
                            self.assertIsNone(parameter.grad)
                            self.assertEqual(parameter.requires_grad, flag)
                        apply_accepted_factors(
                            model,
                            directions,
                            (coefficient, coefficient),
                            row_block=64,
                        )
                        committed_output = _combined_stress_output(
                            model, inputs
                        ).detach()
                        committed_event = _combined_stress_event(
                            model, inputs
                        ).detach()
                        output_passes += int(
                            torch.allclose(
                                functional_output.float(), committed_output.float(),
                                atol=5e-5, rtol=5e-3
                            )
                        )
                        event_passes += int(
                            torch.allclose(
                                functional_event.float(), committed_event.float(),
                                atol=5e-5, rtol=5e-3
                            )
                        )
                        for parameter, pointer in zip(
                            parameters, pointers, strict=True
                        ):
                            self.assertEqual(parameter.data_ptr(), pointer)
                            self.assertIsNone(parameter.grad)
                            self.assertFalse(parameter.requires_grad)
                        comparisons += 1
        self.assertEqual(
            comparisons,
            len(seeds) * len(scales) * len(coefficients),
        )
        self.assertEqual(output_passes, comparisons)
        self.assertEqual(event_passes, comparisons)
        self.assertEqual(zero_identity_passes, comparisons)


if __name__ == "__main__":
    unittest.main()
