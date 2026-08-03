from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_edit_method.contracts import MethodContractError
from project.run_scripts.ode_edit_method.derivatives import (
    ActuatorDirectionalHook,
    all_layer_directional_derivatives,
    directional_gradient_scope,
)
from project.run_scripts.ode_edit_method.functional_trial import LowRankFunctionalTrial
from project.run_scripts.ode_edit_method.hooks import FactorDirection
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation


class TinyNetwork(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(3, 4, bias=True)
        self.second = torch.nn.Linear(4, 2, bias=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.tanh(self.first(value)))


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
        self.assertEqual(snapshot["counters"]["N_field"], 1)
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
            with metrics.state_forward():
                first = _loss(self.model, self.inputs)
            with metrics.state_forward():
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
        self.assertEqual(counters["N_state_fwd"], 2)
        self.assertEqual(counters["N_field"], 1)
        self.assertEqual(counters["N_bw"], 1)


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


if __name__ == "__main__":
    unittest.main()
