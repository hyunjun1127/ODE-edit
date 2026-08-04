from __future__ import annotations

import copy
import hashlib
import threading
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError, ODEBFStateError
from project.run_scripts.ode_bf.functional import (
    CumulativeBF16FunctionalTrial,
    WaypointFactor,
    assemble_effective_bf16,
    independent_native_alpha_bf16_fixture,
    tensor_sha256,
)
from project.run_scripts.ode_bf.transaction import AtomicBatchTransaction
from project.run_scripts.ode_bf.woodbury import (
    full_projector_certificate,
    solve_alpha_woodbury,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class InteractingB10Network(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(12, 14, bias=False, dtype=torch.bfloat16)
        self.middle = torch.nn.Linear(14, 13, bias=False, dtype=torch.bfloat16)
        self.last = torch.nn.Linear(13, 11, bias=False, dtype=torch.bfloat16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.nn.functional.silu(self.first(value))
        value = torch.nn.functional.silu(self.middle(value))
        return self.last(value)


def _factor(
    weight_name: str,
    layer: int,
    shape: tuple[int, int],
    *,
    cycle: int = 0,
    step: int = 0,
    ordinal: int = 0,
    theta: float = 1.0,
    seed: int = 1,
) -> WaypointFactor:
    generator = torch.Generator().manual_seed(seed)
    out_features, in_features = shape
    left = 0.02 * torch.randn((out_features, 10), generator=generator, dtype=torch.float64)
    right = 0.02 * torch.randn((in_features, 10), generator=generator, dtype=torch.float64)
    # Every request contributes to shared row/column geometry.
    left[0] += torch.linspace(0.001, 0.01, 10, dtype=torch.float64)
    right[0] += torch.linspace(0.01, 0.001, 10, dtype=torch.float64)
    return WaypointFactor(
        weight_name,
        layer,
        cycle,
        step,
        ordinal,
        theta,
        left,
        right,
    )


class CumulativeFunctionalTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.model = InteractingB10Network()
        self.inputs = torch.randn((5, 12), dtype=torch.bfloat16)
        self.shapes = {
            "first.weight": (14, 12),
            "middle.weight": (13, 14),
            "last.weight": (11, 13),
        }
        self.layers = {"first.weight": 4, "middle.weight": 5, "last.weight": 6}

    def _factors(self) -> dict[str, tuple[WaypointFactor, ...]]:
        return {
            name: (_factor(name, self.layers[name], shape, seed=index + 3),)
            for index, (name, shape) in enumerate(self.shapes.items())
        }

    def test_true_rank10_q0_matches_independent_native_bytes_logits_and_event(self) -> None:
        factors = self._factors()
        parameters = dict(self.model.named_parameters())
        native = copy.deepcopy(self.model)
        with torch.no_grad():
            for name, factor_list in factors.items():
                observed, stats = assemble_effective_bf16(parameters[name], factor_list, row_block=3)
                expected = independent_native_alpha_bf16_fixture(parameters[name], factor_list[0])
                self.assertTrue(torch.equal(observed.cpu(), expected))
                self.assertEqual(stats.dense_fp32_full_delta_live, 0)
                self.assertLess(stats.maximum_fp32_block_elements, parameters[name].numel())
                dict(native.named_parameters())[name].copy_(expected)
        pointers = {name: parameter.data_ptr() for name, parameter in parameters.items()}
        versions = {name: parameter._version for name, parameter in parameters.items()}
        rng = torch.get_rng_state().clone()
        trial = CumulativeBF16FunctionalTrial(self.model, factors, row_block=3)
        with trial:
            virtual_output = self.model(self.inputs)
        native_output = native(self.inputs)
        self.assertTrue(torch.equal(virtual_output, native_output))
        self.assertTrue(torch.equal(torch.log_softmax(virtual_output.float(), -1), torch.log_softmax(native_output.float(), -1)))
        self.assertEqual(trial.max_live_effective_weights, 1)
        self.assertEqual(trial.replacement_linear_calls, 3)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, parameter in parameters.items():
            self.assertEqual(parameter.data_ptr(), pointers[name])
            self.assertEqual(parameter._version, versions[name])
            self.assertIsNone(parameter.grad)

    def test_entry_relative_multi_waypoint_order_is_canonical(self) -> None:
        parameter = dict(self.model.named_parameters())["first.weight"]
        first = _factor("first.weight", 4, self.shapes["first.weight"], step=0, theta=0.25, seed=51)
        second = _factor("first.weight", 4, self.shapes["first.weight"], step=1, theta=0.5, seed=52)
        ordered, _ = assemble_effective_bf16(parameter, (first, second), row_block=2)
        reversed_input, _ = assemble_effective_bf16(parameter, (second, first), row_block=2)
        self.assertTrue(torch.equal(ordered, reversed_input))
        manual = parameter.float()
        for factor in (first, second):
            update = (factor.right.float() @ factor.left.float().T).T
            manual = manual + torch.tensor(factor.theta, dtype=torch.float32) * update
        self.assertTrue(torch.equal(ordered, manual.to(torch.bfloat16)))

    def test_dense_native_and_exact_woodbury_endpoint_quantize_identically(self) -> None:
        native_model = copy.deepcopy(self.model)
        wb_model = copy.deepcopy(self.model)
        generator = torch.Generator().manual_seed(71)
        for index, (name, shape) in enumerate(self.shapes.items()):
            out_features, in_features = shape
            keys = torch.randn((in_features, 10), generator=generator, dtype=torch.float64)
            residual = 0.02 * torch.randn((out_features, 10), generator=generator, dtype=torch.float64)
            projector = torch.eye(in_features, dtype=torch.float64)
            regularization = 2.0
            dense_q = torch.linalg.solve(
                regularization * torch.eye(in_features, dtype=torch.float64) + projector @ keys @ keys.T,
                projector @ keys,
            )
            wb_q = solve_alpha_woodbury(
                projector,
                keys,
                history_keys=None,
                regularization=regularization,
                projector_certificate=full_projector_certificate(
                    projector,
                    source_sha256=_digest(f"projector-{index}"),
                ),
            ).q
            native_factor = WaypointFactor(name, self.layers[name], 0, 0, 0, 1.0, residual, dense_q)
            wb_factor = WaypointFactor(name, self.layers[name], 0, 0, 0, 1.0, residual, wb_q)
            native_candidate, _ = assemble_effective_bf16(dict(native_model.named_parameters())[name], (native_factor,), row_block=2)
            wb_candidate, _ = assemble_effective_bf16(dict(wb_model.named_parameters())[name], (wb_factor,), row_block=2)
            self.assertTrue(torch.equal(native_candidate, wb_candidate))
            with torch.no_grad():
                dict(native_model.named_parameters())[name].copy_(native_candidate)
                dict(wb_model.named_parameters())[name].copy_(wb_candidate)
        native_output = native_model(self.inputs)
        wb_output = wb_model(self.inputs)
        self.assertTrue(torch.equal(native_output, wb_output))

    def test_duplicate_factor_order_and_partial_invalid_request_fail_closed(self) -> None:
        parameter = dict(self.model.named_parameters())["first.weight"]
        factor = _factor("first.weight", 4, self.shapes["first.weight"], seed=91)
        with self.assertRaisesRegex(ODEBFContractError, "duplicate"):
            assemble_effective_bf16(parameter, (factor, factor), row_block=2)
        bad_left = factor.left.clone()
        bad_left[0, 4] = torch.nan
        with self.assertRaisesRegex(ODEBFContractError, "non-finite"):
            WaypointFactor("first.weight", 4, 0, 0, 0, 1.0, bad_left, factor.right)


class AtomicBatchTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(101)
        self.model = InteractingB10Network()
        self.parameters = dict(self.model.named_parameters())
        self.factors = {
            name: (_factor(name, layer, tuple(parameter.shape), seed=110 + index),)
            for index, (name, parameter) in enumerate(self.parameters.items())
            for layer in (4 + index,)
        }
        self.candidates = {
            name: assemble_effective_bf16(parameter, self.factors[name], row_block=2)[0]
            for name, parameter in self.parameters.items()
        }
        self.lock = threading.RLock()

    def _transaction(self, identity: str) -> AtomicBatchTransaction:
        transaction = AtomicBatchTransaction(
            self.parameters,
            transaction_id=identity,
            mutation_lock=self.lock,
        )
        for name, candidate in self.candidates.items():
            transaction.stage(name, candidate)
        return transaction

    def test_early_middle_late_faults_restore_complete_prebatch_state(self) -> None:
        original = {name: parameter.detach().clone() for name, parameter in self.parameters.items()}
        pointers = {name: parameter.data_ptr() for name, parameter in self.parameters.items()}
        for index in (0, 1, 2):
            transaction = self._transaction(f"fault-{index}")
            with self.assertRaisesRegex(RuntimeError, "injected"):
                transaction.commit(post_commit_verify=lambda: True, fault_after_writes=index)
            for name, parameter in self.parameters.items():
                self.assertTrue(torch.equal(parameter, original[name]))
                self.assertEqual(parameter.data_ptr(), pointers[name])

    def test_postcommit_allten_failure_rolls_back_and_success_commits_once(self) -> None:
        original = {name: parameter.detach().clone() for name, parameter in self.parameters.items()}
        failing = self._transaction("postverify-failure")
        with self.assertRaisesRegex(ODEBFStateError, "post-commit"):
            failing.commit(post_commit_verify=lambda: False)
        for name, parameter in self.parameters.items():
            self.assertTrue(torch.equal(parameter, original[name]))
        successful = self._transaction("success")
        receipt = successful.commit(
            post_commit_verify=lambda: all(
                torch.equal(self.parameters[name].cpu(), candidate.cpu())
                for name, candidate in self.candidates.items()
            )
        )
        self.assertEqual(receipt.commit_count, 1)
        self.assertEqual(receipt.rollback_count, 0)
        self.assertEqual(len(receipt.touched_weights), 3)
        self.assertTrue(receipt.post_commit_verified)
        for name, candidate in self.candidates.items():
            self.assertEqual(dict(receipt.parameter_sha256)[name], tensor_sha256(candidate))


if __name__ == "__main__":
    unittest.main()
