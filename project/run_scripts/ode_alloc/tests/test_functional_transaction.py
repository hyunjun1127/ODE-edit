from __future__ import annotations

import copy
import threading
import unittest

import torch

from project.run_scripts.ode_alloc.accounting import ComputeLedger
from project.run_scripts.ode_alloc.functional import (
    FP32FactorOverlay,
    QuantizedBF16FunctionalTrial,
    independent_native_weight_cpu_fixture,
    quantized_effective_weight,
)
from project.run_scripts.ode_alloc.gauge import FactorPair, FixedEnergyGauge
from project.run_scripts.ode_alloc.transaction import AtomicLayerTransaction


class TinyBF16Network(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
        self.second = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.nn.functional.silu(self.first(value)))


def _factors() -> dict[str, FactorPair]:
    return {
        "first.weight": FactorPair(
            4,
            torch.tensor([[0.2], [-0.1], [0.3], [0.4]]),
            torch.tensor([[0.5], [-0.2], [0.1]]),
        ),
        "second.weight": FactorPair(
            5,
            torch.tensor([[0.2], [-0.3]]),
            torch.tensor([[0.4], [0.1], [-0.2], [0.3]]),
        ),
    }


class FunctionalTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.model = TinyBF16Network()
        self.factors = _factors()
        self.inputs = torch.tensor(
            [[0.2, -0.4, 0.8], [0.1, 0.3, -0.7]], dtype=torch.bfloat16
        )

    def test_q_zero_matches_independent_native_bytes_output_and_event(self) -> None:
        gauge = FixedEnergyGauge(
            {pair.layer: pair for pair in self.factors.values()},
            basis_energy_epsilon=1.0e-24,
            max_abs_centered_q=20.0,
        )
        reading = gauge.evaluate(gauge.zeros())
        self.assertTrue(torch.equal(reading.ratios, torch.ones_like(reading.ratios)))
        native = copy.deepcopy(self.model)
        native_parameters = dict(native.named_parameters())
        original_parameters = dict(self.model.named_parameters())
        for name, pair in self.factors.items():
            expected = independent_native_weight_cpu_fixture(original_parameters[name], pair)
            observed = quantized_effective_weight(original_parameters[name], pair, 1.0)
            self.assertTrue(torch.equal(expected, observed))
            with torch.no_grad():
                native_parameters[name].copy_(expected)

        pointers = {name: value.data_ptr() for name, value in original_parameters.items()}
        versions = {name: value._version for name, value in original_parameters.items()}
        rng = torch.get_rng_state().clone()
        trial = QuantizedBF16FunctionalTrial(
            self.model,
            self.factors,
            {layer: float(ratio) for layer, ratio in reading.ratio_by_layer().items()},
            row_block=1,
        )
        with trial:
            virtual_output = self.model(self.inputs)
        native_output = native(self.inputs)
        self.assertTrue(torch.equal(virtual_output, native_output))
        virtual_event = torch.log_softmax(virtual_output.float(), dim=-1)
        native_event = torch.log_softmax(native_output.float(), dim=-1)
        self.assertTrue(torch.equal(virtual_event, native_event))
        self.assertEqual(trial.max_live_effective_weights, 1)
        self.assertEqual(trial.replacement_linear_calls, 2)
        self.assertLess(
            trial.max_fp32_delta_block_elements,
            max(parameter.numel() for parameter in original_parameters.values()),
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        for name, parameter in original_parameters.items():
            self.assertEqual(parameter.data_ptr(), pointers[name])
            self.assertEqual(parameter._version, versions[name])
            self.assertIsNone(parameter.grad)

    def test_fp32_overlay_is_gradient_only_and_dense_delta_free(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        ratios = {
            pair.layer: torch.tensor(1.0, dtype=torch.float32, requires_grad=True)
            for pair in self.factors.values()
        }
        overlay = FP32FactorOverlay(self.model, self.factors, ratios)
        self.assertFalse(overlay.verdict_eligible)
        pointers = {
            name: parameter.data_ptr()
            for name, parameter in self.model.named_parameters()
        }
        versions = {
            name: parameter._version
            for name, parameter in self.model.named_parameters()
        }
        rng = torch.get_rng_state().clone()
        with overlay:
            loss = self.model(self.inputs).float().square().mean()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        loss.backward()
        self.assertTrue(all(value.grad is not None for value in ratios.values()))
        for name, parameter in self.model.named_parameters():
            self.assertIsNone(parameter.grad)
            self.assertEqual(parameter.data_ptr(), pointers[name])
            self.assertEqual(parameter._version, versions[name])

    def test_one_transaction_commit_and_fault_rollback(self) -> None:
        parameters = dict(self.model.named_parameters())
        candidates = {
            name: quantized_effective_weight(parameters[name], pair, 1.0)
            for name, pair in self.factors.items()
        }
        lock = threading.RLock()
        ledger = ComputeLedger()
        transaction = AtomicLayerTransaction(parameters, mutation_lock=lock)
        for name, candidate in candidates.items():
            transaction.stage(name, candidate)
        self.assertTrue(
            all(value.device.type == "cpu" for value in transaction._staged.values())
        )
        receipt = transaction.commit(ledger=ledger)
        self.assertEqual(receipt.commit_count, 1)
        self.assertEqual(ledger.commit_count, 1)
        for name, candidate in candidates.items():
            self.assertTrue(torch.equal(parameters[name], candidate))

        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        pointers = {name: parameter.data_ptr() for name, parameter in parameters.items()}
        rng = torch.get_rng_state().clone()
        faulting = AtomicLayerTransaction(parameters, mutation_lock=lock)
        for name, parameter in parameters.items():
            faulting.stage(name, torch.zeros_like(parameter))
        with self.assertRaisesRegex(RuntimeError, "injected"):
            faulting.commit(fault_after_writes=1)
        for name, parameter in parameters.items():
            self.assertTrue(torch.equal(parameter, before[name]))
            self.assertEqual(parameter.data_ptr(), pointers[name])
            self.assertIsNone(parameter.grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))


if __name__ == "__main__":
    unittest.main()
