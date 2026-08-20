from __future__ import annotations

import inspect
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError, ODEBFStateError
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_fp32_transaction as transaction_module,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_fp32_transaction import (
    FP32TransactionMode,
    OfficialStyleFP32SequentialTransaction,
    RESIDUAL_RESERVE_LAYER_ORDER,
)


class ResidualReserveFP32TransactionTests(unittest.TestCase):
    @staticmethod
    def bindings(dtype: torch.dtype = torch.bfloat16):
        result = {}
        for ordinal, layer in enumerate(RESIDUAL_RESERVE_LAYER_ORDER):
            value = torch.tensor(
                [
                    [0.25 + ordinal, -0.75 - ordinal, 1.25 + ordinal],
                    [-1.5 - ordinal, 2.0 + ordinal, -2.5 - ordinal],
                ],
                dtype=dtype,
            )
            result[layer] = (
                f"model.layers.{layer}.mlp.down_proj.weight",
                torch.nn.Parameter(value, requires_grad=False),
            )
        return result

    @staticmethod
    def updates():
        return {
            layer: torch.tensor(
                [
                    [0.0031 * layer, -0.0027 * layer, 0.0019 * layer],
                    [-0.0043 * layer, 0.0051 * layer, -0.0067 * layer],
                ],
                dtype=torch.float32,
            )
            for layer in RESIDUAL_RESERVE_LAYER_ORDER
        }

    @staticmethod
    def snapshot(bindings):
        return {
            layer: {
                "value": parameter.detach().clone(),
                "hash": tensor_sha256(parameter),
                "pointer": int(parameter.data_ptr()),
                "dtype": parameter.dtype,
                "device": parameter.device,
            }
            for layer, (_, parameter) in bindings.items()
        }

    def assert_entry_restored(self, bindings, entry) -> None:
        for layer, (_, parameter) in bindings.items():
            self.assertEqual(int(parameter.data_ptr()), entry[layer]["pointer"])
            self.assertEqual(parameter.dtype, entry[layer]["dtype"])
            self.assertEqual(parameter.device, entry[layer]["device"])
            self.assertEqual(tensor_sha256(parameter), entry[layer]["hash"])
            self.assertTrue(torch.equal(parameter, entry[layer]["value"]))

    @staticmethod
    def prepare_and_commit(transaction):
        prepared = transaction.prepare_authoritative_commit()
        return prepared, transaction.commit_prepared(prepared.identity_sha256)

    def test_official_reference_equality_bf16_single_and_five_layer(self) -> None:
        bindings = self.bindings(torch.bfloat16)
        updates = self.updates()
        expected = {
            layer: parameter.detach().clone()
            for layer, (_, parameter) in bindings.items()
        }
        transaction = OfficialStyleFP32SequentialTransaction(
            bindings,
            mode=FP32TransactionMode.AUTHORITATIVE,
            transaction_id="bf16-reference",
        )
        for ordinal, layer in enumerate(RESIDUAL_RESERVE_LAYER_ORDER):
            update_before = updates[layer].clone()
            with torch.no_grad():
                expected[layer][...] = expected[layer] + updates[layer].float()
            transaction.apply_layer_fp32(layer, updates[layer])
            self.assertTrue(torch.equal(bindings[layer][1], expected[layer]))
            self.assertTrue(torch.equal(updates[layer], update_before))
            if ordinal == 0:
                self.assertEqual(
                    tensor_sha256(bindings[layer][1]), tensor_sha256(expected[layer])
                )
        prepared, receipt = self.prepare_and_commit(transaction)
        self.assertEqual(prepared.logical_outer_commit_count, 0)
        self.assertEqual(prepared.persistent_commit_count, 0)
        self.assertEqual(
            prepared.future_final_receipt.identity_sha256,
            receipt.identity_sha256,
        )
        self.assertEqual(receipt.native_storage_assignment_count, 5)
        self.assertEqual(receipt.storage_cast_boundary_count, 5)
        self.assertEqual(receipt.logical_outer_commit_count, 1)
        self.assertEqual(receipt.rollback_count, 0)

    def test_official_reference_equality_fp32_and_update_immutability(self) -> None:
        bindings = self.bindings(torch.float32)
        updates = self.updates()
        expected = {
            layer: parameter.detach().clone()
            for layer, (_, parameter) in bindings.items()
        }
        transaction = OfficialStyleFP32SequentialTransaction(
            bindings,
            mode=FP32TransactionMode.AUTHORITATIVE,
            transaction_id="fp32-reference",
        )
        for layer in RESIDUAL_RESERVE_LAYER_ORDER:
            pointer = int(bindings[layer][1].data_ptr())
            with torch.no_grad():
                expected[layer][...] = expected[layer] + updates[layer].float()
            receipt = transaction.apply_layer_fp32(layer, updates[layer])
            self.assertEqual(updates[layer].dtype, torch.float32)
            self.assertTrue(torch.equal(bindings[layer][1], expected[layer]))
            self.assertEqual(int(bindings[layer][1].data_ptr()), pointer)
            self.assertFalse(receipt.storage_cast_required)
        _, receipt = self.prepare_and_commit(transaction)
        self.assertEqual(receipt.persistent_commit_count, 1)

    def test_pointer_and_storage_dtype_stay_fixed_after_every_apply(self) -> None:
        bindings = self.bindings()
        entry = self.snapshot(bindings)
        transaction = OfficialStyleFP32SequentialTransaction(
            bindings,
            mode=FP32TransactionMode.AUTHORITATIVE,
            transaction_id="identity",
        )
        for layer, update in self.updates().items():
            receipt = transaction.apply_layer_fp32(layer, update)
            parameter = bindings[layer][1]
            self.assertEqual(int(parameter.data_ptr()), entry[layer]["pointer"])
            self.assertEqual(parameter.dtype, entry[layer]["dtype"])
            self.assertEqual(parameter.device, entry[layer]["device"])
            self.assertEqual(receipt.entry_parameter_pointer, entry[layer]["pointer"])
            self.assertEqual(receipt.post_storage_parameter_pointer, entry[layer]["pointer"])
        self.prepare_and_commit(transaction)

    def test_shadow_restores_exact_bytes_and_pointers_without_commit(self) -> None:
        bindings = self.bindings()
        entry = self.snapshot(bindings)
        transaction = OfficialStyleFP32SequentialTransaction(
            bindings,
            mode=FP32TransactionMode.SHADOW,
            transaction_id="shadow",
        )
        for layer, update in self.updates().items():
            transaction.apply_layer_fp32(layer, update)
        receipt = transaction.finish_shadow()
        self.assert_entry_restored(bindings, entry)
        self.assertEqual(receipt.logical_outer_commit_count, 0)
        self.assertEqual(receipt.persistent_commit_count, 0)
        self.assertEqual(receipt.rollback_count, 1)
        self.assertEqual(receipt.candidate_materialization_count, 0)

    def test_authoritative_commits_once_and_keeps_exact_endpoint(self) -> None:
        bindings = self.bindings()
        updates = self.updates()
        expected = {
            layer: parameter.detach().clone()
            for layer, (_, parameter) in bindings.items()
        }
        transaction = OfficialStyleFP32SequentialTransaction(
            bindings,
            mode=FP32TransactionMode.AUTHORITATIVE,
            transaction_id="authoritative",
        )
        for layer in RESIDUAL_RESERVE_LAYER_ORDER:
            with torch.no_grad():
                expected[layer][...] = expected[layer] + updates[layer].float()
            transaction.apply_layer_fp32(layer, updates[layer])
        prepared, receipt = self.prepare_and_commit(transaction)
        for layer in RESIDUAL_RESERVE_LAYER_ORDER:
            self.assertTrue(torch.equal(bindings[layer][1], expected[layer]))
        self.assertEqual(receipt.logical_outer_commit_count, 1)
        self.assertEqual(receipt.native_storage_assignment_count, 5)
        self.assertFalse(receipt.restored_entry_bytes)
        with self.assertRaises(ODEBFStateError):
            transaction.commit_prepared(prepared.identity_sha256)

    def test_prepare_is_nonmutating_and_binds_exact_future_receipt(self) -> None:
        bindings = self.bindings()
        updates = self.updates()
        transaction = OfficialStyleFP32SequentialTransaction(
            bindings,
            mode=FP32TransactionMode.AUTHORITATIVE,
            transaction_id="prepared",
        )
        for layer in RESIDUAL_RESERVE_LAYER_ORDER:
            transaction.apply_layer_fp32(layer, updates[layer])
        before = self.snapshot(bindings)
        prepared = transaction.prepare_authoritative_commit()
        after = self.snapshot(bindings)
        self.assertEqual(
            tuple(item["hash"] for item in before.values()),
            tuple(item["hash"] for item in after.values()),
        )
        self.assertFalse(transaction.finalized)
        self.assertEqual(prepared.preparation_parameter_mutation_count, 0)
        self.assertEqual(prepared.logical_outer_commit_count, 0)
        self.assertEqual(prepared.persistent_commit_count, 0)
        receipt = transaction.commit_prepared(prepared.identity_sha256)
        self.assertIs(receipt, prepared.future_final_receipt)

    def test_wrong_prepared_identity_and_postprepare_corruption_restore(self) -> None:
        for failure in ("wrong-identity", "postprepare-corruption"):
            with self.subTest(failure=failure):
                bindings = self.bindings()
                entry = self.snapshot(bindings)
                transaction = OfficialStyleFP32SequentialTransaction(
                    bindings,
                    mode=FP32TransactionMode.AUTHORITATIVE,
                    transaction_id=failure,
                )
                for layer, update in self.updates().items():
                    transaction.apply_layer_fp32(layer, update)
                prepared = transaction.prepare_authoritative_commit()
                if failure == "postprepare-corruption":
                    with torch.no_grad():
                        bindings[6][1][0, 0] += 1.0
                with self.assertRaises(ODEBFStateError):
                    transaction.commit_prepared(
                        "wrong" if failure == "wrong-identity" else prepared.identity_sha256
                    )
                self.assert_entry_restored(bindings, entry)
                self.assertTrue(transaction.finalized)
                self.assertEqual(transaction.rollback_count, 1)

    def test_direct_unprepared_commit_restores_and_fails(self) -> None:
        bindings = self.bindings()
        entry = self.snapshot(bindings)
        transaction = OfficialStyleFP32SequentialTransaction(
            bindings,
            mode=FP32TransactionMode.AUTHORITATIVE,
            transaction_id="unprepared",
        )
        for layer, update in self.updates().items():
            transaction.apply_layer_fp32(layer, update)
        with self.assertRaises(ODEBFStateError):
            transaction.commit_outer()
        self.assert_entry_restored(bindings, entry)

    def test_order_completeness_and_mode_fail_closed(self) -> None:
        updates = self.updates()
        for operation in ("out-of-order", "duplicate", "missing", "shadow-commit"):
            with self.subTest(operation=operation):
                bindings = self.bindings()
                entry = self.snapshot(bindings)
                mode = (
                    FP32TransactionMode.SHADOW
                    if operation == "shadow-commit"
                    else FP32TransactionMode.AUTHORITATIVE
                )
                transaction = OfficialStyleFP32SequentialTransaction(
                    bindings, mode=mode, transaction_id=operation
                )
                with self.assertRaises((ODEBFContractError, ODEBFStateError)):
                    if operation == "out-of-order":
                        transaction.apply_layer_fp32(5, updates[5])
                    elif operation == "duplicate":
                        transaction.apply_layer_fp32(4, updates[4])
                        transaction.apply_layer_fp32(4, updates[4])
                    elif operation == "missing":
                        transaction.apply_layer_fp32(4, updates[4])
                        transaction.commit_outer()
                    else:
                        for layer in RESIDUAL_RESERVE_LAYER_ORDER:
                            transaction.apply_layer_fp32(layer, updates[layer])
                        transaction.commit_outer()
                self.assert_entry_restored(bindings, entry)
                self.assertEqual(transaction.rollback_count, 1)

    def test_injected_failure_after_two_layers_restores_all(self) -> None:
        bindings = self.bindings()
        entry = self.snapshot(bindings)
        updates = self.updates()
        original = transaction_module._official_native_assignment
        call_count = 0

        def fail_on_third(parameter, update):
            nonlocal call_count
            if call_count == 2:
                raise RuntimeError("injected failure after layer2")
            call_count += 1
            original(parameter, update)

        transaction = OfficialStyleFP32SequentialTransaction(
            bindings,
            mode=FP32TransactionMode.AUTHORITATIVE,
            transaction_id="injected-failure",
        )
        with mock.patch.object(
            transaction_module,
            "_official_native_assignment",
            side_effect=fail_on_third,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                for layer in RESIDUAL_RESERVE_LAYER_ORDER:
                    transaction.apply_layer_fp32(layer, updates[layer])
        self.assert_entry_restored(bindings, entry)
        self.assertEqual(transaction.rollback_count, 1)

    def test_invalid_update_contracts_restore_exact(self) -> None:
        cases = {
            "nonfinite": torch.full((2, 3), float("nan"), dtype=torch.float32),
            "shape": torch.ones((3, 2), dtype=torch.float32),
            "dtype": torch.ones((2, 3), dtype=torch.float64),
            "device": torch.ones((2, 3), dtype=torch.float32, device="meta"),
        }
        for label, bad_update in cases.items():
            with self.subTest(label=label):
                bindings = self.bindings()
                entry = self.snapshot(bindings)
                transaction = OfficialStyleFP32SequentialTransaction(
                    bindings,
                    mode=FP32TransactionMode.AUTHORITATIVE,
                    transaction_id=f"invalid-{label}",
                )
                with self.assertRaises(ODEBFContractError):
                    transaction.apply_layer_fp32(4, bad_update)
                self.assert_entry_restored(bindings, entry)
                self.assertEqual(transaction.rollback_count, 1)

    def test_external_hash_and_pointer_failures_restore_exact(self) -> None:
        for failure in ("hash", "pointer"):
            with self.subTest(failure=failure):
                bindings = self.bindings()
                entry = self.snapshot(bindings)
                updates = self.updates()
                transaction = OfficialStyleFP32SequentialTransaction(
                    bindings,
                    mode=FP32TransactionMode.AUTHORITATIVE,
                    transaction_id=f"external-{failure}",
                )
                transaction.apply_layer_fp32(4, updates[4])
                parameter = bindings[5][1]
                with torch.no_grad():
                    if failure == "hash":
                        parameter[0, 0] = parameter[0, 0] + 1.0
                    else:
                        parameter.data = parameter.detach().clone()
                with self.assertRaises(ODEBFStateError):
                    transaction.apply_layer_fp32(5, updates[5])
                self.assert_entry_restored(bindings, entry)

    def test_precast_and_actual_post_storage_telemetry_are_separate(self) -> None:
        bindings = self.bindings(torch.bfloat16)
        update = torch.full((2, 3), 1.0e-4, dtype=torch.float32)
        transaction = OfficialStyleFP32SequentialTransaction(
            bindings,
            mode=FP32TransactionMode.SHADOW,
            transaction_id="telemetry",
        )
        receipt = transaction.apply_layer_fp32(4, update)
        self.assertGreater(receipt.pre_cast_fp32_update_energy, 0.0)
        self.assertEqual(receipt.actual_post_storage_delta32_energy, 0.0)
        self.assertNotEqual(
            receipt.pre_cast_fp32_update_sha256,
            receipt.actual_post_storage_delta32_sha256,
        )
        self.assertEqual(receipt.postcast_decision_influence_count, 0)
        transaction.abort_and_rollback()

    def test_context_exception_and_explicit_abort_restore(self) -> None:
        bindings = self.bindings()
        entry = self.snapshot(bindings)
        with self.assertRaisesRegex(RuntimeError, "caller failure"):
            with OfficialStyleFP32SequentialTransaction(
                bindings,
                mode=FP32TransactionMode.AUTHORITATIVE,
                transaction_id="context",
            ) as transaction:
                transaction.apply_layer_fp32(4, self.updates()[4])
                raise RuntimeError("caller failure")
        self.assert_entry_restored(bindings, entry)

        transaction = OfficialStyleFP32SequentialTransaction(
            bindings,
            mode=FP32TransactionMode.AUTHORITATIVE,
            transaction_id="abort",
        )
        transaction.apply_layer_fp32(4, self.updates()[4])
        receipt = transaction.abort_and_rollback()
        self.assert_entry_restored(bindings, entry)
        self.assertEqual(receipt.logical_outer_commit_count, 0)

    def test_prohibited_symbols_and_compute_ledger(self) -> None:
        source = inspect.getsource(transaction_module)
        for prohibited in (
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "CumulativeBF16FunctionalTrial",
            "assemble_effective_bf16",
            "effective_bf16_sha256",
            "p1r52_sequential_contract",
            "PIR-U",
            "PIRU",
            "FPIQ",
            "add_",
        ):
            self.assertNotIn(prohibited, source)

        bindings = self.bindings()
        transaction = OfficialStyleFP32SequentialTransaction(
            bindings,
            mode=FP32TransactionMode.SHADOW,
            transaction_id="compute-ledger",
        )
        for layer, update in self.updates().items():
            transaction.apply_layer_fp32(layer, update)
        receipt = transaction.finish_shadow()
        self.assertEqual(receipt.model_forward_count, 0)
        self.assertEqual(receipt.model_backward_count, 0)
        self.assertEqual(receipt.candidate_materialization_count, 0)
        self.assertEqual(receipt.external_materializer_call_count, 0)


if __name__ == "__main__":
    unittest.main()
