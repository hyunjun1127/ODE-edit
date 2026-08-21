from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_update_binding as binding_module,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_fp32_transaction import (
    FP32TransactionMode,
    OfficialStyleFP32SequentialTransaction,
    RESIDUAL_RESERVE_LAYER_ORDER,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_update_binding import (
    EXACT_SINGLE_CONSTRUCTION_STATUS,
    UpdateMatchOrientation,
    apply_prepared_low_rank_update,
    build_prepared_low_rank_update,
)


class ResidualReserveUpdateBindingTests(unittest.TestCase):
    @staticmethod
    def build(layer: int, *, transpose: bool = False, scale: float = 1.0):
        rows, columns = (4, 3) if transpose else (3, 4)
        left = scale * torch.arange(1, rows + 1, dtype=torch.float32).reshape(-1, 1)
        right = torch.linspace(0.1, 0.1 * columns, columns).reshape(-1, 1)
        return left, right, build_prepared_low_rank_update(
            construction_id=f"construction-{layer}-{'t' if transpose else 'd'}",
            layer=layer,
            weight_name=f"model.layers.{layer}.mlp.down_proj.weight",
            parameter_shape=(3, 4),
            beta_applied_left32=left,
            q_side32=right,
        )

    def test_direct_and_transpose_match_official_and_exact_factor_product(self) -> None:
        for transpose, expected in (
            (False, UpdateMatchOrientation.DIRECT.value),
            (True, UpdateMatchOrientation.TRANSPOSE_VIEW.value),
        ):
            left, right, result = self.build(4, transpose=transpose)
            raw = left @ right.T
            official = raw if raw.shape == (3, 4) else raw.T
            reconstructed = result.factor.left @ result.factor.right.T
            self.assertEqual(result.receipt.matched_orientation, expected)
            self.assertTrue(torch.equal(result.matched_update32, official))
            self.assertTrue(torch.equal(reconstructed, result.matched_update32))
            self.assertEqual(
                tensor_sha256(result.matched_update32),
                result.receipt.matched_update_sha256,
            )
            self.assertEqual(result.receipt.dense_construction_count, 1)
            self.assertEqual(result.receipt.second_dense_construction_count, 0)
            self.assertEqual(result.receipt.fp64_algorithm_tensor_count, 0)
            self.assertEqual(result.receipt.beta_application_count, 1)
            self.assertEqual(result.receipt.beta_reapplication_count, 0)
            self.assertEqual(
                result.receipt.raw_update_pointer,
                result.receipt.matched_update_pointer,
            )

    def test_exact_object_pointer_version_survive_m3a_application(self) -> None:
        parameters = {
            layer: (
                f"model.layers.{layer}.mlp.down_proj.weight",
                torch.nn.Parameter(
                    torch.full((3, 4), 0.25, dtype=torch.bfloat16),
                    requires_grad=False,
                ),
            )
            for layer in RESIDUAL_RESERVE_LAYER_ORDER
        }
        transaction = OfficialStyleFP32SequentialTransaction(
            parameters,
            mode=FP32TransactionMode.AUTHORITATIVE,
            transaction_id="exact-object",
        )
        applied = []
        for layer in RESIDUAL_RESERVE_LAYER_ORDER:
            _, _, construction = self.build(
                layer,
                transpose=layer == 4,
                scale=0.01 * layer,
            )
            pointer = int(construction.matched_update32.data_ptr())
            version = int(construction.matched_update32._version)
            item = apply_prepared_low_rank_update(transaction, construction)
            applied.append(item)
            self.assertEqual(
                int(construction.matched_update32.data_ptr()),
                pointer,
            )
            self.assertEqual(int(construction.matched_update32._version), version)
            self.assertEqual(item.exact_object_apply_count, 1)
            self.assertEqual(
                item.m3a_layer_receipt.pre_cast_fp32_update_sha256,
                construction.receipt.matched_update_sha256,
            )
            self.assertEqual(
                item.raw_free_payload()["equivalence_status"],
                EXACT_SINGLE_CONSTRUCTION_STATUS,
            )
        transaction.prepare_authoritative_commit()
        transaction.abort_and_rollback()
        self.assertEqual(len(applied), 5)

    def test_inputs_immutable_factor_nonalias_and_receipts_raw_free(self) -> None:
        left, right, result = self.build(5)
        left_guard = (int(left.data_ptr()), int(left._version), tensor_sha256(left))
        right_guard = (int(right.data_ptr()), int(right._version), tensor_sha256(right))
        self.assertEqual(
            left_guard,
            (int(left.data_ptr()), int(left._version), tensor_sha256(left)),
        )
        self.assertEqual(
            right_guard,
            (int(right.data_ptr()), int(right._version), tensor_sha256(right)),
        )
        self.assertNotEqual(int(result.factor.left.data_ptr()), int(left.data_ptr()))
        self.assertNotEqual(int(result.factor.right.data_ptr()), int(right.data_ptr()))
        self.assertFalse(
            any(
                isinstance(value, torch.Tensor)
                for value in result.receipt.raw_free_payload().values()
            )
        )
        with self.assertRaises(FrozenInstanceError):
            result.receipt.layer = 6

    def test_invalid_shape_dtype_and_nonfinite_fail_closed(self) -> None:
        good = torch.ones((3, 1), dtype=torch.float32)
        other = torch.ones((4, 1), dtype=torch.float32)
        base = dict(
            construction_id="bad",
            layer=4,
            weight_name="model.layers.4.mlp.down_proj.weight",
            parameter_shape=(3, 4),
            beta_applied_left32=good,
            q_side32=other,
        )
        with self.assertRaises((ValueError, ODEBFContractError)):
            build_prepared_low_rank_update(**{**base, "parameter_shape": (2, 2)})
        with self.assertRaises(ODEBFContractError):
            build_prepared_low_rank_update(
                **{**base, "beta_applied_left32": good.to(torch.float64)}
            )
        bad = good.clone()
        bad[0, 0] = float("nan")
        with self.assertRaises(ODEBFContractError):
            build_prepared_low_rank_update(
                **{**base, "beta_applied_left32": bad}
            )

    def test_prohibited_paths_and_compute_contract(self) -> None:
        source = inspect.getsource(binding_module)
        for prohibited in (
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "CumulativeBF16FunctionalTrial",
            "assemble_effective_bf16",
            "effective_bf16_sha256",
            "p1r52_pir",
            "p1r52_fpiq",
        ):
            self.assertNotIn(prohibited, source)
        _, _, result = self.build(8)
        payload = result.receipt.raw_free_payload()
        self.assertEqual(payload["dense_construction_count"], 1)
        self.assertEqual(payload["second_dense_construction_count"], 0)
        self.assertEqual(payload["fp64_algorithm_tensor_count"], 0)
        self.assertEqual(payload["postcast_decision_influence_count"], 0)


if __name__ == "__main__":
    unittest.main()
