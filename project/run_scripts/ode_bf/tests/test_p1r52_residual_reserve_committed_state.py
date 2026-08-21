from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import inspect
import math
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError, ODEBFStateError
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_committed_state as state_module,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_committed_state import (
    CommittedGrossLoadLedger,
    LayerCommittedStateAnchor,
    bind_authoritative_transaction_factors,
    initialize_committed_gross_load_state,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_fp32_transaction import (
    FP32PreparedCommitReceipt,
    FP32TransactionMode,
    FP32TransactionReceipt,
    OfficialStyleFP32SequentialTransaction,
    RESIDUAL_RESERVE_LAYER_ORDER,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_pc_inventory import (
    LowRankFP32Factor,
    LowRankFactorSource,
    SealedPrevalidatedCovariance,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_update_binding import (
    EXACT_SINGLE_CONSTRUCTION_STATUS,
    AppliedPreparedLowRankUpdate,
    PreparedLowRankUpdate,
    apply_prepared_low_rank_update,
    build_prepared_low_rank_update,
)


class ResidualReserveCommittedStateTests(unittest.TestCase):
    PARAMETER_SHAPE = (3, 4)

    @staticmethod
    def covariance(layer: int) -> SealedPrevalidatedCovariance:
        value = torch.diag(
            torch.tensor(
                [1.0, 1.2, 1.4, 1.6],
                dtype=torch.float32,
            )
            + 0.01 * (layer - 4)
        )
        return SealedPrevalidatedCovariance(
            layer,
            value,
            f"sealed-covariance-{layer}",
        )

    def anchors(self):
        return tuple(
            LayerCommittedStateAnchor(
                layer,
                self.covariance(layer),
                10.0 + layer,
            )
            for layer in RESIDUAL_RESERVE_LAYER_ORDER
        )

    @staticmethod
    def constructions(scale: float = 1.0):
        result = []
        for ordinal, layer in enumerate(RESIDUAL_RESERVE_LAYER_ORDER):
            left = scale * torch.tensor(
                [
                    [0.5 + 0.05 * ordinal],
                    [-0.25 + 0.02 * ordinal],
                    [0.75 - 0.03 * ordinal],
                ],
                dtype=torch.float32,
            )
            right = torch.tensor(
                [
                    [0.4],
                    [-0.3 + 0.01 * ordinal],
                    [0.2 + 0.02 * ordinal],
                    [0.1],
                ],
                dtype=torch.float32,
            )
            result.append(
                build_prepared_low_rank_update(
                    construction_id=f"construction-{scale}-{layer}",
                    layer=layer,
                    weight_name=f"model.layers.{layer}.mlp.down_proj.weight",
                    parameter_shape=(
                        ResidualReserveCommittedStateTests.PARAMETER_SHAPE
                    ),
                    beta_applied_left32=left,
                    q_side32=right,
                )
            )
        return tuple(result)

    @staticmethod
    def factors(scale: float = 1.0):
        return tuple(
            item.factor
            for item in ResidualReserveCommittedStateTests.constructions(scale)
        )

    @staticmethod
    def prepared_transaction(
        constructions: tuple[PreparedLowRankUpdate, ...],
        transaction_id: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ):
        parameters = {
            layer: (
                f"model.layers.{layer}.mlp.down_proj.weight",
                torch.nn.Parameter(
                    torch.full(
                        ResidualReserveCommittedStateTests.PARAMETER_SHAPE,
                        0.25 + 0.1 * (layer - 4),
                        dtype=dtype,
                    ),
                    requires_grad=False,
                ),
            )
            for layer in RESIDUAL_RESERVE_LAYER_ORDER
        }
        transaction = OfficialStyleFP32SequentialTransaction(
            parameters,
            mode=FP32TransactionMode.AUTHORITATIVE,
            transaction_id=transaction_id,
        )
        applied = tuple(
            apply_prepared_low_rank_update(transaction, construction)
            for construction in constructions
        )
        prepared = transaction.prepare_authoritative_commit()
        return parameters, transaction, prepared, applied

    @staticmethod
    def parameter_snapshot(parameters):
        return {
            layer: (
                int(parameter.data_ptr()),
                parameter.detach().clone(),
            )
            for layer, (_, parameter) in parameters.items()
        }

    def assert_parameter_snapshot(self, parameters, snapshot) -> None:
        for layer, (_, parameter) in parameters.items():
            self.assertEqual(int(parameter.data_ptr()), snapshot[layer][0])
            self.assertTrue(torch.equal(parameter, snapshot[layer][1]))

    def ledger(self, state_id: str = "case-01") -> CommittedGrossLoadLedger:
        return CommittedGrossLoadLedger(
            initialize_committed_gross_load_state(
                self.anchors(),
                state_id=state_id,
            )
        )

    @staticmethod
    def contains_tensor(value) -> bool:
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(
                ResidualReserveCommittedStateTests.contains_tensor(item)
                for item in value.values()
            )
        if isinstance(value, (list, tuple)):
            return any(
                ResidualReserveCommittedStateTests.contains_tensor(item)
                for item in value
            )
        return False

    def stage(
        self,
        ledger: CommittedGrossLoadLedger,
        constructions: tuple[PreparedLowRankUpdate, ...],
        transaction_id: str,
    ):
        parameters, transaction, prepared, applied = self.prepared_transaction(
            constructions,
            transaction_id,
        )
        stage = ledger.stage_authoritative_commit(
            transaction,
            prepared,
            applied,
        )
        return parameters, transaction, prepared, stage

    @staticmethod
    def commit(ledger, transaction, prepared, stage):
        preview = ledger.preview_staged_publication(
            stage.stage_identity,
            prepared.identity_sha256,
        )
        return ledger.commit_staged_with_transaction(
            transaction,
            stage.stage_identity,
            prepared.identity_sha256,
            preview.identity_sha256,
        )

    def test_w0_state_is_zero_immutable_and_independent_reset(self) -> None:
        first = self.ledger("same-case").state
        second = self.ledger("same-case").state
        self.assertEqual(first.identity_sha256, second.identity_sha256)
        self.assertEqual(first.version, 0)
        self.assertEqual(first.commit_count, 0)
        self.assertEqual(first.committed_transaction_ids, ())
        for layer in first.layers:
            self.assertEqual(layer.committed_precast_factors, ())
            self.assertEqual(layer.structural_p_constant, 0.0)
            self.assertEqual(layer.cumulative_precast_lambda, 0.0)
            self.assertEqual(
                layer.cumulative_actual_post_storage_load_observation,
                0.0,
            )
        self.assertFalse(self.contains_tensor(first.raw_free_payload()))
        with self.assertRaises(FrozenInstanceError):
            first.version = 1

    def test_recursive_p_and_lambda_match_dense_two_commit_fixture(self) -> None:
        ledger = self.ledger()
        committed: list[tuple[LowRankFP32Factor, ...]] = []
        for ordinal, constructions in enumerate(
            (self.constructions(1.0), self.constructions(-0.4))
        ):
            _, transaction, prepared, stage = self.stage(
                ledger,
                constructions,
                f"authoritative-{ordinal}",
            )
            result = self.commit(ledger, transaction, prepared, stage)
            committed.append(tuple(item.factor for item in constructions))
            self.assertEqual(result.state.version, ordinal + 1)
            self.assertEqual(result.state.commit_count, ordinal + 1)
            self.assertEqual(result.receipt.factor_append_count, 5)
            self.assertEqual(result.receipt.logical_commit_count, 1)
            for layer_index, layer_state in enumerate(result.state.layers):
                self.assertTrue(
                    all(
                        factor.source
                        is LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32
                        for factor in layer_state.committed_precast_factors
                    )
                )
                self.assertEqual(
                    len(layer_state.committed_precast_factors),
                    ordinal + 1,
                )
                dense = sum(
                    (
                        factor_set[layer_index].left
                        @ factor_set[layer_index].right.T
                        for factor_set in committed
                    ),
                    torch.zeros(self.PARAMETER_SHAPE, dtype=torch.float32),
                )
                covariance = layer_state.covariance.value
                expected_p = float(torch.trace(dense @ covariance @ dense.T))
                expected_lambda = sum(
                    float(
                        torch.sum(
                            (factor_set[layer_index].left
                            @ factor_set[layer_index].right.T)
                            ** 2
                        )
                    )
                    / layer_state.pretrained_weight_norm_squared
                    for factor_set in committed
                )
                self.assertAlmostEqual(
                    layer_state.structural_p_constant,
                    expected_p,
                    places=6,
                )
                self.assertAlmostEqual(
                    layer_state.cumulative_precast_lambda,
                    expected_lambda,
                    places=7,
                )
                layer_receipt = result.receipt.layer_receipts[layer_index]
                self.assertAlmostEqual(
                    layer_receipt.p_increment,
                    2.0 * layer_receipt.p_cross + layer_receipt.p_self,
                    places=14,
                )

    def test_stage_excludes_proposal_and_abort_restores_exact_state(self) -> None:
        ledger = self.ledger()
        before = ledger.state
        before_payload = before.raw_free_payload()
        parameters, _, _, stage = self.stage(
            ledger,
            self.constructions(),
            "abort-me",
        )
        live_endpoint = self.parameter_snapshot(parameters)
        self.assertIs(ledger.state, before)
        self.assertEqual(ledger.state.raw_free_payload(), before_payload)
        self.assertEqual(stage.current_state_mutation_count, 0)
        self.assertEqual(stage.proposal_decision_influence_count_before_commit, 0)
        receipt = ledger.abort_staged(stage.stage_identity)
        self.assertIs(ledger.state, before)
        self.assertEqual(receipt.before_state_identity, receipt.after_state_identity)
        self.assertEqual(
            receipt.before_decision_identity,
            receipt.after_decision_identity,
        )
        self.assertEqual(receipt.state_mutation_count, 0)
        self.assertEqual(receipt.rollback_count, 1)
        self.assertEqual(
            sum(
                len(layer.committed_precast_factors)
                for layer in ledger.state.layers
            ),
            0,
        )
        self.assertTrue(
            any(
                not torch.equal(parameters[layer][1], live_endpoint[layer][1])
                for layer in RESIDUAL_RESERVE_LAYER_ORDER
            )
        )

    def test_authoritative_complete_gate_rejects_shadow_partial_and_failed(self) -> None:
        constructions = self.constructions()
        shadow_parameters = {
            layer: (
                f"model.layers.{layer}.mlp.down_proj.weight",
                torch.nn.Parameter(
                    torch.zeros(self.PARAMETER_SHAPE, dtype=torch.bfloat16),
                    requires_grad=False,
                ),
            )
            for layer in RESIDUAL_RESERVE_LAYER_ORDER
        }
        shadow = OfficialStyleFP32SequentialTransaction(
            shadow_parameters,
            mode=FP32TransactionMode.SHADOW,
            transaction_id="shadow",
        )
        for construction in constructions:
            apply_prepared_low_rank_update(shadow, construction)
        with self.assertRaises(ODEBFContractError):
            shadow.prepare_authoritative_commit()

        _, transaction, valid, applied = self.prepared_transaction(
            constructions,
            "valid",
        )
        partial = replace(
            valid,
            layer_receipts=valid.layer_receipts[:-1],
            native_storage_assignment_count=4,
            storage_cast_boundary_count=4,
        )
        failed = replace(valid, rollback_count=1)
        forged_numeric_cast = replace(
            valid,
            numeric_storage_cast_count=valid.numeric_storage_cast_count + 1,
        )
        forged_rounding_count = replace(
            valid,
            rounding_mismatch_count=valid.rounding_mismatch_count + 1,
        )
        for receipt in (
            partial,
            failed,
            forged_numeric_cast,
            forged_rounding_count,
        ):
            with self.assertRaises(ODEBFContractError):
                bind_authoritative_transaction_factors(receipt, applied)

        nominal = tuple(
            replace(
                item,
                construction=replace(
                    item.construction,
                    factor=LowRankFP32Factor(
                        item.construction.factor.layer,
                        item.construction.factor.parameter_shape,
                        item.construction.factor.left,
                        item.construction.factor.right,
                        LowRankFactorSource.NOMINAL_REFERENCE_PRECAST_FP32,
                    ),
                ),
            )
            for item in applied
        )
        with self.assertRaises((ODEBFContractError, ODEBFStateError)):
            bind_authoritative_transaction_factors(valid, nominal)
        transaction.abort_and_rollback()

    def test_prelabeled_successful_factor_stage_rejected_and_restored(self) -> None:
        constructions = self.constructions()
        parameters, transaction, prepared, applied = self.prepared_transaction(
            constructions,
            "false-success",
        )
        successful = tuple(
            replace(
                item,
                construction=replace(
                    item.construction,
                    factor=LowRankFP32Factor(
                        item.construction.factor.layer,
                        item.construction.factor.parameter_shape,
                        item.construction.factor.left,
                        item.construction.factor.right,
                        LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32,
                    ),
                ),
            )
            for item in applied
        )
        ledger = self.ledger()
        before = ledger.state
        entry = {
            layer: (
                int(parameter.data_ptr()),
                torch.full_like(parameter, 0.25 + 0.1 * (layer - 4)),
            )
            for layer, (_, parameter) in parameters.items()
        }
        with self.assertRaises((ODEBFContractError, ODEBFStateError)):
            ledger.stage_authoritative_commit(
                transaction,
                prepared,
                successful,
            )
        self.assert_parameter_snapshot(parameters, entry)
        self.assertIs(ledger.state, before)
        self.assertEqual(ledger.state.version, 0)
        self.assertEqual(
            sum(
                len(layer.committed_precast_factors)
                for layer in ledger.state.layers
            ),
            0,
        )

    def test_duplicate_out_of_order_and_reused_stage_fail_closed(self) -> None:
        ledger = self.ledger()
        parameters, transaction, prepared, applied = self.prepared_transaction(
            self.constructions(),
            "bad-order",
        )
        entry = {
            layer: (
                int(parameter.data_ptr()),
                torch.full_like(parameter, 0.25 + 0.1 * (layer - 4)),
            )
            for layer, (_, parameter) in parameters.items()
        }
        before_identity = ledger.state.identity_sha256
        with self.assertRaises(ODEBFContractError):
            ledger.stage_authoritative_commit(
                transaction,
                prepared,
                tuple(reversed(applied)),
            )
        self.assert_parameter_snapshot(parameters, entry)
        self.assertEqual(ledger.state.identity_sha256, before_identity)

        parameters, transaction, prepared, _ = self.stage(
            ledger,
            self.constructions(),
            "duplicate-stage",
        )
        entry = {
            layer: (
                int(parameter.data_ptr()),
                torch.full_like(parameter, 0.25 + 0.1 * (layer - 4)),
            )
            for layer, (_, parameter) in parameters.items()
        }
        with self.assertRaises(ODEBFStateError):
            ledger.stage_authoritative_commit(
                transaction,
                prepared,
                applied,
            )
        self.assert_parameter_snapshot(parameters, entry)
        self.assertEqual(ledger.state.identity_sha256, before_identity)

        _, transaction, prepared, stage = self.stage(
            ledger,
            self.constructions(),
            "once",
        )
        result = self.commit(ledger, transaction, prepared, stage)
        committed_identity = ledger.state.identity_sha256
        with self.assertRaises(ODEBFStateError):
            ledger.commit_staged_with_transaction(
                transaction,
                stage.stage_identity,
                prepared.identity_sha256,
                "reused-preview-identity",
            )
        self.assertEqual(result.state.version, 1)

        _, reused_transaction, reused_prepared, reused_applied = (
            self.prepared_transaction(
                self.constructions(),
                "once",
            )
        )
        with self.assertRaises(ODEBFStateError):
            ledger.stage_authoritative_commit(
                reused_transaction,
                reused_prepared,
                reused_applied,
            )
        self.assertEqual(ledger.state.identity_sha256, committed_identity)

    def test_valid_prepare_stage_coordinated_commit_binds_all_identities(self) -> None:
        ledger = self.ledger()
        constructions = self.constructions()
        _, transaction, prepared, stage = self.stage(
            ledger,
            constructions,
            "coordinated",
        )
        before = ledger.state
        preview = ledger.preview_staged_publication(
            stage.stage_identity,
            prepared.identity_sha256,
        )
        result = self.commit(ledger, transaction, prepared, stage)
        self.assertTrue(transaction.finalized)
        self.assertIs(transaction.final_receipt, prepared.future_final_receipt)
        self.assertEqual(result.state.version, 1)
        self.assertIs(ledger.state, result.state)
        self.assertIsNot(ledger.state, before)
        self.assertEqual(
            stage.prepared_commit_identity,
            prepared.identity_sha256,
        )
        self.assertEqual(
            stage.future_final_transaction_receipt_identity,
            prepared.future_final_receipt.identity_sha256,
        )
        self.assertEqual(
            result.receipt.prepared_commit_identity,
            prepared.identity_sha256,
        )
        self.assertEqual(
            result.receipt.final_transaction_receipt_identity,
            result.transaction_receipt.identity_sha256,
        )
        self.assertEqual(
            preview.ledger_commit_receipt_identity,
            result.receipt.identity_sha256,
        )
        self.assertEqual(preview.after_state_identity, result.state.identity_sha256)
        self.assertEqual(
            preview.after_decision_identity,
            result.state.decision_identity_sha256,
        )
        for construction, layer_state, layer_receipt in zip(
            constructions,
            result.state.layers,
            result.receipt.layer_receipts,
            strict=True,
        ):
            self.assertEqual(len(layer_state.committed_precast_factors), 1)
            committed_factor = layer_state.committed_precast_factors[0]
            self.assertIs(
                committed_factor.source,
                LowRankFactorSource.SUCCESSFUL_COMMITTED_PRECAST_FP32,
            )
            self.assertEqual(
                tensor_sha256(construction.factor.left),
                tensor_sha256(committed_factor.left),
            )
            self.assertEqual(
                tensor_sha256(construction.factor.right),
                tensor_sha256(committed_factor.right),
            )
            self.assertNotEqual(
                int(construction.factor.left.data_ptr()),
                int(committed_factor.left.data_ptr()),
            )
            self.assertNotEqual(
                int(construction.factor.right.data_ptr()),
                int(committed_factor.right.data_ptr()),
            )
            self.assertEqual(layer_receipt.provenance_transition_count, 1)
            self.assertEqual(
                layer_receipt.prepared_factor_source,
                "AUTHORITATIVE_PREPARED_PRECAST_FP32",
            )
            self.assertEqual(
                layer_receipt.committed_factor_source,
                "SUCCESSFUL_COMMITTED_PRECAST_FP32",
            )
            self.assertTrue(
                layer_receipt.source_transition_numerical_tensor_byte_identity
            )
            self.assertTrue(
                layer_receipt.source_transition_numerical_tensor_nonalias
            )
            self.assertEqual(
                layer_receipt.factor_update_equivalence_status,
                EXACT_SINGLE_CONSTRUCTION_STATUS,
            )
            self.assertEqual(
                layer_receipt.construction_receipt_identity,
                construction.receipt.identity_sha256,
            )
            self.assertEqual(
                layer_state.committed_construction_receipt_ids,
                (construction.receipt.identity_sha256,),
            )

    def test_publication_preview_is_primitive_immutable_and_identity_gated(self) -> None:
        ledger = self.ledger("preview-gate")
        parameters, transaction, prepared, stage = self.stage(
            ledger,
            self.constructions(),
            "preview-first",
        )
        before = ledger.state
        entry = {
            layer: (
                int(parameter.data_ptr()),
                torch.full_like(parameter, 0.25 + 0.1 * (layer - 4)),
            )
            for layer, (_, parameter) in parameters.items()
        }
        preview = ledger.preview_staged_publication(
            stage.stage_identity,
            prepared.identity_sha256,
        )
        self.assertFalse(self.contains_tensor(preview.raw_free_payload()))
        self.assertEqual(preview.layers, RESIDUAL_RESERVE_LAYER_ORDER)
        self.assertEqual(preview.factor_append_counts, (1,) * 5)
        self.assertEqual(preview.native_storage_assignment_count, 5)
        with self.assertRaises(FrozenInstanceError):
            preview.after_version = 99
        with self.assertRaises(ODEBFStateError):
            ledger.commit_staged_with_transaction(
                transaction,
                stage.stage_identity,
                prepared.identity_sha256,
                "0" * 64,
            )
        self.assert_parameter_snapshot(parameters, entry)
        self.assertIs(ledger.state, before)
        self.assertEqual(ledger.state.version, 0)

        parameters, transaction, prepared, stage = self.stage(
            ledger,
            self.constructions(),
            "preview-second",
        )
        entry = {
            layer: (
                int(parameter.data_ptr()),
                torch.full_like(parameter, 0.25 + 0.1 * (layer - 4)),
            )
            for layer, (_, parameter) in parameters.items()
        }
        with self.assertRaises(ODEBFStateError):
            ledger.commit_staged_with_transaction(
                transaction,
                stage.stage_identity,
                prepared.identity_sha256,
                preview.identity_sha256,
            )
        self.assert_parameter_snapshot(parameters, entry)
        self.assertIs(ledger.state, before)
        self.assertEqual(ledger.state.version, 0)

    def test_corrupt_construction_fields_fail_closed_with_exact_w_and_ledger(self) -> None:
        def corrupt(
            applied: tuple[AppliedPreparedLowRankUpdate, ...],
            field: str,
            value,
        ) -> tuple[AppliedPreparedLowRankUpdate, ...]:
            first = applied[0]
            altered_receipt = replace(
                first.construction.receipt,
                **{field: value},
            )
            altered = replace(
                first,
                construction=replace(
                    first.construction,
                    receipt=altered_receipt,
                ),
            )
            return (altered,) + applied[1:]

        cases = (
            ("layer", 5),
            ("parameter_shape", (2, 6)),
            ("matched_orientation", "TRANSPOSE_VIEW"),
            ("prepared_factor_identity", "0" * 64),
            ("matched_update_sha256", "1" * 64),
            ("matched_update_pointer", -1),
            ("matched_update_version", 99),
        )
        for field, value in cases:
            with self.subTest(field=field):
                ledger = self.ledger(f"corrupt-{field}")
                before = ledger.state
                parameters, transaction, prepared, applied = (
                    self.prepared_transaction(
                        self.constructions(),
                        f"corrupt-{field}",
                    )
                )
                entry = {
                    layer: (
                        int(parameter.data_ptr()),
                        torch.full_like(
                            parameter,
                            0.25 + 0.1 * (layer - 4),
                        ),
                    )
                    for layer, (_, parameter) in parameters.items()
                }
                with self.assertRaises((ODEBFContractError, ODEBFStateError)):
                    ledger.stage_authoritative_commit(
                        transaction,
                        prepared,
                        corrupt(applied, field, value),
                    )
                self.assert_parameter_snapshot(parameters, entry)
                self.assertIs(ledger.state, before)
                self.assertEqual(ledger.state.version, 0)

    def test_reused_construction_receipt_rejected_and_restored(self) -> None:
        constructions = self.constructions()
        ledger = self.ledger("receipt-reuse")
        _, first_transaction, first_prepared, first_stage = self.stage(
            ledger,
            constructions,
            "receipt-first",
        )
        self.commit(ledger, first_transaction, first_prepared, first_stage)
        committed_identity = ledger.state.identity_sha256

        parameters, transaction, prepared, applied = self.prepared_transaction(
            constructions,
            "receipt-second",
        )
        entry = {
            layer: (
                int(parameter.data_ptr()),
                torch.full_like(parameter, 0.25 + 0.1 * (layer - 4)),
            )
            for layer, (_, parameter) in parameters.items()
        }
        with self.assertRaises(ODEBFContractError):
            ledger.stage_authoritative_commit(transaction, prepared, applied)
        self.assert_parameter_snapshot(parameters, entry)
        self.assertEqual(ledger.state.identity_sha256, committed_identity)
        self.assertEqual(ledger.state.version, 1)

    def test_postapply_update_mutation_rejected_and_restored(self) -> None:
        constructions = self.constructions()
        ledger = self.ledger("update-mutation")
        before = ledger.state
        parameters, transaction, prepared, applied = self.prepared_transaction(
            constructions,
            "update-mutation",
        )
        entry = {
            layer: (
                int(parameter.data_ptr()),
                torch.full_like(parameter, 0.25 + 0.1 * (layer - 4)),
            )
            for layer, (_, parameter) in parameters.items()
        }
        constructions[0].matched_update32.add_(1.0)
        with self.assertRaises(ODEBFStateError):
            ledger.stage_authoritative_commit(transaction, prepared, applied)
        self.assert_parameter_snapshot(parameters, entry)
        self.assertIs(ledger.state, before)
        self.assertEqual(ledger.state.version, 0)

    def test_legacy_ledger_only_publish_is_disabled_and_restores_w(self) -> None:
        ledger = self.ledger()
        before = ledger.state
        parameters, _, _, stage = self.stage(
            ledger,
            self.constructions(),
            "ledger-only-disabled",
        )
        entry = {
            layer: (
                int(parameter.data_ptr()),
                torch.full_like(parameter, 0.25 + 0.1 * (layer - 4)),
            )
            for layer, (_, parameter) in parameters.items()
        }
        with self.assertRaises(ODEBFStateError):
            ledger.commit_staged(stage.stage_identity)
        self.assertIs(ledger.state, before)
        self.assertEqual(ledger.state.version, 0)
        self.assert_parameter_snapshot(parameters, entry)

    def test_poststage_live_hash_corruption_restores_w_and_aborts_ledger(self) -> None:
        ledger = self.ledger()
        before_state = ledger.state
        before_identity = before_state.identity_sha256
        parameters, transaction, prepared, stage = self.stage(
            ledger,
            self.constructions(),
            "poststage-corruption",
        )
        entry = {
            layer: (
                int(parameter.data_ptr()),
                torch.full_like(parameter, 0.25 + 0.1 * (layer - 4)),
            )
            for layer, (_, parameter) in parameters.items()
        }
        with torch.no_grad():
            parameters[7][1][0, 0] += 1.0
        with self.assertRaises(ODEBFStateError):
            self.commit(ledger, transaction, prepared, stage)
        self.assert_parameter_snapshot(parameters, entry)
        self.assertTrue(transaction.finalized)
        self.assertEqual(transaction.rollback_count, 1)
        self.assertIs(ledger.state, before_state)
        self.assertEqual(ledger.state.identity_sha256, before_identity)

    def test_wrong_prepared_identity_and_direct_final_receipt_rejected(self) -> None:
        ledger = self.ledger()
        parameters, transaction, prepared, applied = self.prepared_transaction(
            self.constructions(),
            "wrong-prepared",
        )
        entry = {
            layer: (
                int(parameter.data_ptr()),
                torch.full_like(parameter, 0.25 + 0.1 * (layer - 4)),
            )
            for layer, (_, parameter) in parameters.items()
        }
        wrong = replace(prepared, transaction_id="wrong")
        with self.assertRaises(ODEBFStateError):
            ledger.stage_authoritative_commit(
                transaction,
                wrong,
                applied,
            )
        self.assert_parameter_snapshot(parameters, entry)
        self.assertEqual(ledger.state.version, 0)

        _, finalized_transaction, finalized_prepared, finalized_applied = (
            self.prepared_transaction(
                self.constructions(),
                "direct-final",
            )
        )
        final_receipt = finalized_transaction.commit_prepared(
            finalized_prepared.identity_sha256
        )
        self.assertIsInstance(final_receipt, FP32TransactionReceipt)
        self.assertNotIsInstance(final_receipt, FP32PreparedCommitReceipt)
        with self.assertRaises(ODEBFContractError):
            ledger.stage_authoritative_commit(
                finalized_transaction,
                final_receipt,
                finalized_applied,
            )
        self.assertEqual(ledger.state.version, 0)

    def test_public_wildcard_export_resolves_anchor(self) -> None:
        namespace: dict[str, object] = {}
        exec(
            "from project.run_scripts.ode_bf."
            "p1r52_residual_reserve_committed_state import *",
            namespace,
        )
        self.assertIs(
            namespace["LayerCommittedStateAnchor"],
            LayerCommittedStateAnchor,
        )
        self.assertNotIn("CommittedLayerStateAnchor", namespace)

    def test_actual_post_observation_does_not_change_route_inputs(self) -> None:
        constructions = self.constructions()
        first = self.ledger("observation-pair")
        second = self.ledger("observation-pair")
        _, first_transaction, first_prepared, first_applied = (
            self.prepared_transaction(
                constructions,
                "observation-bf16",
            )
        )
        _, second_transaction, second_prepared, second_applied = (
            self.prepared_transaction(
                constructions,
                "observation-fp32",
                dtype=torch.float32,
            )
        )
        first_stage = first.stage_authoritative_commit(
            first_transaction,
            first_prepared,
            first_applied,
        )
        second_stage = second.stage_authoritative_commit(
            second_transaction,
            second_prepared,
            second_applied,
        )
        first_result = self.commit(
            first,
            first_transaction,
            first_prepared,
            first_stage,
        )
        second_result = self.commit(
            second,
            second_transaction,
            second_prepared,
            second_stage,
        )
        self.assertEqual(
            first_result.state.decision_identity_sha256,
            second_result.state.decision_identity_sha256,
        )
        self.assertNotEqual(
            first_result.state.identity_sha256,
            second_result.state.identity_sha256,
        )
        for left, right in zip(
            first_result.state.layers,
            second_result.state.layers,
            strict=True,
        ):
            self.assertEqual(left.structural_p_constant, right.structural_p_constant)
            self.assertEqual(
                left.cumulative_precast_lambda,
                right.cumulative_precast_lambda,
            )
            self.assertNotEqual(
                left.cumulative_actual_post_storage_load_observation,
                right.cumulative_actual_post_storage_load_observation,
            )

    def test_binding_receipt_binds_factor_update_and_actual_observation(self) -> None:
        constructions = self.constructions()
        _, transaction, prepared, applied = self.prepared_transaction(
            constructions,
            "binding",
        )
        bindings = bind_authoritative_transaction_factors(prepared, applied)
        for construction, layer_receipt, binding in zip(
            constructions,
            prepared.layer_receipts,
            bindings,
            strict=True,
        ):
            self.assertEqual(
                binding.prepared_factor_identity,
                construction.factor.identity_sha256,
            )
            self.assertEqual(
                binding.prepared_factor_source,
                "AUTHORITATIVE_PREPARED_PRECAST_FP32",
            )
            self.assertEqual(
                binding.committed_factor_source,
                "SUCCESSFUL_COMMITTED_PRECAST_FP32",
            )
            self.assertNotEqual(
                binding.prepared_factor_identity,
                binding.committed_factor_identity,
            )
            self.assertEqual(binding.provenance_transition_count, 1)
            self.assertEqual(
                binding.pre_cast_update_sha256,
                layer_receipt.pre_cast_fp32_update_sha256,
            )
            self.assertEqual(
                binding.actual_post_storage_delta_sha256,
                layer_receipt.actual_post_storage_delta32_sha256,
            )
            self.assertEqual(binding.transaction_id, prepared.transaction_id)
            self.assertEqual(
                binding.factor_update_equivalence_status,
                EXACT_SINGLE_CONSTRUCTION_STATUS,
            )
            self.assertEqual(
                binding.construction_receipt_identity,
                construction.receipt.identity_sha256,
            )
            self.assertEqual(
                binding.raw_free_payload()["second_dense_update_tensor_creation_count"],
                0,
            )
        transaction.abort_and_rollback()

    def test_state_and_receipts_are_raw_free_and_inputs_stay_immutable(self) -> None:
        ledger = self.ledger()
        constructions = self.constructions()
        identities = tuple(
            item.factor.identity_sha256 for item in constructions
        )
        _, transaction, prepared, stage = self.stage(
            ledger,
            constructions,
            "immutable",
        )
        result = self.commit(ledger, transaction, prepared, stage)
        self.assertEqual(
            identities,
            tuple(item.factor.identity_sha256 for item in constructions),
        )
        self.assertFalse(self.contains_tensor(result.state.raw_free_payload()))
        self.assertFalse(self.contains_tensor(result.receipt.raw_free_payload()))
        self.assertEqual(result.receipt.shadow_commit_count, 0)
        self.assertEqual(result.receipt.failed_commit_count, 0)
        self.assertEqual(result.receipt.duplicate_commit_count, 0)
        self.assertTrue(
            all(
                item.factor_append_count == 1
                and item.covariance_right_matmul_count == 1
                and item.dense_materialization_count == 0
                for item in result.receipt.layer_receipts
            )
        )

    def test_prohibited_paths_and_compute_ledger(self) -> None:
        source = inspect.getsource(state_module)
        for prohibited in (
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "CumulativeBF16FunctionalTrial",
            "assemble_effective_bf16",
            "effective_bf16_sha256",
            "PIR-U",
            "PIRU",
            "FPIQ",
            "p1r52_sequential_contract",
            "left @ right.T",
        ):
            self.assertNotIn(prohibited, source)
        ledger = self.ledger()
        _, transaction, prepared, stage = self.stage(
            ledger,
            self.constructions(),
            "compute",
        )
        result = self.commit(ledger, transaction, prepared, stage)
        self.assertEqual(result.receipt.model_forward_count, 0)
        self.assertEqual(result.receipt.model_backward_count, 0)
        self.assertEqual(result.receipt.dense_materialization_count, 0)
        self.assertEqual(result.receipt.post_storage_decision_influence_count, 0)
        self.assertTrue(
            all(
                math.isfinite(item.p_after)
                and math.isfinite(item.lambda_after)
                and math.isfinite(item.actual_load_observation_after)
                for item in result.receipt.layer_receipts
            )
        )


if __name__ == "__main__":
    unittest.main()
