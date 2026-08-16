from __future__ import annotations

import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.functional import WaypointFactor, tensor_sha256
from project.run_scripts.ode_bf.p1_backend import (
    CandidateBF16FunctionalTrial,
    CovarianceActionReceipt,
    FULL_CURRENT_RESIDUAL_DEFINITION,
    FULL_CURRENT_RESIDUAL_DIVISOR,
    P1DynamicField,
    P1LayerField,
    PinnedCovarianceRegistry,
    SignedProgressReceipt,
    full_current_residual,
)
from project.run_scripts.ode_bf.p1_controller import (
    AcceptedLayerContribution,
    P1ControllerLock,
    assert_projector_only_difference,
    build_p1_routing_problem,
    project_shared_raw_velocity,
    rebind_shared_problem,
    solve_matched_raw_velocity,
)
from project.run_scripts.ode_bf.woodbury import (
    ProjectorCertificate,
    WoodburyCertificate,
    WoodburyMethod,
)


class _TwoLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(7, 6, bias=False, dtype=torch.bfloat16)
        self.second = torch.nn.Linear(6, 5, bias=False, dtype=torch.bfloat16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(value))


class CandidateAndCovarianceTests(unittest.TestCase):
    def test_complete_bf16_candidate_trial_is_exact_and_pure(self) -> None:
        torch.manual_seed(4101)
        model = _TwoLinear()
        inputs = torch.randn(3, 7).to(torch.bfloat16)
        candidates = {
            "first.weight": (model.first.weight.float() + 0.02).to(torch.bfloat16),
            "second.weight": (model.second.weight.float() - 0.03).to(torch.bfloat16),
        }
        pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
        versions = {name: value._version for name, value in model.named_parameters()}
        rng = torch.get_rng_state().clone()
        expected = torch.nn.functional.linear(
            torch.nn.functional.linear(inputs, candidates["first.weight"]),
            candidates["second.weight"],
        )
        with CandidateBF16FunctionalTrial(model, candidates) as trial:
            observed = model(inputs)
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(trial.max_live_candidate_weights, 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(
            {name: value.data_ptr() for name, value in model.named_parameters()}, pointers
        )
        self.assertEqual(
            {name: value._version for name, value in model.named_parameters()}, versions
        )

    def test_streamed_pinned_covariance_action_matches_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "stats.npz"
            matrix = np.arange(64, dtype=np.float32).reshape(8, 8) / np.float32(100.0)
            count = np.asarray(25, dtype=np.int64)
            np.savez(stats, **{"mom2.mom2": matrix, "mom2.count": count})
            digest = hashlib.sha256(stats.read_bytes()).hexdigest()
            registry = PinnedCovarianceRegistry(
                easyedit_root=root,
                covariance_spec={4: ("stats.npz", stats.stat().st_size, digest)},
            )
            q = torch.arange(80, dtype=torch.float32).reshape(10, 8).T
            pointer = q.data_ptr()
            version = q._version
            action, gram, receipt = registry.action(4, q, row_block=3)
            expected = torch.from_numpy(matrix @ q.numpy() / np.float32(25.0))
            self.assertTrue(torch.allclose(action, expected, rtol=1.0e-6, atol=1.0e-7))
            self.assertTrue(
                torch.allclose(gram, q.double().T @ action.double(), rtol=0.0, atol=0.0)
            )
            self.assertEqual(receipt.matrix_shape, (8, 8))
            self.assertEqual(receipt.q_shape, (8, 10))
            self.assertEqual((q.data_ptr(), q._version), (pointer, version))


def _field() -> tuple[P1DynamicField, SignedProgressReceipt]:
    layers: list[P1LayerField] = []
    projector = ProjectorCertificate("1" * 64, 1.0, 1.0, "artifact-unverified", 1.0e-10)
    certificate = WoodburyCertificate(
        WoodburyMethod.GENERAL_LU,
        projector,
        10,
        2.0,
        1.0e-12,
        1.0e-12,
        True,
        True,
    )
    generator = torch.Generator().manual_seed(4102)
    for ordinal, layer in enumerate((4, 5, 6, 7, 8)):
        left = torch.randn(12, 10, generator=generator)
        right = torch.randn(11, 10, generator=generator)
        covariance_action = 0.01 * right
        factor = WaypointFactor(
            f"layers.{layer}.weight", layer, 0, 0, ordinal, 1.0, left, right
        )
        covariance_receipt = CovarianceActionReceipt(
            layer,
            str(layer) * 64,
            100,
            (11, 11),
            100000,
            (11, 10),
            tensor_sha256(right),
            tensor_sha256(covariance_action),
            tensor_sha256(right.double().T @ covariance_action.double()),
            True,
            0.0,
        )
        layers.append(
            P1LayerField(
                layer,
                factor.weight_name,
                right,
                right,
                left,
                FULL_CURRENT_RESIDUAL_DEFINITION,
                FULL_CURRENT_RESIDUAL_DIVISOR,
                right,
                factor,
                float(torch.sum((left.T @ left) * (right.T @ right))),
                covariance_action,
                right.double().T @ covariance_action.double(),
                covariance_receipt,
                certificate,
                torch.empty((12, 0), dtype=torch.float64),
            )
        )
    dynamic = P1DynamicField(
        0,
        "a" * 64,
        torch.randn(12, 10, generator=generator),
        torch.randn(12, 10, generator=generator),
        tuple(layers),
        "b" * 64,
        0,
        0,
    )
    signed = SignedProgressReceipt(
        dynamic.identity_sha256,
        (1.0, 0.8, 0.6, 0.4, 0.2),
        (),
        "c" * 64,
        10,
        100,
        True,
    )
    return dynamic, signed


class MatchedRoutingTests(unittest.TestCase):
    def test_full_current_residual_changes_only_legacy_preshare_scaling(self) -> None:
        generator = torch.Generator().manual_seed(4103)
        target = torch.randn((12, 10), generator=generator, dtype=torch.float32)
        current = torch.randn((12, 10), generator=generator, dtype=torch.float32)
        key = torch.randn((11, 10), generator=generator, dtype=torch.float32)
        q = torch.randn((11, 10), generator=generator, dtype=torch.float32)
        source_identity = {
            "target": tensor_sha256(target),
            "current": tensor_sha256(current),
            "key": tensor_sha256(key),
            "q": tensor_sha256(q),
        }
        full = full_current_residual(target, current)
        for layer_index, legacy_divisor in enumerate((5, 4, 3, 2, 1)):
            with self.subTest(layer_index=layer_index):
                legacy = full / float(legacy_divisor)
                observed = full_current_residual(target, current)
                self.assertTrue(torch.equal(observed, full))
                self.assertTrue(
                    torch.allclose(
                        legacy * float(legacy_divisor),
                        observed,
                        rtol=1.0e-7,
                        atol=1.0e-7,
                    )
                )
                if legacy_divisor > 1:
                    self.assertFalse(torch.equal(legacy, observed))
        self.assertEqual(
            {
                "target": tensor_sha256(target),
                "current": tensor_sha256(current),
                "key": tensor_sha256(key),
                "q": tensor_sha256(q),
            },
            source_identity,
        )

    def test_full_residual_receipt_locks_divisor_q_factor_and_arm_identity(self) -> None:
        field, _ = _field()
        for layer in field.layers:
            payload = layer.raw_free_payload()
            self.assertEqual(
                payload["residual_definition"], FULL_CURRENT_RESIDUAL_DEFINITION
            )
            self.assertEqual(
                payload["residual_divisor"], FULL_CURRENT_RESIDUAL_DIVISOR
            )
            self.assertEqual(
                payload["current_residual_sha256"], tensor_sha256(layer.residual)
            )
            self.assertEqual(payload["q_sha256"], tensor_sha256(layer.q))
            self.assertEqual(payload["factor_sha256"], layer.factor_identity())
            self.assertEqual(payload["layer_arm_sha256"], layer.arm_identity())
            self.assertGreater(payload["current_residual_frobenius_norm"], 0.0)

    def test_authorized_field_builders_have_no_remaining_layer_divisor(self) -> None:
        from project.run_scripts.ode_bf.p1_backend import (
            build_p1_dynamic_field,
            build_p1_frozen_field_from_capture,
        )

        for builder in (build_p1_dynamic_field, build_p1_frozen_field_from_capture):
            source = inspect.getsource(builder)
            self.assertIn("full_current_residual(", source)
            self.assertNotIn("len(layers) - layer_index", source)

    def test_controller_lock_includes_common_trust_ratio_without_alias_branch(self) -> None:
        lock = P1ControllerLock()
        self.assertEqual(lock.minimum_progress, 1.0e-8)
        self.assertEqual(lock.rho_accept, 0.1)
        self.assertEqual(lock.backtracking, (1.0, 0.5, 0.25))
        self.assertIn("rho_accept", inspect.getsource(P1ControllerLock.identity))

    def test_shared_raw_velocity_is_unchanged_by_arm_local_projection(self) -> None:
        field, signed = _field()
        lock = P1ControllerLock()
        layers = tuple(item.layer for item in field.layers)
        empty = {layer: () for layer in layers}
        load = {layer: 0.0 for layer in layers}
        source = build_p1_routing_problem(
            field, signed, accepted_by_layer=empty, committed_load_by_layer=load, lock=lock
        )
        raw = solve_matched_raw_velocity(source)
        self.assertGreaterEqual(
            raw.maximum_feasible_progress,
            source.problem.requested_progress,
        )
        accepted = {}
        for ordinal, item in enumerate(field.layers):
            tiny = WaypointFactor(
                item.weight_name,
                item.layer,
                0,
                0,
                ordinal,
                0.01,
                item.residual,
                item.q,
            )
            accepted[item.layer] = (AcceptedLayerContribution.from_field(item, tiny),)
        barrier = build_p1_routing_problem(
            field, signed, accepted_by_layer=accepted, committed_load_by_layer=load, lock=lock
        )
        rebound = rebind_shared_problem(source, barrier)
        self.assertTrue(np.array_equal(rebound.capacity_metric, source.problem.capacity_metric))
        self.assertTrue(np.array_equal(rebound.signed_progress, source.problem.signed_progress))
        projected = project_shared_raw_velocity(source, barrier, raw)
        self.assertEqual(projected.raw_velocity_sha256, raw.velocity_sha256)
        self.assertEqual(projected.result.raw_velocity_identity, raw.velocity_sha256)
        self.assertIsNotNone(projected.result.values)

    def test_nonpositive_progress_is_not_clipped_into_a_direction(self) -> None:
        field, signed = _field()
        signed = SignedProgressReceipt(
            field.identity_sha256,
            (-1.0, -0.5, 0.0, -0.1, -2.0),
            (4, 5, 6, 7, 8),
            "d" * 64,
            10,
            100,
            True,
        )
        with self.assertRaisesRegex(ODEBFContractError, "PROGRESS_INFEASIBLE"):
            build_p1_routing_problem(
                field,
                signed,
                accepted_by_layer={layer: () for layer in (4, 5, 6, 7, 8)},
                committed_load_by_layer={layer: 0.0 for layer in (4, 5, 6, 7, 8)},
                lock=P1ControllerLock(),
            )

    def test_source_guard_allows_only_projector_label_difference(self) -> None:
        common = {"field": "f" * 64, "raw": "r" * 64, "budget": 8}
        assert_projector_only_difference(
            {**common, "barrier_projector": "identity"},
            {**common, "barrier_projector": "cbf"},
        )
        with self.assertRaises(ODEBFContractError):
            assert_projector_only_difference(
                {**common, "barrier_projector": "identity"},
                {**common, "barrier_projector": "cbf", "budget": 7},
            )


if __name__ == "__main__":
    unittest.main()
