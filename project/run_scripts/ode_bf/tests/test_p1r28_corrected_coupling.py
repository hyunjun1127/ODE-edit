from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError, ODEBFStateError
from project.run_scripts.ode_bf.p1r28_corrected_coupling import (
    AffineTrustDomain,
    AffineVelocitySlope,
    AppliedCoefficient,
    P1R24MatchedDemand,
    P1R28CouplingStatus,
    RawCoefficientSlope,
    VelocityCoefficient,
    VelocitySlope,
    coordinate_identity_receipt,
    largest_feasible_lag_scale,
    semantic_only_coupling,
    stall_state_receipt,
)
from project.run_scripts.ode_bf.p1_backend import (
    _validate_dynamic_factor_capacity,
    observe_dynamic_factor_capacity,
)
from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24RoutingStatus,
    p1r24_disable_historical,
    solve_p1r24_matched_routing,
)
from project.run_scripts.ode_bf.fixed_e8_soft_routing import FixedE8Arm
from project.run_scripts.ode_bf.routing import (
    QuadraticBarrier,
    ZeroActionRoutingProblem,
)
from project.run_scripts.ode_bf.scalable_batched_runtime import DynamicRefreshLedger


class P1R28CorrectedCouplingTests(unittest.TestCase):
    def domain(
        self,
        constant=(1.0, 1.0, 1.0, 1.0, 1.0),
        cross=(0.0, 0.0, 0.0, 0.0, 0.0),
        lag=(0.0, 0.0, 0.0, 0.0, 0.0),
    ) -> AffineTrustDomain:
        return AffineTrustDomain(constant, cross, lag, 1.0)

    def demand(self, semantic: float, lag: float) -> P1R24MatchedDemand:
        return P1R24MatchedDemand(semantic, lag, "d" * 64)

    def test_coordinate_identity_and_h_once(self) -> None:
        raw = RawCoefficientSlope((8.0, 4.0, 2.0, 1.0, 0.5))
        velocity = VelocityCoefficient((2.0, 1.0, 0.5, 0.25, 0.125))
        receipt = coordinate_identity_receipt(raw, velocity)
        self.assertEqual(receipt["ratio"], 1.0)
        self.assertEqual(receipt["h_application_count"], 1)
        self.assertEqual(receipt["second_h_division_count"], 0)
        self.assertAlmostEqual(
            raw.to_velocity().contract(velocity),
            raw.contract(velocity.to_applied()),
        )

    def test_illegal_coordinate_contractions_are_unrepresentable(self) -> None:
        with self.assertRaises(ODEBFContractError):
            VelocitySlope((1, 1, 1, 1, 1)).contract(  # type: ignore[arg-type]
                AppliedCoefficient((1, 1, 1, 1, 1))
            )
        with self.assertRaises(ODEBFContractError):
            RawCoefficientSlope((1, 1, 1, 1, 1)).contract(  # type: ignore[arg-type]
                VelocityCoefficient((1, 1, 1, 1, 1))
            )

    def test_full_lag_feasible_selects_one(self) -> None:
        slopes = AffineVelocitySlope(
            VelocitySlope((2, 2, 2, 2, 2)),
            VelocitySlope((0, 0, 0, 0, 0)),
        )
        selected = largest_feasible_lag_scale(
            slopes, self.demand(1.0, 1.0), self.domain()
        )
        self.assertEqual(selected.status, P1R28CouplingStatus.JOINT_WRITE)
        self.assertEqual(selected.lag_scale, 1.0)

    def test_tracking_conflict_rectified_at_exact_root(self) -> None:
        # rmax=sqrt(5)*max(1-2 lambda,0); rho=.5 is feasible at lambda0
        # but not lambda1.  The global polynomial solve must select an interior
        # boundary without assuming monotonicity or using a lambda grid.
        slopes = AffineVelocitySlope(
            VelocitySlope((1, 1, 1, 1, 1)),
            VelocitySlope((-2, -2, -2, -2, -2)),
        )
        selected = largest_feasible_lag_scale(
            slopes, self.demand(0.5, 0.0), self.domain()
        )
        self.assertEqual(
            selected.status, P1R28CouplingStatus.TRACKING_CONFLICT_RECTIFIED
        )
        self.assertGreater(selected.lag_scale, 0.0)
        self.assertLess(selected.lag_scale, 1.0)
        self.assertGreaterEqual(selected.r_max + 1.0e-8, selected.rho)

    def test_nonmonotone_feasibility_chooses_largest_component(self) -> None:
        slopes = AffineVelocitySlope(
            VelocitySlope((2.0, -1.0, 0.2, 0.2, 0.2)),
            VelocitySlope((-3.0, 4.0, 0.0, 0.0, 0.0)),
        )
        selected = largest_feasible_lag_scale(
            slopes, self.demand(2.0, 0.0), self.domain()
        )
        self.assertEqual(selected.lag_scale, 1.0)
        self.assertEqual(selected.status, P1R28CouplingStatus.JOINT_WRITE)

    def test_pure_semantic_feasible_cannot_be_destroyed_by_tracking(self) -> None:
        slopes = AffineVelocitySlope(
            VelocitySlope((1, 0, 0, 0, 0)),
            VelocitySlope((-2, 0, 0, 0, 0)),
        )
        selected = largest_feasible_lag_scale(
            slopes, self.demand(0.25, 0.0), self.domain()
        )
        self.assertNotEqual(selected.status, P1R28CouplingStatus.SEMANTIC_STALL)
        self.assertGreaterEqual(selected.lag_scale, 0.0)

    def test_semantic_no_direction_and_stall_totality(self) -> None:
        slopes = AffineVelocitySlope(
            VelocitySlope((-1, -1, -1, -1, -1)),
            VelocitySlope((0, 0, 0, 0, 0)),
        )
        selected = semantic_only_coupling(
            slopes, self.demand(1.0, 0.0), self.domain()
        )
        self.assertEqual(
            selected.status, P1R28CouplingStatus.SEMANTIC_NO_POSITIVE_DIRECTION
        )
        receipt = stall_state_receipt(
            w_before_sha256="w",
            w_after_sha256="w",
            z_before_sha256="z",
            z_after_sha256="z",
        )
        self.assertTrue(receipt["w_held"] and receipt["z_held"])
        with self.assertRaises(ODEBFContractError):
            stall_state_receipt(
                w_before_sha256="w",
                w_after_sha256="w",
                z_before_sha256="z0",
                z_after_sha256="z1",
            )

    def test_matched_demand_is_one_total_not_per_layer(self) -> None:
        slopes = AffineVelocitySlope(
            VelocitySlope((1, 2, 3, 4, 5)),
            VelocitySlope((0, 0, 0, 0, 0)),
        )
        selected = semantic_only_coupling(
            slopes, self.demand(3.0, 0.0), self.domain()
        )
        self.assertEqual(selected.rho, 3.0)
        self.assertNotEqual(selected.rho, 15.0)

    def test_invalid_trust_geometry_fails_closed(self) -> None:
        with self.assertRaises(ODEBFContractError):
            self.domain(constant=(1, 1, -1, 1, 1))

    def test_zero_capacity_observation_and_explicit_totality(self) -> None:
        residual = torch.zeros((4, 1), dtype=torch.float32)
        q = torch.ones((3, 1), dtype=torch.float32)
        observed = observe_dynamic_factor_capacity(
            residual,
            q,
            layer=4,
            request_order_sha256="a" * 64,
            accepted_waypoint=4,
        )
        self.assertEqual(observed["value_class"], "ZERO")
        self.assertEqual(observed["factor_capacity"], 0.0)
        self.assertEqual(
            _validate_dynamic_factor_capacity(
                0.0, allow_zero_capacity=True, observation=observed
            ),
            0.0,
        )
        with self.assertRaises(ODEBFContractError):
            _validate_dynamic_factor_capacity(
                0.0, allow_zero_capacity=False, observation=observed
            )

    def test_positive_and_nonfinite_capacity_classes(self) -> None:
        positive = observe_dynamic_factor_capacity(
            torch.ones((4, 1), dtype=torch.float32),
            torch.ones((3, 1), dtype=torch.float32),
            layer=4,
            request_order_sha256="b" * 64,
            accepted_waypoint=0,
        )
        self.assertEqual(positive["value_class"], "POSITIVE")
        self.assertGreater(
            _validate_dynamic_factor_capacity(
                float(positive["factor_capacity"]),
                allow_zero_capacity=True,
                observation=positive,
            ),
            0.0,
        )
        nonfinite = observe_dynamic_factor_capacity(
            torch.tensor([[float("nan")]], dtype=torch.float32),
            torch.ones((1, 1), dtype=torch.float32),
            layer=4,
            request_order_sha256="c" * 64,
            accepted_waypoint=0,
        )
        self.assertEqual(nonfinite["value_class"], "NONFINITE")
        with self.assertRaises(ODEBFContractError):
            _validate_dynamic_factor_capacity(
                float("nan"),
                allow_zero_capacity=True,
                observation=nonfinite,
            )

    def test_zero_semantic_field_totalizes_before_tracking(self) -> None:
        slopes = AffineVelocitySlope(
            VelocitySlope((0, 0, 0, 0, 0)),
            VelocitySlope((1, 1, 1, 1, 1)),
        )
        zero_domain = self.domain(constant=(0, 0, 0, 0, 0))
        selected = largest_feasible_lag_scale(
            slopes, self.demand(0.0, 1.0), zero_domain
        )
        self.assertEqual(
            selected.status,
            P1R28CouplingStatus.SEMANTIC_NO_POSITIVE_DIRECTION,
        )
        self.assertEqual(selected.lag_scale, 0.0)

    def _zero_barrier(self, label: str) -> QuadraticBarrier:
        zero = np.zeros(5, dtype=np.float64)
        return QuadraticBarrier(
            label, 0.0, zero, np.diag(zero), 0.0, "layer-local-diagonal"
        )

    def test_zero_action_problem_preserves_exact_zero_without_fake_radius(self) -> None:
        zero = np.zeros(5, dtype=np.float64)
        problem = ZeroActionRoutingProblem(
            zero,
            np.diag(np.full(5, 1.0e-12, dtype=np.float64)),
            np.diag(zero),
            0.0,
            np.ones(5, dtype=np.float64),
            0.0,
            0.0,
            self._zero_barrier("historical"),
            self._zero_barrier("pretrained"),
        )
        disabled = p1r24_disable_historical(problem)
        self.assertIsInstance(disabled, ZeroActionRoutingProblem)
        routing = solve_p1r24_matched_routing(
            disabled, arm=FixedE8Arm.NEUTRAL, rho_write=0.0
        )
        self.assertEqual(routing.status, P1R24RoutingStatus.SEMANTIC_NO_WRITE)
        self.assertEqual(routing.velocity, (0.0,) * 5)
        self.assertEqual(disabled.trust_radius, 0.0)
        with self.assertRaisesRegex(ODEBFContractError, "nonzero action geometry"):
            ZeroActionRoutingProblem(
                np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
                np.diag(np.full(5, 1.0e-12, dtype=np.float64)),
                np.diag(zero),
                0.0,
                np.ones(5, dtype=np.float64),
                0.0,
                0.0,
                self._zero_barrier("historical"),
                self._zero_barrier("pretrained"),
            )

    def test_dynamic_refresh_allows_only_explicit_constant_totality(self) -> None:
        value = "a" * 64
        ledger = DynamicRefreshLedger(expected_steps=2)
        ledger.record(
            step_index=0,
            accepted_state_sha256=value,
            target_sha256="b" * 64,
            key_inventory_sha256="c" * 64,
            slope_sha256="d" * 64,
            field_sha256="e" * 64,
            field_invocation_index=1,
        )
        with self.assertRaisesRegex(
            ODEBFStateError, "accepted physical state did not advance"
        ):
            ledger.record(
                step_index=1,
                accepted_state_sha256=value,
                target_sha256="b" * 64,
                key_inventory_sha256="c" * 64,
                slope_sha256="d" * 64,
                field_sha256="e" * 64,
                field_invocation_index=2,
            )
        ledger = DynamicRefreshLedger(expected_steps=2)
        for step in range(2):
            ledger.record(
                step_index=step,
                accepted_state_sha256=value,
                target_sha256="b" * 64,
                key_inventory_sha256="c" * 64,
                slope_sha256="d" * 64,
                field_sha256="e" * 64,
                field_invocation_index=step + 1,
                constant_state_totality=bool(step),
            )
        self.assertTrue(ledger.finalize()["records"][1]["constant_state_totality"])

    def test_runtime_source_closes_forbidden_paths(self) -> None:
        root = Path(__file__).resolve().parents[4]
        runtime = (root / "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py").read_text()
        controller = (root / "project/run_scripts/ode_bf/p1r28_corrected_coupling.py").read_text()
        sbatch = (root / "project/run_scripts/session05_ode_bf_p1r28_corrected_coupling.sbatch").read_text()
        self.assertIn("target_next = current_target.clone()", runtime)
        self.assertIn("stall_state_receipt", runtime)
        self.assertIn("p1r24-anchor-scale-comparison.json", runtime)
        self.assertIn("factor_capacity_observer", runtime)
        self.assertIn("allow_zero_action_totality", runtime)
        self.assertIn("same_sample_p1r24_anchor_energy_ratio", runtime)
        self.assertIn("functional_p_inner_probe_count\": 0", runtime)
        self.assertIn("largest_feasible_lag_scale", runtime)
        self.assertIn("forbidden_h_raw_slope_dot_applied_count\": 0", controller)
        self.assertNotIn("fixed_e8_soft_routing", controller)
        self.assertNotIn("bisect(", controller.casefold())
        self.assertNotIn("callback", sbatch.casefold())
        self.assertIn("MODELS=(llama3-8b-inst qwen2.5-7b-inst)", sbatch)

    def test_technical_result_parent_is_tightly_scoped_to_p1r28(self) -> None:
        from project.run_scripts.ode_bf import p1_runtime

        root = Path("/tmp/p1r28-namespace-fixture")
        expected_parent = root / "local/odebf/results"
        role = "P1R28_B1_RS_PAIR"
        result_name = "fixture-result"

        def expected_name(_alias: str, _role: str) -> str:
            self.assertEqual(_role, role)
            return result_name

        with (
            mock.patch.object(p1_runtime, "_source_freeze"),
            mock.patch(
                "project.run_scripts.ode_bf.p1r24_atomic_strength_panel.expected_p1r24_result_name",
                side_effect=expected_name,
            ),
            mock.patch.object(Path, "exists", return_value=True),
        ):
            with self.assertRaises(p1_runtime.P1OutputRootCollision):
                p1_runtime.run_p1(
                    repo_root=root,
                    alias="llama3-8b-inst",
                    output_root=expected_parent / "p1r28-tech-r3" / result_name,
                    source_head="f" * 40,
                    atomic_strength_recovery_role=role,
                )
            with self.assertRaisesRegex(
                p1_runtime.ODEBFContractError, "output namespace differs"
            ):
                p1_runtime.run_p1(
                    repo_root=root,
                    alias="llama3-8b-inst",
                    output_root=expected_parent / "other-tech-r3" / result_name,
                    source_head="f" * 40,
                    atomic_strength_recovery_role=role,
                )


if __name__ == "__main__":
    unittest.main()
