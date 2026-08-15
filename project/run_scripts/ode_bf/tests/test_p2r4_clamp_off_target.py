from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r24_atomic_strength import P1R24AliasTargetLock
from project.run_scripts.ode_bf.p2r1_rms_tangent_target import (
    P2R1RMSState,
    p2r1_target_update,
)
from project.run_scripts.ode_bf.p2r4_clamp_off_target import (
    P2R4_CLAMP_POLICY,
    p2r4_clamp_off_target_update,
)


class P2R4ClampOffTargetTests(unittest.TestCase):
    def test_would_hit_is_observation_only_and_preclamp_is_applied(self) -> None:
        lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")
        origin = torch.ones(4, 2, dtype=torch.float32)
        direction = torch.full_like(origin, -0.745)
        current = origin + direction
        semantic = torch.ones_like(origin)
        preservation = torch.zeros_like(origin)
        state = P2R1RMSState.zero()

        frozen_on = p2r1_target_update(
            current,
            origin,
            semantic,
            preservation,
            state,
            alias=lock.alias,
            microstep_index=0,
            lock=lock,
        )
        off = p2r4_clamp_off_target_update(
            current,
            origin,
            semantic,
            preservation,
            state,
            alias=lock.alias,
            microstep_index=0,
            lock=lock,
        )
        expected = (
            current.to(torch.float64)
            + float(frozen_on.receipt["eta_m"]) * frozen_on.field.to(torch.float64)
        ).to(torch.float32)

        self.assertGreater(frozen_on.receipt["clamp_hit_count"], 0)
        self.assertTrue(torch.equal(off.target_next, expected))
        self.assertFalse(torch.equal(off.target_next, frozen_on.target_next))
        self.assertTrue(torch.equal(off.field, frozen_on.field))
        self.assertTrue(
            torch.equal(
                off.state_next.running_squared_gradient,
                frozen_on.state_next.running_squared_gradient,
            )
        )
        self.assertEqual(off.receipt["clamp_policy"], P2R4_CLAMP_POLICY)
        self.assertEqual(
            off.receipt["clamp_would_hit_count"],
            frozen_on.receipt["clamp_hit_count"],
        )
        self.assertEqual(off.receipt["clamp_hit_count"], 0)
        self.assertEqual(off.receipt["clamp_application_count"], 0)
        self.assertEqual(off.receipt["identity_projection_application_count"], 1)
        self.assertEqual(off.receipt["clamp_decision_influence_count"], 0)
        self.assertEqual(off.receipt["norm_rescale_count"], 0)
        self.assertEqual(off.receipt["velocity_clip_count"], 0)
        self.assertEqual(
            off.receipt["pre_clamp_displacement_norm"],
            off.receipt["post_clamp_displacement_norm"],
        )

    def test_no_hit_matches_frozen_on_numerically(self) -> None:
        lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")
        origin = torch.ones(7, 3)
        semantic = torch.full_like(origin, 0.01)
        preservation = torch.zeros_like(origin)
        state = P2R1RMSState.zero()
        frozen_on = p2r1_target_update(
            origin,
            origin,
            semantic,
            preservation,
            state,
            alias=lock.alias,
            microstep_index=0,
            lock=lock,
        )
        off = p2r4_clamp_off_target_update(
            origin,
            origin,
            semantic,
            preservation,
            state,
            alias=lock.alias,
            microstep_index=0,
            lock=lock,
        )
        self.assertEqual(frozen_on.receipt["clamp_hit_count"], 0)
        self.assertTrue(torch.equal(off.target_next, frozen_on.target_next))

    def test_exact_24_step_rms_state_chain_has_no_outer_reset(self) -> None:
        torch.manual_seed(204)
        lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")
        origin = torch.ones(6, 2)
        target = origin.clone()
        state = P2R1RMSState.zero()
        squared: list[torch.Tensor] = []
        for microstep in range(24):
            semantic = torch.randn_like(origin) * (0.001 + microstep * 1e-5)
            squared.append(semantic.to(torch.float64).square())
            update = p2r4_clamp_off_target_update(
                target,
                origin,
                semantic,
                torch.zeros_like(origin),
                state,
                alias=lock.alias,
                microstep_index=microstep,
                lock=lock,
            )
            target, state = update.target_next, update.state_next
            self.assertEqual(state.completed_microsteps, microstep + 1)
            self.assertTrue(
                torch.allclose(
                    state.running_squared_gradient,
                    torch.stack(squared).mean(dim=0),
                    atol=0.0,
                    rtol=1e-12,
                )
            )
        self.assertEqual(state.completed_microsteps, 24)

    def test_nonfinite_input_fails_closed(self) -> None:
        lock = P1R24AliasTargetLock.for_alias("qwen2.5-7b-inst")
        origin = torch.ones(3, 1)
        semantic = torch.ones_like(origin)
        semantic[0, 0] = float("inf")
        with self.assertRaises(ODEBFContractError):
            p2r4_clamp_off_target_update(
                origin,
                origin,
                semantic,
                torch.zeros_like(origin),
                P2R1RMSState.zero(),
                alias=lock.alias,
                microstep_index=0,
                lock=lock,
            )


if __name__ == "__main__":
    unittest.main()
