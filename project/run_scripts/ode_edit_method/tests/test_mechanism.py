from __future__ import annotations

import gc
import unittest
import weakref
from unittest.mock import PropertyMock, patch

import torch

from project.run_scripts.ode_edit_method.contracts import (
    Arm,
    ArmRunResult,
    LayerProposal,
    MethodContractError,
    ProposalBatch,
    ProposalSemantics,
    StepRecord,
)
from project.run_scripts.ode_edit_method.hooks import FactorDirection
from project.run_scripts.ode_edit_method.mechanism import (
    capture_field_mechanism,
    summarize_mechanism,
    trajectory_summary,
)
from project.run_scripts.ode_edit_method.p1_orchestration import P1SequentialArmGuard


def _direction(layer: int, shift: float) -> FactorDirection:
    return FactorDirection(
        layer=layer,
        weight_name=f"layers.{layer}.weight",
        left=torch.tensor(
            [[0.2 + shift, -0.4], [0.7, 0.3 + shift]], dtype=torch.float64
        ),
        right=torch.tensor(
            [[0.3, -0.1 + shift], [0.5 + shift, 0.2], [-0.4, 0.6]],
            dtype=torch.float64,
        ),
    )


def _batch(state: str, shift: float, slopes: tuple[float, float]) -> ProposalBatch:
    directions = (_direction(0, shift), _direction(1, -shift))
    return ProposalBatch(
        snapshot_id=state,
        proposals=tuple(
            LayerProposal(
                layer=direction.layer,
                state_id=state,
                direction_id=direction.direction_id,
                payload=direction,
            )
            for direction in directions
        ),
        slopes=slopes,
        semantics=ProposalSemantics.CURRENT_SAME_SNAPSHOT,
    )


def _step(
    position: int,
    coefficients: tuple[float, float],
    *,
    accepted: bool,
    before: float,
    after: float,
) -> StepRecord:
    return StepRecord(
        arm=Arm.FULL_ODE_EDIT,
        position=position,
        layer=None,
        snapshot_id=f"state-{position}",
        direction_ids=(f"d-{position}-0", f"d-{position}-1"),
        solver_coefficients=coefficients,
        applied_coefficients=coefficients,
        accepted=accepted,
        reason="accepted" if accepted else "trust-ratio-reject",
        hard_phi_before=before,
        hard_phi_after=after,
        smooth_phi_before=before + 0.5,
        smooth_phi_after=after + 0.5,
        trust_ratio=1.0 if accepted else 0.0,
    )


class MechanismDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def _stream_result(
        *,
        edit_id: str,
        terminal: str,
        appended: bool,
        direct_z: int = 1,
    ) -> ArmRunResult:
        if direct_z == 0:
            return ArmRunResult(
                arm=Arm.FULL_ODE_EDIT,
                edit_id=edit_id,
                status="event_hit",
                direct_z_compute_count=0,
                terminal_state_id=terminal,
                omega_appended=True,
                steps=(),
            )
        step = _step(0, (0.2, 0.1), accepted=True, before=4.0, after=2.0)
        return ArmRunResult(
            arm=Arm.FULL_ODE_EDIT,
            edit_id=edit_id,
            status="event_hit" if appended else "resolution_cap_unresolved",
            direct_z_compute_count=1,
            terminal_state_id=terminal,
            omega_appended=appended,
            steps=(step,),
            failure_type=None if appended else "resolution_cap_unresolved",
        )

    def test_capture_retains_only_string_direction_identities(self) -> None:
        batch = _batch("entry", 0.0, (1.0, 3.0))
        left_tensor = batch.proposals[0].payload.left
        tensor_reference = weakref.ref(left_tensor)
        with patch.object(
            FactorDirection,
            "direction_id",
            new_callable=PropertyMock,
            side_effect=AssertionError("mechanism capture rehashed U/V"),
        ):
            first, previous = capture_field_mechanism(
                batch, None, normalization_epsilon=1e-12
            )
        self.assertEqual(first["normalized_layer_efficiency"], [0.25, 0.75])
        self.assertTrue(all(isinstance(value, str) for value in previous.values()))
        self.assertFalse(
            any(isinstance(value, torch.Tensor) for value in previous.values())
        )
        second, current = capture_field_mechanism(
            _batch("child", 0.17, (4.0, 1.0)),
            previous,
            normalization_epsilon=1e-12,
        )
        self.assertEqual(
            second["transition_from_previous"]["direction_id_transition_rate"],
            1.0,
        )
        self.assertTrue(all(isinstance(value, str) for value in current.values()))

        del left_tensor
        del batch
        gc.collect()
        self.assertIsNone(tensor_reference())

    def test_omega_receipt_never_infers_rollback_state(self) -> None:
        step = _step(0, (0.2, 0.1), accepted=True, before=4.0, after=2.5)
        common = dict(
            arm=Arm.FULL_ODE_EDIT,
            edit_id="edit",
            status="event_hit",
            direct_z_compute_count=1,
            terminal_state_id="terminal",
            steps=(step,),
        )
        appended = ArmRunResult(**common, omega_appended=True)
        not_appended = ArmRunResult(
            **{**common, "status": "resolution_cap_unresolved"},
            omega_appended=False,
            failure_type="resolution_cap_unresolved",
        )
        values = []
        for result in (appended, not_appended):
            summary = trajectory_summary(
                result,
                (),
                controller_gpu_seconds=2.0,
                progress_epsilon=1e-12,
                failure_rollback_required=False,
                failure_rollback_exact=None,
                successful_terminal_endpoint_exact=None,
                arm_isolation_restore_exact=True,
            )
            state = summary["transaction_state"]
            self.assertFalse(state["omega_used_for_rollback_inference"])
            values.append(
                (
                    state["failure_rollback_exact"],
                    state["successful_terminal_endpoint_exact"],
                    state["arm_isolation_restore_exact"],
                )
            )
        self.assertEqual(values[0], values[1])

    def test_adaptive_drift_and_progress_use_existing_records_only(self) -> None:
        first, previous = capture_field_mechanism(
            _batch("entry", 0.0, (1.0, 3.0)),
            None,
            normalization_epsilon=1e-12,
        )
        second, _current = capture_field_mechanism(
            _batch("child", 0.2, (4.0, 1.0)),
            previous,
            normalization_epsilon=1e-12,
        )
        result = ArmRunResult(
            arm=Arm.FULL_ODE_EDIT,
            edit_id="edit",
            status="resolution_cap_unresolved",
            direct_z_compute_count=1,
            terminal_state_id="entry",
            omega_appended=False,
            failure_type="resolution_cap_unresolved",
            steps=(
                _step(0, (0.4, 0.2), accepted=True, before=4.0, after=3.0),
                _step(1, (0.1, 0.5), accepted=True, before=3.0, after=2.0),
                _step(2, (0.1, 0.5), accepted=False, before=2.0, after=2.4),
            ),
        )
        summary = summarize_mechanism(
            Arm.FULL_ODE_EDIT,
            result,
            (first, second),
            layers=(0, 1),
            support_tolerance=1e-12,
            controller_gpu_seconds=2.0,
            event_history=(),
            progress_epsilon=1e-12,
            failure_rollback_required=True,
            failure_rollback_exact=True,
            successful_terminal_endpoint_exact=None,
            arm_isolation_restore_exact=True,
        )
        self.assertEqual(summary["field_count"], 2)
        self.assertEqual(summary["direction_id_transition_rate"]["values"], [1.0])
        self.assertEqual(summary["layer_ranking_drift"]["values"], [1.0])
        self.assertEqual(summary["allocation_coefficient_cosine"]["comparisons"], 1)
        self.assertEqual(summary["support_turnover"]["values"], [0.0])
        self.assertFalse(summary["c_geometry"]["available"])
        self.assertFalse(summary["c_geometry"]["controller_compute_contaminated"])
        trajectory = summary["trajectory"]
        self.assertEqual(trajectory["start_hard_phi"], 4.0)
        self.assertEqual(trajectory["end_accepted_hard_phi"], 2.0)
        self.assertEqual(trajectory["last_trial_hard_phi"], 2.4)
        self.assertEqual(trajectory["hard_progress_per_controller_gpu_second"], 1.0)
        self.assertTrue(
            trajectory["transaction_state"]["failure_rollback_exact"]
        )

    def test_p1_sequential_state_machine_preserves_history_and_arm_isolation(self) -> None:
        cases = ("2022", "12498", "20964", "768")
        guard = P1SequentialArmGuard(Arm.FULL_ODE_EDIT, cases)
        guard.note_arm_start_restore()

        guard.begin_edit(
            order_position=0,
            case_id=cases[0],
            pre_edit_state_id="W0",
            omega_before={0: 0.0, 1: 0.0},
            history_length_before=0,
            cache_identity="full/position-0.pt",
        )
        guard.finish_edit(
            result=self._stream_result(
                edit_id="e0", terminal="W1", appended=True
            ),
            post_edit_state_id="W1",
            omega_after={0: 0.1, 1: 0.2},
            history_length_after=1,
        )

        guard.begin_edit(
            order_position=1,
            case_id=cases[1],
            pre_edit_state_id="W1",
            omega_before={0: 0.1, 1: 0.2},
            history_length_before=1,
            cache_identity="full/position-1.pt",
        )
        guard.finish_edit(
            result=self._stream_result(
                edit_id="e1", terminal="W1", appended=False
            ),
            post_edit_state_id="W1",
            omega_after={0: 0.1, 1: 0.2},
            history_length_after=1,
        )

        guard.begin_edit(
            order_position=2,
            case_id=cases[2],
            pre_edit_state_id="W1",
            omega_before={0: 0.1, 1: 0.2},
            history_length_before=1,
            cache_identity="full/position-2.pt",
        )
        guard.finish_edit(
            result=self._stream_result(
                edit_id="e2", terminal="W2", appended=True
            ),
            post_edit_state_id="W2",
            omega_after={0: 0.4, 1: 0.3},
            history_length_after=2,
        )

        guard.begin_edit(
            order_position=3,
            case_id=cases[3],
            pre_edit_state_id="W2",
            omega_before={0: 0.4, 1: 0.3},
            history_length_before=2,
            cache_identity="full/position-3.pt",
        )
        guard.finish_edit(
            result=self._stream_result(
                edit_id="e3", terminal="W2", appended=True, direct_z=0
            ),
            post_edit_state_id="W2",
            omega_after={0: 0.4, 1: 0.3},
            history_length_after=3,
        )
        guard.note_arm_end_restore(exact=True)
        summary = guard.summary()
        self.assertEqual(summary["arm_start_baseline_restore_count"], 1)
        self.assertEqual(summary["arm_end_baseline_restore_count"], 1)
        self.assertEqual(summary["case_level_w0_restore_count"], 0)
        self.assertEqual(summary["direct_z_compute_counts"], [1, 1, 1, 0])
        self.assertEqual(summary["cache_identity_count"], 4)
        self.assertFalse(summary["p1_cross_arm_h0_over_D_native_computed"])
        self.assertEqual(summary["rows"][1]["pre_edit_state_id"], "W1")
        self.assertEqual(summary["rows"][1]["post_edit_state_id"], "W1")
        self.assertEqual(summary["rows"][2]["pre_edit_state_id"], "W1")
        self.assertEqual(
            [row["order_position"] for row in summary["rows"]], [0, 1, 2, 3]
        )

        independent = P1SequentialArmGuard(Arm.FULL_ODE_EDIT, cases)
        independent.note_arm_start_restore()
        independent.begin_edit(
            order_position=0,
            case_id=cases[0],
            pre_edit_state_id="W0",
            omega_before={0: 0.0, 1: 0.0},
            history_length_before=0,
            cache_identity="other-arm/position-0.pt",
        )

        broken = P1SequentialArmGuard(Arm.FULL_ODE_EDIT, cases)
        broken.note_arm_start_restore()
        broken.begin_edit(
            order_position=0,
            case_id=cases[0],
            pre_edit_state_id="W0",
            omega_before={0: 0.0, 1: 0.0},
            history_length_before=0,
            cache_identity="broken/position-0.pt",
        )
        broken.finish_edit(
            result=self._stream_result(
                edit_id="broken-e0", terminal="W1", appended=True
            ),
            post_edit_state_id="W1",
            omega_after={0: 0.1, 1: 0.2},
            history_length_after=1,
        )
        with self.assertRaisesRegex(MethodContractError, "retained endpoint"):
            broken.begin_edit(
                order_position=1,
                case_id=cases[1],
                pre_edit_state_id="W0",
                omega_before={0: 0.1, 1: 0.2},
                history_length_before=1,
                cache_identity="broken/position-1.pt",
            )


if __name__ == "__main__":
    unittest.main()
