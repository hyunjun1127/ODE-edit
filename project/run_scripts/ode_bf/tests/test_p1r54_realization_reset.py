from __future__ import annotations

import torch

from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1r54_realization_policy import (
    RealizationPolicy,
    RealizationTransitionController,
)
from project.run_scripts.ode_bf.p1r54_realization_reset import (
    CELLS,
    ResetExecutionScope,
    build_sequential_binding,
    source_equivalence_receipt,
)


def _tensor(offset: float) -> torch.Tensor:
    return (torch.arange(12, dtype=torch.float32).reshape(4, 3) + offset).contiguous()


def test_reset_and_continuation_diverge_only_after_k1_transition() -> None:
    origin = _tensor(0.0)
    current = origin.clone()
    command = _tensor(1.0)
    physical = _tensor(0.25)
    reset = RealizationTransitionController(
        RealizationPolicy.POST_WRITE_W_REALIZATION_RESET, request_count=3
    )
    continuation = RealizationTransitionController(
        RealizationPolicy.TARGET_Z_CONTINUATION, request_count=3
    )
    for controller in (reset, continuation):
        controller.observe_entry(
            step_index=0,
            controller_anchor=current,
            current_terminal=current,
            target_origin=origin,
            teacher_sha256="a" * 64,
            target_new_nll_by_request=(1.0, 2.0, 3.0),
        )
    reset_anchor, reset_receipt = reset.advance(
        step_index=0,
        controller_anchor=current,
        commanded_target=command,
        current_terminal=current,
        next_physical_terminal=physical,
        target_origin=origin,
        teacher_sha256="a" * 64,
        commanded_target_new_nll_by_request=(0.5, 0.6, 0.7),
    )
    continuation_anchor, continuation_receipt = continuation.advance(
        step_index=0,
        controller_anchor=current,
        commanded_target=command,
        current_terminal=current,
        next_physical_terminal=physical,
        target_origin=origin,
        teacher_sha256="a" * 64,
        commanded_target_new_nll_by_request=(0.5, 0.6, 0.7),
    )
    assert reset_receipt["commanded_target_sha256"] == continuation_receipt["commanded_target_sha256"]
    assert reset_receipt["current_terminal_sha256"] == continuation_receipt["current_terminal_sha256"]
    assert reset_receipt["next_physical_terminal_sha256"] == continuation_receipt["next_physical_terminal_sha256"]
    assert tensor_sha256(reset_anchor) == tensor_sha256(physical)
    assert tensor_sha256(continuation_anchor) == tensor_sha256(command)


def test_reset_k2_anchor_and_last_command_are_separate() -> None:
    origin = _tensor(0.0)
    current = origin.clone()
    controller = RealizationTransitionController(
        RealizationPolicy.POST_WRITE_W_REALIZATION_RESET, request_count=3
    )
    for step in range(8):
        controller.observe_entry(
            step_index=step,
            controller_anchor=current,
            current_terminal=current,
            target_origin=origin,
            teacher_sha256="b" * 64,
            target_new_nll_by_request=(float(step + 1),) * 3,
        )
        command = current + 1.0
        physical = current + 0.25
        current, receipt = controller.advance(
            step_index=step,
            controller_anchor=current,
            commanded_target=command,
            current_terminal=current,
            next_physical_terminal=physical,
            target_origin=origin,
            teacher_sha256="b" * 64,
            commanded_target_new_nll_by_request=(0.5,) * 3,
            terminal_post_write_nll_by_request=((0.25,) * 3 if step == 7 else None),
        )
        assert receipt["next_controller_anchor_is_z_oracle"] is False
    assert tensor_sha256(controller.terminal_commanded_target()) == receipt["last_commanded_target_sha256"]
    terminal = controller.terminal_receipt()
    assert terminal["origin_refresh_count"] == 0
    assert terminal["teacher_refresh_count"] == 1
    assert terminal["reset_added_model_forward_count"] == 0
    assert terminal["reset_added_backward_count"] == 0
    assert terminal["reset_added_materialization_count"] == 0
    assert all(row["post_write_target_new_nll_by_request"] is not None for row in terminal["transition_rows"])


def test_four_cell_mapping_and_shared_sequential_loop() -> None:
    assert [item.cell for item in CELLS] == [0, 1, 2, 3]
    assert [item.scope for item in CELLS] == [
        ResetExecutionScope.INDEPENDENT_B100,
        ResetExecutionScope.INDEPENDENT_B100,
        ResetExecutionScope.SEQUENTIAL_10XB100,
        ResetExecutionScope.SEQUENTIAL_10XB100,
    ]
    assert [item.arm.value for item in CELLS] == ["FZ", "PDZ", "FZ", "PDZ"]
    for config in CELLS:
        receipt = source_equivalence_receipt(config.cell)
        assert receipt["new_continuation_execution_count"] == 0
        assert receipt["divergence_first_allowed_outer"] == 2
    for config in CELLS[2:]:
        binding = build_sequential_binding(config)
        assert binding.realization_controller_factory is not None
        assert binding.writer_arm == "C3-KSTEP-CACHE"
        assert binding.metadata["continuation_gpu_execution_count"] == 0
