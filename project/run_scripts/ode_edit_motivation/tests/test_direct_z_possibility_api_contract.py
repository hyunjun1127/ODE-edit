"""Regression guard for direct-z runner cross-module call signatures.

The direct-z runner deliberately composes reusable helpers from the bridge,
quarter-step controller, MV-1 metrics, and direct-z fidelity module.  This
test parses the runner rather than executing it, then binds every such call
against the *actual* public callable signature.  It therefore catches a
positional/keyword contract drift before a GPU run can produce zero outcomes.
"""

from __future__ import annotations

import ast
import inspect
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from project.run_scripts.ode_edit_motivation import direct_z_fidelity
from project.run_scripts.ode_edit_motivation import easyedit_bridge
from project.run_scripts.ode_edit_motivation import mv1_calibration
from project.run_scripts.ode_edit_motivation import quarter_step_refresh


_RUNNER_PATH = Path(__file__).resolve().parents[1] / "direct_z_possibility.py"
_SENTINEL = object()


_FUNCTION_TARGETS: dict[str, Callable[..., Any]] = {
    # direct_z_fidelity
    "aggregate_sequence_spill": direct_z_fidelity.aggregate_sequence_spill,
    "aggregate_shared_delta_fidelity": direct_z_fidelity.aggregate_shared_delta_fidelity,
    "combine_unit_c_single_layer_proposals": (
        direct_z_fidelity.combine_unit_c_single_layer_proposals
    ),
    "make_batch_position_patch": direct_z_fidelity.make_batch_position_patch,
    "solve_five_direction_ridge": direct_z_fidelity.solve_five_direction_ridge,
    "unwrap_layer_output": direct_z_fidelity.unwrap_layer_output,
    # mv1_calibration
    "_feature_hash": mv1_calibration._feature_hash,
    "_layer_by_weight": mv1_calibration._layer_by_weight,
    "_state_identity": mv1_calibration._state_identity,
    "assert_exact_action_contract": mv1_calibration.assert_exact_action_contract,
    "build_exact_teacher_batch": mv1_calibration.build_exact_teacher_batch,
    "build_unit_c_actions": mv1_calibration.build_unit_c_actions,
    "proposal_c_energy": mv1_calibration.proposal_c_energy,
    "rewrite_metrics": mv1_calibration.rewrite_metrics,
    "scale_proposal": mv1_calibration.scale_proposal,
    # quarter_step_refresh
    "_assert_step_energy": quarter_step_refresh._assert_step_energy,
    "_build_policy_path": quarter_step_refresh._build_policy_path,
    "_build_score_mix_action": quarter_step_refresh._build_score_mix_action,
    "_capture_source_snapshot": quarter_step_refresh._capture_source_snapshot,
    "_combine_steps": quarter_step_refresh._combine_steps,
    "_commit_qstep_action": quarter_step_refresh._commit_action,
    "_proposal_panel": quarter_step_refresh._proposal_panel,
}

_BRIDGE_METHODS: dict[str, Callable[..., Any]] = {
    "preflight": easyedit_bridge.EasyEditBridge.preflight,
    "load": easyedit_bridge.EasyEditBridge.load,
    "load_or_compute_direct_z": easyedit_bridge.EasyEditBridge.load_or_compute_direct_z,
    "propose_synchronous_memit_factors": (
        easyedit_bridge.EasyEditBridge.propose_synchronous_memit_factors
    ),
    "propose_ordered_memit_factors": (
        easyedit_bridge.EasyEditBridge.propose_ordered_memit_factors
    ),
}


class _CrossModuleCallCollector(ast.NodeVisitor):
    """Collect only runner calls whose target is in this contract surface."""

    def __init__(self) -> None:
        self.calls: dict[str, list[ast.Call]] = defaultdict(list)

    def visit_Call(self, node: ast.Call) -> None:
        name: str | None = None
        if isinstance(node.func, ast.Name):
            if node.func.id in _FUNCTION_TARGETS or node.func.id == "EasyEditBridge":
                name = node.func.id
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "bridge"
            and node.func.attr in _BRIDGE_METHODS
        ):
            name = f"bridge.{node.func.attr}"
        if name is not None:
            self.calls[name].append(node)
        self.generic_visit(node)


def _bind_ast_call(
    signature_target: Callable[..., Any],
    call: ast.Call,
    *,
    bound_receiver: bool,
) -> None:
    """Bind a syntax-level call using sentinels without executing it."""

    if any(isinstance(argument, ast.Starred) for argument in call.args):
        raise AssertionError(f"line {call.lineno}: starred positional arguments hide API binding")
    if any(keyword.arg is None for keyword in call.keywords):
        raise AssertionError(f"line {call.lineno}: **kwargs hides API binding")
    positional = [_SENTINEL for _ in call.args]
    keywords = {str(keyword.arg): _SENTINEL for keyword in call.keywords}
    receiver = (_SENTINEL,) if bound_receiver else ()
    inspect.signature(signature_target).bind(*receiver, *positional, **keywords)


class DirectZPossibilityApiContractTest(unittest.TestCase):
    def test_runner_cross_module_calls_bind_to_actual_signatures(self) -> None:
        source = _RUNNER_PATH.read_text(encoding="utf-8")
        compile(source, str(_RUNNER_PATH), "exec")
        collector = _CrossModuleCallCollector()
        collector.visit(ast.parse(source, filename=str(_RUNNER_PATH)))

        expected = {
            *(_FUNCTION_TARGETS),
            "EasyEditBridge",
            *(f"bridge.{name}" for name in _BRIDGE_METHODS),
        }
        self.assertEqual(
            set(collector.calls),
            expected,
            "the runner cross-module API surface changed; update this contract test",
        )

        for name, calls in collector.calls.items():
            if name == "EasyEditBridge":
                target = easyedit_bridge.EasyEditBridge
                bound_receiver = False
            elif name.startswith("bridge."):
                target = _BRIDGE_METHODS[name.removeprefix("bridge.")]
                bound_receiver = True
            else:
                target = _FUNCTION_TARGETS[name]
                bound_receiver = False
            for call in calls:
                with self.subTest(target=name, line=call.lineno):
                    _bind_ast_call(target, call, bound_receiver=bound_receiver)


if __name__ == "__main__":
    unittest.main()
