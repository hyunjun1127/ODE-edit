"""Static signature guard for the Alpha direct-z GPU runner."""

from __future__ import annotations

import ast
import inspect
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

from project.run_scripts.ode_edit_motivation import alphaedit_factors
from project.run_scripts.ode_edit_motivation import alphaedit_proposal_adapter
from project.run_scripts.ode_edit_motivation import alphaedit_reference
from project.run_scripts.ode_edit_motivation import direct_z_alpha_possibility
from project.run_scripts.ode_edit_motivation import direct_z_alpha_replay
from project.run_scripts.ode_edit_motivation import direct_z_fidelity
from project.run_scripts.ode_edit_motivation import direct_z_possibility
from project.run_scripts.ode_edit_motivation import easyedit_bridge
from project.run_scripts.ode_edit_motivation import mv1_calibration
from project.run_scripts.ode_edit_motivation import projector_adapter
from project.run_scripts.ode_edit_motivation import quarter_step_refresh


_RUNNER = Path(__file__).resolve().parents[1] / "direct_z_alpha_possibility.py"
_SENTINEL = object()

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "compare_low_rank_proposals": alphaedit_factors.compare_low_rank_proposals,
    "load_alphaedit_solver_config": alphaedit_reference.load_alphaedit_solver_config,
    "load_alpha_replay_lock": direct_z_alpha_replay.load_alpha_replay_lock,
    "preflight_model_replay": direct_z_alpha_replay.preflight_model_replay,
    "combine_unit_c_single_layer_proposals": direct_z_fidelity.combine_unit_c_single_layer_proposals,
    "solve_five_direction_ridge": direct_z_fidelity.solve_five_direction_ridge,
    "_capture_teacher": direct_z_possibility._capture_teacher,
    "_evaluate_arm": direct_z_possibility._evaluate_arm,
    "_outcome_payload": direct_z_possibility._outcome_payload,
    "generate_directz_selection": direct_z_possibility.generate_directz_selection,
    "load_heldout_evaluation_case": direct_z_possibility.load_heldout_evaluation_case,
    "_feature_hash": mv1_calibration._feature_hash,
    "_layer_by_weight": mv1_calibration._layer_by_weight,
    "_state_identity": mv1_calibration._state_identity,
    "assert_exact_action_contract": mv1_calibration.assert_exact_action_contract,
    "build_exact_teacher_batch": mv1_calibration.build_exact_teacher_batch,
    "build_unit_c_actions": mv1_calibration.build_unit_c_actions,
    "proposal_c_energy": mv1_calibration.proposal_c_energy,
    "scale_proposal": mv1_calibration.scale_proposal,
    "_assert_step_energy": quarter_step_refresh._assert_step_energy,
    "_build_policy_path": quarter_step_refresh._build_policy_path,
    "_build_score_mix_action": quarter_step_refresh._build_score_mix_action,
    "_capture_source_snapshot": quarter_step_refresh._capture_source_snapshot,
    "_combine_steps": quarter_step_refresh._combine_steps,
    "_commit_qstep_action": quarter_step_refresh._commit_action,
    "_proposal_panel": quarter_step_refresh._proposal_panel,
}

_METHODS: dict[tuple[str, str], Callable[..., Any]] = {
    ("bridge", "load"): easyedit_bridge.EasyEditBridge.load,
    ("bridge", "load_or_compute_direct_z"): easyedit_bridge.EasyEditBridge.load_or_compute_direct_z,
    ("adapter", "propose_ordered"): alphaedit_proposal_adapter.AlphaEditProposalAdapter.propose_ordered,
    ("adapter", "propose_synchronous"): alphaedit_proposal_adapter.AlphaEditProposalAdapter.propose_synchronous,
    ("adapter", "proposal_right_leak"): alphaedit_proposal_adapter.AlphaEditProposalAdapter.proposal_right_leak,
    ("projector_bank", "assert_stat_current"): projector_adapter.AlphaEditProjectorBank.assert_stat_current,
    ("projector_bank", "assert_hash_current"): projector_adapter.AlphaEditProjectorBank.assert_hash_current,
    ("projector_bank", "metadata"): projector_adapter.AlphaEditProjectorBank.metadata,
    ("replay", "assert_current"): direct_z_alpha_replay.ModelReplayPreflight.assert_current,
    ("replay", "record_for_request"): direct_z_alpha_replay.ModelReplayPreflight.record_for_request,
}

_CONSTRUCTORS: dict[str, Callable[..., Any]] = {
    "EasyEditBridge": easyedit_bridge.EasyEditBridge,
    "AlphaEditProposalAdapter": alphaedit_proposal_adapter.AlphaEditProposalAdapter,
}


class _Collector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: dict[str, list[ast.Call]] = defaultdict(list)

    def visit_Call(self, node: ast.Call) -> None:
        name: str | None = None
        if isinstance(node.func, ast.Name):
            if node.func.id in _FUNCTIONS or node.func.id in _CONSTRUCTORS:
                name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            key = (node.func.value.id, node.func.attr)
            if key in _METHODS:
                name = f"{key[0]}.{key[1]}"
        if name is not None:
            self.calls[name].append(node)
        self.generic_visit(node)


def _bind(target: Callable[..., Any], call: ast.Call, *, receiver: bool) -> None:
    if any(isinstance(argument, ast.Starred) for argument in call.args):
        raise AssertionError(f"line {call.lineno}: starred positional argument")
    if any(keyword.arg is None for keyword in call.keywords):
        raise AssertionError(f"line {call.lineno}: **kwargs hides binding")
    positional = [_SENTINEL for _ in call.args]
    keywords = {str(keyword.arg): _SENTINEL for keyword in call.keywords}
    prefix = (_SENTINEL,) if receiver else ()
    inspect.signature(target).bind(*prefix, *positional, **keywords)


class DirectZAlphaApiContractTests(unittest.TestCase):
    def test_all_cross_module_calls_bind(self) -> None:
        source = _RUNNER.read_text(encoding="utf-8")
        compile(source, str(_RUNNER), "exec")
        collector = _Collector()
        collector.visit(ast.parse(source, filename=str(_RUNNER)))
        expected = {
            *_FUNCTIONS,
            *_CONSTRUCTORS,
            *(f"{owner}.{name}" for owner, name in _METHODS),
        }
        self.assertEqual(set(collector.calls), expected)
        for name, calls in collector.calls.items():
            if name in _FUNCTIONS:
                target, receiver = _FUNCTIONS[name], False
            elif name in _CONSTRUCTORS:
                target, receiver = _CONSTRUCTORS[name], False
            else:
                owner, method = name.split(".", 1)
                target, receiver = _METHODS[(owner, method)], True
            for call in calls:
                with self.subTest(target=name, line=call.lineno):
                    _bind(target, call, receiver=receiver)

    def test_receipt_helper_keys_and_bf_endpoint_binding_are_static(self) -> None:
        tree = ast.parse(_RUNNER.read_text(encoding="utf-8"), filename=str(_RUNNER))
        run_event = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_event"
        )
        assigned: dict[str, ast.Dict] = {}
        for node in ast.walk(run_event):
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in {"feature", "action"}
                and isinstance(node.value, ast.Dict)
            ):
                assigned[node.target.id] = node.value
        self.assertEqual(set(assigned), {"feature", "action"})

        def literal_keys(node: ast.Dict) -> set[str]:
            return {
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }

        feature_keys = literal_keys(assigned["feature"])
        action_keys = literal_keys(assigned["action"])
        self.assertIn("target_identity", feature_keys)
        self.assertIn("per_hop_c_energy", action_keys)

        target_identity = next(
            value
            for key, value in zip(
                assigned["feature"].keys, assigned["feature"].values, strict=True
            )
            if isinstance(key, ast.Constant) and key.value == "target_identity"
        )
        path_action_hashes = next(
            value
            for key, value in zip(
                assigned["action"].keys, assigned["action"].values, strict=True
            )
            if isinstance(key, ast.Constant) and key.value == "path_action_hashes"
        )
        self.assertIsInstance(target_identity, ast.Dict)
        self.assertIsInstance(path_action_hashes, ast.Dict)
        self.assertIn("origin_lineage_id", literal_keys(target_identity))
        self.assertIn("bf_evaluation_endpoint_binding", literal_keys(path_action_hashes))

    def test_bf_endpoint_binding_reproduces_evaluated_energy(self) -> None:
        binding = direct_z_alpha_possibility._bf_endpoint_receipt_binding(
            proposal_hash="a" * 64,
            endpoint_scale_from_raw=0.5,
            raw_path_c_energy=4.0,
            evaluation_endpoint_c_energy=1.0,
        )
        self.assertEqual(
            binding,
            {
                "proposal_direction_hash": "a" * 64,
                "endpoint_scale_from_raw": 0.5,
                "raw_path_c_energy": 4.0,
                "evaluation_endpoint_c_energy": 1.0,
            },
        )
        with self.assertRaises(direct_z_alpha_possibility.ContractError):
            direct_z_alpha_possibility._bf_endpoint_receipt_binding(
                proposal_hash="a" * 64,
                endpoint_scale_from_raw=0.5,
                raw_path_c_energy=4.0,
                evaluation_endpoint_c_energy=2.0,
            )

    def test_zero_proposal_has_zero_leak_without_adapter_call(self) -> None:
        class Adapter:
            def __init__(self) -> None:
                self.calls = 0

            def proposal_right_leak(self, proposal: Any) -> Any:
                self.calls += 1
                return SimpleNamespace(leak_ratio=0.25)

        adapter = Adapter()
        zero = SimpleNamespace(
            factors=(
                SimpleNamespace(left=torch.zeros(2, 1), right=torch.ones(2, 1)),
            )
        )
        self.assertTrue(direct_z_alpha_possibility._proposal_is_effectively_zero(zero))
        self.assertEqual(
            direct_z_alpha_possibility._zero_safe_right_leak_ratio(adapter, zero),
            0.0,
        )
        self.assertEqual(adapter.calls, 0)

        nonzero = SimpleNamespace(
            factors=(
                SimpleNamespace(left=torch.ones(2, 1), right=torch.ones(2, 1)),
            )
        )
        self.assertFalse(
            direct_z_alpha_possibility._proposal_is_effectively_zero(nonzero)
        )
        self.assertEqual(
            direct_z_alpha_possibility._zero_safe_right_leak_ratio(adapter, nonzero),
            0.25,
        )
        self.assertEqual(adapter.calls, 1)

    def test_summary_provenance_requires_every_completed_event(self) -> None:
        key = "genuine_alpha_bf_all_hops_genuine"
        results = [
            {"pass": True, "technical": {key: True}},
            {"pass": True, "technical": {key: True}},
        ]
        self.assertTrue(
            direct_z_alpha_possibility._aggregate_event_technical_provenance(
                results,
                planned_case_count=2,
                key=key,
            )
        )
        self.assertFalse(
            direct_z_alpha_possibility._aggregate_event_technical_provenance(
                results[:1],
                planned_case_count=2,
                key=key,
            )
        )
        results[1]["technical"][key] = False
        self.assertFalse(
            direct_z_alpha_possibility._aggregate_event_technical_provenance(
                results,
                planned_case_count=2,
                key=key,
            )
        )


if __name__ == "__main__":
    unittest.main()
