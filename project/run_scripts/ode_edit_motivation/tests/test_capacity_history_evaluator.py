from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_edit_motivation.capacity_history_analysis import (
    BRANCH_ALPHA_NATIVE,
    BRANCHES,
    CHECKPOINT_EVENT,
    CONTRACT_KEYS,
    STREAM_SCHEMA,
    load_stream,
)
from project.run_scripts.ode_edit_motivation.capacity_history_controller import (
    RUN_IDS as CONTROLLER_RUN_IDS,
)
from project.run_scripts.ode_edit_motivation.capacity_history_evaluator import (
    EVALUATOR_RUN_IDS,
    ControllerActionEvidence,
    _execution_envelope,
    _routing_diagnostic,
    _verify_history_chain,
    evaluator_policy_parameters,
    merge_checkpoint_streams,
    run_capacity_history_evaluator,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import SanitizedJsonlWriter


def _nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        result = {str(key).lower() for key in value}
        for child in value.values():
            result.update(_nested_keys(child))
        return result
    if isinstance(value, (list, tuple)):
        result: set[str] = set()
        for child in value:
            result.update(_nested_keys(child))
        return result
    return set()


class CapacityHistoryEvaluatorTests(unittest.TestCase):
    def test_evaluation_decode_follows_all_controller_barrier(self) -> None:
        source = inspect.getsource(run_capacity_history_evaluator)
        self.assertLess(
            source.index("verify_all_controllers("),
            source.index("evaluation_loader("),
        )

    def test_policy_and_compact_schema_have_no_nfe_key(self) -> None:
        self.assertNotIn("nfe", _nested_keys(evaluator_policy_parameters()))
        self.assertNotIn("nfe", {key.lower() for key in CONTRACT_KEYS})

    def test_evaluator_identity_is_model_branch_and_controller_locked(self) -> None:
        for branch in BRANCHES:
            for model_alias, run_id in EVALUATOR_RUN_IDS[branch].items():
                _execution_envelope(branch, model_alias, run_id)
                self.assertNotEqual(run_id, CONTROLLER_RUN_IDS[branch][model_alias])
        with self.assertRaisesRegex(Exception, "identity differs"):
            _execution_envelope(
                BRANCHES[1],
                "llama3-8b-inst",
                EVALUATOR_RUN_IDS[BRANCHES[0]]["llama3-8b-inst"],
            )

    def test_alpha_history_chain_is_exactly_once_and_linked(self) -> None:
        actions = []
        previous = None
        for edit_index in range(1, 5):
            before_id = previous or "0" * 64
            after_id = f"{edit_index}" * 64
            actions.append(
                ControllerActionEvidence(
                    feature={
                        "case_id": str(edit_index),
                        "history_before": {
                            "edit_count": edit_index - 1,
                            "history_id": before_id,
                            "append_policy": (
                                "post-accepted-edit-once-history-fixed-within-edit"
                            ),
                        },
                        "history_after": {
                            "edit_count": edit_index,
                            "history_id": after_id,
                            "append_policy": (
                                "post-accepted-edit-once-history-fixed-within-edit"
                            ),
                        },
                        "history_append": {"edit_id": str(edit_index)},
                    },
                    action={},
                    result={"history_edit_count": edit_index},
                    proposal_manifests=(),
                )
            )
            previous = after_id
        _verify_history_chain(BRANCH_ALPHA_NATIVE, actions)
        broken = list(actions)
        broken[-1] = ControllerActionEvidence(
            feature={
                **broken[-1].feature,
                "history_before": {
                    **broken[-1].feature["history_before"],
                    "history_id": "f" * 64,
                },
            },
            action={},
            result=broken[-1].result,
            proposal_manifests=(),
        )
        with self.assertRaisesRegex(Exception, "history chain differs"):
            _verify_history_chain(BRANCH_ALPHA_NATIVE, broken)

    def test_routing_diagnostic_observes_overload_suppression_and_reroute(self) -> None:
        terms = []
        coefficients = []
        for index in range(5):
            overloaded = index == 0
            terms.append(
                {
                    "psi_before": 1.0 if overloaded else 0.1,
                    "linear": 0.2 if overloaded else 0.0,
                    "quadratic": 1.0,
                    "barrier": 1.0 if overloaded else 2.0,
                    "coefficient_cap": 0.0 if overloaded else 1.0,
                    "overloaded": overloaded,
                }
            )
            coefficients.append(0.0 if overloaded or index > 1 else 0.25)
        result = _routing_diagnostic(
            branch=BRANCHES[1],
            feature={
                "round_diagnostics": [
                    {
                        "accepted": True,
                        "coefficients": coefficients,
                        "capacity_terms": terms,
                    }
                ]
            },
            accepted_round_count=1,
        )
        self.assertEqual(result["overloaded_layer_observations"], 1)
        self.assertEqual(result["suppressed_overloaded_observations"], 1)
        self.assertEqual(result["capacity_reroute_round_count"], 1)
        self.assertTrue(result["capacity_barrier_exact"])

    def test_merge_resequences_four_isolated_branch_streams(self) -> None:
        contract = {"same": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for branch in BRANCHES:
                path = root / f"{branch}.jsonl"
                with SanitizedJsonlWriter(
                    path, branch, schema_version=STREAM_SCHEMA
                ) as writer:
                    for edit_index in range(1, 5):
                        writer.write(
                            CHECKPOINT_EVENT,
                            {
                                "model_alias": "llama3-8b-inst",
                                "branch": branch,
                                "edit_index": edit_index,
                                "fixed_contract": contract,
                            },
                        )
                paths[branch] = path
            merged = merge_checkpoint_streams(
                branch_paths=paths,
                output_path=root / "merged.jsonl",
                model_alias="llama3-8b-inst",
                run_id="capacity-history-merge-test",
            )
            records = load_stream(merged)
        self.assertEqual(len(records), 16)
        self.assertEqual(
            [record["branch"] for record in records],
            [branch for branch in BRANCHES for _ in range(4)],
        )


if __name__ == "__main__":
    unittest.main()
