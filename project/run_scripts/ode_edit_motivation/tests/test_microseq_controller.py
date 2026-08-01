from __future__ import annotations

import inspect
import unittest

from project.run_scripts.ode_edit_motivation.contracts import ExpectedFileIdentity
from project.run_scripts.ode_edit_motivation.manifests import build_counterfact_selection
from project.run_scripts.ode_edit_motivation.microseq_analysis import (
    BRANCH_NATIVE,
    BRANCH_ODE,
)
from project.run_scripts.ode_edit_motivation.microseq_controller import (
    APPLICATION_MODES,
    MICROSEQ_RANK_START,
    MICROSEQ_RANK_STOP,
    RUN_IDS,
    _execution_envelope,
    _salted_rank,
    select_microseq_cases,
)
import project.run_scripts.ode_edit_motivation.microseq_controller as controller


class MicroseqControllerTests(unittest.TestCase):
    def test_selection_is_exact_disjoint_rank_slice(self) -> None:
        case_ids = tuple(str(index) for index in range(300))
        identity = ExpectedFileIdentity(sha256="a" * 64, size=123)
        canonical = build_counterfact_selection(
            case_ids,
            source_identity=identity,
            split_counts=(20, 60, 20),
        )
        selection = select_microseq_cases(case_ids, canonical=canonical)
        ranked = tuple(sorted(case_ids, key=_salted_rank))
        self.assertEqual(
            selection.case_ids, ranked[MICROSEQ_RANK_START:MICROSEQ_RANK_STOP]
        )
        self.assertTrue(set(selection.case_ids).isdisjoint(ranked[:MICROSEQ_RANK_START]))
        self.assertEqual(len(selection.manifest_id), 64)
        reversed_selection = select_microseq_cases(
            tuple(reversed(case_ids)), canonical=canonical
        )
        self.assertEqual(selection, reversed_selection)

    def test_controller_identity_is_exact(self) -> None:
        for branch, by_model in RUN_IDS.items():
            for model, run_id in by_model.items():
                _execution_envelope(branch, model, run_id)
        with self.assertRaisesRegex(Exception, "identity differs"):
            _execution_envelope(
                BRANCH_ODE, "llama3-8b-inst", RUN_IDS[BRANCH_NATIVE]["llama3-8b-inst"]
            )

    def test_application_modes_are_branch_locked(self) -> None:
        self.assertEqual(APPLICATION_MODES[BRANCH_NATIVE], "easyedit_exact")
        self.assertEqual(APPLICATION_MODES[BRANCH_ODE], "low_rank_addmm")

    def test_controller_source_has_no_evaluation_field_loader(self) -> None:
        source = inspect.getsource(controller)
        for forbidden in (
            "neighborhood_prompts",
            "generation_prompts",
            "target_true",
            "load_counterfact_evaluation",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
