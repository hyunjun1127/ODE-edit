from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze_s1_llama_gh.py"
SPEC = importlib.util.spec_from_file_location("analyze_s1_llama_gh", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DYNAMIC_ARMS = MODULE.DYNAMIC_ARMS
audit_progress_localization = MODULE.audit_progress_localization


def _arm(residual: float) -> dict:
    return {"nodes": [{"nonlinear_progress_residual": residual}]}


class GHAnalysisProgressAuditTests(unittest.TestCase):
    def test_detects_unlocalized_dynamic_apply(self) -> None:
        source = """def _run_dynamic_arm():
    coefficients = (velocity * step_size)
    trajectory.apply(proposal, coefficients)

def next_function():
    pass
"""
        contract = "r(W_n+\\rho_nD_n)=r(W_n)+h and h\\rho_nu_n^\\star"
        arms = {name: _arm(float(index + 1)) for index, name in enumerate(DYNAMIC_ARMS)}
        audit = audit_progress_localization(source, contract, arms)
        self.assertTrue(audit["contract_requires_scalar_exact_progress_localization"])
        self.assertTrue(audit["dynamic_source_applies_unlocalized_coefficients"])
        self.assertEqual(audit["dynamic_node_rho_or_root_receipt_count"], 0)
        self.assertEqual(audit["maximum_absolute_nonlinear_progress_residual"], 5.0)
        self.assertFalse(audit["matched_progress_attribution_valid"])

    def test_rho_presence_prevents_unlocalized_classification(self) -> None:
        source = """def _run_dynamic_arm():
    rho = localize()
    coefficients = (velocity * step_size)
    trajectory.apply(proposal, coefficients)

def next_function():
    pass
"""
        contract = "r(W_n+\\rho_nD_n)=r(W_n)+h and h\\rho_nu_n^\\star"
        arms = {name: _arm(0.0) for name in DYNAMIC_ARMS}
        audit = audit_progress_localization(source, contract, arms)
        self.assertTrue(audit["dynamic_source_mentions_rho"])
        self.assertFalse(audit["dynamic_source_applies_unlocalized_coefficients"])


if __name__ == "__main__":
    unittest.main()
