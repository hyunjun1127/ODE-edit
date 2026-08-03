from __future__ import annotations

import unittest

from project.run_scripts.ode_edit_method.cached_graph import (
    MEMIT_PROPOSAL_CAPABILITIES,
    CachedAcceptedTrialSequence,
    ProposalBackendCapabilities,
    TrialGraphMode,
    assess_cached_graph_feasibility,
)
from project.run_scripts.ode_edit_method.contracts import MethodContractError


class CachedGraphFeasibilityTests(unittest.TestCase):
    def test_current_memit_path_fails_closed_to_no_grad(self) -> None:
        feasibility = assess_cached_graph_feasibility(MEMIT_PROPOSAL_CAPABILITIES)
        self.assertFalse(feasibility.supported)
        self.assertEqual(
            feasibility.selected_mode,
            TrialGraphMode.NO_GRAD_REBUILD_AFTER_COMMIT,
        )
        self.assertEqual(
            feasibility.reason,
            "proposal-requires-committed-dense-snapshot",
        )
        with self.assertRaises(MethodContractError):
            CachedAcceptedTrialSequence("delta-a", MEMIT_PROPOSAL_CAPABILITIES)

    def test_supported_backend_enforces_backward_cache_commit_order(self) -> None:
        capabilities = ProposalBackendCapabilities(
            proposal_reads_committed_dense_weights=False,
            proposal_uses_no_grad=False,
            virtual_state_identity_supported=True,
        )
        sequence = CachedAcceptedTrialSequence("delta-a", capabilities)
        with self.assertRaises(MethodContractError):
            sequence.authorize_commit("delta-a")
        with self.assertRaises(MethodContractError):
            sequence.complete_backward("delta-b")
        sequence.complete_backward("delta-a")
        with self.assertRaises(MethodContractError):
            sequence.authorize_commit("delta-a")
        sequence.cache_next_field("delta-a", "field-next")
        self.assertEqual(sequence.authorize_commit("delta-a"), "field-next")
        with self.assertRaises(MethodContractError):
            sequence.authorize_commit("delta-a")


if __name__ == "__main__":
    unittest.main()
