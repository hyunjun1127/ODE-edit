"""Fail-closed feasibility and ordering for accepted-trial graph caching."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contracts import MethodContractError


class TrialGraphMode(str, Enum):
    CACHED_BEFORE_COMMIT = "cached-before-commit"
    NO_GRAD_REBUILD_AFTER_COMMIT = "no-grad-rebuild-after-commit"


@dataclass(frozen=True, slots=True)
class ProposalBackendCapabilities:
    proposal_reads_committed_dense_weights: bool
    proposal_uses_no_grad: bool
    virtual_state_identity_supported: bool


@dataclass(frozen=True, slots=True)
class CachedGraphFeasibility:
    supported: bool
    selected_mode: TrialGraphMode
    reason: str


def assess_cached_graph_feasibility(
    capabilities: ProposalBackendCapabilities,
) -> CachedGraphFeasibility:
    if capabilities.proposal_reads_committed_dense_weights:
        return CachedGraphFeasibility(
            supported=False,
            selected_mode=TrialGraphMode.NO_GRAD_REBUILD_AFTER_COMMIT,
            reason="proposal-requires-committed-dense-snapshot",
        )
    if capabilities.proposal_uses_no_grad:
        return CachedGraphFeasibility(
            supported=False,
            selected_mode=TrialGraphMode.NO_GRAD_REBUILD_AFTER_COMMIT,
            reason="proposal-path-is-no-grad",
        )
    if not capabilities.virtual_state_identity_supported:
        return CachedGraphFeasibility(
            supported=False,
            selected_mode=TrialGraphMode.NO_GRAD_REBUILD_AFTER_COMMIT,
            reason="virtual-trial-state-identity-unavailable",
        )
    return CachedGraphFeasibility(
        supported=True,
        selected_mode=TrialGraphMode.CACHED_BEFORE_COMMIT,
        reason="functional-state-and-identity-supported",
    )


# The current read-only EasyEdit bridge captures/hashes committed target
# weights and constructs synchronous proposals under torch.no_grad().
MEMIT_PROPOSAL_CAPABILITIES = ProposalBackendCapabilities(
    proposal_reads_committed_dense_weights=True,
    proposal_uses_no_grad=True,
    virtual_state_identity_supported=False,
)


class CachedAcceptedTrialSequence:
    """Enforce backward/cache-before-commit and exact delta identity."""

    def __init__(
        self,
        delta_id: str,
        capabilities: ProposalBackendCapabilities,
    ) -> None:
        if not delta_id:
            raise MethodContractError("accepted trial delta identity is empty")
        self.delta_id = delta_id
        self.feasibility = assess_cached_graph_feasibility(capabilities)
        if not self.feasibility.supported:
            raise MethodContractError(
                "cached accepted-trial graph unavailable: " + self.feasibility.reason
            )
        self._backward_complete = False
        self._next_field_id: str | None = None
        self._committed = False

    def complete_backward(self, delta_id: str) -> None:
        self._assert_delta(delta_id)
        if self._backward_complete or self._committed:
            raise MethodContractError("cached trial backward ordering is invalid")
        self._backward_complete = True

    def cache_next_field(self, delta_id: str, field_id: str) -> None:
        self._assert_delta(delta_id)
        if not self._backward_complete or self._next_field_id is not None or not field_id:
            raise MethodContractError("next field cache must follow one completed backward")
        self._next_field_id = field_id

    def authorize_commit(self, delta_id: str) -> str:
        self._assert_delta(delta_id)
        if not self._backward_complete or self._next_field_id is None or self._committed:
            raise MethodContractError("commit preceded backward/next-field cache")
        self._committed = True
        return self._next_field_id

    def _assert_delta(self, delta_id: str) -> None:
        if delta_id != self.delta_id:
            raise MethodContractError("cached graph delta differs from committed delta")
