from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from project.run_scripts.barrier_guided_ode.errors import BGODEScientificBoundary
from project.run_scripts.ode_edit_motivation.hooks import capture_snapshot, tensor_sha256
from project.run_scripts.ode_edit_motivation.contracts import (
    EditRequest,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
)
from project.run_scripts.barrier_guided_ode.s1_alphaedit_runtime import (
    AtomicWeightTrajectory,
    SerialForwardJVPBackend,
)
from project.run_scripts.barrier_guided_ode.s1_contract import S1Tokenization
from project.run_scripts.barrier_guided_ode.s1_streaming_events import StreamingTrieLayout


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.embedding = torch.nn.Embedding(11, 4)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4, bias=False) for _ in range(5)]
        )
        self.head = torch.nn.Linear(4, 11, bias=False)

    def forward(self, input_ids, use_cache=False):
        value = self.embedding(input_ids)
        for layer in self.layers:
            value = torch.tanh(layer(value))
        return SimpleNamespace(logits=self.head(value))


class _ToyAttentionConfig:
    def __init__(self):
        self._attn_implementation_internal = "sdpa"

    @property
    def _attn_implementation(self):
        return self._attn_implementation_internal


def _proposal(model):
    names = tuple(f"layers.{index}.weight" for index in range(5))
    request = EditRequest("case", "{}", "subject", "target")
    snapshot = capture_snapshot(
        model,
        model_id="toy",
        requests=(request,),
        context_id="ctx",
        hparams={"layers": list(range(5))},
        weight_names=names,
    )
    factors = tuple(
        LowRankFactor(
            weight_name=name,
            left=torch.tensor([[0.03], [-0.02], [0.01], [0.04]], dtype=torch.float32),
            right=torch.tensor([[0.02], [0.01], [-0.03], [0.05]], dtype=torch.float32),
            expected_weight_sha256=snapshot.parameter(name).sha256,
            native_update_transposed=False,
        )
        for name in names
    )
    return MemitFactorProposal(
        snapshot=snapshot,
        factors=factors,
        semantics=ProposalSemantics.ORDERED_GAUSS_SEIDEL,
        solver_name="toy-p-inside",
        residual_denominator=None,
    )


def _tokenization():
    return S1Tokenization(
        prompt_text="toy",
        prompt_token_ids=(1, 2),
        target_token_ids=(3,),
        source_token_ids=(4,),
        boundary_string="<|eot_id|>",
        boundary_token_id=10,
        tokenizer_name_or_path="toy",
        tokenizer_observed_commit="toy",
        tokenizer_expected_revision="toy-revision",
        tokenizer_vocab_size=11,
        identity="0" * 64,
    )


def test_serial_forward_jvp_matches_locked_central_fd_for_all_five_directions(monkeypatch):
    model = _ToyModel().eval()
    proposal = _proposal(model)
    layout = StreamingTrieLayout.build(
        source_tokens=(4,), target_tokens=(3,), boundary_token=10, vocabulary_size=11
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    backend = SerialForwardJVPBackend(model, _tokenization())
    observations = backend.observe(
        layout=layout, proposal=proposal, validate_first_node_fd=True
    )
    assert set(observations) == set(layout.internal_prefixes)
    assert backend.finite_difference_receipt is not None
    assert backend.finite_difference_receipt.allclose
    assert backend.finite_difference_receipt.direction_count == 5
    assert backend.ledger.jvp_call_count == 5 * len(layout.internal_prefixes)
    assert backend.ledger.finite_difference_forward_count == 10


def test_jvp_eager_attention_boundary_restores_entry_backend(monkeypatch):
    model = _ToyModel().eval()
    model.config = _ToyAttentionConfig()
    proposal = _proposal(model)
    layout = StreamingTrieLayout.build(
        source_tokens=(4,), target_tokens=(3,), boundary_token=10, vocabulary_size=11
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    backend = SerialForwardJVPBackend(model, _tokenization())
    backend.observe(layout=layout, proposal=proposal, validate_first_node_fd=True)
    assert model.config._attn_implementation == "sdpa"
    assert backend.ledger.attention_backend == "eager-forward-ad-reference"
    assert backend.ledger.attention_backend_switch_count == 1


def test_atomic_trajectory_restores_pointer_and_bytes_on_success_and_failure(monkeypatch):
    model = _ToyModel().eval()
    proposal = _proposal(model)
    names = tuple(factor.weight_name for factor in proposal.factors)
    parameters = dict(model.named_parameters())
    pointers = {name: parameters[name].data_ptr() for name in names}
    hashes = {name: tensor_sha256(parameters[name]) for name in names}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    with AtomicWeightTrajectory(model, names) as trajectory:
        receipt = trajectory.apply(proposal, torch.full((5,), 0.2, dtype=torch.float32))
        assert receipt.assignment_count == 5
        assert any(tensor_sha256(parameters[name]) != hashes[name] for name in names)
    assert all(parameters[name].data_ptr() == pointers[name] for name in names)
    assert all(tensor_sha256(parameters[name]) == hashes[name] for name in names)

    with pytest.raises(RuntimeError, match="injected"):
        with AtomicWeightTrajectory(model, names) as trajectory:
            trajectory.apply(proposal, torch.full((5,), 0.1, dtype=torch.float32))
            raise RuntimeError("injected")
    assert all(parameters[name].data_ptr() == pointers[name] for name in names)
    assert all(tensor_sha256(parameters[name]) == hashes[name] for name in names)


def test_atomic_trajectory_rejects_non_fp32_coefficients(monkeypatch):
    model = _ToyModel().eval()
    proposal = _proposal(model)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    with pytest.raises(BGODEScientificBoundary, match="coefficients"):
        with AtomicWeightTrajectory(model, [factor.weight_name for factor in proposal.factors]) as trajectory:
            trajectory.apply(proposal, torch.ones(5, dtype=torch.float64))
