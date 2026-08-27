from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from project.run_scripts.barrier_guided_ode.r2.errors import PrefixEventBoundary
from project.run_scripts.barrier_guided_ode.r2.stage_b_contract import (
    seal_model_vocabulary,
    seal_tokenization,
)
from project.run_scripts.barrier_guided_ode.r2.stage_b_probe import run_stage_b


class _Tokenizer:
    name_or_path = "sealed/fake"
    _commit_hash = "a" * 40

    def __len__(self) -> int:
        return 8

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        table = {
            ("P S", True): [4],
            (" new", False): [1, 2],
            (" old", False): [1, 3],
            ("P S new", True): [4, 1, 2],
            ("P S old", True): [4, 1, 3],
        }
        return table[(text, add_special_tokens)]


def _sample() -> object:
    return SimpleNamespace(
        edit_request=SimpleNamespace(prompt="P {}", subject="S", target_new="new"),
        target_true="old",
    )


def test_r2_tokenization_has_no_termination_choice_and_supports_vocab_mismatch() -> None:
    tokenization = seal_tokenization(_Tokenizer(), _sample(), model_alias="llama3-8b-inst")
    receipt = tokenization.receipt()
    assert receipt["termination_token_count"] == 0
    assert "boundary" not in "\n".join(receipt)
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=10),
        get_output_embeddings=lambda: SimpleNamespace(weight=torch.zeros((10, 3))),
    )
    vocabulary = seal_model_vocabulary(model, tokenization)
    assert vocabulary.tokenizer_vocabulary_size == 8
    assert vocabulary.output_head_vocabulary_size == 10


def test_equal_source_target_tokenization_fails_close() -> None:
    tokenizer = _Tokenizer()
    tokenizer.encode = lambda text, add_special_tokens: [4] if add_special_tokens else [1, 2]  # type: ignore[method-assign]
    with pytest.raises(PrefixEventBoundary, match="empty or equal"):
        seal_tokenization(tokenizer, _sample(), model_alias="llama3-8b-inst")


def test_stage_b_source_orders_validation_before_fp64_solver_and_has_no_dynamic_write() -> None:
    source = Path(inspect.getfile(run_stage_b)).read_text(encoding="utf-8")
    proposal = source.index("propose_validated_ordered(")
    solve = source.index("solve_two_equality_rayleighian(")
    assert proposal < solve
    assert '"dynamic_writer_action_count": 0' in source
    assert "solution.physical_coefficient_fp32(0.25)" in source
    assert "trajectory.apply(proposal, coefficients)" in source  # technical fidelity only


def test_stage_b_array_mapping_and_cap_are_fixed() -> None:
    path = Path(inspect.getfile(run_stage_b)).parents[2] / "session05_bgode_r2_stage_b.sbatch"
    source = path.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1%2" in source
    assert '0)\n    readonly MODEL_ALIAS="llama3-8b-inst"' in source
    assert '1)\n    readonly MODEL_ALIAS="qwen2.5-7b-inst"' in source
    assert "--gres=gpu:1" in source
