from __future__ import annotations

import pytest
import torch

from project.run_scripts.barrier_guided_ode.event_trie import (
    JointFirstDepartureTrie,
    TerminationConvention,
)
from project.run_scripts.barrier_guided_ode.s1_contract import (
    CANONICAL_ORDER_SHA256,
    CANONICAL_STREAM_SHA256,
    NonpositiveNativeHorizon,
    S1_MODEL_BINDINGS,
    S1SampleSemanticBoundary,
    load_sealed_s1_sample,
    seal_s1_model_vocabulary,
    seal_s1_tokenization,
    s1_model_binding,
    validate_s1_horizon,
)
from project.run_scripts.barrier_guided_ode.s1_streaming_events import (
    PrefixDirectionalLogits,
    StreamingTrieLayout,
    aggregate_streaming_moments,
    evaluate_streaming_state,
)


class _Tokenizer:
    name_or_path = "sealed-llama3-tokenizer"
    _commit_hash = "tokenizer-commit"

    def __len__(self):
        return 128_256

    def convert_tokens_to_ids(self, token):
        return 128_009 if token == "<|eot_id|>" else -1

    def encode(self, text, *, add_special_tokens):
        # The locked first request is enough for a deterministic semantic
        # boundary fixture; production seals the actual tokenizer bytes.
        mapping = {
            "The occupation of Justin I is": (10, 11, 12),
            " actor": (21,),
            " politician": (22, 23),
            "The occupation of Justin I is actor": (10, 11, 12, 21),
            "The occupation of Justin I is politician": (10, 11, 12, 22, 23),
        }
        return list(mapping[text])


class _QwenTokenizer(_Tokenizer):
    name_or_path = "sealed-qwen2.5-tokenizer"

    def __len__(self):
        return 151_665

    def convert_tokens_to_ids(self, token):
        return 151_645 if token == "<|im_end|>" else -1


def test_locked_first_request_and_token_boundary_are_exact():
    sample = load_sealed_s1_sample()
    assert sample.case_id == "19795"
    assert sample.batch_label == "B1"
    assert sample.request_ordinal == 0
    assert sample.stream_root_sha256 == CANONICAL_STREAM_SHA256
    assert sample.all_request_order_sha256 == CANONICAL_ORDER_SHA256
    assert sample.request_sha256 == "285a3add6f31d8546b0d76689a016bc0f9f7d8e58c95f54e87bba48ad5cabc65"
    tokenization = seal_s1_tokenization(_Tokenizer(), sample)
    assert tokenization.boundary_string == "<|eot_id|>"
    assert tokenization.boundary_token_id == 128_009
    assert tokenization.target_token_ids == (21,)
    assert tokenization.source_token_ids == (22, 23)


def test_qwen_matched_model_binding_and_fixed_boundary_are_exact():
    sample = load_sealed_s1_sample()
    binding = s1_model_binding("qwen2.5-7b-inst")
    assert tuple(S1_MODEL_BINDINGS) == ("llama3-8b-inst", "qwen2.5-7b-inst")
    assert binding.revision == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert binding.boundary_string == "<|im_end|>"
    assert binding.boundary_token_id == 151_645
    tokenization = seal_s1_tokenization(
        _QwenTokenizer(), sample, model_alias="qwen2.5-7b-inst"
    )
    assert tokenization.model_alias == "qwen2.5-7b-inst"
    assert tokenization.boundary_string == "<|im_end|>"
    assert tokenization.boundary_token_id == 151_645
    assert tokenization.target_token_ids == (21,)
    assert tokenization.source_token_ids == (22, 23)


def test_qwen_wrong_boundary_and_unknown_model_fail_close():
    sample = load_sealed_s1_sample()

    class WrongQwen(_QwenTokenizer):
        def convert_tokens_to_ids(self, token):
            return 151_643

    with pytest.raises(S1SampleSemanticBoundary, match="token identity"):
        seal_s1_tokenization(
            WrongQwen(), sample, model_alias="qwen2.5-7b-inst"
        )
    with pytest.raises(S1SampleSemanticBoundary, match="unsupported"):
        s1_model_binding("unknown-model")


def test_qwen_event_vocabulary_binds_model_output_not_tokenizer_length():
    sample = load_sealed_s1_sample()
    tokenization = seal_s1_tokenization(
        _QwenTokenizer(), sample, model_alias="qwen2.5-7b-inst"
    )

    class Output:
        weight = torch.empty((152_064, 1), dtype=torch.float32)

    class Config:
        vocab_size = 152_064

    class Model:
        config = Config()

        @staticmethod
        def get_output_embeddings():
            return Output()

    binding = seal_s1_model_vocabulary(Model(), tokenization)
    assert binding.tokenizer_vocab_size == 151_665
    assert binding.config_vocab_size == 152_064
    assert binding.output_head_vocab_size == 152_064
    assert binding.event_vocab_size == 152_064


def test_model_config_output_vocabulary_mismatch_fails_close():
    sample = load_sealed_s1_sample()
    tokenization = seal_s1_tokenization(_Tokenizer(), sample)

    class Output:
        weight = torch.empty((128_257, 1), dtype=torch.float32)

    class Config:
        vocab_size = 128_256

    class Model:
        config = Config()

        @staticmethod
        def get_output_embeddings():
            return Output()

    with pytest.raises(S1SampleSemanticBoundary, match="config/output"):
        seal_s1_model_vocabulary(Model(), tokenization)


def test_token_boundary_and_native_horizon_fail_close():
    sample = load_sealed_s1_sample()

    class Wrong(_Tokenizer):
        def convert_tokens_to_ids(self, token):
            return 128_001

    with pytest.raises(S1SampleSemanticBoundary, match="token identity"):
        seal_s1_tokenization(Wrong(), sample)
    for value in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(NonpositiveNativeHorizon):
            validate_s1_horizon(value)
    assert validate_s1_horizon(0.125) == 0.125


def _observations(layout, logits, tangent):
    return {
        prefix: PrefixDirectionalLogits(
            logits=logits[index].clone(), tangent_logits=tangent[index].clone()
        )
        for index, prefix in enumerate(layout.internal_prefixes)
    }


def test_streaming_reducer_matches_materialized_partition_and_autograd_scores():
    generator = torch.Generator().manual_seed(41)
    vocabulary = 7
    source, target, boundary = (1, 2), (1, 3, 4), 6
    layout = StreamingTrieLayout.build(
        source_tokens=source,
        target_tokens=target,
        boundary_token=boundary,
        vocabulary_size=vocabulary,
    )
    brute = JointFirstDepartureTrie(
        source_tokens=source,
        target_tokens=target,
        termination=TerminationConvention("eot", boundary),
        vocabulary_size=vocabulary,
    )
    logits = [torch.randn(vocabulary, generator=generator) for _ in layout.internal_prefixes]
    tangent = [torch.randn(vocabulary, 3, generator=generator) for _ in layout.internal_prefixes]
    observations = _observations(layout, logits, tangent)
    initial = evaluate_streaming_state(layout, observations)
    materialized = brute.evaluate_bruteforce(
        {prefix: logits[index] for index, prefix in enumerate(layout.internal_prefixes)}
    )
    assert initial.target_log_probability == pytest.approx(
        float(materialized.target_log_probability), abs=3e-7
    )
    assert initial.source_log_probability == pytest.approx(
        float(materialized.source_log_probability), abs=3e-7
    )
    assert initial.normalization_log_residual < 2e-5

    moments = aggregate_streaming_moments(
        initial=initial, observations=observations, reference_time=0.0
    )
    def materialized_logs(beta):
        rows = {
            prefix: logits[index] + tangent[index] @ beta
            for index, prefix in enumerate(layout.internal_prefixes)
        }
        values = []
        prefix_mass = {(): torch.zeros(())}
        normalized = {}
        for prefix in brute.internal_prefixes:
            normalized[prefix] = torch.log_softmax(rows[prefix], 0)
            for token in brute.child_tokens_by_prefix[prefix]:
                prefix_mass[prefix + (token,)] = prefix_mass[prefix] + normalized[prefix][token]
        for event in brute.events:
            if event.token is None:
                values.append(prefix_mass[event.prefix])
            else:
                values.append(prefix_mass[event.prefix] + normalized[event.prefix][event.token])
        return torch.stack(values)

    beta = torch.zeros(3, requires_grad=True)
    event_logs = materialized_logs(beta)
    scores = torch.autograd.functional.jacobian(materialized_logs, beta)
    probabilities = event_logs.detach().exp()
    centered = (probabilities[:, None] * scores).sum(0)
    fisher = scores.T @ (probabilities[:, None] * scores)
    progress = scores[0] - scores[1]
    assert torch.linalg.vector_norm(centered).item() < 3e-6
    assert torch.allclose(moments.fisher.matrix, fisher, atol=3e-6, rtol=3e-6)
    assert torch.allclose(moments.progress_sensitivity, progress, atol=3e-6, rtol=3e-6)
    assert torch.linalg.vector_norm(moments.anchor_gradient).item() < 3e-6
