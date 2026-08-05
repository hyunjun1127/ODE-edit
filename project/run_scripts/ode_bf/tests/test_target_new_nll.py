from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import re
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1_backend import _controller_margin_loss
from project.run_scripts.ode_bf.request_digest import ordered_request_digest_v1
from project.run_scripts.ode_bf.target_new_nll import (
    RoutingObjective,
    evaluate_routing_objective,
    select_locked_routing_objective,
)


class _TokenBatch(dict[str, torch.Tensor]):
    def to(self, device: torch.device) -> "_TokenBatch":
        return _TokenBatch({name: value.to(device) for name, value in self.items()})


class _Tokenizer:
    def __init__(self, *, llama: bool = False) -> None:
        self.padding_side = "right"
        self.llama = llama
        self._vocabulary = {"<pad>": 0, "<bos>": 1}
        self.calls: list[tuple[tuple[str, ...], bool, str | None, str]] = []
        self.fail_substring: str | None = None

    def _token_id(self, token: str) -> int:
        if token not in self._vocabulary:
            self._vocabulary[token] = len(self._vocabulary)
        return self._vocabulary[token]

    def ids(self, text: str) -> list[int]:
        pieces = text.split()
        result = [1] if self.llama else []
        result.extend(self._token_id(piece) for piece in pieces)
        return result

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> Mapping[str, Any]:
        values = (texts,) if isinstance(texts, str) else tuple(texts)
        self.calls.append((values, padding, return_tensors, self.padding_side))
        if self.fail_substring is not None and any(
            self.fail_substring in value for value in values
        ):
            raise RuntimeError("tokenizer fixture failure")
        rows = [self.ids(value) for value in values]
        if return_tensors != "pt":
            return {"input_ids": rows[0] if isinstance(texts, str) else rows}
        width = max(len(row) for row in rows)
        padded: list[list[int]] = []
        attention: list[list[int]] = []
        for row in rows:
            pad = width - len(row)
            if padding and self.padding_side == "left":
                padded.append([0] * pad + row)
                attention.append([0] * pad + [1] * len(row))
            elif padding:
                padded.append(row + [0] * pad)
                attention.append([1] * len(row) + [0] * pad)
            elif pad:
                raise AssertionError("fixture only supports padded variable rows")
            else:
                padded.append(row)
                attention.append([1] * len(row))
        return _TokenBatch(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        )


class _ToyCausalLM(torch.nn.Module):
    def __init__(self, *, llama: bool = False) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros((256, 256), dtype=torch.float32))
        self.config = SimpleNamespace(_name_or_path="llama-fixture" if llama else "qwen-fixture")
        self.forward_calls = 0
        self.generate_calls = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: Any,
    ) -> SimpleNamespace:
        del attention_mask
        self.forward_calls += 1
        return SimpleNamespace(logits=self.table[input_ids])

    def generate(self, *_: Any, **__: Any) -> None:
        self.generate_calls += 1
        raise AssertionError("routing objective must never generate")


class _MutatingToyCausalLM(_ToyCausalLM):
    def forward(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        with torch.no_grad():
            self.table.add_(0.01)
        return super().forward(*args, **kwargs)


class _ScaledToyCausalLM(_ToyCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: Any,
    ) -> SimpleNamespace:
        del attention_mask
        self.forward_calls += 1
        return SimpleNamespace(logits=self.table[input_ids] * self.scale[0])


@dataclass(frozen=True)
class _HeldOutTrap:
    def __str__(self) -> str:
        raise AssertionError("held-out content entered routing")


@contextmanager
def _assert_raises(exception: type[BaseException], match: str):
    try:
        yield
    except exception as exc:
        if re.search(match, str(exc)) is None:
            raise AssertionError(
                f"{exception.__name__} message {str(exc)!r} does not match {match!r}"
            ) from exc
    else:
        raise AssertionError(f"{exception.__name__} was not raised")


def _requests(
    *,
    new_lengths: Sequence[int] | None = None,
    true_lengths: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    new_lengths = tuple(new_lengths or (1,) * 10)
    true_lengths = tuple(true_lengths or (1,) * 10)
    assert len(new_lengths) == len(true_lengths) == 10
    return [
        {
            "request_sha256": f"{ordinal + 1:064x}",
            "prompt": "{} relation",
            "subject": f"subject{ordinal}",
            "target_new": " ".join(
                [f"new{ordinal}"]
                + [f"new{ordinal}_{index}" for index in range(1, new_lengths[ordinal])]
            ),
            "target_true": " ".join(
                [f"old{ordinal}"]
                + [f"old{ordinal}_{index}" for index in range(1, true_lengths[ordinal])]
            ),
        }
        for ordinal in range(10)
    ]


def _contexts() -> tuple[tuple[str, ...], ...]:
    return (("{}",), tuple(f"locked-context-{index} {{}}" for index in range(5)))


def _suffix_ids(tokenizer: _Tokenizer, target: object) -> list[int]:
    ids = tokenizer.ids(f" {target}")
    return ids[1:] if tokenizer.llama else ids


def _configure(
    model: _ToyCausalLM,
    tokenizer: _Tokenizer,
    requests: Sequence[Mapping[str, object]],
    *,
    new_logit: float = 1.0,
    old_logit: float = 1.0,
) -> None:
    with torch.no_grad():
        model.table.zero_()
        for request in requests:
            prefix = str(request["prompt"]).format(str(request["subject"]))
            prefix_ids = tokenizer.ids(prefix)
            for key, logit in (("target_new", new_logit), ("target_true", old_logit)):
                if key not in request:
                    continue
                previous = prefix_ids[-1]
                for token in _suffix_ids(tokenizer, request[key]):
                    model.table[previous, token] = logit
                    previous = token


def _set_logits(
    model: _ToyCausalLM,
    tokenizer: _Tokenizer,
    requests: Sequence[Mapping[str, object]],
    *,
    key: str,
    logit: float,
) -> None:
    with torch.no_grad():
        for request in requests:
            prefix = str(request["prompt"]).format(str(request["subject"]))
            previous = tokenizer.ids(prefix)[-1]
            for token in _suffix_ids(tokenizer, request[key]):
                model.table[previous, token] = logit
                previous = token


def _evaluate(
    model: _ToyCausalLM,
    tokenizer: _Tokenizer,
    requests: Sequence[Mapping[str, object]],
    objective: RoutingObjective | str = RoutingObjective.TARGET_NEW_NLL,
):
    selected = select_locked_routing_objective(objective)
    return evaluate_routing_objective(
        model,
        tokenizer,
        requests,
        objective=selected,
        contexts=_contexts()
        if selected is RoutingObjective.TARGET_NEW_NLL
        else None,
    )


def test_margin_is_fixture_identical_to_the_current_controller_loss() -> None:
    tokenizer = _Tokenizer()
    tokenizer.padding_side = "left"
    requests = _requests()
    model = _ToyCausalLM()
    _configure(model, tokenizer, requests, new_logit=0.3, old_logit=1.7)

    expected, expected_per_request, expected_processed = _controller_margin_loss(
        model, tokenizer, requests
    )
    observed = _evaluate(model, tokenizer, requests, RoutingObjective.MARGIN)

    torch.testing.assert_close(observed.loss, expected)
    torch.testing.assert_close(observed.per_request_values, expected_per_request)
    assert observed.processed_token_count == expected_processed
    assert observed.model_forward_count == 10
    assert observed.context_group_sizes == (1,)
    assert observed.context_count == 1
    assert observed.target_true_suffix_token_counts == (1,) * 10


def test_target_new_has_uniform_request_and_suffix_token_weighting() -> None:
    tokenizer = _Tokenizer()
    requests = _requests(new_lengths=(1, 2, 2, 2, 2, 2, 2, 2, 2, 2))
    model = _ToyCausalLM()
    _configure(model, tokenizer, requests, new_logit=2.0)
    _set_logits(model, tokenizer, requests[:1], key="target_new", logit=-2.0)

    observed = _evaluate(model, tokenizer, requests)
    values = observed.per_request_values.detach()
    token_weighted = sum(
        value * count for value, count in zip(values, observed.suffix_token_counts)
    ) / sum(observed.suffix_token_counts)

    torch.testing.assert_close(observed.loss.detach(), values.mean())
    assert observed.suffix_token_counts == (1, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    assert not torch.isclose(observed.loss.detach(), token_weighted)
    assert observed.context_group_sizes == (1, 5)
    assert observed.context_count == 6
    assert observed.model_forward_count == 60


def test_target_new_supports_no_grad_candidate_measurement() -> None:
    tokenizer = _Tokenizer()
    requests = _requests()
    model = _ToyCausalLM()
    _configure(model, tokenizer, requests, new_logit=2.0)

    with torch.no_grad():
        observed = _evaluate(model, tokenizer, requests)

    assert not observed.graph_requires_grad
    assert observed.model_forward_count == 60


def test_target_new_streamed_request_gradient_matches_monolithic_objective() -> None:
    tokenizer = _Tokenizer()
    requests = _requests(new_lengths=(1, 2, 3, 1, 2, 3, 1, 2, 3, 1))
    model = _ScaledToyCausalLM()
    _configure(model, tokenizer, requests, new_logit=1.75)

    monolithic = _evaluate(model, tokenizer, requests)
    expected = torch.autograd.grad(monolithic.loss, model.scale)[0]
    streamed = evaluate_routing_objective(
        model,
        tokenizer,
        requests,
        objective=RoutingObjective.TARGET_NEW_NLL,
        contexts=_contexts(),
        gradient_input=model.scale,
    )

    assert streamed.input_gradient is not None
    torch.testing.assert_close(streamed.input_gradient, expected)
    torch.testing.assert_close(streamed.loss, monolithic.loss.detach())
    torch.testing.assert_close(
        streamed.per_request_values, monolithic.per_request_values.detach()
    )
    assert streamed.backward_count == 10
    assert streamed.model_forward_count == 60
    assert not streamed.graph_requires_grad


def test_target_new_contexts_are_uniformly_weighted_across_all_six_surfaces() -> None:
    tokenizer = _Tokenizer()
    requests = _requests()
    model = _ToyCausalLM()
    contexts = (
        ("{}",),
        tuple(f"{{}} trailing-context-{index}" for index in range(5)),
    )
    context_logits = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
    with torch.no_grad():
        model.table.zero_()
        for request in requests:
            target_token = _suffix_ids(tokenizer, request["target_new"])[0]
            flattened = tuple(template for group in contexts for template in group)
            for template, logit in zip(flattened, context_logits, strict=True):
                prefix = template.format(str(request["prompt"])).format(
                    str(request["subject"])
                )
                previous = tokenizer.ids(prefix)[-1]
                model.table[previous, target_token] = logit
    observed = evaluate_routing_objective(
        model,
        tokenizer,
        requests,
        objective=RoutingObjective.TARGET_NEW_NLL,
        contexts=contexts,
    )
    target_token = _suffix_ids(tokenizer, requests[0]["target_new"])[0]
    expected_surfaces = []
    flattened = tuple(template for group in contexts for template in group)
    for template in flattened:
        prefix = template.format(str(requests[0]["prompt"])).format(
            str(requests[0]["subject"])
        )
        previous = tokenizer.ids(prefix)[-1]
        expected_surfaces.append(
            -torch.log_softmax(model.table[previous], dim=0)[target_token]
        )
    expected = torch.stack(expected_surfaces).mean()

    torch.testing.assert_close(observed.per_request_values[0], expected)
    torch.testing.assert_close(observed.loss, observed.per_request_values.mean())
    assert observed.context_count == 6


def test_target_true_never_influences_target_new_value_gradient_or_tokenizer_trace() -> None:
    tokenizer = _Tokenizer()
    requests = _requests()
    for request in requests:
        request["target_true"] = "old-forbidden"
    model = _ToyCausalLM()
    _configure(model, tokenizer, requests, new_logit=1.25, old_logit=-2.0)
    tokenizer.fail_substring = "old-forbidden"

    with_old = _evaluate(model, tokenizer, requests)
    grad_with_old = torch.autograd.grad(with_old.loss, model.table)[0]
    calls_with_old = tuple(tokenizer.calls)
    without_old = [{key: value for key, value in request.items() if key != "target_true"} for request in requests]
    tokenizer.calls.clear()
    without_old_result = _evaluate(model, tokenizer, without_old)
    grad_without_old = torch.autograd.grad(without_old_result.loss, model.table)[0]

    torch.testing.assert_close(with_old.loss, without_old_result.loss)
    torch.testing.assert_close(grad_with_old, grad_without_old)
    assert all(
        "old-forbidden" not in text
        for texts, _, _, _ in calls_with_old
        for text in texts
    )
    assert all(
        "old-forbidden" not in text
        for texts, _, _, _ in tokenizer.calls
        for text in texts
    )
    assert with_old.target_true_suffix_token_counts == (None,) * 10


def test_old_only_degradation_is_not_target_new_progress_but_new_nll_reduction_is() -> None:
    tokenizer = _Tokenizer()
    requests = _requests()
    model = _ToyCausalLM()
    _configure(model, tokenizer, requests, new_logit=0.5, old_logit=0.5)
    target_before = _evaluate(model, tokenizer, requests)

    mutated_old = [dict(request) for request in requests]
    for ordinal, request in enumerate(mutated_old):
        request["target_true"] = f"diagnostic-old-only-{ordinal}"
    target_after_old_degradation = _evaluate(model, tokenizer, mutated_old)
    old_nll_before = 1.0
    old_nll_after = 5.0
    legacy_margin_before = float(target_before.loss.detach()) - old_nll_before
    legacy_margin_after = float(target_after_old_degradation.loss.detach()) - old_nll_after

    assert legacy_margin_before - legacy_margin_after > 0.0
    torch.testing.assert_close(target_before.loss, target_after_old_degradation.loss)

    _set_logits(model, tokenizer, requests, key="target_new", logit=4.0)
    target_after_new_improvement = _evaluate(model, tokenizer, requests)
    assert (target_after_old_degradation.loss - target_after_new_improvement.loss).item() > 0.0


def test_b10_order_and_raw_free_span_identities_are_explicit() -> None:
    tokenizer = _Tokenizer()
    requests = _requests(new_lengths=(1, 2, 3, 1, 2, 3, 1, 2, 3, 1))
    model = _ToyCausalLM()
    _configure(model, tokenizer, requests, new_logit=1.0)

    observed = _evaluate(model, tokenizer, requests)
    reversed_observed = _evaluate(model, tokenizer, list(reversed(requests)))

    assert observed.request_sha256 == tuple(request["request_sha256"] for request in requests)
    assert observed.request_order_sha256 == ordered_request_digest_v1(observed.request_sha256)
    assert observed.suffix_token_counts == (1, 2, 3, 1, 2, 3, 1, 2, 3, 1)
    assert len(set(observed.target_span_identities)) == 10
    assert observed.request_order_sha256 != reversed_observed.request_order_sha256
    assert observed.target_span_sha256 != reversed_observed.target_span_sha256


def test_locked_llama_and_non_llama_suffix_shifts_score_one_token_targets() -> None:
    for llama in (False, True):
        tokenizer = _Tokenizer(llama=llama)
        requests = _requests()
        model = _ToyCausalLM(llama=llama)
        _configure(model, tokenizer, requests, new_logit=2.0)

        observed = _evaluate(model, tokenizer, requests)

        assert observed.objective is RoutingObjective.TARGET_NEW_NLL
        assert observed.suffix_token_counts == (1,) * 10
        assert observed.model_forward_count == 60


def test_llama_multi_token_suffix_alignment_and_streamed_gradient() -> None:
    lengths = (2, 3, 4, 2, 3, 4, 2, 3, 4, 2)
    tokenizer = _Tokenizer(llama=True)
    requests = _requests(new_lengths=lengths)
    model = _ScaledToyCausalLM()
    model.config = SimpleNamespace(_name_or_path="llama-fixture")
    _configure(model, tokenizer, requests, new_logit=2.0)

    observed = evaluate_routing_objective(
        model,
        tokenizer,
        requests,
        objective=RoutingObjective.TARGET_NEW_NLL,
        contexts=_contexts(),
        gradient_input=model.scale,
    )

    assert observed.suffix_token_counts == lengths
    assert observed.model_forward_count == 60
    assert observed.backward_count == 10
    assert observed.input_gradient is not None
    assert tuple(observed.input_gradient.shape) == (1,)
    assert bool(torch.isfinite(observed.input_gradient).all())


def test_padding_side_is_restored_even_when_tokenization_raises() -> None:
    tokenizer = _Tokenizer()
    requests = _requests()
    model = _ToyCausalLM()
    _configure(model, tokenizer, requests)

    _evaluate(model, tokenizer, requests)
    assert tokenizer.padding_side == "right"

    tokenizer.fail_substring = "new0"
    with _assert_raises(RuntimeError, "fixture failure"):
        _evaluate(model, tokenizer, requests)
    assert tokenizer.padding_side == "right"


def test_no_generation_or_held_out_inputs_are_used() -> None:
    tokenizer = _Tokenizer()
    requests = _requests()
    for request in requests:
        request["paraphrase_prompts"] = _HeldOutTrap()
        request["neighborhood_prompts"] = _HeldOutTrap()
    model = _ToyCausalLM()
    _configure(model, tokenizer, requests)

    _evaluate(model, tokenizer, requests)

    assert model.generate_calls == 0


def test_nonpositive_or_invalid_inputs_fail_closed() -> None:
    tokenizer = _Tokenizer()
    requests = _requests()
    model = _ToyCausalLM()
    _configure(model, tokenizer, requests)

    with _assert_raises(ODEBFContractError, "not locked"):
        select_locked_routing_objective("margin")
    with _assert_raises(ODEBFContractError, "joint B10"):
        _evaluate(model, tokenizer, requests[:-1])
    with _assert_raises(ODEBFContractError, "contexts are absent"):
        evaluate_routing_objective(
            model,
            tokenizer,
            requests,
            objective=RoutingObjective.TARGET_NEW_NLL,
        )
    with _assert_raises(ODEBFContractError, "context groups differ"):
        evaluate_routing_objective(
            model,
            tokenizer,
            requests,
            objective=RoutingObjective.TARGET_NEW_NLL,
            contexts=(("{}",),),
        )
    repeated = [dict(request) for request in requests]
    repeated[-1]["request_sha256"] = repeated[0]["request_sha256"]
    with _assert_raises(ODEBFContractError, "duplicate"):
        _evaluate(model, tokenizer, repeated)
    malformed = [dict(request) for request in requests]
    malformed[-1]["request_sha256"] = "not-a-sha256"
    with _assert_raises(ODEBFContractError, "invalid SHA"):
        _evaluate(model, tokenizer, malformed)
    empty_suffix = [dict(request) for request in requests]
    empty_suffix[0]["target_new"] = " "
    with _assert_raises(ODEBFContractError, "suffix is empty"):
        _evaluate(model, tokenizer, empty_suffix)
    tokenizer.padding_side = "center"
    with _assert_raises(ODEBFContractError, "padding side differs"):
        _evaluate(model, tokenizer, requests)
    tokenizer.padding_side = "right"

    with torch.no_grad():
        model.table.fill_(float("inf"))
    with _assert_raises(ODEBFContractError, "non-finite"):
        _evaluate(model, tokenizer, requests)

    mutating = _MutatingToyCausalLM()
    _configure(mutating, tokenizer, requests)
    with _assert_raises(ODEBFContractError, "mutated model state"):
        _evaluate(mutating, tokenizer, requests)


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value))
    return suite


if __name__ == "__main__":
    unittest.main()
