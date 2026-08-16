from __future__ import annotations

import hashlib
import types
import unittest

import torch

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.barriers import functional_replay_risk
from project.run_scripts.ode_bf.p1_replay import (
    Theta0TeacherCache,
    build_outer_entry_pretrained_cache,
    build_theta0_teacher_cache,
    capture_pretrained_entry_kl,
    evaluate_functional_replay_pair,
    evaluate_next_token_log_probs,
    samplewise_teacher_kl,
)


class _Batch(dict):
    def to(self, device: torch.device) -> "_Batch":
        return _Batch({key: value.to(device) for key, value in self.items()})


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    @staticmethod
    def _tokens(value: str) -> list[int]:
        return [1] + [2 + (ord(char) % 13) for char in value]

    def __call__(
        self,
        values: str | list[str],
        *,
        padding: bool = False,
        return_tensors: str | None = None,
    ):
        del padding
        if isinstance(values, str):
            return {"input_ids": self._tokens(values)}
        rows = [self._tokens(value) for value in values]
        if return_tensors is None:
            return {"input_ids": rows}
        maximum = max(map(len, rows))
        padded = [row + [0] * (maximum - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (maximum - len(row)) for row in rows]
        return _Batch(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        )


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor(0.5, dtype=torch.bfloat16), requires_grad=False
        )
        self.config = types.SimpleNamespace(
            _name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ):
        del attention_mask
        vocabulary = torch.arange(32, dtype=torch.float32, device=input_ids.device)
        logits = -torch.abs(
            vocabulary.reshape(1, 1, -1)
            - (input_ids % 29).unsqueeze(-1).float()
        ) * self.scale.float()
        return types.SimpleNamespace(logits=logits.to(torch.bfloat16))


def _requests(count: int = 10) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "request_sha256": hashlib.sha256(f"p-{index}".encode()).hexdigest(),
            "prompt": "subject {}",
            "subject": str(index),
            "target_new": "target",
            "target_true": "old",
        }
        for index in range(count)
    )


class P1ReplayTests(unittest.TestCase):
    def test_samplewise_kl_is_zero_for_identity_and_positive_for_shift(self) -> None:
        teacher = torch.log_softmax(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1
        )
        self.assertTrue(
            torch.equal(
                samplewise_teacher_kl(teacher, teacher),
                torch.zeros(2, dtype=torch.float64),
            )
        )
        observed = torch.log_softmax(
            torch.tensor([[0.0, 1.0], [1.0, 0.0]]), dim=1
        )
        self.assertTrue(torch.all(samplewise_teacher_kl(teacher, observed) > 0.0))

    def test_noop_functional_pair_uses_theta0_and_no_generation(self) -> None:
        model = _Model()
        tokenizer = _Tokenizer()
        requests = _requests()
        log_probs, receipt = evaluate_next_token_log_probs(model, tokenizer, requests)
        cache = Theta0TeacherCache(
            tuple(str(item["request_sha256"]) for item in requests),
            {
                str(item["request_sha256"]): log_probs[index].clone()
                for index, item in enumerate(requests)
            },
            canonical_hash([str(item["request_sha256"]) for item in requests]),
            receipt.value_sha256,
            1,
            receipt.processed_token_count,
        )
        entry_kl, _ = capture_pretrained_entry_kl(
            model, tokenizer, requests, cache
        )
        pair = evaluate_functional_replay_pair(
            model,
            tokenizer,
            model_alias="llama3-8b-inst",
            history_requests=(),
            history_entry_nll=torch.empty((0,), dtype=torch.float64),
            pretrained_requests=requests,
            pretrained_entry_kl=entry_kl,
            theta0_cache=cache,
            trial_factors_by_weight={},
            historical_budget=1.0e-3,
            pretrained_budget=1.0e-3,
            smoothmax_temperature=1.0e-2,
        )
        self.assertTrue(pair.historical.passed)
        self.assertTrue(pair.pretrained.passed)
        self.assertEqual(pair.historical.item_count, 0)
        self.assertEqual(pair.pretrained.item_count, 10)
        self.assertEqual(pair.pretrained.signed_mean_damage, 0.0)
        self.assertEqual(pair.pretrained_trial_receipt.generation_call_count, 0)

    def test_fixed_outer_entry_cache_catches_cumulative_drift(self) -> None:
        model = _Model()
        tokenizer = _Tokenizer()
        population = _requests(160)
        pointer = model.scale.data_ptr()
        version = model.scale._version
        rng = torch.get_rng_state().clone()
        theta0 = build_theta0_teacher_cache(
            model,
            tokenizer,
            population,
            chunk_size=10,
        )
        with torch.no_grad():
            model.scale.fill_(0.75)
        outer_version = model.scale._version
        cache = build_outer_entry_pretrained_cache(
            model,
            tokenizer,
            population,
            theta0,
            outer_entry_snapshot_sha256="a" * 64,
        )
        self.assertEqual(model.scale.data_ptr(), pointer)
        self.assertEqual(model.scale._version, outer_version)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        identities = tuple(str(item["request_sha256"]) for item in population[:10])
        fixed_entry, fixed_receipt, baseline = cache.select(identities)
        self.assertEqual(baseline.baseline_kind, "outer_entry")
        self.assertEqual(baseline.outer_entry_snapshot_sha256, "a" * 64)
        self.assertEqual(baseline.entry_kl_identity_sha256, fixed_receipt.value_sha256)

        with torch.no_grad():
            model.scale.fill_(1.25)
        trial_kl, _ = capture_pretrained_entry_kl(
            model,
            tokenizer,
            population[:10],
            theta0,
        )
        fixed_risk = functional_replay_risk(
            barrier="pretrained-theta0-teacher",
            entry_values=fixed_entry.tolist(),
            trial_values=trial_kl.tolist(),
            sample_sha256=identities,
            budget=1.0,
            smooth_max_temperature=1.0e-2,
        )
        moving_risk = functional_replay_risk(
            barrier="pretrained-theta0-teacher",
            entry_values=trial_kl.tolist(),
            trial_values=trial_kl.tolist(),
            sample_sha256=identities,
            budget=1.0,
            smooth_max_temperature=1.0e-2,
        )
        repeated, repeated_receipt, repeated_baseline = cache.select(identities)
        self.assertGreater(fixed_risk.mean_positive_damage, 0.0)
        self.assertEqual(moving_risk.mean_positive_damage, 0.0)
        self.assertTrue(torch.equal(repeated, fixed_entry))
        self.assertEqual(repeated_receipt, fixed_receipt)
        self.assertEqual(repeated_baseline, baseline)
        self.assertEqual(model.scale.data_ptr(), pointer)
        self.assertGreaterEqual(model.scale._version, outer_version)
        self.assertGreater(model.scale._version, version)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_fixed_outer_entry_cache_selection_fails_closed(self) -> None:
        model = _Model()
        tokenizer = _Tokenizer()
        population = _requests(160)
        theta0 = build_theta0_teacher_cache(model, tokenizer, population)
        cache = build_outer_entry_pretrained_cache(
            model,
            tokenizer,
            population,
            theta0,
            outer_entry_snapshot_sha256="b" * 64,
        )
        identities = tuple(str(item["request_sha256"]) for item in population[:10])
        with self.assertRaisesRegex(Exception, "selection differs"):
            cache.select(identities[:-1] + (identities[0],))
        cache.entry_kl_by_request[identities[0]].add_(1.0)
        with self.assertRaisesRegex(Exception, "value mutated"):
            cache.select(identities)


if __name__ == "__main__":
    unittest.main()
