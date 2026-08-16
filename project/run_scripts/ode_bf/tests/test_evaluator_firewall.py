from __future__ import annotations

import ast
import hashlib
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from project.run_scripts.ode_bf.benchmark import verify_pinned_evaluator_sources
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.evaluator import (
    evaluate_counterfact_rewrite_batch,
    evaluate_zsre_rewrite_batch,
)
from project.run_scripts.ode_bf.firewall import (
    assert_ast_firewall,
    assert_no_alias_specific_controller_branch,
    evaluator_request_hash,
    validate_controller_batch,
)


ALPHAEDIT_ROOT = Path("/mnt/raid5/janghj/00.KE/00.Experiment/00.EAIR_parametric/AlphaEdit")


class FakeEncoding(dict):
    def to(self, device: object) -> "FakeEncoding":
        del device
        return self


class FakeTokenizer:
    def __init__(self, *, llama: bool) -> None:
        self.llama = llama
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def _id(self, token: str) -> int:
        if token not in self._token_to_id:
            identity = len(self._token_to_id) + 2
            self._token_to_id[token] = identity
            self._id_to_token[identity] = token
        return self._token_to_id[token]

    def _encode(self, value: str) -> list[int]:
        tokens = [self._id(token) for token in value.strip().split() if token]
        return ([1] if self.llama else []) + tokens

    def __call__(
        self,
        values: str | list[str],
        *,
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, object]:
        del padding
        scalar = isinstance(values, str)
        sequences = [self._encode(values)] if scalar else [self._encode(value) for value in values]
        if return_tensors is None:
            return {"input_ids": sequences[0] if scalar else sequences}
        maximum = max(len(sequence) for sequence in sequences)
        padded = [sequence + [0] * (maximum - len(sequence)) for sequence in sequences]
        attention = [[1] * len(sequence) + [0] * (maximum - len(sequence)) for sequence in sequences]
        return FakeEncoding(
            input_ids=torch.tensor(padded, dtype=torch.long),
            attention_mask=torch.tensor(attention, dtype=torch.long),
        )

    def decode(self, values: int | list[int]) -> str:
        if isinstance(values, int):
            values = [values]
        return " ".join(self._id_to_token[value] for value in values if value not in (0, 1))


class FakeCausalLM(torch.nn.Module):
    def __init__(self, *, llama: bool) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16), requires_grad=False)
        self.config = types.SimpleNamespace(
            _name_or_path="Meta-Llama-fake" if llama else "Qwen-fake"
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        del attention_mask
        vocabulary = torch.arange(128, dtype=torch.float32).reshape(1, 1, -1)
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32).reshape(1, -1, 1)
        seeds = input_ids.float().unsqueeze(-1)
        logits = torch.cos((seeds + positions + 1.0) * (vocabulary + 1.0) * 0.017)
        return types.SimpleNamespace(logits=logits)


def _canonical_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(selected) != 1:
        raise AssertionError(f"canonical function {name} not found")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _requests() -> list[dict[str, object]]:
    requests: list[dict[str, object]] = []
    for index in range(10):
        request: dict[str, object] = {
            "case_id": index,
            "prompt": f"relation-{index} {{}}",
            "relation_id": f"R{index}",
            "subject": f"entity-{index}",
            "target_new": "alpha beta",
            "target_true": "gamma delta",
            "request_sha256": "",
        }
        request["request_sha256"] = evaluator_request_hash(request)
        requests.append(request)
    return requests


class EvaluatorParityTests(unittest.TestCase):
    def test_pinned_evaluator_source_hashes(self) -> None:
        observed = verify_pinned_evaluator_sources(ALPHAEDIT_ROOT)
        self.assertEqual(len(observed), 3)

    def test_counterfact_exact_function_parity_for_both_aliases(self) -> None:
        canonical = _canonical_function(
            ALPHAEDIT_ROOT / "experiments/py/eval_utils_counterfact.py",
            "test_batch_prediction",
        )
        requests = _requests()
        prefixes = [str(request["prompt"]).format(str(request["subject"])) for request in requests]
        for llama, alias in ((True, "llama3-8b-inst"), (False, "qwen2.5-7b-inst")):
            tokenizer = FakeTokenizer(llama=llama)
            model = FakeCausalLM(llama=llama)
            expected_probs, _ = canonical(
                model,
                tokenizer,
                prefixes,
                [0] * 10,
                "alpha beta",
                "gamma delta",
            )
            receipt = evaluate_counterfact_rewrite_batch(
                model,
                tokenizer,
                requests,
                model_alias=alias,
            )
            assert receipt.counterfact_scores is not None
            observed = [
                {
                    "target_new": score.target_new_nll,
                    "target_true": score.target_true_nll,
                }
                for score in receipt.counterfact_scores
            ]
            self.assertEqual(observed, expected_probs)
            self.assertEqual(receipt.generation_call_count, 0)

    def test_zsre_exact_teacher_forced_position_parity_for_both_aliases(self) -> None:
        canonical = _canonical_function(
            ALPHAEDIT_ROOT / "experiments/py/eval_utils_zsre.py",
            "test_batch_prediction_acc",
        )
        requests = _requests()
        for llama, alias in ((True, "llama3-8b-inst"), (False, "qwen2.5-7b-inst")):
            tokenizer = FakeTokenizer(llama=llama)
            model = FakeCausalLM(llama=llama)
            expected: list[tuple[int, ...]] = []
            for request in requests:
                prefix = str(request["prompt"]).format(str(request["subject"]))
                target_tokens = tokenizer(" alpha beta")["input_ids"]
                if llama:
                    target_tokens = target_tokens[1:]
                prompts = [
                    prefix + tokenizer.decode(target_tokens[:index])
                    if not llama or index == 0
                    else prefix + " " + tokenizer.decode(target_tokens[:index])
                    for index in range(len(target_tokens))
                ]
                targets = [tokenizer.decode(target_tokens[index]) for index in range(len(target_tokens))]
                expected.append(tuple(int(value) for value in canonical(model, tokenizer, prompts, targets)))
            receipt = evaluate_zsre_rewrite_batch(
                model,
                tokenizer,
                requests,
                model_alias=alias,
            )
            self.assertEqual(receipt.batch_success.per_case_position_bits, tuple(expected))
            self.assertEqual(receipt.generation_call_count, 0)


class FirewallTests(unittest.TestCase):
    def test_controller_schema_rejects_heldout_fields_and_duplicates(self) -> None:
        controller = []
        for request in _requests():
            item = dict(request)
            item["authorized_rewrite_prefixes"] = ()
            controller.append(item)
        identities = validate_controller_batch(controller)
        self.assertEqual(len(identities), 10)
        contaminated = [dict(item) for item in controller]
        contaminated[0]["paraphrase_prompts"] = ["forbidden"]
        with self.assertRaisesRegex(ODEBFContractError, "held-out"):
            validate_controller_batch(contaminated)
        duplicated = [dict(item) for item in controller]
        duplicated[-1] = duplicated[0]
        with self.assertRaisesRegex(ODEBFContractError, "duplicate"):
            validate_controller_batch(duplicated)

    def test_ast_firewall_and_alias_branch_guards(self) -> None:
        root = Path(__file__).resolve().parents[1]
        scientific = [
            root / name
            for name in (
                "contracts.py",
                "benchmark.py",
                "first_hit.py",
                "history.py",
                "sampling.py",
                "accounting.py",
                "analysis.py",
                "woodbury.py",
                "routing.py",
                "functional.py",
                "transaction.py",
                "evaluator.py",
            )
        ]
        assert_ast_firewall(scientific)
        assert_no_alias_specific_controller_branch(
            [root / name for name in ("first_hit.py", "history.py", "sampling.py", "routing.py")]
        )
        with tempfile.TemporaryDirectory() as directory:
            forbidden = Path(directory) / "bad.py"
            forbidden.write_text("model.generate()\n", encoding="utf-8")
            with self.assertRaisesRegex(ODEBFContractError, "generate"):
                assert_ast_firewall([forbidden])


if __name__ == "__main__":
    unittest.main()
