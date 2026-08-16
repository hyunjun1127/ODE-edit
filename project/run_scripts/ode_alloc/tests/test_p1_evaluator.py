from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_alloc.contracts import Arm, ODEAllocContractError
from project.run_scripts.ode_alloc.p1_contracts import EndpointFreezeToken
from project.run_scripts.ode_alloc.p1_evaluator import (
    EvaluationCase,
    evaluate_frozen_endpoint,
    load_evaluation_cases_after_freeze,
)
from project.run_scripts.ode_alloc.p1_scoring import (
    TeacherForcedScorer,
    TeacherForcedSpec,
)
from project.run_scripts.ode_alloc.selection import project_request_identity


class FakeTokenizer:
    bos_token_id = 1
    unk_token_id = 0
    pad_token_id = 2
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        body = [3 + (ord(character) % 20) for character in text]
        return ([self.bos_token_id] if add_special_tokens else []) + body


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float32), requires_grad=False)
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> types.SimpleNamespace:
        self.calls.append((input_ids.detach().clone(), attention_mask.detach().clone()))
        vocab = 32
        positions = torch.arange(vocab, device=input_ids.device).view(1, 1, -1)
        logits = -torch.abs(positions - input_ids.unsqueeze(-1)).float()
        return types.SimpleNamespace(logits=logits)


def _freeze() -> EndpointFreezeToken:
    return EndpointFreezeToken.create(
        arm=Arm.NATIVE,
        source_head="1" * 40,
        parameter_sha256="2" * 64,
        action_sha256="3" * 64,
        edit_count=4,
    )


class P1EvaluatorTests(unittest.TestCase):
    def test_scorer_right_padding_microbatch_suffix_and_cleanup(self) -> None:
        model = FakeModel()
        specs = (
            TeacherForcedSpec("a", "short", "x"),
            TeacherForcedSpec("b", "a longer prompt", "y"),
            TeacherForcedSpec("c", "mid", "z"),
        )
        result = TeacherForcedScorer(
            model, FakeTokenizer(), microbatch_size=2
        ).score(specs, gradient=False)
        self.assertEqual(result.model_forward_calls, 2)
        self.assertEqual(set(result.logp), {"a", "b", "c"})
        self.assertEqual(len(result.panel_id or ""), 64)
        first_ids, first_attention = model.calls[0]
        self.assertEqual(tuple(first_ids.shape[:1]), (2,))
        self.assertTrue(torch.equal(first_attention[:, 0], torch.ones(2, dtype=torch.long)))
        self.assertTrue(torch.all(first_attention.sum(dim=1) > 0))
        with self.assertRaises(ODEAllocContractError):
            TeacherForcedScorer(model, FakeTokenizer(), microbatch_size=2).score(
                (specs[0], specs[0]), gradient=False
            )

        class BoundaryChangingTokenizer(FakeTokenizer):
            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                result = super().encode(text, add_special_tokens=add_special_tokens)
                if add_special_tokens and text == "prompt target":
                    result[-1] += 1
                return result

        with self.assertRaises(ODEAllocContractError):
            TeacherForcedScorer(
                model, BoundaryChangingTokenizer(), microbatch_size=1
            ).score(
                (TeacherForcedSpec("boundary", "prompt", "target"),),
                gradient=False,
            )

    def test_scorer_gradient_path_retains_graph(self) -> None:
        class GradientModel(FakeModel):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.tensor(1.0, requires_grad=True)

            def forward(self, **kwargs: object) -> types.SimpleNamespace:
                output = super().forward(**kwargs)
                return types.SimpleNamespace(logits=output.logits * self.scale)

        model = GradientModel()
        result = TeacherForcedScorer(model, FakeTokenizer(), microbatch_size=1).score(
            (TeacherForcedSpec("a", "prompt", "target"),), gradient=True
        )
        gradient = torch.autograd.grad(result.logp["a"], model.scale)[0]
        self.assertTrue(torch.isfinite(gradient))
        self.assertIsNone(result.panel_id)

    def test_loader_requires_valid_endpoint_freeze_and_exact_request_hash(self) -> None:
        rows = []
        inner = []
        approved = []
        for case_id in range(4):
            row = {
                "case_id": case_id,
                "requested_rewrite": {
                    "prompt": "{} relation",
                    "relation_id": "R",
                    "subject": f"subject-{case_id}",
                    "target_new": {"str": f"new-{case_id}"},
                    "target_true": {"str": f"old-{case_id}"},
                },
                "paraphrase_prompts": [f"paraphrase-{case_id}"],
                "neighborhood_prompts": [f"neighbor-{case_id}"],
                "generation_prompts": ["unused"],
            }
            identity = project_request_identity(json.dumps(row).encode())
            rows.append(row)
            approved.append({"case_id": case_id, "request_hash": identity.request_hash})
            inner.append(
                {
                    "case_id": case_id,
                    "prompt": "{} relation",
                    "relation_id": "R",
                    "subject": f"subject-{case_id}",
                    "target_new": f"new-{case_id}",
                    "target_old": f"old-{case_id}",
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            cases = load_evaluation_cases_after_freeze(
                path,
                freeze=_freeze(),
                approved=approved,
                inner_requests=inner,
            )
            self.assertEqual(tuple(item.case_id for item in cases), (0, 1, 2, 3))
            bad = EndpointFreezeToken(
                Arm.NATIVE, "1" * 40, "2" * 64, "3" * 64, 4, "0" * 64
            )
            with self.assertRaises(ODEAllocContractError):
                load_evaluation_cases_after_freeze(
                    path, freeze=bad, approved=approved, inner_requests=inner
                )

    def test_endpoint_evaluator_is_teacher_forced_and_accounted_separately(self) -> None:
        cases = tuple(
            EvaluationCase(
                case_id=index,
                request_hash=str(index) * 64,
                prompt="{} relation",
                subject=f"subject-{index}",
                target_new=f"new-{index}",
                target_old=f"old-{index}",
                paraphrases=(f"para-{index}",),
                neighborhoods=(f"neighbor-{index}",),
            )
            for index in range(4)
        )
        anchors = tuple(
            {
                "prompt": "{} anchor",
                "subject": f"anchor-{index}",
                "target_old": "old",
            }
            for index in range(16)
        )
        result = evaluate_frozen_endpoint(
            FakeModel(),
            FakeTokenizer(),
            freeze=_freeze(),
            cases=cases,
            anchor_requests=anchors,
            theta0_anchor_teacher=(0.0,) * 16,
        )
        self.assertEqual(result["model_generate_calls"], 0)
        self.assertIs(result["action_feedback_to_controller"], False)
        self.assertEqual(len(result["case_metrics"]), 4)
        self.assertEqual(result["anchor_preservation"]["count"], 16)
        self.assertGreater(result["accounting"]["model_forward_calls"], 0)


if __name__ == "__main__":
    unittest.main()
