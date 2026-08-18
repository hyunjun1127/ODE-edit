from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r52_accepted_z_observation import (
    AcceptedZBinding,
    AbsoluteAcceptedZActivationOverlay,
    OfficialNativeZCapture,
    _lookup_geometry,
)
from project.run_scripts.ode_bf.p1_evaluator import CounterFactEvaluationCase
from project.run_scripts.ode_bf.p1r52_accepted_z_observation_panel import (
    ROLES,
    expected_result_name,
)


class AcceptedZObservationTests(unittest.TestCase):
    def _binding(self) -> AcceptedZBinding:
        return AcceptedZBinding(
            role=ROLES[0],
            source="native-test",
            layer=8,
            module_name="layer",
            fact_token="subject_last",
            request_order_sha256=canonical_hash(list(range(100))),
            accepted_z=torch.arange(300, dtype=torch.float32).reshape(3, 100),
        )

    def test_absolute_overlay_preserves_locality_and_model_state(self) -> None:
        class Toy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = torch.nn.Identity()
                self.marker = torch.nn.Parameter(torch.tensor([7.0]))

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.layer(value)

        model = Toy()
        before = model.marker.detach().clone()
        positions = tuple((1, 1, 2, 2, 0, 0) for _ in range(100))
        prefix_counts = (4,) * 100
        with AbsoluteAcceptedZActivationOverlay(
            model, self._binding(), positions, prefix_counts
        ) as overlay:
            for ordinal in range(100):
                value = torch.zeros((6, 4, 3), dtype=torch.float32)
                observed = model(value)
                selected = self._binding().accepted_z[:, ordinal]
                self.assertTrue(torch.equal(observed[0, 1], selected))
                self.assertTrue(torch.equal(observed[3, 2], selected))
                self.assertEqual(int(torch.count_nonzero(observed[4:])), 0)
        receipt = overlay.raw_free_payload()
        self.assertEqual(receipt["hook_call_count"], 100)
        self.assertEqual(receipt["patched_row_count"], 400)
        self.assertEqual(receipt["locality_unpatched_row_count"], 200)
        self.assertEqual(receipt["added_backward_count"], 0)
        self.assertEqual(receipt["added_generation_call_count"], 0)
        self.assertTrue(torch.equal(model.marker, before))

    def test_binding_is_raw_free_and_proxy_free(self) -> None:
        receipt = self._binding().raw_free_payload()
        self.assertEqual(receipt["accepted_z_shape"], [3, 100])
        self.assertEqual(receipt["proxy_or_imputation_count"], 0)
        self.assertNotIn("accepted_z", receipt)

    def test_official_capture_binds_exact_return_order(self) -> None:
        from easyeditor.models.memit import memit_main

        requests = tuple(
            {
                "request_sha256": canonical_hash({"request": index}),
                "subject": f"subject-{index}",
            }
            for index in range(100)
        )
        hparams = SimpleNamespace(
            layers=[4, 5, 6, 7, 8],
            layer_module_tmp="layer{}",
            fact_token="subject_last",
        )

        def fake_compute_z(_model, _tokenizer, request, _hparams):
            return torch.full((3,), float(requests.index(request)))

        with patch.object(memit_main, "compute_z", fake_compute_z):
            capture = OfficialNativeZCapture(
                role="official-memit-sequential",
                requests=requests,
                hparams=hparams,
            )
            with capture:
                for request in requests:
                    memit_main.compute_z(None, None, request, hparams)
            binding = capture.finalize()
        self.assertEqual(list(binding.accepted_z.shape), [3, 100])
        self.assertTrue(torch.equal(binding.accepted_z[:, 99], torch.full((3,), 99.0)))

    def test_lookup_geometry_counts_rewrite_rephrase_and_locality(self) -> None:
        class FakeBatch(dict):
            pass

        class FakeTokenizer:
            padding_side = "left"

            def __call__(self, texts, padding=False, return_tensors=None):
                values = [texts] if isinstance(texts, str) else list(texts)
                lengths = [max(2, len(value.split()) + 1) for value in values]
                if return_tensors == "pt":
                    maximum = max(lengths)
                    attention = torch.zeros((len(values), maximum), dtype=torch.long)
                    for row, length in enumerate(lengths):
                        attention[row, maximum - length :] = 1
                    return FakeBatch(
                        input_ids=torch.ones_like(attention), attention_mask=attention
                    )
                rows = [list(range(length)) for length in lengths]
                return {"input_ids": rows[0] if isinstance(texts, str) else rows}

            def encode(self, text):
                return list(range(max(2, len(text.split()) + 1)))

        requests = tuple(
            {
                "request_sha256": canonical_hash({"request": index}),
                "subject": f"Subject{index}",
            }
            for index in range(100)
        )
        cases = tuple(
            CounterFactEvaluationCase(
                index,
                request["request_sha256"],
                f"{request['subject']} rewrite",
                (
                    f"{request['subject']} paraphrase one",
                    f"{request['subject']} paraphrase two",
                ),
                (f"neighborhood {index}",),
                "new target",
                "true target",
            )
            for index, request in enumerate(requests)
        )
        positions, prefix_counts, receipt = _lookup_geometry(
            FakeTokenizer(), requests, cases, fact_token="subject_last"
        )
        self.assertEqual(len(positions), 100)
        self.assertEqual(prefix_counts, (6,) * 100)
        self.assertEqual(receipt["patched_row_count"], 600)
        self.assertEqual(receipt["locality_unpatched_row_count"], 200)

    def test_literal_unrelated_brace_is_preserved_by_lookup_kernel(self) -> None:
        class LiteralTokenizer:
            padding_side = "left"

            def encode(self, text):
                return list(range(max(2, len(text.split()) + 1)))

            def __call__(self, texts, padding=False, return_tensors=None):
                if isinstance(texts, str):
                    return {"input_ids": self.encode(texts)}
                lengths = [len(self.encode(text)) for text in texts]
                maximum = max(lengths)
                attention = torch.zeros((len(texts), maximum), dtype=torch.long)
                for row, length in enumerate(lengths):
                    attention[row, maximum - length :] = 1
                return {
                    "input_ids": torch.ones_like(attention),
                    "attention_mask": attention,
                }

        tokenizer = LiteralTokenizer()
        requests = tuple(
            {
                "request_sha256": canonical_hash({"literal": index}),
                "subject": f"Subject{index}",
            }
            for index in range(100)
        )
        cases = tuple(
            CounterFactEvaluationCase(
                index,
                request["request_sha256"],
                f"{request['subject']} rewrite",
                (f"{request['subject']} literal {{Vinay/he}}",),
                (f"neighbor {index}",),
                "new",
                "true",
            )
            for index, request in enumerate(requests)
        )
        _, prefix_counts, receipt = _lookup_geometry(
            tokenizer, requests, cases, fact_token="subject_last"
        )
        self.assertEqual(prefix_counts, (4,) * 100)
        self.assertIn("repr_tools", receipt["lookup_kernel"])

    def test_four_cell_names_are_create_once_distinct(self) -> None:
        names = [expected_result_name(role) for role in ROLES]
        self.assertEqual(len(set(names)), 4)
        self.assertTrue(all("accepted-z-rephrase-obs" in name for name in names))

    def test_runtime_disables_batch_entry_metrics_and_binds_actual_delta(self) -> None:
        from project.run_scripts.ode_bf import p1_runtime, p1r52_sequential_runtime

        runtime = Path(p1_runtime.__file__).read_text(encoding="utf-8")
        sequential = Path(p1r52_sequential_runtime.__file__).read_text(encoding="utf-8")
        self.assertIn('"accepted-z-rephrase-obs",', runtime)
        self.assertIn("accepted_z_reference_root", sequential)
        self.assertIn("sealed_commit_weight_exact", sequential)
        self.assertIn("accepted-z observation mutated model state", sequential)


if __name__ == "__main__":
    unittest.main()
