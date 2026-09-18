"""CPU fake-model checks of preparation calls, shift, atomicity and failures."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from project.run_scripts.single_layer_edit_preserving_correction.alltoken import tensor_sha256
from .generated_reference import (
    GeneratedReferenceBuilder, ReferencePreparationError, atomic_json,
)
from .generated_teacher import CPUFixture, GeneratedTeacherStore, file_sha256, token_sha256
from .test_generated_teacher import binding


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(6, 4, bias=False)
        nn.init.zeros_(self.mlp.down_proj.weight)

    def forward(self, hidden):
        residual = hidden
        state = self.post_attention_layernorm(hidden)
        keys = torch.cat((state, state[..., :2]), dim=-1)
        return residual + self.mlp.down_proj(keys)


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeBlock() for _ in range(6)])
        self.calls = []
        self.raise_on_tf = False

    def forward(self, *, input_ids, attention_mask, position_ids, use_cache, return_dict):
        if use_cache or not return_dict or not torch.equal(attention_mask, torch.ones_like(input_ids)):
            raise RuntimeError("WRONG_FAKE_DECODER_PROTOCOL")
        if not torch.equal(position_ids, torch.arange(input_ids.shape[1]).unsqueeze(0)):
            raise RuntimeError("WRONG_POSITION_SHIFT")
        self.calls.append(input_ids.shape[1])
        hidden = torch.stack((position_ids.float(), input_ids.float(), torch.ones_like(input_ids).float(),
                              torch.zeros_like(input_ids).float()), dim=-1)
        for layer in self.layers:
            hidden = layer(hidden)
        if self.raise_on_tf and len(self.calls) > 3:
            raise RuntimeError("INJECTED_DECODER_FAILURE")
        return SimpleNamespace(last_hidden_state=hidden)


class FakeHead(nn.Module):
    def __init__(self, stop):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(8, 4))
        self.stop = stop
        self.mismatch = self.nonfinite = False

    def forward(self, hidden):
        logits = torch.full((*hidden.shape[:-1], 8), -4.0)
        logits[..., 3] = 0
        logits[..., 4] = 0  # exact maximum tie must select token 3
        ended = hidden[..., 0] >= 128 + self.stop - 1
        logits[..., 0] = torch.where(ended, 1.0, -4.0)
        if self.mismatch and hidden.ndim == 2:
            logits[0, 6] = 2
        if self.nonfinite:
            logits[..., 7] = torch.nan
        return logits


class FakeModel(nn.Module):
    def __init__(self, stop=3):
        super().__init__()
        self.model = FakeDecoder()
        self.lm_head = FakeHead(stop)
        self.generation_config = SimpleNamespace(eos_token_id=0)
        self.config = SimpleNamespace(eos_token_id=7)
        self.eval().requires_grad_(False)


def inputs(root):
    rows = []
    for index in range(640):
        prompt = [1] + [2 + index // 6**j % 6 for j in range(4)] + [2] * 124
        row = dict(role="R512" if index < 512 else "Dev128", ordinal=index if index < 512 else index - 512,
                   source_row_id=f"fixture-{index}", input_ids=prompt, prompt_token_sha256=token_sha256(prompt),
                   window_token_sha256="6" * 64, source_role="S64" if index < 64 else "Reserve320" if index < 384 else
                   "AdditionalTrain128" if index < 512 else "Dev128", source_text_sha256="7" * 64, window_start=0)
        rows.append(row)
    path = atomic_json(root / "inputs.json", rows)
    return path


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="en-reference-adapter-fixture-")
        self.root = Path(self.temporary.name)
        self.inputs = inputs(self.root)
        self.fixture = CPUFixture(8, 6, 4, file_sha256(self.inputs))
        self.tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.tf32
        self.temporary.cleanup()

    def builder(self, stop=3, **kwargs):
        model = FakeModel(stop)
        identity = binding()
        identity["w0_sha256"] = tensor_sha256(model.model.layers[4].mlp.down_proj.weight)
        builder = GeneratedReferenceBuilder(model, identity, self.root / "teacher", self.inputs,
                                            cpu_fixture=self.fixture, **kwargs)
        return model, builder

    def test_one_token_actual_eos_and_joint_capture(self):
        model, builder = self.builder(stop=1)
        member, capsule, work = builder.build_document(0)
        self.assertEqual(capsule["y0"], [0])
        self.assertEqual(capsule["score_positions"], [128])
        self.assertEqual(capsule["stop_reason"], "eos")
        self.assertFalse(capsule["length_censored"])
        self.assertEqual(model.model.calls, [129, 129])
        self.assertEqual(work["generation_input_tokens"], 129)
        self.assertEqual(work["canonical_TF_decoder_forwards"], 1)
        self.assertEqual(work["key_capture_forwards"], 0)
        self.assertEqual(np.load(builder.root / member["keys"]["path"]).shape, (129, 6))
        self.assertFalse((builder.root / "manifest.json").exists())
        self.assertTrue((builder.root / "R512/000/completion.json").exists())

    def test_middle_eos_shift_tie_and_positions(self):
        model, builder = self.builder(stop=17)
        _, capsule, work = builder.build_document(0)
        self.assertEqual(capsule["y0"], [3] * 16 + [0])
        self.assertEqual(len(capsule["tf_input_ids"]), 145)
        self.assertEqual(capsule["score_positions"], list(range(128, 145)))
        self.assertEqual(model.model.calls, list(range(129, 146)) + [145])
        self.assertEqual(work["generation_input_tokens"], sum(range(129, 146)))
        self.assertEqual(work["canonical_TF_head_rows"], 17)

    def test_256_length_cap_has_no_fake_eos(self):
        model, builder = self.builder(stop=300)
        _, capsule, work = builder.build_document(0)
        self.assertEqual(capsule["y0"], [3] * 256)
        self.assertTrue(capsule["length_censored"])
        self.assertEqual(capsule["stop_reason"], "max_new_tokens")
        self.assertEqual(len(capsule["input_ids"] + capsule["y0"]), 385)
        self.assertEqual(len(capsule["tf_input_ids"]), 384)
        self.assertEqual(capsule["score_positions"][-1], 383)
        self.assertEqual(work["generation_input_tokens"], 65664)
        self.assertEqual(model.model.calls[-1], 384)

    def test_actual_eos_at256_is_not_censored(self):
        _, builder = self.builder(stop=256)
        _, capsule, _ = builder.build_document(0)
        self.assertEqual(capsule["y0"][-1], 0)
        self.assertEqual(capsule["actual_length"], 256)
        self.assertFalse(capsule["length_censored"])

    def test_tf_mismatch_keeps_failure_not_ready(self):
        model, builder = self.builder(stop=3)
        model.lm_head.mismatch = True
        with self.assertRaisesRegex(ReferencePreparationError, "TF_ARGMAX_MISMATCH"):
            builder.build_document(0)
        folder = builder.root / "R512/000"
        parity = json.loads((folder / "generation-TF-parity.json").read_text())
        self.assertEqual(parity["mismatch_positions"], [128])
        self.assertTrue((folder / "logp.npy").exists())
        self.assertTrue((folder / "failure.json").exists())
        self.assertFalse((folder / "capsule.json").exists())
        self.assertFalse((folder / "completion.json").exists())
        with self.assertRaisesRegex(ReferencePreparationError, "PARTIAL_DOCUMENT"):
            builder.build_document(0)

    def test_nonfinite_is_technical_not_fallback(self):
        model, builder = self.builder()
        model.lm_head.nonfinite = True
        with self.assertRaisesRegex(ReferencePreparationError, "NONFINITE_GENERATION"):
            builder.build_document(0)
        failure = json.loads((builder.root / "R512/000/failure.json").read_text())
        self.assertFalse(failure["fallback"])
        self.assertFalse(failure["complete"])

    def test_failure_removes_only_own_hooks(self):
        model, builder = self.builder(stop=3)
        model.model.raise_on_tf = True
        with self.assertRaisesRegex(RuntimeError, "INJECTED_DECODER_FAILURE"):
            builder.build_document(0)
        layer = model.model.layers[4]
        self.assertEqual(len(layer.mlp.down_proj._forward_pre_hooks), 0)
        self.assertEqual(len(layer.post_attention_layernorm._forward_pre_hooks), 0)
        builder._guard()

    def test_mutated_model_mode_and_weight_rejected(self):
        model, builder = self.builder()
        model.train()
        with self.assertRaisesRegex(ReferencePreparationError, "MODEL_STATE"):
            builder.generate(builder.inputs[0]["input_ids"])
        model.eval()
        with torch.no_grad():
            model.model.layers[4].mlp.down_proj.weight.add_(1)
        with self.assertRaisesRegex(ReferencePreparationError, "MODEL_STATE"):
            builder.generate(builder.inputs[0]["input_ids"])

    def test_atomic_no_clobber_and_io_failure_preserved(self):
        model, builder = self.builder(stop=1)
        builder.build_document(0)
        before = (builder.root / "R512/000/completion.json").read_bytes()
        with self.assertRaisesRegex(ReferencePreparationError, "REUSE_MUST_BE_EXPLICIT"):
            builder.build_document(0)
        self.assertEqual((builder.root / "R512/000/completion.json").read_bytes(), before)
        with self.assertRaises(FileExistsError):
            atomic_json(builder.root / "preparation.json", {})
        with patch("project.run_scripts.en_execution_reuse.generated_reference.atomic_array",
                   side_effect=OSError(28, "synthetic ENOSPC")):
            with self.assertRaises(OSError):
                builder.build_document(1)
        self.assertFalse((builder.root / "R512/001/completion.json").exists())

    def test_complete_document_reuse_is_exact_and_no_model_call(self):
        model, builder = self.builder(stop=1, reuse_completed=True)
        first = builder.build_document(0)
        calls = len(model.model.calls)
        second = builder.build_document(0)
        self.assertEqual(first, second)
        self.assertEqual(len(model.model.calls), calls)
        path = builder.root / first[0]["logp"]["path"]
        with path.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(ReferencePreparationError, "PAYLOAD_IDENTITY"):
            builder.build_document(0)

    def test_full640_store_is_compatible_and_dev_separate(self):
        model, builder = self.builder(stop=1)
        result = builder.build()
        self.assertEqual(result["document_counts"], dict(R512=512, Dev128=128))
        self.assertEqual(result["position_counts"], dict(R512=512, Dev128=128))
        self.assertFalse(result["production_ready"])
        self.assertEqual(len(model.model.calls), 1280)
        store = GeneratedTeacherStore(builder.root, result["path"], expected_manifest_sha256=result["sha256"],
                                      inputs_path=self.inputs, expected_binding=builder.binding,
                                      cpu_fixture=self.fixture, require_upstream_cache=True)
        with store.document(639) as doc:
            self.assertEqual(doc.capsule["role"], "Dev128")
            self.assertEqual(doc.capsule["ordinal"], 127)
            self.assertEqual(doc.logp.shape, (1, 8))
        with self.assertRaisesRegex(ReferencePreparationError, "COMPLETE_STORE_ALREADY_EXISTS"):
            builder.build()

    def test_runtime_eos_binding_rejects_mismatch(self):
        model = FakeModel(stop=1)
        identity = binding()
        identity["w0_sha256"] = tensor_sha256(model.model.layers[4].mlp.down_proj.weight)
        identity["generation"]["eos_token_ids"] = [7]
        with self.assertRaisesRegex(ReferencePreparationError, "NO_KV_GENERATION_POLICY"):
            GeneratedReferenceBuilder(model, identity, self.root / "teacher", self.inputs, cpu_fixture=self.fixture)


if __name__ == "__main__":
    unittest.main()
