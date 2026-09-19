"""CPU synthetic numerical/coverage integration, not actual8B/GPU evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from project.run_scripts.single_layer_edit_preserving_correction.alltoken import WEIGHT, tensor_sha256
from .endpoint_session import EndpointSession, RuntimePolicy
from .generated_oracle import FiniteTrialModelOverflow, GeneratedReferenceOracle
from .generated_teacher import (
    CPUFixture, DATA_ID, SCHEMA, GeneratedTeacherError, GeneratedTeacherStore, canonical_sha256,
    file_sha256, make_capsule, token_sha256,
)
from .test_generated_teacher import binding, descriptor, write_json


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(6, 4, bias=False)
        nn.init.zeros_(self.mlp.down_proj.weight)

    def forward(self, hidden, **_kwargs):
        state = self.post_attention_layernorm(hidden)
        keys = torch.cat((state, state[..., :2]), dim=-1)
        return (hidden + self.mlp.down_proj(keys),)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(6)])
        self.norm = nn.Identity()
        self.calls = 0

    def _update_causal_mask(self, *args):
        return None

    def rotary_emb(self, hidden, position_ids):
        return (None, None)

    def forward(self, *, input_ids, attention_mask, position_ids, **_kwargs):
        self.calls += 1
        length = (1, 9, 17, 256, 300, 1)[int(input_ids[0, 1]) - 2]
        position = position_ids.float() / 512
        hidden = torch.stack((position, torch.ones_like(position),
                              torch.full_like(position, (128 + length - 1) / 512),
                              input_ids.float() / 32), dim=-1)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(32, dtype=torch.float32).reshape(8, 4) / 160)
        self.shapes = []

    def forward(self, hidden):
        self.shapes.append(tuple(hidden.shape))
        logits = F.linear(hidden, self.weight)
        bias = torch.zeros_like(logits)
        bias[..., 3] = 3
        bias[..., 0] = torch.where(hidden[..., 0] >= hidden[..., 2], 6.0, -5.0)
        return logits + bias


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.lm_head = Decoder(), Head()
        self.config = SimpleNamespace(model_type="llama", pretraining_tp=1,
                                      _attn_implementation="eager", vocab_size=8)
        self.eval().requires_grad_(False)


def make_store(root, model):
    identity, rows, members = binding(), [], []
    identity["w0_sha256"] = tensor_sha256(dict(model.named_parameters())[WEIGHT])
    counts = dict(R512=0, Dev128=0)
    for index in range(640):
        role, ordinal = ("R512", index) if index < 512 else ("Dev128", index - 512)
        prompt = [1] + [2 + index // 6**j % 6 for j in range(4)] + [2] * 124
        row = dict(role=role, ordinal=ordinal, source_row_id=f"oracle-fixture-{index}",
                   input_ids=prompt, prompt_token_sha256=token_sha256(prompt), window_token_sha256="6" * 64,
                   source_role="S64" if index < 64 else "Reserve320" if index < 384 else
                   "AdditionalTrain128" if index < 512 else "Dev128", source_text_sha256="7" * 64, window_start=0)
        rows.append(row)
        length = (1, 9, 17, 256, 300, 1)[index % 6]
        y0 = [3] * min(length, 256)
        if length <= 256:
            y0[-1] = 0
        cap = make_capsule(row, y0, identity)
        counts[role] += len(y0)
        folder = root / role / str(ordinal)
        folder.mkdir(parents=True)
        path = folder / "capsule.json"
        write_json(path, cap)
        member = dict(index=index, role=role, ordinal=ordinal, source_row_id=row["source_row_id"],
                      capsule=descriptor(root, path))
        keys, residual = [], []
        layer = model.model.layers[4]
        handles = [layer.mlp.down_proj.register_forward_pre_hook(lambda _m, args: keys.append(args[0])),
                   layer.post_attention_layernorm.register_forward_pre_hook(lambda _m, args: residual.append(args[0]))]
        try:
            with torch.no_grad():
                hidden = model.model(input_ids=torch.tensor([cap["tf_input_ids"]]),
                                     attention_mask=torch.tensor([cap["attention_mask"]]),
                                     position_ids=torch.tensor([cap["position_ids"]])).last_hidden_state
                logp = model.lm_head(hidden[0, cap["score_positions"]]).log_softmax(-1)
                assert logp.argmax(-1).tolist() == y0
        finally:
            for handle in handles:
                handle.remove()
        for kind, value in (("logp", logp), ("keys", keys[0][0]), ("residual", residual[0][0])):
            array = value.numpy().copy()
            path = folder / (kind + ".npy")
            np.save(path, array, allow_pickle=False)
            member[kind] = descriptor(root, path, array)
        members.append(member)
    inputs = root / "inputs.json"
    write_json(inputs, rows)
    manifest = dict(schema=SCHEMA, data_id=DATA_ID, status="COMPLETE", production_ready=False,
                    vocabulary_size=8, key_size=6, hidden_size=4, inputs_sha256=file_sha256(inputs),
                    binding=identity, binding_sha256=canonical_sha256(identity), documents=members,
                    document_counts=dict(R512=512, Dev128=128), position_counts=counts,
                    upstream_cache_status="COMPLETE")
    path = root / "manifest.json"
    write_json(path, manifest)
    return GeneratedTeacherStore(root, path, expected_manifest_sha256=file_sha256(path), inputs_path=inputs,
                                 expected_binding=identity, cpu_fixture=CPUFixture(8, 6, 4, file_sha256(inputs)),
                                 require_upstream_cache=True)


class GeneratedOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        cls.temporary = tempfile.TemporaryDirectory(prefix="en-oracle-cpu-")
        cls.root = Path(cls.temporary.name)
        cls.model = Model()
        cls.store = make_store(cls.root, cls.model)
        cls.oracle = GeneratedReferenceOracle(cls.model, cls.store)
        cls.W0 = dict(cls.model.named_parameters())[WEIGHT].detach().clone()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = cls.tf32

    def setUp(self):
        self.oracle.reset_counters()
        self.model.lm_head.shapes.clear()
        self.candidate = self.W0 + torch.arange(24, dtype=torch.float32).reshape(4, 6) / 2400

    def session(self, weight):
        policy = RuntimePolicy(shape=(4, 6), device="cpu", model_epoch=1,
                               input_identity="all640", source_identity="CPU_FIXTURE",
                               epoch_getter=lambda: 1, input_identity_getter=lambda: "all640",
                               source_identity_getter=lambda: "CPU_FIXTURE")
        session = EndpointSession("fixture-model", "cold-B1", policy)
        handle = session.bind_native(weight)
        self.addCleanup(session.close)
        return session, handle

    def test_preload_one_bank_no_prefix_model_calls_and_bounded_payload(self):
        self.assertEqual(len(self.oracle.caches), 640)
        self.assertEqual(self.oracle.setup_receipt["work"]["retained_prefix_documents_loaded"], 640)
        self.assertEqual(self.oracle.setup_receipt["prefix_forwards"], 0)
        self.assertLess(self.oracle.resident_cache_bytes, 18 * 2**30)
        with self.assertRaisesRegex(MemoryError, "CPU_PREFIX_CACHE_BUDGET"):
            GeneratedReferenceOracle(self.model, self.store, max_cache_bytes=1)
        self.assertEqual(self.oracle.caches[4].keys.shape, (1, 384, 6))

    def test_full512_gradient_legacy_and_resident_exact_independent(self):
        with patch.object(self.oracle, "_sync", wraps=self.oracle._sync) as sync:
            legacy = self.oracle.kl(self.candidate, gradient=True)
            legacy_sync_calls = sync.call_count
        legacy_receipt = self.oracle.last_sweep
        self.assertEqual(legacy_receipt["coverage"]["documents"], 512)
        self.assertEqual(legacy_receipt["coverage"]["backward_documents"], 512)
        self.assertEqual(legacy_receipt["work"]["autograd_calls"], 512)
        self.assertEqual(legacy_receipt["work"]["prefix_byte_checks"], 2)
        self.assertEqual(legacy_receipt["work"]["gradient_sweeps"], 1)
        self.assertEqual(self.oracle.work["weight_validation_calls"], 513)
        self.assertEqual(self.oracle.work["peak_live_document_graphs"], 1)
        cache_pointers = [c.keys.data_ptr() for c in self.oracle.caches]
        self.oracle.reset_counters()
        session, handle = self.session(self.candidate)
        with patch.object(self.oracle, "_sync", wraps=self.oracle._sync) as sync:
            resident = self.oracle.kl(self.candidate, gradient=True, session=session, handle=handle)
            resident_sync_calls = sync.call_count
        self.assertEqual(legacy_sync_calls, resident_sync_calls)
        self.assertEqual(legacy_sync_calls, 512 * 8 + 2)  # One final accumulator transfer boundary.
        self.assertEqual(legacy[0], resident[0])
        self.assertTrue(torch.equal(legacy[1], resident[1]))
        self.assertEqual(legacy[2], resident[2])
        self.assertEqual(resident[1].dtype, torch.float64)
        self.assertEqual(resident[1].device.type, "cpu")
        self.assertEqual(self.oracle.work["weight_validation_calls"], 0)
        self.assertEqual(self.oracle.work["cached_backward_documents"], 512)
        self.assertEqual(self.oracle.work["autograd_calls"], 512)
        self.assertEqual(self.oracle.work["peak_live_document_graphs"], 1)
        self.assertEqual(cache_pointers, [c.keys.data_ptr() for c in self.oracle.caches])
        self.assertEqual(session.work["gradient_leaf_clones"], 1)
        self.assertEqual(session.work["inference_transfer_calls"], 1)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.model.parameters()))

    def test_dev_observer_full128_no_gradient_and_full_T_heads(self):
        result = self.oracle.kl(self.candidate, role="Dev128")
        receipt = self.oracle.last_sweep
        self.assertIsNone(result[1])
        self.assertEqual(len(result[2]), 128)
        self.assertEqual(receipt["coverage"]["documents"], 128)
        self.assertEqual(receipt["coverage"]["backward_documents"], 0)
        self.assertEqual(self.oracle.work["head_calls"], 128)
        self.assertEqual(self.model.lm_head.shapes, [(r["scored_positions"], 4) for r in result[2]])
        session, handle = self.session(self.candidate)
        resident = self.oracle.kl(self.candidate, role="Dev128", session=session, handle=handle)
        self.assertEqual(result[0], resident[0])
        self.assertEqual(result[2], resident[2])
        self.assertEqual(session.work["gradient_leaf_clones"], 0)
        self.assertEqual(self.oracle.last_sweep["work"]["weight_validation_calls"], 0)
        for role, gradient in (("Dev128", True), ("Report256", False), ("S64", False)):
            with self.assertRaises(ValueError):
                self.oracle.kl(self.candidate, role=role, gradient=gradient)
        with self.assertRaises(TypeError):
            self.oracle.kl(self.candidate, indices=[0])

    def test_bounded_physical_AD_is_exact_and_not_full_sweep(self):
        saved = dict(self.model.named_parameters())[WEIGHT].detach().clone()
        result = self.oracle.check_documents(self.candidate, [0, 2, 4], gradient=True)
        self.assertFalse(result["full_bank_sweep"])
        self.assertFalse(result["full_bank_validation_pass"])
        self.assertTrue(result["scalar_exact"])
        self.assertTrue(result["rows_exact"])
        self.assertTrue(result["gradient_exact"])
        self.assertEqual(result["gradient_relative_l2"], 0)
        self.assertEqual(result["physical"]["receipt"]["documents"], 3)
        self.assertEqual(self.oracle.work["physical_exact_restores"], 1)
        self.assertEqual(self.oracle.work["gradient_sweeps"], 0)
        self.assertEqual(self.oracle.work["kl_sweeps"], 0)
        self.assertIsNone(self.oracle.last_sweep)
        self.assertTrue(torch.equal(saved, dict(self.model.named_parameters())[WEIGHT]))
        self.assertEqual(self.oracle.work["autograd_calls"], 6)

    def test_runtime_skip_preserves_full512_loss_gradient_without_payload_or_prefix_audits(self):
        expected = self.oracle.kl(self.candidate, gradient=True)
        with patch.object(self.store, 'verify_payloads', False), \
             patch.object(self.oracle, '_digest_cache', side_effect=AssertionError('prefix rehash')), \
             patch.object(self.store, '_check_file', side_effect=AssertionError('payload rehash')):
            self.oracle.reset_counters()
            actual = self.oracle.kl(self.candidate, gradient=True)
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[2], expected[2])
        self.assertTrue(torch.equal(actual[1], expected[1]))
        receipt = self.oracle.last_sweep
        self.assertEqual(receipt['coverage']['documents'], 512)
        self.assertEqual(receipt['coverage']['backward_documents'], 512)
        self.assertEqual(receipt['work']['prefix_byte_checks'], 0)
        self.assertEqual(receipt['runtime_validation'], 'SKIPPED_USER_DIRECTED')
        self.assertEqual(actual[1].device.type, 'cpu')

    def test_document_mean_uses_actual_lengths_not_length_weighting(self):
        result = self.oracle.check_documents(self.candidate, [0, 2, 4], gradient=False)
        rows = result["cached"]["rows"]
        self.assertEqual(result["cached"]["mean"], sum(r["loss"] for r in rows) / 3)
        weighted = sum(r["loss"] * r["scored_positions"] for r in rows) / sum(r["scored_positions"] for r in rows)
        self.assertNotEqual(result["cached"]["mean"], weighted)

    def test_model_guard_and_numpy_alias_cache_mutation_rejected(self):
        array = self.oracle.caches[0].keys.numpy()
        before = array.copy()
        try:
            array[0, 0, 0] += 1
            with self.assertRaisesRegex(RuntimeError, "CPU_PREFIX_CACHE_BYTES"):
                self.oracle.kl(self.W0)
            self.assertIsNone(self.oracle.last_sweep)
        finally:
            array[...] = before
        self.model.train()
        try:
            with self.assertRaisesRegex(RuntimeError, "FROZEN_MODEL"):
                self.oracle.kl(self.W0)
        finally:
            self.model.eval()

    def test_technical_failure_restores_physical_parameter_and_no_complete_receipt(self):
        saved = dict(self.model.named_parameters())[WEIGHT].detach().clone()
        original = self.oracle._teacher
        calls = 0
        def failure(index):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic teacher evidence failure during physical route")
            return original(index)
        with patch.object(self.oracle, "_teacher", side_effect=failure):
            with self.assertRaisesRegex(OSError, "teacher evidence"):
                self.oracle.check_documents(self.candidate, [0])
        self.assertTrue(torch.equal(saved, dict(self.model.named_parameters())[WEIGHT]))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.parameters()))
        self.assertIsNone(self.oracle.last_sweep)

    def test_mismatched_session_or_stale_weight_rejected(self):
        session, handle = self.session(self.candidate)
        # The real optimizer clones WN at entry; a new object with the same
        # bytes is valid while the resident native owner remains unchanged.
        session.validate_source(handle, self.candidate.clone())
        with self.assertRaisesRegex(ValueError, "SESSION_AND_HANDLE"):
            self.oracle.kl(self.candidate, session=session)
        with self.assertRaisesRegex(RuntimeError, "WEIGHT_HANDLE_BYTES"):
            self.oracle.kl(self.W0, session=session, handle=handle)
        for indices in ([], [0, 0], list(range(17)), [640], [True]):
            with self.assertRaises(ValueError):
                self.oracle.check_documents(self.W0, indices)
        with self.assertRaisesRegex(ValueError, "DEV_GRADIENT"):
            self.oracle.check_documents(self.W0, [512], gradient=True)

    def test_optin_trial_logits_overflow_preserves_partial_costs_and_session(self):
        session, native = self.session(self.W0)
        candidate = session.bind_candidate(self.candidate, dict(trial=0))
        original = self.oracle._head
        calls = 0
        def nonfinite_second(hidden, positions):
            nonlocal calls
            calls += 1
            result = original(hidden, positions)
            return result if calls == 1 else torch.full_like(result, torch.inf)
        with patch.object(self.oracle, "_head", side_effect=nonfinite_second):
            with self.assertRaises(FiniteTrialModelOverflow) as caught:
                self.oracle.kl(self.candidate, session=session, handle=candidate, allow_trial_overflow=True)
        receipt = self.oracle.last_sweep
        self.assertEqual(receipt, caught.exception.receipt)
        self.assertFalse(receipt["complete"])
        self.assertFalse(receipt["coverage"]["complete"])
        self.assertEqual(receipt["documents"], 1)
        self.assertEqual(receipt["attempted_documents"], 2)
        self.assertEqual(receipt["failed_document"]["index"], 1)
        self.assertEqual(receipt["failed_document"]["reason"], "MODEL_LOGITS_NONFINITE")
        self.assertEqual(len(receipt["partial_rows"]), 1)
        self.assertEqual(receipt["work"]["head_calls"], 2)
        self.assertEqual(receipt["work"]["cached_suffix_forwards"], 2)
        self.assertEqual(receipt["work"]["teacher_documents_read"], 2)
        self.assertEqual(receipt["work"]["autograd_calls"], 0)
        self.assertEqual(receipt["work"]["kl_sweeps"], 0)
        self.assertEqual(receipt["work"]["prefix_byte_checks"], 2)
        self.assertTrue(receipt["endpoint_borrow_exit_validated"])
        self.assertTrue(receipt["parent_full_teacher_recheck_required"])
        self.assertFalse(receipt["entire_fixed_teacher_bank_rechecked"])
        self.assertNotIn("mean", receipt)
        session.validate_handle(native, full=True)
        session.validate_handle(candidate, full=True)
        next_weight = self.W0 + (self.candidate - self.W0) / 2
        next_candidate = session.bind_candidate(next_weight, dict(trial=1))
        result = self.oracle.kl(next_weight, session=session, handle=next_candidate, allow_trial_overflow=True)
        self.assertEqual(len(result[2]), 512)
        self.assertTrue(self.oracle.last_sweep["complete"])
        self.assertEqual(self.oracle.work["kl_sweeps"], 1)

    def test_optin_nonfinite_model_loss_is_partial_legacy_trial_only(self):
        target = "project.run_scripts.en_execution_reuse.generated_oracle.signed_forward_kl"
        with patch(target, return_value=torch.tensor(float("inf"), dtype=torch.float64)):
            with self.assertRaises(FiniteTrialModelOverflow):
                self.oracle.kl(self.candidate, allow_trial_overflow=True)
        receipt = self.oracle.last_sweep
        self.assertEqual(receipt["documents"], 0)
        self.assertEqual(receipt["attempted_documents"], 1)
        self.assertEqual(receipt["failed_document"]["reason"], "MODEL_LOSS_NONFINITE")
        self.assertFalse(receipt["endpoint_borrow_exit_validated"])
        self.assertEqual(receipt["work"]["kl_sweeps"], 0)

    def test_default_base_accepted_gradient_and_dev_have_no_overflow_optin(self):
        original = self.oracle._head
        with patch.object(self.oracle, "_head", side_effect=lambda h, p: torch.full_like(original(h, p), torch.nan)):
            for gradient in (False, True):
                with self.subTest(gradient=gradient), self.assertRaises(FloatingPointError) as caught:
                    self.oracle.kl(self.candidate, gradient=gradient)
                self.assertNotIsInstance(caught.exception, FiniteTrialModelOverflow)
                self.assertIsNone(self.oracle.last_sweep)
        for kwargs in (dict(gradient=True), dict(role="Dev128")):
            with self.assertRaisesRegex(ValueError, "TRIAL_OVERFLOW_ONLY"):
                self.oracle.kl(self.candidate, allow_trial_overflow=True, **kwargs)
        with patch("torch.autograd.grad", return_value=(torch.full_like(self.candidate, torch.nan),)):
            with self.assertRaisesRegex(FloatingPointError, "NONFINITE_OR_NONFP32_REFERENCE_GRADIENT") as caught:
                self.oracle.kl(self.candidate, gradient=True)
            self.assertNotIsInstance(caught.exception, FiniteTrialModelOverflow)
            self.assertIsNone(self.oracle.last_sweep)

    def test_teacher_nonfinite_even_with_bad_logits_is_always_technical(self):
        session, native = self.session(self.W0)
        candidate = session.bind_candidate(self.candidate, dict(trial=0))
        original_head, original_teacher = self.oracle._head, self.oracle._teacher
        with patch.object(self.oracle, "_teacher", side_effect=lambda i: original_teacher(i).fill_(torch.nan)), \
                patch.object(self.oracle, "_head", side_effect=lambda h, p: original_head(h, p).fill_(torch.inf)):
            with self.assertRaisesRegex(FloatingPointError, "NONFINITE_FIXED_TEACHER") as caught:
                self.oracle.kl(self.candidate, session=session, handle=candidate, allow_trial_overflow=True)
        self.assertNotIsInstance(caught.exception, FiniteTrialModelOverflow)
        self.assertIsNone(self.oracle.last_sweep)
        self.assertEqual(self.oracle.work["finite_trial_model_overflows"], 0)

    def test_teacher_corruption_oom_and_arbitrary_arithmetic_errors_are_not_trial_overflow(self):
        target = "project.run_scripts.en_execution_reuse.generated_oracle.signed_forward_kl"
        cases = (("_teacher", GeneratedTeacherError("PAYLOAD_SHA")),
                 ("_head", torch.OutOfMemoryError("synthetic model OOM")))
        for name, error in cases:
            with self.subTest(name=name), patch.object(self.oracle, name, side_effect=error):
                with self.assertRaises(type(error)) as caught:
                    self.oracle.kl(self.candidate, allow_trial_overflow=True)
                self.assertIs(caught.exception, error)
                self.assertIsNone(self.oracle.last_sweep)
        with patch(target, side_effect=FloatingPointError("UNRELATED_ARITHMETIC_ERROR")):
            with self.assertRaisesRegex(FloatingPointError, "UNRELATED_ARITHMETIC_ERROR") as caught:
                self.oracle.kl(self.candidate, allow_trial_overflow=True)
            self.assertNotIsInstance(caught.exception, FiniteTrialModelOverflow)
        self.assertEqual(self.oracle.work["finite_trial_model_overflows"], 0)

    def test_endpoint_or_prefix_mutation_overrides_captured_trial_overflow(self):
        session, native = self.session(self.W0)
        candidate_weight = self.candidate.clone()
        candidate = session.bind_candidate(candidate_weight, dict(trial=0))
        original = self.oracle._head
        def corrupt_endpoint(hidden, positions):
            result = original(hidden, positions)
            candidate_weight.numpy()[0, 0] += 1
            return result.fill_(torch.inf)
        with patch.object(self.oracle, "_head", side_effect=corrupt_endpoint):
            with self.assertRaises(RuntimeError) as caught:
                self.oracle.kl(candidate_weight, session=session, handle=candidate, allow_trial_overflow=True)
        self.assertNotIsInstance(caught.exception, FiniteTrialModelOverflow)
        self.assertIsNone(self.oracle.last_sweep)
        array = self.oracle.caches[0].keys.numpy()
        saved = array.copy()
        def corrupt_prefix(hidden, positions):
            result = original(hidden, positions)
            array[0, 0, 0] += 1
            return result.fill_(torch.inf)
        try:
            with patch.object(self.oracle, "_head", side_effect=corrupt_prefix):
                with self.assertRaisesRegex(RuntimeError, "CPU_PREFIX_CACHE_BYTES") as caught:
                    self.oracle.kl(self.candidate, allow_trial_overflow=True)
            self.assertNotIsInstance(caught.exception, FiniteTrialModelOverflow)
            self.assertIsNone(self.oracle.last_sweep)
        finally:
            array[...] = saved


if __name__ == "__main__":
    unittest.main()
