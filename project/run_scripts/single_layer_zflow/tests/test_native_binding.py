"""CPU construction/reduction fixtures, not actual Llama parity evidence."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch

from project.run_scripts.single_layer_zflow.native_binding import (
    BoundSequence, aggregate_native_keys, build_training_sequences,
    capture_native_keys, native_subject_last_index, normalize_requests,
    pack_sequences, request_incidence, select_l4_projector,
    validate_native_config,
)


class CharTokenizer:
    bos_token_id, unk_token_id, pad_token_id = 1, 2, 0
    padding_side = "right"

    def encode(self, text):
        return [self.bos_token_id] + [ord(char) + 3 for char in text]

    def __call__(self, text):
        return {"input_ids": self.encode(text)}

    def decode(self, ids):
        return "".join(chr(i - 3) for i in ids if i > 2)


CONTEXTS = [["{}"], [f"Prefix{n}. {{}}" for n in range(5)]]
REQUESTS = [
    {"case_id": 9, "prompt": "{} resides in", "subject": "Alice",
     "target_new": {"str": " Rome", "id": "Q1"}},
    {"case_id": 17, "prompt": "{} was founded at", "subject": "Acme",
     "target_new": {"str": "Paris France", "id": "Q2"}},
]


class TestNativeBinding(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()
        self.batch = build_training_sequences(self.tok, REQUESTS, CONTEXTS)

    def test_normalization_is_deep_copy_and_space_once(self):
        original = deepcopy(REQUESTS)
        got = normalize_requests(REQUESTS)
        self.assertEqual(REQUESTS, original)
        self.assertEqual(got[0]["target_new"]["str"], " Rome")
        self.assertEqual(got[1]["target_new"]["str"], " Paris France")
        got[0]["target_new"]["str"] = "mutated"
        self.assertEqual(REQUESTS, original)

    def test_empty_target_and_duplicate_case_fail_closed(self):
        bad = deepcopy(REQUESTS)
        bad[0]["target_new"]["str"] = ""
        with self.assertRaises(ValueError):
            normalize_requests(bad)
        with self.assertRaises(ValueError):
            normalize_requests([REQUESTS[0], REQUESTS[0]])

    def test_exact_native_decode_join_and_prediction_shift(self):
        for row in self.batch.sequences:
            if row.kind != "edit":
                continue
            request = self.batch.requests[row.request_index]
            target = self.tok(request["target_new"]["str"])["input_ids"]
            if target[0] in (self.tok.bos_token_id, self.tok.unk_token_id):
                target = target[1:]
            contexts = [item for group in CONTEXTS for item in group]
            prompt = contexts[row.context_index].format(request["prompt"]) + self.tok.decode(target[:-1])
            ids = self.tok(prompt.format(request["subject"]))["input_ids"]
            native_rewriting_targets = [-100] * len(ids)
            native_rewriting_targets[len(ids) - len(target):len(ids)] = target
            positions = [i for i, label in enumerate(native_rewriting_targets) if label != -100]
            self.assertEqual(row.ids, tuple(ids))
            self.assertEqual(row.edit_positions, tuple(positions))
            self.assertEqual(row.edit_labels, tuple(target))
            self.assertEqual(row.edit_positions[-1], len(ids) - 1)

    def test_native_subject_last_and_essence_lookup(self):
        self.assertEqual(native_subject_last_index(self.tok, "Prefix {} suffix", "Alice"),
                         len(self.tok.encode("Prefix Alice")) - 1)
        essences = [s for s in self.batch.sequences if s.kind == "essence"]
        self.assertEqual(len(essences), 2)
        for row, request in zip(essences, self.batch.requests):
            self.assertEqual(row.ids, tuple(self.tok.encode(request["subject"] + " is a")))
            self.assertEqual(row.kl_pos, len(self.tok.encode(request["subject"])) - 1)
            self.assertEqual(row.kl_weight, .5)
            self.assertEqual(row.edit_labels, ())

    def test_logical_weights_request_context_token_not_microbatch(self):
        self.assertAlmostEqual(sum(sum(s.edit_weights) for s in self.batch.sequences), 1.)
        self.assertAlmostEqual(sum(s.kl_weight for s in self.batch.sequences), 1.)
        for request_index in range(2):
            own = [s for s in self.batch.sequences if s.request_index == request_index and s.kind == "edit"]
            self.assertEqual(len(own), 6)
            for row in own:
                self.assertAlmostEqual(sum(row.edit_weights), 1 / 12)
        whole = pack_sequences(self.batch.sequences, 0)
        parts = [pack_sequences(self.batch.sequences[i:i + 3], 0) for i in range(0, 14, 3)]
        self.assertEqual(whole["edit_weights"].dtype, torch.float64)
        self.assertAlmostEqual(sum(float(p["edit_weights"].sum()) for p in parts), 1.)
        self.assertAlmostEqual(sum(float(p["kl_weights"].sum()) for p in parts), 1.)

    def test_right_padding_native_arange_and_selected_metadata(self):
        packed = pack_sequences(self.batch.sequences[:7], 0)
        width = packed["input_ids"].shape[1]
        for i, row in enumerate(self.batch.sequences[:7]):
            self.assertEqual(packed["input_ids"][i, :len(row.ids)].tolist(), list(row.ids))
            self.assertTrue(torch.all(packed["input_ids"][i, len(row.ids):] == 0))
            self.assertEqual(int(packed["attention_mask"][i].sum()), len(row.ids))
            self.assertTrue(torch.equal(packed["position_ids"][i], torch.arange(width)))
        self.assertEqual(packed["kl_rows"].tolist(), [6])
        self.assertEqual(packed["kl_cols"].tolist(), [self.batch.sequences[6].kl_pos])
        self.assertTrue(torch.all(packed["attention_mask"][packed["edit_rows"], packed["edit_cols"]] == 1))

    def test_context_group_mean_differs_from_six_context_mean(self):
        raw = torch.tensor([[10., 20.]] + [[0., 0.]] * 5 + [[20., 40.]] + [[2., 4.]] * 5)
        keys = aggregate_native_keys(raw)
        self.assertTrue(torch.equal(keys, torch.tensor([[5., 11.], [10., 22.]])))
        self.assertFalse(torch.equal(keys[:, 0], raw[:6].mean(0)))

    def test_source_group_reduction_order_matches_reference(self):
        gen = torch.Generator().manual_seed(1209)
        raw = torch.randn(18, 13, generator=gen)
        expected = []
        for offset in range(0, 18, 6):
            expected.append(torch.stack([raw[offset:offset + 1].mean(0),
                                         raw[offset + 1:offset + 6].mean(0)], 0).mean(0))
        self.assertTrue(torch.equal(aggregate_native_keys(raw), torch.stack(expected).T))

    def test_incidence_identity_keeps_request_and_context_order(self):
        self.assertTrue(torch.equal(request_incidence(self.batch), torch.eye(2, dtype=torch.float64)))
        self.assertEqual([s.request_index for s in self.batch.key_sequences], [0] * 6 + [1] * 6)
        self.assertEqual([s.context_index for s in self.batch.key_sequences], list(range(6)) * 2)
        self.assertEqual(self.batch.metadata["request_order"], [9, 17])
        self.assertFalse(self.batch.metadata["key_and_training_cache_shared"])

    def test_native_context_padding_and_positions_fail_closed(self):
        self.tok.padding_side = "left"
        with self.assertRaises(ValueError):
            build_training_sequences(self.tok, REQUESTS, CONTEXTS)
        self.tok.padding_side = "right"
        with self.assertRaises(ValueError):
            build_training_sequences(self.tok, REQUESTS, [["{}"], ["one {}"]])
        row = BoundSequence((1, 2), 0, 1, "edit", 0, 1, (2,), (5,), (1.,))
        with self.assertRaises(ValueError):
            pack_sequences([row], 0)

    def test_projector_binding_is_explicit_zero_index_view(self):
        stack = torch.arange(45.).reshape(5, 3, 3)
        selected, meta = select_l4_projector(stack)
        self.assertEqual(selected.data_ptr(), stack.data_ptr())
        self.assertTrue(torch.equal(selected, stack[0]))
        self.assertEqual(meta["physical_layer"], 4)
        self.assertEqual(meta["source_stack_index"], 0)
        with self.assertRaises(ValueError):
            select_l4_projector(stack, (5, 6, 7, 8, 9))

    def test_native_config_singleton_without_mutation(self):
        config = {"layers": [4], "fact_token": "subject_last", "L2": 1, "blue": True,
                  "rewrite_module_tmp": "model.layers.{}.mlp.down_proj", "layer_module_tmp": "model.layers.{}"}
        before = deepcopy(config)
        result = validate_native_config(config)
        self.assertEqual(config, before)
        self.assertFalse(result["native_target_optimizer_called"])
        config["layers"] = [8]
        with self.assertRaises(ValueError):
            validate_native_config(config)

    def test_partial_and_nonfinite_context_keys_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_native_keys(torch.zeros(5, 3))
        with self.assertRaises(ValueError):
            aggregate_native_keys(torch.full((6, 3), float("nan")))

    def test_noop_incidence_empty(self):
        batch = build_training_sequences(self.tok, [], CONTEXTS)
        self.assertEqual(request_incidence(batch).shape, (0, 0))
        self.assertEqual(aggregate_native_keys(torch.empty(0, 3)).shape, (3, 0))

    def test_key_capture_cpu_fixture_is_read_only_and_keeps_columns(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Module()
                self.mlp.down_proj = torch.nn.Linear(3, 3, bias=False)
                with torch.no_grad():
                    self.mlp.down_proj.weight.copy_(torch.eye(3))

            def forward(self, hidden):
                return self.mlp.down_proj(hidden)

        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(256, 3)
                with torch.no_grad():
                    self.embed_tokens.weight.copy_(torch.arange(768.).reshape(256, 3))
                self.layers = torch.nn.ModuleList([Block() for _ in range(5)])

            def forward(self, input_ids, attention_mask, position_ids, use_cache, return_dict):
                self.asserted = (not use_cache) and return_dict
                hidden = self.embed_tokens(input_ids)
                for layer in self.layers:
                    hidden = layer(hidden)
                return SimpleNamespace(last_hidden_state=hidden)

        model = torch.nn.Module()
        model.model = Base()
        model.requires_grad_(False).eval()
        original = {name: value.clone() for name, value in model.state_dict().items()}
        keys, receipt = capture_native_keys(model, self.batch, pad_token_id=0, microbatch=2)
        context_keys = torch.stack([model.model.embed_tokens.weight[row.ids[row.lookup_index]]
                                    for row in self.batch.key_sequences])
        expected = aggregate_native_keys(context_keys)
        self.assertTrue(torch.equal(keys, expected))
        self.assertTrue(model.model.asserted)
        self.assertEqual(receipt["native_z_calls"], 0)
        self.assertEqual(receipt["history_append"], 0)
        self.assertEqual(receipt["key_shape"], [3, 2])
        self.assertTrue(all(torch.equal(value, original[name])
                            for name, value in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        model.train()
        with self.assertRaises(ValueError):
            capture_native_keys(model, self.batch, pad_token_id=0)


if __name__ == "__main__":
    unittest.main()
