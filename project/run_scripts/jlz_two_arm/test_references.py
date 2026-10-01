"""Bounded CPU selection/loss tests, not actual tokenizer/model certification."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from . import references as ref


def record(case_id, subject=None, relation="P1", target="new"):
    subject = f"subject-{case_id}" if subject is None else subject
    return {"case_id": case_id, "requested_rewrite": {
        "subject": subject, "relation_id": relation, "prompt": "{} is located in",
        "target_true": {"str": "old"}, "target_new": {"str": target}},
        "paraphrase_prompts": [f"paraphrase one {case_id}", f"paraphrase two {case_id}"],
        "neighborhood_prompts": [f"neighborhood {case_id}"]}


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(ref.DEFAULT_CONFIG)
        self.pool = [record(i, relation=f"P{i % 5}") for i in range(5000, 5100)]

    def test_normalization_is_nfc_whitespace_not_lowercase(self):
        self.assertEqual(ref.norm("  Cafe\u0301\t A\n"), "Café A")
        self.assertNotEqual(ref.norm("A"), ref.norm("a"))

    def test_four_filters_in_sealed_order(self):
        fixed = [record(1, "fixed", "P1")]
        same_claim = record(2, "fixed", "P1")
        same_prompt = record(3, "other", "P3")
        same_prompt["requested_rewrite"]["prompt"] = "paraphrase one 1"
        same_subject = record(4, "fixed", "P2")
        same_subject["requested_rewrite"]["prompt"] = "different {}"
        keep = record(5, "allowed", "P9")
        pool, counts = ref.filter_general_pool([*fixed, same_claim, same_prompt, same_subject, keep], fixed)
        self.assertEqual([r["case_id"] for r in pool], [5])
        self.assertEqual(list(counts.values()), [4, 3, 2, 1, 1])

    def test_general_matches_inspector_exact_rank(self):
        design = Path(__file__).parents[3] / "plans/global/2026-10-02-jlz-two-arm-v2/inspect_design.py"
        # __file__ parents: jlz_two_arm, run_scripts, project, repository.
        spec = importlib.util.spec_from_file_location("sealed_jlz_inspector", design)
        inspector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inspector)
        for batch in range(20):
            current = [record(200 + batch, relation=f"P{batch % 5}")]
            actual = ref.select_general(self.pool, current, self.cfg, batch)
            self.assertEqual(actual, inspector.select_general(self.pool, current, self.cfg, batch))
            self.assertEqual(len(actual), 16)
            self.assertEqual(len({r["case_id"] for r in actual}), 16)
            self.assertTrue(all(r["requested_rewrite"]["relation_id"] == f"P{batch % 5}" for r in actual[:8]))

    def test_short_relation_fills_global_without_duplicates(self):
        chosen = ref.select_general(self.pool, [record(2, relation="absent")])
        self.assertEqual(len(chosen), 16)
        self.assertEqual(len({r["case_id"] for r in chosen}), 16)

    def test_supersession_and_current_exclusion(self):
        past = [record(1, "a", target="old"), record(2, "b"), record(3, "a", target="latest")]
        before = copy.deepcopy(past)
        self.assertEqual({r["case_id"] for r in ref.select_replay(past, [])}, {2, 3})
        self.assertEqual({r["case_id"] for r in ref.select_replay(past, [record(4, "a")])}, {2})
        self.assertEqual(past, before)
        self.assertEqual(ref.select_replay([], [record(4)]), [])

    def test_quartiles_and_small_past(self):
        for count in (1, 3, 4, 15, 16, 17, 100):
            past = [record(i) for i in range(count)]
            chosen = ref.select_replay(past, [], self.cfg, 7)
            self.assertEqual(len(chosen), min(16, count))
            self.assertEqual(len({ref.claim(r) for r in chosen}), len(chosen))
            if count >= 16:
                for q in range(4):
                    lo, hi = count * q // 4, count * (q + 1) // 4
                    self.assertEqual(sum(lo <= r["case_id"] < hi for r in chosen), 4)

    def test_full20_and_independent_pilot_prefix(self):
        stream = [record(i, relation=f"P{i % 5}") for i in range(2000)]
        rows = ref.build_reference_schedule(self.pool, stream)
        self.assertEqual(len(rows["main"]), 20)
        self.assertEqual(len(rows["pilot"]), 2)
        self.assertEqual([c for b in rows["main"] for c in b["current_ids"]], list(range(2000)))
        self.assertEqual([c for b in rows["pilot"] for c in b["current_ids"]], list(range(8)))
        self.assertEqual(rows["main"][0]["replay_count"], 0)
        self.assertEqual(rows["pilot"][0]["replay_count"], 0)
        self.assertEqual(rows["pilot"][1]["replay_count"], 4)
        for b in rows["main"]:
            self.assertTrue(all(c < b["batch_index"] * 100 for c in b["replay_ids"]))
        self.assertEqual(rows, ref.build_reference_schedule(self.pool, stream))

    def test_token_means_group_means_and_empty_replay(self):
        general = ref.prompt_token_means([[1., 3.], [8.]])
        self.assertEqual(general, [2., 8.])
        self.assertEqual(ref.group_mean_reference_loss(general, [], 100), 31.25)
        self.assertEqual(ref.group_mean_reference_loss(general, [2., 4.], 100), 331.25)
        with self.assertRaisesRegex(ValueError, "EMPTY_TARGET"):
            ref.prompt_token_means([[]])

    def test_group_loss_preserves_torch_gradient(self):
        import torch
        values = torch.tensor([2., 8., 2., 4.], requires_grad=True)
        loss = ref.group_mean_reference_loss(list(values[:2]), list(values[2:]), 100)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.tensor([3.125, 3.125, 50., 50.]))

    def test_teacher_identity_changes_with_tokens_and_rejects_padding(self):
        args = dict(model_sha256="m", tokenizer_sha256="t", input_ids=[1, 2, 3],
                    target_ids=[3, 4], prediction_positions=[1, 2], attention_mask=[1, 1, 1],
                    teacher_sha256="teacher")
        a = ref.teacher_identity(**args)
        b = ref.teacher_identity(**{**args, "input_ids": [1, 9, 3]})
        self.assertNotEqual(a["identity_sha256"], b["identity_sha256"])
        with self.assertRaisesRegex(ValueError, "PAD_POSITION"):
            ref.teacher_identity(**{**args, "attention_mask": [1, 1, 0]})
        with self.assertRaisesRegex(ValueError, "GENERAL_TEACHER_IDENTITY"):
            ref.teacher_identity(**{**args, "model_epoch": "armA"})

    def test_verification_and_identity_errors(self):
        raw = ref.compact([record(1)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_bytes(raw)
            rows, receipt = ref.verified_records(path, ref.digest(raw))
            self.assertEqual(len(rows), 1)
            self.assertEqual(receipt["bytes"], len(raw))
            with self.assertRaisesRegex(ValueError, "SHA_MISMATCH"):
                ref.verified_records(path, "wrong")
        with self.assertRaisesRegex(ValueError, "REFERENCE_CONFIG_CHANGED"):
            ref.select_general(self.pool, [], {"seed": 2})
        with self.assertRaisesRegex(ValueError, "duplicate case"):
            ref.filter_general_pool([record(1), record(1)], [])


if __name__ == "__main__":
    unittest.main()
