from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1_evaluator import (
    BatchEntryObservationSeal,
    CounterFactEvaluationCase,
    EndpointActionFreeze,
    evaluate_counterfact_success_accuracy_batch,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    build_pre_post_final_request_rows,
)
from project.run_scripts.ode_bf.request_digest import ordered_request_digest_v1
from project.run_scripts.ode_bf.tests.test_evaluator_firewall import (
    FakeCausalLM,
    FakeTokenizer,
)


def _cases() -> tuple[CounterFactEvaluationCase, ...]:
    return tuple(
        CounterFactEvaluationCase(
            index,
            canonical_hash({"request": index}),
            f"rewrite-{index}",
            (f"paraphrase-{index}-0", f"paraphrase-{index}-1"),
            (f"neighborhood-{index}",),
            "alpha beta",
            "gamma delta",
        )
        for index in range(10)
    )


class P1R52SequentialEvaluatorTests(unittest.TestCase):
    def test_pinned_alphaedit_accuracy_kernel_identity(self) -> None:
        source = Path(
            "/mnt/raid5/janghj/00.KE/02.LTE/AlphaEdit/experiments/py/"
            "eval_utils_counterfact.py"
        )
        self.assertEqual(
            hashlib.sha256(source.read_bytes()).hexdigest(),
            "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145",
        )
        text = source.read_text(encoding="utf-8")
        self.assertIn("def test_batch_prediction(", text)
        self.assertIn("logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item()", text)

    def test_entry_observation_seal_binds_weight_bytes_and_has_zero_influence(self) -> None:
        requests = [canonical_hash({"request": index}) for index in range(10)]
        order = ordered_request_digest_v1(requests)
        first = BatchEntryObservationSeal(
            "P1R52-SEQUENTIAL-ENTRY",
            0,
            order,
            canonical_hash({"weights": 1}),
            canonical_hash({"weights": 1}),
        )
        second = BatchEntryObservationSeal(
            "P1R52-SEQUENTIAL-ENTRY",
            0,
            order,
            canonical_hash({"weights": 2}),
            canonical_hash({"weights": 2}),
        )
        self.assertNotEqual(first.identity(), second.identity())
        self.assertEqual(first.controller_influence_count, 0)
        self.assertEqual(first.routing_influence_count, 0)
        with self.assertRaises(Exception):
            BatchEntryObservationSeal(
                "P1R52-SEQUENTIAL-ENTRY",
                0,
                order,
                canonical_hash({"weights": 1}),
                canonical_hash({"weights": 1}),
                controller_influence_count=1,
            )

    def test_one_logits_pass_yields_success_and_strict_accuracy(self) -> None:
        cases = _cases()
        freeze = EndpointActionFreeze(
            "P1R52-SEQUENTIAL",
            0,
            ordered_request_digest_v1([case.request_sha256 for case in cases]),
            canonical_hash({"endpoint": 1}),
            8,
        )
        model = FakeCausalLM(llama=True)
        receipt = evaluate_counterfact_success_accuracy_batch(
            model,
            FakeTokenizer(llama=True),
            cases,
            model_alias="llama3-8b-inst",
            freeze=freeze,
        )
        self.assertEqual(receipt.primary.model_forward_count, 10)
        self.assertEqual(receipt.added_model_forward_count, 0)
        self.assertEqual(receipt.added_backward_count, 0)
        self.assertEqual(receipt.added_generation_call_count, 0)
        self.assertEqual(
            receipt.rewrite_success.per_request_bits,
            receipt.primary.efficacy.per_case_bits,
        )
        self.assertEqual(
            receipt.paraphrase_success.per_request_bits,
            receipt.primary.generalization.per_case_bits,
        )
        # The two-token fixture verifies that the NLL preference and strict
        # all-token argmax views are independent definitions.
        paired = [
            (success, accuracy)
            for success_row, accuracy_row in zip(
                receipt.paraphrase_success.per_request_bits,
                receipt.paraphrase_accuracy.per_request_bits,
                strict=True,
            )
            for success, accuracy in zip(success_row, accuracy_row, strict=True)
        ]
        self.assertTrue(any(success != accuracy for success, accuracy in paired))

    def test_aliases_denominators_and_strict_paraphrase_aggregation_are_stable(self) -> None:
        cases = _cases()
        freeze = EndpointActionFreeze(
            "P1R52-SEQUENTIAL",
            0,
            ordered_request_digest_v1([case.request_sha256 for case in cases]),
            canonical_hash({"endpoint": 2}),
            8,
        )
        receipt = evaluate_counterfact_success_accuracy_batch(
            FakeCausalLM(llama=False),
            FakeTokenizer(llama=False),
            cases,
            model_alias="qwen2.5-7b-inst",
            freeze=freeze,
        )
        payload = receipt.raw_free_payload()
        self.assertEqual(payload["paraphrase_success"], payload["rephrase_success"])
        self.assertEqual(payload["paraphrase_acc"], payload["rephrase_acc"])
        self.assertEqual(receipt.rewrite_success.prompt_denominator, 10)
        self.assertEqual(receipt.paraphrase_success.prompt_denominator, 20)
        self.assertEqual(receipt.paraphrase_accuracy.strict_request_denominator, 10)
        self.assertEqual(
            receipt.paraphrase_accuracy.strict_all_prompt_bits,
            tuple(int(all(bits)) for bits in receipt.paraphrase_accuracy.per_request_bits),
        )
        self.assertEqual(receipt.controller_influence_count, 0)

    def test_pre_post_final_request_table_presence_and_denominator_alignment(self) -> None:
        base_cases = _cases()
        freeze = EndpointActionFreeze(
            "P1R52-SEQUENTIAL",
            0,
            ordered_request_digest_v1([case.request_sha256 for case in base_cases]),
            canonical_hash({"endpoint": 3}),
            8,
        )
        payload = evaluate_counterfact_success_accuracy_batch(
            FakeCausalLM(llama=True),
            FakeTokenizer(llama=True),
            base_cases,
            model_alias="llama3-8b-inst",
            freeze=freeze,
        ).raw_free_payload()
        request_batches = []
        pre = []
        post = []
        final = []
        for round_index in range(10):
            requests = tuple(
                {
                    "case_id": round_index * 10 + request_index,
                    "request_sha256": canonical_hash(
                        {"round": round_index, "request": request_index}
                    ),
                }
                for request_index in range(10)
            )
            request_batches.append(requests)
            order = ordered_request_digest_v1(
                [request["request_sha256"] for request in requests]
            )
            rows = [copy.deepcopy(payload) for _ in range(3)]
            for row in rows:
                row["legacy_primary"]["request_order_sha256"] = order
            pre.append(rows[0])
            post.append(rows[1])
            final.append(rows[2])
        table = build_pre_post_final_request_rows(
            request_batches, pre, post, final
        )
        self.assertEqual(len(table), 100)
        self.assertEqual(
            set(table[0]),
            {
                "round",
                "history_width_at_entry",
                "case_id",
                "request_index",
                "request_sha256",
                "entry_pre",
                "immediate_post",
                "final_W10",
                "deltas",
                "identity_sha256",
            },
        )
        bad_final = copy.deepcopy(final)
        bad_final[0]["rewrite_acc"]["per_request_required"][0] += 1
        with self.assertRaises(Exception):
            build_pre_post_final_request_rows(
                request_batches, pre, post, bad_final
            )


if __name__ == "__main__":
    unittest.main()
