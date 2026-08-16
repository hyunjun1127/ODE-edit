from __future__ import annotations

import hashlib
import inspect
import unittest

from project.run_scripts.ode_bf import alpha_backend, evaluator, p0_runtime
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.request_digest import (
    ORDERED_REQUEST_DIGEST_SCHEMA,
    ordered_request_digest_v1,
)


def _values() -> list[str]:
    return [f"{index + 1:064x}" for index in range(10)]


class OrderedRequestDigestTests(unittest.TestCase):
    def test_positive_is_versioned_deterministic_and_not_legacy_concatenation(self) -> None:
        values = _values()
        first = ordered_request_digest_v1(values)
        second = ordered_request_digest_v1(tuple(values))
        legacy = hashlib.sha256("".join(values).encode("ascii")).hexdigest()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, legacy)

    def test_reorder_duplicate_missing_and_schema_version_fail_closed(self) -> None:
        values = _values()
        reordered = values.copy()
        reordered[0], reordered[1] = reordered[1], reordered[0]
        self.assertNotEqual(
            ordered_request_digest_v1(values),
            ordered_request_digest_v1(reordered),
        )
        duplicated = values.copy()
        duplicated[-1] = duplicated[0]
        with self.assertRaisesRegex(ODEBFContractError, "duplicate"):
            ordered_request_digest_v1(duplicated)
        with self.assertRaisesRegex(ODEBFContractError, "exactly ten"):
            ordered_request_digest_v1(values[:-1])
        with self.assertRaisesRegex(ODEBFContractError, "schema"):
            ordered_request_digest_v1(values, schema_version="v2")

    def test_variable_length_concatenation_collision_is_rejected(self) -> None:
        common = [f"{index + 3:064x}" for index in range(8)]
        ambiguous_left = ["a", "bc", *common]
        ambiguous_right = ["ab", "c", *common]
        self.assertEqual("".join(ambiguous_left), "".join(ambiguous_right))
        for values in (ambiguous_left, ambiguous_right):
            with self.assertRaisesRegex(ODEBFContractError, "invalid SHA-256"):
                ordered_request_digest_v1(values)

    def test_missing_nonlowercase_and_nonstring_values_fail_closed(self) -> None:
        for invalid in (None, "A" * 64, "f" * 63, 7):
            values = _values()
            values[4] = invalid  # type: ignore[assignment]
            with self.assertRaisesRegex(ODEBFContractError, "invalid SHA-256"):
                ordered_request_digest_v1(values)

    def test_alpha_and_both_evaluators_import_the_shared_function(self) -> None:
        self.assertIs(alpha_backend.ordered_request_digest_v1, ordered_request_digest_v1)
        self.assertIs(evaluator.ordered_request_digest_v1, ordered_request_digest_v1)
        self.assertIs(p0_runtime.ordered_request_digest_v1, ordered_request_digest_v1)
        for function in (
            alpha_backend.capture_native_and_wb_joint_endpoint,
            evaluator.evaluate_counterfact_rewrite_batch,
            evaluator.evaluate_zsre_rewrite_batch,
            p0_runtime.run_p0,
        ):
            source = inspect.getsource(function)
            self.assertIn("ordered_request_digest_v1", source)
            self.assertNotIn('hashlib.sha256("".join', source)
        self.assertEqual(
            ORDERED_REQUEST_DIGEST_SCHEMA,
            "ode-edit-s04-ode-bf-ordered-request-digest/v1",
        )


if __name__ == "__main__":
    unittest.main()
