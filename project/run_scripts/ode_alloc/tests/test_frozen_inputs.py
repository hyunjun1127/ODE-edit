from __future__ import annotations

import unittest

from project.run_scripts.ode_alloc.contracts import (
    ODEAllocContractError,
    canonical_hash,
)
from project.run_scripts.ode_alloc.frozen_inputs import (
    REQUIRED_INPUTS,
    OneShotEditInputs,
)


class FrozenInputTests(unittest.TestCase):
    def test_all_expensive_inputs_are_built_exactly_once_and_reused(self) -> None:
        inputs = OneShotEditInputs("edit-1")
        calls = {name: 0 for name in REQUIRED_INPUTS}
        payloads = {}

        def builder(name: str):
            def build():
                calls[name] += 1
                return {"name": name, "value": [calls[name]]}

            return build

        for name in REQUIRED_INPUTS:
            payloads[name] = inputs.capture_from(
                name,
                builder(name),
                canonical_hash,
            )
            self.assertIs(inputs.read(name), payloads[name])
        receipt = inputs.seal()
        self.assertEqual(dict(receipt.capture_counts), {name: 1 for name in REQUIRED_INPUTS})
        self.assertEqual(calls, {name: 1 for name in REQUIRED_INPUTS})
        self.assertEqual(len(receipt.receipt_id), 64)
        self.assertIs(inputs.seal(), receipt)
        with self.assertRaises(ODEAllocContractError):
            inputs.capture_from("direct_z", builder("direct_z"), canonical_hash)
        self.assertEqual(calls["direct_z"], 1)

    def test_incomplete_or_mutated_capture_fails_closed(self) -> None:
        inputs = OneShotEditInputs("edit-2")
        payload = inputs.capture_from(
            "direct_z", lambda: {"values": [1]}, canonical_hash
        )
        with self.assertRaises(ODEAllocContractError):
            inputs.seal()
        payload["values"].append(2)
        with self.assertRaises(ODEAllocContractError):
            inputs.read("direct_z")


if __name__ == "__main__":
    unittest.main()
