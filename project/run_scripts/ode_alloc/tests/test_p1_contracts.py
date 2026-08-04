from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_alloc.contracts import Arm, ODEAllocContractError
from project.run_scripts.ode_alloc.p1_contracts import (
    EXPECTED_BASE,
    P1_CASE_ORDER,
    P1_REQUEST_HASHES,
    P1_RUN_TOKEN,
    DamageAggregate,
    EndpointFreezeToken,
    HistoryRequest,
    P1Policy,
    assert_p1_seal,
    differentiable_damage_aggregate,
    expected_p1_result_name,
    historical_damage_aggregate,
    pretrained_damage_aggregate,
    remove_obsolete_requests,
    request_key,
)


ROOT = Path(__file__).resolve().parents[1]


class P1ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = json.loads((ROOT / "numerical_lock_proposal.json").read_text())
        cls.seal = json.loads((ROOT / "split_anchor_seal_candidate.json").read_text())
        cls.policy = P1Policy.from_lock(cls.lock)

    def test_exact_seal_order_anchor_count_and_policy(self) -> None:
        assert_p1_seal(self.seal)
        self.assertEqual(
            tuple(item["case_id"] for item in self.seal["p1_order"]),
            P1_CASE_ORDER,
        )
        self.assertEqual(
            tuple(item["request_hash"] for item in self.seal["p1_order"]),
            P1_REQUEST_HASHES,
        )
        self.assertEqual(len(self.seal["pretrained_anchor"]), 16)
        self.assertEqual(self.policy.solver_budget.fixed_k, 8)
        self.assertEqual(self.policy.solver_budget.quantized_trial_budget, 25)
        self.assertEqual(self.policy.max_abs_centered_q, math.log(4.0))

    def test_lock_change_fails_closed(self) -> None:
        changed = json.loads(json.dumps(self.lock))
        changed["solver"]["fixed_k"] = 7
        with self.assertRaises(ODEAllocContractError):
            P1Policy.from_lock(changed)
        changed = json.loads(json.dumps(self.lock))
        changed["allocation_gauge"]["global_scale"] = 1.0
        with self.assertRaises(ODEAllocContractError):
            P1Policy.from_lock(changed)

    def test_damage_aggregation_is_fixed_scale_mean_plus_smoothmax(self) -> None:
        history = historical_damage_aggregate((0.0, 0.001, 0.002), policy=self.policy)
        self.assertIsInstance(history, DamageAggregate)
        self.assertEqual(history.mean, 0.001)
        self.assertAlmostEqual(
            history.aggregate, 0.5 * (history.mean + history.smooth_max)
        )
        self.assertAlmostEqual(
            history.barrier_value, self.policy.historical_scale - history.aggregate
        )
        empty = historical_damage_aggregate((), policy=self.policy)
        self.assertEqual(empty.raw, ())
        self.assertEqual(empty.aggregate, 0.0)
        anchors = pretrained_damage_aggregate((0.0,) * 16, policy=self.policy)
        self.assertEqual(anchors.aggregate, 0.0)
        with self.assertRaises(ODEAllocContractError):
            pretrained_damage_aggregate((0.0,) * 15, policy=self.policy)

    def test_differentiable_damage_has_finite_gradient_and_empty_zero(self) -> None:
        q = torch.tensor([0.2, -0.2], dtype=torch.float64, requires_grad=True)
        candidates = (q[0] + 0.3, q[1] + 0.4)
        reading = differentiable_damage_aggregate(
            candidates,
            (0.6, 0.7),
            fixed_scale=0.001,
            smooth_temperature=0.01,
            zero_reference=q,
        )
        gradient = torch.autograd.grad(reading.aggregate, q)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        empty = differentiable_damage_aggregate(
            (),
            (),
            fixed_scale=0.001,
            smooth_temperature=0.01,
            zero_reference=q,
        )
        self.assertEqual(float(empty.aggregate.detach()), 0.0)
        self.assertEqual(tuple(empty.raw.shape), (0,))

    def test_history_obsolete_collision_is_audit_only(self) -> None:
        request = {
            "case_id": 1,
            "prompt": "{} is in",
            "relation_id": "R",
            "subject": "Alpha",
            "target_new": " New",
            "target_old": " Old",
        }
        same = HistoryRequest(1, "a" * 64, request_key(request), request)
        other_request = dict(request, case_id=2, subject="Beta")
        other = HistoryRequest(2, "b" * 64, request_key(other_request), other_request)
        kept, obsolete = remove_obsolete_requests((same, other), request)
        self.assertEqual(tuple(item.case_id for item in kept), (2,))
        self.assertEqual(tuple(item.case_id for item in obsolete), (1,))

    def test_endpoint_freeze_is_hash_bound_and_result_name_closed(self) -> None:
        token = EndpointFreezeToken.create(
            arm=Arm.ODE_ALLOC,
            source_head=EXPECTED_BASE,
            parameter_sha256="a" * 64,
            action_sha256="b" * 64,
            edit_count=4,
        )
        token.validate()
        bad = EndpointFreezeToken(
            token.arm,
            token.source_head,
            token.parameter_sha256,
            token.action_sha256,
            token.edit_count,
            "0" * 64,
        )
        with self.assertRaises(ODEAllocContractError):
            bad.validate()
        self.assertEqual(
            expected_p1_result_name("llama3-8b-inst", "c" * 64, P1_RUN_TOKEN),
            "s04-p1-matched-llama3-8b-inst-cccccccc",
        )
        with self.assertRaises(ODEAllocContractError):
            expected_p1_result_name("llama3-8b-inst", "c" * 64, "retry")


if __name__ == "__main__":
    unittest.main()
