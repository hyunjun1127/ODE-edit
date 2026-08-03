from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_edit_method.contracts import (
    ControllerConfig,
    MethodContractError,
)
from project.run_scripts.ode_edit_method.events import (
    ControllerRequest,
    InformationFirewall,
    event_from_log_likelihoods,
    normalize_object_text,
)
from project.run_scripts.ode_edit_method.hooks import (
    FactorDirection,
    TorchCheckpoint,
    TorchFactorTrial,
    terminal_net_c_energy,
)


class InformationFirewallTests(unittest.TestCase):
    def test_counterfact_projection_drops_evaluation_fields(self) -> None:
        row = {
            "case_id": 17,
            "requested_rewrite": {
                "prompt": "{} lives in",
                "subject": "Ada",
                "target_new": {"str": "Paris"},
                "target_true": {"str": "London"},
            },
            "paraphrase_prompts": ["evaluation secret"],
            "neighborhood_prompts": ["held-out secret"],
            "generation_prompts": ["generation secret"],
        }
        request = ControllerRequest.from_counterfact_row(row)
        self.assertEqual(request.case_id, "17")
        self.assertFalse(hasattr(request, "paraphrase_prompts"))
        self.assertNotIn("secret", repr(request))

        firewall = InformationFirewall(
            request,
            evaluation_payload={"paraphrases": row["paraphrase_prompts"]},
        )
        with self.assertRaises(MethodContractError):
            firewall.open_evaluation()
        action_hash = firewall.freeze_action({"alpha": 0.5})
        self.assertEqual(len(action_hash), 64)
        self.assertEqual(
            firewall.open_evaluation(),
            {"paraphrases": ["evaluation secret"]},
        )

    def test_event_uses_length_normalized_log_likelihoods(self) -> None:
        reading = event_from_log_likelihoods(
            target_new=(-1.0, -3.0),
            target_old=(-2.0, -2.5),
            tau=0.2,
        )
        self.assertEqual(reading.context_margins, (1.0, -0.5))
        self.assertAlmostEqual(reading.hard_phi, 0.5)
        self.assertEqual(normalize_object_text("Paris"), " Paris")
        self.assertEqual(normalize_object_text(" Paris"), " Paris")

    def test_controller_schema_has_no_model_specific_branch(self) -> None:
        fields = set(ControllerConfig.__dataclass_fields__)
        for forbidden in (
            "model",
            "model_alias",
            "llama",
            "qwen",
            "fallback",
            "sign_rule",
        ):
            self.assertNotIn(forbidden, fields)


class TrialRollbackTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)).double()
        self.direction = FactorDirection(
            layer=0,
            weight_name="0.weight",
            left=torch.tensor([[0.2], [-0.4]], dtype=torch.float64),
            right=torch.tensor([[0.5], [0.3], [-0.1]], dtype=torch.float64),
        )

    def test_reject_and_exception_restore_weight_and_rng_exactly(self) -> None:
        checkpoint = TorchCheckpoint.capture(self.model, ("0.weight",))
        with TorchFactorTrial(self.model, (self.direction,), (0.7,)):
            _ = torch.rand(4)
        checkpoint.assert_exact(self.model, include_rng=True)

        with self.assertRaisesRegex(RuntimeError, "failure"):
            with TorchFactorTrial(self.model, (self.direction,), (0.7,)):
                _ = torch.rand(4)
                raise RuntimeError("failure")
        checkpoint.assert_exact(self.model, include_rng=True)

    def test_terminal_net_energy_is_subdivision_invariant(self) -> None:
        entry = TorchCheckpoint.capture(self.model, ("0.weight",))
        covariance = {0: torch.eye(3, dtype=torch.float64)}
        names = {0: "0.weight"}
        with TorchFactorTrial(self.model, (self.direction,), (0.8,)) as trial:
            trial.commit()
        one_step = terminal_net_c_energy(self.model, entry, covariance, names)
        entry.restore(self.model)

        for _ in range(4):
            with TorchFactorTrial(self.model, (self.direction,), (0.2,)) as trial:
                trial.commit()
        four_steps = terminal_net_c_energy(self.model, entry, covariance, names)
        self.assertAlmostEqual(one_step[0], four_steps[0], places=13)


if __name__ == "__main__":
    unittest.main()
