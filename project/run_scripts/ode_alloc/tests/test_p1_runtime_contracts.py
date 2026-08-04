from __future__ import annotations

import contextlib
import io
import inspect
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_alloc.accounting import ComputeLedger, assert_matched_call_identity
from project.run_scripts.ode_alloc.contracts import Arm, SolverBudget
from project.run_scripts.ode_alloc.functional import (
    independent_native_weight_cpu_fixture,
    quantized_effective_weight,
)
from project.run_scripts.ode_alloc.gauge import FactorPair, FixedEnergyGauge
from project.run_scripts.ode_alloc.p1_firewall import (
    assert_no_alias_specific_scientific_branch,
    assert_p1_evaluator_firewall,
    assert_p1_inner_firewall,
    assert_projector_is_only_adaptive_arm_branch,
)
from project.run_scripts.ode_alloc.p1_runtime import (
    ProgressRecorder,
    RecordingProjector,
    _discard_easyedit_console_output,
    _preservation_specs,
    _projector_for_arm,
)
from project.run_scripts.ode_alloc.p1_contracts import (
    HistoryRequest,
    P1Policy,
    request_key,
)
from project.run_scripts.ode_alloc.p0_runtime import ScorePanel
from project.run_scripts.ode_alloc.p1_runtime import EditBasis, EditProblem
from project.run_scripts.ode_alloc.solver import (
    CBFProjector,
    CandidateVerdict,
    ConstraintLinearization,
    EvaluationCost,
    FieldEvaluation,
    FixedGridSolver,
    GenericProjector,
    LinearizationEvaluation,
)
from project.run_scripts.ode_alloc.transaction import AtomicLayerTransaction


ROOT = Path(__file__).resolve().parents[1]


class P1RuntimeContractTests(unittest.TestCase):
    def test_concrete_gradient_callback_emits_three_finite_tangent_rows(self) -> None:
        class Tokenizer:
            bos_token_id = 1
            unk_token_id = 0
            pad_token_id = 2
            eos_token_id = 2

            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                body = [3 + (ord(character) % 13) for character in text]
                return ([self.bos_token_id] if add_special_tokens else []) + body

        class CausalFixture(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = torch.nn.Linear(32, 4, bias=False, dtype=torch.bfloat16)
                self.second = torch.nn.Linear(4, 32, bias=False, dtype=torch.bfloat16)
                for parameter in self.parameters():
                    parameter.requires_grad_(False)

            def forward(
                self,
                *,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                use_cache: bool,
            ) -> types.SimpleNamespace:
                del attention_mask, use_cache
                encoded = torch.nn.functional.one_hot(
                    input_ids, num_classes=32
                ).to(dtype=torch.bfloat16)
                return types.SimpleNamespace(
                    logits=self.second(torch.nn.functional.silu(self.first(encoded)))
                )

        model = CausalFixture()
        factors = {
            "first.weight": FactorPair(
                0,
                torch.tensor([[0.2], [-0.1], [0.3], [0.1]]),
                torch.linspace(0.01, 0.32, 32).reshape(32, 1),
            ),
            "second.weight": FactorPair(
                1,
                torch.linspace(0.02, 0.64, 32).reshape(32, 1),
                torch.tensor([[0.2], [-0.2], [0.1], [0.3]]),
            ),
        }
        gauge = FixedEnergyGauge(
            {pair.layer: pair for pair in factors.values()},
            basis_energy_epsilon=1e-24,
            max_abs_centered_q=1.3862943611198906,
        )
        prompt_templates = tuple(
            (f"ctx-{index}", f"prefix-{index} {{}} relation")
            for index in range(6)
        )
        entry = ScorePanel(
            {key: -2.0 for key, _ in prompt_templates},
            {key: -2.5 for key, _ in prompt_templates},
            "a" * 64,
            "b" * 64,
        )
        direct = ScorePanel(
            {key: -1.0 for key, _ in prompt_templates},
            {key: -2.0 for key, _ in prompt_templates},
            "c" * 64,
            "d" * 64,
        )
        basis = EditBasis(
            factors_by_weight=factors,
            active_factors=factors,
            gauge=gauge,
            entry_panel=entry,
            direct_z_panel=direct,
            native_panel=entry,
            native_event=None,  # not consulted by the gradient callback
            native_weight_hashes={},
            entry_weight_hashes={},
            entry_pointers={},
            frozen_receipt=None,
            underlying_counts={},
            replay_writer_calls=1,
            oneshot=None,
        )
        anchors = tuple(
            {"prompt": "{} anchor", "subject": f"Anchor {index}", "target_old": "Old"}
            for index in range(16)
        )
        lock = json.loads(
            (ROOT / "numerical_lock_proposal.json").read_text(encoding="utf-8")
        )
        problem = EditProblem(
            model=model,
            tokenizer=Tokenizer(),
            request={
                "case_id": 1,
                "prompt": "{} relation",
                "relation_id": "R",
                "subject": "Subject",
                "target_new": "New",
                "target_old": "Old",
            },
            prompt_templates=prompt_templates,
            history=(),
            history_prompt_templates=(),
            anchors=anchors,
            theta0_anchor_teacher=(0.0,) * 16,
            basis=basis,
            policy=P1Policy.from_lock(lock),
        )
        nominal, linearization, cost = problem._gradient_state(
            torch.zeros(2, dtype=torch.float64)
        )
        self.assertTrue(torch.isfinite(nominal).all())
        self.assertAlmostEqual(float(nominal.sum()), 0.0, places=12)
        self.assertEqual(tuple(linearization.matrix.shape), (3, 2))
        self.assertEqual(linearization.labels, ("E", "H", "P"))
        self.assertTrue(torch.isfinite(linearization.matrix).all())
        self.assertEqual((cost.backward_calls, cost.constraint_vjp_calls), (4, 3))

    def test_easyedit_request_bearing_console_output_is_discarded(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with _discard_easyedit_console_output():
                print("request-bearing-stdout")
                print("request-bearing-stderr", file=sys.stderr)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_history_uses_each_prior_requests_authorized_context_panel(self) -> None:
        prior = {
            "case_id": 7,
            "prompt": "{} was born in",
            "relation_id": "R-born",
            "subject": "Prior Subject",
            "target_new": "Prior Target",
            "target_old": "Old Target",
        }
        history = HistoryRequest(7, "a" * 64, request_key(prior), prior)
        panel = tuple(
            (f"history-{index}", f"prefix-{index} {{}} was born in")
            for index in range(6)
        )
        anchors = tuple(
            {"prompt": "{} anchor", "subject": f"Anchor {index}", "target_old": "Old"}
            for index in range(16)
        )
        specs, groups, anchor_ids = _preservation_specs(
            (history,), anchors, (panel,)
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 6)
        self.assertEqual(len(anchor_ids), 16)
        self.assertEqual(specs[0].prompt, "prefix-0 Prior Subject was born in")
        with self.assertRaises(Exception):
            _preservation_specs((history,), anchors, ())

    def test_inner_evaluator_alias_firewalls_pass_and_reject_foreign_fixture(self) -> None:
        runtime = ROOT / "p1_runtime.py"
        evaluator = ROOT / "p1_evaluator.py"
        assert_p1_inner_firewall((runtime, ROOT / "p1_contracts.py"))
        assert_p1_evaluator_firewall(evaluator)
        assert_no_alias_specific_scientific_branch(runtime)
        assert_projector_is_only_adaptive_arm_branch(runtime)
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.py"
            bad.write_text('value = "paraphrase_prompts"\n', encoding="utf-8")
            with self.assertRaises(Exception):
                assert_p1_inner_firewall((bad,))
            bad.write_text("model.generate()\n", encoding="utf-8")
            with self.assertRaises(Exception):
                assert_p1_evaluator_firewall(bad)
            bad.write_text(
                "def choose(arm):\n"
                "    if arm is Arm.ODE_ALLOC:\n"
                "        return 1\n",
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                assert_projector_is_only_adaptive_arm_branch(bad)

    def test_generic_ode_use_one_solver_path_and_projector_is_only_hook(self) -> None:
        source = inspect.getsource(_projector_for_arm)
        self.assertIn("GenericProjector", source)
        self.assertIn("CBFProjector", source)
        self.assertNotIn("MODEL_ALIASES", source)
        budget = SolverBudget(2, 3, (1.0, 0.5, 0.25), 6)

        def direction(step: int, q: torch.Tensor) -> FieldEvaluation:
            del step
            return FieldEvaluation(
                torch.tensor([0.2, -0.2], dtype=torch.float64),
                EvaluationCost(model_forward_calls=1, processed_tokens=10, backward_calls=1),
            )

        def linearize(step: int, q: torch.Tensor) -> LinearizationEvaluation:
            del step, q
            return LinearizationEvaluation(
                ConstraintLinearization(
                    matrix=torch.tensor(
                        [[1.0, -1.0], [-1.0, 1.0], [0.5, -0.5]],
                        dtype=torch.float64,
                    ),
                    barrier_values=torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64),
                    kappa=1.0,
                    labels=("E", "H", "P"),
                ),
                EvaluationCost(constraint_vjp_calls=3, backward_calls=3),
            )

        def verdict(q: torch.Tensor) -> CandidateVerdict:
            return CandidateVerdict(
                True,
                float(torch.sum(q.square())),
                f"event-{float(q[0]):.8f}",
                model_forward_calls=1,
                processed_tokens=10,
            )

        ledgers = []
        endpoints = []
        for projector in (
            RecordingProjector(GenericProjector(), []),
            RecordingProjector(CBFProjector(slack_penalty_weight=100.0), []),
        ):
            ledger = ComputeLedger()
            endpoint = FixedGridSolver(budget=budget, projector=projector).run(
                torch.zeros(2, dtype=torch.float64),
                direction_fn=direction,
                linearize_fn=linearize,
                verdict_fn=verdict,
                ledger=ledger,
            )
            ledgers.append(ledger)
            endpoints.append(endpoint)
            self.assertEqual(len(projector.records), 2)
            self.assertEqual(ledger.quantized_trial_calls, 7)
        assert_matched_call_identity(ledgers[0], ledgers[1])
        self.assertEqual(endpoints[0].completed_steps, endpoints[1].completed_steps)

    def test_q0_and_nonzero_fixed_energy_quantized_transaction(self) -> None:
        base_a = torch.zeros((3, 2), dtype=torch.bfloat16)
        base_b = torch.zeros((2, 2), dtype=torch.bfloat16)
        pairs = {
            0: FactorPair(
                0,
                torch.tensor([[1.0], [0.5], [0.25]], dtype=torch.float64),
                torch.tensor([[0.5], [1.0]], dtype=torch.float64),
            ),
            1: FactorPair(
                1,
                torch.tensor([[0.25], [0.75]], dtype=torch.float64),
                torch.tensor([[1.0], [0.5]], dtype=torch.float64),
            ),
        }
        gauge = FixedEnergyGauge(
            pairs, basis_energy_epsilon=1e-24, max_abs_centered_q=1.3862943611198906
        )
        zero = gauge.evaluate(gauge.zeros())
        self.assertTrue(torch.equal(zero.ratios, torch.ones_like(zero.ratios)))
        self.assertTrue(
            torch.equal(
                quantized_effective_weight(base_a, pairs[0], 1.0),
                independent_native_weight_cpu_fixture(base_a, pairs[0]),
            )
        )
        nonzero = gauge.evaluate(
            {0: torch.tensor(0.2), 1: torch.tensor(-0.2)}
        )
        self.assertTrue(
            torch.allclose(nonzero.energy_before, nonzero.energy_after, rtol=1e-12, atol=1e-12)
        )
        parameters = {
            "a": torch.nn.Parameter(base_a.clone(), requires_grad=False),
            "b": torch.nn.Parameter(base_b.clone(), requires_grad=False),
        }
        transaction = AtomicLayerTransaction(parameters, mutation_lock=threading.RLock())
        transaction.stage("a", quantized_effective_weight(base_a, pairs[0], float(nonzero.ratios[0])))
        transaction.stage("b", quantized_effective_weight(base_b, pairs[1], float(nonzero.ratios[1])))
        transaction.commit()
        self.assertFalse(torch.equal(parameters["a"], base_a))

    def test_progress_receipts_are_ordered_create_once_and_raw_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            recorder = ProgressRecorder(raw)
            digest = recorder.complete("model-loaded", alias="locked", count=1)
            self.assertEqual(len(digest), 64)
            self.assertEqual(recorder.last_stage, "model-loaded")
            self.assertGreater(recorder.wall_seconds, 0.0)
            with self.assertRaises(Exception):
                recorder.complete("bad", prompt="raw")


if __name__ == "__main__":
    unittest.main()
