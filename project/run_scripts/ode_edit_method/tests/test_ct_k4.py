from __future__ import annotations

import ast
import math
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
import inspect
from types import SimpleNamespace
from contextlib import AbstractContextManager
from typing import Any

import torch

import project.run_scripts.ode_edit_method.ct_k4_evaluation as ct_k4_evaluation

from project.run_scripts.ode_edit_motivation.contracts import (
    LowRankFactor,
    MemitFactorProposal,
    ParameterRecord,
    ProposalSemantics as MotivationSemantics,
    SnapshotManifest,
)
from project.run_scripts.ode_edit_method.controller import solve_full_transport_qp
from project.run_scripts.ode_edit_method.ct_k4 import (
    CT_ARM_ORDER,
    CT_CUMULATIVE,
    CTArm,
    CT_K,
    CT_LAMBDAS,
    NAIVE_QUARTER_RESIDUAL_FRACTION,
    assert_shared_direct_z_identities,
    observe_first_hit,
    run_ct_arm,
)
from project.run_scripts.ode_edit_method.contracts import (
    LayerProposal,
    MethodContractError,
    ProposalBatch,
    ProposalSemantics,
)
from project.run_scripts.ode_edit_method.easyedit_backend import (
    EasyEditBackendCheckpoint,
    EasyEditMemitBackend,
)
from project.run_scripts.ode_edit_method.ct_k4_evaluation import (
    _token_agreement,
    evaluate_frozen_endpoint,
    make_firewall,
    teacher_forced_token_accuracy,
)
from project.run_scripts.ode_edit_method.events import ControllerRequest
from project.run_scripts.ode_edit_method.event_strength import assert_raw_free
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.contracts import EventReading
from project.run_scripts.session03_ct_k4_common import (
    _evaluation_firewall_metadata,
    _PostActionEvaluationInstrumentation,
    _PostActionForwardCounter,
    _require_session03_output_root,
    dry_plan,
    expected_output_root,
    run as run_session03,
)
from project.run_scripts.ode_edit_method.hooks import (
    FactorDirection,
    TorchCheckpoint,
    apply_accepted_factors,
    set_cumulative_factors_from_checkpoint,
)
from project.run_scripts.ode_edit_method.lock import controller_config, load_lock
from project.run_scripts.ode_edit_method.ct_k4_lock import (
    CT_K4_LOCK_PATH,
    load_ct_k4_lock,
)
from project.run_scripts.ode_edit_method.memit_adapter import synchronous_normalization


def _snapshot() -> SnapshotManifest:
    return SnapshotManifest(
        model_id="toy/model@" + "a" * 40,
        context_id="b" * 64,
        request_ids=("c" * 64,),
        hparams_sha256="d" * 64,
        parameters=(
            ParameterRecord(
                name="layers.0.weight",
                sha256="e" * 64,
                shape=(3, 2),
                dtype="torch.float64",
            ),
            ParameterRecord(
                name="layers.1.weight",
                sha256="f" * 64,
                shape=(3, 2),
                dtype="torch.float64",
            ),
        ),
        provenance_ids=("0" * 64,),
    )


def _raw_proposal() -> MemitFactorProposal:
    return MemitFactorProposal(
        snapshot=_snapshot(),
        factors=(
            LowRankFactor(
                "layers.0.weight",
                torch.tensor([[2.0], [0.5], [-1.0]], dtype=torch.float64),
                torch.tensor([[1.5], [-0.25]], dtype=torch.float64),
                "e" * 64,
            ),
            LowRankFactor(
                "layers.1.weight",
                torch.tensor([[0.25], [-0.75], [1.25]], dtype=torch.float64),
                torch.tensor([[0.5], [2.0]], dtype=torch.float64),
                "f" * 64,
            ),
        ),
        semantics=MotivationSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
        solver_name="toy/session03-raw",
        residual_denominator=2,
    )


class _TwoLinear(torch.nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 3, bias=False, dtype=dtype)]
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers[0](value)


class _ToyTokenBatch(dict):
    def to(self, device: torch.device | str):
        return _ToyTokenBatch(
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in self.items()
            }
        )


class _ToyTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.padding_side = "right"

    @staticmethod
    def _token_id(word: str) -> int:
        return 3 + sum(word.encode("utf-8")) % 53

    def encode(self, text: str, **_kwargs):
        return [1, *(self._token_id(word) for word in text.split())]

    def apply_chat_template(
        self,
        conversations,
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ):
        if not add_generation_prompt or tokenize:
            raise AssertionError("toy chat-template contract differs")
        return [
            f"[USER] {conversation[0]['content']} [ASSISTANT]"
            for conversation in conversations
        ]

    def __call__(
        self,
        texts,
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ):
        if isinstance(texts, str):
            texts = [texts]
        if not padding or not truncation or return_tensors != "pt":
            raise AssertionError("toy tokenizer invocation differs")
        rows = [self.encode(text)[:max_length] for text in texts]
        width = max(len(row) for row in rows)
        padded = []
        masks = []
        for row in rows:
            count = width - len(row)
            if self.padding_side == "left":
                padded.append([self.pad_token_id] * count + row)
                masks.append([0] * count + [1] * len(row))
            else:
                padded.append(row + [self.pad_token_id] * count)
                masks.append([1] * len(row) + [0] * count)
        return _ToyTokenBatch(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        )


class _ToyConfig:
    def __init__(self) -> None:
        self.torch_dtype = "torch.float64"
        self.use_cache = False

    def to_dict(self):
        return {"torch_dtype": self.torch_dtype, "use_cache": self.use_cache}


class _ToyCausalLM(torch.nn.Module):
    def __init__(self, tokenizer: _ToyTokenizer, *, fail: bool = False) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.config = _ToyConfig()
        self._wrong_token_id = tokenizer._token_id("wrong")
        self._fail = fail

    def forward(self, input_ids: torch.Tensor, **_kwargs):
        if self._fail:
            raise RuntimeError("injected evaluator failure")
        vocabulary = 64
        chosen = torch.zeros_like(input_ids)
        chosen[:, :-1] = input_ids[:, 1:]
        chosen[:, -1] = 2
        chosen = torch.where(
            chosen == self._wrong_token_id,
            torch.full_like(chosen, 2),
            chosen,
        )
        logits = torch.full(
            (*input_ids.shape, vocabulary),
            -10.0,
            dtype=self.anchor.dtype,
            device=input_ids.device,
        )
        logits.scatter_(2, chosen.unsqueeze(-1), 10.0)
        logits = logits + self.anchor * 0.0
        return SimpleNamespace(logits=logits)


def _locked_teacher_forced_reference(
    model: torch.nn.Module,
    tokenizer: _ToyTokenizer,
    hparams: Any,
    prompts: tuple[str, ...],
    targets: tuple[str, ...],
    *,
    locality: bool,
):
    rendered = prompts
    if not locality and bool(getattr(hparams, "use_chat_template", False)):
        rendered = tuple(
            tokenizer.apply_chat_template(
                [[{"role": "user", "content": prompt}] for prompt in prompts],
                add_generation_prompt=True,
                tokenize=False,
            )
        )
    combined = tuple(
        prompt + " " + target
        for prompt, target in zip(rendered, targets, strict=True)
    )
    max_length = max(
        int(hparams.max_length), max(len(tokenizer.encode(row)) for row in combined) + 1
    )
    original = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        combined_batch = tokenizer(
            list(combined),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_batch = tokenizer(
            list(rendered),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            predicted = model(**combined_batch).logits.argmax(dim=-1)
        values = []
        for index in range(len(prompts)):
            prompt_count = int(
                (prompt_batch["input_ids"][index] != tokenizer.pad_token_id).sum()
            )
            padding_count = int(
                (combined_batch["input_ids"][index] == tokenizer.pad_token_id).sum()
            )
            start = padding_count + prompt_count
            answer = predicted[index, start - 1 : -1].tolist()
            label = combined_batch["input_ids"][index, start:].tolist()
            values.append(
                answer
                if locality
                else sum(left == right for left, right in zip(answer, label, strict=True))
                / len(label)
            )
        return values
    finally:
        tokenizer.padding_side = original


class _ToyTrial(AbstractContextManager):
    def __init__(self, backend: "_ToyTransportBackend", coefficients: tuple[float, ...]):
        self.backend = backend
        self.applied_coefficients = coefficients

    def __enter__(self):
        self.backend.trial_state = self.backend.state + 1
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.backend.trial_state = None
        return False


class _ToyTransportBackend:
    layers = (0, 1)

    def __init__(self) -> None:
        self.state = 0
        self.trial_state: int | None = None
        self.mechanism_field_history = ()

    def current_state_id(self) -> str:
        return f"state-{self.state}"

    def checkpoint(self):
        return SimpleNamespace(state_id=self.current_state_id(), state=self.state)

    def restore(self, checkpoint):
        self.state = checkpoint.state

    def assert_checkpoint(self, checkpoint):
        if self.current_state_id() != checkpoint.state_id:
            raise AssertionError("checkpoint differs")

    def prepare_event_target(self, frozen_target):
        return None

    def event(self, request):
        effective = self.state if self.trial_state is None else self.trial_state
        margin = -1.0 if effective == 0 else 1.0
        return EventReading(
            hard_phi=-margin,
            smooth_phi=-margin,
            context_margins=(margin,),
            nfe=2,
            target_new_log_likelihoods=(margin,),
            target_old_log_likelihoods=(0.0,),
            event_mode="toy",
            decision_deficits=(-margin,),
        )

    def build_transport_field(self, frozen_target):
        state = self.current_state_id()
        directions = tuple(
            FactorDirection(
                layer,
                f"layers.{layer}.weight",
                torch.ones((1, 1), dtype=torch.float64),
                torch.ones((1, 1), dtype=torch.float64),
            )
            for layer in self.layers
        )
        proposals = tuple(
            LayerProposal(layer, state, direction.direction_id, direction)
            for layer, direction in zip(self.layers, directions, strict=True)
        )
        return (
            ProposalBatch(
                state,
                proposals,
                (1.0, 0.5),
                ProposalSemantics.CURRENT_SAME_SNAPSHOT,
            ),
            (0.5, 0.5),
        )

    def trial(self, batch, coefficients):
        return _ToyTrial(self, tuple(coefficients))

    def commit(self, batch, coefficients):
        self.state += 1
        return tuple(coefficients)

    def commit_frozen_cumulative(self, entry, batch, coefficients, fraction):
        self.state = int(round(fraction * 4))
        return tuple(fraction * value for value in coefficients)

    def terminal_net_energy(self, entry):
        return {0: float(self.state), 1: float(self.state)}


class CTK4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = controller_config(load_lock())

    def test_raw_c_coefficients_exactly_reconstruct_synchronous_factors(self) -> None:
        raw = _raw_proposal()
        normalized = synchronous_normalization(
            raw,
            {0: torch.eye(2, dtype=torch.float64), 1: torch.eye(2, dtype=torch.float64)},
            {"layers.0.weight": 0, "layers.1.weight": 1},
            {0: 1.0, 1: 0.5},
            unit_c_norm_epsilon=1e-12,
            unit_c_identity_atol=2e-5,
            unit_c_identity_rtol=2e-5,
        )
        self.assertEqual(normalized.batch.layers, (0, 1))
        for raw_factor, proposal, coefficient in zip(
            raw.factors,
            normalized.batch.proposals,
            normalized.raw_coefficients,
            strict=True,
        ):
            direction = proposal.payload
            expected = raw_factor.left @ raw_factor.right.T
            observed = (direction.left * coefficient) @ direction.right.T
            torch.testing.assert_close(observed, expected, atol=1e-12, rtol=1e-12)

    def test_capacity_corrector_preserves_full_progress_and_raw_radius(self) -> None:
        result = solve_full_transport_qp(
            layers=(0, 1, 2),
            slopes=(2.0, 1.0, 0.25),
            raw_coefficients=(0.3, 0.4, 0.2),
            omega={0: 0.0, 1: 2.0, 2: 0.5},
            denominators={0: 1.0, 1: 3.0, 2: 2.0},
            config=self.config,
        )
        expected = 2.0 * 0.3 + 1.0 * 0.4 + 0.25 * 0.2
        self.assertAlmostEqual(result.full_progress, expected, places=12)
        self.assertAlmostEqual(result.qp.predicted_progress, expected, places=8)
        self.assertLessEqual(
            result.qp.coefficient_norm,
            math.sqrt(0.3**2 + 0.4**2 + 0.2**2)
            + self.config.qp_trust_tolerance,
        )
        self.assertFalse(result.zero_slope_raw_completion)

    def test_zero_slope_uses_raw_completion_without_model_branch(self) -> None:
        raw = (0.3, 0.4)
        result = solve_full_transport_qp(
            layers=(0, 1),
            slopes=(0.0, self.config.slope_epsilon / 2.0),
            raw_coefficients=raw,
            omega={0: 0.0, 1: 0.0},
            denominators={0: 1.0, 1: 1.0},
            config=self.config,
        )
        self.assertTrue(result.zero_slope_raw_completion)
        self.assertEqual(result.qp.coefficients, raw)

    def test_equal_time_lambda_schedule_and_naive_negative_control(self) -> None:
        remaining = 1.0
        endpoints = []
        for value in CT_LAMBDAS:
            remaining *= 1.0 - value
            endpoints.append(1.0 - remaining)
        self.assertEqual(len(endpoints), CT_K)
        for observed, expected in zip(endpoints, CT_CUMULATIVE, strict=True):
            self.assertAlmostEqual(observed, expected, places=15)
        self.assertAlmostEqual(NAIVE_QUARTER_RESIDUAL_FRACTION, 0.31640625)

    def test_first_hit_is_observed_without_freezing_non_es_arms(self) -> None:
        observed, stop = observe_first_hit(
            None, completed_step=2, hit=True, early_stop_enabled=False
        )
        self.assertEqual(observed, 2)
        self.assertFalse(stop)
        self.assertEqual(
            observe_first_hit(
                observed, completed_step=3, hit=True, early_stop_enabled=False
            ),
            (2, False),
        )
        self.assertEqual(
            observe_first_hit(
                None, completed_step=2, hit=True, early_stop_enabled=True
            ),
            (2, True),
        )

    def test_d_runs_all_four_after_observed_hit_but_es_freezes(self) -> None:
        request = ControllerRequest("1", "{} is", "Ada", "Paris", "London")
        primary_backend = _ToyTransportBackend()
        primary_metrics = EditInstrumentation("toy-primary")
        primary = run_ct_arm(
            CTArm.ODE_REFRESH_CT_K4,
            request=request,
            backend=primary_backend,
            frozen_target=object(),
            denominators={0: 1.0, 1: 1.0},
            config=self.config,
            instrumentation=primary_metrics,
        )
        snapshot = primary_metrics.finalize().to_dict()
        self.assertEqual(len(primary.steps), 4)
        self.assertEqual(primary.first_hit_step, 1)
        self.assertFalse(snapshot["first_hit"])
        self.assertEqual(snapshot["counters"]["N_field"], 4)

        es_backend = _ToyTransportBackend()
        es_metrics = EditInstrumentation("toy-es")
        early = run_ct_arm(
            CTArm.ODE_REFRESH_CT_K4_ES,
            request=request,
            backend=es_backend,
            frozen_target=object(),
            denominators={0: 1.0, 1: 1.0},
            config=self.config,
            instrumentation=es_metrics,
        )
        es_snapshot = es_metrics.finalize().to_dict()
        self.assertEqual(len(early.steps), 1)
        self.assertEqual(early.first_hit_step, 1)
        self.assertTrue(es_snapshot["first_hit"])

    def test_shared_direct_z_requires_one_global_compute_and_exact_arm_hashes(self) -> None:
        identity = {
            "tensor_sha256": "a" * 64,
            "artifact_sha256": "b" * 64,
            "artifact_size": 123,
            "source_state_id": "c" * 64,
        }
        rows = {arm: dict(identity) for arm in CT_ARM_ORDER}
        assert_shared_direct_z_identities(identity, rows, global_n_z=1)
        with self.assertRaisesRegex(MethodContractError, "global N_z"):
            assert_shared_direct_z_identities(identity, rows, global_n_z=2)
        rows[CTArm.BF_FROZEN_CT_K4]["tensor_sha256"] = "d" * 64
        with self.assertRaisesRegex(MethodContractError, "bf-frozen"):
            assert_shared_direct_z_identities(identity, rows, global_n_z=1)

    def test_frozen_backend_requires_entry_scoped_full_unique_batch(self) -> None:
        model = _TwoLinear(torch.float64).eval()
        entry_weights = TorchCheckpoint.capture(model, ("layers.0.weight",))
        snapshot = _snapshot()
        entry = EasyEditBackendCheckpoint(entry_weights, snapshot)
        direction = FactorDirection(
            0,
            "layers.0.weight",
            torch.ones((3, 1), dtype=torch.float64),
            torch.ones((2, 1), dtype=torch.float64),
        )
        proposal = LayerProposal(
            0, snapshot.state_id, direction.direction_id, direction
        )
        batch = ProposalBatch(
            snapshot.state_id,
            (proposal,),
            (1.0,),
            ProposalSemantics.CURRENT_SAME_SNAPSHOT,
        )
        fake = SimpleNamespace(_entry_snapshot=snapshot, model=model)
        mismatch = ProposalBatch(
            "wrong-state",
            (
                LayerProposal(
                    0, "wrong-state", direction.direction_id, direction
                ),
            ),
            (1.0,),
            ProposalSemantics.CURRENT_SAME_SNAPSHOT,
        )
        with self.assertRaisesRegex(MethodContractError, "entry-scoped"):
            EasyEditMemitBackend.commit_frozen_cumulative(
                fake, entry, mismatch, (1.0,), 0.25
            )
        fake.weight_by_layer = {0: "layers.0.weight", 1: "missing.weight"}
        # A full checkpoint with more targets than the batch must fail before write.
        extra_entry = EasyEditBackendCheckpoint(
            SimpleNamespace(weight_names=("layers.0.weight", "missing.weight")),
            snapshot,
        )
        with self.assertRaisesRegex(MethodContractError, "weight set"):
            EasyEditMemitBackend.commit_frozen_cumulative(
                fake, extra_entry, batch, (1.0,), 0.25
            )

    def test_frozen_cumulative_final_bytes_equal_one_shot_bf16(self) -> None:
        torch.manual_seed(13)
        model = _TwoLinear(torch.bfloat16).eval()
        direction = FactorDirection(
            0,
            "layers.0.weight",
            torch.tensor([[0.37], [-0.19], [0.11]], dtype=torch.float32),
            torch.tensor([[0.23], [-0.41]], dtype=torch.float32),
        )
        coefficient = (0.73,)
        entry = TorchCheckpoint.capture(model, (direction.weight_name,), backup_device="cpu")
        probe = torch.tensor([[0.31, -0.27]], dtype=torch.bfloat16)

        apply_accepted_factors(model, (direction,), coefficient)
        one_shot_weight = model.layers[0].weight.detach().clone()
        one_shot_output = model(probe).detach().clone()
        entry.restore(model)

        for fraction in CT_CUMULATIVE:
            set_cumulative_factors_from_checkpoint(
                model, entry, (direction,), coefficient, fraction
            )
        self.assertTrue(torch.equal(model.layers[0].weight, one_shot_weight))
        self.assertTrue(torch.equal(model(probe), one_shot_output))
        self.assertIsNone(model.layers[0].weight.grad)

    def test_frozen_float64_fixture_has_constant_target_displacement(self) -> None:
        model = _TwoLinear(torch.float64).eval()
        with torch.no_grad():
            model.layers[0].weight.zero_()
        direction = FactorDirection(
            0,
            "layers.0.weight",
            torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64),
            torch.tensor([[0.5], [0.25]], dtype=torch.float64),
        )
        entry = TorchCheckpoint.capture(model, (direction.weight_name,))
        previous = model.layers[0].weight.detach().clone()
        increments = []
        for fraction in CT_CUMULATIVE:
            set_cumulative_factors_from_checkpoint(
                model, entry, (direction,), (0.5,), fraction
            )
            current = model.layers[0].weight.detach().clone()
            increments.append(current - previous)
            previous = current
        for current in increments[1:]:
            torch.testing.assert_close(current, increments[0], atol=0.0, rtol=0.0)

    def test_fresh_panel_and_firewall_are_prelocked(self) -> None:
        lock = load_ct_k4_lock()
        self.assertEqual(lock["selection"]["p1_case_ids"], ["17503", "1534", "14652", "9774"])
        self.assertFalse(set(lock["selection"]["p1_case_ids"]) & {"2022", "12498", "20964", "768"})
        self.assertTrue(lock["evaluation"]["action_freeze_required"])
        self.assertFalse(lock["evaluation"]["controller_access"])
        self.assertFalse(lock["resources"]["submission_authorized"])
        raw = json.loads(CT_K4_LOCK_PATH.read_text(encoding="utf-8"))
        for mutate in (
            lambda row: row["selection"]["p1_case_ids"].__setitem__(0, "2022"),
            lambda row: row["policy"].__setitem__("K", 5),
            lambda row: row["evaluation"].__setitem__("controller_access", True),
        ):
            changed = json.loads(json.dumps(raw))
            mutate(changed)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "lock.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(MethodContractError):
                    load_ct_k4_lock(path)

    def test_evaluation_payload_stays_closed_until_action_freeze(self) -> None:
        request = ControllerRequest("1", "{} is", "Ada", "Paris", "London")
        private = {
            "paraphrase_prompts": ("Ada lives in",),
            "neighborhood_prompts": ("Grace lives in",),
            "target_true": "London",
        }
        firewall = make_firewall(request, private)
        self.assertEqual(firewall.controller_request, request)
        self.assertFalse(hasattr(firewall.controller_request, "paraphrase_prompts"))
        with self.assertRaisesRegex(MethodContractError, "closed"):
            firewall.open_evaluation()
        firewall.freeze_action({"terminal_state_id": "a" * 64})
        self.assertEqual(firewall.open_evaluation(), private)
        self.assertAlmostEqual(
            _token_agreement(((1, 2), (3, 4)), ((1, 9), (3, 4))),
            0.75,
        )
        source = inspect.getsource(make_firewall)
        self.assertNotIn("open_evaluation", source)
        controller_source = inspect.getsource(run_ct_arm)
        for forbidden in (
            "open_evaluation",
            "paraphrase",
            "neighborhood",
            "generation",
            "evaluation_payload",
        ):
            self.assertNotIn(forbidden, controller_source)

    def test_local_teacher_forced_evaluator_matches_locked_reference(self) -> None:
        for use_chat_template in (False, True):
            for prompts, targets in (
                (("short",), ("correct",)),
                (("short", "a much longer prompt"), ("correct", "wrong")),
            ):
                for locality in (False, True):
                    with self.subTest(
                        use_chat_template=use_chat_template,
                        batch=len(prompts),
                        locality=locality,
                    ):
                        tokenizer = _ToyTokenizer()
                        model = _ToyCausalLM(tokenizer).eval()
                        hparams = SimpleNamespace(
                            max_length=2,
                            use_chat_template=use_chat_template,
                        )
                        pointer = model.anchor.data_ptr()
                        version = model.anchor._version
                        rng = torch.get_rng_state().clone()
                        actual = teacher_forced_token_accuracy(
                            model,
                            tokenizer,
                            hparams,
                            prompts,
                            targets,
                            locality=locality,
                        )
                        reference = _locked_teacher_forced_reference(
                            model,
                            tokenizer,
                            hparams,
                            prompts,
                            targets,
                            locality=locality,
                        )
                        self.assertEqual(actual, reference)
                        self.assertEqual(tokenizer.padding_side, "right")
                        self.assertEqual(model.anchor.data_ptr(), pointer)
                        self.assertEqual(model.anchor._version, version)
                        self.assertIsNone(model.anchor.grad)
                        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_local_teacher_forced_left_padding_and_one_token_target(self) -> None:
        tokenizer = _ToyTokenizer()
        model = _ToyCausalLM(tokenizer).eval()
        hparams = SimpleNamespace(max_length=1, use_chat_template=False)
        prompts = ("one", "one two three four")
        targets = ("answer", "answer")
        predicted = teacher_forced_token_accuracy(
            model,
            tokenizer,
            hparams,
            prompts,
            targets,
            locality=True,
        )
        reference = _locked_teacher_forced_reference(
            model,
            tokenizer,
            hparams,
            prompts,
            targets,
            locality=True,
        )
        self.assertEqual(predicted, reference)
        self.assertEqual([len(row) for row in predicted], [1, 1])

    def test_local_teacher_forced_restores_padding_on_exception(self) -> None:
        tokenizer = _ToyTokenizer()
        model = _ToyCausalLM(tokenizer, fail=True).eval()
        hparams = SimpleNamespace(max_length=8, use_chat_template=False)
        with self.assertRaisesRegex(RuntimeError, "injected evaluator failure"):
            teacher_forced_token_accuracy(
                model,
                tokenizer,
                hparams,
                ("short",),
                ("answer",),
            )
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertIsNone(model.anchor.grad)

    def test_local_evaluator_has_no_easyedit_evaluator_import_or_generation(self) -> None:
        source = inspect.getsource(ct_k4_evaluation)
        self.assertNotIn("easyeditor.evaluate", source)
        tree = ast.parse(source)
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        self.assertFalse(any(name.startswith("easyeditor") for name in imported))
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "generate"
                for node in ast.walk(tree)
            )
        )

    def test_frozen_endpoint_uses_five_post_action_forwards(self) -> None:
        tokenizer = _ToyTokenizer()
        model = _ToyCausalLM(tokenizer).eval()
        runtime = SimpleNamespace(model=model, tokenizer=tokenizer)
        hparams = SimpleNamespace(max_length=2, use_chat_template=True)
        request = ControllerRequest("1", "{} lives", "Ada", "Paris", "London")
        firewall = make_firewall(
            request,
            {
                "paraphrase_prompts": ("Ada resides", "Ada is located"),
                "neighborhood_prompts": ("Grace lives", "Turing lived"),
                "target_true": "London",
            },
        )
        firewall.freeze_action({"terminal_state_id": "a" * 64})
        baseline = TorchCheckpoint.capture(model, ("anchor",), backup_device="cpu")
        endpoint = TorchCheckpoint.capture(model, ("anchor",), backup_device="cpu")
        metrics = EditInstrumentation("toy-local-evaluator")
        post_action = _PostActionForwardCounter(model)
        with post_action:
            with post_action.scope("terminal_residual"):
                model(input_ids=torch.tensor([[1, 2]], dtype=torch.long))
            result = evaluate_frozen_endpoint(
                runtime=runtime,
                hparams=hparams,
                request=request,
                firewall=firewall,
                baseline_checkpoint=baseline,
                endpoint_checkpoint=endpoint,
                instrumentation=_PostActionEvaluationInstrumentation(
                    metrics, post_action
                ),
            )
        endpoint.assert_exact(model, include_rng=True)
        post = post_action.finalize()
        snapshot = metrics.finalize().to_dict()
        self.assertFalse(result["generation_executed"])
        self.assertEqual(snapshot["counters"]["N_eval"], 5)
        self.assertEqual(snapshot["counters"]["N_model_fwd"], 0)
        self.assertEqual(post["N_post_action_model_fwd"], 6)
        self.assertEqual(
            post["post_action_model_fwd_scope_counts"],
            {"terminal_residual": 1, "endpoint_metrics": 5},
        )

    def test_paired_dry_plans_are_common_and_non_authorizing(self) -> None:
        lock = load_ct_k4_lock()
        for stage, expected_cases in (
            ("p0", ["2022"]),
            ("p1", ["17503", "1534", "14652", "9774"]),
        ):
            plan = dry_plan(lock, stage)
            self.assertFalse(plan["submission_authorized"])
            self.assertEqual(plan["case_ids"], expected_cases)
            self.assertEqual(plan["pair_gpu"], 2)
            self.assertEqual(plan["server1_project_gpu_cap"], 3)
            self.assertEqual(
                [job["model_alias"] for job in plan["jobs"]],
                ["llama3-8b-inst", "qwen2.5-7b-inst"],
            )
            self.assertTrue(
                all(lock["proposal_id"][:8] in job["output_root"] for job in plan["jobs"])
            )
            if stage == "p0":
                self.assertTrue(
                    all("session03-ct-k4-p0-r3-" in job["output_root"] for job in plan["jobs"])
                )
            else:
                self.assertTrue(
                    all("session03-ct-k4-p1-r1-" in job["output_root"] for job in plan["jobs"])
                )
        source = inspect.getsource(run_session03)
        self.assertNotIn("if args.model_alias", source)
        self.assertNotIn("elif args.model_alias", source)

    def test_session03_output_root_is_exact_create_once_and_symlink_safe(self) -> None:
        proposal = "a" * 64
        for stage in ("p0", "p1"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                repo = Path(directory).resolve()
                relative = expected_output_root(stage, "llama3-8b-inst", proposal)
                expected = repo / relative
                created = _require_session03_output_root(repo, expected, relative)
                self.assertEqual(created, expected)
                self.assertTrue(created.is_dir())
                with self.assertRaises(FileExistsError):
                    _require_session03_output_root(repo, expected, relative)

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            relative = expected_output_root("p0", "llama3-8b-inst", proposal)
            with self.assertRaises(ValueError):
                _require_session03_output_root(
                    repo,
                    repo / "local/results/wrong-basename",
                    relative,
                )
            with self.assertRaises(ValueError):
                _require_session03_output_root(
                    repo,
                    repo.parent / "escaped-session03-root",
                    Path("../escaped-session03-root"),
                )

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            target = repo / "real-results"
            target.mkdir()
            (repo / "local").mkdir()
            (repo / "local/results").symlink_to(target, target_is_directory=True)
            relative = expected_output_root("p0", "llama3-8b-inst", proposal)
            with self.assertRaises(ValueError):
                _require_session03_output_root(repo, repo / relative, relative)

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            relative = expected_output_root("p0", "llama3-8b-inst", proposal)
            (repo / "local/results").mkdir(parents=True)
            (repo / relative).symlink_to(repo / "missing-target")
            with self.assertRaises(FileExistsError):
                _require_session03_output_root(repo, repo / relative, relative)

    def test_tracked_session03_schemas_are_raw_free_before_model_load(self) -> None:
        forbidden = {
            "prompt",
            "subject",
            "target_new",
            "target_old",
            "raw_context",
            "context_templates",
            "templates",
            "evaluation",
            "generation",
        }
        for stage in ("p0", "p1"):
            skeleton = {
                "schema_version": f"ode-edit-session03-ct-k4-{stage}-manifest/v1",
                "evaluation_firewall": _evaluation_firewall_metadata(stage),
            }
            assert_raw_free(skeleton)

        tree = ast.parse(textwrap.dedent(inspect.getsource(run_session03)))
        tracked_names = {"manifest", "record", "summary"}
        observed: dict[str, set[str]] = {name: set() for name in tracked_names}

        def collect_keys(node: ast.AST) -> set[str]:
            return {
                key.value
                for item in ast.walk(node)
                if isinstance(item, ast.Dict)
                for key in item.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in tracked_names:
                    observed[target.id].update(collect_keys(node.value))
        self.assertEqual(set(observed), tracked_names)
        self.assertTrue(all(observed.values()))
        for name, keys in observed.items():
            self.assertFalse(forbidden & keys, (name, forbidden & keys))

    def test_es_freezes_controller_but_post_action_is_separately_counted(self) -> None:
        request = ControllerRequest("1", "{} is", "Ada", "Paris", "London")
        backend = _ToyTransportBackend()
        metrics = EditInstrumentation("toy-es-post-action")
        model = _TwoLinear(torch.float64).eval()
        probe = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
        metrics.attach_model(model)
        result = run_ct_arm(
            CTArm.ODE_REFRESH_CT_K4_ES,
            request=request,
            backend=backend,
            frozen_target=object(),
            denominators={0: 1.0, 1: 1.0},
            config=self.config,
            instrumentation=metrics,
        )
        metrics.detach_model()
        self.assertEqual(result.first_hit_step, 1)
        self.assertFalse(metrics.tracks_model_forwards)
        frozen_controller = {
            name: metrics._counters[name]
            for name in (
                "N_model_fwd",
                "N_field",
                "N_bw",
                "N_trial",
                "N_write",
                "N_z",
            )
        }

        post_action = _PostActionForwardCounter(model)
        with post_action:
            with post_action.scope("terminal_residual"):
                model(probe)
            adapter = _PostActionEvaluationInstrumentation(metrics, post_action)
            adapter.increment("N_eval")
            with adapter.component("evaluation"):
                model(probe)
        post = post_action.finalize()
        snapshot = metrics.finalize().to_dict()
        self.assertTrue(snapshot["first_hit"])
        self.assertEqual(snapshot["counters"]["N_model_fwd"], 0)
        self.assertEqual(snapshot["counters"]["N_field"], 1)
        self.assertEqual(snapshot["counters"]["N_eval"], 1)
        self.assertEqual(
            {name: snapshot["counters"][name] for name in frozen_controller},
            frozen_controller,
        )
        self.assertEqual(post["N_post_action_model_fwd"], 2)
        self.assertEqual(
            post["post_action_model_fwd_scope_counts"],
            {"terminal_residual": 1, "endpoint_metrics": 1},
        )
        self.assertEqual(snapshot["component_wall_seconds"]["evaluation"], 0.0)
        assert_raw_free({"compute": snapshot, **post})

    def test_post_action_hook_exception_cleanup_preserves_model_state(self) -> None:
        torch.manual_seed(91)
        model = _TwoLinear(torch.float64).eval()
        probe = torch.tensor([[0.17, -0.29]], dtype=torch.float64)
        parameter = model.layers[0].weight
        pointer = parameter.data_ptr()
        version = parameter._version
        requires_grad = parameter.requires_grad
        rng = torch.get_rng_state().clone()
        counter = _PostActionForwardCounter(model)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with counter:
                with counter.scope("terminal_residual"):
                    model(probe)
                    raise RuntimeError("injected post-action failure")
        self.assertFalse(counter.attached)
        self.assertEqual(parameter.data_ptr(), pointer)
        self.assertEqual(parameter._version, version)
        self.assertEqual(parameter.requires_grad, requires_grad)
        self.assertIsNone(parameter.grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))


if __name__ == "__main__":
    unittest.main()
