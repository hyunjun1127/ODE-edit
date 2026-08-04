from __future__ import annotations

import inspect
import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F

from project.run_scripts.ode_edit_method.contracts import (
    LayerProposal,
    MethodContractError,
    ProposalBatch,
    ProposalSemantics,
)
from project.run_scripts.ode_edit_method.events import ControllerRequest
from project.run_scripts.ode_edit_method.easyedit_backend import EasyEditMemitBackend
from project.run_scripts.ode_edit_method.event_strength import assert_raw_free
from project.run_scripts.ode_edit_method.hard_batch import (
    HARD_BATCH_ARM_ORDER,
    CounterFactEndpointMetrics,
    CounterFactJointReading,
    CounterFactPairReading,
    CounterFactPairSpec,
    HardBatchArm,
    efficacy_specs,
    fixed_horizon_lambdas,
    frozen_cumulative_fractions,
    native_floor_verdict,
    naive_repeated_residual_fraction,
    run_hard_batch_arm,
    score_counterfact_nll_pairs,
    zsre_correct_position_counts,
)
from project.run_scripts.ode_edit_method.hard_batch_backend import (
    JOINT_BATCH_SIZE,
    JointHardBatchEasyEditBackend,
)
from project.run_scripts.ode_edit_method.hard_batch_lock import (
    BATCH_CASE_IDS,
    HARD_BATCH_LOCK_PATH,
    load_hard_batch_lock,
)
from project.run_scripts.ode_edit_method.hard_batch_gpu_resource_r1 import (
    load_gpu_resource_lock,
)
from project.run_scripts.ode_edit_method.gpu_resource import inspect_gpu_resource
from project.run_scripts.ode_edit_method.hooks import (
    FactorDirection,
    TorchCheckpoint,
    apply_accepted_factors,
    set_cumulative_factors_from_checkpoint,
)
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.lock import controller_config, load_lock
from project.run_scripts.ode_edit_method.tests.test_ct_k4 import (
    _ToyTransportBackend,
    _TwoLinear,
)
from project.run_scripts.session03_hard_b10_common import (
    _assert_direct_z_identity,
    dry_plan,
)


def _requests() -> tuple[ControllerRequest, ...]:
    return tuple(
        ControllerRequest(str(index), "{} is", f"subject-{index}", "new", "old")
        for index in range(JOINT_BATCH_SIZE)
    )


class _SourceTokenizer:
    pad_token_id = 0
    padding_side = "right"

    @staticmethod
    def _id(value: str) -> int:
        return 3 + sum(value.encode("utf-8")) % 61

    def encode(self, text: str, *, add_special_tokens: bool = True):
        values = [self._id(value) for value in text.split()]
        return [1, *values] if add_special_tokens else values

    def __call__(self, texts, *, add_special_tokens: bool = True, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [
                self.encode(value, add_special_tokens=add_special_tokens)
                for value in texts
            ]
        }


class _BoundaryTokenizer(_SourceTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = True):
        values = super().encode(text, add_special_tokens=add_special_tokens)
        if text.startswith("boundary ") and len(values) > 1:
            values[-1] = (values[-1] + 7) % 63 + 1
        return values


class _SourceModel(torch.nn.Module):
    def __init__(self, name: str = "toy") -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.config = SimpleNamespace(_name_or_path=name)

    def forward(self, input_ids: torch.Tensor, **_kwargs):
        vocabulary = 72
        rows, width = input_ids.shape
        positions = torch.arange(width, device=input_ids.device).view(1, width, 1)
        ids = torch.arange(vocabulary, device=input_ids.device).view(1, 1, -1)
        centers = ((input_ids * 7 + positions.squeeze(-1) * 3 + 5) % vocabulary).unsqueeze(-1)
        logits = -(ids - centers).abs().to(torch.float64)
        return SimpleNamespace(logits=logits + self.anchor * 0.0)


class _FailingSourceModel(_SourceModel):
    def forward(self, input_ids: torch.Tensor, **_kwargs):
        raise RuntimeError("injected evaluator failure")


class _FakeCuda:
    def __init__(self) -> None:
        self.available = True
        self.count = 1
        self.index = 0
        self.properties = SimpleNamespace(
            name="NVIDIA RTX A6000",
            uuid="GPU-2087c567-ec90-0aa2-09c5-c5174daec87a",
            total_memory=50899386368,
        )
        self.capability = (8, 6)
        self.capacity_total = 50899386368 - 256 * 1024**2
        self.reserved = 16 * 1024**3
        self.allocated = 15 * 1024**3
        self.free = self.capacity_total - self.reserved

    def is_available(self):
        return self.available

    def device_count(self):
        return self.count

    def current_device(self):
        return self.index

    def get_device_properties(self, _index):
        return self.properties

    def get_device_capability(self, _index):
        return self.capability

    def mem_get_info(self, _index):
        return self.free, self.capacity_total

    def memory_allocated(self, _index):
        return self.allocated

    def memory_reserved(self, _index):
        return self.reserved


def _source_reference(
    model: torch.nn.Module,
    tokenizer: _SourceTokenizer,
    spec: CounterFactPairSpec,
) -> tuple[float, float]:
    prefix_len = len(tokenizer(spec.prefix)["input_ids"][0])
    full_rows = [
        tokenizer.encode(f"{spec.prefix} {target}")
        for target in (spec.target_new, spec.target_true)
    ]
    target_rows = [
        tokenizer(f" {target}")["input_ids"][0]
        for target in (spec.target_new, spec.target_true)
    ]
    width = max(map(len, full_rows))
    input_ids = torch.zeros((2, width), dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for index, row in enumerate(full_rows):
        input_ids[index, : len(row)] = torch.tensor(row)
        mask[index, : len(row)] = 1
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=mask).logits
    values = []
    for index, targets in enumerate(target_rows):
        loss = torch.tensor(0.0, dtype=torch.float32)
        for offset, token in enumerate(targets):
            loss += float(
                -F.log_softmax(logits[index, prefix_len + offset - 1], dim=0)[token]
            )
        loss /= len(targets)
        values.append(float(loss))
    return values[0], values[1]


def _joint_reading(success_count: int) -> CounterFactJointReading:
    rows = []
    for index in range(JOINT_BATCH_SIZE):
        success = index < success_count
        rows.append(
            CounterFactPairReading(
                case_id=str(index),
                role="efficacy",
                target_new_nll=0.0 if success else 2.0,
                target_true_nll=1.0,
                success=success,
                target_new_token_count=1,
                target_true_token_count=1,
                source_prefix_token_count=2,
                target_new_start=2,
                target_true_start=2,
            )
        )
    return CounterFactJointReading(tuple(rows), 1, 20, 0.0)


class _HardToyBackend(_ToyTransportBackend):
    def __init__(self, instrumentation: EditInstrumentation) -> None:
        super().__init__()
        self.requests = _requests()
        self.request = self.requests[0]
        self.instrumentation = instrumentation
        self.build_count = 0

    def capture_current_checkpoint(self):
        return self.checkpoint()

    @property
    def last_request_event_readings(self):
        reading = super().event(self.request)
        return tuple(reading for _ in range(JOINT_BATCH_SIZE))

    def direct_z_residual_norms(self, _frozen_target):
        value = float(max(0, 20 - self.state))
        return tuple(value + index / 100.0 for index in range(JOINT_BATCH_SIZE))

    def build_transport_field(self, frozen_target):
        self.build_count += 1
        self.instrumentation.increment("N_bw")
        return super().build_transport_field(frozen_target)

    def build_native_terminal(self, frozen_target):
        return super().build_transport_field(frozen_target)[0]

    def assert_genuine_joint_batch(self, _batch):
        return {
            "batch_size": JOINT_BATCH_SIZE,
            "factor_ranks": [JOINT_BATCH_SIZE] * len(self.layers),
            "singleton_decomposition": False,
        }


class _FailingHardToyBackend(_HardToyBackend):
    def commit(self, batch, coefficients):
        if self.state == 2:
            raise RuntimeError("injected joint transaction failure")
        return super().commit(batch, coefficients)


class HardBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = controller_config(load_lock())

    def test_k4_k10_hash_compatible_schedules_and_k20_equal_time(self) -> None:
        self.assertEqual(
            fixed_horizon_lambdas(4), (0.25, 1 / 3, 0.5, 1.0)
        )
        self.assertEqual(
            fixed_horizon_lambdas(10),
            tuple(1 / (10 - index) for index in range(10)),
        )
        remaining = 1.0
        for observed, expected in zip(
            fixed_horizon_lambdas(20),
            frozen_cumulative_fractions(20),
            strict=True,
        ):
            remaining *= 1.0 - observed
            self.assertAlmostEqual(1.0 - remaining, expected, places=14)
        self.assertAlmostEqual(
            naive_repeated_residual_fraction(10), 0.3486784401, places=15
        )

    def test_k20_frozen_final_bytes_output_and_event_equal_one_shot_bf16(self) -> None:
        torch.manual_seed(83)
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
        expected_weight = model.layers[0].weight.detach().clone()
        expected_output = model(probe).detach().clone()
        expected_event = float(expected_output.float().square().mean())
        entry.restore(model)
        for fraction in frozen_cumulative_fractions(20):
            set_cumulative_factors_from_checkpoint(
                model, entry, (direction,), coefficient, fraction
            )
        self.assertTrue(torch.equal(model.layers[0].weight, expected_weight))
        observed_output = model(probe).detach()
        self.assertTrue(torch.equal(observed_output, expected_output))
        self.assertEqual(
            float(observed_output.float().square().mean()), expected_event
        )

    def test_linear_resolution_and_nonlinear_second_cycle_are_separate(self) -> None:
        for resolution in (10, 20):
            remaining = 1.0
            for value in fixed_horizon_lambdas(resolution):
                remaining *= 1.0 - value
            self.assertAlmostEqual(remaining, 0.0, places=15)

        def cycle(value: float, resolution: int) -> float:
            for step in fixed_horizon_lambdas(resolution):
                value += step * math.tanh(1.0 - value)
            return value

        k10 = cycle(0.0, 10)
        k20 = cycle(0.0, 20)
        second_cycle = cycle(k10, 10)
        self.assertNotAlmostEqual(k10, k20, places=10)
        self.assertNotAlmostEqual(k20, second_cycle, places=10)

    def test_counterfact_nll_matches_pinned_source_semantics(self) -> None:
        tokenizer = _SourceTokenizer()
        model = _SourceModel().eval()
        spec = CounterFactPairSpec(
            "1", "efficacy", "Ada lives in", "Paris", "London", True
        )
        expected = _source_reference(model, tokenizer, spec)
        observed = score_counterfact_nll_pairs(model, tokenizer, (spec,))
        self.assertAlmostEqual(observed.rows[0].target_new_nll, expected[0])
        self.assertAlmostEqual(observed.rows[0].target_true_nll, expected[1])
        self.assertEqual(observed.success_count, int(expected[0] < expected[1]))
        self.assertEqual(observed.model_forward_count, 1)

        boundary = _BoundaryTokenizer()
        boundary_expected = _source_reference(model, boundary, spec)
        boundary_observed = score_counterfact_nll_pairs(model, boundary, (spec,))
        self.assertEqual(
            (
                boundary_observed.rows[0].target_new_nll,
                boundary_observed.rows[0].target_true_nll,
            ),
            boundary_expected,
        )

    def test_counterfact_nll_exception_restores_runtime_state(self) -> None:
        tokenizer = _SourceTokenizer()
        model = _FailingSourceModel().eval()
        spec = CounterFactPairSpec(
            "1", "efficacy", "Ada lives in", "Paris", "London", True
        )
        pointer = model.anchor.data_ptr()
        version = model.anchor._version
        requires_grad = model.anchor.requires_grad
        cpu_rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "evaluator failure"):
            score_counterfact_nll_pairs(model, tokenizer, (spec,))
        self.assertEqual(tokenizer.padding_side, "right")
        self.assertEqual(model.anchor.data_ptr(), pointer)
        self.assertEqual(model.anchor._version, version)
        self.assertEqual(model.anchor.requires_grad, requires_grad)
        self.assertIsNone(model.anchor.grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))

    def test_efficacy_prompt_and_integer_native_floor_are_locked(self) -> None:
        requests = _requests()
        specs = efficacy_specs(requests)
        self.assertEqual(
            tuple(item.prefix for item in specs),
            tuple(item.prompt.format(item.subject) for item in requests),
        )
        native = CounterFactEndpointMetrics(
            _joint_reading(10), _joint_reading(8), _joint_reading(9)
        )
        ours = CounterFactEndpointMetrics(
            _joint_reading(9), _joint_reading(9), _joint_reading(9)
        )
        verdict = native_floor_verdict(native, ours)
        self.assertFalse(verdict["pass"])
        self.assertFalse(verdict["metrics"]["efficacy"]["pass"])
        self.assertEqual(
            verdict["metrics"]["efficacy"]["sealed_noninferiority_margin_count"],
            0,
        )
        zsre = zsre_correct_position_counts((1, 7, 3), (1, 2, 3))
        self.assertEqual(zsre["correct_position_count"], 2)
        self.assertEqual(zsre["denominator"], 3)

        mismatched = CounterFactEndpointMetrics(
            CounterFactJointReading(
                rows=tuple(
                    CounterFactPairReading(
                        case_id=f"different-{index}",
                        role="efficacy",
                        target_new_nll=row.target_new_nll,
                        target_true_nll=row.target_true_nll,
                        success=row.success,
                        target_new_token_count=row.target_new_token_count,
                        target_true_token_count=row.target_true_token_count,
                        source_prefix_token_count=row.source_prefix_token_count,
                        target_new_start=row.target_new_start,
                        target_true_start=row.target_true_start,
                    )
                    for index, row in enumerate(_joint_reading(9).rows)
                ),
                model_forward_count=1,
                input_token_count=20,
                evaluator_wall_seconds=0.0,
            ),
            _joint_reading(9),
            _joint_reading(9),
        )
        with self.assertRaisesRegex(MethodContractError, "denominator"):
            native_floor_verdict(native, mismatched)

    def test_hard_batch_numeric_schemas_pass_raw_firewall(self) -> None:
        reading = _joint_reading(7)
        endpoint = CounterFactEndpointMetrics(reading, reading, reading)
        metrics = EditInstrumentation("hard-b10-firewall")
        snapshot = metrics.finalize().to_dict()
        assert_raw_free(
            {
                "benchmark": reading.to_dict(),
                "endpoint_metrics": endpoint.to_dict(),
                "compute": snapshot,
                "native_floor": native_floor_verdict(endpoint, endpoint),
                "autoregressive_decoding_count": 0,
            },
            "hard-B10 focused schema",
        )

    def test_gpu_resource_separates_identity_inventory_and_capacity(self) -> None:
        lock = load_gpu_resource_lock()
        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
            cuda = _FakeCuda()
            receipt = inspect_gpu_resource(
                cuda, model_alias=alias, resource_lock=lock
            )
            self.assertNotEqual(
                receipt.cuda_device_property_total_bytes,
                lock["physical_inventory_total_bytes"],
            )
            self.assertNotEqual(
                receipt.cuda_mem_get_info_total_bytes,
                receipt.cuda_device_property_total_bytes,
            )
            self.assertGreaterEqual(
                receipt.reusable_capacity_bytes,
                receipt.required_capacity_bytes,
            )
            self.assertEqual(receipt.safety_reserve_bytes, 2 * 1024**3)

    def test_gpu_resource_fails_wrong_identity_units_and_capacity(self) -> None:
        lock = load_gpu_resource_lock()

        cuda = _FakeCuda()
        cuda.properties.uuid = "GPU-00000000-0000-0000-0000-000000000000"
        with self.assertRaisesRegex(MethodContractError, "identity"):
            inspect_gpu_resource(cuda, model_alias="llama3-8b-inst", resource_lock=lock)

        for attribute, value, pattern in (
            ("count", 2, "count"),
            ("index", 1, "index"),
        ):
            cuda = _FakeCuda()
            setattr(cuda, attribute, value)
            with self.assertRaisesRegex(MethodContractError, pattern):
                inspect_gpu_resource(
                    cuda, model_alias="llama3-8b-inst", resource_lock=lock
                )

        wrong_units = deepcopy(lock)
        wrong_units["physical_inventory_total_bytes"] = 49140
        with self.assertRaisesRegex(MethodContractError, "identity"):
            inspect_gpu_resource(
                _FakeCuda(),
                model_alias="llama3-8b-inst",
                resource_lock=wrong_units,
            )

        qwen_required = (
            lock["conservative_peak_forecast_bytes"]["qwen2.5-7b-inst"]
            + lock["safety_reserve_bytes"]
        )
        cuda = _FakeCuda()
        cuda.capacity_total = qwen_required - 1
        cuda.reserved = 16 * 1024**3
        cuda.free = cuda.capacity_total - cuda.reserved
        with self.assertRaisesRegex(MethodContractError, "below"):
            inspect_gpu_resource(
                cuda, model_alias="qwen2.5-7b-inst", resource_lock=lock
            )

        cuda = _FakeCuda()
        cuda.free = qwen_required - cuda.reserved - 1
        with self.assertRaisesRegex(MethodContractError, "below"):
            inspect_gpu_resource(
                cuda, model_alias="qwen2.5-7b-inst", resource_lock=lock
            )

        for invalid in (0, float("inf"), True):
            cuda = _FakeCuda()
            cuda.free = invalid
            with self.assertRaises(MethodContractError):
                inspect_gpu_resource(
                    cuda, model_alias="llama3-8b-inst", resource_lock=lock
                )

    def test_joint_geometry_rejects_singleton_factor_decomposition(self) -> None:
        requests = _requests()
        motivation = tuple(SimpleNamespace(request_id=f"request-{i}") for i in range(10))
        fake = SimpleNamespace(
            requests=requests,
            motivation_requests=motivation,
            layers=(0, 1),
            _direct_z=SimpleNamespace(
                values=torch.zeros((4, 10)),
                request_ids=tuple(item.request_id for item in motivation),
            ),
            joint_direct_z_receipts=tuple({"column": i} for i in range(10)),
        )

        def batch(rank: int) -> ProposalBatch:
            proposals = []
            for layer in (0, 1):
                direction = FactorDirection(
                    layer,
                    f"layers.{layer}.weight",
                    torch.ones((3, rank)),
                    torch.ones((2, rank)),
                )
                proposals.append(
                    LayerProposal(layer, "state", direction.direction_id, direction)
                )
            return ProposalBatch(
                "state", tuple(proposals), (1.0, 1.0),
                ProposalSemantics.CURRENT_SAME_SNAPSHOT,
            )

        result = JointHardBatchEasyEditBackend.assert_genuine_joint_batch(
            fake, batch(10)
        )
        self.assertEqual(result["factor_ranks"], [10, 10])
        self.assertFalse(result["singleton_decomposition"])
        with self.assertRaisesRegex(MethodContractError, "rank"):
            JointHardBatchEasyEditBackend.assert_genuine_joint_batch(fake, batch(1))

    def test_joint_direct_z_uses_one_bridge_call_for_ten_requests(self) -> None:
        requests = _requests()
        motivation = tuple(SimpleNamespace(request_id=f"request-{i}") for i in range(10))
        values = torch.arange(40, dtype=torch.float32).reshape(4, 10)
        result = SimpleNamespace(values=values, source_state_id="entry")
        bridge = SimpleNamespace(load_or_compute_direct_z=mock.Mock(return_value=result))
        fake = SimpleNamespace(
            requests=requests,
            motivation_requests=motivation,
            _direct_z=None,
            _direct_z_tensor_sha256=None,
            bridge=bridge,
            model=object(),
            tokenizer=object(),
            hparams=object(),
            contexts=object(),
            runtime=SimpleNamespace(spec=SimpleNamespace(snapshot_name="toy")),
            direct_z_cache_root=Path("cache"),
            direct_z_cache_path=Path("cache/joint.pt"),
            current_state_id=lambda: "entry",
        )
        fake._compute_direct_z_panel = MethodType(
            EasyEditMemitBackend._compute_direct_z_panel, fake
        )
        observed = EasyEditMemitBackend.compute_joint_direct_z(fake, requests)
        self.assertIs(observed, result)
        bridge.load_or_compute_direct_z.assert_called_once()
        self.assertEqual(
            bridge.load_or_compute_direct_z.call_args.args[2], motivation
        )

    def test_joint_backend_rejects_duplicate_or_non_ten_panels_before_super(self) -> None:
        requests = list(_requests())
        with mock.patch.object(EasyEditMemitBackend, "__init__", return_value=None):
            with self.assertRaisesRegex(MethodContractError, "ten unique"):
                JointHardBatchEasyEditBackend(
                    requests=requests[:9], oracle_epsilon=1e-6
                )
            requests[-1] = ControllerRequest(
                requests[0].case_id,
                "{} differs",
                "different",
                "new",
                "old",
            )
            with self.assertRaisesRegex(MethodContractError, "ten unique"):
                JointHardBatchEasyEditBackend(
                    requests=requests, oracle_epsilon=1e-6
                )

    def test_fixed_budget_keeps_running_then_selects_earliest_exact_hit(self) -> None:
        metrics = EditInstrumentation("hard-b10-k20")
        backend = _HardToyBackend(metrics)
        result = run_hard_batch_arm(
            HardBatchArm.REFRESH_K20_T1_RESOLUTION,
            backend=backend,
            frozen_target=object(),
            denominators={0: 1.0, 1: 1.0},
            config=self.config,
            instrumentation=metrics,
            benchmark=lambda: _joint_reading(10 if backend.state >= 2 else 0),
            stagnation_epsilon=1e-8,
        )
        snapshot = metrics.finalize().to_dict()
        self.assertEqual(len(result.steps), 20)
        self.assertEqual(result.first_exact_hit_step, 2)
        self.assertEqual(result.rollout_terminal_state_id, "state-20")
        self.assertEqual(result.selected_terminal_state_id, "state-2")
        self.assertEqual(backend.state, 2)
        self.assertEqual(snapshot["counters"]["N_field"], 20)
        self.assertEqual(snapshot["counters"]["N_bw"], 20)
        self.assertFalse(snapshot["first_hit"])

    def test_two_cycle_budget_and_atomic_failure_rollback(self) -> None:
        metrics = EditInstrumentation("hard-b10-t2")
        backend = _HardToyBackend(metrics)
        result = run_hard_batch_arm(
            HardBatchArm.REFRESH_2XK10_T2_CORRECTION,
            backend=backend,
            frozen_target=object(),
            denominators={0: 1.0, 1: 1.0},
            config=self.config,
            instrumentation=metrics,
            benchmark=lambda: _joint_reading(0),
            stagnation_epsilon=1e-8,
        )
        self.assertEqual(result.field_build_count, 20)
        self.assertEqual(tuple(step.correction_cycle for step in result.steps[:11]), (0,) * 10 + (1,))
        metrics.finalize()

        failing_metrics = EditInstrumentation("hard-b10-failure")
        failing = _FailingHardToyBackend(failing_metrics)
        with self.assertRaisesRegex(RuntimeError, "joint transaction"):
            run_hard_batch_arm(
                HardBatchArm.REFRESH_K10_T1,
                backend=failing,
                frozen_target=object(),
                denominators={0: 1.0, 1: 1.0},
                config=self.config,
                instrumentation=failing_metrics,
                benchmark=lambda: _joint_reading(0),
                stagnation_epsilon=1e-8,
            )
        self.assertEqual(failing.state, 0)
        self.assertIsNone(failing.trial_state)

    def test_lock_seals_panel_common_policy_and_fails_closed(self) -> None:
        lock = load_hard_batch_lock()
        self.assertEqual(tuple(lock["selection"]["batch_case_ids"]), BATCH_CASE_IDS)
        self.assertEqual(lock["policy"]["edit_batch_size"], 10)
        self.assertEqual(lock["policy"]["K20"], 20)
        self.assertEqual(
            tuple(lock["policy"]["arms"]),
            tuple(arm.value for arm in HARD_BATCH_ARM_ORDER),
        )
        self.assertEqual(
            lock["benchmark"]["native_floor_margin_counts"],
            {"efficacy": 0, "generalization": 0, "locality": 0},
        )
        self.assertFalse(lock["resources"]["submission_authorized"])
        raw = json.loads(HARD_BATCH_LOCK_PATH.read_text(encoding="utf-8"))
        for key, value in (
            ("edit_batch_size", 1),
            ("online_early_stop", True),
            ("model_specific_policy", True),
        ):
            changed = json.loads(json.dumps(raw))
            changed["policy"][key] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "lock.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(MethodContractError):
                    load_hard_batch_lock(path)

    def test_dry_plan_and_direct_z_ten_receipt_distribution(self) -> None:
        lock = load_hard_batch_lock()
        for stage in ("p0", "p1"):
            plan = dry_plan(lock, stage)
            self.assertFalse(plan["submission_authorized"])
            self.assertEqual(plan["edit_batch_size"], 10)
            self.assertEqual(plan["pair_gpu"], 2)
            self.assertEqual(plan["server1_project_gpu_cap"], 3)
            self.assertEqual(
                plan["arms"], [arm.value for arm in HARD_BATCH_ARM_ORDER]
            )
            self.assertTrue(
                all(
                    f"session03-hard-b10-k20-{stage}-" in job["output_root"]
                    for job in plan["jobs"]
                )
            )
        identity = {
            "tensor_sha256": "a" * 64,
            "artifact_sha256": "b" * 64,
            "artifact_size": 123,
            "source_state_id": "c" * 64,
        }
        receipt_rows = tuple(
            {
                "column_index": index,
                "case_id": str(index),
                "request_id": f"request-{index}",
                **identity,
            }
            for index in range(10)
        )
        identities = {arm: dict(identity) for arm in HARD_BATCH_ARM_ORDER}
        receipts = {arm: receipt_rows for arm in HARD_BATCH_ARM_ORDER}
        counts = {
            arm: 10 if index == 0 else 0
            for index, arm in enumerate(HARD_BATCH_ARM_ORDER)
        }
        _assert_direct_z_identity(
            identity, receipt_rows, identities, receipts, counts
        )
        counts[HardBatchArm.REFRESH_K10_T1] = 1
        with self.assertRaisesRegex(MethodContractError, "accounting"):
            _assert_direct_z_identity(
                identity, receipt_rows, identities, receipts, counts
            )

    def test_action_source_has_no_heldout_or_model_alias_branch(self) -> None:
        source = inspect.getsource(run_hard_batch_arm)
        for forbidden in (
            "model_alias",
            "paraphrase",
            "neighborhood",
            "generation",
            "open_evaluation",
            "mark_first_hit",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
