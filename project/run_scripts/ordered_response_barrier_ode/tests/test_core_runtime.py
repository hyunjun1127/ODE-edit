from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from project.run_scripts.ordered_response_barrier_ode import artifacts
from project.run_scripts.ordered_response_barrier_ode.adapters import (
    CallbackMethodAdapter,
    FixedZArtifact,
    LayerBuild,
)
from project.run_scripts.ordered_response_barrier_ode.contracts import (
    ArmId,
    NumericalMethodBoundary,
    StaleStateBoundary,
    TechnicalBoundary,
    assert_full_fp32,
    build_frozen_anchor,
    canonical_arm_configs,
    normalized_potential,
    response_statistics,
)
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import (
    GroupedFP32Overlay,
    OverlayDelta,
    tensor_sha256,
)
from project.run_scripts.ordered_response_barrier_ode.integrator import (
    OrderedResponseIntegrator,
    terminal_flow_status,
)
from project.run_scripts.ordered_response_barrier_ode.preflight import (
    ORDER_ROOT,
    PreflightBoundary,
    STREAM_ROOT,
    cell_spec,
    validate_round0_common_gate,
)
from project.run_scripts.ordered_response_barrier_ode.runtime import (
    FamilyRuntime,
    _arm_dtype_scope,
    _classify_failure_status,
    _fd_absolute_tolerance,
    _install_model_forward_counter,
    _model_forward_count,
    _reconcile_arm_accounting,
    _response_identity,
    _stock_official_parity_receipt,
    _stock_repr_tools,
    _tensor_state_identity,
    _tensor_content_state_identity,
    _terminal_receipt,
)
from project.run_scripts.ordered_response_barrier_ode.semantic import (
    SemanticObservation,
    build_compute_z_semantic_inventory,
    observe_semantic_predicate,
)
from project.run_scripts.ordered_response_barrier_ode.telemetry import ArmTelemetry, build_step_record
from project.run_scripts.ordered_response_barrier_ode.terminal_jvp import (
    TerminalResponseObserver,
    seal_eager_attention,
)
from project.run_scripts.ordered_response_barrier_ode.tests.test_artifacts import (
    _round as raw_round_fixture,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ToyLinearModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(3, 2, bias=False) for _ in range(9)])
        self.config = SimpleNamespace(pad_token_id=0)


def _delta(layer: int, version: int, coefficient: float = 1.0) -> OverlayDelta:
    return OverlayDelta(
        weight_name=f"layers.{layer}.weight",
        layer=layer,
        left=torch.tensor([[1.0], [2.0]]),
        right=torch.tensor([[1.0], [-1.0], [0.5]]),
        coefficient=coefficient,
        built_state_version=version,
        build_identity=_sha(f"build-{layer}-{version}".encode()),
    )


class ArmAndFP32Tests(unittest.TestCase):
    def test_exact_arm_order_and_clock(self) -> None:
        values = canonical_arm_configs(sweeps=4)
        self.assertEqual(tuple(value.arm.value for value in values), ("O", "QCL", "NQFIX", "ORBFH", "JAC", "ORBHit"))
        self.assertEqual(tuple(value.step_size for value in values[1:]), (0.25,) * 5)
        self.assertTrue(all(not value.first_hit_controls_dynamics for value in values))

    def test_full_fp32_gate_is_active(self) -> None:
        model = ToyLinearModel().float()
        old_cuda = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            receipt = assert_full_fp32(model)
            self.assertFalse(receipt["autocast_enabled"])
            model.layers[0] = model.layers[0].double()
            with self.assertRaises(TechnicalBoundary):
                assert_full_fp32(model)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_cuda
            torch.backends.cudnn.allow_tf32 = old_cudnn

    def test_terminal_graph_uses_shared_undetached_repr_tools_contract(self) -> None:
        model = ToyLinearModel().float()

        class ReprTools:
            calls: list[dict[str, object]] = []

            @classmethod
            def get_reprs_at_word_tokens(cls, **kwargs: object) -> torch.Tensor:
                cls.calls.append(dict(kwargs))
                batch = len(kwargs["words"])  # type: ignore[arg-type]
                value = model.layers[0].weight.sum(dim=1)
                return value.unsqueeze(0).expand(batch, -1)

        class StockModule:
            class ComputeZBinding:
                __globals__ = {"repr_tools": ReprTools}

                def __call__(self, *_args: object, **_kwargs: object) -> None:
                    return None

            compute_z = ComputeZBinding()

            @staticmethod
            def get_module_input_output_at_words(*_args: object, **_kwargs: object) -> torch.Tensor:
                raise AssertionError("family-specific detached wrapper must not be called")

        family = object.__new__(FamilyRuntime)
        family.model = model
        family.tokenizer = object()
        family.module = StockModule()
        family._repr_tools = _stock_repr_tools(family.module)
        family.hparams = SimpleNamespace(
            fact_token="subject_last",
            layer_module_tmp="layers.{}.weight",
        )
        family.requests = (
            {"prompt": "{} was born in", "subject": "Ada"},
            {"prompt": "{} works in", "subject": "Grace"},
        )

        terminal = family.terminal_graph()
        self.assertEqual(tuple(terminal.shape), (2, 2))
        self.assertTrue(terminal.requires_grad)
        terminal.sum().backward()
        self.assertIsNotNone(model.layers[0].weight.grad)
        self.assertEqual(len(ReprTools.calls), 1)
        self.assertEqual(ReprTools.calls[0]["track"], "out")
        self.assertEqual(ReprTools.calls[0]["subtoken"], "last")


class OverlayTests(unittest.TestCase):
    def test_tensor_hash_accepts_transposed_singleton_final_dimension(self) -> None:
        flat = torch.arange(32, dtype=torch.float32)
        transposed = flat.reshape(1, -1).T
        self.assertEqual(transposed.stride(), (1, 32))
        self.assertEqual(tensor_sha256(transposed), tensor_sha256(flat.reshape(-1, 1)))

    def test_canonical_rectangular_factor_and_shadow(self) -> None:
        model = ToyLinearModel().float()
        entry = model.layers[4].weight.detach().clone()
        value = _delta(4, 0, 0.25)
        with GroupedFP32Overlay(model, {4: "layers.4.weight"}) as overlay:
            overlay.append(value)
            shadow = overlay.materialize_shadow()["layers.4.weight"]
            expected = entry + 0.25 * value.left @ value.right.T
            self.assertTrue(torch.equal(shadow, expected))
            self.assertTrue(torch.equal(model.layers[4].weight, entry))

    def test_terminal_suspend_rebases_version_only_after_byte_restore(self) -> None:
        model = ToyLinearModel().float()
        entry = model.layers[4].weight.detach().clone()
        with GroupedFP32Overlay(model, {4: "layers.4.weight"}) as overlay:
            overlay.append(_delta(4, 0))
            shadow = overlay.materialize_shadow()["layers.4.weight"]
            with overlay.suspend(authoritative=True):
                with torch.no_grad():
                    model.layers[4].weight.copy_(shadow)
                    model.layers[4].weight.copy_(entry)
            overlay.assert_w0_unchanged(full_bytes=True)
            self.assertEqual(overlay.physical_write_count, 1)

    def test_terminal_suspend_rejects_leak(self) -> None:
        model = ToyLinearModel().float()
        with self.assertRaises(TechnicalBoundary):
            with GroupedFP32Overlay(model, {4: "layers.4.weight"}) as overlay:
                with overlay.suspend(authoritative=True):
                    with torch.no_grad():
                        model.layers[4].weight.add_(1.0)

    def test_jac_exact_sweep_token(self) -> None:
        model = ToyLinearModel().float()
        with GroupedFP32Overlay(model, {4: "layers.4.weight", 5: "layers.5.weight"}) as overlay:
            token = overlay.seal_sweep_entry(0)
            overlay.append(_delta(4, 0), sweep_token=token)
            overlay.append(_delta(5, 0), sweep_token=token)
            overlay.close_sweep(token)
            with self.assertRaises(StaleStateBoundary):
                overlay.append(_delta(4, 0), sweep_token=token)


class TinyTokenizer:
    bos_token_id = None
    unk_token_id = None
    pad_token_id = 0
    eos_token_id = 0
    padding_side = "right"

    def encode(self, text, return_tensors=None, add_special_tokens=False):
        del add_special_tokens
        values = [1 + (ord(char) % 5) for char in text]
        tensor = torch.tensor([values], dtype=torch.long)
        return tensor if return_tensors == "pt" else values

    def decode(self, values):
        if isinstance(values, torch.Tensor):
            values = values.tolist()
        return "".join(chr(96 + int(value)) for value in values)

    def __call__(self, texts, return_tensors=None, padding=False):
        del return_tensors, padding
        rows = [self.encode(text) for text in texts]
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class TinySemanticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.config = SimpleNamespace(pad_token_id=0)

    def forward(self, *, input_ids, attention_mask, position_ids, use_cache=False):
        del attention_mask, position_ids, use_cache
        logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32, device=input_ids.device)
        # Every encoded target ID is strict at its aligned token position.
        logits.scatter_(2, input_ids.unsqueeze(-1), 3.0)
        logits = logits + self.anchor
        return SimpleNamespace(logits=logits)


class SemanticTests(unittest.TestCase):
    def test_multitoken_context_inventory_and_batched_observation(self) -> None:
        tokenizer = TinyTokenizer()
        requests = [{"case_id": 1, "prompt": "{} works", "subject": "A", "target_new": " BC"}]
        inventory = build_compute_z_semantic_inventory(
            tokenizer=tokenizer,
            requests=requests,
            context_templates=[["{}"], ["prefix {}"]],
            request_order_sha256=_sha(b"order"),
        )
        self.assertGreater(inventory.event_count, inventory.request_count)
        observation = observe_semantic_predicate(TinySemanticModel(), inventory, microbatch_size=2)
        self.assertEqual(observation.event_count, inventory.event_count)


class EntryAlreadyHitTests(unittest.TestCase):
    def test_w0_noop_is_evaluated_and_derived(self) -> None:
        model = ToyLinearModel().float()
        names = {layer: f"layers.{layer}.weight" for layer in (4, 5, 6, 7, 8)}
        target = torch.zeros((2, 1), dtype=torch.float32)
        fixed = FixedZArtifact(target, tensor_sha256(target), _sha(b"order"), _sha(b"contexts"))
        version = {"value": 0}
        adapter = CallbackMethodAdapter(
            family="MEMIT",
            layers=(4, 5, 6, 7, 8),
            weight_names_by_layer=names,
            compute_fixed_z=lambda: fixed,
            capture_terminal=lambda: target.clone(),
            build_official_form_layer=lambda **_: (_ for _ in ()).throw(AssertionError("no build")),
            run_official_endpoint=lambda **_: {},
            finalize_terminal_state=lambda **_: {},
            evaluate_current_endpoint=lambda **kwargs: {
                "status": kwargs["status"],
                "physical_write_count": 0,
                "history_append_count": 0,
            },
            capture_method_state_identity=lambda: _sha(b"state"),
        )
        adapter.use_fixed_z(fixed)
        overlay = GroupedFP32Overlay(model, names)
        integrator = OrderedResponseIntegrator(
            adapter=adapter,
            overlay=overlay,
            jvp=object(),
            observe_semantic=lambda: SemanticObservation(True, 1, 1, (True,), 1, 3.0, 3.0, 1.0, 0),
        )
        result = integrator.run_dynamic(canonical_arm_configs()[3], fixed)
        self.assertEqual(result.status, "ENTRY_ALREADY_HIT")
        self.assertIsNotNone(result.derived_endpoint)
        self.assertEqual(result.endpoint["physical_write_count"], 0)

    def test_official_w0_noop_evaluates_without_factor_or_write(self) -> None:
        model = ToyLinearModel().float()
        names = {layer: f"layers.{layer}.weight" for layer in (4, 5, 6, 7, 8)}
        target = torch.zeros((2, 1), dtype=torch.float32)
        fixed = FixedZArtifact(target, tensor_sha256(target), _sha(b"order"), _sha(b"contexts"))
        calls = {"official": 0, "evaluate": 0}

        def run_official(**_):
            calls["official"] += 1
            raise AssertionError("entry-hit Official writer must not run")

        def evaluate(**kwargs):
            calls["evaluate"] += 1
            return {
                "status": kwargs["status"],
                "evaluation": {"rows": 1},
                "physical_write_count": 0,
                "history_append_count": 0,
            }

        adapter = CallbackMethodAdapter(
            family="MEMIT",
            layers=(4, 5, 6, 7, 8),
            weight_names_by_layer=names,
            compute_fixed_z=lambda: fixed,
            capture_terminal=lambda: target.clone(),
            build_official_form_layer=lambda **_: (_ for _ in ()).throw(AssertionError("no build")),
            run_official_endpoint=run_official,
            finalize_terminal_state=lambda **_: {},
            evaluate_current_endpoint=evaluate,
            capture_method_state_identity=lambda: _sha(b"state"),
        )
        adapter.use_fixed_z(fixed)
        result = OrderedResponseIntegrator(
            adapter=adapter,
            overlay=GroupedFP32Overlay(model, names),
            jvp=object(),
            observe_semantic=lambda: SemanticObservation(True, 1, 1, (True,), 1, 3.0, 3.0, 1.0, 0),
        ).run_official(canonical_arm_configs()[0], fixed)
        self.assertEqual(result.status, "ENTRY_ALREADY_HIT")
        self.assertEqual(calls, {"official": 0, "evaluate": 1})
        self.assertEqual(result.telemetry["factor_build_count"], 0)
        self.assertEqual(result.telemetry["physical_write_count"], 0)
        self.assertEqual(result.telemetry["history_append_count"], 0)


class MechanismTelemetryTests(unittest.TestCase):
    def test_request_g_r_and_qcl_command_geometry_are_preserved(self) -> None:
        target = torch.tensor([[10.0, 0.0]], dtype=torch.float32)
        entry = torch.zeros_like(target)
        anchor = build_frozen_anchor(
            target=target,
            entry_terminal=entry,
            target_sha256=_sha(b"target"),
            request_order_sha256=_sha(b"order"),
        )
        statistics = response_statistics(
            anchor=anchor,
            terminal=entry,
            response=torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        )
        self.assertAlmostEqual(statistics.per_request_g[0], 0.2)
        self.assertAlmostEqual(statistics.per_request_r[0], 0.04)
        self.assertIsNone(statistics.per_request_g[1])
        self.assertIsNone(statistics.per_request_r[1])
        potential, _ = normalized_potential(anchor, entry)
        record = build_step_record(
            arm=ArmId.QUOTA_CLOSED_LOOP,
            sweep=0,
            layer=4,
            visit_ordinal=0,
            built_state_version=0,
            resulting_state_version=1,
            residual_denominator=5,
            build_identity=_sha(b"build"),
            residual_sha256=_sha(b"residual"),
            keys_sha256=_sha(b"keys"),
            solver_identity=_sha(b"solver"),
            factor_rank=1,
            u=1.0,
            h=0.25,
            entry_potential=potential,
            potential_before=potential,
            potential_after=0.405,
            target=target,
            command_residual=torch.tensor([[2.0, 0.0]], dtype=torch.float32),
            anchor_scales=anchor.scales,
            anchor_active=anchor.active,
            terminal_before=entry,
            terminal_after=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            direction_frobenius=2.0,
            response=None,
            statistics=None,
            semantic=SemanticObservation(False, 0, 2, (False, True), 1, 0.0, 0.0, 1.0, 0),
            virtual_state_identity_sha256=_sha(b"virtual"),
        )
        # Locked D_R0 uses alpha times the full current residual: (2.5-1)^2/10^2.
        self.assertAlmostEqual(record.per_request_D_R0[0], 0.0225)
        self.assertIsNone(record.per_request_D_R0[1])
        self.assertAlmostEqual(record.per_request_q_res[0], 0.9)
        # The actual QCL writer command remains alpha*(R/5)=0.5.
        self.assertAlmostEqual(record.per_request_writer_command_R0_squared[0], 0.0025)
        self.assertAlmostEqual(record.command_norm_mean, 1.25)
        self.assertAlmostEqual(record.writer_command_norm_mean, 0.25)
        self.assertLess(record.per_request_actual_potential_change[0], 0.0)
        self.assertFalse(record.per_request_actual_potential_worsened[0])
        self.assertEqual(record.action_geometry_status, "FROBENIUS_ONLY_CREG_NOT_BOUND")
        self.assertEqual(record.native_creg_action_status, "TELEMETRY_WITHHELD")
        self.assertAlmostEqual(record.applied_update_frobenius_squared, 0.25)
        self.assertAlmostEqual(record.resolution_stable_path_increment_frobenius_squared, 1.0)

    def test_zero_command_cross_response_and_orthogonal_energy_are_observed(self) -> None:
        target = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        terminal = torch.zeros_like(target)
        anchor = build_frozen_anchor(
            target=target,
            entry_terminal=terminal,
            target_sha256=tensor_sha256(target),
            request_order_sha256=_sha(b"order"),
        )
        response = torch.tensor([[0.0], [2.0]], dtype=torch.float32)
        statistics = response_statistics(anchor=anchor, terminal=terminal, response=response)
        self.assertEqual(statistics.u, 0.0)
        semantic = SemanticObservation(
            False, 1, 1, (False,), 0, 0.0, 0.0, 0.0, 0
        )
        record = build_step_record(
            arm=ArmId.ORDERED_RESPONSE_FIXED_HORIZON,
            sweep=0,
            layer=4,
            visit_ordinal=0,
            built_state_version=0,
            resulting_state_version=0,
            residual_denominator=1,
            build_identity=_sha(b"build"),
            residual_sha256=_sha(b"residual"),
            keys_sha256=_sha(b"keys"),
            solver_identity=_sha(b"solver"),
            factor_rank=1,
            u=0.0,
            h=0.25,
            entry_potential=0.5,
            potential_before=0.5,
            potential_after=0.5,
            target=target,
            command_residual=target,
            anchor_scales=anchor.scales,
            anchor_active=anchor.active,
            terminal_before=terminal,
            terminal_after=terminal,
            direction_frobenius=3.0,
            response=response,
            statistics=statistics,
            semantic=semantic,
            virtual_state_identity_sha256=_sha(b"virtual"),
        )
        self.assertEqual(record.per_request_D_R0, (None,))
        self.assertEqual(record.per_request_D_model_R0, (None,))
        self.assertEqual(record.per_request_response_orthogonal_R0_squared, (4.0,))
        self.assertEqual(record.per_request_zero_command_cross_response_R0_squared, (None,))
        self.assertEqual(record.per_request_metric_status, ("SKIPPED_ZERO_COMMAND",))
        self.assertEqual(record.per_request_actual_potential_change, (0.0,))
        self.assertEqual(record.per_request_actual_potential_worsened, (False,))

    def test_request_zero_command_preserves_shared_batch_cross_response(self) -> None:
        target = torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float32)
        entry = torch.zeros_like(target)
        anchor = build_frozen_anchor(
            target=target,
            entry_terminal=entry,
            target_sha256=tensor_sha256(target),
            request_order_sha256=_sha(b"order"),
        )
        terminal_before = torch.tensor([[0.0, 2.0], [0.0, 0.0]], dtype=torch.float32)
        terminal_after = torch.tensor([[0.25, 2.5], [0.0, 0.0]], dtype=torch.float32)
        response = torch.tensor([[0.5, 0.5], [0.0, 0.0]], dtype=torch.float32)
        statistics = response_statistics(
            anchor=anchor, terminal=terminal_before, response=response
        )
        self.assertEqual(statistics.u, 1.0)
        record = build_step_record(
            arm=ArmId.ORDERED_RESPONSE_FIXED_HORIZON,
            sweep=0,
            layer=4,
            visit_ordinal=0,
            built_state_version=0,
            resulting_state_version=1,
            residual_denominator=1,
            build_identity=_sha(b"build"),
            residual_sha256=_sha(b"residual"),
            keys_sha256=_sha(b"keys"),
            solver_identity=_sha(b"solver"),
            factor_rank=1,
            u=1.0,
            h=0.25,
            entry_potential=0.5,
            potential_before=0.25,
            potential_after=0.265625,
            target=target,
            command_residual=target - terminal_before,
            anchor_scales=anchor.scales,
            anchor_active=anchor.active,
            terminal_before=terminal_before,
            terminal_after=terminal_after,
            direction_frobenius=1.0,
            response=response,
            statistics=statistics,
            semantic=SemanticObservation(False, 2, 2, (False, False), 0, 0.0, 0.0, 0.0, 0),
            virtual_state_identity_sha256=_sha(b"virtual"),
        )
        self.assertIsNone(record.per_request_zero_command_cross_response_R0_squared[0])
        self.assertAlmostEqual(record.per_request_zero_command_cross_response_R0_squared[1], 0.0625)
        self.assertEqual(
            record.per_request_metric_status[1], "ZERO_COMMAND_CROSS_RESPONSE_OBSERVED"
        )
        self.assertAlmostEqual(record.per_request_D_R0[1], 0.0625)
        self.assertTrue(record.per_request_actual_potential_worsened[1])

    def test_arm_payload_reconciles_step_path_and_terminal_net_frobenius(self) -> None:
        target = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        terminal = torch.zeros_like(target)
        anchor = build_frozen_anchor(
            target=target,
            entry_terminal=terminal,
            target_sha256=tensor_sha256(target),
            request_order_sha256=_sha(b"order"),
        )
        semantic = SemanticObservation(False, 1, 1, (False,), 0, 0.0, 0.0, 0.0, 0)
        record = build_step_record(
            arm=ArmId.NO_QUOTA_FIXED,
            sweep=0,
            layer=4,
            visit_ordinal=0,
            built_state_version=0,
            resulting_state_version=1,
            residual_denominator=1,
            build_identity=_sha(b"build"),
            residual_sha256=_sha(b"residual"),
            keys_sha256=_sha(b"keys"),
            solver_identity=_sha(b"solver"),
            factor_rank=1,
            u=1.0,
            h=0.25,
            entry_potential=0.5,
            potential_before=0.5,
            potential_after=0.28125,
            target=target,
            command_residual=target,
            anchor_scales=anchor.scales,
            anchor_active=anchor.active,
            terminal_before=terminal,
            terminal_after=torch.tensor([[0.25], [0.0]], dtype=torch.float32),
            direction_frobenius=4.0,
            response=None,
            statistics=None,
            semantic=semantic,
            virtual_state_identity_sha256=_sha(b"virtual"),
        )
        telemetry = ArmTelemetry(
            arm=ArmId.NO_QUOTA_FIXED.value,
            steps=[record],
            terminal_net_frobenius=1.5,
            terminal_net_frobenius_squared=2.25,
        )
        payload = telemetry.payload()
        self.assertEqual(payload["sum_step_action_frobenius_squared"], 1.0)
        self.assertEqual(payload["physical_euler_path_frobenius_length"], 1.0)
        self.assertEqual(payload["resolution_stable_path_frobenius"], 2.0)
        self.assertEqual(payload["resolution_stable_path_frobenius_squared"], 4.0)
        self.assertEqual(payload["terminal_net_frobenius"], 1.5)
        self.assertEqual(payload["terminal_net_frobenius_squared"], 2.25)
        self.assertAlmostEqual(record.applied_update_frobenius_squared, 1.0)
        self.assertAlmostEqual(record.resolution_stable_path_increment_frobenius_squared, 4.0)

    def test_transient_hit_is_not_reported_as_terminal_hit(self) -> None:
        self.assertEqual(
            terminal_flow_status(
                stalled=False,
                terminal_hit=False,
                first_hit_latched=True,
            ),
            "HORIZON_TRANSIENT_HIT_TERMINAL_MISS",
        )
        self.assertEqual(
            terminal_flow_status(
                stalled=True,
                terminal_hit=False,
                first_hit_latched=True,
            ),
            "FLOW_STALLED_AFTER_TRANSIENT_HIT",
        )


class AttentionBackendTests(unittest.TestCase):
    def test_eager_backend_is_source_and_config_sealed(self) -> None:
        model = ToyLinearModel().float()
        model.config._attn_implementation = "eager"
        model.config._attn_implementation_internal = "eager"
        receipt = seal_eager_attention(model)
        self.assertEqual(receipt["observed_backend"], "eager")
        self.assertEqual(len(receipt["config_identity_sha256"]), 64)
        self.assertEqual(len(receipt["model_class_source_sha256"]), 64)
        model.config._attn_implementation_internal = "sdpa"
        with self.assertRaises(TechnicalBoundary):
            seal_eager_attention(model)


class RuntimeReceiptTests(unittest.TestCase):
    def test_dtype_scope_exposes_stock_memit_ephemeral_fp64_exception(self) -> None:
        official_memit = _arm_dtype_scope(family="MEMIT", arm=ArmId.OFFICIAL)
        self.assertTrue(official_memit["model_storage_full_fp32"])
        self.assertTrue(official_memit["orbode_dynamic_path_full_fp32"])
        self.assertEqual(
            official_memit["official_native_ephemeral_solve_dtype"], "torch.float64"
        )
        self.assertFalse(official_memit["unqualified_all_algorithm_full_fp32"])
        dynamic_memit = _arm_dtype_scope(
            family="MEMIT", arm=ArmId.ORDERED_RESPONSE_FIXED_HORIZON
        )
        self.assertEqual(
            dynamic_memit["official_native_ephemeral_solve_dtype"], "torch.float32"
        )
        self.assertTrue(dynamic_memit["unqualified_all_algorithm_full_fp32"])

    def test_dynamic_cross_layer_accounting_reconciles_and_rejects_solve_drift(self) -> None:
        state = _sha(b"method-state")
        payload = {
            "status": "HORIZON_SEMANTIC_MISS",
            "telemetry": {
                "factor_build_count": 2,
                "nonzero_action_visit_count": 2,
                "jvp_call_count": 2,
            },
            "adapter_ledger": {
                "key_capture_count": 2,
                "layer_factorization_count": 2,
                "solve_count": 2,
                "official_endpoint_count": 0,
                "terminal_finalize_count": 1,
            },
            "endpoint": {
                "physical_write_count": 1,
                "history_append_count": 1,
                "terminal_history_key_capture_count": 5,
            },
            "overlay": {
                "physical_terminal_write_count": 1,
                "shadow_materialization_count": 1,
                "factor_count": 2,
            },
            "derived_endpoint": None,
            "jvp_ledger": {
                "model_forward_invocation_count": 2,
                "jvp_call_count": 2,
            },
            "model_forward_invocation_count": 10,
        }
        receipt = _reconcile_arm_accounting(
            payload,
            family="AlphaEdit",
            arm=ArmId.ORDERED_RESPONSE_FIXED_HORIZON,
            endpoint_evaluator_count=1,
            method_state_entry_sha256=state,
            method_state_after_sha256=state,
        )
        self.assertEqual(receipt["status"], "EXACT_CROSS_LAYER_RECONCILIATION_PASS")
        self.assertEqual(receipt["authoritative_terminal_commit_count"], 1)
        bad = dict(payload)
        bad["adapter_ledger"] = dict(payload["adapter_ledger"], solve_count=1)
        with self.assertRaisesRegex(TechnicalBoundary, "build/key/solve/action"):
            _reconcile_arm_accounting(
                bad,
                family="AlphaEdit",
                arm=ArmId.ORDERED_RESPONSE_FIXED_HORIZON,
                endpoint_evaluator_count=1,
                method_state_entry_sha256=state,
                method_state_after_sha256=state,
            )

    def test_content_state_identity_detects_content_and_can_ignore_restore_version(self) -> None:
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
        entry_content = _tensor_content_state_identity(
            "cache", {"value": tensor}, include_version=False
        )
        entry_versioned = _tensor_content_state_identity(
            "cache", {"value": tensor}, include_version=True
        )
        with torch.no_grad():
            tensor.add_(1.0)
        changed_content = _tensor_content_state_identity(
            "cache", {"value": tensor}, include_version=False
        )
        self.assertNotEqual(entry_content, changed_content)
        with torch.no_grad():
            tensor.copy_(torch.tensor([1.0, 2.0], dtype=torch.float32))
        self.assertEqual(
            entry_content,
            _tensor_content_state_identity("cache", {"value": tensor}, include_version=False),
        )
        self.assertNotEqual(
            entry_versioned,
            _tensor_content_state_identity("cache", {"value": tensor}, include_version=True),
        )

    def test_active_element_sign_contradiction_is_hard_failure(self) -> None:
        with self.assertRaisesRegex(TechnicalBoundary, "response identity failed"):
            _response_identity(
                torch.tensor([1000.0, 1.0], dtype=torch.float32),
                torch.tensor([1000.0, -100.0], dtype=torch.float32),
                absolute_tolerance=99.0,
            )

    def test_nonfinite_never_promotes_without_full_oracle_exclusion(self) -> None:
        status, detected = _classify_failure_status(
            NumericalMethodBoundary("non-finite JVP response")
        )
        self.assertTrue(detected)
        self.assertEqual(
            status, "TECHNICAL_INVALID_NONFINITE_ORACLE_EXCLUSION_INCOMPLETE"
        )
        finite_status, finite_detected = _classify_failure_status(
            NumericalMethodBoundary("finite locked numerical boundary")
        )
        self.assertFalse(finite_detected)
        self.assertEqual(finite_status, "NUMERICAL_METHOD_BOUNDARY")

    def test_top_level_model_forward_counter_is_actual_hook_count(self) -> None:
        model = ToyLinearModel().float()
        handle = _install_model_forward_counter(model)
        try:
            model.layers[0](torch.zeros((1, 3), dtype=torch.float32))
            self.assertEqual(_model_forward_count(model), 0)
            # A top-level call is what the production counter intentionally
            # counts; internal module calls are not double-counted.
            with self.assertRaises(NotImplementedError):
                model(torch.zeros((1, 3), dtype=torch.float32))
            self.assertEqual(_model_forward_count(model), 1)
        finally:
            handle.remove()

    def test_tensor_state_identity_hashes_each_tensor_once(self) -> None:
        tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
        expected = hashlib.sha256()
        expected.update(b"prefix\0")
        expected.update(b"only\0")
        expected.update(tensor_sha256(tensor).encode("ascii"))
        expected.update(b"\0")
        self.assertEqual(_tensor_state_identity("prefix", {"only": tensor}), expected.hexdigest())

    def test_runtime_receipt_passes_round0_to_remaining_common_gate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for cell_id in range(4):
                spec = cell_spec(cell_id)
                task_root = root / f"task-{cell_id}"
                task_root.mkdir(parents=True)
                round_publication = artifacts.reduce_round_payload(raw_round_fixture())
                round_path = task_root / "round-00-result.json"
                round_bytes = (
                    json.dumps(round_publication, sort_keys=True) + "\n"
                ).encode("utf-8")
                round_path.write_bytes(round_bytes)
                round_path.chmod(0o600)
                round_file_sha256 = hashlib.sha256(round_bytes).hexdigest()
                native_memit_exception = spec.writer_family == "MEMIT"
                result = {
                    "schema": "orbode.server1.cell-result.v1",
                    "status": "TERMINAL_VALID",
                    "cell_id": cell_id,
                    "model_alias": spec.model_alias,
                    "writer_family": spec.writer_family,
                    "wave": "round0",
                    "rounds": [0],
                    "request_count": 100,
                    "primary_arms": ["O", "QCL", "NQFIX", "ORBFH", "JAC"],
                    "primary_endpoint_count": 500,
                    "scientific_attempted_count": 500,
                    "terminal_valid_count": 500,
                    "technical_failure_count": 0,
                    "fixed_z_compute_count": 100,
                    "fixed_z_recompute_count": 0,
                    "source_head": "h",
                    "source_tree": "t",
                    "stream_root": STREAM_ROOT,
                    "order_root": ORDER_ROOT,
                    "full_fp32": True,
                    "full_fp32_claim_scope": "MODEL_STORAGE_AND_ORBODE_DYNAMIC_PATH",
                    "official_native_memit_ephemeral_fp64_solve_exception": native_memit_exception,
                    "all_algorithm_solves_full_fp32": not native_memit_exception,
                    "unqualified_full_fp32_claim": not native_memit_exception,
                    "fast_runtime_preamble_count": 1,
                    "fast_runtime_preamble_status": "PASS_THIS_WAVE",
                    "round_results": [round_publication],
                    "round_publication_receipts": [
                        {
                            "round_index": 0,
                            "relative_path": "round-00-result.json",
                            "file_sha256": round_file_sha256,
                            "identity_sha256": round_publication["identity_sha256"],
                        }
                    ],
                    "round_publication_count": 1,
                    "round_publication_identity_root": artifacts.canonical_hash(
                        [round_publication["identity_sha256"]]
                    ),
                    "round_publication_file_sha256_root": artifacts.canonical_hash(
                        [round_file_sha256]
                    ),
                    "literal_prompt_target_token_prediction_publication_count": 0,
                }
                result_path = task_root / "result.json"
                result_bytes = (json.dumps(result, sort_keys=True) + "\n").encode("utf-8")
                result_path.write_bytes(result_bytes)
                result_path.chmod(0o600)
                receipt = _terminal_receipt(
                    cell_id=cell_id,
                    wave="round0",
                    round_indices=(0,),
                    model_alias=spec.model_alias,
                    writer_family=spec.writer_family,
                    run_token=f"token-{cell_id}",
                    request_count=100,
                    primary_endpoint_count=500,
                    entry_already_hit_count=0,
                    fixed_z_compute_count=100,
                    source_head="h",
                    source_tree="t",
                    result_sha256=hashlib.sha256(result_bytes).hexdigest(),
                    round_publication_identities=[
                        str(round_publication["identity_sha256"])
                    ],
                    round_publication_file_sha256=[round_file_sha256],
                )
                path = task_root / "terminal-receipt.json"
                path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
                path.chmod(0o600)
            gate = validate_round0_common_gate(root, source_head="h", source_tree="t")
            self.assertEqual(gate["status"], "ROUND0_COMMON_INTEGRITY_PASS")
            self.assertEqual(gate["endpoint_count"], 2000)
            tampered = root / "task-2" / "result.json"
            tampered.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(PreflightBoundary):
                validate_round0_common_gate(root, source_head="h", source_tree="t")


class RuntimePreambleMathTests(unittest.TestCase):
    def test_stock_official_parity_is_wrapper_direct_not_dynamic_qcl(self) -> None:
        endpoint = {
            "selected_weight_endpoint_sha256": _sha(b"weights"),
            "terminal_activation_sha256": _sha(b"activation"),
            "semantic_observation": {"strict": [True], "logit": 1.25},
            "evaluation": {"rewrite": {"target_new_nll": 0.25}},
        }
        receipt = _stock_official_parity_receipt(endpoint, dict(endpoint))
        self.assertTrue(receipt["exact_parity"])
        self.assertFalse(receipt["dynamic_qcl_one_pass_used_as_stock_oracle"])
        changed = dict(endpoint)
        changed["terminal_activation_sha256"] = _sha(b"different")
        mismatch = _stock_official_parity_receipt(endpoint, changed)
        self.assertFalse(mismatch["exact_parity"])
        self.assertFalse(mismatch["terminal_activation_sha256_equal"])
        missing = dict(endpoint)
        missing.pop("terminal_activation_sha256")
        absent = _stock_official_parity_receipt(missing, missing)
        self.assertFalse(absent["exact_parity"])
        self.assertFalse(absent["terminal_activation_sha256_present"])

    def test_fd_receipt_has_locked_error_cosine_sign_and_l2_fields(self) -> None:
        model = ToyLinearModel().float()
        input_value = torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float32)
        build = LayerBuild(
            layer=4,
            weight_name="layers.4.weight",
            left=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            right=torch.tensor([[1.0], [-1.0], [0.5]], dtype=torch.float32),
            built_state_version=0,
            residual_denominator=1,
            residual_sha256=_sha(b"residual"),
            keys_sha256=_sha(b"keys"),
            solver_identity=_sha(b"solver"),
        )
        with GroupedFP32Overlay(model, {4: "layers.4.weight"}) as overlay:
            observer = TerminalResponseObserver(
                model=model,
                overlay=overlay,
                capture_terminal_graph=lambda: model.layers[4](input_value).T,
            )
            reference = observer.observe(build, expected_state_version=0)
            epsilon = 2.0 ** -8
            receipt = observer.audit_central_difference(
                build,
                expected_state_version=0,
                epsilon=epsilon,
                absolute_tolerance=_fd_absolute_tolerance(reference.terminal, epsilon),
                relative_tolerance=2.0 ** -5,
            )
        self.assertTrue(receipt.allclose)
        self.assertGreater(receipt.cosine_similarity, 0.0)
        self.assertGreaterEqual(receipt.sign_agreement_fraction, 0.0)
        self.assertLessEqual(receipt.sign_agreement_fraction, 1.0)
        self.assertGreaterEqual(receipt.relative_l2_error, 0.0)


if __name__ == "__main__":
    unittest.main()
