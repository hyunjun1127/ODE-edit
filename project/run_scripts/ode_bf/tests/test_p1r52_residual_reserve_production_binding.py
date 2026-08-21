from __future__ import annotations

from dataclasses import replace
import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_bf.contracts import (
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1r52_residual_reserve_nominal_shadow import (
    ShadowWeightStateIdentity,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_pc_inventory import (
    SealedPrevalidatedCovariance,
)
from project.run_scripts.ode_bf import (
    p1r52_residual_reserve_production_binding as binding_module,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_production_binding import (
    PREFIX_OBSERVATION_STATUS,
    build_residual_reserve_production_binding,
)
from project.run_scripts.ode_bf.scalable_batched_model import ScalableCapturePlan
from project.run_scripts.ode_bf.scalable_batched_runtime import (
    StreamingBatchPlan,
    StreamingPhysicalCapture,
)


LAYERS = (4, 5, 6, 7, 8)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(5204)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(4, 3, bias=False) for _ in range(9)]
        )
        with torch.no_grad():
            for layer in LAYERS:
                value = torch.randn((3, 4), generator=generator)
                self.layers[layer].weight = torch.nn.Parameter(
                    value.to(dtype=torch.float32), requires_grad=False
                )


def _capture_plan() -> ScalableCapturePlan:
    streaming_identity = canonical_hash({"stream": "phase-a-production-test"})
    streaming = StreamingBatchPlan(
        request_count=2,
        microbatch_size=2,
        request_identity_sha256=(canonical_hash([0]), canonical_hash([1])),
        request_lengths=(3, 4),
        bucket_order=(0, 1),
        inverse_order=(0, 1),
        batches=(),
        identity_sha256=streaming_identity,
    )
    return ScalableCapturePlan(
        streaming_plan=streaming,
        batches=(),
        request_order_sha256=canonical_hash(["r0", "r1"]),
        context_sha256=canonical_hash(["context"]),
        identity_sha256=canonical_hash(["capture-plan"]),
    )


def _weight_state(binding) -> tuple[ShadowWeightStateIdentity, ...]:
    return tuple(
        ShadowWeightStateIdentity(
            layer=item.layer,
            weight_name=item.weight_name,
            shape=tuple(item.parameter.shape),
            dtype=str(item.parameter.dtype),
            device=str(item.parameter.device),
            sha256=tensor_sha256(item.parameter),
            pointer=int(item.parameter.data_ptr()),
        )
        for item in binding.layer_bindings
    )


class ResidualReserveProductionBindingTests(unittest.TestCase):
    def fixture(self):
        model = _FakeModel()
        hparams = SimpleNamespace(
            layers=list(LAYERS),
            rewrite_module_tmp="layers.{}",
            layer_module_tmp="layers.{}",
            L2=0.25,
        )
        projector = torch.stack(
            [torch.eye(4, dtype=torch.float32) * (1.0 + 0.01 * index)
             for index in range(5)]
        )
        covariances = tuple(
            SealedPrevalidatedCovariance(
                layer,
                torch.eye(4, dtype=torch.float32) * (0.5 + layer * 0.01),
                canonical_hash({"covariance": layer}),
            )
            for layer in LAYERS
        )
        touched = {
            f"layers.{layer}.weight": model.layers[layer].weight
            for layer in LAYERS
        }
        binding = build_residual_reserve_production_binding(
            model=model,
            capture_plan=_capture_plan(),
            hparams=hparams,
            touched_weights=touched,
            projector32=projector,
            projector_artifact_identity=canonical_hash({"projector": "pinned"}),
            covariances=covariances,
            gross_state_id="phase-a-case-test",
            q_residual_tolerance=1.0e-5,
            q_condition_max_dimension=64,
        )
        return model, hparams, projector, covariances, binding

    @staticmethod
    def fake_capture(model, plan, hparams):
        del hparams
        total = sum(
            model.layers[layer].weight.detach().float().sum()
            for layer in LAYERS
        )
        terminal = torch.tensor(
            [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5]],
            dtype=torch.float32,
        ) + total * 1.0e-4
        keys = {
            layer: torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
                dtype=torch.float32,
            ) + total * (layer + 1) * 1.0e-6
            for layer in LAYERS
        }
        payload = {
            "terminal": tensor_sha256(terminal),
            "keys": {str(k): tensor_sha256(v) for k, v in keys.items()},
        }
        return StreamingPhysicalCapture(
            keys_by_layer=keys,
            terminal_z=terminal,
            batch_plan_sha256=plan.streaming_plan.identity_sha256,
            model_state_sha256=canonical_hash({"model": float(total)}),
            physical_forward_count=2,
            processed_token_count=12,
            padded_token_count=16,
            identity_sha256=canonical_hash(payload),
        )

    def test_live_parameter_context_and_zero_history_binding(self):
        model, _, projector, covariances, binding = self.fixture()
        self.assertEqual(tuple(x.layer for x in binding.layer_bindings), LAYERS)
        for item in binding.layer_bindings:
            self.assertIs(item.parameter, model.layers[item.layer].weight)
        for index, context in enumerate(binding.alpha_contexts):
            self.assertEqual(context.projector32.dtype, torch.float32)
            self.assertEqual(context.committed_covariance32.dtype, torch.float32)
            self.assertEqual(context.regularization32.dtype, torch.float32)
            self.assertTrue(torch.equal(context.projector32.cpu(), projector[index]))
            self.assertEqual(torch.count_nonzero(context.committed_covariance32), 0)
            self.assertNotEqual(context.projector32.data_ptr(), projector[index].data_ptr())
        for source, sealed in zip(covariances, binding.covariances, strict=True):
            self.assertEqual(source.artifact_identity, sealed.artifact_identity)
            self.assertEqual(tensor_sha256(source.value), tensor_sha256(sealed.value))
            self.assertNotEqual(source.value.data_ptr(), sealed.value.data_ptr())
        state = binding.gross_ledger.state
        self.assertEqual((state.version, state.commit_count), (0, 0))
        self.assertTrue(all(not layer.committed_precast_factors for layer in state.layers))
        receipt = binding.receipt
        self.assertEqual(set(receipt.live_storage_dtypes), {"torch.float32"})
        self.assertEqual(receipt.algorithm_dtype, "torch.float32")
        self.assertEqual(receipt.model_floating_parameter_dtype, "torch.float32")
        self.assertEqual(receipt.model_floating_parameter_count, 9)
        self.assertEqual(receipt.authoritative_prepared_update_required_dtype, "torch.float32")
        self.assertEqual(receipt.numeric_storage_cast_count, 0)
        self.assertEqual(receipt.bf16_path_call_count, 0)
        self.assertEqual(receipt.bf16_path_decision_influence_count, 0)
        self.assertEqual(receipt.autocast_count, 0)
        self.assertEqual(receipt.downcast_count, 0)
        self.assertEqual(receipt.quantization_count, 0)
        for layer_receipt in receipt.layer_receipts:
            self.assertEqual(layer_receipt.live_storage_dtype, "torch.float32")
            self.assertEqual(
                layer_receipt.authoritative_prepared_update_required_dtype,
                "torch.float32",
            )
            self.assertEqual(layer_receipt.numeric_storage_cast_count, 0)
            self.assertEqual(layer_receipt.bf16_path_call_count, 0)
            self.assertEqual(layer_receipt.bf16_path_decision_influence_count, 0)
        self.assertEqual(receipt.gross_state_version, 0)
        self.assertEqual(receipt.gross_factor_count, 0)
        self.assertEqual(receipt.binding_model_forward_count, 0)
        self.assertEqual(receipt.binding_model_backward_count, 0)
        self.assertEqual(receipt.alpha_history_consume_count, 0)
        self.assertEqual(receipt.alpha_history_append_count, 0)
        self.assertEqual(receipt.alpha_history_finalize_count, 0)

    def test_prefix_capture_refreshes_after_native_earlier_layer_write(self):
        model, _, _, _, binding = self.fixture()
        provider = binding.prefix_provider
        with mock.patch.object(
            binding_module,
            "capture_scalable_physical_state",
            side_effect=self.fake_capture,
        ):
            first = provider(4, _weight_state(binding))
            before = model.layers[4].weight.detach().clone()
            with torch.no_grad():
                update = torch.full(
                    model.layers[4].weight.shape, 0.03125, dtype=torch.float32
                )
                expected = before + update.float()
                model.layers[4].weight[...] = (
                    model.layers[4].weight + update.float()
                )
            self.assertEqual(model.layers[4].weight.dtype, torch.float32)
            self.assertTrue(torch.equal(model.layers[4].weight, expected))
            self.assertEqual((model.layers[4].weight - before).dtype, torch.float32)
            second = provider(5, _weight_state(binding))
        self.assertNotEqual(
            tensor_sha256(first.current_terminal32),
            tensor_sha256(second.current_terminal32),
        )
        self.assertNotEqual(
            tensor_sha256(first.joint_keys32),
            tensor_sha256(second.joint_keys32),
        )
        self.assertEqual(len(provider.receipts), 2)
        receipt = provider.receipts[-1]
        self.assertEqual(receipt.status, PREFIX_OBSERVATION_STATUS)
        self.assertEqual(receipt.logical_capture_group_count, 1)
        self.assertEqual(receipt.physical_forward_count, 2)
        self.assertEqual(receipt.model_backward_count, 0)
        self.assertEqual(receipt.materialization_count, 0)
        self.assertEqual(receipt.heldout_evaluator_count, 0)
        self.assertEqual(receipt.action_influence_count, 0)

    def test_one_complete_prefix_sweep_and_input_immutability(self):
        _, _, projector, covariances, binding = self.fixture()
        projector_sha = tensor_sha256(projector)
        covariance_sha = tuple(tensor_sha256(item.value) for item in covariances)
        with mock.patch.object(
            binding_module,
            "capture_scalable_physical_state",
            side_effect=self.fake_capture,
        ):
            for layer in LAYERS:
                observed = binding.prefix_provider(layer, _weight_state(binding))
                self.assertEqual(observed.current_terminal32.dtype, torch.float32)
                self.assertEqual(observed.joint_keys32.dtype, torch.float32)
                self.assertFalse(observed.current_terminal32.requires_grad)
                self.assertFalse(observed.joint_keys32.requires_grad)
        binding.prefix_provider.assert_complete(expected_sweeps=1)
        self.assertEqual(tensor_sha256(projector), projector_sha)
        self.assertEqual(
            tuple(tensor_sha256(item.value) for item in covariances), covariance_sha
        )

    def test_capture_failure_or_mutation_restores_weights_and_publishes_nothing(self):
        model, _, _, _, binding = self.fixture()
        pointers = tuple(item.parameter.data_ptr() for item in binding.layer_bindings)
        hashes = tuple(tensor_sha256(item.parameter) for item in binding.layer_bindings)
        ledger_identity = binding.gross_ledger.state.identity_sha256

        def fail_after_mutation(model, plan, hparams):
            del plan, hparams
            with torch.no_grad():
                model.layers[4].weight.add_(1.0)
            raise RuntimeError("injected capture failure")

        with mock.patch.object(
            binding_module,
            "capture_scalable_physical_state",
            side_effect=fail_after_mutation,
        ):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                binding.prefix_provider(4, _weight_state(binding))
        self.assertEqual(
            tuple(item.parameter.data_ptr() for item in binding.layer_bindings), pointers
        )
        self.assertEqual(
            tuple(tensor_sha256(item.parameter) for item in binding.layer_bindings), hashes
        )
        self.assertEqual(binding.prefix_provider.receipts, ())
        self.assertEqual(binding.gross_ledger.state.identity_sha256, ledger_identity)

        def return_after_mutation(model, plan, hparams):
            with torch.no_grad():
                model.layers[4].weight.add_(0.5)
            return self.fake_capture(model, plan, hparams)

        with mock.patch.object(
            binding_module,
            "capture_scalable_physical_state",
            side_effect=return_after_mutation,
        ):
            with self.assertRaises(ODEBFStateError):
                binding.prefix_provider(4, _weight_state(binding))
        self.assertEqual(
            tuple(tensor_sha256(item.parameter) for item in binding.layer_bindings), hashes
        )
        self.assertEqual(binding.prefix_provider.receipts, ())
        self.assertEqual(binding.gross_ledger.state.identity_sha256, ledger_identity)

        original = model.layers[4].weight

        def return_after_mapping_swap(model, plan, hparams):
            model.layers[4].weight = torch.nn.Parameter(
                model.layers[4].weight.detach().clone(), requires_grad=False
            )
            return self.fake_capture(model, plan, hparams)

        with mock.patch.object(
            binding_module,
            "capture_scalable_physical_state",
            side_effect=return_after_mapping_swap,
        ):
            with self.assertRaises(ODEBFStateError):
                binding.prefix_provider(4, _weight_state(binding))
        self.assertIs(model.layers[4].weight, original)
        self.assertEqual(
            tuple(tensor_sha256(item.parameter) for item in binding.layer_bindings), hashes
        )
        self.assertEqual(binding.prefix_provider.receipts, ())
        self.assertEqual(binding.gross_ledger.state.identity_sha256, ledger_identity)

    def test_stale_context_and_capture_prestate_fail_before_capture(self):
        _, _, _, _, binding = self.fixture()
        pre = _weight_state(binding)
        forged = (replace(pre[0], sha256=canonical_hash(["forged"])), *pre[1:])
        with mock.patch.object(
            binding_module, "capture_scalable_physical_state"
        ) as capture:
            with self.assertRaises(ODEBFStateError):
                binding.prefix_provider(4, forged)
            capture.assert_not_called()
        with torch.no_grad():
            binding.alpha_contexts[0].projector32.add_(0.0)
        with mock.patch.object(
            binding_module, "capture_scalable_physical_state"
        ) as capture:
            with self.assertRaises(ODEBFStateError):
                binding.prefix_provider(4, pre)
            capture.assert_not_called()

        _, _, _, _, covariance_binding = self.fixture()
        with torch.no_grad():
            covariance_binding.covariances[0].value.add_(0.0)
        with mock.patch.object(
            binding_module, "capture_scalable_physical_state"
        ) as capture:
            with self.assertRaises(ODEBFStateError):
                covariance_binding.prefix_provider(
                    4, _weight_state(covariance_binding)
                )
            capture.assert_not_called()

    def test_bad_live_binding_orientation_and_covariance_fail_close(self):
        model = _FakeModel()
        hparams = SimpleNamespace(
            layers=list(LAYERS), rewrite_module_tmp="layers.{}", L2=0.25
        )
        projector = torch.stack([torch.eye(4) for _ in LAYERS]).float()
        covariances = tuple(
            SealedPrevalidatedCovariance(
                layer, torch.eye(4), canonical_hash({"cov": layer})
            )
            for layer in LAYERS
        )
        touched = {
            f"layers.{layer}.weight": model.layers[layer].weight for layer in LAYERS
        }
        copied = dict(touched)
        copied["layers.4.weight"] = torch.nn.Parameter(
            touched["layers.4.weight"].detach().clone(), requires_grad=False
        )
        kwargs = dict(
            model=model,
            capture_plan=_capture_plan(),
            hparams=hparams,
            projector32=projector,
            projector_artifact_identity="projector",
            covariances=covariances,
            gross_state_id="bad-fixture",
            q_residual_tolerance=1.0e-5,
            q_condition_max_dimension=64,
        )
        with self.assertRaises(ODEBFContractError):
            build_residual_reserve_production_binding(
                touched_weights=copied, **kwargs
            )
        model.layers[0].weight = torch.nn.Parameter(
            model.layers[0].weight.detach().to(dtype=torch.bfloat16)
        )
        with self.assertRaisesRegex(ODEBFContractError, "unquantized FP32"):
            build_residual_reserve_production_binding(
                touched_weights=touched, **kwargs
            )
        model.layers[0].weight = torch.nn.Parameter(
            model.layers[0].weight.detach().to(dtype=torch.float32)
        )
        model.layers[1].weight = torch.nn.Parameter(
            torch.ones(model.layers[1].weight.shape, dtype=torch.int8),
            requires_grad=False,
        )
        with self.assertRaisesRegex(ODEBFContractError, "unquantized FP32"):
            build_residual_reserve_production_binding(
                touched_weights=touched, **kwargs
            )
        model.layers[1].weight = torch.nn.Parameter(
            model.layers[1].weight.detach().to(dtype=torch.float32),
            requires_grad=False,
        )
        with self.assertRaises(ODEBFContractError):
            build_residual_reserve_production_binding(
                touched_weights=touched, projector32=torch.eye(4),
                **{k: v for k, v in kwargs.items() if k != "projector32"}
            )
        bad_covariances = (
            SealedPrevalidatedCovariance(4, torch.eye(3), "bad-cov"),
            *covariances[1:],
        )
        with self.assertRaises(ODEBFContractError):
            build_residual_reserve_production_binding(
                touched_weights=touched,
                covariances=bad_covariances,
                **{k: v for k, v in kwargs.items() if k != "covariances"},
            )

    def test_nonfinite_capture_and_wrong_order_fail_close(self):
        _, _, _, _, binding = self.fixture()
        with self.assertRaises(ODEBFStateError):
            binding.prefix_provider(5, _weight_state(binding))

        def nonfinite_capture(model, plan, hparams):
            observed = self.fake_capture(model, plan, hparams)
            terminal = observed.terminal_z.clone()
            terminal[0, 0] = float("nan")
            return replace(observed, terminal_z=terminal)

        with mock.patch.object(
            binding_module,
            "capture_scalable_physical_state",
            side_effect=nonfinite_capture,
        ):
            with self.assertRaises(ODEBFContractError):
                binding.prefix_provider(4, _weight_state(binding))
        self.assertEqual(binding.prefix_provider.receipts, ())

    def test_autocast_fails_before_binding_or_capture(self):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(ODEBFStateError, "autocast"):
                self.fixture()
        _, _, _, _, binding = self.fixture()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            with mock.patch.object(
                binding_module, "capture_scalable_physical_state"
            ) as capture:
                with self.assertRaisesRegex(ODEBFStateError, "autocast"):
                    binding.prefix_provider(4, _weight_state(binding))
                capture.assert_not_called()

    def test_prohibited_paths_and_public_surface(self):
        source = inspect.getsource(binding_module)
        prohibited = (
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "assemble_effective_bf16",
            "effective_bf16_sha256",
            "p1r52_frozen_pi_quota_writer",
            "p1r52_pir_sequential_writer",
            "compute_z(",
        )
        for symbol in prohibited:
            self.assertNotIn(symbol, source)
        self.assertEqual(source.count("torch.float64"), 1)
        self.assertIn("pretrained_weight_norm_squared", source)
        self.assertIn("capture_scalable_physical_state", source)


if __name__ == "__main__":
    unittest.main()
