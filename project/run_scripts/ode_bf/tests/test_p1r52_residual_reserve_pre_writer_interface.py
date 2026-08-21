from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import unittest

import torch

from project.run_scripts.ode_bf.contracts import (
    ODEBFContractError,
    ODEBFStateError,
)
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1r39_normalized_gradient_target import (
    _full_current_residual_step,
)
from project.run_scripts.ode_bf.p1r51_requestwise_semantic_allocation import (
    P1R51ControllerState,
)
from project.run_scripts.ode_bf.p1r52_r42_safe_kdc import (
    P1R52_INSTRUCTION_ID,
    P1R52_METHOD_ID,
    P1R52_REPAIR_REASON,
    P1R52_REPAIR_REVISION,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_pre_writer_interface import (
    P1R52_PRE_WRITER_INTERFACE_STATUS,
    P1R52PreWriterInput,
    P1R52PreWriterObservationReceipt,
    P1R52PreWriterObserver,
    observe_native_il1_pre_writer_input,
    reconstruct_native_il1_selected_target,
)
from project.run_scripts.ode_bf.p1r52_target_depth import (
    P1R52TargetDepth,
    P1R52TargetDepthInner,
    reassemble_p1r52_outer_target,
)
from project.run_scripts.ode_bf.scalable_batched_model import (
    ScalableObjectiveResult,
)
from project.run_scripts.ode_bf.scalable_batched_runtime import (
    scalable_ordered_request_digest,
)


REQUESTS = ("1" * 64,)
REQUEST_ORDER = scalable_ordered_request_digest(REQUESTS)


def _endpoint() -> ScalableObjectiveResult:
    return ScalableObjectiveResult(
        0.3,
        (0.3,),
        REQUEST_ORDER,
        "3" * 64,
        ("4" * 64,),
        (1,),
        1,
        0,
        2,
        2,
        None,
        None,
        "6" * 64,
        "7" * 64,
        "8" * 64,
    )


class _NoActionObserver(P1R52PreWriterObserver):
    def __init__(self) -> None:
        self.observed: P1R52PreWriterInput | None = None

    def observe(
        self, pre_writer_input: P1R52PreWriterInput
    ) -> P1R52PreWriterObservationReceipt:
        self.observed = pre_writer_input
        return P1R52PreWriterObservationReceipt.no_action(
            pre_writer_input,
            observer_id="focused-no-action-observer",
        )


class _MutatingObserver(P1R52PreWriterObserver):
    def __init__(
        self,
        parameter: torch.nn.Parameter,
        *,
        fail: bool,
        mutate_target: bool = False,
    ) -> None:
        self.parameter = parameter
        self.fail = fail
        self.mutate_target = mutate_target

    def observe(
        self, pre_writer_input: P1R52PreWriterInput
    ) -> P1R52PreWriterObservationReceipt:
        with torch.no_grad():
            self.parameter.add_(1.0)
            if self.mutate_target:
                pre_writer_input.selected_target.target_step.target_next.add_(0.5)
        if self.fail:
            raise RuntimeError("injected pre-writer hook failure")
        return P1R52PreWriterObservationReceipt.no_action(
            pre_writer_input,
            observer_id="focused-mutating-observer",
        )


class _ForgedComputeObserver(P1R52PreWriterObserver):
    def observe(
        self, pre_writer_input: P1R52PreWriterInput
    ) -> P1R52PreWriterObservationReceipt:
        return replace(
            P1R52PreWriterObservationReceipt.no_action(
                pre_writer_input,
                observer_id="focused-forged-compute-observer",
            ),
            model_forward_count=1,
        )


class _EntryContractMutationObserver(P1R52PreWriterObserver):
    def __init__(self, touched, mutation: str) -> None:
        self.touched = touched
        self.mutation = mutation
        self.mutated_version: int | None = None

    def observe(
        self, pre_writer_input: P1R52PreWriterInput
    ) -> P1R52PreWriterObservationReceipt:
        name = next(iter(self.touched))
        parameter = self.touched[name]
        if self.mutation == "parameter_add_zero":
            with torch.no_grad():
                parameter.add_(0)
            self.mutated_version = int(parameter._version)
        elif self.mutation == "requires_grad":
            parameter.requires_grad_(True)
        elif self.mutation == "mapping_swap":
            self.touched[name] = torch.nn.Parameter(
                parameter.detach().clone(),
                requires_grad=parameter.requires_grad,
            )
        elif self.mutation == "same_pointer_stride":
            parameter.data = parameter.data.as_strided(
                tuple(parameter.shape), tuple(reversed(parameter.stride()))
            )
        elif self.mutation == "target_add_zero":
            with torch.no_grad():
                pre_writer_input.selected_target.target_step.target_next.add_(0)
            self.mutated_version = int(
                pre_writer_input.selected_target.target_step.target_next._version
            )
        else:
            raise AssertionError("unknown focused mutation")
        return P1R52PreWriterObservationReceipt.no_action(
            pre_writer_input,
            observer_id=f"focused-{self.mutation}-observer",
        )


class P1R52ResidualReservePreWriterInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entry = torch.tensor([[0.4], [0.2], [0.8]], dtype=torch.float32)
        self.terminal = self.entry - 0.05
        self.target = self.entry + torch.tensor(
            [[0.1], [0.05], [-0.1]], dtype=torch.float32
        )
        self.gradient = torch.tensor(
            [[-0.2], [-0.1], [0.2]], dtype=torch.float64
        )
        self.state = P1R51ControllerState((1.0,), (0.1,))
        self.endpoint = _endpoint()
        self.touched = {
            "model.layers.4.mlp.down_proj.weight": torch.nn.Parameter(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
                requires_grad=False,
            ),
            "model.layers.8.mlp.down_proj.weight": torch.nn.Parameter(
                torch.tensor([[0.5, -0.5], [1.5, -1.5]], dtype=torch.bfloat16),
                requires_grad=False,
            ),
        }

    def inner(self, *, inner_index: int, depth: int, step_index: int = 2):
        target = self.target + 0.01 * inner_index
        receipt = {
            "schema": "focused-native-p1r52-selected-target/v1",
            "instruction_id": P1R52_INSTRUCTION_ID,
            "method_id": P1R52_METHOD_ID,
            "repair_revision": P1R52_REPAIR_REVISION,
            "repair_reason": P1R52_REPAIR_REASON,
            "k": step_index,
            "selection_by_request": ["PRIMARY"],
            "entry_norm_calibration_count": int(inner_index == 0),
        }
        step = _full_current_residual_step(
            target_next=target,
            current_target=self.entry,
            current_terminal=self.terminal,
            nll_gradient=self.gradient,
            aggregate_gradient=2.0 * self.gradient,
            held=(False,),
            receipt=receipt,
        )
        return P1R52TargetDepthInner(
            inner_index,
            step_index * depth + inner_index,
            step,
            self.endpoint,
            self.state,
        )

    def outer(self, *, depth: P1R52TargetDepth = P1R52TargetDepth.IL1):
        inners = tuple(
            self.inner(inner_index=index, depth=depth.inner_count)
            for index in range(depth.inner_count)
        )
        return reassemble_p1r52_outer_target(
            outer_step_index=2,
            depth=depth,
            outer_entry_target=self.entry,
            current_terminal=self.terminal,
            inner_steps=inners,
            outer_entry_nll_gradient=self.gradient,
            outer_entry_aggregate_gradient=2.0 * self.gradient,
            physical_state_sha256_before="a" * 64,
            physical_state_sha256_after_inners="a" * 64,
            teacher_sha256_before="b" * 64,
            teacher_sha256_after_inners="b" * 64,
            history_cache_sha256_before="c" * 64,
            history_cache_sha256_after_inners="c" * 64,
            factor_inventory_sha256_before="d" * 64,
            factor_inventory_sha256_after_inners="d" * 64,
        )

    @staticmethod
    def snapshot(parameters):
        return {
            name: (
                parameter,
                int(parameter.data_ptr()),
                tensor_sha256(parameter),
                parameter.detach().clone(),
                bool(parameter.requires_grad),
                tuple(parameter.shape),
                parameter.dtype,
                parameter.device,
                parameter.layout,
                tuple(parameter.stride()),
                int(parameter.storage_offset()),
                int(parameter._version),
            )
            for name, parameter in parameters.items()
        }

    def assert_snapshot(self, snapshot) -> None:
        for name, (
            parameter,
            pointer,
            sha256,
            value,
            requires_grad,
            shape,
            dtype,
            device,
            layout,
            stride,
            storage_offset,
            _version,
        ) in snapshot.items():
            self.assertIs(self.touched[name], parameter)
            self.assertEqual(int(parameter.data_ptr()), pointer)
            self.assertEqual(tensor_sha256(parameter), sha256)
            self.assertTrue(torch.equal(parameter, value))
            self.assertEqual(bool(parameter.requires_grad), requires_grad)
            self.assertEqual(tuple(parameter.shape), shape)
            self.assertEqual(parameter.dtype, dtype)
            self.assertEqual(parameter.device, device)
            self.assertEqual(parameter.layout, layout)
            self.assertEqual(tuple(parameter.stride()), stride)
            self.assertEqual(int(parameter.storage_offset()), storage_offset)

    def test_native_il1_bridge_is_exact_for_six_tensors_endpoint_and_state(self):
        outer = self.outer()
        selected, receipt = reconstruct_native_il1_selected_target(
            outer, step_index=2
        )
        self.assertEqual(receipt.status, P1R52_PRE_WRITER_INTERFACE_STATUS)
        self.assertEqual(receipt.exact_six_tensor_match_count, 6)
        self.assertTrue(receipt.endpoint_identity_match)
        self.assertTrue(receipt.next_state_match)
        self.assertIs(selected.target_step, outer.inner_steps[-1].target_step)
        self.assertIs(selected.selected_endpoint, outer.selected_endpoint)
        self.assertIs(selected.next_state, outer.next_state)
        self.assertIs(selected.receipt, outer.inner_steps[-1].target_step.receipt)
        for name in (
            "target_next",
            "target_displacement",
            "required_displacement",
            "write_velocity",
            "nll_gradient",
            "combined_gradient",
        ):
            self.assertTrue(
                torch.equal(
                    getattr(selected.target_step, name),
                    getattr(outer.target_step, name),
                )
            )

    def test_il3_and_mixed_outer_components_fail_before_hook(self):
        with self.assertRaises(ODEBFContractError):
            reconstruct_native_il1_selected_target(self.outer(depth=P1R52TargetDepth.IL3_FULL), step_index=2)
        outer = self.outer()
        forged_step = replace(
            outer.target_step,
            target_next=outer.target_step.target_next + 0.01,
        )
        with self.assertRaises(ODEBFStateError):
            reconstruct_native_il1_selected_target(
                replace(outer, target_step=forged_step), step_index=2
            )
        with self.assertRaises(ODEBFStateError):
            reconstruct_native_il1_selected_target(
                replace(outer, selected_endpoint=replace(self.endpoint)),
                step_index=2,
            )
        with self.assertRaises(ODEBFStateError):
            reconstruct_native_il1_selected_target(
                replace(outer, next_state=replace(self.state)), step_index=2
            )

    def test_observer_seals_exact_pre_j0_weight_request_and_action_freeze(self):
        observer = _NoActionObserver()
        before = self.snapshot(self.touched)
        receipt = observe_native_il1_pre_writer_input(
            observer,
            outer=self.outer(),
            step_index=2,
            touched=self.touched,
            request_identities=REQUESTS,
            request_order_sha256=REQUEST_ORDER,
        )
        self.assert_snapshot(before)
        self.assertIsNotNone(observer.observed)
        assert observer.observed is not None
        sealed = observer.observed.receipt
        self.assertEqual(sealed.request_identities, REQUESTS)
        self.assertEqual(sealed.request_order_sha256, REQUEST_ORDER)
        self.assertEqual(len(sealed.entry_weight_identities), 2)
        self.assertEqual(sealed.target_recompute_count, 0)
        self.assertEqual(sealed.writer_action_count, 0)
        self.assertEqual(sealed.target_controller_influence_count, 0)
        self.assertEqual(receipt.model_forward_count, 0)
        self.assertEqual(receipt.model_backward_count, 0)
        self.assertEqual(receipt.materialization_count, 0)
        self.assertEqual(receipt.heldout_evaluator_count, 0)

    def test_hook_failure_and_mutation_restore_exact_w_before_j0(self):
        for fail in (True, False):
            with self.subTest(fail=fail):
                before = self.snapshot(self.touched)
                observer = _MutatingObserver(
                    next(iter(self.touched.values())), fail=fail
                )
                expected = RuntimeError if fail else ODEBFStateError
                with self.assertRaises(expected):
                    observe_native_il1_pre_writer_input(
                        observer,
                        outer=self.outer(),
                        step_index=2,
                        touched=self.touched,
                        request_identities=REQUESTS,
                        request_order_sha256=REQUEST_ORDER,
                    )
                self.assert_snapshot(before)

    def test_target_mutation_is_rejected_and_restored_with_w_exact(self):
        outer = self.outer()
        target_before = outer.inner_steps[-1].target_step.target_next.clone()
        before = self.snapshot(self.touched)
        observer = _MutatingObserver(
            next(iter(self.touched.values())),
            fail=False,
            mutate_target=True,
        )
        with self.assertRaises(ODEBFStateError):
            observe_native_il1_pre_writer_input(
                observer,
                outer=outer,
                step_index=2,
                touched=self.touched,
                request_identities=REQUESTS,
                request_order_sha256=REQUEST_ORDER,
            )
        self.assert_snapshot(before)
        self.assertTrue(
            torch.equal(outer.inner_steps[-1].target_step.target_next, target_before)
        )

    def test_nonzero_compute_or_influence_receipt_fails_before_j0(self):
        before = self.snapshot(self.touched)
        with self.assertRaises(ODEBFStateError):
            observe_native_il1_pre_writer_input(
                _ForgedComputeObserver(),
                outer=self.outer(),
                step_index=2,
                touched=self.touched,
                request_identities=REQUESTS,
                request_order_sha256=REQUEST_ORDER,
            )
        self.assert_snapshot(before)

    def test_live_weight_and_target_contract_mutations_fail_and_restore(self):
        for mutation in (
            "parameter_add_zero",
            "requires_grad",
            "mapping_swap",
            "same_pointer_stride",
            "target_add_zero",
        ):
            with self.subTest(mutation=mutation):
                outer = self.outer()
                target = outer.inner_steps[-1].target_step.target_next
                target_pointer = int(target.data_ptr())
                target_sha256 = tensor_sha256(target)
                target_requires_grad = bool(target.requires_grad)
                target_version = int(target._version)
                before = self.snapshot(self.touched)
                observer = _EntryContractMutationObserver(
                    self.touched, mutation
                )
                with self.assertRaises(ODEBFStateError):
                    observe_native_il1_pre_writer_input(
                        observer,
                        outer=outer,
                        step_index=2,
                        touched=self.touched,
                        request_identities=REQUESTS,
                        request_order_sha256=REQUEST_ORDER,
                    )
                self.assert_snapshot(before)
                self.assertEqual(int(target.data_ptr()), target_pointer)
                self.assertEqual(tensor_sha256(target), target_sha256)
                self.assertEqual(bool(target.requires_grad), target_requires_grad)
                if mutation == "parameter_add_zero":
                    original_version = next(iter(before.values()))[-1]
                    self.assertIsNotNone(observer.mutated_version)
                    self.assertGreater(observer.mutated_version, original_version)
                if mutation == "target_add_zero":
                    self.assertIsNotNone(observer.mutated_version)
                    self.assertGreater(observer.mutated_version, target_version)

    def test_default_disabled_hook_is_additive_and_precedes_j0_field(self):
        root = Path(__file__).resolve().parents[1]
        runtime_path = root / "p1_scalable_batched_experiment.py"
        runtime = runtime_path.read_text(encoding="utf-8")
        from project.run_scripts.ode_bf import p1_scalable_batched_experiment

        parameter = inspect.signature(
            p1_scalable_batched_experiment._run_ode_arm
        ).parameters["p1r52_pre_writer_observer"]
        self.assertIsNone(parameter.default)
        guard = runtime.index("if p1r52_pre_writer_observer is not None:", runtime.index("target_step = outer52.target_step"))
        hook = runtime.index("observe_native_il1_pre_writer_input(", guard)
        field = runtime.index("field = build_scalable_dynamic_field(", hook)
        materialize = runtime.index("materializer.materialize(", field)
        self.assertLess(guard, hook)
        self.assertLess(hook, field)
        self.assertLess(field, materialize)
        self.assertNotIn(
            "p1r52_pre_writer_observer",
            runtime[runtime.index("else:\n                    target_step = p1r24_target_step("):guard],
        )

    def test_prohibited_paths_are_absent(self):
        source = Path(
            inspect.getsourcefile(observe_native_il1_pre_writer_input) or ""
        ).read_text(encoding="utf-8")
        for prohibited in (
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "assemble_effective_bf16",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "solve_residual_reserve_pc_router",
            "run_residual_reserve_phase_a_outer",
            "evaluate_scalable_primary",
            "P1HistoryLedger",
        ):
            self.assertNotIn(prohibited, source)


if __name__ == "__main__":
    unittest.main()
