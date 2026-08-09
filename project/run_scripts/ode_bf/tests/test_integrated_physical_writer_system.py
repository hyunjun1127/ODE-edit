from __future__ import annotations

import inspect
import threading
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as torch_functional

from project.run_scripts.ode_bf.contracts import ODEBFContractError, ODEBFStateError
from project.run_scripts.ode_bf.integrated_physical_writer import (
    PHYSICAL_WRITER_H,
    PhysicalWriterComputePhase,
    PhysicalWriterForwardRole,
    PhysicalWriterGroupLedger,
    batched_two_output_coefficient_vjp,
)
from project.run_scripts.ode_bf.integrated_physical_writer_runtime import (
    PhysicalThetaOverlay,
    commit_verify_restore_integrated_endpoint,
)
from project.run_scripts.ode_bf.integrated_physical_writer_terminal import (
    IntegratedHeldoutResidualOverlay,
    IntegratedPrimaryEvaluation,
    _paired_payload,
    run_integrated_terminal_panel,
)
from project.run_scripts.ode_bf.integrated_physical_writer_experiment import (
    IntegratedActionFreeze,
)
from project.run_scripts.ode_bf.request_digest import ordered_request_digest_v1
from project.run_scripts.ode_bf.p1_integrated_physical_writer_panel import (
    validate_p1r14_source_closure,
)
from project.run_scripts.ode_bf.tests.test_integrated_physical_writer import (
    _RuntimeModel,
    _runtime_field,
)


class _HeldoutModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target = torch.nn.Identity()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.target(value)


class IntegratedPhysicalWriterSystemTests(unittest.TestCase):
    def test_compute_records_are_physical_and_postfreeze_is_terminal_only(self) -> None:
        ledger = PhysicalWriterGroupLedger()
        with self.assertRaisesRegex(ODEBFContractError, "physical VJP"):
            ledger.record(
                group_id="invalid-vjp",
                phase=PhysicalWriterComputePhase.PRODUCTION,
                role=PhysicalWriterForwardRole.WRITER_VJP,
                step_index=0,
                microbatch_ordinal=0,
                model_forward_calls=0,
                physical_microbatch_graphs=1,
                autograd_backend_invocations=0,
                backward_calls=0,
                processed_tokens=0,
            )
        ledger.freeze_actions("0" * 64)
        with self.assertRaisesRegex(ODEBFContractError, "after freeze"):
            ledger.record(
                group_id="late-production",
                phase=PhysicalWriterComputePhase.PRODUCTION,
                role=PhysicalWriterForwardRole.TARGET_FACTOR,
                step_index=0,
                microbatch_ordinal=None,
                model_forward_calls=1,
                physical_microbatch_graphs=1,
                autograd_backend_invocations=1,
                backward_calls=1,
                processed_tokens=1,
            )
        terminal = ledger.record(
            group_id="terminal",
            phase=PhysicalWriterComputePhase.TERMINAL,
            role=PhysicalWriterForwardRole.TERMINAL,
            step_index=None,
            microbatch_ordinal=None,
            model_forward_calls=1,
            physical_microbatch_graphs=1,
            autograd_backend_invocations=0,
            backward_calls=0,
            processed_tokens=1,
        )
        self.assertEqual(terminal.phase, PhysicalWriterComputePhase.TERMINAL)

    def test_five_layer_theta_overlay_vjp_matches_finite_difference(self) -> None:
        model = _RuntimeModel()
        field = _runtime_field()
        input_ids = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long)
        attention = torch.ones_like(input_ids)
        theta = torch.zeros(5, dtype=torch.float32, requires_grad=True)
        with PhysicalThetaOverlay(model, field, theta):
            loss = model(input_ids=input_ids, attention_mask=attention).logits.float().square().mean()
        gradient = torch.autograd.grad(loss, theta)[0].detach().to(dtype=torch.float64)
        epsilon = 1.0e-3
        finite = []
        for index in range(5):
            values = []
            for sign in (-1.0, 1.0):
                probe = torch.zeros(5, dtype=torch.float32)
                probe[index] = sign * epsilon
                probe.requires_grad_(True)
                with torch.no_grad(), PhysicalThetaOverlay(model, field, probe):
                    values.append(
                        float(
                            model(
                                input_ids=input_ids,
                                attention_mask=attention,
                            ).logits.float().square().mean()
                        )
                    )
            finite.append((values[1] - values[0]) / (2.0 * epsilon))
        np.testing.assert_allclose(
            gradient.numpy(), np.asarray(finite), rtol=5.0e-3, atol=5.0e-3
        )

    def test_real_overlay_two_loss_vjp_fd_and_single_h_candidate(self) -> None:
        model = _RuntimeModel()
        field = _runtime_field()
        input_ids = torch.tensor(
            [[2, 3, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]],
            dtype=torch.long,
        )
        attention = torch.ones_like(input_ids)
        labels = torch.tensor([13, 14, 15], dtype=torch.long)

        def losses(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            captured: list[torch.Tensor] = []
            hook = model.target.register_forward_hook(
                lambda _module, _args, output: captured.append(output)
            )
            try:
                with PhysicalThetaOverlay(model, field, theta):
                    logits = model(
                        input_ids=input_ids, attention_mask=attention
                    ).logits
            finally:
                hook.remove()
            self.assertEqual(len(captured), 1)
            edit = torch_functional.cross_entropy(
                logits[:, -1, :].float(), labels
            )
            transport = 0.5 * captured[0].float().square().mean()
            return edit, transport

        theta = torch.zeros(5, dtype=torch.float32, requires_grad=True)
        edit, transport = losses(theta)
        vjp = batched_two_output_coefficient_vjp(edit, transport, theta)
        epsilon = 1.0e-3
        edit_fd: list[float] = []
        transport_fd: list[float] = []
        for index in range(5):
            edit_values: list[float] = []
            transport_values: list[float] = []
            for sign in (-1.0, 1.0):
                probe = torch.zeros(5, dtype=torch.float32)
                probe[index] = sign * epsilon
                probe.requires_grad_(True)
                with torch.no_grad():
                    observed_edit, observed_transport = losses(probe)
                edit_values.append(float(observed_edit))
                transport_values.append(float(observed_transport))
            edit_fd.append((edit_values[1] - edit_values[0]) / (2.0 * epsilon))
            transport_fd.append(
                (transport_values[1] - transport_values[0]) / (2.0 * epsilon)
            )
        np.testing.assert_allclose(
            -np.asarray(vjp.a_edit), np.asarray(edit_fd), rtol=1.0e-2, atol=1.0e-3
        )
        np.testing.assert_allclose(
            -np.asarray(vjp.a_transport),
            np.asarray(transport_fd),
            rtol=1.0e-2,
            atol=1.0e-3,
        )
        direction = np.maximum(
            np.asarray(vjp.a_edit) + np.asarray(vjp.a_transport), 0.0
        )
        if float(np.linalg.norm(direction)) == 0.0:
            direction = np.ones(5, dtype=np.float64)
        direction = direction / float(np.max(np.abs(direction)))
        candidate = torch.as_tensor(
            PHYSICAL_WRITER_H * direction, dtype=torch.float32
        ).requires_grad_(True)
        double_h = torch.as_tensor(
            PHYSICAL_WRITER_H**2 * direction, dtype=torch.float32
        ).requires_grad_(True)
        self.assertFalse(torch.equal(candidate, double_h))
        with torch.no_grad():
            candidate_losses = losses(candidate)
            double_h_losses = losses(double_h)
        self.assertNotEqual(
            tuple(float(item) for item in candidate_losses),
            tuple(float(item) for item in double_h_losses),
        )

    def test_heldout_additive_assignment_and_locality_nohook(self) -> None:
        model = _HeldoutModel()
        residual = torch.arange(40, dtype=torch.float32).reshape(4, 10) / 10.0
        positions = tuple((0, 1, 2, 0) for _ in range(10))
        counts = (2,) * 10
        overlay = IntegratedHeldoutResidualOverlay(
            model, "target", residual, positions, counts
        )
        observed = []
        with overlay:
            for request in range(10):
                value = torch.zeros((4, 3, 4), dtype=torch.bfloat16)
                observed.append(model(value).detach().clone())
        receipt = overlay.raw_free_payload()
        self.assertEqual(receipt["hook_call_count"], 10)
        self.assertEqual(receipt["patched_row_count"], 20)
        self.assertEqual(receipt["locality_nohook_row_count"], 20)
        self.assertEqual(receipt["maximum_authoritative_assignment_error"], 0.0)
        for request, value in enumerate(observed):
            expected = residual[:, request].to(dtype=torch.bfloat16)
            self.assertTrue(torch.equal(value[0, 0], expected))
            self.assertTrue(torch.equal(value[1, 1], expected))
            self.assertEqual(int(torch.count_nonzero(value[2:])), 0)

    def test_source_closure_rejects_every_alias_dispatch_form_and_legacy_import(self) -> None:
        fixtures = {
            "import.py": "from project.run_scripts.ode_bf import fixed_e8_runtime\n",
            "mapping.py": "X={'llama3-8b-inst': {'h': 0.5}}\n",
            "if.py": "x = 1 if alias == 'qwen2.5-7b-inst' else 2\n",
            "match.py": "match alias:\n case 'llama3-8b-inst':\n  x=1\n",
            "identifier.py": "def f():\n return ZERO_POSITIVE_DIRECTION\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, source in fixtures.items():
                (root / name).write_text(source, encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(ODEBFContractError):
                    validate_p1r14_source_closure(root, paths=(name,))
            good = root / "good.py"
            good.write_text("VALUE='ONE_ARM'\n", encoding="utf-8")
            receipt = validate_p1r14_source_closure(root, paths=(good.name,))
            self.assertEqual(receipt["legacy_runtime_import_count"], 0)

    def test_endpoint_false_postverify_rolls_back(self) -> None:
        model = _RuntimeModel()
        for module in model.writers.values():
            module.weight.data = module.weight.data.to(dtype=torch.bfloat16)
        field = _runtime_field()
        from project.run_scripts.ode_bf.integrated_physical_writer_runtime import (
            IntegratedFactorAccumulator,
        )

        accumulator = IntegratedFactorAccumulator()
        accumulator.append(field, (0.01,) * 5, step_index=0)
        factors = accumulator.factors_by_weight()
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if name in factors
        }
        before = {name: value.detach().clone() for name, value in parameters.items()}
        pointers = {name: value.data_ptr() for name, value in parameters.items()}
        with self.assertRaisesRegex(ODEBFStateError, "post-commit joint verification"):
            commit_verify_restore_integrated_endpoint(
                parameters,
                factors,
                accumulated_step_count=1,
                row_block=4,
                transaction_id="r14-false-postverify",
                mutation_lock=threading.RLock(),
                post_commit_verify=lambda: False,
            )
        self.assertTrue(
            all(
                torch.equal(parameter, before[name])
                and parameter.data_ptr() == pointers[name]
                for name, parameter in parameters.items()
            )
        )

    def test_k0_typed_prefix_runs_postfreeze_without_invented_transaction(self) -> None:
        model = _HeldoutModel()
        requests = [
            {
                "request_sha256": f"{index + 1:064x}",
                "prompt": "{}",
                "subject": "subject",
                "target_new": " target",
            }
            for index in range(10)
        ]
        order = ordered_request_digest_v1(
            [str(item["request_sha256"]) for item in requests]
        )
        freeze = IntegratedActionFreeze(
            order,
            "a" * 64,
            0,
            1,
            "NO_PHYSICAL_W_ONLY_DIRECTION",
        )
        metric_rows = {
            "efficacy": {
                "per_case_bits": [[0] for _ in range(10)],
                "per_case_required": [1] * 10,
                "numerator": 0,
                "denominator": 10,
            },
            "generalization": {
                "per_case_bits": [[0, 0] for _ in range(10)],
                "per_case_required": [2] * 10,
                "numerator": 0,
                "denominator": 20,
            },
            "locality-preservation": {
                "per_case_bits": [[0] * 10 for _ in range(10)],
                "per_case_required": [10] * 10,
                "numerator": 0,
                "denominator": 100,
            },
        }
        primary = {
            "request_order_sha256": order,
            "evaluation_case_identity_sha256": "b" * 64,
            "target_span_sha256": "c" * 64,
            "evaluator_source_sha256": "d" * 64,
            "aggregator_source_sha256": "e" * 64,
            "metrics": metric_rows,
        }
        evaluation = IntegratedPrimaryEvaluation(
            primary, {}, None, "f" * 64
        )
        native = SimpleNamespace(
            direct_z=tuple(torch.zeros(4) for _ in range(10)),
            native_candidates=(),
            raw_free_payload=lambda: {"identity_sha256": "1" * 64},
        )
        hparams = SimpleNamespace(
            fact_token="subject_last", layers=(4, 5, 6, 7, 8)
        )
        lookup = (
            tuple((0,) for _ in range(10)),
            (1,) * 10,
            {"identity_sha256": "2" * 64},
        )
        with mock.patch(
            "project.run_scripts.ode_bf.p1_evaluator.load_counterfact_cases_after_freeze",
            return_value=tuple(object() for _ in range(10)),
        ), mock.patch(
            "project.run_scripts.ode_bf.integrated_physical_writer_terminal.integrated_heldout_lookup_geometry",
            return_value=lookup,
        ), mock.patch(
            "project.run_scripts.ode_bf.integrated_physical_writer_terminal.evaluate_integrated_primary",
            return_value=evaluation,
        ), mock.patch(
            "project.run_scripts.ode_bf.integrated_physical_writer_terminal.capture_cold_z_base",
            return_value=torch.zeros((4, 10)),
        ), mock.patch(
            "project.run_scripts.ode_bf.integrated_physical_writer_terminal.evaluate_exact_functional_p",
            return_value={"identity_sha256": "3" * 64},
        ), mock.patch(
            "project.run_scripts.ode_bf.integrated_physical_writer_terminal.capture_p1_native_entry",
            return_value=native,
        ), mock.patch(
            "project.run_scripts.ode_bf.integrated_physical_writer_terminal.CandidateBF16FunctionalTrial",
            side_effect=lambda *_args, **_kwargs: nullcontext(),
        ), mock.patch(
            "project.run_scripts.ode_bf.integrated_physical_writer_terminal.commit_verify_restore_integrated_endpoint"
        ) as transaction:
            panel = run_integrated_terminal_panel(
                model,
                object(),
                model_alias="llama3-8b-inst",
                requests=requests,
                dataset_path=Path("unused"),
                hparams=hparams,
                target_layer_name="target",
                final_target=torch.ones((4, 10)),
                factors_by_weight={},
                accepted_transition_count=0,
                action_freeze=freeze,
                p_anchor_microbatches=tuple(tuple(requests) for _ in range(6)),
                p_teacher_log_probs=tuple(torch.zeros((10, 2)) for _ in range(6)),
                p_entry_kl=tuple(torch.zeros(10) for _ in range(6)),
                projector=torch.eye(4).repeat(5, 1, 1),
                contexts=(("{}",), ("{}",) * 5),
                mutation_lock=threading.RLock(),
            )
        transaction.assert_not_called()
        self.assertEqual(
            panel.endpoint_transaction["status"],
            "INACTIVE_NO_CERTIFIED_WEIGHT_TRANSITION",
        )
        self.assertEqual(
            panel.endpoint_transaction["transaction_commit_count"], 0
        )
        self.assertTrue(panel.endpoint_transaction["final_w0_restore_exact"])
        self.assertEqual(
            set(panel.paired_native_floor["metrics"]),
            {"efficacy", "generalization", "locality-preservation"},
        )
        legacy_rows = dict(metric_rows)
        legacy_rows["locality"] = legacy_rows.pop("locality-preservation")
        legacy_primary = {**primary, "metrics": legacy_rows}
        legacy_evaluation = IntegratedPrimaryEvaluation(
            legacy_primary, {}, None, "0" * 64
        )
        with self.assertRaises(ODEBFContractError):
            _paired_payload(legacy_evaluation, legacy_evaluation)
        source = inspect.getsource(_paired_payload)
        self.assertIn("for metric in PrimaryMetric", source)
        self.assertNotIn('"locality"', source)
        for broken in (
            replace(freeze, accepted_transition_count=1),
            replace(freeze, attempted_field_count=0),
            replace(freeze, terminal_status="UNRELATED_ZERO_PREFIX"),
        ):
            with self.subTest(broken=broken), self.assertRaises(
                ODEBFContractError
            ):
                run_integrated_terminal_panel(
                    model,
                    object(),
                    model_alias="llama3-8b-inst",
                    requests=requests,
                    dataset_path=Path("unused"),
                    hparams=hparams,
                    target_layer_name="target",
                    final_target=torch.ones((4, 10)),
                    factors_by_weight={},
                    accepted_transition_count=0,
                    action_freeze=broken,
                    p_anchor_microbatches=tuple(tuple(requests) for _ in range(6)),
                    p_teacher_log_probs=tuple(torch.zeros((10, 2)) for _ in range(6)),
                    p_entry_kl=tuple(torch.zeros(10) for _ in range(6)),
                    projector=torch.eye(4).repeat(5, 1, 1),
                    contexts=(("{}",), ("{}",) * 5),
                    mutation_lock=threading.RLock(),
                )


if __name__ == "__main__":
    unittest.main()
