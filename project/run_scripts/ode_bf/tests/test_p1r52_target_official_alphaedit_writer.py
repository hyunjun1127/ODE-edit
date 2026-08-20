from __future__ import annotations

import copy
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch

from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1r52_target_official_alphaedit_writer import (
    PHASE_A_RESULT_NAME,
    PHASE_A_ROLE,
    PHASE_A_TECH_R1_RESULT_NAME,
    PHASE_A_TECH_R2_RESULT_NAME,
    PHASE_B_RESULT_NAME,
    PHASE_B_ROLE,
    _writer_gap,
    accepted_z_cache_template,
    isolated_alphaedit_module_state,
    run_phase_b,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    expected_p1r52_sequential_result_name,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import P1R52_B100X10_SCALE
from project.run_scripts.ode_bf.scalable_batched_native import run_official_native_apply
from project.run_scripts.ode_bf.scalable_batched_field import build_scalable_dynamic_field


class _Hparams:
    layers = [4, 5, 6, 7, 8]
    clamp_norm_factor = 4
    device = 0


class P1R52TargetOfficialWriterTests(unittest.TestCase):
    def _requests(self, count: int = 100) -> list[dict[str, object]]:
        return [
            {
                "case_id": index,
                "request_sha256": f"{index:064x}",
                "prompt": "{} is",
                "subject": f"s{index}",
                "target_new": " x",
            }
            for index in range(count)
        ]

    def test_phase_a_result_identity(self) -> None:
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst",
                PHASE_A_ROLE,
                scale=P1R52_B100X10_SCALE,
            ),
            PHASE_A_RESULT_NAME,
        )
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst",
                PHASE_A_ROLE,
                scale=P1R52_B100X10_SCALE,
                attempt_suffix="tech-r1",
            ),
            PHASE_A_TECH_R1_RESULT_NAME,
        )
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst",
                PHASE_A_ROLE,
                scale=P1R52_B100X10_SCALE,
                attempt_suffix="tech-r2",
            ),
            PHASE_A_TECH_R2_RESULT_NAME,
        )

    def test_b100_scalable_field_binds_empty_history_capacity_zero(self) -> None:
        requests = self._requests()
        state = torch.zeros(3, 100)
        captured = {layer: torch.zeros(7, 100) for layer in (4, 5, 6, 7, 8)}
        observed = SimpleNamespace(model_forward_count=0, current_z=state)
        with mock.patch(
            "project.run_scripts.ode_bf.scalable_batched_field.build_p1_dynamic_field",
            return_value=observed,
        ) as builder:
            result = build_scalable_dynamic_field(
                object(),
                object(),
                requests,
                _Hparams(),
                torch.eye(7),
                (("{}",),),
                target_state=state,
                current_terminal=state,
                captured_keys_by_layer=captured,
                accepted_waypoint=0,
                covariance_registry=object(),
                projector_sha256="0" * 64,
                residual_tolerance=1e-8,
                ledger=object(),
            )
        self.assertIs(result, observed)
        self.assertEqual(builder.call_args.kwargs["expected_batch_size"], 100)
        self.assertEqual(builder.call_args.kwargs["maximum_history_columns"], 0)

    def test_accepted_z_cache_is_exact_and_ordered(self) -> None:
        requests = self._requests()
        accepted = torch.arange(17 * 100, dtype=torch.float32).reshape(17, 100)
        with tempfile.TemporaryDirectory() as directory:
            with accepted_z_cache_template(
                requests,
                accepted,
                _Hparams(),
                parent=Path(directory),
            ) as (template, receipt):
                self.assertEqual(receipt["accepted_z_sha256"], tensor_sha256(accepted))
                self.assertEqual(receipt["native_alphaedit_compute_z_expected_count"], 0)
                for index, request in enumerate(requests):
                    path = Path(template.format(8, 4, request["case_id"]))
                    self.assertTrue(np.array_equal(np.load(path)["v_star"], accepted[:, index].numpy()))

    def test_phase_b_result_identity(self) -> None:
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst",
                PHASE_B_ROLE,
                scale=P1R52_B100X10_SCALE,
            ),
            PHASE_B_RESULT_NAME,
        )

    def test_phase_b_exact_writer_and_firewall_contract_is_explicit(self) -> None:
        source = inspect.getsource(run_phase_b)
        self.assertIn("expected_native_compute_z_call_count=0", source)
        self.assertIn("cache_history_width=(round_index - 1) * 100", source)
        self.assertIn("reset_cache=round_index == 1", source)
        self.assertIn('"r52_structural_h": 0', source)
        self.assertIn('"r52_p_barrier": 0', source)
        self.assertIn('"r52_energy_capacity_barrier": 0', source)
        self.assertIn('"materialization_authoritative_count": 1', source)
        self.assertIn('"batch_entry_evaluator_count": 0', source)
        self.assertIn("_restore_exact_w0", source)

    def test_writer_gap_is_signed_w_minus_z(self) -> None:
        z = {"rewrite_target_new_nll_mean": 1.0, "rephrase_target_new_nll_mean": 2.0}
        writer = {"rewrite_target_new_nll_mean": 1.25, "rephrase_target_new_nll_mean": 1.5}
        self.assertEqual(
            _writer_gap(writer, z),
            {
                "rewrite_W_minus_z_target_new_nll": 0.25,
                "rephrase_W_minus_z_target_new_nll": -0.5,
            },
        )

    def test_official_entrypoint_consumes_cache_without_compute_z(self) -> None:
        from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

        model = torch.nn.Linear(3, 3, bias=False)
        touched = {"weight": model.weight}
        requests = self._requests(100)
        accepted = torch.arange(300, dtype=torch.float32).reshape(3, 100)
        state = {
            name: copy.deepcopy(getattr(alpha_main, name, None))
            for name in ("P_loaded", "P_loaded_from", "cache_c_new")
        }
        tensor_state = {
            name: getattr(alpha_main, name, None)
            for name in ("P", "cache_c")
        }

        def fake_compute_ks(*_args, **_kwargs):
            return torch.ones(2, 3, dtype=torch.bfloat16)

        def fake_apply(inner_model, _tokenizer, inner_requests, _hparams, **kwargs):
            template = kwargs["cache_template"]
            for index, request in enumerate(inner_requests):
                loaded = np.load(Path(template.format(8, 4, request["case_id"])))['v_star']
                self.assertTrue(np.array_equal(loaded, accepted[:, index].numpy()))
            alpha_main.compute_ks()
            alpha_main.P = torch.eye(3).repeat(5, 1, 1)
            alpha_main.P_loaded = True
            alpha_main.P_loaded_from = "fixture"
            alpha_main.cache_c = torch.ones(5, 3, 3)
            alpha_main.cache_c_new = True
            originals = {"weight": inner_model.weight.detach().clone()}
            with torch.no_grad():
                inner_model.weight.add_(1)
            return inner_model, originals

        try:
            with tempfile.TemporaryDirectory() as directory:
                with accepted_z_cache_template(
                    requests,
                    accepted,
                    _Hparams(),
                    parent=Path(directory),
                ) as (template, _receipt):
                    with mock.patch.object(alpha_main, "compute_ks", fake_compute_ks), mock.patch.object(
                        alpha_main, "apply_AlphaEdit_to_model", fake_apply
                    ):
                        payload, originals = run_official_native_apply(
                            model,
                            object(),
                            requests,
                            _Hparams(),
                            touched=touched,
                            reset_cache=True,
                            cache_history_width=0,
                            cache_template=template,
                            expected_native_compute_z_call_count=0,
                            accepted_z_source="P1R52_K8_TERMINAL_TARGET",
                        )
            self.assertEqual(payload["native_alphaedit_compute_z_call_count"], 0)
            self.assertTrue(payload["accepted_z_cache_template_used"])
            self.assertEqual(payload["accepted_z_source"], "P1R52_K8_TERMINAL_TARGET")
            self.assertEqual(payload["alphaedit_dynamic_cache_contract"]["static_projection"]["sha256"], tensor_sha256(alpha_main.P))
            self.assertEqual(set(originals), set(touched))
        finally:
            for name, value in tensor_state.items():
                if value is None and hasattr(alpha_main, name):
                    delattr(alpha_main, name)
                elif value is not None:
                    setattr(alpha_main, name, value)
            for name, value in state.items():
                setattr(alpha_main, name, value)

    def test_alpha_module_isolation_restores_globals(self) -> None:
        from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

        alpha_main.P_loaded = False
        alpha_main.cache_c_new = False
        with isolated_alphaedit_module_state() as receipt:
            alpha_main.P_loaded = True
            alpha_main.cache_c_new = True
            alpha_main.P = torch.ones(1)
            alpha_main.cache_c = torch.ones(1)
        self.assertTrue(receipt["restored"])
        self.assertFalse(alpha_main.P_loaded)
        self.assertFalse(alpha_main.cache_c_new)


if __name__ == "__main__":
    unittest.main()
