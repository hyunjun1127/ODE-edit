from __future__ import annotations

import ast
import inspect
import sys
import types
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_bf.p1r52_official_sequential_baseline_panel import (
    verify_memit_artifacts,
)
from project.run_scripts.ode_bf.p1r52_official_sequential_baselines import (
    run_official_memit_apply,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    MEMIT_ROLE,
    NATIVE_CORRECTED_ROLE,
    expected_p1r52_sequential_result_name,
    run_p1r52_sequential,
)
from project.run_scripts.ode_bf.scalable_batched_native import (
    run_official_native_apply,
)
from project.run_scripts.session05_ode_bf_p1r52_official_sequential_baselines_dry_plan import (
    build_plan,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((2, 2)), requires_grad=False)


def requests() -> list[dict[str, str]]:
    return [
        {"request_sha256": str(index).zfill(64), "target_new": " x"}
        for index in range(10)
    ]


class P1R52OfficialSequentialBaselineTests(unittest.TestCase):
    def test_alphaedit_reset_once_then_cache_continuity(self) -> None:
        model = TinyModel()
        module = types.ModuleType("easyeditor.models.alphaedit.AlphaEdit_main")
        module.cache_c_new = False
        module.P_loaded = False
        module.P_loaded_from = None
        reset_values: list[bool] = []

        def compute_ks(*args, **kwargs):
            del args, kwargs
            return torch.ones((2, 2), dtype=torch.bfloat16)

        module.compute_ks = compute_ks

        def apply(*args, **kwargs):
            del args
            reset = bool(kwargs["reset_cache"])
            reset_values.append(reset)
            if reset or not module.cache_c_new:
                module.cache_c = torch.zeros((1, 2, 2), dtype=torch.float32)
                module.cache_c_new = True
            module.compute_ks()
            module.cache_c.add_(torch.eye(2).unsqueeze(0))
            module.P = torch.eye(2).unsqueeze(0)
            module.P_loaded = True
            module.P_loaded_from = "/locked/P.pt"
            entry = model.weight.detach().clone()
            with torch.no_grad():
                model.weight.add_(1.0)
            return model, {"weight": entry}

        module.apply_AlphaEdit_to_model = apply
        easyeditor = types.ModuleType("easyeditor")
        models = types.ModuleType("easyeditor.models")
        alphaedit = types.ModuleType("easyeditor.models.alphaedit")
        alphaedit.AlphaEdit_main = module
        modules = {
            "easyeditor": easyeditor,
            "easyeditor.models": models,
            "easyeditor.models.alphaedit": alphaedit,
            "easyeditor.models.alphaedit.AlphaEdit_main": module,
        }
        with mock.patch.dict(sys.modules, modules):
            first, _ = run_official_native_apply(
                model, object(), requests(), object(), touched={"weight": model.weight},
                reset_cache=True, cache_history_width=0,
            )
            second, _ = run_official_native_apply(
                model, object(), requests(), object(), touched={"weight": model.weight},
                reset_cache=False, cache_history_width=10,
            )
        self.assertEqual(reset_values, [True, False])
        first_cache = first["alphaedit_dynamic_cache_contract"]
        second_cache = second["alphaedit_dynamic_cache_contract"]
        self.assertEqual(first_cache["exit"]["sha256"], second_cache["entry"]["sha256"])
        self.assertTrue(second_cache["solver_consumed_entry_cache"])
        self.assertGreater(second_cache["exit"]["frobenius_norm"], first_cache["exit"]["frobenius_norm"])
        self.assertTrue(first_cache["static_projection_distinct_from_dynamic_cache"])

    def test_memit_calls_official_entrypoint_and_reuses_only_static_covariance(self) -> None:
        model = TinyModel()
        module = types.ModuleType("easyeditor.models.memit.memit_main")
        module.COV_CACHE = {}
        calls: list[dict[str, object]] = []

        def apply(*args, **kwargs):
            del args
            calls.append(dict(kwargs))
            for layer in range(4, 9):
                module.COV_CACHE[("llama", f"layer-{layer}")] = torch.eye(2)
            entry = model.weight.detach().clone()
            with torch.no_grad():
                model.weight.add_(1.0)
            return model, {"weight": entry}

        module.apply_memit_to_model = apply
        easyeditor = types.ModuleType("easyeditor")
        models = types.ModuleType("easyeditor.models")
        memit = types.ModuleType("easyeditor.models.memit")
        memit.memit_main = module
        modules = {
            "easyeditor": easyeditor,
            "easyeditor.models": models,
            "easyeditor.models.memit": memit,
            "easyeditor.models.memit.memit_main": module,
        }
        with mock.patch.dict(sys.modules, modules):
            first, _ = run_official_memit_apply(
                model, object(), requests(), object(), touched={"weight": model.weight}
            )
            second, _ = run_official_memit_apply(
                model, object(), requests(), object(), touched={"weight": model.weight}
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[0],
            {
                "copy": False,
                "return_orig_weights": True,
                "cache_template": None,
                "keep_original_weight": False,
            },
        )
        self.assertEqual(first["covariance_cache"]["entry"]["entry_count"], 0)
        self.assertEqual(second["covariance_cache"]["entry_reused_count"], 5)
        self.assertFalse(second["target_z_cache"]["historical_decision_state"])

    def test_adapters_do_not_reimplement_official_solver(self) -> None:
        for function, entrypoint in (
            (run_official_native_apply, "apply_AlphaEdit_to_model"),
            (run_official_memit_apply, "apply_memit_to_model"),
        ):
            source = inspect.getsource(function)
            tree = ast.parse(source)
            attributes = {
                node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
            }
            self.assertIn(entrypoint, attributes)
            self.assertNotIn("solve", attributes)
            self.assertNotIn("lstsq", attributes)

    def test_runtime_binds_corrected_reset_to_first_batch_only(self) -> None:
        source = inspect.getsource(run_p1r52_sequential)
        self.assertIn("reset_cache=round_index == 1", source)
        self.assertIn("cache_history_width=history_width", source)

    def test_roles_results_and_two_cell_dry_plan(self) -> None:
        plan = build_plan("a" * 40)
        self.assertEqual(plan["job_count"], 2)
        self.assertEqual(plan["array"], "0-1%2")
        self.assertEqual(
            [item["role"] for item in plan["jobs"]],
            [NATIVE_CORRECTED_ROLE, MEMIT_ROLE],
        )
        self.assertIn("cache-on-corrected", expected_p1r52_sequential_result_name("llama3-8b-inst", NATIVE_CORRECTED_ROLE))
        self.assertIn("official-memit", expected_p1r52_sequential_result_name("llama3-8b-inst", MEMIT_ROLE))

    def test_memit_sources_hparams_and_covariance_inventory_are_locked(self) -> None:
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[4]
        receipt = verify_memit_artifacts(repo_root, hash_covariances=False)
        self.assertEqual(receipt["hparams_sha256"], "2b81838b49b1a5e0e4ef41f229216fda473200093fcbed6bd040f07a70d43818")
        self.assertEqual(len(receipt["covariance_sha256"]), 5)


if __name__ == "__main__":
    unittest.main()
