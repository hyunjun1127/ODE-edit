from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFStateError
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p1r52_c_writer_kstep import CKStepWriterRuntime
from project.run_scripts.ode_bf.p1r52_c_writer_kstep_cache import (
    KStepBatchEntryCachePolicy,
    snapshot_alpha_module_cache,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    expected_p1r52_sequential_result_name,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import P1R52_B100X10_SCALE
from project.run_scripts.ode_bf.p1r54_fz_finalz_oneshot_analysis import (
    build_writer_cadence_comparison,
)
from project.run_scripts.ode_bf.p1r54_fz_finalz_oneshot_sequential import (
    RESULT_NAME,
    ROLE,
    build_binding,
    control_source_equivalence,
    validate_finalz_batch_rows,
)
from project.run_scripts.ode_bf.writer_cadence import WriterCadence


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])


class _CachePolicy:
    arm = "C3-KSTEP-CACHE"

    def __init__(self) -> None:
        self.closed: list[int] = []

    def close_no_write_step(self, step_index: int) -> None:
        self.closed.append(step_index)


def _runtime() -> tuple[CKStepWriterRuntime, _Model, _CachePolicy, torch.Tensor]:
    model = _Model()
    policy = _CachePolicy()
    target = torch.arange(200, dtype=torch.float32).reshape(2, 100)

    def resolver(_outer: object, *, step_index: int):
        selected = SimpleNamespace(
            target_step=SimpleNamespace(target_next=target + step_index)
        )
        return selected, SimpleNamespace(identity_sha256=f"{step_index:064x}")

    runtime = CKStepWriterRuntime(
        arm="C3-KSTEP-CACHE",
        model=model,
        tokenizer=object(),
        requests=tuple(
            {"request_sha256": f"{index:064x}"} for index in range(100)
        ),
        hparams=SimpleNamespace(layers=(0,), rewrite_module_tmp="layers.{}"),
        projector=torch.eye(2),
        contexts=(),
        covariance_registry=object(),
        projector_sha256="c" * 64,
        controller_lock=object(),
        objective_plan=object(),
        capture_plan=object(),
        dataset_path=Path("/does/not/open"),
        private_root=Path("/does/not/write"),
        job_ledger=object(),
        alpha_cache_status="ALPHA_CACHE_BATCH_ENTRY_SNAPSHOT",
        cache_policy=policy,
        selected_target_resolver=resolver,
        heldout_step_indices=(7,),
        writer_cadence=WriterCadence.FINAL_Z_ONESHOT,
    )
    return runtime, model, policy, target


class P1R54FZFinalZOneShotTests(unittest.TestCase):
    def test_writer_cadence_and_binding_are_exact(self) -> None:
        binding = build_binding()
        self.assertIs(binding.writer_cadence, WriterCadence.FINAL_Z_ONESHOT)
        self.assertEqual(binding.target_subcycle_schedule.total_field_evaluations, 8)
        self.assertEqual(binding.heldout_step_indices, (7,))
        self.assertEqual(WriterCadence.KSTEP_EACH_OUTER.writer_calls_per_block, 8)
        self.assertEqual(WriterCadence.FINAL_Z_ONESHOT.writer_calls_per_block, 1)
        self.assertEqual(
            [WriterCadence.FINAL_Z_ONESHOT.writes_at(i) for i in range(8)],
            [False] * 7 + [True],
        )

    def test_k1_k7_target_only_and_weight_cache_freeze(self) -> None:
        runtime, model, policy, target = _runtime()
        pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
        versions = {name: value._version for name, value in model.named_parameters()}
        hashes = {
            name: tensor_sha256(value) for name, value in model.named_parameters()
        }
        executions = [runtime.execute(object(), step_index=i) for i in range(7)]
        self.assertEqual(policy.closed, list(range(7)))
        self.assertEqual(executions[0].selected_target_sha256, tensor_sha256(target))
        self.assertTrue(all(item.entry_weight_sha256 == item.commit_weight_sha256 for item in executions))
        self.assertTrue(all(item.compute["logical_commit_count"] == 0 for item in executions))
        self.assertTrue(all(item.compute["native_apply_count"] == 0 for item in executions))
        self.assertTrue(all(item.compute["heldout_evaluator_count"] == 0 for item in executions))
        self.assertTrue(all(item.writer_receipt["intermediate_target_writer_influence_count"] == 0 for item in executions))
        self.assertEqual(
            pointers, {name: value.data_ptr() for name, value in model.named_parameters()}
        )
        self.assertEqual(
            versions, {name: value._version for name, value in model.named_parameters()}
        )
        self.assertEqual(
            hashes,
            {name: tensor_sha256(value) for name, value in model.named_parameters()},
        )

    def test_cache_is_untouched_until_k8_then_appended_once(self) -> None:
        alpha = SimpleNamespace()
        snapshot = snapshot_alpha_module_cache(alpha)
        policy = KStepBatchEntryCachePolicy(
            arm="C3-KSTEP-CACHE",
            batch_index=1,
            history_keys_by_layer=None,
            history_version=0,
            alpha_main=alpha,
            alpha_entry_snapshot=snapshot,
            writer_cadence=WriterCadence.FINAL_Z_ONESHOT,
        )
        for step in range(7):
            policy.close_no_write_step(step)
            self.assertFalse(hasattr(alpha, "cache_c"))
        with self.assertRaises(ODEBFStateError):
            policy.commit_after_k8()
        policy.prepare_c3_step(7)
        alpha.cache_c = torch.ones(2, 100)
        policy.close_c3_step(
            7,
            {
                "logical_history_width_at_entry": 0,
                "logical_history_width_after_append": 100,
                "solver_consumed_entry_cache": False,
                "entry": {"sha256": "d" * 64},
                "exit": {"sha256": "e" * 64},
            },
        )
        self.assertFalse(hasattr(alpha, "cache_c"))
        commit = policy.commit_after_k8()
        self.assertEqual(commit.receipt["consume_count"], 1)
        self.assertEqual(commit.receipt["append_count"], 1)
        self.assertEqual(commit.receipt["no_write_closed_step_count"], 7)
        self.assertEqual(commit.receipt["entry_width"], 0)
        self.assertEqual(commit.receipt["exit_width"], 100)

    def test_full_run_counts_chain_and_stale_entry_negative(self) -> None:
        rows = []
        entry = {"layer": "w0"}
        for index in range(1, 11):
            commit = {"layer": f"w{index}"}
            rows.append(
                {
                    "batch_index": index,
                    "writer_call_count": 1,
                    "layer_apply_count": 5,
                    "field_evaluation_count": 8,
                    "rho_capture_count": 1,
                    "rho_refresh_count": 0,
                    "no_write_prefix_count": 7,
                    "entry_weight_sha256": entry,
                    "commit_weight_sha256": commit,
                    "cache_entry_width": (index - 1) * 100,
                    "cache_exit_width": index * 100,
                }
            )
            entry = commit
        receipt = validate_finalz_batch_rows(rows)
        self.assertEqual(receipt["writer_call_count"], 10)
        self.assertEqual(receipt["layer_apply_count"], 50)
        self.assertEqual(receipt["commit_to_next_entry_match_count"], 9)
        broken = [dict(row) for row in rows]
        broken[4]["entry_weight_sha256"] = {"layer": "stale"}
        with self.assertRaises(ODEBFStateError):
            validate_finalz_batch_rows(broken)

    def test_control_identity_dispatch_and_analysis_counts(self) -> None:
        equivalence = control_source_equivalence()
        self.assertEqual(equivalence["single_variable"], "writer_cadence")
        self.assertEqual(equivalence["target_proposal_difference_count_at_k1"], 0)
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst", ROLE, scale=P1R52_B100X10_SCALE
            ),
            RESULT_NAME,
        )
        shared = {
            "status": "TERMINAL_VALID",
            "completed_batch_count": 10,
            "valid_request_count": 1000,
            "target_field_evaluation_count": 80,
            "W0_restored": True,
            "stream_root": "r",
            "stream_order": "o",
        }
        comparison = build_writer_cadence_comparison(
            treatment={
                **shared,
                "K_writer_call_count": 10,
                "writer_layer_apply_count": 50,
            },
            control={
                **shared,
                "K_writer_call_count": 80,
                "writer_layer_apply_count": 400,
            },
        )
        self.assertEqual(comparison["writer_call_count"]["delta"], -70)
        self.assertEqual(comparison["layer_apply_count"]["delta"], -350)

    def test_thin_binding_does_not_reimplement_science_or_writer(self) -> None:
        from project.run_scripts.ode_bf import p1r54_fz_finalz_oneshot_sequential as module

        source = inspect.getsource(module)
        for prohibited in (
            "run_official_native_apply",
            "accepted_z_cache_template",
            "KStepBatchEntryCachePolicy(",
            "_fp32_target_and_j0(",
            "loss.backward(",
        ):
            self.assertNotIn(prohibited, source)
        self.assertIn("run_phase3(", source)
        self.assertIn("EnergyFreeLocalZPolicy(", source)


if __name__ == "__main__":
    unittest.main()
