from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from project.run_scripts.ode_bf.compute_progress_simplex_runtime import (
    FixedRankHistoricalSketch,
)
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.fixed_e8_soft_routing import FixedE8Arm
from project.run_scripts.ode_bf.fixed_e8_runtime import FixedE8ProblemReceipt
from project.run_scripts.ode_bf.historical_h0_sequential_runtime import (
    EVALUATION_ROUNDS,
    HISTORICAL_SKETCH_RANK,
    HISTORY_COUNTS,
    METHODS,
    _run_ours_round,
    expected_historical_h0_result_name,
)
from project.run_scripts.ode_bf.historical_h0_sequential_selection import (
    verify_historical_h0_fresh_seal,
)
from project.run_scripts.ode_bf.p1_historical_h0_sequential_panel import (
    LOCK_SCHEMA,
    load_and_validate_historical_h0_lock,
)
from project.run_scripts.ode_bf.p1_scalable_batched_experiment import (
    _run_ode_arm,
    _with_fixed_rank_historical_barrier,
)
from project.run_scripts.ode_bf.progress_simplex_routing import (
    StructuralOnlyRoutingInventory,
    solve_progress_simplex_routing,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem
from project.run_scripts.ode_bf.scalable_batched_runtime import UniformRequestAccumulator
from project.run_scripts import session05_ode_bf_historical_h0_sequential_dry_plan as dry


ROOT = Path(__file__).resolve().parents[4]
LOCKS = ROOT / "project/run_scripts/ode_bf/locks"


def _barrier(label: str, linear: tuple[float, ...]) -> QuadraticBarrier:
    size = len(linear)
    return QuadraticBarrier(
        label,
        0.0,
        np.asarray(linear, dtype=np.float64),
        np.eye(size, dtype=np.float64) * 0.01,
        1.0,
        "layer-local-diagonal",
    )


def _problem(*, historical: bool) -> RoutingProblem:
    return RoutingProblem(
        np.asarray((0.5, 0.3, 0.2, 0.1, 0.05), dtype=np.float64),
        np.eye(5, dtype=np.float64),
        np.eye(5, dtype=np.float64),
        100.0,
        np.ones(5, dtype=np.float64),
        1.0e-6,
        1.0e-8,
        _barrier(
            "historical",
            (0.0, 8.0, 0.0, 0.0, 0.0) if historical else (0.0,) * 5,
        ),
        _barrier("pretrained", (8.0, 0.0, 0.0, 0.0, 0.0)),
    )


class Full6StructuralHistoricalTest(unittest.TestCase):
    def test_fresh_seal_and_new_lock_are_rooted(self) -> None:
        seal = verify_historical_h0_fresh_seal(
            json.loads(
                (LOCKS / "p1r20_historical_h0_fresh_cf_b100_seal.json").read_text()
            )
        )
        lock, _ = load_and_validate_historical_h0_lock(
            LOCKS / "numerical_lock_s05_historical_h0_sequential.json"
        )
        self.assertEqual(lock["schema_version"], LOCK_SCHEMA)
        self.assertEqual(lock["fresh_seal_root"], seal["root_digest"])
        self.assertEqual(lock["history"]["round_entry_counts"], list(HISTORY_COUNTS))
        self.assertEqual(HISTORICAL_SKETCH_RANK, 100)
        self.assertEqual(EVALUATION_ROUNDS, (1, 5, 10))

    def test_full_matrix_and_dry_plan_repeat(self) -> None:
        first = dry.build_plan("0" * 40)
        self.assertEqual(first, dry.build_plan("0" * 40))
        self.assertEqual(first["trajectory_count"], 10)
        self.assertEqual(first["ode_trajectory_count"], 8)
        self.assertEqual(first["model_level_alphaedit_count"], 2)
        self.assertEqual(first["array_max_concurrent_gpu"], 4)
        self.assertEqual([job["method"] for job in first["jobs"][:5]], list(METHODS))
        self.assertEqual(len({job["result_name"] for job in first["jobs"]}), 10)
        repair = dry.build_plan(
            "1" * 40, attempt_namespace="tech-r1", include_alpha=False
        )
        self.assertEqual(repair["trajectory_count"], 8)
        self.assertEqual(repair["ode_trajectory_count"], 8)
        self.assertEqual(repair["model_level_alphaedit_count"], 0)
        self.assertTrue(
            all(job["result_name"].endswith("-tech-r1-v1") for job in repair["jobs"])
        )
        repair_r2 = dry.build_plan(
            "1" * 40, attempt_namespace="tech-r2", include_alpha=False
        )
        self.assertEqual(repair_r2["trajectory_count"], 8)
        self.assertTrue(
            all(job["result_name"].endswith("-tech-r2-v1") for job in repair_r2["jobs"])
        )
        repair_r3 = dry.build_plan(
            "2" * 40, attempt_namespace="tech-r3", include_alpha=False
        )
        self.assertTrue(
            all(job["result_name"].endswith("-tech-r3-v1") for job in repair_r3["jobs"])
        )

    def test_full_six_global_weighting_is_partition_invariant(self) -> None:
        per_request = (1.0, 2.0, 4.0, 8.0, 16.0)
        full = UniformRequestAccumulator(5)
        full.add(range(5), per_request)
        split = UniformRequestAccumulator(5)
        split.add((3, 1), (8.0, 2.0))
        split.add((4,), (16.0,))
        split.add((0, 2), (1.0, 4.0))
        self.assertEqual(full.finalize(), split.finalize())
        self.assertEqual(full.finalize()[0], sum(per_request) / len(per_request))
        source = inspect.getsource(_run_ode_arm)
        self.assertIn("full_six_slope", source)
        self.assertIn("slope_plan =", source)
        self.assertIn("else objective_plan", source)

    def test_fixed_rank_h_transaction_rollback_commit_once(self) -> None:
        sketch = FixedRankHistoricalSketch(100, {4: 3, 5: 3})
        values = {4: np.ones((10, 3)), 5: np.ones((10, 3)) * 2.0}
        first = sketch.stage(values)
        sketch.finalize(transaction_committed=False)
        self.assertEqual(sketch.history_item_count, 0)
        self.assertEqual(sketch.commit_count, 0)
        second = sketch.stage(values)
        self.assertEqual(first, second)
        sketch.finalize(transaction_committed=True)
        self.assertEqual(sketch.history_item_count, 10)
        self.assertEqual(sketch.commit_count, 1)
        with self.assertRaisesRegex(Exception, "no staged"):
            sketch.finalize(transaction_committed=True)

    def test_t2_h_can_change_soft_allocation_without_model_calls(self) -> None:
        p_only = solve_progress_simplex_routing(
            _problem(historical=False),
            StructuralOnlyRoutingInventory(0, "a" * 64, "b" * 64, "c" * 64),
            arm=FixedE8Arm.SOFT,
            alpha_req=1.0,
        )
        with_h = solve_progress_simplex_routing(
            _problem(historical=True),
            StructuralOnlyRoutingInventory(10, "a" * 64, "b" * 64, "c" * 64),
            arm=FixedE8Arm.SOFT,
            alpha_req=1.0,
        )
        self.assertGreater(
            np.max(np.abs(np.asarray(p_only.velocity) - np.asarray(with_h.velocity))),
            1.0e-8,
        )
        source = inspect.getsource(_run_ours_round)
        self.assertIn("historical_h_controller_input_count", source)
        self.assertIn("historical_h_decision_influence_count", source)
        self.assertIn("maximum_records=40", source)
        self.assertNotIn("maximum_records=10", source)

    def test_projected_key_h_barrier_matches_manual_quadratic(self) -> None:
        layers = (4, 5, 6, 7, 8)
        sketch = FixedRankHistoricalSketch(10, {layer: 2 for layer in layers})
        historical_rows = {
            layer: np.asarray(
                [[1.0 + layer / 10.0, (-1.0) ** row] for row in range(10)],
                dtype=np.float64,
            )
            for layer in layers
        }
        sketch.stage(historical_rows, item_count=10)
        sketch.finalize(transaction_committed=True)
        names = {layer: f"weight-{layer}" for layer in layers}
        w0 = {name: torch.zeros((2, 2), dtype=torch.float64) for name in names.values()}
        entry = {
            name: torch.full((2, 2), 0.01 * layer, dtype=torch.float64)
            for layer, name in names.items()
        }
        fields = tuple(
            SimpleNamespace(
                layer=layer,
                weight_name=names[layer],
                residual=torch.tensor([[1.0], [0.5]], dtype=torch.float64),
                q=torch.tensor([[0.25], [0.75]], dtype=torch.float64),
            )
            for layer in layers
        )
        base = FixedE8ProblemReceipt(
            _problem(historical=False),
            (0.0,) * 5,
            (0.0,) * 5,
            (0.0,) * 5,
            "d" * 64,
        )
        observed = _with_fixed_rank_historical_barrier(
            base,
            SimpleNamespace(layers=fields),
            sketch,
            entry_values=entry,
            trajectory_w0_values=w0,
        )
        velocity = np.asarray((1.0, 0.8, 0.6, 0.4, 0.2), dtype=np.float64)
        manual = 0.0
        for index, layer in enumerate(layers):
            baseline = entry[names[layer]].numpy() @ historical_rows[layer].T
            proposal = (
                fields[index].residual.numpy()
                @ (fields[index].q.numpy().T @ historical_rows[layer].T)
            )
            manual += float(
                np.sum((baseline + 0.125 * velocity[index] * proposal) ** 2)
            )
        self.assertAlmostEqual(
            observed.problem.historical.value(velocity), manual, places=10
        )
        self.assertEqual(sketch.history_item_count, 10)

    def test_k8_full6_and_55_evaluation_source_contract(self) -> None:
        source = inspect.getsource(_run_ode_arm)
        self.assertIn("for step_index in range(P1R23_GRID_COUNT)", source)
        self.assertIn('"FULL_SIX_FIXED"', source)
        runtime_source = (
            ROOT / "project/run_scripts/ode_bf/historical_h0_sequential_runtime.py"
        ).read_text()
        self.assertIn("batches=stream_batches[:round_index]", runtime_source)
        self.assertIn("if round_index in EVALUATION_ROUNDS", runtime_source)
        self.assertIn("postcommit_cumulative_b10_evaluation_count", runtime_source)
        self.assertNotIn('method == "MEMIT"', runtime_source)

    def test_result_namespace_and_launcher_matrix(self) -> None:
        names = {
            expected_historical_h0_result_name(alias, method)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for method in METHODS
        }
        self.assertEqual(len(names), 10)
        source = (
            ROOT / "project/run_scripts/session05_ode_bf_historical_h0_sequential.sbatch"
        ).read_text()
        self.assertIn("#SBATCH --array=0-7%4", source)
        self.assertIn("#SBATCH --nodelist=devbox", source)
        self.assertNotIn("MEMIT", source)
        self.assertNotIn("ALPHAEDIT", source)

    def test_source_manifest_rehashes_every_member(self) -> None:
        path = LOCKS / "source_manifest_s05_historical_h0_sequential.json"
        value = json.loads(path.read_text())
        root = value.pop("root_digest")
        self.assertEqual(root, canonical_hash(value))
        self.assertEqual(value["entry_count"], len(value["entries"]))
        for item in value["entries"]:
            member = ROOT / item["path"]
            payload = member.read_bytes()
            self.assertEqual(member.stat().st_mode, item["mode"])
            self.assertEqual(len(payload), item["size"])
            self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
            object_id = subprocess.run(
                ["git", "hash-object", str(member)],
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            self.assertEqual(object_id, item["object_id"])


if __name__ == "__main__":
    unittest.main()
