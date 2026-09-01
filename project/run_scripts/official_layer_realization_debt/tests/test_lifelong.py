from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from project.run_scripts.official_layer_realization_debt.contracts import Method
from project.run_scripts.official_layer_realization_debt.lifelong_journal import (
    EditableStateSnapshot,
    extend_hash_chain,
    restore_checkpoint,
    save_checkpoint,
    write_json_once,
)
from project.run_scripts.official_layer_realization_debt.lifelong_metrics import (
    action_realization_profiles,
    rank_concordance,
    sustained_onset,
)
from project.run_scripts.official_layer_realization_debt.lifelong_probe import (
    _minimum_action,
)
from project.run_scripts.official_layer_realization_debt.lifelong_stream import (
    verify_existing_1k_prefix,
    verify_lifelong_stream,
)
from project.run_scripts.official_layer_realization_debt.metrics import (
    weight_action_energy,
)
from project.run_scripts.official_layer_realization_debt.observer import (
    OfficialLayerObserver,
)


class _FakeCacheModule:
    def __init__(self) -> None:
        self.COV_CACHE = {("model", "layer"): torch.eye(2)}


class _ComputeModule:
    def compute_z(self, model, value):
        return torch.tensor([float(value)], device=next(model.parameters()).device)

    def get_module_input_output_at_words(self, *args, **kwargs):
        return torch.zeros(1, 1)


class LifelongFocusedTests(unittest.TestCase):
    def test_primary_update_profile_uses_magnitude_not_squared(self) -> None:
        touched = {
            f"layer.{index}": torch.nn.Parameter(torch.tensor([float(index + 1)]))
            for index in range(5)
        }
        originals = {
            name: torch.zeros_like(parameter) for name, parameter in touched.items()
        }
        action = weight_action_energy(touched, originals)
        self.assertEqual(
            [round(row["update_magnitude_share"], 6) for row in action["layers"]],
            [round(value / 15.0, 6) for value in (1, 2, 3, 4, 5)],
        )
        self.assertNotEqual(
            [row["update_magnitude_share"] for row in action["layers"]],
            [row["energy_share"] for row in action["layers"]],
        )

    def test_action_realization_keeps_observation_units_separate(self) -> None:
        debt = {
            "records": [
                {
                    "request_sha256": "a" * 64,
                    "layers": [
                        {"layer": layer, "rho": 1.0, "allocation_norm_over_R1": 0.2}
                        for layer in (4, 5, 6, 7, 8)
                    ],
                }
            ]
        }
        action = {
            "layers": [
                {"layer": layer, "frobenius_magnitude": float(index + 1)}
                for index, layer in enumerate((4, 5, 6, 7, 8))
            ]
        }
        value = action_realization_profiles(debt, action)
        self.assertEqual(value["request_as_independent_weight_sample_count"], 0)
        self.assertEqual(value["primary_update_quantity"], "frobenius_magnitude")
        self.assertEqual(value["request_count"], 1)

    def test_registered_completion_minimum_action_and_unreachable(self) -> None:
        r = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        responses = (
            torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        )
        value = _minimum_action(r, responses, (1.0, 4.0))
        self.assertAlmostEqual(value["minimum_expected_completion_action"], 8.5)
        self.assertAlmostEqual(value["unreachable_fraction"], 0.0, places=12)
        self.assertEqual(value["effective_rank"], 2)

    def test_editable_state_rollback_exact(self) -> None:
        fake = _FakeCacheModule()
        touched = {"weight": torch.nn.Parameter(torch.tensor([1.0, 2.0]))}
        with mock.patch(
            "project.run_scripts.official_layer_realization_debt.lifelong_journal._method_module",
            return_value=fake,
        ):
            state = EditableStateSnapshot.capture(Method.MEMIT, touched)
            pointer = int(touched["weight"].data_ptr())
            with torch.no_grad():
                touched["weight"].add_(9.0)
            restore = state.restore(touched)
        self.assertTrue(restore["weights"]["exact"])
        self.assertEqual(int(touched["weight"].data_ptr()), pointer)
        self.assertTrue(torch.equal(touched["weight"], torch.tensor([1.0, 2.0])))

    def test_observer_fixed_z_replay_does_not_call_optimizer(self) -> None:
        module = _ComputeModule()
        model = torch.nn.Linear(1, 1)
        observer = OfficialLayerObserver(
            method=Method.MEMIT,
            request_sha256=("b" * 64,),
            capture_layers=False,
            replay_z=(torch.tensor([7.0]),),
        )
        with observer.observe(module):
            value = module.compute_z(model, 1.0)
        payload = observer.payload(module)
        self.assertTrue(torch.equal(value.cpu(), torch.tensor([7.0])))
        self.assertEqual(payload["direct_z_optimizer_compute_count"], 0)
        self.assertEqual(payload["direct_z_shared_replay_count"], 1)

    def test_journal_create_once_and_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.json"
            digest = write_json_once(path, {"value": 1})
            self.assertEqual(json.loads(path.read_text())["value"], 1)
            with self.assertRaises(FileExistsError):
                write_json_once(path, {"value": 2})
            self.assertNotEqual(extend_hash_chain("0" * 64, digest, 1), "0" * 64)

    def test_checkpoint_is_weights_only_recoverable(self) -> None:
        fake = _FakeCacheModule()
        touched = {"weight": torch.nn.Parameter(torch.tensor([1.0, 2.0]))}
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "project.run_scripts.official_layer_realization_debt.lifelong_journal._method_module",
            return_value=fake,
        ), mock.patch.object(torch.cuda, "get_rng_state_all", return_value=[]), mock.patch.object(
            torch.cuda, "is_available", return_value=False
        ):
            path = Path(directory) / "state.pt"
            saved = save_checkpoint(
                path=path,
                method=Method.MEMIT,
                touched=touched,
                batch_index=1,
                accepted_edit_count=100,
                stream_root="a" * 64,
                order_root="b" * 64,
                journal_chain_root="c" * 64,
                source={"head": "d" * 40, "tree": "e" * 40},
                evaluator_identity="f" * 64,
            )
            with torch.no_grad():
                touched["weight"].add_(7.0)
            recovered = restore_checkpoint(
                path, method=Method.MEMIT, touched=touched
            )
        self.assertEqual(saved["accepted_edit_count"], 100)
        self.assertTrue(recovered["rng_restore_exact"])
        self.assertTrue(torch.equal(touched["weight"], torch.tensor([1.0, 2.0])))

    def test_sustained_failure_and_concordance(self) -> None:
        rows = [
            {"batch_index": index, "value": value}
            for index, value in enumerate((0.0, 2.0, 3.0, 4.0), start=1)
        ]
        self.assertEqual(
            sustained_onset(rows, field="value", threshold=1.0, consecutive=3), 2
        )
        concordance = rank_concordance([1, 2, 3], [2, 4, 6])
        self.assertAlmostEqual(concordance["spearman"], 1.0)
        self.assertAlmostEqual(concordance["kendall"], 1.0)

    def test_real_seal_verifies_and_extends_exact_1k(self) -> None:
        seal_path = Path(
            "/data/janghj/ODE-edit/local/state/official-layer-realization-debt-lifelong-b100-v1/stream-seal-v1.json"
        )
        if not seal_path.exists():
            self.skipTest("local outcome-free stream seal is not built")
        seal = verify_lifelong_stream(json.loads(seal_path.read_text()))
        prefix = Path(__file__).resolve().parents[2] / (
            "ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json"
        )
        receipt = verify_existing_1k_prefix(seal, prefix)
        self.assertEqual(receipt["status"], "EXACT_1K_PREFIX_PASS")


if __name__ == "__main__":
    unittest.main()
