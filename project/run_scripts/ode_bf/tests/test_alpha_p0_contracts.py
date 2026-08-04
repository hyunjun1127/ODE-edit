from __future__ import annotations

import copy
import tempfile
import threading
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch

from project.run_scripts.ode_bf.accounting import ComputeLedger
from project.run_scripts.ode_bf.alpha_backend import (
    _normalize_requests,
    capture_native_and_wb_joint_endpoint,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p0_runtime import (
    _evaluation_payload,
    expected_result_name,
)
from project.run_scripts.ode_bf.benchmark import (
    CounterFactRequestScore,
    counterfact_batch_receipt,
)
from project.run_scripts.ode_bf.evaluator import ModelEvaluationReceipt


class TinyAlphaModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.touched = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros((3, 12), dtype=torch.bfloat16),
                    requires_grad=False,
                )
                for _ in range(5)
            ]
        )


def _requests() -> list[dict[str, object]]:
    return [
        {
            "case_id": index,
            "request_sha256": f"{index + 1:064x}",
            "prompt": "{} relation",
            "subject": f"entity-{index}",
            "target_new": "new",
        }
        for index in range(10)
    ]


class AlphaJointCaptureTests(unittest.TestCase):
    def test_normalization_requires_one_distinct_joint_batch(self) -> None:
        normalized = _normalize_requests(_requests())
        self.assertEqual(len(normalized), 10)
        self.assertTrue(all(str(item["target_new"]).startswith(" ") for item in normalized))
        with self.assertRaisesRegex(ODEBFContractError, "joint B10"):
            _normalize_requests(_requests()[:1])
        duplicated = _requests()
        duplicated[-1]["request_sha256"] = duplicated[0]["request_sha256"]
        with self.assertRaisesRegex(ODEBFContractError, "duplicate"):
            _normalize_requests(duplicated)

    def test_pinned_native_is_called_once_and_underlying_keys_once_per_layer(self) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Importing from timm.models.hub is deprecated.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Can't initialize NVML",
                category=UserWarning,
            )
            from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main
            from easyeditor.util import nethook

        model = TinyAlphaModel()
        hparams = types.SimpleNamespace(
            layers=[4, 5, 6, 7, 8],
            rewrite_module_tmp="layer{}",
            layer_module_tmp="layer{}",
            fact_token="subject_last",
            L2=2.0,
        )
        names = {
            f"layer{layer}.weight": model.touched[index]
            for index, layer in enumerate(hparams.layers)
        }
        counters = {"execute": 0, "z": 0, "keys": 0, "io": 0}

        def fake_get_parameter(observed_model: object, name: str) -> torch.nn.Parameter:
            self.assertIs(observed_model, model)
            return names[name]

        def fake_z(*args: object, **kwargs: object) -> torch.Tensor:
            del args, kwargs
            counters["z"] += 1
            leaf = torch.ones((), requires_grad=True)
            torch.autograd.backward(leaf)
            return torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        base_keys = torch.zeros((10, 12), dtype=torch.float32)
        base_keys[:, :10] = torch.eye(10, dtype=torch.float32)

        def fake_keys(*args: object, **kwargs: object) -> torch.Tensor:
            del args, kwargs
            counters["keys"] += 1
            return base_keys.clone()

        def fake_io(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            del args, kwargs
            counters["io"] += 1
            return torch.zeros((10, 12)), torch.zeros((10, 3))

        def fake_execute(
            observed_model: object,
            tokenizer: object,
            requests: list[dict[str, object]],
            observed_hparams: object,
            cache_template: object = None,
        ) -> dict[str, torch.Tensor]:
            del tokenizer, cache_template
            self.assertIs(observed_model, model)
            self.assertEqual(len(requests), 10)
            self.assertIs(observed_hparams, hparams)
            counters["execute"] += 1
            direct = [alpha_main.compute_z() for _ in requests]
            zs = torch.stack(direct, dim=1)
            entry = {name: value.detach().clone() for name, value in names.items()}
            deltas: dict[str, torch.Tensor] = {}
            for index, layer in enumerate(hparams.layers):
                keys = alpha_main.compute_ks(model, None, requests, hparams, layer, []).T
                current = alpha_main.get_module_input_output_at_words()[1].T
                residual = (zs - current) / (len(hparams.layers) - index)
                p = alpha_main.P[index]
                system = p @ (keys @ keys.T + alpha_main.cache_c[index])
                system = system + hparams.L2 * torch.eye(keys.shape[0])
                update = torch.linalg.solve(system, p @ keys @ residual.T).T
                name = f"layer{layer}.weight"
                with torch.no_grad():
                    names[name].copy_((names[name].float() + update.float()).bfloat16())
                deltas[name] = update.detach().cpu()
            for layer in hparams.layers:
                alpha_main.compute_ks(model, None, requests, hparams, layer, [])
            with torch.no_grad():
                for name, value in names.items():
                    value.copy_(entry[name])
            return deltas

        original_globals = {
            name: copy.copy(getattr(alpha_main, name, None))
            for name in ("P_loaded", "P_loaded_from", "cache_c_new")
        }
        with tempfile.TemporaryDirectory() as directory:
            projector_path = Path(directory) / "projector.pt"
            torch.save(torch.eye(12).repeat(5, 1, 1), projector_path)
            ledger = ComputeLedger()
            with (
                mock.patch.object(nethook, "get_parameter", side_effect=fake_get_parameter),
                mock.patch.object(alpha_main, "compute_z", side_effect=fake_z),
                mock.patch.object(alpha_main, "compute_ks", side_effect=fake_keys),
                mock.patch.object(
                    alpha_main,
                    "get_module_input_output_at_words",
                    side_effect=fake_io,
                ),
                mock.patch.object(alpha_main, "execute_AlphaEdit", side_effect=fake_execute),
            ):
                captured = capture_native_and_wb_joint_endpoint(
                    model,
                    object(),
                    _requests(),
                    hparams,
                    projector_path,
                    [["{}"]],
                    projector_sha256="a" * 64,
                    mutation_lock=threading.RLock(),
                    ledger=ledger,
                    model_residual_tolerance=1.0e-8,
                )

        self.assertEqual(counters, {"execute": 1, "z": 10, "keys": 5, "io": 5})
        self.assertEqual(captured.initialization.direct_z_initializations, 10)
        self.assertEqual(captured.initialization.shared_key_computes, 5)
        self.assertTrue(
            all(
                torch.equal(captured.native_candidates[name], captured.wb_candidates[name])
                for name in names
            )
        )
        self.assertTrue(
            all(tensor_sha256(value) == captured.entry_sha256[name] for name, value in names.items())
        )
        self.assertGreater(ledger.counters["target_backward"], 0)
        self.assertEqual(
            {name: getattr(alpha_main, name, None) for name in original_globals},
            original_globals,
        )


class P0ReceiptTests(unittest.TestCase):
    def test_result_namespace_and_outcome_sealing(self) -> None:
        self.assertEqual(
            expected_result_name("llama3-8b-inst"),
            "s04-p0-native-wb-b10-llama3-8b-inst-23fe5621",
        )
        scores = tuple(CounterFactRequestScore(0.1, 0.2) for _ in range(10))
        receipt = ModelEvaluationReceipt(
            counterfact_batch_receipt(scores),
            "a" * 64,
            "b" * 64,
            (2,) * 10,
            10,
            20,
            0,
            scores,
        )
        payload = _evaluation_payload(receipt)
        self.assertIn("batch_success_sha256", payload)
        self.assertNotIn("batch_success", payload)
        self.assertNotIn("request_success_vector", str(payload))


if __name__ == "__main__":
    unittest.main()
