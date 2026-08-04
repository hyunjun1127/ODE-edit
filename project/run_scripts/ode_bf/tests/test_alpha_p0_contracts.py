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
    ALPHA_SOLVE_DTYPE,
    ALPHA_SOLVE_REFERENCE,
    _normalize_requests,
    canonical_alpha_fp32_solve,
    capture_native_and_wb_joint_endpoint,
    validate_joint_alpha_solve_geometry,
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
from project.run_scripts.ode_bf.woodbury import (
    ProjectorCertificate,
    full_projector_certificate,
    solve_alpha_woodbury,
)


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
        counters = {"z": 0, "keys": 0, "io": 0}

        def fake_get_parameter(observed_model: object, name: str) -> torch.nn.Parameter:
            self.assertIs(observed_model, model)
            return names[name]

        def fake_z(*args: object, **kwargs: object) -> torch.Tensor:
            del args, kwargs
            counters["z"] += 1
            leaf = torch.ones((), requires_grad=True)
            torch.autograd.backward(leaf)
            return torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        base_keys = torch.zeros((10, 12), dtype=torch.bfloat16)
        base_keys[:, :10] = torch.eye(10, dtype=torch.bfloat16)

        def fake_keys(*args: object, **kwargs: object) -> torch.Tensor:
            del args, kwargs
            counters["keys"] += 1
            return base_keys.clone()

        def fake_io(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
            del args, kwargs
            counters["io"] += 1
            return (
                torch.zeros((10, 12), dtype=torch.bfloat16),
                torch.zeros((10, 3), dtype=torch.bfloat16),
            )

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
                    model_residual_tolerance=1.0e-5,
                )

        self.assertEqual(counters, {"z": 10, "keys": 5, "io": 5})
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
        self.assertEqual(len(captured.dense_solve_receipts), 5)
        self.assertTrue(
            all(
                receipt.reference == ALPHA_SOLVE_REFERENCE
                and receipt.solve_dtype == str(ALPHA_SOLVE_DTYPE)
                and receipt.input_dtypes[1] == "torch.bfloat16"
                and receipt.joint_key_rank == 10
                and receipt.passed
                for receipt in captured.dense_solve_receipts
            )
        )
        self.assertEqual(
            {name: getattr(alpha_main, name, None) for name in original_globals},
            original_globals,
        )


class CanonicalAlphaSolveTests(unittest.TestCase):
    def _fixture(
        self,
        *,
        dimension: int = 16,
        output_dimension: int = 7,
        general: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(101)
        if general:
            projector = torch.eye(dimension, dtype=torch.float32)
            projector += 0.01 * torch.randn(
                (dimension, dimension),
                generator=generator,
                dtype=torch.float32,
            )
        else:
            basis, _ = torch.linalg.qr(
                torch.randn(
                    (dimension, dimension - 3),
                    generator=generator,
                    dtype=torch.float64,
                )
            )
            projector = (basis @ basis.T).to(dtype=torch.float32)
        keys = torch.randn(
            (dimension, 10),
            generator=generator,
            dtype=torch.float32,
        )
        keys[:10] += torch.eye(10, dtype=torch.float32)
        covariance = 0.02 * torch.eye(dimension, dtype=torch.float32)
        residual = torch.randn(
            (output_dimension, 10),
            generator=generator,
            dtype=torch.float32,
        )
        return projector, keys, covariance, residual

    def test_unadapted_bf16_key_mismatch_reproduces_and_adapter_passes(self) -> None:
        projector, keys, covariance, residual = self._fixture()
        keys_bf16 = (
            keys.to(dtype=torch.bfloat16).T.contiguous().T
        )
        with self.assertRaises(RuntimeError):
            _ = projector @ keys_bf16
        result = canonical_alpha_fp32_solve(
            projector,
            keys_bf16,
            covariance.to(dtype=torch.float64),
            residual.to(dtype=torch.bfloat16),
            layer=4,
            regularization=torch.tensor(1.0, dtype=torch.float64),
            solve_device="cpu",
            residual_tolerance=1.0e-5,
        )
        self.assertEqual(result.update.dtype, torch.float32)
        self.assertEqual(
            result.receipt.input_dtypes,
            (
                "torch.float32",
                "torch.bfloat16",
                "torch.float64",
                "torch.bfloat16",
            ),
        )
        self.assertEqual(set(result.receipt.normalized_dtypes), {"torch.float32"})
        self.assertTrue(result.receipt.passed)

    def test_locked_model_geometries_are_joint_rank10_not_singletons(self) -> None:
        llama = validate_joint_alpha_solve_geometry(
            (14_336, 14_336),
            (14_336, 10),
            (14_336, 14_336),
            (4_096, 10),
        )
        qwen = validate_joint_alpha_solve_geometry(
            (18_944, 18_944),
            (18_944, 10),
            (18_944, 18_944),
            (3_584, 10),
        )
        self.assertEqual(llama.output_shape, (14_336, 4_096))
        self.assertEqual(qwen.output_shape, (18_944, 3_584))
        with self.assertRaisesRegex(ODEBFContractError, "joint B10"):
            validate_joint_alpha_solve_geometry(
                (14_336, 14_336),
                (14_336, 1),
                (14_336, 14_336),
                (4_096, 1),
            )

    def test_float64_oracle_certificate_and_input_rng_alias_purity(self) -> None:
        projector, keys, covariance, residual = self._fixture(general=True)
        keys_bf16 = keys.to(dtype=torch.bfloat16).requires_grad_(True)
        tensors = (projector, keys_bf16, covariance, residual)
        pointers = tuple(value.data_ptr() for value in tensors)
        versions = tuple(value._version for value in tensors)
        rng = torch.get_rng_state().clone()
        result = canonical_alpha_fp32_solve(
            projector,
            keys_bf16,
            covariance,
            residual,
            layer=5,
            regularization=1.25,
            solve_device="cpu",
            residual_tolerance=1.0e-5,
        )
        p64 = projector.double()
        k64 = keys_bf16.detach().double()
        c64 = covariance.double()
        r64 = residual.double()
        a64 = p64 @ (k64 @ k64.T + c64) + 1.25 * torch.eye(
            projector.shape[0],
            dtype=torch.float64,
        )
        b64 = p64 @ k64 @ r64.T
        oracle = torch.linalg.solve(a64, b64)
        self.assertTrue(
            torch.allclose(
                result.update.double(),
                oracle,
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        )
        self.assertIsNotNone(result.receipt.condition_estimate)
        self.assertLessEqual(result.receipt.relative_residual, 1.0e-5)
        self.assertEqual(tuple(value.data_ptr() for value in tensors), pointers)
        self.assertEqual(tuple(value._version for value in tensors), versions)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertIsNone(keys_bf16.grad)

    def test_all_fp32_adapter_matches_pinned_source_order_exactly(self) -> None:
        projector, keys, covariance, residual = self._fixture()
        source_a = projector @ (keys @ keys.T + covariance)
        source_a = source_a + 1.0 * torch.eye(keys.shape[0], dtype=torch.float32)
        source_b = projector @ keys @ residual.T
        source_update = torch.linalg.solve(source_a, source_b)
        adapted = canonical_alpha_fp32_solve(
            projector,
            keys,
            covariance,
            residual,
            layer=6,
            regularization=1.0,
            solve_device="cpu",
            residual_tolerance=1.0e-5,
        )
        self.assertTrue(torch.equal(adapted.update, source_update))
        self.assertEqual(
            adapted.receipt.normalized_storage_reused,
            (True, True, True, True),
        )

    def test_dense_matches_exact_and_general_woodbury_certificates(self) -> None:
        for general in (False, True):
            projector, keys, _, residual = self._fixture(general=general)
            covariance = torch.zeros_like(projector)
            dense = canonical_alpha_fp32_solve(
                projector,
                keys,
                covariance,
                residual,
                layer=7,
                regularization=1.0,
                solve_device="cpu",
                residual_tolerance=1.0e-5,
            )
            if general:
                certificate = ProjectorCertificate(
                    "b" * 64,
                    0.1,
                    0.1,
                    "artifact-unverified",
                    1.0e-10,
                )
            else:
                certificate = full_projector_certificate(
                    projector,
                    source_sha256="a" * 64,
                    tolerance=1.0e-5,
                )
            wb = solve_alpha_woodbury(
                projector,
                keys,
                history_keys=None,
                regularization=1.0,
                projector_certificate=certificate,
                residual_tolerance=1.0e-5,
            )
            wb_update = wb.q @ residual.double().T
            self.assertTrue(wb.certificate.passed)
            self.assertTrue(
                torch.allclose(
                    dense.update.double(),
                    wb_update,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
            )

    def test_nonfinite_and_rank_one_inputs_fail_closed(self) -> None:
        projector, keys, covariance, residual = self._fixture()
        bad = keys.clone()
        bad[0, 0] = float("nan")
        with self.assertRaisesRegex(ODEBFContractError, "non-finite"):
            canonical_alpha_fp32_solve(
                projector,
                bad,
                covariance,
                residual,
                layer=8,
                regularization=1.0,
                solve_device="cpu",
                residual_tolerance=1.0e-5,
            )
        rank_one = torch.ones_like(keys)
        with self.assertRaisesRegex(ODEBFContractError, "certificate"):
            canonical_alpha_fp32_solve(
                projector,
                rank_one,
                covariance,
                residual,
                layer=8,
                regularization=1.0,
                solve_device="cpu",
                residual_tolerance=1.0e-5,
            )


class P0ReceiptTests(unittest.TestCase):
    def test_result_namespace_and_outcome_sealing(self) -> None:
        self.assertEqual(
            expected_result_name("llama3-8b-inst"),
            "s04-p0-native-wb-b10-r1-llama3-8b-inst-23fe5621",
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
