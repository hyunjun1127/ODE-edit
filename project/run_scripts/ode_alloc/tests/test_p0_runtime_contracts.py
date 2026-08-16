from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

from project.run_scripts.ode_alloc.contracts import ODEAllocContractError, canonical_json
from project.run_scripts.ode_alloc.gauge import FactorPair
from project.run_scripts.ode_alloc.p0_runtime import (
    DIAGNOSTIC_STAGES,
    DiagnosticStageRecorder,
    _allowlisted_exception_frames,
    _atomic_diagnostic_write_once,
    _canonical_write_once,
    expected_result_name,
    tensor_sha256,
    write_diagnostic_failure_once,
)


class P0RuntimeContractTests(unittest.TestCase):
    def test_output_name_is_lock_hash_bound_and_alias_closed(self) -> None:
        digest = "a" * 64
        self.assertEqual(
            expected_result_name("llama3-8b-inst", digest, "r3"),
            "s04-p0-native-identity-r3-llama3-8b-inst-aaaaaaaa",
        )
        with self.assertRaises(ODEAllocContractError):
            expected_result_name("foreign", digest, "r3")
        with self.assertRaises(ODEAllocContractError):
            expected_result_name("llama3-8b-inst", digest, "r2")

    def test_tensor_hash_is_layout_stable_and_value_sensitive(self) -> None:
        value = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
        self.assertEqual(tensor_sha256(value), tensor_sha256(value.clone()))
        noncontiguous = value.T.contiguous().T
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertTrue(torch.equal(value, noncontiguous))
        self.assertEqual(tensor_sha256(value), tensor_sha256(noncontiguous))
        changed = value.clone()
        changed[0, 0] = 7
        self.assertNotEqual(tensor_sha256(value), tensor_sha256(changed))

    @staticmethod
    def _reference_tensor_hash(value: torch.Tensor) -> str:
        detached = value.detach()
        digest = hashlib.sha256()
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(canonical_json(tuple(detached.shape)).encode("ascii"))
        logical = detached.contiguous().to(device="cpu").reshape(-1)
        digest.update(logical.view(torch.uint8).numpy().tobytes(order="C"))
        return digest.hexdigest()

    def test_factor_shape_stride_reproduction_and_reference_bytes(self) -> None:
        for key_rows, residual_rows in (
            (14336, 4096),
            (18944, 3584),
        ):
            key = torch.arange(key_rows, dtype=torch.float64).reshape(1, key_rows).T
            residual = torch.arange(residual_rows, dtype=torch.float64).reshape(
                1, residual_rows
            ).T
            self.assertEqual(tuple(key.shape), (key_rows, 1))
            self.assertEqual(tuple(residual.shape), (residual_rows, 1))
            self.assertEqual(key.dtype, torch.float64)
            self.assertEqual(residual.dtype, torch.float64)
            self.assertTrue(key.is_contiguous())
            self.assertNotEqual(key.stride()[-1], 1)
            with self.assertRaises(RuntimeError):
                key.contiguous().to(device="cpu").view(torch.uint8)
            self.assertEqual(tensor_sha256(key), self._reference_tensor_hash(key))
            self.assertEqual(
                tensor_sha256(residual), self._reference_tensor_hash(residual)
            )
            key_standard = torch.arange(key_rows, dtype=torch.float64).reshape(
                key_rows, 1
            )
            self.assertEqual(key_standard.stride()[-1], 1)
            self.assertEqual(tensor_sha256(key), tensor_sha256(key_standard))

    def test_tensor_hash_dtype_empty_zero_and_nonfinite_contract(self) -> None:
        for dtype in (torch.float64, torch.float32, torch.bfloat16):
            finite = torch.tensor([0.0, -0.0, 1.5, -2.0], dtype=dtype).reshape(2, 2)
            self.assertEqual(tensor_sha256(finite), self._reference_tensor_hash(finite))
            self.assertEqual(tensor_sha256(finite), tensor_sha256(finite.clone()))
            empty = torch.empty((3, 0), dtype=dtype)
            self.assertEqual(tensor_sha256(empty), self._reference_tensor_hash(empty))
            zero = torch.zeros((), dtype=dtype)
            self.assertEqual(tensor_sha256(zero), self._reference_tensor_hash(zero))

        nonfinite = torch.tensor([float("nan"), float("inf"), -float("inf")])
        self.assertEqual(tensor_sha256(nonfinite), tensor_sha256(nonfinite.clone()))
        with self.assertRaises(ODEAllocContractError):
            FactorPair(0, nonfinite.reshape(3, 1), torch.ones(2, 1))

    def test_tensor_hash_preserves_pointer_version_grad_and_rng(self) -> None:
        value = (
            torch.arange(24, dtype=torch.float64)
            .reshape(4, 6)
            .clone()
            .detach()
            .requires_grad_(True)
        )
        value.grad = torch.ones_like(value)
        pointer = value.data_ptr()
        version = value._version
        grad_pointer = value.grad.data_ptr()
        grad_copy = value.grad.clone()
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state().clone()

        first = tensor_sha256(value)
        second = tensor_sha256(value)

        self.assertEqual(first, second)
        self.assertEqual(value.data_ptr(), pointer)
        self.assertEqual(value._version, version)
        self.assertIsNotNone(value.grad)
        self.assertEqual(value.grad.data_ptr(), grad_pointer)
        self.assertTrue(torch.equal(value.grad, grad_copy))
        self.assertEqual(random.getstate(), python_rng)
        observed_numpy = np.random.get_state()
        self.assertEqual(observed_numpy[0], numpy_rng[0])
        self.assertTrue(np.array_equal(observed_numpy[1], numpy_rng[1]))
        self.assertEqual(observed_numpy[2:], numpy_rng[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))

    def test_terminal_metadata_is_create_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            first = _canonical_write_once(path, {"status": "PASS"})
            self.assertEqual(len(first), 64)
            with self.assertRaises(FileExistsError):
                _canonical_write_once(path, {"status": "REWRITE"})

    def test_atomic_diagnostic_write_cleans_temporary_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "receipt.json"
            with mock.patch(
                "project.run_scripts.ode_alloc.p0_runtime.os.link",
                side_effect=OSError("injected publish failure"),
            ):
                with self.assertRaises(OSError):
                    _atomic_diagnostic_write_once(path, {"stage": "test"})
            self.assertFalse(path.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_stage_recorder_is_ordered_create_once_and_state_invariant(self) -> None:
        self.assertEqual(
            DIAGNOSTIC_STAGES,
            (
                "post_execute_memit",
                "post_factor_hash",
                "post_factor_contract",
                "pre_direct_z_score",
                "post_direct_z_score",
                "pre_native_writer",
                "post_native_writer",
                "pre_q0_functional",
                "post_q0_functional",
                "post_q0_commit",
                "post_identity_verdict",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            model = torch.nn.Linear(3, 2)
            parameter = next(model.parameters())
            pointer = parameter.data_ptr()
            version = parameter._version
            state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            torch_rng = torch.get_rng_state().clone()

            recorder = DiagnosticStageRecorder(raw)
            receipt = recorder.complete(DIAGNOSTIC_STAGES[0])

            self.assertEqual(len(receipt), 64)
            self.assertGreater(recorder.wall_seconds, 0.0)
            self.assertEqual(recorder.last_completed_stage, DIAGNOSTIC_STAGES[0])
            self.assertEqual(parameter.data_ptr(), pointer)
            self.assertEqual(parameter._version, version)
            self.assertIsNone(parameter.grad)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, state[name]))
            self.assertEqual(random.getstate(), python_rng)
            observed_numpy = np.random.get_state()
            self.assertEqual(observed_numpy[0], numpy_rng[0])
            self.assertTrue(np.array_equal(observed_numpy[1], numpy_rng[1]))
            self.assertEqual(observed_numpy[2:], numpy_rng[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            with self.assertRaises(ODEAllocContractError):
                recorder.complete(DIAGNOSTIC_STAGES[2])
            with self.assertRaises(FileExistsError):
                DiagnosticStageRecorder(raw)

    def test_structural_receipt_contains_only_raw_free_tensor_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            raw.mkdir()
            recorder = DiagnosticStageRecorder(raw)
            digest = recorder.write_structure_once(
                {"opaque-weight-name": (torch.ones(3, 1), torch.ones(2, 1))}
            )
            self.assertEqual(len(digest), 64)
            path = raw / "diagnostic" / "factor_structure.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(value),
                {"schema_version", "layer_count", "mapping_keyset_sha256", "factors"},
            )
            self.assertEqual(value["layer_count"], 1)
            self.assertNotIn("opaque-weight-name", path.read_text(encoding="utf-8"))
            factor = value["factors"][0]
            self.assertEqual(set(factor), {"ordinal", "rank", "key", "residual"})
            self.assertEqual(
                set(factor["key"]),
                {"shape", "dtype", "device_class", "finite", "sha256"},
            )

    def test_diagnostic_failure_is_allowlisted_raw_free_and_create_once(self) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "result"
            raw = output_root / "raw"
            raw.mkdir(parents=True)
            recorder = DiagnosticStageRecorder(raw)
            recorder.complete(DIAGNOSTIC_STAGES[0])
            try:
                raise RuntimeError("never serialize this message")
            except RuntimeError as exc:
                frames = _allowlisted_exception_frames(
                    exc, repo_root=repo_root, easyedit_root=repo_root
                )
                self.assertTrue(frames)
                self.assertEqual(set(frames[-1]), {"file", "function", "line"})
                digest = write_diagnostic_failure_once(
                    output_root,
                    exc,
                    repo_root=repo_root,
                    easyedit_root=repo_root,
                )
                self.assertIsNone(
                    write_diagnostic_failure_once(
                        output_root,
                        exc,
                        repo_root=repo_root,
                        easyedit_root=repo_root,
                    )
                )
            self.assertIsNotNone(digest)
            path = output_root / "diagnostic_failure.json"
            encoded = path.read_text(encoding="utf-8")
            self.assertNotIn("never serialize this message", encoded)
            value = json.loads(encoded)
            self.assertEqual(value["last_completed_stage"], DIAGNOSTIC_STAGES[0])
            self.assertEqual(value["exception_type"], "RuntimeError")
            self.assertEqual(value["scientific_outcome_count"], 0)
            self.assertEqual(
                set(value["allowlisted_frames"][-1]), {"file", "function", "line"}
            )

    def test_easyedit_frame_is_relative_and_source_free(self) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        with tempfile.TemporaryDirectory() as directory:
            easyedit_root = Path(directory)
            source = easyedit_root / "easyeditor" / "models" / "memit" / "fixture.py"
            source.parent.mkdir(parents=True)
            namespace: dict[str, object] = {}
            exec(
                compile(
                    "def fail():\n    raise RuntimeError('raw message')\n",
                    str(source),
                    "exec",
                ),
                namespace,
            )
            try:
                namespace["fail"]()
            except RuntimeError as exc:
                frames = _allowlisted_exception_frames(
                    exc, repo_root=repo_root, easyedit_root=easyedit_root
                )
            easyedit_frames = [
                frame for frame in frames if frame["file"].startswith("easyedit/")
            ]
            self.assertEqual(len(easyedit_frames), 1)
            self.assertEqual(
                easyedit_frames[0]["file"],
                "easyedit/easyeditor/models/memit/fixture.py",
            )
            self.assertEqual(set(easyedit_frames[0]), {"file", "function", "line"})


if __name__ == "__main__":
    unittest.main()
