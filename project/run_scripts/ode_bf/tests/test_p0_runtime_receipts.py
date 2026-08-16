from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from project.run_scripts.ode_bf.accounting import (
    JointInitializationReceipt,
    LayerFactorReceipt,
)
from project.run_scripts.ode_bf.alpha_backend import (
    CapturedNativeWBEndpoint,
    FOUR_PATH_ORDER,
    FourPathLayerReceipt,
    FourPathSolveReceipt,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p0_runtime import (
    StageRecorder,
    _atomic_write_once,
    _post_joint_factor_contract_payload,
    classify_four_path_diagnostic,
    classify_w64_technical_candidate,
    expected_result_name,
    write_failure_once,
)


class P0RuntimeReceiptTests(unittest.TestCase):
    @staticmethod
    def _captured_endpoint_fixture() -> CapturedNativeWBEndpoint:
        order_sha256 = "a" * 64
        factor_receipts = tuple(
            LayerFactorReceipt(
                layer=layer,
                key_shape=(4, 10),
                residual_shape=(3, 10),
                numerical_rank=10,
                dtype="torch.float32",
                device_class="cpu",
            )
            for layer in range(2)
        )
        initialization = JointInitializationReceipt(
            editor_invocations=1,
            request_count=10,
            direct_z_initializations=10,
            target_initializations=10,
            hidden_singleton_editor_calls=0,
            shared_key_computes=2,
            covariance_reads=0,
            projector_reads=2,
            native_factor_computes=2,
            covariance_decision_enabled=False,
            factors=factor_receipts,
            request_order_sha256=order_sha256,
        )
        layers = []
        for layer in range(2):
            path_receipts = tuple(
                FourPathSolveReceipt(
                    layer=layer,
                    path=path,
                    reference="fixture-source-order",
                    source_identity_sha256="b" * 64,
                    source_hashes=(("P", "c" * 64),),
                    source_shapes=(("P", (4, 4)),),
                    source_dtypes=(("P", "torch.float32"),),
                    solve_shape=(4, 10),
                    update_shape=(3, 4),
                    solve_dtype=(
                        "torch.float64" if path == "W64" else "torch.float32"
                    ),
                    assembler_dtype="torch.float32",
                    solve_device_class="cpu",
                    joint_rank=10,
                    residual_rhs_kind="thin_rhs_G",
                    normalized_backward_residual=0.0,
                    system_norm_estimate=1.0,
                    system_norm_kind="fixture",
                    condition_estimate=1.0,
                    condition_kind="fixture",
                    factor_relative_error_to_d32=0.0,
                    update_relative_error_to_d32=0.0,
                    endpoint_relative_to_n32_edit_error=0.0,
                    finite=True,
                    certificate_passed=True,
                    wall_seconds=0.0,
                    gpu_seconds=0.0,
                    allocated_bytes_after=0,
                )
                for path in FOUR_PATH_ORDER
            )
            layers.append(
                FourPathLayerReceipt(
                    layer=layer,
                    weight_name_sha256="d" * 64,
                    request_order_sha256=order_sha256,
                    source_identity_sha256="b" * 64,
                    path_receipts=path_receipts,
                    comparisons_to_n32=(),
                    pairwise_comparisons=(),
                    adjacent_comparisons=(),
                    first_adjacent_byte_boundary=None,
                )
            )
        return CapturedNativeWBEndpoint(
            native_candidates={},
            wb_candidates={},
            wb_factors={},
            path_candidates={},
            path_factors={},
            four_path_layer_receipts=tuple(layers),
            entry_weights={},
            entry_sha256={},
            direct_z_sha256=tuple(f"{index:064x}" for index in range(10)),
            key_sha256_by_layer=tuple((layer, "e" * 64) for layer in range(2)),
            dense_solve_receipts=(),
            woodbury_certificates=(),
            initialization=initialization,
            target_backward_count=10,
        )

    def test_actual_four_path_receipt_serializes_through_post_capture_stage(self) -> None:
        captured = self._captured_endpoint_fixture()
        payload = _post_joint_factor_contract_payload(captured)
        self.assertEqual(payload["prospective_path"]["solve_device_classes"], ["cpu"])
        self.assertNotIn("device_class", payload["prospective_path"])
        with tempfile.TemporaryDirectory() as directory:
            recorder = StageRecorder(Path(directory))
            recorder.record("post_joint_factor_contract", payload)
            recorder.record(
                "post_four_path_construction",
                {
                    "path_order": list(FOUR_PATH_ORDER),
                    "layers": [
                        asdict(item) for item in captured.four_path_layer_receipts
                    ],
                },
            )
            first = json.loads(
                (Path(directory) / "stage-01-post_joint_factor_contract.json").read_text(
                    encoding="utf-8"
                )
            )
            second = json.loads(
                (Path(directory) / "stage-02-post_four_path_construction.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(
            first["payload"]["prospective_path"]["solve_device_classes"],
            ["cpu"],
        )
        self.assertEqual(second["payload"]["layers"][0]["path_receipts"][0]["solve_device_class"], "cpu")

    def test_post_capture_receipt_rejects_loose_schema(self) -> None:
        captured = self._captured_endpoint_fixture()
        captured.four_path_layer_receipts = (
            SimpleNamespace(
                path_receipts=(SimpleNamespace(device_class="cpu"),),
            ),
        )
        with self.assertRaisesRegex(ODEBFContractError, "schema"):
            _post_joint_factor_contract_payload(captured)

    def test_runner_has_no_legacy_four_path_device_class_access(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "p0_runtime.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertFalse(
            [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr == "device_class"
            ]
        )

    def test_w64_candidate_ignores_w32_diagnostic_failure_without_fallback(self) -> None:
        common = {
            "source_inputs_identical": True,
            "required_paths_finite": True,
            "required_certificates_pass": True,
            "n32_w64_benchmark_bits_exact": True,
            "n32_w64_decisions_exact": True,
            "strict_w64_virtual_commit_exact": True,
            "rollback_and_restore_exact": True,
            "boundary_touched": False,
            "request_and_span_parity": True,
        }
        self.assertEqual(
            classify_w64_technical_candidate(**common),
            "W64_TECHNICAL_CANDIDATE_NO_NATIVE_EQUIVALENCE_CLAIM",
        )
        for gate in (
            "source_inputs_identical",
            "required_paths_finite",
            "required_certificates_pass",
            "n32_w64_benchmark_bits_exact",
            "n32_w64_decisions_exact",
            "strict_w64_virtual_commit_exact",
            "rollback_and_restore_exact",
            "request_and_span_parity",
        ):
            failed = dict(common)
            failed[gate] = False
            self.assertEqual(
                classify_w64_technical_candidate(**failed),
                "W64_TECHNICAL_HARD_GATE_FAIL",
            )
        touched = dict(common)
        touched["boundary_touched"] = True
        self.assertEqual(
            classify_w64_technical_candidate(**touched),
            "W64_TECHNICAL_HARD_GATE_FAIL",
        )

    def test_four_path_classification_is_predeclared_and_fail_closed(self) -> None:
        common = {
            "source_inputs_identical": True,
            "all_finite": True,
            "all_certificates_pass": True,
            "benchmark_bits_exact": True,
            "decisions_exact": True,
            "strict_wb_virtual_commit_exact": True,
            "rollback_and_restore_exact": True,
            "boundary_touched": False,
            "parity_established": True,
        }
        self.assertEqual(
            classify_four_path_diagnostic(
                all_endpoint_bytes_exact=True,
                **common,
            ),
            "BYTE_EXACT",
        )
        self.assertEqual(
            classify_four_path_diagnostic(
                all_endpoint_bytes_exact=False,
                **common,
            ),
            "NUMERIC_PATH_DIVERGENCE_CANDIDATE",
        )
        ambiguous = dict(common)
        ambiguous["boundary_touched"] = True
        self.assertEqual(
            classify_four_path_diagnostic(
                all_endpoint_bytes_exact=False,
                **ambiguous,
            ),
            "NUMERICALLY_AMBIGUOUS",
        )
        non_equivalent = dict(common)
        non_equivalent["benchmark_bits_exact"] = False
        self.assertEqual(
            classify_four_path_diagnostic(
                all_endpoint_bytes_exact=False,
                **non_equivalent,
            ),
            "NON_EQUIVALENT",
        )

    def test_stage_receipts_are_create_once_and_rng_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = StageRecorder(root)
            before = torch.get_rng_state().clone()
            first = recorder.record("post_fixture", {"shape": [12, 10]})
            after = torch.get_rng_state().clone()
            self.assertTrue(torch.equal(before, after))
            self.assertEqual(len(first), 64)
            self.assertEqual(recorder.last_stage, "post_fixture")
            with self.assertRaises(FileExistsError):
                _atomic_write_once(root / "stage-01-post_fixture.json", {"x": 1})

    def test_failure_receipt_hashes_message_and_allowlists_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            output = repo / "local/odebf/results" / expected_result_name(
                "llama3-8b-inst"
            )
            secret = "prompt-subject-target-must-not-appear"
            try:
                raise RuntimeError(secret)
            except RuntimeError as exc:
                digest, receipt = write_failure_once(output, exc, repo_root=repo)
            self.assertEqual(len(digest), 64)
            self.assertEqual(receipt["exception_class"], "RuntimeError")
            self.assertEqual(
                receipt["exception_message_sha256"],
                hashlib.sha256(secret.encode("utf-8")).hexdigest(),
            )
            raw = (output / "failure.json").read_text(encoding="utf-8")
            self.assertNotIn(secret, raw)
            self.assertFalse(receipt["retry_permitted"])
            with self.assertRaises(FileExistsError):
                try:
                    raise RuntimeError("second")
                except RuntimeError as exc:
                    write_failure_once(output, exc, repo_root=repo)

    def test_failure_namespace_cannot_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            with self.assertRaisesRegex(ODEBFContractError, "namespace"):
                try:
                    raise RuntimeError("opaque")
                except RuntimeError as exc:
                    write_failure_once(repo / "outside", exc, repo_root=repo)


if __name__ == "__main__":
    unittest.main()
