import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_edit_motivation.artifacts import JsonlArtifactWriter
from project.run_scripts.ode_edit_motivation.contracts import (
    ContractError,
    ContextManifest,
    EditRequest,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
    freeze_provenance,
)
from project.run_scripts.ode_edit_motivation.hooks import (
    ConcurrentWeightMutationError,
    ForwardCapture,
    TemporaryExactMemitApplication,
    TemporaryLowRankApplication,
    TensorHashRuntimeError,
    _classify_tensor_hash_runtime_error,
    assert_tensor_sha256_device_parity,
    capture_snapshot,
    tensor_sha256,
)


class HookAndArtifactTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.model.weight.copy_(
                torch.tensor(
                    [[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]],
                    dtype=torch.float64,
                )
            )
        self.request = EditRequest.from_mapping(
            {
                "case_id": "toy",
                "prompt": "{} is in",
                "subject": "Ada",
                "target_new": "London",
            }
        )
        self.contexts = ContextManifest.freeze([["{}"]], source="unit-test")
        self.snapshot = capture_snapshot(
            self.model,
            model_id="toy-linear",
            requests=[self.request],
            context_id=self.contexts.manifest_id,
            hparams={"layers": [0], "weight": 2.0},
            weight_names=["weight"],
        )
        self.factor = LowRankFactor(
            weight_name="weight",
            left=torch.tensor([[1.0], [2.0]], dtype=torch.float64),
            right=torch.tensor([[0.5], [-1.0], [2.0]], dtype=torch.float64),
            expected_weight_sha256=self.snapshot.parameter("weight").sha256,
        )
        self.proposal = MemitFactorProposal(
            snapshot=self.snapshot,
            factors=(self.factor,),
            semantics=ProposalSemantics.ORDERED_GAUSS_SEIDEL,
            solver_name="unit-test",
        )

    def test_temporary_apply_and_bit_exact_rollback(self):
        original = self.model.weight.detach().clone()
        original_hash = tensor_sha256(self.model.weight)
        with TemporaryLowRankApplication(self.model, self.proposal):
            torch.testing.assert_close(
                self.model.weight,
                original + self.factor.left @ self.factor.right.T,
            )
        torch.testing.assert_close(self.model.weight, original, rtol=0, atol=0)
        self.assertEqual(tensor_sha256(self.model.weight), original_hash)

    def test_tensor_hash_uses_cpu_logical_contiguous_bytes(self):
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            with self.subTest(dtype=dtype):
                value = torch.arange(12, dtype=dtype).reshape(3, 4).T
                self.assertFalse(value.is_contiguous())
                canonical = value.detach().cpu().contiguous().view(torch.uint8)
                expected = hashlib.sha256(
                    canonical.numpy().tobytes(order="C")
                ).hexdigest()
                self.assertEqual(tensor_sha256(value), expected)

    def test_tensor_hash_device_parity_rejects_cpu(self):
        with self.assertRaisesRegex(
            ContractError,
            "requires available CUDA",
        ):
            assert_tensor_sha256_device_parity(torch.device("cpu"))

    def test_tensor_hash_runtime_categories_are_fixed(self):
        examples = {
            "CUDA out of memory. SENSITIVE": "cuda_out_of_memory",
            "an illegal memory access was encountered": (
                "cuda_illegal_memory_access"
            ),
            "misaligned address": "cuda_misaligned_address",
            "device-side assert triggered": "cuda_device_assert",
            "CUBLAS_STATUS_EXECUTION_FAILED": "cuda_cublas",
            "CUSOLVER_STATUS_INTERNAL_ERROR": "cuda_cusolver",
            "CUDA error: SENSITIVE": "cuda_runtime",
            "view size is not compatible with stride": "tensor_layout",
            "DefaultCPUAllocator cannot allocate memory": "host_out_of_memory",
            "SENSITIVE_UNKNOWN": "unknown_runtime",
        }
        for message, expected in examples.items():
            with self.subTest(message=message):
                self.assertEqual(
                    _classify_tensor_hash_runtime_error(RuntimeError(message)),
                    expected,
                )

    def test_tensor_hash_runtime_error_contains_metadata_not_raw_message(self):
        tensor = torch.zeros((2, 3), dtype=torch.float32)
        error = TensorHashRuntimeError(
            phase="device_to_cpu",
            category="cuda_runtime",
            tensor=tensor,
        )
        self.assertEqual(error.phase, "device_to_cpu")
        self.assertEqual(error.category, "cuda_runtime")
        self.assertEqual(error.shape, (2, 3))
        self.assertEqual(error.numel, 6)
        self.assertNotIn("SENSITIVE", str(error))

    def test_temporary_apply_detects_mutation_and_still_restores(self):
        original_hash = tensor_sha256(self.model.weight)
        with self.assertRaises(ConcurrentWeightMutationError):
            with TemporaryLowRankApplication(self.model, self.proposal):
                with torch.no_grad():
                    self.model.weight[0, 0] += 0.25
        self.assertEqual(tensor_sha256(self.model.weight), original_hash)

    def test_exact_application_matches_easyedit_float_add_and_rolls_back(self):
        precise_factor = LowRankFactor(
            weight_name="weight",
            left=torch.tensor(
                [[1.0000000001], [2.0000000003]],
                dtype=torch.float64,
            ),
            right=torch.tensor(
                [[0.3333333333], [-0.1428571429], [1.0000000007]],
                dtype=torch.float64,
            ),
            expected_weight_sha256=self.snapshot.parameter("weight").sha256,
            native_update_transposed=True,
        )
        proposal = MemitFactorProposal(
            snapshot=self.snapshot,
            factors=(precise_factor,),
            semantics=ProposalSemantics.ORDERED_GAUSS_SEIDEL,
            solver_name="unit-test/ordered",
        )
        original = self.model.weight.detach().clone()
        # EasyEdit's raw factor call for Llama/Qwen is
        # adj_k @ resid.T followed by update-shape transposition.
        expected = original + (
            precise_factor.right @ precise_factor.left.T
        ).T.float()
        with TemporaryLowRankApplication(self.model, proposal):
            no_dense_value = self.model.weight.detach().clone()
        self.assertFalse(torch.equal(no_dense_value, expected))
        with TemporaryExactMemitApplication(self.model, proposal):
            torch.testing.assert_close(self.model.weight, expected, rtol=0, atol=0)
        torch.testing.assert_close(self.model.weight, original, rtol=0, atol=0)

    def test_forward_capture_is_removed(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        value = torch.tensor([[1.0, 2.0]])
        with ForwardCapture(model, "0") as capture:
            output = model(value)
            torch.testing.assert_close(capture.input, value)
            torch.testing.assert_close(capture.output, output)
        capture.output = None
        model(value * 2)
        self.assertIsNone(capture.output)

    def test_jsonl_writer_binds_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("immutable", encoding="utf-8")
            provenance = freeze_provenance([source], label="unit-test")
            artifact = root / "events.jsonl"
            with JsonlArtifactWriter(
                artifact,
                run_id="run-1",
                provenance=provenance,
                contexts=self.contexts,
                snapshot=self.snapshot,
            ) as writer:
                writer.write("measurement", {"value": torch.tensor(1.25)})
            records = [
                json.loads(line)
                for line in artifact.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["sequence"] for record in records], [0, 1])
            self.assertEqual(records[1]["payload"]["value"], 1.25)
            self.assertEqual(records[1]["snapshot_id"], self.snapshot.snapshot_id)


if __name__ == "__main__":
    unittest.main()
