import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    ContextManifest,
    EditRequest,
    FileRecord,
    LowRankFactor,
    MemitFactorProposal,
    ParameterRecord,
    ProposalSemantics,
    ProvenanceManifest,
    SnapshotManifest,
)
from project.run_scripts.ode_edit_motivation.easyedit_bridge import (
    CovarianceCacheMissError,
)
from project.run_scripts.ode_edit_motivation.gpu_runtime import (
    FixedModelRuntime,
    GpuIdentity,
)
from project.run_scripts.ode_edit_motivation.hooks import tensor_sha256
from project.run_scripts.ode_edit_motivation.manifests import (
    CounterFactSelectionManifest,
    fixed_model_spec,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import (
    ExactWeightBranch,
    MV0Error,
    NativeDelta,
    NativeExecution,
    REPOSITORY_ROOT,
    RollbackError,
    SanitizedJsonlWriter,
    _apply_bridge_exact,
    _apply_native_deltas,
    _local_run_directory,
    _safe_payload,
    _slurm_runtime_state,
    compare_tensors,
    run_mv0,
)


class _ShapeAdapter:
    @staticmethod
    def upd_matrix_match_shape(matrix, shape):
        if tuple(matrix.shape) == tuple(shape):
            return matrix
        if tuple(matrix.T.shape) == tuple(shape):
            return matrix.T
        raise ValueError("shape")


class _ToyMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = torch.nn.Linear(2, 2, bias=False)


class _ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _ToyMlp()


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_ToyLayer() for _ in range(5)])
        self.config = types.SimpleNamespace(use_cache=False)


class _FakeBridge:
    def __init__(self, provenance):
        self.provenance = provenance

    def preflight(self):
        return self.provenance

    def load(self):
        return types.SimpleNamespace()


class MV0FidelityCpuTests(unittest.TestCase):
    def test_slurm_identity_is_bound_to_exact_audited_smoke(self):
        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "15501",
                "SLURM_JOB_NAME": "odeedit_mv0_qwen_smoke",
                "SLURMD_NODENAME": "devbox",
            },
            clear=False,
        ):
            self.assertEqual(
                _slurm_runtime_state(
                    "qwen2.5-7b-inst",
                    "mv0_qwen_smoke_v1",
                ),
                {
                    "under_slurm": True,
                    "job_id": "15501",
                    "job_name": "odeedit_mv0_qwen_smoke",
                    "node": "devbox",
                },
            )
            with self.assertRaises(MV0Error):
                _slurm_runtime_state(
                    "llama3-8b-inst",
                    "mv0_llama_smoke_v1",
                )
            os.environ["SLURM_JOB_NAME"] = "odeedit_mv0_pair_c3"
            self.assertEqual(
                _slurm_runtime_state(
                    "llama3-8b-inst",
                    "mv0_llama_c3_v1",
                )["job_id"],
                "15501",
            )

    def setUp(self):
        self.model = torch.nn.Linear(3, 2, bias=False, dtype=torch.float32)
        with torch.no_grad():
            self.model.weight.copy_(
                torch.tensor(
                    [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]],
                    dtype=torch.float32,
                )
            )
        self.base = self.model.weight.detach().clone()
        self.base_hash = tensor_sha256(self.model.weight)
        adjusted = torch.tensor([[0.25], [-0.5], [1.5]], dtype=torch.float64)
        residual = torch.tensor([[2.0], [-3.0]], dtype=torch.float64)
        factor = LowRankFactor(
            weight_name="weight",
            left=residual,
            right=adjusted,
            expected_weight_sha256=self.base_hash,
        )
        snapshot = SnapshotManifest(
            model_id="toy",
            context_id="c" * 64,
            request_ids=("r" * 64,),
            hparams_sha256="h" * 64,
            parameters=(
                ParameterRecord(
                    name="weight",
                    sha256=self.base_hash,
                    shape=(2, 3),
                    dtype="torch.float32",
                ),
            ),
        )
        self.execution = NativeExecution(
            proposal=MemitFactorProposal(
                snapshot=snapshot,
                factors=(factor,),
                semantics=ProposalSemantics.ORDERED_GAUSS_SEIDEL,
                solver_name="unit-test/native",
                residual_denominator=None,
            ),
            deltas=(
                NativeDelta(
                    weight_name="weight",
                    adjusted_keys=adjusted,
                    residuals=residual,
                    raw_update_transposed=True,
                ),
            ),
            forward_calls=1,
            solve_count=1,
        )
        self.bindings = types.SimpleNamespace(memit_main=_ShapeAdapter())

    def test_tensor_comparison_subtracts_base_once_from_each_branch(self):
        base = torch.tensor([10.0, -4.0])
        reference = torch.tensor([11.0, -2.0])
        candidate = torch.tensor([11.0, -2.0])

        metrics = compare_tensors(reference, candidate, base=base)

        expected_norm = float(torch.sqrt(torch.tensor(5.0)))
        self.assertEqual(metrics["reference_norm"], expected_norm)
        self.assertEqual(metrics["candidate_norm"], expected_norm)
        self.assertEqual(metrics["relative_l2_error"], 0.0)
        self.assertEqual(metrics["max_abs_error"], 0.0)
        self.assertTrue(metrics["exact_bytes"])

    def test_exact_bridge_reconstructs_native_gemm_order(self):
        with ExactWeightBranch(self.model, {"weight": self.base_hash}):
            _apply_native_deltas(
                self.model,
                self.execution.deltas,
                self.bindings,
            )
            native = self.model.weight.detach().clone()
        self.assertTrue(torch.equal(self.model.weight, self.base))
        with ExactWeightBranch(self.model, {"weight": self.base_hash}):
            _apply_bridge_exact(self.model, self.execution, self.bindings)
            bridge = self.model.weight.detach().clone()
        self.assertTrue(torch.equal(native, bridge))
        self.assertTrue(
            compare_tensors(native, bridge, base=self.base)["exact_bytes"]
        )
        self.assertEqual(tensor_sha256(self.model.weight), self.base_hash)

    def test_branch_rolls_back_even_when_body_raises(self):
        with self.assertRaisesRegex(RuntimeError, "body"):
            with ExactWeightBranch(self.model, {"weight": self.base_hash}):
                with torch.no_grad():
                    self.model.weight.add_(10)
                raise RuntimeError("body")
        self.assertTrue(torch.equal(self.model.weight, self.base))

    def test_artifact_firewall_rejects_raw_fields_and_tensors(self):
        with self.assertRaises(MV0Error):
            _safe_payload({"prompt": "{} is"})
        with self.assertRaises(MV0Error):
            _safe_payload({"values": torch.ones(2)})
        safe = _safe_payload({"case_id": "7", "metric": 0.25})
        self.assertEqual(safe["case_id"], "7")

    def test_jsonl_contains_only_sanitized_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with SanitizedJsonlWriter(path, "run-1") as writer:
                writer.write("case", {"case_id": "1", "pass": True})
            raw = path.read_text(encoding="utf-8")
            record = json.loads(raw)
        self.assertEqual(record["payload"]["case_id"], "1")
        self.assertNotIn("target_new", raw)

    def test_output_is_restricted_to_repo_local(self):
        with self.assertRaises(Exception):
            _local_run_directory("/tmp", "mv0-test")
        local = REPOSITORY_ROOT / "local"
        with tempfile.TemporaryDirectory(dir=local) as directory:
            root = Path(directory) / "raw"
            destination = _local_run_directory(root, "mv0-test")
            self.assertTrue(destination.is_dir())

    def _run_mv0_with_fakes(self, output_root, side_effect):
        model_spec = fixed_model_spec("llama3-8b-inst")
        model = _ToyModel().eval()
        runtime = FixedModelRuntime(
            spec=model_spec,
            model=model,
            tokenizer=types.SimpleNamespace(),
            gpu=GpuIdentity(1, 0, "fake", 1, None),
            observed_model_commit=model_spec.revision,
            observed_tokenizer_commit=model_spec.revision,
            tokenizer_policy="test",
        )
        requests = tuple(
            EditRequest.from_mapping(
                {
                    "case_id": str(index),
                    "prompt": "{} is",
                    "subject": f"s{index}",
                    "target_new": f"t{index}",
                }
            )
            for index in range(3)
        )
        selection = CounterFactSelectionManifest(
            seed="test",
            source_sha256="a" * 64,
            source_size=1,
            source_row_count=3,
            calibration=("0", "1", "2"),
            confirmatory=(),
            untouched=(),
        )
        provenance = ProvenanceManifest(
            label="test",
            files=(
                FileRecord(
                    path="/tmp/easyedit-test-source",
                    sha256="b" * 64,
                    size=1,
                ),
            ),
        )
        contexts = ContextManifest.freeze(
            (("{}",), ("CONTEXT-PLAINTEXT-MUST-NOT-PERSIST {}",)),
            source="unit-test",
        )
        hparams = types.SimpleNamespace(
            layers=(4,),
            rewrite_module_tmp="model.layers.{}.mlp.down_proj",
        )
        fake_bridge = _FakeBridge(provenance)
        with (
            mock.patch(
                "project.run_scripts.ode_edit_motivation.mv0_fidelity.preflight_fixed_artifacts",
                return_value=provenance,
            ),
            mock.patch(
                "project.run_scripts.ode_edit_motivation.mv0_fidelity._git_runtime_state",
                return_value={
                    "commit": "c" * 40,
                    "tracked_worktree_clean": True,
                },
            ),
            mock.patch(
                "project.run_scripts.ode_edit_motivation.mv0_fidelity.generate_counterfact_selection",
                return_value=selection,
            ),
            mock.patch(
                "project.run_scripts.ode_edit_motivation.mv0_fidelity.load_counterfact_requests",
                return_value=requests,
            ),
            mock.patch(
                "project.run_scripts.ode_edit_motivation.mv0_fidelity.EasyEditBridge",
                return_value=fake_bridge,
            ),
            mock.patch(
                "project.run_scripts.ode_edit_motivation.mv0_fidelity._load_hparams",
                return_value=hparams,
            ),
            mock.patch(
                "project.run_scripts.ode_edit_motivation.mv0_fidelity._freeze_contexts",
                return_value=contexts,
            ),
            mock.patch(
                "project.run_scripts.ode_edit_motivation.mv0_fidelity._covariance_specs",
                return_value=(),
            ),
            mock.patch(
                "project.run_scripts.ode_edit_motivation.mv0_fidelity._load_verified_covariances",
                return_value=({}, ()),
            ),
            mock.patch(
                "project.run_scripts.ode_edit_motivation.mv0_fidelity._run_event",
                side_effect=side_effect,
            ) as run_event,
            mock.patch.object(torch.cuda, "reset_peak_memory_stats"),
            mock.patch.object(torch.cuda, "max_memory_allocated", return_value=0),
            mock.patch.object(torch.cuda, "max_memory_reserved", return_value=0),
        ):
            summary = run_mv0(
                easyedit_root="/tmp",
                model_alias="llama3-8b-inst",
                run_id="fake-run",
                output_root=output_root,
                case_count=3,
                model_loader=lambda alias: runtime,
            )
        return summary, run_event

    @staticmethod
    def _success_result(request):
        return {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "pass": True,
            "rollback_exact": True,
            "bridge_comparison": {
                "final_delta": {"relative_l2_error": 0.0},
                "teacher_forced": {
                    "relative_l2_error": 0.0,
                    "nll_abs_error": 0.0,
                },
            },
        }

    def test_run_mv0_does_not_persist_frozen_context_plaintext(self):
        def succeed(**kwargs):
            return self._success_result(kwargs["request"])

        local = REPOSITORY_ROOT / "local"
        with tempfile.TemporaryDirectory(dir=local) as directory:
            output_root = Path(directory) / "raw"
            summary, run_event = self._run_mv0_with_fakes(output_root, succeed)
            self.assertTrue(summary["all_pass"])
            self.assertEqual(run_event.call_count, 3)
            for path in (Path(summary["output_directory"]).glob("*")):
                if path.is_file():
                    self.assertNotIn(
                        "CONTEXT-PLAINTEXT-MUST-NOT-PERSIST",
                        path.read_text(encoding="utf-8"),
                    )
            manifest = json.loads(
                (Path(summary["output_directory"]) / "manifest.json").read_text()
            )
            self.assertEqual(
                set(manifest["contexts"]),
                {
                    "manifest_id",
                    "source",
                    "group_sizes",
                    "raw_templates_persisted",
                },
            )
            self.assertFalse(manifest["contexts"]["raw_templates_persisted"])

    def test_run_mv0_records_fatal_failure_then_aborts_remaining_cases(self):
        calls = 0

        def fail_second(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RollbackError("must not persist")
            return self._success_result(kwargs["request"])

        local = REPOSITORY_ROOT / "local"
        with tempfile.TemporaryDirectory(dir=local) as directory:
            summary, run_event = self._run_mv0_with_fakes(
                Path(directory) / "raw",
                fail_second,
            )
            self.assertEqual(run_event.call_count, 2)
            self.assertEqual(summary["run_status"], "aborted")
            self.assertEqual(summary["abort_failure_type"], "RollbackError")
            self.assertEqual(summary["planned_case_count"], 3)
            self.assertEqual(summary["attempted_case_count"], 2)
            self.assertEqual(summary["not_run_due_to_abort_count"], 1)
            self.assertEqual(summary["pass_count"], 1)
            self.assertEqual(summary["failure_count"], 2)
            self.assertFalse(summary["all_pass"])
            self.assertEqual(
                summary["case_results"][-1]["status"],
                "not_run_due_to_abort",
            )
            events = (
                Path(summary["output_directory"]) / "events.jsonl"
            ).read_text(encoding="utf-8")
            self.assertIn('"failure_type":"RollbackError"', events)
            self.assertNotIn("must not persist", events)

    def test_covariance_guard_failure_is_fatal_and_denominator_preserving(self):
        def fail_guard(**kwargs):
            del kwargs
            raise CovarianceCacheMissError("must not persist")

        local = REPOSITORY_ROOT / "local"
        with tempfile.TemporaryDirectory(dir=local) as directory:
            summary, run_event = self._run_mv0_with_fakes(
                Path(directory) / "raw",
                fail_guard,
            )
            self.assertEqual(run_event.call_count, 1)
            self.assertEqual(summary["abort_failure_type"], "CovarianceCacheMissError")
            self.assertEqual(summary["attempted_case_count"], 1)
            self.assertEqual(summary["failure_count"], 3)
            self.assertFalse(summary["all_pass"])


if __name__ == "__main__":
    unittest.main()
