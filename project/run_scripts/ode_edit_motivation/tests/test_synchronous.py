import hashlib
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    ContextManifest,
    ContractError,
    EditRequest,
    ExpectedFileIdentity,
    ProposalSemantics,
    freeze_provenance,
)
from project.run_scripts.ode_edit_motivation.direct_z import DirectZCache
from project.run_scripts.ode_edit_motivation.easyedit_bridge import (
    CovarianceCacheMissError,
    CovarianceCacheSpec,
    EasyEditBindings,
    EasyEditBridge,
    guard_precomputed_layer_stats,
)
from project.run_scripts.ode_edit_motivation.hooks import (
    capture_snapshot,
    tensor_sha256,
)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(2, 3, bias=False, dtype=torch.float64),
                torch.nn.Linear(2, 3, bias=False, dtype=torch.float64),
            ]
        )
        self.config = SimpleNamespace(_name_or_path="org/toy")


class MockBridge(EasyEditBridge):
    def __init__(self, bindings):
        self._bindings = bindings

    def load(self):
        self._bindings.provenance.assert_current()
        return self._bindings


class DirectZAndSynchronousTests(unittest.TestCase):
    def setUp(self):
        self.request_a = EditRequest.from_mapping(
            {
                "case_id": "a",
                "prompt": "{} lives in",
                "subject": "Ada",
                "target_new": "London",
            }
        )
        self.request_b = EditRequest.from_mapping(
            {
                "case_id": "b",
                "prompt": "{} lives in",
                "subject": "Grace",
                "target_new": "Paris",
            }
        )
        self.requests = (self.request_a, self.request_b)
        self.contexts = ContextManifest.freeze([["{}"]], source="unit-test")

    def test_direct_z_computes_once_then_requires_full_identity_for_reuse(self):
        model = ToyModel()
        hparams = {"layers": [0, 1], "mode": "test"}
        snapshot = capture_snapshot(
            model,
            model_id="toy",
            requests=self.requests,
            context_id=self.contexts.manifest_id,
            hparams=hparams,
            weight_names=("layers.0.weight", "layers.1.weight"),
        )
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            cache = DirectZCache(directory, "direct-z.pt")
            first = cache.load_or_compute(
                source_snapshot=snapshot,
                z_layer=1,
                compute=lambda: calls.append("computed")
                or torch.arange(6, dtype=torch.float64).reshape(3, 2),
            )
            self.assertEqual(calls, ["computed"])
            with self.assertRaises(ContractError):
                DirectZCache(directory, "direct-z.pt").load_or_compute(
                    source_snapshot=snapshot,
                    z_layer=1,
                    compute=lambda: self.fail("must not recompute"),
                )
            second = DirectZCache(directory, "direct-z.pt").load_or_compute(
                source_snapshot=snapshot,
                z_layer=1,
                compute=lambda: self.fail("must reuse pinned artifact"),
                expected_identity=ExpectedFileIdentity(
                    sha256=first.artifact.sha256,
                    size=first.artifact.size,
                ),
            )
            self.assertEqual(second.artifact_id, first.artifact_id)
            torch.testing.assert_close(second.values, first.values)
            self.assertEqual(calls, ["computed"])

    def test_layer_stats_guard_fails_before_miss_or_recompute(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("approved", encoding="utf-8")
            cache = (
                root
                / "toy"
                / "wikipedia_stats"
                / "layers.0_float64_mom2_10.npz"
            )
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b"precomputed")
            model = ToyModel()
            calls = []

            def original(*args, **kwargs):
                calls.append((args, kwargs))
                return "loaded"

            memit_main = SimpleNamespace(layer_stats=original)
            layer_stats = SimpleNamespace(load_dataset=lambda *args, **kwargs: "dataset")
            bindings = EasyEditBindings(
                memit_main=memit_main,
                compute_ks=SimpleNamespace(),
                compute_z=SimpleNamespace(),
                memit_hparams=SimpleNamespace(),
                nethook=SimpleNamespace(),
                generate=SimpleNamespace(),
                layer_stats=layer_stats,
                repr_tools=SimpleNamespace(),
                provenance=freeze_provenance([source], label="mock"),
            )
            with guard_precomputed_layer_stats(bindings, [cache]):
                result = memit_main.layer_stats(
                    model,
                    None,
                    "layers.0",
                    root,
                    "wikipedia",
                    ["mom2"],
                    sample_size=10,
                    precision="float64",
                    force_recompute=False,
                )
                self.assertEqual(result, "loaded")
                with self.assertRaises(CovarianceCacheMissError):
                    memit_main.layer_stats(
                        model,
                        None,
                        "layers.0",
                        root,
                        "wikipedia",
                        ["mom2"],
                        sample_size=10,
                        precision="float64",
                        force_recompute=True,
                    )
                with self.assertRaises(CovarianceCacheMissError):
                    layer_stats.load_dataset("wikipedia")
            self.assertIs(memit_main.layer_stats, original)
            self.assertEqual(len(calls), 1)

    def test_ordered_and_synchronous_proposals_are_labeled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("approved", encoding="utf-8")
            provenance = freeze_provenance([source], label="mock-easyedit")
            model = ToyModel()
            model.eval()
            with torch.no_grad():
                for index, layer in enumerate(model.layers):
                    layer.weight.fill_(index + 1)
            original_hashes = [
                tensor_sha256(layer.weight)
                for layer in model.layers
            ]
            hparams = SimpleNamespace(
                layers=[0, 1],
                rewrite_module_tmp="layers.{}",
                layer_module_tmp="layers.{}",
                fact_token="subject_last",
                clamp_norm_factor=1.0,
                stats_dir=str(root),
                mom2_dataset="wikipedia",
                mom2_n_samples=10,
                mom2_dtype="float64",
                mom2_update_weight=1.0,
            )

            memit_main = SimpleNamespace(
                CONTEXT_TEMPLATES_CACHE=None,
                COV_CACHE={},
            )

            def original_layer_stats(*args, **kwargs):
                return torch.eye(2, dtype=torch.float64)

            layer_stats = SimpleNamespace(
                load_dataset=lambda *args, **kwargs: self.fail(
                    "dataset path must stay guarded"
                )
            )
            memit_main.layer_stats = original_layer_stats

            def get_cov(
                current_model,
                tokenizer,
                layer_name,
                dataset,
                sample_size,
                precision,
                force_recompute,
                hparams,
            ):
                del tokenizer
                return memit_main.layer_stats(
                    current_model,
                    None,
                    layer_name,
                    hparams.stats_dir,
                    dataset,
                    ["mom2"],
                    sample_size=sample_size,
                    precision=precision,
                    force_recompute=force_recompute,
                    hparams=hparams,
                )

            memit_main.get_cov = get_cov
            original_compute_z = lambda *args, **kwargs: self.fail(
                "ordered path must inject frozen direct-z"
            )
            memit_main.compute_z = original_compute_z

            def execute_memit(
                current_model,
                current_tokenizer,
                requests,
                current_hparams,
                cache_template=None,
            ):
                self.assertIsNone(cache_template)
                for request in requests:
                    memit_main.compute_z(
                        current_model,
                        current_tokenizer,
                        request,
                        current_hparams,
                        current_hparams.layers[-1],
                        memit_main.CONTEXT_TEMPLATES_CACHE,
                    )
                for layer in current_hparams.layers:
                    get_cov(
                        current_model,
                        current_tokenizer,
                        current_hparams.rewrite_module_tmp.format(layer),
                        current_hparams.mom2_dataset,
                        current_hparams.mom2_n_samples,
                        current_hparams.mom2_dtype,
                        False,
                        current_hparams,
                    )
                current_model.train()
                current_model.requires_grad_(False)
                random.random()
                np.random.random()
                torch.rand(1)
                return {
                    f"layers.{layer}.weight": (
                        torch.eye(2, dtype=torch.float64),
                        torch.ones(3, 2, dtype=torch.float64),
                    )
                    for layer in current_hparams.layers
                }

            memit_main.execute_memit = execute_memit
            synchronous_grad_modes = []

            def compute_synchronous_keys(
                model,
                tokenizer,
                requests,
                hp,
                layer,
                contexts,
            ):
                synchronous_grad_modes.append(torch.is_grad_enabled())
                return torch.eye(2, dtype=torch.float64)

            compute_ks = SimpleNamespace(compute_ks=compute_synchronous_keys)
            current_z_calls = []

            def get_current_z(*args, **kwargs):
                current_z_calls.append((args, kwargs))
                synchronous_grad_modes.append(torch.is_grad_enabled())
                return torch.zeros(2, 3, dtype=torch.float32)

            compute_z = SimpleNamespace(
                get_module_input_output_at_words=get_current_z
            )
            bindings = EasyEditBindings(
                memit_main=memit_main,
                compute_ks=compute_ks,
                compute_z=compute_z,
                memit_hparams=SimpleNamespace(),
                nethook=SimpleNamespace(),
                generate=SimpleNamespace(),
                layer_stats=layer_stats,
                repr_tools=SimpleNamespace(),
                provenance=provenance,
            )
            bridge = MockBridge(bindings)

            source_snapshot = capture_snapshot(
                model,
                model_id="toy",
                requests=self.requests,
                context_id=self.contexts.manifest_id,
                hparams=hparams,
                weight_names=("layers.0.weight", "layers.1.weight"),
                provenance_ids=(provenance.manifest_id,),
            )
            direct_z = DirectZCache(root, "direct-z.pt").load_or_compute(
                source_snapshot=source_snapshot,
                z_layer=1,
                compute=lambda: torch.ones(3, 2, dtype=torch.float64),
            )
            covariance_specs = []
            for layer in hparams.layers:
                path = (
                    root
                    / "toy"
                    / "wikipedia_stats"
                    / f"layers.{layer}_float64_mom2_10.npz"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"covariance-{layer}".encode()
                path.write_bytes(payload)
                covariance_specs.append(
                    CovarianceCacheSpec(
                        layer=layer,
                        path=str(path),
                        identity=ExpectedFileIdentity(
                            sha256=hashlib.sha256(payload).hexdigest(),
                            size=len(payload),
                        ),
                    )
                )

            rng_before = torch.get_rng_state().clone()
            python_rng_before = random.getstate()
            numpy_rng_before = np.random.get_state()
            ordered = bridge.propose_memit_factors(
                model,
                None,
                self.requests,
                hparams,
                self.contexts,
                model_id="toy",
                direct_z=direct_z,
                covariance_caches=covariance_specs,
            )
            self.assertEqual(
                ordered.semantics,
                ProposalSemantics.ORDERED_GAUSS_SEIDEL,
            )
            self.assertIn("ordered-gauss-seidel", ordered.solver_name)
            self.assertFalse(ordered.is_synchronous)
            self.assertFalse(model.training)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
            self.assertEqual(random.getstate(), python_rng_before)
            numpy_rng_after = np.random.get_state()
            self.assertEqual(numpy_rng_after[0], numpy_rng_before[0])
            self.assertTrue(np.array_equal(numpy_rng_after[1], numpy_rng_before[1]))
            self.assertEqual(numpy_rng_after[2:], numpy_rng_before[2:])
            self.assertIs(memit_main.compute_z, original_compute_z)

            synchronous = bridge.propose_synchronous_memit_factors(
                model,
                None,
                self.requests,
                hparams,
                self.contexts,
                direct_z,
                covariance_specs,
                model_id="toy",
            )
            self.assertEqual(
                synchronous.semantics,
                ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            )
            self.assertTrue(synchronous.is_synchronous)
            self.assertEqual(synchronous.residual_denominator, 2)
            self.assertEqual(len(current_z_calls), 1)
            self.assertEqual(synchronous_grad_modes, [False, False, False])
            for factor in synchronous.factors:
                torch.testing.assert_close(
                    factor.left,
                    torch.full((3, 2), 0.5, dtype=torch.float64),
                )
                self.assertEqual(factor.left.dtype, torch.float64)
            ordered.assert_same_entry_snapshot(synchronous)
            with self.assertRaises(ContractError):
                ordered.assert_same_snapshot(synchronous)
            self.assertEqual(
                [tensor_sha256(layer.weight) for layer in model.layers],
                original_hashes,
            )
            self.assertIs(memit_main.layer_stats, original_layer_stats)
            self.assertEqual(memit_main.COV_CACHE, {})


if __name__ == "__main__":
    unittest.main()
