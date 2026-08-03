from __future__ import annotations

import hashlib
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    FileRecord,
    ProvenanceManifest,
)
from project.run_scripts.ode_edit_method.contracts import (
    LayerProposal,
    MethodContractError,
    ProposalBatch,
    ProposalSemantics,
)
from project.run_scripts.ode_edit_method.easyedit_backend import EasyEditMemitBackend
from project.run_scripts.ode_edit_method.functional_trial import (
    QuantizedFullLinearFunctionalTrial,
)
from project.run_scripts.ode_edit_method.hooks import FactorDirection
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.events import ControllerRequest
from project.run_scripts.ode_edit_method.preflight import assert_static_lock_identities
from project.run_scripts.ode_edit_method.preflight import CovarianceRuntimeContract


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Linear(3, 2, bias=False),
                torch.nn.Linear(3, 2, bias=False),
            ]
        )


class _Bridge:
    def __init__(self) -> None:
        self.bindings = SimpleNamespace(
            provenance=SimpleNamespace(manifest_id="pinned-source-manifest")
        )

    def load(self):
        return self.bindings


def _backend(alias: str, root: str) -> EasyEditMemitBackend:
    model = _TinyModel().double().eval()
    spec = SimpleNamespace(
        alias=alias,
        layers=(0, 1),
        snapshot_name=f"fixed/{alias}@revision",
    )
    runtime = SimpleNamespace(spec=spec, model=model, tokenizer=object())
    hparams = SimpleNamespace(
        layers=(0, 1),
        rewrite_module_tmp="blocks.{}",
        model_name=f"fixed/{alias}",
    )
    contexts = SimpleNamespace(manifest_id="context-id", templates=(("{}",),))
    request = ControllerRequest(
        case_id="fixture",
        prompt="{} is",
        subject="Ada",
        target_new="Paris",
        target_old="London",
    )
    return EasyEditMemitBackend(
        runtime=runtime,  # type: ignore[arg-type]
        bridge=_Bridge(),  # type: ignore[arg-type]
        hparams=hparams,
        contexts=contexts,
        request=request,
        covariance_specs=(),
        covariance_contract=SimpleNamespace(
            cache_path_by_layer={0: f"{root}/cov-0", 1: f"{root}/cov-1"},
            layer_names=("blocks.0", "blocks.1"),
            assert_current=lambda: None,
        ),  # type: ignore[arg-type]
        covariance_by_layer={
            0: torch.eye(3, dtype=torch.float64),
            1: torch.eye(3, dtype=torch.float64),
        },
        direct_z_cache_root=root,
        direct_z_cache_path=f"{root}/direct-z.pt",
        tau=0.1,
        unit_c_norm_epsilon=1e-12,
        unit_c_identity_atol=2e-5,
        unit_c_identity_rtol=2e-5,
        instrumentation=EditInstrumentation(f"fixture-{alias}"),
    )


class ConcreteBackendTests(unittest.TestCase):
    def test_adaptive_trial_uses_quantized_full_linear_backend(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            backend = _backend("llama3-8b-inst", root)
            state = backend.current_state_id()
            direction = FactorDirection(
                layer=0,
                weight_name="blocks.0.weight",
                left=torch.tensor([[0.2], [-0.4]], dtype=torch.float64),
                right=torch.tensor([[0.3], [0.1], [-0.2]], dtype=torch.float64),
            )
            proposal = LayerProposal(
                layer=0,
                state_id=state,
                direction_id=direction.direction_id,
                payload=direction,
            )
            for semantics in (
                ProposalSemantics.CURRENT_SAME_SNAPSHOT,
                ProposalSemantics.ENTRY_FROZEN_REBOUND,
                ProposalSemantics.CURRENT_COORDINATE,
            ):
                with self.subTest(semantics=semantics.value):
                    adaptive = ProposalBatch(
                        snapshot_id=state,
                        proposals=(proposal,),
                        slopes=(1.0,),
                        semantics=semantics,
                    )
                    trial = backend.trial(adaptive, (0.25,))
                    self.assertIs(type(trial), QuantizedFullLinearFunctionalTrial)
                    self.assertEqual(trial.row_block, 64)
                    with trial:
                        pass

            native = ProposalBatch(
                snapshot_id=state,
                proposals=(proposal,),
                slopes=(1.0,),
                semantics=ProposalSemantics.NATIVE_ORDERED_TERMINAL,
            )
            with self.assertRaisesRegex(MethodContractError, "adaptive proposal"):
                backend.trial(native, (0.25,))

    def test_static_lock_identities_bind_to_verified_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data" / "counterfact" / "counterfact.json"
            hparams = root / "hparams" / "MEMIT" / "fixture.yaml"
            dataset.parent.mkdir(parents=True)
            hparams.parent.mkdir(parents=True)
            dataset.write_bytes(b"dataset")
            hparams.write_bytes(b"hparams")

            def record(path: Path) -> FileRecord:
                payload = path.read_bytes()
                return FileRecord(
                    path=str(path.resolve()),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    size=len(payload),
                )

            records = (record(dataset), record(hparams))
            manifest = ProvenanceManifest(label="fixture", files=records)
            covariance_contract = CovarianceRuntimeContract.capture(
                manifest,
                {0: str(dataset), 1: str(hparams)},
                ("layer.0", "layer.1"),
            )
            covariance_contract.assert_current()
            model_lock = {
                "hparams_relative_path": "hparams/MEMIT/fixture.yaml",
                "hparams_sha256": records[1].sha256,
                "hparams_size_bytes": records[1].size,
            }
            selection_lock = {
                "dataset_relative_path": "data/counterfact/counterfact.json",
                "dataset_sha256": records[0].sha256,
                "dataset_size_bytes": records[0].size,
            }
            assert_static_lock_identities(
                root,
                fixed_artifacts=manifest,
                model_lock=model_lock,
                selection_lock=selection_lock,
            )
            model_lock["hparams_sha256"] = "0" * 64
            with self.assertRaisesRegex(MethodContractError, "tracked lock identity"):
                assert_static_lock_identities(
                    root,
                    fixed_artifacts=manifest,
                    model_lock=model_lock,
                    selection_lock=selection_lock,
                )
            dataset.write_bytes(b"changed")
            with self.assertRaisesRegex(MethodContractError, "covariance artifact changed"):
                covariance_contract.assert_current()

    def test_both_models_share_one_concrete_backend_class(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            llama = _backend("llama3-8b-inst", root)
            qwen = _backend("qwen2.5-7b-inst", root)
        self.assertIs(type(llama), EasyEditMemitBackend)
        self.assertIs(type(qwen), EasyEditMemitBackend)
        self.assertEqual(llama.layers, qwen.layers)

    def test_unchanged_state_checks_do_not_dense_rehash_target_weights(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            backend = _backend("llama3-8b-inst", root)
            checkpoint = backend.checkpoint()
            expected = backend.current_state_id()
            with mock.patch.object(
                backend, "_capture_snapshot", wraps=backend._capture_snapshot
            ) as dense_snapshot:
                self.assertEqual(backend.current_state_id(), expected)
                self.assertEqual(backend.current_state_id(), expected)
                dense_snapshot.assert_not_called()
            with torch.no_grad():
                backend.model.blocks[0].weight.add_(1.0)
            with self.assertRaisesRegex(MethodContractError, "guarded backend"):
                backend.current_state_id()
            backend.restore(checkpoint)
            backend.assert_checkpoint(checkpoint)

    def test_verified_covariance_source_is_transient_and_mutation_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            backend = _backend("llama3-8b-inst", root)
            source = backend.covariance_by_layer[0]
            pointer = source.data_ptr()
            version = source._version
            first = backend._covariance_source_for_solve(0)
            second = backend._covariance_source_for_solve(0)
            self.assertIs(first, source)
            self.assertIs(second, source)
            self.assertEqual(source.data_ptr(), pointer)
            self.assertEqual(source._version, version)
            self.assertFalse(hasattr(backend, "_solve_covariance_by_layer"))
            with torch.no_grad():
                backend.covariance_by_layer[0].add_(1.0)
            with self.assertRaisesRegex(MethodContractError, "changed in memory"):
                backend._covariance_source_for_solve(0)

    def test_native_preflight_reuse_is_scoped_and_restored(self) -> None:
        from project.run_scripts.ode_edit_method import easyedit_backend as module

        with tempfile.TemporaryDirectory() as root:
            backend = _backend("llama3-8b-inst", root)
            original = module.bridge_module._preflight_covariance_caches
            with backend._reuse_verified_covariance_preflight():
                replacement = module.bridge_module._preflight_covariance_caches
                self.assertIsNot(replacement, original)
                contract, paths, names = replacement(
                    backend.model,
                    backend.hparams,
                    backend.layers,
                    (),
                )
                self.assertIs(contract, backend.covariance_contract)
                self.assertEqual(set(paths), {0, 1})
                self.assertEqual(names, ("blocks.0", "blocks.1"))
            self.assertIs(module.bridge_module._preflight_covariance_caches, original)

    def test_mid_commit_failure_uses_single_outer_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            backend = _backend("llama3-8b-inst", root)
            checkpoint = backend.checkpoint()
            before_python = random.getstate()
            before_numpy = np.random.get_state()
            state = backend.current_state_id()
            directions = (
                FactorDirection(
                    layer=0,
                    weight_name="blocks.0.weight",
                    left=torch.tensor([[0.2], [-0.4]], dtype=torch.float64),
                    right=torch.tensor([[0.3], [0.1], [-0.2]], dtype=torch.float64),
                ),
                FactorDirection(
                    layer=1,
                    weight_name="blocks.1.weight",
                    left=torch.tensor([[0.1], [0.5]], dtype=torch.float64),
                    right=torch.tensor([[0.2], [-0.3], [0.4]], dtype=torch.float64),
                ),
            )
            batch = ProposalBatch(
                snapshot_id=state,
                proposals=tuple(
                    LayerProposal(
                        layer=direction.layer,
                        state_id=state,
                        direction_id=direction.direction_id,
                        payload=direction,
                    )
                    for direction in directions
                ),
                slopes=(1.0, 1.0),
                semantics=ProposalSemantics.CURRENT_SAME_SNAPSHOT,
            )
            from project.run_scripts.ode_edit_method import hooks as hook_module

            original = hook_module._apply_factor_update_
            calls = 0

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected concrete mid-commit failure")
                return original(*args, **kwargs)

            with mock.patch.object(
                hook_module, "_apply_factor_update_", side_effect=fail_second
            ):
                with self.assertRaisesRegex(RuntimeError, "mid-commit"):
                    backend.commit(batch, (0.25, 0.5))
            _ = random.random()
            _ = np.random.random()
            backend.restore(checkpoint)
            backend.assert_checkpoint(checkpoint)
            self.assertEqual(random.getstate(), before_python)
            observed_numpy = np.random.get_state()
            self.assertEqual(observed_numpy[0], before_numpy[0])
            self.assertTrue(np.array_equal(observed_numpy[1], before_numpy[1]))
            self.assertEqual(observed_numpy[2:], before_numpy[2:])


if __name__ == "__main__":
    unittest.main()
