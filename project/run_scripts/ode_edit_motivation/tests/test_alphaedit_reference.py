from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from project.run_scripts.ode_edit_motivation.alphaedit_reference import (
    load_alphaedit_solver_config,
)
from project.run_scripts.ode_edit_motivation.contracts import (
    ContractError,
    ExpectedFileIdentity,
)
from project.run_scripts.ode_edit_motivation.manifests import FixedModelSpec


class AlphaEditReferenceTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict[str, ExpectedFileIdentity], FixedModelSpec]:
        alpha_body = '''from .AlphaEdit_hparams import AlphaEditHyperParams\n\ndef compute_ks(model: AlphaEditHyperParams):\n    return model + 1\n'''
        memit_body = '''from .memit_hparams import MEMITHyperParams\n\ndef compute_ks(model: MEMITHyperParams):\n    return model + 1\n'''
        files = {
            "easyeditor/models/alphaedit/AlphaEdit_main.py": "solve = 'alpha'\n",
            "easyeditor/models/alphaedit/compute_ks.py": alpha_body,
            "easyeditor/models/alphaedit/compute_z.py": "value = 1\n",
            "easyeditor/models/alphaedit/AlphaEdit_hparams.py": "value = 2\n",
            "easyeditor/models/memit/compute_ks.py": memit_body,
            "hparams/AlphaEdit/llama3-8b.yaml": '''alg_name: AlphaEdit
model_name: repo/model
stats_dir: ./data/stats
P_loc: ./P.pt
device: 0
layers: [4, 5, 6, 7, 8]
clamp_norm_factor: 1
layer_selection: all
fact_token: subject_last
v_num_grad_steps: 1
v_lr: 0.1
v_loss_layer: 1
v_weight_decay: 0.1
kl_factor: 0.1
mom2_adjustment: true
mom2_update_weight: 1
rewrite_module_tmp: model.layers.{}.mlp.down_proj
layer_module_tmp: model.layers.{}
mlp_module_tmp: model.layers.{}.mlp
attn_module_tmp: model.layers.{}.self_attn
ln_f_module: model.norm
lm_head_module: lm_head
mom2_dataset: wikipedia
mom2_n_samples: 1
mom2_dtype: float32
nullspace_threshold: 0.02
L2: 1
''',
        }
        identities = {}
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            data = path.read_bytes()
            identities[relative] = ExpectedFileIdentity(
                sha256=hashlib.sha256(data).hexdigest(), size=len(data)
            )
        spec = FixedModelSpec(
            alias="llama3-8b-inst",
            repository_id="repo/model",
            revision="a" * 40,
            hparams_path="unused",
            covariance_paths=("a", "b", "c", "d", "e"),
            projector_path="P.pt",
        )
        return identities, spec

    def test_loads_pinned_reference_and_checks_common_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities, spec = self._fixture(root)
            hparams = SimpleNamespace(
                layers=[4, 5, 6, 7, 8],
                fact_token="subject_last",
                rewrite_module_tmp="model.layers.{}.mlp.down_proj",
                layer_module_tmp="model.layers.{}",
            )
            with (
                patch(
                    "project.run_scripts.ode_edit_motivation.alphaedit_reference.FIXED_FILE_IDENTITIES",
                    identities,
                ),
                patch(
                    "project.run_scripts.ode_edit_motivation.alphaedit_reference.fixed_model_spec",
                    return_value=spec,
                ),
            ):
                config, manifest = load_alphaedit_solver_config(
                    root, "llama3-8b-inst", memit_hparams=hparams
                )
            self.assertTrue(config.key_extractor_equivalent)
            self.assertEqual(config.layers, (4, 5, 6, 7, 8))
            self.assertEqual(config.l2, 1.0)
            self.assertEqual(config.reference_manifest_id, manifest.manifest_id)

    def test_rejects_key_extractor_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identities, spec = self._fixture(root)
            path = root / "easyeditor/models/memit/compute_ks.py"
            path.write_text(path.read_text().replace("model + 1", "model + 2"))
            data = path.read_bytes()
            identities["easyeditor/models/memit/compute_ks.py"] = ExpectedFileIdentity(
                sha256=hashlib.sha256(data).hexdigest(), size=len(data)
            )
            with (
                patch(
                    "project.run_scripts.ode_edit_motivation.alphaedit_reference.FIXED_FILE_IDENTITIES",
                    identities,
                ),
                patch(
                    "project.run_scripts.ode_edit_motivation.alphaedit_reference.fixed_model_spec",
                    return_value=spec,
                ),
            ):
                with self.assertRaises(ContractError):
                    load_alphaedit_solver_config(root, "llama3-8b-inst")


if __name__ == "__main__":
    unittest.main()
