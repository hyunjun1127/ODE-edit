import hashlib
import importlib.util
import os
import py_compile
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    ContextManifest,
    ContractError,
    EditRequest,
    ExpectedFileIdentity,
    ParameterRecord,
    ProvenanceManifest,
    ProvenanceMismatchError,
    SnapshotManifest,
    orient_easyedit_factor,
    preflight_pinned_files,
    sanitize_edit_requests,
)
from project.run_scripts.ode_edit_motivation.easyedit_bridge import (
    APPROVED_ALPHAEDIT_REFERENCE_FILES,
    APPROVED_EASYEDIT_FILES,
    EasyEditBridge,
)


class ContractTests(unittest.TestCase):
    @staticmethod
    def _clear_easyeditor_modules():
        for name in tuple(sys.modules):
            if name == "easyeditor" or name.startswith("easyeditor."):
                sys.modules.pop(name, None)

    def test_request_is_sanitized_and_canonical(self):
        request = EditRequest.from_mapping(
            {
                "case_id": 7,
                "prompt": "{} lives in",
                "subject": "Ada",
                "target_new": {"str": "London"},
            }
        )
        self.assertEqual(request.case_id, "7")
        self.assertEqual(request.to_easyedit()["target_new"], " London")
        self.assertEqual(request, EditRequest.from_mapping(request.to_dict()))
        self.assertEqual(len(request.request_id), 64)

        with self.assertRaises(ContractError):
            EditRequest.from_mapping(
                {
                    **request.to_dict(),
                    "prompt": "{subject} lives in",
                }
            )
        with self.assertRaises(ContractError):
            sanitize_edit_requests([request.to_dict(), request.to_dict()])
        for unsafe_case_id in (".", ".."):
            with self.assertRaises(ContractError):
                EditRequest.from_mapping(
                    {**request.to_dict(), "case_id": unsafe_case_id}
                )

    def test_contexts_are_deep_frozen(self):
        mutable = [["{}"], ["Because this is known. {}"]]
        manifest = ContextManifest.freeze(mutable, source="unit-test")
        mutable[0][0] = "changed {}"
        self.assertEqual(manifest.templates[0][0], "{}")
        self.assertEqual(manifest, ContextManifest.freeze(manifest.to_easyedit(), source="unit-test"))

    def test_pinned_preflight_checks_full_hash_and_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"frozen")
            digest = hashlib.sha256(b"frozen").hexdigest()
            manifest = preflight_pinned_files(
                {
                    "artifact.bin": ExpectedFileIdentity(
                        sha256=digest,
                        size=6,
                    )
                },
                label="test",
                base_dir=directory,
            )
            self.assertEqual(manifest.files[0].sha256, digest)
            path.write_bytes(b"mutated")
            with self.assertRaises(ProvenanceMismatchError):
                manifest.assert_current()

        with self.assertRaises(ContractError):
            ExpectedFileIdentity(sha256="7f5fc9b1", size=1)

    def test_easyedit_factor_orientation_avoids_dense_update(self):
        adjusted_keys = torch.arange(6, dtype=torch.float64).reshape(3, 2)
        residuals = torch.arange(4, dtype=torch.float64).reshape(2, 2)
        record = ParameterRecord(
            name="weight",
            sha256="a" * 64,
            shape=(2, 3),
            dtype="torch.float64",
        )
        snapshot = SnapshotManifest(
            model_id="toy",
            context_id="c" * 64,
            request_ids=("r" * 64,),
            hparams_sha256="h" * 64,
            parameters=(record,),
        )
        factor = orient_easyedit_factor(
            adjusted_keys,
            residuals,
            weight_name="weight",
            weight_shape=record.shape,
            expected_weight_sha256=record.sha256,
        )
        self.assertEqual(factor.weight_shape, record.shape)
        self.assertTrue(factor.native_update_transposed)
        torch.testing.assert_close(
            factor.left @ factor.right.T,
            (adjusted_keys @ residuals.T).T,
        )
        self.assertEqual(len(snapshot.snapshot_id), 64)

    def test_bridge_preflight_requires_complete_pins_without_importing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = {}
            for index, relative in enumerate(APPROVED_EASYEDIT_FILES):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = f"approved-{index}".encode()
                path.write_bytes(payload)
                pins[relative] = {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                }
            manifest = EasyEditBridge(root, expected_files=pins).preflight()
            self.assertEqual(len(manifest.files), len(APPROVED_EASYEDIT_FILES))
            with self.assertRaises(ContractError):
                EasyEditBridge(
                    root,
                    expected_files={
                        key: value
                        for key, value in pins.items()
                        if key != APPROVED_EASYEDIT_FILES[-1]
                    },
                )

    def test_bridge_alphaedit_reference_opt_in_is_complete_and_verified(self):
        self._clear_easyeditor_modules()
        bridge = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                approved = APPROVED_EASYEDIT_FILES + APPROVED_ALPHAEDIT_REFERENCE_FILES
                pins = {}
                for index, relative in enumerate(approved):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = f"VALUE = {index}\n".encode()
                    path.write_bytes(payload)
                    pins[relative] = {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                    }
                bridge = EasyEditBridge(
                    root,
                    expected_files=pins,
                    include_alphaedit_reference=True,
                )
                bindings = bridge.load()
                self.assertEqual(
                    len(bindings.provenance.files),
                    len(approved),
                )
                for index, relative in enumerate(APPROVED_ALPHAEDIT_REFERENCE_FILES):
                    module_name = relative.removesuffix(".py").replace("/", ".")
                    self.assertEqual(
                        sys.modules[module_name].VALUE,
                        len(APPROVED_EASYEDIT_FILES) + index,
                    )
                with self.assertRaises(ContractError):
                    EasyEditBridge(
                        root,
                        expected_files={
                            key: value
                            for key, value in pins.items()
                            if key != APPROVED_ALPHAEDIT_REFERENCE_FILES[-1]
                        },
                        include_alphaedit_reference=True,
                    )
        finally:
            if bridge is not None:
                bridge._remove_import_finder()
            self._clear_easyeditor_modules()

    def test_bridge_uses_inert_namespace_and_skips_package_initializers(self):
        self._clear_easyeditor_modules()
        bridge = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                marker = root / "initializer-ran"
                bytecode_marker = root / "bytecode-ran"
                for relative in (
                    "easyeditor/__init__.py",
                    "easyeditor/models/__init__.py",
                    "easyeditor/models/memit/__init__.py",
                    "easyeditor/models/rome/__init__.py",
                    "easyeditor/util/__init__.py",
                ):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        "from pathlib import Path\n"
                        f"Path({str(marker)!r}).write_text('executed')\n",
                        encoding="utf-8",
                    )
                pins = {}
                for index, relative in enumerate(APPROVED_EASYEDIT_FILES):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = f"VALUE = {index}\n".encode()
                    if index == 0:
                        malicious = (
                            "from pathlib import Path\n"
                            f"Path({str(bytecode_marker)!r}).write_text('executed')\n"
                        ).encode()
                        payload += b"#" * (len(malicious) - len(payload))
                        path.write_bytes(malicious)
                        fixed_mtime = 1_700_000_000
                        os.utime(path, (fixed_mtime, fixed_mtime))
                        py_compile.compile(
                            str(path),
                            cfile=importlib.util.cache_from_source(str(path)),
                            doraise=True,
                            invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP,
                        )
                        path.write_bytes(payload)
                        os.utime(path, (fixed_mtime, fixed_mtime))
                    else:
                        path.write_bytes(payload)
                    pins[relative] = {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                    }

                bridge = EasyEditBridge(root, expected_files=pins)
                bindings = bridge.load()

                self.assertFalse(marker.exists())
                self.assertFalse(bytecode_marker.exists())
                self.assertEqual(bindings.memit_main.VALUE, 0)
                self.assertIsNone(bindings.memit_main.__cached__)
                self.assertEqual(
                    bindings.memit_main.__file__,
                    str((root / APPROVED_EASYEDIT_FILES[0]).resolve()),
                )
                repo_sources = {
                    str(Path(module.__file__).resolve())
                    for name, module in sys.modules.items()
                    if (
                        (name == "easyeditor" or name.startswith("easyeditor."))
                        and getattr(module, "__file__", None) is not None
                    )
                }
                self.assertEqual(
                    repo_sources,
                    {
                        str((root / relative).resolve())
                        for relative in APPROVED_EASYEDIT_FILES
                    },
                )
                self.assertTrue(
                    getattr(
                        sys.modules["easyeditor"],
                        "__ode_edit_inert_namespace__",
                        False,
                    )
                )
        finally:
            if bridge is not None:
                bridge._remove_import_finder()
            self._clear_easyeditor_modules()

    def test_bridge_rejects_preloaded_easyeditor_modules(self):
        self._clear_easyeditor_modules()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pins = {}
                for index, relative in enumerate(APPROVED_EASYEDIT_FILES):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = f"VALUE = {index}\n".encode()
                    path.write_bytes(payload)
                    pins[relative] = {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                    }
                sys.modules["easyeditor"] = types.ModuleType("easyeditor")

                with self.assertRaisesRegex(ImportError, "fresh process"):
                    EasyEditBridge(root, expected_files=pins).load()
        finally:
            self._clear_easyeditor_modules()

    def test_bridge_cleans_inert_namespaces_if_source_disappears_during_install(self):
        self._clear_easyeditor_modules()
        bridge = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pins = {}
                for index, relative in enumerate(APPROVED_EASYEDIT_FILES):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = f"VALUE = {index}\n".encode()
                    path.write_bytes(payload)
                    pins[relative] = {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                    }
                bridge = EasyEditBridge(root, expected_files=pins)
                provenance = bridge.preflight()
                disappearing = root / APPROVED_EASYEDIT_FILES[-1]

                def delete_after_check(_manifest):
                    disappearing.unlink()

                with mock.patch.object(
                    ProvenanceManifest,
                    "assert_current",
                    autospec=True,
                    side_effect=delete_after_check,
                ):
                    with self.assertRaises(FileNotFoundError):
                        bridge.load()

                self.assertFalse(
                    any(
                        name == "easyeditor" or name.startswith("easyeditor.")
                        for name in sys.modules
                    )
                )
                self.assertIsNone(bridge._import_finder)
        finally:
            if bridge is not None:
                bridge._remove_import_finder()
            self._clear_easyeditor_modules()

    def test_bridge_cleans_modules_and_finder_on_post_import_provenance_failure(self):
        self._clear_easyeditor_modules()
        bridge = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                pins = {}
                for index, relative in enumerate(APPROVED_EASYEDIT_FILES):
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    payload = f"VALUE = {index}\n".encode()
                    path.write_bytes(payload)
                    pins[relative] = {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size": len(payload),
                    }
                bridge = EasyEditBridge(root, expected_files=pins)
                bridge.preflight()
                with mock.patch.object(
                    ProvenanceManifest,
                    "assert_current",
                    side_effect=(None, ProvenanceMismatchError("changed")),
                ):
                    with self.assertRaises(ProvenanceMismatchError):
                        bridge.load()

                self.assertFalse(
                    any(
                        name == "easyeditor" or name.startswith("easyeditor.")
                        for name in sys.modules
                    )
                )
                self.assertIsNone(bridge._import_finder)
        finally:
            if bridge is not None:
                bridge._remove_import_finder()
            self._clear_easyeditor_modules()


if __name__ == "__main__":
    unittest.main()
