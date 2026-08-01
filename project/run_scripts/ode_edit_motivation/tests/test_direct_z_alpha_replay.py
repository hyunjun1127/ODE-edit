"""CPU-only tests for the read-only Alpha direct-z replay lock."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_edit_motivation.contracts import (
    ContractError,
    ProvenanceMismatchError,
)
from project.run_scripts.ode_edit_motivation.direct_z_alpha_replay import (
    ALPHA_REPLAY_LOCK_SCHEMA,
    DirectZAlphaReplayLock,
    load_alpha_replay_lock,
    preflight_model_replay,
)


def _identity(payload: bytes) -> dict[str, object]:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def _artifact(case_id: str, request_id: str, payload: bytes) -> dict[str, object]:
    identity = _identity(payload)
    return {
        "case_id": case_id,
        "request_id": request_id,
        "artifact_sha256": identity["sha256"],
        "artifact_size": identity["size"],
        "tensor_sha256": "a" * 64,
        "origin_lineage_id": "b" * 64,
        "target_token_sha256": "c" * 64,
        "reference_c_energy": 0.125,
    }


def _lock_mapping(
    *,
    llama_artifact: dict[str, object],
    qwen_artifact: dict[str, object],
    manifest: bytes,
    summary: bytes,
) -> dict[str, object]:
    manifest_identity = _identity(manifest)
    summary_identity = _identity(summary)
    return {
        "schema_version": ALPHA_REPLAY_LOCK_SCHEMA,
        "selection_sha256": "d" * 64,
        "rank_slice": [9, 10],
        "case_order": ["case-1"],
        "models": {
            "llama3-8b-inst": {
                "source_run_id": "llama-source",
                "source_manifest": manifest_identity,
                "source_summary": summary_identity,
                "artifacts": [llama_artifact],
            },
            "qwen2.5-7b-inst": {
                "source_run_id": "qwen-source",
                "source_manifest": manifest_identity,
                "source_summary": summary_identity,
                "artifacts": [qwen_artifact],
            },
        },
    }


class AlphaReplayLockTests(unittest.TestCase):
    def _write_lock(self, root: Path, mapping: dict[str, object]) -> Path:
        path = root / "lock.json"
        path.write_text(json.dumps(mapping), encoding="utf-8")
        return path

    @staticmethod
    def _write_source_run(
        output_root: Path,
        run_id: str,
        *,
        manifest: bytes,
        summary: bytes,
        request_id: str,
        artifact: bytes,
    ) -> None:
        source = output_root / run_id
        (source / "direct_z").mkdir(parents=True)
        (source / "manifest.json").write_bytes(manifest)
        (source / "summary.json").write_bytes(summary)
        (source / "direct_z" / f"{request_id}.pt").write_bytes(artifact)

    def test_successful_preflight_and_request_lookup_are_read_only(self) -> None:
        manifest = b'{"manifest":true}'
        summary = b'{"summary":true}'
        llama_payload = b"llama-direct-z"
        qwen_payload = b"qwen-direct-z"
        llama_request = "1" * 64
        qwen_request = "1" * 64
        mapping = _lock_mapping(
            llama_artifact=_artifact("case-1", llama_request, llama_payload),
            qwen_artifact=_artifact("case-1", qwen_request, qwen_payload),
            manifest=manifest,
            summary=summary,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = self._write_lock(root, mapping)
            output_root = root / "output"
            output_root.mkdir()
            self._write_source_run(
                output_root,
                "llama-source",
                manifest=manifest,
                summary=summary,
                request_id=llama_request,
                artifact=llama_payload,
            )
            lock = load_alpha_replay_lock(lock_path)
            before = sorted(path.relative_to(output_root) for path in output_root.rglob("*"))
            preflight = preflight_model_replay(
                output_root,
                "llama3-8b-inst",
                ("case-1",),
                replay_lock=lock,
            )
            after = sorted(path.relative_to(output_root) for path in output_root.rglob("*"))

        self.assertEqual(before, after)
        self.assertEqual(preflight.record_for_request(llama_request).case_id, "case-1")
        self.assertEqual(
            preflight.artifact_path_for_request(llama_request).name,
            f"{llama_request}.pt",
        )
        self.assertEqual(lock.lock_id, lock.canonical_identity.sha256)
        self.assertEqual(lock.canonical_identity.size, len(lock.canonical_bytes))
        with self.assertRaises(ContractError):
            preflight.record_for_request("f" * 64)

    def test_schema_hash_and_duplicate_errors_fail_closed(self) -> None:
        manifest = b"m"
        summary = b"s"
        artifact = _artifact("case-1", "1" * 64, b"z")
        mapping = _lock_mapping(
            llama_artifact=artifact,
            qwen_artifact=dict(artifact),
            manifest=manifest,
            summary=summary,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_unknown = dict(mapping)
            bad_unknown["unexpected"] = True
            with self.assertRaises(ContractError):
                load_alpha_replay_lock(self._write_lock(root, bad_unknown))

            bad_hash = json.loads(json.dumps(mapping))
            bad_hash["models"]["llama3-8b-inst"]["artifacts"][0]["tensor_sha256"] = "bad"
            with self.assertRaises(ContractError):
                load_alpha_replay_lock(self._write_lock(root, bad_hash))

            duplicate_json = '{"schema_version":"x","schema_version":"x"}'
            duplicate_path = root / "duplicate.json"
            duplicate_path.write_text(duplicate_json, encoding="utf-8")
            with self.assertRaises(ContractError):
                load_alpha_replay_lock(duplicate_path)

            zero_energy = json.loads(json.dumps(mapping))
            zero_energy["models"]["llama3-8b-inst"]["artifacts"][0]["reference_c_energy"] = 0
            with self.assertRaises(ContractError):
                DirectZAlphaReplayLock.from_mapping(zero_energy)

    def test_order_mismatch_and_byte_mismatch_are_rejected(self) -> None:
        manifest = b"manifest"
        summary = b"summary"
        llama_request = "1" * 64
        payload = b"direct-z"
        mapping = _lock_mapping(
            llama_artifact=_artifact("case-1", llama_request, payload),
            qwen_artifact=_artifact("case-1", llama_request, b"qwen"),
            manifest=manifest,
            summary=summary,
        )
        lock = DirectZAlphaReplayLock.from_mapping(mapping)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_source_run(
                root,
                "llama-source",
                manifest=manifest,
                summary=summary,
                request_id=llama_request,
                artifact=payload,
            )
            with self.assertRaises(ContractError):
                preflight_model_replay(
                    root,
                    "llama3-8b-inst",
                    ("different-case",),
                    replay_lock=lock,
                )
            (root / "llama-source" / "direct_z" / f"{llama_request}.pt").write_bytes(
                b"changed"
            )
            with self.assertRaises(ProvenanceMismatchError):
                preflight_model_replay(
                    root,
                    "llama3-8b-inst",
                    ("case-1",),
                    replay_lock=lock,
                )

    def test_symlinked_artifact_is_rejected_before_hashing(self) -> None:
        manifest = b"manifest"
        summary = b"summary"
        request_id = "1" * 64
        payload = b"direct-z"
        mapping = _lock_mapping(
            llama_artifact=_artifact("case-1", request_id, payload),
            qwen_artifact=_artifact("case-1", request_id, b"qwen"),
            manifest=manifest,
            summary=summary,
        )
        lock = DirectZAlphaReplayLock.from_mapping(mapping)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_source_run(
                root,
                "llama-source",
                manifest=manifest,
                summary=summary,
                request_id=request_id,
                artifact=payload,
            )
            artifact_path = root / "llama-source" / "direct_z" / f"{request_id}.pt"
            target = root / "outside.pt"
            target.write_bytes(payload)
            artifact_path.unlink()
            os.symlink(target, artifact_path)
            with self.assertRaises(ContractError):
                preflight_model_replay(
                    root,
                    "llama3-8b-inst",
                    ("case-1",),
                    replay_lock=lock,
                )


if __name__ == "__main__":
    unittest.main()
