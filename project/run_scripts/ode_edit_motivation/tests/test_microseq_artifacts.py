from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    LowRankFactor,
    MemitFactorProposal,
    ParameterRecord,
    ProposalSemantics,
    SnapshotManifest,
)
from project.run_scripts.ode_edit_motivation.hooks import tensor_sha256
from project.run_scripts.ode_edit_motivation.microseq_artifacts import (
    MicroseqArtifactError,
    load_proposal_artifact,
    write_proposal_artifact,
)


def _snapshot(weight: torch.Tensor) -> SnapshotManifest:
    return SnapshotManifest(
        model_id="unit-model",
        context_id="unit-context",
        request_ids=("request",),
        hparams_sha256="a" * 64,
        parameters=(
            ParameterRecord(
                name="layer.weight",
                sha256=tensor_sha256(weight),
                shape=tuple(weight.shape),
                dtype=str(weight.dtype),
            ),
        ),
        provenance_ids=("b" * 64,),
    )


def _proposal() -> tuple[MemitFactorProposal, SnapshotManifest]:
    weight = torch.zeros((3, 4), dtype=torch.float32)
    snapshot = _snapshot(weight)
    factor = LowRankFactor(
        weight_name="layer.weight",
        left=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        right=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        expected_weight_sha256=snapshot.parameter("layer.weight").sha256,
    )
    return (
        MemitFactorProposal(
            snapshot=snapshot,
            factors=(factor,),
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="unit",
            residual_denominator=5,
        ),
        snapshot,
    )


class MicroseqArtifactTests(unittest.TestCase):
    def test_tensor_only_round_trip(self) -> None:
        proposal, snapshot = _proposal()
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_proposal_artifact(
                directory, action_id="edit-1-step-1", proposal=proposal
            )
            replay = load_proposal_artifact(manifest, snapshot=snapshot)
            tensor_payload = torch.load(
                Path(directory) / "edit-1-step-1.pt",
                map_location="cpu",
                weights_only=True,
            )
        self.assertEqual(replay.semantics, proposal.semantics)
        self.assertEqual(replay.residual_denominator, 5)
        self.assertEqual(set(tensor_payload), {"factor_0_left", "factor_0_right"})
        self.assertTrue(torch.equal(replay.factors[0].left, proposal.factors[0].left))

    def test_rejects_wrong_entry_state(self) -> None:
        proposal, snapshot = _proposal()
        changed_snapshot = _snapshot(torch.ones((3, 4), dtype=torch.float32))
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_proposal_artifact(
                directory, action_id="edit-1-step-1", proposal=proposal
            )
            with self.assertRaisesRegex(MicroseqArtifactError, "identity/state"):
                load_proposal_artifact(manifest, snapshot=changed_snapshot)
        self.assertNotEqual(snapshot.state_id, changed_snapshot.state_id)

    def test_rejects_tensor_file_tampering(self) -> None:
        proposal, snapshot = _proposal()
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_proposal_artifact(
                directory, action_id="edit-1-step-1", proposal=proposal
            )
            tensor_path = Path(directory) / "edit-1-step-1.pt"
            tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(MicroseqArtifactError, "byte identity"):
                load_proposal_artifact(manifest, snapshot=snapshot)

    def test_rejects_manifest_tensor_hash_tampering(self) -> None:
        proposal, snapshot = _proposal()
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_proposal_artifact(
                directory, action_id="edit-1-step-1", proposal=proposal
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["factors"][0]["left_sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(MicroseqArtifactError, "identity differs"):
                load_proposal_artifact(manifest, snapshot=snapshot)


if __name__ == "__main__":
    unittest.main()
