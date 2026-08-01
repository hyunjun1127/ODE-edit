import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    ExpectedFileIdentity,
    LowRankFactor,
    MemitFactorProposal,
    ParameterRecord,
    ProposalSemantics,
    SnapshotManifest,
    sha256_file,
)
from project.run_scripts.ode_edit_motivation.manifests import FixedModelSpec
from project.run_scripts.ode_edit_motivation.projector_adapter import (
    AlphaEditProjectorBank,
)


class AlphaEditProjectorAdapterTests(unittest.TestCase):
    def _fixture(self, root: Path):
        path = root / "projector.pt"
        projectors = torch.stack(
            (
                torch.diag(torch.tensor([1.0, 0.0, 1.0])),
                torch.diag(torch.tensor([0.0, 1.0, 1.0])),
            )
        )
        torch.save(projectors, path)
        identity = ExpectedFileIdentity(
            sha256=sha256_file(path),
            size=path.stat().st_size,
        )
        spec = FixedModelSpec(
            alias="toy",
            repository_id="toy/model",
            revision="0" * 40,
            hparams_path="unused.yaml",
            covariance_paths=("c0", "c1"),
            projector_path="projector.pt",
            layers=(4, 5),
        )
        snapshot = SnapshotManifest(
            model_id="toy/model@" + "0" * 40,
            context_id="1" * 64,
            request_ids=("2" * 64,),
            hparams_sha256="3" * 64,
            parameters=(
                ParameterRecord(
                    name="layer4.weight",
                    shape=(2, 3),
                    dtype="torch.float32",
                    sha256="4" * 64,
                ),
                ParameterRecord(
                    name="layer5.weight",
                    shape=(2, 3),
                    dtype="torch.float32",
                    sha256="5" * 64,
                ),
            ),
            provenance_ids=("6" * 64,),
        )
        proposal = MemitFactorProposal(
            snapshot=snapshot,
            factors=(
                LowRankFactor(
                    weight_name="layer4.weight",
                    left=torch.tensor([[1.0], [2.0]]),
                    right=torch.tensor([[1.0], [2.0], [3.0]]),
                    expected_weight_sha256="4" * 64,
                ),
                LowRankFactor(
                    weight_name="layer5.weight",
                    left=torch.tensor([[3.0], [4.0]]),
                    right=torch.tensor([[4.0], [5.0], [6.0]]),
                    expected_weight_sha256="5" * 64,
                ),
            ),
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="toy",
            residual_denominator=2,
        )
        return path, identity, spec, proposal

    def test_projects_right_factors_without_writing_source(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path, identity, spec, proposal = self._fixture(root)
            before = path.read_bytes()
            with mock.patch(
                "project.run_scripts.ode_edit_motivation.projector_adapter.FIXED_FILE_IDENTITIES",
                {"projector.pt": identity},
            ):
                bank = AlphaEditProjectorBank.open(root, spec)
                result = bank.project_proposal(
                    proposal,
                    layer_by_weight={"layer4.weight": 4, "layer5.weight": 5},
                    solver_suffix="unit",
                )
            self.assertTrue(
                torch.equal(
                    result.proposal.factors[0].right,
                    torch.tensor([[1.0], [0.0], [3.0]]),
                )
            )
            self.assertTrue(
                torch.equal(
                    result.proposal.factors[1].right,
                    torch.tensor([[0.0], [5.0], [6.0]]),
                )
            )
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(result.proposal.snapshot, proposal.snapshot)
            bank.assert_hash_current()

    def test_dimension_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _, identity, spec, proposal = self._fixture(root)
            bad_snapshot = SnapshotManifest(
                model_id=proposal.snapshot.model_id,
                context_id=proposal.snapshot.context_id,
                request_ids=proposal.snapshot.request_ids,
                hparams_sha256=proposal.snapshot.hparams_sha256,
                parameters=(
                    ParameterRecord(
                        name="layer4.weight",
                        shape=(2, 4),
                        dtype="torch.float32",
                        sha256="4" * 64,
                    ),
                    proposal.snapshot.parameter("layer5.weight"),
                ),
                provenance_ids=proposal.snapshot.provenance_ids,
            )
            bad = MemitFactorProposal(
                snapshot=bad_snapshot,
                factors=(
                    LowRankFactor(
                        weight_name="layer4.weight",
                        left=torch.ones(2, 1),
                        right=torch.ones(4, 1),
                        expected_weight_sha256="4" * 64,
                    ),
                ),
                semantics=proposal.semantics,
                solver_name="bad",
                residual_denominator=2,
            )
            with mock.patch(
                "project.run_scripts.ode_edit_motivation.projector_adapter.FIXED_FILE_IDENTITIES",
                {"projector.pt": identity},
            ):
                bank = AlphaEditProjectorBank.open(root, spec)
                with self.assertRaisesRegex(ValueError, "dimension"):
                    bank.project_proposal(
                        bad,
                        layer_by_weight={"layer4.weight": 4},
                        solver_suffix="unit",
                    )


if __name__ == "__main__":
    unittest.main()
