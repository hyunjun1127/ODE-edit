import unittest

import torch

from project.run_scripts.ode_edit_motivation.alphaedit_factors import (
    AlphaEditFactorError,
    AlphaEditProposalKind,
    alphaedit_factor_right_leak,
    alphaedit_proposal_kind,
    compare_low_rank_proposals,
    make_isolated_alphaedit_proposal,
    make_posthoc_alphaedit_proposal,
    make_unprojected_isolated_alphaedit_proposal,
    solve_isolated_alphaedit_factor,
    solve_unprojected_isolated_alphaedit_factor,
)
from project.run_scripts.ode_edit_motivation.contracts import (
    LowRankFactor,
    MemitFactorProposal,
    ParameterRecord,
    ProposalSemantics,
    SnapshotManifest,
)


def _snapshot(*, weight_shape: tuple[int, int]) -> SnapshotManifest:
    return SnapshotManifest(
        model_id="toy/model@" + "a" * 40,
        context_id="b" * 64,
        request_ids=("c" * 64,),
        hparams_sha256="d" * 64,
        parameters=(
            ParameterRecord(
                name="layer.weight",
                shape=weight_shape,
                dtype="torch.float64",
                sha256="e" * 64,
            ),
        ),
        provenance_ids=("f" * 64,),
    )


class IsolatedAlphaEditFactorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.keys = torch.tensor(
            [[1.0, -0.4], [0.3, 1.5], [-0.8, 0.2]], dtype=torch.float64
        )
        self.residuals = torch.tensor(
            [[0.7, -1.1], [0.5, 0.9], [-0.2, 0.4], [1.2, -0.3]],
            dtype=torch.float64,
        )
        # The identity works even before relying on P's intended symmetry.
        self.projector = torch.tensor(
            [[1.0, 0.1, 0.0], [0.0, 0.8, -0.2], [0.05, 0.0, 0.9]],
            dtype=torch.float64,
        )
        self.l2 = 0.7

    def _native_dense_update(self) -> torch.Tensor:
        projected_keys = self.projector @ self.keys
        system = self.l2 * torch.eye(3, dtype=torch.float64) + projected_keys @ self.keys.T
        return torch.linalg.solve(system, projected_keys @ self.residuals.T)

    def test_woodbury_factor_matches_direct_dense_alpha_solve_float64(self) -> None:
        native = self._native_dense_update()
        factor = solve_isolated_alphaedit_factor(
            keys=self.keys,
            residuals=self.residuals,
            projector=self.projector,
            l2=self.l2,
            weight_name="layer.weight",
            weight_shape=tuple(native.T.shape),
            expected_weight_sha256="e" * 64,
        )

        self.assertTrue(factor.native_update_transposed)
        torch.testing.assert_close(
            factor.left @ factor.right.T,
            native.T,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_native_and_transposed_weight_orientations_are_both_exact(self) -> None:
        native = self._native_dense_update()
        direct = solve_isolated_alphaedit_factor(
            keys=self.keys,
            residuals=self.residuals,
            projector=self.projector,
            l2=self.l2,
            weight_name="layer.weight",
            weight_shape=tuple(native.shape),
            expected_weight_sha256="e" * 64,
        )
        transposed = solve_isolated_alphaedit_factor(
            keys=self.keys,
            residuals=self.residuals,
            projector=self.projector,
            l2=self.l2,
            weight_name="layer.weight",
            weight_shape=tuple(native.T.shape),
            expected_weight_sha256="e" * 64,
        )

        self.assertFalse(direct.native_update_transposed)
        self.assertTrue(transposed.native_update_transposed)
        torch.testing.assert_close(direct.left @ direct.right.T, native)
        torch.testing.assert_close(transposed.left @ transposed.right.T, native.T)

    def test_same_alpha_base_posthoc_bp_and_genuine_solve_are_distinct(self) -> None:
        native = self._native_dense_update()
        snapshot = _snapshot(weight_shape=tuple(native.T.shape))
        genuine = make_isolated_alphaedit_proposal(
            snapshot=snapshot,
            keys=self.keys,
            residuals=self.residuals,
            projector=self.projector,
            l2=self.l2,
            weight_name="layer.weight",
            solver_suffix="unit",
            residual_denominator=5,
        )
        unprojected = make_unprojected_isolated_alphaedit_proposal(
            snapshot=snapshot,
            keys=self.keys,
            residuals=self.residuals,
            l2=self.l2,
            weight_name="layer.weight",
            solver_suffix="unit",
            residual_denominator=5,
        )
        posthoc = make_posthoc_alphaedit_proposal(
            unprojected_alpha_base=unprojected,
            projector=self.projector,
            solver_suffix="unit",
        )

        self.assertEqual(
            alphaedit_proposal_kind(genuine),
            AlphaEditProposalKind.GENUINE_ISOLATED_FIRST_EDIT,
        )
        self.assertEqual(
            alphaedit_proposal_kind(unprojected),
            AlphaEditProposalKind.UNPROJECTED_ISOLATED_FIRST_EDIT,
        )
        self.assertEqual(
            alphaedit_proposal_kind(posthoc),
            AlphaEditProposalKind.POSTHOC_RIGHT_PROJECTED,
        )
        unprojected_dense = torch.linalg.solve(
            self.l2 * torch.eye(3, dtype=torch.float64) + self.keys @ self.keys.T,
            self.keys @ self.residuals.T,
        )
        torch.testing.assert_close(
            unprojected.factors[0].left @ unprojected.factors[0].right.T,
            unprojected_dense.T,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            posthoc.factors[0].left @ posthoc.factors[0].right.T,
            unprojected_dense.T @ self.projector,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_posthoc_rejects_a_non_alpha_base(self) -> None:
        native = self._native_dense_update()
        snapshot = _snapshot(weight_shape=tuple(native.T.shape))
        non_alpha = MemitFactorProposal(
            snapshot=snapshot,
            factors=(
                solve_unprojected_isolated_alphaedit_factor(
                    keys=self.keys,
                    residuals=self.residuals,
                    l2=self.l2,
                    weight_name="layer.weight",
                    weight_shape=tuple(native.T.shape),
                    expected_weight_sha256="e" * 64,
                ),
            ),
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="memit-like-but-not-alpha-tag",
            residual_denominator=1,
        )
        with self.assertRaises(AlphaEditFactorError):
            make_posthoc_alphaedit_proposal(
                unprojected_alpha_base=non_alpha,
                projector=self.projector,
                solver_suffix="unit",
            )

    def test_right_leak_and_proposal_comparison_use_factor_grams(self) -> None:
        snapshot = _snapshot(weight_shape=(2, 3))
        reference_factor = LowRankFactor(
            weight_name="layer.weight",
            left=torch.tensor([[2.0], [1.0]], dtype=torch.float64),
            right=torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64),
            expected_weight_sha256="e" * 64,
        )
        candidate_factor = LowRankFactor(
            weight_name="layer.weight",
            left=torch.tensor([[1.0], [0.5]], dtype=torch.float64),
            right=torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64),
            expected_weight_sha256="e" * 64,
        )
        reference = MemitFactorProposal(
            snapshot=snapshot,
            factors=(reference_factor,),
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="reference",
            residual_denominator=1,
        )
        candidate = MemitFactorProposal(
            snapshot=snapshot,
            factors=(candidate_factor,),
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="candidate",
            residual_denominator=1,
        )
        projector = torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))

        leak = alphaedit_factor_right_leak(reference_factor, projector)
        dense_reference = reference_factor.left @ reference_factor.right.T
        dense_leak = dense_reference @ (torch.eye(3, dtype=torch.float64) - projector)
        self.assertAlmostEqual(leak.update_frobenius, float(torch.linalg.matrix_norm(dense_reference)))
        self.assertAlmostEqual(leak.right_leak_frobenius, float(torch.linalg.matrix_norm(dense_leak)))
        self.assertAlmostEqual(
            leak.right_leak_ratio,
            float(torch.linalg.matrix_norm(dense_leak) / torch.linalg.matrix_norm(dense_reference)),
        )

        comparison = compare_low_rank_proposals(reference, candidate)
        dense_candidate = candidate_factor.left @ candidate_factor.right.T
        self.assertAlmostEqual(
            comparison.cosine,
            float(
                torch.sum(dense_reference * dense_candidate)
                / (torch.linalg.matrix_norm(dense_reference) * torch.linalg.matrix_norm(dense_candidate))
            ),
        )
        self.assertAlmostEqual(
            comparison.relative_frobenius_to_reference,
            float(torch.linalg.matrix_norm(dense_reference - dense_candidate) / torch.linalg.matrix_norm(dense_reference)),
        )

    def test_invalid_inputs_fail_closed(self) -> None:
        native = self._native_dense_update()
        common = {
            "keys": self.keys,
            "residuals": self.residuals,
            "projector": self.projector,
            "l2": self.l2,
            "weight_name": "layer.weight",
            "weight_shape": tuple(native.T.shape),
            "expected_weight_sha256": "e" * 64,
        }
        with self.assertRaises(AlphaEditFactorError):
            solve_isolated_alphaedit_factor(**{**common, "l2": 0.0})
        with self.assertRaises(AlphaEditFactorError):
            solve_isolated_alphaedit_factor(
                **{**common, "residuals": self.residuals[:, :1]}
            )
        bad_projector = self.projector.clone()
        bad_projector[0, 0] = float("nan")
        with self.assertRaises(AlphaEditFactorError):
            solve_isolated_alphaedit_factor(**{**common, "projector": bad_projector})
        with self.assertRaises(AlphaEditFactorError):
            solve_isolated_alphaedit_factor(
                **{**common, "expected_weight_sha256": "not-a-digest"}
            )


if __name__ == "__main__":
    unittest.main()
