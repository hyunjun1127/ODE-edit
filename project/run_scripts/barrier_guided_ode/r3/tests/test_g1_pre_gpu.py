from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from project.run_scripts.barrier_guided_ode.r3.actuators import NormalizedActuatorBasis
from project.run_scripts.barrier_guided_ode.r3.g1_probe import _jsonable, run_g1
from project.run_scripts.barrier_guided_ode.r3.natural import load_natural_manifest, natural_case
from project.run_scripts.ode_edit_motivation.contracts import (
    LowRankFactor,
    MemitFactorProposal,
    ParameterRecord,
    ProposalSemantics,
    SnapshotManifest,
)


def test_natural_manifest_seals_one_common_multitoken_case_without_synthetic_replacement() -> None:
    manifest, _ = load_natural_manifest()
    assert sum(row["status"] == "AVAILABLE" for row in manifest["cases"]) == 2
    assert sum(row["status"] == "NATURAL_TOPOLOGY_UNAVAILABLE" for row in manifest["cases"]) == 6
    for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        case = natural_case(alias, "unequal-non-prefix")
        assert case["ordinal"] == 26 and case["case_id"] == "17454"
        assert max(len(case["target_token_ids"]), len(case["source_token_ids"])) >= 2
    assert all(
        row["status"] == "NATURAL_TOPOLOGY_UNAVAILABLE"
        for row in manifest["cases"]
        if row["status"] != "AVAILABLE"
    )


def test_g1_runtime_orders_official_horizon_before_validated_dictionary_and_write() -> None:
    source = Path(inspect.getfile(run_g1)).read_text(encoding="utf-8")
    official = source.index("run_official_native_apply(")
    validated = source.index("propose_validated_ordered(")
    solve = source.index("solve_factor_space(")
    write = source.index("trajectory.apply(")
    assert official < validated < solve < write
    assert "observe_all_prefix_fd(" in source
    assert "W0ConditionalSeal.capture(initial)" in source
    assert "step_size = horizon / 32.0" in source
    assert "build_ledger.release_node()" in source


def test_normalized_proposal_preserves_order_and_uses_normalized_factors() -> None:
    factors = (
        LowRankFactor("a.weight", torch.tensor([[2.0]]), torch.tensor([[3.0]]), "a" * 64, False),
        LowRankFactor("b.weight", torch.tensor([[4.0]]), torch.tensor([[5.0]]), "b" * 64, False),
    )
    basis = NormalizedActuatorBasis.from_factors(factors)
    snapshot = SnapshotManifest(
        model_id="m",
        context_id="c",
        request_ids=("r",),
        hparams_sha256="h",
        parameters=(
            ParameterRecord("a.weight", "a" * 64, (1, 1), "torch.float32"),
            ParameterRecord("b.weight", "b" * 64, (1, 1), "torch.float32"),
        ),
    )
    proposal = MemitFactorProposal(
        snapshot=snapshot,
        factors=factors,
        semantics=ProposalSemantics.ORDERED_GAUSS_SEIDEL,
        solver_name="native",
        residual_denominator=None,
    )
    normalized = basis.normalized_proposal(proposal)
    assert tuple(value.weight_name for value in normalized.factors) == ("a.weight", "b.weight")
    assert normalized.factors == basis.normalized_factors
    assert normalized.solver_name.endswith("/bgode-r3-frobenius-normalized")


def test_g1_array_mapping_and_cap_are_locked() -> None:
    path = Path(inspect.getfile(run_g1)).parents[2] / "session05_bgode_r3_g1.sbatch"
    source = path.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1%2" in source
    assert '0)\n    readonly MODEL_ALIAS="llama3-8b-inst"' in source
    assert '1)\n    readonly MODEL_ALIAS="qwen2.5-7b-inst"' in source
    assert "--gres=gpu:1" in source


def test_observation_receipt_tensor_serialization_is_value_preserving() -> None:
    payload = {
        "matrix": torch.tensor([[1.0, -2.0], [3.5, 4.0]], dtype=torch.float64),
        "scalar": torch.tensor(0.25, dtype=torch.float64),
    }
    assert _jsonable(payload) == {
        "matrix": [[1.0, -2.0], [3.5, 4.0]],
        "scalar": 0.25,
    }
