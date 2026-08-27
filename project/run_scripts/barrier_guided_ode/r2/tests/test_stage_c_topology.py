from __future__ import annotations

import inspect
from pathlib import Path

from project.run_scripts.barrier_guided_ode.r2.stage_b_probe import run_stage_b
from project.run_scripts.barrier_guided_ode.r2.stage_c_contract import TOPOLOGY_NAMES, cell_case, load_stage_c_manifest
from project.run_scripts.session05_bgode_r2_stage_c import run_id


def test_stage_c_manifest_is_outcome_independent_and_has_six_exact_cases() -> None:
    manifest, observed_sha = load_stage_c_manifest()
    assert len(observed_sha) == 64
    assert manifest["case_count"] == 6
    assert manifest["selection_influence"] == "TOKENIZER_IDS_ONLY_NO_MODEL_OUTPUT_EFFICACY_LOCALITY"
    assert manifest["termination_token_count"] == 0
    assert len({row["case_identity"] for row in manifest["cases"]}) == 6


def test_stage_c_cells_cover_each_model_and_topology_once() -> None:
    cases = tuple(cell_case(index) for index in range(6))
    assert tuple(case.topology for case in cases[:3]) == TOPOLOGY_NAMES
    assert tuple(case.topology for case in cases[3:]) == TOPOLOGY_NAMES
    assert {case.model_alias for case in cases} == {"llama3-8b-inst", "qwen2.5-7b-inst"}
    assert all(run_id(case.model_alias, case.topology).endswith("fp64-two-equality-v1") for case in cases)


def test_stage_c_topologies_have_the_locked_prefix_relations() -> None:
    for index in range(6):
        case = cell_case(index)
        target, source = case.target_token_ids, case.source_token_ids
        if case.topology == "both-multi-unequal-nonprefix":
            assert len(target) > 1 and len(source) > 1 and len(target) != len(source)
            assert target[: len(source)] != source and source[: len(target)] != target
        elif case.topology == "source-prefix-target":
            assert len(source) < len(target) and target[: len(source)] == source
        else:
            assert len(target) < len(source) and source[: len(target)] == target


def test_shared_probe_keeps_default_stage_b_and_adds_only_typed_stage_c_overrides() -> None:
    signature = inspect.signature(run_stage_b)
    assert signature.parameters["event_token_override"].default is None
    assert signature.parameters["terminal_schema"].default.endswith("stage-b-technical-closure/v1")
    source = Path(inspect.getfile(run_stage_b)).read_text(encoding="utf-8")
    assert "scalar_localizer_count\": 0" in source
    assert "probability_floor_count\": 0" in source
    assert "solve_two_equality_rayleighian" in source


def test_stage_c_array_is_six_cells_throttled_to_cap_three() -> None:
    path = Path(inspect.getfile(run_stage_b)).parents[2] / "session05_bgode_r2_stage_c.sbatch"
    source = path.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-5%3" in source
    assert source.count('MODEL_ALIAS="llama3-8b-inst"') == 3
    assert source.count('MODEL_ALIAS="qwen2.5-7b-inst"') == 3
