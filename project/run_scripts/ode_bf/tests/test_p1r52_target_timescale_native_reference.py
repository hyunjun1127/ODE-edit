from __future__ import annotations

import json
from pathlib import Path

import pytest

from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    expected_p1r52_sequential_result_name,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import P1R52_B100X10_SCALE
from project.run_scripts.ode_bf.p1r52_target_timescale_native_reference import (
    METHODS,
    POLICY_PROVENANCE,
    ROLES,
    expected_result_name,
    is_target_timescale_native_reference_role,
    method_for_role,
    role_for_cell,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
LOCK = REPO_ROOT / (
    "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p1r52_target_timescale_native_b100_user_override_v1.json"
)


def test_native_roles_and_names_are_exact_and_distinct() -> None:
    assert len(ROLES) == len(METHODS) == 2
    assert len(set(ROLES)) == len(set(METHODS)) == 2
    for index, (role, method) in enumerate(zip(ROLES, METHODS, strict=True)):
        assert role_for_cell(index) == role
        assert method_for_role(role) == method
        assert is_target_timescale_native_reference_role(role)
        assert expected_p1r52_sequential_result_name(
            "llama3-8b-inst", role, scale=P1R52_B100X10_SCALE
        ) == expected_result_name(role)
    assert "tech-r2" in expected_result_name(ROLES[1])
    with pytest.raises(ODEBFContractError):
        role_for_cell(2)
    with pytest.raises(ODEBFContractError):
        method_for_role("arbitrary-native")


def test_native_lock_binds_user_override_and_reference_only_scope() -> None:
    value = json.loads(LOCK.read_text(encoding="utf-8"))
    root = value.pop("root_digest")
    assert root == canonical_hash(value)
    assert value["policy_provenance"] == POLICY_PROVENANCE
    assert value["selected_batch"] == "B1"
    assert value["selected_request_count"] == 100
    assert value["project_gpu_cap"] == 4
    assert value["reference_boundary"] == {
        "comparison_only": True,
        "promotion_influence_count": 0,
        "selection_influence_count": 0,
        "target_timescale_parameter_influence_count": 0,
    }
    assert [item["role"] for item in value["methods"]] == list(ROLES)
    assert [item["method"] for item in value["methods"]] == list(METHODS)


def test_native_binding_reuses_official_implementation_without_science_copy() -> None:
    source = (
        REPO_ROOT
        / "project/run_scripts/ode_bf/p1r52_target_timescale_native_reference.py"
    ).read_text(encoding="utf-8")
    assert "_run_native_method(" in source
    assert "apply_AlphaEdit_to_model(" not in source
    assert "apply_memit_to_model(" not in source
    assert "USER_DIRECTED_NATIVE_REFERENCE_RUN_OVERRIDE" in source
    assert "target_timescale_field_execution_count\": 0" in source
    assert 'memit_easyedit_root=Path("/data/janghj/EasyEdit")' in source
