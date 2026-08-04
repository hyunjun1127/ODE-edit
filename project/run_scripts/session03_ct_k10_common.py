"""Session 03 CT-K10 experiment wiring over the shared fixed-horizon runner."""

from __future__ import annotations

from typing import Any, Mapping

from project.run_scripts.ode_edit_method.ct_k10 import (
    CT_K10_ARM_ORDER,
    CTK10Arm,
    assert_shared_direct_z_identities,
    run_ct_k10_arm,
)
from project.run_scripts.ode_edit_method.ct_k10_lock import (
    CT_K10_LOCK_PATH,
    load_ct_k10_lock,
)
from project.run_scripts.session03_ct_k4_common import (
    Session03CTSpec,
    build_parser as _build_parser,
    dry_plan as _dry_plan,
    expected_output_root as _expected_output_root,
    run as _run,
)


CT_K10_SPEC = Session03CTSpec(
    slug="ct-k10",
    lock_path=CT_K10_LOCK_PATH,
    load_lock=load_ct_k10_lock,
    arm_order=tuple(CT_K10_ARM_ORDER),
    run_arm=run_ct_k10_arm,
    assert_shared_direct_z=assert_shared_direct_z_identities,
    native_arm=CTK10Arm.NATIVE_MEMIT,
    one_shot_arm=CTK10Arm.BF_ONESHOT_FULL,
    frozen_arm=CTK10Arm.BF_FROZEN_CT_K10,
    finite_reference_arm=CTK10Arm.ODE_REFRESH_CT_K10,
    field_build_counts={
        CTK10Arm.BF_FROZEN_CT_K10: 1,
        CTK10Arm.ODE_REFRESH_CT_K4: 4,
        CTK10Arm.ODE_REFRESH_CT_K10: 10,
    },
    field_build_max_counts={},
    first_step_nonzero_arms=frozenset(
        {CTK10Arm.BF_FROZEN_CT_K10, CTK10Arm.ODE_REFRESH_CT_K10}
    ),
    authorization_env_prefix="ODEEDIT_SESSION03_CT_K10",
    execution_tokens={
        "p0": "session03-ct-k10-p0-v1",
        "p1": "session03-ct-k10-p1-v1",
    },
    output_prefixes={
        "p0": "session03-ct-k10-p0",
        "p1": "session03-ct-k10-p1",
    },
)


def expected_output_root(stage: str, model_alias: str, proposal_id: str):
    return _expected_output_root(stage, model_alias, proposal_id, CT_K10_SPEC)


def dry_plan(lock: Mapping[str, Any], stage: str) -> dict[str, Any]:
    return _dry_plan(lock, stage, CT_K10_SPEC)


def build_parser(stage: str):
    return _build_parser(stage, CT_K10_SPEC)


def run(args: Any) -> int:
    return _run(args, CT_K10_SPEC)
