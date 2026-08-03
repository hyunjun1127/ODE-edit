"""Fail-closed V4 macro-step/rejection-retry lock projection."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .contracts import ControllerConfig, MethodContractError, canonical_hash
from .lock import controller_config
from .oracle_absolute_lock import (
    ORACLE_ABSOLUTE_LOCK_PATH,
    file_sha256,
    load_oracle_absolute_lock,
)


V4_MACROSTEP_LOCK_PATH = Path(__file__).with_name("v4_macrostep_retry.json")
V4_EVENT_HASH = "a963fef57899ba6865d3d7f9734f7de9e641682f618ee387066f527ff241332d"
V4_MODELS = ("llama3-8b-inst", "qwen2.5-7b-inst")
V4_CASE_IDS = ("2022", "12498", "20964", "768")


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MethodContractError(f"V4 lock section {name} is absent")
    return value


def load_v4_macrostep_lock(
    path: str | Path = V4_MACROSTEP_LOCK_PATH,
) -> dict[str, Any]:
    candidate = Path(path).resolve(strict=True)
    try:
        raw = json.loads(
            candidate.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                MethodContractError(f"non-finite V4 lock constant: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MethodContractError("V4 macro-step lock is invalid JSON") from exc
    root = _mapping("root", raw)
    frozen = _mapping("v3_frozen", root.get("v3_frozen"))
    overrides = _mapping("controller_overrides", root.get("controller_overrides"))
    retry = _mapping("retry_policy", root.get("retry_policy"))
    unchanged = _mapping("unchanged_controller", root.get("unchanged_controller"))
    selection = _mapping("selection", root.get("selection"))
    baselines = _mapping("v3_native_baselines", root.get("v3_native_baselines"))
    resources = _mapping("resources", root.get("resources"))
    boundary = _mapping("execution_boundary", root.get("execution_boundary"))
    if (
        root.get("schema_version")
        != "ode-edit-v4-macrostep-retry-development-lock/v1"
        or root.get("status") != "TECHNICAL_V4_FULL_ONLY_P1_EXECUTION_LOCK"
        or root.get("instruction_id")
        != "ODEEDIT-S02-V4-MACROSTEP-RETRY-P1-PAIR-V1"
        or root.get("parent_instruction_id")
        != "ODEEDIT-S02-ORACLE-ABSOLUTE-MEAN-MARGIN-V3-P0P1"
        or root.get("implementation_base")
        != "55556881ac593e1300c3cff90753f17731f3df93"
        or frozen.get("proposal_id")
        != "bc8fc256737d4e82b83f1937633ca94709d59e51969612611e20981aa70c4c23"
        or frozen.get("lock_sha256")
        != "cd4ffeb2310e1d261cea0457cd683293e53cf92b136b6d6eb01cd5c4af579882"
        or frozen.get("event_canonical_sha256") != V4_EVENT_HASH
        or frozen.get("rho") != 0.5
        or frozen.get("required_mean_margin") != 0.0
        or frozen.get("q_new_floor") != 0.5
        or any(
            frozen.get(name) is not False
            for name in (
                "q_margin_decision",
                "per_context_decision",
                "native_nll_equality_used",
            )
        )
        or dict(overrides)
        != {"h0_fraction": 0.5, "h_max_fraction": 1.0, "kappa": 1.25}
        or retry.get("enabled") is not True
        or retry.get("same_state_field_reuse") is not True
        or retry.get("progress_retry_scale") != "gamma_down**retry_index"
        or retry.get("radius_contraction_preserved") is not True
        or retry.get("candidate_replay_fail_closed") is not True
        or retry.get("extra_field_or_backward") is not False
        or dict(unchanged)
        != {
            "beta": 0.8,
            "gamma_down": 0.5,
            "gamma_up": 1.5,
            "eta_reject": 0.1,
            "eta_expand": 0.75,
            "s_max": 6,
            "max_rejections_per_state": 4,
        }
        or tuple(selection.get("models", ())) != V4_MODELS
        or tuple(selection.get("case_ids", ())) != V4_CASE_IDS
        or selection.get("seed") != 17
        or tuple(selection.get("arms", ())) != ("full-ode-edit",)
        or selection.get("execution_axis") != "full-arm-sequential-edit-inner"
        or selection.get("dtype_policy") != "checkpoint-original"
        or selection.get("model_specific_policy") is not False
        or selection.get("evaluation") is not False
        or selection.get("generation") is not False
        or set(baselines) != set(V4_MODELS)
        or resources.get("server1_project_gpu_cap") != 3
        or resources.get("gpu_per_job") != 1
        or resources.get("cpu_per_job") != 8
        or resources.get("host_memory_mib_per_job") != 65000
        or resources.get("time") != "08:00:00"
        or resources.get("pair_gpu") != 2
        or boundary.get("submission_authorized") is not False
        or boundary.get("retry_authorized") is not False
        or boundary.get("token") != "v4-macrostep-retry-p1-full-only"
        or boundary.get("scientific_outcome_count") != 0
    ):
        raise MethodContractError("V4 macro-step lock differs")
    expected_baselines = {
        "llama3-8b-inst": (
            "local/results/session02-oracle-absolute-mean-margin-v3-p1-"
            "llama3-8b-inst-bc8fc256",
            "a02568fe4ea4c41cdb6a1cadb2f56a775ccca891dfdbbd17c23613f3db4f3a75",
        ),
        "qwen2.5-7b-inst": (
            "local/results/session02-oracle-absolute-mean-margin-v3-p1-"
            "qwen2.5-7b-inst-bc8fc256",
            "29fae415b6f285083ebe3a82750ffa538865f611e7e1c2e1e1b2d3e9f0ee3269",
        ),
    }
    for alias, (expected_root, expected_hash) in expected_baselines.items():
        baseline = _mapping(f"baseline {alias}", baselines.get(alias))
        if (
            baseline.get("root") != expected_root
            or baseline.get("terminal_manifest_sha256") != expected_hash
        ):
            raise MethodContractError("V4 Native baseline identity differs")
    v3 = load_oracle_absolute_lock(ORACLE_ABSOLUTE_LOCK_PATH)
    if (
        v3["proposal_id"] != frozen["proposal_id"]
        or v3["lock_sha256"] != frozen["lock_sha256"]
        or canonical_hash(v3["event"]) != V4_EVENT_HASH
    ):
        raise MethodContractError("V4 frozen V3 event identity differs")
    payload = dict(root)
    payload["proposal_id"] = canonical_hash(root)
    payload["lock_sha256"] = file_sha256(candidate)
    return payload


def v4_controller_config(
    base_lock: Mapping[str, Any], v4_lock: Mapping[str, Any]
) -> ControllerConfig:
    base = controller_config(base_lock)
    overrides = _mapping("controller_overrides", v4_lock["controller_overrides"])
    config = replace(
        base,
        h0_fraction=float(overrides["h0_fraction"]),
        h_max_fraction=float(overrides["h_max_fraction"]),
        kappa=float(overrides["kappa"]),
    )
    before = base.to_dict()
    after = config.to_dict()
    changed = {name for name in before if before[name] != after[name]}
    if changed != {"h0_fraction", "h_max_fraction", "kappa"}:
        raise MethodContractError("V4 controller changed outside its three locks")
    return config


__all__ = [
    "V4_CASE_IDS",
    "V4_MACROSTEP_LOCK_PATH",
    "V4_MODELS",
    "load_v4_macrostep_lock",
    "v4_controller_config",
]
