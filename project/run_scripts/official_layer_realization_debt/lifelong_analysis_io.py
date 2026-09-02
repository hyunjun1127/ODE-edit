"""Fail-closed, CPU-only I/O for the four-arm lifelong analysis.

This module deliberately does not import torch or any experiment runtime.  It
reads JSON, hashes checkpoint bytes, and checks the append-only chains without
deserialising model/checkpoint state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping, Sequence


INSTRUCTION_ID = (
    "ODEEDIT-S06-OFFICIAL-LAYER-REALIZATION-DEBT-LIFELONG-"
    "FOURARM-EXHAUSTIVE-ANALYSIS-V1"
)
CHECKPOINT_BATCHES = (0, 10, 15, 20, 30, 50, 75, 100)
LAYERS = (4, 5, 6, 7, 8)
Q_LABELS = ("pre-L4", "pre-L5", "pre-L6", "pre-L7", "pre-L8", "post-L8")
IDEAL_Q = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
CELL_ORDER = (
    ("llama3-8b-inst", "memit"),
    ("llama3-8b-inst", "alphaedit"),
    ("qwen2.5-7b-inst", "memit"),
    ("qwen2.5-7b-inst", "alphaedit"),
)


class AnalysisBoundary(RuntimeError):
    """Raised when raw evidence cannot support a common valid denominator."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise AnalysisBoundary(f"missing analysis input: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise AnalysisBoundary(f"input is not a regular non-symlink file: {path}")
    return info


def load_json(path: Path) -> Any:
    regular_file(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisBoundary(f"invalid JSON input: {path}: {exc}") from exc


def verify_embedded_identity(payload: Mapping[str, Any], label: str) -> None:
    identity = payload.get("identity_sha256")
    if identity is None:
        return
    stripped = dict(payload)
    stripped.pop("identity_sha256")
    if identity != canonical_hash(stripped):
        raise AnalysisBoundary(f"embedded canonical identity differs: {label}")


def extend_hash_chain(previous: str, member_sha256: str, ordinal: int) -> str:
    return canonical_hash(
        {
            "previous": previous,
            "member_sha256": member_sha256,
            "ordinal": int(ordinal),
        }
    )


@dataclass(frozen=True, slots=True)
class TechnicalExclusion:
    job_id: str
    classification: str
    reason: str
    paths: tuple[Path, ...] = ()

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "TechnicalExclusion":
        return cls(
            job_id=str(payload["job_id"]),
            classification=str(payload.get("classification", "PURE_TECHNICAL")),
            reason=str(payload["reason"]),
            paths=tuple(Path(value) for value in payload.get("paths", ())),
        )


@dataclass(frozen=True, slots=True)
class ArmSpec:
    model: str
    method: str
    job_id: str
    result_path: Path
    expected_source_head: str
    expected_source_tree: str
    scheduler_state: str = "COMPLETED"
    scheduler_exit_code: str = "0:0"
    resource: Mapping[str, Any] = field(default_factory=dict)
    logs: tuple[Path, ...] = ()
    receipts: tuple[Path, ...] = ()
    technical_exclusions: tuple[TechnicalExclusion, ...] = ()

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ArmSpec":
        source = payload["expected_source"]
        return cls(
            model=str(payload["model"]),
            method=str(payload["method"]),
            job_id=str(payload["job_id"]),
            result_path=Path(payload["result_path"]),
            expected_source_head=str(source["head"]),
            expected_source_tree=str(source["tree"]),
            scheduler_state=str(payload.get("scheduler_state", "COMPLETED")),
            scheduler_exit_code=str(payload.get("scheduler_exit_code", "0:0")),
            resource=dict(payload.get("resource", {})),
            logs=tuple(Path(value) for value in payload.get("logs", ())),
            receipts=tuple(Path(value) for value in payload.get("receipts", ())),
            technical_exclusions=tuple(
                TechnicalExclusion.from_json(value)
                for value in payload.get("technical_exclusions", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class InputLock:
    arms: tuple[ArmSpec, ...]
    stream_root: str
    order_root: str
    sentinel_order_root: str
    stream_sha256: str
    easyedit_root: Path
    easyedit_head: str
    easyedit_tree: str
    evaluator_identity: str
    expected_checkpoint_batches: tuple[int, ...] = CHECKPOINT_BATCHES
    instruction_id: str = INSTRUCTION_ID

    @classmethod
    def from_path(cls, path: Path) -> "InputLock":
        payload = load_json(path)
        if not isinstance(payload, Mapping):
            raise AnalysisBoundary("input lock is not a JSON object")
        arms = tuple(ArmSpec.from_json(value) for value in payload.get("arms", ()))
        observed = {(arm.model, arm.method) for arm in arms}
        if len(arms) != 4 or observed != set(CELL_ORDER):
            raise AnalysisBoundary(f"input lock must contain exact four arms: {observed}")
        stream = payload["stream"]
        easyedit = payload["easyedit"]
        value = cls(
            arms=arms,
            stream_root=str(stream["root"]),
            order_root=str(stream["order"]),
            sentinel_order_root=str(stream["sentinel_order"]),
            stream_sha256=str(stream["sha256"]),
            easyedit_root=Path(easyedit["root"]),
            easyedit_head=str(easyedit["head"]),
            easyedit_tree=str(easyedit["tree"]),
            evaluator_identity=str(payload.get("evaluator_identity", "NOT_RECORDED_SCHEMA")),
            expected_checkpoint_batches=tuple(
                int(value) for value in payload.get("checkpoint_batches", CHECKPOINT_BATCHES)
            ),
            instruction_id=str(payload.get("instruction_id", INSTRUCTION_ID)),
        )
        if value.instruction_id != INSTRUCTION_ID:
            raise AnalysisBoundary("analysis instruction identity differs")
        if value.expected_checkpoint_batches != CHECKPOINT_BATCHES:
            raise AnalysisBoundary("checkpoint schedule differs from 0/1k/1.5k/2k/3k/5k/7.5k/10k")
        return value


@dataclass(slots=True)
class ValidatedArm:
    spec: ArmSpec
    result: dict[str, Any]
    journals: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]
    raw_members: list[dict[str, Any]]
    raw_member_root: str
    external_members: list[dict[str, Any]]


def _member(
    path: Path,
    kind: str,
    expected_sha: str | None = None,
    expected_bytes: int | None = None,
    *,
    required_mode: int | None = None,
) -> dict[str, Any]:
    info = regular_file(path)
    if expected_bytes is not None and info.st_size != expected_bytes:
        raise AnalysisBoundary(f"member bytes differ: {path}: {info.st_size} != {expected_bytes}")
    if required_mode is not None and stat.S_IMODE(info.st_mode) != required_mode:
        raise AnalysisBoundary(
            f"member mode differs: {path}: {stat.S_IMODE(info.st_mode):04o} != {required_mode:04o}"
        )
    digest = sha256_file(path)
    if expected_sha is not None and digest != expected_sha:
        raise AnalysisBoundary(f"member SHA differs: {path}: {digest} != {expected_sha}")
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "bytes": int(info.st_size),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "sha256": digest,
    }


def _bool_all(mapping: Mapping[str, Any], label: str) -> None:
    failed = sorted(key for key, value in mapping.items() if isinstance(value, bool) and not value)
    if failed:
        raise AnalysisBoundary(f"false invariant(s) in {label}: {failed}")


def _validate_model_binding(result: Mapping[str, Any]) -> None:
    binding = result.get("model_binding", {})
    if binding.get("full_fp32") is not True:
        raise AnalysisBoundary("FULL_FP32 binding differs")
    dtypes = result.get("initial_w0", {}).get("dtypes", {})
    restored = result.get("terminal_w0_restore", {}).get("dtypes", {})
    if len(dtypes) != 5 or set(dtypes.values()) != {"torch.float32"} or restored != dtypes:
        raise AnalysisBoundary("editable parameter FP32 inventory differs")
    padding = binding.get("padding_gate", {})
    if padding.get("status") != "PASS" or padding.get("padding_side") != "left":
        raise AnalysisBoundary("left-padding semantic identity gate differs")
    if not padding.get("batch_reorder_identity") or not padding.get("padding_length_identity"):
        raise AnalysisBoundary("left-padding reorder/length identity differs")


def _validate_journal(
    payload: Mapping[str, Any],
    *,
    arm: ArmSpec,
    batch_index: int,
    expected_entry: Mapping[str, str],
    prior_cache_exit: str | None,
    expected_request_order: Sequence[str] | None = None,
    expected_batch_order_sha256: str | None = None,
) -> tuple[Mapping[str, str], str | None]:
    if (
        payload.get("status") != "B100_ATOMIC_TERMINAL_VALID"
        or payload.get("model") != arm.model
        or payload.get("method") != arm.method
        or int(payload.get("batch_index", -1)) != batch_index
        or int(payload.get("request_count", -1)) != 100
        or int(payload.get("accepted_edit_start", -1)) != (batch_index - 1) * 100
        or int(payload.get("accepted_edit_end", -1)) != batch_index * 100
    ):
        raise AnalysisBoundary(f"journal terminal/denominator differs: {arm.model}/{arm.method}/B{batch_index}")
    observer = payload.get("apply", {}).get("layer_realization_observer", {})
    debt = observer.get("residual_debt", {})
    if (
        observer.get("status") != "TERMINAL_OBSERVATION_VALID"
        or int(observer.get("request_count", -1)) != 100
        or int(observer.get("direct_z_compute_count", -1)) != 100
        or int(observer.get("direct_z_recompute_count", -1)) != 0
        or int(observer.get("layer_loop_pass_through_call_count", -1)) != 5
        or int(observer.get("layer_loop_observation_copy_count", -1)) != 5
        or int(observer.get("terminal_post_L8_forward_count", -1)) != 1
        or int(observer.get("decision_or_update_tensor_mutation_count", -1)) != 0
        or int(observer.get("raw_prompt_logit_publish_count", -1)) != 0
        or int(debt.get("nonfinite_count", -1)) != 0
        or len(debt.get("records", ())) != 100
    ):
        raise AnalysisBoundary(f"observer invariant differs: {arm.model}/{arm.method}/B{batch_index}")
    if float(debt.get("maximum_recurrence_closure_relative_error", float("inf"))) >= 1e-4:
        raise AnalysisBoundary(f"recurrence closure differs: {arm.model}/{arm.method}/B{batch_index}")
    weight_action = observer.get("weight_action", {})
    if (
        observer.get("module_global_identity_restored") is not True
        or weight_action.get("scalar_reduction_dtype") != "float64"
        or [int(row.get("layer", -1)) for row in weight_action.get("layers", ())] != list(LAYERS)
        or not math.isclose(float(weight_action.get("share_sum", math.nan)), 1.0, rel_tol=1e-9, abs_tol=1e-9)
        or not math.isclose(float(weight_action.get("update_magnitude_share_sum", math.nan)), 1.0, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise AnalysisBoundary(f"observer weight/scalar invariant differs: {arm.model}/{arm.method}/B{batch_index}")
    audit = payload.get("apply", {}).get("official_call_audit", {})
    expected_ks = 10 if arm.method == "alphaedit" else 5
    if (
        int(audit.get("compute_ks_call_count", -1)) != expected_ks
        or int(audit.get("torch_linalg_solve_call_count", -1)) != 5
        or audit.get("module_compute_ks_identity_restored") is not True
        or audit.get("torch_linalg_solve_identity_restored") is not True
        or int(audit.get("output_or_decision_mutation_count", -1)) != 0
    ):
        raise AnalysisBoundary(f"Official production call audit differs: {arm.model}/{arm.method}/B{batch_index}")
    if len(payload.get("endpoint", {}).get("records", ())) != 100:
        raise AnalysisBoundary(f"endpoint denominator differs: {arm.model}/{arm.method}/B{batch_index}")
    if len(payload.get("action_realization", {}).get("requests", ())) != 100:
        raise AnalysisBoundary(f"action-realization denominator differs: {arm.model}/{arm.method}/B{batch_index}")
    if expected_request_order is not None:
        observed_orders = {
            "observer": [str(row["request_sha256"]) for row in debt["records"]],
            "endpoint": [str(row["request_sha256"]) for row in payload["endpoint"]["records"]],
            "action": [str(row["request_sha256"]) for row in payload["action_realization"]["requests"]],
        }
        if any(values != list(expected_request_order) for values in observed_orders.values()):
            raise AnalysisBoundary(f"B100 request membership/order differs: {arm.model}/{arm.method}/B{batch_index}")
        if len(set(expected_request_order)) != 100:
            raise AnalysisBoundary(f"sealed B100 contains duplicate requests: B{batch_index}")
    if expected_batch_order_sha256 is not None and payload.get("request_order_sha256") != expected_batch_order_sha256:
        raise AnalysisBoundary(f"B100 ordered digest differs: {arm.model}/{arm.method}/B{batch_index}")
    weight = payload.get("weight_continuity", {})
    if dict(weight.get("entry_sha256", {})) != dict(expected_entry):
        raise AnalysisBoundary(f"W commit-to-entry link differs: {arm.model}/{arm.method}/B{batch_index}")
    if not weight.get("entry_matches_previous_commit") or not weight.get("official_original_copy_matches_entry"):
        raise AnalysisBoundary(f"W continuity check differs: {arm.model}/{arm.method}/B{batch_index}")
    cache = payload.get("cache_continuity", {})
    checks = cache.get("checks", {})
    expected_cache_checks = (
        {
            "append_width_exact",
            "consume_width_exact",
            "entry_exit_sha_link",
            "entry_width_exact",
            "exit_width_exact",
            "prior_cache_consumed",
            "reset_only_at_first_entry",
            "static_projection_separate",
        }
        if arm.method == "alphaedit"
        else {
            "entry_reuse_after_first",
            "historical_decision_state_false",
            "request_history_width_zero",
            "silent_reset_zero",
            "static_covariance_identity_link",
        }
    )
    if set(checks) != expected_cache_checks:
        raise AnalysisBoundary(f"cache check schema differs: {arm.model}/{arm.method}/B{batch_index}")
    _bool_all(checks, f"cache {arm.model}/{arm.method}/B{batch_index}")
    if prior_cache_exit is not None and str(cache.get("entry_sha256")) != prior_cache_exit:
        raise AnalysisBoundary(f"cache commit-to-entry link differs: {arm.model}/{arm.method}/B{batch_index}")
    if arm.method == "alphaedit":
        if (
            int(cache.get("entry_width", -1)) != (batch_index - 1) * 100
            or int(cache.get("exit_width", -1)) != batch_index * 100
            or int(cache.get("append_width", -1)) != 100
            or int(cache.get("consume_width", -1)) != (batch_index - 1) * 100
        ):
            raise AnalysisBoundary(f"AlphaEdit cache width differs: {arm.model}/B{batch_index}")
    elif arm.method == "memit":
        if any(int(cache.get(key, -1)) != 0 for key in ("entry_width", "exit_width", "append_width", "consume_width")):
            raise AnalysisBoundary(f"MEMIT static cache width differs: {arm.model}/B{batch_index}")
    else:
        raise AnalysisBoundary(f"unknown method: {arm.method}")
    if batch_index == 1 and payload.get("first_valid_gate", {}).get("status") != "FIRST_B100_OBSERVER_ON_OFF_PARITY_PASS":
        raise AnalysisBoundary(f"B1 observer parity gate differs: {arm.model}/{arm.method}")
    return dict(weight.get("commit_sha256", {})), (
        None if cache.get("exit_sha256") is None else str(cache.get("exit_sha256"))
    )


def _validate_checkpoint_probe(
    checkpoint: Mapping[str, Any], *, arm: ArmSpec, batch_index: int,
    expected_sentinel_order: Sequence[str] | None = None,
    expected_sentinel_order_sha256: str | None = None,
    expected_current_order: Sequence[str] | None = None,
) -> None:
    """Validate all scientific checkpoint forks without loading state tensors."""
    probe = checkpoint["probe"]
    if expected_sentinel_order_sha256 is not None and probe.get("sentinel_order_sha256") != expected_sentinel_order_sha256:
        raise AnalysisBoundary(f"checkpoint sentinel order digest differs: {arm.model}/{arm.method}/B{batch_index}")
    expected_forks = ("primary", "reset_cache_fork") if arm.method == "alphaedit" else ("primary",)
    if arm.method == "memit" and probe.get("reset_cache_fork") is not None:
        raise AnalysisBoundary(f"unexpected MEMIT reset-cache fork: {arm.model}/B{batch_index}")
    for fork_name in expected_forks:
        fork = probe.get(fork_name)
        if not isinstance(fork, Mapping):
            raise AnalysisBoundary(f"checkpoint fork absent: {arm.model}/{arm.method}/B{batch_index}/{fork_name}")
        ordered = fork.get("ordered", {})
        observer = ordered.get("layer_realization_observer", {})
        debt = observer.get("residual_debt", {})
        expected_replay = 100 if fork_name == "reset_cache_fork" else 0
        if (
            observer.get("status") != "TERMINAL_OBSERVATION_VALID"
            or int(observer.get("request_count", -1)) != 100
            or int(observer.get("direct_z_compute_count", -1)) != 100
            or int(observer.get("direct_z_recompute_count", -1)) != 0
            or int(observer.get("direct_z_shared_replay_count", -1)) != expected_replay
            or int(observer.get("layer_loop_pass_through_call_count", -1)) != 5
            or int(observer.get("layer_loop_observation_copy_count", -1)) != 5
            or int(observer.get("terminal_post_L8_forward_count", -1)) != 1
            or int(observer.get("decision_or_update_tensor_mutation_count", -1)) != 0
            or int(observer.get("raw_prompt_logit_publish_count", -1)) != 0
            or int(debt.get("nonfinite_count", -1)) != 0
            or len(debt.get("records", ())) != 100
            or len(observer.get("weight_action", {}).get("layers", ())) != 5
            or float(debt.get("maximum_recurrence_closure_relative_error", math.inf)) >= 1e-4
        ):
            raise AnalysisBoundary(
                f"checkpoint observer differs: {arm.model}/{arm.method}/B{batch_index}/{fork_name}"
            )
        if expected_sentinel_order is not None:
            observed = [str(row["request_sha256"]) for row in debt["records"]]
            if observed != list(expected_sentinel_order) or len(set(observed)) != 100:
                raise AnalysisBoundary(
                    f"checkpoint sentinel membership/order differs: {arm.model}/{arm.method}/B{batch_index}/{fork_name}"
                )
        audit = ordered.get("official_call_audit", {})
        expected_ks = 10 if arm.method == "alphaedit" else 5
        if (
            int(audit.get("compute_ks_call_count", -1)) != expected_ks
            or int(audit.get("torch_linalg_solve_call_count", -1)) != 5
            or audit.get("module_compute_ks_identity_restored") is not True
            or audit.get("torch_linalg_solve_identity_restored") is not True
            or int(audit.get("output_or_decision_mutation_count", -1)) != 0
        ):
            raise AnalysisBoundary(
                f"checkpoint Official call audit differs: {arm.model}/{arm.method}/B{batch_index}/{fork_name}"
            )
        same = fork.get("same_entry", {})
        restore = same.get("restore", {})
        if (
            int(same.get("activation_forward_count", -1)) != 6
            or int(same.get("additional_compute_z", -1)) != 0
            or int(same.get("additional_key_compute", -1)) != 0
            or int(same.get("additional_solve", -1)) != 0
            or len(same.get("layers", ())) != 5
            or restore.get("weights", {}).get("exact") is not True
            or restore.get("cache", {}).get("exact") is not True
        ):
            raise AnalysisBoundary(
                f"checkpoint same-entry/restore differs: {arm.model}/{arm.method}/B{batch_index}/{fork_name}"
            )
        geometry = fork.get("completion_geometry", {})
        waypoints = geometry.get("waypoints", ())
        scalar_fields = (
            "A0",
            "V_to_go",
            "Vbar",
            "minimum_completion_budget_h_normalized",
            "unreachable_fraction",
        )
        if (
            len(waypoints) != 6
            or [int(row.get("waypoint", -1)) for row in waypoints] != list(range(6))
            or [int(row.get("remaining_layer_count", -1)) for row in waypoints]
            != [5, 4, 3, 2, 1, 0]
            or geometry.get("full_parameter_space_claim") is not False
            or not all(
                isinstance(geometry.get(key), (int, float))
                and math.isfinite(float(geometry[key]))
                for key in scalar_fields
            )
        ):
            raise AnalysisBoundary(
                f"checkpoint completion geometry differs: {arm.model}/{arm.method}/B{batch_index}/{fork_name}"
            )
    functional = checkpoint.get("functional", {})
    def validate_panel(
        panel: Mapping[str, Any], *, expected_count: int,
        expected_order: Sequence[str] | None, low_cost: bool,
    ) -> None:
        records = list(panel.get("records", ()))
        observed_order = [str(row.get("request_sha256")) for row in records]
        if (
            int(panel.get("request_count", -1)) != expected_count
            or len(records) != expected_count
            or len(set(observed_order)) != expected_count
            or int(panel.get("controller_or_selection_influence_count", -1)) != 0
            or (expected_order is not None and observed_order != list(expected_order))
        ):
            raise AnalysisBoundary(f"functional panel denominator/order differs: {arm.model}/{arm.method}/B{batch_index}")
        expected_counts = {
            "rewrite_target_new": 1,
            "rewrite_target_true": 1,
            "rephrase_target_new": 2,
            "locality_target_true": 10,
        }
        for record in records:
            if int(record.get("raw_prompt_logit_generation_publish_count", -1)) != 0:
                raise AnalysisBoundary(f"functional raw publication differs: {arm.model}/{arm.method}/B{batch_index}")
            for metric, count in expected_counts.items():
                value = record.get(metric, {})
                if low_cost and metric in {"rephrase_target_new", "locality_target_true"}:
                    if value.get("status") != "NOT_RECORDED_LOW_COST_BATCH" or "count" in value:
                        raise AnalysisBoundary(f"retention missing-policy differs: {arm.model}/{arm.method}/B{batch_index}")
                elif int(value.get("count", -1)) != count or not all(
                    isinstance(value.get(key), (int, float)) and math.isfinite(float(value[key]))
                    for key in ("nll_mean", "margin_mean", "strict_count")
                ):
                    raise AnalysisBoundary(f"functional metric denominator differs: {arm.model}/{arm.method}/B{batch_index}/{metric}")
    if batch_index == 0:
        validate_panel(
            functional.get("sentinel_pre_edit", {}),
            expected_count=100,
            expected_order=expected_sentinel_order,
            low_cost=False,
        )
    elif (
        int(functional.get("controller_or_selection_influence_count", -1)) != 0
        or not isinstance(functional.get("retention"), Mapping)
    ):
        raise AnalysisBoundary(f"checkpoint functional boundary differs: {arm.model}/{arm.method}/B{batch_index}")
    else:
        validate_panel(
            functional.get("current", {}),
            expected_count=100,
            expected_order=expected_current_order,
            low_cost=False,
        )
        retention = functional["retention"]
        if set(retention) != {"earliest", "recent", "hash_stratified"}:
            raise AnalysisBoundary(f"retention cohort schema differs: {arm.model}/{arm.method}/B{batch_index}")
        for name, count in (("earliest", 32), ("recent", 32), ("hash_stratified", 64)):
            validate_panel(
                retention[name], expected_count=count, expected_order=None, low_cost=True
            )


def validate_arm(spec: ArmSpec, lock: InputLock, *, expected_batches: int = 100, expected_requests: int = 10_000, expected_checkpoints: int = 8) -> ValidatedArm:
    if spec.scheduler_state != "COMPLETED" or spec.scheduler_exit_code != "0:0":
        raise AnalysisBoundary(f"scheduler terminal differs: job {spec.job_id}")
    cell_root = spec.result_path.parent.resolve()
    result_member = _member(spec.result_path, "CELL_RESULT", required_mode=0o600)
    result_member["relative_path"] = spec.result_path.resolve().relative_to(cell_root).as_posix()
    result = load_json(spec.result_path)
    if not isinstance(result, dict):
        raise AnalysisBoundary(f"cell result is not an object: {spec.result_path}")
    verify_embedded_identity(result, str(spec.result_path))
    if (
        result.get("status") != "LIFELONG_10K_TERMINAL_VALID"
        or result.get("model") != spec.model
        or result.get("method") != spec.method
        or result.get("source", {}).get("head") != spec.expected_source_head
        or result.get("source", {}).get("tree") != spec.expected_source_tree
        or result.get("source", {}).get("tracked_clean") is not True
        or int(result.get("valid_batch_denominator", -1)) != expected_batches
        or int(result.get("valid_request_denominator", -1)) != expected_requests
        or int(result.get("checkpoint_denominator", -1)) != expected_checkpoints
        or int(result.get("nonfinite_count", -1)) != 0
        or int(result.get("rollback_violation_count", -1)) != 0
        or int(result.get("target_recomputation_count", -1)) != 0
        or result.get("scientific_promotion") is not False
    ):
        raise AnalysisBoundary(f"cell terminal/schema differs: {spec.model}/{spec.method}")
    stream = result.get("stream", {})
    if (
        stream.get("root") != lock.stream_root
        or stream.get("order") != lock.order_root
        or stream.get("sentinel_order") != lock.sentinel_order_root
        or stream.get("sha256") != lock.stream_sha256
        or int(stream.get("sample_duplication_count", -1)) != 0
    ):
        raise AnalysisBoundary(f"stream/order identity differs: {spec.model}/{spec.method}")
    stream_path = Path(stream["path"])
    raw_members = [result_member]
    external = [_member(stream_path, "STREAM_SEAL", lock.stream_sha256)]
    stream_seal = load_json(stream_path)
    training_by_batch: dict[int, list[str]] = {}
    batch_digests: list[str] = []
    sentinel_order: list[str] | None = None
    if expected_batches == 100 and expected_requests == 10_000:
        unsigned_stream = dict(stream_seal)
        unsigned_stream.pop("root_digest", None)
        if (
            stream_seal.get("root_digest") != lock.stream_root
            or canonical_hash(unsigned_stream) != lock.stream_root
            or stream_seal.get("training_order_sha256") != lock.order_root
            or stream_seal.get("sentinel_order_sha256") != lock.sentinel_order_root
            or int(stream_seal.get("training_request_count", -1)) != 10_000
            or int(stream_seal.get("sentinel_request_count", -1)) != 100
            or int(stream_seal.get("batch_count", -1)) != 100
            or int(stream_seal.get("batch_size", -1)) != 100
            or int(stream_seal.get("sample_duplication_count", -1)) != 0
        ):
            raise AnalysisBoundary(f"sealed stream content differs: {spec.model}/{spec.method}")
        training = list(stream_seal.get("training", ()))
        sentinel_order = [str(row["request_sha256"]) for row in stream_seal.get("sentinel", ())]
        if len(training) != 10_000 or len(sentinel_order) != 100 or len(set(sentinel_order)) != 100:
            raise AnalysisBoundary("sealed stream membership denominator differs")
        training_by_batch = {
            batch: [str(row["request_sha256"]) for row in training if int(row["batch_index"]) == batch]
            for batch in range(1, 101)
        }
        batch_digests = list(stream_seal.get("batch_ordered_request_digest_v1", ()))
        if any(len(values) != 100 or len(set(values)) != 100 for values in training_by_batch.values()) or len(batch_digests) != 100:
            raise AnalysisBoundary("sealed stream B100 partition differs")
    _validate_model_binding(result)
    if result.get("terminal_w0_restore", {}).get("exact") is not True or result.get("terminal_cache_restore", {}).get("exact") is not True:
        raise AnalysisBoundary(f"terminal W0/cache restore differs: {spec.model}/{spec.method}")
    if result.get("terminal_w0_restore", {}).get("sha256") != result.get("initial_w0", {}).get("sha256"):
        raise AnalysisBoundary(f"terminal W0 bytes differ: {spec.model}/{spec.method}")

    journal_refs = result.get("journals", ())
    checkpoint_refs = result.get("checkpoints", ())
    if len(journal_refs) != expected_batches or len(checkpoint_refs) != expected_checkpoints:
        raise AnalysisBoundary(f"raw member count differs: {spec.model}/{spec.method}")
    checkpoint_by_batch = {int(value["batch_index"]): value for value in checkpoint_refs}
    expected_cp = lock.expected_checkpoint_batches if expected_checkpoints == 8 else tuple(sorted(checkpoint_by_batch))
    if tuple(sorted(checkpoint_by_batch)) != tuple(expected_cp):
        raise AnalysisBoundary(f"checkpoint schedule differs: {spec.model}/{spec.method}")

    chain = canonical_hash({"campaign_id": result["campaign_id"], "genesis": True})
    checkpoints: list[dict[str, Any]] = []
    journals: list[dict[str, Any]] = []

    def validate_checkpoint(ref: Mapping[str, Any], batch_index: int) -> None:
        nonlocal chain
        path = Path(ref["path"])
        member = _member(path, "CHECKPOINT_JSON", str(ref["sha256"]), required_mode=0o600)
        member["relative_path"] = path.resolve().relative_to(cell_root).as_posix()
        checkpoint = load_json(path)
        if (
            checkpoint.get("model") != spec.model
            or checkpoint.get("method") != spec.method
            or int(checkpoint.get("batch_index", -1)) != batch_index
            or int(checkpoint.get("accepted_edit_count", -1)) != batch_index * 100
            or checkpoint.get("journal_chain_root") != chain
        ):
            raise AnalysisBoundary(f"checkpoint binding differs: {spec.model}/{spec.method}/B{batch_index}")
        state = checkpoint.get("state", {})
        state_path = Path(state["path"])
        state_member = _member(
            state_path,
            "CHECKPOINT_STATE_BYTES_ONLY",
            str(state["sha256"]),
            int(state["bytes"]),
            required_mode=0o600,
        )
        state_member["relative_path"] = state_path.resolve().relative_to(cell_root).as_posix()
        if (
            int(state.get("batch_index", -1)) != batch_index
            or int(state.get("accepted_edit_count", -1)) != batch_index * 100
            or state.get("journal_chain_root") != chain
            or ref.get("state") != state
        ):
            raise AnalysisBoundary(f"checkpoint state binding differs: {spec.model}/{spec.method}/B{batch_index}")
        probe = checkpoint.get("probe", {})
        contamination = probe.get("probe_contamination", {})
        if (
            int(probe.get("sentinel_request_count", -1)) != 100
            or int(probe.get("direct_z_recompute_count", -1)) != 0
            or int(probe.get("direct_z_optimizer_count", -1)) != 100
            or contamination.get("weight_restore_exact") is not True
            or contamination.get("cache_restore_exact") is not True
            or int(contamination.get("accepted_history_append_count", -1)) != 0
            or int(contamination.get("decision_influence_count", -1)) != 0
        ):
            raise AnalysisBoundary(f"checkpoint probe contamination differs: {spec.model}/{spec.method}/B{batch_index}")
        if batch_index > 0 and not isinstance(checkpoint.get("functional"), Mapping):
            raise AnalysisBoundary(f"checkpoint functional panel absent: {spec.model}/{spec.method}/B{batch_index}")
        _validate_checkpoint_probe(
            checkpoint,
            arm=spec,
            batch_index=batch_index,
            expected_sentinel_order=sentinel_order,
            expected_sentinel_order_sha256=(lock.sentinel_order_root if sentinel_order is not None else None),
            expected_current_order=(training_by_batch.get(batch_index) if batch_index > 0 else None),
        )
        raw_members.extend((member, state_member))
        checkpoints.append(checkpoint)
        chain = extend_hash_chain(chain, str(ref["sha256"]), 10_000 + batch_index)
        if ref.get("chain_after") != chain:
            raise AnalysisBoundary(f"checkpoint chain differs: {spec.model}/{spec.method}/B{batch_index}")

    validate_checkpoint(checkpoint_by_batch[0], 0)
    expected_entry = dict(result.get("initial_w0", {}).get("sha256", {}))
    prior_cache_exit: str | None = None
    for offset, ref in enumerate(journal_refs, start=1):
        if int(ref.get("batch_index", -1)) != offset or int(ref.get("request_count", -1)) != 100:
            raise AnalysisBoundary(f"journal reference order differs: {spec.model}/{spec.method}/B{offset}")
        path = Path(ref["path"])
        member = _member(path, "B100_JOURNAL", str(ref["sha256"]), required_mode=0o600)
        member["relative_path"] = path.resolve().relative_to(cell_root).as_posix()
        journal = load_json(path)
        expected_entry, prior_cache_exit = _validate_journal(
            journal,
            arm=spec,
            batch_index=offset,
            expected_entry=expected_entry,
            prior_cache_exit=prior_cache_exit,
            expected_request_order=training_by_batch.get(offset),
            expected_batch_order_sha256=(str(batch_digests[offset - 1]) if batch_digests else None),
        )
        if dict(ref.get("weight_commit_sha256", {})) != dict(expected_entry):
            raise AnalysisBoundary(f"journal commit reference differs: {spec.model}/{spec.method}/B{offset}")
        if str(ref.get("cache_exit_sha256")) != str(prior_cache_exit):
            raise AnalysisBoundary(f"journal cache reference differs: {spec.model}/{spec.method}/B{offset}")
        chain = extend_hash_chain(chain, str(ref["sha256"]), offset)
        if ref.get("chain_after") != chain:
            raise AnalysisBoundary(f"journal chain differs: {spec.model}/{spec.method}/B{offset}")
        journals.append(journal)
        raw_members.append(member)
        if offset in checkpoint_by_batch:
            validate_checkpoint(checkpoint_by_batch[offset], offset)
    if result.get("journal_chain_root") != chain:
        raise AnalysisBoundary(f"terminal chain root differs: {spec.model}/{spec.method}")
    if result.get("terminal_committed_weight_sha256") != expected_entry:
        raise AnalysisBoundary(f"terminal committed W identity differs: {spec.model}/{spec.method}")
    for kind, paths in (("SLURM_LOG", spec.logs), ("EXECUTION_RECEIPT", spec.receipts)):
        external.extend(_member(path, kind) for path in paths)
    for exclusion in spec.technical_exclusions:
        external.extend(_member(path, "PURE_TECHNICAL_EXCLUSION") for path in exclusion.paths)
    if len(raw_members) != 117 and expected_batches == 100 and expected_checkpoints == 8:
        raise AnalysisBoundary(f"canonical raw member count differs: {spec.model}/{spec.method}")
    raw_root = canonical_hash(
        [
            [row["relative_path"], row["sha256"], row["bytes"]]
            for row in sorted(raw_members, key=lambda value: value["relative_path"])
        ]
    )
    return ValidatedArm(
        spec=spec,
        result=result,
        journals=journals,
        checkpoints=checkpoints,
        raw_members=raw_members,
        raw_member_root=raw_root,
        external_members=external,
    )


def validate_campaign(lock: InputLock) -> list[ValidatedArm]:
    by_cell = {(arm.model, arm.method): arm for arm in lock.arms}
    validated = [validate_arm(by_cell[key], lock) for key in CELL_ORDER]
    result_streams = {
        (
            arm.result["stream"]["root"],
            arm.result["stream"]["order"],
            arm.result["stream"]["sentinel_order"],
        )
        for arm in validated
    }
    if len(result_streams) != 1:
        raise AnalysisBoundary("four-arm stream/order denominator differs")
    functional_signatures = []
    for arm in validated:
        checkpoints = []
        for checkpoint in arm.checkpoints:
            functional = checkpoint["functional"]
            if int(checkpoint["batch_index"]) == 0:
                cohorts = {
                    "sentinel_pre_edit": [
                        row["request_sha256"]
                        for row in functional["sentinel_pre_edit"]["records"]
                    ]
                }
                sampling = None
            else:
                cohorts = {
                    "current": [row["request_sha256"] for row in functional["current"]["records"]],
                    **{
                        f"retention_{name}": [row["request_sha256"] for row in panel["records"]]
                        for name, panel in functional["retention"].items()
                    },
                }
                sampling = functional["sampling_identity"]
            checkpoints.append(
                [int(checkpoint["batch_index"]), sampling, canonical_hash(cohorts)]
            )
        functional_signatures.append(checkpoints)
    if any(value != functional_signatures[0] for value in functional_signatures[1:]):
        raise AnalysisBoundary("four-arm checkpoint functional cohorts differ")
    return validated


def write_json_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)


def member_root(rows: Iterable[Mapping[str, Any]]) -> str:
    return canonical_hash(
        [
            [str(row["path"]), int(row["bytes"]), str(row["sha256"])]
            for row in sorted(rows, key=lambda value: str(value["path"]))
        ]
    )


__all__ = [
    "AnalysisBoundary",
    "ArmSpec",
    "CELL_ORDER",
    "CHECKPOINT_BATCHES",
    "IDEAL_Q",
    "INSTRUCTION_ID",
    "InputLock",
    "LAYERS",
    "Q_LABELS",
    "TechnicalExclusion",
    "ValidatedArm",
    "canonical_hash",
    "extend_hash_chain",
    "load_json",
    "member_root",
    "sha256_file",
    "validate_arm",
    "validate_campaign",
    "write_json_once",
]
