"""Raw-free factual analyzer for matched BGODE-R1 S1 terminals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "ode-edit-bgode-r1-s1-matched-analysis/v1"
TERMINAL_SCHEMA = "ode-edit-bgode-r1-s1-six-arm-terminal/v1"
TERMINAL_STATUS = "BGODE_R1_S1_TERMINAL_PASS"
ARM_ORDER = (
    "official-native-alphaedit-bypass",
    "plain-dynamic",
    "fisher-only-dynamic",
    "full-moving-barrier-dynamic",
    "one-step-full-barrier",
    "frozen-field-n4-split",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_terminal(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f"regular mode0600 terminal required: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != TERMINAL_SCHEMA
        or payload.get("status") != TERMINAL_STATUS
        or tuple(payload.get("arm_order", ())) != ARM_ORDER
        or payload.get("scientific_promotion") is not False
    ):
        raise ValueError(f"BGODE terminal header differs: {path}")
    sample = payload.get("sample")
    model = payload.get("model")
    if (
        not isinstance(sample, Mapping)
        or sample.get("case_id") != "19795"
        or sample.get("batch_label") != "B1"
        or sample.get("request_ordinal") != 0
        or not isinstance(model, Mapping)
        or model.get("dtype") != "torch.float32"
    ):
        raise ValueError(f"BGODE terminal scientific binding differs: {path}")
    inventory = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "mode": "0600",
        "source_head": payload.get("source_head"),
        "source_tree": payload.get("source_tree"),
        "run_id": payload.get("run_id"),
        "model_alias": model.get("model_alias"),
    }
    return payload, inventory


def _nll(observation: Mapping[str, Any], key: str) -> float | None:
    value = observation.get(key)
    if not isinstance(value, Mapping):
        return None
    result = value.get("nll")
    return float(result) if isinstance(result, (int, float)) and not isinstance(result, bool) else None


def _exact(observation: Mapping[str, Any], key: str) -> bool | None:
    value = observation.get(key)
    if not isinstance(value, Mapping) or not isinstance(value.get("exact_satisfied"), bool):
        return None
    return bool(value["exact_satisfied"])


def _sum_nodes(nodes: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    total = 0.0
    for node in nodes:
        value: Any = node
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
    return total


def _arm_row(payload: Mapping[str, Any], arm: str) -> dict[str, Any]:
    model_alias = str(payload["model"]["model_alias"])
    arm_payload = payload["arms"][arm]
    observation = payload["observations"][arm]
    event = observation["event"]
    nodes = tuple(arm_payload.get("nodes", ()))
    compute = arm_payload.get("compute") if isinstance(arm_payload.get("compute"), Mapping) else {}
    horizon = float(payload["native_horizon"]["T_AE"])
    native_apply = arm_payload.get("official_apply") if isinstance(arm_payload.get("official_apply"), Mapping) else {}
    edit_core = (
        float(native_apply["edit_core_wall_seconds"])
        if isinstance(native_apply.get("edit_core_wall_seconds"), (int, float))
        else _sum_nodes(nodes, ("node_total_wall_seconds",))
    )
    return {
        "model_alias": model_alias,
        "arm": arm,
        "terminal_log_odds": float(event["log_odds"]),
        "target_log_odds": float(payload["native_horizon"]["r_AE"]),
        "absolute_progress_error": abs(float(event["log_odds"]) - float(payload["native_horizon"]["r_AE"])),
        "target_log_probability": float(event["target_log_probability"]),
        "source_log_probability": float(event["source_log_probability"]),
        "pair_mass": float(event["pair_mass"]),
        "normalization_log_residual": float(event["normalization_log_residual"]),
        "rewrite_target_new_nll": _nll(observation, "rewrite_target_new"),
        "rewrite_target_true_nll": _nll(observation, "rewrite_target_true"),
        "rephrase_target_new_nll": _nll(observation, "rephrase_target_new"),
        "rephrase_target_true_nll": _nll(observation, "rephrase_target_true"),
        "rewrite_exact": _exact(observation, "rewrite_target_new"),
        "rephrase_exact": _exact(observation, "rephrase_target_new"),
        "locality_forward_kl": float(observation["locality_forward_kl"]),
        "controller_influence_count": int(observation.get("controller_influence_count", 0)),
        "dictionary_build_count": int(arm_payload.get("dictionary_build_count", 0)),
        "physical_final_write_count": int(arm_payload.get("physical_final_write_count", 0)),
        "physical_layer_apply_count": int(arm_payload.get("physical_layer_apply_count", 0)),
        "model_forward_invocations": int(compute.get("model_forward_invocations", 0)),
        "jvp_call_count": int(compute.get("jvp_call_count", 0)),
        "finite_difference_forward_count": int(compute.get("finite_difference_forward_count", 0)),
        "primal_prefix_count": int(compute.get("primal_prefix_count", 0)),
        "score_buffer_peak_bytes": int(compute.get("score_buffer_peak_bytes", 0)),
        "jvp_component_wall_seconds": float(compute.get("wall_seconds", 0.0)),
        "dictionary_wall_seconds": _sum_nodes(nodes, ("dictionary_wall_seconds",)),
        "physical_action_wall_seconds": _sum_nodes(nodes, ("physical_action", "wall_seconds")),
        "edit_core_wall_seconds": edit_core,
        "cumulative_action_energy": _sum_nodes(nodes, ("update_energy", "total_energy")),
        "history_append_inside_node_count": int(arm_payload.get("history_append_inside_node_count", 0)),
        "heldout_decision_influence_count": int(arm_payload.get("heldout_decision_influence_count", 0)),
        "native_horizon": horizon,
    }


def _node_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    model_alias = str(payload["model"]["model_alias"])
    rows: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        for node in payload["arms"][arm].get("nodes", ()):
            numerical = node.get("numerical") if isinstance(node.get("numerical"), Mapping) else {}
            energy = node.get("update_energy") if isinstance(node.get("update_energy"), Mapping) else {}
            row = {
                "model_alias": model_alias,
                "arm": arm,
                "node": int(node["node"]),
                "time_entry": float(node["time_entry"]),
                "step_size": float(node["step_size"]),
                "entry_log_odds": float(node["event_entry"]["log_odds"]),
                "exit_log_odds": float(node["event_exit"]["log_odds"]),
                "expected_exit_log_odds": float(node["expected_exit_log_odds"]),
                "nonlinear_progress_residual": float(node["nonlinear_progress_residual"]),
                "moving_reference_kl_entry": float(node["moving_reference_kl_entry"]),
                "anchored_kl_entry": float(node["anchored_kl_entry"]),
                "matrix_rank": numerical.get("matrix_rank"),
                "pinv_tolerance": numerical.get("pinv_tolerance"),
                "equality_residual": numerical.get("equality_residual"),
                "stationarity_residual": numerical.get("stationarity_residual"),
                "score_centering_norm": float(node["score_centering_norm"]),
                "update_energy": float(energy["total_energy"]),
                "layer4_share": energy.get("per_layer_share", {}).get("model.layers.4.mlp.down_proj.weight"),
                "layer5_share": energy.get("per_layer_share", {}).get("model.layers.5.mlp.down_proj.weight"),
                "layer6_share": energy.get("per_layer_share", {}).get("model.layers.6.mlp.down_proj.weight"),
                "layer7_share": energy.get("per_layer_share", {}).get("model.layers.7.mlp.down_proj.weight"),
                "layer8_share": energy.get("per_layer_share", {}).get("model.layers.8.mlp.down_proj.weight"),
                "coefficients": _canonical(node["coefficients"]),
                "velocity": _canonical(node["velocity"]),
                "dictionary_wall_seconds": float(node["dictionary_wall_seconds"]),
                "serial_jvp_wall_seconds": float(node["serial_jvp_wall_seconds"]),
                "physical_action_wall_seconds": float(node["physical_action"]["wall_seconds"]),
                "normalization_log_residual_entry": float(node["event_entry"]["normalization_log_residual"]),
                "normalization_log_residual_exit": float(node["event_exit"]["normalization_log_residual"]),
            }
            rows.append(row)
    return rows


def _max_normalization(payload: Mapping[str, Any]) -> float:
    values = [float(payload["initial_event"]["normalization_log_residual"])]
    for arm in ARM_ORDER:
        values.append(float(payload["observations"][arm]["event"]["normalization_log_residual"]))
        for node in payload["arms"][arm].get("nodes", ()):
            values.extend(
                (
                    float(node["event_entry"]["normalization_log_residual"]),
                    float(node["event_exit"]["normalization_log_residual"]),
                )
            )
    return max(values)


def _vector_l2(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))


def _model_summary(payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in rows}
    plain = by_arm["plain-dynamic"]
    fisher = by_arm["fisher-only-dynamic"]
    full = by_arm["full-moving-barrier-dynamic"]
    one = by_arm["one-step-full-barrier"]
    frozen = by_arm["frozen-field-n4-split"]
    full_nodes = payload["arms"]["full-moving-barrier-dynamic"]["nodes"]
    fisher_nodes = payload["arms"]["fisher-only-dynamic"]["nodes"]
    coefficient_differences = [
        _vector_l2(full_node["coefficients"], fisher_node["coefficients"])
        for full_node, fisher_node in zip(full_nodes, fisher_nodes, strict=True)
    ]
    return {
        "schema": SCHEMA,
        "model_alias": payload["model"]["model_alias"],
        "run_id": payload["run_id"],
        "source_head": payload["source_head"],
        "source_tree": payload["source_tree"],
        "sample_identity": payload["sample_identity"],
        "tokenization_identity": payload["tokenization"]["identity"],
        "fixed_z_sha256": payload["fixed_z"]["tensor_sha256"],
        "fixed_z_compute_count": payload["fixed_z"]["compute_count"],
        "fixed_z_recompute_count": payload["fixed_z"]["recompute_count"],
        "native_horizon": payload["native_horizon"],
        "dtype": payload["dtype"],
        "controller_firewall": payload["controller_firewall"],
        "resources": payload["resources"],
        "accepted_z_observation": payload["observations"]["fixed_z_star"],
        "w0_observation": payload["observations"]["w0"],
        "H1_event_validity": {
            "status": "TERMINAL_VALID_PARTITION_RECEIPTS_PASS",
            "max_recorded_normalization_log_residual": _max_normalization(payload),
            "termination": payload["termination"],
        },
        "H2_barrier_attribution": {
            "status": "FACTUAL_ONLY_ACTION_AND_PROGRESS_NOT_MATCHED",
            "full_minus_fisher_terminal_log_odds": full["terminal_log_odds"] - fisher["terminal_log_odds"],
            "full_minus_fisher_absolute_progress_error": full["absolute_progress_error"] - fisher["absolute_progress_error"],
            "full_minus_fisher_rewrite_target_new_nll": full["rewrite_target_new_nll"] - fisher["rewrite_target_new_nll"],
            "full_minus_fisher_rephrase_target_new_nll": full["rephrase_target_new_nll"] - fisher["rephrase_target_new_nll"],
            "full_minus_fisher_locality_forward_kl": full["locality_forward_kl"] - fisher["locality_forward_kl"],
            "full_minus_fisher_cumulative_action_energy": full["cumulative_action_energy"] - fisher["cumulative_action_energy"],
            "full_minus_plain_terminal_log_odds": full["terminal_log_odds"] - plain["terminal_log_odds"],
            "full_minus_plain_absolute_progress_error": full["absolute_progress_error"] - plain["absolute_progress_error"],
            "coefficient_l2_full_vs_fisher_by_node": coefficient_differences,
            "terminal_moving_reference_kl": "NOT_RECORDED_SCHEMA_GAP",
            "promotion": False,
        },
        "H3_ode_attribution": {
            "status": (
                "ONE_STEP_EQUALS_FROZEN_WITHIN_RECORDED_PRECISION_DYNAMIC_DIFFERS"
                if abs(one["terminal_log_odds"] - frozen["terminal_log_odds"]) <= 1.0e-6
                else "ONE_STEP_FROZEN_AND_DYNAMIC_ENDPOINTS_DIFFER"
            ),
            "one_step_minus_frozen_log_odds": one["terminal_log_odds"] - frozen["terminal_log_odds"],
            "one_step_minus_frozen_rewrite_target_new_nll": one["rewrite_target_new_nll"] - frozen["rewrite_target_new_nll"],
            "full_dynamic_minus_frozen_log_odds": full["terminal_log_odds"] - frozen["terminal_log_odds"],
            "full_dynamic_minus_frozen_rewrite_target_new_nll": full["rewrite_target_new_nll"] - frozen["rewrite_target_new_nll"],
            "convergence_claim": False,
        },
        "scientific_promotion": False,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return str(value)


def _model_report(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    alias = summary["model_alias"]
    lines = [
        f"# BGODE-R1 S1 {alias} 단일-request 사실 보고서",
        "",
        "> 경계: canonical Phase123 B1 request000(case19795) 한 건, B=1, Claim-A sequence log-odds, fixed native z* 1회, 여섯 arm 순차 실행입니다. 이 결과는 promotion·CBF invariance·barrier monotonicity·global locality·Euler convergence를 주장하지 않습니다.",
        "",
        "## 핵심 경계",
        "",
        f"- run: `{summary['run_id']}`",
        f"- source: `{summary['source_head']}` / `{summary['source_tree']}`",
        f"- sample identity: `{summary['sample_identity']}`",
        f"- tokenization identity: `{summary['tokenization_identity']}`",
        f"- fixed z SHA: `{summary['fixed_z_sha256']}`; compute/recompute={summary['fixed_z_compute_count']}/{summary['fixed_z_recompute_count']}",
        f"- T_AE={_fmt(summary['native_horizon']['T_AE'])}; r0={_fmt(summary['native_horizon']['r0'])}; r_AE={_fmt(summary['native_horizon']['r_AE'])}",
        f"- FULL-FP32 parameter inventory: `{summary['dtype']['parameter_dtype_counts']}`; autocast/TF32/BF16/FP16/cast={int(summary['dtype']['autocast_enabled'])}/{int(summary['dtype']['tf32_enabled'])}/{summary['dtype']['bf16_conversion_count']}/{summary['dtype']['fp16_conversion_count']}/{summary['dtype']['numeric_storage_cast_count']}",
        "",
        "## arm별 terminal",
        "",
        "| arm | log-odds | |r-r_AE| | LOC KL | cumulative action energy | writes/layers |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {arm} | {r} | {err} | {loc} | {energy} | {writes}/{layers} |".format(
                arm=row["arm"],
                r=_fmt(row["terminal_log_odds"]),
                err=_fmt(row["absolute_progress_error"]),
                loc=_fmt(row["locality_forward_kl"]),
                energy=_fmt(row["cumulative_action_energy"]),
                writes=row["physical_final_write_count"],
                layers=row["physical_layer_apply_count"],
            )
        )
    lines.extend(
        [
            "",
            "### Rewrite",
            "",
            "| arm | target-new NLL | target-true NLL | exact success |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['arm']} | {_fmt(row['rewrite_target_new_nll'])} | {_fmt(row['rewrite_target_true_nll'])} | {_fmt(row['rewrite_exact'])} |"
        )
    lines.extend(
        [
            "",
            "### Rephrase",
            "",
            "| arm | target-new NLL | target-true NLL | exact success |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['arm']} | {_fmt(row['rephrase_target_new_nll'])} | {_fmt(row['rephrase_target_true_nll'])} | {_fmt(row['rephrase_exact'])} |"
        )
    z = summary["accepted_z_observation"]
    w0 = summary["w0_observation"]
    lines.extend(
        [
            "",
            "## accepted z* / W0 observation",
            "",
            "| state | rewrite new/true NLL | rephrase new/true NLL | rewrite/rephrase exact | locality KL |",
            "|---|---:|---:|---:|---:|",
            f"| W0 | {_fmt(_nll(w0, 'rewrite_target_new'))}/{_fmt(_nll(w0, 'rewrite_target_true'))} | {_fmt(_nll(w0, 'rephrase_target_new'))}/{_fmt(_nll(w0, 'rephrase_target_true'))} | {_fmt(_exact(w0, 'rewrite_target_new'))}/{_fmt(_exact(w0, 'rephrase_target_new'))} | {_fmt(w0['locality_forward_kl'])} |",
            f"| fixed z* | {_fmt(_nll(z, 'rewrite_target_new'))}/{_fmt(_nll(z, 'rewrite_target_true'))} | {_fmt(_nll(z, 'rephrase_target_new'))}/{_fmt(_nll(z, 'rephrase_target_true'))} | {_fmt(_exact(z, 'rewrite_target_new'))}/{_fmt(_exact(z, 'rephrase_target_new'))} | {_fmt(z['locality_forward_kl'])} |",
            "",
            "## H1/H2/H3 사실 판정",
            "",
            f"- H1 event validity: `{summary['H1_event_validity']['status']}`; max normalization log residual={_fmt(summary['H1_event_validity']['max_recorded_normalization_log_residual'], 9)}.",
            f"- H2 barrier attribution: `{summary['H2_barrier_attribution']['status']}`. Full−Fisher progress error={_fmt(summary['H2_barrier_attribution']['full_minus_fisher_absolute_progress_error'])}, rewrite NLL={_fmt(summary['H2_barrier_attribution']['full_minus_fisher_rewrite_target_new_nll'])}, locality KL={_fmt(summary['H2_barrier_attribution']['full_minus_fisher_locality_forward_kl'])}. Terminal moving-reference KL은 `NOT_RECORDED_SCHEMA_GAP`입니다.",
            f"- H3 ODE attribution: `{summary['H3_ode_attribution']['status']}`. One-step−Frozen log-odds={_fmt(summary['H3_ode_attribution']['one_step_minus_frozen_log_odds'], 9)}, Full-dynamic−Frozen={_fmt(summary['H3_ode_attribution']['full_dynamic_minus_frozen_log_odds'])}.",
            "",
            "## 계산량",
            "",
            "| arm | dictionary | model forward | JVP | FD forward | edit-core s | dictionary s | JVP s | peak event buffer bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['dictionary_build_count']} | {row['model_forward_invocations']} | {row['jvp_call_count']} | {row['finite_difference_forward_count']} | {_fmt(row['edit_core_wall_seconds'], 3)} | {_fmt(row['dictionary_wall_seconds'], 3)} | {_fmt(row['jvp_component_wall_seconds'], 3)} | {row['score_buffer_peak_bytes']} |"
        )
    lines.extend(
        [
            "",
            f"전체 wall={_fmt(summary['resources']['total_wall_seconds'], 3)}s, model load={_fmt(summary['resources']['model_load_wall_seconds'], 3)}s, peak allocated/reserved={summary['resources']['peak_gpu_allocated_bytes']}/{summary['resources']['peak_gpu_reserved_bytes']} bytes.",
            "",
            "## 제한",
            "",
            "- 단일 request pilot이므로 mean/median/p90이나 population claim은 정의하지 않습니다.",
            "- Full과 Fisher의 terminal progress/action energy가 matched되지 않아 H2의 load-bearing barrier claim은 성립하지 않습니다.",
            "- dictionary helper 내부 physical forward 수는 terminal schema에서 `NOT_RESOLVED_SOURCE_LEVEL_HELPER_INTERNALS`입니다.",
            "- scientific_promotion=false.",
            "",
        ]
    )
    return "\n".join(lines)


def _matched_report(summaries: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# BGODE-R1 S1 Llama/Qwen matched 사실 비교",
        "",
        "> 같은 canonical B1 request000 bytes/order/context와 같은 여섯 arm 정의를 사용했습니다. 모델마다 outcome-free termination과 native T_AE는 독립이며, 한 모델의 결과가 다른 모델 선택에 영향을 주지 않았습니다. promotion=false.",
        "",
        "## 모델별 horizon / H1-H3",
        "",
        "| model | termination | T_AE | H1 max norm residual | Full−Fisher progress error | Full−Fisher rewrite NLL | Full−Frozen log-odds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        h2 = summary["H2_barrier_attribution"]
        h3 = summary["H3_ode_attribution"]
        term = summary["H1_event_validity"]["termination"]
        lines.append(
            f"| {summary['model_alias']} | {term['boundary_string']} ({term['boundary_token_id']}) | {_fmt(summary['native_horizon']['T_AE'])} | {_fmt(summary['H1_event_validity']['max_recorded_normalization_log_residual'], 9)} | {_fmt(h2['full_minus_fisher_absolute_progress_error'])} | {_fmt(h2['full_minus_fisher_rewrite_target_new_nll'])} | {_fmt(h3['full_dynamic_minus_frozen_log_odds'])} |"
        )
    lines.extend(
        [
            "",
            "## arm별 cross-model terminal",
            "",
            "| model | arm | log-odds | progress error | rewrite new NLL | rephrase new NLL | locality KL | cumulative action energy | edit-core s |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['model_alias']} | {row['arm']} | {_fmt(row['terminal_log_odds'])} | {_fmt(row['absolute_progress_error'])} | {_fmt(row['rewrite_target_new_nll'])} | {_fmt(row['rephrase_target_new_nll'])} | {_fmt(row['locality_forward_kl'])} | {_fmt(row['cumulative_action_energy'])} | {_fmt(row['edit_core_wall_seconds'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## 해석 경계",
            "",
            "- H1은 각 모델 내부 event partition receipt의 정규화 사실만 확인합니다.",
            "- H2는 Full/Fisher/Plain의 progress와 action이 matched되지 않으면 barrier가 load-bearing이라고 판정하지 않습니다.",
            "- H3는 dynamic endpoint와 one-step/frozen endpoint의 차이를 관찰할 뿐 수렴을 주장하지 않습니다.",
            "- Llama/Qwen T_AE와 tokenizer termination이 다르므로 raw wall·strength의 cross-model 인과 비교는 하지 않습니다.",
            "- scientific_promotion=false.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _write_once(path, (_canonical(value) + "\n").encode("utf-8"))


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("CSV rows must be nonempty")
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def analyze(terminals: Sequence[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"create-once analysis root exists: {output}")
    loaded = [_read_terminal(path) for path in terminals]
    models = [str(payload["model"]["model_alias"]) for payload, _ in loaded]
    if len(models) != len(set(models)):
        raise ValueError("matched analysis model aliases must be unique")
    sample_identities = {payload["sample_identity"] for payload, _ in loaded}
    request_sha = {payload["sample"]["request_sha256"] for payload, _ in loaded}
    if len(sample_identities) != 1 or len(request_sha) != 1:
        raise ValueError("matched terminals do not use the same sample bytes/order")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    all_rows: list[dict[str, Any]] = []
    all_nodes: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for payload, _ in loaded:
        alias = str(payload["model"]["model_alias"])
        rows = [_arm_row(payload, arm) for arm in ARM_ORDER]
        nodes = _node_rows(payload)
        summary = _model_summary(payload, rows)
        summaries.append(summary)
        all_rows.extend(rows)
        all_nodes.extend(nodes)
        _write_json(output / f"{alias}-summary.json", summary)
        _write_once(output / f"{alias}-arm-summary.csv", _csv_bytes(rows))
        _write_once(output / f"{alias}-node-trajectory.csv", _csv_bytes(nodes))
        _write_once(
            output / f"{alias}-factual-ko.md",
            (_model_report(summary, rows) + "\n").encode("utf-8"),
        )
    matched = {
        "schema": SCHEMA,
        "status": "BGODE_R1_S1_MATCHED_ANALYSIS_PASS",
        "model_count": len(summaries),
        "sample_identity": next(iter(sample_identities)),
        "request_sha256": next(iter(request_sha)),
        "models": summaries,
        "scientific_promotion": False,
    }
    _write_json(output / "matched-summary.json", matched)
    _write_once(output / "matched-arm-summary.csv", _csv_bytes(all_rows))
    _write_once(output / "matched-node-trajectory.csv", _csv_bytes(all_nodes))
    _write_once(
        output / "bgode-r1-s1-llama-qwen-matched-factual-ko.md",
        (_matched_report(summaries, all_rows) + "\n").encode("utf-8"),
    )
    inventory = {
        "schema": "ode-edit-bgode-r1-s1-analysis-input-inventory/v1",
        "terminals": [inventory for _, inventory in loaded],
    }
    _write_json(output / "artifact-inventory.json", inventory)
    member_paths = sorted(
        path for path in output.iterdir() if path.name not in {"analysis-manifest.json", "rooted-receipt.json"}
    )
    entries = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "mode": "0600",
        }
        for path in member_paths
    ]
    manifest = {
        "schema": "ode-edit-bgode-r1-s1-analysis-manifest/v1",
        "entries": entries,
        "entries_root_sha256": _sha256_bytes(_canonical(entries).encode("utf-8")),
    }
    _write_json(output / "analysis-manifest.json", manifest)
    manifest_path = output / "analysis-manifest.json"
    receipt = {
        "schema": "ode-edit-bgode-r1-s1-analysis-rooted-receipt/v1",
        "status": "BGODE_R1_S1_MATCHED_ANALYSIS_PASS",
        "analysis_manifest_sha256": _sha256_file(manifest_path),
        "entries_root_sha256": manifest["entries_root_sha256"],
        "terminal_sha256": {
            item["model_alias"]: item["sha256"] for item in inventory["terminals"]
        },
        "sample_identity": next(iter(sample_identities)),
        "scientific_promotion": False,
    }
    receipt["receipt_identity"] = _sha256_bytes(_canonical(receipt).encode("utf-8"))
    _write_json(output / "rooted-receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--terminal", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = analyze(args.terminal, args.output)
    print(_canonical(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
