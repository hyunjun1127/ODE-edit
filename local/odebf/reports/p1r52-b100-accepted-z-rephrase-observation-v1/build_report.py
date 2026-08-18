#!/usr/bin/env python3
"""Build the raw-free P1R52 B100 accepted-z rewrite/rephrase report package."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
INSTRUCTION_ID = "ODEEDIT-S05-P1R52-LLAMA-B100-ACCEPTED-Z-REPHRASE-OBS-V1"
BASE_REPAIR = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r52-b100-accepted-z-rephrase-obs-tech-r1-v1"
)
R52_REPAIR = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r52-b100-accepted-z-rephrase-obs-tech-r1-r52-v1"
)
FOUR_ARM = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r52-llama-seq-10xb100-fourarm-r1-v1"
    "/local/odebf/reports/p1r52-llama-sequential-10xb100-fourarm-v1"
)

ARMS = {
    "memit": {
        "label": "Official EasyEdit MEMIT Sequential",
        "role": "official-memit-sequential",
        "obs": BASE_REPAIR / "local/odebf/results/"
        "s05-p1r52-official-memit-sequential-10xb100-accepted-z-rephrase-obs-tech-r1-v1",
    },
    "alphaedit": {
        "label": "Official EasyEdit AlphaEdit Sequential (cache_c ON)",
        "role": "native-alphaedit-sequential-cache-on-corrected",
        "obs": BASE_REPAIR / "local/odebf/results/"
        "s05-p1r52-official-alphaedit-sequential-cache-on-10xb100-accepted-z-rephrase-obs-tech-r1-v1",
    },
    "r52_h_on": {
        "label": "P1R52 Repair-R1 Soft Sequential / Structural-H ON",
        "role": "r52-soft-sequential-h",
        "obs": R52_REPAIR / "local/odebf/results/"
        "s05-p1r52-llama-soft-sequential-structuralh-on-10xb100-accepted-z-rephrase-obs-tech-r1-r52-v1",
    },
    "r52_h_off": {
        "label": "P1R52 Repair-R1 Soft Sequential / AlphaCache ON, Structural-H OFF",
        "role": "r52-soft-sequential-alphacache-on-structuralh-off",
        "obs": R52_REPAIR / "local/odebf/results/"
        "s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb100-accepted-z-rephrase-obs-tech-r1-r52-v1",
    },
}

W_AGGREGATE = FOUR_ARM / "p1r52-b100-four-arm-aggregate.json"
W_BATCH = FOUR_ARM / "p1r52-b100-four-arm-batch-checkpoints.json"
W_REQUEST = FOUR_ARM / "p1r52-b100-four-arm-request-retention.json"
W_ROUTING = FOUR_ARM / "p1r52-b100-r52-controller-routing.json"
W_TARGET_WRITER = FOUR_ARM / "p1r52-b100-r52-target-writer-step-v2.json"
W_TERMINAL_Z = FOUR_ARM / "p1r52-b100-r52-terminal-z-w-v2.json"
W_INTEGRITY = FOUR_ARM / "p1r52-b100-four-arm-integrity-compute.json"
W_MANIFEST = FOUR_ARM / "manifest-v2.json"
W_RECEIPT = FOUR_ARM / "rooted-receipt-v2.json"


def load(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing/nonregular input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def dump_once(name: str, value: Any) -> Path:
    path = HERE / name
    write_once(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
    )
    return path


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise RuntimeError("empty distribution")
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def distribution(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    return {
        "count": len(rows),
        "mean": statistics.fmean(rows),
        "median": statistics.median(rows),
        "p90": quantile(rows, 0.9),
        "max": max(rows),
    }


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(float(value) for value in values)


def flatten(values: Iterable[Iterable[float]]) -> list[float]:
    return [float(item) for row in values for item in row]


def metric_view(metric: dict[str, Any], index: int) -> dict[str, Any]:
    new = [float(x) for x in metric["target_new_nll_by_request"][index]]
    true = [float(x) for x in metric["target_true_nll_by_request"][index]]
    return {
        "bits": list(metric["per_request_bits"][index]),
        "correct": int(metric["per_request_correct"][index]),
        "required": int(metric["per_request_required"][index]),
        "rate": int(metric["per_request_correct"][index])
        / int(metric["per_request_required"][index]),
        "strict_all_prompts_pass": int(metric["strict_all_prompt_bits"][index]),
        "target_new_nll": new,
        "target_true_nll": true,
        "target_new_nll_mean": mean(new),
        "target_true_nll_mean": mean(true),
        "target_true_minus_new_margin_mean": mean(t - n for n, t in zip(new, true, strict=True)),
    }


def summarize_panel(rows: list[dict[str, Any]], panel: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc"):
        values = [row[panel][metric] for row in rows]
        new_values = flatten(value["target_new_nll"] for value in values)
        true_values = flatten(value["target_true_nll"] for value in values)
        metrics[metric] = {
            "prompt_numerator": sum(value["correct"] for value in values),
            "prompt_denominator": sum(value["required"] for value in values),
            "prompt_rate": sum(value["correct"] for value in values)
            / sum(value["required"] for value in values),
            "strict_request_numerator": sum(
                value["strict_all_prompts_pass"] for value in values
            ),
            "strict_request_denominator": len(values),
            "strict_request_rate": sum(
                value["strict_all_prompts_pass"] for value in values
            )
            / len(values),
            "target_new_nll": distribution(new_values),
            "target_true_nll": distribution(true_values),
            "target_true_minus_new_margin": distribution(
                t - n for n, t in zip(new_values, true_values, strict=True)
            ),
            "bit_vector_sha256": canonical_hash([value["bits"] for value in values]),
            "strict_bit_vector_sha256": canonical_hash(
                [value["strict_all_prompts_pass"] for value in values]
            ),
        }
    return metrics


def w_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "bits": list(metric["bits"]),
        "correct": int(metric["correct"]),
        "required": int(metric["required"]),
        "rate": float(metric["rate"]),
        "strict_all_prompts_pass": int(metric["strict_all_prompts_pass"]),
        "target_new_nll": [float(metric["target_new_nll_mean"])],
        "target_true_nll": [float(metric["target_old_nll_mean"])],
        "target_new_nll_mean": float(metric["target_new_nll_mean"]),
        "target_true_nll_mean": float(metric["target_old_nll_mean"]),
        "target_true_minus_new_margin_mean": float(
            metric["target_old_minus_new_margin_mean"]
        ),
    }


def normalize_sealed_summary(panel: dict[str, Any]) -> dict[str, Any]:
    """Normalize the sealed four-arm summary without changing its statistics."""
    out: dict[str, Any] = {}
    for metric in ("rewrite_success", "rewrite_acc", "paraphrase_success", "paraphrase_acc"):
        value = panel[metric]
        out[metric] = {
            "prompt_numerator": value["numerator"],
            "prompt_denominator": value["denominator"],
            "prompt_rate": value["rate"],
            "strict_request_numerator": value["strict_numerator"],
            "strict_request_denominator": value["strict_denominator"],
            "strict_request_rate": value["strict_rate"],
            "target_new_nll": value["target_new_nll"],
            "target_true_nll": value["target_old_nll"],
            "target_true_minus_new_margin": value["margin"],
            "bit_vector_sha256": value["bit_vector_sha256"],
            "strict_bit_vector_sha256": value["strict_bit_vector_sha256"],
        }
    return out


def count_text(metric: dict[str, Any], strict: bool = False) -> str:
    prefix = "strict_request_" if strict else "prompt_"
    return (
        f"{metric[prefix+'numerator']}/{metric[prefix+'denominator']} "
        f"({100*metric[prefix+'rate']:.2f}%)"
    )


def nll_text(metric: dict[str, Any]) -> str:
    return f"{metric['target_new_nll']['mean']:.6f}"


def main() -> None:
    input_paths = [
        W_AGGREGATE,
        W_BATCH,
        W_REQUEST,
        W_ROUTING,
        W_TARGET_WRITER,
        W_TERMINAL_Z,
        W_INTEGRITY,
        W_MANIFEST,
        W_RECEIPT,
    ]
    w_aggregate = load(W_AGGREGATE)
    w_batch = load(W_BATCH)
    w_requests = load(W_REQUEST)
    w_routing = load(W_ROUTING)
    w_target_writer = load(W_TARGET_WRITER)
    w_terminal_z = load(W_TERMINAL_Z)
    w_integrity = load(W_INTEGRITY)
    if w_requests.get("row_count") != 4000:
        raise RuntimeError("sealed W request denominator differs")
    if w_batch.get("row_count") != 40:
        raise RuntimeError("sealed W batch denominator differs")
    w_batch_by_key = {
        (row["arm"], int(row["round"])): row for row in w_batch["rows"]
    }
    w_by_key = {
        (row["arm"], int(row["round"]), int(row["request_index"])): row
        for row in w_requests["rows"]
    }
    if len(w_by_key) != 4000:
        raise RuntimeError("sealed W request key uniqueness differs")

    per_request: list[dict[str, Any]] = []
    obs_receipts: list[dict[str, Any]] = []
    for arm, config in ARMS.items():
        root = config["obs"]
        terminal = root / "terminal.json"
        manifest = root / "manifest.json"
        input_paths.extend([terminal, manifest])
        terminal_value = load(terminal)
        load(manifest)
        if (
            terminal_value.get("round_count") != 10
            or terminal_value.get("request_count") != 1000
            or terminal_value.get("batch_entry_metrics_status")
            != "REMOVED_BY_USER_AMENDMENT"
        ):
            raise RuntimeError(f"observation terminal denominator/policy differs: {arm}")
        for batch in range(1, 11):
            path = root / f"raw/batches/b{batch:02d}/accepted-z-rephrase-observation.json"
            input_paths.append(path)
            receipt = load(path)
            obs_receipts.append(receipt)
            if (
                receipt.get("round") != batch
                or receipt.get("role") != config["role"]
                or receipt.get("observation_only") is not True
                or receipt.get("sealed_commit_weight_exact") is not True
                or receipt.get("batch_entry_W_metrics_count") != 0
                or receipt.get("added_backward_count") != 0
                or receipt.get("added_generation_call_count") != 0
                or receipt.get("action_influence_count") != 0
                or receipt["binding"].get("proxy_or_imputation_count") != 0
                or receipt["binding"].get("native_definition_preserved") is not True
                or receipt["overlay"].get("maximum_assignment_error") != 0.0
            ):
                raise RuntimeError(f"observation invariant differs: {arm}/B{batch}")
            scores = receipt["scores"]
            for index in range(100):
                old = w_by_key[(arm, batch, index)]
                z = {
                    metric: metric_view(scores[metric], index)
                    for metric in (
                        "rewrite_success",
                        "rewrite_acc",
                        "paraphrase_success",
                        "paraphrase_acc",
                    )
                }
                post = {
                    metric: w_metric(old["immediate_post"][metric])
                    for metric in (
                        "rewrite_success",
                        "rewrite_acc",
                        "paraphrase_success",
                        "paraphrase_acc",
                    )
                }
                final = {
                    metric: w_metric(old["final_W10"][metric])
                    for metric in (
                        "rewrite_success",
                        "rewrite_acc",
                        "paraphrase_success",
                        "paraphrase_acc",
                    )
                }
                per_request.append(
                    {
                        "arm": arm,
                        "role": config["role"],
                        "round": batch,
                        "request_index": index,
                        "case_id": old["case_id"],
                        "request_order_sha256": receipt["binding"][
                            "request_order_sha256"
                        ],
                        "history_width_at_entry": old["history_width_at_entry"],
                        "accepted_z": z,
                        "W_immediate_post": post,
                        "W_final_W10": final,
                        "Wpost_minus_z_rewrite_new_nll": post["rewrite_success"][
                            "target_new_nll_mean"
                        ]
                        - z["rewrite_success"]["target_new_nll_mean"],
                        "Wpost_minus_z_rephrase_new_nll": post["paraphrase_success"][
                            "target_new_nll_mean"
                        ]
                        - z["paraphrase_success"]["target_new_nll_mean"],
                        "Wfinal_minus_z_rewrite_new_nll": final["rewrite_success"][
                            "target_new_nll_mean"
                        ]
                        - z["rewrite_success"]["target_new_nll_mean"],
                        "Wfinal_minus_z_rephrase_new_nll": final[
                            "paraphrase_success"
                        ]["target_new_nll_mean"]
                        - z["paraphrase_success"]["target_new_nll_mean"],
                        "identity_sha256": canonical_hash(
                            {
                                "arm": arm,
                                "round": batch,
                                "request_index": index,
                                "case_id": old["case_id"],
                                "request_order_sha256": receipt["binding"][
                                    "request_order_sha256"
                                ],
                                "observation_identity": receipt["identity_sha256"],
                                "W_identity": old["identity_sha256"],
                            }
                        ),
                    }
                )

    if len(per_request) != 4000 or len(obs_receipts) != 40:
        raise RuntimeError("accepted-z observation denominator differs")

    aggregate_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_rows = [row for row in per_request if row["arm"] == arm]
        sealed_arm = w_aggregate["arms"][arm]
        aggregate_rows.append(
            {
                "arm": arm,
                "label": ARMS[arm]["label"],
                "request_count": len(arm_rows),
                "accepted_z": summarize_panel(arm_rows, "accepted_z"),
                "W_immediate_post": normalize_sealed_summary(
                    sealed_arm["immediate_post_B100_panels"]
                ),
                "W_final_W10": normalize_sealed_summary(sealed_arm["final_W10_B1000"]),
                "NLL_aggregation_unit": "PROMPT_LEVEL_MATCHED_DENOMINATOR",
                "Wpost_minus_z_rewrite_new_nll": distribution(
                    row["Wpost_minus_z_rewrite_new_nll"] for row in arm_rows
                ),
                "Wpost_minus_z_rephrase_new_nll": distribution(
                    row["Wpost_minus_z_rephrase_new_nll"] for row in arm_rows
                ),
                "Wfinal_minus_z_rewrite_new_nll": distribution(
                    row["Wfinal_minus_z_rewrite_new_nll"] for row in arm_rows
                ),
                "Wfinal_minus_z_rephrase_new_nll": distribution(
                    row["Wfinal_minus_z_rephrase_new_nll"] for row in arm_rows
                ),
            }
        )
        for batch in range(1, 11):
            rows = [row for row in arm_rows if row["round"] == batch]
            batch_rows.append(
                {
                    "arm": arm,
                    "round": batch,
                    "request_count": len(rows),
                    "accepted_z": summarize_panel(rows, "accepted_z"),
                    "W_immediate_post": normalize_sealed_summary(
                        w_batch_by_key[(arm, batch)]["immediate_post"]
                    ),
                    "W_final_W10": summarize_panel(rows, "W_final_W10"),
                    "NLL_aggregation_unit": {
                        "accepted_z": "PROMPT_LEVEL",
                        "W_immediate_post": "PROMPT_LEVEL",
                        "W_final_W10": "PER_REQUEST_PROMPT_MEAN_FOR_BATCH_AGE_VIEW",
                    },
                    "Wpost_minus_z_rewrite_new_nll": distribution(
                        row["Wpost_minus_z_rewrite_new_nll"] for row in rows
                    ),
                    "Wpost_minus_z_rephrase_new_nll": distribution(
                        row["Wpost_minus_z_rephrase_new_nll"] for row in rows
                    ),
                    "Wfinal_minus_z_rewrite_new_nll": distribution(
                        row["Wfinal_minus_z_rewrite_new_nll"] for row in rows
                    ),
                    "Wfinal_minus_z_rephrase_new_nll": distribution(
                        row["Wfinal_minus_z_rephrase_new_nll"] for row in rows
                    ),
                }
            )

    hard_cohorts: dict[str, Any] = {}
    for arm in ARMS:
        rows = [row for row in per_request if row["arm"] == arm]
        categories = {
            "z_rewrite_success_fail": [
                row
                for row in rows
                if row["accepted_z"]["rewrite_success"]["strict_all_prompts_pass"] == 0
            ],
            "z_rephrase_strict_fail": [
                row
                for row in rows
                if row["accepted_z"]["paraphrase_success"]["strict_all_prompts_pass"]
                == 0
            ],
            "z_rewrite_success_Wpost_failure": [
                row
                for row in rows
                if row["accepted_z"]["rewrite_success"]["strict_all_prompts_pass"] == 1
                and row["W_immediate_post"]["rewrite_success"][
                    "strict_all_prompts_pass"
                ]
                == 0
            ],
            "z_rephrase_success_Wpost_failure": [
                row
                for row in rows
                if row["accepted_z"]["paraphrase_success"]["strict_all_prompts_pass"]
                == 1
                and row["W_immediate_post"]["paraphrase_success"][
                    "strict_all_prompts_pass"
                ]
                == 0
            ],
        }
        hard_cohorts[arm] = {
            name: {
                "count": len(values),
                "request_keys": [
                    [row["round"], row["request_index"], row["case_id"]]
                    for row in values
                ],
            }
            for name, values in categories.items()
        }

    route_summary: dict[str, Any] = {}
    for arm in ("r52_h_on", "r52_h_off"):
        target = [row for row in w_target_writer["rows"] if row["arm"] == arm]
        routing = [row for row in w_routing["rows"] if row["arm"] == arm]
        terminal_z = [row for row in w_terminal_z["rows"] if row["arm"] == arm]
        route_summary[arm] = {
            "step_rows": len(target),
            "request_decisions": {
                "PRIMARY": sum(row["primary"] for row in target),
                "RESCUE": sum(row["rescue"] for row in target),
                "CURRENT": sum(row["current"] for row in target),
                "ACTIVE_GRADIENT": sum(row["active_gradient"] for row in target),
                "CLAMP_HIT": sum(row["clamp_hit"] for row in target),
            },
            "fallback_count": sum(bool(row["fallback"]) for row in routing),
            "negative_actual_count": sum(bool(row["negative_actual"]) for row in routing),
            "h_status_counts": dict(Counter(row["h_status"] for row in routing)),
            "history_widths": sorted(set(row["history_width"] for row in routing)),
            "alpha_cache_append_total": sum(row["alpha_cache_append"] for row in routing),
            "alpha_cache_consume_total": sum(row["alpha_cache_consume"] for row in routing),
            "structural_h_decision_influence_sum": sum(
                row["structural_h_decision_influence"] for row in routing
            ),
            "terminal_target_objective_full_six": terminal_z,
            "target_objective_boundary": (
                "METHOD_TARGET_OBJECTIVE_FULL_SIX_NOT_REWRITE_OR_REPHRASE_NLL"
            ),
        }

    observation_compute: dict[str, Any] = {}
    for arm in ARMS:
        receipts = [row for row in obs_receipts if row["role"] == ARMS[arm]["role"]]
        observation_compute[arm] = {
            "batch_count": len(receipts),
            "model_forward_count": sum(row["model_forward_count"] for row in receipts),
            "processed_token_count": sum(row["processed_token_count"] for row in receipts),
            "wall_seconds": sum(row["wall_seconds"] for row in receipts),
            "added_backward_count": sum(row["added_backward_count"] for row in receipts),
            "added_generation_call_count": sum(
                row["added_generation_call_count"] for row in receipts
            ),
            "action_influence_count": sum(row["action_influence_count"] for row in receipts),
            "hook_call_count": sum(row["overlay"]["hook_call_count"] for row in receipts),
            "patched_row_count": sum(row["overlay"]["patched_row_count"] for row in receipts),
            "locality_unpatched_row_count": sum(
                row["overlay"]["locality_unpatched_row_count"] for row in receipts
            ),
            "maximum_assignment_error": max(
                row["overlay"]["maximum_assignment_error"] for row in receipts
            ),
        }

    inputs = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(set(input_paths), key=str)
    ]
    request_path = dump_once(
        "p1r52-accepted-z-per-request.json",
        {
            "schema": "p1r52-b100-accepted-z-observation/per-request/v1",
            "row_count": len(per_request),
            "rows": per_request,
        },
    )
    batch_path = dump_once(
        "p1r52-accepted-z-per-batch.json",
        {
            "schema": "p1r52-b100-accepted-z-observation/per-batch/v1",
            "row_count": len(batch_rows),
            "rows": batch_rows,
        },
    )
    aggregate_path = dump_once(
        "p1r52-accepted-z-aggregates.json",
        {
            "schema": "p1r52-b100-accepted-z-observation/aggregates/v1",
            "row_count": len(aggregate_rows),
            "rows": aggregate_rows,
            "common_W0": w_aggregate["common_W0_B1_panel"],
            "batch_entry_aggregate_policy": "REMOVED_NONCANONICAL_UNUSED",
        },
    )
    hard_path = dump_once(
        "p1r52-accepted-z-hard-cohorts.json",
        {
            "schema": "p1r52-b100-accepted-z-observation/hard-cohorts/v1",
            "arms": hard_cohorts,
            "classification_boundary": "ASSOCIATION_ONLY_NO_ISOLATED_CAUSAL_CLAIM",
        },
    )
    compute_path = dump_once(
        "p1r52-accepted-z-routing-compute.json",
        {
            "schema": "p1r52-b100-accepted-z-observation/routing-compute/v1",
            "accepted_z_observation": observation_compute,
            "r52_route_summary": route_summary,
            "sealed_W_integrity_compute": w_integrity,
        },
    )

    lines = [
        "# P1R52 Llama 10×B100 accepted-z rewrite/rephrase 관측 보고서",
        "",
        "> 네 방법의 method-native accepted z를 각 방법의 실제 target layer/subject-last 위치에 직접 주입한 observation-only 결과입니다. 기존 물리 W 결과는 봉인된 10×B100 보고서에서 결합했습니다.",
        "",
        "## 핵심 경계",
        "",
        "- 네 arm 모두 10개 B100, arm당 1,000 requests를 사용하며 missing/imputation/proxy는 0입니다.",
        "- z 관측은 writer/router/controller/history/cache/selection/materialization 영향 0, 추가 backward/generation 0입니다.",
        "- method-specific batch-entry W_(b-1) aggregate는 사용자 지시에 따라 측정·분석·보고하지 않습니다.",
        "- 기존 `full-six z target_new NLL`은 방법 내부 target-objective panel이며, 이 보고서의 rewrite/rephrase NLL과 다른 지표입니다.",
        "- 방법마다 accepted-z 생성 정의는 native implementation을 보존했습니다. 따라서 raw NLL은 나란히 제시하지만 완전히 동일한 latent intervention이라는 인과 주장은 하지 않습니다.",
        "",
        "## B1–B10 및 전체: accepted-z와 물리 W target-new NLL",
        "",
        "|Arm|범위|z rewrite|z rephrase|W post rewrite|W post rephrase|W10 rewrite|W10 rephrase|Wpost−z rewrite|Wpost−z rephrase|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in batch_rows + aggregate_rows:
        scope = f"B{row['round']}" if "round" in row else "전체 1000"
        z = row["accepted_z"]
        post = row["W_immediate_post"]
        final = row["W_final_W10"]
        lines.append(
            f"|{ARMS[row['arm']]['label']}|{scope}|"
            f"{nll_text(z['rewrite_success'])}|{nll_text(z['paraphrase_success'])}|"
            f"{nll_text(post['rewrite_success'])}|{nll_text(post['paraphrase_success'])}|"
            f"{nll_text(final['rewrite_success'])}|{nll_text(final['paraphrase_success'])}|"
            f"{row['Wpost_minus_z_rewrite_new_nll']['mean']:+.6f}|"
            f"{row['Wpost_minus_z_rephrase_new_nll']['mean']:+.6f}|"
        )

    lines += [
        "",
        "## 전체 1,000 requests 성공·정확도 절대값",
        "",
        "|Arm|Panel|EFF/rewrite success|Rewrite Acc|GEN/rephrase success|GEN strict|Rephrase Acc|Acc strict|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        for panel, label in (
            ("accepted_z", "accepted z"),
            ("W_immediate_post", "W immediate-post"),
            ("W_final_W10", "W final-W10"),
        ):
            value = row[panel]
            lines.append(
                f"|{row['label']}|{label}|{count_text(value['rewrite_success'])}|"
                f"{count_text(value['rewrite_acc'])}|{count_text(value['paraphrase_success'])}|"
                f"{count_text(value['paraphrase_success'], True)}|"
                f"{count_text(value['paraphrase_acc'])}|"
                f"{count_text(value['paraphrase_acc'], True)}|"
            )

    lines += [
        "",
        "## accepted-z NLL 분포와 z→W gap",
        "",
        "|Arm|z rewrite mean/median/p90/max|z rephrase mean/median/p90/max|Wpost−z rewrite mean|Wpost−z rephrase mean|W10−z rewrite mean|W10−z rephrase mean|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        zr = row["accepted_z"]["rewrite_success"]["target_new_nll"]
        zg = row["accepted_z"]["paraphrase_success"]["target_new_nll"]
        lines.append(
            f"|{row['label']}|{zr['mean']:.6f}/{zr['median']:.6f}/{zr['p90']:.6f}/{zr['max']:.6f}|"
            f"{zg['mean']:.6f}/{zg['median']:.6f}/{zg['p90']:.6f}/{zg['max']:.6f}|"
            f"{row['Wpost_minus_z_rewrite_new_nll']['mean']:+.6f}|"
            f"{row['Wpost_minus_z_rephrase_new_nll']['mean']:+.6f}|"
            f"{row['Wfinal_minus_z_rewrite_new_nll']['mean']:+.6f}|"
            f"{row['Wfinal_minus_z_rephrase_new_nll']['mean']:+.6f}|"
        )

    lines += ["", "## R52 controller / route / history", ""]
    for arm in ("r52_h_on", "r52_h_off"):
        route = route_summary[arm]
        decisions = route["request_decisions"]
        lines.append(
            f"- {ARMS[arm]['label']}: PRIMARY `{decisions['PRIMARY']}`, RESCUE `{decisions['RESCUE']}`, "
            f"CURRENT `{decisions['CURRENT']}`, active-gradient `{decisions['ACTIVE_GRADIENT']}`, "
            f"clamp-hit `{decisions['CLAMP_HIT']}`, fallback `{route['fallback_count']}`, "
            f"negative physical progress `{route['negative_actual_count']}`, history widths `{route['history_widths']}`, "
            f"H status `{route['h_status_counts']}`."
        )

    lines += [
        "",
        "## Hard / realization-loss cohort",
        "",
        "|Arm|z rewrite fail|z rephrase strict fail|z rewrite success→Wpost fail|z rephrase strict success→Wpost fail|",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, value in hard_cohorts.items():
        lines.append(
            f"|{ARMS[arm]['label']}|{value['z_rewrite_success_fail']['count']}|"
            f"{value['z_rephrase_strict_fail']['count']}|"
            f"{value['z_rewrite_success_Wpost_failure']['count']}|"
            f"{value['z_rephrase_success_Wpost_failure']['count']}|"
        )

    lines += ["", "## Observation compute", "", "|Arm|F|B|generation|tokens|wall s|hook calls|action influence|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, value in observation_compute.items():
        lines.append(
            f"|{ARMS[arm]['label']}|{value['model_forward_count']}|"
            f"{value['added_backward_count']}|{value['added_generation_call_count']}|"
            f"{value['processed_token_count']}|{value['wall_seconds']:.2f}|"
            f"{value['hook_call_count']}|{value['action_influence_count']}|"
        )

    lines += [
        "",
        "## Machine-readable artifacts",
        "",
        f"- aggregate: `{aggregate_path}` (4 rows)",
        f"- per-batch: `{batch_path}` (40 rows)",
        f"- per-request: `{request_path}` (4,000 rows)",
        f"- hard cohorts: `{hard_path}`",
        f"- routing/compute: `{compute_path}`",
        "",
        "scientific_promotion=false",
    ]
    report_path = HERE / "p1r52-b100-accepted-z-rephrase-observation-factual-ko.md"
    write_once(report_path, ("\n".join(lines) + "\n").encode())

    outputs = [aggregate_path, batch_path, request_path, hard_path, compute_path, report_path]
    members = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "lines": len(path.read_text(encoding="utf-8").splitlines()),
            "sha256": sha(path),
        }
        for path in outputs
    ]
    manifest_value = {
        "schema": "p1r52-b100-accepted-z-observation/report-manifest/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "PASS",
        "members": members,
        "members_root_sha256": canonical_hash(members),
        "inputs": inputs,
        "inputs_root_sha256": canonical_hash(inputs),
    }
    manifest_path = dump_once("analysis-manifest.json", manifest_value)
    receipt_value = {
        "schema": "p1r52-b100-accepted-z-observation/rooted-receipt/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "PASS",
        "report_path": str(report_path),
        "report_sha256": sha(report_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha(manifest_path),
        "manifest_root": manifest_value["members_root_sha256"],
        "input_root": manifest_value["inputs_root_sha256"],
        "accepted_z_batches": 40,
        "per_request_rows": 4000,
        "missing_count": 0,
        "imputation_count": 0,
        "proxy_count": 0,
        "batch_entry_W_metric_count": 0,
        "added_backward_count": 0,
        "added_generation_call_count": 0,
        "action_influence_count": 0,
        "scientific_promotion": False,
    }
    receipt_value["root_digest"] = canonical_hash(receipt_value)
    receipt_path = dump_once("rooted-analysis-receipt.json", receipt_value)

    review_members = outputs + [manifest_path, receipt_path]
    checks = {
        "four_arms": len(aggregate_rows) == 4,
        "forty_batches": len(batch_rows) == 40,
        "four_thousand_requests": len(per_request) == 4000,
        "native_z_defined": all(
            row["binding"]["native_definition_preserved"] for row in obs_receipts
        ),
        "proxy_imputation_zero": all(
            row["binding"]["proxy_or_imputation_count"] == 0 for row in obs_receipts
        ),
        "observation_only": all(row["observation_only"] for row in obs_receipts),
        "action_influence_zero": all(
            row["action_influence_count"] == 0 for row in obs_receipts
        ),
        "additional_backward_generation_zero": all(
            row["added_backward_count"] == 0
            and row["added_generation_call_count"] == 0
            for row in obs_receipts
        ),
        "batch_entry_W_metrics_zero": all(
            row["batch_entry_W_metrics_count"] == 0 for row in obs_receipts
        ),
        "sealed_W_exact": all(row["sealed_commit_weight_exact"] for row in obs_receipts),
        "manifest_members_rehash": all(
            sha(Path(member["path"])) == member["sha256"]
            and Path(member["path"]).stat().st_size == member["bytes"]
            for member in members
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("independent raw-free review differs")
    review_value = {
        "schema": "p1r52-b100-accepted-z-observation/independent-rehash/v1",
        "status": "PASS",
        "checks": checks,
        "members": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
            for path in review_members
        ],
        "members_root_sha256": canonical_hash(
            [
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
                for path in review_members
            ]
        ),
    }
    review_value["root_digest"] = canonical_hash(review_value)
    review_path = dump_once("independent-rawfree-rehash-review.json", review_value)
    print(
        json.dumps(
            {
                "status": "PASS",
                "report": str(report_path),
                "report_sha256": sha(report_path),
                "manifest": str(manifest_path),
                "manifest_sha256": sha(manifest_path),
                "receipt": str(receipt_path),
                "receipt_sha256": sha(receipt_path),
                "review": str(review_path),
                "review_sha256": sha(review_path),
                "rows": {"aggregate": 4, "batch": 40, "request": 4000},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
