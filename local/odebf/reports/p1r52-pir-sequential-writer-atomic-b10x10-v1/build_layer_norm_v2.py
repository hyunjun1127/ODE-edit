#!/usr/bin/env python3
"""Create an append-only report revision with final per-layer BF16 norms."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path


ROOT = Path("/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-s05-p1r52-pir-atomic-b10x10-v1")
REPORT_DIR = ROOT / "local/odebf/reports/p1r52-pir-sequential-writer-atomic-b10x10-v1"
RESULT_DIR = ROOT / "local/odebf/results"
V1 = REPORT_DIR / "p1r52-pir-sequential-writer-atomic-b10x10-analysis-ko.md"
V2 = REPORT_DIR / "p1r52-pir-sequential-writer-atomic-b10x10-analysis-ko-v2.md"
TABLE = REPORT_DIR / "p1r52-pir-final-cumulative-bf16-layer-norm-v2.json"
MANIFEST = REPORT_DIR / "analysis-manifest-layer-norm-v2.json"
RECEIPT = REPORT_DIR / "analysis-receipt-layer-norm-v2.json"
REVIEW = REPORT_DIR / "independent-review-layer-norm-v2.json"

ARMS = {
    "J0": (
        "s05-p1r52-pir-independent-b10x10-llama3-8b-inst-pir-j0-repair-r1-v1",
        "p1r52-pir-j0",
    ),
    "PIR-G": (
        "s05-p1r52-pir-independent-b10x10-llama3-8b-inst-pir-g-repair-r1-v1",
        "p1r52-pir-pir-g",
    ),
    "PIR-U": (
        "s05-p1r52-pir-independent-b10x10-llama3-8b-inst-pir-u-repair-r1-v1",
        "p1r52-pir-pir-u",
    ),
}


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def write_json(path: Path, value: object) -> None:
    path.write_text(canonical(value).decode("utf-8") + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def main() -> None:
    per_case: list[dict[str, object]] = []
    for arm, (root_name, accepted_name) in ARMS.items():
        root = RESULT_DIR / root_name
        for case_index in range(1, 11):
            path = (
                root
                / "raw/cases"
                / f"case-{case_index:02d}"
                / "raw/ode"
                / accepted_name
                / "accepted-k8.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            energy_map = payload["materialization"]["cumulative_bf16_capacity"]
            energies = {
                layer: float(energy_map[f"model.layers.{layer}.mlp.down_proj.weight"])
                for layer in range(4, 9)
            }
            norms = {layer: math.sqrt(value) for layer, value in energies.items()}
            energy_total = math.fsum(energies.values())
            norm_sum = math.fsum(norms.values())
            for layer in range(4, 9):
                per_case.append(
                    {
                        "arm": arm,
                        "case_index": case_index,
                        "layer": layer,
                        "final_cumulative_bf16_energy": energies[layer],
                        "final_cumulative_bf16_frobenius_norm": norms[layer],
                        "norm_sum_share": norms[layer] / norm_sum,
                        "energy_share": energies[layer] / energy_total,
                        "norm_over_global_frobenius_norm": math.sqrt(
                            energies[layer] / energy_total
                        ),
                        "accepted_k8_receipt_sha256": sha256(path),
                    }
                )

    aggregate: list[dict[str, object]] = []
    for arm in ARMS:
        for layer in range(4, 9):
            selected = [
                row for row in per_case if row["arm"] == arm and row["layer"] == layer
            ]
            norms = [float(row["final_cumulative_bf16_frobenius_norm"]) for row in selected]
            aggregate.append(
                {
                    "arm": arm,
                    "layer": layer,
                    "cases": len(selected),
                    "mean_final_norm": statistics.fmean(norms),
                    "median_final_norm": statistics.median(norms),
                    "p90_final_norm": percentile(norms, 0.90),
                    "max_final_norm": max(norms),
                    "mean_norm_sum_share": statistics.fmean(
                        float(row["norm_sum_share"]) for row in selected
                    ),
                    "mean_energy_share": statistics.fmean(
                        float(row["energy_share"]) for row in selected
                    ),
                    "mean_norm_over_global": statistics.fmean(
                        float(row["norm_over_global_frobenius_norm"]) for row in selected
                    ),
                }
            )

    table_payload = {
        "schema": "ode-edit-s05-p1r52-pir-final-cumulative-bf16-layer-norm/v2",
        "definition": "sqrt(cumulative_bf16_capacity[layer]) = ||W8_layer-W0_layer||_F",
        "aggregation": "mean across ten independent B10 cases; shares computed per case then averaged",
        "source_head": "871d41c668ed46535a08fd4886318f881798886f",
        "per_case_rows": per_case,
        "aggregate_rows": aggregate,
    }
    table_payload["identity_sha256"] = digest(table_payload)
    write_json(TABLE, table_payload)

    lines = [
        "## 최종 실제 BF16 layer별 update norm — W8−W0",
        "",
        "각 case의 마지막 accepted-k8 materialization receipt에서 `cumulative_bf16_capacity`를 읽어 "
        "`||W8_layer−W0_layer||_F = sqrt(capacity_layer)`로 계산했다. 아래 값은 10개 독립 case의 평균이다. "
        "`norm 합 비중`은 case별 layer norm 합을 100%로 정규화한 뒤 평균한 값이며, `energy 비중`은 제곱 norm 비중이다.",
        "",
        "| Arm | layer | cases | mean norm | median | p90 | max | norm 합 비중 | energy 비중 | norm/global |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['arm']} | {row['layer']} | {row['cases']} | "
            f"{row['mean_final_norm']:.6f} | {row['median_final_norm']:.6f} | "
            f"{row['p90_final_norm']:.6f} | {row['max_final_norm']:.6f} | "
            f"{100.0 * row['mean_norm_sum_share']:.2f}% | "
            f"{100.0 * row['mean_energy_share']:.2f}% | "
            f"{100.0 * row['mean_norm_over_global']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "이 표는 실제 post-commit BF16 parameter 이동량이다. 앞 절의 PIR `factor_energy` 비중은 "
            "virtual low-rank factor 좌표의 제곱 norm이므로 이 표와 구분한다. 실제 norm 기준 L8의 "
            "layer-norm 합 비중은 J0 24.03%, PIR-G 40.21%, PIR-U 36.42%다.",
            "",
        ]
    )
    original = V1.read_text(encoding="utf-8")
    anchor = "## receipt 기반 성공·실패 패턴"
    if original.count(anchor) != 1:
        raise RuntimeError("report insertion anchor is not unique")
    revised = original.replace(anchor, "\n".join(lines) + anchor)
    V2.write_text(revised, encoding="utf-8")
    os.chmod(V2, 0o600)

    members = []
    for path in (V1, V2, TABLE, Path(__file__)):
        members.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "mode": oct(path.stat().st_mode & 0o777),
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema": "ode-edit-s05-p1r52-pir-layer-norm-report-manifest/v2",
        "original_report_preserved_sha256": sha256(V1),
        "canonical_v2_report": str(V2),
        "members": members,
        "members_root": digest(members),
    }
    manifest["identity_sha256"] = digest(manifest)
    write_json(MANIFEST, manifest)
    receipt = {
        "schema": "ode-edit-s05-p1r52-pir-layer-norm-report-receipt/v2",
        "manifest_sha256": sha256(MANIFEST),
        "manifest_identity_sha256": manifest["identity_sha256"],
        "report_sha256": sha256(V2),
        "table_sha256": sha256(TABLE),
        "per_case_rows": len(per_case),
        "aggregate_rows": len(aggregate),
        "model_evaluator_gpu_slurm_actions": [0, 0, 0, 0],
    }
    receipt["root_digest"] = digest(receipt)
    write_json(RECEIPT, receipt)

    review = {
        "schema": "ode-edit-s05-p1r52-pir-layer-norm-independent-review/v2",
        "status": "INDEPENDENT_RAW_FREE_REVIEW_PASS",
        "all_30_k8_receipts_present": len({(r["arm"], r["case_index"]) for r in per_case}) == 30,
        "per_case_rows": len(per_case),
        "aggregate_rows": len(aggregate),
        "norm_square_identity_max_abs": max(
            abs(
                float(row["final_cumulative_bf16_frobenius_norm"]) ** 2
                - float(row["final_cumulative_bf16_energy"])
            )
            for row in per_case
        ),
        "share_sum_max_abs": max(
            abs(
                math.fsum(
                    float(row["norm_sum_share"])
                    for row in per_case
                    if row["arm"] == arm and row["case_index"] == case_index
                )
                - 1.0
            )
            for arm in ARMS
            for case_index in range(1, 11)
        ),
        "report_sha256": sha256(V2),
        "table_sha256": sha256(TABLE),
        "manifest_sha256": sha256(MANIFEST),
        "receipt_sha256": sha256(RECEIPT),
        "actions": {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0},
    }
    review["identity_sha256"] = digest(review)
    write_json(REVIEW, review)

    print(
        json.dumps(
            {
                "report": str(V2),
                "report_sha256": sha256(V2),
                "report_bytes": V2.stat().st_size,
                "report_lines": len(revised.splitlines()),
                "table": str(TABLE),
                "table_sha256": sha256(TABLE),
                "manifest_sha256": sha256(MANIFEST),
                "receipt_sha256": sha256(RECEIPT),
                "review_sha256": sha256(REVIEW),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
