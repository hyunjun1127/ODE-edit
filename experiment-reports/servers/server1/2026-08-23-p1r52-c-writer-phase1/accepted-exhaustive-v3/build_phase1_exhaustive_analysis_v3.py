#!/usr/bin/env python3
"""Publish the final Phase-1 report with explicit accuracy and W-minus-z tables."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-exhaustive-analysis-v2"
V1 = HERE.parent / "p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-exhaustive-analysis-v1"
ARMS = ("alphaedit", "memit", "c0", "c1", "c3")
REPORT = "p1r52-c-writer-phase1-full-fp32-sequential-tech-r1-exhaustive-factual-ko.md"


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write_once(path: Path, payload: bytes) -> Path:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"create-once target exists: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)
    return path


def dump_json(name: str, value: Any) -> Path:
    return write_once(HERE / name, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())


def dump_csv(name: str, rows: list[dict[str, Any]]) -> Path:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return write_once(HERE / name, buf.getvalue().encode())


def main() -> None:
    v2_receipt = load(V2 / "rooted-analysis-receipt.json")
    if v2_receipt["status"] != "TERMINAL_VALID_EXHAUSTIVE_ANALYSIS_COMPLETE":
        raise RuntimeError("v2 input is not terminal-valid")
    arms = load(V1 / "arm-aggregate-summary.json")["arms"]
    nll_rows = load(V1 / "nll-distribution-aggregates.json")["rows"]
    accepted = load(V2 / "accepted-z-native-z-summary.json")["rows"]
    immediate = load(V2 / "immediate-w-summary.json")["rows"]
    by_z = {(row["arm"], row["prompt"]): row for row in accepted}
    by_w = {(row["arm"], row["prompt"]): row for row in immediate}
    nll = {(row["arm"], row["endpoint"], row["prompt"], row["target"]): row for row in nll_rows}

    endpoint_rows = []
    for arm in ARMS:
        for endpoint, table, metric_scope in (("accepted_z_or_native_z", by_z, "z"), ("immediate_W", by_w, "W_immediate")):
            for prompt in ("rewrite", "rephrase"):
                row = table[(arm, prompt)]
                success = arms[arm][metric_scope][f"{prompt}_success"]
                accuracy = arms[arm][metric_scope][f"{prompt}_accuracy"]
                endpoint_rows.append({
                    "arm": arm, "endpoint": endpoint, "prompt": prompt,
                    "z_provenance": arms[arm]["z_provenance"] if endpoint.startswith("accepted") else "NOT_APPLICABLE",
                    "success_numerator": success["numerator"], "success_denominator": success["denominator"],
                    "success_strict_numerator": success["strict_numerator"], "success_strict_denominator": success["strict_denominator"],
                    "accuracy_numerator": accuracy["numerator"], "accuracy_denominator": accuracy["denominator"],
                    "accuracy_strict_numerator": accuracy["strict_numerator"], "accuracy_strict_denominator": accuracy["strict_denominator"],
                    **{key: row[key] for key in row if key.startswith("target_")},
                })

    gap_rows = []
    for arm in ARMS:
        for prompt in ("rewrite", "rephrase"):
            row = nll[(arm, "W_MINUS_Z", prompt, "new")]
            gap_rows.append({"arm": arm, "prompt": prompt, "target": "new", **{key: row[key] for key in ("count", "mean", "median", "p90", "max", "min")}})

    outputs = [
        dump_json("endpoint-success-accuracy-nll.json", {"schema": "phase1/endpoint-success-accuracy-nll/v3", "row_count": len(endpoint_rows), "rows": endpoint_rows}),
        dump_csv("endpoint-success-accuracy-nll.csv", endpoint_rows),
        dump_json("w-minus-z-gap.json", {"schema": "phase1/w-minus-z-gap/v3", "row_count": len(gap_rows), "rows": gap_rows}),
        dump_csv("w-minus-z-gap.csv", gap_rows),
    ]

    lines = [
        "# P1R52 C-writer Phase 1 FULL-FP32 Alpha-cache sequential 5-arm — 최종 상세 사실 보고서", "",
        "> **최우선 실행 경계:** C0/C1/C3는 이전 J0와 달리 각 B100에서 P1R52 accepted-z를 K1→K8까지 전부 최적화한 뒤 writer를 한 번만 적용했다. K-step 표는 write 이전 target/z trajectory이지 중간 writer endpoint가 아니다.", "",
        "> 이 문서는 v2 상세 분석을 그대로 보존하면서 요청된 accuracy 및 W−z NLL gap을 명시적으로 보강한 canonical v3다. 모든 arm은 FULL FP32이며 BF16/FP16/autocast/quantization/numeric cast=0, imputation=0, scientific promotion=false다.", "",
        "## A. accepted-z / native-z: success·accuracy·NLL", "",
        "AlphaEdit native-z, MEMIT latent-z, P1R52 accepted-z는 provenance가 다르다. 특히 MEMIT 값을 P1R52 accepted-z로 해석하지 않는다.", "",
        "|arm|prompt|success|accuracy|success strict|accuracy strict|target-new mean/median/p90|max|target-true mean/median/p90|max|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in endpoint_rows:
        if row["endpoint"] != "accepted_z_or_native_z":
            continue
        lines.append(f"|{row['arm']}|{row['prompt']}|{row['success_numerator']}/{row['success_denominator']}|{row['accuracy_numerator']}/{row['accuracy_denominator']}|{row['success_strict_numerator']}/{row['success_strict_denominator']}|{row['accuracy_strict_numerator']}/{row['accuracy_strict_denominator']}|{row['target_new_mean']:.6f}/{row['target_new_median']:.6f}/{row['target_new_p90']:.6f}|{row['target_new_max']:.6f}|{row['target_true_mean']:.6f}/{row['target_true_median']:.6f}/{row['target_true_p90']:.6f}|{row['target_true_max']:.6f}|")
    lines += ["", "## B. immediate-post W: success·accuracy·NLL", "", "|arm|prompt|success|accuracy|success strict|accuracy strict|target-new mean/median/p90|max|target-true mean/median/p90|max|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in endpoint_rows:
        if row["endpoint"] != "immediate_W":
            continue
        lines.append(f"|{row['arm']}|{row['prompt']}|{row['success_numerator']}/{row['success_denominator']}|{row['accuracy_numerator']}/{row['accuracy_denominator']}|{row['success_strict_numerator']}/{row['success_strict_denominator']}|{row['accuracy_strict_numerator']}/{row['accuracy_strict_denominator']}|{row['target_new_mean']:.6f}/{row['target_new_median']:.6f}/{row['target_new_p90']:.6f}|{row['target_new_max']:.6f}|{row['target_true_mean']:.6f}/{row['target_true_median']:.6f}/{row['target_true_p90']:.6f}|{row['target_true_max']:.6f}|")
    lines += ["", "## C. W−z target-new NLL gap", "", "|arm|prompt|mean/median/p90|max|min|", "|---|---|---:|---:|---:|"]
    for row in gap_rows:
        lines.append(f"|{row['arm']}|{row['prompt']}|{row['mean']:.6f}/{row['median']:.6f}/{row['p90']:.6f}|{row['max']:.6f}|{row['min']:.6f}|")
    detailed = (V2 / REPORT).read_text(encoding="utf-8").splitlines()
    lines += ["", "---", "", "## D. 전체 상세 본문", ""] + detailed[1:]

    report_path = write_once(HERE / REPORT, ("\n".join(lines) + "\n").encode())
    outputs.append(report_path)
    members = [{"path": str(Path(__file__).resolve()), "bytes": Path(__file__).stat().st_size, "sha256": sha_file(Path(__file__).resolve())}]
    members += [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path)} for path in outputs]
    manifest = {
        "schema": "phase1/exhaustive-analysis-manifest/v3", "input_v2_receipt_identity_sha256": v2_receipt["identity_sha256"],
        "input_artifact_inventory_root_sha256": v2_receipt["input_artifact_inventory_root_sha256"],
        "members": members, "member_root_sha256": canonical_sha(members), "scientific_promotion": False,
    }
    manifest_path = dump_json("analysis-manifest.json", manifest)
    receipt = {
        "schema": "phase1/rooted-analysis-receipt/v3", "status": "TERMINAL_VALID_EXHAUSTIVE_ANALYSIS_COMPLETE",
        "report": {"path": str(report_path), "sha256": sha_file(report_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha_file(manifest_path)},
        "analysis_member_root_sha256": manifest["member_root_sha256"],
        "input_artifact_inventory_root_sha256": manifest["input_artifact_inventory_root_sha256"],
        "row_counts": {"endpoint_success_accuracy_nll": len(endpoint_rows), "w_minus_z_gap": len(gap_rows), "v2_prompt_endpoint_rows": 30000},
        "technical_failures": 0, "scientific_failures": 0, "retries": 0, "imputations": 0, "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_sha(receipt)
    dump_json("rooted-analysis-receipt.json", receipt)


if __name__ == "__main__":
    main()
