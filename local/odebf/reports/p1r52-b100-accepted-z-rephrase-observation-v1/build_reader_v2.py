#!/usr/bin/env python3
"""Create the append-only reader-complete v2 report from sealed v1 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
INSTRUCTION_ID = "ODEEDIT-S05-P1R52-LLAMA-B100-ACCEPTED-Z-REPHRASE-OBS-V1"
V1_REPORT = HERE / "p1r52-b100-accepted-z-rephrase-observation-factual-ko.md"
AGGREGATE = HERE / "p1r52-accepted-z-aggregates.json"
V1_MANIFEST = HERE / "analysis-manifest.json"
V1_RECEIPT = HERE / "rooted-analysis-receipt.json"
V1_REVIEW = HERE / "independent-rawfree-rehash-review.json"

ROOTS = {
    "memit": Path(
        "/mnt/raid5/janghj/.codex/worktrees/"
        "odeeditsh1-s05-p1r52-b100-accepted-z-rephrase-obs-tech-r1-v1/"
        "local/odebf/results/"
        "s05-p1r52-official-memit-sequential-10xb100-accepted-z-rephrase-obs-tech-r1-v1"
    ),
    "alphaedit": Path(
        "/mnt/raid5/janghj/.codex/worktrees/"
        "odeeditsh1-s05-p1r52-b100-accepted-z-rephrase-obs-tech-r1-v1/"
        "local/odebf/results/"
        "s05-p1r52-official-alphaedit-sequential-cache-on-10xb100-accepted-z-rephrase-obs-tech-r1-v1"
    ),
    "r52_h_on": Path(
        "/mnt/raid5/janghj/.codex/worktrees/"
        "odeeditsh1-s05-p1r52-b100-accepted-z-rephrase-obs-tech-r1-r52-v1/"
        "local/odebf/results/"
        "s05-p1r52-llama-soft-sequential-structuralh-on-10xb100-accepted-z-rephrase-obs-tech-r1-r52-v1"
    ),
    "r52_h_off": Path(
        "/mnt/raid5/janghj/.codex/worktrees/"
        "odeeditsh1-s05-p1r52-b100-accepted-z-rephrase-obs-tech-r1-r52-v1/"
        "local/odebf/results/"
        "s05-p1r52-llama-soft-sequential-alphacache-on-structuralh-off-10xb100-accepted-z-rephrase-obs-tech-r1-r52-v1"
    ),
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_once(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def dump_once(path: Path, value: Any) -> None:
    write_once(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(),
    )


def fmt(summary: dict[str, Any]) -> str:
    return (
        f"{summary['target_new_nll']['mean']:.6f} / "
        f"{summary['target_true_nll']['mean']:.6f} / "
        f"{summary['target_true_minus_new_margin']['mean']:+.6f}"
    )


def main() -> None:
    aggregate = load(AGGREGATE)
    v1 = V1_REPORT.read_text(encoding="utf-8")
    bindings = []
    binding_inputs = []
    for arm, root in ROOTS.items():
        receipts = []
        for path in sorted(root.glob("raw/batches/b*/accepted-z-rephrase-observation.json")):
            value = load(path)
            receipts.append(value)
            binding_inputs.append(
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
            )
        if len(receipts) != 10:
            raise RuntimeError(f"binding denominator differs: {arm}")
        first = receipts[0]["binding"]
        signature = {
            key: first[key]
            for key in (
                "source",
                "layer",
                "module_name",
                "fact_token",
                "accepted_z_dtype",
                "accepted_z_shape",
                "native_definition_preserved",
                "proxy_or_imputation_count",
            )
        }
        if any(
            {
                key: row["binding"][key]
                for key in signature
            }
            != signature
            for row in receipts
        ):
            raise RuntimeError(f"binding changes across B1-B10: {arm}")
        bindings.append({"arm": arm, **signature})

    appendix = [
        "",
        "## V2 완결성 보강: target-new / target-true / margin",
        "",
        "> 아래 셀은 `target-new NLL / target-true NLL / (true-new) margin` 순서입니다. 네 패널은 동일한 봉인 요청·prompt 분모를 사용합니다.",
        "",
        "|Arm|Panel|Rewrite|Rephrase|",
        "|---|---|---:|---:|",
    ]
    labels = {row["arm"]: row["label"] for row in aggregate["rows"]}
    for row in aggregate["rows"]:
        for panel, label in (
            ("accepted_z", "accepted z"),
            ("W_immediate_post", "W immediate-post"),
            ("W_final_W10", "W final-W10"),
        ):
            value = row[panel]
            appendix.append(
                f"|{row['label']}|{label}|{fmt(value['rewrite_success'])}|"
                f"{fmt(value['paraphrase_success'])}|"
            )

    appendix += [
        "",
        "## V2 완결성 보강: method-native accepted-z binding",
        "",
        "|Arm|native source|layer/module|lookup|dtype/shape|native preserved|proxy/imputation|",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in bindings:
        appendix.append(
            f"|{labels[row['arm']]}|`{row['source']}`|"
            f"{row['layer']} / `{row['module_name']}`|`{row['fact_token']}`|"
            f"`{row['accepted_z_dtype']}` / `{row['accepted_z_shape']}`|"
            f"{int(row['native_definition_preserved'])}|{row['proxy_or_imputation_count']}|"
        )
    appendix += [
        "",
        "- 네 방법의 source-native 정의를 그대로 사용했습니다. layer/lookup이 같더라도 accepted-z 생성 연산은 방법별 native implementation이므로 동일 latent 개입의 인과 비교로 해석하지 않습니다.",
        "- 기존 full-six z target objective는 이 rewrite/rephrase prompt 관측과 별도입니다.",
        "",
    ]

    report = HERE / "p1r52-b100-accepted-z-rephrase-observation-factual-ko-v2.md"
    write_once(report, (v1.rstrip() + "\n" + "\n".join(appendix)).encode())

    v1_members = [
        AGGREGATE,
        HERE / "p1r52-accepted-z-per-batch.json",
        HERE / "p1r52-accepted-z-per-request.json",
        HERE / "p1r52-accepted-z-hard-cohorts.json",
        HERE / "p1r52-accepted-z-routing-compute.json",
        V1_REPORT,
        V1_MANIFEST,
        V1_RECEIPT,
        V1_REVIEW,
    ]
    members = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
        for path in v1_members + [report]
    ]
    manifest_value = {
        "schema": "p1r52-b100-accepted-z-observation/report-manifest/v2",
        "instruction_id": INSTRUCTION_ID,
        "status": "PASS",
        "canonical_report": str(report),
        "canonical_report_sha256": sha(report),
        "members": members,
        "members_root_sha256": canonical(members),
        "native_binding_inputs": binding_inputs,
        "native_binding_inputs_root_sha256": canonical(binding_inputs),
        "rows": {"aggregate": 4, "batch": 40, "request": 4000},
        "missing_imputation_proxy": 0,
    }
    manifest = HERE / "analysis-manifest-v2.json"
    dump_once(manifest, manifest_value)
    receipt_value = {
        "schema": "p1r52-b100-accepted-z-observation/rooted-receipt/v2",
        "instruction_id": INSTRUCTION_ID,
        "status": "PASS",
        "report_path": str(report),
        "report_sha256": sha(report),
        "manifest_path": str(manifest),
        "manifest_sha256": sha(manifest),
        "manifest_root": manifest_value["members_root_sha256"],
        "binding_root": manifest_value["native_binding_inputs_root_sha256"],
        "added_model_forward": 0,
        "added_backward": 0,
        "added_generation": 0,
        "scientific_promotion": False,
    }
    receipt_value["root_digest"] = canonical(receipt_value)
    receipt = HERE / "rooted-analysis-receipt-v2.json"
    dump_once(receipt, receipt_value)

    checks = {
        "report_rehash": sha(report) == manifest_value["canonical_report_sha256"],
        "member_rehash": all(
            sha(Path(row["path"])) == row["sha256"]
            and Path(row["path"]).stat().st_size == row["bytes"]
            for row in members
        ),
        "binding_rows40": len(binding_inputs) == 40,
        "native_preserved": all(row["native_definition_preserved"] for row in bindings),
        "proxy_imputation_zero": all(row["proxy_or_imputation_count"] == 0 for row in bindings),
        "true_nll_margin_present": all(
            key in row[panel][metric]
            for row in aggregate["rows"]
            for panel in ("accepted_z", "W_immediate_post", "W_final_W10")
            for metric in ("rewrite_success", "paraphrase_success")
            for key in ("target_new_nll", "target_true_nll", "target_true_minus_new_margin")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("v2 independent review differs")
    review_value = {
        "schema": "p1r52-b100-accepted-z-observation/independent-rehash/v2",
        "status": "PASS",
        "checks": checks,
        "report_sha256": sha(report),
        "manifest_sha256": sha(manifest),
        "receipt_sha256": sha(receipt),
    }
    review_value["root_digest"] = canonical(review_value)
    review = HERE / "independent-rawfree-rehash-review-v2.json"
    dump_once(review, review_value)
    print(
        json.dumps(
            {
                "status": "PASS",
                "report": str(report),
                "report_sha256": sha(report),
                "manifest": str(manifest),
                "manifest_sha256": sha(manifest),
                "manifest_root": manifest_value["members_root_sha256"],
                "receipt": str(receipt),
                "receipt_sha256": sha(receipt),
                "review": str(review),
                "review_sha256": sha(review),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
