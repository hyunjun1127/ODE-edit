"""Analysis-only final-W10 NLL backfill for P1R54 sequential reports."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from .contracts import ODEBFContractError, canonical_hash


METHOD_ALPHA = "OFFICIAL-ALPHAEDIT-CACHE"
METHOD_MEMIT = "OFFICIAL-MEMIT"
METHOD_FZ = "P1R54-FZ-SEQUENTIAL"
METHODS = (METHOD_ALPHA, METHOD_MEMIT, METHOD_FZ)
FZ_FINAL_W10_SHA256 = "7f90dc19fbf02164c55406a8149ad77e72357597081127f4ed0b1b248e3d378b"
NATIVE_TRANSFER_SHA256 = {
    "alphaedit/immediate-post-final-w10-requests.json": "fe2310e31545cc5da019661f278420638d7292970e0a87a323afa9f7dff4e21d",
    "alphaedit/terminal.json": "c4376a89bf58db155b876b8d754172d7af9e28af0ac5982d3c13b29a231fd244",
    "alphaedit/manifest.json": "b9ad408c3e8fec00205788653815a7bcecb63305135c154583b0326b072f4d09",
    "memit/immediate-post-final-w10-requests.json": "eb2c3ef08fc41ed65490e9c23f7c50ce1d8927b53f7f586e13c262370c55febf",
    "memit/terminal.json": "4ec82d71123e6007f7931ab3895580057899fdbeea28dadf74bef3be0b67bc12",
    "memit/manifest.json": "06a0325b216c930527edb85d2685eb2a3e96672a9b8582e83b25fe461601c8d9",
    "provenance/artifact-inventory.json": "7af9b0d5f5be7c0672b25932da09845af98667889dbc385e14ab2f6ab2d14d87",
    "provenance/terminal-integrity.json": "d2ff6a4f050312910b026ff46357cfbe6a04f4525197a849f1a8f92ab2b5d6af",
    "provenance/analysis-manifest.json": "208cf89fe811ffe8aaafce5c0a1ecbc28f825a6a4dd52abbd8a62949a104f4e7",
    "provenance/rooted-analysis-receipt.json": "7e18ef5a86a96b543e17f9f4ffd04ae983a4232be9e0e1525cfc4ee91a8a3e03",
}
NATIVE_TRANSFER_BYTES = {
    "alphaedit/immediate-post-final-w10-requests.json": 3163574,
    "memit/immediate-post-final-w10-requests.json": 3149957,
}
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
ORDER_ROOT = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rooted(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError(f"analysis input differs: {path}")
    value = json.loads(path.read_bytes())
    identity = value.pop("identity_sha256", None)
    if identity != canonical_hash(value):
        raise ODEBFContractError(f"analysis rooted identity differs: {path}")
    value["identity_sha256"] = identity
    return value


def validate_native_transfer(root: Path) -> dict[str, Any]:
    """Full-rehash the create-once SH1 native final-W10 transfer."""
    for relative, expected_sha in NATIVE_TRANSFER_SHA256.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or (path.stat().st_mode & 0o777) != 0o600:
            raise ODEBFContractError(f"native transfer member type/mode differs: {path}")
        if sha256_file(path) != expected_sha:
            raise ODEBFContractError(f"native transfer member SHA differs: {path}")
        expected_bytes = NATIVE_TRANSFER_BYTES.get(relative)
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise ODEBFContractError(f"native transfer member bytes differ: {path}")

    integrity = _read_rooted(root / "provenance/terminal-integrity.json")
    inventory = json.loads((root / "provenance/artifact-inventory.json").read_bytes())
    if (
        integrity.get("stream_root_sha256") != STREAM_ROOT
        or integrity.get("stream_order_sha256") != ORDER_ROOT
        or integrity.get("dtype") != "FULL_FP32_PASS_ALL_5_ARMS"
        or integrity.get("terminal_status") != "TERMINAL_VALID_5_OF_5"
        or inventory.get("row_count") != 692
    ):
        raise ODEBFContractError("native transfer provenance differs")

    method_evidence: dict[str, Any] = {}
    request_order: list[tuple[int, int, str]] | None = None
    for dirname, expected_role in (
        ("alphaedit", "native-alphaedit-sequential-cache-on-corrected"),
        ("memit", "official-memit-sequential"),
    ):
        method_root = root / dirname
        table_path = method_root / "immediate-post-final-w10-requests.json"
        table = _read_rooted(table_path)
        terminal = _read_rooted(method_root / "terminal.json")
        manifest = _read_rooted(method_root / "manifest.json")
        rows = table.get("rows", ())
        order = [(int(row["round"]), int(row["request_index"]), str(row["request_sha256"])) for row in rows]
        if (
            table.get("schema") != "ode-edit-s05-p1r52-sequential-b100x10-pre-post-final-requests/v1"
            or table.get("role") != expected_role
            or table.get("row_count") != 1000
            or len(rows) != 1000
            or len(set(order)) != 1000
            or len({row[2] for row in order}) != 1000
            or any(sum(1 for value in order if value[0] == round_id) != 100 for round_id in range(1, 11))
            or terminal.get("role") != expected_role
            or terminal.get("request_count") != 1000
            or terminal.get("round_count") != 10
            or terminal.get("terminal_W0_restore_count") != 1
            or manifest.get("role") != expected_role
            or manifest.get("request_count") != 1000
            or manifest.get("round_count") != 10
            or manifest.get("W0_restored") is not True
            or manifest.get("terminal_sha256") != sha256_file(method_root / "terminal.json")
            or manifest.get("comparison_table_sha256", {}).get(table_path.name) != sha256_file(table_path)
        ):
            raise ODEBFContractError(f"native transfer schema/denominator differs: {dirname}")
        for row in rows:
            payload = dict(row)
            identity = payload.pop("identity_sha256", None)
            if identity != canonical_hash(payload):
                raise ODEBFContractError(f"native request identity differs: {dirname}")
            for stage in ("immediate_post", "final_W10"):
                for metric in ("rewrite_success", "paraphrase_success"):
                    values = row[stage][metric]
                    for key in (
                        "target_new_nll_mean",
                        "target_old_nll_mean",
                        "target_old_minus_new_margin_mean",
                    ):
                        if not math.isfinite(float(values[key])):
                            raise ODEBFContractError(f"native NLL is nonfinite: {dirname}")
        if request_order is not None and order != request_order:
            raise ODEBFContractError("native AlphaEdit/MEMIT request order differs")
        request_order = order
        method_evidence[dirname] = {
            "role": expected_role,
            "rows": len(rows),
            "rounds": 10,
            "unique_requests": len({row[2] for row in order}),
            "table_sha256": sha256_file(table_path),
            "terminal_sha256": sha256_file(method_root / "terminal.json"),
            "manifest_sha256": sha256_file(method_root / "manifest.json"),
        }
    result: dict[str, Any] = {
        "status": "FULL_REHASH_PASS",
        "root": str(root),
        "stream_root": STREAM_ROOT,
        "order_root": ORDER_ROOT,
        "methods": method_evidence,
        "compatibility_field_note": (
            "target_old_nll_mean is the transferred schema name for "
            "target_true_nll_by_request/request-mean"
        ),
        "member_sha256": dict(NATIVE_TRANSFER_SHA256),
    }
    result["identity_sha256"] = canonical_hash(result)
    return result


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values]
    if not clean or any(not math.isfinite(value) for value in clean):
        raise ODEBFContractError("final-W NLL vector is empty or nonfinite")
    ordered = sorted(clean)
    return {
        "denominator": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p90_nearest_rank": ordered[math.ceil(0.90 * len(ordered)) - 1],
        "max": ordered[-1],
    }


def _request_means(rows: Sequence[Sequence[float]]) -> list[float]:
    output = []
    for row in rows:
        values = [float(value) for value in row]
        if not values:
            raise ODEBFContractError("per-request prompt NLL denominator is empty")
        output.append(statistics.fmean(values))
    return output


def summarize_fz_final_w10(path: Path) -> dict[str, Any]:
    if sha256_file(path) != FZ_FINAL_W10_SHA256:
        raise ODEBFContractError("FZ sequential final-W10 identity differs")
    value = _read_rooted(path)
    if value.get("request_count") != 1000 or value.get("cohort_count") != 10:
        raise ODEBFContractError("FZ sequential final-W10 denominator differs")
    vectors: dict[tuple[str, str], list[float]] = {
        (prompt, target): []
        for prompt in ("rewrite", "rephrase")
        for target in ("target_new", "target_true")
    }
    for cohort in value["cohorts"]:
        scores = cohort["scores"]
        for prompt, metric in (
            ("rewrite", "rewrite_success"),
            ("rephrase", "paraphrase_success"),
        ):
            for target in ("target_new", "target_true"):
                vectors[(prompt, target)].extend(
                    _request_means(scores[metric][f"{target}_nll_by_request"])
                )
    return _method_summary(METHOD_FZ, vectors, path)


def summarize_native_final_w10(root: Path, *, method: str) -> dict[str, Any]:
    if method not in (METHOD_ALPHA, METHOD_MEMIT):
        raise ODEBFContractError("native final-W method differs")
    terminal = _read_rooted(root / "terminal.json")
    manifest = _read_rooted(root / "manifest.json")
    table_path = root / "immediate-post-final-w10-requests.json"
    table = _read_rooted(table_path)
    expected_role = (
        "native-alphaedit-sequential-cache-on-corrected"
        if method == METHOD_ALPHA
        else "official-memit-sequential"
    )
    if (
        table.get("row_count") != 1000
        or len(table.get("rows", ())) != 1000
        or table.get("role") != expected_role
        or terminal.get("role") != expected_role
        or terminal.get("request_count") != 1000
        or terminal.get("round_count") != 10
        or manifest.get("role") != expected_role
        or manifest.get("W0_restored") is not True
        or manifest.get("terminal_sha256") != sha256_file(root / "terminal.json")
        or manifest.get("comparison_table_sha256", {}).get(table_path.name)
        != sha256_file(table_path)
    ):
        raise ODEBFContractError("native sequential final-W terminal boundary differs")
    vectors: dict[tuple[str, str], list[float]] = {
        (prompt, target): []
        for prompt in ("rewrite", "rephrase")
        for target in ("target_new", "target_true")
    }
    for row in table["rows"]:
        final = row["final_W10"]
        for prompt, metric in (
            ("rewrite", "rewrite_success"),
            ("rephrase", "paraphrase_success"),
        ):
            vectors[(prompt, "target_new")].append(
                float(final[metric]["target_new_nll_mean"])
            )
            vectors[(prompt, "target_true")].append(
                float(final[metric]["target_old_nll_mean"])
            )
    result = _method_summary(method, vectors, table_path)
    result["terminal_sha256"] = sha256_file(root / "terminal.json")
    result["terminal_identity"] = terminal["identity_sha256"]
    result["manifest_sha256"] = sha256_file(root / "manifest.json")
    result["manifest_identity"] = manifest["identity_sha256"]
    result["source_head"] = terminal.get("source_head")
    result["compatibility_field_note"] = (
        "target_old_nll_mean maps to target_true_nll_by_request/request-mean"
    )
    return result


def _method_summary(
    method: str,
    vectors: Mapping[tuple[str, str], Sequence[float]],
    source_path: Path,
) -> dict[str, Any]:
    metrics = {
        prompt: {
            target: summarize(vectors[(prompt, target)])
            for target in ("target_new", "target_true")
        }
        for prompt in ("rewrite", "rephrase")
    }
    if any(
        metrics[prompt][target]["denominator"] != 1000
        for prompt in metrics
        for target in metrics[prompt]
    ):
        raise ODEBFContractError("final-W NLL request denominator differs")
    result: dict[str, Any] = {
        "method": method,
        "request_denominator": 1000,
        "metrics": metrics,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_bytes": source_path.stat().st_size,
        "imputation_count": 0,
        "model_or_evaluator_action_count": 0,
    }
    result["identity_sha256"] = canonical_hash(result)
    return result


def update_headline_csv(source: str, summaries: Mapping[str, Mapping[str, Any]]) -> str:
    rows = list(csv.DictReader(io.StringIO(source)))
    if {row["method"] for row in rows}.intersection(METHODS) != set(METHODS):
        raise ODEBFContractError("headline baseline method inventory differs")
    fields = list(rows[0])
    for row in rows:
        method = row["method"]
        if method not in summaries:
            continue
        metrics = summaries[method]["metrics"]
        for prompt in ("rewrite", "rephrase"):
            stats = metrics[prompt]["target_new"]
            for name in ("mean", "median", "p90_nearest_rank", "max"):
                row[f"W_{prompt}_{name}"] = repr(float(stats[name]))
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def update_report_markdown(
    source: str,
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    start_heading: str | None = None,
) -> str:
    lines = source.splitlines()
    start_index = 0
    if start_heading is not None:
        try:
            start_index = lines.index(start_heading)
        except ValueError as exc:
            raise ODEBFContractError("report sequential heading differs") from exc
    replaced: set[str] = set()
    for index, line in enumerate(lines):
        if index < start_index:
            continue
        for method in METHODS:
            if not line.startswith(f"| {method} |"):
                continue
            cells = [cell.strip() for cell in line.split("|")]
            if len(cells) != 11:
                raise ODEBFContractError("report headline row schema differs")
            metrics = summaries[method]["metrics"]
            cells[7] = _compact(metrics["rewrite"]["target_new"])
            cells[9] = _compact(metrics["rephrase"]["target_new"])
            lines[index] = "| " + " | ".join(cells[1:-1]) + " |"
            replaced.add(method)
    if replaced != set(METHODS):
        raise ODEBFContractError("report baseline row inventory differs")
    details = [
        "",
        "## Baseline final-W10 NLL backfill",
        "",
        "동일 canonical B1→B10 stream의 terminal W10 1,000-request raw에서 request-level prompt mean을 집계했다. Official AlphaEdit-cache/MEMIT는 server1의 기존 sealed raw를 create-once 전송해 재사용했고 신규 native 재실행은 0이다. FZ continuation은 기존 server4 sealed raw를 재분석했다. imputation=0.",
        "",
        "| Method | Prompt | target-new mean/median/p90/max | target-true mean/median/p90/max | requests |",
        "|---|---|---|---|---:|",
    ]
    for method in METHODS:
        for prompt in ("rewrite", "rephrase"):
            metrics = summaries[method]["metrics"][prompt]
            details.append(
                f"| {method} | {prompt.title()} | {_compact(metrics['target_new'])} | "
                f"{_compact(metrics['target_true'])} | {metrics['target_new']['denominator']} |"
            )
    details.extend([
        "",
        "호환 필드 `target_old_nll_mean`은 이 전송 schema에서 `target_true_nll_by_request`의 request mean에 대응한다.",
        "",
        "기존 report의 `NR`은 source package에 raw final-W10 NLL이 없었다는 뜻이며 성능 0을 뜻하지 않는다. 본 v2는 새 native raw와 기존 FZ raw를 SHA로 결속해 해당 칸을 채운 분석-only revision이다.",
    ])
    insert_at = next(
        (
            index
            for index, line in enumerate(lines)
            if index >= start_index
            and (
                line.startswith("## Rewrite/Rephrase")
                or line.startswith("## Sequential−Independent")
            )
        ),
        len(lines),
    )
    lines[insert_at:insert_at] = details + [""]
    lines = [
        line.replace(
            "Reset W는 terminal W10의 10 cohort 평가다. Native v5의 terminal W10 NLL 미기록 필드는 NR이다.",
            "Reset W와 세 baseline W NLL은 모두 terminal W10의 동일 10-cohort/1,000-request 평가다. Native 두 arm은 server1 기존 sealed final-W10 raw, FZ continuation은 server4 기존 sealed final-W10 raw를 사용했으며 신규 native 재실행은 0이다.",
        )
        for line in lines
    ]
    return "\n".join(lines) + "\n"


def _compact(stats: Mapping[str, Any]) -> str:
    return "/".join(f"{float(stats[name]):.6f}" for name in ("mean", "median", "p90_nearest_rank", "max"))


def detailed_rows(summaries: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        for prompt in ("rewrite", "rephrase"):
            for target in ("target_new", "target_true"):
                stats = summaries[method]["metrics"][prompt][target]
                rows.append({"method": method, "prompt": prompt, "target": target, **stats})
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ODEBFContractError("analysis CSV is empty")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def build_manifest(
    package: Path,
    *,
    source_head: str,
    source_tree: str,
    external_inputs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    members = []
    for path in sorted(package.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path.name in ("analysis-manifest.json", "rooted-receipt.json"):
            continue
        members.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    member_root = canonical_hash([[row["name"], row["sha256"], row["bytes"]] for row in members])
    manifest: dict[str, Any] = {
        "schema": "p1r54-reset-report-manifest/v2",
        "package": package.name,
        "source": source_head,
        "tree": source_tree,
        "members": members,
        "member_root": member_root,
        "external_inputs": list(external_inputs),
        "external_input_root": canonical_hash(list(external_inputs)),
        "scientific_promotion": False,
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    write_json(package / "analysis-manifest.json", manifest)
    receipt: dict[str, Any] = {
        "schema": "p1r54-reset-report-receipt/v2",
        "package": package.name,
        "manifest_sha256": sha256_file(package / "analysis-manifest.json"),
        "manifest_identity": manifest["identity_sha256"],
        "member_root": member_root,
        "analysis_sha256": sha256_file(package / "analysis.json"),
        "report_sha256": sha256_file(package / "report-ko.md"),
        "external_input_root": manifest["external_input_root"],
        "source_mutation": 0,
        "raw_mutation": 0,
        "imputation_count": 0,
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    write_json(package / "rooted-receipt.json", receipt)
    return manifest, receipt


__all__ = [
    "METHODS",
    "detailed_rows",
    "summarize",
    "summarize_fz_final_w10",
    "summarize_native_final_w10",
    "validate_native_transfer",
    "update_headline_csv",
    "update_report_markdown",
]
