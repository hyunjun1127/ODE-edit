#!/usr/bin/env python3
"""Build the terminal Llama-only P4-Euler calibration report package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import torch

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.functional import tensor_sha256


NOT_RECORDED = "NOT_RECORDED_CALIBRATION_RUNTIME_TELEMETRY_OMISSION"
INSTRUCTION = "ODEEDIT-S05-P4-EULER-PROJECTED-SEMANTIC-ODE-V1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path) -> None:
    observed = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise RuntimeError(f"non-regular input: {path}")


def _json(path: Path, *, rooted: bool = True) -> dict[str, Any]:
    _regular(path)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"non-object JSON: {path}")
    if rooted:
        body = dict(value)
        identity = body.pop("identity_sha256", None)
        if identity != canonical_hash(body):
            raise RuntimeError(f"identity mismatch: {path}")
    return value


def _tensor(path: Path, expected_sha: str) -> torch.Tensor:
    _regular(path)
    if (path.stat().st_mode & 0o777) != 0o600:
        raise RuntimeError(f"snapshot mode mismatch: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype is not torch.float32
        or value.ndim != 2
        or value.shape[1] != 10
        or not bool(torch.isfinite(value).all())
        or tensor_sha256(value) != expected_sha
    ):
        raise RuntimeError(f"snapshot mismatch: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty table: {path.name}")
    with path.open("x", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o600)


def _member(path: Path, root: Path, *, role: str) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "relative_path": str(path.relative_to(root)),
        "role": role,
        "bytes": len(data),
        "lines": data.count(b"\n"),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _archive_only_receipt(qwen_root: Path) -> dict[str, Any]:
    if qwen_root.is_symlink() or not qwen_root.is_dir():
        raise RuntimeError("Qwen archival root differs")
    members: list[dict[str, Any]] = []
    for path in sorted(qwen_root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Qwen archival symlink differs")
        if not path.is_file():
            continue
        members.append(
            {
                "relative_path": str(path.relative_to(qwen_root)),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
    terminal = qwen_root / "terminal.json"
    _regular(terminal)
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-qwen-stage2-archival-exclusion/v1",
        "instruction_id": INSTRUCTION,
        "status": "USER_DIRECTED_SCIENTIFIC_SCOPE_EXCLUSION",
        "labels": [
            "CALIBRATION_DIAGNOSTIC_ONLY",
            "SCIENTIFIC_INTERPRETATION_EXCLUDED",
        ],
        "root": str(qwen_root.resolve()),
        "member_count": len(members),
        "total_bytes": sum(row["bytes"] for row in members),
        "member_root": canonical_hash({"members": members}),
        "terminal_file_sha256": _sha(terminal),
        "payload_parse_count": 0,
        "metric_aggregate_count": 0,
        "scientific_input_count": 0,
        "numerical_lock_influence_count": 0,
        "za_zb_submission_count": 0,
        "raw_root_preserved": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def build(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError("report output is create-once")
    os.umask(0o077)
    args.output_root.mkdir(parents=True, mode=0o700)

    stage1 = args.stage1_root.resolve()
    stage2 = args.stage2_root.resolve()
    terminal_path = stage2 / "terminal.json"
    terminal = _json(terminal_path)
    if (
        terminal.get("alias") != "llama3-8b-inst"
        or terminal.get("status") != "STAGE2_MODEL_CELL_SCIENTIFIC_HOLD"
        or terminal.get("selected_h") != 0.25
        or terminal.get("selected_target_horizon") != 1.25
    ):
        raise RuntimeError("Llama Stage2 terminal differs")

    inputs: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    endpoint_summary_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    microstep_rows: list[dict[str, Any]] = []
    stage2_values: dict[tuple[str, int], tuple[dict[str, Any], torch.Tensor]] = {}
    for arm, directory in (("Z+", "Zplus"), ("Z±", "Zpm")):
        comparison_path = stage2 / "comparisons" / f"{directory}.json"
        comparison = _json(comparison_path)
        inputs.append({"path": str(comparison_path), "sha256": _sha(comparison_path)})
        arm_values: dict[int, tuple[dict[str, Any], torch.Tensor]] = {}
        for microsteps in (5, 10):
            path = stage2 / "trajectories" / directory / f"M{microsteps}.json"
            row = _json(path)
            snapshot = stage2 / "snapshots" / row["target_snapshot_relative_path"]
            state = _tensor(snapshot, row["target_sha256"])
            arm_values[microsteps] = (row, state)
            stage2_values[(arm, microsteps)] = (row, state)
            inputs.extend(
                [
                    {"path": str(path), "sha256": _sha(path)},
                    {"path": str(snapshot), "sha256": _sha(snapshot)},
                ]
            )
            compute_rows.append(
                {
                    "stage": "Stage2",
                    "arm": arm,
                    "M": microsteps,
                    "h": row["h"],
                    "T_z": row["target_horizon"],
                    "requests": 10,
                    "executed_microsteps": row["executed_microsteps"],
                    "autograd_grad_calls": row["actual_autograd_grad_call_count"],
                    "duplicate_evaluations": row["duplicate_autograd_evaluation_count"],
                    "clamp_hits": row["clamp_hit_count"],
                    "clamp_denominator_request_microsteps": row["clamp_denominator"],
                    "clamp_fraction": row["clamp_fraction"],
                    "inner_wall_seconds": row["wall_time_seconds"],
                    "W_pointer_version_changes": row["W0_pointer_version_change_count"],
                    "W_bytes_changes": row["W0_bytes_change_count"],
                    "optimizer": row["optimizer"],
                    "adam_states": row["adam_state_count"],
                    "backward_calls": row["loss_backward_count"],
                    "parameter_gradients": row["parameter_gradient_count"],
                }
            )
            for step in range(1, microsteps + 1):
                microstep_rows.append(
                    {
                        "arm": arm,
                        "M": microsteps,
                        "step": step,
                        "h": row["h"],
                        "T_z": row["target_horizon"],
                        "autograd_grad_call": 1,
                        "target_objective_total": NOT_RECORDED,
                        "semantic_component": NOT_RECORDED,
                        "kl_component": NOT_RECORDED,
                        "decay_component": NOT_RECORDED,
                        "raw_gradient_norm": NOT_RECORDED,
                        "projected_gradient_norm": NOT_RECORDED,
                        "euler_movement_norm": NOT_RECORDED,
                        "clamp_hit_by_request": NOT_RECORDED,
                        "step_receipt_root": row["integrator_receipt"][
                            "step_identity_sha256"
                        ],
                        "recording_note": (
                            "The runtime verified the step in-memory but persisted "
                            "only the rooted aggregate step identity."
                        ),
                    }
                )
        m5, z5 = arm_values[5]
        m10, z10 = arm_values[10]
        absolute = torch.linalg.vector_norm(z5 - z10, dim=0)
        obs5 = m5["final_value_only_observation"]
        obs10 = m10["final_value_only_observation"]
        for ordinal in range(10):
            endpoint_rows.append(
                {
                    "arm": arm,
                    "request_ordinal": ordinal,
                    "m5_new_nll": obs5["new_nll_by_request"][ordinal],
                    "m5_true_nll": obs5["old_nll_by_request"][ordinal],
                    "m5_new_minus_true_margin": obs5[
                        "new_minus_old_margin_by_request"
                    ][ordinal],
                    "m5_train_new_beats_true": (
                        obs5["new_minus_old_margin_by_request"][ordinal] > 0
                    ),
                    "m10_new_nll": obs10["new_nll_by_request"][ordinal],
                    "m10_true_nll": obs10["old_nll_by_request"][ordinal],
                    "m10_new_minus_true_margin": obs10[
                        "new_minus_old_margin_by_request"
                    ][ordinal],
                    "m10_train_new_beats_true": (
                        obs10["new_minus_old_margin_by_request"][ordinal] > 0
                    ),
                    "m5_state_norm": float(torch.linalg.vector_norm(z5[:, ordinal])),
                    "m10_state_norm": float(torch.linalg.vector_norm(z10[:, ordinal])),
                    "m5_m10_absolute_endpoint_distance": float(absolute[ordinal]),
                    "m5_origin_displacement": NOT_RECORDED,
                    "m10_origin_displacement": NOT_RECORDED,
                    "relative_d_i": NOT_RECORDED,
                    "relative_d_i_distribution_authority": (
                        "comparison receipt summary only; request vector was not persisted"
                    ),
                }
            )
        summary = comparison["endpoint_discrepancy"]
        endpoint_summary_rows.append(
            {
                "arm": arm,
                "requests": 10,
                "d_i_mean": summary["mean"],
                "d_i_median": summary["median"],
                "d_i_p90": summary["p90"],
                "d_i_max": summary["max"],
                "threshold_median": 0.10,
                "threshold_p90": 0.25,
                "threshold_max": 0.50,
                "threshold_pass": comparison["threshold_pass"],
                "M5_clamp_hits": m5["clamp_hit_count"],
                "M5_clamp_denominator": m5["clamp_denominator"],
                "M5_clamp_fraction": m5["clamp_fraction"],
                "M10_clamp_hits": m10["clamp_hit_count"],
                "M10_clamp_denominator": m10["clamp_denominator"],
                "M10_clamp_fraction": m10["clamp_fraction"],
            }
        )

    stage1_curve_rows: list[dict[str, Any]] = []
    prefix_distance_rows: list[dict[str, Any]] = []
    for arm, directory in (("Z+", "Zplus"), ("Z±", "Zpm")):
        path = stage1 / "trajectories" / directory / "h-0.25.json"
        trajectory = _json(path)
        inputs.append({"path": str(path), "sha256": _sha(path)})
        snapshots: dict[int, torch.Tensor] = {}
        for row in trajectory["prefix_rows"]:
            prefix = int(row["M"])
            snapshot = stage1 / "snapshots" / row["target_snapshot_relative_path"]
            snapshots[prefix] = _tensor(snapshot, row["target_sha256"])
            inputs.append({"path": str(snapshot), "sha256": _sha(snapshot)})
            stage1_curve_rows.append(
                {
                    "arm": arm,
                    "h": 0.25,
                    "M": prefix,
                    "T_z": row["target_horizon"],
                    "requests": 10,
                    "new_nll_mean": row["new_nll"]["mean"],
                    "new_nll_median": row["new_nll"]["median"],
                    "new_nll_p90": row["new_nll"]["p90"],
                    "new_nll_max": row["new_nll"]["max"],
                    "true_nll_mean": row["old_nll"]["mean"],
                    "true_nll_median": row["old_nll"]["median"],
                    "margin_mean": row["new_minus_old_margin"]["mean"],
                    "margin_median": row["new_minus_old_margin"]["median"],
                    "displacement_mean": row["displacement"]["mean"],
                    "displacement_median": row["displacement"]["median"],
                    "displacement_p90": row["displacement"]["p90"],
                    "displacement_max": row["displacement"]["max"],
                    "raw_field_norm_mean": row["raw_field_norm"]["mean"],
                    "raw_field_norm_median": row["raw_field_norm"]["median"],
                    "raw_field_norm_p90": row["raw_field_norm"]["p90"],
                    "raw_field_norm_max": row["raw_field_norm"]["max"],
                    "clamp_hits": row["clamp_hit_count"],
                    "clamp_denominator_request_microsteps": row[
                        "clamp_denominator"
                    ],
                    "clamp_fraction": row["clamp_fraction"],
                }
            )
        for left, right in ((1, 3), (3, 5), (5, 10)):
            distance = torch.linalg.vector_norm(
                snapshots[right] - snapshots[left], dim=0
            )
            for ordinal, value in enumerate(distance):
                prefix_distance_rows.append(
                    {
                        "arm": arm,
                        "request_ordinal": ordinal,
                        "from_M": left,
                        "to_M": right,
                        "absolute_endpoint_distance": float(value),
                    }
                )

    qwen_receipt = _archive_only_receipt(args.qwen_root.resolve())
    _write_json(args.output_root / "qwen-archival-exclusion.json", qwen_receipt)
    _write_csv(args.output_root / "llama-endpoint-requests.csv", endpoint_rows)
    _write_csv(args.output_root / "llama-endpoint-summary.csv", endpoint_summary_rows)
    _write_csv(args.output_root / "llama-stage1-h025-curve.csv", stage1_curve_rows)
    _write_csv(
        args.output_root / "llama-stage1-prefix-distances.csv", prefix_distance_rows
    )
    _write_csv(args.output_root / "llama-microstep-ledger.csv", microstep_rows)
    _write_csv(args.output_root / "llama-compute.csv", compute_rows)

    analysis: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-stage2-llama-only-analysis/v1",
        "instruction_id": INSTRUCTION,
        "scope": "LLAMA_ONLY_STAGE2_PLUS",
        "status": "SCIENTIFIC_HOLD",
        "hold_reason": (
            "Euler discretization refinement gate failure at locked T_z=1.25"
        ),
        "model": "llama3-8b-inst",
        "calibration_unit": "B1_CASE01_PERMANENT_CALIBRATION_ONLY",
        "selected_h": 0.25,
        "selected_target_horizon": 1.25,
        "stage2_endpoint_discrepancy": endpoint_summary_rows,
        "stage2_terminal_identity": terminal["identity_sha256"],
        "stage2_terminal_file_sha256": _sha(terminal_path),
        "invariants": {
            "finite": True,
            "W0_restored": terminal["W0_restored"],
            "full_fp32": terminal["full_fp32"],
            "optimizer": terminal["optimizer"],
            "adam_state_count": terminal["adam_state_count"],
            "loss_backward_count": terminal["loss_backward_count"],
            "parameter_gradient_count": terminal["parameter_gradient_count"],
            "actual_autograd_grad_call_count": terminal[
                "actual_autograd_grad_call_count"
            ],
            "duplicate_autograd_evaluation_count": terminal[
                "duplicate_autograd_evaluation_count"
            ],
            "writer_materialization_count": terminal[
                "writer_materialization_count"
            ],
            "cache_append_count": terminal["cache_append_count"],
            "heldout_access_count": terminal["heldout_access_count"],
            "native_access_count": terminal["native_access_count"],
        },
        "recording_coverage": {
            "endpoint_request_nll_margin": "RECORDED",
            "endpoint_tensor_pair_distance": "RECORDED",
            "endpoint_d_i_distribution": "RECORDED",
            "endpoint_d_i_by_request": NOT_RECORDED,
            "origin_displacement_by_request": NOT_RECORDED,
            "microstep_objective_components": NOT_RECORDED,
            "microstep_gradient_and_movement_norms": NOT_RECORDED,
            "reason": (
                "The executed runner retained detailed step receipts in memory but "
                "persisted only their canonical aggregate identity. No replay was "
                "performed after the scientific HOLD."
            ),
        },
        "performance_fields": {
            "heldout": "NOT_RECORDED_CALIBRATION_TRAIN_ONLY",
            "rewrite": "NOT_RECORDED_CALIBRATION_TRAIN_ONLY",
            "rephrase": "NOT_RECORDED_CALIBRATION_TRAIN_ONLY",
            "locality": "NOT_RECORDED_CALIBRATION_TRAIN_ONLY",
        },
        "qwen_scientific_input_count": 0,
        "qwen_metric_parse_count_during_report_finalization": 0,
        "scientific_promotion": False,
        "za_zb_submission_count": 0,
    }
    analysis["root_digest"] = canonical_hash(analysis)
    _write_json(args.output_root / "llama-analysis.json", analysis)

    report = f"""# P4-Euler Stage2 Llama 전용 최종 보고서

> Scope: **LLAMA_ONLY_STAGE2_PLUS**. 공통 hyperparameter transfer가 Qwen 편집 성공을 만들지 못해 Qwen은 진단 전용으로 제외하고, Stage2+ 과학 분석은 Llama에만 한정한다.

## 최종 판정

- 상태: **SCIENTIFIC_HOLD**
- 사유: **Euler discretization refinement gate failure at locked T_z=1.25**
- 잠금: Llama `h=0.25`, `T_z=1.25`; Stage2 `M5,h=.25` 대 `M10,h=.125`.
- `Z+` d_i mean/median/p90/max = 0.599766/0.599819/0.819999/0.837496.
- `Z±` d_i mean/median/p90/max = 0.722633/0.737896/0.865267/1.022678.
- 사전 임계 median≤.10, p90≤.25, max≤.50를 두 arm 모두 초과했다. 이는 고정 T_z에서의 수치 refinement 실패이며 편집 성능 결론이 아니다.
- ZA/ZB 제출 0, 재시도·regrid·T_z·tolerance 변경 0, scientific promotion=false.

## Stage1 h=.25 strength curve와 Stage2 연결

Stage1은 동일 h=.25의 한 10-step trajectory에서 M1/M3/M5/M10 prefix를 재사용했다. `llama-stage1-h025-curve.csv`가 arm별 train new/true NLL, margin, displacement, raw-field norm과 request×microstep clamp 분모를 제공한다. Stage2 M5는 같은 h/T_z endpoint이고, M10은 같은 T_z를 h=.125로 세분화했다. M5와 M10의 endpoint가 크게 달라져 refinement gate를 통과하지 못했다.

Stage1의 M1/M3/M5/M10 간 request별 절대 이동은 `llama-stage1-prefix-distances.csv`에 있다. Stage2 실행은 중간 state를 저장하지 않았으므로 M5와 M10 trajectory가 **어느 최초 microstep에서** 갈라졌는지는 기록으로 판정할 수 없다. 원인은 추정하지 않는다.

## Endpoint 및 request-level 산출

- `llama-endpoint-summary.csv`: d_i 분포와 임계, clamp numerator/denominator/fraction.
- `llama-endpoint-requests.csv`: ordinal별 M5/M10 train new/true NLL, margin, state norm, M5↔M10 절대 endpoint 거리.
- request별 d_i와 ||z_M-y|| 벡터는 runner가 저장하지 않아 `{NOT_RECORDED}`이다. authoritative d_i distribution만 Stage2 comparison receipt에 남아 있다.
- `train_new_beats_true`는 train target log-odds의 관측 필드이며 heldout Eff/Gen 성공 판정이 아니다.

## Microstep, objective, gradient, movement

`llama-microstep-ledger.csv`는 30개 실행 microstep(arm별 M5+M10)의 순서, h/T_z, autograd 1회/step 및 rooted step identity를 기록한다. 다만 runtime이 in-memory step receipt의 objective semantic/KL/decay, raw/projected gradient norm, movement, request clamp vector를 파일에 영속화하지 않았다. 해당 열은 `{NOT_RECORDED}`이며 HOLD 이후 GPU replay를 하지 않았다.

## 불변식과 compute

- Stage2: 4 trajectories, 30 microsteps, autograd.grad 30, duplicate evaluation 0.
- W pointer/version/bytes 변화 0 및 terminal W0 restore PASS.
- FULL FP32, autocast/quantization 0; optimizer/Adam/backward/parameter-gradient 0.
- Z+ clamp M5 0/50, M10 0/100. Z± M5 6/50=.12, M10 0/100. 모두 <.50.
- writer/cache append/heldout/Native access 모두 0.
- `llama-compute.csv`에 trajectory별 inner wall time과 ledger가 있다. 성공 Slurm allocation은 Stage1 354초 + Stage2 152초 = 506 GPU초이며, inner trajectory 합은 299.982초 + 113.179초 = 413.161초다. 성공 실행의 setup/report overhead는 92.839초다. 선행 기술 실패 3회의 Llama GPU allocation은 47초이며 science delta는 0이다.

## Claim 경계

B1_CASE01은 영구 calibration-only다. heldout/rewrite/rephrase/locality/evaluator는 `NOT_RECORDED_CALIBRATION_TRAIN_ONLY`이며 confirmatory denominator나 성능 claim에 포함되지 않는다. Qwen은 별도 `qwen-archival-exclusion.json`에 root 무결성만 보존되며 이 분석의 scientific input 또는 numerical-lock input이 아니다.
"""
    report_path = args.output_root / "report-ko.md"
    report_path.write_text(report, encoding="utf-8")
    os.chmod(report_path, 0o600)

    output_members = [
        _member(path, args.output_root, role="LLAMA_SCIENTIFIC_REPORT")
        for path in sorted(args.output_root.iterdir())
        if path.name not in {"qwen-archival-exclusion.json"}
    ]
    archive_member = _member(
        args.output_root / "qwen-archival-exclusion.json",
        args.output_root,
        role="NON_SCIENTIFIC_ARCHIVAL_EXCLUSION",
    )
    manifest: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-stage2-llama-only-analysis-manifest/v1",
        "instruction_id": INSTRUCTION,
        "scope": "LLAMA_ONLY_STAGE2_PLUS",
        "scientific_inputs": sorted(inputs, key=lambda row: row["path"]),
        "scientific_input_model_aliases": ["llama3-8b-inst"],
        "qwen_scientific_input_count": 0,
        "outputs": output_members,
        "archival_exclusion_output": archive_member,
    }
    manifest["root_digest"] = canonical_hash(manifest)
    manifest_path = args.output_root / "analysis-manifest.json"
    _write_json(manifest_path, manifest)
    receipt: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-stage2-llama-only-rooted-receipt/v1",
        "instruction_id": INSTRUCTION,
        "scope": "LLAMA_ONLY_STAGE2_PLUS",
        "status": "TERMINAL_SCIENTIFIC_HOLD",
        "hold_reason": (
            "Euler discretization refinement gate failure at locked T_z=1.25"
        ),
        "output_root": str(args.output_root.resolve()),
        "manifest_sha256": _sha(manifest_path),
        "manifest_root": manifest["root_digest"],
        "llama_analysis_root": analysis["root_digest"],
        "qwen_archival_exclusion_identity": qwen_receipt["identity_sha256"],
        "qwen_scientific_input_count": 0,
        "B1_confirmatory_eligibility": False,
        "za_zb_submission_count": 0,
        "scientific_promotion": False,
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    receipt_path = args.output_root / "rooted-receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "output_root": str(args.output_root.resolve()),
        "manifest_root": manifest["root_digest"],
        "receipt_identity": receipt["identity_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
