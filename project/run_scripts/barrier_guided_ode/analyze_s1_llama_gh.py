#!/usr/bin/env python3
"""Build the GH-owned factual analysis package for the BGODE-R1 Llama S1 pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_SCHEMA = "ode-edit-bgode-r1-s1-six-arm-terminal/v1"
EXPECTED_STATUS = "BGODE_R1_S1_TERMINAL_PASS"
EXPECTED_ARM_ORDER = (
    "official-native-alphaedit-bypass",
    "plain-dynamic",
    "fisher-only-dynamic",
    "full-moving-barrier-dynamic",
    "one-step-full-barrier",
    "frozen-field-n4-split",
)
DYNAMIC_ARMS = EXPECTED_ARM_ORDER[1:]
VERDICT = "TECHNICAL_PASS_SCIENTIFIC_HOLD_MISSING_EXACT_PROGRESS_LOCALIZATION"


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_receipt(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"expected regular non-symlink file: {path}")
    payload = path.read_bytes()
    return {
        "path": display_path or str(path),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def git_blob(repo: Path, commit: str, relative_path: str) -> tuple[bytes, dict[str, Any]]:
    payload = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return payload, {
        "kind": "git_blob",
        "source_head": commit,
        "path": relative_path,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def _function_block(source: str, name: str) -> str:
    marker = f"def {name}("
    start = source.find(marker)
    if start < 0:
        raise ValueError(f"missing function {name}")
    following = source.find("\ndef ", start + len(marker))
    return source[start:] if following < 0 else source[start:following]


def audit_progress_localization(source: str, contract: str, arms: Mapping[str, Any]) -> dict[str, Any]:
    block = _function_block(source, "_run_dynamic_arm")
    direct_scale = "coefficients = (velocity * step_size)" in block
    trajectory_apply = "trajectory.apply(proposal, coefficients)" in block
    rho_in_dynamic = "rho" in block
    contract_equation = "r(W_n+\\rho_nD_n)" in contract
    contract_effective = "h\\rho_nu_n^\\star" in contract
    dynamic_rho_rows = sum(
        int("rho" in node or "root_localization_count" in node)
        for arm in DYNAMIC_ARMS
        for node in arms[arm]["nodes"]
    )
    residuals = [
        float(node["nonlinear_progress_residual"])
        for arm in DYNAMIC_ARMS
        for node in arms[arm]["nodes"]
    ]
    return {
        "contract_requires_scalar_exact_progress_localization": contract_equation and contract_effective,
        "dynamic_source_directly_scales_velocity_by_step_size": direct_scale,
        "dynamic_source_applies_unlocalized_coefficients": direct_scale and trajectory_apply and not rho_in_dynamic,
        "dynamic_source_mentions_rho": rho_in_dynamic,
        "dynamic_node_rho_or_root_receipt_count": dynamic_rho_rows,
        "dynamic_node_count": len(residuals),
        "maximum_absolute_nonlinear_progress_residual": max(abs(value) for value in residuals),
        "terminal_absolute_nonlinear_progress_residual_by_arm": {
            arm: abs(float(arms[arm]["nodes"][-1]["nonlinear_progress_residual"]))
            for arm in DYNAMIC_ARMS
        },
        "matched_progress_attribution_valid": False,
    }


def _sum_node_energy(arm: Mapping[str, Any]) -> float:
    return sum(float(node["update_energy"]["total_energy"]) for node in arm.get("nodes", ()))


def endpoint_rows(terminal: Mapping[str, Any]) -> list[dict[str, Any]]:
    observations = terminal["observations"]
    r_ae = float(terminal["native_horizon"]["r_AE"])
    rows: list[dict[str, Any]] = []
    for arm_name in EXPECTED_ARM_ORDER:
        arm = terminal["arms"][arm_name]
        observation = observations[arm_name]
        event = observation["event"]
        compute = arm.get("compute", {})
        rows.append(
            {
                "arm": arm_name,
                "steps": int(arm.get("steps", 1)),
                "terminal_log_odds": float(event["log_odds"]),
                "terminal_progress_residual_vs_native": float(event["log_odds"]) - r_ae,
                "pair_mass": float(event["pair_mass"]),
                "rewrite_target_new_nll": float(observation["rewrite_target_new"]["nll"]),
                "rewrite_target_true_nll": float(observation["rewrite_target_true"]["nll"]),
                "rewrite_exact": bool(observation["rewrite_target_new"]["exact_satisfied"]),
                "rephrase_target_new_nll": float(observation["rephrase_target_new"]["nll"]),
                "rephrase_target_true_nll": float(observation["rephrase_target_true"]["nll"]),
                "rephrase_exact": bool(observation["rephrase_target_new"]["exact_satisfied"]),
                "locality_forward_kl": float(observation["locality_forward_kl"]),
                "update_energy": _sum_node_energy(arm),
                "dictionary_builds": int(arm.get("dictionary_build_count", 1)),
                "physical_writes": int(arm["physical_final_write_count"]),
                "physical_layer_applies": int(arm["physical_layer_apply_count"]),
                "model_forwards": compute.get("model_forward_invocations", "NOT_RECORDED_NATIVE"),
                "jvp_calls": compute.get("jvp_call_count", "NOT_RECORDED_NATIVE"),
                "primal_prefix_count": compute.get("primal_prefix_count", "NOT_RECORDED_NATIVE"),
                "controller_core_wall_seconds": compute.get("wall_seconds", "NOT_RECORDED_NATIVE"),
            }
        )
    return rows


def node_rows(terminal: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm_name in DYNAMIC_ARMS:
        for node in terminal["arms"][arm_name]["nodes"]:
            shares = node["update_energy"]["per_layer_share"]
            rows.append(
                {
                    "arm": arm_name,
                    "node": int(node["node"]) + 1,
                    "time_entry": float(node["time_entry"]),
                    "step_size": float(node["step_size"]),
                    "entry_log_odds": float(node["event_entry"]["log_odds"]),
                    "exit_log_odds": float(node["event_exit"]["log_odds"]),
                    "expected_exit_log_odds": float(node["expected_exit_log_odds"]),
                    "nonlinear_progress_residual": float(node["nonlinear_progress_residual"]),
                    "pair_mass_exit": float(node["event_exit"]["pair_mass"]),
                    "anchored_kl_entry": float(node["anchored_kl_entry"]),
                    "moving_reference_kl_entry": float(node["moving_reference_kl_entry"]),
                    "update_energy": float(node["update_energy"]["total_energy"]),
                    "maximum_layer_energy_share": max(float(value) for value in shares.values()),
                    "maximum_absolute_coefficient": max(abs(float(value)) for value in node["coefficients"]),
                    "rho": "NOT_RECORDED_NOT_APPLIED",
                    "root_localization_count": 0,
                    "dictionary_refreshed": bool(node["dictionary_refreshed"]),
                }
            )
    return rows


def compute_rows(terminal: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm_name in EXPECTED_ARM_ORDER:
        arm = terminal["arms"][arm_name]
        compute = arm.get("compute", {})
        rows.append(
            {
                "arm": arm_name,
                "dictionary_build_count": int(arm.get("dictionary_build_count", 1)),
                "physical_final_write_count": int(arm["physical_final_write_count"]),
                "physical_layer_apply_count": int(arm["physical_layer_apply_count"]),
                "model_forward_invocations": compute.get("model_forward_invocations", "NOT_RECORDED_NATIVE"),
                "jvp_call_count": compute.get("jvp_call_count", "NOT_RECORDED_NATIVE"),
                "primal_prefix_count": compute.get("primal_prefix_count", "NOT_RECORDED_NATIVE"),
                "finite_difference_forward_count": compute.get(
                    "finite_difference_forward_count", "NOT_RECORDED_NATIVE"
                ),
                "controller_core_wall_seconds": compute.get("wall_seconds", "NOT_RECORDED_NATIVE"),
                "dictionary_wall_seconds": sum(
                    float(node["dictionary_wall_seconds"]) for node in arm.get("nodes", ())
                ),
                "physical_write_wall_seconds": sum(
                    float(node["physical_action"]["wall_seconds"]) for node in arm.get("nodes", ())
                ),
                "score_buffer_peak_bytes": compute.get("score_buffer_peak_bytes", "NOT_RECORDED_NATIVE"),
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    path.chmod(0o600)


def _fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def build_report(
    terminal: Mapping[str, Any],
    endpoints: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    progress: Mapping[str, Any],
    terminal_receipt: Mapping[str, Any],
    contract_receipt: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
) -> str:
    by_arm = {row["arm"]: row for row in endpoints}
    one = by_arm["one-step-full-barrier"]
    frozen = by_arm["frozen-field-n4-split"]
    full = by_arm["full-moving-barrier-dynamic"]
    fisher = by_arm["fisher-only-dynamic"]
    plain = by_arm["plain-dynamic"]
    native = by_arm["official-native-alphaedit-bypass"]
    w0 = terminal["observations"]["w0"]
    fixed_z = terminal["observations"]["fixed_z_star"]
    max_normalization = max(
        abs(float(terminal["observations"][arm]["event"]["normalization_log_residual"]))
        for arm in EXPECTED_ARM_ORDER
    )
    report: list[str] = [
        "# BGODE-R1 S1 Llama request000 — GH 독립 사실 분석",
        "",
        "> **최종 판정:** 실행은 6-arm, FULL-FP32, event normalization, Official AlphaEdit adapter fidelity, "
        "atomic W0 복구를 통과한 유효한 기술 pilot이다. 그러나 최종 수학 설계가 요구한 node별 "
        "scalar exact-progress localization `rho_n`가 dynamic 실행 경로에 구현되지 않았다. 실제 action은 "
        "`beta=h*u`를 곧바로 적용했고 terminal progress residual이 arm별로 크게 달랐다. 따라서 이 결과는 "
        "`unlocalized Euler diagnostic`으로 보존하며, matched-progress 조건의 barrier 인과 효과·ODE 우위·scientific "
        "promotion을 주장하지 않는다.",
        "",
        f"- verdict: `{VERDICT}`",
        f"- sample: Llama B1 request000, case `{terminal['sample']['case_id']}`, single request only",
        f"- raw terminal: `{terminal_receipt['sha256']}`, {terminal_receipt['bytes']} bytes, mode {terminal_receipt['mode']}",
        f"- exact execution source: `{terminal['source_head']}` / `{terminal['source_tree']}`",
        f"- exact source blob SHA: `{source_receipt['sha256']}`",
        f"- final design SHA: `{contract_receipt['sha256']}`",
        "- scientific promotion: `false`",
        "",
        "## 1. 가장 중요한 구현-수학 경계",
        "",
        "최종 설계는 각 node에서 `D_n=h sum_j u_j B_j`를 만든 뒤 "
        "`r(W_n+rho_n D_n)=r(W_n)+h`를 만족하는 scalar root `rho_n`을 찾고 "
        "`beta_n=h rho_n u_n`을 적용하도록 요구한다. exact 실행 source의 `_run_dynamic_arm()`은 "
        "`coefficients=(velocity*step_size)`를 만들고 즉시 `trajectory.apply()`를 호출한다. dynamic block에는 "
        "`rho` 또는 root-localization 호출이 없고 terminal node에도 rho/root receipt가 없다.",
        "",
        "|검사|관측|판정|",
        "|---|---:|---|",
        f"|설계의 scalar exact localization 요구|{_fmt(progress['contract_requires_scalar_exact_progress_localization'])}|required|",
        f"|dynamic source의 `h*u` direct apply|{_fmt(progress['dynamic_source_applies_unlocalized_coefficients'])}|observed|",
        f"|dynamic rho/root receipt|{progress['dynamic_node_rho_or_root_receipt_count']}/{progress['dynamic_node_count']}|absent|",
        f"|최대 abs nonlinear progress residual|{_fmt(progress['maximum_absolute_nonlinear_progress_residual'])}|matched-progress FAIL|",
        f"|최대 event normalization log residual|{_fmt(max_normalization, 8)}|PASS|",
        "",
        "이 경계는 단순 telemetry 누락이 아니다. terminal `r`이 Native target `r_AE`와 크게 달라져 Full/Fisher/Plain이 "
        "같은 semantic progress에서 비교되지 않는다. barrier attribution H2와 dynamic-relinearization H3의 모델 증거는 "
        "이 pilot만으로 닫히지 않는다.",
        "",
        "## 2. Arm endpoint",
        "",
        "낮은 NLL/KL이 좋지만, progress가 일치하지 않으므로 arm 순위는 기술적 관측값이다.",
        "",
        "|arm|steps|terminal r|r-rAE|Rewrite new/true NLL|Rephrase new/true NLL|Loc KL|energy|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in endpoints:
        report.append(
            "|{arm}|{steps}|{r:.6f}|{res:+.6f}|{rw:.6f}/{rwt:.6f}|{rp:.6f}/{rpt:.6f}|{loc:.6f}|{energy:.6f}|".format(
                arm=row["arm"],
                steps=row["steps"],
                r=row["terminal_log_odds"],
                res=row["terminal_progress_residual_vs_native"],
                rw=row["rewrite_target_new_nll"],
                rwt=row["rewrite_target_true_nll"],
                rp=row["rephrase_target_new_nll"],
                rpt=row["rephrase_target_true_nll"],
                loc=row["locality_forward_kl"],
                energy=row["update_energy"],
            )
        )
    report.extend(
        [
            "",
            "### 사실 해석",
            "",
            f"- Native는 `r={native['terminal_log_odds']:.6f}`로 `r_AE`를 그대로 재현했다. adapter relative "
            f"Frobenius는 `{terminal['native_adapter_fidelity']['relative_frobenius']:.8g}`로 "
            f"tolerance `{terminal['native_adapter_fidelity']['tolerance']}`를 통과했다.",
            f"- Fisher는 terminal residual `{fisher['terminal_progress_residual_vs_native']:+.6f}`, Full은 "
            f"`{full['terminal_progress_residual_vs_native']:+.6f}`이다. Fisher가 Rewrite/Rephrase/Loc/energy 모두 "
            "Full보다 낮게 관측됐지만 progress mismatch 때문에 barrier가 해롭다는 결론은 금지한다.",
            f"- Plain은 node 3에서 coefficient abs max `{max(float(row['maximum_absolute_coefficient']) for row in nodes if row['arm']=='plain-dynamic'):.6f}`, "
            f"total energy `{plain['update_energy']:.3f}`, Loc KL `{plain['locality_forward_kl']:.3f}`로 붕괴했다. "
            "이는 unlocalized first-order step의 불안정성 증거이지 Full barrier의 causal 승리 증거가 아니다.",
            f"- one-step과 frozen-N4 endpoint는 Rewrite new delta `{abs(one['rewrite_target_new_nll']-frozen['rewrite_target_new_nll']):.9f}`, "
            f"Rephrase new delta `{abs(one['rephrase_target_new_nll']-frozen['rephrase_target_new_nll']):.9f}`, "
            f"Loc KL delta `{abs(one['locality_forward_kl']-frozen['locality_forward_kl']):.9f}`로 사실상 동일하다. "
            "이는 frozen-field split identity를 지지하지만 dynamic ODE 우위를 지지하지 않는다.",
            "",
            "## 3. Node progress residual",
            "",
            "|arm|K1|K2|K3|K4|terminal abs|",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    node_by_arm = {arm: [row for row in nodes if row["arm"] == arm] for arm in DYNAMIC_ARMS}
    for arm in DYNAMIC_ARMS:
        values = [float(row["nonlinear_progress_residual"]) for row in node_by_arm[arm]]
        padded = values + [math.nan] * (4 - len(values))
        report.append(
            f"|{arm}|{_fmt(padded[0])}|{_fmt(padded[1])}|{_fmt(padded[2])}|{_fmt(padded[3])}|{abs(values[-1]):.6f}|"
        )
    report.extend(
        [
            "",
            "`a^T u=1`의 solver equality residual과 위 nonlinear residual은 다른 양이다. 전자는 local tangent equation, "
            "후자는 실제 nonlinear W update가 의도한 `h`만큼 진행했는지를 측정한다. 이번 pilot은 전자를 통과했지만 "
            "후자를 local root로 교정하지 않았다.",
            "",
            "## 4. z→W와 single-request 성능",
            "",
            f"- W0 Rewrite/Rephrase new NLL: `{w0['rewrite_target_new']['nll']:.6f}` / `{w0['rephrase_target_new']['nll']:.6f}`.",
            f"- fixed native z*: `{fixed_z['rewrite_target_new']['nll']:.6f}` / `{fixed_z['rephrase_target_new']['nll']:.6f}`. "
            "Rewrite는 exact target을 만족했지만 Rephrase는 만족하지 않았다.",
            f"- Native W: `{native['rewrite_target_new_nll']:.6f}` / `{native['rephrase_target_new_nll']:.6f}`. "
            f"fixed-z 대비 W gap은 `{native['rewrite_target_new_nll']-fixed_z['rewrite_target_new']['nll']:+.6f}` / "
            f"`{native['rephrase_target_new_nll']-fixed_z['rephrase_target_new']['nll']:+.6f}`이다.",
            "- 단일 request pilot이므로 success rate, 평균 우위, generalization/locality promotion으로 확장하지 않는다.",
            "",
            "## 5. 기술 gate와 compute",
            "",
            f"- terminal status `{terminal['status']}`; FULL-FP32 parameter tensors "
            f"`{terminal['dtype']['parameter_dtype_counts'].get('torch.float32', 0)}/291`; BF16/FP16/quantization/autocast/cast 0.",
            f"- total wall `{terminal['resources']['total_wall_seconds']:.3f}s`; model load "
            f"`{terminal['resources']['model_load_wall_seconds']:.3f}s`; peak allocated "
            f"`{terminal['resources']['peak_gpu_allocated_bytes']}` bytes; reserved `{terminal['resources']['peak_gpu_reserved_bytes']}` bytes.",
            f"- fixed z compute/recompute `{terminal['fixed_z']['compute_count']}/{terminal['fixed_z']['recompute_count']}`; "
            "Euler history append 0; heldout/locality controller influence 0; terminal W0 pointer+bytes rollback gate PASS.",
            f"- Plain first-node JVP-vs-FD max abs error `{terminal['arms']['plain-dynamic']['finite_difference']['max_absolute_error']:.8f}` "
            "with configured allclose PASS.",
            "- Native physical writes/layers `1/5`; each dynamic N4 arm `4/20`; one-step `1/5`. "
            "dictionary helper internal physical forward count는 `NOT_RESOLVED_SOURCE_LEVEL_HELPER_INTERNALS`로 보존한다.",
            "",
            "## 6. 과학 판정",
            "",
            "|hypothesis|판정|근거|",
            "|---|---|---|",
            "|H1 event validity|SUPPORTED_FOR_THIS_PILOT|normalization residual finite/small, tokenizer boundary fixed, JVP-vs-FD PASS|",
            "|H2 barrier attribution|OPEN / INVALID_MATCHED_PROGRESS_PANEL|Full/Fisher/Plain terminal progress와 energy가 불일치|",
            "|H3 ODE attribution|OPEN|one-step=frozen identity는 확인, dynamic benefit은 localization 없는 trajectory라 판정 불가|",
            "|Native fidelity|PASS|Official endpoint relative Frobenius within tolerance|",
            "|Scientific promotion|FALSE|B=1 단일 sample + missing localization|",
            "",
            "다음 유효 실험은 현재 결과의 threshold 조정이나 imputation이 아니라, exact design대로 node별 monotonic bracket과 "
            "`rho_n` root를 구현하고 `rho`, actual/linear progress, subdivision count를 저장한 뒤 동일 sample·동일 terminal progress에서 "
            "Fisher/Full/Plain을 재비교하는 것이다. Qwen 결과는 SH1 소유의 별도 package로 유지한다.",
            "",
            "## 7. 산출물",
            "",
            "- `arm-endpoints.csv`: 6-arm endpoint NLL/event/KL/energy/compute",
            "- `node-trajectory.csv`: dynamic node별 actual/expected progress, residual, energy, coefficient, rho absence",
            "- `compute.csv`: dictionary/write/JVP/forward/메모리 ledger",
            "- `analysis-summary.json`: 핵심 gate·verdict·비교 경계",
            "- `analysis-manifest.json`, `rooted-analysis-receipt.json`: input/output byte identity",
            "",
        ]
    )
    return "\n".join(report)


def _root(rows: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        f"{row['sha256']} {row.get('mode', 'git')} {row['bytes']} {row['path']}\n"
        for row in rows
    ]
    return sha256_bytes("".join(sorted(lines)).encode("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    path.chmod(0o600)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve(strict=True)
    terminal_path = args.terminal.resolve(strict=True)
    direct_z_path = args.direct_z.resolve(strict=True)
    contract_path = args.contract.resolve(strict=True)
    output = args.output.resolve(strict=False)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)

    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("schema") != EXPECTED_SCHEMA or terminal.get("status") != EXPECTED_STATUS:
        raise ValueError("terminal schema/status mismatch")
    if tuple(terminal.get("arm_order", ())) != EXPECTED_ARM_ORDER:
        raise ValueError("arm order mismatch")
    if set(terminal["arms"]) != set(EXPECTED_ARM_ORDER):
        raise ValueError("arm inventory mismatch")
    if terminal["sample"]["batch_label"] != "B1" or terminal["sample"]["request_ordinal"] != 0:
        raise ValueError("sample is not sealed B1 request000")
    if terminal["dtype"]["parameter_dtype_counts"] != {"torch.float32": 291}:
        raise ValueError("FULL-FP32 inventory mismatch")
    if any(
        terminal["dtype"][name]
        for name in (
            "autocast_enabled",
            "bf16_conversion_count",
            "fp16_conversion_count",
            "numeric_storage_cast_count",
            "quantized_parameter_count",
            "tf32_enabled",
        )
    ):
        raise ValueError("low precision or cast path observed")
    if terminal["scientific_promotion"]:
        raise ValueError("unexpected scientific promotion")

    source_bytes, source_receipt = git_blob(
        repo, terminal["source_head"], "project/run_scripts/barrier_guided_ode/s1_experiment.py"
    )
    contract_bytes = contract_path.read_bytes()
    progress = audit_progress_localization(
        source_bytes.decode("utf-8"), contract_bytes.decode("utf-8"), terminal["arms"]
    )
    if not progress["contract_requires_scalar_exact_progress_localization"]:
        raise ValueError("final design exact-localization text not found")
    if not progress["dynamic_source_applies_unlocalized_coefficients"]:
        raise ValueError("execution source no longer matches unlocalized pilot")
    if progress["dynamic_node_rho_or_root_receipt_count"] != 0:
        raise ValueError("unexpected dynamic root receipt")

    endpoints = endpoint_rows(terminal)
    nodes = node_rows(terminal)
    compute = compute_rows(terminal)
    one = next(row for row in endpoints if row["arm"] == "one-step-full-barrier")
    frozen = next(row for row in endpoints if row["arm"] == "frozen-field-n4-split")
    equivalence = {
        "rewrite_target_new_nll_abs_delta": abs(
            float(one["rewrite_target_new_nll"]) - float(frozen["rewrite_target_new_nll"])
        ),
        "rephrase_target_new_nll_abs_delta": abs(
            float(one["rephrase_target_new_nll"]) - float(frozen["rephrase_target_new_nll"])
        ),
        "locality_forward_kl_abs_delta": abs(
            float(one["locality_forward_kl"]) - float(frozen["locality_forward_kl"])
        ),
        "terminal_log_odds_abs_delta": abs(
            float(one["terminal_log_odds"]) - float(frozen["terminal_log_odds"])
        ),
    }
    if max(equivalence.values()) > 1.0e-4:
        raise ValueError("one-step/frozen-field identity unexpectedly failed")

    terminal_receipt = file_receipt(terminal_path)
    direct_z_receipt = file_receipt(direct_z_path)
    contract_receipt = file_receipt(contract_path)
    report = build_report(
        terminal,
        endpoints,
        nodes,
        progress,
        terminal_receipt,
        contract_receipt,
        source_receipt,
    )
    report_path = output / "report-ko.md"
    report_path.write_text(report, encoding="utf-8")
    report_path.chmod(0o600)
    write_csv(output / "arm-endpoints.csv", endpoints)
    write_csv(output / "node-trajectory.csv", nodes)
    write_csv(output / "compute.csv", compute)

    summary = {
        "schema": "ode-edit-bgode-r1-s1-llama-gh-independent-analysis-summary/v1",
        "verdict": VERDICT,
        "terminal_status": terminal["status"],
        "technical_execution_valid": True,
        "matched_progress_attribution_valid": False,
        "scientific_promotion": False,
        "source_head": terminal["source_head"],
        "source_tree": terminal["source_tree"],
        "sample_identity": terminal["sample_identity"],
        "stream_root": terminal["sample"]["stream_root_sha256"],
        "order_root": terminal["sample"]["all_request_order_sha256"],
        "progress_localization_audit": progress,
        "one_step_frozen_equivalence": equivalence,
        "native_adapter_fidelity": terminal["native_adapter_fidelity"],
        "event_normalization_max_abs_residual": max(
            abs(float(terminal["observations"][arm]["event"]["normalization_log_residual"]))
            for arm in EXPECTED_ARM_ORDER
        ),
        "raw_mutation_count": 0,
        "imputation_count": 0,
        "gpu_replay_count": 0,
    }
    _write_json(output / "analysis-summary.json", summary)

    input_members = [terminal_receipt, direct_z_receipt, contract_receipt, source_receipt]
    output_names = (
        "report-ko.md",
        "arm-endpoints.csv",
        "node-trajectory.csv",
        "compute.csv",
        "analysis-summary.json",
    )
    output_members = [file_receipt(output / name, display_path=name) for name in output_names]
    manifest = {
        "schema": "ode-edit-bgode-r1-s1-llama-gh-independent-analysis-manifest/v1",
        "verdict": VERDICT,
        "source_head": terminal["source_head"],
        "source_tree": terminal["source_tree"],
        "sample_identity": terminal["sample_identity"],
        "stream_root": terminal["sample"]["stream_root_sha256"],
        "order_root": terminal["sample"]["all_request_order_sha256"],
        "input_members": input_members,
        "input_members_root": _root(input_members),
        "output_members": output_members,
        "output_members_root": _root(output_members),
        "scientific_promotion": False,
        "raw_mutation_count": 0,
        "imputation_count": 0,
        "gpu_replay_count": 0,
    }
    manifest["identity_sha256"] = sha256_bytes(canonical_json(manifest))
    manifest_path = output / "analysis-manifest.json"
    _write_json(manifest_path, manifest)
    manifest_receipt = file_receipt(manifest_path, display_path="analysis-manifest.json")
    receipt = {
        "schema": "ode-edit-bgode-r1-s1-llama-gh-rooted-analysis-receipt/v1",
        "status": VERDICT,
        "analysis_manifest_sha256": manifest_receipt["sha256"],
        "analysis_manifest_identity": manifest["identity_sha256"],
        "input_members_root": manifest["input_members_root"],
        "output_members_root": manifest["output_members_root"],
        "report_sha256": next(row["sha256"] for row in output_members if row["path"] == "report-ko.md"),
        "scientific_promotion": False,
        "raw_mutation_count": 0,
        "imputation_count": 0,
        "gpu_replay_count": 0,
    }
    receipt["identity_sha256"] = sha256_bytes(canonical_json(receipt))
    receipt_path = output / "rooted-analysis-receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "output": str(output),
        "verdict": VERDICT,
        "report": file_receipt(report_path),
        "manifest": file_receipt(manifest_path),
        "manifest_identity": manifest["identity_sha256"],
        "receipt": file_receipt(receipt_path),
        "receipt_identity": receipt["identity_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--direct-z", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(analyze(parse_args()), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
