"""Read-only S terminal analysis, with no model/runtime or adaptive selection.

Inputs must be externally frozen terminal/failure cells with exact marker SHA.
Only consumed JSON is rehashed here: local tensors are not loaded or replayed.
The nine scientific endpoints are distinct from seven executed paths, and the
two T4 prefixes never receive invented fractions of their parent's wall time.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import platform
import re
from pathlib import Path

import numpy as np

from project.run_scripts.native_response_ode_v31.analysis import distribution, summarize
from .publication import digest, member, safe_path, verify_package, write_once
from .sweep import MODELS, endpoint_binding, endpoints, paths

MISSING = "NOT_RECORDED_SCHEMA_GAP"
REPORT_BASE = "experiment-reports/servers/server1/alpha-jv-llama-diagnosis-sweep-2026-09-07-v1"
SUCCESS = "S_NINE_ENDPOINTS_COMPLETE"
ORDER = ("PRE_EDIT", "O_NATIVE", *(s.candidate_id for s in endpoints()))
KINDS = {"RS": "rewrite", "PS": "rephrase", "NS": "locality"}


class AnalysisBoundary(ValueError):
    """Integrity/schema boundary; no imputation or scientific threshold repair."""


def require(condition, label):
    if not condition:
        raise AnalysisBoundary(label)


def _finite(value):
    if isinstance(value, float):
        require(math.isfinite(value), "NONFINITE_RECORDED_INPUT")
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


def _ratio(n, d):
    return n / d if d else None


def _unique(rows, keys, label):
    index = {tuple(r[k] for k in keys): r for r in rows}
    require(len(index) == len(rows), label)
    return index


def _evaluation(evaluation, records, entry, model, candidate, expected):
    """Reuse the strict canonical reducer; raw strings/token arrays stay local."""
    facts, prompts, public = summarize(evaluation, records, entry)
    for metric in KINDS:
        require(facts[metric + "_d"] == expected[metric], "EVALUATION_DENOMINATOR_" + metric)
    requests = []
    ids = {int(r["case_id"]): r["request_sha256"] for r in records}
    for row in prompts:
        row.update(model=model, candidate=candidate, request_sha256=ids[int(row["case_id"])])
    _unique(prompts, ("case_id", "metric", "prompt_index"), "DUPLICATE_PROMPT")
    for metric, kind in KINDS.items():
        requested_kind = kind + ("_target_true" if metric == "NS" else "_target_new")
        token_rows = [r for r in public["rows"] if r["kind"] == requested_kind]
        facts[kind + "_accuracy_n"] = sum(r["correct_token_count"] for r in token_rows)
        facts[kind + "_accuracy_d"] = sum(r["target_token_count"] for r in token_rows)
        facts[kind + "_accuracy_rate"] = _ratio(facts[kind + "_accuracy_n"], facts[kind + "_accuracy_d"])
        strict_success, strict_accuracy = [], []
        for case in ids:
            rows = [r for r in prompts if r["case_id"] == case and r["metric"] == metric]
            strict_success.append(all(r["success"] for r in rows))
            strict_accuracy.append(all(r["strict_teacher_forced"] for r in rows))
            requests.append(dict(model=model, candidate=candidate, case_id=case, request_sha256=ids[case],
                metric=metric, prompt_count=len(rows), success_n=sum(r["success"] for r in rows),
                strict_success=int(strict_success[-1]), strict_teacher_forced=int(strict_accuracy[-1]),
                target_new_nll_mean=float(np.mean([r["target_new_nll"] for r in rows])),
                target_true_nll_mean=float(np.mean([r["target_true_nll"] for r in rows])),
                target_new_nll_max=max(r["target_new_nll"] for r in rows),
                nll_advantage_mean=float(np.mean([r["nll_advantage"] for r in rows]))))
        facts[kind + "_strict_request_success_n"] = sum(strict_success)
        facts[kind + "_strict_request_accuracy_n"] = sum(strict_accuracy)
        facts[kind + "_strict_request_d"] = len(records)
    return facts, prompts, requests


def paired(rows, *, request_level=False):
    """Paired deltas have exact request/hash/prompt joins; ties stay equal."""
    keys = ("model", "candidate", "case_id", "metric") + (() if request_level else ("prompt_index",))
    lookup = _unique(rows, keys, "DUPLICATE_PAIRED_KEY")
    fields = (("target_new_nll_mean", "target_true_nll_mean", "nll_advantage_mean", "strict_success", "strict_teacher_forced")
              if request_level else ("target_new_nll", "target_true_nll", "nll_advantage", "success", "strict_teacher_forced"))
    groups = {}
    for row in rows:
        for reference in ("O_NATIVE", "JV-BASE"):
            if row["candidate"] in (reference, "PRE_EDIT"):
                continue
            refkey = tuple(reference if k == "candidate" else row[k] for k in keys)
            require(refkey in lookup, "MISSING_PAIRED_REFERENCE")
            ref = lookup[refkey]
            require(ref["request_sha256"] == row["request_sha256"], "PAIRED_REQUEST_SHA")
            for field in fields:
                group = (row["model"], row["candidate"], reference, row["metric"], field)
                groups.setdefault(group, []).append(float(row[field]) - float(ref[field]))
    output = []
    for (model, candidate, ref, metric, field), values in sorted(groups.items()):
        lower = field.startswith("target_new_nll")
        if metric == "NS" and ("nll" in field):
            lower = not lower
        sign = -1 if lower else 1
        output.append(dict(model=model, candidate=candidate, reference=ref, metric=metric, field=field,
            unit="REQUEST_REDUCTION" if request_level else "PAIRED_PROMPT", **distribution(values),
            better=sum(sign * v > 0 for v in values), equal=sum(v == 0 for v in values),
            worse=sum(sign * v < 0 for v in values), favorable_direction="lower" if lower else "higher",
            interpretation="preference-direction descriptive delta; not a causal or safety criterion"))
    return output


NODE_FIELDS = ("node", "t", "V_before", "V_after", "V0", "V_ratio", "E", "qN", "qN_ref", "gain",
    "response_sq", "native_path_length_normalized", "actual_barrier_increment", "dissipation_residual",
    "speed_bound", "speed_excess", "velocity_mismatch", "model_error_normalized", "model_error_raw_activation",
    "finite_step_dissipation_defect", "KKT_stationarity", "KKT_complementarity", "state_version",
    "exit_state_version", "node_wall_seconds", "training_semantic_forward_count", "training_semantic_wall_seconds")
LAYER_FIELDS = ("direction_raw_action", "direction_frobenius_squared", "raw_native_velocity_action",
    "normalized_native_velocity_action", "history_velocity_action", "L2_velocity_action", "frobenius_velocity_squared")


def _path_tables(model, path, result, records, fixed_z, entry_w):
    nodes, layers, shadows, first_hit = [], [], [], []
    raw_nodes = result["nodes"]
    require(result.get("status") == "TERMINAL_VALID" and len(raw_nodes) == path.config.N, "PATH_COMPLETENESS")
    semantic_by_case = {int(r["case_id"]): [] for r in records}
    for index, raw in enumerate(raw_nodes):
        require(raw["node"] == index, "NODE_ORDINAL")
        for key, value in (("lambda_response", path.config.lambda_response), ("T", path.config.T),
                           ("N", path.config.N), ("h", path.config.h)):
            require(raw.get(key) == value, "NODE_CONFIG_" + key)
        require(raw.get("fixed_z_sha256") == fixed_z and raw.get("source_entry_sha256") == entry_w, "NODE_ENTRY_Z")
        for key in ("inner_weight_mutation_count", "inner_history_append_count", "controller_heldout_access_count"):
            require(raw.get(key) == 0, "NODE_FORBIDDEN_" + key)
        active = raw.get("active_layers", [])
        require(len(set(active)) == len(active) and set(active) <= set(range(4, 9)), "ACTIVE_LAYER_MAP")
        for name in ("c", "g", "q_layers", "raw_physical_coefficients", "actual_step_coefficients", "predicted_target_contribution"):
            require(len(raw.get(name, [])) == len(active), "NODE_COLUMN_SHAPE_" + name)
        physical = {r["layer"]: r for r in raw.get("actual_physical", [])}
        require(len(physical) == len(raw.get("actual_physical", [])), "PHYSICAL_LAYER_DUPLICATE")
        require(not physical or set(physical) == set(range(4, 9)), "PHYSICAL_LAYER_COMPLETENESS")
        actions = {r["layer"]: r for r in raw.get("layer_actions", [])}
        require(set(actions) == set(active), "NATIVE_ACTION_LAYER_MAP")
        step_total = sum(r["actual_step_DeltaW_squared"] for r in physical.values()) if physical else None
        net_total = sum(r["actual_net_DeltaW_squared"] for r in physical.values()) if physical else None
        node = dict(model=model, path_id=path.path_id, **{k: raw.get(k, MISSING) for k in NODE_FIELDS},
            lambda_response=path.config.lambda_response, T=path.config.T, N=path.config.N, h=path.config.h,
            support_count=sum(c > 0 for c in raw["c"]), active_direction_count=len(active),
            actual_step_DeltaW_squared=step_total, actual_net_DeltaW_squared=net_total,
            actual_step_DeltaW_norm=math.sqrt(step_total) if step_total is not None else MISSING,
            actual_net_DeltaW_norm=math.sqrt(net_total) if net_total is not None else MISSING,
            raw_integrated_work_increment=path.config.h * raw["qN"] * raw["qN_ref"],
            normalized_integrated_work_increment=path.config.h * raw["qN"])
        for field, value in raw.get("compute_seconds", {}).items():
            node["seconds_" + field] = value
        semantic = raw.get("training_semantic_observation")
        if semantic is not None:
            require(len(semantic["request_strict"]) == len(records), "SEMANTIC_REQUEST_COUNT")
            for record, hit in zip(records, semantic["request_strict"], strict=True):
                require(isinstance(hit, bool), "SEMANTIC_BOOLEAN")
                semantic_by_case[int(record["case_id"])].append(hit)
            for name in ("all_strict", "strict_event_count", "event_count", "request_strict_count", "strict_tie_count"):
                node["training_" + name] = semantic.get(name, MISSING)
        residuals = raw.get("raw_residual_norms_by_request")
        if residuals is not None:
            require(len(residuals) == len(records), "RESIDUAL_REQUEST_COUNT")
            node.update({"residual_" + k: v for k, v in distribution(residuals).items()})
        for layer in range(4, 9):
            p = physical.get(layer, {})
            a = actions.get(layer, {})
            row = dict(model=model, path_id=path.path_id, node=index, t=raw["t"], layer=layer,
                native_direction_active=layer in active, qref=raw["qN_ref"], h=path.config.h,
                **{k: a.get(k, MISSING) for k in LAYER_FIELDS},
                **{k: p.get(k, MISSING) for k in ("actual_step_DeltaW_squared", "actual_net_DeltaW_squared", "actual_nonzero")})
            if p:
                row["actual_step_energy_share"] = _ratio(p["actual_step_DeltaW_squared"], step_total)
                row["actual_net_energy_share"] = _ratio(p["actual_net_DeltaW_squared"], net_total)
            if layer in active:
                j = active.index(layer)
                row.update(c=raw["c"][j], g=raw["g"][j], q=raw["q_layers"][j],
                    raw_physical_coefficient=raw["raw_physical_coefficients"][j],
                    actual_step_coefficient=raw["actual_step_coefficients"][j],
                    signed_predicted_target_contribution=raw["predicted_target_contribution"][j],
                    H_diagonal=raw.get("full_H", [[None] * len(active)] * len(active))[j][j],
                    G_diagonal=raw.get("G", [[None] * len(active)] * len(active))[j][j])
                for field in ("raw_native_velocity_action", "normalized_native_velocity_action", "history_velocity_action", "L2_velocity_action"):
                    row[field.replace("velocity_action", "work_increment")] = path.config.h * a[field]
            layers.append(row)
        node["L8_step_energy_share"] = _ratio(physical[8]["actual_step_DeltaW_squared"], step_total) if 8 in physical else MISSING
        nodes.append(node)
        sh = raw.get("shadows", {})
        for single in sh.get("single_layer", []):
            shadows.append(dict(model=model, path_id=path.path_id, node=index, comparator="SINGLE_LAYER_SHADOW",
                layer=single["layer"], objective_improvement=single.get("objective_improvement", MISSING),
                joint_objective_improvement=sh.get("joint_objective_improvement", MISSING),
                native_cosine=single.get("native_cosine"), Rturn=single.get("Rturn"),
                physical_coefficient_difference=single.get("physical_coefficient_difference", MISSING),
                outcome_evaluation="NOT_EXECUTED_SHADOW_ONLY", controller_influence_count=0))
    for record in records:
        seq = semantic_by_case[int(record["case_id"])]
        complete = len(seq) == path.config.N
        hit = next((i + 1 for i, value in enumerate(seq) if value), None) if complete else None
        first_hit.append(dict(model=model, path_id=path.path_id, case_id=record["case_id"],
            request_sha256=record["request_sha256"], observed_nodes=len(seq),
            status="POST_NODE_OBSERVATION_ONLY" if complete else MISSING,
            first_observed_hit_node=hit, first_observed_hit_time=hit * path.config.h if hit else None,
            terminal_strict=seq[-1] if complete else None,
            transient_hit_terminal_miss=bool(hit and not seq[-1]) if complete else None,
            entry_already_hit=MISSING, first_hit_controls_dynamics=False))
    return nodes, layers, shadows, first_hit


def analyze_cell(terminal, result, cohort, source, *, compute=None, components=()):
    """Pure bounded reducer for one complete model (also used by CPU fixtures)."""
    _finite((terminal, result, compute, list(components)))
    model = terminal.get("model_alias")
    require(model in MODELS and result.get("model") == model, "MODEL_IDENTITY")
    require(terminal.get("status") == result.get("status") == SUCCESS, "CELL_NOT_COMPLETE")
    for key, expected in (("JV_endpoints", 9), ("actual_paths", 7), ("completed_nodes", 34),
        ("official_endpoints", 1), ("requests_per_endpoint", len(cohort["records"])),
        ("fixed_z_capture_count", 1), ("fixed_z_recompute_count", 0), ("qref_capture_count", 1),
        ("audit_outcomes_opened", 0), ("source_science_change_count", 0), ("tolerance_change_count", 0)):
        require(terminal.get(key) == expected, "TERMINAL_" + key)
    require(terminal.get("W0_restore") is True and terminal.get("M_restore") is True and result.get("entry_restore") is True, "ENTRY_RESTORE")
    for key in ("head", "tree"):
        require(terminal.get("source_" + key) == source[key], "EXECUTION_SOURCE_" + key)
    for key in ("entry_W_sha256", "entry_M_sha256"):
        require(result.get(key) == terminal.get(key), "COMMON_" + key)
    require(terminal.get("S_DEV_order_sha256") == cohort["request_order_sha256"], "S_DEV_ORDER")
    require(result.get("first_fidelity", {}).get("status") == "PASS", "FIRST_FIDELITY_NOT_PASS")
    specs = {s.candidate_id: s for s in endpoints()}
    parents = {p.path_id: p for p in paths()}
    require(set(result["endpoints"]) == set(specs), "NINE_ENDPOINT_INVENTORY")
    require(set(terminal["completed"]) == set(specs) | {"O_NATIVE"} and len(terminal["completed"]) == 10, "COMPLETED_LABELS")
    trajectory = _unique(result["trajectories"], ("path_id",), "DUPLICATE_PATH")
    require(set(k[0] for k in trajectory) == set(parents), "SEVEN_PATH_INVENTORY")
    records = cohort["records"]
    require(len({r["case_id"] for r in records}) == len(records), "REQUEST_DUPLICATION")
    require(digest([r["request_sha256"] for r in records]) == cohort["request_order_sha256"], "REQUEST_ORDER_ROOT")
    tables = {k: [] for k in ("main", "prompts", "requests", "clocks", "nodes", "layers", "endpoint_layers", "shadows", "first_hit", "compute", "integrity")}
    raw_entry = result["entry_evaluation"]
    raw_endpoints = {"PRE_EDIT": {"evaluation": raw_entry}, "O_NATIVE": result["Official_endpoint"]}
    raw_endpoints.update({k: v["observation"] for k, v in result["endpoints"].items()})
    for candidate in ORDER:
        endpoint = raw_endpoints[candidate]
        require(candidate == "PRE_EDIT" or endpoint.get("status") == "TERMINAL_VALID", "ENDPOINT_NOT_VALID")
        facts, prompts, requests = _evaluation(endpoint["evaluation"], records, raw_entry, model, candidate, cohort["evaluation"]["denominators"])
        row = dict(model=model, candidate=candidate, status="TERMINAL_VALID", **facts,
            endpoint_W_sha256=endpoint.get("selected_weight_endpoint_sha256", terminal["entry_W_sha256"] if candidate == "PRE_EDIT" else MISSING),
            terminal_frobenius_squared=endpoint.get("terminal_net_frobenius_squared", MISSING),
            retained_history_count=endpoint.get("history_append_count", "NOT_APPLICABLE" if candidate == "PRE_EDIT" else MISSING))
        if candidate in specs:
            spec = specs[candidate]
            binding = endpoint_binding(spec, parents[spec.path_id])
            require(result["endpoints"][candidate]["binding"] == binding, "EXACT_CANDIDATE_BINDING")
            for key in ("effective_T", "effective_N", "parent_T", "parent_N", "h", "lambda_response", "normalization_id"):
                require(endpoint.get(key) == binding[key], "ENDPOINT_CLOCK_" + key)
            require(endpoint.get("history_append_count") == binding["history_append_count"], "ENDPOINT_HISTORY_COUNT")
            path_result = trajectory[(spec.path_id,)]["result"]
            require(path_result["endpoints"][candidate] == endpoint, "PATH_ENDPOINT_BYTES_BINDING")
            node = path_result["nodes"][spec.completed_nodes - 1]
            row.update(binding, V_ratio=node["V_ratio"], E_T=node["E"],
                native_path_length_normalized=node["native_path_length_normalized"],
                raw_integrated_work=node["E"] * node["qN_ref"],
                native_net_raw=path_result.get("native_net_raw", MISSING) if not spec.derived_observation_only else MISSING,
                native_net_normalized=path_result.get("native_net_normalized", MISSING) if not spec.derived_observation_only else MISSING,
                endpoint_cost="SHARED_PARENT_PATH_NOT_ALLOCATED" if spec.path_id == "S-PATH-T4" else "PATH_TOTAL_IN_COMPUTE_TABLE")
            tables["clocks"].append(dict(model=model, **binding))
        else:
            row.update(V_ratio=MISSING, E_T="NOT_APPLICABLE_NO_EULER_TRAJECTORY", endpoint_cost="ENTRY_OR_OFFICIAL_COMPONENT")
        tables["main"].append(row)
        tables["prompts"].extend(prompts)
        tables["requests"].extend(requests)
        for weight, energy in endpoint.get("terminal_net_frobenius_squared_by_weight", {}).items():
            layer = re.search(r"(?:^|\.)layers\.(\d+)\.", weight)
            tables["endpoint_layers"].append(dict(model=model, candidate=candidate,
                weight=weight, layer=int(layer.group(1)) if layer else MISSING,
                actual_endpoint_DeltaW_squared=energy,
                energy_share=_ratio(energy, endpoint["terminal_net_frobenius_squared"]),
                provenance="RECORDED_ACTUAL_FP32_ENDPOINT_MINUS_COMMON_COLD_ENTRY"))
        tables["integrity"].append(dict(model=model, candidate=candidate,
            evaluator_identity=digest(endpoint["evaluation"]), same_input_as_entry=True,
            status="CANONICAL_EVALUATOR_REDUCER_PASS", same_entry_W=terminal["entry_W_sha256"],
            same_entry_M=terminal["entry_M_sha256"], fixed_z=result["fixed_z_bundle_sha256"],
            lifecycle="OBSERVATION_ONLY_PREFIX" if candidate in specs and specs[candidate].derived_observation_only else "ENTRY_OR_TERMINAL"))
    for path in paths():
        path_result = trajectory[(path.path_id,)]["result"]
        for name, rows in zip(("nodes", "layers", "shadows", "first_hit"),
            _path_tables(model, path, path_result, records, result["fixed_z_bundle_sha256"], terminal["entry_W_sha256"]), strict=True):
            tables[name].extend(rows)
        tables["compute"].append(dict(model=model, scope="ACTUAL_PATH", component=path.path_id,
            endpoint_ids=list(path.endpoint_ids), actual_nodes=len(path_result["nodes"]),
            main_JVP_calls=path_result.get("main_jvp_count", MISSING),
            dictionary_build_count=path_result.get("dictionary_build_count", MISSING),
            native_solve_count=path_result.get("solve_count", MISSING),
            wall_seconds=path_result.get("total_write_and_endpoint_seconds", MISSING),
            parent_prefix_double_count=0, **{"JVP_" + k: v for k, v in path_result.get("jvp_ledger", {}).items()}))
    for component in components:
        accounting = component.get("accounting", {})
        payload = component.get("payload", {})
        tables["compute"].append(dict(model=model, scope="SCOPED_RUNTIME_COMPONENT", **accounting,
            evaluation_seconds=payload.get("evaluation_seconds", MISSING),
            sum_as_total=False, nested_component_warning="fixed_target_and_entry includes first_fidelity"))
        if accounting.get("component") == "O_NATIVE" and isinstance(payload.get("actual_physical"), dict):
            official = next(row for row in tables["main"] if row["candidate"] == "O_NATIVE")
            for field in ("native_net_raw", "native_net_normalized", "frobenius_net_sq"):
                official[field] = payload["actual_physical"].get(field, MISSING)
        target = payload.get("target_generation_accounting")
        if target is not None:
            tables["compute"].append(dict(model=model, scope="NESTED_TARGET_GENERATION", component="stock_compute_z",
                **target, sum_as_total=False))
    if compute is not None:
        tables["compute"].append(dict(model=model, scope="PROCESS_TOTAL", component="PROCESS",
            wall_seconds=compute.get("process_residency_seconds", MISSING),
            model_load_seconds=compute.get("model_load_seconds", MISSING),
            model_forward_invocations=compute.get("model_forward_invocations", MISSING),
            allocated_GPU_seconds=compute.get("scheduler_GPU_seconds", MISSING),
            authoritative_allocation="SEPARATE_SCHEDULER_LEDGER_REQUIRED", sum_nested_components=False))
    return tables


def runtime_integrity(runtime, model, source, cohort):
    """Read recorded source/dtype contract, not a fresh model inventory."""
    require(runtime.get("model_alias") == model and runtime.get("family") == "AlphaEdit", "RUNTIME_MODEL_FAMILY")
    require(runtime.get("source") == source, "RUNTIME_SOURCE")
    require(runtime.get("model_storage_forward_dtype") == "float32" and runtime.get("controller_dtype") == "float64", "RUNTIME_DTYPE")
    require(runtime.get("autocast") is False and runtime.get("tf32") is False, "RUNTIME_MIXED_PRECISION")
    require(runtime.get("independent_cold_S_DEV") is True and runtime.get("D_restore_dependency") is False, "RUNTIME_COLD_FIXTURE")
    require(runtime.get("request_order_sha256") == cohort["request_order_sha256"] and runtime.get("sample_root") == cohort["ordered_root"], "RUNTIME_SAMPLE_ROOT")
    counts = runtime.get("dtype", {}).get("parameter_dtype_counts", {})
    require(counts and all(name in ("float32", "torch.float32") or value == 0 for name, value in counts.items()), "PARAMETER_DTYPE_INVENTORY")
    require(runtime.get("official", {}).get("tracked_clean") is True, "OFFICIAL_TRACKED_IDENTITY")
    return dict(model=model, status="RECORDED_RUNTIME_BINDING_PASS", parameter_dtype_counts=counts,
        model_storage_forward_dtype=runtime["model_storage_forward_dtype"], controller_dtype=runtime["controller_dtype"],
        contexts_sha256=runtime.get("contexts_sha256", MISSING), hparams_identity=runtime.get("hparams", MISSING),
        official_source=runtime["official"], HF_snapshot=runtime.get("hf_snapshot", MISSING),
        GPU_model=runtime.get("cuda_device", MISSING), GPU_ID=runtime.get("cuda_visible", MISSING),
        scheduler_job=runtime.get("slurm_job_id", MISSING), timing=runtime.get("timing", MISSING))


def _read(path, inputs):
    fact = member(path)
    value = json.loads(safe_path(path).read_text())
    _finite(value)
    require(member(path) == fact, "INPUT_CHANGED_DURING_READ")
    inputs[str(path)] = fact
    return value


def load_frozen(seal_path):
    """A caller must pin both cells after scheduler completion; never polls."""
    inputs = {}
    seal = _read(Path(seal_path).absolute(), inputs)
    require(seal.get("frozen") is True and seal.get("model_or_gpu_actions") == 0, "FROZEN_INPUT_SEAL_REQUIRED")
    require(set(seal.get("cells", {})) == {"0", "1"}, "TWO_MODEL_FROZEN_MARKERS_REQUIRED")
    root = safe_path(seal["root"])
    sample = _read(root / "sample.lock.json", inputs)
    require(digest({k: v for k, v in sample.items() if k != "manifest_identity"}) == sample.get("manifest_identity"), "SAMPLE_MANIFEST_IDENTITY")
    cohort = sample["cohorts"]["S_DEV"]
    require(len(cohort["records"]) == 100 and cohort["evaluation"]["denominators"] == {"RS": 100, "PS": 200, "NS": 1000}, "S_DEV_100_200_1000")
    require(digest(cohort["records"]) == cohort["ordered_root"], "S_DEV_RECORD_ROOT")
    source = _read(root / "source.lock.json", inputs)
    tables = {k: [] for k in ("main", "prompts", "requests", "clocks", "nodes", "layers", "endpoint_layers", "shadows", "first_hit", "compute", "integrity", "runtime", "status")}
    # Optional scheduler/resource provenance is read only and never treated as
    # scientific trajectory evidence or newly inferred GPU-hour accounting.
    provenance = {}
    for name in ("runtime.lock.json", "science.lock.json", "resource.lock.json", "assets.lock.json",
                 "held-inspection.json", "gpu-hour-ledger-terminal.json", "technical-exclusions.json"):
        if (root / name).exists():
            value = _read(root / name, inputs)
            provenance[name] = dict(path=str(root / name), sha256=inputs[str(root / name)]["sha256"],
                                   identity=digest(value))
    accounting = None
    for name in ("accounting", "scheduler", "json_inventory", "historical_recovery"):
        if name in seal:
            fact = seal[name]
            require(member(fact["path"]) == fact, "FROZEN_EXTERNAL_IDENTITY_" + name)
            value = _read(Path(fact["path"]), inputs)
            if name == "accounting":
                require(digest({k: v for k, v in value.items() if k != "identity"}) == value["identity"], "ACCOUNTING_IDENTITY")
                require(all(r["state"] == "COMPLETED" and r["live_state"] is None for r in value["jobs"]), "SCHEDULER_NOT_TERMINAL")
                accounting = value
            if name == "json_inventory":
                for item in value["members"]:
                    require(member(item["path"]) == item, "RAW_JSON_REHASH")
                    inputs[item["path"]] = item
    for cell, model in enumerate(MODELS):
        directory = root / f"cell-{cell}"
        pin = seal["cells"][str(cell)]
        require(pin.get("marker") in ("terminal.json", "failure-boundary.json"), "FROZEN_MARKER_KIND")
        marker = directory / pin["marker"]
        require(member(marker)["sha256"] == pin["sha256"], "FROZEN_MARKER_SHA")
        state = _read(marker, inputs)
        require(not (directory / ("failure-boundary.json" if pin["marker"] == "terminal.json" else "terminal.json")).exists(), "AMBIGUOUS_TERMINAL_FAILURE")
        if pin["marker"] == "failure-boundary.json":
            partial = []
            for node_path in sorted((directory / "paths").glob("*/node-*.json")):
                raw = _read(node_path, inputs)
                partial.append({k: raw.get(k, MISSING) for k in ("node", "arm", "T", "N", "lambda_response")})
            tables["status"].append(dict(model=model, status=state.get("status", MISSING),
                terminal_valid=False, scientific_JV_endpoint_denominator=0, completed_labels=state.get("completed", []),
                partial_node_count=len(partial), stage=state.get("stage", MISSING),
                exception_type=state.get("exception_type", MISSING),
                original_exception_receipt_identity=digest(state.get("original_exception_receipt", {})),
                entry_restore=state.get("entry_restore", MISSING), imputation_count=0,
                classification="RECORDED_BOUNDARY_NO_AUTOMATIC_TECHNICAL_OR_SCIENTIFIC_RECLASSIFICATION"))
            continue
        result = _read(directory / "sweep-result.json", inputs)
        compute = _read(directory / "compute-accounting.json", inputs)
        runtime = _read(directory / "runtime.lock.json", inputs)
        tables["runtime"].append(runtime_integrity(runtime, model, source, cohort))
        fidelity = result['first_fidelity']
        if fidelity.get('execution_device_type') == 'cuda':
            require(all(fidelity.get(k) is True for k in ('M_restore','RNG_unchanged','W0_bytes_restore','W0_pointer_restore')), 'FIDELITY_RESTORE')
            tables.setdefault('fidelity', []).append(dict(model=model, status=fidelity['status'],
                executed_device=fidelity['execution_device_type'],
                JVP_calls=fidelity['jvp_ledger']['jvp_call_count'],
                FD_forwards=fidelity['jvp_ledger']['finite_difference_forward_count'],
                dictionary_build_count=fidelity['dictionary_build_count'],
                cosine_min=fidelity['fd_cosine_min'], relative_L2_max=fidelity['fd_relative_L2_max'],
                fd_epsilons=fidelity['fd_epsilons'],
                authoritative_write_count=fidelity['authoritative_write_count'],
                history_append_count=fidelity['history_append_count'],
                W0_M_RNG_restore=True))
        components = [_read(p, inputs) for p in sorted((directory / "components").glob("*.json"))]
        reduced = analyze_cell(state, result, cohort, source, compute=compute, components=components)
        require(state["model_alias"] == model, "CELL_MODEL_MAP")
        for name, rows in reduced.items():
            tables[name].extend(rows)
        tables["status"].append(dict(model=model, status=SUCCESS, terminal_valid=True,
            scientific_JV_endpoint_denominator=9, Official_endpoint_count=1, W0_observation_count=1,
            partial_node_count=0, W0_restore=True, M_restore=True, fixed_z_capture_count=1,
            fixed_z_recompute_count=0, imputation_count=0))
    tables["paired_prompt"] = paired(tables["prompts"])
    tables["paired_request"] = paired(tables["requests"], request_level=True)
    from .analysis_details import augment_tables
    augment_tables(tables, accounting)
    return dict(tables=tables, inputs=list(inputs.values()), source=source,
        sample_identity=sample["manifest_identity"], order_identity=cohort["request_order_sha256"],
        freeze_seal_identity=digest(seal), provenance=provenance,
        accounting=accounting,
        complete_models=sum(r["terminal_valid"] for r in tables["status"]))


def _csv(rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: json.dumps(v, sort_keys=True, allow_nan=False) if isinstance(v, (dict, list, tuple)) else v
                     for k, v in row.items()} for row in rows)
    return stream.getvalue().encode()


def plot_pngs(tables):
    """Pure table-to-PNG renderer; identical calls must return identical bytes."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    output = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "figure.dpi": 110,
                         "savefig.dpi": 110, "path.simplify": False, "axes.unicode_minus": False}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 7), squeeze=False)
        for i, model in enumerate(MODELS):
            main = {r["candidate"]: r for r in tables["main"] if r["model"] == model}
            for j, metric in enumerate(KINDS):
                ax = axes[i, j]
                ax.bar(range(len(ORDER)), [100 * main[c][metric + "_rate"] if c in main else np.nan for c in ORDER])
                ax.set(xticks=range(len(ORDER)), xticklabels=ORDER, ylim=(0, 100), ylabel=f"{metric} (%)", title=model)
                ax.tick_params(axis="x", labelrotation=75)
        fig.tight_layout()
        stream = io.BytesIO(); fig.savefig(stream, format="png", metadata={"Software": "ODE-edit deterministic Python"}); plt.close(fig)
        output["endpoint-rates.png"] = stream.getvalue()
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
        for i, model in enumerate(MODELS):
            for path in paths():
                rows = [r for r in tables["nodes"] if r["model"] == model and r["path_id"] == path.path_id]
                axes[i, 0].plot([r["t"] for r in rows], [r["V_ratio"] if isinstance(r["V_ratio"], (int, float)) else np.nan for r in rows], label=path.path_id)
                axes[i, 1].plot([r["t"] for r in rows], [r["actual_step_DeltaW_squared"] for r in rows], label=path.path_id)
            axes[i, 0].set(title=model, xlabel="Actual path time", ylabel="V / V0")
            axes[i, 1].set(title="Layer-wise Update Magnitude", xlabel="Actual path time", ylabel="Total actual FP32 step squared Frobenius")
            axes[i, 0].legend(fontsize=6)
        fig.tight_layout()
        stream = io.BytesIO(); fig.savefig(stream, format="png", metadata={"Software": "ODE-edit deterministic Python"}); plt.close(fig)
        output["path-physics.png"] = stream.getvalue()
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), squeeze=False)
        for i, model in enumerate(MODELS):
            rows = [r for r in tables["layers"] if r["model"] == model]
            keys = [(p.path_id, n) for p in paths() for n in range(p.config.N)]
            lookup = {(r["path_id"], r["node"], r["layer"]): r for r in rows}
            for j, field in enumerate(("actual_step_DeltaW_squared", "actual_step_energy_share")):
                def number(value):
                    return float(value) if isinstance(value, (int, float)) else np.nan
                values = np.array([[number(lookup.get((path, n, l), {}).get(field)) for path, n in keys] for l in range(4, 9)], dtype=float)
                pic = axes[i, j].imshow(values, aspect="auto", interpolation="nearest")
                axes[i, j].set(title="Layer-wise Update Magnitude", ylabel=model, yticks=range(5), yticklabels=range(4, 9),
                    xlabel="Actual path-node order (34 distinct nodes); " + ("FP32 step squared Frobenius" if j == 0 else "within-node share"))
                fig.colorbar(pic, ax=axes[i, j])
        fig.tight_layout()
        stream = io.BytesIO(); fig.savefig(stream, format="png", metadata={"Software": "ODE-edit deterministic Python"}); plt.close(fig)
        output["layer-update-magnitude.png"] = stream.getvalue()
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), squeeze=False)
        candidates = ORDER[2:]
        deltas = tables.get("paired_prompt", paired(tables["prompts"]))
        for i, model in enumerate(MODELS):
            for j, field in enumerate(("target_new_nll", "target_true_nll")):
                for metric in ("RS", "PS"):
                    matched = {r["candidate"]: r for r in deltas if r["model"] == model and r["metric"] == metric
                               and r["field"] == field and r["reference"] == "O_NATIVE"}
                    axes[i, j].plot(range(len(candidates)), [matched[c]["mean"] if c in matched else np.nan for c in candidates],
                        marker="o", label=metric)
                axes[i, j].axhline(0., color="grey", linewidth=.6)
                axes[i, j].set(xticks=range(len(candidates)), xticklabels=candidates, title=model + " / " + field,
                               ylabel="Paired mean NLL delta from O_NATIVE")
                axes[i, j].tick_params(axis="x", labelrotation=65)
                axes[i, j].legend()
        fig.tight_layout()
        stream = io.BytesIO(); fig.savefig(stream, format="png", metadata={"Software": "ODE-edit deterministic Python"}); plt.close(fig)
        output["paired-nll-deltas.png"] = stream.getvalue()
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), squeeze=False)
        labels = [p.path_id for p in paths()]
        for i, model in enumerate(MODELS):
            lookup = {r["component"]: r for r in tables["compute"] if r["model"] == model and r["scope"] == "ACTUAL_PATH"}
            for j, field in enumerate(("wall_seconds", "main_JVP_calls")):
                axes[i, j].bar(range(len(labels)), [lookup[p].get(field, np.nan) if p in lookup else np.nan for p in labels])
                axes[i, j].set(xticks=range(len(labels)), xticklabels=labels, ylabel=field,
                    title=model + " / actual path (prefix costs shared)")
                axes[i, j].tick_params(axis="x", labelrotation=60)
        fig.tight_layout()
        stream = io.BytesIO(); fig.savefig(stream, format="png", metadata={"Software": "ODE-edit deterministic Python"}); plt.close(fig)
        output["path-compute.png"] = stream.getvalue()
    return output


def _md_table(rows, columns):
    def cell(value):
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"] +
        ["| " + " | ".join(cell(r.get(c, MISSING)) for c in columns) + " |" for r in rows])


def report_ko(package):
    from .analysis_details import detailed_sections
    tables = package["tables"]
    headline = []
    for row in tables["main"]:
        summary = {"model": row["model"], "candidate": row["candidate"]}
        for key in KINDS:
            summary[key] = f"{row[key+'_n']}/{row[key+'_d']} ({100*row[key+'_rate']:.2f}%)"
        summary.update(rewrite_new_NLL=row["rewrite_new_nll_mean"], rephrase_new_NLL=row["rephrase_new_nll_mean"],
                       rephrase_new_p90=row["rephrase_new_nll_p90"], V_ratio=row["V_ratio"])
        headline.append(summary)
    return "\n\n".join([
        "# AlphaEdit JV D/S — S_DEV B100 실제 sweep 상세 사실 보고서",
        "## 범위·지표·분모",
        f"완료 모델 {package['complete_models']}/2. 모델별 W0 1개, Official O 1개, JV 후보 9개를 구분한다. "
        "9 후보는 7 실제 trajectory/34 node이며 T1·BASE는 T4 경로의 저장 prefix다. S_DEV100 동일 raw request/order, "
        "cold W0/M0, 모델별 fixed-z 1회·N0_SOURCE 및 qref를 공유한다. 새 audit300·sequential/lifelong 실행은 포함하지 않는다. "
        "실패/partial 셀은 run_registry에 남기며 완료 모델의 endpoint 분모로 대체하거나 보간하지 않는다.",
        "RS: rewrite target-new NLL < target-true NLL (100 requests, 각 1 prompt). PS: 같은 부등식의 rephrase prompt (200). "
        "NS: locality target-true NLL < target-new NLL (1000). 모두 strict inequality이며 tie는 실패다. "
        "NLL은 낮을수록 해당 정답열의 likelihood가 높다. new/true를 둘 다 공개하며 preference와 new-NLL 향상을 혼동하지 않는다. "
        "Token accuracy와 all-token strict, request별 all-prompt strict를 별도 기록한다. PS prompt 분모200과 strict-request 분모100은 다르다.",
        _md_table(headline, ("model", "candidate", "RS", "PS", "NS", "rewrite_new_NLL", "rephrase_new_NLL", "rephrase_new_p90", "V_ratio")),
        detailed_sections(package, _md_table),
        "## 실행 clock·동일성",
        _md_table(tables["clocks"], ("model", "candidate_id", "path_id", "lambda_response", "effective_T", "effective_N", "h", "parent_T", "parent_N", "derived_observation_only")),
        "λ 비교는 .01/.0316227766/.1/.316227766/1의 T2/N4 actual endpoint다. N2/N8은 T2 resolution 축, "
        "T1/T4는 h=.5 horizon 축으로 분리한다. prefix는 history append0의 observation이며 sequential resume checkpoint가 아니다. "
        "후보 선택·추가 grid·실패 제거·outcome-based gate는 하지 않았다.",
        "## Request-paired 성능·tail",
        "main.csv에 rewrite/rephrase/locality의 target-new·true NLL mean/median/p90/max와 success/strict/accuracy 절대 분모를 기록했다. "
        "paired_prompt.csv와 paired_request.csv는 O_NATIVE 및 JV-BASE 대비 exact request/hash/prompt 결합의 delta mean/median/p90/max, "
        "better/equal/worse를 담는다. request_reduction은 prompt 평균/최대·all-prompt strict일 뿐 원래 PS/NS prompt 지표를 대체하지 않는다.",
        "## Node·layer·physics·first-hit",
        "nodes.csv/layers.csv는 모든 실제 경로 node를 한 번만 집계한다. actual_step_DeltaW_squared는 연속 materialized FP32 상태의 실제 차, "
        "actual_net_DeltaW_squared는 동일 cold-entry 대비 실제 net 차다. step norm/energy의 합을 net norm으로 부르지 않는다. "
        "native/raw/normalized velocity action, history/L2, h×action work, signed g_l*c_l, c/q/raw physical coefficient를 구분한다. "
        "L8 share는 total physical magnitude와 함께만 해석하며 share 변화만으로 유익한 재배분을 주장하지 않는다. "
        "0 total의 share는 NA, inactive direction의 미기록 q/c는 NA이며 Official Euler node는 만들지 않는다. "
        "KKT/finite defect/model error는 저장된 intrinsic 값을 그대로 공개한다. finite 성능저하/stall은 과학 관찰이며 제외하지 않는다.",
        "first_hit.csv는 저장된 post-node training strict predicate에서 처음 관측된 hit다. entry predicate가 없는 경우 "
        "ENTRY_ALREADY_HIT를 추론하지 않으며 first-hit에 따른 dynamics 변경0. same-state single-layer는 shadow만이며 actual endpoint 효과가 아니다.",
        "## 계산량·시간",
        "compute.csv는 실제 경로와 scoped runtime component 및 process total을 구분한다. fixed_target_and_entry는 first_fidelity를 포함한다. "
        "nested component를 합산해 총시간으로 만들지 않으며 T4의 세 endpoint에 비용을 임의 배분하지 않는다. "
        "native dictionary/solve, model.forward, main/diagnostic JVP, evaluator·semantic wall을 기록된 범위에서만 사용한다. "
        "FLOPs/FLOP-equivalent를 역추정하지 않는다. scheduler GPU-hour는 별도 authoritative ledger가 필요하다. "
        "W0/O와 JV의 endpoint/직접비교는 가능하지만 load/setup/fidelity 비용을 edit-core로 혼합하지 않는다.",
        "## 검증 범위·한계",
        f"execution source: {package['source']['head']} / {package['source']['tree']}. "
        f"S_DEV order: {package['order_identity']}. 분석은 sealed terminal JSON 및 실제 소비 입력의 before/after SHA를 검증했다. "
        "local raw tensor/chronological journal의 production reconstruction은 수행하지 않았으며 STORED_NOT_RECONSTRUCTED다. "
        "D의 W5/context bytes는 실행 중 별도 확보됐지만 exact W9/B10-z replay 및 D intervention은 아직 실행하지 않았다. "
        "원래 terminal의 D unavailable 문구는 실행 소스 봉인 시점 상태이며 현재 파일 확보와 구분한다. S는 D 결과의 대체 근거가 아니다. "
        "NOT_RECORDED_SCHEMA_GAP은 0/추정값으로 채우지 않는다. "
        "모든 관계는 contemporaneous descriptive association이며 causal·lifelong readiness·locality guarantee 또는 자동 promotion 주장은 없다. scientific_promotion=false.",
        "## 산출물",
        "main/prompts/requests/clocks/nodes/layers/endpoint_layers/shadows/first_hit/compute/integrity/runtime/status/paired_prompt/paired_request CSV와 tables.json, "
        "code-generated PNG, plot-reproduction.json, input-manifest.json, source-manifest.json, manifest.json, rooted-receipt.json. "
        "manifest의 각 member bytes/SHA/rows와 plot 재실행 byte identity가 이 보고서에 결속된다.",
    ]) + "\n"


def build_package(seal_path, output, *, analysis_source, approved_base=None):
    """Create once, frozen-data only; source identity is supplied by integration."""
    import matplotlib
    package = load_frozen(seal_path)
    output = Path(output).absolute()
    base = Path(approved_base).absolute() if approved_base else Path(__file__).absolute().parents[3] / REPORT_BASE
    safe_path(output, root=base, make_parents=True)
    require(not output.exists(), "CREATE_ONCE_REPORT_EXISTS")
    output.mkdir(mode=0o700)
    tables = package["tables"]
    for name, rows in tables.items():
        write_once(output / (name + ".csv"), _csv(rows), root=output)
    write_once(output / "tables.json", tables, root=output)
    if package.get('accounting'):
        write_once(output / 'gpu-hour-ledger-terminal.json', package['accounting'], root=output)
    first = plot_pngs(tables)
    second = plot_pngs(tables)
    require(first == second, "PNG_DETERMINISTIC_REPRODUCTION")
    for name, payload in first.items():
        write_once(output / name, payload, root=output)
    inputs = package["inputs"]
    write_once(output / "input-manifest.json", dict(members=inputs, members_root=digest(inputs),
        scope="EXACT_CONSUMED_FROZEN_JSON_ONLY", raw_tensor_replay_verified=False), root=output)
    write_once(output / "source-manifest.json", dict(analysis_source=analysis_source,
        execution_source=package["source"], analyzer=member(__file__),
        dependencies=[member(p) for p in (Path(__file__).with_name("analysis_details.py"), Path(__file__).with_name("sweep.py"), Path(__file__).with_name("publication.py"),
            Path(__file__).parents[1] / "native_response_ode_v31/analysis.py")]), root=output)
    plot_input = member(output / "tables.json")
    write_once(output / "plot-reproduction.json", dict(status="BYTE_STABLE_REEXECUTION_PASS", executions=2,
        plot_input=plot_input, exact_command=f"python3 -m project.run_scripts.alpha_jv_llama_diagnosis.analysis_s plot --tables {output/'tables.json'} --output CREATE_ONCE_DIRECTORY",
        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__,
        backend="Agg", font="DejaVu Sans", DPI=110, code=member(__file__),
        outputs=[member(output / name, relative_to=output) for name in first]), root=output)
    write_once(output / "factual-report-ko.md", report_ko(package).encode(), root=output)
    for fact in inputs:
        require(member(fact["path"]) == fact, "INPUT_SHA_AFTER_CHANGED")
    members = [member(p, relative_to=output) for p in sorted(output.iterdir())]
    manifest = dict(schema="alpha-jv-ds.S-terminal-analysis-manifest.v1", members=members,
        members_root=digest(members), input_members_root=digest(inputs),
        row_counts={k: len(v) for k, v in tables.items()}, source=package["source"],
        sample_identity=package["sample_identity"], order_identity=package["order_identity"],
        input_unchanged=True, imputation_count=0, interpolation_count=0, scientific_promotion=False)
    write_once(output / "manifest.json", manifest, root=output)
    receipt = dict(schema="alpha-jv-ds.S-terminal-analysis-rooted-receipt.v1", scope="FROZEN_S_EXPERIMENT_FACTUAL_ANALYSIS",
        members_root=manifest["members_root"], manifest_sha256=member(output / "manifest.json")["sha256"],
        complete_model_count=package["complete_models"], expected_model_count=2,
        JV_endpoint_count=9 * package["complete_models"], expected_JV_endpoint_count=18,
        evaluated_observation_count=len(tables["main"]), expected_observation_count=22,
        table_rows=manifest["row_counts"], input_unchanged=True, plot_byte_reproduction=True,
        frozen_input_identity=package["freeze_seal_identity"], raw_reconstruction_verified=False,
        model_load_count=0, GPU_action_count=0, Slurm_action_count=0, imputation_count=0,
        audit_execution_count=0, scientific_promotion=False)
    receipt["identity"] = digest(receipt)
    write_once(output / "rooted-receipt.json", receipt, root=output)
    return verify_package(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--frozen-input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--analysis-head", required=True)
    build.add_argument("--analysis-tree", required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("--tables", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "plot":
        output = safe_path(args.output, make_parents=True)
        output.mkdir(mode=0o700, exist_ok=False)
        tables = json.loads(safe_path(args.tables).read_text())
        for name, payload in plot_pngs(tables).items():
            write_once(output / name, payload, root=output)
        return
    result = build_package(args.frozen_input, args.output,
        analysis_source=dict(head=args.analysis_head, tree=args.analysis_tree))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
