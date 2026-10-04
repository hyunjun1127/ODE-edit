"""CPU-only compact publication of already stored V13 B1 scalar results.

This module imports no model/runtime libraries, runs no evaluator or scheduler
commands, and never changes the execution attempt. Large context/per-owner
records stay local; only pooled/by-layer aggregates and small collector outputs
are exported. Existing raw inventory hashes are reused with current stat checks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


BRANCHES = ("RT", "RD", "MT", "MD", "CD")
METRICS = (
    "action_norm", "target_norm", "error_norm", "norm_ratio",
    "directional_ratio", "cosine", "relative_error",
    "zero_target_leakage_over_anchor",
)
CHANNELS = ("ideal", "cast", "effective", "actual")
GAPS = (
    "inherited_gap_norm", "direct_error_norm", "final_virtual_gap_norm",
    "inherited_direct_error_dot",
)
STAT_FIELDS = ("count", "undefined", "mean", "median", "min", "max", "RMS")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    present = [float(x) for x in values if x is not None]
    require(all(math.isfinite(x) for x in present), "nonfinite stored scalar")
    return {
        "count": len(present), "undefined": len(values) - len(present),
        "mean": statistics.mean(present) if present else None,
        "median": statistics.median(present) if present else None,
        "min": min(present) if present else None,
        "max": max(present) if present else None,
        "RMS": math.sqrt(statistics.mean(x*x for x in present)) if present else None,
    }


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def create_or_exact_reuse(path, data, replace_generated=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.is_file(), "publication target is not a regular file")
        if path.read_bytes() != data:
            require(replace_generated, "publication file already exists with different bytes: " + str(path))
            with path.open("wb") as handle:
                handle.write(data)
    else:
        with path.open("xb") as handle:
            handle.write(data)


def csv_bytes(rows):
    require(bool(rows), "empty generated CSV")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


class StoredAttempt:
    def __init__(self, attempt):
        self.root = attempt.resolve(strict=True)
        self.receipts = {}
        inventory_path = self.root / "cpu-report/inventory.json"
        inventory_bytes = inventory_path.read_bytes()
        inventory = json.loads(inventory_bytes)
        require(inventory["no_checkpoint"] and inventory["raw_local_KEEP"],
                "original inventory storage policy mismatch")
        self.inventory = {entry["path"]: entry for entry in inventory["files"]}

    def read(self, relative):
        path = self.root / relative
        require(path.is_file() and not path.is_symlink(), "not a regular input")
        status = path.stat()
        prior = self.inventory.get(str(path))
        data = path.read_bytes()
        if prior is not None:
            require(status.st_size == prior["bytes"] and
                    status.st_mtime_ns == prior["mtime_ns"] and
                    status.st_ino == prior["inode"], "raw changed since inventory")
            digest = prior["sha256"]
            verification = "EXISTING_COLLECTOR_FULL_SHA_PLUS_CURRENT_STAT"
        else:
            digest = hashlib.sha256(data).hexdigest()
            verification = "CURRENT_SMALL_NAMED_INPUT_FULL_SHA"
        self.receipts[relative] = {
            "path": str(path), "bytes": len(data), "sha256": digest,
            "verification": verification,
        }
        return data

    def load(self, relative):
        return json.loads(self.read(relative))


def realization_records(branch, actions, layers):
    records = []
    rows = actions["rows"]
    require(len(rows) == 3500, "expected 100 owners x 7 native rows x 5 layers")
    identities = {(r["layer"], r["global_row"]) for r in rows}
    require(len(identities) == len(rows), "duplicate native action identity")
    require({r["owner"] for r in rows} == set(range(100)), "owner coverage")
    for scope in ("native_mean", "canonical", "rewrite", "KL"):
        if scope == "native_mean":
            scoped = [dict(layer=int(layer), owner=owner, **{
                channel: values[channel][owner] for channel in CHANNELS})
                for layer, values in actions["mean"].items()
                for owner in range(100)]
        else:
            scoped = [r for r in rows if (r["canonical"] if scope == "canonical"
                       else r["kind"] == ("kl" if scope == "KL" else "rewrite"))]
        expected = {"native_mean": 500, "canonical": 500,
                    "rewrite": 3000, "KL": 500}[scope]
        require(len(scoped) == expected, "native scope coverage: " + scope)
        channels = CHANNELS if scope == "native_mean" else CHANNELS + ("net_entry_change",)
        for layer in ("ALL",) + tuple(layers):
            selected = scoped if layer == "ALL" else [r for r in scoped if r["layer"] == layer]
            for channel in channels:
                metrics = [r[channel] for r in selected]
                record = {
                    "branch": branch, "scope": scope, "channel": channel,
                    "layer": layer, "rows": len(metrics),
                    "zero_target_count": sum(x["target_norm"] == 0 for x in metrics),
                    "fields": {key: stats([x[key] for x in metrics]) for key in METRICS},
                }
                require(all((x["norm_ratio"] is None) == (x["target_norm"] == 0)
                            for x in metrics), "zero-target ratio semantics")
                by_owner = defaultdict(list)
                for row, metric in zip(selected, metrics):
                    by_owner[row["owner"]].append(metric["error_norm"])
                record["owner_error_weighted_RMS"] = stats([
                    math.sqrt(statistics.mean(x*x for x in values))
                    for values in by_owner.values()])
                record["owner_error_max"] = stats([max(v) for v in by_owner.values()])
                if channel == "actual" and scope != "native_mean":
                    record["gaps"] = {key: stats([r[key] for r in selected]) for key in GAPS}
                records.append(record)
    return records


def realization_csv(records):
    rows = []
    for record in records:
        identity = {key: record[key] for key in
                    ("branch", "scope", "channel", "layer", "rows", "zero_target_count")}
        fields = dict(record["fields"])
        fields["owner_error_weighted_RMS"] = record["owner_error_weighted_RMS"]
        fields["owner_error_max"] = record["owner_error_max"]
        fields.update(record.get("gaps", {}))
        for metric, values in fields.items():
            rows.append(dict(identity, metric=metric, **values))
    return rows


def shares_records(branch, shares, layers):
    result = []
    require(shares["currency"] == "effective_FP32_direct_action_over_entry_anchor",
            "share currency changed")
    for scope in ("mean", "canonical", "rewrite", "KL"):
        if scope in ("mean", "canonical"):
            rows = shares[scope]
        else:
            role = "kl" if scope == "KL" else "rewrite"
            rows = [row for row, kind in zip(shares["contexts"], shares["context_roles"])
                    if kind == role]
        expected = 600 if scope == "rewrite" else 100
        require(len(rows) == expected, "share scope coverage")
        counts = dict(Counter(row["status"] for row in rows))
        fields = {
            "share_L1": [r["share_L1"] for r in rows],
            "realized_relative_total": [r["realized_relative_total"] for r in rows],
        }
        for index, layer in enumerate(layers):
            for kind in ("planned_share", "direct_share"):
                fields[kind + "_layer_" + str(layer)] = [
                    None if r[kind] is None else r[kind][index] for r in rows]
        for field, values in fields.items():
            result.append(dict(branch=branch, scope=scope, metric=field, rows=len(rows),
                               status_counts=json.dumps(counts, sort_keys=True), **stats(values)))
    return result


def compact_writer(branch, writer, additivity, restore):
    require(restore["verified"] and restore["W_H_RNG"] and
            restore["plan_cache_context_ledger_nonselected"], "branch RAM restore failed")
    require(writer["history_appends"] == 5 and not writer["checkpoint_saved"],
            "history/checkpoint policy")
    history = {int(row["layer"]): row for row in writer["history"]}
    rows = []
    for layer, value in writer["layers"].items():
        layer = int(layer)
        solver = value["solver"]
        hist = history[layer]
        local = additivity["checks"][str(layer)]
        parity = value["ideal_effective_parity"]
        require(solver["numerical_projection_verified"] and parity["pass_"] and
                local["pass_"], "stored solver/local parity failed")
        require(hist["append_count"] == 1 and hist["columns"] == 100 and
                hist["rewrite_only"] and not hist["KL_in_history"] and hist["CPU_FP32"],
                "history is not once/layer rewrite-only CPU FP32")
        rows.append(dict(
            branch=branch, layer=layer, solver_kind=solver["kind"],
            solver_status=solver["status"], nominal_geometry_status=solver["nominal_geometry_status"],
            constraints=solver["shape"]["constraints"], rank=solver["rank"],
            positive_discarded_count=solver["positive_discarded_count"], cutoff=solver["cutoff"],
            cutoff_rule=solver["cutoff_rule"], retained_condition=solver["retained_condition"],
            original_columns_retained=solver["original_columns_retained"],
            incompatible_target_norm=solver["incompatible_target_norm"],
            weighted_target_mismatch=solver["weighted_target_mismatch"],
            weighted_target_norm=solver["weighted_target_norm"],
            projected_target_residual=solver["projected_target_residual"],
            numerical_tolerance=solver["numerical_tolerance"],
            numerical_projection_verified=solver["numerical_projection_verified"],
            metric_SPD=solver["metric"]["SPD"], metric_jitter=solver["metric"]["jitter"],
            metric_skew_relative=solver["metric"]["skew_relative"],
            ideal_Q=value["ideal_Q"], effective_Q=value["effective_Q"],
            ideal_update_norm=solver["ideal_update_norm"],
            cast_update_norm=value["cast_update_norm"],
            effective_update_norm=value["effective_update_norm"],
            ideal_effective_parity=parity["pass_"], parity_failed_elements=parity["failed_elements"],
            parity_max_absolute=parity["max_absolute"], parity_atol=parity["atol"], parity_rtol=parity["rtol"],
            local_additivity_pass=local["pass_"], local_failed_elements=local["failed_elements"],
            local_max_absolute=local["max_absolute"],
            history_appends=hist["append_count"], history_columns=hist["columns"],
            history_rewrite_only=hist["rewrite_only"], history_CPU_FP32=hist["CPU_FP32"],
            history_before_sha256=hist["before"], history_after_sha256=hist["after"],
            final_native_key_sha256=hist["key_sha256"], weight_after_sha256=value["weight_after"],
            branch_fresh_upper_capture=solver["branch_fresh_upper_capture"],
            solver_seconds=solver["seconds"], layer_phase_seconds=value["seconds"],
            metric_factor_seconds=solver["metric"]["factor_seconds"],
            branch_writer_phase_seconds=writer["seconds"],
        ))
    return {
        "branch": branch, "layers": rows, "RAM_restore": restore,
        "ideal_Q_sum": math.fsum(row["ideal_Q"] for row in rows),
        "effective_Q_sum": math.fsum(row["effective_Q"] for row in rows),
        "history_appends": writer["history_appends"],
        "capture_forward_calls": writer["capture_forward_calls"],
        "writer_phase_seconds": writer["seconds"],
        "writer_phase_excludes_official_endpoint_evaluation": True,
        "checkpoint_saved": writer["checkpoint_saved"],
    }


def self_test():
    sample = stats([0, 1, 1000, None])
    require(sample["count"] == 3 and sample["undefined"] == 1 and
            sample["median"] == 1 and sample["max"] == 1000, "aggregate/null regression")
    empty = stats([None, None])
    require(empty["mean"] is None and empty["undefined"] == 2, "all-null regression")
    require(stats([-2, 1])["min"] == -2, "signed directional regression")
    require(b"null" in encoded(empty), "JSON null preservation")


def publish(attempt, output, replace_generated=False):
    self_test()
    source = StoredAttempt(attempt)
    output = output.resolve()
    require(source.root not in (output, *output.parents), "cannot write into original attempt")
    lock = source.load("execution.lock.json")
    metrics = source.load("cpu-report/metrics.json")
    require(metrics["status"] == "COMPLETED" and
            all(value == "COMPLETE" for value in metrics["coverage"].values()),
            "this complete-publication exporter requires completed stored endpoints")
    terminal = source.load("B1/terminal.json")
    require(terminal["fit_count"] == 1 and terminal["no_B2"] and
            not terminal["checkpoint_saved"] and tuple(terminal["branches_completed"]) == BRANCHES,
            "B1 scope changed")
    plan = source.load("B1/plan.json")
    layers = plan["layers"]
    require(layers == [4, 5, 6, 7, 8] and len(plan["ids"]) == 100 and
            plan["terminal_z_same_forward"] and not plan["persisted_plan_tensors"],
            "plan/input/terminal scope")
    fit = source.load("B1/fit/fit.json")
    candidates = list(fit["candidates"].values())
    require(len(candidates) == fit["requests"] == 100 and fit["request_evaluations"] == 2500 and
            fit["request_updates"] == 2400 and fit["terminal_extra_backward"] == 0 and
            fit["terminal_extra_forward"] == 0, "stored fit counts changed")
    fit_summary = {key: value for key, value in fit.items() if key != "candidates"}
    fit_summary.update(
        final_J=stats([row["J"] for row in candidates]),
        terminal_candidate_counts=dict(Counter(row["candidate"] for row in candidates)),
        terminal_stop_reason_counts=dict(Counter(row["reason"] for row in candidates)),
        fit_count=1, unique_requests=100, endpoint_branch_count=5,
        branches_are_not_500_unique_requests=True,
        source_commit=lock["source_commit"], source_tree=lock["source_tree"],
    )
    realization, shares, writers = [], [], []
    for branch in BRANCHES:
        actions = source.load("B1/" + branch + "/actions.json")
        realization.extend(realization_records(branch, actions, layers))
        shares.extend(shares_records(branch, source.load("B1/" + branch + "/shares.json"), layers))
        writers.append(compact_writer(
            branch, source.load("B1/" + branch + "/writer.json"),
            source.load("B1/" + branch + "/local-additivity.json"),
            source.load("B1/" + branch + "/restore.json")))
    copies = {
        "metrics.json": "metrics.json", "comparison-B1.csv": "comparison-B1.csv",
        "cost.json": "cost.json", "terminal.json": "collector-terminal.json",
        "inventory.json": "collector-inventory.json", "report-ko.md": "collector-report-ko.md",
    }
    for original, destination in copies.items():
        create_or_exact_reuse(output / destination, source.read("cpu-report/" + original))
    # Preserve every actual/net by-layer record. Intermediate ideal/cast/effective
    # context channels are pooled, while native-mean channels also remain by-layer.
    realization = [row for row in realization if row["scope"] == "native_mean" or
                   row["channel"] in ("actual", "net_entry_change") or row["layer"] == "ALL"]
    summary = {
        "schema": 1, "source_commit": lock["source_commit"], "source_tree": lock["source_tree"],
        "scope": "stored same-fit B1 scalar telemetry only; no new model/evaluator/scheduler calls",
        "definitions": {
            "target": "All action metrics compare with planner D, including RT/MT; their solve RHS is tracking T=z_virtual-h_actual, not D.",
            "actual": "Own-layer local action at actual key; not cumulative net entry hidden displacement.",
            "net_entry_change": "Final actual hidden minus clean entry hidden, compared separately with D.",
            "native_mean": "Stored native FP32 nested rewrite group mean action per owner and layer.",
            "canonical": "Canonical rewrite context subset; do not add its rows to rewrite totals.",
            "pooling": "Uniform row pooling over requested scope and layers; by-layer records remain separate. Mean/median/max are all retained.",
            "layout": "All channels pooled. Every native-mean channel and actual/net context channel also has per-layer aggregates. No context or per-owner raw arrays.",
            "zero_target": "D exactly zero: ratios null; anchor-normalized leakage retained. Tiny positive D ratios are not clipped or silently excluded.",
            "owner_error_weighted_RMS": "Within each owner, equal-context mean squared error then sqrt; aggregate across owners. No per-owner arrays exported.",
            "shares": "Effective FP32 direct action divided by entry anchor. realized_relative_total is a relative magnitude sum, not a realization ratio or requested feasibility test.",
            "cost": "Original cost.json retains collector generation-time scheduler snapshot. Final scheduler accounting, if supplied, is a separate receipt.",
        },
        "records": realization,
        "source_receipts": dict(sorted(source.receipts.items())),
        "raw_local_KEEP": True, "new_GPU_model_evaluator_calls": 0,
        "broad_raw_rehash": False, "checkpoint_saved": False,
    }
    outputs = {
        "realization-summary.json": (json.dumps(summary, ensure_ascii=False, sort_keys=True,
                                                separators=(",", ":"), allow_nan=False) + "\n").encode(),
        "realization-summary.csv": csv_bytes(realization_csv(realization)),
        "shares-summary.csv": csv_bytes(shares),
        "writer-summary.json": encoded({"source_commit": lock["source_commit"], "branches": writers}),
        "writer-summary.csv": csv_bytes([row for branch in writers for row in branch["layers"]]),
        "fit-summary.json": encoded(fit_summary),
    }
    for filename, data in outputs.items():
        create_or_exact_reuse(output / filename, data, replace_generated)
    for branch in BRANCHES:
        pooled = next(row for row in realization if row["branch"] == branch and
                      row["scope"] == "native_mean" and row["channel"] == "actual" and row["layer"] == "ALL")
        require(pooled["rows"] == 500 and pooled["zero_target_count"] == 96 and
                pooled["fields"]["directional_ratio"]["count"] == 404,
                "pooled native mean zero/defined count mismatch")
    files = []
    for filename in sorted(set(copies.values()) | set(outputs)):
        data = (output / filename).read_bytes()
        files.append(dict(path=str(output / filename), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return dict(status="COMPACT_STORED_RESULT_PUBLICATION_READY", source_commit=lock["source_commit"],
                output=str(output), files=files, artifact_count=len(files),
                realization_records=len(realization), writer_layer_records=25,
                source_receipts=len(source.receipts), self_test="PASS", assertions="PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replace-generated", action="store_true",
                        help="Replace this script's six generated outputs only; never replace exact collector copies or raw inputs")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
        print(json.dumps({"self_test": "PASS", "GPU_model_evaluator_calls": 0}))
        return
    require(arguments.attempt is not None and arguments.output is not None,
            "--attempt and --output required")
    print(json.dumps(publish(arguments.attempt, arguments.output, arguments.replace_generated),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
