"""Afterany, CPU-only independent raw reducer for the sealed two-arm task.

No model/native/evaluator imports, Slurm writes, polling, or incomplete-file
repair. Scheduler terminal and scientific completeness are separate facts.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import unicodedata


INSTRUCTION = "ODEEDIT-USER-GH-SH4-JLZ-TWOARM-BS100X20-20261002-R1"
WIDTH = {"R": 1, "P": 2, "N": 10}
TAG_KIND = {"RS": "R", "PS": "P", "NS": "N"}
REPORT_SUFFIX = Path("experiment-reports/servers/server4/jlz-twoarm-bs100x20-20261002-v1")
COMPLETE_STATUSES = {
    "COMPLETED", "COMPLETED_WITH_NUMERICAL_WARNINGS", "PILOT_COMMITTED_EVALUATED",
    "TWENTY_BATCHES_COMMITTED_EVALUATED", "TWO_BATCHES_COMMITTED_EVALUATED",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value, *, ascii=False):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=ascii, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "REFUSE_OVERWRITE: " + str(path))
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.rename(path)


def csv_table(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row):
    return row["kind"], int(row["case_id"]), int(row["prompt_index"])


def extract_rows(document):
    """Accept scalar JLZ rows and original native RS/PS/NS scalar rows only."""
    if isinstance(document, dict) and isinstance(document.get("rows"), list):
        return [dict(r, identity_convention="JLZ_UTF8_KIND") for r in document["rows"]]
    require(isinstance(document, dict) and isinstance(document.get("metrics"), dict),
            "UNSUPPORTED_OBSERVATION_SCHEMA")
    rows = []
    for tag, kind in TAG_KIND.items():
        if tag in document["metrics"]:
            rows.extend(dict(r, kind=kind, identity_convention="NATIVE_ASCII")
                        for r in document["metrics"][tag]["rows"])
    return rows


def expected_identity(record, kind, index, convention):
    rw = record["requested_rewrite"]
    prompts = ([rw["prompt"].format(rw["subject"])] if kind == "R" else
               record["paraphrase_prompts"] if kind == "P" else record["neighborhood_prompts"])
    require(len(prompts) == WIDTH[kind], "DATASET_PROMPT_CARDINALITY")
    base = [int(record["case_id"]), index, prompts[index],
            rw["target_new"]["str"], rw["target_true"]["str"]]
    if convention == "JLZ_UTF8_KIND":
        base.insert(1, kind)
    return digest(base, ascii=convention == "NATIVE_ASCII")


def is_success(row):
    return row["true_nll"] < row["new_nll"] if row["kind"] == "N" else row["new_nll"] < row["true_nll"]


def validate_rows(rows, records_by_id, expected_ids_by_kind):
    """Validate order, raw denominators, prompt/target identity and finite values."""
    require(bool(rows), "EMPTY_OBSERVATION")
    require(len({row_key(r) for r in rows}) == len(rows), "DUPLICATE_OBSERVATION_ROW")
    for kind in WIDTH:
        group = [r for r in rows if r["kind"] == kind]
        wanted = [(int(cid), i) for cid in expected_ids_by_kind.get(kind, []) for i in range(WIDTH[kind])]
        require([(int(r["case_id"]), int(r["prompt_index"])) for r in group] == wanted,
                "CASE_PROMPT_ORDER_DENOMINATOR: " + kind)
    active, versions = {}, {}
    for case in expected_ids_by_kind.get("R", []):
        rw = records_by_id[case]["requested_rewrite"]
        claim = (unicodedata.normalize("NFC", " ".join(rw["subject"].split())), rw.get("relation_id"))
        target = rw["target_new"].get("id", rw["target_new"]["str"])
        active[case] = True
        for old_case, old_target in versions.get(claim, []):
            if old_target != target:
                active[old_case] = False
        versions.setdefault(claim, []).append((case, target))
    for row in rows:
        require(row["kind"] in WIDTH, "UNKNOWN_METRIC_KIND")
        record = records_by_id[int(row["case_id"])]
        require(row["identity"] == expected_identity(record, row["kind"], int(row["prompt_index"]),
                                                      row.get("identity_convention", "JLZ_UTF8_KIND")),
                "PROMPT_TARGET_IDENTITY")
        for name in ("new_nll", "true_nll"):
            require(isinstance(row[name], (int, float)) and not isinstance(row[name], bool)
                    and math.isfinite(row[name]), "NONFINITE_NLL")
        if "margin" in row:
            require(math.isfinite(row["margin"]) and
                    math.isclose(row["margin"], row["true_nll"] - row["new_nll"], abs_tol=1e-12),
                    "MARGIN_ARITHMETIC")
        if "success" in row:
            require(bool(row["success"]) == is_success(row), "SUCCESS_STRICT_TIE_FAILURE")
        if "active_at_endpoint" in row:
            require(bool(row["active_at_endpoint"]) == active[int(row["case_id"])],
                    "ACTIVE_SUPERSEDED_IDENTITY")
        for target in ("new", "true"):
            present = [target + k in row for k in ("_token_correct", "_token_count", "_strict")]
            require(all(present) or not any(present), "PARTIAL_TF_FIELDS")
            if all(present):
                correct, count = row[target + "_token_correct"], row[target + "_token_count"]
                require(type(correct) is int and type(count) is int and 0 <= correct <= count and count > 0,
                        "INVALID_TF_COUNTS")
                require(bool(row[target + "_strict"]) == (correct == count), "STRICT_TOKEN_MISMATCH")
    return True


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def reduce_rows(rows):
    """Independent arithmetic; never trust stored aggregate success or means."""
    output = {}
    for kind in WIDTH:
        group = [r for r in rows if r["kind"] == kind]
        if not group:
            continue
        desired = "true" if kind == "N" else "new"
        count = len(group)
        desired_nll = [r[desired + "_nll"] for r in group]
        summary = dict(
            numerator=sum(is_success(r) for r in group), denominator=count,
            ties=sum(r["true_nll"] == r["new_nll"] for r in group), desired=desired,
            true_nll=statistics.fmean(r["true_nll"] for r in group),
            new_nll=statistics.fmean(r["new_nll"] for r in group),
            desired_nll=statistics.fmean(desired_nll),
            margin_true_minus_new=statistics.fmean(r["true_nll"] - r["new_nll"] for r in group),
            desired_nll_q95=quantile(desired_nll, .95), desired_nll_q99=quantile(desired_nll, .99),
            desired_nll_max=max(desired_nll),
        )
        summary["preference_rate"] = summary["numerator"] / count
        if all(desired + "_token_count" in r for r in group):
            tokens = sum(r[desired + "_token_count"] for r in group)
            correct = sum(r[desired + "_token_correct"] for r in group)
            strict = sum(bool(r[desired + "_strict"]) for r in group)
            summary.update(tf_status="RECORDED", token_correct=correct, token_count=tokens,
                           token_micro=correct / tokens,
                           prompt_macro=statistics.fmean(r[desired + "_token_correct"] /
                                                        r[desired + "_token_count"] for r in group),
                           strict_numerator=strict, strict_denominator=count, strict_rate=strict / count)
        else:
            summary.update(tf_status="NOT_RECORDED", token_correct=None, token_count=None,
                           token_micro=None, prompt_macro=None, strict_numerator=None,
                           strict_denominator=None, strict_rate=None)
        output[kind] = summary
    return output


def paired(before, after, comparison, *, qualification="SAME_RAW_IDENTITY"):
    """Before may be a fixed subset; never silently intersect missing rows."""
    left, right = {row_key(r): r for r in before}, {row_key(r): r for r in after}
    require(len(left) == len(before) and len(right) == len(after), "PAIRED_DUPLICATE")
    require(set(left) <= set(right), "PAIRED_MISSING_ROWS")
    summaries, identities = [], []
    for kind in WIDTH:
        for split in ("all", "active", "superseded"):
            if split != "all" and not all("active_at_endpoint" in right[k] for k in left if k[0] == kind):
                continue
            keys = [k for k in left if k[0] == kind and
                    (split == "all" or bool(right[k]["active_at_endpoint"]) == (split == "active"))]
            if not keys:
                continue
            for key in keys:
                for target in ("new", "true"):
                    if target + "_token_count" in left[key] and target + "_token_count" in right[key]:
                        require(left[key][target + "_token_count"] == right[key][target + "_token_count"],
                                "PAIRED_TOKEN_COUNT_IDENTITY")
            lost = [k for k in keys if is_success(left[k]) and not is_success(right[k])]
            gained = [k for k in keys if not is_success(left[k]) and is_success(right[k])]
            before_success = sum(is_success(left[k]) for k in keys)
            row = dict(comparison=comparison, kind=kind, split=split, qualification=qualification,
                       denominator=len(keys), before_success=before_success,
                       after_success=sum(is_success(right[k]) for k in keys),
                       lost=len(lost), gained=len(gained), retained=before_success - len(lost),
                       retention_if_before_success=(before_success - len(lost)) / before_success if before_success else None,
                       new_nll_delta=statistics.fmean(right[k]["new_nll"] - left[k]["new_nll"] for k in keys),
                       true_nll_delta=statistics.fmean(right[k]["true_nll"] - left[k]["true_nll"] for k in keys))
            desired = "true" if kind == "N" else "new"
            if all(desired + "_strict" in left[k] and desired + "_strict" in right[k] for k in keys):
                row.update(tf_strict_lost=sum(bool(left[k][desired + "_strict"]) and
                                             not bool(right[k][desired + "_strict"]) for k in keys),
                           tf_strict_gained=sum(not bool(left[k][desired + "_strict"]) and
                                               bool(right[k][desired + "_strict"]) for k in keys))
            summaries.append(row)
            identities.append(dict(comparison=comparison, kind=kind, split=split,
                                   lost_ids=[list(k) for k in lost], gained_ids=[list(k) for k in gained]))
    return summaries, identities


def validate_state(state):
    require(set(state) == {"W", "H"}, "W_H_STATE_SCHEMA")
    for field in ("W", "H"):
        require(set(state[field]) == set(map(str, range(4, 9))), "ALL_FIVE_STATE_LAYERS")
        require(all(isinstance(h, str) and re.fullmatch("[0-9a-f]{64}", h) for h in state[field].values()),
                "STATE_SHA_FORMAT")


def validate_ledger(ledger, case_ids, *, stage, schedule=None):
    size, batches, cap = (4, 2, 32) if stage == "pilot" else (100, 20, 120)
    require(isinstance(ledger, list) and len(ledger) <= batches, "LEDGER_LENGTH")
    previous = None
    calls = history = 0
    warnings = []
    for index, row in enumerate(ledger):
        require(row["batch"] == index + 1 and row["case_ids"] == case_ids[index * size:(index + 1) * size],
                "BATCH_REQUEST_ORDER")
        require(row["commit"] is True and row["history_appends"] == 5, "COMMIT_HISTORY_CARDINALITY")
        validate_state(row["entry"])
        validate_state(row["post"])
        require(previous is None or previous == row["entry"], "W_H_CHAIN_CONTINUITY")
        previous = row["post"]
        solver = row["solver"]
        count = solver["calls"]
        require(type(count) is int and 2 <= count <= cap, "SCIENCE_ORACLE_CAP")
        budget = solver.get("budget", solver.get("account"))
        require(isinstance(budget, dict) and budget["cap"] == cap and budget["actual_calls"] == count,
                "SHARED_BUDGET_IDENTITY")
        events = budget["events"]
        require(len(events) == count and events[0] == "initial" and events[-1] == "final"
                and events.count("final") == 1 and budget["final_reserved"] and budget["final_started"],
                "FINAL_RESERVE_AND_CALL_LEDGER")
        require(solver.get("final_recomputed") is True and solver.get("commit_eligible") is True,
                "FRESH_FINAL_REQUIRED")
        require(solver["status"] in {"CONVERGED", "POLICY_ZERO_STEP", "BUDGET_STOP",
                "STALLED_AT_PRECISION", "NOT_CONVERGED", "LINESEARCH_FAILED"},
                "INVALID_COMMITTED_SOLVER_STATUS")
        if solver["status"] != "CONVERGED":
            warnings.append(dict(batch=index + 1, type="FINITE_SOLVER_STATUS", status=solver["status"],
                                 action="RECORD_ONLY_USER_DIRECTED"))
        if schedule is not None:
            planned = schedule[index]
            refs = row["references"]
            for field, alias in (("current_ids", "ids"), ("general_ids", "general"), ("replay_ids", "replay")):
                require(refs[field] == planned.get(field, planned.get(alias)), "REFERENCE_MEMBERSHIP: " + field)
            require(len(refs["general_ids"]) == 16 and len(refs["replay_ids"]) <= 16,
                    "REFERENCE_CARDINALITY")
            require(not set(refs["replay_ids"]) & set(row["case_ids"]), "CURRENT_IN_REPLAY")
        calls += count
        history += row["history_appends"]
    return dict(commits=len(ledger), calls=calls, history_appends=history,
                chain_links=max(0, len(ledger) - 1), expected_commits=batches,
                reference_membership="VERIFIED" if schedule is not None else "NOT_RECORDED",
                warnings=warnings)


def accounting_snapshot(job_ids, run_command=subprocess.run):
    """Exactly one read-only accounting command for the supplied physical IDs."""
    require(job_ids and len(set(job_ids)) == len(job_ids)
            and all(re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job) for job in job_ids), "EXACT_JOB_ALLOWLIST")
    fields = "JobIDRaw,JobName,User,State,ExitCode,ElapsedRaw,AllocTRES,Start,End,NodeList"
    argv = ["sacct", "-X", "-j", ",".join(job_ids), "--parsable2", "--noheader", "--format=" + fields]
    at = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        result = run_command(argv, capture_output=True, text=True, check=False)
    except OSError as error:
        return dict(status="ACCOUNTING_NOT_AVAILABLE", observed_at=at, exact_jobs=job_ids,
                    argv=argv, error=str(error), parents=[], allocated_gpu_seconds=None)
    rows = []
    for line in result.stdout.splitlines():
        values = line.split("|")
        if not values or values[0] not in job_ids:
            continue  # Never add batch/extern steps or unrelated IDs.
        row = dict(zip(fields.split(","), values))
        require(row["JobIDRaw"] not in {r["JobIDRaw"] for r in rows}, "DUPLICATE_PARENT_ACCOUNTING")
        match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", row.get("AllocTRES", ""))
        allocated = row.get("AllocTRES", "")
        gpu = int(match.group(1)) if match else (0 if allocated and "gres/gpu" not in allocated else None)
        seconds = int(row["ElapsedRaw"]) if row.get("ElapsedRaw", "").isdigit() else None
        row.update(allocated_gpus=gpu, allocated_gpu_seconds=gpu * seconds if gpu is not None and seconds is not None else None)
        rows.append(row)
    return dict(status="SNAPSHOT" if result.returncode == 0 else "ACCOUNTING_ERROR",
                observed_at=at, exact_jobs=job_ids, argv=argv, returncode=result.returncode,
                stderr=result.stderr, parents=rows, missing_jobs=sorted(set(job_ids) - {r["JobIDRaw"] for r in rows}),
                allocated_gpu_seconds=sum(r["allocated_gpu_seconds"] or 0 for r in rows),
                allocated_gpu_seconds_complete=all(r["allocated_gpu_seconds"] is not None for r in rows)
                and len(rows) == len(job_ids), allocation_is_not_utilization=True,
                parent_only=True, repeated_queries=0)


def _stream_path(config):
    value = config.get("stream_path", config.get("stream"))
    if value is None:
        value = config.get("assets", {}).get("stream", {})
    return Path(value["path"] if isinstance(value, dict) else value)


def _load_observation(path, records, ids_by_kind):
    obj = read(path)
    rows = extract_rows(obj)
    validate_rows(rows, records, ids_by_kind)
    return obj, rows


def _stat_member(path, run):
    before = path.stat()
    value = sha(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), "INPUT_CHANGED_DURING_HASH")
    try:
        name = path.relative_to(run).as_posix()
    except ValueError:
        name = str(path)
    return dict(path=name, bytes=after.st_size, sha256=value)


def collect(run, config_path, scheduler):
    run, config_path = Path(run), Path(config_path)
    dest = run / "report"
    dest.mkdir(parents=True, exist_ok=False)
    config = read(config_path)
    record_list = read(_stream_path(config))
    records = {int(r["case_id"]): r for r in record_list}
    case_ids = [int(r["case_id"]) for r in record_list[:2000]]
    require(len(case_ids) == 2000 and len(records) == len(record_list), "FIXED_FIRST2000_ORDER")
    if config.get("case_ids") is not None:
        require(config["case_ids"] == case_ids, "CONFIG_CASE_ORDER")
    expected_stream_sha = config.get("stream_sha256", config.get("contract", {}).get("references", {}).get("stream_sha256"))
    if expected_stream_sha is not None:
        require(sha(_stream_path(config)) == expected_stream_sha, "DATASET_BYTES")
    report = dict(instruction_id=INSTRUCTION, status="NOT_COMPLETE", scheduler=scheduler,
                  source_commit=config.get("source_commit", "NOT_RECORDED"),
                  config_sha256=sha(config_path), analysis_sha256=sha(__file__),
                  checkpoint_saved=False, exact_resume="NOT_AVAILABLE",
                  no_new_model_or_gpu_calls=True, independent_reducer=True,
                  independent_agent_postrun_audit="NOT_PERFORMED_BY_COLLECTOR",
                  numerical_fidelity_policy="RECORD_ONLY_USER_DIRECTED",
                  numerical_certification="NOT_ESTABLISHED", science_interpretation="GH_OWNED",
                  artifact_broadcast="NO_BROADCAST_NOT_REQUIRED", errors=[], chains={}, baselines=[])
    prep_path = run / "prep/receipt.json"
    route_path = run / "prep/route.json"
    try:
        prep, route = read(prep_path), read(route_path)
        require(prep["status"] == "STRUCTURAL_COMPLETION" and prep["W0_restored"] is True,
                "PREP_STRUCTURAL_COMPLETION")
        require(prep["config_sha256"] == report["config_sha256"] == route["config_sha256"],
                "PREP_ROUTE_CONFIG_IDENTITY")
        require(0 <= route["extra_whole_batch_calls"] <= 6, "SHARED_B100_TECHNICAL_BUDGET")
        report["prep"] = dict(status=prep["status"], route=route["route"],
                              technical_calls=route["extra_whole_batch_calls"],
                              seconds=prep.get("seconds"), baseline_pilots=prep.get("baseline_pilots", "NOT_RECORDED"),
                              numerical_certification="NOT_ESTABLISHED")
    except Exception as error:
        report["errors"].append(dict(scope="prep", type=type(error).__name__, error=str(error)))
    tables, final_tables, transition_rows, transition_ids, costs, layers, fits = [], [], [], [], [], [], []
    all_final, w0 = {}, None
    shared_w0 = run / "prep/W00-observations.json"
    try:
        _, w0 = _load_observation(shared_w0, records, {k: case_ids for k in WIDTH})
        for kind, value in reduce_rows(w0).items():
            tables.append(dict(method="W0", stage="shared", batch=0, panel="first2000", kind=kind, **value))
    except Exception as error:
        report["errors"].append(dict(scope="W0", type=type(error).__name__, error=str(error)))
    for stage, size, batches in (("pilot", 4, 2), ("main", 100, 20)):
        for arm in ("A", "B"):
            name, folder = stage + "-" + arm, run / (stage + "-" + arm)
            chain = dict(status="NOT_COMPLETE", commits=0, expected_commits=batches,
                         failed_attempts=[], failed_calls=0, failed_calls_unrecorded=0)
            report["chains"][name] = chain
            for path in sorted(folder.glob("B*-failure.json")):
                failure = read(path)
                candidate = failure.get("candidate", {})
                count = failure.get("attempted_oracles")
                if count is None:
                    count = candidate.get("solver", {}).get("calls")
                if count is None:
                    count = (failure.get("solver_evidence") or {}).get("budget", {}).get("actual_calls")
                if count is None:
                    chain["failed_calls_unrecorded"] += 1
                else:
                    require(type(count) is int and count >= 0, "FAILED_ORACLE_COUNT")
                    chain["failed_calls"] += count
                chain["failed_attempts"].append(dict(path=str(path), sha256=sha(path),
                    error=failure.get("error", "NOT_RECORDED"),
                    rollback_verified=failure.get("rollback_verified", "NOT_RECORDED"),
                    attempted_oracles=count, seconds=failure.get("seconds")))
            try:
                ledger = read(folder / "ledger.json") if (folder / "ledger.json").exists() else []
                terminal = read(folder / "terminal.json") if (folder / "terminal.json").exists() else {"status": "MISSING"}
                chain.update(validate_ledger(ledger, case_ids, stage=stage,
                                             schedule=config.get("schedules", {}).get(stage)))
                chain["terminal"] = terminal
                if ledger:
                    runtime = read(folder / "runtime.json")
                    require(runtime.get("fresh_W0") is True and runtime.get("W0_H0") == ledger[0]["entry"],
                            "FRESH_CHAIN_W0_ENTRY")
                    require(runtime.get("config_sha256") == report["config_sha256"], "RUNTIME_CONFIG_BINDING")
                    if (run / "execution.lock.json").exists():
                        require(runtime.get("execution_lock_sha256") == sha(run / "execution.lock.json"),
                                "RUNTIME_SOURCE_LOCK_BINDING")
                    reuse = read(folder / "W00-reuse.json")
                    require(reuse["same_W0"] == ledger[0]["entry"]
                            and reuse["member"]["sha256"] == sha(shared_w0), "W0_REUSE_IDENTITY")
                    if w0 is not None:
                        require(read(shared_w0)["state"] == ledger[0]["entry"], "SHARED_W0_STATE_IDENTITY")
                    chain["runtime"] = {k: runtime.get(k, "NOT_RECORDED") for k in
                                        ("source_commit", "config_sha256", "execution_lock_sha256",
                                         "torch", "transformers", "gpu", "job_id")}
                if "commits" in terminal:
                    require(terminal["commits"] == len(ledger), "TERMINAL_COMMIT_COUNT")
                observed, atwrite = {}, []
                for item in ledger:
                    batch = item["batch"]
                    receipt_path = folder / f"B{batch:03d}-committed.json"
                    require(receipt_path.exists(), "MISSING_ATOMIC_COMMIT_RECEIPT")
                    receipt = read(receipt_path)
                    require(receipt == item, "COMMIT_LEDGER_DISAGREEMENT")
                    require(not (folder / f"B{batch:03d}-failure.json").exists(), "FAILURE_WITH_COMMIT")
                    require(item.get("checkpoint_saved") is False, "COMMIT_NO_CHECKPOINT_POLICY")
                    require(item.get("commit_gate", {}).get("weight_bitwise") is True,
                            "FINAL_MATERIALIZATION_COMMIT_IDENTITY")
                    require(item.get("config_sha256") == report["config_sha256"], "COMMIT_CONFIG_BINDING")
                    require([r["layer"] for r in item.get("layer_allocation", [])] == list(range(4, 9)),
                            "ALL_FIVE_LAYER_DIAGNOSTICS")
                    for layer in item.get("layer_allocation", []):
                        layers.append(dict(stage=stage, arm=arm, batch=batch, **layer))
                    for label, diagnostic in item.get("commit_gate", {}).items():
                        if isinstance(diagnostic, dict) and diagnostic.get("original_verdict") == "FAIL":
                            chain["warnings"].append(dict(batch=batch, type="FINITE_NUMERICAL_DIFFERENCE",
                                                          diagnostic=label, original_verdict="FAIL",
                                                          action="RECORD_ONLY_USER_DIRECTED"))
                    fit = {"stage": stage, "arm": arm, "batch": batch,
                           "burden_eta": 0 if arm == "A" else 1}
                    for key in ("nll", "kl", "general", "replay", "omega"):
                        values = item.get("fit", {}).get(key)
                        if isinstance(values, list) and values and all(isinstance(x, dict) for x in values):
                            require(key in ("general", "replay") and
                                    [x["case_id"] for x in values] == item["references"][key + "_ids"],
                                    "FINAL_REFERENCE_LOSS_MEMBERSHIP")
                            values = [x["value"] for x in values]
                        if isinstance(values, (int, float)):
                            require(math.isfinite(values), "NONFINITE_FINAL_FIT")
                            fit[key] = values
                        elif isinstance(values, list) and all(isinstance(x, (int, float)) for x in values):
                            require(all(math.isfinite(x) for x in values), "NONFINITE_FINAL_FIT")
                            fit[key + "_count"] = len(values)
                            fit[key + "_sum"] = sum(values)
                            fit[key + "_mean"] = statistics.fmean(values) if values else None
                        else:
                            fit[key] = "NOT_RECORDED"
                    fits.append(fit)
                    path = folder / f"W{batch:02d}-observations.json"
                    seen = case_ids[:batch * size]
                    current = case_ids[(batch - 1) * size:batch * size]
                    expected = {"R": seen, "P": seen,
                                "N": seen if batch in (5, 10, 20) else current}
                    obs, rows = _load_observation(path, records, expected)
                    require(obs.get("state") == item["post"], "OBSERVER_ENDPOINT_STATE")
                    require(obs.get("no_mutation") is True and obs.get("optimizer_feedback") is False,
                            "OBSERVER_MUTATION_OR_FEEDBACK")
                    if item.get("observation", {}).get("sha256"):
                        require(sha(path) == item["observation"]["sha256"], "OBSERVER_RECEIPT_SHA")
                    observed[batch] = rows
                    current_rows = [r for r in rows if r["case_id"] in set(current)]
                    atwrite.extend(current_rows)
                    for panel, selected in (("allseen_RP_current_or_milestone_N", rows), ("current", current_rows)):
                        for kind, summary in reduce_rows(selected).items():
                            tables.append(dict(method="JLZ_" + arm, stage=stage, batch=batch,
                                               panel=panel, kind=kind, **summary))
                    costs.append(dict(stage=stage, method="JLZ_" + arm, batch=batch,
                                      calls=item["solver"]["calls"], solver_status=item["solver"]["status"],
                                      history_appends=item["history_appends"],
                                      batch_seconds=item.get("seconds"), component_timers="NOT_SEPARATED"))
                if len(ledger) == batches and terminal.get("status") in COMPLETE_STATUSES:
                    chain["status"] = "COMPLETE_CPU_VALIDATED"
                    final = observed[batches]
                    # Pilot W02 records current N only; allseen N is not invented.
                    final_keys = {row_key(a) for a in final}
                    common_atwrite = [r for r in atwrite if row_key(r) in final_keys]
                    summaries, ids = paired(common_atwrite, final, name + ":atwrite_to_final")
                    transition_rows.extend(summaries)
                    transition_ids.extend(ids)
                    if stage == "main":
                        require(len(final) == 26000, "W20_FIRST2000_RPN_DENOMINATOR")
                        all_final["JLZ_" + arm] = final
                        for kind, summary in reduce_rows(final).items():
                            final_tables.append(dict(method="JLZ_" + arm, endpoint="W20", mode="NEW_S4",
                                                     comparison_scope="SAME_FIRST2000", kind=kind, **summary))
                        for label, before in (("W10_samefirst1000_to_W20", observed[10]),
                                              ("W0_to_W20", w0)):
                            if before is not None:
                                summaries, ids = paired(before, final, name + ":" + label)
                                transition_rows.extend(summaries)
                                transition_ids.extend(ids)
                else:
                    chain["status"] = "NOT_COMPLETE"
            except Exception as error:
                chain.update(status="VALIDATION_FAILED", error_type=type(error).__name__, error=str(error))
                report["errors"].append(dict(scope=name, type=type(error).__name__, error=str(error)))
    for baseline in config.get("baselines", []):
        item = dict(name=baseline["name"], mode=baseline.get("mode", "NOT_RECORDED"), status="NOT_COMPLETE",
                    metadata_receipts=baseline.get("metadata_receipts", baseline.get("receipt", [])))
        report["baselines"].append(item)
        try:
            path = Path(baseline["raw_path"])
            obj, rows = _load_observation(path, records, {k: case_ids for k in WIDTH})
            raw_sha = sha(path)
            original_receipt = baseline.get("receipt", {})
            expected_sha = baseline.get("sha256", original_receipt.get("sha256"))
            if expected_sha:
                require(raw_sha == expected_sha, "BASELINE_RAW_SHA")
            if original_receipt.get("bytes") is not None:
                require(path.stat().st_size == original_receipt["bytes"], "BASELINE_RAW_SIZE")
            item.update(status="RAW_CPU_VALIDATED", raw_path=str(path), raw_sha256=raw_sha,
                        raw_bytes=path.stat().st_size,
                        endpoint=baseline.get("endpoint", "W20"),
                        endpoint_provenance="BOUND_BY_CONFIG_METADATA_RECEIPTS",
                        hardware_comparison="CROSS_HOST_QUALITY_NOT_MATCHED_SPEED")
            require(item["endpoint"] in (20, "W20"), "BASELINE_ENDPOINT_NOT_W20")
            commit_path = baseline.get("commit_receipt_path", baseline.get("commit", {}).get("path"))
            if commit_path:
                commit = read(commit_path)
                require(commit.get("batch") == 20, "BASELINE_COMMIT_BATCH_NOT20")
                require(commit.get("seen_requests") == 2000, "BASELINE_COMMIT_NOT_FIRST2000")
                endpoint = commit.get("endpoint", commit.get("post"))
                if obj.get("state") is not None:
                    require(obj["state"] == endpoint, "BASELINE_ENDPOINT_STATE")
                    binding = "RAW_STATE_BOUND_TO_ORIGINAL_B020_COMMIT"
                elif isinstance(endpoint, dict) and "weights" in endpoint and "weight_state" in obj:
                    require(obj.get("weight_state") == endpoint["weights"], "BASELINE_ENDPOINT_WEIGHTS")
                    if "cache" in endpoint:
                        require(obj.get("cache_sha256") == endpoint["cache"], "BASELINE_ENDPOINT_HISTORY")
                    binding = "RAW_STATE_BOUND_TO_ORIGINAL_B020_COMMIT"
                elif baseline["name"] == "MEMIT-H" and obj.get("evaluation_type") == "ALL_SEEN_RPN":
                    # Original metrics.merge emits scalar rows but omits state.
                    # Preserve that limit; a hash/filename is not a new tensor
                    # verification. The immutable source creates this B020
                    # merge between its recorded endpoint/observer guards.
                    require(obj.get("requests") == 2000 and obj.get("current_rows_reused") is True
                            and commit.get("nonfinite") == 0 and commit.get("observer_mutation") == 0
                            and commit.get("observer_context_rng_ledger_unchanged") is True,
                            "MEMIT_H_SOURCE_COMMIT_OBSERVER_BINDING")
                    binding = "ORIGINAL_B020_PRODUCER_AND_COMMIT_BOUND_RAW_STATE_FIELD_NOT_RECORDED"
                else:
                    raise ValueError("BASELINE_COMMIT_ENDPOINT_SCHEMA")
                commit_sha = sha(commit_path)
                if baseline.get("commit", {}).get("sha256"):
                    require(commit_sha == baseline["commit"]["sha256"], "BASELINE_COMMIT_RECEIPT_SHA")
                item.update(endpoint_provenance=binding, commit_receipt_sha256=commit_sha)
            all_final[baseline["name"]] = rows
            for kind, summary in reduce_rows(rows).items():
                final_tables.append(dict(method=baseline["name"], endpoint="W20", mode=item["mode"],
                                         comparison_scope="SAME_FIRST2000_RUNTIME_DIFFERENCES_RECORDED", kind=kind, **summary))
            for arm in ("JLZ_A", "JLZ_B"):
                if arm in all_final:
                    summaries, ids = paired(rows, all_final[arm], baseline["name"] + "_to_" + arm,
                                           qualification="DATASET_PROMPT_TARGET_IDENTITY_VERIFIED_RUNTIME_DIFFERENCES_RECORDED")
                    transition_rows.extend(summaries)
                    transition_ids.extend(ids)
        except Exception as error:
            item.update(status="NOT_COMPLETE", error_type=type(error).__name__, error=str(error))
            report["errors"].append(dict(scope="baseline:" + item["name"], type=type(error).__name__, error=str(error)))
    if "JLZ_A" in all_final and "JLZ_B" in all_final:
        summaries, ids = paired(all_final["JLZ_A"], all_final["JLZ_B"], "JLZ_A_to_JLZ_B")
        transition_rows.extend(summaries)
        transition_ids.extend(ids)
    main_chains = [v for k, v in report["chains"].items() if k.startswith("main-")]
    pilot_chains = [v for k, v in report["chains"].items() if k.startswith("pilot-")]
    report["main_totals"] = {key: sum(v.get(key, 0) for v in main_chains)
                             for key in ("commits", "calls", "history_appends", "failed_calls", "failed_calls_unrecorded")}
    report["pilot_totals"] = {key: sum(v.get(key, 0) for v in pilot_chains)
                              for key in ("commits", "calls", "history_appends", "failed_calls", "failed_calls_unrecorded")}
    report["budget_contract"] = dict(main_commits=40, main_history_appends=200, main_oracle_ceiling=4800,
                                     pilot_commits=4, pilot_history_appends=20, pilot_oracle_ceiling=128,
                                     shared_B100_technical_ceiling=6, allocation_is_not_utilization=True)
    if report["main_totals"]["calls"] + report["main_totals"]["failed_calls"] > 4800 \
            or report["pilot_totals"]["calls"] + report["pilot_totals"]["failed_calls"] > 128:
        report["errors"].append(dict(scope="total_budget", error="SCIENCE_TOTAL_CAP"))
    require(all(p.get("calls", 0) <= 2400 for p in main_chains), "ARM_SCIENCE_CAP")
    if not report["errors"] and all(c["status"] == "COMPLETE_CPU_VALIDATED" for c in report["chains"].values()) \
            and all(b["status"] == "RAW_CPU_VALIDATED" for b in report["baselines"]):
        report["status"] = "COMPLETED_WITH_RECORDED_NUMERICAL_STATUS"
    report["baseline_scope_complete"] = {b["name"] for b in report["baselines"] if b["status"] == "RAW_CPU_VALIDATED"} \
        == {"MEMIT-H", "BASE_MEMIT", "BASE_ALPHAEDIT"}
    if not report["baseline_scope_complete"]:
        report["status"] = "NOT_COMPLETE"
    report["no_checkpoint_audit"] = "NO_NEW_RESUME_TENSOR_FOUND_IN_SCOPED_OUTPUT"
    inventory = []
    for subdir in ("prep", "pilot-A", "pilot-B", "main-A", "main-B"):
        root = run / subdir
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                continue  # Shared inputs are never followed, rehashed or copied.
            if path.is_file():
                inventory.append(_stat_member(path, run))
                if path.suffix in (".pt", ".pth", ".safetensors", ".npy", ".npz") and subdir != "prep":
                    report["no_checkpoint_audit"] = "UNEXPECTED_PERSISTENT_TENSOR_REQUIRES_REVIEW"
                    report["status"] = "NOT_COMPLETE"
    inventory.append(_stat_member(config_path, run))
    report["artifact_inventory"] = dict(members=len(inventory), bytes=sum(x["bytes"] for x in inventory),
                                         member_root_sha256=digest(inventory), no_input_model_or_checkpoint_rehash=True)
    atomic_json(run / "paired-identities.json", transition_ids)
    csv_table(dest / "endpoint-metrics.csv", tables)
    csv_table(dest / "final-comparison.csv", final_tables)
    csv_table(dest / "paired-transitions.csv", transition_rows)
    csv_table(dest / "compute.csv", costs)
    csv_table(dest / "layer-allocation.csv", layers)
    csv_table(dest / "fit-reference-losses.csv", fits)
    csv_table(dest / "parent-accounting.csv", scheduler.get("parents", []))
    atomic_json(dest / "artifact-manifest.json", inventory)
    atomic_json(dest / "collection.json", report)
    _write_report(dest / "report-ko.md", report, final_tables)
    from .plot import generate
    generate(dest)
    package = [_stat_member(p, dest) for p in sorted(dest.iterdir()) if p.is_file()]
    atomic_json(dest / "package-manifest.json", dict(members=package, root_sha256=digest(package), raw_free=True))
    target = config.get("report_path")
    if target and Path(target).resolve() != dest.resolve():
        target = Path(target)
        require(target.is_absolute() and tuple(target.parts[-len(REPORT_SUFFIX.parts):]) == REPORT_SUFFIX.parts,
                "REPORT_PATH_OUTSIDE_TASK_SCOPE")
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(dest.iterdir()):
            destination = target / path.name
            require(not destination.exists(), "REPORT_PUBLICATION_ALREADY_EXISTS")
            with path.open("rb") as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)
    return report


def _write_report(path, report, rows):
    lines = ["# JLZ A/B BS100×20 자동 CPU 사실 보고", "",
             "상태: `" + report["status"] + "`. Scheduler 종료와 과학적 완결성은 별도입니다.", "",
             "## 최종 first2000 W20 비교", "",
             "| 방법 | 지표 | 성공/분모 | preference | TF token-micro | TF prompt-macro | TF strict | true NLL | new NLL |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(x):
        return "NOT_RECORDED" if x is None else f"{x:.6f}"
    for row in rows:
        lines.append("| " + " | ".join([row["method"], row["kind"],
                      f"{row['numerator']}/{row['denominator']}", fmt(row["preference_rate"]),
                      fmt(row["token_micro"]), fmt(row["prompt_macro"]), fmt(row["strict_rate"]),
                      fmt(row["true_nll"]), fmt(row["new_nll"])]) + " |")
    lines += ["", "R/P는 new NLL < true NLL, N은 true NLL < new NLL이며 tie는 실패입니다. "
              "TF 지표는 자유생성 정확도가 아닙니다. 과거 baseline의 다른 host wall을 matched speed로 비교하지 않습니다.",
              "", "## 실행·회계", "",
              f"Main commit {report['main_totals']['commits']}/40, history append {report['main_totals']['history_appends']}/200, "
              f"committed-prefix oracle {report['main_totals']['calls']}/4800, "
              f"failed-attempt oracle {report['main_totals']['failed_calls']} "
              f"(미기록 실패 {report['main_totals']['failed_calls_unrecorded']}건). "
              f"Pilot committed/failed oracle {report['pilot_totals']['calls']}/{report['pilot_totals']['failed_calls']} (총 상한128).",
              "", "Oracle 수는 초기·수용/거절 trial·원본 재검사·fresh final을 포함합니다. "
              "Teacher/geometry/observer 비용은 oracle 분모와 별개이며 미분리 component 시간은 NOT_SEPARATED입니다.",
              "", "Parent allocated GPU-sec: " + str(report["scheduler"].get("allocated_gpu_seconds", "NOT_RECORDED")) +
              (" (COMPLETE)" if report["scheduler"].get("allocated_gpu_seconds_complete") else " (PARTIAL_OR_NOT_RECORDED)") +
              "; allocation은 utilization이 아닙니다. Batch timer와 parent allocation은 중첩 합산하지 않습니다.",
              "", "## 판정과 한계", "",
              "수치 근접성·비수렴·finite 품질 저하는 RECORD_ONLY_USER_DIRECTED이며 원 수치판정을 PASS로 바꾸지 않습니다. "
              "numerical_certification=NOT_ESTABLISHED. 누락/구조 오류는 완료로 표기하지 않습니다.",
              "", "checkpoint_saved=false; exact_resume=NOT_AVAILABLE. 과거 입력 CP를 새로 만들거나 삭제하지 않았습니다. "
              "Collector는 원 scalar row만 독립 산술 검산했으며 모델 재실행 및 별도 독립 agent 사후감사를 수행하지 않았습니다.",
              "", "[전체 곡선](endpoint-metrics.csv) · [최종 비교](final-comparison.csv) · "
              "[paired 전이](paired-transitions.csv) · [비용](compute.csv) · [회계](parent-accounting.csv) · "
              "[층별 배분](layer-allocation.csv) · [학습/reference 목적](fit-reference-losses.csv) · "
              "[검산 receipt](collection.json) · [artifact SHA](artifact-manifest.json)",
              "", "![관측 preference](quality.svg)", "", "![Reference KL](reference-kl.svg)",
              "", "그림은 원 CSV에서 `python -m project.run_scripts.jlz_two_arm.plot REPORT_DIR`로 재생성합니다. "
              "N 곡선은 allseen 관측5/10/20만 표시합니다. 독립 반복/인과효과로 해석하지 않습니다.",
              "", "Raw/prompt/teacher/tensor/fullstdout는 Git 보고 package에 포함하지 않습니다. "
              "NO_BROADCAST_NOT_REQUIRED. 과학적 종합 해석·방법 선택은 GH가 별도로 수행합니다."]
    if report["errors"]:
        lines += ["", "## 미완료·검산 오류", ""]
        lines += ["- `" + item["scope"] + "`: " + item["error"] for item in report["errors"]]
    with Path(path).open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--jobs", required=True, help="Exact comma-separated parent IDs; one sacct snapshot only")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    scheduler = accounting_snapshot(args.jobs.split(","))
    result = collect(args.run, args.config or args.run / "config.json", scheduler)
    # An afterany collection with preserved incomplete evidence is a completed
    # collection, not a claim that the science jobs passed.
    print(json.dumps({"collection_status": result["status"], "report": str(args.run / "report/report-ko.md")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
