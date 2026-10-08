"""One-shot independent allocation accounting for exact official manifest IDs.

This never discovers jobs, polls, mutates Slurm, loads a model or uses W&B. Parent
allocation rows alone determine time/CPU/GPU cost. Step rows may supply a peak
MaxRSS observation, but never add another allocation or duration. Query failures
and missing accounting fields stay unknown, not fabricated zero cost.
"""
from __future__ import annotations

import math
import re
import subprocess


SCHEMA = "official-server1-bounded-accounting-v1"
FIELDS = ("JobIDRaw", "State", "ElapsedRaw", "AllocCPUS", "AllocTRES", "TotalCPU", "MaxRSS")
STATES = frozenset(("PENDING", "RUNNING", "COMPLETING", "COMPLETED", "FAILED", "CANCELLED",
    "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REQUEUED",
    "REQUEUE_FED", "REVOKED", "SPECIAL_EXIT", "SUSPENDED", "CONFIGURING", "RESIZING", "STOPPED"))
TERMINAL = STATES - {"PENDING", "RUNNING", "COMPLETING", "SUSPENDED", "CONFIGURING", "RESIZING"}
UNKNOWN = frozenset(("", "Unknown", "UNKNOWN", "N/A", "None", "(null)"))
MAX_JOBS, MAX_ROWS, MAX_BYTES = 256, 4096, 4 << 20


class AccountingError(ValueError):
    """Public error codes only; never embeds scheduler stdout/stderr."""


def require(value, code):
    if not value:
        raise AccountingError(code)


def command(argv, *, timeout):
    """Exactly one bounded process; stderr is never copied into any receipt."""
    try:
        value = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        raise AccountingError("ACCOUNTING_QUERY_TIMEOUT") from error
    except OSError as error:
        raise AccountingError("ACCOUNTING_COMMAND_UNAVAILABLE") from error
    require(value.returncode == 0, "ACCOUNTING_QUERY_FAILED")
    return value.stdout


def _integer(value, code):
    if value in UNKNOWN:
        return None
    require(re.fullmatch(r"\d+", value) is not None, code)
    return int(value)


def _seconds(value):
    if value in UNKNOWN:
        return None
    match = re.fullmatch(r"(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", value)
    require(match is not None, "ACCOUNTING_TOTAL_CPU_CLOCK")
    days, hours, minutes, seconds = match.groups()
    seconds = float(seconds)
    require(int(minutes) < 60 and seconds < 60 and math.isfinite(seconds), "ACCOUNTING_TOTAL_CPU_CLOCK")
    return int(days or 0) * 86400 + int(hours or 0) * 3600 + int(minutes) * 60 + seconds


def _memory(value):
    if value in UNKNOWN:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGTPE]?)", value)
    require(match is not None, "ACCOUNTING_MEMORY_UNIT")
    number, unit = match.groups()
    # The sole query sets --units=K, so an unqualified value is KiB.
    result = float(number) * 1024 ** ("KMGTPE".index(unit or "K") + 1)
    require(math.isfinite(result) and result >= 0, "ACCOUNTING_MEMORY_FINITE")
    return int(result)


def _allocation(value):
    if value in UNKNOWN:
        return None, None
    result, gpu_types = {}, []
    for item in value.split(","):
        pieces = item.split("=")
        require(len(pieces) == 2, "ACCOUNTING_ALLOC_TRES_SCHEMA")
        key, count = pieces
        require(key not in result and (key in ("cpu", "node", "billing", "mem")
            or re.fullmatch(r"gres/gpu(?::[A-Za-z0-9_-]+)?", key) is not None),
            "ACCOUNTING_ALLOC_TRES_WHITELIST")
        if key == "mem":
            require(_memory(count) is not None, "ACCOUNTING_ALLOC_MEMORY_REQUIRED")
            result[key] = count
        else:
            parsed = _integer(count, "ACCOUNTING_ALLOC_TRES_COUNT")
            require(parsed is not None, "ACCOUNTING_ALLOC_TRES_COUNT")
            result[key] = parsed
            if key.startswith("gres/gpu:"):
                gpu_types.append(parsed)
    generic, typed = result.get("gres/gpu"), sum(gpu_types)
    require(generic is None or not gpu_types or generic == typed, "ACCOUNTING_GPU_GENERIC_TYPED_CONFLICT")
    # A present complete allocation TRES without a GPU genuinely has zero GPU
    # allocation. An absent TRES is UNKNOWN and must not be treated this way.
    return result, generic if generic is not None else typed


def _empty(key, job_id, reason):
    return dict(key=key, job_id=job_id, accounting_status="NOT_OBSERVED", state=None,
        elapsed_seconds=None, allocated_cpus=None, allocated_tres=None, allocated_GPU_count=None,
        total_CPU_seconds=None, max_RSS_bytes=None, max_RSS_source=None,
        allocation_GPU_seconds=None, allocation_CPU_seconds=None,
        terminal_accounting_observation=False, uncertainty=[reason])


def _parse(stdout, keys):
    require(type(stdout) is str and len(stdout.encode()) <= MAX_BYTES, "ACCOUNTING_OUTPUT_BOUND")
    lines = stdout.splitlines()
    require(len(lines) <= MAX_ROWS, "ACCOUNTING_ROW_BOUND")
    parents, step_peaks, step_ids = {}, {}, set()
    for line in lines:
        require(bool(line), "ACCOUNTING_EMPTY_ROW")
        cells = line.split("|")
        if len(cells) == len(FIELDS) + 1 and cells[-1] == "":
            cells.pop()
        require(len(cells) == len(FIELDS), "ACCOUNTING_ROW_SCHEMA")
        job, state, elapsed, cpus, tres, cpu, rss = cells
        match = re.fullmatch(r"(\d+)(?:\.([A-Za-z0-9_-]+))?", job)
        require(match is not None and match[1] in keys, "ACCOUNTING_UNREQUESTED_OR_INVALID_JOB")
        parent, step = match.groups()
        if step is not None:
            require(job not in step_ids, "ACCOUNTING_DUPLICATE_STEP")
            step_ids.add(job)
            memory = _memory(rss)
            if memory is not None:
                step_peaks.setdefault(parent, []).append(memory)
            # Do not add step ElapsedRaw/TotalCPU/AllocTRES to parent cost.
            continue
        require(parent not in parents, "ACCOUNTING_DUPLICATE_PARENT")
        normalized_state = state.split()[0].rstrip("+") if state not in UNKNOWN else None
        require(normalized_state is None or normalized_state in STATES, "ACCOUNTING_STATE_ENUM")
        elapsed, cpus, cpu, memory = (_integer(elapsed, "ACCOUNTING_ELAPSED_RAW_INTEGER"),
            _integer(cpus, "ACCOUNTING_ALLOC_CPUS_INTEGER"), _seconds(cpu), _memory(rss))
        allocation, gpu = _allocation(tres)
        require(allocation is None or cpus is None or "cpu" not in allocation or allocation["cpu"] == cpus,
                "ACCOUNTING_ALLOC_CPU_CONFLICT")
        row = _empty(keys[parent], parent, "")
        row.update(accounting_status="OBSERVED_PARENT_ALLOCATION", state=normalized_state,
            elapsed_seconds=elapsed, allocated_cpus=cpus, allocated_tres=allocation,
            allocated_GPU_count=gpu, total_CPU_seconds=cpu, max_RSS_bytes=memory,
            max_RSS_source="PARENT_ROW" if memory is not None else None,
            allocation_GPU_seconds=elapsed * gpu if elapsed is not None and gpu is not None else None,
            allocation_CPU_seconds=elapsed * cpus if elapsed is not None and cpus is not None else None,
            terminal_accounting_observation=normalized_state in TERMINAL,
            uncertainty=[])
        parents[parent] = row
    for parent, row in parents.items():
        if row["max_RSS_bytes"] is None and step_peaks.get(parent):
            row["max_RSS_bytes"] = max(step_peaks[parent])
            row["max_RSS_source"] = "MAX_REPORTED_STEP_TASK_PEAK_NOT_SUM"
        for field in ("state", "elapsed_seconds", "allocated_cpus", "allocated_tres", "total_CPU_seconds", "max_RSS_bytes"):
            if row[field] is None:
                row["uncertainty"].append("MISSING_" + field.upper())
        if not row["terminal_accounting_observation"]:
            row["uncertainty"].append("LIVE_OR_UNKNOWN_ACCOUNTING_NOT_FINAL")
    return [parents.get(job, _empty(key, job, "PARENT_ACCOUNTING_ROW_NOT_AVAILABLE")) for job, key in keys.items()]


def collect_costs(manifest, *, runner=None, timeout=10):
    """Return only whitelisted allocation metadata, retaining unknowns as None.

    The caller must first verify this exact manifest/source/execution-lock. This
    function independently rejects duplicate/unregistered IDs and profile drift.
    ``runner(argv, timeout=10)`` is the fake-exec seam; production calls sacct once.
    """
    require(type(timeout) in (int, float) and 0 < timeout <= 10, "ACCOUNTING_TIMEOUT_BOUND")
    require(type(manifest) is dict and manifest.get("schema") == "official-server1-job-manifest-v1",
            "ACCOUNTING_MANIFEST_SCHEMA")
    source = manifest.get("source")
    require(type(source) is dict and set(source) == {"main_commit", "official_tree"}
            and all(type(value) is str and re.fullmatch(r"[a-f0-9]{40}", value) for value in source.values())
            and type(manifest.get("plan_sha256")) is str
            and re.fullmatch(r"[a-f0-9]{64}", manifest["plan_sha256"]), "ACCOUNTING_SOURCE_PLAN_IDENTITY")
    jobs, profiles = manifest.get("jobs"), manifest.get("profiles")
    require(type(jobs) is dict and 0 < len(jobs) <= MAX_JOBS and type(profiles) is list
            and len(profiles) == len(jobs) and {row["key"] for row in profiles} == set(jobs),
            "ACCOUNTING_EXACT_MANIFEST_PROFILES")
    require(all(type(job) is str and re.fullmatch(r"\d+", job) for job in jobs.values())
            and len(set(jobs.values())) == len(jobs), "ACCOUNTING_EXACT_UNIQUE_PARENT_IDS")
    keys = {job: key for key, job in jobs.items()}
    argv = ["sacct", "--noheader", "--parsable2", "--units=K", "--jobs=" + ",".join(keys),
            "--format=JobIDRaw%40,State%40,ElapsedRaw%20,AllocCPUS%20,AllocTRES%4096,TotalCPU%40,MaxRSS%40"]
    error_code = None
    try:
        rows = _parse((runner or command)(argv, timeout=timeout), keys)
    except Exception as error:
        code = str(error)
        error_code = code if type(error) is AccountingError and re.fullmatch(r"[A-Z0-9_]{1,100}", code) \
            else "ACCOUNTING_QUERY_OR_PARSE_UNAVAILABLE"
        rows = [_empty(key, job, error_code) for job, key in keys.items()]
    sums = {}
    for field in ("allocation_GPU_seconds", "allocation_CPU_seconds", "total_CPU_seconds"):
        available = [row[field] for row in rows if row[field] is not None]
        sums["known_" + field] = math.fsum(available) if available else None
        sums[field + "_unknown_jobs"] = sum(row[field] is None for row in rows)
    observed = sum(row["accounting_status"] == "OBSERVED_PARENT_ALLOCATION" for row in rows)
    return dict(schema=SCHEMA, status="ACCOUNTING_UNAVAILABLE" if error_code else
                "ACCOUNTING_COMPLETE_ID_COVERAGE" if observed == len(rows) else "ACCOUNTING_PARTIAL_ID_COVERAGE",
        source=manifest.get("source"), plan_sha256=manifest.get("plan_sha256"), rows=rows,
        requested_parent_jobs=len(rows), observed_parent_jobs=observed, totals=sums,
        all_costs_final=all(row["terminal_accounting_observation"] and not row["uncertainty"] for row in rows),
        error_code=error_code, query_calls=1, query_timeout_seconds=timeout,
        parent_allocations_only_no_step_double_count=True, MaxRSS_is_task_peak_not_aggregate_memory=True,
        allocation_seconds_not_GPU_compute_seconds=True, scientific_success_not_inferred=True,
        missing_not_zero=True, GPU=0, model_forward_calls=0, automatic_retry=False)
