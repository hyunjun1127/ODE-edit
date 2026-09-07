"""Source-pinned S launch preparation and campaign accounting; no submission.

Only SH1's integration owner queries Slurm and executes held submission/release.
This module consumes the resulting exact evidence and checks it before model
loading. Prior null-budget packages and all failed attempts remain immutable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess

from .contracts import INSTRUCTION_ID, RUNTIME_HEAD, CONTRACT_SHA256
from .publication import digest, member, safe_path, write_once
from .resources import validate_sbatch_memory
from .sweep import dry_plan

BUDGET_SHA256 = "54789342ea78151442a09ec07696db9ce80caae6f39a665c2ca4568b881a3c15"
CAMPAIGN_GPU_SECONDS = 172800
CELL_WALL_SECONDS = 28800
JOB_NAME = "odeedit_alpha_jv_ds_sweep_s1"
PACKAGE = "project/run_scripts/alpha_jv_llama_diagnosis"
PYTHON = "/mnt/raid5/janghj/EasyEdit/.venv/bin/python"
LIVE_STATES = frozenset(("RUNNING", "COMPLETING", "CONFIGURING"))
TERMINAL_STATES = frozenset(("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"))


class LaunchBoundary(RuntimeError):
    """Missing or mismatched execution evidence; no permissive continuation."""


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE).strip()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise LaunchBoundary("EVIDENCE_SHA256")
    return value


def _int(value, name, *, minimum=0):
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)) or int(value) < minimum:
        raise LaunchBoundary(name)
    return int(value)


def _json(path):
    path = safe_path(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise LaunchBoundary("LOCK_NOT_OBJECT")
    return value


def verify_budget_authority(path):
    value = _json(path)
    expected = dict(instruction_id=INSTRUCTION_ID, campaign="D_PLUS_S_COMBINED",
                    total_GPU_hours=48, total_GPU_seconds=CAMPAIGN_GPU_SECONDS,
                    per_cell_or_track_budget=False, server="server1", node="devbox",
                    project_GPU_cap=2, GPUs_per_process=1, mem_MiB_per_GPU=182272)
    if any(value.get(key) != required for key, required in expected.items()):
        raise LaunchBoundary("CAMPAIGN_BUDGET_AUTHORITY_MISMATCH")
    authority = value.get("authority", {})
    if authority.get("sha256") != BUDGET_SHA256:
        raise LaunchBoundary("BUDGET_ADDENDUM_IDENTITY")
    fact = member(authority["path"])
    if fact["sha256"] != BUDGET_SHA256 or fact["bytes"] != 4400:
        raise LaunchBoundary("BUDGET_ADDENDUM_REHASH")
    return value


def empty_campaign_registry():
    return dict(schema="alpha-jv-ds.campaign-jobs.v1", instruction_id=INSTRUCTION_ID,
                budget_authority_sha256=BUDGET_SHA256, jobs=[])


def _registered_jobs(registry):
    if (registry.get("instruction_id") != INSTRUCTION_ID
            or registry.get("budget_authority_sha256") != BUDGET_SHA256):
        raise LaunchBoundary("CAMPAIGN_REGISTRY_AUTHORITY")
    result = {}
    for job in registry.get("jobs", []):
        key = str(job["job_id"])
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", key) or key in result:
            raise LaunchBoundary("REGISTERED_JOB_IDENTITY")
        if (job.get("instruction_id") != INSTRUCTION_ID or job.get("track") not in ("D", "S")
                or job.get("node") != "devbox" or job.get("gpus") != 1
                or not str(job.get("job_name", "")).startswith("odeedit_alpha_jv_ds_")
                or not job.get("owner") or not job.get("source_head") or not job.get("run_root")):
            raise LaunchBoundary("REGISTERED_JOB_OWNERSHIP")
        source_head = str(job["source_head"])
        run_root = Path(job["run_root"])
        if (not re.fullmatch(r"[0-9a-f]{40}", source_head) or not run_root.is_absolute()
                or "/local/alpha-jv-llama-diagnosis-sweep/" not in str(run_root)
                or ".." in run_root.parts):
            raise LaunchBoundary("REGISTERED_TASK_SOURCE_PATH")
        _int(job["wall_seconds"], "REGISTERED_WALL", minimum=1)
        result[key] = job
    return result


def _duration(value):
    """Parse squeue %M as [days-]hours:minutes:seconds or minutes:seconds."""
    if not isinstance(value, str):
        raise LaunchBoundary("LIVE_RESIDENCY_UNRESOLVED")
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2})(?::(\d{2}))?", value)
    if not match:
        raise LaunchBoundary("LIVE_RESIDENCY_UNRESOLVED")
    days, first, second, third = match.groups()
    if third is None:
        if days is not None or int(second) >= 60:
            raise LaunchBoundary("LIVE_RESIDENCY_UNRESOLVED")
        return 60 * int(first) + int(second)
    if int(second) >= 60 or int(third) >= 60:
        raise LaunchBoundary("LIVE_RESIDENCY_UNRESOLVED")
    return int(days or 0) * 86400 + int(first) * 3600 + int(second) * 60 + int(third)


def reconcile_accounting(registry, *, sacct_text, squeue_text, sacct_ok, squeue_ok):
    """Rebuild totals from exact registered task IDs; no live scheduler calls.

    Required sacct columns (without header):
    JobID|User|JobName|State|ElapsedRaw|AllocTRES|TimelimitRaw.
    Use -X to omit step duplication. squeue: %i|%T|%M for these exact task IDs.
    A CANCELLED sacct row still in live COMPLETING is charged at the greater
    sacct or live allocated-runtime value, never assumed released.
    """
    registered = _registered_jobs(registry)
    if not sacct_ok or not squeue_ok:
        raise LaunchBoundary("ACCOUNTING_QUERY_FAILED")
    live = {}
    for line in squeue_text.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split("|")
        if len(parts) != 3 or parts[0] not in registered or parts[0] in live:
            raise LaunchBoundary("LIVE_JOB_OWNERSHIP_OR_SCHEMA")
        if parts[1] not in LIVE_STATES | {"PENDING"}:
            raise LaunchBoundary("LIVE_JOB_STATE_UNRESOLVED")
        live[parts[0]] = (parts[1], _duration(parts[2]))
    actual = {}
    for parts in csv.reader(io.StringIO(sacct_text), delimiter="|"):
        if not parts or not any(parts):
            continue
        if parts[-1] == "":
            parts = parts[:-1]
        if len(parts) != 7:
            raise LaunchBoundary("SACCT_SCHEMA")
        job_id, owner, name, state, elapsed, tres, minutes = parts
        if "." in job_id:  # Even accidental step records cannot double charge.
            if job_id.split(".", 1)[0] not in registered:
                raise LaunchBoundary("UNREGISTERED_ACCOUNTING_STEP")
            continue
        if job_id not in registered or job_id in actual:
            raise LaunchBoundary("ACCOUNTING_JOB_UNREGISTERED_OR_DUPLICATE")
        job = registered[job_id]
        if owner != job["owner"] or name != job["job_name"]:
            raise LaunchBoundary("ACCOUNTING_JOB_OWNERSHIP")
        seconds = _int(elapsed, "ACCOUNTING_ELAPSED")
        limit = _int(minutes, "ACCOUNTING_TIME_LIMIT", minimum=1) * 60
        if limit != job["wall_seconds"]:
            raise LaunchBoundary("ACCOUNTING_TIME_LIMIT_DRIFT")
        state = state.split()[0].rstrip("+")
        if state not in LIVE_STATES | TERMINAL_STATES | {"PENDING"}:
            raise LaunchBoundary("ACCOUNTING_STATE_UNRESOLVED")
        gpu_fields = []
        for field in tres.split(","):
            if not field.startswith("gres/gpu"):
                continue
            match = re.fullmatch(r"gres/gpu(?:[:][^=,]+)?=([0-9]+)", field)
            if not match:
                raise LaunchBoundary("ALLOCATED_GPU_ACCOUNTING_MALFORMED")
            gpu_fields.append(match.group(1))
        if state != "PENDING" and not gpu_fields:
            raise LaunchBoundary("ALLOCATED_GPU_ACCOUNTING_MISSING")
        if gpu_fields and any(int(value) != 1 for value in gpu_fields):
            raise LaunchBoundary("ALLOCATED_GPU_COUNT_DRIFT")
        if state == "PENDING" and seconds != 0:
            raise LaunchBoundary("QUEUED_TIME_IS_NOT_GPU_RESIDENCY")
        if state in LIVE_STATES and job_id not in live:
            raise LaunchBoundary("UNCLEARED_ACCOUNTING_ALLOCATION")
        live_state, live_seconds = live.get(job_id, (None, 0))
        if live_state in LIVE_STATES:
            seconds = max(seconds, live_seconds)
        elif live_state == "PENDING" and state != "PENDING":
            raise LaunchBoundary("SCHEDULER_STATE_DISAGREEMENT")
        charged = 0 if state == "PENDING" and live_state not in LIVE_STATES else seconds
        outstanding = max(0, limit - charged) if state == "PENDING" or state in LIVE_STATES or live_state in LIVE_STATES else 0
        actual[job_id] = dict(job_id=job_id, track=job["track"], state=state,
                             live_state=live_state, charged_GPU_seconds=charged,
                             reserved_remaining_GPU_seconds=outstanding,
                             source_head=job["source_head"], run_root=job["run_root"])
    if set(actual) != set(registered):
        raise LaunchBoundary("REGISTERED_JOB_ACCOUNTING_MISSING")
    charged = sum(x["charged_GPU_seconds"] for x in actual.values())
    reserved = sum(x["reserved_remaining_GPU_seconds"] for x in actual.values())
    result = dict(schema="alpha-jv-ds.accounting-snapshot.v1", instruction_id=INSTRUCTION_ID,
                  jobs=list(actual.values()), registered_job_count=len(registered),
                  charged_GPU_seconds=charged, reserved_running_or_queued_GPU_seconds=reserved,
                  remaining_unreserved_GPU_seconds=CAMPAIGN_GPU_SECONDS - charged - reserved,
                  campaign_ceiling_GPU_seconds=CAMPAIGN_GPU_SECONDS,
                  scientific_exclusion_does_not_exclude_cost=True,
                  queue_wait_charged=False, registry_identity=digest(registry),
                  sacct_sha256=hashlib.sha256(sacct_text.encode()).hexdigest(),
                  squeue_sha256=hashlib.sha256(squeue_text.encode()).hexdigest(),
                  scheduler_queries_successful=True)
    result["identity"] = digest(result)
    return result


def reserve_first_s_wave(accounting):
    body = dict(accounting)
    if digest({k: v for k, v in body.items() if k != "identity"}) != body.get("identity"):
        raise LaunchBoundary("ACCOUNTING_SNAPSHOT_IDENTITY")
    if (body.get("instruction_id") != INSTRUCTION_ID
            or body.get("campaign_ceiling_GPU_seconds") != CAMPAIGN_GPU_SECONDS
            or body.get("scheduler_queries_successful") is not True
            or body.get("registered_job_count") != len(body.get("jobs", []))):
        raise LaunchBoundary("ACCOUNTING_CAMPAIGN_BOUNDARY")
    charged = sum(_int(x["charged_GPU_seconds"], "CHARGED_GPU_SECONDS") for x in body["jobs"])
    outstanding = sum(_int(x["reserved_remaining_GPU_seconds"], "RESERVED_GPU_SECONDS") for x in body["jobs"])
    if (charged != body["charged_GPU_seconds"] or outstanding != body["reserved_running_or_queued_GPU_seconds"]
            or body["remaining_unreserved_GPU_seconds"] != CAMPAIGN_GPU_SECONDS - charged - outstanding):
        raise LaunchBoundary("ACCOUNTING_TOTAL_IDENTITY")
    reserved = 2 * CELL_WALL_SECONDS
    remaining = accounting["remaining_unreserved_GPU_seconds"]
    if remaining < reserved:
        raise LaunchBoundary("CAMPAIGN_BUDGET_INSUFFICIENT_FOR_TWO_MODEL_WAVE")
    return dict(new_wave_reservation_GPU_seconds=reserved, new_wave_reservation_GPU_hours=16,
                remaining_after_wave_reservation_GPU_seconds=remaining - reserved,
                per_cell_wall_seconds=CELL_WALL_SECONDS, array="0-1%2",
                first_wave_is_not_entire_campaign_budget=True)


def preparelaunch(root, repo, *, budget_lock, sample_lock, assets_lock, cpu_checks_lock,
                  campaign_registry, accounting_snapshot=None, python_executable=PYTHON):
    """Create a new launch namespace and locks; never call sbatch/scontrol."""
    repo = Path(repo).absolute()
    root = safe_path(root, root=repo / "local/alpha-jv-llama-diagnosis-sweep", make_parents=True)
    if python_executable != PYTHON:
        raise LaunchBoundary("PINNED_PYTHON_EXECUTABLE")
    if root.exists():
        raise LaunchBoundary("CREATE_ONCE_ATTEMPT_ALREADY_EXISTS")
    budget = verify_budget_authority(budget_lock)
    registered = _registered_jobs(campaign_registry)
    if accounting_snapshot is None:
        if registered:
            raise LaunchBoundary("REGISTERED_ATTEMPTS_REQUIRE_ACCOUNTING")
        accounting_snapshot = reconcile_accounting(campaign_registry, sacct_text="", squeue_text="", sacct_ok=True, squeue_ok=True)
    if accounting_snapshot["registry_identity"] != digest(campaign_registry):
        raise LaunchBoundary("ACCOUNTING_REGISTRY_MISMATCH")
    reservation = reserve_first_s_wave(accounting_snapshot)
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise LaunchBoundary("TRACKED_EXECUTION_SOURCE_DIRTY")
    head, tree = _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD^{tree}")
    script = repo / PACKAGE / "server1_sweep.sbatch"
    validate_sbatch_memory(script.read_text())
    for relative in (PACKAGE + "/server1_sweep.sbatch", PACKAGE + "/gpu_runtime.py", PACKAGE + "/launch.py"):
        data = _git(repo, "ls-files", "--", relative)
        if data != relative:
            raise LaunchBoundary("LAUNCH_SOURCE_NOT_COMMITTED: " + relative)
    source_members = [member(repo / name, relative_to=repo) for name in _git(repo, "ls-files", PACKAGE).splitlines()]
    bindings = {name: member(path) for name, path in (("budget", budget_lock), ("sample", sample_lock), ("assets", assets_lock), ("cpu_checks", cpu_checks_lock))}
    sample, assets, checks = _json(sample_lock), _json(assets_lock), _json(cpu_checks_lock)
    if checks.get("status") not in ("PASS_CPU_ONLY", "PASS", "CPU_PASS") or not assets:
        raise LaunchBoundary("CPU_ASSET_BINDING_REQUIRED")
    root.mkdir(mode=0o700)
    source = dict(head=head, tree=tree, repo=str(repo), branch=_git(repo, "branch", "--show-current"),
                  instruction_id=INSTRUCTION_ID, runtime_parent=RUNTIME_HEAD,
                  contract_sha256=CONTRACT_SHA256, members=source_members, members_root=digest(source_members))
    resource = dict(schema="alpha-jv-ds.launch-resource.v1", instruction_id=INSTRUCTION_ID,
                    user_budget_authority=bindings["budget"], budget_addendum_sha256=BUDGET_SHA256,
                    campaign_GPU_seconds=CAMPAIGN_GPU_SECONDS, campaign_GPU_hours=48,
                    charged_GPU_seconds=accounting_snapshot["charged_GPU_seconds"],
                    previously_reserved_GPU_seconds=accounting_snapshot["reserved_running_or_queued_GPU_seconds"],
                    project_GPU_cap=2, GPU_per_process=1, mem_MiB=182272, cpus_per_task=8,
                    node="devbox", export="NONE", no_requeue=True, **reservation,
                    accounting_snapshot_identity=accounting_snapshot["identity"],
                    requires_fresh_live_resource_check_before_submit=True,
                    current_allocations_mutated=False, old_budget_inheritance_count=0)
    values = {"source.lock.json": source, "resource.lock.json": resource,
              "budget-authority.lock.json": budget, "sample.lock.json": sample,
              "assets.lock.json": assets, "cpu-checks.lock.json": checks,
              "science.lock.json": dict(instruction_id=INSTRUCTION_ID, contract_sha256=CONTRACT_SHA256,
                  runtime_parent=RUNTIME_HEAD, plan=dry_plan(), stock_compute_z_unchanged=True,
                  inner_fixed_z_recompute_count=0, normalization_primary="N0_SOURCE", scientific_promotion=False),
              "prior-campaign-registry.json": campaign_registry,
              "admission-accounting.json": accounting_snapshot}
    for name, value in values.items():
        write_once(root / name, value, root=root)
    locks = [member(root / name, relative_to=root) for name in sorted(values)]
    launch = dict(schema="alpha-jv-ds.launch.v1", instruction_id=INSTRUCTION_ID, repo=str(repo),
                  run_root=str(root), source_head=head, source_tree=tree, job_name=JOB_NAME,
                  python_executable=python_executable, caller_input_bindings=bindings,
                  model_mapping={"0": "llama3-8b-inst", "1": "qwen2.5-7b-inst"},
                  source_script=member(script), locks=locks, locks_root=digest(locks),
                  suggested_held_command=["sbatch", "--parsable", "--hold",
                    f"--output={root}/slurm-%A_%a.out", f"--error={root}/slurm-%A_%a.err",
                    str(script), str(repo), str(root)],
                  status="PREPARED_NOT_SUBMITTED", premodel_gate="launch.validate_launch",
                  actual_GPU_fidelity="RUNTIME_NORMAL_SIGNAL_GATE_NOT_YET_RUN")
    launch["identity"] = digest(launch)
    write_once(root / "launch.lock.json", launch, root=root)
    return launch


def register_submission(root, *, array_job_id, owner, owner_uid, held_inspection_sha256,
                        live_admission_sha256):
    """Called only after SH1's actual held inspection; no scheduler action."""
    root = safe_path(root)
    launch = _json(root / "launch.lock.json")
    array_id = str(_int(array_job_id, "ARRAY_JOB_ID", minimum=1))
    _sha(held_inspection_sha256); _sha(live_admission_sha256)
    jobs = [dict(job_id=f"{array_id}_{cell}", array_job_id=array_id, cell=cell,
                 instruction_id=INSTRUCTION_ID, track="S", attempt_id=root.name,
                 source_head=launch["source_head"], source_tree=launch["source_tree"],
                 source_repo=launch["repo"], run_root=str(root), owner=owner, owner_uid=int(owner_uid),
                 job_name=JOB_NAME, node="devbox", gpus=1, wall_seconds=CELL_WALL_SECONDS)
            for cell in (0, 1)]
    value = dict(schema="alpha-jv-ds.submission-registration.v1", instruction_id=INSTRUCTION_ID,
                 budget_authority_sha256=BUDGET_SHA256, launch_identity=launch["identity"], jobs=jobs,
                 held_inspection_sha256=held_inspection_sha256, live_admission_sha256=live_admission_sha256)
    _registered_jobs(value)
    value["identity"] = digest(value)
    write_once(root / "submission-registration.json", value, root=root)
    return value


def validate_launch(repo, root, cell, *, environ=None, current_uid=None):
    """Mechanical CPU guard called before any model import/load or GPU use."""
    repo, root = Path(repo).absolute(), safe_path(root)
    env = os.environ if environ is None else environ
    uid = os.getuid() if current_uid is None else current_uid
    cell = _int(cell, "CELL_MAPPING")
    if cell not in (0, 1):
        raise LaunchBoundary("CELL_MAPPING")
    launch = _json(root / "launch.lock.json")
    if digest({k: v for k, v in launch.items() if k != "identity"}) != launch.get("identity"):
        raise LaunchBoundary("LAUNCH_LOCK_IDENTITY")
    if launch["repo"] != str(repo) or launch["run_root"] != str(root):
        raise LaunchBoundary("LAUNCH_PATH_BINDING")
    if (_git(repo, "rev-parse", "HEAD") != launch["source_head"]
            or _git(repo, "rev-parse", "HEAD^{tree}") != launch["source_tree"]
            or _git(repo, "status", "--porcelain", "--untracked-files=no")):
        raise LaunchBoundary("QUEUED_SOURCE_DRIFT_OR_DIRTY")
    observed = [member(root / x["path"], relative_to=root) for x in launch["locks"]]
    if observed != launch["locks"] or digest(observed) != launch["locks_root"]:
        raise LaunchBoundary("LOCK_BYTES_DRIFT")
    source = _json(root / "source.lock.json")
    source_members = [member(repo / x["path"], relative_to=repo) for x in source["members"]]
    if source_members != source["members"] or digest(source_members) != source["members_root"]:
        raise LaunchBoundary("SOURCE_MEMBER_DRIFT")
    verify_budget_authority(root / "budget-authority.lock.json")
    registration = _json(root / "submission-registration.json")
    if (digest({k: v for k, v in registration.items() if k != "identity"}) != registration.get("identity")
            or registration.get("launch_identity") != launch["identity"]):
        raise LaunchBoundary("SUBMISSION_REGISTRATION_IDENTITY")
    jobs = _registered_jobs(registration)
    task = f"{env.get('SLURM_ARRAY_JOB_ID')}_{env.get('SLURM_ARRAY_TASK_ID')}"
    if task not in jobs or str(cell) != env.get("SLURM_ARRAY_TASK_ID"):
        raise LaunchBoundary("UNREGISTERED_RUNTIME_TASK")
    job = jobs[task]
    if (job["owner_uid"] != uid or job["source_head"] != source["head"]
            or job["run_root"] != str(root) or job["source_repo"] != str(repo)
            or env.get("SLURM_JOB_NAME") != job["job_name"]):
        raise LaunchBoundary("RUNTIME_JOB_OWNERSHIP_OR_SOURCE")
    return dict(status="PREMODEL_MECHANICAL_PASS", cell=cell,
                model_alias=launch["model_mapping"][str(cell)], source_head=source["head"],
                source_tree=source["tree"], task_id=task, launch_identity=launch["identity"],
                actual_GPU_fidelity="NOT_CLAIMED_BY_PREMODEL_CPU_CHECK")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    check = sub.add_parser("premodel")
    check.add_argument("--repo", type=Path, required=True)
    check.add_argument("--run-root", type=Path, required=True)
    check.add_argument("--cell", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_launch(args.repo, args.run_root, args.cell), sort_keys=True))


if __name__ == "__main__":
    main()
