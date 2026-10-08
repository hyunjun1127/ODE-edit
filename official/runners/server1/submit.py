"""Create-once, held-inspected Slurm registration for server1 official runners.

This module never runs a model, changes an old job, waits for a free GPU, or
retries sbatch. A plan and a reviewed main commit are required before registration.
Qualification PLAN/source binding is not an actual GPU qualification receipt.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import uuid

from official.experiments.prepare import digest, file_sha, write_new


SCHEMA = "official-server1-slurm-plan-v1"
METHODS = ("FT", "MEMIT", "MEMIT_FE")
ACTIVE = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}
DEFAULT_PYTHON = "/mnt/raid5/janghj/EasyEdit/.venv/bin/python"
DEFAULT_RESOURCES = dict(node="devbox", partition="gpu", qos="lab_gpu_s1",
                         cpus=8, memory_MiB=65536, wall_seconds=48 * 3600,
                         collector_memory_MiB=24576, collector_wall_seconds=4 * 3600,
                         storage_reserve_bytes=64 * (1 << 30), inode_reserve=10000)
SOURCE_ENV = "OFFICIAL_CODE_COMMIT"
TREE_ENV = "OFFICIAL_TREE_SHA256"


class RegistrationError(RuntimeError):
    """A precise control-path failure, never a scientific performance gate."""


def require(value, code):
    if not value:
        raise RegistrationError(code)


def command(argv, *, cwd=None):
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RegistrationError(f"COMMAND_FAILED: {argv[0]}: {result.returncode}: {result.stderr.strip()}")
    return result.stdout.strip()


def read(path):
    return json.loads(Path(path).read_text())


def member(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "INPUT_REGULAR_FILE_REQUIRED")
    before = path.stat()
    checksum = file_sha(path)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "INPUT_CHANGED_DURING_HASH")
    return dict(path=str(path), bytes=after.st_size, sha256=checksum)


def verify(row):
    require(set(row) >= {"path", "bytes", "sha256"}, "INPUT_MEMBER_SCHEMA")
    actual = member(row["path"])
    require(all(actual[key] == row[key] for key in actual), "INPUT_MEMBER_CHANGED")
    return Path(row["path"])


def metadata(text):
    """Preserve present-empty NodeList rather than mistaking it for missing."""
    # Actual Slurm keys include Socks/Node and ReqB:S:C:T. Their boundaries
    # must terminate AllocTRES rather than becoming part of its final GPU count.
    matches = list(re.finditer(r"(?<!\S)([A-Za-z][A-Za-z0-9_/:]*)=", text))
    result = {}
    for index, match in enumerate(matches):
        require(match[1] not in result, "DUPLICATE_SLURM_METADATA")
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[match[1]] = text[match.end():end].strip()
    return result


def gpu_count(tres):
    counts = {}
    for token in tres.split(","):
        if token.startswith("gres/gpu"):
            parsed = re.fullmatch(r"(gres/gpu(?::[^=,]+)?)=(\d+)", token)
            require(parsed and parsed[1] not in counts, "GPU_TRES_NUMERIC_OR_DUPLICATE")
            counts[parsed[1]] = int(parsed[2])
    generic = counts.get("gres/gpu")
    typed = sum(value for key, value in counts.items() if key != "gres/gpu")
    require(generic is None or not typed or typed == generic, "GPU_TRES_GENERIC_TYPED_CONFLICT")
    return generic if generic is not None else typed


def dependencies(text):
    if text in ("", "(null)", "None", "N/A"):
        return []
    require("?" not in text, "OR_DEPENDENCY_NOT_ADMISSIBLE")
    result = []
    for condition in text.split(","):
        pieces = re.sub(r"\([^)]*\)", "", condition).split(":")
        require(pieces[0] in ("afterok", "afterany") and len(pieces) > 1,
                "UNSUPPORTED_DEPENDENCY_TYPE")
        for job in pieces[1:]:
            require(re.fullmatch(r"\d+(?:_\d+)?", job), "NONNUMERIC_DEPENDENCY_ID")
            result.append((pieces[0], job))
    return sorted(set(result))


def graph_width(rows):
    """Weighted DAG antichain: no job-name patterns, array/step double counting."""
    nodes = {str(row["key"]): row for row in rows}
    require(len(nodes) == len(rows), "DUPLICATE_JOB_NODE")
    ancestors, visiting = {}, set()
    def visit(key):
        if key in ancestors:
            return ancestors[key]
        require(key not in visiting, "DEPENDENCY_CYCLE")
        visiting.add(key)
        result = set()
        for parent in nodes[key]["parents"]:
            if parent in nodes:
                result.add(parent)
                result.update(visit(parent))
        visiting.remove(key)
        ancestors[key] = result
        return result
    for key in nodes:
        visit(key)
    slots = [(key, index) for key, row in nodes.items() for index in range(row["gpus"])]
    require(len(slots) <= 256, "BOUNDED_GPU_DAG_LIMIT")
    matching = {}
    def augment(left, seen):
        for right in slots:
            if left[0] in ancestors[right[0]] and right not in seen:
                seen.add(right)
                if right not in matching or augment(matching[right], seen):
                    matching[right] = left
                    return True
        return False
    matched = sum(augment(left, set()) for left in slots)
    return len(slots) - matched


def frontier(rows):
    gpu_nodes = {str(row["key"]) for row in rows if row["gpus"]}
    ancestors = set()
    lookup = {str(row["key"]): row for row in rows}
    def walk(key, seen):
        for parent in lookup[key]["parents"]:
            if parent in lookup and parent not in seen:
                seen.add(parent)
                walk(parent, seen)
    for key in gpu_nodes:
        found = set()
        walk(key, found)
        ancestors.update(found & gpu_nodes)
    return sorted(gpu_nodes - ancestors)


def inventory(*, runner=command, owner=None, node="devbox", exclude=()):
    """All current own queue allocations and pending requests, not name patterns."""
    owner = owner or getpass.getuser()
    ids = runner(["squeue", "--array", "--noheader", "--user", owner, "--format=%i"]).splitlines()
    require(len(ids) <= 256, "BOUNDED_OWNER_QUEUE_LIMIT")
    rows, seen = [], set()
    for display in ids:
        display = display.strip()
        require(re.fullmatch(r"\d+(?:_\d+)?", display), "UNKNOWN_ARRAY_OR_STEP_ID")
        detail = runner(["scontrol", "show", "job", display, "--oneliner"])
        value = metadata(detail)
        require(value.get("UserId", "").split("(")[0] == owner, "OWNER_METADATA_MISMATCH")
        raw = value.get("JobId")
        require(raw and raw.isdigit(), "RAW_JOB_ID_REQUIRED")
        if "ArrayJobId" in value:
            array, task = value["ArrayJobId"], value.get("ArrayTaskId", "")
            require(array.isdigit() and task.isdigit(), "ARRAY_PARENT_TASK_INDEX_REQUIRED")
            logical_key = array + "_" + task
            require(display == logical_key, "ARRAY_DISPLAY_METADATA_MISMATCH")
        else:
            logical_key = raw
            require(display == raw, "QUEUE_RAW_DISPLAY_ID_MISMATCH")
        if logical_key in seen or raw in set(map(str, exclude)) or logical_key in set(map(str, exclude)):
            continue
        seen.add(logical_key)
        state = value.get("JobState")
        require(state in ACTIVE, "QUEUE_STATE_CHANGED_RECHECK_REQUIRED")
        require("NodeList" in value and "ReqNodeList" in value, "ALLOCATION_NODE_METADATA_REQUIRED")
        allocated = value["NodeList"]
        requested = value["ReqNodeList"]
        requested_gpus = gpu_count(value.get("ReqTRES", ""))
        if not requested_gpus:
            continue
        if node not in (allocated, requested):
            require(state != "PENDING" or requested not in ("", "(null)"),
                    "PENDING_POSSIBLE_NODE_UNRESOLVED")
            continue
        require(requested == node or allocated == node, "MULTINODE_ALLOCATION_UNSUPPORTED")
        allocated_gpus = gpu_count(value.get("AllocTRES", ""))
        if state in ("RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"):
            require(allocated_gpus > 0 and allocated == node, "ACTUAL_GPU_ALLOCATION_REQUIRED")
        else:
            require(allocated in ("", "(null)") and not allocated_gpus, "PENDING_ALREADY_ALLOCATED")
        rows.append(dict(key=logical_key, raw_job_id=raw, display_id=display, gpus=requested_gpus,
                         allocated_gpus=allocated_gpus, state=state,
                         parents=[job for _, job in dependencies(value.get("Dependency", ""))],
                         detail=value, original=detail))
    return dict(owner=owner, node=node, jobs=rows, frontier=frontier(rows),
                allocated_gpus=sum(row["allocated_gpus"] for row in rows),
                admitted_DAG_width=graph_width(rows), selection="ALL_OWNER_NODE_GPU_JOBS_NOT_NAME_PATTERNS",
                arrays="EXPANDED_TASK_IDS_INDEX0_VALID; throttle not credited to conservative width")


def _job(key, mode, method, dataset, config, output, parents=(), *, gpus=1):
    return dict(key=key, mode=mode, method=method, dataset=dataset,
                config=member(config), output=str(Path(output).absolute()),
                parents=list(parents), gpus=gpus)


def build_pipeline(configs, output_root, *, main_commit, official_tree, inputs,
                   existing_frontier=(), cap=3, resources=None, purpose="pipeline",
                   python=DEFAULT_PYTHON):
    """Prepare only; GPU receipts/READY do not exist until sealed jobs run."""
    require(purpose in ("pipeline", "qualification"), "UNKNOWN_PLAN_PURPOSE")
    output_root = Path(output_root).absolute()
    jobs = [_job("qual-" + method.lower(), "qualification", method, "cf",
                 configs[(method, "cf")], output_root / ("qualification-" + method.lower()))
            for method in METHODS]
    if purpose == "pipeline":
        qual = [job["key"] for job in jobs]
        jobs.append(_job("base-w0", "base_w0", None, "cf", configs[("FT", "cf")],
                         output_root / "base-w0", qual))
        for method in METHODS:
            jobs.append(_job("cf-" + method.lower(), "chain", method, "cf",
                             configs[(method, "cf")], output_root / ("cf-" + method.lower()), ["base-w0"]))
        cf = ["cf-" + method.lower() for method in METHODS]
        jobs.append(_job("zsre-smoke", "smoke", "FT", "zsre", configs[("FT", "zsre")],
                         output_root / "zsre-smoke", cf))
        for method in METHODS:
            jobs.append(_job("zsre-" + method.lower(), "chain", method, "zsre",
                             configs[(method, "zsre")], output_root / ("zsre-" + method.lower()), ["zsre-smoke"]))
    jobs.append(_job("collector", "collect", None, "cf", configs[("FT", "cf")],
                     output_root / "collector", [job["key"] for job in jobs], gpus=0))
    plan = dict(schema=SCHEMA, purpose=purpose, server="server1", model="llama3", cap=cap,
                source=dict(main_commit=main_commit, official_tree=official_tree),
                inputs=list(inputs), existing_frontier=list(map(str, existing_frontier)),
                resources=dict(DEFAULT_RESOURCES, **(resources or {})), jobs=jobs,
                python=str(Path(python).absolute()), actual_GPU_qualification=False,
                checkpoint_policy="LATEST_ONE_FP32_W_NATIVE_STATE_RNG_CURSOR_W20_KEEP",
                automatic_retry=False, recurring_monitor=False, old_jobs_mutation=False)
    validate_plan(plan)
    return plan


def build_resume(config, output, *, method, dataset, main_commit, official_tree,
                 checkpoint_identity, checkpoint_latest, inputs, existing_frontier=(),
                 cap=3, resources=None, python=DEFAULT_PYTHON, collector_output=None):
    """New control attempt, same science/output/source/checkpoint identity."""
    require(method in METHODS and dataset in ("cf", "zsre"), "RESUME_PROFILE")
    jobs = [_job("resume-" + method.lower(), "chain", method, dataset, config, output)]
    jobs[0]["resume"] = True
    jobs.append(_job("collector", "collect", None, dataset, config,
                     collector_output or Path(output).parent / ("resume-collector-" + uuid.uuid4().hex),
                     [jobs[0]["key"]], gpus=0))
    plan = dict(schema=SCHEMA, purpose="resume", server="server1", model="llama3", cap=cap,
                source=dict(main_commit=main_commit, official_tree=official_tree),
                checkpoint_identity=checkpoint_identity, checkpoint_latest=member(checkpoint_latest),
                inputs=list(inputs), existing_frontier=list(map(str, existing_frontier)),
                resources=dict(DEFAULT_RESOURCES, **(resources or {})), jobs=jobs,
                python=str(Path(python).absolute()), actual_GPU_qualification=False,
                checkpoint_policy="LATEST_ONE_FP32_W_NATIVE_STATE_RNG_CURSOR_W20_KEEP",
                automatic_retry=False, recurring_monitor=False, old_jobs_mutation=False)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    require(plan.get("schema") == SCHEMA and plan.get("server") == "server1"
            and plan.get("model") == "llama3", "PLAN_SCOPE")
    require(plan.get("purpose") in ("pipeline", "qualification", "resume"), "UNKNOWN_PLAN_PURPOSE")
    require(type(plan.get("cap")) is int and 1 <= plan["cap"] <= 3, "PLAN_STRICTER_CAP")
    require(not plan.get("automatic_retry") and not plan.get("recurring_monitor")
            and not plan.get("old_jobs_mutation") and plan.get("actual_GPU_qualification") is False,
            "CONTROL_NOT_SCIENCE_PASS_OR_RETRY")
    source = plan["source"]
    require(all(re.fullmatch("[a-f0-9]{40}", source.get(key, ""))
                for key in ("main_commit", "official_tree")), "EXACT_MAIN_SOURCE_REQUIRED")
    resources = plan["resources"]
    require(resources["node"] == "devbox" and resources["partition"] == "gpu"
            and resources["qos"] == "lab_gpu_s1", "SERVER1_NATIVE_RESOURCE_PROFILE")
    require(1 <= resources["cpus"] <= 8 and 0 < resources["memory_MiB"] <= 183296
            and 0 < resources["collector_memory_MiB"] <= 24576
            and 0 < resources["wall_seconds"] <= 48 * 3600
            and 0 < resources["collector_wall_seconds"] <= 4 * 3600, "RESOURCE_CEILINGS")
    require(bool(plan["inputs"]) and all(re.fullmatch(r"\d+(?:_\d+)?", job)
            for job in plan["existing_frontier"]), "INPUTS_OR_CURRENT_FRONTIER")
    keys = set()
    for job in plan["jobs"]:
        require(job["key"] not in keys and re.fullmatch(r"[a-z0-9_-]+", job["key"]), "JOB_KEY")
        require(set(job["parents"]) <= keys, "TOPOLOGICAL_JOB_ORDER")
        keys.add(job["key"])
        require(job["dataset"] in ("cf", "zsre") and job["mode"] in
                ("qualification", "base_w0", "chain", "smoke", "collect"), "RUNNER_MODE")
        require(job["method"] in METHODS or (job["method"] is None and
                job["mode"] in ("base_w0", "collect")), "METHOD_NOT_OURS")
        require(job["gpus"] == (0 if job["mode"] == "collect" else 1), "GPU_LABEL_IDENTITY")
        require(Path(job["output"]).is_absolute(), "ABSOLUTE_LOCAL_OUTPUT")
        value = read(verify(job["config"]))
        require(value.get("model") == "llama3" and value.get("dataset") == job["dataset"]
                and (job["method"] is None or value.get("method") == job["method"]), "CONFIG_PROFILE")
        stream = value.get("stream", {})
        require(stream.get("requests") == 2000 and stream.get("batch_size") == 100
                and stream.get("batches") == 20, "EXACT_2K_20B_CONFIG")
    for row in plan["inputs"]:
        verify(row)
    require(graph_width(plan["jobs"]) <= plan["cap"], "NEW_DAG_EXCEEDS_CAP")
    if plan["purpose"] == "resume":
        checkpoint = verify(plan["checkpoint_latest"])
        pointer = read(checkpoint)
        require(pointer["identity_sha256"] == digest(plan["checkpoint_identity"])
                and plan["checkpoint_identity"]["code_commit"] == source["main_commit"]
                and plan["checkpoint_identity"]["official_tree_sha256"] == source["official_tree"],
                "RESUME_SAME_IMMUTABLE_SOURCE_IDENTITY")
        chain = next(job for job in plan["jobs"] if job.get("resume"))
        require(type(pointer.get("batch")) is int and 0 <= pointer["batch"] < 20
                and pointer.get("final_W20") is False
                and not (Path(chain["output"]) / "COMPLETE.json").exists(),
                "RESUME_ALREADY_COMPLETE_NO_DUPLICATE_ADMISSION")
        require(checkpoint.is_relative_to(chain["output"]), "RESUME_CHECKPOINT_OWN_OUTPUT")
        config = read(verify(chain["config"]))
        if "config_sha256" in config:
            require(plan["checkpoint_identity"]["config_sha256"] == config["config_sha256"],
                    "RESUME_SAME_SCIENTIFIC_CONFIG")
    return plan


def freeze_source(plan, attempt, *, repository=None, runner=command):
    repository = Path(repository or Path(__file__).resolve().parents[3])
    source = plan["source"]
    require(runner(["git", "rev-parse", source["main_commit"] + ":official"], cwd=repository)
            == source["official_tree"], "OFFICIAL_TREE_MISMATCH")
    runner(["git", "merge-base", "--is-ancestor", source["main_commit"], "origin/main"], cwd=repository)
    archive = attempt / "official-source.tar"
    require(not archive.exists(), "IMMUTABLE_SOURCE_ARCHIVE_EXISTS")
    runner(["git", "archive", "--format=tar", "--output=" + str(archive),
            source["main_commit"], "official"], cwd=repository)
    target = attempt / "source"
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        require(members and len(members) == len({row.name for row in members}) and
                all((row.isfile() or row.isdir()) and row.name.split("/")[0] == "official"
                    and not Path(row.name).is_absolute() and ".." not in Path(row.name).parts
                    for row in members), "OFFICIAL_ONLY_SAFE_ARCHIVE")
        bundle.extractall(target, filter="data")
    return dict(archive=member(archive), directory=str(target),
                members=[member(path) for path in sorted(target.rglob("*")) if path.is_file()],
                main_membership_verified=True, official_tree_verified=True,
                dirty_worktree_not_archived=True)


def runtime_argv(plan, job, lock_path):
    argv = [plan["python"], "-B", "-u", "-m", "official.runners.server1.run",
            "--mode", job["mode"], "--dataset", job["dataset"],
            "--config", job["config"]["path"], "--output", job["output"],
            "--source-lock", str(lock_path)]
    if job["method"] is not None:
        argv += ["--method", job["method"]]
    if job.get("resume"):
        argv.append("--resume")
    if job["mode"] == "collect":
        argv += ["--job-manifest", str(Path(lock_path).parent / "job-manifest.json")]
    return argv


def launcher(plan, job, source, lock_path):
    env = dict(PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS=str(plan["resources"]["cpus"]),
               MKL_NUM_THREADS=str(plan["resources"]["cpus"]), OPENBLAS_NUM_THREADS=str(plan["resources"]["cpus"]),
               OFFICIAL_CODE_COMMIT=plan["source"]["main_commit"],
               OFFICIAL_TREE_SHA256=plan["source"]["official_tree"],
               OFFICIAL_EXECUTION_LOCK=str(lock_path), OFFICIAL_JOB_KEY=job["key"],
               WANDB_CONSOLE="off", WANDB_DISABLE_CODE="true", WANDB_SAVE_CODE="false")
    if not job["gpus"]:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return ("#!/bin/bash\nset -euo pipefail\n" +
            "".join("export " + key + "=" + shlex.quote(value) + "\n" for key, value in env.items()) +
            "cd " + shlex.quote(str(source)) + "\nexec " +
            shlex.join(runtime_argv(plan, job, lock_path)) + "\n")


def memory_MiB(value):
    parsed = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)", value)
    require(parsed, "SLURM_MEMORY_UNIT")
    return int(float(parsed[1]) * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 ** 2}[parsed[2]])


def seconds(value):
    if value in ("UNLIMITED", "INFINITE", "N/A", ""):
        return None
    day, _, clock = value.rpartition("-")
    parts = list(map(int, clock.split(":")))
    require(len(parts) in (2, 3), "SLURM_WALL_CLOCK")
    if len(parts) == 2:
        parts.insert(0, 0)
    return int(day or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def resource_preflight(plan, *, runner=command):
    resources = plan["resources"]
    node_text = runner(["scontrol", "show", "node", resources["node"], "--oneliner"])
    partition_text = runner(["scontrol", "show", "partition", resources["partition"], "--oneliner"])
    qos_text = runner(["sacctmgr", "-n", "-P", "show", "qos", resources["qos"],
                       "format=Name,MaxWall,MaxTRESPJ,MaxTRESPU"])
    node, partition = metadata(node_text), metadata(partition_text)
    require(node.get("NodeName") == resources["node"] and
            partition.get("PartitionName") == resources["partition"], "CANONICAL_NODE_PARTITION")
    require(int(node["CPUTot"]) >= resources["cpus"] and int(node["RealMemory"]) >= resources["memory_MiB"],
            "PHYSICAL_CPU_MEMORY_CEILING")
    require(gpu_count(node.get("CfgTRES", "")) >= 1, "NODE_GPU_MISSING")
    limit = seconds(partition.get("MaxTime", ""))
    require(limit is None or resources["wall_seconds"] <= limit, "PARTITION_WALL_CEILING")
    qos_rows = [row.split("|") for row in qos_text.splitlines() if row]
    require(len(qos_rows) == 1 and qos_rows[0][0] == resources["qos"] and len(qos_rows[0]) >= 4,
            "QOS_PROFILE_REQUIRED")
    limit = seconds(qos_rows[0][1])
    require(limit is None or resources["wall_seconds"] <= limit, "QOS_WALL_CEILING")
    per_job = dict(token.split("=", 1) for token in qos_rows[0][2].split(",") if "=" in token)
    require("mem" not in per_job or resources["memory_MiB"] <= memory_MiB(per_job["mem"]), "QOS_MEMORY_CEILING")
    require("cpu" not in per_job or resources["cpus"] <= int(per_job["cpu"]), "QOS_CPU_CEILING")
    require("gres/gpu" not in per_job or gpu_count(qos_rows[0][2]) >= 1, "QOS_GPU_CEILING")
    per_user = dict(token.split("=", 1) for token in qos_rows[0][3].split(",") if "=" in token)
    require("gres/gpu" not in per_user or plan["cap"] <= gpu_count(qos_rows[0][3]),
            "STRICTER_QOS_USER_GPU_CEILING")
    return dict(node=node_text, partition=partition_text, qos=qos_text,
                actual_free_GPU_wait=False, wall_is_ceiling_not_ETA=True)


def sbatch_argv(plan, job, script, attempt, ids):
    resources = plan["resources"]
    parents = [("afterany" if job["mode"] in ("collect", "smoke") else "afterok", ids[parent])
               for parent in job["parents"]]
    if not job["parents"]:
        parents += [("afterany", old) for old in plan["existing_frontier"]]
    groups = {}
    for kind, parent in parents:
        groups.setdefault(kind, []).append(str(parent))
    conditions = [kind + ":" + ":".join(values) for kind, values in sorted(groups.items())]
    wall = resources["wall_seconds"] if job["gpus"] else resources["collector_wall_seconds"]
    argv = ["sbatch", "--parsable", "--hold", "--export=NONE", "--no-requeue",
            "--kill-on-invalid-dep=yes",
            "--partition=" + resources["partition"], "--qos=" + resources["qos"],
            "--nodelist=" + resources["node"], "--cpus-per-task=" + str(resources["cpus"]),
            "--mem=" + str(resources["memory_MiB"] if job["gpus"] else resources["collector_memory_MiB"]) + "M",
            "--time=" + str(wall // 60), "--job-name=official-s1-" + job["key"],
            "--chdir=" + str(attempt / "source"),
            "--output=" + str(attempt / "logs" / (job["key"] + "-%j.out")),
            "--error=" + str(attempt / "logs" / (job["key"] + "-%j.err"))]
    if job["gpus"]:
        argv.append("--gres=gpu:1")
    if conditions:
        argv.append("--dependency=" + ",".join(conditions))
    return argv + [str(script)], sorted(parents)


def inspect_held(plan, job, job_id, script, attempt, expected_dependencies, *, runner=command, owner=None):
    owner = owner or getpass.getuser()
    text = runner(["scontrol", "show", "job", str(job_id), "--oneliner"])
    value, resources = metadata(text), plan["resources"]
    require(value.get("JobId") == str(job_id) and value.get("UserId", "").split("(")[0] == owner,
            "HELD_EXACT_OWNER_ID")
    require(value.get("JobState") == "PENDING" and value.get("Reason") == "JobHeldUser"
            and value.get("NodeList") in ("", "(null)") and not gpu_count(value.get("AllocTRES", "")),
            "HELD_NOT_ALLOCATED_STATE")
    require(value.get("Command") == str(script) and value.get("WorkDir") == str(attempt / "source")
            and value.get("JobName") == "official-s1-" + job["key"], "HELD_FULLARGV_SOURCE")
    require(value.get("ReqNodeList") == resources["node"] and value.get("Partition") == resources["partition"]
            and value.get("QOS") == resources["qos"] and value.get("Requeue") == "0", "HELD_NODE_QOS_REQUEUE")
    require(int(value["NumCPUs"]) == resources["cpus"] and gpu_count(value.get("ReqTRES", "")) == job["gpus"],
            "HELD_EXACT_CPU_GPU")
    requested_memory = resources["memory_MiB"] if job["gpus"] else resources["collector_memory_MiB"]
    require(memory_MiB(value["MinMemoryNode"]) == requested_memory, "HELD_EXACT_MEMORY")
    requested_wall = resources["wall_seconds"] if job["gpus"] else resources["collector_wall_seconds"]
    require(seconds(value["TimeLimit"]) == requested_wall, "HELD_EXACT_WALL")
    require(dependencies(value.get("Dependency", "")) == sorted(expected_dependencies), "HELD_EXACT_DEPENDENCIES")
    batch_script = runner(["scontrol", "write", "batch_script", str(job_id), "-"])
    require(batch_script.strip() == Path(script).read_text().strip(), "HELD_ACTUAL_BATCH_SCRIPT_BYTES")
    return dict(job_id=str(job_id), key=job["key"], original=text, parsed=value,
                script=member(script), runtime_argv=runtime_argv(plan, job, attempt / "execution-lock.json"),
                Slurm_internal_batch_script_verified=True,
                GPU_qualification_claim=False, scientific_completion_claim=False)


def register(plan, attempt, *, repository=None, runner=command, owner=None):
    """One deliberate registration pass. Failure preserves exact partial IDs."""
    validate_plan(plan)
    owner = owner or getpass.getuser()
    attempt = Path(attempt).absolute()
    if (attempt / "submission.json").exists():
        existing = read(attempt / "submission.json")
        require(existing["plan_sha256"] == digest(plan), "DUPLICATE_ATTEMPT_DIFFERENT_PLAN")
        return dict(existing, duplicate_submission_avoided=True)
    require(not attempt.exists(), "REGISTRATION_ATTEMPT_EXISTS_NO_AUTOMATIC_RETRY")
    current = inventory(runner=runner, owner=owner, node=plan["resources"]["node"])
    require(current["frontier"] == sorted(plan["existing_frontier"]), "FRESH_FULL_OWNER_GPU_FRONTIER")
    require(current["allocated_gpus"] <= plan["cap"] and current["admitted_DAG_width"] <= plan["cap"],
            "LEGACY_OVERCAP_BLOCKS_NEW_ADMISSION")
    resource_receipt = resource_preflight(plan, runner=runner)
    storage_parent = attempt.parent
    while not storage_parent.exists():
        storage_parent = storage_parent.parent
    filesystem = os.statvfs(storage_parent)
    resources = plan["resources"]
    require(filesystem.f_bavail * filesystem.f_frsize >= resources["storage_reserve_bytes"]
            and filesystem.f_favail >= resources["inode_reserve"], "STORAGE_OR_INODE_RESERVE")
    resource_receipt["storage"] = dict(path=str(storage_parent),
        available_bytes=filesystem.f_bavail * filesystem.f_frsize, available_inodes=filesystem.f_favail,
        reserved_planning_bytes=resources["storage_reserve_bytes"], inode_reserve=resources["inode_reserve"],
        no_cleanup=True)
    prospective = current["jobs"] + [dict(job, key="new-" + job["key"],
        parents=["new-" + parent for parent in job["parents"]] or plan["existing_frontier"])
        for job in plan["jobs"]]
    require(graph_width(prospective) <= plan["cap"], "COMBINED_OWNER_NEW_DAG_CAP")
    attempt.mkdir(parents=True)
    (attempt / "logs").mkdir()
    write_new(attempt / "plan.json", plan)
    write_new(attempt / "resource-preflight.json", dict(resource_receipt, inventory=current))
    frozen = freeze_source(plan, attempt, repository=repository, runner=runner)
    lock_path = attempt / "execution-lock.json"
    launchers = {}
    for job in plan["jobs"]:
        script = attempt / (job["key"] + ".sh")
        with script.open("x") as stream:
            stream.write(launcher(plan, job, frozen["directory"], lock_path))
        script.chmod(0o700)
        launchers[job["key"]] = member(script)
    lock = dict(schema="official-server1-execution-lock-v1", plan_sha256=digest(plan),
                source=plan["source"], frozen_source=frozen, inputs=plan["inputs"],
                launchers=launchers, job_configs=[job["config"] for job in plan["jobs"]],
                plan=member(attempt / "plan.json"), profiles=plan["jobs"], python=plan["python"],
                owner=owner, server="server1", effective_cap=plan["cap"],
                required_actual_qualification="RUNTIME_ONLY_NOT_CONTROL_PASS")
    write_new(lock_path, lock)
    write_new(attempt / "registration-pass-started.json", dict(plan_sha256=digest(plan),
              automatic_retry=False, no_existing_job_mutation=True, effective_cap=plan["cap"]))
    ids, submitted, held, released = {}, [], [], []
    try:
        for job in plan["jobs"]:
            argv, expected = sbatch_argv(plan, job, Path(launchers[job["key"]]["path"]), attempt, ids)
            stdout = runner(argv)
            require(re.fullmatch(r"\d+(?:;[^\s;]+)?", stdout), "SBATCH_ACTUAL_ID_REQUIRED")
            job_id = stdout.split(";")[0]
            require(job_id not in ids.values(), "SBATCH_DUPLICATE_ID")
            ids[job["key"]] = job_id
            receipt = dict(key=job["key"], job_id=job_id, argv=argv, stdout=stdout,
                           dependencies=expected, scientific_source=plan["source"])
            submitted.append(receipt)
            write_new(attempt / ("submitted-" + job["key"] + ".json"), receipt)
        for job in plan["jobs"]:
            expected = next(row["dependencies"] for row in submitted if row["key"] == job["key"])
            held.append(inspect_held(plan, job, ids[job["key"]], Path(launchers[job["key"]]["path"]),
                                     attempt, expected, runner=runner, owner=owner))
        for row in frozen["members"] + list(launchers.values()) + plan["inputs"]:
            verify(row)
        validate_plan(plan)
        fresh = inventory(runner=runner, owner=owner, node=plan["resources"]["node"], exclude=ids.values())
        new_rows = [dict(job, key=ids[job["key"]], parents=[ids[parent] for parent in job["parents"]]
                        or plan["existing_frontier"]) for job in plan["jobs"]]
        require(fresh["allocated_gpus"] <= plan["cap"] and graph_width(fresh["jobs"] + new_rows) <= plan["cap"],
                "FRESH_PRE_RELEASE_COMBINED_CAP")
        write_new(attempt / "held-inspection.json", dict(jobs=held, fresh_inventory=fresh,
                  all_owner_source_input_argv_resources_dependencies_verified=True,
                  actual_GPU_qualification=False, release_authorized=True))
        # Collector inputs must already exist before any release: fast upstream
        # technical failures can satisfy afterany immediately.
        write_new(attempt / "job-manifest.json", dict(schema="official-server1-job-manifest-v1",
                  jobs=ids, submitted=submitted, plan_sha256=digest(plan), source=plan["source"],
                  profiles=plan["jobs"], execution_lock=member(lock_path),
                  actual_GPU_qualification=False, scientific_completion=False))
        # Downstream first: afterany collector cannot race an upstream release.
        for job in reversed(plan["jobs"]):
            stdout = runner(["scontrol", "release", ids[job["key"]]])
            released.append(dict(key=job["key"], job_id=ids[job["key"]], stdout=stdout))
        snapshots = {key: metadata(runner(["scontrol", "show", "job", job_id, "--oneliner"]))
                     for key, job_id in ids.items()}
        result = dict(schema="official-server1-submission-v1", plan_sha256=digest(plan),
                      jobs=ids, submitted=submitted, released=released, initial_snapshot=snapshots,
                      source=plan["source"], execution_lock=member(lock_path),
                      effective_cap=plan["cap"], status="REGISTERED_RELEASED_BOUNDED_HANDOFF",
                      actual_GPU_qualification=False, W_and_metrics_resume_verified=False,
                      online_validation="NEXT_ACTUAL_STARTUP_NOT_CONTROL_CERTIFICATION",
                      automatic_retry=False, recurring_monitor=False)
        write_new(attempt / "submission.json", result)
        return result
    except BaseException as error:
        write_new(attempt / "registration-failure.json", dict(status="REGISTRATION_BLOCKED_OR_PARTIAL",
                  error_type=type(error).__name__, error=str(error), actual_ids=ids,
                  submitted=submitted, held=held, released=released, source=plan["source"],
                  old_jobs_mutated=False, automatic_retry=False, source_raw_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--repository")
    parser.add_argument("--register", action="store_true", help="Actual one-pass sbatch/inspection/release")
    args = parser.parse_args()
    plan = validate_plan(read(args.plan))
    if not args.register:
        print(json.dumps(dict(status="PLAN_VALIDATED_NOT_SUBMITTED", plan_sha256=digest(plan),
                              GPU_qualification=False), sort_keys=True))
        return
    result = register(plan, args.attempt, repository=args.repository)
    print(json.dumps(dict(status=result["status"], jobs=result["jobs"], source=result["source"]), sort_keys=True))


if __name__ == "__main__":
    main()
