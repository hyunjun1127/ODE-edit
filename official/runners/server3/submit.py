"""Freeze, register, inspect and release server3 Qwen baseline Slurm jobs.

There are three serial stages.  Qualification contains CF W0 and four method
resume-parity jobs.  CF contains eight physical edit chains.  zsRE contains its
own W0, a cold one-batch FT smoke, and six fresh edit chains.  The selected CF
BLUE main row is a grid alias.  No command cancels an old job.

Recovery is implemented only for failed physical edit chains with a valid
checkpoint.  A failed W0, method qualification, or zsRE smoke leaves downstream
jobs pending and blocks stage re-registration.  Those failures require an
explicit, separately reviewed immutable retry/dependency repair; this module
does not report the stage complete or silently restart them.
"""

import argparse
import copy
import fcntl
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

from official.experiments.prepare import build_matrix, digest, file_sha, load_plan


ROOT = Path(__file__).resolve().parents[3]
SERVER = "server3"
NODE = "ubuntu"
PARTITION = "gpu"
QOS = "lab_gpu_s3"
JOB_PREFIX = "official_s3_qwen_"
GRID = (1, 10, 95)
METHODS = ("FT", "MEMIT", "ALPHAEDIT", "ALPHAEDIT_BLUE", "MEMIT_FE", "SPHERE")
# Historical W2000 MEMIT and AlphaEdit CF checkpoints are reserved for a
# separately bound generation evaluation. Their new official CF edit chains
# and refit qualifications are excluded; zsRE and the CF clamp control remain.
CF_NEW_CHAIN_METHODS = ("FT", "ALPHAEDIT_BLUE", "MEMIT_FE", "SPHERE")
AGENT_SEALS = ("agents/server3/experiment-ready-paths-20260919-v1.json",
               "agents/server4/p4-hf-consumed-closure-seal.json",
               "agents/server4/alphaedit-runtime-path-seal.json")
DEFAULT_MEMORY_MIB = 116736  # 114 GiB, below the server3 119 GiB policy ceiling.
GIB = 1 << 30


class Blocked(RuntimeError):
    """A missing or changed prerequisite must stop before GPU registration."""


def require(condition, code):
    if not condition:
        raise Blocked(code)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Create a receipt once; an identical replay is harmless."""
    path = Path(path)
    data = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        require(path.read_bytes() == data, "REFUSE_OVERWRITE:" + str(path))


def command(argv, *, cwd=None, env=None, timeout=60):
    result = subprocess.run([str(x) for x in argv], cwd=cwd, env=env, text=True,
                            capture_output=True, timeout=timeout)
    if result.returncode:
        # Do not echo output from auth checks, which can contain private details.
        raise Blocked("COMMAND_FAILED:" + str(argv[0]) + ":" + str(result.returncode)
                      + ":" + result.stderr.strip()[:500])
    return result.stdout.strip()


def member(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), "MISSING_FILE:" + str(path))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha(path)}


def verify_member(record):
    path = Path(record["path"])
    require(path.is_file() and path.stat().st_size == record["bytes"],
            "INPUT_MISSING_OR_SIZE_CHANGED:" + str(path))
    require(file_sha(path) == record["sha256"], "INPUT_SHA_CHANGED:" + str(path))
    return path


def official_tree_sha256(source_root):
    """SHA256 of sorted official/ regular paths and their SHA256 contents.

    A Git archive has exactly the tracked files.  Bytecode caches are excluded
    so importing the frozen package does not change the identity.
    """
    official = Path(source_root) / "official"
    require(official.is_dir(), "OFFICIAL_TREE_MISSING")
    require(not any(p.is_symlink() for p in official.rglob("*")),
            "OFFICIAL_TREE_SYMLINK")
    files = [p for p in official.rglob("*") if p.is_file() and not p.is_symlink()
             and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    require(files, "OFFICIAL_TREE_EMPTY")
    hasher = hashlib.sha256()
    for path in sorted(files, key=lambda p: p.relative_to(source_root).as_posix()):
        relative = path.relative_to(source_root).as_posix()
        hasher.update(relative.encode() + b"\0" + file_sha(path).encode() + b"\n")
    return hasher.hexdigest()


def verify_tracking_closure(source_root):
    """The runner and its tracking implementation must share the frozen tree."""
    source_root = Path(source_root)
    for name in ("official/tracking/client.py", "official/tracking/schema.py",
                 "official/tracking/method.py", "official/tracking/worker.py"):
        require((source_root / name).is_file(), "OFFICIAL_TRACKING_SOURCE_MISSING:" + name)
    runner = source_root / "official/runners/server3/run.py"
    require(runner.is_file(), "FROZEN_RUNNER_MISSING")
    runner_source = runner.read_text()
    require("official.tracking" in runner_source,
            "RUNNER_OFFICIAL_TRACKING_IMPORT_MISSING")
    require("project.run_scripts.experiment_tracking" not in runner_source,
            "RUNNER_LEGACY_TRACKING_IMPORT")


def git_main_identity():
    commit = command(["git", "rev-parse", "HEAD"], cwd=ROOT)
    require(commit == command(["git", "rev-parse", "refs/remotes/origin/main"], cwd=ROOT),
            "SOURCE_MUST_BE_PUBLISHED_MAIN_COMMIT")
    sealed_paths = ("official", *AGENT_SEALS)
    require(not command(["git", "status", "--porcelain", "--", *sealed_paths], cwd=ROOT),
            "OFFICIAL_TREE_UNCOMMITTED")
    for name in ("official/runners/server3/run.py", "official/runners/server3/submit.py",
                 "official/evaluation/factual.py", "official/tracking/client.py",
                 "official/tracking/schema.py", "official/tracking/method.py",
                 "official/tracking/worker.py", *sealed_paths[1:]):
        require(command(["git", "ls-files", "--", name], cwd=ROOT) == name,
                "SOURCE_NOT_TRACKED:" + name)
    return commit


def checked_archive_members(tar):
    members = tar.getmembers()
    require(members and len({x.name for x in members}) == len(members), "ARCHIVE_DUPLICATE")
    for entry in members:
        parts = Path(entry.name).parts
        require(parts and parts[0] in ("official", "agents") and
                entry.name.rstrip("/") == Path(entry.name).as_posix() and
                not any(part in (".", "..") for part in entry.name.split("/")) and
                not Path(entry.name).is_absolute() and
                (entry.isfile() or entry.isdir()) and
                (parts[0] != "agents" or not entry.isfile() or
                 entry.name in AGENT_SEALS), "UNSAFE_SOURCE_ARCHIVE")
    return members


def unpack_source_archive(archive, source):
    """Extract only regular, canonically named files in the sealed closure."""
    source = Path(source)
    source.mkdir()
    with tarfile.open(archive) as tar:
        for entry in checked_archive_members(tar):
            target = source / entry.name
            if entry.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(entry) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)


def source_lock_for_archive(attempt, commit, sha, *, parent_attempt=None):
    archive = attempt / "official.tar"
    source = attempt / "source"
    require(official_tree_sha256(source) == sha, "SOURCE_TREE_HASH_MISMATCH")
    verify_tracking_closure(source)
    result = {"code_commit": commit, "official_tree_sha256": sha,
              "archive": member(archive), "source": str(source),
              "source_members": [member(p) for p in sorted(source.rglob("*")) if p.is_file()],
              "agent_seals": [member(source / path) for path in AGENT_SEALS]}
    if parent_attempt is not None:
        result["source_parent_attempt"] = str(parent_attempt)
    return result


def verify_source_snapshot(attempt, lock):
    """Check a prior archive, extracted tree and every locked file byte."""
    attempt = Path(attempt).resolve(strict=True)
    source = attempt / "source"
    archive = attempt / "official.tar"
    require(lock.get("source") == str(source) and
            lock.get("archive", {}).get("path") == str(archive),
            "SOURCE_ARCHIVE_SCOPE_MISMATCH")
    require(not archive.is_symlink() and not source.is_symlink(),
            "SOURCE_ARCHIVE_OR_ROOT_SYMLINK")
    require(re.fullmatch(r"[0-9a-f]{40}", lock.get("code_commit", "")) and
            re.fullmatch(r"[0-9a-f]{64}", lock.get("official_tree_sha256", "")),
            "SOURCE_IDENTITY_INVALID")
    verify_member(lock["archive"])
    require(official_tree_sha256(source) == lock["official_tree_sha256"],
            "FROZEN_OFFICIAL_TREE_CHANGED")
    verify_tracking_closure(source)
    require(not any(p.is_symlink() for p in source.rglob("*")),
            "FROZEN_SOURCE_SYMLINK")
    actual = {str(p.resolve()) for p in source.rglob("*") if p.is_file()
              and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    records = lock["source_members"]
    expected = {record["path"] for record in records}
    require(len(records) == len(expected) and actual == expected and
            all(Path(path).is_relative_to(source) for path in expected),
            "FROZEN_SOURCE_MEMBER_SET_CHANGED")
    for record in records:
        verify_member(record)
    seals = lock["agent_seals"]
    require({x["path"] for x in seals} == {str(source / path) for path in AGENT_SEALS} and
            len(seals) == len(AGENT_SEALS) and
            all(x in records for x in seals), "AGENT_SEAL_SET_CHANGED")
    for record in seals:
        verify_member(record)
    relative = {str(Path(x["path"]).relative_to(source)): x for x in records}
    with tarfile.open(archive) as tar:
        archive_files = {entry.name for entry in checked_archive_members(tar)
                         if entry.isfile()}
    require(archive_files == set(relative), "SOURCE_ARCHIVE_MEMBER_SET_CHANGED")
    return relative


def freeze_source(attempt):
    commit = git_main_identity()
    archive = attempt / "official.tar"
    source = attempt / "source"
    command(["git", "archive", "--format=tar", "--output=" + str(archive),
             commit, "official", *AGENT_SEALS], cwd=ROOT)
    unpack_source_archive(archive, source)
    sha = official_tree_sha256(source)
    require(sha == official_tree_sha256(ROOT), "SOURCE_CHANGED_DURING_FREEZE")
    return source_lock_for_archive(attempt, commit, sha)


def inherit_source(attempt, parent_attempt, output_root, expected_stage):
    """Make an independent, byte-identical task-local copy of a released stage."""
    output_root = Path(output_root).resolve(strict=True)
    attempt = Path(attempt)
    require(attempt.parent == output_root / "attempts" and
            attempt.name.startswith("attempt-"), "SOURCE_CHILD_SCOPE")
    parent_attempt = Path(parent_attempt).resolve(strict=True)
    require(parent_attempt.parent == output_root / "attempts" and
            parent_attempt.name.startswith("attempt-") and
            (parent_attempt / "release.json").is_file(), "SOURCE_PARENT_SCOPE_OR_RELEASE")
    parent = read(parent_attempt / "execution.lock.json")
    require(parent.get("stage") == expected_stage and
            parent.get("output_root") == str(output_root) and
            parent.get("owner") == getpass.getuser(), "SOURCE_PARENT_STAGE_OR_OWNER")
    prior_members = verify_source_snapshot(parent_attempt, parent)
    archive = attempt / "official.tar"
    with (parent_attempt / "official.tar").open("rb") as src, archive.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    require(file_sha(archive) == parent["archive"]["sha256"] and
            archive.stat().st_size == parent["archive"]["bytes"],
            "INHERITED_ARCHIVE_HASH_MISMATCH")
    unpack_source_archive(archive, attempt / "source")
    result = source_lock_for_archive(attempt, parent["code_commit"],
                                     parent["official_tree_sha256"],
                                     parent_attempt=parent_attempt)
    copied = {str(Path(x["path"]).relative_to(result["source"])): x
              for x in result["source_members"]}
    require(copied.keys() == prior_members.keys() and
            all((copied[name]["bytes"], copied[name]["sha256"]) ==
                (record["bytes"], record["sha256"])
                for name, record in prior_members.items()),
            "INHERITED_SOURCE_MEMBERS_CHANGED")
    return result


def stage_source(stage, attempt, output_root, qualification=None, selection=None):
    if stage == "qualify":
        return freeze_source(attempt)
    if stage == "cf":
        require(qualification is not None, "CF_QUALIFICATION_SOURCE_MISSING")
        return inherit_source(attempt, qualification["attempt"], output_root, "qualify")
    require(stage == "zsre" and selection is not None, "ZSRE_SELECTION_SOURCE_MISSING")
    return inherit_source(attempt, selection["source_attempt"], output_root, "cf")


def matrix_rows(matrix_root):
    contract, profiles = load_plan()
    rows = {r["run_id"]: r for r in build_matrix(contract, profiles) if r["model"] == "qwen25"}
    require(len(rows) == 17, "QWEN_MATRIX_SIZE")
    root = Path(matrix_root).resolve(strict=True)
    files = {}
    for run_id, row in rows.items():
        path = root / "configs" / (run_id + ".json")
        require(path.is_file() and read(path) == row, "MATRIX_CONFIG_MISMATCH:" + run_id)
        files[run_id] = path
    return rows, files


def stage_specs(stage, rows, files, output_root, selection=None):
    dataset = "cf" if stage in ("qualify", "cf") else "zsre"
    if stage == "qualify":
        specs = [{"key": "qwen25-cf-w0", "kind": "w0", "dataset": "cf",
                  "output": str(output_root / "shared-w0" / "qwen25-cf")}]
        for method in CF_NEW_CHAIN_METHODS:
            run_id = ("qwen25-cf-alphaedit_blue-l2-1" if method == "ALPHAEDIT_BLUE"
                      else f"qwen25-cf-{method.lower()}")
            specs.append({"key": "qwen25-cf-qualify-" + method.lower(),
                          "kind": "qualify", "method": method, "run_id": run_id,
                          "config": str(files[run_id]),
                          "config_sha256": file_sha(files[run_id]),
                          "output": str(output_root / "qualification" / method)})
        return specs
    configs = []
    for method in CF_NEW_CHAIN_METHODS if stage == "cf" else METHODS:
        run_id = f"qwen25-{dataset}-{method.lower()}"
        if dataset == "cf" and method == "ALPHAEDIT_BLUE":
            continue  # Selected grid result is an alias, not a second chain.
        if dataset == "zsre" and method == "ALPHAEDIT_BLUE":
            require(selection is not None, "ZSRE_BLUE_SELECTION_REQUIRED")
            config = Path(selection["selected_zsre_config"])
            require(file_sha(config) == selection["selected_zsre_config_sha256"],
                    "SELECTED_CONFIG_CHANGED")
            row = read(config)
            require(row["run_id"] == run_id and row["hparams"]["L2"] in GRID,
                    "SELECTED_ZSRE_L2_INVALID")
        else:
            row, config = rows[run_id], files[run_id]
        configs.append((run_id, row, config))
    if stage == "cf":
        for l2 in GRID:
            run_id = f"qwen25-cf-alphaedit_blue-l2-{l2}"
            configs.append((run_id, rows[run_id], files[run_id]))
        for method in ("ALPHAEDIT", "ALPHAEDIT_BLUE"):
            run_id = f"qwen25-cf-{method.lower()}-clamp075"
            configs.append((run_id, rows[run_id], files[run_id]))
    require(len(configs) == (8 if stage == "cf" else 6), "PHYSICAL_CHAIN_COUNT")
    specs = []
    if stage == "zsre":
        specs.append({"key": "qwen25-zsre-w0", "kind": "w0", "dataset": dataset,
                      "output": str(output_root / "shared-w0" / "qwen25-zsre")})
    if stage == "zsre":
        run_id, row, config = next(x for x in configs if x[1]["method"] == "FT")
        specs.append({"key": "qwen25-zsre-ft-b1-smoke", "kind": "smoke",
                      "run_id": run_id, "config": str(config), "config_sha256": file_sha(config),
                      "output": str(output_root / "qualification" / "qwen25-zsre-ft-b1")})
    for run_id, row, config in configs:
        require(row["model"] == "qwen25" and row["dataset"] == dataset and
                row["execution_kind"] != "selected_grid_alias", "BAD_RUN_SCOPE:" + run_id)
        specs.append({"key": run_id, "kind": "edit", "run_id": run_id,
                      "config": str(config), "config_sha256": file_sha(config),
                      "output": str(output_root / "runs" / run_id)})
    return specs


def stream_member(matrix_root, dataset):
    stream = Path(matrix_root).resolve() / (dataset + "-stream.json")
    lock = Path(matrix_root).resolve() / (dataset + "-stream.lock.json")
    source_lock = ROOT / "official" / "hparams" / (dataset + "-stream.lock.json")
    require(stream.is_file() and lock.is_file(), "PREPARED_STREAM_MISSING:" + dataset)
    receipt = read(lock)
    require(receipt == read(source_lock) and receipt["stream_sha256"] == file_sha(stream),
            "PREPARED_STREAM_LOCK_MISMATCH:" + dataset)
    return member(stream), member(lock)


def model_snapshot_bytes(assets):
    path = Path(assets.get("model_snapshot", ""))
    require(path.is_dir(), "QWEN_MODEL_SNAPSHOT_MISSING")
    weights = [p for p in path.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".bin")]
    total = sum(p.stat().st_size for p in weights)
    require(total >= 10 * GIB, "QWEN_MODEL_WEIGHT_BYTES_MISSING")
    return total


def checkpoint_bytes(config):
    """Actual FP32 edited W and native cache_c tensor bytes for one chain."""
    layers = config["hparams"]["layers"]
    require(layers and all(type(x) is int for x in layers), "INVALID_LAYER_COUNT")
    hidden, intermediate = 3584, 18944  # Locked Qwen2.5-7B-Instruct geometry.
    weight = len(layers) * hidden * intermediate * 4
    history = (len(layers) * intermediate * intermediate * 4 if
               config["method"] in ("ALPHAEDIT", "ALPHAEDIT_BLUE", "SPHERE") else 0)
    return weight + history + 8 * (1 << 20)  # RNG/context/serializer overhead.


def storage_reserve(output_root, specs, assets):
    model_bytes = model_snapshot_bytes(assets)
    sizes = []
    for spec in specs:
        if spec["kind"] in ("edit", "smoke", "qualify"):
            one_checkpoint = checkpoint_bytes(read(spec["config"]))
            # Qualification keeps continuous/B3 and resumed/B3 side by side.
            sizes.extend([one_checkpoint] * (2 if spec["kind"] == "qualify" else 1))
    # Retained W20 (and bounded qualification/smoke) plus one atomic temporary
    # copy at the largest row; raw factual/generation and filesystem headroom.
    reserve = sum(sizes) + (max(sizes) if sizes else 0) + 16 * GIB
    output_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_root).free
    require(free >= reserve, f"OUTPUT_DISK_LOW:free={free}:reserve={reserve}")
    return {"model_weight_bytes": model_bytes, "planned_checkpoint_bytes": sum(sizes),
            "atomic_peak_bytes": max(sizes) if sizes else 0,
            "raw_and_headroom_bytes": 16 * GIB,
            "reserve_bytes": reserve, "free_bytes": free}


def full_program_storage_reserve(output_root, rows):
    """Bound the complete Qwen assignment before its first GPU registration.

    Qualification's two B3 branches, all CF and zsRE W20 checkpoints and the
    independent zsRE B1 smoke remain local.  No unapproved cleanup or storage
    migration is assumed to make a later stage fit.
    """
    qualify = [rows["qwen25-cf-alphaedit_blue-l2-1"] if method == "ALPHAEDIT_BLUE"
               else rows[f"qwen25-cf-{method.lower()}"] for method in CF_NEW_CHAIN_METHODS]
    cf = [rows[f"qwen25-cf-{method.lower()}"] for method in CF_NEW_CHAIN_METHODS
          if method != "ALPHAEDIT_BLUE"]
    cf += [rows[f"qwen25-cf-alphaedit_blue-l2-{l2}"] for l2 in GRID]
    cf += [rows[f"qwen25-cf-{method.lower()}-clamp075"]
           for method in ("ALPHAEDIT", "ALPHAEDIT_BLUE")]
    zsre = [rows[f"qwen25-zsre-{method.lower()}"] for method in METHODS]
    sizes = ([checkpoint_bytes(row) for row in qualify for _ in range(2)] +
             [checkpoint_bytes(row) for row in cf] +
             [checkpoint_bytes(row) for row in zsre] +
             [checkpoint_bytes(rows["qwen25-zsre-ft"])])
    retained = sum(sizes)
    atomic_peak = max(sizes)
    reserve = retained + atomic_peak + 16 * GIB
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_root).free
    require(free >= reserve,
            f"FULL_PROGRAM_DISK_LOW:free={free}:reserve={reserve}")
    return {"retained_checkpoint_bytes": retained,
            "atomic_peak_bytes": atomic_peak,
            "raw_and_headroom_bytes": 16 * GIB,
            "reserve_bytes": reserve, "free_bytes": free,
            "qualification_checkpoint_copies": 2 * len(qualify),
            "cf_physical_chains": len(cf), "zsre_physical_chains": len(zsre),
            "zsre_smoke_checkpoints": 1,
            "cleanup_assumed": False}


def bind_runtime_python(requested, assets):
    """Bind the invocation path and binary bytes to the asset manifest."""
    declared = assets.get("runtime_python")
    require(isinstance(declared, str) and Path(declared).is_absolute(),
            "RUNTIME_PYTHON_MANIFEST_PATH_INVALID")
    declared_path = Path(declared).absolute()
    requested_path = Path(requested).absolute()
    require(requested_path == declared_path,
            "RUNTIME_PYTHON_MANIFEST_PATH_MISMATCH")
    require(requested_path.is_file() and os.access(requested_path, os.X_OK),
            "RUNTIME_PYTHON_NOT_EXECUTABLE")
    binary = member(requested_path)
    # Keep the venv/bin/python invocation path.  Resolving its symlink before
    # execution would turn off the virtual environment's package isolation.
    return requested_path, binary


def policy_cap(memory_mib, caps_file=None):
    local = (Path(caps_file) if caps_file else
             (ROOT / "servers/local/gpu-caps.tsv" if
              (ROOT / "servers/local/gpu-caps.tsv").is_file() else
              Path("/data/janghj/ODE-edit/servers/local/gpu-caps.tsv")))
    tracked = ROOT / "control" / "gpu-concurrency-policy.tsv"
    memory = ROOT / "servers" / "slurm-memory-policy.tsv"
    require(local.is_file() and tracked.is_file() and memory.is_file(), "SERVER3_POLICY_MISSING")
    def row(path):
        matches = [line.split("\t") for line in path.read_text().splitlines()
                   if line.startswith("server3\t")]
        require(len(matches) == 1, "SERVER3_POLICY_AMBIGUOUS:" + str(path))
        return matches[0]
    loc, track, mem = row(local), row(tracked), row(memory)
    require(loc[1] == NODE and mem[1] == NODE, "SERVER3_NODE_POLICY_CHANGED")
    cap = min(int(loc[2]), int(track[1]), 1)
    require(cap == 1, "SERVER3_PROJECT_CAP_BLOCKED")
    require(memory_mib > 0 and memory_mib <= min(int(loc[3]), int(mem[3]) * 1024),
            "SERVER3_MEMORY_REQUEST_EXCEEDS_POLICY")
    return {"cap": cap, "local": member(local), "tracked": member(tracked),
            "memory": member(memory), "memory_mib": memory_mib}


def check_wandb(source, env_file):
    require(os.environ.get("WANDB_MODE", "online") == "online" and
            os.environ.get("WANDB_DISABLED", "false").lower() not in ("true", "1"),
            "WANDB_ONLINE_REQUIRED")
    # This is a cheap authenticated remote read.  No run is created and no
    # credential is printed or copied into an attempt.
    require(Path(env_file).is_file(), "WANDB_ENV_FILE_MISSING")
    # --export=NONE does not carry caller credentials.  Probe only the home
    # credential source that the launcher explicitly restores.
    allow = ("PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
    env = {key: os.environ[key] for key in allow if key in os.environ}
    env.update(HOME=pwd.getpwuid(os.getuid()).pw_dir, USER=getpass.getuser(),
               PYTHONPATH=str(source), WANDB_MODE="online", CUDA_VISIBLE_DEVICES="")
    settings_script = ("import json,sys\n"
                       "from official.tracking.schema import load_env\n"
                       "settings=load_env(sys.argv[1])\n"
                       "print(json.dumps({'python':settings['ODEEDIT_WANDB_PYTHON'],"
                       "'base_url':settings['WANDB_BASE_URL']}))\n")
    settings = json.loads(command([sys.executable, "-c", settings_script, str(env_file)],
                                  cwd=source, env=env, timeout=20))
    sdk_python = Path(settings["python"])
    require(sdk_python.is_file() and os.access(sdk_python, os.X_OK), "WANDB_SDK_PYTHON_MISSING")
    script = ("import sys,wandb\n"
              "api=wandb.Api(overrides={'base_url':sys.argv[1]},timeout=20)\n"
              "projects=api.projects(entity='wkdguswns2256')\n"
              "assert any(p.name=='layer allocation' for p in projects), 'PROJECT_NOT_VISIBLE'\n")
    result = subprocess.run([str(sdk_python), "-c", script, settings["base_url"]], text=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=env, timeout=40)
    require(result.returncode == 0, "WANDB_ONLINE_AUTH_OR_PROJECT_BLOCKED")
    return {"status": "ONLINE_PROJECT_READ_VERIFIED", "entity": "wkdguswns2256",
            "project": "layer allocation", "sdk_python": str(sdk_python.resolve())}


def node_jobs():
    # -w omits pending jobs with an explicit ReqNodeList but no allocation yet.
    # Inspect all current jobs and retain every GPU request for this node.
    raw = command(["squeue", "-h", "-r", "-o", "%i|%u|%j|%T|%b|%R"])
    jobs = []
    for line in raw.splitlines():
        fields = line.split("|", 5)
        require(len(fields) == 6, "SQUEUE_PARSE")
        job_id, user, name, state, gres, reason = fields
        require(re.fullmatch(r"[0-9]+(?:_[0-9]+)?", job_id) is not None,
                "SQUEUE_JOB_ID_PARSE")
        detail = command(["scontrol", "show", "job", "-o", job_id])
        if NODE not in ((parse_field(detail, "ReqNodeList") or "") + "," +
                        (parse_field(detail, "NodeList") or "")):
            continue
        match = re.search(r"\bReqTRES=([^ ]+)", detail)
        gpu = re.search(r"gres/gpu(?:\:[^=,]+)?=(\d+)", match.group(1)) if match else None
        if not gpu:
            gpu = re.search(r"gres/gpu(?::[^: ,]+)?:([0-9]+)", gres)
        if gpu:
            jobs.append({"job": job_id, "user": user, "name": name,
                         "state": state, "gpus": int(gpu.group(1)), "reason": reason})
    return jobs


def no_duplicate_jobs(keys):
    names = {JOB_PREFIX + key for key in keys}
    raw = command(["squeue", "-h", "-r", "-u", getpass.getuser(), "-o", "%i|%j"])
    for line in raw.splitlines():
        fields = line.split("|", 1)
        require(len(fields) == 2, "SQUEUE_DEDUP_PARSE")
        require(fields[1] not in names, "DUPLICATE_ACTIVE_SERVER3_BASELINE_JOB")


def attempt_lock(output_root):
    path = Path(output_root) / ".server3-submit.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return descriptor


def safe_attempt(attempt, output_root):
    path = Path(attempt).resolve()
    root = Path(output_root).resolve()
    require(path.parent == root / "attempts" and
            re.fullmatch(r"attempt-[A-Za-z0-9_-]{1,56}", path.name) is not None,
            "ATTEMPT_MUST_BE_UNDER_OUTPUT_ROOT")
    require(not path.exists(), "ATTEMPT_ALREADY_EXISTS")
    return path


def registered_keys(output_root):
    keys = set()
    for path in (Path(output_root) / "attempts").glob("attempt-*/registered-*.json"):
        receipt = read(path)
        keys.add(receipt["key"])
    return keys


def launcher(spec, source_lock, python, stream, assets, attempt, wandb_env, resume=False):
    source = Path(source_lock["source"])
    module = "official.runners.server3.run"
    common = ["--assets", assets["path"], "--stream", stream["path"],
              "--output", spec["output"]]
    if spec["kind"] == "w0":
        args = ["w0", "--dataset", spec["dataset"], *common]
    elif spec["kind"] == "qualify":
        args = ["qualify", "--config", spec["config"], *common]
    else:
        args = ["execute", "--config", spec["config"], *common]
        if spec["kind"] == "smoke":
            args += ["--stop-after-batch", "1"]
        elif resume:
            args.append("--resume")
    verify = [str(python), "-B", "-m", "official.runners.server3.submit",
              "verify-runtime", "--attempt", str(attempt), "--key", spec["key"]]
    run = [str(python), "-B", "-m", module, *args]
    script = ("#!/bin/bash\nset -euo pipefail\n"
              "export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
              "export WANDB_MODE=online WANDB_CONSOLE=off WANDB_SAVE_CODE=false WANDB_DISABLE_CODE=true\n"
              "export WANDB_ENTITY=wkdguswns2256\n"
              "export WANDB_PROJECT='layer allocation'\n"
              "export PYTHONPATH=" + shlex.quote(str(source)) + "\n"
              "export HOME=" + shlex.quote(pwd.getpwuid(os.getuid()).pw_dir) + "\n"
              "export ODEEDIT_SOURCE_LOCK=" + shlex.quote(str(attempt / "source-lock.json")) + "\n"
              "export ODEEDIT_CODE_COMMIT=" + shlex.quote(source_lock["code_commit"]) + "\n"
              "export ODEEDIT_OFFICIAL_TREE_SHA256=" + shlex.quote(source_lock["official_tree_sha256"]) + "\n"
              "export ODEEDIT_WANDB_ENV_FILE=" + shlex.quote(str(wandb_env)) + "\n"
              "export ODEEDIT_ATTEMPT_ID=" + shlex.quote(attempt.name + "-" + spec["key"]) + "\n"
              "cd " + shlex.quote(str(source)) + "\n"
              + shlex.join(verify) + "\n"
              "export ODEEDIT_WANDB_PROJECT_VERIFIED=1\n"
              "exec " + shlex.join(run) + "\n")
    path = attempt / "launchers" / (spec["key"] + ".sh")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream_file:
        stream_file.write(script)
    path.chmod(0o700)
    return member(path)


def preflight(specs, python, source_lock, stream, assets, attempt, *, resume=False):
    source = Path(source_lock["source"])
    env = {**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1",
           "WANDB_MODE": "online", "WANDB_ENTITY": "wkdguswns2256",
           "WANDB_PROJECT": "layer allocation",
           "ODEEDIT_WANDB_PROJECT_VERIFIED": "1",
           "ODEEDIT_SOURCE_LOCK": str(attempt / "source-lock.json"),
           "ODEEDIT_CODE_COMMIT": source_lock["code_commit"],
           "ODEEDIT_OFFICIAL_TREE_SHA256": source_lock["official_tree_sha256"]}
    for spec in specs:
        if spec["kind"] == "w0":
            continue  # Each editing config checks the same W0 asset closure.
        argv = [str(python), "-B", "-m", "official.runners.server3.run", "preflight",
                "--config", spec["config"], "--assets", assets["path"],
                "--stream", stream["path"], "--output", spec["output"]]
        if resume:
            argv.append("--resume")
        command(argv, cwd=source, env=env, timeout=300)


def sbatch_args(spec, attempt, memory_mib, wall, dependency, source=None):
    source = Path(source) if source is not None else attempt / "source"
    argv = ["sbatch", "--parsable", "--hold", "--partition=" + PARTITION,
            "--qos=" + QOS, "--nodelist=" + NODE, "--nodes=1", "--ntasks=1",
            "--cpus-per-task=8", "--gres=gpu:1", "--export=NONE", "--no-requeue",
            "--mem=" + str(memory_mib) + "M", "--time=" + wall,
            "--job-name=" + JOB_PREFIX + spec["key"],
            "--chdir=" + str(source),
            "--output=" + str(attempt / "logs" / (spec["key"] + "-%j.out")),
            "--error=" + str(attempt / "logs" / (spec["key"] + "-%j.err"))]
    if dependency:
        argv.append("--dependency=" + dependency)
    return argv + [str(attempt / "launchers" / (spec["key"] + ".sh"))]


def parse_field(detail, key):
    match = re.search(r"(?:^| )" + re.escape(key) + r"=([^ ]+)", detail)
    return match.group(1) if match else None


def dependency_set(value):
    if value in (None, "", "(null)"):
        return set()
    return set(re.sub(r"\([^)]*\)", "", value).split(","))


def inspect_one(job, spec, attempt, expected_dependency, memory_mib, source=None):
    source = Path(source) if source is not None else attempt / "source"
    detail = command(["scontrol", "show", "job", "-o", job])
    require(parse_field(detail, "JobId") == job and
            parse_field(detail, "JobName") == JOB_PREFIX + spec["key"] and
            (parse_field(detail, "UserId") or "").startswith(getpass.getuser() + "("),
            "HELD_JOB_IDENTITY_MISMATCH:" + job)
    require(parse_field(detail, "JobState") == "PENDING" and
            parse_field(detail, "Reason") == "JobHeldUser", "JOB_NOT_HELD:" + job)
    require(parse_field(detail, "ReqNodeList") == NODE and
            parse_field(detail, "Partition") == PARTITION and
            parse_field(detail, "QOS") == QOS and
            parse_field(detail, "Requeue") == "0", "HELD_RESOURCE_MISMATCH:" + job)
    require(parse_field(detail, "Command") == str(attempt / "launchers" / (spec["key"] + ".sh")) and
            parse_field(detail, "WorkDir") == str(source),
            "HELD_COMMAND_MISMATCH:" + job)
    require("gres/gpu:1" in detail and
            (("mem=" + str(memory_mib) + "M") in detail or
             ("mem=" + str(memory_mib // 1024) + "G") in detail),
            "HELD_GPU_OR_MEMORY_MISMATCH:" + job)
    actual_dep = parse_field(detail, "Dependency")
    require(dependency_set(actual_dep) == dependency_set(expected_dependency),
            "HELD_DEPENDENCY_MISMATCH:" + job)
    observed = command(["scontrol", "write", "batch_script", job, "-"])
    script = (attempt / "launchers" / (spec["key"] + ".sh")).read_text().strip()
    require(observed.strip() == script, "HELD_SCRIPT_BYTES_CHANGED:" + job)
    return {"job": job, "key": spec["key"], "detail": detail,
            "dependency": expected_dependency}


def verify_runtime(attempt, key):
    attempt = Path(attempt).resolve(strict=True)
    lock = read(attempt / "execution.lock.json")
    require(key in {s["key"] for s in lock["specs"]}, "UNKNOWN_LAUNCHER_KEY")
    if lock["stage"] in ("cf", "zsre"):
        require("source_parent_attempt" in lock, "INHERITED_SOURCE_PARENT_MISSING")
    source_owner = Path(lock["archive"]["path"]).parent
    verify_source_snapshot(source_owner, lock)
    if "source_parent_attempt" in lock:
        parent_attempt = Path(lock["source_parent_attempt"]).resolve(strict=True)
        origin_stage = (read(Path(lock["origin_attempt"]) / "execution.lock.json")["stage"]
                        if lock["stage"] == "resume" else lock["stage"])
        expected_stage = {"cf": "qualify", "zsre": "cf"}.get(origin_stage)
        require(expected_stage is not None and
                parent_attempt.parent == Path(lock["output_root"]) / "attempts" and
                (parent_attempt / "release.json").is_file(), "SOURCE_PARENT_SCOPE_OR_RELEASE")
        parent = read(parent_attempt / "execution.lock.json")
        require(parent["stage"] == expected_stage and
                parent["output_root"] == lock["output_root"] and
                parent["owner"] == lock["owner"] and
                parent["code_commit"] == lock["code_commit"] and
                parent["official_tree_sha256"] == lock["official_tree_sha256"] and
                parent["archive"]["sha256"] == lock["archive"]["sha256"],
                "SOURCE_PARENT_IDENTITY_CHANGED")
        verify_source_snapshot(parent_attempt, parent)
    for record in lock["inputs"]:
        verify_member(record)
    require(Path(lock["python"]).resolve(strict=True) ==
            Path(lock["runtime_python_binary"]["path"]),
            "RUNTIME_PYTHON_SYMLINK_TARGET_CHANGED")
    verify_member(lock["runtime_python_binary"])
    for record in lock["agent_seals"]:
        verify_member(record)
    for script in lock["launchers"]:
        verify_member(script)
    check_wandb(lock["source"], lock["wandb_env"])
    if lock["stage"] == "cf":
        verify_qualification(Path(lock["output_root"]), source=lock)
    if lock["stage"] == "zsre" and next(s for s in lock["specs"]
                                           if s["key"] == key)["kind"] == "edit":
        verify_zsre_smoke(attempt, lock)
    if lock["stage"] == "resume":
        origin_attempt = Path(lock["origin_attempt"])
        origin = read(origin_attempt / "execution.lock.json")
        if origin["stage"] == "cf":
            verify_qualification(Path(lock["output_root"]), source=lock)
        elif origin["stage"] == "zsre":
            verify_zsre_smoke(origin_attempt, origin)
    require(lock["code_commit"] and lock["official_tree_sha256"], "SOURCE_IDENTITY_MISSING")
    return {"status": "FROZEN_INPUTS_VALID", "key": key,
            "code_commit": lock["code_commit"],
            "official_tree_sha256": lock["official_tree_sha256"]}


def accounted_job(job, key, attempt, source):
    """Use durable accounting for terminal jobs after Slurm purges scontrol."""
    require(str(job).isdigit(), "ACCOUNTING_JOB_ID_INVALID")
    raw = command(["sacct", "-X", "-j", str(job), "-n", "-P",
                   "--format=JobIDRaw,State%50,ExitCode,JobName%100,User%64,SubmitLine%2000,WorkDir%500"])
    rows = [line.split("|", 6) for line in raw.splitlines() if line.strip()]
    require(len(rows) == 1 and len(rows[0]) == 7 and rows[0][0] == str(job),
            "ACCOUNTING_ROW_MISSING_OR_AMBIGUOUS:" + str(job))
    job_id, state, exit_code, name, user, submit_line, workdir = rows[0]
    script = str(Path(attempt) / "launchers" / (key + ".sh"))
    require(name == JOB_PREFIX + key and user == getpass.getuser() and
            workdir == str(source), "ACCOUNTING_JOB_IDENTITY_MISMATCH:" + job_id)
    require(submit_line.startswith("sbatch ") and
            shlex.split(submit_line)[-1] == script,
            "ACCOUNTING_SUBMIT_SCRIPT_MISMATCH:" + job_id)
    return {"job": job_id, "state": state.split()[0], "exit_code": exit_code,
            "job_name": name, "script": script, "workdir": workdir}


def successful_job(job, key, attempt, source):
    row = accounted_job(job, key, attempt, source)
    require(row["state"] == "COMPLETED" and row["exit_code"] == "0:0",
            "PREREQUISITE_JOB_NOT_SUCCESSFUL:" + str(job))
    return row


def verify_qualification(output_root, source=None):
    root = Path(output_root).resolve()
    attempts = []
    for path in (root / "attempts").glob("attempt-*/execution.lock.json"):
        value = read(path)
        if value.get("stage") == "qualify" and (path.parent / "release.json").is_file():
            attempts.append((path.parent, value))
    require(len(attempts) == 1, "CF_GPU_QUALIFICATION_NOT_UNIQUE_OR_MISSING")
    attempt, lock = attempts[0]
    if source:
        require(lock["code_commit"] == source["code_commit"] and
                lock["official_tree_sha256"] == source["official_tree_sha256"],
                "CF_QUALIFICATION_SOURCE_MISMATCH")
    stream_sha = next(x["sha256"] for x in lock["inputs"]
                      if x["path"].endswith("cf-stream.json"))
    expected_specs = [("qwen25-cf-w0", "w0", None)] + [
        ("qwen25-cf-qualify-" + method.lower(), "qualify", method)
        for method in CF_NEW_CHAIN_METHODS]
    actual_specs = [(item.get("key"), item.get("kind"), item.get("method"))
                    for item in lock["specs"]]
    require(actual_specs == expected_specs,
            "CF_QUALIFICATION_DAG_MISMATCH")
    evidence = []
    for spec in lock["specs"]:
        registration = read(attempt / ("registered-" + spec["key"] + ".json"))
        successful_job(registration["job"], spec["key"], attempt, lock["source"])
        if spec["kind"] == "w0":
            path = Path(spec["output"]) / "w0-receipt.json"
            require(path.is_file(), "CF_W0_RECEIPT_MISSING")
            receipt = read(path)
            expected = {"model": "qwen25", "dataset": "cf", "endpoint": "W0",
                        "stream_sha256": stream_sha, "code_commit": lock["code_commit"],
                        "official_tree_sha256": lock["official_tree_sha256"],
                        "observed_requests": 2000}
            require(all(receipt.get(k) == v for k, v in expected.items()),
                    "CF_W0_RECEIPT_IDENTITY_MISMATCH")
            require(receipt.get("receipt_sha256") == digest({
                k: v for k, v in receipt.items() if k != "receipt_sha256"}),
                "CF_W0_RECEIPT_DIGEST_MISMATCH")
            cases = Path(spec["output"]) / "w0-cases.json"
            require(cases.is_file() and file_sha(cases) ==
                    receipt.get("factual", {}).get("cases_sha256") and
                    receipt.get("generation_identity_sha256"),
                    "CF_W0_SCIENCE_PROVENANCE_MISSING")
        else:
            path = Path(spec["output"]) / "resume-parity.json"
            require(path.is_file(), "METHOD_GPU_QUALIFICATION_MISSING:" + spec["method"])
            receipt = read(path)
            expected = {"status": "PASS", "method": spec["method"],
                        "run_id": spec["run_id"],
                        "config_sha256": read(spec["config"])["config_sha256"],
                        "stream_sha256": stream_sha,
                        "code_commit": lock["code_commit"],
                        "official_tree_sha256": lock["official_tree_sha256"],
                        "completed_batch": 3}
            require(all(receipt.get(k) == v for k, v in expected.items()),
                    "METHOD_GPU_QUALIFICATION_IDENTITY_OR_PARITY_MISSING:" + spec["method"])
            expected_checks = ("weight_history_context_rng_cursor_hashes_equal",
                               "b3_raw_factual_equal", "b3_summary_equal",
                               "identical_checkpoint_identity",
                               "reloaded_exact_b2_checkpoint")
            checks = receipt.get("checks", {})
            require(all(checks.get(name) is True for name in expected_checks) and
                    receipt.get("state_hashes") and receipt.get("factual_sha256"),
                    "METHOD_GPU_QUALIFICATION_NOT_EXACT:" + spec["method"])
        evidence.append(member(path))
    return {"attempt": str(attempt), "receipts": evidence,
            "code_commit": lock["code_commit"],
            "official_tree_sha256": lock["official_tree_sha256"]}


def verify_zsre_smoke(attempt, lock):
    smoke = next(s for s in lock["specs"] if s["kind"] == "smoke")
    registration = read(Path(attempt) / ("registered-" + smoke["key"] + ".json"))
    successful_job(registration["job"], smoke["key"], attempt, lock["source"])
    path = Path(smoke["output"]) / "smoke-receipt.json"
    require(path.is_file(), "ZSRE_B1_SMOKE_RECEIPT_MISSING")
    receipt = read(path)
    stream_sha = next(x["sha256"] for x in lock["inputs"]
                      if x["path"].endswith("zsre-stream.json"))
    expected = {"completed_batch": 1, "run_id": smoke["run_id"],
                "config_sha256": read(smoke["config"])["config_sha256"],
                "stream_sha256": stream_sha,
                "code_commit": lock["code_commit"],
                "official_tree_sha256": lock["official_tree_sha256"]}
    require(all(receipt.get(k) == v for k, v in expected.items()) and
            receipt.get("checkpoint_identity"), "ZSRE_B1_SMOKE_INVALID")
    return member(path)


def select_blue(matrix_root, output_root, cf_attempt):
    rows, files = matrix_rows(matrix_root)
    attempt = Path(cf_attempt).resolve(strict=True)
    lock = read(attempt / "execution.lock.json")
    require(lock["stage"] == "cf" and Path(lock["output_root"]) == Path(output_root).resolve(),
            "CF_ATTEMPT_SCOPE")
    stream_sha = next(x["sha256"] for x in lock["inputs"]
                      if x["path"].endswith("cf-stream.json"))
    scored = []
    for l2 in GRID:
        run_id = f"qwen25-cf-alphaedit_blue-l2-{l2}"
        job_file = attempt / ("registered-" + run_id + ".json")
        require(job_file.is_file(), "GRID_JOB_NOT_REGISTERED:" + run_id)
        registration = read(job_file)
        successful_job(registration["job"], run_id, attempt, lock["source"])
        result = Path(output_root) / "runs" / run_id / "evaluations" / "w20.json"
        require(result.is_file(), "GRID_W20_MISSING:" + run_id)
        value = read(result)
        expected = {"run_id": run_id, "config_sha256": rows[run_id]["config_sha256"],
                    "stream_sha256": stream_sha, "code_commit": lock["code_commit"],
                    "official_tree_sha256": lock["official_tree_sha256"],
                    "completed_batch": 20}
        require(all(value.get(k) == v for k, v in expected.items()),
                "GRID_W20_IDENTITY_MISMATCH:" + run_id)
        require(value.get("checkpoint_sha256") and value.get("factual_sha256"),
                "GRID_W20_PROVENANCE_MISSING:" + run_id)
        run_root = Path(output_root) / "runs" / run_id
        pointer = run_root / "checkpoint" / "latest.json"
        require(pointer.is_file(), "GRID_W20_CHECKPOINT_POINTER_MISSING:" + run_id)
        checkpoint = read(pointer)
        filename = checkpoint.get("file", "")
        checkpoint_file = pointer.parent / filename
        require(Path(filename).name == filename and
                checkpoint.get("batch") == 20 and
                checkpoint.get("sha256") == value["checkpoint_sha256"] and
                checkpoint == value.get("checkpoint") and
                checkpoint_file.is_file() and
                file_sha(checkpoint_file) == checkpoint["sha256"],
                "GRID_W20_CHECKPOINT_HASH_MISMATCH:" + run_id)
        cases = run_root / "evaluations" / "w20-cases.json"
        require(cases.is_file() and
                value.get("factual", {}).get("cases_path") == str(cases.resolve()) and
                file_sha(cases) == value["factual_sha256"] ==
                value["factual"]["cases_sha256"],
                "GRID_W20_FACTUAL_HASH_MISMATCH:" + run_id)
        generation = value.get("generation") or {}
        require(generation.get("identity_sha256") and
                Path(generation.get("rows_path", "")).is_file(),
                "GRID_W20_GENERATION_PROVENANCE_MISSING:" + run_id)
        summary = value.get("factual", {}).get("summary", {})
        keys = ("Score", "Specificity", "Efficacy", "Generalization")
        require(all(type(summary.get(k)) in (float, int) and
                    math.isfinite(summary[k]) for k in keys),
                "GRID_UNROUNDED_METRICS_MISSING:" + run_id)
        scored.append({"run_id": run_id, "l2": l2, "job": registration["job"],
                       "metrics": {k: summary[k] for k in keys}, "w20": member(result)})
    winner = choose_blue_winner(scored)
    original = copy.deepcopy(rows["qwen25-zsre-alphaedit_blue"])
    require(original["hparams"]["L2"] is None, "ZSRE_BLUE_ALREADY_SELECTED")
    original["hparams"]["L2"] = winner["l2"]
    original["selection_dependency"] = "qwen25-cf-blue-l2-grid"
    original["config_sha256"] = digest({k: v for k, v in original.items()
                                        if k != "config_sha256"})
    selected_path = Path(output_root) / "selected-configs" / "qwen25-zsre-alphaedit_blue.json"
    write_new(selected_path, original)
    selection = {"status": "CF_BLUE_GRID_SELECTED", "source_attempt": str(attempt),
                 "code_commit": lock["code_commit"],
                 "official_tree_sha256": lock["official_tree_sha256"],
                 "criteria": ["Score", "Specificity", "Efficacy", "Generalization", "lowest_L2"],
                 "grid": scored, "winner_run_id": winner["run_id"], "chosen_L2": winner["l2"],
                 "selected_zsre_config": str(selected_path),
                 "selected_zsre_config_sha256": file_sha(selected_path)}
    selection_path = Path(output_root) / "selected-configs" / "qwen25-blue-selection.json"
    write_new(selection_path, selection)
    alias = {"run_id": "qwen25-cf-alphaedit_blue", "execution_kind": "selected_grid_alias",
             "winner_run_id": winner["run_id"], "chosen_L2": winner["l2"],
             "source_w20": winner["w20"], "selection": member(selection_path),
             "no_new_edit_chain": True}
    write_new(Path(output_root) / "aliases" / "qwen25-cf-alphaedit_blue.json", alias)
    return selection


def choose_blue_winner(scored):
    require({row["l2"] for row in scored} == set(GRID) and len(scored) == len(GRID),
            "BLUE_GRID_INCOMPLETE")
    keys = ("Score", "Specificity", "Efficacy", "Generalization")
    for row in scored:
        require(all(type(row["metrics"].get(k)) in (int, float) and
                    math.isfinite(row["metrics"][k]) for k in keys),
                "BLUE_GRID_METRICS_INVALID")
    return max(scored, key=lambda row: tuple(row["metrics"][k] for k in keys)
               + (-row["l2"],))


def submit_held(args):
    assets_path = Path(args.assets).resolve(strict=True)
    assets_data = read(assets_path)
    output_root = Path(assets_data["output_root"]).resolve()
    require(output_root.is_absolute() and output_root != ROOT and
            not str(output_root).startswith(str(ROOT / "official")), "OUTPUT_ROOT_INVALID")
    output_root.mkdir(parents=True, exist_ok=True)
    lock_descriptor = attempt_lock(output_root)
    try:
        attempt = safe_attempt(args.attempt, output_root)
        rows, files = matrix_rows(args.matrix_root)
        selection = None
        if args.stage == "zsre":
            require(args.selection is not None, "ZSRE_BLUE_SELECTION_REQUIRED")
            selection = read(args.selection)
            require(selection.get("status") == "CF_BLUE_GRID_SELECTED" and
                    selection.get("chosen_L2") in GRID, "BLUE_SELECTION_INVALID")
            require(Path(args.selection).resolve() == output_root / "selected-configs" /
                    "qwen25-blue-selection.json", "BLUE_SELECTION_SCOPE")
            recomputed = select_blue(args.matrix_root, output_root,
                                     selection["source_attempt"])
            require(recomputed == selection, "BLUE_SELECTION_CHANGED")
            require(selection["selected_zsre_config_sha256"] ==
                    file_sha(selection["selected_zsre_config"]), "SELECTED_CONFIG_CHANGED")
        specs = stage_specs(args.stage, rows, files, output_root, selection)
        qualification = verify_qualification(output_root) if args.stage == "cf" else None
        existing = registered_keys(output_root)
        require(not existing.intersection(s["key"] for s in specs),
                "PREVIOUS_REGISTRATION_EXISTS_USE_RESUME")
        no_duplicate_jobs(s["key"] for s in specs)
        dataset = "cf" if args.stage in ("qualify", "cf") else "zsre"
        stream, stream_lock = stream_member(args.matrix_root, dataset)
        policy = policy_cap(args.memory_mib, args.gpu_caps)
        storage = storage_reserve(output_root, specs, assets_data)
        if args.stage == "qualify":
            storage["whole_program"] = full_program_storage_reserve(output_root, rows)
        python, runtime_binary = bind_runtime_python(args.python, assets_data)
        attempt.mkdir(parents=True)
        (attempt / "logs").mkdir()
        source_lock = stage_source(args.stage, attempt, output_root, qualification, selection)
        source = Path(source_lock["source"])
        if args.stage == "cf":
            verify_qualification(output_root, source=source_lock)
        if args.stage == "zsre":
            require(selection["code_commit"] == source_lock["code_commit"] and
                    selection["official_tree_sha256"] ==
                    source_lock["official_tree_sha256"], "ZSRE_SELECTION_SOURCE_MISMATCH")
            verify_qualification(output_root, source=source_lock)
        write_new(attempt / "source-lock.json", {
            "code_commit": source_lock["code_commit"],
            "official_tree_sha256": source_lock["official_tree_sha256"]})
        wandb = check_wandb(source, args.wandb_env)
        require(Path(assets_data.get("wandb_env", "")).resolve() ==
                Path(args.wandb_env).resolve() and
                Path(assets_data.get("wandb_sdk_python", "")).resolve() ==
                Path(wandb["sdk_python"]), "WANDB_ASSET_BINDING_MISMATCH")
        assets = member(assets_path)
        preflight(specs, python, source_lock, stream, assets, attempt)
        inputs = [assets, stream, stream_lock, member(attempt / "source-lock.json"),
                  runtime_binary,
                  member(args.wandb_env),
                  policy["local"], policy["tracked"],
                  policy["memory"]]
        inputs.extend(member(s["config"]) for s in specs if "config" in s)
        if args.selection:
            inputs.append(member(args.selection))
        launchers = [launcher(s, source_lock, python, stream, assets, attempt,
                              Path(args.wandb_env).resolve()) for s in specs]
        execution = {**source_lock, "stage": args.stage, "owner": getpass.getuser(),
                     "output_root": str(output_root), "python": str(python),
                     "runtime_python_binary": runtime_binary,
                     "wandb_env": str(Path(args.wandb_env).resolve()),
                     "assets_path": str(assets_path), "stream_path": stream["path"],
                     "specs": specs, "inputs": inputs, "launchers": launchers,
                     "storage": storage, "policy": policy, "wandb": wandb,
                     "created_utc": datetime.now(timezone.utc).isoformat()}
        write_new(attempt / "execution.lock.json", execution)
        before = node_jobs()
        external = [j["job"] for j in before if j["gpus"] > 0]
        # Cap one: wait for all observed node GPU work.  Existing jobs are
        # grandfathered and never cancelled or modified.
        previous = None
        registered = []
        for spec in specs:
            deps = []
            if previous:
                deps.append("afterok:" + previous)
            elif external:
                deps.append("afterany:" + ":".join(external))
            dependency = ",".join(deps)
            argv = sbatch_args(spec, attempt, args.memory_mib, args.wall, dependency)
            job = command(argv).split(";")[0]
            require(job.isdigit(), "SBATCH_ID_INVALID")
            receipt = {"key": spec["key"], "job": job, "dependency": dependency,
                       "argv": argv, "state": "HELD"}
            write_new(attempt / ("registered-" + spec["key"] + ".json"), receipt)
            registered.append(receipt)
            previous = job
        inspected = [inspect_one(r["job"], spec, attempt, r["dependency"], args.memory_mib)
                     for r, spec in zip(registered, specs)]
        current = node_jobs()
        other = [j for j in current if j["job"] not in {x["job"] for x in registered}]
        require({j["job"] for j in other} <= {j["job"] for j in before},
                "ADMISSION_RACE_KEEP_HELD")
        verify_runtime(attempt, specs[0]["key"])
        inspection = {"status": "ALL_HELD_INSPECTED", "jobs": inspected,
                      "existing_before": before, "existing_before_release": other,
                      "external_afterany": external, "max_new_concurrent_gpus": 1,
                      "old_jobs_mutated": False}
        write_new(attempt / "held-inspection.json", inspection)
        return {"status": "HELD", "attempt": str(attempt),
                "jobs": {x["key"]: x["job"] for x in registered}}
    finally:
        os.close(lock_descriptor)


def resume_held(args):
    """Register one held recovery job against the exact prior frozen chain.

    The immediate downstream job, if one exists, is rewired only by release()
    after the new held job and the old pending job have both been inspected.
    """
    prior_attempt = Path(args.prior_attempt).resolve(strict=True)
    prior = read(prior_attempt / "execution.lock.json")
    output_root = Path(prior["output_root"])
    descriptor = attempt_lock(output_root)
    try:
        attempt = safe_attempt(args.attempt, output_root)
        require(prior["owner"] == getpass.getuser() and
                (prior_attempt / "release.json").is_file(),
                "PRIOR_ATTEMPT_NOT_RELEASED_OR_NOT_OWNED")
        match = [s for s in prior["specs"] if s["key"] == args.run_id and s["kind"] == "edit"]
        require(len(match) == 1, "RESUME_RUN_NOT_IN_PRIOR_ATTEMPT")
        spec = match[0]
        verify_runtime(prior_attempt, spec["key"])
        prior_registration = read(prior_attempt / ("registered-" + spec["key"] + ".json"))
        old_job = prior_registration["job"]
        prior_job = accounted_job(old_job, spec["key"], prior_attempt, prior["source"])
        require(prior_job["state"] in
                ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "CANCELLED"),
                "PRIOR_JOB_NOT_TERMINAL_FAILURE")
        require(not (Path(spec["output"]) / "evaluations" / "w20.json").exists(),
                "W20_EXISTS_REVIEW_BEFORE_RESUME")
        origin_attempt = Path(prior.get("origin_attempt", prior_attempt)).resolve(strict=True)
        origin = read(origin_attempt / "execution.lock.json")
        require(origin["stage"] in ("cf", "zsre"), "RESUME_ORIGIN_STAGE_INVALID")
        position = next(i for i, row in enumerate(origin["specs"])
                        if row["key"] == spec["key"])
        descendants = origin["specs"][position + 1:]
        successor = None
        if descendants:
            successor_spec = descendants[0]
            successor_receipt = read(origin_attempt / (
                "registered-" + successor_spec["key"] + ".json"))
            successor_job = successor_receipt["job"]
            successor_detail = command(["scontrol", "show", "job", "-o", successor_job])
            require(parse_field(successor_detail, "JobState") == "PENDING" and
                    (parse_field(successor_detail, "UserId") or "").startswith(
                        getpass.getuser() + "(") and
                    parse_field(successor_detail, "Command") == str(
                        origin_attempt / "launchers" / (successor_spec["key"] + ".sh")) and
                    dependency_set(parse_field(successor_detail, "Dependency")) ==
                    {"afterok:" + old_job}, "DOWNSTREAM_JOB_CHANGED")
            successor = {"job": successor_job, "key": successor_spec["key"],
                         "attempt": str(origin_attempt),
                         "old_dependency": "afterok:" + old_job}
        no_duplicate_jobs([spec["key"]])
        policy = policy_cap(args.memory_mib, args.gpu_caps)
        assets_path = Path(prior["assets_path"])
        assets_data = read(assets_path)
        require(Path(assets_data["output_root"]).resolve() == output_root,
                "RESUME_OUTPUT_ROOT_CHANGED")
        storage = storage_reserve(output_root, [spec], assets_data)
        source_lock = {k: prior[k] for k in ("code_commit", "official_tree_sha256",
                        "archive", "source", "source_members",
                        "agent_seals")}
        if "source_parent_attempt" in prior:
            source_lock["source_parent_attempt"] = prior["source_parent_attempt"]
        source = Path(source_lock["source"])
        python = Path(prior["python"])
        require(python.is_file() and os.access(python, os.X_OK), "RUNTIME_PYTHON_MISSING")
        require(python.resolve(strict=True) ==
                Path(prior["runtime_python_binary"]["path"]),
                "RUNTIME_PYTHON_SYMLINK_TARGET_CHANGED")
        wandb = check_wandb(source, prior["wandb_env"])
        stream = member(prior["stream_path"])
        assets = member(assets_path)
        attempt.mkdir(parents=True)
        (attempt / "logs").mkdir()
        write_new(attempt / "source-lock.json", {
            "code_commit": source_lock["code_commit"],
            "official_tree_sha256": source_lock["official_tree_sha256"]})
        preflight([spec], python, source_lock, stream, assets, attempt, resume=True)
        inputs = [assets, stream, member(attempt / "source-lock.json"),
                  prior["runtime_python_binary"],
                  member(prior["wandb_env"]), member(spec["config"]),
                  policy["local"], policy["tracked"], policy["memory"]]
        launcher_member = launcher(spec, source_lock, python, stream, assets,
                                   attempt, prior["wandb_env"], resume=True)
        execution = {**source_lock, "stage": "resume", "owner": getpass.getuser(),
                     "origin_attempt": str(origin_attempt), "prior_attempt": str(prior_attempt),
                     "prior_job": old_job, "successor": successor,
                     "output_root": str(output_root), "python": str(python),
                     "runtime_python_binary": prior["runtime_python_binary"],
                     "wandb_env": prior["wandb_env"], "assets_path": str(assets_path),
                     "stream_path": stream["path"], "specs": [spec], "inputs": inputs,
                     "launchers": [launcher_member], "storage": storage,
                     "policy": policy, "wandb": wandb,
                     "created_utc": datetime.now(timezone.utc).isoformat()}
        write_new(attempt / "execution.lock.json", execution)
        before = node_jobs()
        descendant_jobs = {read(origin_attempt / ("registered-" + s["key"] + ".json"))["job"]
                           for s in descendants}
        external = [j["job"] for j in before if j["gpus"] > 0 and
                    j["job"] not in descendant_jobs]
        dependency = "afterany:" + ":".join(external) if external else ""
        argv = sbatch_args(spec, attempt, args.memory_mib, args.wall, dependency, source)
        job = command(argv).split(";")[0]
        require(job.isdigit(), "SBATCH_ID_INVALID")
        registration = {"key": spec["key"], "job": job, "dependency": dependency,
                        "argv": argv, "state": "HELD", "resume_of": old_job}
        write_new(attempt / ("registered-" + spec["key"] + ".json"), registration)
        inspected = inspect_one(job, spec, attempt, dependency, args.memory_mib, source)
        current = node_jobs()
        require({j["job"] for j in current if j["job"] != job} <=
                {j["job"] for j in before}, "ADMISSION_RACE_KEEP_HELD")
        verify_runtime(attempt, spec["key"])
        write_new(attempt / "held-inspection.json", {
            "status": "ALL_HELD_INSPECTED", "jobs": [inspected],
            "existing_before": before, "external_afterany": external,
            "max_new_concurrent_gpus": 1, "old_jobs_mutated": False})
        return {"status": "HELD_RESUME", "attempt": str(attempt),
                "run_id": spec["key"], "job": job, "prior_job": old_job,
                "successor_pending": successor}
    finally:
        os.close(descriptor)


def release(attempt):
    attempt = Path(attempt).resolve(strict=True)
    lock = read(attempt / "execution.lock.json")
    root = Path(lock["output_root"])
    descriptor = attempt_lock(root)
    try:
        require((attempt / "held-inspection.json").is_file(), "HELD_INSPECTION_REQUIRED")
        require(not (attempt / "release.json").exists(), "ALREADY_RELEASED")
        verify_runtime(attempt, lock["specs"][0]["key"])
        require(shutil.disk_usage(root).free >= lock["storage"]["reserve_bytes"],
                "OUTPUT_DISK_LOW_BEFORE_RELEASE")
        policy_cap(lock["policy"]["memory_mib"], lock["policy"]["local"]["path"])
        check_wandb(lock["source"], lock["wandb_env"])
        receipts = [read(attempt / ("registered-" + s["key"] + ".json"))
                    for s in lock["specs"]]
        for spec, receipt in zip(lock["specs"], receipts):
            inspect_one(receipt["job"], spec, attempt, receipt["dependency"],
                        lock["policy"]["memory_mib"], lock["source"])
        # A new unrelated allocation between registration and release keeps
        # everything held.  New jobs need an explicit dependency update first.
        before = {j["job"] for j in read(attempt / "held-inspection.json")["existing_before"]}
        now = node_jobs()
        own = {r["job"] for r in receipts}
        require({j["job"] for j in now if j["job"] not in own} <= before,
                "NEW_NODE_JOB_SINCE_INSPECTION_KEEP_HELD")
        successor = lock.get("successor")
        if successor:
            successor_attempt = Path(successor["attempt"])
            successor_detail = command(["scontrol", "show", "job", "-o", successor["job"]])
            require(parse_field(successor_detail, "JobState") == "PENDING" and
                    (parse_field(successor_detail, "UserId") or "").startswith(
                        getpass.getuser() + "(") and
                    parse_field(successor_detail, "Command") == str(
                        successor_attempt / "launchers" / (successor["key"] + ".sh")) and
                    dependency_set(parse_field(successor_detail, "Dependency")) ==
                    {successor["old_dependency"]}, "DOWNSTREAM_CHANGED_KEEP_HELD")
            new_dependency = "afterok:" + receipts[0]["job"]
            command(["scontrol", "update", "JobId=" + successor["job"],
                     "Dependency=" + new_dependency])
            observed = command(["scontrol", "show", "job", "-o", successor["job"]])
            require(dependency_set(parse_field(observed, "Dependency")) ==
                    {new_dependency}, "DOWNSTREAM_REWIRE_NOT_VERIFIED_KEEP_HELD")
            write_new(attempt / "downstream-rewire.json", {
                "job": successor["job"], "key": successor["key"],
                "old_dependency": successor["old_dependency"],
                "new_dependency": new_dependency,
                "same_existing_job_and_script": True})
        for receipt in reversed(receipts):
            command(["scontrol", "release", receipt["job"]])
            write_new(attempt / ("released-" + receipt["key"] + ".json"),
                      {"key": receipt["key"], "job": receipt["job"], "released": True})
        result = {"status": "RELEASED", "attempt": str(attempt),
                  "jobs": {r["key"]: r["job"] for r in receipts},
                  "code_commit": lock["code_commit"],
                  "official_tree_sha256": lock["official_tree_sha256"]}
        write_new(attempt / "release.json", result)
        return result
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="Print the exact physical stage DAG without mutation")
    p.add_argument("--matrix-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--stage", choices=("qualify", "cf", "zsre"), required=True)
    p.add_argument("--selection", type=Path)
    p = sub.add_parser("submit-held", help="Preflight, freeze and register every stage job held")
    p.add_argument("--matrix-root", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--wandb-env", type=Path, required=True)
    p.add_argument("--stage", choices=("qualify", "cf", "zsre"), required=True)
    p.add_argument("--selection", type=Path)
    p.add_argument("--memory-mib", type=int, default=DEFAULT_MEMORY_MIB)
    p.add_argument("--gpu-caps", type=Path)
    p.add_argument("--wall", default="7-00:00:00")
    p = sub.add_parser("verify-runtime", help="Verify frozen source and inputs before a GPU process")
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--key", required=True)
    p = sub.add_parser("select-blue", help="Select CF W20 grid winner and seal zsRE L2")
    p.add_argument("--matrix-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--cf-attempt", type=Path, required=True)
    p = sub.add_parser("resume-held", help="Register one checkpoint recovery job held")
    p.add_argument("--prior-attempt", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--memory-mib", type=int, default=DEFAULT_MEMORY_MIB)
    p.add_argument("--gpu-caps", type=Path)
    p.add_argument("--wall", default="7-00:00:00")
    p = sub.add_parser("release", help="Reinspect held jobs then release in reverse order")
    p.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            rows, files = matrix_rows(args.matrix_root)
            selection = read(args.selection) if args.selection else None
            specs = stage_specs(args.stage, rows, files, args.output_root, selection)
            result = {"stage": args.stage, "jobs": [x["key"] for x in specs],
                      "physical_edit_chains": sum(x["kind"] == "edit" for x in specs),
                      "ready_to_submit": False}
        elif args.command == "submit-held":
            result = submit_held(args)
        elif args.command == "verify-runtime":
            result = verify_runtime(args.attempt, args.key)
        elif args.command == "select-blue":
            result = select_blue(args.matrix_root, args.output_root, args.cf_attempt)
        elif args.command == "resume-held":
            result = resume_held(args)
        else:
            result = release(args.attempt)
        print(json.dumps(result, sort_keys=True, indent=2))
    except (Blocked, KeyError, ValueError, FileNotFoundError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
