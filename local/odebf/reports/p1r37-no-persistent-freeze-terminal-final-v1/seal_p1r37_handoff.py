#!/usr/bin/env python3
"""Create and independently verify the raw-free P1R37 result/report/code handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


INSTRUCTION_ID = (
    "ODEEDIT-S05-P1R37-P1R36-NO-PERSISTENT-FREEZE-"
    "INDEPENDENT-B10X10-V1"
)
SOURCE_HEAD = "27fd7714532599abfe169dc9cccf83b7a66735b9"
SOURCE_TREE = "587e1ac06cc1dda64400ee202499048249269582"
SOURCE_PARENT = "0f0907b881eec8d4efdce02b3bd860821bf1b928"
P36_HEAD = "87efd168fc9680cabde878cd3b595e98db66d8fa"
P36_TREE = "0df5a4318b7886145e6b083d48db982de56b37ed"
BRANCH = "codex/p1r37-p1r36-no-persistent-freeze-independent-b10x10-v1"
WORKTREE = Path(
    "/mnt/raid5/janghj/ODE-edit/local/worktrees/"
    "odeeditsh2-s05-p1r37-no-persistent-freeze-v1"
)
ANALYSIS = Path(
    "/mnt/raid5/janghj/ODE-edit/local/odebf/analysis/"
    "p1r37-no-persistent-freeze-terminal-final-v1"
)
AGGREGATOR = Path(
    "/mnt/raid5/janghj/ODE-edit/local/odebf/analysis/"
    "p1r37-no-persistent-freeze-terminal-v1/aggregate_p1r37.py"
)
DESTINATION = Path(
    "/mnt/raid5/janghj/ODE-edit/local/source-handoff/"
    "P1R37_NO_PERSISTENT_FREEZE_TERMINAL_SH2_V1"
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def write_once(path: Path, value: Any) -> None:
    payload = canonical(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def copy_private(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"regular source required: {source}")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    target.chmod(0o600)


def entry(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    mode = stat.S_IMODE(path.lstat().st_mode)
    if not path.is_file() or path.is_symlink() or mode != 0o600:
        raise ValueError(f"private regular file required: {relative}")
    return {
        "path": relative,
        "mode": "0600",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def scheduler_receipt() -> dict[str, Any]:
    output = run(
        [
            "sacct",
            "-X",
            "-j",
            "19528",
            "--format=JobIDRaw,State,ExitCode,Elapsed,Start,End,AllocTRES,NodeList",
            "--parsable2",
            "--noheader",
        ]
    )
    records: dict[str, dict[str, str]] = {}
    for values in csv.reader(output.splitlines(), delimiter="|"):
        if not values or not values[0]:
            continue
        values += [""] * (8 - len(values))
        records[values[0]] = {
            "job_id_raw": values[0],
            "state": values[1],
            "exit_code": values[2],
            "elapsed": values[3],
            "start": values[4],
            "end": values[5],
            "alloc_tres": values[6],
            "node": values[7],
        }
    memory_output = run(
        [
            "sacct",
            "-j",
            "19528",
            "--format=JobIDRaw,State,MaxRSS",
            "--parsable2",
            "--noheader",
        ]
    )
    max_rss: dict[str, str] = {}
    for values in csv.reader(memory_output.splitlines(), delimiter="|"):
        if len(values) >= 3 and values[0].endswith(".batch"):
            max_rss[values[0].removesuffix(".batch")] = values[2]
    mapping = [
        (0, "19529", "llama3-8b-inst", "RS", "Neutral"),
        (1, "19530", "llama3-8b-inst", "RS", "Soft"),
        (2, "19531", "llama3-8b-inst", "BG", "Neutral"),
        (3, "19532", "llama3-8b-inst", "BG", "Soft"),
        (4, "19533", "qwen2.5-7b-inst", "RS", "Neutral"),
        (5, "19534", "qwen2.5-7b-inst", "RS", "Soft"),
        (6, "19535", "qwen2.5-7b-inst", "BG", "Neutral"),
        (7, "19528", "qwen2.5-7b-inst", "BG", "Soft"),
    ]
    tasks = []
    for array_index, raw_id, alias, allocation, arm in mapping:
        record = dict(records[raw_id])
        if record["state"] != "COMPLETED" or record["exit_code"] != "0:0":
            raise ValueError(f"task not terminal PASS: {raw_id}")
        if record["node"] != "server2":
            raise ValueError(f"task node differs: {raw_id}")
        record.update(
            {
                "array_index": array_index,
                "alias": alias,
                "allocation": allocation,
                "arm": arm,
                "max_rss": max_rss.get(raw_id),
            }
        )
        tasks.append(record)
    return {
        "schema": "ode-edit-s05-p1r37-scheduler-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "array_job_id": "19528",
        "array": "0-7%4",
        "server2_gpu_cap": 4,
        "terminal_tasks": tasks,
        "completed_exit0_count": len(tasks),
    }


def create_bundle(stage: Path) -> dict[str, Any]:
    if run(["git", "rev-parse", "HEAD"], cwd=WORKTREE).strip() != SOURCE_HEAD:
        raise ValueError("source head differs")
    if run(["git", "rev-parse", "HEAD^{tree}"], cwd=WORKTREE).strip() != SOURCE_TREE:
        raise ValueError("source tree differs")
    if run(["git", "rev-parse", "HEAD^"], cwd=WORKTREE).strip() != SOURCE_PARENT:
        raise ValueError("source parent differs")
    if run(["git", "rev-parse", BRANCH], cwd=WORKTREE).strip() != SOURCE_HEAD:
        raise ValueError("source branch differs")
    if run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=WORKTREE
    ).strip():
        raise ValueError("tracked source dirty")

    bundle = stage / "code/p1r37-source.bundle"
    bundle.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    run(["git", "bundle", "create", str(bundle), BRANCH], cwd=WORKTREE)
    bundle.chmod(0o600)

    patch_path = stage / "code/p1r37-source-delta-from-p1r36.patch"
    with patch_path.open("wb") as handle:
        subprocess.run(
            ["git", "diff", "--no-ext-diff", f"{P36_HEAD}..{SOURCE_HEAD}"],
            cwd=WORKTREE,
            check=True,
            stdout=handle,
        )
        handle.flush()
        os.fsync(handle.fileno())
    patch_path.chmod(0o600)

    with tempfile.TemporaryDirectory(prefix="p1r37-bundle-verify-") as temporary:
        temporary_path = Path(temporary)
        bare = temporary_path / "empty.git"
        checkout = temporary_path / "checkout"
        run(["git", "init", "--bare", "--quiet", str(bare)])
        # Verifying in an empty object database proves prerequisites=0.
        verify_output = run(["git", "-C", str(bare), "bundle", "verify", str(bundle)])
        run(
            [
                "git",
                "--git-dir",
                str(bare),
                "fetch",
                str(bundle),
                f"refs/heads/{BRANCH}:refs/heads/imported",
            ]
        )
        run(["git", "--git-dir", str(bare), "fsck", "--strict", "--full"])
        imported = run(
            ["git", "--git-dir", str(bare), "rev-parse", "refs/heads/imported"]
        ).strip()
        if imported != SOURCE_HEAD:
            raise ValueError("imported bundle head differs")
        for object_id in (SOURCE_HEAD, SOURCE_TREE, P36_HEAD, P36_TREE):
            run(["git", "--git-dir", str(bare), "cat-file", "-e", object_id])
        checkout.mkdir(mode=0o700)
        run(
            [
                "git",
                f"--git-dir={bare}",
                f"--work-tree={checkout}",
                "checkout",
                "--force",
                SOURCE_HEAD,
                "--",
                ".",
            ]
        )
        if sha256_file(
            checkout
            / "project/run_scripts/ode_bf/p1r37_instantaneous_freeze.py"
        ) != "a233b77c350d845eed54617bca309f0cff4980fd4c69ccfb39c4a56fa0251de3":
            raise ValueError("detached checkout source differs")

    return {
        "schema": "ode-edit-s05-p1r37-source-object-manifest/v1",
        "instruction_id": INSTRUCTION_ID,
        "branch": BRANCH,
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "source_parent": SOURCE_PARENT,
        "p1r36_source_head": P36_HEAD,
        "p1r36_source_tree": P36_TREE,
        "bundle_sha256": sha256_file(bundle),
        "bundle_bytes": bundle.stat().st_size,
        "bundle_prerequisites": 0,
        "empty_bare_verify": "PASS",
        "empty_bare_import": "PASS",
        "strict_fsck": "PASS",
        "detached_checkout": "PASS",
        "verify_output_sha256": hashlib.sha256(verify_output.encode()).hexdigest(),
        "source_patch_sha256": sha256_file(patch_path),
        "source_patch_bytes": patch_path.stat().st_size,
    }


def main() -> int:
    if DESTINATION.exists():
        raise FileExistsError(f"create-once destination exists: {DESTINATION}")
    DESTINATION.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(DESTINATION.parent, 0o700)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".P1R37_NO_PERSISTENT_FREEZE_TERMINAL_SH2_V1.",
            dir=DESTINATION.parent,
        )
    )
    staging.chmod(0o700)
    try:
        analysis_files = [
            "p1r37-no-persistent-freeze-independent-b10x10-terminal-comparison-ko.md",
            "p1r37-comparison-per-case.json",
            "p1r37-comparison-per-case.csv",
            "p1r37-comparison-per-step.json",
            "p1r37-comparison-per-step.csv",
            "p1r37-terminal-summary.json",
            "p1r37-result-identities.json",
            "p1r37-independent-terminal-review-receipt.json",
            "p1r37-focused-tests-receipt.json",
        ]
        for name in analysis_files:
            copy_private(ANALYSIS / name, staging / "analysis" / name)
        copy_private(AGGREGATOR, staging / "analysis/aggregate_p1r37.py")

        lock_dir = WORKTREE / "project/run_scripts/ode_bf/locks"
        copy_private(
            lock_dir
            / "numerical_lock_s05_p1r37_no_persistent_freeze_independent_b10x10.json",
            staging / "locks/p1r37-numerical-lock.json",
        )
        copy_private(
            lock_dir
            / "source_manifest_s05_p1r37_no_persistent_freeze_independent_b10x10.json",
            staging / "locks/p1r37-source-manifest.json",
        )
        state_dir = WORKTREE / "local/odebf/state/p1r37-no-persistent-freeze-b10x10"
        copy_private(
            state_dir / "s05-p1r37-no-persistent-freeze-27fd77145325-v1.intent.json",
            staging / "submission/p1r37-submission-intent.json",
        )
        copy_private(
            state_dir
            / "s05-p1r37-no-persistent-freeze-27fd77145325-v1.submission-receipt.json",
            staging / "submission/p1r37-submission-receipt.json",
        )
        write_once(staging / "submission/p1r37-scheduler-terminal-receipt.json", scheduler_receipt())

        source_manifest = create_bundle(staging)
        write_once(staging / "code/p1r37-source-object-manifest.json", source_manifest)

        analysis_entries = [
            entry(staging, path)
            for path in sorted((staging / "analysis").iterdir())
        ]
        analysis_manifest = {
            "schema": "ode-edit-s05-p1r37-analysis-manifest/v1",
            "instruction_id": INSTRUCTION_ID,
            "attempts": 80,
            "endpoints": 73,
            "typed_incomplete": 7,
            "common_endpoint_pairs": 71,
            "per_case_rows": 80,
            "per_step_rows": 631,
            "task_scoped_freeze_classification": "NEAR_IDENTICAL_OUTCOME",
            "scientific_promotion": False,
            "files": analysis_entries,
        }
        analysis_manifest["analysis_root"] = canonical_hash(analysis_manifest["files"])
        write_once(staging / "analysis-manifest.json", analysis_manifest)

        payload_paths = sorted(
            path
            for path in staging.rglob("*")
            if path.is_file()
            and path.name not in {"package-manifest.json", "handoff-receipt.json"}
        )
        payload_entries = [entry(staging, path) for path in payload_paths]
        package_root = canonical_hash(payload_entries)
        package_manifest = {
            "schema": "ode-edit-s05-p1r37-terminal-handoff-manifest/v1",
            "instruction_id": INSTRUCTION_ID,
            "package_id": "P1R37_NO_PERSISTENT_FREEZE_TERMINAL_SH2_V1",
            "raw_free": True,
            "source_head": SOURCE_HEAD,
            "source_tree": SOURCE_TREE,
            "attempts": 80,
            "endpoints": 73,
            "typed_incomplete": 7,
            "common_endpoint_pairs": 71,
            "payload_file_count": len(payload_entries),
            "payload_bytes": sum(value["bytes"] for value in payload_entries),
            "files": payload_entries,
            "package_root": package_root,
            "scientific_promotion": False,
        }
        write_once(staging / "package-manifest.json", package_manifest)
        manifest_sha = sha256_file(staging / "package-manifest.json")

        report = staging / "analysis/p1r37-no-persistent-freeze-independent-b10x10-terminal-comparison-ko.md"
        receipt = {
            "schema": "ode-edit-s05-p1r37-terminal-handoff-receipt/v1",
            "instruction_id": INSTRUCTION_ID,
            "package_id": "P1R37_NO_PERSISTENT_FREEZE_TERMINAL_SH2_V1",
            "destination": str(DESTINATION),
            "create_once": True,
            "directory_mode": "0700",
            "file_mode": "0600",
            "package_manifest_sha256": manifest_sha,
            "package_root": package_root,
            "report_path": report.relative_to(staging).as_posix(),
            "report_sha256": sha256_file(report),
            "report_bytes": report.stat().st_size,
            "report_lines": report.read_bytes().count(b"\n"),
            "source_head": SOURCE_HEAD,
            "source_tree": SOURCE_TREE,
            "source_bundle_sha256": source_manifest["bundle_sha256"],
            "source_bundle_bytes": source_manifest["bundle_bytes"],
            "source_bundle_prerequisites": 0,
            "empty_bare_import_strict_fsck_detached_checkout": "PASS",
            "job_id": "19528",
            "terminal_task_count": 8,
            "attempts": 80,
            "endpoints": 73,
            "typed_incomplete": 7,
            "technical_failures": 0,
            "model_gpu_slurm_source_result_mutation_after_terminal_analysis": 0,
            "excluded": [
                "RAW_PROMPTS",
                "RAW_TARGETS",
                "GENERATIONS",
                "TENSOR_PAYLOADS",
                "MODEL_WEIGHTS",
                "DATA_CACHE",
                "CREDENTIALS",
                "RUNTIME_LOGS",
            ],
            "task_scoped_freeze_classification": "NEAR_IDENTICAL_OUTCOME",
            "scientific_promotion": False,
        }
        receipt["handoff_root"] = canonical_hash(receipt)
        write_once(staging / "handoff-receipt.json", receipt)

        # Verify every declared payload immediately before atomic publication.
        for expected in payload_entries:
            observed = entry(staging, staging / expected["path"])
            if observed != expected:
                raise ValueError(f"payload changed: {expected['path']}")
        if canonical_hash(payload_entries) != package_root:
            raise ValueError("package root differs")
        for directory in [staging, *[path for path in staging.rglob("*") if path.is_dir()]]:
            directory.chmod(0o700)
        if DESTINATION.exists():
            raise FileExistsError("destination appeared during sealing")
        os.rename(staging, DESTINATION)

        result = {
            "destination": str(DESTINATION),
            "package_manifest_sha256": sha256_file(DESTINATION / "package-manifest.json"),
            "package_root": package_root,
            "handoff_receipt_sha256": sha256_file(DESTINATION / "handoff-receipt.json"),
            "handoff_root": receipt["handoff_root"],
            "report_sha256": receipt["report_sha256"],
            "report_bytes": receipt["report_bytes"],
            "report_lines": receipt["report_lines"],
            "source_bundle_sha256": source_manifest["bundle_sha256"],
            "source_bundle_bytes": source_manifest["bundle_bytes"],
            "payload_file_count": package_manifest["payload_file_count"],
            "payload_bytes": package_manifest["payload_bytes"],
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
