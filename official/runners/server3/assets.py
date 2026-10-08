"""Read-only, fail-closed server3 asset and host preflight.

This module hashes small metadata and source datasets, but never loads a model,
tensor, numpy archive, or scientific record. Large SHA checks are opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


OFFICIAL = Path(__file__).resolve().parents[2]
WORKTREE = OFFICIAL.parent
CONTRACT = OFFICIAL / "hparams/contract.json"
TOKENIZERS = OFFICIAL / "hparams/tokenizers.lock.json"
GENERATION = OFFICIAL / "hparams/generation.lock.json"
HISTORICAL = WORKTREE / "agents/server3/experiment-ready-paths-20260919-v1.json"
MODEL_SEAL = WORKTREE / "agents/server4/p4-hf-consumed-closure-seal.json"
SIZE_SEAL = WORKTREE / "agents/server4/alphaedit-runtime-path-seal.json"
LAYERS = ("4", "5", "6", "7", "8")
DEFAULT_MIN_FREE_BYTES = 32 * 1024**3
DEFAULT_MIN_FREE_INODES = 1000
SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _path(value, base: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).absolute()


def _block(result: dict, code: str, path: Path | str | None, detail: str) -> None:
    item = {"code": code, "path": str(path) if path is not None else None, "detail": detail}
    if item not in result["blockers"]:
        result["blockers"].append(item)


def _file(result: dict, label: str, value, base: Path, *, expected_sha=None,
          expected_bytes=None, hash_content=False, unverified_blocks=False) -> dict:
    path = _path(value, base)
    member = {"path": str(path) if path else None, "exists": False,
              "bytes": None, "sha256": expected_sha, "observed_sha256": None,
              "verification": "MISSING"}
    if path is None:
        _block(result, "PATH_NOT_SET", label, f"{label} must have an explicit file path")
        return member
    if not path.is_file():
        _block(result, "FILE_MISSING", path, f"{label} is absent or is not a regular file")
        return member
    try:
        member["bytes"] = path.stat().st_size
        member["exists"] = True
    except OSError as exc:
        _block(result, "FILE_STAT_FAILED", path, str(exc))
        return member
    if member["bytes"] <= 0:
        _block(result, "FILE_EMPTY", path, f"{label} is empty")
    if expected_bytes is not None and member["bytes"] != expected_bytes:
        _block(result, "FILE_SIZE_MISMATCH", path,
               f"{label}: expected {expected_bytes} bytes, observed {member['bytes']}")
    if expected_sha is not None and (not isinstance(expected_sha, str) or not SHA.fullmatch(expected_sha)):
        _block(result, "SHA_LOCK_INVALID", path, f"{label} has no valid pinned SHA256")
    elif expected_sha is None:
        _block(result, "SHA_LOCK_MISSING", path, f"{label} has no pinned SHA256")
    if hash_content:
        try:
            member["observed_sha256"] = _sha(path)
        except OSError as exc:
            _block(result, "SHA_READ_FAILED", path, str(exc))
        else:
            if member["observed_sha256"] != expected_sha:
                _block(result, "SHA_MISMATCH", path, f"{label} differs from its pinned SHA256")
                member["verification"] = "SHA256_MISMATCH"
            else:
                member["verification"] = "SHA256_VERIFIED"
    elif expected_bytes is not None and member["bytes"] == expected_bytes:
        member["verification"] = "HISTORICAL_SHA_AND_SIZE_ONLY"
    else:
        member["verification"] = "HISTORICAL_SHA_ONLY"
    if unverified_blocks and member["verification"] != "SHA256_VERIFIED":
        _block(result, "CONTENT_UNVERIFIED", path,
               f"{label} requires an exact SHA256 check before execution")
    return member


def _git(args: list[str]) -> str | None:
    try:
        run = subprocess.run(["git", "-C", str(WORKTREE), *args],
                             capture_output=True, text=True, timeout=15, check=True)
        return run.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _official_tree_sha256(result: dict) -> str | None:
    """Hash the archive's actual official/ members using the submitter's format."""
    files = []
    for directory, names, filenames in os.walk(OFFICIAL, followlinks=False):
        current = Path(directory)
        for name in list(names):
            candidate = current / name
            if name == "__pycache__":
                names.remove(name)
            elif candidate.is_symlink():
                _block(result, "SOURCE_TREE_SYMLINK", candidate,
                       "Frozen official source may not contain directory symlinks")
                names.remove(name)
        for name in filenames:
            if name.endswith(".pyc"):
                continue
            candidate = current / name
            if candidate.is_symlink() or not candidate.is_file():
                _block(result, "SOURCE_TREE_NONREGULAR", candidate,
                       "Frozen official source requires regular files")
                continue
            files.append(candidate)
    value = hashlib.sha256()
    try:
        for member in sorted(files, key=lambda p: p.relative_to(WORKTREE).as_posix()):
            relative = member.relative_to(WORKTREE).as_posix()
            value.update(relative.encode() + b"\0" + _sha(member).encode() + b"\n")
    except OSError as exc:
        _block(result, "SOURCE_TREE_READ_FAILED", OFFICIAL, str(exc))
        return None
    return value.hexdigest()


def _source(result: dict) -> None:
    observed = _official_tree_sha256(result)
    if not (WORKTREE / ".git").exists():
        lock_path = _path(os.environ.get("ODEEDIT_SOURCE_LOCK"), WORKTREE)
        code_env = os.environ.get("ODEEDIT_CODE_COMMIT")
        tree_env = os.environ.get("ODEEDIT_OFFICIAL_TREE_SHA256")
        result["source"] = {"kind": "frozen_archive", "source_root": str(WORKTREE),
                            "source_lock": str(lock_path) if lock_path else None,
                            "git_head": None, "code_commit": code_env,
                            "official_tree_sha256": observed,
                            "declared_official_tree_sha256": tree_env,
                            "provenance": "source lock, env repeat, and actual official file digest"}
        if (WORKTREE.name != "source" or lock_path is None or
                lock_path.parent != WORKTREE.parent or lock_path.name != "source-lock.json"):
            _block(result, "SOURCE_ARCHIVE_LAYOUT_INVALID", WORKTREE,
                   "Frozen source and adjacent source-lock.json are required")
            return
        if not lock_path.is_file():
            _block(result, "SOURCE_LOCK_MISSING", lock_path,
                   "Frozen source lock is absent")
            return
        try:
            lock = json.loads(lock_path.read_text())
        except (OSError, ValueError) as exc:
            _block(result, "SOURCE_LOCK_INVALID", lock_path, str(exc))
            return
        if (not isinstance(lock, dict) or not re.fullmatch(r"[0-9a-f]{40}",
                                                          str(lock.get("code_commit", "")))):
            _block(result, "SOURCE_CODE_COMMIT_INVALID", lock_path,
                   "Source lock has no valid commit identifier")
        if lock.get("code_commit") != code_env:
            _block(result, "SOURCE_COMMIT_ENV_MISMATCH", lock_path,
                   "Commit env value differs from source lock")
        if (lock.get("official_tree_sha256") != tree_env or
                not isinstance(tree_env, str) or not SHA.fullmatch(tree_env)):
            _block(result, "SOURCE_TREE_ENV_MISMATCH", lock_path,
                   "Tree env value differs from source lock")
        if observed != tree_env:
            _block(result, "SOURCE_TREE_MISMATCH", OFFICIAL,
                   "Actual frozen official tree differs from source lock")
        result["source"]["git_head"] = lock.get("code_commit")
        result["source"]["source_lock_sha256"] = _sha(lock_path)
        return
    head = _git(["rev-parse", "HEAD"])
    main = _git(["rev-parse", "refs/remotes/origin/main"])
    committed_tree = _git(["rev-parse", "HEAD:official"])
    status = _git(["status", "--porcelain", "--untracked-files=all", "--", "official"])
    result["source"] = {"kind": "git_worktree", "git_head": head,
                        "code_commit": head, "published_main_commit": main,
                        "committed_official_tree": committed_tree,
                        "official_tree_sha256": observed,
                        "official_worktree_clean": status == "" if status is not None else None,
                        "provenance": "git HEAD and committed official tree; no execution source sealed"}
    if not head or not main or not committed_tree or status is None:
        _block(result, "SOURCE_GIT_UNAVAILABLE", WORKTREE, "Could not inspect source provenance")
    else:
        if head != main:
            _block(result, "SOURCE_NOT_ORIGIN_MAIN", WORKTREE,
                   "Execution source must equal the published origin/main commit")
        if status:
            _block(result, "SOURCE_DIRTY", OFFICIAL,
                   "official/ has modified or untracked files; commit and seal execution source")


def _model(result: dict, manifest: dict, base: Path, contract: dict,
           tokenizers: dict, hash_large: bool) -> None:
    snapshot = _path(manifest.get("model_snapshot"), base)
    revision = contract["models"]["qwen25"]["revision"]
    model_id = contract["models"]["qwen25"]["model_id"]
    model = {"path": str(snapshot) if snapshot else None, "model_id": model_id,
             "revision": revision, "exists": bool(snapshot and snapshot.is_dir()),
             "verification": "MISSING", "weight_shards": {}, "tokenizer_files": {},
             "metadata_files": {}}
    result["assets"]["model_snapshot"] = model
    if snapshot is None:
        _block(result, "PATH_NOT_SET", "model_snapshot", "Pinned Qwen snapshot path is required")
        return
    if not snapshot.is_dir():
        _block(result, "MODEL_SNAPSHOT_MISSING", snapshot,
               f"Qwen model snapshot at revision {revision} is absent")
        return
    seal = json.loads(MODEL_SEAL.read_text())["models"]["qwen2.5-7b-inst"]
    model["historical_seal"] = {"path": str(MODEL_SEAL), "sha256": _sha(MODEL_SEAL)}
    if seal["revision"] != revision or seal["native_name"] != model_id:
        _block(result, "MODEL_SEAL_MISMATCH", MODEL_SEAL,
               "Historical model seal differs from the experiment contract")
    parts = snapshot.parts
    repo_segment = "models--Qwen--Qwen2.5-7B-Instruct"
    if len(parts) < 3 or parts[-2] != "snapshots" or parts[-1] != revision or repo_segment not in parts:
        _block(result, "MODEL_REVISION_UNPROVEN", snapshot,
               "Snapshot path must bind the Qwen repository and pinned revision")
    else:
        model["verification"] = "HF_SNAPSHOT_PATH_ONLY"
    sealed = {row["relative_path"]: row for row in seal["required_members"]}
    config = snapshot / "config.json"
    if manifest.get("model_config_sha256") not in (None, sealed["config.json"]["sha256"]):
        _block(result, "MODEL_CONFIG_LOCK_MISMATCH", config,
               "Local model config SHA differs from historical seal")
    model["config"] = _file(result, "model config", str(config), base,
                            expected_sha=sealed["config.json"]["sha256"],
                            expected_bytes=sealed["config.json"]["size"], hash_content=True)
    if config.is_file():
        try:
            parsed = json.loads(config.read_text())
            if parsed.get("model_type") != "qwen2":
                _block(result, "MODEL_CONFIG_MISMATCH", config,
                       "Pinned model must have Qwen2 architecture")
        except (OSError, ValueError) as exc:
            _block(result, "MODEL_CONFIG_UNREADABLE", config, str(exc))
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text())["weight_map"]
            shards = sorted(set(weight_map.values()))
            if not shards or any(Path(name).name != name for name in shards):
                raise ValueError("invalid shard names")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            _block(result, "MODEL_INDEX_INVALID", index, str(exc))
            shards = []
    else:
        _block(result, "MODEL_INDEX_MISSING", index, "Qwen safetensors index is absent")
        shards = []
    locked_shards = {name for name, row in sealed.items() if row["kind"] == "weight_shard"}
    if set(shards) != locked_shards:
        _block(result, "MODEL_SHARD_MAP_MISMATCH", index,
               "Weight index shard set differs from pinned historical model seal")
    for name, row in sorted(sealed.items()):
        if name == "config.json":
            continue
        path = snapshot / name
        if row["kind"] == "weight_shard":
            member = _file(result, f"model shard {name}", str(path), base,
                           expected_sha=row["sha256"], expected_bytes=row["size"],
                           hash_content=hash_large)
            if not hash_large and member["exists"]:
                link = os.readlink(path) if path.is_symlink() else None
                member["blob_link_target"] = link
                if link != row["link_target"]:
                    _block(result, "MODEL_SHARD_PROVENANCE_UNPROVEN", path,
                           "Shard must be a pinned Hugging Face blob link or pass exact SHA256")
                else:
                    member["verification"] = "HF_BLOB_LINK_AND_HISTORICAL_SIZE"
            model["weight_shards"][name] = member
        else:
            model["metadata_files"][name] = _file(
                result, f"model metadata {name}", str(path), base,
                expected_sha=row["sha256"], expected_bytes=row["size"], hash_content=True)
    lock = tokenizers["audits"]["qwen25-cf"]["tokenizer_files_sha256"]
    for name, expected in sorted(lock.items()):
        model["tokenizer_files"][name] = _file(
            result, f"tokenizer {name}", str(snapshot / name), base,
            expected_sha=expected, hash_content=True)
    model["tokenizer_lock_path"] = str(TOKENIZERS)
    model["tokenizer_lock_sha256"] = _sha(TOKENIZERS)
    if not any(item["code"].startswith("MODEL_") or item["code"] in
               ("FILE_MISSING", "FILE_SIZE_MISMATCH", "SHA_MISMATCH") for item in result["blockers"]):
        model["verification"] = ("SHA256_VERIFIED" if hash_large else
                                  "REVISION_AND_HF_BLOB_LINKS_WITH_HISTORICAL_SIZES")


def _physical(result: dict, manifest: dict, base: Path, historical: dict,
              hash_large: bool) -> None:
    expected = historical["models"]["qwen2.5-7b-inst"]["assets"]
    size_seal = None
    if SIZE_SEAL.is_file():
        size_seal = json.loads(SIZE_SEAL.read_text())["models"]["qwen2.5-7b-inst"]
        size_members = {(row["kind"], str(row["layer"])): row
                        for row in size_seal["members"]
                        if row["kind"] in ("covariance", "projector")}
    else:
        size_members = {}
        _block(result, "HISTORICAL_SIZE_SEAL_MISSING", SIZE_SEAL,
               "Tracked historical C0/P size receipt is absent")
    supplied_cov = manifest.get("covariance")
    if not isinstance(supplied_cov, dict):
        supplied_cov = {}
        _block(result, "COVARIANCE_MAP_MISSING", "covariance",
               "Explicit L4–L8 physical covariance map is required")
    if set(supplied_cov) != set(LAYERS):
        _block(result, "COVARIANCE_LAYER_MAP_INVALID", "covariance",
               "Covariance map must contain exactly layers 4, 5, 6, 7, 8")
    output = {}
    for layer in LAYERS:
        entry = supplied_cov.get(layer)
        if not isinstance(entry, dict):
            entry = {}
        pinned = expected["covariance"][layer]
        size_record = size_members.get(("covariance", layer))
        if (not size_record or size_record["sha256"] != pinned["sha256"] or
                size_record["shape"] != pinned["shape"] or
                size_record["dtype"] != pinned["dtype"] or
                not pinned["path"].endswith(size_record["relative_path"])):
            _block(result, "COVARIANCE_SIZE_SEAL_MISMATCH", f"covariance.{layer}",
                   f"L{layer} historical size receipt disagrees with path/hash/shape")
            expected_bytes = None
        else:
            expected_bytes = size_record["size"]
            if entry.get("bytes") != expected_bytes:
                _block(result, "COVARIANCE_SIZE_DECLARATION_MISMATCH",
                       entry.get("path"), f"L{layer} declared size differs from historical receipt")
        path = _path(entry.get("path"), base)
        if path is not None and str(path) != pinned["path"]:
            _block(result, "COVARIANCE_PATH_PROVENANCE_MISMATCH", path,
                   f"L{layer} differs from the historical physical path")
        for field in ("sha256", "shape", "dtype", "npz_key"):
            if entry.get(field) != pinned[field]:
                _block(result, "COVARIANCE_PROVENANCE_MISMATCH", path or f"covariance.{layer}",
                       f"L{layer} {field} differs from historical binding")
        member = _file(result, f"C0 L{layer}", entry.get("path"), base,
                       expected_sha=pinned["sha256"], expected_bytes=expected_bytes,
                       hash_content=hash_large, unverified_blocks=True)
        member.update(shape=pinned["shape"], dtype=pinned["dtype"], npz_key=pinned["npz_key"],
                      historical_sha256=pinned["sha256"],
                      historical_provenance=str(HISTORICAL))
        output[layer] = member
    result["assets"]["covariance"] = output
    entry = manifest.get("projector")
    if not isinstance(entry, dict):
        entry = {}
        _block(result, "PROJECTOR_MAP_MISSING", "projector", "Explicit projector map is required")
    pinned = expected["projector"]
    size_record = size_members.get(("projector", "None"))
    if (not size_record or size_record["sha256"] != pinned["sha256"] or
            size_record["shape"] != pinned["shape"] or
            size_record["dtype"] != pinned["dtype"] or
            not pinned["path"].endswith(size_record["relative_path"])):
        _block(result, "PROJECTOR_SIZE_SEAL_MISMATCH", "projector",
               "Historical projector size receipt disagrees with path/hash/shape")
        expected_bytes = None
    else:
        expected_bytes = size_record["size"]
        if entry.get("bytes") != expected_bytes:
            _block(result, "PROJECTOR_SIZE_DECLARATION_MISMATCH", entry.get("path"),
                   "Declared projector size differs from historical receipt")
    path = _path(entry.get("path"), base)
    if path is not None and str(path) != pinned["path"]:
        _block(result, "PROJECTOR_PATH_PROVENANCE_MISMATCH", path,
               "Projector differs from historical physical path")
    for field in ("sha256", "shape", "dtype", "layer_mapping"):
        if entry.get(field) != pinned[field]:
            _block(result, "PROJECTOR_PROVENANCE_MISMATCH", path or "projector",
                   f"Projector {field} differs from historical binding")
    member = _file(result, "P projector", entry.get("path"), base,
                   expected_sha=pinned["sha256"], expected_bytes=expected_bytes,
                   hash_content=hash_large, unverified_blocks=True)
    member.update(shape=pinned["shape"], dtype=pinned["dtype"],
                  layer_mapping=pinned["layer_mapping"],
                  historical_sha256=pinned["sha256"], historical_provenance=str(HISTORICAL))
    result["assets"]["projector"] = member
    result["physical_provenance"] = {
        "historical_receipt": str(HISTORICAL), "historical_receipt_sha256": _sha(HISTORICAL),
        "historical_size_seal": str(SIZE_SEAL),
        "historical_size_seal_sha256": _sha(SIZE_SEAL) if size_seal else None,
        "claim": "historical path/hash/declared shape only until current exact SHA and native tensor validation"}


def _generation(result: dict, value, base: Path, hash_large: bool,
                require_generation: bool) -> None:
    lock = json.loads(GENERATION.read_text())
    path = _path(value, base)
    generation = {"manifest_path": str(path) if path else None,
                  "manifest_sha256": None, "identity_sha256": lock["reference_identity_sha256"],
                  "files": {}, "required": require_generation}
    result["assets"]["generation_reference"] = generation
    if not require_generation:
        generation["verification"] = "NOT_REQUIRED_FOR_THIS_STAGE"
        return
    native_generator = OFFICIAL / "evaluation/generation/native_generator.py"
    native_sha = _sha(native_generator)
    generation["native_generator_sha256"] = native_sha
    # This reviewed common source contains the Qwen2 cache/position route.
    # Source identity plus tiny CPU family fixtures establishes only software
    # compatibility; the target-model GPU B3 qualification remains separate.
    known_gpt_only_sha = "ef3d3daf20c174cfa3627a3852e2d593eb506718ac8ddde62a909a26c87bd2c3"
    reviewed_qwen2_sha = "601bfb8b3d5514d0d2b13a11d025b02fd53f0f1fe39325daa94ed7307c3ec520"
    if native_sha != reviewed_qwen2_sha:
        _block(result, "QWEN_GENERATION_MODEL_FAMILY_UNSUPPORTED" if native_sha == known_gpt_only_sha
               else "QWEN_GENERATION_SOURCE_NOT_REVIEWED", native_generator,
               "Qwen2 generator source differs from the reviewed CPU-compatible bytes")
    else:
        generation["qwen2_software_compatibility"] = "CPU_FAMILY_FIXTURE_ONLY_GPU_NOT_QUALIFIED"
    runtime_python = result.get("runtime", {}).get("python")
    if runtime_python and Path(runtime_python).is_file():
        # Probe the actual scientific interpreter with the same HOME-only
        # resource lookup as the --export=NONE launcher.  The native generator
        # needs punkt/punkt_tab; no downloader or fallback is permitted.
        probe = ("import json; from official.evaluation.generation.assets "
                 "import nltk_binding; b=nltk_binding(); "
                 "print(json.dumps({'family':b['required_family'],"
                 "'version':b['nltk_version']}))")
        env = {"HOME": str(Path.home()), "USER": os.environ.get("USER", "janghj"),
               "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "PYTHONPATH": str(WORKTREE), "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            observed = subprocess.run([runtime_python, "-c", probe], cwd=WORKTREE,
                                      env=env, capture_output=True, text=True,
                                      timeout=30)
            if observed.returncode:
                raise ValueError("native tokenizer binding failed")
            generation["tokenizer"] = json.loads(observed.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            _block(result, "GENERATION_TOKENIZER_UNAVAILABLE", runtime_python,
                   "Native NLTK tokenizer resource is not available in the sealed runtime")
            generation["tokenizer"] = {"verification": "NOT_AVAILABLE"}
    if path is None or not path.is_file():
        _block(result, "GENERATION_MANIFEST_MISSING", path or "generation_reference_manifest",
               "FLU/CON reference manifest is absent")
        for name, pinned in lock["reference_files"].items():
            generation["files"][name] = {"path": None, "sha256": pinned["sha256"],
                                         "bytes": None, "expected_bytes": pinned["bytes"],
                                         "verification": "MISSING_MANIFEST"}
        return
    generation["manifest_sha256"] = _sha(path)
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        _block(result, "GENERATION_MANIFEST_INVALID", path, str(exc))
        return
    if manifest.get("identity_sha256") != lock["reference_identity_sha256"]:
        _block(result, "GENERATION_IDENTITY_MISMATCH", path,
               "FLU/CON reference identity differs from generation.lock.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        files = {}
        _block(result, "GENERATION_FILE_MAP_INVALID", path, "Reference file map is absent")
    for name, pinned in lock["reference_files"].items():
        row = files.get(name)
        if not isinstance(row, dict):
            row = {}
        if row.get("sha256") != pinned["sha256"] or row.get("bytes") != pinned["bytes"]:
            _block(result, "GENERATION_LOCK_MISMATCH", path,
                   f"{name} differs from generation.lock.json")
        member = _file(result, f"FLU/CON {name}", row.get("path"), path.parent,
                       expected_sha=pinned["sha256"], expected_bytes=pinned["bytes"],
                       hash_content=hash_large, unverified_blocks=True)
        member["source_url"] = pinned["source_url"]
        generation["files"][name] = member
    generation["verification"] = "SHA256_VERIFIED" if all(
        x["verification"] == "SHA256_VERIFIED" for x in generation["files"].values()
    ) else "MANIFEST_AND_SIZE_ONLY"


def _runtime(result: dict, manifest: dict, base: Path) -> None:
    path = _path(manifest.get("runtime_python"), base)
    runtime = {"python": str(path) if path else None, "version": None,
               "packages": {}, "verification": "MISSING"}
    result["runtime"] = runtime
    if path is None or not path.is_file() or not os.access(path, os.X_OK):
        _block(result, "RUNTIME_PYTHON_MISSING", path or "runtime_python",
               "Executable scientific Python runtime is required")
        return
    probe = ("import json,sys,importlib.metadata as m; "
             "names=('torch','transformers','numpy','scipy','scikit-learn','safetensors'); "
             "installed={d.metadata['Name'].lower():d.version for d in m.distributions() "
             "if d.metadata.get('Name')}; "
             "print(json.dumps({'version':list(sys.version_info[:3]),"
             "'packages':{n:installed.get(n) for n in names}}))")
    try:
        run = subprocess.run([str(path), "-I", "-c", probe], capture_output=True,
                             text=True, timeout=30, check=True)
        measured = json.loads(run.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        _block(result, "RUNTIME_PROBE_FAILED", path, str(exc))
        return
    runtime["version"] = measured["version"]
    runtime["packages"] = measured["packages"]
    runtime["verification"] = "METADATA_PROBED"
    if measured["version"][:2] < [3, 10]:
        _block(result, "RUNTIME_PYTHON_TOO_OLD", path, "Python 3.10 or newer is required")
    for package, version in measured["packages"].items():
        if version is None:
            _block(result, "RUNTIME_PACKAGE_MISSING", path, f"{package} is absent")


def _storage(result: dict, manifest: dict, base: Path) -> None:
    path = _path(manifest.get("output_root"), base)
    min_bytes = manifest.get("min_free_bytes", DEFAULT_MIN_FREE_BYTES)
    min_inodes = manifest.get("min_free_inodes", DEFAULT_MIN_FREE_INODES)
    storage = {"output_root": str(path) if path else None,
               "minimum_free_bytes": min_bytes, "minimum_free_inodes": min_inodes,
               "available_bytes": None, "available_inodes": None}
    result["storage"] = storage
    if path is None or not path.is_dir():
        _block(result, "OUTPUT_ROOT_MISSING", path or "output_root",
               "Existing writable output root is required")
        return
    if not os.access(path, os.W_OK | os.X_OK):
        _block(result, "OUTPUT_ROOT_NOT_WRITABLE", path,
               "Output root is not writable by this user")
    if not isinstance(min_bytes, int) or isinstance(min_bytes, bool) or min_bytes < DEFAULT_MIN_FREE_BYTES:
        _block(result, "DISK_BUDGET_INVALID", path,
               f"min_free_bytes must be at least {DEFAULT_MIN_FREE_BYTES}")
        min_bytes = DEFAULT_MIN_FREE_BYTES
    if not isinstance(min_inodes, int) or isinstance(min_inodes, bool) or min_inodes < DEFAULT_MIN_FREE_INODES:
        _block(result, "INODE_BUDGET_INVALID", path,
               f"min_free_inodes must be at least {DEFAULT_MIN_FREE_INODES}")
        min_inodes = DEFAULT_MIN_FREE_INODES
    try:
        stat = os.statvfs(path)
    except OSError as exc:
        _block(result, "STORAGE_STAT_FAILED", path, str(exc))
        return
    storage["available_bytes"] = stat.f_bavail * stat.f_frsize
    storage["available_inodes"] = stat.f_favail
    if storage["available_bytes"] < min_bytes:
        _block(result, "DISK_SPACE_LOW", path,
               f"Available {storage['available_bytes']} bytes below budget {min_bytes}")
    if storage["available_inodes"] < min_inodes:
        _block(result, "INODES_LOW", path,
               f"Available {storage['available_inodes']} inodes below budget {min_inodes}")


def _wandb(result: dict, manifest: dict, base: Path) -> None:
    """Inspect nonsecret locations only; this cannot certify account access."""
    env_path = _path(manifest.get("wandb_env"), base)
    sdk_path = _path(manifest.get("wandb_sdk_python"), base)
    status = {"env_path": str(env_path) if env_path else None,
              "env_exists": bool(env_path and env_path.is_file()),
              "sdk_python": str(sdk_path) if sdk_path else None,
              "sdk_version": None, "online_startup_validated": False,
              "online_startup_pending": True,
              "project_auth_prechecked": False,
              "verification": "LOCAL_PATHS_ONLY"}
    result["wandb"] = status
    if env_path is None or not env_path.is_file():
        _block(result, "WANDB_ENV_MISSING", env_path or "wandb_env",
               "Server3 W&B config path is absent; values are never read by preflight")
    if sdk_path is None or not sdk_path.is_file() or not os.access(sdk_path, os.X_OK):
        _block(result, "WANDB_SDK_PYTHON_MISSING", sdk_path or "wandb_sdk_python",
               "Executable isolated W&B SDK Python is required")
    else:
        scientific = _path(manifest.get("runtime_python"), base)
        if (scientific and scientific.is_file() and
                scientific.parent.parent == sdk_path.parent.parent):
            _block(result, "WANDB_SDK_NOT_ISOLATED", sdk_path,
                   "W&B SDK must use its configured isolated Python runtime")
        try:
            run = subprocess.run([str(sdk_path), "-I", "-c",
                                  "import importlib.metadata as m;print(m.version('wandb'))"],
                                 capture_output=True, text=True, timeout=30, check=True)
            status["sdk_version"] = run.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            _block(result, "WANDB_SDK_PROBE_FAILED", sdk_path, str(exc))
    # submit-held and verify-runtime set this only after their bounded SDK
    # project/auth read. The real online Tracker start still runs before GPU.
    if os.environ.get("ODEEDIT_WANDB_PROJECT_VERIFIED") == "1":
        status["project_auth_prechecked"] = True
        status["verification"] = "PROJECT_AUTH_PRECHECKED_BY_SUBMIT"
        result["pending"].append({"code": "WANDB_ONLINE_STARTUP_PENDING",
                                  "path": str(sdk_path) if sdk_path else None,
                                  "detail": "Tracker online startup must pass before model load"})
    else:
        _block(result, "WANDB_CREDENTIAL_MISSING", sdk_path or "wandb_sdk_python",
               "No successful bounded SDK project/auth precheck is available")


def _asset_identity(contract: dict, tokenizers: dict, historical: dict) -> dict:
    """Expected scientific identities only, independent of host and check state."""
    model = contract["models"]["qwen25"]
    seal = json.loads(MODEL_SEAL.read_text())["models"]["qwen2.5-7b-inst"]
    generation = json.loads(GENERATION.read_text())
    physical = historical["models"]["qwen2.5-7b-inst"]["assets"]
    sources = {dataset: json.loads((OFFICIAL / f"hparams/{dataset}-stream.lock.json").read_text())
               ["source_sha256"] for dataset in ("cf", "zsre")}
    return {
        "model_id": model["model_id"], "model_revision": model["revision"],
        "model_files_sha256": {row["relative_path"]: row["sha256"]
                               for row in seal["required_members"]},
        "tokenizer_files_sha256": tokenizers["audits"]["qwen25-cf"]["tokenizer_files_sha256"],
        "source_sha256": sources,
        "covariance_sha256": {layer: physical["covariance"][layer]["sha256"]
                              for layer in LAYERS},
        "projector": {"sha256": physical["projector"]["sha256"],
                      "layer_mapping": physical["projector"]["layer_mapping"]},
        "generation_reference_identity_sha256": generation["reference_identity_sha256"],
        "generation_reference_files_sha256": {
            name: row["sha256"] for name, row in generation["reference_files"].items()},
    }


def preflight(asset_manifest_path, *, hash_large=False, require_generation=True) -> dict:
    """Return a reviewable manifest; ``ready_to_submit`` is true only with no blockers.

    The input is a local JSON path. Even an all-green preflight is metadata and
    byte-integrity readiness, not a model load, tensor validation, or GPU smoke.
    """
    manifest_path = Path(asset_manifest_path).expanduser().absolute()
    result = {"schema": "official.server3.asset-preflight.v1", "server": "server3",
              "model": "qwen25", "manifest_path": str(manifest_path),
              "manifest_sha256": None, "assets": {}, "blockers": [], "pending": [],
              "ready_to_submit": False, "scientific_io": False,
              "downloads": 0, "recomputed_assets": 0,
              "large_sha_requested": bool(hash_large),
              "qualification_limit": "No model load, tensor validation, native edit, or GPU smoke"}
    if not manifest_path.is_file():
        _block(result, "ASSET_MANIFEST_MISSING", manifest_path, "Local asset manifest is absent")
        return result
    try:
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be a JSON object")
    except (OSError, ValueError) as exc:
        _block(result, "ASSET_MANIFEST_INVALID", manifest_path, str(exc))
        return result
    result["manifest_sha256"] = _sha(manifest_path)
    contract = json.loads(CONTRACT.read_text())
    tokenizers = json.loads(TOKENIZERS.read_text())
    historical = json.loads(HISTORICAL.read_text())
    result["model_identity"] = contract["models"]["qwen25"]
    result["asset_identity"] = _asset_identity(contract, tokenizers, historical)
    result["assets_sha256"] = hashlib.sha256(json.dumps(
        result["asset_identity"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode()).hexdigest()
    result["locks"] = {"contract": {"path": str(CONTRACT), "sha256": _sha(CONTRACT)},
                       "tokenizers": {"path": str(TOKENIZERS), "sha256": _sha(TOKENIZERS)},
                       "generation": {"path": str(GENERATION), "sha256": _sha(GENERATION)},
                       "cf_stream": {"path": str(OFFICIAL / 'hparams/cf-stream.lock.json')},
                       "zsre_stream": {"path": str(OFFICIAL / 'hparams/zsre-stream.lock.json')}}
    _source(result)
    _model(result, manifest, manifest_path.parent, contract, tokenizers, hash_large)
    for dataset, key in (("cf", "cf_source"), ("zsre", "zsre_source")):
        lock_path = OFFICIAL / f"hparams/{dataset}-stream.lock.json"
        lock = json.loads(lock_path.read_text())
        result["locks"][f"{dataset}_stream"]["sha256"] = _sha(lock_path)
        member = _file(result, f"{dataset} source", manifest.get(key), manifest_path.parent,
                       expected_sha=lock["source_sha256"], hash_content=True)
        member["stream_sha256"] = lock["stream_sha256"]
        member["source_lock_path"] = str(lock_path)
        member["source_provenance"] = "official stream source SHA; no dataset parse or rewrite"
        result["assets"][key] = member
    _physical(result, manifest, manifest_path.parent, historical, hash_large)
    _runtime(result, manifest, manifest_path.parent)
    _generation(result, manifest.get("generation_reference_manifest"),
                manifest_path.parent, hash_large, require_generation)
    _storage(result, manifest, manifest_path.parent)
    _wandb(result, manifest, manifest_path.parent)
    result["ready_to_submit"] = not result["blockers"]
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", "--manifest", dest="assets", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hash-large", action="store_true",
                        help="Read and verify multi-GB C0/P and FLU/CON bytes")
    parser.add_argument("--no-generation", action="store_true",
                        help="Inventory pre-generation stage without FLU/CON assets")
    args = parser.parse_args(argv)
    report = preflight(args.assets, hash_large=args.hash_large,
                       require_generation=not args.no_generation)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        from official.experiments.prepare import write_new
        write_new(args.output, report)
    print(encoded, end="")
    return 0 if report["ready_to_submit"] else 2


if __name__ == "__main__":
    sys.exit(main())
