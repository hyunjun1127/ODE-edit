"""CPU-only, create-once SL-ZFlow source/input freeze and verification.

No model loading, CUDA probing, scheduler calls, network access or asset writes.
The original CPU-reference Git blobs and actual execution source are separate
inventories. Large pretrained shard content hashes are explicitly inherited
from the exact earlier full-hash manifest; current size/header/index coverage
is checked. This is not a new full weight-content rehash or Llama parity test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
from typing import Any

from .native_binding import verify_native_source, validate_native_config


PUBLICATION = "69b467d380715f236fad81141d00b53723e01ba7"
PUBLICATION_TREE = "c0ae949fd666d20b08c2137bc373e78cffbfa244"
REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
FIRST1000_ROOT = "40fe1afb318a4a5da9630425213644a729a1e85fb4e78c18e9a4dcd2870eefdd"
TASK_ROOT = Path("/mnt/raid5/janghj/ODE-edit/local/single-layer-zflow/20260916-v1")
REPO_ROOT = Path("/mnt/raid5/janghj/ODE-edit")
LOCAL = REPO_ROOT / "local"
PREEDIT_LOCK = LOCAL / "fixed10k-preedit-eval/attempt-v1/execution.lock.json"
PREEDIT_LOCK_SHA = "7ea4991018fb18b1cb4dc520acf3e6adb386ad39112440fa2f57af9e33bdf167"
NATIVE_SOURCE = LOCAL / "blue-l4-progress-barrier/attempt-v1/imports/imports/blue-source"
ARCHIVE = LOCAL / "checkpoint-archives/server4-migration-20260911-v1/new177-v1"
N4_CLOSURE = ARCHIVE / "closure/local/blue-alphaedit-l4-oneshot-sequential/attempt-v1"
N4_SUPPLEMENT = ARCHIVE / "supplement/local/blue-alphaedit-l4-oneshot-sequential/attempt-v1"
CONTEXTS = N4_CLOSURE / "main-llama/B01/contexts.json"
CONTEXTS_SHA = "33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e"
CONTEXTS_SEMANTIC_SHA = "cf14b8571136b0413ded1ea8aa283a004ecbddbd818a6a8744b0296b63f72191"
NATIVE_CONFIG = N4_SUPPLEMENT / "blue-l4-config.json"
NATIVE_CONFIG_SHA = "2392ab8392476ed019985e4292a8c0aa2a3957f36a06a704eb7590e8e8c99c5d"
PROJECTOR = Path("/mnt/raid5/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt")
PROJECTOR_SHA = "6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec"
PROJECTOR_BYTES = 4110419877
NETHOOK_SHA = "75d1b6353aae539f56aed3f721fdae9762776414e7a97f2c13734fc77bb29ce1"
MANIFEST_REL = "plans/global/2026-09-16-single-layer-zflow-sh2-dispatch/source-input-manifest.json"
MANIFEST_SHA = "e0f64857e070e4e2123c57adc77e757225d1d36eba86c58a18d8718ce454e423"
USER_REL = "plans/global/2026-09-16-single-layer-zflow-sh2-dispatch/authoritative-user-message.md"
DISPATCH_REL = "messages/head/2026-09-16-sh2-single-layer-zflow-seq1000.md"
CHECKS_REL = "audits/global/2026-09-16-single-layer-zflow-pipeline-checks.json"
PACKAGE = "project/run_scripts/single_layer_zflow"
LOADER_REL = "scripts/fixed_counterfact.py"
LOADER_SHA = "8201a570b19a9e4e93ec7528d6a9131d5c7b203ddf333cba83d73f31220f9000"
EVALUATOR_SHA = {
    "project/run_scripts/blue_alphaedit_sequential_comparison/__init__.py": "c3804026f89cfd012a8050971055ed5feb21a11be3190807bc9c984a44f9a844",
    "project/run_scripts/blue_alphaedit_sequential_comparison/evaluation.py": "d82bc6f0090c3f572ecc23a2de03e8af544e910e1f3534b331ffd8486f96085e",
    "project/run_scripts/blue_alphaedit_sequential_comparison/integrity.py": "1587c9dda24b618afe9695ebd3150f4a3f62ed3330af8a9b10307b02908753e8",
    "project/run_scripts/alphaedit_strength_neutral_barrier/evaluator.py": "4f5af6dbf8854c79aedc5e36134fddfcb2995b60f05d270604f8ea735672d93e",
    "project/run_scripts/alphaedit_strength_neutral_barrier/contracts.py": "1c4aab5f974b4f02531693b733f546ad981b98b26d9260f900f6d074a061a88e",
    "project/run_scripts/ordered_response_barrier_ode/__init__.py": "aa7b332461eea6e6b2317da82e9c537f0c645aea592832445edf33f40e2e5a90",
    "project/run_scripts/ordered_response_barrier_ode/contracts.py": "e94e1d2c17827bc89078010b8b5ca735470b5f376b70e9cee902921afb1dd514",
    "project/run_scripts/ordered_response_barrier_ode/counterfact_locality_evaluator.py": "51551523511a87901c8edd0755f2f2f6f6f16bfd5fd31169e55cd12c7ad89fe6",
}


def encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_member(path: str | Path, *, expected_sha: str | None = None,
                expected_bytes: int | None = None, full_hash: bool = True,
                allow_symlink: bool = False) -> dict[str, Any]:
    path = Path(path).absolute()
    require(allow_symlink or not path.is_symlink(), f"unexpected symlink: {path}")
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
    if expected_bytes is not None:
        require(before.st_size == expected_bytes, f"size mismatch: {path}")
    actual = file_sha(path) if full_hash else expected_sha
    require(actual is not None and len(actual) == 64, f"missing full source hash: {path}")
    if expected_sha is not None:
        require(actual == expected_sha, f"hash mismatch: {path}")
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"file changed during verification: {path}")
    return {"path": str(path), "realpath": str(path.resolve(strict=True)),
            "bytes": before.st_size, "sha256": actual,
            "verification": "FULL_SHA256" if full_hash else "INHERITED_FULL_SHA256_CURRENT_SIZE",
            "symlink": path.is_symlink()}


def exclusive_write(path: Path, data: bytes) -> None:
    """Never overwrite; reject a symlink in any existing output ancestor."""
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), f"output symlink forbidden: {parent}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True).stdout


def reference_blobs(source_root: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Original16 are read from the exact published commit, not edited runtime."""
    tree = git_bytes(source_root, "rev-parse", f"{PUBLICATION}^{{tree}}").decode().strip()
    require(tree == PUBLICATION_TREE, "reference publication tree mismatch")
    manifest_bytes = git_bytes(source_root, "show", f"{PUBLICATION}:{MANIFEST_REL}")
    require(hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA, "reference manifest mismatch")
    manifest = json.loads(manifest_bytes)
    require(len(manifest["members"]) == 16, "reference must have 16 members")
    blobs: dict[str, bytes] = {}
    for member in manifest["members"]:
        relative = member["path"]
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "unsafe reference member")
        raw = git_bytes(source_root, "show", f"{PUBLICATION}:{relative}")
        require(len(raw) == member["bytes"] and hashlib.sha256(raw).hexdigest() == member["sha256"],
                f"reference original mismatch: {relative}")
        require(relative not in blobs, "duplicate reference member")
        blobs[relative] = raw
    inherited = json.loads(blobs[CHECKS_REL])["source_manifest"]
    require(len(inherited) == 12, "inherited source manifest must have 12 members")
    for relative, member in inherited.items():
        require(relative in blobs and len(blobs[relative]) == member["bytes"] and
                hashlib.sha256(blobs[relative]).hexdigest() == member["sha256"],
                f"inherited reference mismatch: {relative}")
    for relative in (MANIFEST_REL, USER_REL, DISPATCH_REL):
        blobs[relative] = git_bytes(source_root, "show", f"{PUBLICATION}:{relative}")
    return blobs, {"publication_head": PUBLICATION, "publication_tree": tree,
                   "original_members": 16, "inherited_members": 12,
                   "authority_companion_members": 3,
                   "reference_llama_adapter_implemented": False,
                   "reference_durable_commit_implemented": False,
                   "reference_test_count": 30,
                   "original_source_bytes_preserved": True}


def execution_source(source_root: Path) -> dict[str, Any]:
    paths = [PACKAGE, LOADER_REL, *EVALUATOR_SHA]
    dirty = git_bytes(source_root, "status", "--porcelain", "--untracked-files=all", "--", *paths)
    require(not dirty.strip(), "execution source not committed/clean; freeze after all code is final")
    tracked = git_bytes(source_root, "ls-files", "-z", "--", *paths).decode().split("\0")
    tracked = sorted(p for p in tracked if p)
    require(tracked and any(p.endswith("native_binding.py") for p in tracked),
            "actual runtime source missing from tracked source")
    members = []
    for relative in tracked:
        require(Path(relative).suffix in (".py", ".md", ".json", ".sbatch", ".sh", ".toml"),
                f"unexpected source payload extension: {relative}")
        expected = EVALUATOR_SHA.get(relative, LOADER_SHA if relative == LOADER_REL else None)
        member = file_member(source_root / relative, expected_sha=expected)
        member["relative"] = relative
        require(git_bytes(source_root, "show", f"HEAD:{relative}") == (source_root / relative).read_bytes(),
                f"working source not HEAD bytes: {relative}")
        members.append(member)
    for relative in (LOADER_REL, *EVALUATOR_SHA):
        require(relative in tracked, f"missing execution closure: {relative}")
    return {"root": str(source_root),
            "head": git_bytes(source_root, "rev-parse", "HEAD").decode().strip(),
            "tree": git_bytes(source_root, "rev-parse", "HEAD^{tree}").decode().strip(),
            "members": members, "member_root": digest(members), "source_scope_clean": True,
            "evaluator_original_sha_match": True,
            "excluded_historical_barrier_package_init": True}


def safetensor_header(path: Path, expected_keys: set[str]) -> dict[str, Any]:
    """Header/index coverage without loading model tensors or reading all data."""
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        require(len(length_bytes) == 8, "truncated safetensors file")
        length = struct.unpack("<Q", length_bytes)[0]
        require(2 <= length <= (32 << 20), "invalid safetensors header length")
        raw = stream.read(length)
    require(len(raw) == length, "truncated safetensors header")
    header = json.loads(raw)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    require(set(tensors) == expected_keys, f"index/header tensor inventory mismatch: {path}")
    intervals = []
    for key, value in tensors.items():
        start, end = value["data_offsets"]
        require(isinstance(start, int) and isinstance(end, int) and 0 <= start <= end,
                f"invalid tensor offsets: {key}")
        intervals.append((start, end))
    intervals.sort()
    require(intervals and intervals[0][0] == 0 and all(a[1] == b[0] for a, b in zip(intervals, intervals[1:])),
            "incomplete/overlapping tensor data coverage")
    require(8 + length + intervals[-1][1] == path.stat().st_size, "safetensors physical length mismatch")
    return {"path": str(path), "header_sha256": hashlib.sha256(raw).hexdigest(),
            "header_bytes": length, "tensor_count": len(tensors),
            "index_membership_exact": True, "data_offsets_cover_file": True}


def model_inventory(model_path: Path, previous_members: list[dict[str, Any]]) -> dict[str, Any]:
    model_path = model_path.absolute()
    prior = {m["path"]: m for m in previous_members if Path(m["path"]).parent == model_path}
    index_path = model_path / "model.safetensors.index.json"
    require(str(index_path) in prior, "model index missing from inherited full-hash manifest")
    index_member = file_member(index_path, expected_sha=prior[str(index_path)]["sha256"],
                               expected_bytes=prior[str(index_path)]["bytes"], allow_symlink=True)
    index = json.loads(index_path.read_bytes())
    weight_map = index["weight_map"]
    shards = set(weight_map.values())
    require(shards and all(Path(s).name == s and s.endswith(".safetensors") for s in shards),
            "unsafe/empty shard index")
    required = {"config.json", "generation_config.json", "model.safetensors.index.json",
                "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", *shards}
    require({Path(path).name for path in prior} == required, "model manifest membership differs")
    members, headers = [], []
    for name in sorted(required):
        path = model_path / name
        old = prior[str(path)]
        members.append(file_member(path, expected_sha=old["sha256"], expected_bytes=old["bytes"],
                                   full_hash=name not in shards, allow_symlink=True))
        if name in shards:
            headers.append(safetensor_header(path, {key for key, value in weight_map.items() if value == name}))
    return {"path": str(model_path), "revision": model_path.name,
            "members": members, "index_member": index_member, "shard_headers": headers,
            "weight_map_entries": len(weight_map), "shard_count": len(shards),
            "weight_content_rehash": "NOT_REPEATED: reused exact PRE_EDIT full-hash manifest",
            "current_verification": "all size/realpath; all small SHA; every shard header/index/physical-length",
            "gpu_parity": "NOT_CLAIMED"}


def fixed_sample(source_root: Path, dataset_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The official loader returns original records; no reconstruction/reorder."""
    loader_path = source_root / LOADER_REL
    file_member(loader_path, expected_sha=LOADER_SHA)
    spec = importlib.util.spec_from_file_location("sl_zflow_fixed_counterfact", loader_path)
    require(spec is not None and spec.loader is not None, "fixed loader import unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = module.load_prefix(dataset_root, 1000)
    sample = json.loads((dataset_root / "source-sample.lock.json").read_bytes())
    prefix_root = digest(sample["records"][:1000])
    require(prefix_root == FIRST1000_ROOT, "first1000 order root mismatch")
    require(len(records) == len({r["case_id"] for r in records}) == 1000, "prefix unique count mismatch")
    members = [file_member(dataset_root / name) for name in
               ("counterfact.json", "source-sample.lock.json", "receipt.json")]
    return {"dataset_root": str(dataset_root), "dataset_sha256": module.DATASET_SHA,
            "whole_ordered_root": module.ORDER_ROOT, "first1000_ordered_root": prefix_root,
            "source_full_dataset_sha256": module.SOURCE_SHA, "source_sample_sha256": module.SAMPLE_SHA,
            "unique_requests": 1000, "scientific_batches": 10, "batch_size": 100,
            "prefix_records_sha256": digest(records), "case_order_sha256": digest([r["case_id"] for r in records]),
            "case_ids": [r["case_id"] for r in records],
            "batch_boundaries": [{"batch_id": i + 1, "start": i * 100, "stop": (i + 1) * 100,
                                  "case_order_sha256": digest([r["case_id"] for r in records[i*100:(i+1)*100]])}
                                 for i in range(10)], "members": members,
            "selection": "official load_prefix(root,1000), no shuffle/filter/replacement"}, records


def actual_imports(deps_path: Path, source_root: Path) -> dict[str, Any]:
    """CPU imports in a fresh process; do not query CUDA or model availability."""
    code = (
        "import sys,json;sys.path.insert(0,sys.argv[1]);sys.path.insert(1,sys.argv[2]);"
        "import torch,transformers,tokenizers;"
        "import project.run_scripts.single_layer_zflow.native_binding as native;"
        "import project.run_scripts.blue_alphaedit_sequential_comparison.evaluation as ev;"
        "ev.bind_observation_only_package();"
        "import project.run_scripts.alphaedit_strength_neutral_barrier.evaluator as kernel;"
        "import project.run_scripts.ordered_response_barrier_ode.counterfact_locality_evaluator as locality;"
        "print(json.dumps({'python':sys.executable,'python_version':sys.version,"
        "'torch_version':torch.__version__,'torch_path':torch.__file__,"
        "'transformers_version':transformers.__version__,'transformers_path':transformers.__file__,"
        "'tokenizers_version':tokenizers.__version__,'tokenizers_path':tokenizers.__file__,"
        "'native_binding_path':native.__file__,'evaluator_path':ev.__file__,"
        "'evaluator_kernel_path':kernel.__file__,'locality_path':locality.__file__}))"
    )
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # Child-only flags; no user's HOME/HF_HOME or existing process is changed.
    result = subprocess.run([sys.executable, "-c", code, str(deps_path), str(source_root)],
                            cwd=source_root, env=env, check=True, capture_output=True, text=True)
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    require(observed["transformers_version"] == "4.44.2", "transformers version mismatch")
    for key in ("transformers_path", "tokenizers_path"):
        require(Path(observed[key]).is_relative_to(deps_path), f"unexpected dependency import: {key}")
    for key in ("native_binding_path", "evaluator_path", "evaluator_kernel_path", "locality_path"):
        require(Path(observed[key]).is_relative_to(source_root), f"unexpected source import: {key}")
    observed["model_loads"] = observed["cuda_queries"] = 0
    return observed


def prepare(output: Path, source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    output = output.absolute()
    require(output.is_relative_to(TASK_ROOT) and output != TASK_ROOT, "output must be under this task's local namespace")
    require(not any(path.is_symlink() for path in (output, *output.parents)), "output ancestor symlink forbidden")
    require(not output.exists() and not output.is_symlink(), "create-once output already exists")
    originals, reference = reference_blobs(source_root)
    source = execution_source(source_root)
    inherited_lock_member = file_member(PREEDIT_LOCK, expected_sha=PREEDIT_LOCK_SHA)
    inherited = json.loads(PREEDIT_LOCK.read_bytes())
    deps_path = Path(inherited["dependencies"])
    model_path = Path(inherited["snapshot"])
    require(model_path.name == REVISION and inherited["revision"] == REVISION, "pretrained revision mismatch")
    dependencies = []
    for old in inherited["members"]:
        path = Path(old["path"])
        if path.is_relative_to(deps_path):
            dependencies.append(file_member(path, expected_sha=old["sha256"], expected_bytes=old["bytes"]))
    require(dependencies, "empty pinned dependency closure")
    listed = {Path(m["path"]) for m in dependencies}
    actual = {p for p in deps_path.rglob("*") if p.is_file() and p.suffix in (".py", ".so")}
    require(actual <= listed, "unsealed Python/native extension in dependency path")
    model = model_inventory(model_path, inherited["members"])
    native = verify_native_source(NATIVE_SOURCE)
    native["technical_nethook"] = file_member(NATIVE_SOURCE / "util/nethook.py", expected_sha=NETHOOK_SHA)
    native["technical_stock_function_extraction"] = "exact AST functions plus source nethook.Trace; no stock writer import"
    context_member = file_member(CONTEXTS, expected_sha=CONTEXTS_SHA)
    contexts = json.loads(CONTEXTS.read_bytes())
    require(digest(contexts) == CONTEXTS_SEMANTIC_SHA, "native contexts semantic hash mismatch")
    config_member = file_member(NATIVE_CONFIG, expected_sha=NATIVE_CONFIG_SHA)
    config_validation = validate_native_config(json.loads(NATIVE_CONFIG.read_bytes()))
    projector_member = file_member(PROJECTOR, expected_sha=PROJECTOR_SHA, expected_bytes=PROJECTOR_BYTES)
    sample, _records = fixed_sample(source_root, LOCAL / "datasets/counterfact-fixed-10k-v1")
    imports = actual_imports(deps_path, source_root)
    runtime_entries = [file_member(imports["torch_path"]),
                       file_member(imports["python"], allow_symlink=True)]
    # All expensive read-only checks completed before any new output is made.
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    authority_members = []
    for relative, raw in sorted(originals.items()):
        path = output / "authoritative" / relative
        exclusive_write(path, raw)
        authority_members.append(file_member(path))
    source["reference"] = reference
    source["authority_members"] = authority_members
    exclusive_write(output / "source.lock.json", encoded(source) + b"\n")
    source_lock_member = file_member(output / "source.lock.json")
    full_members = [inherited_lock_member, context_member, config_member, projector_member,
                    *sample["members"], *dependencies, *runtime_entries,
                    *native["members"], native["technical_nethook"], *model["members"],
                    *authority_members, source_lock_member]
    # Native verification ports emit plain full-hash members.
    for member in full_members:
        member.setdefault("verification", "FULL_SHA256")
    lock = {"schema_version": 1, "instruction_id": "ODEEDIT-S06-SINGLE-LAYER-ZFLOW-SEQ1000-SH2-V1",
            "status": "SOURCE_INPUT_FROZEN_NOT_LLAMA_VALIDATED", "model_path": str(model_path),
            "deps_path": str(deps_path), "blue_source": str(NATIVE_SOURCE),
            "contexts_path": str(CONTEXTS), "native_config_path": str(NATIVE_CONFIG),
            "projector_path": str(PROJECTOR), "dataset_root": sample["dataset_root"],
            "source_root": str(source_root), "source_head": source["head"], "source_tree": source["tree"],
            "source_lock": source_lock_member, "sample": sample, "model": model,
            "native": native, "native_config_validation": config_validation,
            "contexts": {"sha256": CONTEXTS_SHA, "semantic_sha256": CONTEXTS_SEMANTIC_SHA,
                         "group_sizes": [1, 5], "generated_now": False},
            "projector_binding": {"physical_layer": 4, "source_stack_index": 0,
                                  "local_index": 0, "physical_inventory": [4, 5, 6, 7, 8],
                                  "tensor_loaded_in_this_prepare": False,
                                  "actual_tensor_binding_gate": "REQUIRED_AT_TECHNICAL_RUNTIME"},
            "imports": imports, "prior_full_model_hash_manifest": inherited_lock_member,
            "runtime_verification": "actual CPU versions/imports; Python and torch-entry SHA; complete torch binary closure NOT_CLAIMED",
            "dependencies_count": len(dependencies), "reference": reference,
            "members": full_members, "member_root": digest(full_members),
            "seed": 20260907, "scientific_main_batches": 10, "conditional_n4_max_batches": 10,
            "n4_reuse_decision": "PENDING_PARENT_COMPARABILITY_RAW_REVIEW",
            "model_loads": 0, "GPU_actions": 0, "Slurm_actions": 0,
            "source_input_identity_is_not_llama_parity": True}
    exclusive_write(output / "input.lock.json", encoded(lock) + b"\n")
    receipt = {"status": lock["status"], "input_lock": file_member(output / "input.lock.json"),
               "source_lock": source_lock_member, "member_root": lock["member_root"],
               "original_16_full_sha": True, "inherited_12_full_sha": True,
               "unique_sample_requests": 1000, "model_shards_fresh_content_hash": False,
               "actual_llama_validated": False, "GPU_actions": 0}
    exclusive_write(output / "verification.json", encoded(receipt) + b"\n")
    return receipt


def verify(lock_path: Path) -> dict[str, Any]:
    lock_path = lock_path.absolute()
    receipt = json.loads((lock_path.parent / "verification.json").read_bytes())
    file_member(lock_path, expected_sha=receipt["input_lock"]["sha256"], expected_bytes=receipt["input_lock"]["bytes"])
    lock = json.loads(lock_path.read_bytes())
    require(digest(lock["members"]) == lock["member_root"] == receipt["member_root"], "input member root mismatch")
    source_ref = lock["source_lock"]
    file_member(source_ref["path"], expected_sha=source_ref["sha256"], expected_bytes=source_ref["bytes"])
    source = json.loads(Path(source_ref["path"]).read_bytes())
    require(source["head"] == lock["source_head"] and source["tree"] == lock["source_tree"], "source lock mismatch")
    for member in [*source["members"], *lock["members"]]:
        full_hash = member["verification"] == "FULL_SHA256"
        require(full_hash or member["verification"] == "INHERITED_FULL_SHA256_CURRENT_SIZE", "unknown verification level")
        actual = file_member(member["path"], expected_sha=member["sha256"], expected_bytes=member["bytes"],
                             full_hash=full_hash, allow_symlink=member.get("symlink", False))
        if "realpath" in member:
            require(actual["realpath"] == member["realpath"], "asset realpath changed")
    current_source = execution_source(Path(lock["source_root"]))
    require((current_source["head"], current_source["tree"], current_source["member_root"]) ==
            (source["head"], source["tree"], source["member_root"]), "frozen source changed")
    sample, _ = fixed_sample(Path(lock["source_root"]), Path(lock["dataset_root"]))
    require(sample == lock["sample"], "frozen dataset/sample changed")
    prior = json.loads(Path(lock["prior_full_model_hash_manifest"]["path"]).read_bytes())
    require(model_inventory(Path(lock["model_path"]), prior["members"]) == lock["model"], "model header/index membership changed")
    require(actual_imports(Path(lock["deps_path"]), Path(lock["source_root"])) == lock["imports"],
            "actual runtime imports changed")
    return {"status": "SOURCE_INPUT_REVERIFIED_NOT_LLAMA_VALIDATED", "members": len(lock["members"]),
            "member_root": lock["member_root"], "source_head": lock["source_head"],
            "source_tree": lock["source_tree"], "unique_sample_requests": 1000,
            "large_model_content_sha_reused_not_rehashed": True,
            "GPU_actions": 0, "model_loads": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    check = sub.add_parser("verify")
    check.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output, args.source_root) if args.command == "prepare" else verify(args.lock)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
