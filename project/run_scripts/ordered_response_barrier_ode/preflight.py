"""Fail-closed source, stream, asset, and wave gates for ORBODE.

This module performs provenance checks only.  It deliberately does not import
torch, transformers, EasyEdit, an evaluator, or any model-facing ORBODE code.
The production runtime owns those checks after this mechanical gate succeeds.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import ArtifactBoundary, validate_round_publication


INSTRUCTION_ID = "ODEEDIT-S06-ORDERED-RESPONSE-BARRIER-ODE-FAST-MAIN-SH1-V1"
EXPECTED_SESSION_ID = "01a04939-f93a-7b50-bca0-65438eab2062"
SERVER = "server1"
SLURM_NODE = "devbox"
MEMORY_MIB_PER_TASK = 182_272
GPU_PER_TASK = 1
STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
ORDER_ROOT = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
B1_CASE_ID = 19_795
B1_REQUEST_SHA256 = "285a3add6f31d8546b0d76689a016bc0f9f7d8e58c95f54e87bba48ad5cabc65"
B1_REQUEST_ORDER_SHA256 = "0e6ded7a9103f5dc9588c11748b14d8c575093626be7055e3e4d390811f9ce5c"
STREAM_SEAL_RELATIVE = Path(
    "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json"
)
EXECUTION_LOCK_RELATIVE = Path(
    "project/run_scripts/ordered_response_barrier_ode/locks/server1-gated-execution-v1.json"
)
EXECUTION_SOURCE_PATHS = (
    "project/run_scripts/ordered_response_barrier_ode",
    "project/run_scripts/session06_orbode_server1.py",
    "project/run_scripts/session06_orbode_server1_dry_plan.py",
    "project/run_scripts/session06_orbode_server1_b1.sbatch",
    "project/run_scripts/session06_orbode_server1_round0.sbatch",
    "project/run_scripts/session06_orbode_server1_remaining.sbatch",
)
DATASET_RELATIVE = Path("data/counterfact/counterfact.json")
DATASET_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
DATASET_BYTES = 45_108_470
OFFICIAL_EASYEDIT_HEAD = "14cea8245f06715684592ab55184939b99d70784"
OFFICIAL_EASYEDIT_TREE = "9c52aadbc0883da422badf0a730fff21aaa3a8a7"
OFFICIAL_EASYEDIT_ROOT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/easyeditsh1-official-readonly-v1"
)
EASYEDIT_ARTIFACT_ROOT = Path("/mnt/raid5/janghj/EasyEdit")
HF_HUB_CACHE_ROOT = Path("/mnt/raid5/janghj/.cache/huggingface/hub")
AUTHORITATIVE_ROOT = Path("/mnt/raid5/janghj/ODE-edit")
ARM_ORDER = ("O", "QCL", "NQFIX", "ORBFH", "JAC", "ORBHit")
PRIMARY_ARMS = ARM_ORDER[:5]
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
ROUND0_SCHEMA = "orbode.round0.cell-terminal.v1"


class PreflightBoundary(RuntimeError):
    """Typed mechanical provenance or execution-boundary failure."""


@dataclass(frozen=True, slots=True)
class CellSpec:
    cell_id: int
    model_alias: str
    writer_family: str

    @property
    def label(self) -> str:
        return f"{self.model_alias}x{self.writer_family}"


CELL_SPECS = (
    CellSpec(0, "llama3-8b-inst", "MEMIT"),
    CellSpec(1, "llama3-8b-inst", "AlphaEdit"),
    CellSpec(2, "qwen2.5-7b-inst", "MEMIT"),
    CellSpec(3, "qwen2.5-7b-inst", "AlphaEdit"),
)

WAVE_ROUNDS: dict[str, tuple[int, ...]] = {
    "b1": (0,),
    "round0": (0,),
    "remaining": tuple(range(1, 10)),
}

AUTHORITATIVE_INPUTS: dict[str, dict[str, object]] = {
    "project/proposals/2026-09-03-ordered-response-barrier-ode-edit-proposal.md": {
        "sha256": "03a6fc61258e7643fc281fab8ab0d3ca700bb80f09aaa3eb34591ce5827087be",
        "bytes": 55_900,
        "lines": 1_359,
    },
    "project/proposals/2026-09-03-ordered-response-barrier-ode-edit-gh-fast-main-prompt.md": {
        "sha256": "8548d016fda343f8b8917bcbe43647b58ad99ab6f796ece2fe4ef661faa4c431",
        "bytes": 13_143,
        "lines": 287,
    },
    "plans/global/2026-09-03-ordered-response-barrier-ode-edit-fast-main-table.md": {
        "sha256": "e134ac708c556482b912d7103e12a12347a78958943e7fa3d3da0a473908aa40",
        "bytes": 18_113,
        "lines": 451,
    },
}

OFFICIAL_SOURCE_MEMBERS: dict[str, str] = {
    "easyeditor/models/alphaedit/AlphaEdit_main.py": "1016bf13b4521dddd134f54884f00c1a7fb95c33e477c9ba2fe9e0f275e2fb9a",
    "easyeditor/models/alphaedit/compute_z.py": "f6fd935b1fbbb05cd656618414f8b71f45d2d0458f617de85c40d6013d84234a",
    "easyeditor/models/alphaedit/compute_ks.py": "6c5b53d4a1fae02d7b303fe67acc724b31261e2a82b915de4c3862661023bdba",
    "easyeditor/models/alphaedit/__init__.py": "609bef490ec45743ee9ecd6cd2e01280860d8aa27b3ea96b6b78f0586463f113",
    "easyeditor/models/memit/memit_main.py": "39b0c91b586a136be2b6bdfabbcc17705fb3cc07e7f17e739fe565906d9d6ea7",
    "easyeditor/models/memit/compute_z.py": "a293bf042fba463e5060887160a6c8453b427b91a6c9a6061b5a8cce5ca0187e",
    "easyeditor/models/memit/compute_ks.py": "0b039be47c548f046b71cad4efce63414f53edbaa657bde868a20cdd35a2f5c7",
    "easyeditor/models/memit/memit_hparams.py": "223dc4e663a049214c0528a02eff8c066413fbee9bce98735a2ce8d1c6585f41",
}

MODEL_BINDINGS: dict[str, dict[str, object]] = {
    "llama3-8b-inst": {
        "native_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        "hf_repo": "models--meta-llama--Meta-Llama-3-8B-Instruct",
        "config_sha256": "61f3de03a16ca8046b05dc777bce72717022bc8522152eee61a69072272ef54b",
        "tokenizer_sha256": "e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
        "hparams": {
            "MEMIT": ("project/run_scripts/fixed_z_nonuniqueness/config/memit-llama3-8b.yaml", "1c5bccc160493c078de3066313443fe8cc6bc53ec2c61865d64851a10cb80444"),
            "AlphaEdit": ("project/run_scripts/fixed_z_nonuniqueness/config/alphaedit-llama3-8b.yaml", "63239e48ea8faf78dfcc40d7d4f5aff3ff0832eb4208384d11f4a812819765c3"),
        },
        "projector": (
            "examples/null_space_project_Meta-Llama-3-8B-Instruct.pt",
            4_110_419_877,
            "6d356468c6408dca694c1907ffde31cddb99f7e67fa2e74502910d5afbede5ec",
        ),
        "statistics": {
            4: (822_084_814, "7f5fc9b194d86ce289d607a23d02de5e7d7eb5a0833aaf6ef1698d814353585e"),
            5: (822_084_814, "a99a36207d3b27f129bb5323c11139853a7851c9788c58931621aa08c81fb435"),
            6: (822_084_814, "3a6a0c682f09a8b725e624fec1265adb46a2f7328586acc0b77c08e587e1c495"),
            7: (822_084_814, "a29e2d2ffb408eeba4a507d5ab9523129b2520bafc2874847f8d396796ebc830"),
            8: (822_084_814, "f6bbc2f240343d5244730c6422743344193dc16ee6afe64dc6cd3a97497d9050"),
        },
        "statistics_model_dir": "Meta-Llama-3-8B-Instruct",
    },
    "qwen2.5-7b-inst": {
        "native_name": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "hf_repo": "models--Qwen--Qwen2.5-7B-Instruct",
        "config_sha256": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
        "tokenizer_sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "hparams": {
            "MEMIT": ("project/run_scripts/fixed_z_nonuniqueness/config/memit-qwen2.5-7b.yaml", "2cec6319248f24bf49575bf5dff1824ca8b6965b22ccb9d18a59e8a323463231"),
            "AlphaEdit": ("project/run_scripts/fixed_z_nonuniqueness/config/alphaedit-qwen2.5-7b.yaml", "e980d9b5e4ee68b2dc6497672f9a6803f7a9ec5350888b23eb05b9c90f05e815"),
        },
        "projector": (
            "examples/null_space_project_Qwen2.5-7B-Instruct.pt",
            7_177_504_758,
            "d3a9687d196f7ee06739253813917a632542717ce1166c0433aa7eec70c5c8ff",
        ),
        "statistics": {
            4: (1_435_501_774, "935af1e9a7c5fe690471c668af225b882b3ae49039435f7143cd3c136c1f049b"),
            5: (1_435_501_774, "7e4730511077be9b7370f29e94036d3ca08e7e0f0023395341e7496e16340519"),
            6: (1_435_501_774, "d47f4bb2454555740c6bc46ab7f894e170cf6f5fe4bc6c26a3fc93179e18dc2a"),
            7: (1_435_501_774, "61526068f1e1283d2e8554b514f9c7ef726957a8582b62c181f393e5589c0b0d"),
            8: (1_435_501_774, "f5b00a555c9d860af1732ae3c48b1a04ebe8e3427873081fb0cb4cc8b5b30d46"),
        },
        "statistics_model_dir": "Qwen2.5-7B-Instruct",
    },
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode != 0:
        raise PreflightBoundary(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _assert_no_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise PreflightBoundary(f"missing path component: {current}") from exc
        if stat.S_ISLNK(mode):
            raise PreflightBoundary(f"symlink path component is forbidden: {current}")


def _regular(path: Path, *, expected_bytes: int | None = None) -> os.stat_result:
    _assert_no_symlink_components(path)
    try:
        observed = path.lstat()
    except OSError as exc:
        raise PreflightBoundary(f"missing required file: {path}") from exc
    if not stat.S_ISREG(observed.st_mode):
        raise PreflightBoundary(f"required path is not a regular file: {path}")
    if expected_bytes is not None and observed.st_size != expected_bytes:
        raise PreflightBoundary(f"required file size differs: {path}")
    return observed


def cell_spec(cell_id: int) -> CellSpec:
    try:
        spec = CELL_SPECS[cell_id]
    except (IndexError, TypeError) as exc:
        raise PreflightBoundary(f"unsupported array cell: {cell_id}") from exc
    if spec.cell_id != cell_id:
        raise PreflightBoundary("array mapping is not position-stable")
    return spec


def wave_rounds(wave: str) -> tuple[int, ...]:
    try:
        return WAVE_ROUNDS[wave]
    except KeyError as exc:
        raise PreflightBoundary(f"unsupported ORBODE wave: {wave}") from exc


def validate_authoritative_inputs(authoritative_root: Path) -> list[dict[str, object]]:
    root = authoritative_root.absolute()
    receipts: list[dict[str, object]] = []
    for relative, expected in AUTHORITATIVE_INPUTS.items():
        path = root / relative
        observed = _regular(path, expected_bytes=int(expected["bytes"]))
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        line_count = payload.count(b"\n")
        if digest != expected["sha256"] or line_count != expected["lines"]:
            raise PreflightBoundary(f"authoritative input identity differs: {path}")
        receipts.append(
            {
                "absolute_path": str(path),
                "bytes": observed.st_size,
                "lines": line_count,
                "mode": format(stat.S_IMODE(observed.st_mode), "04o"),
                "sha256": digest,
            }
        )
    return receipts


def validate_source_checkout(repo_root: Path, source_head: str, source_tree: str) -> dict[str, object]:
    root = repo_root.absolute()
    _assert_no_symlink_components(root)
    if _run_git(root, "rev-parse", "HEAD") != source_head:
        raise PreflightBoundary("queued source HEAD differs")
    if _run_git(root, "rev-parse", "HEAD^{tree}") != source_tree:
        raise PreflightBoundary("queued source tree differs")
    dirty = _run_git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise PreflightBoundary("queued source has tracked modifications")
    execution_shadow = _run_git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *EXECUTION_SOURCE_PATHS,
    )
    if execution_shadow:
        raise PreflightBoundary("queued execution source has dirty or untracked shadow files")
    return {
        "absolute_path": str(root),
        "branch": _run_git(root, "branch", "--show-current") or "DETACHED",
        "head": source_head,
        "tracked_clean": True,
        "execution_source_shadow_count": 0,
        "tree": source_tree,
    }


def validate_easyedit_source(easyedit_source_root: Path) -> dict[str, object]:
    root = easyedit_source_root.absolute()
    _assert_no_symlink_components(root)
    head = _run_git(root, "rev-parse", "HEAD")
    tree = _run_git(root, "rev-parse", "HEAD^{tree}")
    if head != OFFICIAL_EASYEDIT_HEAD or tree != OFFICIAL_EASYEDIT_TREE:
        raise PreflightBoundary("stock EasyEdit HEAD/tree differs")
    if _run_git(root, "status", "--porcelain", "--untracked-files=no"):
        raise PreflightBoundary("stock EasyEdit source has tracked modifications")
    members: list[dict[str, object]] = []
    for relative, expected_sha in OFFICIAL_SOURCE_MEMBERS.items():
        path = root / relative
        observed = _regular(path)
        digest = sha256_file(path)
        if digest != expected_sha:
            raise PreflightBoundary(f"Official EasyEdit source differs: {relative}")
        members.append({"bytes": observed.st_size, "path": relative, "sha256": digest})
    return {
        "head": head,
        "implementation_boundary": "ODE_EDIT_HOOK_STOCK_EASYEDIT_READ_ONLY",
        "members": members,
        "members_root": canonical_hash(members),
        "root": str(root),
        "tracked_clean": True,
        "tree": tree,
    }


def validate_stream(repo_root: Path, easyedit_artifact_root: Path) -> dict[str, object]:
    seal_path = repo_root.absolute() / STREAM_SEAL_RELATIVE
    _regular(seal_path)
    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightBoundary("cannot read sealed stream") from exc
    if not isinstance(seal, dict):
        raise PreflightBoundary("sealed stream is not an object")
    body = dict(seal)
    root_digest = body.pop("root_digest", None)
    requests = seal.get("requests")
    batch_digests = seal.get("batch_ordered_request_digest_v1")
    if (
        root_digest != canonical_hash(body)
        or root_digest != STREAM_ROOT
        or seal.get("all_request_order_sha256") != ORDER_ROOT
        or seal.get("edit_batch_size") != 100
        or seal.get("sequential_round_count") != 10
        or seal.get("logical_edit_count") != 1000
        or not isinstance(requests, list)
        or len(requests) != 1000
        or not isinstance(batch_digests, list)
        or len(batch_digests) != 10
        or [item.get("ordinal") for item in requests] != list(range(1000))
        or [item.get("round_index") for item in requests] != [index // 100 for index in range(1000)]
        or [item.get("round_ordinal") for item in requests] != [index % 100 for index in range(1000)]
    ):
        raise PreflightBoundary("sealed stream root/order/shape differs")
    dataset = easyedit_artifact_root.absolute() / DATASET_RELATIVE
    observed = _regular(dataset, expected_bytes=DATASET_BYTES)
    if sha256_file(dataset) != DATASET_SHA256:
        raise PreflightBoundary("CounterFact dataset bytes differ")
    source = seal.get("source")
    if not isinstance(source, dict) or (
        source.get("sha256") != DATASET_SHA256
        or source.get("size_bytes") != DATASET_BYTES
    ):
        raise PreflightBoundary("stream dataset binding differs")
    return {
        "batch_order_digests": batch_digests,
        "dataset": {
            "absolute_path": str(dataset),
            "bytes": observed.st_size,
            "sha256": DATASET_SHA256,
        },
        "order_root": ORDER_ROOT,
        "request_count": 1000,
        "seal_bytes": seal_path.stat().st_size,
        "seal_path": str(seal_path),
        "seal_sha256": sha256_file(seal_path),
        "stream_root": STREAM_ROOT,
    }


def validate_execution_lock(repo_root: Path) -> dict[str, object]:
    path = repo_root.absolute() / EXECUTION_LOCK_RELATIVE
    observed = _regular(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightBoundary("cannot read ORBODE execution lock") from exc
    if not isinstance(value, dict):
        raise PreflightBoundary("ORBODE execution lock is not an object")
    body = dict(value)
    identity = body.pop("identity", None)
    if identity != canonical_hash(body):
        raise PreflightBoundary("ORBODE execution lock identity differs")
    expected_mapping = {
        str(item.cell_id): [item.model_alias, item.writer_family] for item in CELL_SPECS
    }
    if (
        value.get("instruction_id") != INSTRUCTION_ID
        or value.get("cell_mapping") != expected_mapping
        or value.get("wave_rounds") != {key: list(rounds) for key, rounds in WAVE_ROUNDS.items()}
        or value.get("stream_root") != STREAM_ROOT
        or value.get("order_root") != ORDER_ROOT
        or value.get("arms") != list(ARM_ORDER)
        or value.get("dtype") != "FULL_FP32"
        or value.get("T") != 1.0
        or value.get("N") != 4
        or value.get("h") != 0.25
        or value.get("b1_common_gate", {}).get("cell_count") != 4
        or value.get("b1_common_gate", {}).get("request_count_per_cell") != 1
        or value.get("b1_common_gate", {}).get("primary_endpoint_count") != 20
        or value.get("b1_common_gate", {}).get("canonical_case_id") != B1_CASE_ID
        or value.get("b1_common_gate", {}).get("canonical_request_sha256")
        != B1_REQUEST_SHA256
        or value.get("b1_common_gate", {}).get("canonical_request_order_sha256")
        != B1_REQUEST_ORDER_SHA256
        or value.get("b1_common_gate", {}).get("required_before_round0_wave") is not True
        or value.get("slurm", {}).get("mem_mib_per_task") != MEMORY_MIB_PER_TASK
        or value.get("slurm", {}).get("export") != "NONE"
        or value.get("scientific_promotion") is not False
    ):
        raise PreflightBoundary("ORBODE execution lock contract differs")
    return {
        "bytes": observed.st_size,
        "identity": identity,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _resolved_hf_member(path: Path, hub_root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PreflightBoundary(f"missing HF snapshot member: {path}") from exc
    hub = hub_root.resolve(strict=True)
    try:
        resolved.relative_to(hub)
    except ValueError as exc:
        raise PreflightBoundary(f"HF snapshot member escapes cache: {path}") from exc
    if not resolved.is_file():
        raise PreflightBoundary(f"HF snapshot target is not regular: {path}")
    return resolved


def validate_model_artifacts(
    repo_root: Path,
    easyedit_artifact_root: Path,
    hf_hub_cache: Path,
    *,
    deep_hash: bool,
) -> dict[str, object]:
    artifact_root = easyedit_artifact_root.absolute()
    hub = hf_hub_cache.absolute()
    _assert_no_symlink_components(artifact_root)
    _assert_no_symlink_components(hub)
    models: dict[str, object] = {}
    for alias, raw_spec in MODEL_BINDINGS.items():
        spec = dict(raw_spec)
        hparams_receipts: dict[str, object] = {}
        for family, binding in dict(spec["hparams"]).items():
            relative, expected_sha = binding
            path = repo_root.absolute() / str(relative)
            observed = _regular(path)
            digest = sha256_file(path)
            if digest != expected_sha:
                raise PreflightBoundary(f"{alias} {family} hparams differ")
            hparams_receipts[str(family)] = {
                "absolute_path": str(path), "bytes": observed.st_size, "sha256": digest,
            }
        projector_relative, projector_bytes, projector_sha = spec["projector"]
        projector = artifact_root / str(projector_relative)
        observed_projector = _regular(projector, expected_bytes=int(projector_bytes))
        observed_projector_sha = sha256_file(projector) if deep_hash else "DEFERRED_TO_PRE_GPU_DEEP_HASH"
        if deep_hash and observed_projector_sha != projector_sha:
            raise PreflightBoundary(f"{alias} projector differs")
        statistics: list[dict[str, object]] = []
        statistics_dir = (
            artifact_root / "examples/data/stats" / str(spec["statistics_model_dir"])
            / "wikipedia_stats"
        )
        for layer, binding in dict(spec["statistics"]).items():
            expected_bytes, expected_sha = binding
            relative = f"model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz"
            path = statistics_dir / relative
            observed = _regular(path, expected_bytes=int(expected_bytes))
            observed_sha = sha256_file(path) if deep_hash else "DEFERRED_TO_PRE_GPU_DEEP_HASH"
            if deep_hash and observed_sha != expected_sha:
                raise PreflightBoundary(f"{alias} layer{layer} statistics differ")
            statistics.append(
                {
                    "absolute_path": str(path),
                    "bytes": observed.st_size,
                    "expected_sha256": expected_sha,
                    "layer": int(layer),
                    "observed_sha256": observed_sha,
                }
            )
        snapshot = hub / str(spec["hf_repo"]) / "snapshots" / str(spec["revision"])
        config = _resolved_hf_member(snapshot / "config.json", hub)
        tokenizer = _resolved_hf_member(snapshot / "tokenizer.json", hub)
        if sha256_file(config) != spec["config_sha256"] or sha256_file(tokenizer) != spec["tokenizer_sha256"]:
            raise PreflightBoundary(f"{alias} HF config/tokenizer differs")
        models[alias] = {
            "hparams": hparams_receipts,
            "native_name": spec["native_name"],
            "projector": {
                "absolute_path": str(projector),
                "bytes": observed_projector.st_size,
                "expected_sha256": projector_sha,
                "observed_sha256": observed_projector_sha,
            },
            "revision": spec["revision"],
            "snapshot": str(snapshot),
            "snapshot_config_sha256": spec["config_sha256"],
            "snapshot_tokenizer_sha256": spec["tokenizer_sha256"],
            "statistics": statistics,
        }
    return {"deep_hash": deep_hash, "models": models}


def _validate_four_cell_common_gate(
    gate_root: Path,
    *,
    source_head: str,
    source_tree: str,
    wave: str,
    request_count: int,
    gate_status: str,
) -> dict[str, object]:
    if wave not in {"b1", "round0"} or request_count not in {1, 100}:
        raise PreflightBoundary("unsupported ORBODE common-gate contract")
    expected_primary_endpoints = request_count * len(PRIMARY_ARMS)
    root = gate_root.absolute()
    _assert_no_symlink_components(root)
    receipts: list[dict[str, object]] = []
    identities: list[str] = []
    for expected_cell in range(4):
        expected_spec = cell_spec(expected_cell)
        path = root / f"task-{expected_cell}" / "terminal-receipt.json"
        result_path = root / f"task-{expected_cell}" / "result.json"
        round_path = root / f"task-{expected_cell}" / "round-00-result.json"
        observed = _regular(path)
        observed_result = _regular(result_path)
        observed_round = _regular(round_path)
        if stat.S_IMODE(observed.st_mode) != 0o600:
            raise PreflightBoundary(
                f"{wave} terminal receipt mode differs for cell {expected_cell}"
            )
        if stat.S_IMODE(observed_result.st_mode) != 0o600:
            raise PreflightBoundary(
                f"{wave} result mode differs for cell {expected_cell}"
            )
        if stat.S_IMODE(observed_round.st_mode) != 0o600:
            raise PreflightBoundary(
                f"{wave} publication mode differs for cell {expected_cell}"
            )
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PreflightBoundary(f"cannot read {wave} receipt for cell {expected_cell}") from exc
        if not isinstance(receipt, dict):
            raise PreflightBoundary(f"{wave} terminal receipt is not an object")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PreflightBoundary(
                f"cannot read {wave} result for cell {expected_cell}"
            ) from exc
        if not isinstance(result, dict):
            raise PreflightBoundary(f"{wave} result is not an object")
        try:
            round_publication = json.loads(round_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise PreflightBoundary(
                f"cannot read {wave} publication for cell {expected_cell}"
            ) from exc
        if not isinstance(round_publication, dict):
            raise PreflightBoundary(f"{wave} publication is not an object")
        try:
            validate_round_publication(
                round_publication,
                expected_round_index=0,
                expected_request_count=request_count,
            )
        except ArtifactBoundary as exc:
            raise PreflightBoundary(
                f"{wave} nested publication differs for cell {expected_cell}: {exc}"
            ) from exc
        if wave == "b1" and (
            round_publication.get("case_ids") != [B1_CASE_ID]
            or round_publication.get("request_sha256") != [B1_REQUEST_SHA256]
            or round_publication.get("request_order_sha256")
            != B1_REQUEST_ORDER_SHA256
        ):
            raise PreflightBoundary(
                f"B1 canonical first-request identity differs for cell {expected_cell}"
            )
        round_file_sha256 = sha256_file(round_path)
        round_identity = str(round_publication.get("identity_sha256"))
        expected_publication_receipt = {
            "round_index": 0,
            "relative_path": "round-00-result.json",
            "file_sha256": round_file_sha256,
            "identity_sha256": round_identity,
        }
        if (
            result.get("round_results") != [round_publication]
            or result.get("round_publication_receipts")
            != [expected_publication_receipt]
            or result.get("round_publication_count") != 1
            or result.get("round_publication_identity_root")
            != canonical_hash([round_identity])
            or result.get("round_publication_file_sha256_root")
            != canonical_hash([round_file_sha256])
        ):
            raise PreflightBoundary(
                f"{wave} result/publication binding differs for cell {expected_cell}"
            )
        body = dict(receipt)
        identity = body.pop("receipt_identity_sha256", None)
        if identity != canonical_hash(body):
            raise PreflightBoundary(f"{wave} receipt identity differs for cell {expected_cell}")
        native_memit_exception = expected_spec.writer_family == "MEMIT"
        if (
            receipt.get("schema") != ROUND0_SCHEMA
            or receipt.get("status") != "TERMINAL_VALID"
            or receipt.get("cell_id") != expected_cell
            or receipt.get("model_alias") != expected_spec.model_alias
            or receipt.get("writer_family") != expected_spec.writer_family
            or receipt.get("wave") != wave
            or receipt.get("rounds") != [0]
            or receipt.get("request_count") != request_count
            or receipt.get("primary_arms") != list(PRIMARY_ARMS)
            or receipt.get("primary_endpoint_count") != expected_primary_endpoints
            or receipt.get("scientific_attempted_count") != expected_primary_endpoints
            or receipt.get("terminal_valid_count") != expected_primary_endpoints
            or receipt.get("technical_failure_count") != 0
            or receipt.get("fixed_z_compute_count") != request_count
            or receipt.get("fixed_z_recompute_count") != 0
            or receipt.get("model_reload_wave_count") != 1
            or receipt.get("source_head") != source_head
            or receipt.get("source_tree") != source_tree
            or receipt.get("stream_root") != STREAM_ROOT
            or receipt.get("order_root") != ORDER_ROOT
            or receipt.get("full_fp32") is not True
            or receipt.get("full_fp32_claim_scope")
            != "MODEL_STORAGE_AND_ORBODE_DYNAMIC_PATH"
            or receipt.get("official_native_memit_ephemeral_fp64_solve_exception")
            is not native_memit_exception
            or receipt.get("all_algorithm_solves_full_fp32")
            is not (not native_memit_exception)
            or receipt.get("unqualified_full_fp32_claim")
            is not (not native_memit_exception)
            or receipt.get("w0_restore_pass") is not True
            or receipt.get("cache_restore_pass") is not True
            or receipt.get("transaction_pass") is not True
            or receipt.get("jvp_gate_pass") is not True
            or receipt.get("overlay_gate_pass") is not True
            or not isinstance(receipt.get("entry_already_hit_count"), int)
            or int(receipt["entry_already_hit_count"]) < 0
            or receipt.get("result_sha256") != sha256_file(result_path)
            or receipt.get("round_publication_count") != 1
            or receipt.get("round_publication_identity_root")
            != canonical_hash([round_identity])
            or receipt.get("round_publication_file_sha256_root")
            != canonical_hash([round_file_sha256])
            or receipt.get(
                "literal_prompt_target_token_prediction_publication_count"
            )
            != 0
            or result.get("schema") != "orbode.server1.cell-result.v1"
            or result.get("status") != "TERMINAL_VALID"
            or result.get("cell_id") != expected_cell
            or result.get("model_alias") != expected_spec.model_alias
            or result.get("writer_family") != expected_spec.writer_family
            or result.get("wave") != wave
            or result.get("rounds") != [0]
            or result.get("request_count") != request_count
            or result.get("primary_arms") != list(PRIMARY_ARMS)
            or result.get("primary_endpoint_count") != expected_primary_endpoints
            or result.get("scientific_attempted_count") != expected_primary_endpoints
            or result.get("terminal_valid_count") != expected_primary_endpoints
            or result.get("technical_failure_count") != 0
            or result.get("fixed_z_compute_count") != request_count
            or result.get("fixed_z_recompute_count") != 0
            or result.get("source_head") != source_head
            or result.get("source_tree") != source_tree
            or result.get("stream_root") != STREAM_ROOT
            or result.get("order_root") != ORDER_ROOT
            or result.get("full_fp32") is not True
            or result.get("full_fp32_claim_scope")
            != "MODEL_STORAGE_AND_ORBODE_DYNAMIC_PATH"
            or result.get("official_native_memit_ephemeral_fp64_solve_exception")
            is not native_memit_exception
            or result.get("all_algorithm_solves_full_fp32")
            is not (not native_memit_exception)
            or result.get("unqualified_full_fp32_claim")
            is not (not native_memit_exception)
            or result.get(
                "literal_prompt_target_token_prediction_publication_count"
            )
            != 0
            or result.get("fast_runtime_preamble_count") != 1
            or result.get("fast_runtime_preamble_status") != "PASS_THIS_WAVE"
        ):
            raise PreflightBoundary(f"{wave} common integrity differs for cell {expected_cell}")
        receipts.append(receipt)
        identities.append(str(identity))
    total_endpoints = 4 * expected_primary_endpoints
    if sum(int(item["primary_endpoint_count"]) for item in receipts) != total_endpoints:
        raise PreflightBoundary(f"{wave} common endpoint denominator differs")
    return {
        "cell_count": 4,
        "endpoint_count": total_endpoints,
        "request_count": 4 * request_count,
        "receipt_identities": identities,
        "receipt_identity_root": canonical_hash(identities),
        "status": gate_status,
        "wave": wave,
    }


def validate_b1_common_gate(
    b1_root: Path,
    *,
    source_head: str,
    source_tree: str,
) -> dict[str, object]:
    """Validate four terminal-valid first-request cells before B100 round0."""

    return _validate_four_cell_common_gate(
        b1_root,
        source_head=source_head,
        source_tree=source_tree,
        wave="b1",
        request_count=1,
        gate_status="B1_COMMON_INTEGRITY_PASS",
    )


def validate_round0_common_gate(
    round0_root: Path,
    *,
    source_head: str,
    source_tree: str,
) -> dict[str, object]:
    """Validate four terminal-valid B100 round0 cells before remaining rounds."""

    return _validate_four_cell_common_gate(
        round0_root,
        source_head=source_head,
        source_tree=source_tree,
        wave="round0",
        request_count=100,
        gate_status="ROUND0_COMMON_INTEGRITY_PASS",
    )


def run_preflight(
    *,
    repo_root: Path,
    authoritative_root: Path,
    easyedit_source_root: Path,
    easyedit_artifact_root: Path,
    hf_hub_cache: Path,
    source_head: str,
    source_tree: str,
    cell_id: int,
    wave: str,
    deep_artifact_hash: bool,
    b1_root: Path | None = None,
    round0_root: Path | None = None,
) -> dict[str, object]:
    if authoritative_root.absolute() != AUTHORITATIVE_ROOT:
        raise PreflightBoundary("authoritative proposal root differs")
    if easyedit_source_root.absolute() != OFFICIAL_EASYEDIT_ROOT:
        raise PreflightBoundary("server1 clean stock EasyEdit source root differs")
    if easyedit_artifact_root.absolute() != EASYEDIT_ARTIFACT_ROOT:
        raise PreflightBoundary("server1 EasyEdit artifact root differs")
    if hf_hub_cache.absolute() != HF_HUB_CACHE_ROOT:
        raise PreflightBoundary("server1 HF hub cache root differs")
    spec = cell_spec(cell_id)
    rounds = wave_rounds(wave)
    b1_common_gate: Mapping[str, object] | None = None
    round0_common_gate: Mapping[str, object] | None = None
    if wave == "b1":
        if b1_root is not None or round0_root is not None:
            raise PreflightBoundary("b1 wave cannot consume a later-wave common gate")
    elif wave == "round0":
        if b1_root is None:
            raise PreflightBoundary("round0 wave requires a B1 common-gate root")
        if round0_root is not None:
            raise PreflightBoundary("round0 wave cannot consume its own common gate")
        b1_common_gate = validate_b1_common_gate(
            b1_root, source_head=source_head, source_tree=source_tree
        )
    elif wave == "remaining":
        if b1_root is not None:
            raise PreflightBoundary("remaining wave consumes round0 gate, not B1 directly")
        if round0_root is None:
            raise PreflightBoundary("remaining wave requires a round0 common-gate root")
        round0_common_gate = validate_round0_common_gate(
            round0_root, source_head=source_head, source_tree=source_tree
        )
    receipt: dict[str, object] = {
        "action_counts": {"gpu": 0, "model": 0, "slurm": 0},
        "arms": list(ARM_ORDER),
        "authoritative_inputs": validate_authoritative_inputs(authoritative_root),
        "cell": {
            "cell_id": spec.cell_id,
            "label": spec.label,
            "model_alias": spec.model_alias,
            "writer_family": spec.writer_family,
        },
        "easyedit_source": validate_easyedit_source(easyedit_source_root),
        "full_fp32_required": True,
        "instruction_id": INSTRUCTION_ID,
        "execution_lock": validate_execution_lock(repo_root),
        "model_artifacts": validate_model_artifacts(
            repo_root, easyedit_artifact_root, hf_hub_cache, deep_hash=deep_artifact_hash
        ),
        "b1_common_gate": b1_common_gate,
        "round0_common_gate": round0_common_gate,
        "science_lock": {
            "T": 1.0,
            "dynamic_z_recompute_count": 0,
            "h": 0.25,
            "history_or_cache_inner_mutation_count": 0,
            "integrator_steps": 4,
            "scientific_promotion": False,
        },
        "server": {
            "gpu_per_task": GPU_PER_TASK,
            "memory_mib_per_task": MEMORY_MIB_PER_TASK,
            "node": SLURM_NODE,
            "server": SERVER,
        },
        "source": validate_source_checkout(repo_root, source_head, source_tree),
        "status": "PRE_GPU_BINDING_PASS" if deep_artifact_hash else "DRY_PLAN_BINDING_PASS",
        "stream": validate_stream(repo_root, easyedit_artifact_root),
        "wave": {
            "cold_entry_per_cohort": True,
            "model_reload_wave_count": 1,
            "name": wave,
            "round_indices": list(rounds),
            "state_carry_from_b1": False,
            "state_carry_from_round0": False,
        },
    }
    receipt["receipt_identity_sha256"] = canonical_hash(receipt)
    return receipt


def parse_local_cap(path: Path, server: str = SERVER) -> dict[str, object]:
    _regular(path)
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 5 or fields[0] != server:
            continue
        node, cap_text, mem_text, patterns = fields[1:]
        try:
            cap = int(cap_text)
            mem = int(mem_text)
        except ValueError as exc:
            raise PreflightBoundary("local GPU cap row is malformed") from exc
        if node != SLURM_NODE or cap < 1 or mem < MEMORY_MIB_PER_TASK:
            raise PreflightBoundary("local server1 GPU/memory cap cannot admit ORBODE")
        return {
            "job_patterns": patterns,
            "max_project_gpus": cap,
            "mem_mib_per_gpu": mem,
            "node": node,
            "server": server,
        }
    raise PreflightBoundary("local server1 GPU cap row is absent")


def match_project_job(name: str, pattern_text: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in pattern_text.split(",") if pattern)
