"""Outcome-free original-BF16 context-lock reproducibility probe.

This executable only loads one pinned model, verifies the approved EasyEdit
source, and generates the same fresh context set twice after resetting the
fixed RNG seed.  It never reads edit requests, covariance, direct-z, or
evaluation data and has no scheduler submission capability.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

from project.run_scripts.ode_edit_motivation.contracts import (
    ContextManifest,
    ContractError,
    canonical_json,
)
from project.run_scripts.ode_edit_motivation.easyedit_bridge import (
    APPROVED_EASYEDIT_FILES,
    EasyEditBridge,
)
from project.run_scripts.ode_edit_motivation.gpu_runtime import (
    CHECKPOINT_ORIGINAL_DTYPE_POLICY,
    load_fixed_model_checkpoint_original,
    offline_environment,
    seed_runtime,
)
from project.run_scripts.ode_edit_motivation.manifests import (
    FIXED_FILE_IDENTITIES,
)


INSTRUCTION_ID = "ODEEDIT-S02-BF16-CONTEXT-LOCK-PROBE-IMPL-V1"
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
SEED = 17
REPEAT_COUNT = 2
EASYEDIT_ROOT = Path("/mnt/raid5/janghj/EasyEdit")
OUTPUT_PREFIX = "session02-bf16-context-lock-probe-v1-"


class ContextProbeError(RuntimeError):
    """The context probe could not satisfy its fail-closed contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git_head(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    head = completed.stdout.strip().lower()
    if len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise ContextProbeError("Git HEAD is not a full commit identity")
    return head


def _require_offline_environment() -> None:
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise ContextProbeError(f"{key}=1 is required")


def _resolve_output_root(repo: Path, value: Path) -> Path:
    repo = repo.resolve(strict=True)
    candidate = value if value.is_absolute() else repo / value
    candidate = candidate.resolve(strict=False)
    try:
        relative = candidate.relative_to(repo)
    except ValueError as exc:
        raise ContextProbeError("output root escaped the repository") from exc
    if (
        len(relative.parts) != 3
        or relative.parts[:2] != ("local", "results")
        or not relative.name.startswith(OUTPUT_PREFIX)
    ):
        raise ContextProbeError(
            "output root must be one direct local/results context-probe directory"
        )
    if candidate.exists():
        raise FileExistsError(f"output root already exists: {candidate}")
    return candidate


def _pinned_bridge(easyedit_root: Path) -> EasyEditBridge:
    missing = set(APPROVED_EASYEDIT_FILES) - set(FIXED_FILE_IDENTITIES)
    if missing:
        raise ContextProbeError(
            f"approved EasyEdit source lacks fixed identities: {sorted(missing)}"
        )
    pins = {
        relative: {
            "sha256": FIXED_FILE_IDENTITIES[relative].sha256,
            "size": FIXED_FILE_IDENTITIES[relative].size,
        }
        for relative in APPROVED_EASYEDIT_FILES
    }
    return EasyEditBridge(easyedit_root, expected_files=pins)


@contextlib.contextmanager
def _silence_raw_upstream_output() -> Iterator[None]:
    """Prevent generated context strings from entering stdout or Slurm logs."""

    with Path(os.devnull).open("w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def _validate_runtime(runtime: Any, alias: str) -> dict[str, Any]:
    metadata = dict(runtime.metadata())
    required = {
        "model_alias",
        "repository_id",
        "revision",
        "dtype",
        "observed_parameter_dtype",
        "checkpoint_original_dtype",
        "dtype_policy",
        "observed_model_commit",
        "observed_tokenizer_commit",
    }
    if not required <= set(metadata):
        raise ContextProbeError("loaded runtime metadata is incomplete")
    if (
        metadata["model_alias"] != alias
        or metadata["dtype_policy"] != CHECKPOINT_ORIGINAL_DTYPE_POLICY
        or metadata["dtype"] != "torch.bfloat16"
        or metadata["observed_parameter_dtype"] != "torch.bfloat16"
        or metadata["checkpoint_original_dtype"] != "torch.bfloat16"
    ):
        raise ContextProbeError("loaded runtime is not checkpoint-original BF16")
    if (
        metadata["observed_model_commit"] != metadata["revision"]
        or metadata["observed_tokenizer_commit"] != metadata["revision"]
    ):
        raise ContextProbeError("loaded model/tokenizer revision differs")
    parameter_dtypes = {
        parameter.dtype
        for parameter in runtime.model.parameters()
        if parameter.is_floating_point()
    }
    if parameter_dtypes != {torch.bfloat16}:
        raise ContextProbeError("floating parameter dtype set is not exactly BF16")
    if getattr(runtime.model.config, "torch_dtype", None) is not torch.bfloat16:
        raise ContextProbeError("model config dtype is not BF16")
    metadata["config_torch_dtype"] = str(runtime.model.config.torch_dtype)
    return metadata


def _group_sizes(contexts: ContextManifest) -> list[int]:
    return [len(group) for group in contexts.templates]


def _template_bytes(contexts: ContextManifest) -> bytes:
    return canonical_json([list(group) for group in contexts.templates]).encode("utf-8")


def _template_sha256(contexts: ContextManifest) -> str:
    return hashlib.sha256(_template_bytes(contexts)).hexdigest()


def _validate_context_shape(contexts: ContextManifest) -> None:
    if _group_sizes(contexts) != [1, 5] or contexts.templates[0] != ("{}",):
        raise ContextProbeError(
            "EasyEdit context shape differs from base plus five generated templates"
        )


def _contexts_exact(first: ContextManifest, second: ContextManifest) -> bool:
    return (
        first.source == second.source
        and first.templates == second.templates
        and first.manifest_id == second.manifest_id
        and _group_sizes(first) == _group_sizes(second)
        and _template_bytes(first) == _template_bytes(second)
    )


def _file_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "sha256": _sha256(path), "size": path.stat().st_size}


def execute_probe(
    args: argparse.Namespace,
    *,
    runtime_loader: Callable[[str], Any] = load_fixed_model_checkpoint_original,
    bridge_factory: Callable[[Path], Any] = _pinned_bridge,
    seed_fn: Callable[[int], Mapping[str, Any]] = seed_runtime,
    repo_root: Path | None = None,
    git_head_fn: Callable[[Path], str] = _git_head,
) -> int:
    """Run exactly two fresh, identically seeded context generations."""

    _require_offline_environment()
    repo = (repo_root or Path(__file__).resolve().parents[2]).resolve(strict=True)
    output_root = _resolve_output_root(repo, args.output_root)
    easyedit_root = args.easyedit_root.resolve(strict=True)
    if easyedit_root != EASYEDIT_ROOT.resolve(strict=True) and repo_root is None:
        raise ContextProbeError("EasyEdit root differs from the fixed method runtime")

    bridge = bridge_factory(easyedit_root)
    source_manifest = bridge.preflight()
    source_manifest.assert_current()
    with offline_environment():
        runtime = runtime_loader(args.model_alias)
    runtime_metadata = _validate_runtime(runtime, args.model_alias)

    source = f"{args.model_alias}:fresh-seed-{SEED}"
    repeats: list[ContextManifest] = []
    for _repeat_index in range(REPEAT_COUNT):
        seed_fn(SEED)
        with _silence_raw_upstream_output():
            contexts = bridge.freeze_generated_contexts(
                runtime.model,
                runtime.tokenizer,
                source=source,
                fresh=True,
            )
        _validate_context_shape(contexts)
        repeats.append(contexts)
    if len(repeats) != REPEAT_COUNT:
        raise ContextProbeError("context generation repeat count differs")
    source_manifest.assert_current()

    exact_match = _contexts_exact(repeats[0], repeats[1])
    status = "PASS" if exact_match else "NONDETERMINISTIC_CONTEXT_HOLD"
    repeat_summaries = [
        {
            "repeat_index": index,
            "source": contexts.source,
            "manifest_id": contexts.manifest_id,
            "group_sizes": _group_sizes(contexts),
            "templates_sha256": _template_sha256(contexts),
        }
        for index, contexts in enumerate(repeats, start=1)
    ]
    raw_manifest = {
        "schema_version": "ode-edit-session02-bf16-context-raw/v1",
        "raw_local_only": True,
        "model_alias": args.model_alias,
        "seed": SEED,
        "repeats": [
            {
                "repeat_index": index,
                **contexts.to_dict(),
                "group_sizes": _group_sizes(contexts),
                "templates_sha256": _template_sha256(contexts),
            }
            for index, contexts in enumerate(repeats, start=1)
        ],
    }
    summary = {
        "schema_version": "ode-edit-session02-bf16-context-summary/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": status,
        "model": {
            key: runtime_metadata[key]
            for key in (
                "model_alias",
                "repository_id",
                "revision",
                "dtype",
                "observed_parameter_dtype",
                "checkpoint_original_dtype",
                "config_torch_dtype",
                "dtype_policy",
                "observed_model_commit",
                "observed_tokenizer_commit",
            )
        },
        "sampling": {
            "seed": SEED,
            "repeat_count": REPEAT_COUNT,
            "fresh_each_repeat": True,
            "same_process": True,
            "repeat_summaries": repeat_summaries,
            "exact_match": exact_match,
        },
        "scope": {
            "no_edit": True,
            "no_direct_z": True,
            "no_evaluation": True,
            "no_covariance": True,
            "no_dataset": True,
            "scientific_outcome_count": 0,
            "raw_templates_persisted_in_summary": False,
        },
        "hashes": {
            "git_commit": git_head_fn(repo),
            "probe_source_sha256": _sha256(Path(__file__).resolve(strict=True)),
            "easyedit_source_manifest_id": source_manifest.manifest_id,
            "easyedit_source_file_count": len(source_manifest.files),
        },
        "offline": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        },
    }

    output_root.mkdir(parents=True, exist_ok=False)
    context_path = output_root / "context_manifest.json"
    summary_path = output_root / "summary.json"
    _write_json_exclusive(context_path, raw_manifest)
    _write_json_exclusive(summary_path, summary)
    terminal = {
        "schema_version": "ode-edit-session02-bf16-context-terminal/v1",
        "status": status,
        "model_alias": args.model_alias,
        "exact_match": exact_match,
        "files": [_file_record(context_path), _file_record(summary_path)],
    }
    _write_json_exclusive(output_root / "terminal_manifest.json", terminal)
    print(
        json.dumps(
            {
                "status": status,
                "model_alias": args.model_alias,
                "output_root": str(output_root),
                "repeat_ids": [item["manifest_id"] for item in repeat_summaries],
                "exact_match": exact_match,
            },
            sort_keys=True,
        )
    )
    return 0 if exact_match else 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session02-bf16-context-lock-probe",
        description="Generate two fresh original-BF16 context manifests",
        allow_abbrev=False,
    )
    parser.add_argument("--model-alias", choices=MODEL_ALIASES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--easyedit-root", type=Path, default=EASYEDIT_ROOT)
    parser.add_argument("--execute", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        raise ContractError("--execute is required")
    return execute_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
