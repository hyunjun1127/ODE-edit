"""Concrete original-BF16 P0 technical-identity runtime.

This module intentionally exposes no adaptive endpoint search.  It computes the
MEMIT Native factors once, replays the independent EasyEdit Native writer, and
compares that endpoint against the q=0 quantized functional and transactional
paths only.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import os
import random
import resource
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .contracts import MODEL_ALIASES, ODEAllocContractError, canonical_hash, canonical_json
from .events import RewriteGateConfig, RewriteGateReading, evaluate_rewrite_gate
from .frozen_inputs import OneShotEditInputs
from .functional import QuantizedBF16FunctionalTrial, quantized_effective_weight
from .gauge import FactorPair, FixedEnergyGauge, factor_gram_energy
from .p0_artifacts import P0ArtifactGuard, sha256_file
from .selection import (
    assert_seal_source_current,
    load_and_verify_seal_candidate,
    load_projected_request,
)
from .transaction import AtomicLayerTransaction


INSTRUCTION_ID = "ODEEDIT-S04-ODE-ALLOC-P0-LOCK-R1-PAIR-V1"
DIAGNOSTIC_INSTRUCTION_ID = "ODEEDIT-S04-ODE-ALLOC-TENSOR-HASH-R3-P0-PAIR-V1"
SESSION_ID = "019fc5ec-f85b-7770-a73a-1d19be1cd491"
EXPECTED_BASE = "281f4e63611f519995d9b8499895c68a4126002c"
CASE_ID = 21135
REQUEST_SHA256 = "f5626d5c210c4c014af57cb5ed89529f9408ae9e2ad1fab255467713591e2722"
SEAL_ROOT = "1b45cd567d5ef1a8c52bf0a5a6f8b715270f620cbbdc9949bbf7094f87ef8cab"
SEED = 41
R3_RUN_TOKEN = "r3"
DIAGNOSTIC_STAGES = (
    "post_execute_memit",
    "post_factor_hash",
    "post_factor_contract",
    "pre_direct_z_score",
    "post_direct_z_score",
    "pre_native_writer",
    "post_native_writer",
    "pre_q0_functional",
    "post_q0_functional",
    "post_q0_commit",
    "post_identity_verdict",
)


def _hash_bytes(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor, *, row_block: int = 64) -> str:
    """Hash dtype, shape, then C-order logical bytes without numeric conversion.

    Each bounded row chunk is detached, made contiguous, copied to CPU, and
    flattened before its uint8 view.  The byte view therefore preserves the
    tensor dtype's native byte representation while avoiding multidimensional
    last-axis view restrictions.
    """

    value = tensor.detach()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(canonical_json(tuple(value.shape)).encode("ascii"))
    if value.ndim == 0:
        chunks: Iterator[torch.Tensor] = iter((value.reshape(1),))
    elif value.ndim == 1:
        chunks = iter((value,))
    else:
        chunks = (
            value[start : min(start + row_block, value.shape[0])]
            for start in range(0, value.shape[0], row_block)
        )
    for chunk in chunks:
        cpu_contiguous = chunk.contiguous().to(device="cpu")
        logical_1d = cpu_contiguous.reshape(-1)
        raw = logical_1d.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(raw)
    return digest.hexdigest()


def factors_sha256(value: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> str:
    records = []
    for name, (key, residual) in sorted(value.items()):
        records.append(
            {
                "name": name,
                "key": tensor_sha256(key),
                "residual": tensor_sha256(residual),
            }
        )
    return canonical_hash(records)


def expected_result_name(alias: str, numerical_lock_sha256: str, run_token: str) -> str:
    if (
        alias not in MODEL_ALIASES
        or len(numerical_lock_sha256) != 64
        or run_token != R3_RUN_TOKEN
    ):
        raise ODEAllocContractError("P0 result identity is invalid")
    return f"s04-p0-native-identity-{run_token}-{alias}-{numerical_lock_sha256[:8]}"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _canonical_write_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("P0 metadata is create-once")
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise
    return hashlib.sha256(encoded).hexdigest()


def _atomic_diagnostic_write_once(path: Path, value: Mapping[str, Any]) -> str:
    """Publish a complete diagnostic receipt atomically without replacement."""

    if path.exists() or path.is_symlink():
        raise FileExistsError("P0 diagnostic metadata is create-once")
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        linked = True
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    if not linked:
        raise ODEAllocContractError("P0 diagnostic receipt was not published")
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return hashlib.sha256(encoded).hexdigest()


def _tensor_structure(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device_class": tensor.device.type,
        "finite": bool(torch.isfinite(tensor).all().item()),
        "sha256": tensor_sha256(tensor),
    }


class DiagnosticStageRecorder:
    """Ordered, create-once, raw-free stage receipts for the technical rerun."""

    def __init__(self, raw_root: Path) -> None:
        self.root = raw_root / "diagnostic"
        self.stages_root = self.root / "stages"
        if self.root.exists() or self.root.is_symlink():
            raise FileExistsError("P0 diagnostic root is create-once")
        self.root.mkdir(mode=0o700)
        self.stages_root.mkdir(mode=0o700)
        self._completed: list[tuple[str, str]] = []
        self.wall_seconds = 0.0

    @property
    def last_completed_stage(self) -> str | None:
        return self._completed[-1][0] if self._completed else None

    def complete(self, stage: str) -> str:
        ordinal = len(self._completed)
        if ordinal >= len(DIAGNOSTIC_STAGES) or stage != DIAGNOSTIC_STAGES[ordinal]:
            raise ODEAllocContractError("P0 diagnostic stage order differs")
        payload = {
            "schema_version": "ode-alloc-s04-p0-diagnostic-stage-r2/v1",
            "ordinal": ordinal,
            "stage": stage,
        }
        started = time.perf_counter()
        try:
            digest = _atomic_diagnostic_write_once(
                self.stages_root / f"{ordinal:02d}-{stage}.json", payload
            )
        finally:
            self.wall_seconds += time.perf_counter() - started
        self._completed.append((stage, digest))
        return digest

    def write_structure_once(
        self, factors: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> str:
        started = time.perf_counter()
        try:
            return self._write_structure_once(factors)
        finally:
            self.wall_seconds += time.perf_counter() - started

    def _write_structure_once(
        self, factors: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> str:
        rows: list[dict[str, Any]] = []
        for ordinal, (_, (key, residual)) in enumerate(sorted(factors.items())):
            rank = (
                int(key.shape[1])
                if key.ndim == 2
                and residual.ndim == 2
                and key.shape[1] == residual.shape[1]
                else None
            )
            rows.append(
                {
                    "ordinal": ordinal,
                    "rank": rank,
                    "key": _tensor_structure(key),
                    "residual": _tensor_structure(residual),
                }
            )
        payload = {
            "schema_version": "ode-alloc-s04-p0-factor-structure-r2/v1",
            "layer_count": len(rows),
            "mapping_keyset_sha256": canonical_hash(sorted(factors)),
            "factors": rows,
        }
        return _atomic_diagnostic_write_once(self.root / "factor_structure.json", payload)

    def receipt_root_sha256(self) -> str:
        return canonical_hash(self._completed)


def _completed_stage_summary(output_root: Path) -> tuple[str | None, str]:
    stages_root = output_root / "raw" / "diagnostic" / "stages"
    completed: list[tuple[str, str]] = []
    observed_names = (
        {path.name for path in stages_root.iterdir()}
        if stages_root.is_dir() and not stages_root.is_symlink()
        else set()
    )
    gap_seen = False
    for ordinal, stage in enumerate(DIAGNOSTIC_STAGES):
        path = stages_root / f"{ordinal:02d}-{stage}.json"
        if not path.exists() and not path.is_symlink():
            gap_seen = True
            continue
        if path.is_symlink() or gap_seen:
            raise ODEAllocContractError("P0 diagnostic stage receipt set differs")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value != {
            "schema_version": "ode-alloc-s04-p0-diagnostic-stage-r2/v1",
            "ordinal": ordinal,
            "stage": stage,
        }:
            raise ODEAllocContractError("P0 diagnostic stage receipt content differs")
        completed.append((stage, sha256_file(path)))
    expected_names = {
        f"{ordinal:02d}-{stage}.json" for ordinal, (stage, _) in enumerate(completed)
    }
    if observed_names != expected_names:
        raise ODEAllocContractError("P0 diagnostic stage receipt namespace differs")
    return (completed[-1][0] if completed else None, canonical_hash(completed))


def _allowlisted_exception_frames(
    exc: BaseException, *, repo_root: Path, easyedit_root: Path
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    traceback_cursor = exc.__traceback__
    resolved_repo = repo_root.resolve(strict=True)
    resolved_easyedit = easyedit_root.resolve(strict=True)
    while traceback_cursor is not None:
        code = traceback_cursor.tb_frame.f_code
        source = Path(code.co_filename).resolve(strict=False)
        recorded: str | None = None
        with contextlib.suppress(ValueError):
            relative = source.relative_to(resolved_repo)
            if (
                relative.parts[:3] == ("project", "run_scripts", "ode_alloc")
                or relative == Path("project/run_scripts/session04_ode_alloc_p0.py")
            ):
                recorded = relative.as_posix()
        if recorded is None:
            with contextlib.suppress(ValueError):
                relative = source.relative_to(resolved_easyedit)
                if relative.parts[:3] == ("easyeditor", "models", "memit"):
                    recorded = f"easyedit/{relative.as_posix()}"
        if recorded is not None:
            frames.append(
                {
                    "file": recorded,
                    "function": code.co_name,
                    "line": int(traceback_cursor.tb_lineno),
                }
            )
        traceback_cursor = traceback_cursor.tb_next
    return frames


def write_diagnostic_failure_once(
    output_root: Path,
    exc: BaseException,
    *,
    repo_root: Path,
    easyedit_root: Path,
) -> str | None:
    if not output_root.exists() or not output_root.is_dir():
        return None
    destination = output_root / "diagnostic_failure.json"
    if destination.exists() or destination.is_symlink():
        return None
    last_stage, stage_root_sha256 = _completed_stage_summary(output_root)
    structure = output_root / "raw" / "diagnostic" / "factor_structure.json"
    payload = {
        "schema_version": "ode-alloc-s04-p0-diagnostic-failure-r2/v1",
        "status": "FAIL_CLOSED_NO_RETRY",
        "diagnostic_instruction_id": DIAGNOSTIC_INSTRUCTION_ID,
        "last_completed_stage": last_stage,
        "stage_receipt_root_sha256": stage_root_sha256,
        "factor_structure_sha256": (
            sha256_file(structure) if structure.exists() and not structure.is_symlink() else None
        ),
        "exception_type": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
        "allowlisted_frames": _allowlisted_exception_frames(
            exc, repo_root=repo_root, easyedit_root=easyedit_root
        ),
        "scientific_outcome_count": 0,
    }
    return _atomic_diagnostic_write_once(destination, payload)


@dataclass(frozen=True, slots=True)
class ScorePanel:
    new: dict[str, float]
    old: dict[str, float]
    new_logits_sha256: str
    old_logits_sha256: str


class ForwardCounter:
    def __init__(self, model: torch.nn.Module, ledger: ComputeLedger) -> None:
        self.ledger = ledger
        self.handle = model.register_forward_pre_hook(self._observe, with_kwargs=True)

    def _observe(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del module
        self.ledger.increment("model_forward_calls")
        input_ids = kwargs.get("input_ids")
        attention = kwargs.get("attention_mask")
        if input_ids is None and args and isinstance(args[0], torch.Tensor):
            input_ids = args[0]
        if isinstance(attention, torch.Tensor):
            tokens = int(attention.detach().sum().item())
        elif isinstance(input_ids, torch.Tensor):
            tokens = int(input_ids.numel())
        else:
            tokens = 0
        self.ledger.increment("processed_tokens", tokens)

    def close(self) -> None:
        self.handle.remove()


class ComponentTimer:
    def __init__(self) -> None:
        self.wall: dict[str, float] = {}
        self.gpu: dict[str, float] = {}

    @contextlib.contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if name in self.wall:
            raise ODEAllocContractError("P0 timing component repeats")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_event = end_event = None
        started = time.perf_counter()
        try:
            yield
        finally:
            if end_event is not None and start_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                self.gpu[name] = float(start_event.elapsed_time(end_event)) / 1000.0
            else:
                self.gpu[name] = 0.0
            self.wall[name] = time.perf_counter() - started


def _normalized_target(tokenizer: Any, target: str, device: torch.device) -> torch.Tensor:
    locked = target if target.startswith(" ") else " " + target
    ids = tokenizer.encode(locked, return_tensors="pt", add_special_tokens=False)[0]
    if ids.numel() and ids[0].item() in {tokenizer.bos_token_id, tokenizer.unk_token_id}:
        ids = ids[1:]
    if ids.numel() == 0:
        raise ODEAllocContractError("P0 rewrite target tokenization is empty")
    return ids.to(device=device)


def _replace_hidden_at_lookup(direct_z: torch.Tensor, lookup: int):
    def hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
        del module, inputs
        if isinstance(output, torch.Tensor):
            hidden = output
            rest: Sequence[Any] | None = None
            kind = "tensor"
        elif isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            hidden, rest, kind = output[0], output[1:], "tuple"
        elif isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
            hidden, rest, kind = output[0], output[1:], "list"
        else:
            raise ODEAllocContractError("direct-z layer output contract differs")
        updated = hidden.clone()
        updated[0, lookup, :].copy_(direct_z.to(device=hidden.device, dtype=hidden.dtype))
        if kind == "tensor":
            return updated
        assert rest is not None
        if kind == "tuple":
            return (updated, *rest)
        return [updated, *rest]

    return hook


def _score_target(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_templates: Sequence[tuple[str, str]],
    *,
    subject: str,
    target: str,
    direct_z: torch.Tensor | None,
    z_module_name: str | None,
    fact_token: str | None,
) -> tuple[dict[str, float], str]:
    from easyeditor.models.memit.compute_z import find_fact_lookup_idx

    device = next(model.parameters()).device
    target_ids = _normalized_target(tokenizer, target, device)
    digest = hashlib.sha256()
    readings: dict[str, float] = {}
    for context_id, prompt_template in prompt_templates:
        prompt = prompt_template.format(subject)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        base_ids = encoded["input_ids"].to(device)
        if base_ids.shape[1] < 1:
            raise ODEAllocContractError("P0 rewrite prompt tokenization is empty")
        all_ids = torch.cat((base_ids, target_ids.unsqueeze(0)), dim=1)
        attention = torch.ones_like(all_ids)
        handle = None
        if direct_z is not None:
            if z_module_name is None or fact_token is None:
                raise ODEAllocContractError("direct-z scoring metadata is incomplete")
            lookup = find_fact_lookup_idx(
                prompt_template, subject, tokenizer, fact_token, verbose=False
            )
            module = model.get_submodule(z_module_name)
            handle = module.register_forward_hook(_replace_hidden_at_lookup(direct_z, lookup))
        try:
            with torch.no_grad():
                logits = model(
                    input_ids=all_ids,
                    attention_mask=attention,
                    use_cache=False,
                ).logits
        finally:
            if handle is not None:
                handle.remove()
        start = base_ids.shape[1] - 1
        selected = logits[0, start : start + target_ids.numel(), :]
        if selected.shape[0] != target_ids.numel():
            raise ODEAllocContractError("teacher-forced logit window differs")
        digest.update(context_id.encode("ascii"))
        digest.update(str(selected.dtype).encode("ascii"))
        digest.update(canonical_json(tuple(selected.shape)).encode("ascii"))
        digest.update(
            selected.detach().contiguous().to(device="cpu").view(torch.uint8).numpy().tobytes()
        )
        token_logp = torch.log_softmax(selected.float(), dim=-1).gather(
            1, target_ids.unsqueeze(1)
        )
        readings[context_id] = float(token_logp.mean().item())
        del logits, selected, all_ids, attention, base_ids, token_logp
    return readings, digest.hexdigest()


def score_panel(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_templates: Sequence[tuple[str, str]],
    request: Mapping[str, Any],
    *,
    direct_z: torch.Tensor | None = None,
    z_module_name: str | None = None,
    fact_token: str | None = None,
) -> ScorePanel:
    new, new_digest = _score_target(
        model,
        tokenizer,
        prompt_templates,
        subject=request["subject"],
        target=request["target_new"],
        direct_z=direct_z,
        z_module_name=z_module_name,
        fact_token=fact_token,
    )
    old, old_digest = _score_target(
        model,
        tokenizer,
        prompt_templates,
        subject=request["subject"],
        target=request["target_old"],
        direct_z=direct_z,
        z_module_name=z_module_name,
        fact_token=fact_token,
    )
    return ScorePanel(new, old, new_digest, old_digest)


def _event(
    candidate: ScorePanel,
    entry: ScorePanel,
    direct_z: ScorePanel,
    config: RewriteGateConfig,
) -> RewriteGateReading:
    return evaluate_rewrite_gate(
        candidate_new=candidate.new,
        candidate_old=candidate.old,
        entry_new=entry.new,
        direct_z_new=direct_z.new,
        direct_z_old=direct_z.old,
        config=config,
    )


def _event_record(reading: RewriteGateReading) -> dict[str, Any]:
    return {
        "mean_margin": reading.mean_margin,
        "oracle_ratio": reading.oracle_ratio,
        "worst_context_margin_diagnostic_only": reading.worst_context_margin,
        "oracle_mean_margin": reading.oracle_mean_margin,
        "denominator": reading.denominator,
        "feasible": reading.feasible,
        "event_id": reading.event_id,
    }


def _source_freeze(repo: Path, source_head: str) -> None:
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    if observed != source_head:
        raise ODEAllocContractError("P0 source HEAD differs from submission freeze")
    changed = subprocess.run(
        [
            "git", "status", "--porcelain", "--untracked-files=all", "--",
            "project/run_scripts/ode_alloc", "project/run_scripts/session04_ode_alloc_p0.py",
            "project/run_scripts/session04_ode_alloc_p0_r1.sbatch",
            "project/run_scripts/session04_ode_alloc_submit_p0_r1.py",
        ],
        cwd=repo, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    if changed:
        raise ODEAllocContractError("P0 source freeze has changed paths")


def _fresh_contexts_twice(model: Any, tokenizer: Any) -> tuple[list[list[str]], str, str]:
    from easyeditor.models.memit import memit_main

    memit_main.CONTEXT_TEMPLATES_CACHE = None
    _seed_all(SEED)
    first = copy.deepcopy(memit_main.get_context_templates(model, tokenizer))
    first_hash = canonical_hash(first)
    memit_main.CONTEXT_TEMPLATES_CACHE = None
    _seed_all(SEED)
    second = copy.deepcopy(memit_main.get_context_templates(model, tokenizer))
    second_hash = canonical_hash(second)
    if first != second or first_hash != second_hash:
        raise ODEAllocContractError("fresh original-BF16 contexts differ across generation")
    memit_main.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(second)
    return second, first_hash, second_hash


def _flatten_contexts(
    context_templates: Sequence[Sequence[str]], request_prompt: str
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for group in context_templates:
        for template in group:
            authorized = template.format(request_prompt)
            context_id = f"ctx-{len(result):02d}-{_hash_bytes(authorized.encode('utf-8'))[:12]}"
            result.append((context_id, authorized))
    if len(result) != 6:
        raise ODEAllocContractError("fixed MEMIT rewrite context count differs")
    return tuple(result)


def _capture_native_factors(
    model: Any,
    tokenizer: Any,
    request: Mapping[str, Any],
    hparams: Any,
    artifact_guard: P0ArtifactGuard,
    ledger: ComputeLedger,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    torch.Tensor,
    dict[int, torch.Tensor],
    tuple[tuple[int, str], ...],
    dict[str, int],
    Any,
]:
    from easyeditor.models.memit import memit_main

    layer_stats_module = importlib.import_module("easyeditor.models.rome.layer_stats")
    original_z = memit_main.compute_z
    original_ks = memit_main.compute_ks
    original_cov = memit_main.get_cov
    original_load_dataset = layer_stats_module.load_dataset
    captured_z: list[torch.Tensor] = []
    captured_keys: dict[int, torch.Tensor] = {}
    cov_counts: dict[int, int] = {}
    counts = {"direct_z": 0, "key": 0, "covariance": 0, "native_factors": 0}

    def compute_z_once(*args: Any, **kwargs: Any) -> torch.Tensor:
        counts["direct_z"] += 1
        if counts["direct_z"] != 1:
            raise ODEAllocContractError("direct-z was computed more than once")
        value = original_z(*args, **kwargs)
        captured_z.append(value.detach().clone())
        return value

    def compute_ks_once(*args: Any, **kwargs: Any) -> torch.Tensor:
        layer = int(args[4] if len(args) > 4 else kwargs["layer"])
        if layer in captured_keys:
            raise ODEAllocContractError("MEMIT key was computed more than once per layer")
        value = original_ks(*args, **kwargs)
        captured_keys[layer] = value.detach().to(device="cpu").clone()
        counts["key"] += 1
        return value

    def get_cov_once(*args: Any, **kwargs: Any) -> torch.Tensor:
        layer_name = str(args[2] if len(args) > 2 else kwargs["layer_name"])
        try:
            layer = next(layer for layer in hparams.layers if str(layer) in layer_name.split("."))
        except StopIteration as exc:
            raise ODEAllocContractError("covariance layer is outside the locked basis") from exc
        if kwargs.get("force_recompute", False) is not False:
            raise ODEAllocContractError("covariance recomputation was requested")
        cov_counts[layer] = cov_counts.get(layer, 0) + 1
        if cov_counts[layer] != 1:
            raise ODEAllocContractError("covariance was loaded more than once per layer")
        counts["covariance"] += 1
        return original_cov(*args, **kwargs)

    def forbid_dataset(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ODEAllocContractError("Wikipedia cache miss would require dataset access")

    original_backward = torch.autograd.backward

    def counted_backward(*args: Any, **kwargs: Any) -> Any:
        ledger.increment("backward_calls")
        return original_backward(*args, **kwargs)

    try:
        memit_main.compute_z = compute_z_once
        memit_main.compute_ks = compute_ks_once
        memit_main.get_cov = get_cov_once
        layer_stats_module.load_dataset = forbid_dataset
        torch.autograd.backward = counted_backward
        counts["native_factors"] += 1
        deltas = memit_main.execute_memit(
            model, tokenizer, [dict(request)], hparams, cache_template=None
        )
    finally:
        memit_main.compute_z = original_z
        memit_main.compute_ks = original_ks
        memit_main.get_cov = original_cov
        torch.autograd.backward = original_backward
        layer_stats_module.load_dataset = original_load_dataset
    expected_layer_count = len(hparams.layers)
    if (
        counts != {
            "direct_z": 1,
            "key": expected_layer_count,
            "covariance": expected_layer_count,
            "native_factors": 1,
        }
        or len(captured_z) != 1
        or tuple(sorted(captured_keys)) != tuple(sorted(hparams.layers))
        or tuple(sorted(cov_counts)) != tuple(sorted(hparams.layers))
    ):
        raise ODEAllocContractError("frozen Native input receipt counts differ")
    covariance_receipt = tuple(
        (int(layer), artifact_guard.spec["covariance"][str(layer)][2])
        for layer in sorted(hparams.layers)
    )
    return (
        deltas,
        captured_z[0],
        captured_keys,
        covariance_receipt,
        counts,
        memit_main,
    )


def _load_original_bf16(artifact_guard: P0ArtifactGuard) -> tuple[Any, Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from easyeditor.models.memit.memit_hparams import MEMITHyperParams

    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ODEAllocContractError("offline model environment is not locked")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ODEAllocContractError("P0 requires exactly one torch-visible GPU")
    tokenizer = AutoTokenizer.from_pretrained(
        artifact_guard.snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        artifact_guard.snapshot,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hparams = MEMITHyperParams.from_hparams(str(artifact_guard.hparams))
    hparams.device = 0
    hparams.stats_dir = str(artifact_guard.easyedit_root / "examples" / "data" / "stats")
    model.config._name_or_path = hparams.model_name
    observed_dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
    if observed_dtypes != {torch.bfloat16} or model.config.dtype is not torch.bfloat16:
        raise ODEAllocContractError("loaded original model is not uniformly BF16")
    if next(model.parameters()).device != torch.device("cuda:0"):
        raise ODEAllocContractError("P0 model is not on the single visible GPU")
    return model, tokenizer, hparams


def run_p0(
    *,
    repo_root: Path,
    alias: str,
    output_root: Path,
    source_head: str,
    numerical_lock_path: Path,
    artifact_lock_path: Path,
    seal_path: Path,
    run_token: str,
) -> dict[str, Any]:
    started = time.time()
    if alias not in MODEL_ALIASES:
        raise ODEAllocContractError("P0 alias is not approved")
    if source_head == EXPECTED_BASE:
        raise ODEAllocContractError("P0 must run from the new R1 source checkpoint")
    _source_freeze(repo_root, source_head)
    numerical_lock_sha = sha256_file(numerical_lock_path.resolve(strict=True))
    lock_value = json.loads(numerical_lock_path.read_text(encoding="utf-8"))
    if (
        lock_value.get("instruction_id") != INSTRUCTION_ID
        or lock_value.get("execution_seed") != SEED
        or lock_value["p0_identity_lock"].get("case_id") != CASE_ID
        or lock_value["p0_identity_lock"].get("request_sha256") != REQUEST_SHA256
        or lock_value["p0_identity_lock"].get("seal_root_digest") != SEAL_ROOT
    ):
        raise ODEAllocContractError("numerical P0 lock differs")
    expected_name = expected_result_name(alias, numerical_lock_sha, run_token)
    results_parent = (repo_root / "local" / "odealloc" / "results").resolve(strict=True)
    destination = output_root.resolve(strict=False)
    if destination.parent != results_parent or destination.name != expected_name:
        raise ODEAllocContractError("P0 output namespace differs")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("P0 output root is create-once")
    destination.mkdir(mode=0o700)
    raw_root = destination / "raw"
    raw_root.mkdir(mode=0o700)
    diagnostics = DiagnosticStageRecorder(raw_root)

    ledger = ComputeLedger()
    timers = ComponentTimer()
    guard = P0ArtifactGuard(artifact_lock_path, alias)
    with timers.measure("artifact_preflight"):
        artifact_receipt = guard.preflight()
        seal = load_and_verify_seal_candidate(seal_path)
        if seal["root_digest"] != SEAL_ROOT or seal["p0_identity"] != [
            {"case_id": CASE_ID, "request_hash": REQUEST_SHA256}
        ]:
            raise ODEAllocContractError("approved split/anchor seal differs")
        assert_seal_source_current(seal, guard.dataset)
        request = load_projected_request(
            guard.dataset, case_id=CASE_ID, expected_request_hash=REQUEST_SHA256
        )

    torch.cuda.reset_peak_memory_stats(0)
    _seed_all(SEED)
    with timers.measure("model_load"):
        model, tokenizer, hparams = _load_original_bf16(guard)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    forward_counter = ForwardCounter(model, ledger)
    mutation_lock = threading.RLock()
    try:
        with timers.measure("context_generation_twice"):
            contexts, context_hash_first, context_hash_second = _fresh_contexts_twice(
                model, tokenizer
            )
            _canonical_write_once(
                raw_root / "context_templates.json",
                {"schema": "ode-alloc-p0-raw-contexts/v1", "contexts": contexts},
            )
            prompt_templates = _flatten_contexts(contexts, request["prompt"])

        weight_names = tuple(
            f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in hparams.layers
        )
        parameters = dict(model.named_parameters())
        touched = {name: parameters[name] for name in weight_names}
        if any(parameter.dtype is not torch.bfloat16 for parameter in touched.values()):
            raise ODEAllocContractError("MEMIT target parameters are not BF16")
        entry_pointers = {name: parameter.data_ptr() for name, parameter in touched.items()}
        entry_weight_hashes = {name: tensor_sha256(parameter) for name, parameter in touched.items()}

        with timers.measure("entry_teacher_scoring"):
            entry_panel = score_panel(model, tokenizer, prompt_templates, request)

        oneshot = OneShotEditInputs(f"counterfact:{CASE_ID}")
        capture_box: dict[str, Any] = {}

        def build_factors() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
            result = _capture_native_factors(
                model, tokenizer, request, hparams, guard, ledger
            )
            diagnostics.complete("post_execute_memit")
            capture_box["factor_structure_sha256"] = diagnostics.write_structure_once(
                result[0]
            )
            capture_box["direct_z"] = result[1]
            capture_box["keys"] = result[2]
            capture_box["covariance"] = result[3]
            capture_box["counts"] = result[4]
            capture_box["memit_main"] = result[5]
            return result[0]

        diagnostic_wall_before_capture = diagnostics.wall_seconds
        with timers.measure("native_factor_capture_once"):
            deltas = oneshot.capture_from("native_factors", build_factors, factors_sha256)
            diagnostics.complete("post_factor_hash")
            direct_z = oneshot.capture_from(
                "direct_z", lambda: capture_box["direct_z"], tensor_sha256
            )
            keys = oneshot.capture_from(
                "key",
                lambda: capture_box["keys"],
                lambda values: canonical_hash(
                    {str(layer): tensor_sha256(value) for layer, value in sorted(values.items())}
                ),
            )
            covariance = oneshot.capture_from(
                "covariance",
                lambda: capture_box["covariance"],
                lambda value: canonical_hash(value),
            )
            frozen_receipt = oneshot.seal()
        diagnostic_capture_wall = diagnostics.wall_seconds - diagnostic_wall_before_capture
        if not 0.0 <= diagnostic_capture_wall <= timers.wall["native_factor_capture_once"]:
            raise ODEAllocContractError("diagnostic timing exclusion differs")
        timers.wall["native_factor_capture_once"] -= diagnostic_capture_wall
        post_factor_hashes = {name: tensor_sha256(parameter) for name, parameter in touched.items()}
        if post_factor_hashes != entry_weight_hashes:
            raise ODEAllocContractError("execute_memit did not restore W0 byte exactly")

        factors_by_weight: dict[str, FactorPair] = {}
        for layer, name in zip(hparams.layers, weight_names, strict=True):
            key_matrix, residual = deltas[name]
            factors_by_weight[name] = FactorPair(layer, residual, key_matrix)
        diagnostics.complete("post_factor_contract")
        del keys, covariance

        z_module = hparams.layer_module_tmp.format(hparams.layers[-1])
        diagnostics.complete("pre_direct_z_score")
        with timers.measure("direct_z_teacher_scoring"):
            direct_z_panel = score_panel(
                model,
                tokenizer,
                prompt_templates,
                request,
                direct_z=direct_z,
                z_module_name=z_module,
                fact_token=hparams.fact_token,
            )
        diagnostics.complete("post_direct_z_score")

        memit_main = capture_box["memit_main"]
        original_execute = memit_main.execute_memit
        replay_calls = 0

        def replay_once(*args: Any, **kwargs: Any) -> Any:
            nonlocal replay_calls
            del args, kwargs
            replay_calls += 1
            if replay_calls != 1:
                raise ODEAllocContractError("Native factor replay repeated")
            return deltas

        diagnostics.complete("pre_native_writer")
        with timers.measure("independent_easyedit_native_writer"):
            memit_main.execute_memit = replay_once
            try:
                _, w0_snapshots = memit_main.apply_memit_to_model(
                    model,
                    tokenizer,
                    [dict(request)],
                    hparams,
                    copy=False,
                    return_orig_weights=True,
                    cache_template=None,
                )
            finally:
                memit_main.execute_memit = original_execute
        diagnostics.complete("post_native_writer")
        if replay_calls != 1 or set(w0_snapshots) != set(weight_names):
            raise ODEAllocContractError("independent Native writer receipt differs")
        native_weight_hashes = {name: tensor_sha256(touched[name]) for name in weight_names}
        q0_byte_equal: dict[str, bool] = {}
        exact_zero_attestation: dict[int, bool] = {}
        for name, pair in factors_by_weight.items():
            expected = quantized_effective_weight(w0_snapshots[name], pair, 1.0)
            q0_byte_equal[name] = bool(torch.equal(expected, touched[name]))
            if factor_gram_energy(pair) == 0.0:
                exact_zero_attestation[pair.layer] = bool(
                    torch.equal(expected, w0_snapshots[name])
                )
            del expected
        if not all(q0_byte_equal.values()):
            raise ODEAllocContractError("q=0 blocked verdict bytes differ from EasyEdit Native")

        gate_config = RewriteGateConfig(
            rho=float(lock_value["rewrite_gate"]["rho"]),
            denominator_epsilon=float(lock_value["rewrite_gate"]["denominator_epsilon"]),
        )
        with timers.measure("native_teacher_scoring"):
            native_panel = score_panel(model, tokenizer, prompt_templates, request)
            native_event = _event(native_panel, entry_panel, direct_z_panel, gate_config)
        with torch.no_grad():
            for name in weight_names:
                touched[name].copy_(w0_snapshots[name])
        restored_after_native = {name: tensor_sha256(touched[name]) for name in weight_names}
        if restored_after_native != entry_weight_hashes:
            raise ODEAllocContractError("W0 restore after Native was not byte exact")
        for name in weight_names:
            w0_snapshots[name] = w0_snapshots[name].to(device="cpu")
        torch.cuda.empty_cache()

        gauge = FixedEnergyGauge(
            {pair.layer: pair for pair in factors_by_weight.values()},
            basis_energy_epsilon=float(lock_value["allocation_gauge"]["basis_energy_epsilon"]),
            max_abs_centered_q=float(lock_value["allocation_gauge"]["max_abs_centered_q"]),
            quantized_zero_by_layer=exact_zero_attestation,
        )
        gauge_reading = gauge.evaluate(gauge.zeros())
        if not torch.equal(gauge_reading.ratios, torch.ones_like(gauge_reading.ratios)):
            raise ODEAllocContractError("q=0 gauge ratios are not exact one")
        active_factors = {
            name: pair for name, pair in factors_by_weight.items() if pair.layer in gauge.layers
        }
        ratios = {layer: float(value) for layer, value in gauge_reading.ratio_by_layer().items()}

        diagnostics.complete("pre_q0_functional")
        ledger.increment("quantized_trial_calls")
        trial = QuantizedBF16FunctionalTrial(
            model, active_factors, ratios, row_block=64
        )
        with timers.measure("q0_virtual_teacher_scoring"):
            with trial:
                virtual_panel = score_panel(model, tokenizer, prompt_templates, request)
            virtual_event = _event(virtual_panel, entry_panel, direct_z_panel, gate_config)
        if (
            virtual_panel != native_panel
            or virtual_event != native_event
            or virtual_panel.new_logits_sha256 != native_panel.new_logits_sha256
            or virtual_panel.old_logits_sha256 != native_panel.old_logits_sha256
        ):
            raise ODEAllocContractError("q=0 virtual output/event differs from Native")
        max_full_weight_elements = max(touched[name].numel() for name in active_factors)
        if (
            trial.max_live_effective_weights != 1
            or trial.max_fp32_delta_block_elements >= max_full_weight_elements
        ):
            raise ODEAllocContractError("quantized verdict retained a dense FP32 delta")
        diagnostics.complete("post_q0_functional")

        faulting = AtomicLayerTransaction(touched, mutation_lock=mutation_lock)
        for name, pair in factors_by_weight.items():
            faulting.stage(name, quantized_effective_weight(touched[name], pair, 1.0))
        fault_raised = False
        with timers.measure("atomic_fault_rollback"):
            try:
                faulting.commit(fault_after_writes=max(1, len(weight_names) // 2))
            except RuntimeError as exc:
                if "injected" not in str(exc):
                    raise
                fault_raised = True
        if not fault_raised or {
            name: tensor_sha256(touched[name]) for name in weight_names
        } != entry_weight_hashes:
            raise ODEAllocContractError("fault-injected all-layer rollback differs from W0")

        transaction = AtomicLayerTransaction(touched, mutation_lock=mutation_lock)
        for name, pair in factors_by_weight.items():
            transaction.stage(name, quantized_effective_weight(touched[name], pair, 1.0))
        with timers.measure("atomic_q0_commit"):
            commit_receipt = transaction.commit(ledger=ledger)
        committed_weight_hashes = {
            name: tensor_sha256(touched[name]) for name in weight_names
        }
        if committed_weight_hashes != native_weight_hashes:
            raise ODEAllocContractError("q=0 committed bytes differ from Native")
        with timers.measure("committed_teacher_scoring"):
            committed_panel = score_panel(model, tokenizer, prompt_templates, request)
            committed_event = _event(committed_panel, entry_panel, direct_z_panel, gate_config)
        if committed_panel != virtual_panel or committed_event != virtual_event:
            raise ODEAllocContractError("q=0 transactional output/event differs from virtual")
        diagnostics.complete("post_q0_commit")

        cleanup = AtomicLayerTransaction(touched, mutation_lock=mutation_lock)
        for name in weight_names:
            cleanup.stage(name, w0_snapshots[name])
        with timers.measure("cleanup_w0_restore"):
            cleanup.commit()
        final_weight_hashes = {name: tensor_sha256(touched[name]) for name in weight_names}
        if final_weight_hashes != entry_weight_hashes:
            raise ODEAllocContractError("terminal original W0 restore differs")
        if any(
            touched[name].data_ptr() != entry_pointers[name]
            or touched[name].grad is not None
            or touched[name].requires_grad
            for name in weight_names
        ):
            raise ODEAllocContractError("terminal parameter pointer/grad state differs")

        oneshot.assert_current()
        guard.assert_unchanged()
        torch.cuda.synchronize()
        allocated_peak = int(torch.cuda.max_memory_allocated(0))
        reserved_peak = int(torch.cuda.max_memory_reserved(0))
        ledger.observe_peak_memory(allocated_peak)
        ledger.add_seconds("controller_wall_seconds", sum(timers.wall.values()))
        ledger.add_seconds("gpu_seconds", sum(timers.gpu.values()))
        diagnostics.complete("post_identity_verdict")
        manifest: dict[str, Any] = {
            "schema_version": "ode-alloc-s04-p0-native-identity-r3/v1",
            "status": "PASS_TECHNICAL_IDENTITY_ONLY",
            "instruction_id": INSTRUCTION_ID,
            "diagnostic_instruction_id": DIAGNOSTIC_INSTRUCTION_ID,
            "run_token": run_token,
            "session_id": SESSION_ID,
            "source_head": source_head,
            "expected_parent": EXPECTED_BASE,
            "model_alias": alias,
            "case_id": CASE_ID,
            "request_sha256": REQUEST_SHA256,
            "seed": SEED,
            "scientific_outcome_count": 0,
            "heldout_access": {
                "paraphrase": 0,
                "locality": 0,
                "evaluation_generation": 0,
                "evaluation_outcomes": 0,
            },
            "locks": {
                "numerical_lock_sha256": numerical_lock_sha,
                "artifact_lock_sha256": artifact_receipt.lock_sha256,
                "artifact_lock_canonical_sha256": artifact_receipt.lock_canonical_sha256,
                "seal_root_digest": seal["root_digest"],
                "dataset_sha256": artifact_receipt.counterfact,
                "easyedit_sources_root": artifact_receipt.easyedit_sources_root,
                "hparams_sha256": artifact_receipt.hparams,
                "model_revision": artifact_receipt.revision,
                "covariance": list(artifact_receipt.covariance),
            },
            "model": {
                "config_dtype": str(model.config.dtype),
                "observed_parameter_dtype": "torch.bfloat16",
                "parameter_count": parameter_count,
                "visible_cuda_device_count": torch.cuda.device_count(),
            },
            "contexts": {
                "policy": "fresh-original-bf16-generation-twice-exact-match",
                "count": len(prompt_templates),
                "first_sha256": context_hash_first,
                "second_sha256": context_hash_second,
                "exact_match": context_hash_first == context_hash_second,
                "raw_location": "raw/context_templates.json",
            },
            "frozen_inputs": {
                "receipt_id": frozen_receipt.receipt_id,
                "capture_counts": dict(frozen_receipt.capture_counts),
                "underlying_counts": capture_box["counts"],
                "native_replay_writer_calls": replay_calls,
            },
            "diagnostic": {
                "last_completed_stage": diagnostics.last_completed_stage,
                "stage_receipt_root_sha256": diagnostics.receipt_root_sha256(),
                "factor_structure_sha256": capture_box["factor_structure_sha256"],
                "instrumentation_wall_seconds": diagnostics.wall_seconds,
                "excluded_from_scientific_compute": True,
            },
            "gauge": {
                "layers": list(gauge.layers),
                "excluded_exact_zero_layers": list(gauge.excluded_layers),
                "dimension": gauge.dimension,
                "energies": [float(value) for value in gauge.energies],
                "ratios_exact_one": True,
                "energy_exact": bool(
                    torch.equal(gauge_reading.energy_before, gauge_reading.energy_after)
                ),
                "q_cap_hit": False,
            },
            "identity": {
                "native_vs_blocked_q0_parameter_bytes": q0_byte_equal,
                "native_vs_virtual_teacher_logits_exact": True,
                "native_vs_virtual_event_exact": True,
                "virtual_vs_committed_teacher_logits_exact": True,
                "virtual_vs_committed_event_exact": True,
                "native_event": _event_record(native_event),
                "native_logits_sha256": {
                    "new": native_panel.new_logits_sha256,
                    "old": native_panel.old_logits_sha256,
                },
                "virtual_logits_sha256": {
                    "new": virtual_panel.new_logits_sha256,
                    "old": virtual_panel.old_logits_sha256,
                },
                "committed_logits_sha256": {
                    "new": committed_panel.new_logits_sha256,
                    "old": committed_panel.old_logits_sha256,
                },
                "entry_weight_sha256": entry_weight_hashes,
                "native_weight_sha256": native_weight_hashes,
                "committed_weight_sha256": committed_weight_hashes,
                "final_restored_weight_sha256": final_weight_hashes,
            },
            "transaction": {
                "exclusive_lock": True,
                "fault_injected_rollback_exact": fault_raised,
                "experiment_commit_count": commit_receipt.commit_count,
                "cleanup_restore_count": 1,
                "staged_candidate_device": "cpu-bfloat16",
                "original_w0_terminal_restore_exact": True,
            },
            "functional_verdict": {
                "gradient_overlay_used_as_verdict": False,
                "quantized_full_linear": True,
                "max_live_effective_bf16_weights": trial.max_live_effective_weights,
                "max_fp32_delta_block_elements": trial.max_fp32_delta_block_elements,
                "full_fp32_dense_delta_retained": 0,
                "replacement_linear_calls": trial.replacement_linear_calls,
                "duplicate_gemm_calls": trial.replacement_linear_calls,
                "pointer_version_grad_rng_invariant": True,
            },
            "compute": {
                "ledger": ledger.snapshot(),
                "component_wall_seconds": timers.wall,
                "component_gpu_seconds": timers.gpu,
                "peak_allocated_bytes": allocated_peak,
                "peak_reserved_bytes": reserved_peak,
                "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                * 1024,
            },
            "artifact_postflight_unchanged": True,
            "started_unix": started,
            "finished_unix": time.time(),
        }
        manifest["manifest_id"] = canonical_hash(manifest)
        manifest_sha = _canonical_write_once(destination / "manifest.json", manifest)
        summary = {
            "schema_version": "ode-alloc-s04-p0-terminal-summary-r3/v1",
            "status": manifest["status"],
            "model_alias": alias,
            "source_head": source_head,
            "numerical_lock_sha256": numerical_lock_sha,
            "seal_root_digest": SEAL_ROOT,
            "case_id": CASE_ID,
            "scientific_outcome_count": 0,
            "identity_pass": True,
            "event_id": native_event.event_id,
            "capture_counts": dict(frozen_receipt.capture_counts),
            "peak_allocated_bytes": allocated_peak,
            "peak_reserved_bytes": reserved_peak,
            "host_max_rss_bytes": manifest["compute"]["host_max_rss_bytes"],
            "elapsed_seconds": time.time() - started,
        }
        summary["summary_id"] = canonical_hash(summary)
        summary_sha = _canonical_write_once(destination / "summary.json", summary)
        terminal = {
            "schema_version": "ode-alloc-s04-p0-terminal-receipt-r3/v1",
            "status": "PASS",
            "model_alias": alias,
            "manifest_sha256": manifest_sha,
            "summary_sha256": summary_sha,
        }
        terminal["terminal_id"] = canonical_hash(terminal)
        terminal_sha = _canonical_write_once(destination / "terminal.json", terminal)
        return {
            "status": "PASS",
            "alias": alias,
            "root": str(destination),
            "manifest_sha256": manifest_sha,
            "summary_sha256": summary_sha,
            "terminal_sha256": terminal_sha,
            "peak_allocated_bytes": allocated_peak,
            "peak_reserved_bytes": reserved_peak,
        }
    finally:
        forward_counter.close()
        with contextlib.suppress(Exception):
            from easyeditor.models.memit import memit_main

            memit_main.COV_CACHE.clear()
            memit_main.CONTEXT_TEMPLATES_CACHE = None


def write_failure_once(output_root: Path, exc: BaseException) -> str | None:
    if not output_root.exists() or not output_root.is_dir():
        return None
    destination = output_root / "failure.json"
    if destination.exists() or destination.is_symlink():
        return None
    payload = {
        "schema_version": "ode-alloc-s04-p0-failure-r1/v1",
        "status": "FAIL_CLOSED_NO_RETRY",
        "error_type": type(exc).__name__,
        "error_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
        "scientific_outcome_count": 0,
        "failed_unix": time.time(),
    }
    return _canonical_write_once(destination, payload)
