"""MV-0 native MEMIT fidelity and trace-neutrality runner.

The CLI writes only sanitized, compact artifacts beneath the repository's
``local/`` tree.  It never writes Git evidence, downloads data, computes
statistics, or deserializes AlphaEdit projectors.  Per-request direct-z values
are computed once and frozen only inside the exclusive repo-local run folder.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import resource
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch

from .contracts import (
    ContextManifest,
    ContractError,
    EditRequest,
    ExpectedFileIdentity,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
    ProvenanceManifest,
    ProvenanceMismatchError,
    orient_easyedit_factor,
)
from .diagnostic_math import c_inner_product
from .easyedit_bridge import (
    APPROVED_EASYEDIT_FILES,
    CovarianceCacheMissError,
    CovarianceCacheSpec,
    EasyEditBindings,
    EasyEditBridge,
)
from .gpu_runtime import (
    FixedModelRuntime,
    GpuRuntimeError,
    load_fixed_model,
    offline_environment,
    rng_state_hash,
    seed_runtime,
)
from .hooks import (
    ForwardCapture,
    ConcurrentWeightMutationError,
    TemporaryExactMemitApplication,
    WeightHashMismatchError,
    resolve_parameter,
    tensor_sha256,
)
from .manifests import (
    DEFAULT_SELECTION_SEED,
    FIXED_FILE_IDENTITIES,
    MODEL_SPECS,
    CounterFactSelectionManifest,
    FixedModelSpec,
    fixed_model_spec,
    generate_counterfact_selection,
    load_counterfact_requests,
    preflight_fixed_artifacts,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GIT_BIN = Path("/usr/bin/git")
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "local/results/raw/session01_motivation"
)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FORBIDDEN_ARTIFACT_KEYS = {
    "prompt",
    "prompts",
    "subject",
    "target_new",
    "target_true",
    "requested_rewrite",
    "paraphrase_prompts",
    "neighborhood_prompts",
    "attribute_prompts",
    "generation_prompts",
    "locality",
    "ground_truth",
    "evaluation",
}


class MV0Error(RuntimeError):
    """A fail-closed MV-0 contract violation."""


class RecomputeBlockedError(MV0Error):
    """A covariance download/recomputation path was requested."""


class RollbackError(MV0Error):
    """An independent branch did not return to byte-identical state."""


_FATAL_EVENT_ERRORS = (
    ContractError,
    ProvenanceMismatchError,
    MV0Error,
    CovarianceCacheMissError,
    WeightHashMismatchError,
    ConcurrentWeightMutationError,
    torch.cuda.OutOfMemoryError,
    MemoryError,
    OSError,
)


def _safe_payload(value: Any) -> Any:
    """Convert to strict JSON while rejecting raw edit/evaluation fields."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MV0Error("artifact payload contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise MV0Error("raw tensors are forbidden in MV-0 artifacts")
        return _safe_payload(value.detach().cpu().item())
    if dataclasses.is_dataclass(value):
        return _safe_payload(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in _FORBIDDEN_ARTIFACT_KEYS:
                raise MV0Error(f"forbidden artifact field: {key}")
            result[key] = _safe_payload(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_payload(item) for item in value]
    raise MV0Error(f"unsupported artifact value: {type(value).__name__}")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    safe = _safe_payload(payload)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            safe,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class SanitizedJsonlWriter:
    def __init__(
        self,
        path: Path,
        run_id: str,
        *,
        schema_version: str = "ode-edit-mv0/v1",
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.schema_version = schema_version
        self.sequence = 0
        self.handle: Any = None

    def __enter__(self) -> "SanitizedJsonlWriter":
        self.handle = self.path.open("x", encoding="utf-8", newline="\n")
        return self

    def write(self, event: str, payload: Mapping[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("JSONL writer is not open")
        record = _safe_payload(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "sequence": self.sequence,
                "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "event": event,
                "payload": payload,
            }
        )
        self.handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        self.handle.flush()
        self.sequence += 1

    def sync(self) -> None:
        if self.handle is None:
            raise RuntimeError("JSONL writer is not open")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        return False


@contextlib.contextmanager
def _silence_upstream() -> Iterator[None]:
    """Discard EasyEdit prints, which include raw rewrite text."""

    with Path(os.devnull).open("w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def _bridge_pins() -> dict[str, Mapping[str, Any]]:
    """Use the bridge's exact subset while full preflight covers extra sources."""

    missing = set(APPROVED_EASYEDIT_FILES) - set(FIXED_FILE_IDENTITIES)
    if missing:
        raise MV0Error(
            f"bridge source pins are absent from the fixed manifest: {sorted(missing)}"
        )
    return {
        path: {
            "sha256": FIXED_FILE_IDENTITIES[path].sha256,
            "size": FIXED_FILE_IDENTITIES[path].size,
        }
        for path in APPROVED_EASYEDIT_FILES
    }


def _covariance_specs(
    root: Path,
    spec: FixedModelSpec,
) -> tuple[CovarianceCacheSpec, ...]:
    """Return exact contracts for already-computed Wikipedia moments."""

    result = []
    for layer in spec.layers:
        relative = spec.covariance_path_for_layer(layer)
        identity = FIXED_FILE_IDENTITIES[relative]
        result.append(
            CovarianceCacheSpec(
                layer=layer,
                path=str((root / relative).resolve(strict=True)),
                identity=ExpectedFileIdentity(
                    sha256=identity.sha256,
                    size=identity.size,
                ),
            )
        )
    return tuple(result)


def _load_verified_covariances(
    *,
    root: Path,
    runtime: FixedModelRuntime,
    specs: Sequence[CovarianceCacheSpec],
) -> tuple[dict[int, torch.Tensor], tuple[str, ...]]:
    """Load pinned moments for metrics only; never enter dataset code."""

    try:
        import numpy as np
    except ImportError as exc:
        raise MV0Error("NumPy is required to read the pinned covariance") from exc
    moments: dict[int, torch.Tensor] = {}
    paths: list[str] = []
    for cache_spec in specs:
        relative = runtime.spec.covariance_path_for_layer(cache_spec.layer)
        path = (root / relative).resolve(strict=True)
        if str(path) != str(Path(cache_spec.path).resolve(strict=True)):
            raise RecomputeBlockedError("covariance path differs from the fixed model map")
        if (
            path.stat().st_size != cache_spec.identity.size
            or _file_sha256(path) != cache_spec.identity.sha256
        ):
            raise RecomputeBlockedError(
                f"covariance identity changed before metric load: layer {cache_spec.layer}"
            )
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["sample_size"]) != 100_000:
                raise RecomputeBlockedError("covariance sample_size metadata mismatch")
            count = int(archive["mom2.count"])
            raw = archive["mom2.mom2"]
            weight = resolve_parameter(
                runtime.model,
                f"model.layers.{cache_spec.layer}.mlp.down_proj.weight",
            )
            expected_dim = int(weight.shape[1])
            if (
                count <= 0
                or raw.dtype.name != "float32"
                or tuple(raw.shape) != (expected_dim, expected_dim)
            ):
                raise RecomputeBlockedError("covariance payload shape/dtype/count mismatch")
            moment = torch.from_numpy(raw).float() / count
        if not bool(torch.isfinite(moment).all()):
            raise RecomputeBlockedError("covariance contains non-finite values")
        moments[cache_spec.layer] = moment
        paths.append(relative)
    return moments, tuple(paths)


def _load_hparams(
    root: Path,
    spec: FixedModelSpec,
    bindings: EasyEditBindings,
) -> Any:
    path = (root / spec.hparams_path).resolve(strict=True)
    hparams = bindings.memit_hparams.MEMITHyperParams.from_hparams(str(path))
    if (
        tuple(int(layer) for layer in hparams.layers) != spec.layers
        or str(hparams.model_name) != spec.repository_id
        or str(hparams.mom2_dataset) != "wikipedia"
        or int(hparams.mom2_n_samples) != 100_000
        or str(hparams.mom2_dtype) != "float32"
        or str(hparams.rewrite_module_tmp) != "model.layers.{}.mlp.down_proj"
    ):
        raise MV0Error("MEMIT hparams differ from the fixed MV-0 contract")
    hparams.device = 0
    hparams.stats_dir = str((root / "examples/data/stats").resolve(strict=True))
    return hparams


def _freeze_contexts(
    bridge: EasyEditBridge,
    runtime: FixedModelRuntime,
    *,
    seed: int,
) -> ContextManifest:
    seed_runtime(seed)
    with _silence_upstream():
        contexts = bridge.freeze_generated_contexts(
            runtime.model,
            runtime.tokenizer,
            source=f"{runtime.spec.alias}:fresh-seed-{seed}",
            fresh=True,
        )
    flattened = [item for group in contexts.templates for item in group]
    if len(flattened) != 6 or contexts.templates[0] != ("{}",):
        raise MV0Error("EasyEdit did not produce base plus five frozen contexts")
    return contexts


@dataclass(frozen=True, slots=True)
class NativeDelta:
    weight_name: str
    adjusted_keys: torch.Tensor
    residuals: torch.Tensor
    raw_update_transposed: bool


@dataclass(frozen=True, slots=True)
class NativeExecution:
    proposal: MemitFactorProposal
    deltas: tuple[NativeDelta, ...]
    forward_calls: int
    solve_count: int


class _ForwardCounter:
    def __init__(self, model: torch.nn.Module) -> None:
        self.count = 0
        self.handle = model.register_forward_pre_hook(self._hook)

    def _hook(self, module: torch.nn.Module, inputs: Any) -> None:
        del module, inputs
        self.count += 1

    def close(self) -> None:
        self.handle.remove()


def _execution_from_ordered_proposal(
    proposal: MemitFactorProposal,
    *,
    forward_calls: int,
) -> NativeExecution:
    """Recover EasyEdit's raw factor order from the guarded canonical proposal."""

    if proposal.semantics is not ProposalSemantics.ORDERED_GAUSS_SEIDEL:
        raise MV0Error("MV-0 native execution requires an ordered MEMIT proposal")
    deltas: list[NativeDelta] = []
    for factor in proposal.factors:
        if factor.native_update_transposed:
            adjusted, residual = factor.right, factor.left
        else:
            adjusted, residual = factor.left, factor.right
        deltas.append(
            NativeDelta(
                weight_name=factor.weight_name,
                adjusted_keys=adjusted.detach().cpu().clone(),
                residuals=residual.detach().cpu().clone(),
                raw_update_transposed=factor.native_update_transposed,
            )
        )
    return NativeExecution(
        proposal=proposal,
        deltas=tuple(deltas),
        forward_calls=forward_calls,
        solve_count=len(deltas),
    )


def _weight_hashes(
    model: torch.nn.Module,
    names: Iterable[str],
) -> dict[str, str]:
    return {name: tensor_sha256(resolve_parameter(model, name)) for name in names}


class ExactWeightBranch:
    """Restore full parameter copies and verify exact base hashes on every exit."""

    def __init__(
        self,
        model: torch.nn.Module,
        expected_hashes: Mapping[str, str],
    ) -> None:
        self.model = model
        self.expected_hashes = dict(expected_hashes)
        self.backups: dict[str, torch.Tensor] = {}

    def __enter__(self) -> "ExactWeightBranch":
        actual = _weight_hashes(self.model, self.expected_hashes)
        if actual != self.expected_hashes:
            raise RollbackError("independent branch did not start from the base bytes")
        self.backups = {
            name: resolve_parameter(self.model, name).detach().clone()
            for name in self.expected_hashes
        }
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        with torch.no_grad():
            for name, backup in self.backups.items():
                resolve_parameter(self.model, name).copy_(backup)
        actual = _weight_hashes(self.model, self.expected_hashes)
        self.backups.clear()
        if actual != self.expected_hashes:
            error = RollbackError("exact branch rollback hash mismatch")
            if exc_value is not None and hasattr(exc_value, "add_note"):
                exc_value.add_note(str(error))
                return False
            raise error
        return False


def _apply_native_deltas(
    model: torch.nn.Module,
    deltas: Sequence[NativeDelta],
    bindings: EasyEditBindings,
) -> None:
    with torch.no_grad():
        for delta in deltas:
            parameter = resolve_parameter(model, delta.weight_name)
            adjusted = delta.adjusted_keys.to(parameter.device)
            residual = delta.residuals.to(parameter.device)
            update = adjusted @ residual.transpose(0, 1)
            update = bindings.memit_main.upd_matrix_match_shape(
                update, parameter.shape
            )
            parameter[...] += update.float()


def _apply_bridge_exact(
    model: torch.nn.Module,
    execution: NativeExecution,
    bindings: EasyEditBindings,
) -> None:
    """Rebuild native GEMM order from canonical factors, then apply exactly."""

    factors = {
        factor.weight_name: factor for factor in execution.proposal.factors
    }
    with torch.no_grad():
        for delta in execution.deltas:
            factor = factors[delta.weight_name]
            parameter = resolve_parameter(model, delta.weight_name)
            if delta.raw_update_transposed:
                adjusted, residual = factor.right, factor.left
            else:
                adjusted, residual = factor.left, factor.right
            update = adjusted.to(parameter.device) @ residual.to(
                parameter.device
            ).transpose(0, 1)
            update = bindings.memit_main.upd_matrix_match_shape(
                update, parameter.shape
            )
            parameter[...] += update.float()


@dataclass(slots=True)
class TeacherForcedResult:
    logits: torch.Tensor
    nll: float
    context_nll: tuple[float, ...]
    target_token_count: int

    @property
    def logits_hash(self) -> str:
        return tensor_sha256(self.logits)


def teacher_forced_rewrite(
    model: torch.nn.Module,
    tokenizer: Any,
    request: EditRequest,
    contexts: ContextManifest,
) -> TeacherForcedResult:
    """Evaluate full-vocabulary logits at every allowed rewrite target token."""

    target_text = request.to_easyedit()["target_new"]
    target_ids = tokenizer.encode(
        target_text,
        return_tensors="pt",
        add_special_tokens=False,
    )[0]
    if target_ids.numel() == 0:
        raise MV0Error("rewrite target tokenization is empty")
    continuation = tokenizer.decode(target_ids[:-1])
    templates = [item for group in contexts.templates for item in group]
    inputs = [
        context.format(request.prompt).format(request.subject) + continuation
        for context in templates
    ]
    batch = tokenizer(inputs, return_tensors="pt", padding=True).to(
        next(model.parameters()).device
    )
    selected: list[torch.Tensor] = []
    with torch.inference_mode():
        logits = model(**batch).logits
        for row in range(logits.shape[0]):
            positions = torch.nonzero(
                batch["attention_mask"][row], as_tuple=False
            ).flatten()
            if positions.numel() < target_ids.numel():
                raise MV0Error("rewrite input is shorter than its target")
            selected.append(logits[row, positions[-target_ids.numel() :], :].float())
    selected_logits = torch.stack(selected, dim=0)
    targets = target_ids.to(selected_logits.device).expand(
        selected_logits.shape[0], -1
    )
    losses = torch.nn.functional.cross_entropy(
        selected_logits.reshape(-1, selected_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(selected_logits.shape[:2])
    return TeacherForcedResult(
        logits=selected_logits.detach().cpu().contiguous(),
        nll=float(losses.mean().detach().cpu()),
        context_nll=tuple(float(value) for value in losses.mean(dim=1).detach().cpu()),
        target_token_count=int(target_ids.numel()),
    )


@dataclass(slots=True)
class BranchResult:
    teacher: TeacherForcedResult
    updated_weights: dict[str, torch.Tensor]
    updated_hashes: dict[str, str]


def _run_apply_branch(
    *,
    model: torch.nn.Module,
    base_hashes: Mapping[str, str],
    apply: Callable[[], None],
    evaluate: Callable[[], TeacherForcedResult],
) -> BranchResult:
    with ExactWeightBranch(model, base_hashes):
        apply()
        teacher = evaluate()
        hashes = _weight_hashes(model, base_hashes)
        weights = {
            name: resolve_parameter(model, name).detach().cpu().clone()
            for name in base_hashes
        }
        return BranchResult(
            teacher=teacher,
            updated_weights=weights,
            updated_hashes=hashes,
        )


def _run_exact_proposal_branch(
    *,
    model: torch.nn.Module,
    base_hashes: Mapping[str, str],
    proposal: MemitFactorProposal,
    evaluate: Callable[[], TeacherForcedResult],
) -> BranchResult:
    """Evaluate the reusable exact adapter while it is temporarily active."""

    with ExactWeightBranch(model, base_hashes):
        with TemporaryExactMemitApplication(model, proposal):
            teacher = evaluate()
            hashes = _weight_hashes(model, base_hashes)
            weights = {
                name: resolve_parameter(model, name).detach().cpu().clone()
                for name in base_hashes
            }
            return BranchResult(
                teacher=teacher,
                updated_weights=weights,
                updated_hashes=hashes,
            )


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return 0.0 if numerator == 0 else None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _cosine_float64(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    chunk_size: int = 1_048_576,
) -> float | None:
    """Compute bounded cosine without full-tensor FP64 copies."""

    ref_flat = reference.reshape(-1)
    cand_flat = candidate.reshape(-1)
    dot = ref_sq = cand_sq = 0.0
    for start in range(0, ref_flat.numel(), chunk_size):
        ref_chunk = ref_flat[start : start + chunk_size].double()
        cand_chunk = cand_flat[start : start + chunk_size].double()
        dot += float(torch.dot(ref_chunk, cand_chunk))
        ref_sq += float(torch.dot(ref_chunk, ref_chunk))
        cand_sq += float(torch.dot(cand_chunk, cand_chunk))
    cosine = _ratio(dot, math.sqrt(ref_sq) * math.sqrt(cand_sq))
    if cosine is None:
        return None
    return max(-1.0, min(1.0, cosine))


def compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    base: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compare materialized tensors; optionally compare their deltas from base."""

    if reference.shape != candidate.shape:
        raise MV0Error("tensor comparison shape mismatch")
    ref = reference.float()
    cand = candidate.float()
    if base is not None:
        base_value = base.detach().cpu().float()
        ref = ref - base_value
        cand = cand - base_value
    difference = cand - ref
    ref_norm = float(torch.linalg.vector_norm(ref))
    cand_norm = float(torch.linalg.vector_norm(cand))
    diff_norm = float(torch.linalg.vector_norm(difference))
    cosine = _cosine_float64(ref, cand)
    return {
        "reference_norm": ref_norm,
        "candidate_norm": cand_norm,
        "relative_l2_error": _ratio(diff_norm, ref_norm),
        "relative_norm_error": _ratio(abs(cand_norm - ref_norm), ref_norm),
        "max_abs_error": float(difference.abs().max()),
        "cosine": cosine,
        "exact_bytes": tensor_sha256(reference) == tensor_sha256(candidate),
    }


def compare_teacher(
    reference: TeacherForcedResult,
    candidate: TeacherForcedResult,
) -> dict[str, Any]:
    metrics = compare_tensors(reference.logits, candidate.logits)
    metrics.update(
        {
            "nll_abs_error": abs(candidate.nll - reference.nll),
            "nll_relative_error": _ratio(
                abs(candidate.nll - reference.nll), abs(reference.nll)
            ),
            "max_context_nll_abs_error": max(
                abs(left - right)
                for left, right in zip(
                    reference.context_nll, candidate.context_nll
                )
            ),
            "logits_hash_equal": reference.logits_hash == candidate.logits_hash,
        }
    )
    return metrics


def _factor_equal(first: LowRankFactor, second: LowRankFactor) -> bool:
    return (
        first.weight_name == second.weight_name
        and torch.equal(first.left, second.left)
        and torch.equal(first.right, second.right)
        and first.native_update_transposed == second.native_update_transposed
    )


def _c_metrics(
    native: LowRankFactor,
    bridge: LowRankFactor,
    covariance: torch.Tensor,
) -> dict[str, Any]:
    native32 = LowRankFactor(
        native.weight_name,
        native.left.float(),
        native.right.float(),
        native.expected_weight_sha256,
        native.native_update_transposed,
    )
    bridge32 = LowRankFactor(
        bridge.weight_name,
        bridge.left.float(),
        bridge.right.float(),
        bridge.expected_weight_sha256,
        bridge.native_update_transposed,
    )
    native_sq = float(c_inner_product(native32, native32, covariance))
    if native_sq < 0:
        raise MV0Error("native factor has a negative C squared norm")
    native_norm = math.sqrt(native_sq)
    if _factor_equal(native32, bridge32):
        return {
            "native_c_norm": native_norm,
            "bridge_c_norm": native_norm,
            "c_cosine": 1.0 if native_norm else None,
            "relative_c_norm_error": 0.0,
        }
    bridge_sq = float(c_inner_product(bridge32, bridge32, covariance))
    cross = float(c_inner_product(native32, bridge32, covariance))
    bridge_norm = math.sqrt(max(0.0, bridge_sq))
    return {
        "native_c_norm": native_norm,
        "bridge_c_norm": bridge_norm,
        "c_cosine": _ratio(cross, native_norm * bridge_norm),
        "relative_c_norm_error": _ratio(
            abs(bridge_norm - native_norm), native_norm
        ),
    }


def _trace_neutrality(
    *,
    runtime: FixedModelRuntime,
    hparams: Any,
    request: EditRequest,
    contexts: ContextManifest,
    base_hashes: Mapping[str, str],
) -> tuple[TeacherForcedResult, dict[str, Any]]:
    state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(0).clone()
    baseline = teacher_forced_rewrite(
        runtime.model, runtime.tokenizer, request, contexts
    )
    torch.set_rng_state(state)
    torch.cuda.set_rng_state(cuda_state, 0)
    captures: list[ForwardCapture] = []
    with contextlib.ExitStack() as stack:
        for layer in hparams.layers:
            capture = stack.enter_context(
                ForwardCapture(
                    runtime.model,
                    hparams.rewrite_module_tmp.format(layer),
                    retain_input=True,
                    retain_output=True,
                    detach=True,
                    clone=False,
                )
            )
            captures.append(capture)
        traced = teacher_forced_rewrite(
            runtime.model, runtime.tokenizer, request, contexts
        )
    trace_metrics = compare_teacher(baseline, traced)
    weights_equal = _weight_hashes(runtime.model, base_hashes) == dict(base_hashes)
    invoked = all(capture.input is not None and capture.output is not None for capture in captures)
    return baseline, {
        **trace_metrics,
        "weights_unchanged": weights_equal,
        "all_layers_invoked": invoked,
        "pass": bool(
            weights_equal
            and invoked
            and trace_metrics["logits_hash_equal"]
            and trace_metrics["exact_bytes"]
        ),
    }


def _event_seed(seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _run_event(
    *,
    runtime: FixedModelRuntime,
    bridge: EasyEditBridge,
    bindings: EasyEditBindings,
    hparams: Any,
    contexts: ContextManifest,
    covariance_specs: Sequence[CovarianceCacheSpec],
    covariance_moments: Mapping[int, torch.Tensor],
    direct_z_root: Path,
    request: EditRequest,
    seed: int,
) -> dict[str, Any]:
    weight_names = tuple(
        f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in hparams.layers
    )
    base_hashes = _weight_hashes(runtime.model, weight_names)
    event_seed = _event_seed(seed, request.case_id)
    seed_runtime(event_seed)
    baseline, trace = _trace_neutrality(
        runtime=runtime,
        hparams=hparams,
        request=request,
        contexts=contexts,
        base_hashes=base_hashes,
    )
    seed_runtime(event_seed)
    counter = _ForwardCounter(runtime.model)
    try:
        with _silence_upstream():
            direct_z = bridge.load_or_compute_direct_z(
                runtime.model,
                runtime.tokenizer,
                (request,),
                hparams,
                contexts,
                model_id=runtime.spec.snapshot_name,
                local_cache_root=direct_z_root,
                cache_path=f"{request.request_id}.pt",
            )
            ordered = bridge.propose_ordered_memit_factors(
                runtime.model,
                runtime.tokenizer,
                (request,),
                hparams,
                contexts,
                model_id=runtime.spec.snapshot_name,
                direct_z=direct_z,
                covariance_caches=covariance_specs,
            )
    finally:
        counter.close()
    execution = _execution_from_ordered_proposal(
        ordered,
        forward_calls=counter.count,
    )
    if _weight_hashes(runtime.model, weight_names) != base_hashes:
        raise RollbackError("execute_memit did not restore the base state")

    branch_cpu_rng = torch.get_rng_state().clone()
    branch_cuda_rng = torch.cuda.get_rng_state(0).clone()

    def reset_branch_rng() -> None:
        torch.set_rng_state(branch_cpu_rng)
        torch.cuda.set_rng_state(branch_cuda_rng, 0)

    evaluate = lambda: teacher_forced_rewrite(
        runtime.model, runtime.tokenizer, request, contexts
    )
    reset_branch_rng()
    native = _run_apply_branch(
        model=runtime.model,
        base_hashes=base_hashes,
        apply=lambda: _apply_native_deltas(
            runtime.model, execution.deltas, bindings
        ),
        evaluate=evaluate,
    )
    reset_branch_rng()
    replay = _run_apply_branch(
        model=runtime.model,
        base_hashes=base_hashes,
        apply=lambda: _apply_native_deltas(
            runtime.model, execution.deltas, bindings
        ),
        evaluate=evaluate,
    )
    reset_branch_rng()
    bridge_result = _run_exact_proposal_branch(
        model=runtime.model,
        base_hashes=base_hashes,
        proposal=execution.proposal,
        evaluate=evaluate,
    )
    if _weight_hashes(runtime.model, weight_names) != base_hashes:
        raise RollbackError("event branches did not restore the base state")

    layer_metrics: list[dict[str, Any]] = []
    replay_layers: list[dict[str, Any]] = []
    aggregate_ref_sq = aggregate_cand_sq = aggregate_diff_sq = aggregate_dot = 0.0
    aggregate_max = 0.0
    for index, (layer, name) in enumerate(zip(hparams.layers, weight_names)):
        base = resolve_parameter(runtime.model, name).detach().cpu()
        materialized = compare_tensors(
            native.updated_weights[name],
            bridge_result.updated_weights[name],
            base=base,
        )
        replay_metric = compare_tensors(
            native.updated_weights[name],
            replay.updated_weights[name],
            base=base,
        )
        native_factor = execution.proposal.factors[index]
        # Re-orient a fresh bridge factor from the raw pair; this is the local
        # adapter boundary under test.
        raw_delta = execution.deltas[index]
        bridge_factor = orient_easyedit_factor(
            raw_delta.adjusted_keys,
            raw_delta.residuals,
            weight_name=name,
            weight_shape=tuple(base.shape),
            expected_weight_sha256=base_hashes[name],
        )
        covariance = covariance_moments.get(int(layer))
        if covariance is None:
            raise MV0Error(f"verified covariance was not loaded for layer {layer}")
        c_metric = _c_metrics(native_factor, bridge_factor, covariance)
        layer_metrics.append(
            {
                "layer": int(layer),
                "weight_name": name,
                "materialized_delta": materialized,
                "factor": c_metric,
                "native_updated_hash": native.updated_hashes[name],
                "bridge_updated_hash": bridge_result.updated_hashes[name],
            }
        )
        replay_layers.append(
            {
                "layer": int(layer),
                "materialized_delta": replay_metric,
            }
        )
        ref = native.updated_weights[name].float() - base.float()
        cand = bridge_result.updated_weights[name].float() - base.float()
        diff = cand - ref
        aggregate_ref_sq += float(torch.sum(ref * ref))
        aggregate_cand_sq += float(torch.sum(cand * cand))
        aggregate_diff_sq += float(torch.sum(diff * diff))
        aggregate_dot += float(torch.sum(ref * cand))
        aggregate_max = max(aggregate_max, float(diff.abs().max()))

    ref_norm = math.sqrt(aggregate_ref_sq)
    cand_norm = math.sqrt(aggregate_cand_sq)
    final_delta = {
        "reference_norm": ref_norm,
        "candidate_norm": cand_norm,
        "relative_l2_error": _ratio(math.sqrt(aggregate_diff_sq), ref_norm),
        "relative_norm_error": _ratio(abs(cand_norm - ref_norm), ref_norm),
        "max_abs_error": aggregate_max,
        "cosine": _ratio(aggregate_dot, ref_norm * cand_norm),
        "all_layer_hashes_equal": native.updated_hashes == bridge_result.updated_hashes,
    }
    replay_teacher = compare_teacher(native.teacher, replay.teacher)
    bridge_teacher = compare_teacher(native.teacher, bridge_result.teacher)
    fp32_epsilon = torch.finfo(torch.float32).eps
    relative_floor = max(
        fp32_epsilon * 32,
        float(replay_teacher["relative_l2_error"] or 0.0),
        max(
            float(item["materialized_delta"]["relative_l2_error"] or 0.0)
            for item in replay_layers
        ),
    )
    absolute_floor = max(
        fp32_epsilon * 32,
        float(replay_teacher["max_abs_error"]),
        max(
            float(item["materialized_delta"]["max_abs_error"])
            for item in replay_layers
        ),
    )
    equivalence_pass = bool(
        trace["pass"]
        and final_delta["all_layer_hashes_equal"]
        and final_delta["relative_l2_error"] is not None
        and final_delta["relative_l2_error"] <= relative_floor
        and final_delta["max_abs_error"] <= absolute_floor
        and bridge_teacher["relative_l2_error"] is not None
        and bridge_teacher["relative_l2_error"] <= relative_floor
        and bridge_teacher["max_abs_error"] <= absolute_floor
        and bridge_teacher["nll_abs_error"] <= absolute_floor
        and all(
            item["materialized_delta"]["exact_bytes"]
            and item["factor"]["relative_c_norm_error"] <= relative_floor
            and (
                item["factor"]["c_cosine"] is None
                or abs(1.0 - item["factor"]["c_cosine"]) <= relative_floor
            )
            for item in layer_metrics
        )
    )
    return {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "pre_edit_state_hash": hashlib.sha256(
            json.dumps(base_hashes, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "event_seed": event_seed,
        "rng_state_hash": rng_state_hash(),
        "target_token_count": baseline.target_token_count,
        "context_id": contexts.manifest_id,
        "trace_no_write": trace,
        "native_execute": {
            "forward_calls": execution.forward_calls,
            "solve_count": execution.solve_count,
            "cache_template": None,
            "direct_z_mode": "frozen-repo-local-artifact",
            "direct_z_artifact_sha256": direct_z.artifact.sha256,
            "direct_z_artifact_size": direct_z.artifact.size,
            "application_mode": "easyedit-raw-factor-order",
        },
        "self_replay": {
            "layers": replay_layers,
            "teacher_forced": replay_teacher,
        },
        "bridge_comparison": {
            "layers": layer_metrics,
            "final_delta": final_delta,
            "teacher_forced": bridge_teacher,
            "rewrite_progress": {
                "native_nll_reduction": baseline.nll - native.teacher.nll,
                "bridge_nll_reduction": baseline.nll - bridge_result.teacher.nll,
                "absolute_difference": abs(
                    native.teacher.nll - bridge_result.teacher.nll
                ),
            },
            "application_mode": "TemporaryExactMemitApplication",
        },
        "calibrated_bound": {
            "dtype": "torch.float32",
            "relative": relative_floor,
            "absolute": absolute_floor,
        },
        "rollback_exact": True,
        "pass": equivalence_pass,
    }


def _local_run_directory(output_root: str | Path, run_id: str) -> Path:
    if _RUN_ID.fullmatch(run_id) is None or run_id in {".", ".."}:
        raise ContractError("run_id must be a safe ASCII identifier")
    local_root = (REPOSITORY_ROOT / "local").resolve(strict=True)
    root = Path(output_root).expanduser().resolve(strict=False)
    try:
        root.relative_to(local_root)
    except ValueError as exc:
        raise ContractError(
            f"MV-0 output must remain below the repository local/ tree: {local_root}"
        ) from exc
    destination = root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _relative_provenance(
    provenance: ProvenanceManifest,
    easyedit_root: Path,
) -> list[dict[str, Any]]:
    result = []
    for record in provenance.files:
        relative = str(Path(record.path).relative_to(easyedit_root))
        result.append(
            {"relative_path": relative, "sha256": record.sha256, "size": record.size}
        )
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_runtime_state() -> dict[str, Any]:
    """Bind a run to one clean tracked ODE-Edit commit."""

    if not GIT_BIN.is_file() or not os.access(GIT_BIN, os.X_OK):
        raise MV0Error("fixed Git runtime is unavailable")
    commit = subprocess.run(
        (str(GIT_BIN), "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise MV0Error("ODE-Edit HEAD is not a full Git commit")
    tracked_status = subprocess.run(
        (
            str(GIT_BIN),
            "-C",
            str(REPOSITORY_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked_status:
        raise MV0Error("ODE-Edit tracked worktree must be clean at execution")
    return {
        "commit": commit,
        "tracked_worktree_clean": True,
    }


def _slurm_runtime_state(model_alias: str, run_id: str) -> dict[str, Any]:
    """Bind a Slurm-launched artifact to its exact audited job envelope."""

    values = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME"),
    }
    if all(value is None for value in values.values()):
        return {"under_slurm": False}
    if any(value is None for value in values.values()):
        raise MV0Error("partial Slurm identity is forbidden")
    expected_jobs = {
        ("llama3-8b-inst", "mv0_llama_smoke_v1"): (
            "odeedit_mv0_llama_smoke",
            "devbox",
        ),
        ("qwen2.5-7b-inst", "mv0_qwen_smoke_v1"): (
            "odeedit_mv0_qwen_smoke",
            "devbox",
        ),
        ("llama3-8b-inst", "mv0_llama_c3_v3"): (
            "odeedit_mv0_pair_c3v3",
            "devbox",
        ),
        ("qwen2.5-7b-inst", "mv0_qwen_c3_v3"): (
            "odeedit_mv0_pair_c3v3",
            "devbox",
        ),
    }
    expected = expected_jobs.get((model_alias, run_id))
    if expected is None:
        raise MV0Error("Slurm run is outside the audited MV-0 smoke identities")
    expected_name, expected_node = expected
    if (
        re.fullmatch(r"[0-9]+", values["job_id"] or "") is None
        or values["job_name"] != expected_name
        or values["node"] != expected_node
    ):
        raise MV0Error("Slurm job identity does not match the audited envelope")
    return {
        "under_slurm": True,
        "job_id": values["job_id"],
        "job_name": values["job_name"],
        "node": values["node"],
    }


def _selection_cases(
    selection: CounterFactSelectionManifest,
    count: int,
) -> tuple[str, ...]:
    if isinstance(count, bool) or count < 1 or count > 3:
        raise ContractError("MV-0 case count must be between 1 and 3")
    return selection.calibration[:count]


def run_mv0(
    *,
    easyedit_root: str | Path,
    model_alias: str,
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    case_count: int = 3,
    seed: int = 17,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
) -> dict[str, Any]:
    """Execute one model's 1--3 singleton fidelity events."""

    started = time.perf_counter()
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    git_state = _git_runtime_state()
    slurm_state = _slurm_runtime_state(model_alias, run_id)
    # All byte checks and request sanitization happen before model allocation.
    provenance = preflight_fixed_artifacts(root, model_alias=model_alias)
    selection = generate_counterfact_selection(root, seed=selection_seed)
    case_ids = _selection_cases(selection, case_count)
    requests = load_counterfact_requests(root, case_ids)
    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    bridge_provenance = bridge.preflight()
    if not set(record.path for record in bridge_provenance.files).issubset(
        set(record.path for record in provenance.files)
    ):
        raise MV0Error("bridge provenance is outside the fixed full manifest")

    with offline_environment():
        # Validate and import the exact inert EasyEdit source closure before
        # allocating any model/GPU state.
        bindings = bridge.load()
        hparams = _load_hparams(root, spec, bindings)
        seed_runtime(seed)
        runtime = model_loader(model_alias)
        if runtime.spec != spec:
            raise MV0Error("model loader returned a different fixed model spec")
        contexts = _freeze_contexts(bridge, runtime, seed=seed)
        destination = _local_run_directory(output_root, run_id)
        manifest_path = destination / "manifest.json"
        events_path = destination / "events.jsonl"
        summary_path = destination / "summary.json"
        _write_json_exclusive(
            manifest_path,
            {
                "schema_version": "ode-edit-mv0-manifest/v1",
                "run_id": run_id,
                "ode_edit_git": git_state,
                "slurm": slurm_state,
                "model": runtime.metadata(),
                "hparams_relative_path": spec.hparams_path,
                "selection": selection.to_dict(),
                "selected_case_ids": list(case_ids),
                "selected_request_ids": [request.request_id for request in requests],
                "contexts": {
                    "manifest_id": contexts.manifest_id,
                    "source": contexts.source,
                    "group_sizes": [len(group) for group in contexts.templates],
                    "raw_templates_persisted": False,
                },
                "provenance_id": provenance.manifest_id,
                "fixed_files": _relative_provenance(provenance, root),
                "projector_policy": "sha256-and-size-verify-only; never-deserialized",
                "covariance_policy": "verified-read-only; recompute-and-download-blocked",
                "direct_z_policy": "computed-once-and-frozen-under-this-local-run",
                "artifact_firewall": (
                    "structured JSON contains case IDs/full request hashes/metrics "
                    "only; activation-derived direct_z remains local-only"
                ),
            },
        )
        torch.cuda.reset_peak_memory_stats(0)
        event_results: list[dict[str, Any]] = []
        covariance_specs = _covariance_specs(root, spec)
        covariance_moments, loaded_covariances = _load_verified_covariances(
            root=root,
            runtime=runtime,
            specs=covariance_specs,
        )
        direct_z_root = destination / "direct_z"
        abort_failure_type: str | None = None
        with SanitizedJsonlWriter(events_path, run_id) as writer:
            for request in requests:
                edit_weight_names = tuple(
                    f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                    for layer in hparams.layers
                )
                pre_event_hashes = _weight_hashes(
                    runtime.model, edit_weight_names
                )
                try:
                    result = _run_event(
                        runtime=runtime,
                        bridge=bridge,
                        bindings=bindings,
                        hparams=hparams,
                        contexts=contexts,
                        covariance_specs=covariance_specs,
                        covariance_moments=covariance_moments,
                        direct_z_root=direct_z_root,
                        request=request,
                        seed=seed,
                    )
                except _FATAL_EVENT_ERRORS as exc:
                    weights_intact = (
                        _weight_hashes(runtime.model, edit_weight_names)
                        == pre_event_hashes
                    )
                    rollback_exact = bool(
                        weights_intact and not isinstance(exc, RollbackError)
                    )
                    result = {
                        "case_id": request.case_id,
                        "request_id": request.request_id,
                        "failure_type": type(exc).__name__,
                        "failure_class": "fatal_contract_or_state",
                        "rollback_exact": rollback_exact,
                        "pass": False,
                    }
                    abort_failure_type = type(exc).__name__
                except Exception as exc:
                    # Keep technical failures in the intention-to-diagnose
                    # denominator without persisting exception text.
                    if (
                        _weight_hashes(runtime.model, edit_weight_names)
                        != pre_event_hashes
                    ):
                        result = {
                            "case_id": request.case_id,
                            "request_id": request.request_id,
                            "failure_type": "RollbackError",
                            "failure_class": "fatal_contract_or_state",
                            "rollback_exact": False,
                            "pass": False,
                        }
                        abort_failure_type = "RollbackError"
                    else:
                        torch.cuda.empty_cache()
                        result = {
                            "case_id": request.case_id,
                            "request_id": request.request_id,
                            "failure_type": type(exc).__name__,
                            "failure_class": "ordinary_technical",
                            "rollback_exact": True,
                            "pass": False,
                        }
                event_results.append(result)
                writer.write("mv0_case", result)
                if abort_failure_type is not None:
                    break

        elapsed = time.perf_counter() - started
        planned_case_count = len(requests)
        attempted_case_count = len(event_results)
        not_run_case_ids = [
            request.case_id for request in requests[attempted_case_count:]
        ]
        completed = [
            result
            for result in event_results
            if "bridge_comparison" in result
        ]
        max_weight_relative_error = max(
            (
                float(
                    result["bridge_comparison"]["final_delta"][
                        "relative_l2_error"
                    ]
                    or 0.0
                )
                for result in completed
            ),
            default=0.0,
        )
        max_logits_relative_error = max(
            (
                float(
                    result["bridge_comparison"]["teacher_forced"][
                        "relative_l2_error"
                    ]
                    or 0.0
                )
                for result in completed
            ),
            default=0.0,
        )
        max_nll_abs_error = max(
            (
                float(
                    result["bridge_comparison"]["teacher_forced"][
                        "nll_abs_error"
                    ]
                )
                for result in completed
            ),
            default=0.0,
        )
        summary = {
            "schema_version": "ode-edit-mv0-summary/v1",
            "run_id": run_id,
            "model_alias": model_alias,
            "slurm": slurm_state,
            "provenance_id": provenance.manifest_id,
            "selection_manifest_id": selection.manifest_id,
            "context_id": contexts.manifest_id,
            "run_status": (
                "aborted" if abort_failure_type is not None else "completed"
            ),
            "abort_failure_type": abort_failure_type,
            "case_count": planned_case_count,
            "planned_case_count": planned_case_count,
            "attempted_case_count": attempted_case_count,
            "not_run_due_to_abort_count": len(not_run_case_ids),
            "pass_count": sum(bool(result["pass"]) for result in event_results),
            "failure_count": (
                planned_case_count
                - sum(bool(result["pass"]) for result in event_results)
            ),
            "all_pass": bool(
                abort_failure_type is None
                and len(event_results) == planned_case_count
                and all(bool(result["pass"]) for result in event_results)
            ),
            "case_results": [
                {
                    "case_id": result["case_id"],
                    "status": "attempted",
                    "pass": bool(result["pass"]),
                }
                for result in event_results
            ]
            + [
                {
                    "case_id": case_id,
                    "status": "not_run_due_to_abort",
                    "pass": False,
                }
                for case_id in not_run_case_ids
            ],
            "compact_metrics": {
                "max_final_weight_relative_l2_error": max_weight_relative_error,
                "max_teacher_logits_relative_l2_error": max_logits_relative_error,
                "max_teacher_nll_abs_error": max_nll_abs_error,
                "all_rollbacks_exact": all(
                    bool(result["rollback_exact"]) for result in event_results
                ),
            },
            "covariance_files_loaded": list(loaded_covariances),
            "direct_z_artifact_count": (
                len(tuple(direct_z_root.glob("*.pt")))
                if direct_z_root.is_dir()
                else 0
            ),
            "projector_files_loaded": [],
            "resource": {
                "wall_seconds": elapsed,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "host_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "visible_gpu_count": 1,
            },
            "artifacts": {
                "manifest_sha256": _file_sha256(manifest_path),
                "events_sha256": _file_sha256(events_path),
            },
            "git_output_written": False,
        }
        _write_json_exclusive(summary_path, summary)
        return {**summary, "output_directory": str(destination)}


def _cmd_preflight(args: argparse.Namespace) -> int:
    manifest = preflight_fixed_artifacts(
        args.easyedit_root,
        model_alias=args.model,
    )
    selection = generate_counterfact_selection(
        args.easyedit_root,
        seed=args.selection_seed,
    )
    print(
        json.dumps(
            {
                "status": "verified; no model loaded",
                "model": args.model,
                "provenance_id": manifest.manifest_id,
                "selection_manifest_id": selection.manifest_id,
                "selected_rows": len(selection.ordered_case_ids),
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-mv0",
        description="Fail-closed native MEMIT fidelity and trace-neutrality gate",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--easyedit-root", required=True)
    preflight.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    preflight.add_argument("--selection-seed", default=DEFAULT_SELECTION_SEED)
    run = commands.add_parser("run")
    run.add_argument("--easyedit-root", required=True)
    run.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    run.add_argument("--run-id", required=True)
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    run.add_argument("--cases", type=int, choices=(1, 2, 3), default=3)
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--selection-seed", default=DEFAULT_SELECTION_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            return _cmd_preflight(args)
        summary = run_mv0(
            easyedit_root=args.easyedit_root,
            model_alias=args.model,
            run_id=args.run_id,
            output_root=args.output_root,
            case_count=args.cases,
            seed=args.seed,
            selection_seed=args.selection_seed,
        )
        print(
            json.dumps(
                {
                    "run_id": summary["run_id"],
                    "model_alias": summary["model_alias"],
                    "all_pass": summary["all_pass"],
                    "output_directory": summary["output_directory"],
                },
                sort_keys=True,
            )
        )
        return 0 if summary["all_pass"] else 1
    except (ContractError, GpuRuntimeError, MV0Error, OSError) as exc:
        # Never print upstream exception text: it may contain a raw request.
        print(
            json.dumps(
                {
                    "status": "aborted",
                    "error_type": type(exc).__name__,
                    "raw_exception_persisted": False,
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
