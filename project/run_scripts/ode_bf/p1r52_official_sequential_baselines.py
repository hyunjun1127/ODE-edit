"""Official sequential baseline adapters for the P1R52 comparison.

The adapters keep the common frozen stream/evaluator owned by
``p1r52_sequential_runtime`` and bind only method-specific official EasyEdit
entrypoints and cache semantics.  EasyEdit source is never modified here.
"""

from __future__ import annotations

import contextlib
import copy
import io
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .scalable_batched_runtime import scalable_ordered_request_digest


EASYEDIT_ROOT = Path("/mnt/raid5/janghj/EasyEdit")
MEMIT_HPARAMS_PATH = EASYEDIT_ROOT / "hparams/MEMIT/llama3-8b.yaml"
MEMIT_STATS_ROOT = EASYEDIT_ROOT / "examples/data/stats"


def load_official_memit_hparams(*, easyedit_root: Path | None = None) -> Any:
    """Load the frozen Llama MEMIT configuration used by the artifact lock."""

    from easyeditor.models.memit.memit_hparams import MEMITHyperParams

    root = EASYEDIT_ROOT if easyedit_root is None else Path(easyedit_root)
    hparams_path = root / "hparams/MEMIT/llama3-8b.yaml"
    stats_root = root / "examples/data/stats"
    hparams = MEMITHyperParams.from_hparams(str(hparams_path))
    hparams.device = 0
    hparams.stats_dir = str(stats_root)
    exact = {
        "alg_name": "MEMIT",
        "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "layers": [4, 5, 6, 7, 8],
        "fact_token": "subject_last",
        "mom2_dataset": "wikipedia",
        "mom2_n_samples": 100000,
        "mom2_dtype": "float32",
        "mom2_update_weight": 15000,
    }
    if any(getattr(hparams, key) != value for key, value in exact.items()):
        raise ODEBFContractError("Official MEMIT frozen hparams differ")
    return hparams


def _memit_covariance_cache_snapshot(memit_module: Any) -> dict[str, Any]:
    cache = getattr(memit_module, "COV_CACHE", None)
    if not isinstance(cache, dict):
        raise ODEBFStateError("Official MEMIT covariance cache differs")
    rows: list[dict[str, Any]] = []
    for key, value in sorted(cache.items(), key=lambda item: repr(item[0])):
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ODEBFStateError("Official MEMIT covariance cache tensor differs")
        rows.append(
            {
                "key": [str(part) for part in key],
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": bool(torch.isfinite(value).all().item()),
            }
        )
    payload = {
        "cache_kind": "STATIC_MEMIT_COVARIANCE_COMPUTATION_CACHE",
        "entries": rows,
        "entry_count": len(rows),
        "historical_decision_state": False,
        "request_history_width": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _run_official_memit_apply_impl(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    *,
    touched: Mapping[str, torch.nn.Parameter],
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    """Apply one current-B10 Official MEMIT edit to the persistent physical W."""

    from easyeditor.models.memit import memit_main

    entry_sha = {name: tensor_sha256(value) for name, value in touched.items()}
    pointers = {name: int(value.data_ptr()) for name, value in touched.items()}
    request_copy = [copy.deepcopy(dict(item)) for item in requests]
    cache_entry = _memit_covariance_cache_snapshot(memit_main)
    original_backward = torch.autograd.backward
    target_backward_count = 0

    def counted_backward(*args: Any, **kwargs: Any) -> Any:
        nonlocal target_backward_count
        target_backward_count += 1
        return original_backward(*args, **kwargs)

    started = time.perf_counter()
    torch.autograd.backward = counted_backward
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            returned_model, originals = memit_main.apply_memit_to_model(
                model,
                tokenizer,
                request_copy,
                hparams,
                copy=False,
                return_orig_weights=True,
                cache_template=None,
                keep_original_weight=False,
            )
    finally:
        torch.autograd.backward = original_backward
    wall = time.perf_counter() - started
    cache_exit = _memit_covariance_cache_snapshot(memit_main)
    if returned_model is not model or set(originals) != set(touched):
        raise ODEBFContractError("Official MEMIT return contract differs")
    if any(int(touched[name].data_ptr()) != pointers[name] for name in touched):
        raise ODEBFStateError("Official MEMIT pointer differs")
    original_sha = {name: tensor_sha256(value) for name, value in originals.items()}
    if any(original_sha[name] != entry_sha[name] for name in touched):
        raise ODEBFStateError("Official MEMIT original copy differs")
    payload = {
        "schema": "ode-edit-s05-p1r52-official-memit-sequential-apply/v1",
        "entry_sha256": entry_sha,
        "edited_sha256": {name: tensor_sha256(value) for name, value in touched.items()},
        "original_copy_sha256": original_sha,
        "request_order_sha256": scalable_ordered_request_digest(
            [str(item["request_sha256"]) for item in requests]
        ),
        "official_entrypoint": "easyeditor.models.memit.memit_main.apply_memit_to_model",
        "copy": False,
        "return_orig_weights": True,
        "cache_template": None,
        "keep_original_weight": False,
        "physical_weight_persistence": True,
        "current_batch_request_count": len(requests),
        "hidden_controller_state_count": 0,
        "target_backward_count": target_backward_count,
        "target_z_cache": {
            "enabled": False,
            "kind": "OPTIONAL_COMPUTATION_CACHE_ONLY",
            "historical_decision_state": False,
        },
        "covariance_cache": {
            "entry": cache_entry,
            "exit": cache_exit,
            "entry_reused_count": int(cache_entry["entry_count"]),
            "persistent_default_semantics": True,
            "silent_reset_count": 0,
            "historical_decision_influence_count": 0,
        },
        "edit_core_wall_seconds": wall,
        "raw_stdout_stderr_serialized_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload, originals


def run_official_memit_apply(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    *,
    touched: Mapping[str, torch.nn.Parameter],
    observer: Any | None = None,
    call_audit: Any | None = None,
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    """Apply stock MEMIT with an optional call-scoped read-only observer.

    The default ``None`` path is the pre-existing adapter.  Experiment-owned
    observers are entered before the stock entrypoint and are required to
    restore the exact module-global function objects in ``finally``.
    """

    from easyeditor.models.memit import memit_main

    with contextlib.ExitStack() as stack:
        if observer is not None:
            stack.enter_context(observer.observe(memit_main))
        if call_audit is not None:
            stack.enter_context(call_audit.observe(memit_main))
        payload, originals = _run_official_memit_apply_impl(
            model,
            tokenizer,
            requests,
            hparams,
            touched=touched,
        )
        if observer is not None:
            observer.capture_terminal(model, tokenizer)
    if observer is not None:
        observer.capture_weight_action(touched, originals)
        payload["layer_realization_observer"] = observer.payload(memit_main)
    if call_audit is not None:
        payload["official_call_audit"] = call_audit.payload(memit_main)
    if observer is not None or call_audit is not None:
        payload["identity_sha256"] = canonical_hash(
            {key: value for key, value in payload.items() if key != "identity_sha256"}
        )
    return payload, originals


__all__ = [
    "EASYEDIT_ROOT",
    "MEMIT_HPARAMS_PATH",
    "MEMIT_STATS_ROOT",
    "load_official_memit_hparams",
    "run_official_memit_apply",
]
