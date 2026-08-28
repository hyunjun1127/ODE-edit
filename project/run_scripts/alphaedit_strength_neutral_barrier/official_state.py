"""ODE-side binding for stock EasyEdit AlphaEdit global state.

This module mirrors only the state-initialization prelude of the upstream
``apply_AlphaEdit_to_model`` entry point.  It never patches or writes EasyEdit
source.  The scientific writer continues to call upstream ``compute_z``,
``compute_ks``, activation capture, shape matching, projector and cache
objects directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from easyeditor.models.alphaedit import AlphaEdit_main as official
from easyeditor.util import nethook


class OfficialStateBoundary(RuntimeError):
    """The stock AlphaEdit projector/cache state cannot be bound safely."""


def _cache_shape(model: Any, hparams: Any) -> tuple[int, int, int]:
    weight = nethook.get_parameter(
        model,
        f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight",
    )
    name = str(hparams.model_name).lower()
    if "llama" in name or "qwen" in name:
        width = int(weight.shape[1])
    elif "gpt2-xl" in name:
        width = int(weight.shape[0])
    else:
        raise OfficialStateBoundary(
            f"stock AlphaEdit cache shape is undefined for model={hparams.model_name!r}"
        )
    return len(hparams.layers), width, width


def prepare_official_state(
    model: Any,
    tok: Any,
    hparams: Any,
    *,
    reset_cache: bool,
) -> None:
    """Bind stock ``P`` and ``cache_c`` without changing upstream code.

    Projector generation is deliberately not reproduced here: the experiment
    contract seals an existing Official AlphaEdit projector.  Missing
    projector bytes therefore fail closed rather than creating a new artifact.
    Runtime globals are the same objects consumed by upstream AlphaEdit.
    """

    del tok  # Projector computation is intentionally outside this hook.
    projector_path = Path(hparams.P_loc)
    if not projector_path.is_file() or projector_path.is_symlink():
        raise OfficialStateBoundary(
            f"sealed Official AlphaEdit projector is unavailable: {projector_path}"
        )

    if not bool(getattr(official, "P_loaded", False)):
        projector = torch.load(projector_path, map_location="cpu")
        if projector.dtype != torch.float32 or not bool(torch.isfinite(projector).all()):
            raise OfficialStateBoundary("Official AlphaEdit projector must be finite FP32")
        official.P = projector
        official.P_loaded = True

    expected = _cache_shape(model, hparams)
    projector = getattr(official, "P", None)
    if not isinstance(projector, torch.Tensor) or tuple(projector.shape) != expected:
        raise OfficialStateBoundary(
            f"Official projector shape mismatch: expected={expected}, "
            f"observed={getattr(projector, 'shape', None)}"
        )

    if reset_cache:
        official.cache_c_new = False
    cache = getattr(official, "cache_c", None)
    if not bool(getattr(official, "cache_c_new", False)):
        official.cache_c = torch.zeros(expected, dtype=torch.float32, device="cpu")
        official.cache_c_new = True
    elif not isinstance(cache, torch.Tensor) or tuple(cache.shape) != expected:
        raise OfficialStateBoundary(
            f"Official AlphaEdit cache shape mismatch: expected={expected}, "
            f"observed={getattr(cache, 'shape', None)}"
        )
    elif cache.dtype != torch.float32 or not bool(torch.isfinite(cache).all()):
        raise OfficialStateBoundary("Official AlphaEdit cache must be finite FP32")


__all__ = ["OfficialStateBoundary", "prepare_official_state"]
