"""Typed, outcome-independent contracts for the fixed-z screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ScreenError(RuntimeError):
    """Base fail-close error."""


class TechnicalBoundary(ScreenError):
    """A deployment or instrumentation invariant failed."""


class ScientificBoundary(ScreenError):
    """An equality, rank, action, or fixed-z invariant failed."""


class Method(str, Enum):
    ALPHAEDIT = "alphaedit"
    MEMIT = "memit"


@dataclass(frozen=True)
class NumericalLock:
    """Machine-derived FULL_FP32 lock, frozen before candidate outcomes."""

    dtype: str = "float32"
    rho_tangent: float = 0.05
    random_axis_count: int = 4
    duplicate_reference_evaluations: int = 2
    gate_prompt_count: int = 32
    cvar_alpha: float = 0.875
    # 256 * IEEE float32 epsilon.  This is not outcome-tuned.
    fp32_relative_tolerance: float = 3.0517578125e-5
    fp32_absolute_tolerance: float = 3.0517578125e-5
    # Small SVD rank rule is max(m,n)*eps, computed from actual shape.
    rank_rule: str = "rtol=max(m,n)*torch.finfo(float32).eps;atol=0"
    solver: str = "thin_svd_and_small_spd_solve_only"
    dense_inverse_count: int = 0
    seed_namespace: str = "odeedit-s06-fixed-z-nonuniqueness-v1"

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_path: Path
    model_revision: str
    config_sha256: str
    tokenizer_sha256: str
    alpha_hparams: Path
    memit_hparams: Path
    projector: Path
    statistics_root: Path
    statistics_model_dir: str
    pad_policy: str = "pad_equals_eos_explicit"
    official_padding_side: str = "right"
    hook_padding_side: str = "left"

    def payload(self) -> dict[str, Any]:
        row = asdict(self)
        for key, value in tuple(row.items()):
            if isinstance(value, Path):
                row[key] = str(value)
        return row


MODEL_SPECS: dict[str, ModelSpec] = {
    "llama3-8b-inst": ModelSpec(
        alias="llama3-8b-inst",
        model_path=Path("/data/janghj/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"),
        model_revision="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        config_sha256="61f3de03a16ca8046b05dc777bce72717022bc8522152eee61a69072272ef54b",
        tokenizer_sha256="e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7",
        alpha_hparams=Path("project/run_scripts/fixed_z_nonuniqueness/config/alphaedit-llama3-8b.yaml"),
        memit_hparams=Path("project/run_scripts/fixed_z_nonuniqueness/config/memit-llama3-8b.yaml"),
        projector=Path("/data/janghj/EasyEdit/examples/null_space_project_Meta-Llama-3-8B-Instruct.pt"),
        statistics_root=Path("/data/janghj/EasyEdit/examples/data/stats"),
        statistics_model_dir="Meta-Llama-3-8B-Instruct",
    ),
    "qwen2.5-7b-inst": ModelSpec(
        alias="qwen2.5-7b-inst",
        model_path=Path("/data/janghj/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"),
        model_revision="a09a35458c702b33eeacc393d103063234e8bc28",
        config_sha256="7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
        tokenizer_sha256="c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        alpha_hparams=Path("project/run_scripts/fixed_z_nonuniqueness/config/alphaedit-qwen2.5-7b.yaml"),
        memit_hparams=Path("project/run_scripts/fixed_z_nonuniqueness/config/memit-qwen2.5-7b.yaml"),
        projector=Path("/data/janghj/EasyEdit/examples/null_space_project_Qwen2.5-7B-Instruct.pt"),
        statistics_root=Path("/data/janghj/EasyEdit/examples/data/stats"),
        statistics_model_dir="Qwen2.5-7B-Instruct",
    ),
}
