"""Typed locks for evaluation-only lifelong final-weight backfill.

The stored checkpoint schedule is an outcome-blind amendment: unavailable
states are never reconstructed, interpolated, or replayed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


INSTRUCTION_ID = (
    "ODEEDIT-S06-OFFICIAL-LAYER-DEBT-LIFELONG-FINALW-FULL10K-EVAL-V1"
)
SCHEMA = "odeedit.s06.layer-debt.lifelong-finalw-full10k.v1"
ORIGINAL_REQUESTED_CHECKPOINTS = (100, 500, 1000, 2000, 4000, 6000, 8000, 10000)
AMENDED_CHECKPOINTS = (1000, 1500, 2000, 3000, 5000, 7500, 10000)
ABSENT_EXACT_CHECKPOINTS = (100, 500, 4000, 6000, 8000)
LAYERS = (4, 5, 6, 7, 8)
DATASET = Path("/data/janghj/EasyEdit/data/counterfact/counterfact.json")
DATASET_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
STREAM_SEAL = Path(
    "/data/janghj/ODE-edit/local/state/official-layer-realization-debt-"
    "lifelong-b100-v1/stream-seal-v1.json"
)
STREAM_SEAL_SHA256 = "5848dca7323335483039ec125b644dadfb87b0996133f9825f951088cf3a7861"
STREAM_ROOT = "2951b86dfa829ef38b2fb03551a81dc9ce4f11c779af3f0f0c027224ed65cf7d"
ORDER_ROOT = "f86dfc97f50614d8fe3d7ed929326485ea0b1237151f93d7ed7339bae8979c43"
EVALUATOR_IDENTITY = "72b8ecb737157a42d6a055ffd339dc3f907876a0a98165a00cca49ca9bbed07d"
EASYEDIT_ROOT = Path("/data/janghj/EasyEdit-stock-14cea824")
EASYEDIT_HEAD = "14cea8245f06715684592ab55184939b99d70784"
EASYEDIT_TREE = "9c52aadbc0883da422badf0a730fff21aaa3a8a7"
V3_REPORT_ROOT = Path(
    "/data/janghj/ODE-edit/experiment-reports/servers/server4/"
    "official-layer-realization-debt-lifelong-b100x100-2026-09-03-v3"
)
V3_REPORT_SHA256 = "c9daf397b8c9fa679fa89e4f6407d7e03761a11f9ead177e998a97e726e24a8f"
V3_MANIFEST_SHA256 = "47398d3e497edbbf8f2ace5be4f315134653f16ee2698d47945d4769fe0deb15"
V3_RECEIPT_SHA256 = "9882b8bca524f2a34d2a3c1c4fea803aea834552c675a19be2b1e69ab7224af9"


ARM_ROOTS: tuple[tuple[str, str, str], ...] = (
    (
        "llama3-8b-inst",
        "memit",
        "/data/janghj/ODE-edit/local/results/official-layer-realization-debt-"
        "lifelong-b100-v1/campaign-20260902-four-arm-completion-tech-r1/"
        "llama3-8b-inst-memit-lifelong-b100x100",
    ),
    (
        "llama3-8b-inst",
        "alphaedit",
        "/data/janghj/ODE-edit/local/results/official-layer-realization-debt-"
        "lifelong-b100-v1/campaign-20260902-four-arm-completion-tech-r1/"
        "llama3-8b-inst-alphaedit-lifelong-b100x100",
    ),
    (
        "qwen2.5-7b-inst",
        "memit",
        "/data/janghj/ODE-edit/local/results/official-layer-realization-debt-"
        "lifelong-b100-v1/campaign-20260901-tech-r1/"
        "qwen2.5-7b-inst-memit-lifelong-b100x100",
    ),
    (
        "qwen2.5-7b-inst",
        "alphaedit",
        "/data/janghj/ODE-edit/local/results/official-layer-realization-debt-"
        "lifelong-b100-v1/campaign-20260901-v1/"
        "qwen2.5-7b-inst-alphaedit-lifelong-b100x100",
    ),
)


class FinalWeightBoundary(RuntimeError):
    """Fail-close evaluation or artifact-integrity boundary."""


@dataclass(frozen=True, slots=True)
class EvaluationLock:
    checkpoints: tuple[int, ...] = AMENDED_CHECKPOINTS
    original_requested_checkpoints: tuple[int, ...] = ORIGINAL_REQUESTED_CHECKPOINTS
    absent_exact_checkpoints: tuple[int, ...] = ABSENT_EXACT_CHECKPOINTS
    evaluator_batch_size: int = 32
    request_chunk_size: int = 25
    age_early_fraction: float = 0.2
    age_recent_fraction: float = 0.2
    full_fp32: bool = True
    edit_replay_count: int = 0
    checkpoint_interpolation_count: int = 0
    nearest_substitution_count: int = 0
    imputation_count: int = 0
    writer_count: int = 0
    compute_z_count: int = 0
    backward_count: int = 0
    parameter_gradient_count: int = 0
    cache_history_mutation_count: int = 0
    scientific_promotion: bool = False

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def cell_mapping(index: int) -> tuple[str, str, Path]:
    if index < 0 or index >= len(ARM_ROOTS):
        raise FinalWeightBoundary(f"cell index outside 0..3: {index}")
    model, method, root = ARM_ROOTS[index]
    return model, method, Path(root)


__all__ = [
    "ABSENT_EXACT_CHECKPOINTS",
    "AMENDED_CHECKPOINTS",
    "ARM_ROOTS",
    "DATASET",
    "DATASET_SHA256",
    "EASYEDIT_HEAD",
    "EASYEDIT_ROOT",
    "EASYEDIT_TREE",
    "EVALUATOR_IDENTITY",
    "EvaluationLock",
    "FinalWeightBoundary",
    "INSTRUCTION_ID",
    "LAYERS",
    "ORDER_ROOT",
    "ORIGINAL_REQUESTED_CHECKPOINTS",
    "SCHEMA",
    "STREAM_ROOT",
    "STREAM_SEAL",
    "STREAM_SEAL_SHA256",
    "V3_MANIFEST_SHA256",
    "V3_RECEIPT_SHA256",
    "V3_REPORT_ROOT",
    "V3_REPORT_SHA256",
    "cell_mapping",
]
