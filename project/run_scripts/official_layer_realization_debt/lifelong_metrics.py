"""Pure FP64 action--realization and failure-time reductions."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import torch

from project.run_scripts.ode_bf.contracts import canonical_hash

from .contracts import LAYERS, ObservationBoundary


def action_realization_profiles(
    residual_debt: Mapping[str, Any],
    weight_action: Mapping[str, Any],
) -> dict[str, Any]:
    """Pair one batch-level update profile with request-level progress profiles.

    The update profile is deliberately *not* replicated into independent
    request observations.  Request rows retain the common batch profile only
    as a conditioning value.
    """

    weight_rows = list(weight_action["layers"])
    if [int(row["layer"]) for row in weight_rows] != list(LAYERS):
        raise ObservationBoundary("lifelong weight layer inventory differs")
    update = torch.tensor(
        [float(row["frobenius_magnitude"]) for row in weight_rows],
        dtype=torch.float64,
    )
    if not torch.isfinite(update).all() or bool((update < 0).any()):
        raise ObservationBoundary("lifelong update magnitude is invalid")
    update_total = float(update.sum().item())
    if update_total <= 0.0:
        raise ObservationBoundary("lifelong update magnitude total is nonpositive")
    p_update = update / update.sum()

    request_rows: list[dict[str, Any]] = []
    negative_count = 0
    negative_mass = 0.0
    for record in residual_debt["records"]:
        layers = list(record["layers"])
        if [int(row["layer"]) for row in layers] != list(LAYERS):
            raise ObservationBoundary("lifelong progress layer inventory differs")
        progress = torch.tensor(
            [
                float(row["rho"]) * float(row["allocation_norm_over_R1"])
                for row in layers
            ],
            dtype=torch.float64,
        )
        if not torch.isfinite(progress).all():
            raise ObservationBoundary("lifelong target progress is nonfinite")
        positive = torch.clamp(progress, min=0.0)
        positive_total = float(positive.sum().item())
        p_realization = (
            positive / positive.sum()
            if positive_total > 0.0
            else torch.zeros_like(positive)
        )
        negative = progress[progress < 0]
        negative_count += int(negative.numel())
        negative_mass += float((-negative).sum().item())
        tv = 0.5 * torch.abs(p_update - p_realization).sum()
        layer_index = torch.tensor(LAYERS, dtype=torch.float64)
        center = torch.dot(layer_index, p_realization) - torch.dot(
            layer_index, p_update
        )
        row = {
            "request_sha256": str(record["request_sha256"]),
            "update_profile": [float(value) for value in p_update.tolist()],
            "target_progress": [float(value) for value in progress.tolist()],
            "realization_profile": [
                float(value) for value in p_realization.tolist()
            ],
            "positive_progress_total": positive_total,
            "negative_progress_count": int((progress < 0).sum().item()),
            "negative_progress_total": float(
                (-progress[progress < 0]).sum().item()
            ),
            "D_TV": float(tv.item()),
            "delta_center": float(center.item()),
        }
        row["identity_sha256"] = canonical_hash(row)
        request_rows.append(row)
    return {
        "schema": "odeedit.s06.layer-realization-debt.lifelong-profile.v1",
        "weight_observation_unit": "ONE_B100_BATCH_LAYER_UPDATE",
        "activation_observation_unit": "REQUEST_WITHIN_B100",
        "request_as_independent_weight_sample_count": 0,
        "primary_update_quantity": "frobenius_magnitude",
        "squared_update_quantity_role": "TELEMETRY_ONLY",
        "update_profile": [float(value) for value in p_update.tolist()],
        "request_count": len(request_rows),
        "negative_progress_count": negative_count,
        "negative_progress_total": negative_mass,
        "requests": request_rows,
    }


def sustained_onset(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    threshold: float,
    consecutive: int = 3,
    greater_is_violation: bool = True,
) -> int | None:
    """Return the first batch in a preregistered sustained run."""

    if consecutive <= 0 or not math.isfinite(threshold):
        raise ValueError("invalid sustained-onset contract")
    run: list[int] = []
    for row in rows:
        value = float(row[field])
        if not math.isfinite(value):
            raise ObservationBoundary(f"nonfinite failure-time field: {field}")
        violation = value > threshold if greater_is_violation else value < threshold
        if violation:
            run.append(int(row["batch_index"]))
            if len(run) == consecutive:
                return run[0]
        else:
            run.clear()
    return None


def rank_concordance(
    predicted: Iterable[float], actual: Iterable[float]
) -> dict[str, Any]:
    """Dependency-free Spearman/Kendall descriptive concordance."""

    x = torch.tensor(list(predicted), dtype=torch.float64)
    y = torch.tensor(list(actual), dtype=torch.float64)
    if x.numel() != y.numel() or x.numel() < 2 or not all(
        torch.isfinite(value).all() for value in (x, y)
    ):
        return {
            "status": "NOT_IDENTIFIABLE",
            "count": int(min(x.numel(), y.numel())),
        }

    def _ranks(value: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(value, stable=True)
        ranks = torch.empty_like(value)
        ranks[order] = torch.arange(value.numel(), dtype=torch.float64)
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    centered_x, centered_y = rx - rx.mean(), ry - ry.mean()
    denom = torch.linalg.vector_norm(centered_x) * torch.linalg.vector_norm(
        centered_y
    )
    spearman = float(torch.dot(centered_x, centered_y).item() / denom.item())
    concordant = 0
    discordant = 0
    for left in range(x.numel()):
        for right in range(left + 1, x.numel()):
            product = float((x[left] - x[right]) * (y[left] - y[right]))
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    pairs = concordant + discordant
    kendall = (concordant - discordant) / pairs if pairs else float("nan")
    return {
        "status": "DESCRIPTIVE",
        "count": int(x.numel()),
        "spearman": spearman,
        "kendall": kendall,
        "tie_aware": False,
    }


__all__ = [
    "action_realization_profiles",
    "rank_concordance",
    "sustained_onset",
]
