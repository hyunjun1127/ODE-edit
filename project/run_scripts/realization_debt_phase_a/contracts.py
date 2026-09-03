"""Locked definitions and filesystem helpers for Realization Debt Phase A.

This module is deliberately independent of torch and every experiment runtime.
It is safe to import in analysis-only environments.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Iterable


INSTRUCTION_ID = "ODEEDIT-S06-REALIZATION-DEBT-PHASE-A-LIFELONG-V6-V1"
CONTRACT_SHA256 = "b46f8620dda25f44836dab7f8facbb1c8212270ee83744fcc310938045f191a2"
SOURCE_BASE_HEAD = "21da9f0512cde81fa09ccd37b8306051872e3a89"
SOURCE_BASE_TREE = "54cebf2916eb0aa4b59850df566fff2f4c7cb8ec"

ARM_KEYS = ("model", "method")
LAYER_KEYS = ("model", "method", "batch_index", "request_sha256", "layer")
REQUEST_KEYS = ("model", "method", "batch_index", "request_sha256")
ACTION_KEYS = ("model", "method", "batch_index", "layer")

EXPECTED_ARMS = 4
EXPECTED_BATCHES_PER_ARM = 100
EXPECTED_REQUESTS_PER_BATCH = 100
EXPECTED_LAYERS = (4, 5, 6, 7, 8)
EXPECTED_LAYER_ROWS = 200_000
EXPECTED_REQUEST_ROWS = 40_000
EXPECTED_ACTION_ROWS = 2_000

PRIMARY_INPUTS = {
    "production-layer-complete.csv.gz": "2fa6450eafa2cc9a095b25f1f8009bb2da16084a5eeefdd0287a7b568814359e",
    "production-weight-batch-unit-complete.csv.gz": "d3333403c9fd5249d2163e51d3221e356c33e5c3617c1150de729e7294d967a8",
    "production-request-complete.csv.gz": "4e0d6020e97f8c45f6ad246224a2f581097991fc6ca8d08c621c460e1492dff5",
}

CONTEXT_MEMBERS = (
    ("v3", "analysis-manifest.json"),
    ("v3", "rooted-analysis-receipt.json"),
    ("v3", "official-layer-realization-debt-lifelong-fourarm-exhaustive-factual-ko.md"),
    ("v4-r2", "analysis-manifest.json"),
    ("v4-r2", "rooted-analysis-receipt.json"),
    ("v4-r2", "official-layer-debt-lifelong-finalw-full10k-factual-ko.md"),
    ("v5", "analysis-manifest.json"),
    ("v5", "rooted-analysis-receipt.json"),
    ("v5", "official-layer-realization-debt-lifelong-v5-exhaustive-cumulative-factual-ko.md"),
    ("v6", "analysis-manifest.json"),
    ("v6", "rooted-analysis-receipt.json"),
    ("v6", "official-layer-debt-lifelong-counterfact-metrics-v6-factual-ko.md"),
)


class AnalysisBoundary(RuntimeError):
    """Raised when the sealed schema cannot support the locked analysis."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise AnalysisBoundary(f"missing regular file: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise AnalysisBoundary(f"not a regular non-symlink file: {path}")
    return info


def member(path: Path, *, relative_to: Path | None = None, kind: str = "member") -> dict[str, Any]:
    info = regular_file(path)
    shown = path.relative_to(relative_to).as_posix() if relative_to is not None else str(path)
    return {
        "path": shown,
        "kind": kind,
        "bytes": int(info.st_size),
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "sha256": sha256_file(path),
    }


def debt_components(rho: float, tau: float) -> dict[str, float]:
    if not math.isfinite(rho) or not math.isfinite(tau):
        raise AnalysisBoundary("nonfinite rho/tau")
    parallel = (1.0 - rho) ** 2
    under = parallel if 0.0 <= rho < 1.0 else 0.0
    over = (rho - 1.0) ** 2 if rho > 1.0 else 0.0
    opposite = parallel if rho < 0.0 else 0.0
    orthogonal = tau**2
    native = parallel + orthogonal
    return {
        "debt_native": native,
        "debt_under": under,
        "debt_over": over,
        "debt_opposite": opposite,
        "debt_orthogonal": orthogonal,
    }


def normalized_potential_reduction(q_pre: float, q_post: float) -> float:
    if not math.isfinite(q_pre) or not math.isfinite(q_post):
        raise AnalysisBoundary("nonfinite q_pre/q_post")
    return 0.5 * (q_pre**2 - q_post**2)


def write_json_once(path: Path, value: Any, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
        handle.write("\n")


def assert_unique(values: Iterable[Any], expected: int, label: str) -> None:
    observed = len(set(values))
    if observed != expected:
        raise AnalysisBoundary(f"{label} unique count differs: {observed} != {expected}")
