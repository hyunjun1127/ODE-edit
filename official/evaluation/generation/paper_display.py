"""Presentation only: never mutate raw generation metrics or W&B history."""
from decimal import Decimal, ROUND_HALF_UP
import math


def paper_cell(raw_value, *, metric, raw_unit):
    """Accept unrounded raw numbers and return a final x100 table string.

    Explicit units and nonnumeric output prevent accidental display-to-display
    conversion. DEFERRED/missing statuses remain nonnumeric.
    """
    units = {"Flu": "bits", "Con": "cosine_0_to_1"}
    if metric not in units or raw_unit != units[metric]:
        raise ValueError("RAW_GENERATION_UNIT_REQUIRED")
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        if raw_value in ("DEFERRED", "MISSING", "NOT_MEASURED", "UNVERIFIED"):
            return raw_value
        raise ValueError("UNROUNDED_RAW_NUMBER_REQUIRED")
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ValueError("UNROUNDED_RAW_NUMBER_REQUIRED")
    if not math.isfinite(raw_value) or raw_value < 0:
        raise ValueError("INVALID_RAW_GENERATION_VALUE")
    if metric == "Con" and raw_value > 1:
        raise ValueError("RAW_COSINE_OUT_OF_RANGE")
    return str((Decimal(str(raw_value)) * 100).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP))


def paper_generation(summary):
    """Map raw summary means into explicitly named display-only fields."""
    result = {}
    for metric, key, unit_key, count_key in (
        ("Flu", "ngram_entropy", "fluency_unit", "fluency_count"),
        ("Con", "reference_score", "consistency_unit", "consistency_count"),
    ):
        value = summary.get(key)
        if value is None:
            continue
        count = summary.get(count_key)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("MEASURED_GENERATION_COUNT_REQUIRED")
        result[metric + "_paper_x100"] = paper_cell(
            value, metric=metric, raw_unit=summary.get(unit_key))
    return result
