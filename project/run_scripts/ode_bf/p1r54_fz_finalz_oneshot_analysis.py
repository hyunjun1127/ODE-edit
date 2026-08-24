"""Raw-free count comparison for P1R54 FZ writer-cadence reports."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import ODEBFContractError, canonical_hash


def build_writer_cadence_comparison(
    *, treatment: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    if (
        treatment.get("status") != "TERMINAL_VALID"
        or treatment.get("completed_batch_count") != 10
        or treatment.get("valid_request_count") != 1000
        or treatment.get("K_writer_call_count") != 10
        or treatment.get("writer_layer_apply_count") != 50
        or treatment.get("target_field_evaluation_count") != 80
        or treatment.get("W0_restored") is not True
        or control.get("status") != "TERMINAL_VALID"
        or control.get("completed_batch_count") != 10
        or control.get("valid_request_count") != 1000
        or control.get("K_writer_call_count") != 80
        or control.get("writer_layer_apply_count") != 400
        or control.get("target_field_evaluation_count") != 80
        or control.get("W0_restored") is not True
        or treatment.get("stream_root") != control.get("stream_root")
        or treatment.get("stream_order") != control.get("stream_order")
    ):
        raise ODEBFContractError("FZ writer-cadence comparison boundary differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-writer-cadence-comparison/v1",
        "batch_count": 10,
        "request_count": 1000,
        "target_field_evaluation_count": {"control": 80, "treatment": 80},
        "writer_call_count": {"control": 80, "treatment": 10, "delta": -70},
        "layer_apply_count": {"control": 400, "treatment": 50, "delta": -350},
        "theoretical_writer_call_reduction_fraction": 0.875,
        "theoretical_layer_apply_reduction_fraction": 0.875,
        "setup_model_load_evaluator_time_separated": True,
        "causal_wall_time_claim": False,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = ["build_writer_cadence_comparison"]
