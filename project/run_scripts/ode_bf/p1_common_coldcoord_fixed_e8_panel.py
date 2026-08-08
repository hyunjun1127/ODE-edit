"""Reused-Warm-seal and resource contracts for common-cold fixed E8 R10."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import load_rooted_json
from .common_cold_coordinate import (
    COMMON_COLD_H,
    COMMON_COLD_INSTRUCTION_ID,
    CommonColdScale,
    common_cold_source_contract,
)
from .contracts import BATCH_SIZE, MODEL_ALIASES, ODEBFContractError, canonical_hash
from .p1_cold_structp_softp_noveto_panel import (
    PRIOR_P1R5_TECHNICAL_GPU,
    SERVER1_GPU_ALLOCATABLE_CALIBRATION_BYTES,
    SERVER1_GPU_PHYSICAL_TOTAL_BYTES,
)
from .p1_selection import (
    _load_canonical_request_map,
    _sha256_file,
)
from .request_digest import ordered_request_digest_v1
from .resource import forecast_p1_adaptive_b10_memory
from .sampling import LineageSeal, SamplingSeal, StatelessReplaySchedule


COMMON_COLD_SCHEMA_NAMESPACE = "ode-edit-s05-common-coldcoord-fixed-e8-p1r10"
COMMON_COLD_RESULT_TOKEN = "common-coldcoord-fixed-e8-p1r10-r1-v1"
COMMON_COLD_PARENT_HEAD = "3711f371c16e360d809dd5f0b4b1a5271a870c26"
COMMON_COLD_CASE_SEAL_FILE = "p1r10_common_coldcoord_cf_b10_seal.json"
COMMON_COLD_NUMERICAL_LOCK_FILE = (
    "numerical_lock_s05_common_coldcoord_fixed_e8.json"
)
COMMON_COLD_PANEL_LABELS = (
    "RS-NEUTRAL",
    "RS-SOFT",
    "BG-NEUTRAL",
)
COMMON_COLD_ALLOCATION_SECONDS = 86_400
COMMON_COLD_FORECAST_SECONDS = 75_600
WARM_ALLOFF_SOURCE_HEAD = "c3d45eba301d3e449a03a21f7ce65b5aa70d07c2"
WARM_ALLOFF_STREAM_ROOT = (
    "a3e2fbf27e94c3ace4e048027abf1f08f215715388dacc3cf85bb452e8e89157"
)
WARM_ALLOFF_STREAM_FILE_SHA256 = (
    "689dcfbf95b074a946088a31b626f2b64082401a1b525e326ab1e8c46b8a488f"
)
WARM_ALLOFF_BATCH_ORDER_SHA256 = (
    "984fe6ecc419fd0f701dae6032f33556f65a94dbb2850b57196be56c353b015b"
)
WARM_ALLOFF_REQUEST_VECTOR_SHA256 = (
    "f2fe57a602b9fa5d683a5132bc695f858da3e6118b1287641e10e8850cd72d7c"
)
WARM_ALLOFF_CASE_ORDER_SHA256 = (
    "655cef9c436d77490340785060f1f6c84d20371341190e038e5f73b1165f778b"
)
WARM_ALLOFF_COLLISION_ORDER_SHA256 = (
    "088177ba98a83c90706d42793277de01384c5be6970aed2623d8afca0302b9a5"
)
RETIRED_FRESH_DRAFT_SHA256 = (
    "fc4de1df6da8dadf2dc7d37eee5911aa4bd4e3f4677bc964125e4ae21518b55c"
)
RETIRED_FRESH_DRAFT_SIZE = 4_802
WARM_ALLOFF_ROOTS = {
    "llama3-8b-inst": {
        "file_count": 179,
        "tree_sha256": "777933f7adcad03b3607554fa69b1dd507e86e67fa15107d428956f079ffec79",
        "terminal_sha256": "5c8eedcb7b566a41cb70b71ae3b690e193bd102fd5c277265790556838f3b991",
        "manifest_sha256": "ea7c9a31bbfcc166417038d1d6d88b7a3177bf2214368a4ac386b01c6ea4c233",
        "action_freeze_sha256": "e4e05c99552883611372c86a7578266de65518adaa1682a50dc4a60852e148c5",
        "margin_arm_terminal_sha256": "65525a0c32f12c4ddfea8ce634eac6604dde1dd7c6a9885da9de50d08721bb80",
        "newnll_arm_terminal_sha256": "f9d864d0ba0be51147dea39dea7b235ee3d417f88cb1d244af5a57e217e4e06d",
        "newnll_endpoint_evaluation_sha256": "5bd61571162e190e87c678e5a54263cd0d79c6f6b9bab6795b922907786c993f",
        "evaluation_case_identity_sha256": "04012bcc468cea7cf896a09ca29841146e452692e504016b3d3463526c6e4354",
        "target_span_sha256": "26183978152a5a34bb2fea40030b772feaa506efe603fbd3103e5d1771599cf8",
        "context_file_sha256": "ea432a1ee287b4a0da021edebb3ce0d4e8c95e3f46dd074aa39f0de9f91a15be",
        "context_sha256": "27b0b1ae80191e4306935d61c52083dca5f4b8adfcd40e388845786196d91dc4",
        "context_template_sha256": [
            "d56d9071febcde1f8bec20f2d761b06c5dbdfc18cec3d36c68aef82fa20076b6",
            "555684f8c9b3f194a57aada39518f16af4c58551a062ff5591d695593b7cd4f1",
            "9b6c3205672e76b4d0c66448b834df96e1eebc4f9373f078d58c66e251fdde9a",
            "551afbae633bed1d023dda8f2ccba79f44872934a7eb74a2376331407b56dfcc",
            "be3ec081371c6cbe54fdefb0e05f9349a50c10be09738127f6995bab121449a2",
            "806ccd94df21205da2176e38eb9b6144d52b5f86c27ceebaf750751774336a00",
        ],
    },
    "qwen2.5-7b-inst": {
        "file_count": 179,
        "tree_sha256": "a05ec39709e4d83572424c5a3e6d919965157a393d74a6c02b66e158ed4a7785",
        "terminal_sha256": "7d5f1046999cd280b73e73dd95de3dd77052d9215c88834e4bd858c95d839922",
        "manifest_sha256": "af83bdda66f434fb7cb78ce7cdfe84f13e5556ea2b063dfa54c6b00c357d9f54",
        "action_freeze_sha256": "be9c6207e30eccbbc17c14bd2e7ac4a47fcd0e1f191285aa52eff3e98882c5ba",
        "margin_arm_terminal_sha256": "8b8d1f115e0a69d6e146b642d05dabe082a9f8ed376e210cb8792deb5fabe1fa",
        "newnll_arm_terminal_sha256": "d1c7cb27375b3d6014854fec83b88ec61767bf17ff79d865425e174c100d22ae",
        "newnll_endpoint_evaluation_sha256": "6a7f2329f210b3064e0da02a0f922237f5ce016c77af342bbfc9498a71eba06d",
        "evaluation_case_identity_sha256": "04012bcc468cea7cf896a09ca29841146e452692e504016b3d3463526c6e4354",
        "target_span_sha256": "a874d9c818fa0f92ff60f3c34d0bdb9098e2929312b98703803027ec3876e92a",
        "context_file_sha256": "4cf00360f916dfd9f957a88e5dd0e1700316a9514ba796f6b901508a918b7082",
        "context_sha256": "5e93d637462aeb4056881c61c4e4d73940b7cab2875e8b02482532bb297bccdc",
        "context_template_sha256": [
            "d56d9071febcde1f8bec20f2d761b06c5dbdfc18cec3d36c68aef82fa20076b6",
            "1e21f7ce2d23872edf9a4caaabe36289845583c298fb59207bf620662ccdcb01",
            "d72e62084002e66cd4e92324bd211cd485c5a562d548e699ba12caffc4a02300",
            "6bf840cf4d530008a6351df6c9d49d3a26c8c7edaa43d512edd07fa6141bdc6f",
            "c53a52094c6025dcaee645993dd00e245c7e2f20e04b2d0a81f18cdbf89dd286",
            "550162ca09f8489ab42a0b899de4e2339442a45afdf24537ddeb385cb6aaec6c",
        ],
    },
}


def expected_common_cold_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("common cold result alias differs")
    return f"s05-common-coldcoord-fixed-e8-p1r10-r1-{alias}-v1"


def build_common_cold_case_seal(
    dataset_path: Path,
    source_stream_path: Path,
) -> dict[str, Any]:
    """Re-root the exact immutable Warm ALLOFF batch-0 request vector."""

    dataset = dataset_path.resolve(strict=True)
    source_path = source_stream_path.resolve(strict=True)
    if _sha256_file(source_path) != WARM_ALLOFF_STREAM_FILE_SHA256:
        raise ODEBFContractError("common cold warm source seal bytes differ")
    source_stream = json.loads(source_path.read_text(encoding="utf-8"))
    source_root = source_stream.pop("root_digest", None)
    if source_root != WARM_ALLOFF_STREAM_ROOT or source_root != canonical_hash(source_stream):
        raise ODEBFContractError("common cold warm source seal root differs")
    source_requests = source_stream.get("requests")
    if not isinstance(source_requests, list) or len(source_requests) < BATCH_SIZE:
        raise ODEBFContractError("common cold warm source batch differs")
    requests = [
        {
            "ordinal": ordinal,
            "case_id": int(item["case_id"]),
            "request_sha256": str(item["request_sha256"]),
            "collision_sha256": str(item["collision_sha256"]),
        }
        for ordinal, item in enumerate(source_requests[:BATCH_SIZE])
    ]
    if canonical_hash(requests) != WARM_ALLOFF_REQUEST_VECTOR_SHA256:
        raise ODEBFContractError("common cold warm request vector differs")
    source = source_stream.get("source")
    if (
        not isinstance(source, Mapping)
        or dataset.stat().st_size != source.get("size_bytes")
        or _sha256_file(dataset) != source.get("sha256")
    ):
        raise ODEBFContractError("common cold warm dataset bytes differ")
    payload: dict[str, Any] = {
        "schema_version": f"{COMMON_COLD_SCHEMA_NAMESPACE}-reused-warm-b10-seal/v1",
        "instruction_id": COMMON_COLD_INSTRUCTION_ID,
        "status": "REUSED_WARMUP_SEAL_CAUSAL_REGRESSION",
        "benchmark": "counterfact",
        "edit_batch_size": BATCH_SIZE,
        "sequential_batch_count": 1,
        "logical_edit_count": BATCH_SIZE,
        "source": dict(source),
        "warm_source": {
            "source_head": WARM_ALLOFF_SOURCE_HEAD,
            "source_stream_root_digest": WARM_ALLOFF_STREAM_ROOT,
            "source_stream_file_sha256": WARM_ALLOFF_STREAM_FILE_SHA256,
            "source_batch_index": 0,
            "source_batch_order_sha256": WARM_ALLOFF_BATCH_ORDER_SHA256,
            "source_request_vector_sha256": WARM_ALLOFF_REQUEST_VECTOR_SHA256,
            "source_roots": WARM_ALLOFF_ROOTS,
            "selected_historical_arm": "FR-A8-NEWNLL-ALLOFF",
            "matched_margin_arm_also_preserved": True,
            "outcome_selection_bias": "BEST_COMPLETED_WARM_ALLOFF_PANEL_REUSE",
        },
        "retired_fresh_draft": {
            "sha256": RETIRED_FRESH_DRAFT_SHA256,
            "size_bytes": RETIRED_FRESH_DRAFT_SIZE,
            "tracked": False,
            "model_load_count": 0,
            "gpu_action_count": 0,
            "slurm_submission_count": 0,
            "result_root_creation_count": 0,
            "status": "RETIRED_BEFORE_USE_BY_A4",
        },
        "requests": requests,
        "batch_ordered_request_digest_v1": [
            ordered_request_digest_v1(
                [str(item["request_sha256"]) for item in requests]
            )
        ],
        "request_vector_sha256": canonical_hash(requests),
        "case_order_sha256": canonical_hash(
            [int(item["case_id"]) for item in requests]
        ),
        "collision_order_sha256": canonical_hash(
            [str(item["collision_sha256"]) for item in requests]
        ),
        "warm_request_and_order_exact_equal": True,
        "same_case_order_both_aliases": True,
        "heldout_fields_opened_during_selection": 0,
        "panel_kind": "REUSED_WARMUP_SEAL_CAUSAL_REGRESSION",
        "unseen_or_fresh_sample_claim_authorized": False,
        "scientific_promotion_authorized": False,
    }
    payload["root_digest"] = canonical_hash(payload)
    return verify_common_cold_case_seal(payload)


def verify_common_cold_case_seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    observed_root = payload.pop("root_digest", None)
    if observed_root != canonical_hash(payload):
        raise ODEBFContractError("common cold case seal root differs")
    expected = {
        "schema_version": f"{COMMON_COLD_SCHEMA_NAMESPACE}-reused-warm-b10-seal/v1",
        "instruction_id": COMMON_COLD_INSTRUCTION_ID,
        "status": "REUSED_WARMUP_SEAL_CAUSAL_REGRESSION",
        "benchmark": "counterfact",
        "edit_batch_size": BATCH_SIZE,
        "sequential_batch_count": 1,
        "logical_edit_count": BATCH_SIZE,
        "same_case_order_both_aliases": True,
        "heldout_fields_opened_during_selection": 0,
        "panel_kind": "REUSED_WARMUP_SEAL_CAUSAL_REGRESSION",
        "unseen_or_fresh_sample_claim_authorized": False,
        "scientific_promotion_authorized": False,
    }
    if any(payload.get(key) != expected_value for key, expected_value in expected.items()):
        raise ODEBFContractError("common cold case seal policy differs")
    requests = payload.get("requests")
    if not isinstance(requests, list) or len(requests) != BATCH_SIZE:
        raise ODEBFContractError("common cold case seal count differs")
    if [item.get("ordinal") for item in requests] != list(range(BATCH_SIZE)):
        raise ODEBFContractError("common cold case order differs")
    case_ids = [item.get("case_id") for item in requests]
    request_ids = [item.get("request_sha256") for item in requests]
    collision_ids = [item.get("collision_sha256") for item in requests]
    if any(
        len(set(items)) != BATCH_SIZE
        for items in (case_ids, request_ids, collision_ids)
    ):
        raise ODEBFContractError("common cold selected identities are not distinct")
    if canonical_hash(requests) != WARM_ALLOFF_REQUEST_VECTOR_SHA256:
        raise ODEBFContractError("common cold warm request vector differs")
    if payload.get("batch_ordered_request_digest_v1") != [
        ordered_request_digest_v1([str(item) for item in request_ids])
    ] or payload.get("batch_ordered_request_digest_v1") != [
        WARM_ALLOFF_BATCH_ORDER_SHA256
    ]:
        raise ODEBFContractError("common cold request digest differs")
    source = payload.get("source")
    warm = payload.get("warm_source")
    retired = payload.get("retired_fresh_draft")
    if (
        not isinstance(source, Mapping)
        or source.get("relative_contract")
        != "EasyEdit/data/counterfact/counterfact.json"
        or not isinstance(warm, Mapping)
        or warm.get("source_head") != WARM_ALLOFF_SOURCE_HEAD
        or warm.get("source_stream_root_digest") != WARM_ALLOFF_STREAM_ROOT
        or warm.get("source_stream_file_sha256")
        != WARM_ALLOFF_STREAM_FILE_SHA256
        or warm.get("source_batch_order_sha256")
        != WARM_ALLOFF_BATCH_ORDER_SHA256
        or warm.get("source_request_vector_sha256")
        != WARM_ALLOFF_REQUEST_VECTOR_SHA256
        or warm.get("source_batch_index") != 0
        or warm.get("selected_historical_arm") != "FR-A8-NEWNLL-ALLOFF"
        or warm.get("matched_margin_arm_also_preserved") is not True
        or warm.get("source_roots") != WARM_ALLOFF_ROOTS
        or warm.get("outcome_selection_bias")
        != "BEST_COMPLETED_WARM_ALLOFF_PANEL_REUSE"
        or not isinstance(retired, Mapping)
        or retired.get("sha256") != RETIRED_FRESH_DRAFT_SHA256
        or retired.get("size_bytes") != RETIRED_FRESH_DRAFT_SIZE
        or any(
            retired.get(key) != 0
            for key in (
                "model_load_count",
                "gpu_action_count",
                "slurm_submission_count",
                "result_root_creation_count",
            )
        )
        or payload.get("request_vector_sha256")
        != WARM_ALLOFF_REQUEST_VECTOR_SHA256
        or payload.get("case_order_sha256") != WARM_ALLOFF_CASE_ORDER_SHA256
        or payload.get("collision_order_sha256")
        != WARM_ALLOFF_COLLISION_ORDER_SHA256
        or payload.get("warm_request_and_order_exact_equal") is not True
    ):
        raise ODEBFContractError("common cold source/exclusion differs")
    payload["root_digest"] = observed_root
    return payload


def load_common_cold_requests(
    dataset_path: Path, seal: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    value = verify_common_cold_case_seal(seal)
    dataset = dataset_path.resolve(strict=True)
    source = value["source"]
    if dataset.stat().st_size != source["size_bytes"] or _sha256_file(dataset) != source["sha256"]:
        raise ODEBFContractError("common cold dataset bytes differ")
    approved = {
        str(item["request_sha256"]): (
            int(item["case_id"]),
            str(item["collision_sha256"]),
        )
        for item in value["requests"]
    }
    loaded = _load_canonical_request_map(dataset, approved)
    ordered = tuple(loaded[str(item["request_sha256"])] for item in value["requests"])
    if ordered_request_digest_v1(
        [str(item["request_sha256"]) for item in ordered]
    ) != value["batch_ordered_request_digest_v1"][0]:
        raise ODEBFContractError("common cold loaded request order differs")
    return ordered


def common_cold_schedule(base: SamplingSeal) -> StatelessReplaySchedule:
    lineages = tuple(
        LineageSeal(
            item.lineage,
            item.population_sha256,
            int(
                hashlib.sha256(
                    f"{COMMON_COLD_INSTRUCTION_ID}|{item.lineage.value}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:15],
                16,
            ),
            item.sample_count,
            item.allowed_item_sha256,
            item.strata,
        )
        for item in base.lineages
    )
    return StatelessReplaySchedule(SamplingSeal(base.population_sha256, base.items, lineages))


def common_cold_schedule_receipt(schedule: StatelessReplaySchedule) -> dict[str, Any]:
    payload = {
        "schema": f"{COMMON_COLD_SCHEMA_NAMESPACE}-replay-schedule/v1",
        "instruction_id": COMMON_COLD_INSTRUCTION_ID,
        "population_sha256": schedule.seal.population_sha256,
        "lineages": {
            item.lineage.value: {
                "seed": item.seed,
                "seal_sha256": item.identity(),
                "pool_count": len(item.allowed_item_sha256),
                "sample_count": item.sample_count,
            }
            for item in schedule.seal.lineages
        },
        "controller_terminal_distinct": True,
        "arm_identity_in_schedule": False,
        "model_identity_in_schedule": False,
        "state_digest": schedule.state_digest,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


@dataclass(frozen=True, slots=True)
class CommonColdResourceForecast:
    alias: str
    arm_count: int
    bootstrap_count: int
    fields_per_arm: int
    candidates_per_arm: int
    functional_basis_endpoints_per_arm: int
    target_backward_batches_per_arm: int
    conservative_gpu_peak_mib: int
    conservative_host_peak_mib: int
    conservative_time_seconds: int
    physical_total_bytes: int
    allocatable_calibration_bytes: int
    allocation_time_seconds: int
    fits_envelope: bool
    prior_peak_reserved_bytes: int
    parent_forecast_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def forecast_common_cold_panel(
    artifact_lock_path: Path,
    base_model_lock_path: Path,
    alias: str,
) -> CommonColdResourceForecast:
    parent = forecast_p1_adaptive_b10_memory(
        artifact_lock_path, base_model_lock_path, alias
    )
    prior = PRIOR_P1R5_TECHNICAL_GPU[alias]
    gpu = math.ceil(int(prior["peak_reserved_bytes"]) / (1024 * 1024)) + 512 + 8192
    host = max(parent.forecast_host_peak_mib + 3072, 32_000)
    fits = (
        gpu <= SERVER1_GPU_ALLOCATABLE_CALIBRATION_BYTES // (1024 * 1024)
        and host <= 65_000
        and COMMON_COLD_FORECAST_SECONDS <= COMMON_COLD_ALLOCATION_SECONDS
    )
    return CommonColdResourceForecast(
        alias,
        3,
        2,
        8,
        8,
        48,
        8,
        gpu,
        host,
        COMMON_COLD_FORECAST_SECONDS,
        SERVER1_GPU_PHYSICAL_TOTAL_BYTES,
        SERVER1_GPU_ALLOCATABLE_CALIBRATION_BYTES,
        COMMON_COLD_ALLOCATION_SECONDS,
        fits,
        int(prior["peak_reserved_bytes"]),
        parent.identity(),
    )


def validate_common_cold_runtime_gpu_capacity(
    forecast: CommonColdResourceForecast,
    *,
    device_property_total_bytes: int,
    allocatable_total_bytes: int,
    free_bytes: int,
) -> dict[str, Any]:
    values = (device_property_total_bytes, allocatable_total_bytes, free_bytes)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ODEBFContractError("common cold GPU capacity receipt differs")
    # On this locked CUDA device torch.cuda.get_device_properties.total_memory
    # reports the stable allocatable calibration, while the physical inventory
    # total is a separate source/forecast provenance fact.  Comparing those two
    # APIs byte-for-byte is a cross-semantics plumbing error.
    if device_property_total_bytes != forecast.allocatable_calibration_bytes:
        raise ODEBFContractError("common cold stable GPU device identity differs")
    if (
        not forecast.fits_envelope
        or allocatable_total_bytes > device_property_total_bytes
        or free_bytes > allocatable_total_bytes
    ):
        raise ODEBFContractError("common cold allocatable GPU capacity differs")
    required = forecast.conservative_gpu_peak_mib * 1024 * 1024
    if allocatable_total_bytes < required or free_bytes < required:
        raise ODEBFContractError("common cold runtime GPU capacity is insufficient")
    payload = {
        "stable_identity_api": "torch.cuda.get_device_properties.total_memory",
        "stable_device_total_bytes": device_property_total_bytes,
        "physical_inventory_total_bytes": forecast.physical_total_bytes,
        "physical_inventory_role": "LOCKED_FORECAST_PROVENANCE_ONLY",
        "runtime_capacity_api": "torch.cuda.mem_get_info",
        "allocatable_total_bytes": allocatable_total_bytes,
        "free_bytes": free_bytes,
        "required_free_bytes": required,
        "physical_and_allocatable_semantics_separate": True,
        "passed": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def validate_common_cold_lock(
    value: Mapping[str, Any],
    *,
    controller_identity_sha256: str,
    case_root_digest: str,
    population_root_digest: str,
    schedule: StatelessReplaySchedule,
) -> dict[str, Any]:
    payload = dict(value)
    root = payload.pop("root_digest", None)
    if root != canonical_hash(payload):
        raise ODEBFContractError("common cold numerical lock root differs")
    expected = {
        "schema": f"{COMMON_COLD_SCHEMA_NAMESPACE}-numerical-lock/v1",
        "instruction_id": COMMON_COLD_INSTRUCTION_ID,
        "parent_checkpoint": COMMON_COLD_PARENT_HEAD,
        "controller_geometry_identity_sha256": controller_identity_sha256,
        "case_root_digest": case_root_digest,
        "population_root_digest": population_root_digest,
        "schedule_identity_sha256": common_cold_schedule_receipt(schedule)[
            "identity_sha256"
        ],
        "arms": list(COMMON_COLD_PANEL_LABELS),
        "scales": {
            "RS-NEUTRAL": CommonColdScale.ROBUST_SHARED.value,
            "RS-SOFT": CommonColdScale.ROBUST_SHARED.value,
            "BG-NEUTRAL": CommonColdScale.BATCH_GLOBAL.value,
        },
        "residual_policy": "SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1",
        "bootstrap_count_by_scale": 1,
        "bootstrap_h": COMMON_COLD_H,
        "joint_grid_count": 8,
        "joint_h": COMMON_COLD_H,
        "joint_tau_final": 1.0,
        "scientific_retry_count": 0,
        "scientific_rejection_count": 0,
        "hard_h_p_budget_influence_count": 0,
        "native_or_direct_z_cold_access_count": 0,
        "first_hit_observation_only": True,
        "soft_stage2_policy": "CERTIFIED_STAGE1_FALLBACK",
        "method_semantic_identity_sha256": common_cold_source_contract()[
            "identity_sha256"
        ],
        "scientific_promotion_authorized": False,
    }
    if payload != expected:
        raise ODEBFContractError("common cold numerical lock differs")
    payload["root_digest"] = root
    return payload


def load_and_validate_common_cold_lock(
    path: Path,
    **kwargs: Any,
) -> tuple[dict[str, Any], str]:
    value, raw_sha = load_rooted_json(path)
    return validate_common_cold_lock(value, **kwargs), raw_sha


__all__ = [
    "COMMON_COLD_CASE_SEAL_FILE",
    "COMMON_COLD_NUMERICAL_LOCK_FILE",
    "COMMON_COLD_PANEL_LABELS",
    "COMMON_COLD_RESULT_TOKEN",
    "COMMON_COLD_SCHEMA_NAMESPACE",
    "build_common_cold_case_seal",
    "common_cold_schedule",
    "common_cold_schedule_receipt",
    "expected_common_cold_result_name",
    "forecast_common_cold_panel",
    "load_and_validate_common_cold_lock",
    "load_common_cold_requests",
    "validate_common_cold_runtime_gpu_capacity",
    "verify_common_cold_case_seal",
]
