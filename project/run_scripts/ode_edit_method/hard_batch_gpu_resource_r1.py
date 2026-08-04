"""Fail-closed loader for the hard-B10 GPU resource repair receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import MethodContractError, canonical_hash


GPU_RESOURCE_LOCK_PATH = Path(__file__).with_name(
    "hard_batch_gpu_resource_r1.json"
)
GPU_RESOURCE_SCHEMA = "ode-edit-session03-hard-b10-gpu-resource-r1/v1"
GPU_RESOURCE_INSTRUCTION = (
    "ODEEDIT-S03-HARD-B10-P0-GPU-MEMORY-IDENTITY-R1-V1"
)
ALLOWED_UUIDS = (
    "GPU-2087c567-ec90-0aa2-09c5-c5174daec87a",
    "GPU-5d6a1f79-0105-d49c-d8de-40c9438d6d85",
    "GPU-8a497f13-d58c-bdef-5478-611ea200deae",
    "GPU-e3f0bff5-fdfd-b411-a133-d04104e5cfee",
    "GPU-d623a715-ed10-696d-07ec-50b7614c4272",
    "GPU-08613291-f9aa-4b3b-8ee3-b4a48f542f22",
    "GPU-8b402bc3-f797-6774-24e2-d1c920adadfb",
    "GPU-9edcc6a0-bc56-e743-e1d7-05815ef24483",
)


def validate_gpu_resource_lock(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != GPU_RESOURCE_SCHEMA
        or payload.get("instruction_id") != GPU_RESOURCE_INSTRUCTION
        or payload.get("parent_instruction_id")
        != "ODEEDIT-S03-HARD-B10-K20-CYCLE-ATTRIBUTION-P0P1-V1"
        or payload.get("implementation_base_commit")
        != "926dc79808cac0bc82b8a82071d8889e9df1b297"
        or payload.get("scientific_proposal_id")
        != "f563618c4723cd9dac6a1b669da314191b25a9b0b2050dd7ff04e33b2f179f6a"
        or payload.get("scientific_lock_sha256")
        != "e0bb340abcfe26a70a5964b3034e2163ae435daa92aaa7447cdcb4cd38ab6eda"
        or payload.get("stable_identity_api")
        != "torch.cuda.get_device_properties(0)"
        or payload.get("runtime_capacity_api") != "torch.cuda.mem_get_info(0)"
        or payload.get("physical_inventory_api")
        != "NVML nvidia-smi memory.total"
        or payload.get("units") != "bytes"
        or payload.get("expected_visible_device_count") != 1
        or payload.get("expected_visible_index") != 0
        or payload.get("expected_name") != "NVIDIA RTX A6000"
        or tuple(payload.get("expected_capability", ())) != (8, 6)
        or tuple(payload.get("allowed_device_uuids", ())) != ALLOWED_UUIDS
        or payload.get("physical_inventory_total_bytes") != 51527024640
        or payload.get("cuda_device_property_total_bytes") != 50899386368
        or dict(payload.get("conservative_peak_forecast_bytes", {}))
        != {
            "llama3-8b-inst": 43486543872,
            "qwen2.5-7b-inst": 46933255128,
        }
        or payload.get("safety_reserve_bytes") != 2147483648
        or payload.get("physical_inventory_exact_equality_required") is not False
        or payload.get("root_suffix") != "r1"
        or payload.get("submission_authorized") is not False
    ):
        raise MethodContractError("hard-B10 GPU resource lock differs")


def load_gpu_resource_lock(
    path: str | Path = GPU_RESOURCE_LOCK_PATH,
) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MethodContractError("hard-B10 GPU resource lock is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise MethodContractError("hard-B10 GPU resource lock root differs")
    validate_gpu_resource_lock(payload)
    result = dict(payload)
    result["proposal_id"] = canonical_hash(payload)
    result["lock_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return result


__all__ = [
    "GPU_RESOURCE_LOCK_PATH",
    "load_gpu_resource_lock",
    "validate_gpu_resource_lock",
]
