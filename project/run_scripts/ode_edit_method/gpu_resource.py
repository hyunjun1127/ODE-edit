"""Stable GPU identity and runtime-capacity receipts for hard-B10 runs.

Physical inventory, CUDA device properties, and ``cudaMemGetInfo`` capacity
are deliberately separate namespaces.  In particular, no runtime capacity
value is required to equal the NVML physical inventory byte-for-byte.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import MethodContractError


def _integer(name: str, value: Any, *, positive: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MethodContractError(f"GPU resource {name} is not an integer byte value")
    if (positive and value <= 0) or (not positive and value < 0):
        raise MethodContractError(f"GPU resource {name} is out of range")
    return value


def _uuid(value: Any) -> str:
    observed = str(value).strip()
    if observed and not observed.startswith("GPU-") and len(observed) == 36:
        observed = f"GPU-{observed}"
    if not observed.startswith("GPU-") or len(observed) != 40:
        raise MethodContractError("CUDA device UUID is invalid")
    return observed.lower()


@dataclass(frozen=True, slots=True)
class GpuResourceReceipt:
    visible_device_count: int
    visible_index: int
    name: str
    uuid: str
    capability: tuple[int, int]
    cuda_device_property_total_bytes: int
    cuda_mem_get_info_free_bytes: int
    cuda_mem_get_info_total_bytes: int
    torch_memory_allocated_bytes: int
    torch_memory_reserved_bytes: int
    conservative_peak_forecast_bytes: int
    safety_reserve_bytes: int
    required_capacity_bytes: int
    reusable_capacity_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "ode-edit-gpu-resource-receipt/v1",
            "stable_identity_api": "torch.cuda.get_device_properties(0)",
            "runtime_capacity_api": "torch.cuda.mem_get_info(0)",
            "units": "bytes",
            "visible_device_count": self.visible_device_count,
            "visible_index": self.visible_index,
            "name": self.name,
            "uuid": self.uuid,
            "capability": list(self.capability),
            "cuda_device_property_total_bytes": (
                self.cuda_device_property_total_bytes
            ),
            "cuda_mem_get_info_free_bytes": self.cuda_mem_get_info_free_bytes,
            "cuda_mem_get_info_total_bytes": self.cuda_mem_get_info_total_bytes,
            "torch_memory_allocated_bytes": self.torch_memory_allocated_bytes,
            "torch_memory_reserved_bytes": self.torch_memory_reserved_bytes,
            "conservative_peak_forecast_bytes": (
                self.conservative_peak_forecast_bytes
            ),
            "safety_reserve_bytes": self.safety_reserve_bytes,
            "required_capacity_bytes": self.required_capacity_bytes,
            "reusable_capacity_bytes": self.reusable_capacity_bytes,
            "physical_inventory_exact_equality_required": False,
            "capacity_pass": True,
        }


def inspect_gpu_resource(
    cuda_api: Any,
    *,
    model_alias: str,
    resource_lock: Mapping[str, Any],
) -> GpuResourceReceipt:
    """Validate one CUDA-visible device and its usable capacity fail-closed."""

    if not bool(cuda_api.is_available()):
        raise MethodContractError("CUDA resource is unavailable")
    raw_count = cuda_api.device_count()
    count = _integer("visible device count", raw_count, positive=True)
    expected_count = _integer(
        "expected visible device count",
        resource_lock.get("expected_visible_device_count"),
        positive=True,
    )
    if count != expected_count or count != 1:
        raise MethodContractError("CUDA visible device count differs")
    raw_index = cuda_api.current_device()
    index = _integer("visible device index", raw_index, positive=False)
    expected_index = _integer(
        "expected visible device index",
        resource_lock.get("expected_visible_index"),
        positive=False,
    )
    if index != expected_index or index != 0:
        raise MethodContractError("CUDA visible device index differs")

    properties = cuda_api.get_device_properties(index)
    name = str(getattr(properties, "name", ""))
    observed_uuid = _uuid(getattr(properties, "uuid", ""))
    allowed_uuids = tuple(
        _uuid(value) for value in resource_lock.get("allowed_device_uuids", ())
    )
    raw_capability = cuda_api.get_device_capability(index)
    if (
        not isinstance(raw_capability, (tuple, list))
        or len(raw_capability) != 2
    ):
        raise MethodContractError("CUDA capability schema differs")
    capability = (
        _integer("capability major", raw_capability[0], positive=False),
        _integer("capability minor", raw_capability[1], positive=False),
    )
    expected_capability = tuple(resource_lock.get("expected_capability", ()))
    property_total = _integer(
        "device-property total memory",
        getattr(properties, "total_memory", None),
        positive=True,
    )
    expected_property_total = _integer(
        "expected device-property total memory",
        resource_lock.get("cuda_device_property_total_bytes"),
        positive=True,
    )
    physical_total = _integer(
        "physical inventory total memory",
        resource_lock.get("physical_inventory_total_bytes"),
        positive=True,
    )
    if (
        name != resource_lock.get("expected_name")
        or not allowed_uuids
        or observed_uuid not in allowed_uuids
        or capability != expected_capability
        or property_total != expected_property_total
        or physical_total < property_total
    ):
        raise MethodContractError("CUDA stable device identity differs")

    raw_info = cuda_api.mem_get_info(index)
    if not isinstance(raw_info, (tuple, list)) or len(raw_info) != 2:
        raise MethodContractError("CUDA capacity API schema differs")
    free_bytes = _integer("allocatable free memory", raw_info[0], positive=True)
    capacity_total = _integer(
        "allocatable total memory", raw_info[1], positive=True
    )
    allocated_bytes = _integer(
        "torch allocated memory", cuda_api.memory_allocated(index), positive=False
    )
    reserved_bytes = _integer(
        "torch reserved memory", cuda_api.memory_reserved(index), positive=False
    )
    if (
        free_bytes > capacity_total
        or allocated_bytes > reserved_bytes
        or reserved_bytes > capacity_total
        or free_bytes + reserved_bytes > capacity_total
    ):
        raise MethodContractError("CUDA runtime capacity accounting differs")

    forecasts = resource_lock.get("conservative_peak_forecast_bytes")
    if not isinstance(forecasts, Mapping) or model_alias not in forecasts:
        raise MethodContractError("GPU model forecast is absent")
    forecast = _integer(
        "conservative peak forecast", forecasts[model_alias], positive=True
    )
    reserve = _integer(
        "safety reserve", resource_lock.get("safety_reserve_bytes"), positive=True
    )
    required = forecast + reserve
    reusable = free_bytes + reserved_bytes
    if (
        not math.isfinite(float(required))
        or property_total < required
        or capacity_total < required
        or reusable < required
    ):
        raise MethodContractError("CUDA runtime capacity is below the locked forecast")

    return GpuResourceReceipt(
        visible_device_count=count,
        visible_index=index,
        name=name,
        uuid=observed_uuid,
        capability=capability,
        cuda_device_property_total_bytes=property_total,
        cuda_mem_get_info_free_bytes=free_bytes,
        cuda_mem_get_info_total_bytes=capacity_total,
        torch_memory_allocated_bytes=allocated_bytes,
        torch_memory_reserved_bytes=reserved_bytes,
        conservative_peak_forecast_bytes=forecast,
        safety_reserve_bytes=reserve,
        required_capacity_bytes=required,
        reusable_capacity_bytes=reusable,
    )


__all__ = ["GpuResourceReceipt", "inspect_gpu_resource"]
