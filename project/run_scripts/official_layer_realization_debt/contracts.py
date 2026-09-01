"""Immutable deployment and observational contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


INSTRUCTION_ID = "ODEEDIT-S06-OFFICIAL-LAYER-REALIZATION-DEBT-B10X10-V1"
NONCE = "ODEEDIT-GH-SH4-OFFICIAL-LAYER-DEBT-20260901-R1"
SOURCE_BASE_HEAD = "ddc178584ef14efd5d4e1271b3c324e3ebd3e443"
SOURCE_BASE_TREE = "83889fc3d2e32119dafdffc969482de761eedfe2"
EASYEDIT_ROOT = Path("/data/janghj/EasyEdit-stock-14cea824")
EASYEDIT_HEAD = "14cea8245f06715684592ab55184939b99d70784"
EASYEDIT_TREE = "9c52aadbc0883da422badf0a730fff21aaa3a8a7"
DATASET = Path("/data/janghj/EasyEdit/data/counterfact/counterfact.json")
DATASET_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"
STREAM_PARENT = Path(
    "/data/janghj/ODE-edit/local/state/p4-target-side-semantic-barrier-v1/sealed-stream"
)
STREAM_ARCHIVE = STREAM_PARENT / "p4-r52-independent-b10x10-sample-order-transfer-v2.tar"
STREAM_ROOT = STREAM_PARENT / "p4-r52-independent-b10x10-sample-order-transfer-v2"
STREAM_IDENTITY = "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6"
ORDER_IDENTITY = "abe62c071168789b4a1e5ff57d2645ea16a328946362ea7bff166d6d5c76cd5c"
EVALUATOR_IDENTITY = "72b8ecb737157a42d6a055ffd339dc3f907876a0a98165a00cca49ca9bbed07d"
LAYERS = (4, 5, 6, 7, 8)
IDEAL_Q = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)


class ObservationBoundary(RuntimeError):
    """Fail-close technical/instrumentation boundary."""


class Method(str, Enum):
    MEMIT = "memit"
    ALPHAEDIT = "alphaedit"


@dataclass(frozen=True, slots=True)
class ObservationLock:
    layers: tuple[int, ...] = LAYERS
    scalar_reduction_dtype: str = "float64"
    recurrence_relative_tolerance: float = 1.0e-4
    full_fp32: bool = True
    b1_request_count: int = 1
    b10_slice_count: int = 10
    requests_per_slice: int = 10
    total_requests_per_cell: int = 100
    expected_layer_loop_calls: int = 5
    expected_terminal_forward_calls: int = 1
    sample_duplication_count: int = 0
    cross_batch_weight_continuity: int = 0
    cross_batch_history_continuity: int = 0
    scientific_promotion: bool = False

    def payload(self) -> dict[str, Any]:
        return asdict(self)
