"""Production-only helpers for the FP32 residual-reserve Phase-A panel.

The target/controller remains in ``p1_scalable_batched_experiment``.  This
module binds its native IL1 selected target to the audited residual-reserve
adapter and supplies the FP32-only cumulative J0 materialization boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from project.run_scripts.ode_edit_motivation.gpu_runtime import load_fixed_model
from project.run_scripts.ode_edit_motivation.mv0_fidelity import (
    _covariance_specs,
    _load_verified_covariances,
)

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import WaypointFactor, tensor_sha256
from .p1r52_residual_reserve_phase_a_adapter import (
    ResidualReservePhaseAAdapterResult,
    ResidualReservePhaseAArm,
    freeze_residual_reserve_phase_a_action,
    run_residual_reserve_phase_a_outer,
)
from .p1r52_residual_reserve_pre_writer_interface import (
    P1R52IL1NativeSelectedBridgeReceipt,
    reconstruct_native_il1_selected_target,
)
from .p1r52_residual_reserve_production_binding import (
    ResidualReserveProductionBinding,
    build_residual_reserve_production_binding,
)
from .p1r52_residual_reserve_pc_inventory import (
    SealedPrevalidatedCovariance,
)
from .p1r52_target_depth import P1R52TargetDepthOuter


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-RESIDUAL-RESERVE-PCSOFT-WRITER-V1"
METHOD_ID = "P1R52-RESIDUAL-RESERVE-PCSOFT-WRITER-V1"
PHASE_A_ARMS = ("j0", "rr-uniform", "rr-pcsoft", "official-alphaedit")
RR_ARM_BY_TOKEN = {
    "rr-uniform": ResidualReservePhaseAArm.RR_UNIFORM,
    "rr-pcsoft": ResidualReservePhaseAArm.RR_PCSOFT,
}


def expected_phase_a_result_name(
    alias: str,
    arm: str,
    *,
    attempt_suffix: str | None = None,
) -> str:
    if alias not in ("llama3-8b-inst", "qwen2.5-7b-inst") or arm not in PHASE_A_ARMS:
        raise ODEBFContractError("residual-reserve Phase-A result identity differs")
    suffix = "" if attempt_suffix is None else f"-{attempt_suffix}"
    if attempt_suffix is not None and attempt_suffix not in (
        "tech-r1",
        "tech-r2",
        "tech-r3",
    ):
        raise ODEBFContractError("residual-reserve Phase-A attempt suffix differs")
    return f"s05-p1r52-residual-reserve-phase-a-fp32-{alias}-{arm}{suffix}-v1"


def load_phase_a_fp32_model(guard: Any, alias: str) -> tuple[Any, Any, Any, Any]:
    """Reuse the pinned Motivation loader at its accepted FP32 boundary."""

    from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams

    runtime = load_fixed_model(alias)
    model = runtime.model
    tokenizer = runtime.tokenizer
    hparams = AlphaEditHyperParams.from_hparams(str(guard.hparams))
    hparams.device = 0
    hparams.P_loc = str(guard.projector)
    hparams.stats_dir = str(guard.easyedit_root / "examples" / "data" / "stats")
    model.config._name_or_path = hparams.model_name
    floating = tuple(parameter for parameter in model.parameters() if parameter.is_floating_point())
    if (
        not floating
        or any(parameter.dtype is not torch.float32 for parameter in floating)
        or any(parameter.requires_grad for parameter in floating)
        or torch.is_autocast_enabled()
        or torch.is_autocast_enabled("cpu")
        or tuple(int(layer) for layer in hparams.layers) != tuple(guard.spec["layers"])
        or float(hparams.L2) <= 0.0
    ):
        raise ODEBFContractError("residual-reserve FP32 model load differs")
    return model, tokenizer, hparams, runtime


def load_phase_a_covariances(
    runtime: Any,
    *,
    easyedit_root: Path,
) -> tuple[SealedPrevalidatedCovariance, ...]:
    """Reuse the verified read-only covariance loader; never recompute stats."""

    specs = _covariance_specs(easyedit_root, runtime.spec)
    moments, paths = _load_verified_covariances(
        root=easyedit_root,
        runtime=runtime,
        specs=specs,
    )
    if tuple(item.layer for item in specs) != tuple(runtime.spec.layers):
        raise ODEBFContractError("residual-reserve covariance layer order differs")
    result = []
    for spec, relative in zip(specs, paths, strict=True):
        artifact = canonical_hash(
            {
                "layer": spec.layer,
                "path": relative,
                "sha256": spec.identity.sha256,
                "size": spec.identity.size,
                "normalization": "mom2/count",
            }
        )
        result.append(
            SealedPrevalidatedCovariance(spec.layer, moments[spec.layer], artifact)
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ResidualReserveOuterExecution:
    bridge: P1R52IL1NativeSelectedBridgeReceipt
    result: ResidualReservePhaseAAdapterResult
    prefix_capture_receipts: tuple[dict[str, Any], ...]

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "bridge": self.bridge.raw_free_payload(),
            "adapter": self.result.receipt.raw_free_payload(),
            "prefix_captures": list(self.prefix_capture_receipts),
            "publication_identity_sha256": self.result.publication_identity_sha256,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


class ResidualReservePhaseAWriterRuntime:
    """Execute one approved RR adapter call for each native IL1 outer."""

    def __init__(
        self,
        binding: ResidualReserveProductionBinding,
        *,
        arm: ResidualReservePhaseAArm,
        execution_prefix: str,
    ) -> None:
        if not isinstance(binding, ResidualReserveProductionBinding):
            raise ODEBFContractError("residual-reserve production binding differs")
        if not isinstance(arm, ResidualReservePhaseAArm):
            raise ODEBFContractError("residual-reserve production arm differs")
        if not execution_prefix:
            raise ODEBFContractError("residual-reserve execution prefix is empty")
        self.binding = binding
        self.arm = arm
        self.execution_prefix = execution_prefix
        self._executions: list[ResidualReserveOuterExecution] = []

    @property
    def executions(self) -> tuple[ResidualReserveOuterExecution, ...]:
        return tuple(self._executions)

    def execute(
        self,
        outer: P1R52TargetDepthOuter,
        *,
        step_index: int,
    ) -> ResidualReserveOuterExecution:
        selected, bridge = reconstruct_native_il1_selected_target(
            outer,
            step_index=step_index,
        )
        bindings = self.binding.bindings_by_layer()
        action_freeze = freeze_residual_reserve_phase_a_action(selected, bindings)
        capture_count_before = len(self.binding.prefix_provider.receipts)
        result = run_residual_reserve_phase_a_outer(
            selected,
            action_freeze,
            self.arm,
            bindings,
            self.binding.contexts_by_layer(),
            self.binding.prefix_provider,
            self.binding.gross_ledger,
            self.binding.covariances,
            execution_id=f"{self.execution_prefix}:k{step_index + 1}",
        )
        prefix_receipts = tuple(
            item.raw_free_payload()
            for item in self.binding.prefix_provider.receipts[capture_count_before:]
        )
        if len(prefix_receipts) != 10:
            raise ODEBFStateError("residual-reserve outer prefix capture count differs")
        execution = ResidualReserveOuterExecution(bridge, result, prefix_receipts)
        execution.raw_free_payload()
        self._executions.append(execution)
        return execution

    def assert_complete(self) -> None:
        if (
            len(self._executions) != 8
            or self.binding.gross_ledger.state.version != 8
            or any(
                item.result.receipt.committed_state_after_version != index
                for index, item in enumerate(self._executions, start=1)
            )
        ):
            raise ODEBFStateError("residual-reserve K8 execution ledger differs")
        self.binding.prefix_provider.assert_complete(expected_sweeps=16)


class AcceptedPhysicalStateFP32Materializer:
    """J0-only FP32 cumulative endpoint materializer; no BF16 helper calls."""

    def __init__(
        self,
        model: torch.nn.Module,
        entry_values: Mapping[str, torch.Tensor],
        *,
        row_block: int = 64,
    ) -> None:
        parameters = dict(model.named_parameters())
        if (
            set(entry_values) - set(parameters)
            or any(parameters[name].dtype is not torch.float32 for name in entry_values)
            or row_block <= 0
        ):
            raise ODEBFContractError("FP32 J0 materializer entry differs")
        self._parameters = {name: parameters[name] for name in entry_values}
        self._entry = {
            name: value.detach().to(device=parameters[name].device, dtype=torch.float32).clone()
            for name, value in entry_values.items()
        }
        self._pointers = {name: int(value.data_ptr()) for name, value in self._parameters.items()}
        self.row_block = int(row_block)
        self.transition_receipts: list[dict[str, Any]] = []

    def _effective(
        self,
        name: str,
        factors: Sequence[WaypointFactor],
    ) -> torch.Tensor:
        ordered = tuple(sorted(factors, key=lambda item: item.order_key))
        if not ordered or len({item.order_key for item in ordered}) != len(ordered):
            raise ODEBFContractError("FP32 J0 factor inventory differs")
        entry = self._entry[name]
        effective = torch.empty_like(entry)
        with torch.no_grad():
            for start in range(0, entry.shape[0], self.row_block):
                end = min(start + self.row_block, entry.shape[0])
                accumulator = entry[start:end].clone()
                for factor in ordered:
                    if (
                        factor.weight_name != name
                        or tuple(entry.shape) != (factor.left.shape[0], factor.right.shape[0])
                    ):
                        raise ODEBFContractError("FP32 J0 factor shape differs")
                    left = factor.left[start:end].to(device=entry.device, dtype=torch.float32)
                    right = factor.right.to(device=entry.device, dtype=torch.float32)
                    native_order = (right @ left.T).T
                    coefficient = torch.tensor(factor.theta, device=entry.device, dtype=torch.float32)
                    accumulator = accumulator + coefficient * native_order
                effective[start:end].copy_(accumulator)
        if not bool(torch.isfinite(effective).all()):
            raise ODEBFStateError("FP32 J0 endpoint is nonfinite")
        return effective

    def materialize(
        self,
        factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
        *,
        transition_index: int,
    ) -> dict[str, Any]:
        if set(factors_by_weight) != set(self._parameters):
            raise ODEBFContractError("FP32 J0 cumulative factor inventory differs")
        step_energy: dict[str, float] = {}
        cumulative_energy: dict[str, float] = {}
        update_hashes: dict[str, str] = {}
        with torch.no_grad():
            for name in sorted(self._parameters):
                parameter = self._parameters[name]
                if int(parameter.data_ptr()) != self._pointers[name]:
                    raise ODEBFStateError("FP32 J0 parameter pointer differs")
                before = parameter.detach().clone()
                effective = self._effective(name, factors_by_weight[name])
                parameter.copy_(effective)
                step = parameter.detach() - before
                cumulative = parameter.detach() - self._entry[name]
                step_energy[name] = float(torch.sum(step.double().square()).item())
                cumulative_energy[name] = float(torch.sum(cumulative.double().square()).item())
                update_hashes[name] = tensor_sha256(cumulative)
        payload = {
            "schema": "ode-edit-s05-p1r52-rr-phase-a-j0-fp32-materialization/v1",
            "transition_index": int(transition_index),
            "weight_count": len(self._parameters),
            "prepared_fp32_cumulative_update_sha256": update_hashes,
            "realized_fp32_step_energy": step_energy,
            "cumulative_fp32_capacity": cumulative_energy,
            "storage_assignment_count": len(self._parameters),
            "numeric_storage_cast_count": 0,
            "bf16_path_call_count": 0,
            "autocast_count": 0,
            "downcast_count": 0,
            "quantization_count": 0,
            "external_bf16_materializer_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        self.transition_receipts.append(payload)
        return payload

    def restore(self) -> dict[str, Any]:
        with torch.no_grad():
            for name, parameter in self._parameters.items():
                parameter.copy_(self._entry[name])
        if any(
            int(self._parameters[name].data_ptr()) != self._pointers[name]
            or tensor_sha256(self._parameters[name]) != tensor_sha256(self._entry[name])
            for name in self._parameters
        ):
            raise ODEBFStateError("FP32 J0 W0 restore differs")
        payload = {
            "schema": "ode-edit-s05-p1r52-rr-phase-a-fp32-w0-restore/v1",
            "pointer_identity_preserved": True,
            "byte_restored_exact": True,
            "parameter_sha256": {
                name: tensor_sha256(value) for name, value in sorted(self._parameters.items())
            },
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "ode-edit-s05-p1r52-rr-phase-a-j0-fp32-materializer-accounting/v1",
            "transition_count": len(self.transition_receipts),
            "transition_receipt_sha256": [item["identity_sha256"] for item in self.transition_receipts],
            "numeric_storage_cast_count": 0,
            "bf16_path_call_count": 0,
            "autocast_count": 0,
            "downcast_count": 0,
            "quantization_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


def fp32_weight_energy(
    touched: Mapping[str, torch.nn.Parameter],
    entry_values: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    energy: dict[str, float] = {}
    for name, parameter in sorted(touched.items()):
        delta = parameter.detach() - entry_values[name].to(parameter.device, torch.float32)
        energy[name] = float(torch.sum(delta.double().square()).item())
    norms = {name: math.sqrt(max(value, 0.0)) for name, value in energy.items()}
    norm_total = sum(norms.values())
    energy_total = sum(energy.values())
    return {
        "actual_fp32_update_energy": energy,
        "actual_fp32_update_norm": norms,
        "actual_fp32_norm_share": {
            name: (value / norm_total if norm_total else 0.0) for name, value in norms.items()
        },
        "actual_fp32_squared_energy_share": {
            name: (value / energy_total if energy_total else 0.0) for name, value in energy.items()
        },
        "total_actual_fp32_update_energy": energy_total,
        "numeric_storage_cast_count": 0,
        "bf16_path_call_count": 0,
    }


def run_residual_reserve_phase_a_b10(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    alias: str,
    arm: str,
    destination: Path,
    raw_root: Path,
    stages: Any,
    source_head: str,
    stream_batches: Sequence[Sequence[Mapping[str, Any]]],
    stream: Mapping[str, Any],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    covariance_registry: Any,
    projector_sha256: str,
    controller_lock: Any,
    request_by_sha256: Mapping[str, Mapping[str, Any]],
    population_by_sha256: Mapping[str, Mapping[str, Any]],
    schedule: Any,
    theta0_cache: Any,
    dataset_path: Path,
    mutation_lock: Any,
    touched: Mapping[str, torch.nn.Parameter],
    base_receipt: Any,
    base_values: Mapping[str, torch.Tensor],
    job_ledger: Any,
    request_microbatch_size: int,
    runtime: Any,
    **_: Any,
) -> dict[str, Any]:
    """Run one create-once FP32 B10 cell on the frozen first stream batch."""

    from .alpha_backend import seed_all
    from .contracts import BATCH_SIZE, COMMON_SEED
    from .p1_runtime import _atomic_write_once
    from .p1r36_independent_b10x10_runtime import _run_ode_case
    from .p1_scalable_batched_experiment import _model_w0_contract, _run_native_role

    if (
        arm not in PHASE_A_ARMS
        or len(stream_batches) < 1
        or len(stream_batches[0]) != BATCH_SIZE
        or any(
            parameter.dtype is not torch.float32
            for parameter in model.parameters()
            if parameter.is_floating_point()
        )
        or stream.get("batch_ordered_request_digest_v1") is None
    ):
        raise ODEBFContractError("residual-reserve Phase-A cell differs")
    requests = tuple(stream_batches[0])
    seed_all(COMMON_SEED)
    entry_contract = _model_w0_contract(touched)
    if arm == "official-alphaedit":
        result = _run_native_role(
            model,
            tokenizer,
            alias=alias,
            role="OFFICIAL_NATIVE",
            destination=destination,
            raw_root=raw_root,
            source_head=source_head,
            requests=requests,
            hparams=hparams,
            projector=projector,
            contexts=contexts,
            dataset_path=dataset_path,
            touched=touched,
            base_values=base_values,
            request_microbatch_size=request_microbatch_size,
            mutation_lock=mutation_lock,
            job_ledger=job_ledger,
            write_once=_atomic_write_once,
        )
        stages.record(
            "post_residual_reserve_phase_a_cell",
            {
                "arm": arm,
                "status": result["status"],
                "W0_restored": _model_w0_contract(touched) == entry_contract,
            },
        )
        return result

    covariances = (
        None
        if arm == "j0"
        else load_phase_a_covariances(
            runtime,
            easyedit_root=Path("/mnt/raid5/janghj/EasyEdit"),
        )
    )
    method_by_arm = {
        "j0": "P1R52-RR-FP32-J0",
        "rr-uniform": "P1R52-RR-FP32-UNIFORM",
        "rr-pcsoft": "P1R52-RR-FP32-PCSOFT",
    }
    rr_arm = RR_ARM_BY_TOKEN.get(arm)
    case_root = raw_root / "cases" / "case-01"
    result = _run_ode_case(
        model,
        tokenizer,
        requests,
        alias=alias,
        method=method_by_arm[arm],
        case_index=1,
        case_root=case_root,
        hparams=hparams,
        projector=projector,
        contexts=contexts,
        covariance_registry=covariance_registry,
        projector_sha256=projector_sha256,
        controller_lock=controller_lock,
        request_by_sha256=request_by_sha256,
        population_by_sha256=population_by_sha256,
        schedule=schedule,
        theta0_cache=theta0_cache,
        dataset_path=dataset_path,
        mutation_lock=mutation_lock,
        touched=touched,
        base_receipt=base_receipt,
        base_values=base_values,
        request_microbatch_size=request_microbatch_size,
        job_ledger=job_ledger,
        p1r52=True,
        p1r52_residual_reserve_arm=rr_arm,
        p1r52_phase_a_fp32=True,
        p1r52_rr_covariances=covariances,
    )
    case_terminal_path = case_root / "terminal.json"
    case_manifest_path = case_root / "manifest.json"
    case_terminal = __import__("json").loads(case_terminal_path.read_text())
    case_manifest = __import__("json").loads(case_manifest_path.read_text())
    terminal = {
        "schema": "ode-edit-s05-p1r52-residual-reserve-phase-a-b10-terminal/v1",
        "instruction_id": INSTRUCTION_ID,
        "method_id": METHOD_ID,
        "source_head": source_head,
        "alias": alias,
        "arm": arm,
        "request_count": BATCH_SIZE,
        "stream_root_digest": stream.get("root_digest"),
        "stream_first_batch_order_sha256": stream["batch_ordered_request_digest_v1"][0],
        "case_terminal_sha256": case_terminal["identity_sha256"],
        "case_manifest_sha256": case_manifest["identity_sha256"],
        "status": "TECHNICAL_COMPLETE",
        "live_storage_dtype": "torch.float32",
        "numeric_storage_cast_count": 0,
        "autocast_downcast_quantization_count": 0,
        "W0_restored": _model_w0_contract(touched) == entry_contract,
        "job_compute": job_ledger.raw_free_payload(),
    }
    terminal["identity_sha256"] = canonical_hash(terminal)
    terminal_sha = _atomic_write_once(destination / "terminal.json", terminal)
    manifest = {
        "schema": "ode-edit-s05-p1r52-residual-reserve-phase-a-b10-manifest/v1",
        "source_head": source_head,
        "terminal_sha256": terminal_sha,
        "case_terminal_sha256": case_terminal["identity_sha256"],
        "arm": arm,
        "W0_restored": terminal["W0_restored"],
    }
    manifest["identity_sha256"] = canonical_hash(manifest)
    manifest_sha = _atomic_write_once(destination / "manifest.json", manifest)
    stages.record(
        "post_residual_reserve_phase_a_cell",
        {"arm": arm, "status": terminal["status"], "W0_restored": terminal["W0_restored"]},
    )
    return {
        "status": "P1R52_RESIDUAL_RESERVE_PHASE_A_B10_COMPLETE",
        "terminal_sha256": terminal_sha,
        "manifest_sha256": manifest_sha,
        "W0_restored": terminal["W0_restored"],
    }


__all__ = [
    "AcceptedPhysicalStateFP32Materializer",
    "INSTRUCTION_ID",
    "METHOD_ID",
    "PHASE_A_ARMS",
    "RR_ARM_BY_TOKEN",
    "ResidualReserveOuterExecution",
    "ResidualReservePhaseAWriterRuntime",
    "expected_phase_a_result_name",
    "fp32_weight_energy",
    "load_phase_a_covariances",
    "load_phase_a_fp32_model",
]
