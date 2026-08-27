"""Thin BGODE-R1 binding for fixed-z AlphaEdit dictionaries and serial JVPs."""

from __future__ import annotations

import math
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    LowRankFactor,
    MemitFactorProposal,
    SnapshotManifest,
    canonical_json,
    sha256_bytes,
)
from project.run_scripts.ode_edit_motivation.direct_z import FrozenDirectZ
from project.run_scripts.ode_edit_motivation.frozen_target_lineage import proposal_direction_hash
from project.run_scripts.ode_edit_motivation.hooks import (
    assert_snapshot_current,
    resolve_module,
    resolve_parameter,
    tensor_sha256,
)

from .errors import BGODEScientificBoundary
from .s1_contract import S1Tokenization
from .s1_streaming_events import PrefixDirectionalLogits, StreamingTrieLayout


def _snapshot_hashes(snapshot: SnapshotManifest) -> dict[str, str]:
    return {record.name: record.sha256 for record in snapshot.parameters}


def _same_snapshot(left: SnapshotManifest, right: SnapshotManifest) -> bool:
    return (
        left.snapshot_id == right.snapshot_id
        and left.state_id == right.state_id
        and left.model_id == right.model_id
        and left.context_id == right.context_id
        and left.request_ids == right.request_ids
        and left.hparams_sha256 == right.hparams_sha256
        and _snapshot_hashes(left) == _snapshot_hashes(right)
    )


@dataclass(frozen=True, slots=True)
class BGODETargetHop:
    parent_state_id: str
    child_state_id: str
    action_sha256: str
    coefficients: tuple[float, ...]
    applied_parameter_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class BGODETargetAuthority:
    """Authorize immutable W0 z* at exact observed BGODE W descendants."""

    direct_z_tensor_sha256: str
    direct_z_artifact_sha256: str
    direct_z_artifact_size: int
    target_token_sha256: str
    origin_snapshot: SnapshotManifest
    terminal_snapshot: SnapshotManifest
    hops: tuple[BGODETargetHop, ...] = ()

    @classmethod
    def start(
        cls,
        *,
        direct_z: FrozenDirectZ,
        origin_snapshot: SnapshotManifest,
        target_token_ids: torch.Tensor,
    ) -> "BGODETargetAuthority":
        if (
            direct_z.source_snapshot_id != origin_snapshot.snapshot_id
            or direct_z.source_state_id != origin_snapshot.state_id
            or direct_z.model_id != origin_snapshot.model_id
            or direct_z.context_id != origin_snapshot.context_id
            or direct_z.request_ids != origin_snapshot.request_ids
        ):
            raise BGODEScientificBoundary("fixed z* does not bind the BGODE W0 snapshot")
        result = cls(
            direct_z_tensor_sha256=direct_z.tensor_sha256,
            direct_z_artifact_sha256=direct_z.artifact.sha256,
            direct_z_artifact_size=direct_z.artifact.size,
            target_token_sha256=tensor_sha256(target_token_ids.detach().cpu().contiguous()),
            origin_snapshot=origin_snapshot,
            terminal_snapshot=origin_snapshot,
        )
        result.assert_authorizes(
            direct_z=direct_z,
            snapshot=origin_snapshot,
            target_token_ids=target_token_ids,
        )
        return result

    def assert_authorizes(
        self,
        *,
        direct_z: FrozenDirectZ,
        snapshot: SnapshotManifest,
        target_token_ids: torch.Tensor,
    ) -> None:
        if (
            direct_z.tensor_sha256 != self.direct_z_tensor_sha256
            or direct_z.artifact.sha256 != self.direct_z_artifact_sha256
            or direct_z.artifact.size != self.direct_z_artifact_size
            or tensor_sha256(target_token_ids.detach().cpu().contiguous()) != self.target_token_sha256
            or not _same_snapshot(snapshot, self.terminal_snapshot)
        ):
            raise BGODEScientificBoundary("BGODE fixed-target authority mismatch")

    def derive(
        self,
        *,
        proposal: MemitFactorProposal,
        coefficients: torch.Tensor,
        child_snapshot: SnapshotManifest,
        applied_hashes: Mapping[str, str],
    ) -> "BGODETargetAuthority":
        if not _same_snapshot(proposal.snapshot, self.terminal_snapshot):
            raise BGODEScientificBoundary("BGODE action does not start at authority terminal")
        if (
            coefficients.dtype != torch.float32
            or coefficients.ndim != 1
            or coefficients.numel() != len(proposal.factors)
            or not bool(torch.isfinite(coefficients).all())
        ):
            raise BGODEScientificBoundary("BGODE coefficients are invalid")
        parent_hashes = _snapshot_hashes(self.terminal_snapshot)
        child_hashes = _snapshot_hashes(child_snapshot)
        names = tuple(factor.weight_name for factor in proposal.factors)
        if (
            set(applied_hashes) != set(names)
            or any(applied_hashes[name] != child_hashes[name] for name in names)
            or any(child_hashes[name] != parent_hashes[name] for name in set(parent_hashes) - set(names))
        ):
            raise BGODEScientificBoundary("BGODE physical action receipt is incomplete")
        hop = BGODETargetHop(
            parent_state_id=self.terminal_snapshot.state_id,
            child_state_id=child_snapshot.state_id,
            action_sha256=proposal_direction_hash(proposal),
            coefficients=tuple(float(value) for value in coefficients.detach().cpu()),
            applied_parameter_hashes=tuple(sorted(applied_hashes.items())),
        )
        if hop.parent_state_id == hop.child_state_id:
            raise BGODEScientificBoundary("positive BGODE action did not change W")
        return BGODETargetAuthority(
            direct_z_tensor_sha256=self.direct_z_tensor_sha256,
            direct_z_artifact_sha256=self.direct_z_artifact_sha256,
            direct_z_artifact_size=self.direct_z_artifact_size,
            target_token_sha256=self.target_token_sha256,
            origin_snapshot=self.origin_snapshot,
            terminal_snapshot=child_snapshot,
            hops=(*self.hops, hop),
        )


@dataclass(frozen=True, slots=True)
class PhysicalActionReceipt:
    action_sha256: str
    coefficients: tuple[float, ...]
    parameter_hashes: tuple[tuple[str, str], ...]
    assignment_count: int
    numeric_storage_cast_count: int
    wall_seconds: float


class AtomicWeightTrajectory:
    """One-arm exact-W0 transaction supporting multiple physical Euler writes."""

    def __init__(self, model: torch.nn.Module, weight_names: Sequence[str]) -> None:
        names = tuple(weight_names)
        if not names or len(names) != len(set(names)):
            raise BGODEScientificBoundary("trajectory weight inventory is invalid")
        self.model = model
        self.names = names
        self._parameters = {name: resolve_parameter(model, name) for name in names}
        self._backups = {name: value.detach().clone() for name, value in self._parameters.items()}
        self._pointers = {name: int(value.data_ptr()) for name, value in self._parameters.items()}
        self._entry_hashes = {name: tensor_sha256(value) for name, value in self._parameters.items()}
        self._active = False
        self.receipts: list[PhysicalActionReceipt] = []

    def __enter__(self) -> "AtomicWeightTrajectory":
        if self._active:
            raise BGODEScientificBoundary("trajectory is already active")
        self._active = True
        return self

    @staticmethod
    def _dense_update(factor: LowRankFactor, coefficient: float, parameter: torch.Tensor) -> torch.Tensor:
        left = (factor.left * coefficient).to(device=parameter.device, dtype=torch.float32)
        right = factor.right.to(device=parameter.device, dtype=torch.float32)
        if factor.native_update_transposed:
            raw = right @ left.transpose(0, 1)
            dense = raw.transpose(0, 1)
        else:
            dense = left @ right.transpose(0, 1)
        if tuple(dense.shape) != tuple(parameter.shape):
            raise BGODEScientificBoundary("BGODE update orientation differs from live weight")
        return dense

    def apply(
        self,
        proposal: MemitFactorProposal,
        coefficients: torch.Tensor,
    ) -> PhysicalActionReceipt:
        if not self._active:
            raise BGODEScientificBoundary("trajectory action outside active transaction")
        assert_snapshot_current(self.model, proposal.snapshot)
        if (
            coefficients.dtype != torch.float32
            or coefficients.ndim != 1
            or coefficients.numel() != len(proposal.factors)
            or not bool(torch.isfinite(coefficients).all())
        ):
            raise BGODEScientificBoundary("trajectory coefficients are invalid")
        started = time.perf_counter()
        with torch.no_grad():
            for factor, coefficient in zip(proposal.factors, coefficients, strict=True):
                parameter = self._parameters[factor.weight_name]
                if parameter.dtype != torch.float32:
                    raise BGODEScientificBoundary("BGODE physical weight is not FP32")
                update = self._dense_update(factor, float(coefficient.detach().cpu().item()), parameter)
                parameter[...] += update
        torch.cuda.synchronize(parameter.device)
        hashes = tuple(sorted((factor.weight_name, tensor_sha256(self._parameters[factor.weight_name])) for factor in proposal.factors))
        payload = {
            "proposal": proposal_direction_hash(proposal),
            "coefficients": [float(value) for value in coefficients.detach().cpu()],
            "parameter_hashes": dict(hashes),
        }
        receipt = PhysicalActionReceipt(
            action_sha256=sha256_bytes(canonical_json(payload).encode("utf-8")),
            coefficients=tuple(payload["coefficients"]),
            parameter_hashes=hashes,
            assignment_count=len(proposal.factors),
            numeric_storage_cast_count=0,
            wall_seconds=time.perf_counter() - started,
        )
        self.receipts.append(receipt)
        return receipt

    def restore(self) -> None:
        if not self._active:
            return
        with torch.no_grad():
            for name in self.names:
                parameter = self._parameters[name]
                if int(parameter.data_ptr()) != self._pointers[name]:
                    raise BGODEScientificBoundary("live parameter pointer changed inside trajectory")
                parameter.copy_(self._backups[name])
        for name in self.names:
            if tensor_sha256(self._parameters[name]) != self._entry_hashes[name]:
                raise BGODEScientificBoundary("trajectory W0 byte restore failed")
        self._active = False
        # Backups are device tensors.  Release them as soon as the exact restore
        # is verified so six sequential S1 arms do not retain one W0 copy each.
        self._backups.clear()

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        self.restore()
        return False


@dataclass(frozen=True, slots=True)
class FiniteDifferenceReceipt:
    epsilon: float
    absolute_tolerance: float
    relative_tolerance: float
    direction_count: int
    max_absolute_error: float
    allclose: bool


@dataclass(slots=True)
class JVPComputeLedger:
    model_forward_invocations: int = 0
    primal_prefix_count: int = 0
    jvp_call_count: int = 0
    finite_difference_forward_count: int = 0
    score_buffer_peak_bytes: int = 0
    wall_seconds: float = 0.0


class SerialForwardJVPBackend:
    """Correctness-first low-rank hook JVP, one actuator tangent per call."""

    FD_EPSILON = 1.0 / 256.0
    FD_ABSOLUTE_TOLERANCE = 2.0e-2
    FD_RELATIVE_TOLERANCE = 8.0e-2

    def __init__(self, model: torch.nn.Module, tokenization: S1Tokenization) -> None:
        self.model = model
        self.tokenization = tokenization
        self.ledger = JVPComputeLedger()
        self._fd_receipt: FiniteDifferenceReceipt | None = None

    @property
    def finite_difference_receipt(self) -> FiniteDifferenceReceipt | None:
        return self._fd_receipt

    def _function(self, proposal: MemitFactorProposal, prefix: tuple[int, ...]):
        factors = tuple(proposal.factors)
        device = next(self.model.parameters()).device
        factor_devices = tuple(
            (
                factor.left.to(device=device, dtype=torch.float32),
                factor.right.to(device=device, dtype=torch.float32),
            )
            for factor in factors
        )
        ids = torch.tensor(
            [self.tokenization.prompt_token_ids + prefix], dtype=torch.long, device=device
        )

        def function(beta: torch.Tensor) -> torch.Tensor:
            if beta.dtype != torch.float32 or beta.shape != (len(factors),) or beta.device != device:
                raise BGODEScientificBoundary("JVP beta has wrong dtype/shape/device")
            with ExitStack() as stack:
                for index, factor in enumerate(factors):
                    module = resolve_module(self.model, factor.weight_name.removesuffix(".weight"))
                    left, right = factor_devices[index]

                    def hook(_module: torch.nn.Module, args: tuple[Any, ...], output: Any, *, i=index, l=left, r=right):
                        if not isinstance(output, torch.Tensor) or not args or not isinstance(args[0], torch.Tensor):
                            raise BGODEScientificBoundary("AlphaEdit actuator hook saw unexpected module IO")
                        hidden = args[0]
                        delta = (hidden @ r) @ l.transpose(0, 1)
                        return output + beta[i] * delta

                    stack.callback(module.register_forward_hook(hook).remove)
                self.ledger.model_forward_invocations += 1
                return self.model(input_ids=ids, use_cache=False).logits[0, -1].float()

        return function

    def _directional(self, function: Any, direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros_like(direction)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(zero, direction)
            output = function(dual)
            primal, tangent = torch.autograd.forward_ad.unpack_dual(output)
        if tangent is None:
            raise BGODEScientificBoundary("forward-mode JVP returned no tangent")
        self.ledger.jvp_call_count += 1
        return primal.detach(), tangent.detach()

    def _validate_fd(
        self,
        function: Any,
        primal: torch.Tensor,
        tangents: torch.Tensor,
    ) -> None:
        if self._fd_receipt is not None:
            return
        logp = torch.log_softmax(primal, dim=0)
        jvp_scores = tangents - (logp.exp()[:, None] * tangents).sum(dim=0, keepdim=True)
        observed: list[torch.Tensor] = []
        for index in range(tangents.shape[1]):
            direction = torch.zeros(tangents.shape[1], dtype=torch.float32, device=primal.device)
            direction[index] = self.FD_EPSILON
            plus = torch.log_softmax(function(direction), dim=0)
            minus = torch.log_softmax(function(-direction), dim=0)
            self.ledger.finite_difference_forward_count += 2
            observed.append((plus - minus) / (2.0 * self.FD_EPSILON))
        fd = torch.stack(observed, dim=1)
        allclose = bool(torch.allclose(
            jvp_scores,
            fd,
            atol=self.FD_ABSOLUTE_TOLERANCE,
            rtol=self.FD_RELATIVE_TOLERANCE,
        ))
        error = float(torch.max(torch.abs(jvp_scores - fd)).detach().cpu().item())
        receipt = FiniteDifferenceReceipt(
            epsilon=self.FD_EPSILON,
            absolute_tolerance=self.FD_ABSOLUTE_TOLERANCE,
            relative_tolerance=self.FD_RELATIVE_TOLERANCE,
            direction_count=tangents.shape[1],
            max_absolute_error=error,
            allclose=allclose,
        )
        self._fd_receipt = receipt
        if not allclose:
            raise BGODEScientificBoundary("first-node serial JVP/central-FD identity gate failed")

    def observe(
        self,
        *,
        layout: StreamingTrieLayout,
        proposal: MemitFactorProposal,
        validate_first_node_fd: bool,
    ) -> dict[tuple[int, ...], PrefixDirectionalLogits]:
        started = time.perf_counter()
        result: dict[tuple[int, ...], PrefixDirectionalLogits] = {}
        direction_count = len(proposal.factors)
        for node_index, prefix in enumerate(layout.internal_prefixes):
            function = self._function(proposal, prefix)
            primals: list[torch.Tensor] = []
            tangents: list[torch.Tensor] = []
            for index in range(direction_count):
                direction = torch.zeros(direction_count, dtype=torch.float32, device=next(self.model.parameters()).device)
                direction[index] = 1.0
                primal, tangent = self._directional(function, direction)
                primals.append(primal)
                tangents.append(tangent)
            primal = primals[0]
            if any(not torch.equal(primal, candidate) for candidate in primals[1:]):
                raise BGODEScientificBoundary("serial JVP primal bytes differ by direction")
            tangent_matrix = torch.stack(tangents, dim=1).to(dtype=torch.float32)
            self.ledger.primal_prefix_count += 1
            self.ledger.score_buffer_peak_bytes = max(
                self.ledger.score_buffer_peak_bytes,
                int(tangent_matrix.numel() * tangent_matrix.element_size()),
            )
            if validate_first_node_fd and node_index == 0 and self._fd_receipt is None:
                self._validate_fd(function, primal, tangent_matrix)
            result[prefix] = PrefixDirectionalLogits(
                logits=primal.detach().cpu().contiguous(),
                tangent_logits=tangent_matrix.detach().cpu().contiguous(),
            )
        torch.cuda.synchronize(next(self.model.parameters()).device)
        self.ledger.wall_seconds += time.perf_counter() - started
        return result

    def observe_primal(self, layout: StreamingTrieLayout) -> dict[tuple[int, ...], PrefixDirectionalLogits]:
        started = time.perf_counter()
        device = next(self.model.parameters()).device
        result: dict[tuple[int, ...], PrefixDirectionalLogits] = {}
        with torch.inference_mode():
            for prefix in layout.internal_prefixes:
                ids = torch.tensor([self.tokenization.prompt_token_ids + prefix], dtype=torch.long, device=device)
                self.ledger.model_forward_invocations += 1
                self.ledger.primal_prefix_count += 1
                logits = self.model(input_ids=ids, use_cache=False).logits[0, -1].float()
                result[prefix] = PrefixDirectionalLogits(logits=logits.detach().cpu().contiguous())
        torch.cuda.synchronize(device)
        self.ledger.wall_seconds += time.perf_counter() - started
        return result


__all__ = [
    "AtomicWeightTrajectory",
    "BGODETargetAuthority",
    "FiniteDifferenceReceipt",
    "JVPComputeLedger",
    "PhysicalActionReceipt",
    "SerialForwardJVPBackend",
]
