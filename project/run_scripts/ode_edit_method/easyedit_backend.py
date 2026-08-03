"""Concrete, read-only EasyEdit/MEMIT backend for Session 02.

The backend uses one code path for both fixed model aliases.  EasyEdit source
and Wikipedia moments are provenance checked and only read.  A frozen direct-z
target is computed once at the outer-edit entry and reused byte-for-byte while
ODE-Edit relinearizes proposals at descendants owned by this transaction.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from project.run_scripts.ode_edit_motivation import easyedit_bridge as bridge_module
from project.run_scripts.ode_edit_motivation.contracts import (
    EditRequest,
    MemitFactorProposal,
    ProposalSemantics as MotivationProposalSemantics,
    ProvenanceManifest,
    SnapshotManifest,
    orient_easyedit_factor,
)
from project.run_scripts.ode_edit_motivation.direct_z import FrozenDirectZ
from project.run_scripts.ode_edit_motivation.easyedit_bridge import (
    CovarianceCacheSpec,
    EasyEditBridge,
    _EASYEDIT_GLOBAL_LOCK,
    _preserve_model_runtime_state,
)
from project.run_scripts.ode_edit_motivation.gpu_runtime import FixedModelRuntime
from project.run_scripts.ode_edit_motivation.hooks import (
    capture_snapshot,
    resolve_parameter,
    tensor_sha256,
)
from project.run_scripts.ode_edit_motivation.mv1_calibration import proposal_c_energy

from .contracts import (
    EventReading,
    MethodContractError,
    ProposalBatch,
)
from .derivatives import (
    ActuatorDirectionalHook,
    ScalarGateDirectionalReference,
    assert_scalar_gate_matches_hook,
)
from .dense_memit import TransientDenseMemitSolver
from .events import ControllerRequest, measure_differentiable_event, measure_event
from .functional_trial import QuantizedFullLinearFunctionalTrial
from .hooks import TorchCheckpoint, terminal_net_c_energy
from .instrumentation import EditInstrumentation
from .memit_adapter import (
    commit_batch,
    coordinate_batch,
    functional_trial_for_batch,
    native_terminal_batch,
    synchronous_unit_batch,
)
from .mechanism import capture_field_mechanism
from .preflight import CovarianceRuntimeContract


@dataclass(frozen=True, slots=True)
class EasyEditBackendCheckpoint:
    weights: TorchCheckpoint
    snapshot: SnapshotManifest

    @property
    def state_id(self) -> str:
        return self.snapshot.state_id


@dataclass(frozen=True, slots=True)
class ConcreteBackendPreflight:
    fixed_artifact_manifest_id: str
    easyedit_source_manifest_id: str
    context_manifest_id: str
    tokenization_manifest: Mapping[str, Any]
    model_alias: str
    model_revision: str
    hparams_relative_path: str
    covariance_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixed_artifact_manifest_id": self.fixed_artifact_manifest_id,
            "easyedit_source_manifest_id": self.easyedit_source_manifest_id,
            "context_manifest_id": self.context_manifest_id,
            "tokenization_manifest": dict(self.tokenization_manifest),
            "model_alias": self.model_alias,
            "model_revision": self.model_revision,
            "hparams_relative_path": self.hparams_relative_path,
            "covariance_paths": list(self.covariance_paths),
        }


class EasyEditMemitBackend:
    """Concrete :class:`MethodBackend` using pinned MEMIT primitives."""

    def __init__(
        self,
        *,
        runtime: FixedModelRuntime,
        bridge: EasyEditBridge,
        hparams: Any,
        contexts: Any,
        request: ControllerRequest,
        covariance_specs: Sequence[CovarianceCacheSpec],
        covariance_contract: CovarianceRuntimeContract,
        covariance_by_layer: Mapping[int, torch.Tensor],
        direct_z_cache_root: str | Path,
        direct_z_cache_path: str | Path,
        tau: float,
        unit_c_norm_epsilon: float,
        unit_c_identity_atol: float,
        unit_c_identity_rtol: float,
        instrumentation: EditInstrumentation,
        finite_difference_gate_epsilon: float | None = None,
        finite_difference_gate_atol: float = 0.0,
        finite_difference_gate_rtol: float = 0.0,
        record_mechanism: bool = False,
    ) -> None:
        if runtime.spec.alias not in {"llama3-8b-inst", "qwen2.5-7b-inst"}:
            raise MethodContractError("concrete backend received a non-canonical model")
        self.runtime = runtime
        self.model = runtime.model
        self.tokenizer = runtime.tokenizer
        self.bridge = bridge
        self.bindings = bridge.load()
        self.hparams = hparams
        self.contexts = contexts
        self.request = request
        self.motivation_request = EditRequest.from_mapping(
            {
                "case_id": request.case_id,
                "prompt": request.prompt,
                "subject": request.subject,
                "target_new": request.target_new,
            }
        )
        self.covariance_specs = tuple(covariance_specs)
        self.covariance_contract = covariance_contract
        self.covariance_by_layer = {
            int(layer): value for layer, value in covariance_by_layer.items()
        }
        self._covariance_tensor_guards = {
            layer: (
                value.data_ptr(),
                value._version,
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
            )
            for layer, value in self.covariance_by_layer.items()
        }
        self.direct_z_cache_root = Path(direct_z_cache_root).resolve()
        self.direct_z_cache_path = Path(direct_z_cache_path).resolve()
        self.tau = float(tau)
        self.unit_c_norm_epsilon = float(unit_c_norm_epsilon)
        self.unit_c_identity_atol = float(unit_c_identity_atol)
        self.unit_c_identity_rtol = float(unit_c_identity_rtol)
        self.instrumentation = instrumentation
        self.finite_difference_gate_epsilon = (
            None
            if finite_difference_gate_epsilon is None
            else float(finite_difference_gate_epsilon)
        )
        self.finite_difference_gate_atol = float(finite_difference_gate_atol)
        self.finite_difference_gate_rtol = float(finite_difference_gate_rtol)
        if not isinstance(record_mechanism, bool):
            raise MethodContractError("mechanism recording flag must be boolean")
        self.record_mechanism = record_mechanism
        if self.finite_difference_gate_epsilon is not None and (
            not math.isfinite(self.finite_difference_gate_epsilon)
            or self.finite_difference_gate_epsilon <= 0.0
            or self.finite_difference_gate_atol <= 0.0
            or self.finite_difference_gate_rtol <= 0.0
        ):
            raise MethodContractError("P0 finite-difference gate constants are invalid")
        self._layers = tuple(int(layer) for layer in hparams.layers)
        if self._layers != tuple(runtime.spec.layers):
            raise MethodContractError("MEMIT backend layer set differs from fixed model spec")
        self.weight_by_layer = {
            layer: f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            for layer in self._layers
        }
        self.layer_by_weight = {
            name: layer for layer, name in self.weight_by_layer.items()
        }
        if set(self.covariance_by_layer) != set(self._layers):
            raise MethodContractError("backend covariance mapping differs from edit layers")
        if set(self.covariance_contract.cache_path_by_layer) != set(self._layers):
            raise MethodContractError("backend covariance runtime contract differs")
        self._direct_z: FrozenDirectZ | None = None
        self._direct_z_tensor_sha256: str | None = None
        self._entry_snapshot: SnapshotManifest | None = None
        self._current_snapshot = self._capture_snapshot()
        self._parameter_guard = self._capture_parameter_guard()
        self._authorized_states = {self._current_snapshot.state_id}
        self._entry_raw_synchronous: MemitFactorProposal | None = None
        self._entry_synchronous_distance: float | None = None
        self._native_distance: float | None = None
        self._event_history: list[dict[str, Any]] = []
        self._hook_reference_gate: tuple[Mapping[str, Any], ...] | None = None
        self._mechanism_field_history: list[dict[str, Any]] = []
        self._mechanism_previous_directions: Mapping[int, str] | None = None

    def _covariance_source_for_solve(self, layer: int) -> torch.Tensor:
        """Return the guarded CPU source; the solver owns the device copy."""

        try:
            source = self.covariance_by_layer[layer]
            pointer, version, shape, dtype, source_device = (
                self._covariance_tensor_guards[layer]
            )
        except KeyError as exc:
            raise MethodContractError("solve covariance layer is absent") from exc
        if (
            source.data_ptr() != pointer
            or source._version != version
            or tuple(source.shape) != shape
            or str(source.dtype) != dtype
            or str(source.device) != source_device
        ):
            raise MethodContractError("verified covariance tensor changed in memory")
        if source.device.type != "cpu":
            raise MethodContractError("verified covariance source must remain on CPU")
        return source

    @property
    def layers(self) -> tuple[int, ...]:
        return self._layers

    @property
    def native_distance(self) -> float | None:
        return self._native_distance

    @property
    def direct_z_identity(self) -> Mapping[str, Any] | None:
        if self._direct_z is None:
            return None
        return {
            "tensor_sha256": self._direct_z.tensor_sha256,
            "artifact_sha256": self._direct_z.artifact.sha256,
            "artifact_size": self._direct_z.artifact.size,
            "source_state_id": self._direct_z.source_state_id,
        }

    @property
    def event_history(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(value) for value in self._event_history)

    @property
    def hook_reference_gate(self) -> tuple[Mapping[str, Any], ...] | None:
        return self._hook_reference_gate

    @property
    def mechanism_field_history(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(value) for value in self._mechanism_field_history)

    def _record_event(self, source: str, reading: EventReading) -> None:
        self._event_history.append(
            {
                "source": source,
                "hard_phi": reading.hard_phi,
                "smooth_phi": reading.smooth_phi,
                "context_margins": list(reading.context_margins),
                "nfe": reading.nfe,
            }
        )

    def _capture_snapshot(self) -> SnapshotManifest:
        return capture_snapshot(
            self.model,
            model_id=self.runtime.spec.snapshot_name,
            requests=(self.motivation_request,),
            context_id=self.contexts.manifest_id,
            hparams=self.hparams,
            weight_names=tuple(self.weight_by_layer.values()),
            provenance_ids=(self.bindings.provenance.manifest_id,),
        )

    def _capture_parameter_guard(
        self,
    ) -> tuple[tuple[str, int, int, tuple[int, ...], str, str], ...]:
        """Capture a copy-free mutation guard for the cached exact snapshot.

        Exact dense hashes are produced when a proposal snapshot is built and
        after an accepted commit.  Re-hashing every functional trial would
        itself copy every target weight to CPU, so unchanged-state checks use
        storage/version/shape/dtype/device guards between those exact rebuilds.
        """

        rows = []
        for name in sorted(self.layer_by_weight):
            parameter = resolve_parameter(self.model, name)
            rows.append(
                (
                    name,
                    parameter.data_ptr(),
                    parameter._version,
                    tuple(parameter.shape),
                    str(parameter.dtype),
                    str(parameter.device),
                )
            )
        return tuple(rows)

    def _assert_parameter_guard(self) -> None:
        for name, pointer, version, shape, dtype, device in self._parameter_guard:
            parameter = resolve_parameter(self.model, name)
            if (
                parameter.data_ptr() != pointer
                or parameter._version != version
                or tuple(parameter.shape) != shape
                or str(parameter.dtype) != dtype
                or str(parameter.device) != device
                or parameter.grad is not None
            ):
                raise MethodContractError(
                    "target weights changed outside the guarded backend transaction"
                )

    def current_state_id(self) -> str:
        self._assert_parameter_guard()
        return self._current_snapshot.state_id

    def checkpoint(self) -> EasyEditBackendCheckpoint:
        self._assert_parameter_guard()
        snapshot = self._current_snapshot
        checkpoint = EasyEditBackendCheckpoint(
            weights=TorchCheckpoint.capture(
                self.model, tuple(self.weight_by_layer.values())
            ),
            snapshot=snapshot,
        )
        if self._entry_snapshot is None:
            self._entry_snapshot = snapshot
        elif self._entry_snapshot.state_id != snapshot.state_id:
            raise MethodContractError("backend outer edit did not start at its entry state")
        return checkpoint

    def restore(self, checkpoint: EasyEditBackendCheckpoint) -> None:
        if not isinstance(checkpoint, EasyEditBackendCheckpoint):
            raise MethodContractError("backend restore checkpoint type differs")
        checkpoint.weights.restore(self.model)
        self._current_snapshot = checkpoint.snapshot
        self._parameter_guard = self._capture_parameter_guard()
        self._authorized_states = {checkpoint.snapshot.state_id}

    def assert_checkpoint(self, checkpoint: EasyEditBackendCheckpoint) -> None:
        checkpoint.weights.assert_exact(self.model, include_rng=True)
        if self.current_state_id() != checkpoint.snapshot.state_id:
            raise MethodContractError("backend differs from guarded checkpoint")

    def compute_direct_z(self, request: ControllerRequest) -> FrozenDirectZ:
        if request != self.request:
            raise MethodContractError("direct-z request differs from backend edit")
        if self._direct_z is not None:
            raise MethodContractError("direct-z backend was invoked more than once")
        result = self.bridge.load_or_compute_direct_z(
            self.model,
            self.tokenizer,
            (self.motivation_request,),
            self.hparams,
            self.contexts,
            model_id=self.runtime.spec.snapshot_name,
            local_cache_root=self.direct_z_cache_root,
            cache_path=self.direct_z_cache_path,
            expected_identity=None,
        )
        if result.source_state_id != self.current_state_id():
            raise MethodContractError("direct-z origin differs from edit entry")
        self._direct_z = result
        self._direct_z_tensor_sha256 = tensor_sha256(result.values)
        return result

    def _assert_frozen_target(self, target: Any) -> FrozenDirectZ:
        if target is not self._direct_z or not isinstance(target, FrozenDirectZ):
            raise MethodContractError("backend proposal did not receive its frozen direct-z")
        if tensor_sha256(target.values) != self._direct_z_tensor_sha256:
            raise MethodContractError("frozen direct-z bytes changed")
        ProvenanceManifest(label="frozen direct-z", files=(target.artifact,)).assert_current()
        if self.current_state_id() not in self._authorized_states:
            raise MethodContractError("proposal state is outside the outer transaction")
        return target

    def event(self, request: ControllerRequest) -> EventReading:
        if request != self.request:
            raise MethodContractError("event request differs from backend edit")
        reading = measure_event(
            self.model,
            self.tokenizer,
            request,
            self.contexts.templates,
            tau=self.tau,
        )
        self._record_event("event", reading)
        return reading

    @contextmanager
    def _reuse_verified_covariance_preflight(self) -> Iterator[None]:
        """Reuse setup-verified stat/sample guards instead of full rehashes."""

        contract = self.covariance_contract

        def cached_preflight(
            current_model: torch.nn.Module,
            current_hparams: Any,
            layers: Sequence[int],
            covariance_caches: Sequence[CovarianceCacheSpec | Mapping[str, Any]],
        ) -> tuple[CovarianceRuntimeContract, dict[int, str], tuple[str, ...]]:
            observed_specs = tuple(
                CovarianceCacheSpec.from_value(value) for value in covariance_caches
            )
            if (
                current_model is not self.model
                or current_hparams is not self.hparams
                or tuple(int(layer) for layer in layers) != self._layers
                or observed_specs != self.covariance_specs
            ):
                raise MethodContractError(
                    "EasyEdit requested covariance outside the verified contract"
                )
            contract.assert_current()
            return (
                contract,
                contract.cache_path_by_layer,
                contract.layer_names,
            )

        with _EASYEDIT_GLOBAL_LOCK:
            original = bridge_module._preflight_covariance_caches
            bridge_module._preflight_covariance_caches = cached_preflight
            try:
                yield
            finally:
                bridge_module._preflight_covariance_caches = original

    def build_native_terminal(self, frozen_target: Any) -> ProposalBatch:
        target = self._assert_frozen_target(frozen_target)
        if self._entry_snapshot is None or self.current_state_id() != self._entry_snapshot.state_id:
            raise MethodContractError("native terminal proposal is entry-state only")
        with self._reuse_verified_covariance_preflight():
            proposal = self.bridge.propose_ordered_memit_factors(
                self.model,
                self.tokenizer,
                (self.motivation_request,),
                self.hparams,
                self.contexts,
                model_id=self.runtime.spec.snapshot_name,
                direct_z=target,
                covariance_caches=self.covariance_specs,
            )
        # The ordered EasyEdit reference temporarily writes and exactly
        # restores target weights while measuring later layers.  Its bridge
        # verifies dense byte identity; refresh only the copy-free version
        # guard that the temporary restore legitimately advances.
        self._parameter_guard = self._capture_parameter_guard()
        self._native_distance = math.sqrt(
            proposal_c_energy(
                proposal, self.covariance_by_layer, self.layer_by_weight
            )
        )
        return native_terminal_batch(proposal, self.layer_by_weight)

    def _descendant_synchronous_proposal(
        self, target: FrozenDirectZ
    ) -> MemitFactorProposal:
        """Build a same-snapshot field at any backend-authorized descendant.

        The public motivation bridge intentionally caps its historical
        diagnostic lineage.  Session 02 owns a distinct S_max=6 transaction,
        so this ODE-side hook repeats the same pinned read-only MEMIT primitive
        calls while authorizing descendants only through exact committed state
        hashes maintained by this backend.
        """

        layers = self._layers
        covariance_manifest = self.covariance_contract
        covariance_manifest.assert_current()
        self._assert_parameter_guard()
        base_snapshot = self._current_snapshot
        if base_snapshot.state_id not in self._authorized_states:
            raise MethodContractError("synchronous refresh state is not authorized")
        snapshot = SnapshotManifest(
            model_id=base_snapshot.model_id,
            context_id=base_snapshot.context_id,
            request_ids=base_snapshot.request_ids,
            hparams_sha256=base_snapshot.hparams_sha256,
            parameters=base_snapshot.parameters,
            provenance_ids=tuple(
                sorted(
                    {
                        self.bindings.provenance.manifest_id,
                        covariance_manifest.manifest_id,
                        target.artifact_id,
                    }
                )
            ),
        )
        raw_request = self.motivation_request.to_easyedit()
        denominator = len(layers)
        solver = TransientDenseMemitSolver()
        factors = []
        with (
            torch.no_grad(),
            _EASYEDIT_GLOBAL_LOCK,
            _preserve_model_runtime_state(self.model),
        ):
            self._assert_parameter_guard()
            current_z = self.bindings.compute_z.get_module_input_output_at_words(
                self.model,
                self.tokenizer,
                layers[-1],
                context_templates=[raw_request["prompt"]],
                words=[raw_request["subject"]],
                module_template=self.hparams.layer_module_tmp,
                fact_token_strategy=self.hparams.fact_token,
                track="out",
            ).T
            target_on_device = target.values.to(device=current_z.device)
            common_targets = target_on_device - current_z.to(
                dtype=target_on_device.dtype
            )
            for layer in layers:
                keys = self.bindings.compute_ks.compute_ks(
                    self.model,
                    self.tokenizer,
                    [raw_request],
                    self.hparams,
                    layer,
                    self.contexts.to_easyedit(),
                ).T
                if keys.shape[1] % common_targets.shape[1] != 0:
                    raise MethodContractError(
                        "MEMIT key panel is not divisible by request count"
                    )
                targets = common_targets.repeat_interleave(
                    keys.shape[1] // common_targets.shape[1], dim=1
                )
                # The moments were full-hash verified and loaded once during
                # setup.  Reusing this guarded tensor avoids forcing EasyEdit
                # to decompress the same multi-GiB files on every refresh.
                covariance = self._covariance_source_for_solve(layer)
                adjusted_keys = solver.adjusted_keys(
                    covariance,
                    keys.double(),
                    self.hparams.mom2_update_weight,
                )
                residuals = targets.double() / denominator
                name = self.weight_by_layer[layer]
                record = snapshot.parameter(name)
                factors.append(
                    orient_easyedit_factor(
                        adjusted_keys,
                        residuals,
                        weight_name=name,
                        weight_shape=record.shape,
                        expected_weight_sha256=record.sha256,
                    )
                )
            self._assert_parameter_guard()
        covariance_manifest.assert_current()
        self.bindings.provenance.assert_current()
        return MemitFactorProposal(
            snapshot=snapshot,
            factors=tuple(factors),
            semantics=MotivationProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="ode-edit/read-only-synchronous-transient-dense-memit",
            residual_denominator=denominator,
        )

    def _raw_synchronous(self, target: FrozenDirectZ) -> MemitFactorProposal:
        if self._entry_snapshot is None:
            raise MethodContractError("synchronous proposal precedes entry checkpoint")
        if self.current_state_id() == self._entry_snapshot.state_id:
            proposal = self._descendant_synchronous_proposal(target)
            if self._entry_raw_synchronous is None:
                self._entry_raw_synchronous = proposal
            return proposal
        return self._descendant_synchronous_proposal(target)

    def build_synchronous(self, frozen_target: Any) -> ProposalBatch:
        target = self._assert_frozen_target(frozen_target)
        with (
            self.instrumentation.component("proposal"),
            self.instrumentation.model_forward_scope("field"),
        ):
            raw = self._raw_synchronous(target)
            zero_slopes = {layer: 0.0 for layer in self._layers}
            batch = synchronous_unit_batch(
                raw,
                self.covariance_by_layer,
                self.layer_by_weight,
                zero_slopes,
                unit_c_norm_epsilon=self.unit_c_norm_epsilon,
                unit_c_identity_atol=self.unit_c_identity_atol,
                unit_c_identity_rtol=self.unit_c_identity_rtol,
            )
        directions = tuple(item.payload for item in batch.proposals)
        with ActuatorDirectionalHook(
            self.model, directions, instrumentation=self.instrumentation
        ) as hook:
            with (
                self.instrumentation.component("field"),
                self.instrumentation.model_forward_scope("field"),
            ):
                event = measure_differentiable_event(
                    self.model,
                    self.tokenizer,
                    self.request,
                    self.contexts.templates,
                    tau=self.tau,
                )
            field = hook.compute(event.smooth_phi)
        self._record_event("field", event.reading)
        if (
            self.finite_difference_gate_epsilon is not None
            and self._hook_reference_gate is None
        ):
            with ScalarGateDirectionalReference(
                self.model,
                directions,
                instrumentation=self.instrumentation,
            ) as reference:
                with (
                    self.instrumentation.component("reference_gate"),
                    self.instrumentation.model_forward_scope("reference_gate"),
                ):
                    reference_event = measure_differentiable_event(
                        self.model,
                        self.tokenizer,
                        self.request,
                        self.contexts.templates,
                        tau=self.tau,
                    )
                reference_field = reference.compute(reference_event.smooth_phi)
            rows = assert_scalar_gate_matches_hook(
                field,
                reference_field,
                abs_tol=self.finite_difference_gate_atol,
                rel_tol=self.finite_difference_gate_rtol,
            )
            self._hook_reference_gate = tuple(
                {
                    **row,
                    "hard_reference": "scalar-gate-functional-graph",
                    "legacy_one_sided_fd_epsilon": (
                        self.finite_difference_gate_epsilon
                    ),
                    "legacy_one_sided_fd_role": "diagnostic-disabled",
                }
                for row in rows
            )
        slopes = tuple(field.slopes_by_layer[layer] for layer in batch.layers)
        result = replace(batch, slopes=slopes)
        if self.record_mechanism:
            # Only cheap scalar/identity metadata is captured in the method
            # timer.  Exact C contractions are deliberately deferred so P1
            # controller timing and peak memory remain authoritative.
            record, directions_by_layer = capture_field_mechanism(
                result,
                self._mechanism_previous_directions,
                normalization_epsilon=self.unit_c_norm_epsilon,
            )
            record["field_index"] = len(self._mechanism_field_history)
            self._mechanism_field_history.append(record)
            self._mechanism_previous_directions = directions_by_layer
        return result

    def build_coordinate(self, frozen_target: Any, layer: int) -> ProposalBatch:
        return coordinate_batch(self.build_synchronous(frozen_target), layer)

    def entry_synchronous_distance(self) -> float:
        if self._entry_raw_synchronous is None:
            raise MethodContractError("entry trust scale requested before entry field")
        if self._entry_synchronous_distance is None:
            self._entry_synchronous_distance = math.sqrt(
                proposal_c_energy(
                    self._entry_raw_synchronous,
                    self.covariance_by_layer,
                    self.layer_by_weight,
                )
            )
        return self._entry_synchronous_distance

    def rebind_frozen(self, entry_batch: ProposalBatch) -> ProposalBatch:
        return entry_batch.rebind_frozen(self.current_state_id())

    def trial(
        self, batch: ProposalBatch, coefficients: Sequence[float]
    ) -> QuantizedFullLinearFunctionalTrial:
        if batch.snapshot_id != self.current_state_id():
            raise MethodContractError("functional trial proposal is stale")
        return functional_trial_for_batch(self.model, batch, coefficients)

    def commit(
        self, batch: ProposalBatch, coefficients: Sequence[float]
    ) -> tuple[float, ...]:
        parent = self.current_state_id()
        if batch.snapshot_id != parent:
            raise MethodContractError("accepted proposal is stale")
        applied = commit_batch(self.model, batch, coefficients)
        child = self._capture_snapshot()
        if child.state_id == parent:
            raise MethodContractError("positive accepted write preserved state identity")
        self._current_snapshot = child
        self._parameter_guard = self._capture_parameter_guard()
        self._authorized_states.add(child.state_id)
        return applied

    def terminal_net_energy(
        self, entry_checkpoint: EasyEditBackendCheckpoint
    ) -> Mapping[int, float]:
        if not isinstance(entry_checkpoint, EasyEditBackendCheckpoint):
            raise MethodContractError("terminal geometry checkpoint type differs")
        return terminal_net_c_energy(
            self.model,
            entry_checkpoint.weights,
            self.covariance_by_layer,
            self.weight_by_layer,
        )
