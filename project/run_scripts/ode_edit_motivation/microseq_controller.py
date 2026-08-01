"""Controller-only action generation for the four-edit Motivation diagnostic.

This module intentionally has no CounterFact evaluation-row loader.  It emits
four committed edit actions for one model/branch, persists proposal factors
under ignored ``local/``, and exits.  A separate process may evaluate replayed
checkpoints only after both branch controllers are terminal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .contracts import (
    ContractError,
    EditRequest,
    MemitFactorProposal,
    SnapshotManifest,
    canonical_json,
    sha256_bytes,
)
from .easyedit_bridge import CovarianceCacheSpec, EasyEditBridge
from .frozen_target_lineage import FrozenTargetLineage, proposal_direction_hash
from .gpu_runtime import (
    FixedModelRuntime,
    load_fixed_model,
    offline_environment,
    rng_state_hash,
    seed_runtime,
)
from .hooks import (
    TemporaryExactMemitApplication,
    assert_snapshot_current,
    assert_tensor_sha256_device_parity,
)
from .manifests import (
    DEFAULT_SELECTION_SEED,
    MODEL_SPECS,
    CounterFactSelectionManifest,
    fixed_model_spec,
    generate_counterfact_selection,
    load_counterfact_requests,
    preflight_fixed_artifacts,
    scan_counterfact_case_ids,
)
from .microseq_analysis import BRANCH_NATIVE, BRANCH_ODE, BRANCHES, CHAIN_LENGTH
from .microseq_artifacts import (
    apply_proposal_in_disposable_process,
    write_proposal_artifact,
)
from .mv0_fidelity import (
    DEFAULT_OUTPUT_ROOT,
    SanitizedJsonlWriter,
    _bridge_pins,
    _covariance_specs,
    _event_seed,
    _file_sha256,
    _freeze_contexts,
    _git_runtime_state,
    _load_hparams,
    _load_verified_covariances,
    _local_run_directory,
    _relative_provenance,
    _safe_payload,
    _silence_upstream,
    _weight_hashes,
    _write_json_exclusive,
)
from .mv1_calibration import (
    MV1Error,
    _feature_hash,
    _layer_by_weight,
    _state_identity,
    assert_exact_action_contract,
    build_unit_c_actions,
    proposal_c_energy,
    scale_proposal,
    teacher_forced_rewrite_exact,
)
from .quarter_step_refresh import (
    LAYERS,
    PROBE_ACTIONS,
    QSTEP_HOP_FRACTION,
    QSTEP_K,
    QSTEP_PROBE_FRACTION,
    REFRESHED_PREFIX,
    _assert_step_energy,
    _build_policy_path,
    _build_score_mix_action,
    _capture_source_snapshot,
    _commit_action,
    _proposal_panel,
)
from .trajectory import MATCHED_C_ABS_TOL, MATCHED_C_REL_TOL


MICROSEQ_SCHEMA = "ode-edit-microseq-controller/v1"
MICROSEQ_STREAM_SCHEMA = "ode-edit-microseq-controller-stream/v1"
MICROSEQ_RECEIPT_SCHEMA = "ode-edit-microseq-controller-receipt/v1"
MICROSEQ_RUN_SEED = 37
MICROSEQ_RANK_START = 140
MICROSEQ_RANK_STOP = 144
MICROSEQ_POLICY_ID = "always-refresh-k4-d4-scoremix-v1"
MICROSEQ_JOB_NAME = "odeedit_microseq_pair_v1"
APPLICATION_MODES = {
    BRANCH_NATIVE: "easyedit_exact",
    BRANCH_ODE: "low_rank_addmm",
}
RUN_IDS = {
    BRANCH_NATIVE: {
        "llama3-8b-inst": "microseq_native_llama_m0_v1",
        "qwen2.5-7b-inst": "microseq_native_qwen_m0_v1",
    },
    BRANCH_ODE: {
        "llama3-8b-inst": "microseq_ode_llama_m0_v1",
        "qwen2.5-7b-inst": "microseq_ode_qwen_m0_v1",
    },
}


@dataclass(frozen=True, slots=True)
class MicroseqSelection:
    source_sha256: str
    source_size: int
    source_row_count: int
    case_ids: tuple[str, ...]
    prior_order_hash: str

    def __post_init__(self) -> None:
        if (
            len(self.case_ids) != CHAIN_LENGTH
            or len(set(self.case_ids)) != CHAIN_LENGTH
            or self.source_row_count < MICROSEQ_RANK_STOP
            or self.source_size <= 0
        ):
            raise ContractError("microseq selection metadata is invalid")

    @property
    def order_hash(self) -> str:
        return sha256_bytes(canonical_json(list(self.case_ids)).encode("utf-8"))

    @property
    def manifest_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict(False)).encode("utf-8"))

    def to_dict(self, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": "ode-edit-microseq-selection/v1",
            "seed": DEFAULT_SELECTION_SEED,
            "rank_slice": {
                "start": MICROSEQ_RANK_START,
                "stop": MICROSEQ_RANK_STOP,
                "count": CHAIN_LENGTH,
            },
            "source": {
                "sha256": self.source_sha256,
                "size": self.source_size,
                "row_count": self.source_row_count,
            },
            "case_ids": list(self.case_ids),
            "order_hash": self.order_hash,
            "prior_order_hash": self.prior_order_hash,
        }
        if include_id:
            payload["manifest_id"] = self.manifest_id
        return payload


def _salted_rank(case_id: str) -> tuple[bytes, str]:
    return (
        hashlib.sha256(
            DEFAULT_SELECTION_SEED.encode("utf-8")
            + b"\0"
            + str(case_id).encode("utf-8")
        ).digest(),
        str(case_id),
    )


def select_microseq_cases(
    all_case_ids: Sequence[str], *, canonical: CounterFactSelectionManifest
) -> MicroseqSelection:
    ranked = tuple(sorted((str(case_id) for case_id in all_case_ids), key=_salted_rank))
    if (
        len(ranked) != canonical.source_row_count
        or len(set(ranked)) != len(ranked)
        or tuple(ranked[:100]) != canonical.ordered_case_ids
    ):
        raise ContractError("microseq salted order differs from canonical evidence order")
    prior = ranked[:MICROSEQ_RANK_START]
    selected = ranked[MICROSEQ_RANK_START:MICROSEQ_RANK_STOP]
    return MicroseqSelection(
        source_sha256=canonical.source_sha256,
        source_size=canonical.source_size,
        source_row_count=canonical.source_row_count,
        case_ids=selected,
        prior_order_hash=sha256_bytes(canonical_json(list(prior)).encode("utf-8")),
    )


def generate_microseq_selection(easyedit_root: str | Path) -> MicroseqSelection:
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    canonical = generate_counterfact_selection(root, seed=DEFAULT_SELECTION_SEED)
    all_case_ids = scan_counterfact_case_ids(root / "data/counterfact/counterfact.json")
    return select_microseq_cases(all_case_ids, canonical=canonical)


def _execution_envelope(branch: str, model_alias: str, run_id: str) -> None:
    if (
        branch not in BRANCHES
        or model_alias not in MODEL_SPECS
        or run_id != RUN_IDS[branch][model_alias]
    ):
        raise MV1Error("microseq controller identity differs from precommit")


def _slurm_state(branch: str, model_alias: str, run_id: str) -> dict[str, Any]:
    _execution_envelope(branch, model_alias, run_id)
    values = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "role": os.environ.get("ODEEDIT_MICROSEQ_PROCESS_ROLE"),
    }
    if all(value is None for value in values.values()):
        return {"under_slurm": False}
    if any(value is None for value in values.values()):
        raise MV1Error("partial microseq Slurm identity is forbidden")
    if (
        values["job_name"] != MICROSEQ_JOB_NAME
        or values["node"] != "devbox"
        or values["role"] != "controller"
        or not str(values["job_id"]).isdigit()
    ):
        raise MV1Error("microseq Slurm identity differs from fixed envelope")
    return {"under_slurm": True, **values}


def _proposal_artifact_metadata(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "name": path.name,
            "sha256": _file_sha256(path),
            "tensor_name": path.with_suffix(".pt").name,
            "tensor_sha256": _file_sha256(path.with_suffix(".pt")),
        }
        for path in paths
    ]


def _native_descendant(
    *,
    runtime: FixedModelRuntime,
    proposal: MemitFactorProposal,
    request: EditRequest,
    contexts: Any,
    hparams: Any,
    weight_names: Sequence[str],
    provenance_id: str,
) -> SnapshotManifest:
    try:
        with TemporaryExactMemitApplication(runtime.model, proposal):
            return _capture_source_snapshot(
                runtime=runtime,
                request=request,
                contexts=contexts,
                hparams=hparams,
                weight_names=weight_names,
                provenance_id=provenance_id,
            )
    finally:
        assert_snapshot_current(runtime.model, proposal.snapshot)


def _build_edit_action(
    *,
    runtime: FixedModelRuntime,
    bridge: EasyEditBridge,
    hparams: Any,
    contexts: Any,
    covariance_specs: Sequence[CovarianceCacheSpec],
    covariance_moments: Mapping[int, torch.Tensor],
    direct_z_root: Path,
    proposal_root: Path,
    receipt_root: Path,
    request: EditRequest,
    edit_index: int,
    branch: str,
    feature_writer: SanitizedJsonlWriter,
    action_writer: SanitizedJsonlWriter,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed_runtime(_event_seed(MICROSEQ_RUN_SEED, request.case_id))
    layer_by_weight = _layer_by_weight(hparams)
    if tuple(int(layer) for layer in hparams.layers) != LAYERS:
        raise MV1Error("microseq requires exact layers 4--8")
    weight_names = tuple(layer_by_weight)
    origin_hashes = _weight_hashes(runtime.model, weight_names)
    origin_state_identity = _state_identity(runtime)
    initial_rng_hash = rng_state_hash()
    _, target_ids = teacher_forced_rewrite_exact(runtime, request, contexts)
    if (
        _weight_hashes(runtime.model, weight_names) != origin_hashes
        or _state_identity(runtime) != origin_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise ContractError("rewrite-only target tokenization changed controller state")

    bindings = bridge.load()
    origin_snapshot = _capture_source_snapshot(
        runtime=runtime,
        request=request,
        contexts=contexts,
        hparams=hparams,
        weight_names=weight_names,
        provenance_id=bindings.provenance.manifest_id,
    )
    with _silence_upstream():
        direct_z = bridge.load_or_compute_direct_z(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            model_id=runtime.spec.snapshot_name,
            local_cache_root=direct_z_root,
            cache_path=f"{request.request_id}.pt",
        )
        ordered = bridge.propose_ordered_memit_factors(
            runtime.model,
            runtime.tokenizer,
            (request,),
            hparams,
            contexts,
            model_id=runtime.spec.snapshot_name,
            direct_z=direct_z,
            covariance_caches=covariance_specs,
        )
        synchronous = (
            bridge.propose_synchronous_memit_factors(
                runtime.model,
                runtime.tokenizer,
                (request,),
                hparams,
                contexts,
                direct_z,
                covariance_specs,
                model_id=runtime.spec.snapshot_name,
            )
            if branch == BRANCH_ODE
            else None
        )
    if synchronous is not None:
        synchronous.assert_same_entry_snapshot(ordered)
    lineage = FrozenTargetLineage.start(
        direct_z=direct_z,
        origin_snapshot=origin_snapshot,
        target_token_ids=target_ids,
    )
    lineage.assert_authorizes(
        direct_z=direct_z,
        snapshot=origin_snapshot,
        target_token_ids=target_ids,
    )
    native_energy = proposal_c_energy(ordered, covariance_moments, layer_by_weight)
    native_distance = math.sqrt(native_energy)
    if not math.isfinite(native_distance) or native_distance <= 0.0:
        raise ContractError("microseq native C-distance is invalid")

    if branch == BRANCH_NATIVE:
        proposals = (ordered,)
        descendants = (
            _native_descendant(
                runtime=runtime,
                proposal=ordered,
                request=request,
                contexts=contexts,
                hparams=hparams,
                weight_names=weight_names,
                provenance_id=bindings.provenance.manifest_id,
            ),
        )
        step_energies = (native_energy,)
        probe_distance = native_distance * QSTEP_PROBE_FRACTION
        panel_count = 0
    else:
        assert synchronous is not None
        hop_distance = native_distance * QSTEP_HOP_FRACTION
        hop_energy = native_energy / float(QSTEP_K * QSTEP_K)
        probe_distance = native_distance * QSTEP_PROBE_FRACTION
        units = build_unit_c_actions(synchronous, covariance_moments, layer_by_weight)
        assert_exact_action_contract(
            synchronous,
            ordered,
            units,
            expected_factor_names=weight_names,
            expected_action_ids=PROBE_ACTIONS,
            covariance_by_layer=covariance_moments,
            layer_by_weight=layer_by_weight,
        )
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(0).clone()
        panel = _proposal_panel(
            runtime=runtime,
            request=request,
            contexts=contexts,
            target_ids=target_ids,
            actions=units,
            epsilon=probe_distance,
            base_hashes=origin_hashes,
            state_identity=origin_state_identity,
            cpu_rng=cpu_rng,
            cuda_rng=cuda_rng,
            panel_label=f"microseq-refresh-t{edit_index}-w0",
        )
        decision, action = _build_score_mix_action(
            synchronous=synchronous,
            unit_actions=units,
            scores=panel.scores,
            covariance_moments=covariance_moments,
            layer_by_weight=layer_by_weight,
        )
        common_step = scale_proposal(
            action.proposal,
            hop_distance,
            solver_suffix=f"microseq-refresh-t{edit_index}-step-1",
        )
        _assert_step_energy(
            common_step,
            expected=hop_energy,
            covariance_moments=covariance_moments,
            layer_by_weight=layer_by_weight,
        )
        path = _build_policy_path(
            policy=REFRESHED_PREFIX,
            runtime=runtime,
            bridge=bridge,
            hparams=hparams,
            contexts=contexts,
            covariance_specs=covariance_specs,
            covariance_moments=covariance_moments,
            request=request,
            direct_z=direct_z,
            target_ids=target_ids,
            origin_snapshot=origin_snapshot,
            target_lineage=lineage,
            b0=synchronous,
            w0_unit=units,
            decision0=decision,
            action0=action,
            common_step=common_step,
            hop_distance=hop_distance,
            hop_energy=hop_energy,
            probe_distance=probe_distance,
            weight_names=weight_names,
            layer_by_weight=layer_by_weight,
            provenance_id=bindings.provenance.manifest_id,
            base_hashes=origin_hashes,
            base_state_identity=origin_state_identity,
            base_cpu_rng=cpu_rng,
            base_cuda_rng=cuda_rng,
        )
        proposals = path.proposals
        descendants = path.descendant_snapshots
        step_energies = path.step_energies
        panel_count = QSTEP_K
        if len(proposals) != QSTEP_K or any(
            not math.isclose(
                value,
                hop_energy,
                rel_tol=MATCHED_C_REL_TOL,
                abs_tol=MATCHED_C_ABS_TOL,
            )
            for value in step_energies
        ):
            raise ContractError("always-refresh path differs from K=4 matched-C lock")
    if (
        _weight_hashes(runtime.model, weight_names) != origin_hashes
        or _state_identity(runtime) != origin_state_identity
        or rng_state_hash() != initial_rng_hash
    ):
        raise ContractError("microseq action construction changed origin state")

    action_ids = tuple(
        f"t{edit_index:02d}-step{step_index:02d}"
        for step_index in range(1, len(proposals) + 1)
    )
    artifact_paths = tuple(
        write_proposal_artifact(
            proposal_root,
            action_id=action_id,
            proposal=proposal,
        )
        for action_id, proposal in zip(action_ids, proposals, strict=True)
    )
    artifact_metadata = _proposal_artifact_metadata(artifact_paths)
    feature = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "edit_index": edit_index,
        "controller_branch": branch,
        "policy_id": (
            "native-ordered-full-v1" if branch == BRANCH_NATIVE else MICROSEQ_POLICY_ID
        ),
        "application_mode": APPLICATION_MODES[branch],
        "origin_state_identity": origin_state_identity,
        "origin_parameter_hashes": origin_hashes,
        "target_identity": {
            "origin_lineage_id": lineage.lineage_id,
            "direct_z_tensor_sha256": lineage.direct_z_tensor_sha256,
            "direct_z_artifact_sha256": lineage.direct_z_artifact_sha256,
            "target_token_sha256": lineage.target_token_sha256,
            "direct_z_compute_count": 1,
        },
        "native_c_energy": native_energy,
        "native_c_distance": native_distance,
        "probe_c_distance": probe_distance,
        "step_c_energies": list(step_energies),
        "proposal_artifacts": artifact_metadata,
        "controller_only": True,
        "outcome_fields_loaded": False,
    }
    feature["feature_hash"] = _feature_hash(feature)
    path_hashes = [proposal_direction_hash(proposal) for proposal in proposals]
    action_payload = {
        "case_id": request.case_id,
        "request_id": request.request_id,
        "edit_index": edit_index,
        "controller_branch": branch,
        "feature_hash": feature["feature_hash"],
        "branch_order": list(action_ids),
        "path_action_hashes": {branch: path_hashes},
        "proposal_entry_state_ids": [proposal.entry_snapshot_id for proposal in proposals],
        "proposal_descendant_state_ids": [snapshot.state_id for snapshot in descendants],
        "per_hop_c_energy": (
            native_energy if branch == BRANCH_NATIVE else step_energies[0]
        ),
        "application_mode": APPLICATION_MODES[branch],
        "policy_id": (
            "native-ordered-full-v1" if branch == BRANCH_NATIVE else MICROSEQ_POLICY_ID
        ),
        "all_actions_before_outcomes": True,
    }
    action_payload["commitment_hash"] = _feature_hash(action_payload)
    receipt_name, receipt_sha256 = _commit_action(
        feature_writer=feature_writer,
        action_writer=action_writer,
        receipt_root=receipt_root,
        feature=feature,
        action=action_payload,
        expected_branch_order=action_ids,
        feature_event="microseq_controller_feature",
        action_event="microseq_controller_action",
        receipt_schema=MICROSEQ_RECEIPT_SCHEMA,
    )

    for proposal, descendant in zip(proposals, descendants, strict=True):
        observed = apply_proposal_in_disposable_process(
            runtime.model,
            proposal,
            application_mode=APPLICATION_MODES[branch],
        )
        assert_snapshot_current(runtime.model, descendant)
        expected = {
            record.name: record.sha256
            for record in descendant.parameters
            if record.name in observed
        }
        if observed != expected:
            raise ContractError("permanent action hashes differ from precomputed descendant")

    return {
        "schema_version": MICROSEQ_SCHEMA,
        "case_id": request.case_id,
        "request_id": request.request_id,
        "edit_index": edit_index,
        "branch": branch,
        "feature_hash": feature["feature_hash"],
        "commitment_hash": action_payload["commitment_hash"],
        "receipt_name": receipt_name,
        "receipt_sha256": receipt_sha256,
        "direct_z_artifact_sha256": direct_z.artifact.sha256,
        "direct_z_compute_count": 1,
        "proposal_count": len(proposals),
        "proposal_build_count": 1 if branch == BRANCH_NATIVE else QSTEP_K,
        "probe_panel_count": panel_count,
        "terminal_state_id": descendants[-1].state_id,
        "controller_only": True,
        "outcome_fields_loaded": False,
        "wall_seconds": time.perf_counter() - started,
        "pass": True,
    }


def run_microseq_controller(
    *,
    easyedit_root: str | Path,
    branch: str,
    model_alias: str,
    run_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
) -> dict[str, Any]:
    started = time.perf_counter()
    _execution_envelope(branch, model_alias, run_id)
    slurm = _slurm_state(branch, model_alias, run_id)
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    git_state = _git_runtime_state()
    provenance = preflight_fixed_artifacts(root, model_alias=model_alias)
    selection = generate_microseq_selection(root)
    requests = load_counterfact_requests(root, selection.case_ids)
    if len(requests) != CHAIN_LENGTH:
        raise ContractError("microseq controller requires exact four requests")
    bridge = EasyEditBridge(root, expected_files=_bridge_pins())
    bridge_provenance = bridge.preflight()
    if not set(record.path for record in bridge_provenance.files).issubset(
        set(record.path for record in provenance.files)
    ):
        raise MV1Error("microseq bridge provenance is outside fixed manifest")

    with offline_environment():
        assert_tensor_sha256_device_parity(torch.device("cuda", 0))
        bindings = bridge.load()
        hparams = _load_hparams(root, spec, bindings)
        seed_runtime(MICROSEQ_RUN_SEED)
        runtime = model_loader(model_alias)
        if runtime.spec != spec:
            raise MV1Error("microseq model loader returned different spec")
        contexts = _freeze_contexts(bridge, runtime, seed=MICROSEQ_RUN_SEED)
        destination = _local_run_directory(output_root, run_id)
        manifest_path = destination / "manifest.json"
        features_path = destination / "features.jsonl"
        actions_path = destination / "actions.jsonl"
        events_path = destination / "events.jsonl"
        summary_path = destination / "summary.json"
        direct_z_root = destination / "direct_z"
        proposal_root = destination / "proposal_artifacts"
        receipt_root = destination / "action_receipts"
        direct_z_root.mkdir(mode=0o700)
        proposal_root.mkdir(mode=0o700)
        receipt_root.mkdir(mode=0o700)
        manifest = {
            "schema_version": MICROSEQ_SCHEMA,
            "run_id": run_id,
            "controller_branch": branch,
            "application_mode": APPLICATION_MODES[branch],
            "policy_id": (
                "native-ordered-full-v1" if branch == BRANCH_NATIVE else MICROSEQ_POLICY_ID
            ),
            "ode_edit_git": git_state,
            "slurm": slurm,
            "model": runtime.metadata(),
            "selection": selection.to_dict(),
            "contexts": {
                "manifest_id": contexts.manifest_id,
                "group_sizes": [len(group) for group in contexts.templates],
                "raw_templates_persisted": False,
            },
            "provenance_id": provenance.manifest_id,
            "fixed_files": _relative_provenance(provenance, root),
            "constants": {
                "layers": list(LAYERS),
                "chain_length": CHAIN_LENGTH,
                "k": QSTEP_K,
                "hop_fraction": QSTEP_HOP_FRACTION,
                "probe_fraction": QSTEP_PROBE_FRACTION,
                "run_seed": MICROSEQ_RUN_SEED,
            },
            "firewall": {
                "controller_only": True,
                "outcome_fields_loaded": False,
                "separate_evaluator_required": True,
            },
        }
        _safe_payload(manifest)
        _write_json_exclusive(manifest_path, manifest)
        covariance_specs = _covariance_specs(root, spec)
        covariance_moments, loaded_covariances = _load_verified_covariances(
            root=root, runtime=runtime, specs=covariance_specs
        )
        torch.cuda.reset_peak_memory_stats(0)
        results: list[dict[str, Any]] = []
        failure_type: str | None = None
        with (
            SanitizedJsonlWriter(
                features_path, run_id, schema_version=MICROSEQ_STREAM_SCHEMA
            ) as feature_writer,
            SanitizedJsonlWriter(
                actions_path, run_id, schema_version=MICROSEQ_STREAM_SCHEMA
            ) as action_writer,
            SanitizedJsonlWriter(
                events_path, run_id, schema_version=MICROSEQ_STREAM_SCHEMA
            ) as event_writer,
        ):
            for edit_index, request in enumerate(requests, start=1):
                try:
                    result = _build_edit_action(
                        runtime=runtime,
                        bridge=bridge,
                        hparams=hparams,
                        contexts=contexts,
                        covariance_specs=covariance_specs,
                        covariance_moments=covariance_moments,
                        direct_z_root=direct_z_root,
                        proposal_root=proposal_root,
                        receipt_root=receipt_root,
                        request=request,
                        edit_index=edit_index,
                        branch=branch,
                        feature_writer=feature_writer,
                        action_writer=action_writer,
                    )
                except Exception as exc:
                    print(
                        f"MICROSEQ_CONTROLLER_TRACEBACK_BEGIN type={type(exc).__name__}",
                        file=sys.stderr,
                    )
                    for frame, line_number in traceback.walk_tb(exc.__traceback__):
                        print(
                            f'  File "{frame.f_code.co_filename}", line {line_number}, in {frame.f_code.co_name}',
                            file=sys.stderr,
                        )
                    print("MICROSEQ_CONTROLLER_TRACEBACK_END", file=sys.stderr)
                    result = {
                        "case_id": request.case_id,
                        "edit_index": edit_index,
                        "failure_type": type(exc).__name__,
                        "pass": False,
                    }
                    failure_type = type(exc).__name__
                results.append(result)
                event_writer.write("microseq_controller_edit", result)
                if failure_type is not None:
                    break
            sequences = {
                "features": feature_writer.sequence,
                "actions": action_writer.sequence,
                "events": event_writer.sequence,
            }
        receipt_count = len(tuple(receipt_root.glob("*.json")))
        proposal_count = len(tuple(proposal_root.glob("*.json")))
        expected_proposals = CHAIN_LENGTH if branch == BRANCH_NATIVE else CHAIN_LENGTH * QSTEP_K
        exact_counts = bool(
            failure_type is None
            and sequences == {"features": 4, "actions": 4, "events": 4}
            and receipt_count == 4
            and proposal_count == expected_proposals
            and len(tuple(proposal_root.glob("*.pt"))) == expected_proposals
            and len(tuple(direct_z_root.glob("*.pt"))) == 4
        )
        summary = {
            "schema_version": MICROSEQ_SCHEMA,
            "run_id": run_id,
            "model_alias": model_alias,
            "controller_branch": branch,
            "run_status": "aborted" if failure_type else "completed",
            "failure_type": failure_type,
            "all_pass": bool(
                failure_type is None
                and len(results) == CHAIN_LENGTH
                and all(result.get("pass") is True for result in results)
                and exact_counts
            ),
            "selection_manifest_id": selection.manifest_id,
            "stream_sequences": sequences,
            "expected_counts_exact": exact_counts,
            "receipt_count": receipt_count,
            "proposal_artifact_count": proposal_count,
            "direct_z_artifact_count": len(tuple(direct_z_root.glob("*.pt"))),
            "loaded_covariances": list(loaded_covariances),
            "controller_only": True,
            "outcome_fields_loaded": False,
            "disposable_process_exit_required": True,
            "resource": {
                "wall_seconds": time.perf_counter() - started,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "host_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
            "artifacts": {
                "manifest_sha256": _file_sha256(manifest_path),
                "features_sha256": _file_sha256(features_path),
                "actions_sha256": _file_sha256(actions_path),
                "events_sha256": _file_sha256(events_path),
            },
            "git_output_written": False,
        }
        _safe_payload(summary)
        _write_json_exclusive(summary_path, summary)
        return {**summary, "output_directory": str(destination)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-microseq-controller", allow_abbrev=False
    )
    parser.add_argument("--easyedit-root", required=True)
    parser.add_argument("--branch", required=True, choices=BRANCHES)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_microseq_controller(
            easyedit_root=args.easyedit_root,
            branch=args.branch,
            model_alias=args.model,
            run_id=args.run_id,
            output_root=args.output_root,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "aborted",
                    "error_type": type(exc).__name__,
                    "raw_exception_persisted": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
