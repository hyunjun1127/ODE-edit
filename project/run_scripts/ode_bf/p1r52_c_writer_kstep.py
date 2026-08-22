"""Reusable full-FP32 P1R52 K-step C-writer takeover.

The P1R52 controller remains owned by ``_run_ode_arm``.  At each accepted K,
this runtime consumes the exact native IL1 selected target, applies one released
C0/C1/C3 writer, and returns observation-only pre/post metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .accounting import ComputeLedger
from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p0_runtime import ModelForwardCounter
from .p1_evaluator import EndpointActionFreeze, load_counterfact_cases_after_freeze
from .p1r36_independent_b10x10_runtime import _hashes
from .p1r52_accepted_z_observation import evaluate_accepted_z_batch, r52_binding
from .p1r52_joint_pc_execution import JointPCWriterArm
from .p1r52_joint_pc_fp32_runtime import _run_pc_arm_fp32
from .p1r52_joint_pc_independent_fp32_runtime import _assert_no_low_precision_activity, _validate_official_fp32_apply
from .p1r52_joint_pc_runtime import _writer_entry
from .p1r52_residual_reserve_pre_writer_interface import reconstruct_native_il1_selected_target
from .p1r52_target_official_alphaedit_writer import _endpoint_summary, _evaluate_w, _writer_gap, accepted_z_cache_template
from .scalable_batched_native import run_official_native_apply
from .scalable_batched_runtime import P1R23_GRID_COUNT, scalable_ordered_request_digest


ARMS = ("C0-KSTEP", "C1-KSTEP", "C3-KSTEP")


@dataclass(frozen=True, slots=True)
class CKStepExecution:
    arm: str
    step_index: int
    selected_target_sha256: str
    entry_weight_sha256: Mapping[str, str]
    commit_weight_sha256: Mapping[str, str]
    route_receipt: Mapping[str, Any]
    writer_receipt: Mapping[str, Any]
    prefix_capture_receipts: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    compute: Mapping[str, int]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "step_index": self.step_index,
            "selected_target_sha256": self.selected_target_sha256,
            "entry_weight_sha256": dict(self.entry_weight_sha256),
            "commit_weight_sha256": dict(self.commit_weight_sha256),
            "route_receipt": dict(self.route_receipt),
            "writer_receipt": dict(self.writer_receipt),
            "prefix_capture_receipts": [dict(item) for item in self.prefix_capture_receipts],
            "metrics": dict(self.metrics),
            "compute": dict(self.compute),
            "identity_sha256": self.identity_sha256,
        }


class CKStepWriterRuntime:
    def __init__(
        self,
        *,
        arm: str,
        model: torch.nn.Module,
        tokenizer: Any,
        requests: Sequence[Mapping[str, Any]],
        hparams: Any,
        projector: torch.Tensor,
        contexts: Sequence[Sequence[str]],
        covariance_registry: Any,
        projector_sha256: str,
        controller_lock: Any,
        objective_plan: Any,
        capture_plan: Any,
        dataset_path: Path,
        private_root: Path,
        job_ledger: ComputeLedger,
        solve_history_keys_by_layer: Mapping[int, torch.Tensor] | None = None,
        alpha_cache_status: str = "ALPHA_CACHE_OFF_CONTROL",
    ) -> None:
        if arm not in ARMS:
            raise ODEBFContractError("C K-step arm differs")
        if alpha_cache_status not in ("ALPHA_CACHE_OFF_CONTROL", "ALPHA_CACHE_BATCH_ENTRY_SNAPSHOT"):
            raise ODEBFContractError("C K-step cache policy differs")
        self.arm = arm
        self.model = model
        self.tokenizer = tokenizer
        self.requests = tuple(requests)
        self.hparams = hparams
        self.projector = projector
        self.contexts = contexts
        self.covariance_registry = covariance_registry
        self.projector_sha256 = projector_sha256
        self.controller_lock = controller_lock
        self.objective_plan = objective_plan
        self.capture_plan = capture_plan
        self.dataset_path = dataset_path
        self.private_root = private_root
        self.job_ledger = job_ledger
        self.solve_history_keys_by_layer = solve_history_keys_by_layer
        self.alpha_cache_status = alpha_cache_status
        self.executions: list[CKStepExecution] = []
        self._entry_order = scalable_ordered_request_digest([str(item["request_sha256"]) for item in self.requests])

    def execute(self, outer: Any, *, step_index: int) -> CKStepExecution:
        if step_index != len(self.executions) or not 0 <= step_index < P1R23_GRID_COUNT:
            raise ODEBFStateError("C K-step execution order differs")
        selected, bridge = reconstruct_native_il1_selected_target(outer, step_index=step_index)
        target = selected.target_step.target_next.detach().cpu().float().contiguous()
        if target.dtype is not torch.float32 or target.shape[1] != len(self.requests):
            raise ODEBFContractError("C K-step selected target geometry differs")
        entry_hashes = _hashes(self.model_touched)
        freeze = EndpointActionFreeze(
            arm=f"{self.arm}-k{step_index + 1}",
            sequential_batch=step_index,
            request_order_sha256=self._entry_order,
            selected_snapshot_sha256=bridge.identity_sha256,
            fixed_budget_slots_completed=step_index + 1,
        )
        cases = tuple(load_counterfact_cases_after_freeze(
            self.dataset_path, self.requests, freeze, expected_batch_size=len(self.requests)
        ))
        binding = r52_binding(self.arm, self.requests, self.hparams, target)
        counter = ModelForwardCounter(self.model, self.job_ledger)
        try:
            z_observation, _ = evaluate_accepted_z_batch(
                self.model, self.tokenizer, self.requests, cases,
                role=self.arm, round_index=step_index + 1, binding=binding,
                committed_weight_sha256=entry_hashes,
            )
        finally:
            counter.close()
        pre_scores = _evaluate_w(self.model, self.tokenizer, cases, freeze=freeze, ledger=self.job_ledger)
        if self.arm in ("C0-KSTEP", "C1-KSTEP"):
            entry = _writer_entry(
                self.model, self.tokenizer, self.requests, target=target,
                hparams=self.hparams, projector=self.projector, contexts=self.contexts,
                covariance_registry=self.covariance_registry,
                projector_sha256=self.projector_sha256,
                controller_lock=self.controller_lock,
                objective_plan=self.objective_plan, capture_plan=self.capture_plan,
            )
            writer_arm = JointPCWriterArm.C0 if self.arm == "C0-KSTEP" else JointPCWriterArm.C1
            writer = _run_pc_arm_fp32(
                self.model, writer_arm, target=target, entry=entry,
                hparams=self.hparams, projector=self.projector,
                covariance_registry=self.covariance_registry,
                projector_sha256=self.projector_sha256,
                controller_lock=self.controller_lock, capture_plan=self.capture_plan,
                solve_history_keys_by_layer=self.solve_history_keys_by_layer,
            )
            route = (
                entry["control"].raw_free_payload()
                if self.arm == "C0-KSTEP"
                else entry["joint"].receipt.raw_free_payload()
            )
            prefix = tuple(writer["prefix_capture_receipts"])
            writer_receipt = {
                "identity_sha256": canonical_hash(writer),
                "writer": writer,
                "selected_pi": writer["pi"],
                "route_recomputed_at_current_k": True,
                "route_frozen_across_k_count": 0,
            }
            dense_count = len(writer["layers"])
            apply_count = len(writer["layers"])
        else:
            with accepted_z_cache_template(
                self.requests, target, self.hparams,
                parent=self.private_root / f"k{step_index + 1}",
            ) as (cache_template, bridge_receipt):
                counter = ModelForwardCounter(self.model, self.job_ledger)
                try:
                    apply_payload, originals = run_official_native_apply(
                        self.model, self.tokenizer, self.requests, self.hparams,
                        touched=self.model_touched, reset_cache=True,
                        cache_history_width=0, cache_template=cache_template,
                        expected_native_compute_z_call_count=0,
                        accepted_z_source="P1R52_CURRENT_K_ACCEPTED_TARGET",
                    )
                finally:
                    counter.close()
            _validate_official_fp32_apply("C3", apply_payload)
            if any(tensor_sha256(originals[name]) != entry_hashes[name] for name in self.model_touched):
                raise ODEBFStateError("C3 K-step Official writer entry differs")
            route = {
                "schema": "ode-edit-c3-kstep-no-pc-route/v1",
                "status": "OFFICIAL_ALPHAEDIT_DIRECT_ACCEPTED_Z_WRITER",
                "pc_router_call_count": 0,
                "decision_influence_count": 0,
            }
            route["identity_sha256"] = canonical_hash(route)
            prefix = ()
            writer_receipt = {
                "identity_sha256": apply_payload["identity_sha256"],
                "writer": apply_payload,
                "accepted_z_bridge": bridge_receipt,
                "native_compute_z_call_count": 0,
                "hybrid_label": "P1R52_KSTEP_TARGET_PLUS_OFFICIAL_WRITER_8_CALL_HYBRID",
            }
            dense_count = len(tuple(int(item) for item in self.hparams.layers))
            apply_count = dense_count
        _assert_no_low_precision_activity(writer_receipt)
        commit_hashes = _hashes(self.model_touched)
        if commit_hashes == entry_hashes:
            raise ODEBFStateError("C K-step writer produced no physical transition")
        post_scores = _evaluate_w(self.model, self.tokenizer, cases, freeze=freeze, ledger=self.job_ledger)
        z_summary = _endpoint_summary(z_observation["scores"])
        pre_summary = _endpoint_summary(pre_scores)
        post_summary = _endpoint_summary(post_scores)
        metrics = {
            "accepted_z": {"binding": binding.raw_free_payload(), "summary": z_summary, "scores": z_observation["scores"]},
            "pre_writer_W": {"summary": pre_summary, "scores": pre_scores},
            "post_writer_W": {"summary": post_summary, "scores": post_scores},
            "post_W_minus_z_gap": _writer_gap(post_summary, z_summary),
            "heldout_evaluator_decision_influence_count": 0,
        }
        compute = {
            "accepted_z_evaluator_count": 1,
            "pre_writer_evaluator_count": 1,
            "post_writer_evaluator_count": 1,
            "heldout_evaluator_count": 3,
            "dense_update_construction_count": dense_count,
            "native_apply_count": apply_count,
            "logical_commit_count": 1,
            "router_call_count": 1 if self.arm == "C1-KSTEP" else 0,
            "model_backward_added_count": 0,
            "semantic_slope_backward_added_count": 0,
            "bf16_fp16_path_count": 0,
            "numeric_storage_cast_count": 0,
            "retry_count": 0,
        }
        payload = {
            "arm": self.arm,
            "step_index": step_index,
            "selected_target_sha256": tensor_sha256(target),
            "selected_bridge_identity": bridge.identity_sha256,
            "entry_weight_sha256": entry_hashes,
            "commit_weight_sha256": commit_hashes,
            "route_receipt": route,
            "writer_receipt": writer_receipt,
            "prefix_capture_receipts": [dict(item) for item in prefix],
            "metrics": metrics,
            "compute": compute,
            "alpha_cache_status": self.alpha_cache_status,
            "current_K_writer_affects_next_target": True,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        execution = CKStepExecution(
            self.arm, step_index, payload["selected_target_sha256"],
            entry_hashes, commit_hashes, route, writer_receipt, prefix,
            metrics, compute, payload["identity_sha256"],
        )
        self.executions.append(execution)
        return execution

    @property
    def model_touched(self) -> Mapping[str, torch.nn.Parameter]:
        names = {
            f"{self.hparams.rewrite_module_tmp.format(int(layer))}.weight"
            for layer in self.hparams.layers
        }
        parameters = dict(self.model.named_parameters())
        return {name: parameters[name] for name in sorted(names)}

    def assert_complete(self) -> None:
        if len(self.executions) != P1R23_GRID_COUNT or any(
            item.step_index != index for index, item in enumerate(self.executions)
        ):
            raise ODEBFStateError("C K-step K8 completion differs")

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "arm": self.arm,
            "outer_count": len(self.executions),
            "execution_identities": [item.identity_sha256 for item in self.executions],
            "alpha_cache_status": self.alpha_cache_status,
            "full_fp32": True,
            "numeric_storage_cast_count": 0,
            "bf16_fp16_path_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


__all__ = ["ARMS", "CKStepExecution", "CKStepWriterRuntime"]
