"""Post-freeze validation for the single P1R14 integrated writer arm.

This module is intentionally detached from the controller.  It opens held-out
CounterFact fields only after an immutable action-freeze receipt exists, and it
never returns a value to the routing or target clocks.
"""

from __future__ import annotations

import hashlib
import math
import threading
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .accounting import ComputeLedger
from .contracts import BATCH_SIZE, ODEBFContractError, ODEBFStateError, canonical_hash
from .cold_start_target import capture_cold_z_base
from .functional import WaypointFactor, tensor_sha256
from .integrated_physical_writer_experiment import IntegratedActionFreeze
from .integrated_physical_writer_runtime import (
    IntegratedEndpointTransactionReceipt,
    commit_verify_restore_integrated_endpoint,
)
from .p1_backend import CandidateBF16FunctionalTrial, capture_p1_native_entry
from .p1_replay import evaluate_next_token_log_probs, samplewise_teacher_kl
from .request_digest import ordered_request_digest_v1


R14_TERMINAL_SCHEMA = "ode-edit-s05-integrated-physical-writer-terminal/v1"


@dataclass(frozen=True, slots=True)
class IntegratedEndpointFreeze:
    arm: str
    request_order_sha256: str
    selected_snapshot_sha256: str
    accepted_transition_count: int
    action_frozen: bool = True

    def identity(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True, slots=True)
class IntegratedPrimaryEvaluation:
    primary: dict[str, Any]
    score_vectors: dict[str, Any]
    overlay: dict[str, Any] | None
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IntegratedTerminalPanel:
    action_freeze_sha256: str
    w0: IntegratedPrimaryEvaluation
    integrated_w_only: IntegratedPrimaryEvaluation
    integrated_z_oracle: IntegratedPrimaryEvaluation
    native_w_only: IntegratedPrimaryEvaluation
    native_z_oracle: IntegratedPrimaryEvaluation
    integrated_exact_functional_p: dict[str, Any]
    native_exact_functional_p: dict[str, Any]
    native_receipt: dict[str, Any]
    paired_native_floor: dict[str, Any]
    endpoint_transaction: dict[str, Any]
    heldout_lookup: dict[str, Any]
    terminal_compute: dict[str, Any]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def _unwrap(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ODEBFContractError("integrated heldout target output differs")


def _rewrap(original: Any, tensor: torch.Tensor) -> Any:
    if isinstance(original, torch.Tensor):
        return tensor
    if isinstance(original, tuple):
        return (tensor, *original[1:])
    raise ODEBFContractError("integrated heldout target output differs")


class IntegratedHeldoutResidualOverlay(AbstractContextManager["IntegratedHeldoutResidualOverlay"]):
    """Apply one request residual to held-out rewrite/paraphrase rows only."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_name: str,
        residual: torch.Tensor,
        positions_by_forward: Sequence[Sequence[int]],
        patched_row_counts: Sequence[int],
    ) -> None:
        positions = tuple(tuple(int(item) for item in row) for row in positions_by_forward)
        counts = tuple(int(item) for item in patched_row_counts)
        if (
            not isinstance(residual, torch.Tensor)
            or residual.ndim != 2
            or residual.shape[1] != BATCH_SIZE
            or not residual.is_floating_point()
            or not bool(torch.isfinite(residual).all())
            or len(positions) != BATCH_SIZE
            or len(counts) != BATCH_SIZE
            or any(not row for row in positions)
            or any(count <= 0 or count > len(row) for count, row in zip(counts, positions, strict=True))
        ):
            raise ODEBFContractError("integrated heldout residual geometry differs")
        self.model = model
        self.layer_name = layer_name
        self.residual = residual.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self.positions = positions
        self.counts = counts
        self.calls = 0
        self.patched_rows = 0
        self.locality_rows = 0
        self.maximum_assignment_error = 0.0
        self.maximum_realized_delta_error = 0.0
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    @staticmethod
    def _position(value: int, length: int) -> int:
        result = value if value >= 0 else length + value
        if result < 0 or result >= length:
            raise ODEBFContractError("integrated heldout lookup is out of range")
        return result

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
        if self.calls >= BATCH_SIZE:
            raise ODEBFContractError("integrated heldout overlay received extra forward")
        activation = _unwrap(output)
        positions = self.positions[self.calls]
        count = self.counts[self.calls]
        if (
            activation.ndim != 3
            or activation.shape[0] != len(positions)
            or activation.shape[-1] != self.residual.shape[0]
            or not activation.is_floating_point()
            or not bool(torch.isfinite(activation).all())
        ):
            raise ODEBFContractError("integrated heldout activation layout differs")
        request_residual = self.residual[:, self.calls].to(
            device=activation.device, dtype=activation.dtype
        )
        patched = activation.clone()
        for row in range(count):
            position = self._position(positions[row], int(activation.shape[1]))
            before = activation[row, position, :]
            expected = before + request_residual
            patched[row, position, :] = expected
            assigned = patched[row, position, :]
            if not torch.equal(assigned, expected):
                raise ODEBFContractError("integrated heldout assignment is not exact")
            self.maximum_assignment_error = max(
                self.maximum_assignment_error,
                float(torch.max(torch.abs(assigned.float() - expected.float())).cpu()),
            )
            self.maximum_realized_delta_error = max(
                self.maximum_realized_delta_error,
                float(
                    torch.max(
                        torch.abs((assigned - before).float() - request_residual.float())
                    ).cpu()
                ),
            )
        self.patched_rows += count
        self.locality_rows += len(positions) - count
        self.calls += 1
        return _rewrap(output, patched)

    def __enter__(self) -> "IntegratedHeldoutResidualOverlay":
        if self._handle is not None:
            raise ODEBFStateError("integrated heldout overlay is already active")
        self._handle = self.model.get_submodule(self.layer_name).register_forward_hook(
            self._hook
        )
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, exc, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def raw_free_payload(self) -> dict[str, Any]:
        if self._handle is not None or self.calls != BATCH_SIZE:
            raise ODEBFStateError("integrated heldout overlay call/cleanup differs")
        payload = {
            "schema": f"{R14_TERMINAL_SCHEMA}-heldout-additive-overlay",
            "request_count": BATCH_SIZE,
            "hook_call_count": self.calls,
            "patched_row_count": self.patched_rows,
            "locality_nohook_row_count": self.locality_rows,
            "residual_sha256": tensor_sha256(self.residual),
            "authoritative_assignment_definition": "patched=before+request_residual",
            "maximum_authoritative_assignment_error": self.maximum_assignment_error,
            "maximum_realized_delta_error": self.maximum_realized_delta_error,
            "realized_delta_error_role": "BF16_ROUNDING_OBSERVATION_ONLY",
            "absolute_z_replacement_count": 0,
            "decision_influence_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


def integrated_heldout_lookup_geometry(
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    cases: Sequence[Any],
    *,
    fact_token_strategy: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], dict[str, Any]]:
    """Resolve held-out lookup rows after freeze without serializing text."""

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    if len(requests) != BATCH_SIZE or len(cases) != BATCH_SIZE:
        raise ODEBFContractError("integrated heldout case inventory differs")
    padding_side = str(tokenizer.padding_side)
    if padding_side not in ("left", "right"):
        raise ODEBFContractError("integrated heldout padding policy differs")
    positions: list[tuple[int, ...]] = []
    counts: list[int] = []
    rows_receipt: list[dict[str, Any]] = []
    for ordinal, (request, case) in enumerate(zip(requests, cases, strict=True)):
        if str(request["request_sha256"]) != str(case.request_sha256):
            raise ODEBFContractError("integrated heldout request order differs")
        subject = str(request["subject"])
        patched_prefixes = (case.rewrite_prompt,) + tuple(case.paraphrase_prompts)
        all_prefixes = patched_prefixes + tuple(case.neighborhood_prompts)
        raw_lookup: list[int] = []
        prefix_lengths: list[int] = []
        template_hashes: list[str] = []
        for prefix in patched_prefixes:
            if prefix.count(subject) != 1:
                raise ODEBFContractError("integrated heldout subject surface differs")
            template = prefix.replace(subject, "{}", 1)
            raw_lookup.append(
                int(
                    alpha_main.find_fact_lookup_idx(
                        template,
                        subject,
                        tokenizer,
                        fact_token_strategy,
                        verbose=False,
                    )
                )
            )
            prefix_lengths.append(len(tokenizer(prefix)["input_ids"]))
            template_hashes.append(hashlib.sha256(template.encode("utf-8")).hexdigest())
        surfaces = [
            f"{prefix} {suffix}"
            for prefix in all_prefixes
            for suffix in (case.target_new, case.target_true)
        ]
        encoded = tokenizer(surfaces, padding=True, return_tensors="pt")
        attention = encoded["attention_mask"]
        row_positions: list[int] = []
        for row in range(2 * len(patched_prefixes)):
            prefix_index = row // 2
            pad = int(attention.shape[1] - attention[row].sum()) if padding_side == "left" else 0
            raw = raw_lookup[prefix_index]
            row_positions.append(
                pad + prefix_lengths[prefix_index] - 1 if raw == -1 else pad + raw
            )
        row_positions.extend(0 for _ in range(len(surfaces) - len(row_positions)))
        if any(value < 0 or value >= int(attention.shape[1]) for value in row_positions[: 2 * len(patched_prefixes)]):
            raise ODEBFContractError("integrated heldout lookup position differs")
        positions.append(tuple(row_positions))
        counts.append(2 * len(patched_prefixes))
        rows_receipt.append(
            {
                "request_ordinal": ordinal,
                "request_sha256": str(case.request_sha256),
                "patched_row_count": counts[-1],
                "locality_nohook_row_count": len(surfaces) - counts[-1],
                "lookup_sha256": canonical_hash(row_positions),
                "template_sha256": canonical_hash(template_hashes),
            }
        )
    payload = {
        "schema": f"{R14_TERMINAL_SCHEMA}-heldout-lookup",
        "request_count": BATCH_SIZE,
        "padding_side": padding_side,
        "rows": rows_receipt,
        "absolute_z_replacement_count": 0,
        "heldout_controller_access_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return tuple(positions), tuple(counts), payload


def _score_payload(groups: Sequence[Sequence[Any]], *, locality: bool) -> dict[str, Any]:
    per_request = []
    flat_new: list[float] = []
    flat_old: list[float] = []
    for scores in groups:
        rows = []
        for score in scores:
            new = float(score.target_new_nll)
            old = float(score.target_true_nll)
            flat_new.append(new)
            flat_old.append(old)
            rows.append(
                {
                    "target_new_nll": new,
                    "target_old_nll": old,
                    "margin": new - old if locality else old - new,
                }
            )
        per_request.append(rows)
    return {
        "per_request": per_request,
        "mean_target_new_nll": float(np.mean(np.asarray(flat_new, dtype=np.float64))),
        "mean_target_old_nll": float(np.mean(np.asarray(flat_old, dtype=np.float64))),
        "margin_definition": "new-old" if locality else "old-new",
    }


def evaluate_integrated_primary(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    model_alias: str,
    freeze: IntegratedEndpointFreeze,
    overlay: IntegratedHeldoutResidualOverlay | None = None,
) -> IntegratedPrimaryEvaluation:
    """Run the pinned official scorer once and retain raw-free NLL vectors."""

    if not freeze.action_frozen:
        raise ODEBFContractError("integrated heldout evaluation preceded action freeze")
    from .evaluator import _EvaluationStateGuard, _is_llama, _model_device
    from .p1_evaluator import (
        _evaluate_prefixes_pinned,
        counterfact_primary_receipt_from_scores,
    )

    batch = tuple(cases)
    order = ordered_request_digest_v1([str(item.request_sha256) for item in batch])
    if len(batch) != BATCH_SIZE or order != freeze.request_order_sha256:
        raise ODEBFContractError("integrated heldout evaluator order differs")
    llama = _is_llama(model, model_alias)
    device = _model_device(model)
    efficacy: list[tuple[Any, ...]] = []
    generalization: list[tuple[Any, ...]] = []
    locality: list[tuple[Any, ...]] = []
    spans: list[dict[str, Any]] = []
    processed = 0
    context = overlay if overlay is not None else _NullContext()
    with _EvaluationStateGuard(model), torch.no_grad(), context:
        for case in batch:
            prefixes = (
                (case.rewrite_prompt,)
                + tuple(case.paraphrase_prompts)
                + tuple(case.neighborhood_prompts)
            )
            pairs, tokens, span = _evaluate_prefixes_pinned(
                model,
                tokenizer,
                prefixes,
                case.target_new,
                case.target_true,
                llama=llama,
                device=device,
            )
            split = 1 + len(case.paraphrase_prompts)
            efficacy.append(pairs[:1])
            generalization.append(pairs[1:split])
            locality.append(pairs[split:])
            processed += tokens
            spans.append(
                {
                    "request_sha256": case.request_sha256,
                    "rewrite_count": 1,
                    "paraphrase_count": len(case.paraphrase_prompts),
                    "neighborhood_count": len(case.neighborhood_prompts),
                    **span,
                }
            )
    overlay_payload = overlay.raw_free_payload() if overlay is not None else None
    case_identity = canonical_hash([item.raw_free_identity() for item in batch])
    span_sha = canonical_hash(spans)
    primary = counterfact_primary_receipt_from_scores(
        efficacy=efficacy,
        generalization=generalization,
        locality=locality,
        request_order_sha256=order,
        evaluation_case_identity_sha256=case_identity,
        target_span_sha256=span_sha,
        model_forward_count=BATCH_SIZE,
        processed_token_count=processed,
        endpoint_freeze_sha256=freeze.identity(),
    )
    vectors = {
        "efficacy": _score_payload(efficacy, locality=False),
        "generalization": _score_payload(generalization, locality=False),
        "locality": _score_payload(locality, locality=True),
    }
    payload = {
        "primary": primary.raw_free_payload(),
        "score_vectors": vectors,
        "overlay": overlay_payload,
    }
    return IntegratedPrimaryEvaluation(
        payload["primary"], vectors, overlay_payload, canonical_hash(payload)
    )


class _NullContext(AbstractContextManager["_NullContext"]):
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, exc, traceback
        return False


def evaluate_exact_functional_p(
    model: torch.nn.Module,
    tokenizer: Any,
    anchor_microbatches: Sequence[Sequence[Mapping[str, Any]]],
    teacher_log_probs: Sequence[torch.Tensor],
    entry_kl: Sequence[torch.Tensor],
) -> dict[str, Any]:
    if not (
        len(anchor_microbatches) == len(teacher_log_probs) == len(entry_kl) == 6
        and all(len(batch) == BATCH_SIZE for batch in anchor_microbatches)
    ):
        raise ODEBFContractError("integrated terminal P inventory differs")
    signed: list[float] = []
    groups: list[dict[str, Any]] = []
    forwards = 0
    tokens = 0
    for ordinal, batch in enumerate(anchor_microbatches):
        observed, receipt = evaluate_next_token_log_probs(model, tokenizer, batch)
        trial = samplewise_teacher_kl(teacher_log_probs[ordinal], observed)
        baseline = entry_kl[ordinal].detach().to(device="cpu", dtype=torch.float64)
        if trial.shape != baseline.shape or trial.shape != (BATCH_SIZE,):
            raise ODEBFContractError("integrated terminal P value geometry differs")
        delta = trial - baseline
        signed.extend(float(item) for item in delta)
        forwards += receipt.model_forward_count
        tokens += receipt.processed_token_count
        groups.append(
            {
                "ordinal": ordinal,
                "order_sha256": receipt.request_order_sha256,
                "teacher_sha256": tensor_sha256(teacher_log_probs[ordinal]),
                "entry_kl_sha256": tensor_sha256(baseline),
                "trial_kl_sha256": tensor_sha256(trial),
                "signed_delta_sha256": tensor_sha256(delta),
            }
        )
    array = np.asarray(signed, dtype=np.float64)
    if array.shape != (60,) or not np.all(np.isfinite(array)):
        raise ODEBFContractError("integrated terminal P values differ")
    payload = {
        "schema": f"{R14_TERMINAL_SCHEMA}-exact-functional-p",
        "anchor_count": 60,
        "groups": groups,
        "signed_delta": array.tolist(),
        "mean_signed": float(array.mean()),
        "mean_positive": float(np.maximum(array, 0.0).mean()),
        "maximum_positive": float(np.maximum(array, 0.0).max()),
        "model_forward_count": forwards,
        "processed_token_count": tokens,
        "decision_influence_count": 0,
        "terminal_only": True,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _paired_payload(native: IntegratedPrimaryEvaluation, ours: IntegratedPrimaryEvaluation) -> dict[str, Any]:
    # Reconstructing typed receipts from raw payload is deliberately avoided;
    # the exact paired bits are computed directly from already serialized bits.
    rows: dict[str, Any] = {}
    span_parity = all(
        native.primary[key] == ours.primary[key]
        for key in (
            "request_order_sha256",
            "evaluation_case_identity_sha256",
            "target_span_sha256",
            "evaluator_source_sha256",
            "aggregator_source_sha256",
        )
    )
    for metric in ("efficacy", "generalization", "locality"):
        left = native.primary["metrics"][metric]
        right = ours.primary["metrics"][metric]
        native_bits = tuple(tuple(item) for item in left["per_case_bits"])
        ours_bits = tuple(tuple(item) for item in right["per_case_bits"])
        if tuple(left["per_case_required"]) != tuple(right["per_case_required"]):
            raise ODEBFContractError("integrated paired primary denominator differs")
        wins = tuple(tuple(int(o == 1 and n == 0) for n, o in zip(nb, ob, strict=True)) for nb, ob in zip(native_bits, ours_bits, strict=True))
        losses = tuple(tuple(int(n == 1 and o == 0) for n, o in zip(nb, ob, strict=True)) for nb, ob in zip(native_bits, ours_bits, strict=True))
        rows[metric] = {
            "native_numerator": left["numerator"],
            "ours_numerator": right["numerator"],
            "denominator": left["denominator"],
            "win_count": sum(map(sum, wins)),
            "loss_count": sum(map(sum, losses)),
            "win_bits_sha256": canonical_hash([list(item) for item in wins]),
            "loss_bits_sha256": canonical_hash([list(item) for item in losses]),
        }
    payload = {"request_span_parity": span_parity, "metrics": rows}
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def run_integrated_terminal_panel(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    model_alias: str,
    requests: Sequence[Mapping[str, Any]],
    dataset_path: Path,
    hparams: Any,
    target_layer_name: str,
    final_target: torch.Tensor,
    factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    accepted_transition_count: int,
    action_freeze: IntegratedActionFreeze,
    p_anchor_microbatches: Sequence[Sequence[Mapping[str, Any]]],
    p_teacher_log_probs: Sequence[torch.Tensor],
    p_entry_kl: Sequence[torch.Tensor],
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    mutation_lock: threading.RLock,
    row_block: int = 64,
    residual_tolerance: float = 1.0e-8,
) -> IntegratedTerminalPanel:
    """Validate W0, the integrated endpoint, and Native after action freeze."""

    if not action_freeze.action_frozen:
        raise ODEBFContractError("integrated terminal panel preceded action freeze")
    order = ordered_request_digest_v1([str(item["request_sha256"]) for item in requests])
    if order != action_freeze.request_order_sha256:
        raise ODEBFContractError("integrated terminal action-freeze order differs")
    if action_freeze.accepted_transition_count != accepted_transition_count:
        raise ODEBFContractError(
            "integrated terminal accepted-transition freeze differs"
        )
    if (
        action_freeze.attempted_field_count < accepted_transition_count
        or action_freeze.attempted_field_count - accepted_transition_count > 1
    ):
        raise ODEBFContractError(
            "integrated terminal attempted-field freeze differs"
        )
    if accepted_transition_count == 0 and (
        action_freeze.attempted_field_count != 1
        or action_freeze.terminal_status != "NO_PHYSICAL_W_ONLY_DIRECTION"
    ):
        raise ODEBFContractError(
            "integrated zero-transition typed prefix differs"
        )
    freeze = IntegratedEndpointFreeze(
        "INTEGRATED_PHYSICAL_WRITER",
        order,
        action_freeze.selected_snapshot_sha256,
        accepted_transition_count,
    )
    from .p1_evaluator import load_counterfact_cases_after_freeze

    cases = load_counterfact_cases_after_freeze(dataset_path, requests, freeze)
    positions, patched_counts, lookup = integrated_heldout_lookup_geometry(
        tokenizer, requests, cases, fact_token_strategy=hparams.fact_token
    )
    w0 = evaluate_integrated_primary(
        model, tokenizer, cases, model_alias=model_alias, freeze=freeze
    )
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name in factors_by_weight
    }
    integrated_results: dict[str, Any] = {}

    def verify_integrated() -> bool:
        w_only = evaluate_integrated_primary(
            model, tokenizer, cases, model_alias=model_alias, freeze=freeze
        )
        current = capture_cold_z_base(model, tokenizer, requests, hparams)
        residual = (
            final_target.detach().to(device="cpu", dtype=torch.float32) - current
        ).contiguous()
        overlay = IntegratedHeldoutResidualOverlay(
            model, target_layer_name, residual, positions, patched_counts
        )
        oracle = evaluate_integrated_primary(
            model,
            tokenizer,
            cases,
            model_alias=model_alias,
            freeze=freeze,
            overlay=overlay,
        )
        exact_p = evaluate_exact_functional_p(
            model,
            tokenizer,
            p_anchor_microbatches,
            p_teacher_log_probs,
            p_entry_kl,
        )
        integrated_results.update(
            {"w_only": w_only, "z_oracle": oracle, "exact_p": exact_p}
        )
        return True

    if accepted_transition_count < 0:
        raise ODEBFContractError("integrated terminal transition count differs")
    if accepted_transition_count == 0:
        # A k0 NO_PHYSICAL_W_ONLY_DIRECTION is a scientific typed prefix, not
        # a transaction failure.  Its endpoint is W0, so evaluate the frozen
        # target oracle and Native panel without inventing a weight commit.
        if parameters or factors_by_weight:
            raise ODEBFContractError(
                "integrated zero-transition endpoint contains weight factors"
            )
        if not verify_integrated():
            raise ODEBFStateError("integrated zero-transition postverify failed")
        transaction_payload: dict[str, Any] = {
            "schema": f"{R14_TERMINAL_SCHEMA}-inactive-endpoint-transaction",
            "status": "INACTIVE_NO_CERTIFIED_WEIGHT_TRANSITION",
            "accumulated_step_count": 0,
            "transaction_commit_count": 0,
            "transaction_rollback_count": 0,
            "postverify_count": 1,
            "explicit_restore_count": 0,
            "final_w0_restore_exact": True,
            "persistent_commit_count": 0,
            "history_append_count": 0,
        }
        transaction_payload["identity_sha256"] = canonical_hash(
            transaction_payload
        )
    else:
        transaction = commit_verify_restore_integrated_endpoint(
            parameters,
            factors_by_weight,
            accumulated_step_count=accepted_transition_count,
            row_block=row_block,
            transaction_id="p1r14-integrated-endpoint",
            mutation_lock=mutation_lock,
            post_commit_verify=verify_integrated,
        )
        transaction_payload = asdict(transaction)
    integrated_w = integrated_results["w_only"]
    integrated_z = integrated_results["z_oracle"]
    integrated_p = integrated_results["exact_p"]
    if not isinstance(integrated_w, IntegratedPrimaryEvaluation) or not isinstance(
        integrated_z, IntegratedPrimaryEvaluation
    ):
        raise ODEBFStateError("integrated terminal postverify result differs")

    native_ledger = ComputeLedger()
    empty_history = {
        int(layer): torch.empty((projector.shape[1], 0), dtype=torch.float32)
        for layer in hparams.layers
    }
    native = capture_p1_native_entry(
        model,
        tokenizer,
        requests,
        hparams,
        projector,
        contexts,
        history_keys_by_layer=empty_history,
        mutation_lock=mutation_lock,
        ledger=native_ledger,
        residual_tolerance=residual_tolerance,
    )
    native_target = torch.stack(
        [item.detach().to(device="cpu", dtype=torch.float32).view(-1) for item in native.direct_z],
        dim=1,
    ).contiguous()
    with CandidateBF16FunctionalTrial(model, native.native_candidates):
        native_w = evaluate_integrated_primary(
            model, tokenizer, cases, model_alias=model_alias, freeze=freeze
        )
        native_current = capture_cold_z_base(model, tokenizer, requests, hparams)
        native_residual = (native_target - native_current).contiguous()
        native_overlay = IntegratedHeldoutResidualOverlay(
            model,
            target_layer_name,
            native_residual,
            positions,
            patched_counts,
        )
        native_z = evaluate_integrated_primary(
            model,
            tokenizer,
            cases,
            model_alias=model_alias,
            freeze=freeze,
            overlay=native_overlay,
        )
        native_p = evaluate_exact_functional_p(
            model,
            tokenizer,
            p_anchor_microbatches,
            p_teacher_log_probs,
            p_entry_kl,
        )
    paired = _paired_payload(native_w, integrated_w)
    terminal_compute = {
        "schema": f"{R14_TERMINAL_SCHEMA}-compute",
        "phase": "TERMINAL_AFTER_ACTION_FREEZE",
        "w0_primary_forward_count": 10,
        "integrated_primary_forward_count": 20,
        "integrated_exact_p_forward_count": 6,
        "native_primary_forward_count": 20,
        "native_exact_p_forward_count": 6,
        "native_internal_ledger": native_ledger.raw_free_payload(),
        "production_online_cost_inclusion_count": 0,
        "heldout_controller_access_count": 0,
    }
    payload = {
        "action_freeze_sha256": action_freeze.identity(),
        "w0": w0.raw_free_payload(),
        "integrated_w_only": integrated_w.raw_free_payload(),
        "integrated_z_oracle": integrated_z.raw_free_payload(),
        "native_w_only": native_w.raw_free_payload(),
        "native_z_oracle": native_z.raw_free_payload(),
        "integrated_exact_functional_p": integrated_p,
        "native_exact_functional_p": native_p,
        "native_receipt": native.raw_free_payload(),
        "paired_native_floor": paired,
        "endpoint_transaction": transaction_payload,
        "heldout_lookup": lookup,
        "terminal_compute": terminal_compute,
    }
    identity = canonical_hash(payload)
    return IntegratedTerminalPanel(
        action_freeze.identity(),
        w0,
        integrated_w,
        integrated_z,
        native_w,
        native_z,
        integrated_p,
        native_p,
        native.raw_free_payload(),
        paired,
        transaction_payload,
        lookup,
        terminal_compute,
        identity,
    )


__all__ = [
    "IntegratedEndpointFreeze",
    "IntegratedHeldoutResidualOverlay",
    "IntegratedPrimaryEvaluation",
    "IntegratedTerminalPanel",
    "evaluate_exact_functional_p",
    "evaluate_integrated_primary",
    "integrated_heldout_lookup_geometry",
    "run_integrated_terminal_panel",
]
