"""Observation-only accepted-z CounterFact rewrite/rephrase evaluation.

The scientific edit remains owned by the frozen P1R52/Official sequential
runtimes.  This module only captures a method-native accepted target in memory
and evaluates it after the corresponding physical batch action is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence

import torch

from .bg_soft_diagnostics import _rewrap_hook_output, _unwrap_hook_output
from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .p1_evaluator import (
    CounterFactEvaluationCase,
    EndpointActionFreeze,
    evaluate_counterfact_success_accuracy_batch,
)
from .scalable_batched_model import _batch_axis_zero, _restore_batch_axis
from .scalable_batched_runtime import scalable_ordered_request_digest


INSTRUCTION_ID = "ODEEDIT-S05-P1R52-LLAMA-B100-ACCEPTED-Z-REPHRASE-OBS-V1"


@dataclass(frozen=True, slots=True)
class AcceptedZBinding:
    role: str
    source: str
    layer: int
    module_name: str
    fact_token: str
    request_order_sha256: str
    accepted_z: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.layer != 8
            or not self.module_name
            or self.fact_token != "subject_last"
            or len(self.request_order_sha256) != 64
            or self.accepted_z.layout != torch.strided
            or self.accepted_z.ndim != 2
            or self.accepted_z.shape[1] != 100
            or not self.accepted_z.is_floating_point()
            or not bool(torch.isfinite(self.accepted_z).all())
        ):
            raise ODEBFContractError("accepted-z native binding differs")

    def raw_free_payload(self) -> dict[str, Any]:
        payload = {
            "role": self.role,
            "source": self.source,
            "layer": self.layer,
            "module_name": self.module_name,
            "fact_token": self.fact_token,
            "request_order_sha256": self.request_order_sha256,
            "accepted_z_sha256": tensor_sha256(self.accepted_z),
            "accepted_z_shape": list(self.accepted_z.shape),
            "accepted_z_dtype": str(self.accepted_z.dtype),
            "native_definition_preserved": True,
            "proxy_or_imputation_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


class OfficialNativeZCapture:
    """Capture exact EasyEdit ``compute_z`` returns without changing them."""

    def __init__(
        self,
        *,
        role: str,
        requests: Sequence[Mapping[str, Any]],
        hparams: Any,
    ) -> None:
        self.role = role
        self.requests = tuple(requests)
        self.hparams = hparams
        self._module: Any | None = None
        self._original: Any | None = None
        self._values: list[torch.Tensor] = []
        self._request_sha256: list[str] = []
        self.binding: AcceptedZBinding | None = None

    def __enter__(self) -> "OfficialNativeZCapture":
        if self.role == "official-memit-sequential":
            from easyeditor.models.memit import memit_main as native_module

            source = "easyeditor.models.memit.memit_main.compute_z"
        elif self.role == "native-alphaedit-sequential-cache-on-corrected":
            from easyeditor.models.alphaedit import AlphaEdit_main as native_module

            source = "easyeditor.models.alphaedit.AlphaEdit_main.compute_z"
        else:
            raise ODEBFContractError("accepted-z Official role differs")
        original = native_module.compute_z

        def capture(*args: Any, **kwargs: Any) -> torch.Tensor:
            value = original(*args, **kwargs)
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise ODEBFContractError("Official accepted-z return differs")
            request = args[2] if len(args) > 2 else kwargs.get("request")
            if not isinstance(request, Mapping):
                raise ODEBFContractError("Official accepted-z request differs")
            self._values.append(value.detach().to(device="cpu").contiguous().clone())
            self._request_sha256.append(str(request["request_sha256"]))
            return value

        self._module = native_module
        self._original = original
        self._source = source
        native_module.compute_z = capture
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, exc, traceback
        if self._module is not None and self._original is not None:
            self._module.compute_z = self._original
        self._module = None
        self._original = None
        return False

    def finalize(self) -> AcceptedZBinding:
        expected = [str(item["request_sha256"]) for item in self.requests]
        if self._module is not None or self._request_sha256 != expected or len(self._values) != 100:
            raise ODEBFStateError("Official accepted-z capture order/count differs")
        accepted = torch.stack(self._values, dim=1)
        layer = int(self.hparams.layers[-1])
        self.binding = AcceptedZBinding(
            self.role,
            self._source,
            layer,
            str(self.hparams.layer_module_tmp).format(layer),
            str(self.hparams.fact_token),
            scalable_ordered_request_digest(expected),
            accepted,
        )
        return self.binding


def r52_binding(
    role: str,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    terminal_target: torch.Tensor,
) -> AcceptedZBinding:
    layer = int(hparams.layers[-1])
    return AcceptedZBinding(
        role,
        "p1r52_k8_rollout.terminal_target",
        layer,
        str(hparams.layer_module_tmp).format(layer),
        str(hparams.fact_token),
        scalable_ordered_request_digest([str(item["request_sha256"]) for item in requests]),
        terminal_target.detach().to(device="cpu").contiguous().clone(),
    )


def _lookup_geometry(
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    cases: Sequence[CounterFactEvaluationCase],
    *,
    fact_token: str,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], dict[str, Any]]:
    from easyeditor.models.rome import repr_tools

    if len(requests) != 100 or len(cases) != 100:
        raise ODEBFContractError("accepted-z B100 lookup geometry differs")
    padding = tokenizer.padding_side
    if padding not in ("left", "right"):
        raise ODEBFContractError("accepted-z tokenizer padding differs")
    positions_by_request: list[tuple[int, ...]] = []
    prefix_counts: list[int] = []
    identities: list[dict[str, Any]] = []
    for ordinal, (request, case) in enumerate(zip(requests, cases, strict=True)):
        if str(request["request_sha256"]) != case.request_sha256:
            raise ODEBFContractError("accepted-z case/request order differs")
        subject = str(request["subject"])
        patched_prefixes = (case.rewrite_prompt,) + case.paraphrase_prompts
        all_prefixes = patched_prefixes + case.neighborhood_prompts
        raw_positions: list[int] = []
        prefix_lengths: list[int] = []
        for prefix in patched_prefixes:
            if prefix.count(subject) != 1:
                raise ODEBFContractError("accepted-z heldout subject surface differs")
            template = prefix.replace(subject, "{}", 1)
            if fact_token != "subject_last":
                raise ODEBFContractError("accepted-z native lookup strategy differs")
            # This is the exact source kernel used by EasyEdit's
            # find_fact_lookup_idx before its observation-only sentence
            # formatting.  Calling it directly preserves literal braces in a
            # CounterFact paraphrase (for example ``{name/pronoun}``).
            raw_positions.append(
                int(
                    repr_tools.get_words_idxs_in_templates(
                        tok=tokenizer,
                        context_templates=[template],
                        words=[subject],
                        subtoken="last",
                    )[0][0]
                )
            )
            prefix_lengths.append(len(tokenizer(prefix)["input_ids"]))
        rows = [
            f"{prefix} {suffix}"
            for prefix in all_prefixes
            for suffix in (case.target_new, case.target_true)
        ]
        encoded = tokenizer(rows, padding=True, return_tensors="pt")
        attention = encoded["attention_mask"]
        row_positions: list[int] = []
        for row_index in range(2 * len(patched_prefixes)):
            prefix_index = row_index // 2
            pad_count = int(attention.shape[1] - attention[row_index].sum())
            left_pad = pad_count if padding == "left" else 0
            raw = raw_positions[prefix_index]
            position = left_pad + (prefix_lengths[prefix_index] - 1 if raw == -1 else raw)
            if position < 0 or position >= int(attention.shape[1]):
                raise ODEBFContractError("accepted-z lookup position is out of range")
            row_positions.append(position)
        row_positions.extend(0 for _ in range(len(rows) - len(row_positions)))
        positions_by_request.append(tuple(row_positions))
        prefix_counts.append(2 * len(patched_prefixes))
        identities.append(
            {
                "request_ordinal": ordinal,
                "request_sha256": case.request_sha256,
                "patched_prefix_count": len(patched_prefixes),
                "all_prefix_count": len(all_prefixes),
                "lookup_sha256": canonical_hash(row_positions[: 2 * len(patched_prefixes)]),
            }
        )
    receipt = {
        "request_count": 100,
        "fact_token": fact_token,
        "padding_side": padding,
        "lookup_kernel": (
            "easyeditor.models.rome.repr_tools.get_words_idxs_in_templates:subject_last"
        ),
        "geometry_sha256": canonical_hash(identities),
        "patched_row_count": sum(prefix_counts),
        "locality_unpatched_row_count": sum(
            len(rows) - prefix
            for rows, prefix in zip(positions_by_request, prefix_counts, strict=True)
        ),
    }
    receipt["identity_sha256"] = canonical_hash(receipt)
    return tuple(positions_by_request), tuple(prefix_counts), receipt


class AbsoluteAcceptedZActivationOverlay:
    """Replace only rewrite/rephrase lookup activations with native accepted z."""

    def __init__(
        self,
        model: torch.nn.Module,
        binding: AcceptedZBinding,
        positions: Sequence[Sequence[int]],
        prefix_counts: Sequence[int],
    ) -> None:
        self.model = model
        self.binding = binding
        self.positions = tuple(tuple(int(value) for value in row) for row in positions)
        self.prefix_counts = tuple(int(value) for value in prefix_counts)
        if len(self.positions) != 100 or len(self.prefix_counts) != 100:
            raise ODEBFContractError("accepted-z overlay request geometry differs")
        self.calls = 0
        self.patched_rows = 0
        self.locality_rows = 0
        self.max_assignment_error = 0.0
        self._handle: Any | None = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if self.calls >= 100:
            raise ODEBFContractError("accepted-z overlay received an extra forward")
        activation = _unwrap_hook_output(output)
        positions = self.positions[self.calls]
        prefix_count = self.prefix_counts[self.calls]
        batch_first = _batch_axis_zero(activation, len(positions))
        z = self.binding.accepted_z[:, self.calls].to(
            device=batch_first.device, dtype=batch_first.dtype
        )
        if z.numel() != batch_first.shape[-1] or not bool(torch.isfinite(z).all()):
            raise ODEBFContractError("accepted-z overlay hidden geometry differs")
        patched = batch_first.clone()
        for row_index in range(prefix_count):
            position = positions[row_index]
            patched[row_index, position, :] = z
            error = float(torch.max(torch.abs(patched[row_index, position, :].float() - z.float())))
            self.max_assignment_error = max(self.max_assignment_error, error)
        if not torch.equal(patched[prefix_count:], batch_first[prefix_count:]):
            raise ODEBFContractError("accepted-z overlay touched locality rows")
        self.patched_rows += prefix_count
        self.locality_rows += len(positions) - prefix_count
        self.calls += 1
        return _rewrap_hook_output(output, _restore_batch_axis(patched, activation))

    def __enter__(self) -> "AbsoluteAcceptedZActivationOverlay":
        self._handle = self.model.get_submodule(self.binding.module_name).register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, exc, traceback
        if self._handle is not None:
            self._handle.remove()
        self._handle = None
        return False

    def raw_free_payload(self) -> dict[str, Any]:
        if self._handle is not None or self.calls != 100:
            raise ODEBFStateError("accepted-z overlay call/cleanup differs")
        payload = {
            "mode": "METHOD_NATIVE_ABSOLUTE_ACCEPTED_Z_REPLACEMENT",
            "hook_call_count": self.calls,
            "patched_row_count": self.patched_rows,
            "locality_unpatched_row_count": self.locality_rows,
            "maximum_assignment_error": self.max_assignment_error,
            "added_backward_count": 0,
            "added_generation_call_count": 0,
            "controller_router_history_cache_selection_materialization_influence_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


def evaluate_accepted_z_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    cases: Sequence[CounterFactEvaluationCase],
    *,
    role: str,
    round_index: int,
    binding: AcceptedZBinding,
    committed_weight_sha256: Mapping[str, str],
    model_alias: str = "llama3-8b-inst",
) -> tuple[dict[str, Any], float]:
    order = scalable_ordered_request_digest([str(item["request_sha256"]) for item in requests])
    if order != binding.request_order_sha256:
        raise ODEBFContractError("accepted-z observation request order differs")
    positions, prefix_counts, lookup = _lookup_geometry(
        tokenizer, requests, cases, fact_token=binding.fact_token
    )
    freeze = EndpointActionFreeze(
        arm=f"{role}-accepted-z-observation",
        sequential_batch=round_index - 1,
        request_order_sha256=order,
        selected_snapshot_sha256=canonical_hash(dict(committed_weight_sha256)),
        fixed_budget_slots_completed=8 if role.startswith("r52-") else 0,
    )
    overlay = AbsoluteAcceptedZActivationOverlay(model, binding, positions, prefix_counts)
    started = time.perf_counter()
    with overlay:
        evaluated = evaluate_counterfact_success_accuracy_batch(
            model,
            tokenizer,
            cases,
            model_alias=model_alias,
            freeze=freeze,
            expected_batch_size=100,
        )
    wall = time.perf_counter() - started
    scores = evaluated.raw_free_payload()
    payload = {
        "schema": "ode-edit-s05-p1r52-b100-accepted-z-rephrase-observation/v1",
        "instruction_id": INSTRUCTION_ID,
        "role": role,
        "round": round_index,
        "action_freeze_sha256": freeze.identity(),
        "binding": binding.raw_free_payload(),
        "lookup": lookup,
        "overlay": overlay.raw_free_payload(),
        "scores": scores,
        "model_forward_count": int(scores["legacy_primary"]["model_forward_count"]),
        "processed_token_count": int(scores["legacy_primary"]["processed_token_count"]),
        "added_backward_count": 0,
        "added_generation_call_count": 0,
        "batch_entry_W_metrics_count": 0,
        "wall_seconds": wall,
        "observation_only": True,
        "action_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload, wall

__all__ = [
    "AcceptedZBinding",
    "AbsoluteAcceptedZActivationOverlay",
    "INSTRUCTION_ID",
    "OfficialNativeZCapture",
    "evaluate_accepted_z_batch",
    "r52_binding",
]
