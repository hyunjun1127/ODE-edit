"""Raw-free receipts for the P1R4 full-residual arm diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

from .contracts import FIXED_K, ODEBFContractError


DIAGNOSTIC_INSTRUCTION_ID = (
    "ODEEDIT-S04-ODE-BF-FULL-RESIDUAL-ARMS-P1R4-V1"
)
TRIALS_PER_SLOT = 3
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARM_LOCAL_STATUS = {
    "structural_h": "STRUCTURAL_H_INFEASIBLE",
    "structural_p": "STRUCTURAL_P_INFEASIBLE",
    "trust": "TRUST_INFEASIBLE",
    "terminal_h": "TERMINAL_H_INFEASIBLE",
    "terminal_p": "TERMINAL_P_INFEASIBLE",
}
_FORBIDDEN_KEY_FRAGMENTS = (
    "prompt",
    "subject",
    "target_text",
    "target_new",
    "target_true",
    "anchor_text",
    "logit",
    "tensor",
    "token_id",
    "context_text",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_raw_free(value: Any, *, key: str = "root") -> None:
    lowered = key.casefold()
    if any(fragment in lowered for fragment in _FORBIDDEN_KEY_FRAGMENTS):
        raise ODEBFContractError("diagnostic receipt contains a forbidden raw key")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ODEBFContractError("diagnostic receipt contains a non-finite float")
        return
    if isinstance(value, str):
        if any(ord(character) < 0x20 for character in value):
            raise ODEBFContractError("diagnostic receipt contains a control character")
        if key.endswith("sha256") and _SHA256.fullmatch(value) is None:
            raise ODEBFContractError("diagnostic receipt digest differs")
        return
    if isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str):
                raise ODEBFContractError("diagnostic receipt key is not a string")
            _validate_raw_free(nested_value, key=nested_key)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_raw_free(item, key=key)
        return
    raise ODEBFContractError("diagnostic receipt contains an unsupported value")


def _atomic_write_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("diagnostic receipt is create-once")
    _validate_raw_free(value)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("diagnostic temporary receipt exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def first_false_terminal_component(inputs: Mapping[str, bool]) -> str | None:
    order = (
        "structural_h",
        "structural_p",
        "trust",
        "terminal_h",
        "terminal_p",
    )
    if set(inputs) != set(order) or any(
        not isinstance(inputs[name], bool) for name in order
    ):
        raise ODEBFContractError("terminal diagnostic boolean inputs differ")
    return next((name for name in order if not inputs[name]), None)


class P1DiagnosticRecorder:
    """One-file-per-event writer: any interruption leaves a valid prefix."""

    _TRIAL_KEYS = {
        "beta",
        "field_sha256",
        "raw_velocity_sha256",
        "proposal_sha256",
        "entry_snapshot_sha256",
        "trial_snapshot_sha256",
        "predicted_beta_progress",
        "actual_signed_progress",
        "trust_ratio",
        "progress_pass",
        "requested_progress",
        "maximum_feasible_progress",
        "solver",
        "structural_h",
        "structural_p",
        "trust",
        "functional_h",
        "functional_p",
        "authoritative_bf16_pass",
        "first_rejecting_component",
        "accepted",
        "official_success",
        "progress_telemetry",
        "purity",
    }
    _SLOT_KEYS = {
        "accepted",
        "accepted_beta",
        "accepted_t",
        "selected_snapshot_sha256",
        "rejected_field_reuse",
        "trial_receipt_sha256",
        "state_before_sha256",
        "state_after_sha256",
        "history_before_sha256",
        "history_after_sha256",
        "sampler_before_sha256",
        "sampler_after_sha256",
    }
    _TERMINAL_KEYS = {
        "selected_stage",
        "selected_snapshot_sha256",
        "structural_h",
        "structural_p",
        "trust",
        "terminal_h",
        "terminal_p",
        "boolean_inputs",
        "first_false_component",
        "selector",
        "slot_receipt_sha256",
        "arm_local_infeasibility",
    }

    def __init__(self, root: Path, *, arm: str) -> None:
        if arm not in ("F_G", "F_BF", "R_BF"):
            raise ODEBFContractError("terminal diagnostic arm differs")
        if root.exists() or root.is_symlink():
            raise FileExistsError("diagnostic receipt root is create-once")
        root.mkdir(mode=0o700, parents=True)
        self.root = root
        self.arm = arm
        self._trial_hashes: dict[tuple[int, int], str] = {}
        self._slot_hashes: dict[int, str] = {}
        self._terminal_hash: str | None = None

    def write_trial(
        self,
        *,
        slot_index: int,
        trial_ordinal: int,
        payload: Mapping[str, Any],
    ) -> str:
        if set(payload) != self._TRIAL_KEYS:
            raise ODEBFContractError("diagnostic trial schema differs")
        if slot_index not in range(FIXED_K) or trial_ordinal not in range(
            TRIALS_PER_SLOT
        ):
            raise ODEBFContractError("diagnostic trial coordinate differs")
        expected_ordinal = len(self._trial_hashes)
        if expected_ordinal != slot_index * TRIALS_PER_SLOT + trial_ordinal:
            raise ODEBFContractError("diagnostic trials are not a contiguous prefix")
        value = {
            "schema": "ode-edit-s04-ode-bf-p1r4diag-trial/v1",
            "instruction_id": DIAGNOSTIC_INSTRUCTION_ID,
            "arm": self.arm,
            "slot_index": slot_index,
            "stage": slot_index + 1,
            "trial_ordinal": trial_ordinal,
            **dict(payload),
        }
        path = self.root / (
            f"arm-{self.arm}-slot-{slot_index:02d}-trial-{trial_ordinal:02d}.json"
        )
        digest = _atomic_write_once(path, value)
        self._trial_hashes[(slot_index, trial_ordinal)] = digest
        return digest

    def write_slot(self, *, slot_index: int, payload: Mapping[str, Any]) -> str:
        if set(payload) != self._SLOT_KEYS:
            raise ODEBFContractError("diagnostic slot schema differs")
        if slot_index != len(self._slot_hashes) or slot_index not in range(FIXED_K):
            raise ODEBFContractError("diagnostic slots are not a contiguous prefix")
        expected = {
            self._trial_hashes.get((slot_index, trial))
            for trial in range(TRIALS_PER_SLOT)
        }
        if None in expected or payload["trial_receipt_sha256"] != [
            self._trial_hashes[(slot_index, trial)]
            for trial in range(TRIALS_PER_SLOT)
        ]:
            raise ODEBFContractError("diagnostic slot trial links differ")
        value = {
            "schema": "ode-edit-s04-ode-bf-p1r4diag-slot/v1",
            "instruction_id": DIAGNOSTIC_INSTRUCTION_ID,
            "arm": self.arm,
            "slot_index": slot_index,
            "stage": slot_index + 1,
            **dict(payload),
        }
        path = self.root / f"arm-{self.arm}-slot-{slot_index:02d}-summary.json"
        digest = _atomic_write_once(path, value)
        self._slot_hashes[slot_index] = digest
        return digest

    def write_terminal(self, payload: Mapping[str, Any]) -> str:
        if set(payload) != self._TERMINAL_KEYS:
            raise ODEBFContractError("terminal component receipt schema differs")
        if len(self._trial_hashes) != FIXED_K * TRIALS_PER_SLOT or len(
            self._slot_hashes
        ) != FIXED_K:
            raise ODEBFContractError("terminal component receipt lacks K8 telemetry")
        if payload["slot_receipt_sha256"] != [
            self._slot_hashes[slot] for slot in range(FIXED_K)
        ]:
            raise ODEBFContractError("terminal component slot links differ")
        observed_first = first_false_terminal_component(payload["boolean_inputs"])
        if payload["first_false_component"] != observed_first:
            raise ODEBFContractError("terminal first-false ordering differs")
        arm_local = payload["arm_local_infeasibility"]
        if observed_first is None:
            if arm_local is not None:
                raise ODEBFContractError("passing terminal has infeasibility payload")
        elif (
            not isinstance(arm_local, Mapping)
            or arm_local.get("status") != _ARM_LOCAL_STATUS[observed_first]
            or arm_local.get("first_false_component") != observed_first
            or arm_local.get("component_vector") != payload["boolean_inputs"]
        ):
            raise ODEBFContractError("arm-local terminal classification differs")
        value = {
            "schema": "ode-edit-s04-ode-bf-p1r4diag-terminal-components/v1",
            "instruction_id": DIAGNOSTIC_INSTRUCTION_ID,
            "arm": self.arm,
            **dict(payload),
        }
        digest = _atomic_write_once(
            self.root / f"arm-{self.arm}-terminal-components.json", value
        )
        self._terminal_hash = digest
        return digest

    @property
    def trial_hashes(self) -> list[str]:
        return [
            self._trial_hashes[(slot, trial)]
            for slot in range(FIXED_K)
            for trial in range(TRIALS_PER_SLOT)
            if (slot, trial) in self._trial_hashes
        ]

    @property
    def slot_hashes(self) -> list[str]:
        return [
            self._slot_hashes[slot]
            for slot in range(FIXED_K)
            if slot in self._slot_hashes
        ]

    @property
    def terminal_hash(self) -> str | None:
        return self._terminal_hash


def diagnostic_receipt_links(raw_root: Path) -> dict[str, str]:
    """Hash only finalized diagnostic JSON; temporary files are never linked."""

    paths = sorted(raw_root.glob("diagnostic-*.json"))
    diagnostic_root = raw_root / "diagnostics"
    if diagnostic_root.exists():
        if diagnostic_root.is_symlink() or not diagnostic_root.is_dir():
            raise ODEBFContractError("diagnostic receipt root identity differs")
        paths.extend(sorted(diagnostic_root.rglob("*.json")))
    result: dict[str, str] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError("diagnostic receipt is not a regular file")
        relative = path.relative_to(raw_root).as_posix()
        result[relative] = _sha256_file(path)
    return result
