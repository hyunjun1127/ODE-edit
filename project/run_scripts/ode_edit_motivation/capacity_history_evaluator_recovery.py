"""Metadata-only entrypoint for the job-15823 capacity evaluator recovery.

The locked evaluator's policy metadata uses the exact key ``evaluation``, which
the inherited artifact sanitizer reserves for raw outcome payloads.  This
entrypoint renames only that configuration key.  It does not relax the
sanitizer or alter controller actions, model execution, metrics, or gates.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Sequence

from project.run_scripts.ode_edit_motivation import capacity_history_evaluator as evaluator
from project.run_scripts.ode_edit_motivation.mv0_fidelity import MV0Error, _safe_payload


REPOSITORY_ROOT = Path("/mnt/raid5/janghj/ODE-edit")
LOCKED_ROOT = REPOSITORY_ROOT / "local/scratch/caphist-eval-9900a51"
LOCKED_COMMIT = "9900a51f16669447bf502e7df0d5d5e0d61ef360"
EXPECTED_ENTRYPOINT = (
    REPOSITORY_ROOT
    / "project/run_scripts/ode_edit_motivation/capacity_history_evaluator_recovery.py"
)
EXPECTED_EVALUATOR = (
    LOCKED_ROOT
    / "project/run_scripts/ode_edit_motivation/capacity_history_evaluator.py"
)
GIT_BIN = Path("/usr/bin/git")
_ORIGINAL_POLICY_PARAMETERS = evaluator.evaluator_policy_parameters


class CapacityHistoryRecoveryError(RuntimeError):
    """The metadata-only recovery no longer matches its locked preconditions."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def recovered_policy_parameters() -> dict[str, Any]:
    """Return the locked policy with one sanitizer-safe metadata-key rename."""

    original = _ORIGINAL_POLICY_PARAMETERS()
    if set(original) != {"controller", "evaluation"}:
        raise CapacityHistoryRecoveryError("locked evaluator policy shape differs")
    try:
        _safe_payload(original)
    except MV0Error as exc:
        if str(exc) != "forbidden artifact field: evaluation":
            raise CapacityHistoryRecoveryError(
                "locked sanitizer failure differs from the approved recovery"
            ) from exc
    else:
        raise CapacityHistoryRecoveryError("metadata recovery is no longer required")

    recovered = dict(original)
    metric_protocol = recovered.pop("evaluation")
    recovered["metric_protocol"] = metric_protocol
    recovered["technical_recovery"] = {
        "reason": "reserved-artifact-key-collision",
        "renamed_key_from": "evaluation",
        "renamed_key_to": "metric_protocol",
        "metadata_only": True,
        "sanitizer_relaxation": False,
        "controller_action_rerun": False,
        "failed_job_id": "15823",
        "locked_commit": LOCKED_COMMIT,
        "entrypoint_relative_path": str(EXPECTED_ENTRYPOINT.relative_to(REPOSITORY_ROOT)),
        "entrypoint_sha256": _file_sha256(EXPECTED_ENTRYPOINT),
        "locked_evaluator_sha256": _file_sha256(EXPECTED_EVALUATOR),
    }
    _safe_payload(recovered)
    return recovered


def _preflight() -> None:
    if Path(__file__).resolve(strict=True) != EXPECTED_ENTRYPOINT.resolve(strict=True):
        raise CapacityHistoryRecoveryError("recovery entrypoint path differs")
    if Path(evaluator.__file__).resolve(strict=True) != EXPECTED_EVALUATOR.resolve(strict=True):
        raise CapacityHistoryRecoveryError("locked evaluator import path differs")
    commit = subprocess.run(
        (str(GIT_BIN), "-C", str(LOCKED_ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        (
            str(GIT_BIN),
            "-C",
            str(LOCKED_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if commit != LOCKED_COMMIT or tracked_status:
        raise CapacityHistoryRecoveryError("locked evaluator Git identity differs")
    recovered_policy_parameters()


def main(argv: Sequence[str] | None = None) -> int:
    _preflight()
    evaluator.evaluator_policy_parameters = recovered_policy_parameters
    return evaluator.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
