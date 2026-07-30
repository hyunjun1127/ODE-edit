"""Precommitted MV-1 score-mix follow-up runner.

Only two outcome-blind modes exist:

``fold1``
    The fixed hash fold 1 of the original confirmatory 60.  Operationally this
    may be opened once after a gray or architecture-conditional C1 review.

``untouched``
    The exact untouched 20 from the canonical selection manifest.  It may be
    opened after C1 clear, without changing the controller, policies, or q.

The module deliberately contains no C1-result reader or decision logic.  It
reuses the C1 event core, which builds direct-z, synchronous/ordered proposals,
and the six central-FD probes once per event; commits the frozen static and
calibration-only forecast actions before outcomes; evaluates the exact six-arm
order at q=1/256; and retains the existing rollback and artifact firewall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .gpu_runtime import FixedModelRuntime, load_fixed_model
from .manifests import (
    DEFAULT_SELECTION_SEED,
    MODEL_SPECS,
    CounterFactSelectionManifest,
)
from .mv0_fidelity import DEFAULT_OUTPUT_ROOT
from .mv1_analysis import ConfirmatoryFoldManifest
from .mv1_calibration import MV1Error
from .mv1_score_mix_confirmatory import (
    CONFIRMATORY_POLICY_PATHS,
    ScoreMixWaveLock,
    _run_confirmatory_event,
    _score_mix_execution_envelope,
    _score_mix_slurm_state,
    run_score_mix_wave,
    select_score_mix_wave_cases,
)


FOLLOWUP_RUN_SEED = 17
FOLLOWUP_MODES = ("fold1", "untouched")
FOLLOWUP_JOB_NAMES = MappingProxyType(
    {
        "fold1": "odeedit_mv1mix_fold1_pair_v1",
        "untouched": "odeedit_mv1mix_untouched_pair_v1",
    }
)
FOLLOWUP_RUN_IDS = MappingProxyType(
    {
        "fold1": MappingProxyType(
            {
                "llama3-8b-inst": "mv1mix_llama_fold1_v1",
                "qwen2.5-7b-inst": "mv1mix_qwen_fold1_v1",
            }
        ),
        "untouched": MappingProxyType(
            {
                "llama3-8b-inst": "mv1mix_llama_untouched_v1",
                "qwen2.5-7b-inst": "mv1mix_qwen_untouched_v1",
            }
        ),
    }
)
FOLLOWUP_WAVE_LOCKS = MappingProxyType(
    {
        "fold1": ScoreMixWaveLock(
            label="fold1",
            selected_split="confirmatory",
            fold=1,
            case_count=12,
            job_name=FOLLOWUP_JOB_NAMES["fold1"],
            run_ids=FOLLOWUP_RUN_IDS["fold1"],
            run_seed=FOLLOWUP_RUN_SEED,
        ),
        "untouched": ScoreMixWaveLock(
            label="untouched",
            selected_split="untouched",
            fold=None,
            case_count=20,
            job_name=FOLLOWUP_JOB_NAMES["untouched"],
            run_ids=FOLLOWUP_RUN_IDS["untouched"],
            run_seed=FOLLOWUP_RUN_SEED,
        ),
    }
)

# Both modes reuse the exact tracked D0+D1 locks.  This alias is intentionally
# read-only and exists to make that precommit visible to callers and tests.
FOLLOWUP_POLICY_PATHS = MappingProxyType(
    {
        model_alias: MappingProxyType(dict(paths))
        for model_alias, paths in CONFIRMATORY_POLICY_PATHS.items()
    }
)


def followup_wave_lock(mode: str) -> ScoreMixWaveLock:
    if not isinstance(mode, str) or mode not in FOLLOWUP_MODES:
        raise MV1Error("follow-up mode must be exactly fold1 or untouched")
    return FOLLOWUP_WAVE_LOCKS[mode]


def select_followup_cases(
    selection: CounterFactSelectionManifest,
    *,
    mode: str,
) -> tuple[tuple[str, ...], ConfirmatoryFoldManifest]:
    """Select the exact IDs for one of the two precommitted modes."""

    return select_score_mix_wave_cases(
        selection,
        wave=followup_wave_lock(mode),
    )


def _followup_execution_envelope(
    model_alias: str,
    run_id: str,
    *,
    mode: str,
) -> str:
    return _score_mix_execution_envelope(
        model_alias,
        run_id,
        wave=followup_wave_lock(mode),
    )


def _followup_slurm_state(
    model_alias: str,
    run_id: str,
    *,
    mode: str,
) -> dict[str, Any]:
    return _score_mix_slurm_state(
        model_alias,
        run_id,
        wave=followup_wave_lock(mode),
    )


def run_mv1_score_mix_followup(
    *,
    mode: str,
    easyedit_root: str | Path,
    model_alias: str,
    run_id: str,
    static_policy_path: str | Path,
    forecast_policy_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    selection_seed: str = DEFAULT_SELECTION_SEED,
    model_loader: Callable[[str], FixedModelRuntime] = load_fixed_model,
    event_runner: Callable[..., Mapping[str, Any]] = _run_confirmatory_event,
) -> dict[str, Any]:
    """Run one exact follow-up without reading any earlier-wave outcome."""

    wave = followup_wave_lock(mode)
    return run_score_mix_wave(
        wave=wave,
        easyedit_root=easyedit_root,
        model_alias=model_alias,
        run_id=run_id,
        static_policy_path=static_policy_path,
        forecast_policy_path=forecast_policy_path,
        output_root=output_root,
        seed=FOLLOWUP_RUN_SEED,
        selection_seed=selection_seed,
        model_loader=model_loader,
        event_runner=event_runner,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-mv1-score-mix-followup",
        description="Run one precommitted fold1 or untouched MV-1 follow-up.",
    )
    parser.add_argument("--mode", required=True, choices=FOLLOWUP_MODES)
    parser.add_argument("--easyedit-root", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--static-policy", required=True, type=Path)
    parser.add_argument("--forecast-policy", required=True, type=Path)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_mv1_score_mix_followup(
            mode=args.mode,
            easyedit_root=args.easyedit_root,
            model_alias=args.model,
            run_id=args.run_id,
            static_policy_path=args.static_policy,
            forecast_policy_path=args.forecast_policy,
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
