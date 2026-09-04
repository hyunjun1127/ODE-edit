"""Configure the reusable round-0 analyzer for the canonical-NS rerun.

The scientific/mechanism analyzer remains shared with the immutable v1
package.  This thin binding changes only the sealed execution lineage and the
v2 evaluator inventory, where neighborhood target-new and target-true NLLs
are both present and canonical NS has exactly 1,000 prompt pairs per endpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import round0_analysis as analysis


INSTRUCTION_ID = "ODEEDIT-S06-ORRBODE-ROUND0-CANONICAL-NS-RERUN-20260905-V4"
SOURCE_HEAD = "15fd9579821695270c8d7a3e9b8759f9cd360d93"
SOURCE_TREE = "2af82e059054276f123b65a85a93799f67688e06"
KIND_COUNTS = {
    "rewrite_target_new": 100,
    "rewrite_target_true": 100,
    "rephrase_target_new": 200,
    "rephrase_target_true": 200,
    "locality_target_new": 1000,
    "locality_target_true": 1000,
}
REFERENCE_FILES = (
    "experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-exhaustive-2026-09-04-v1/core-performance-summary.csv",
    "experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-exhaustive-2026-09-04-v1/ordered-response-barrier-ode-round0-b100-exhaustive-factual-ko.md",
    "experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-exhaustive-2026-09-04-v1/analysis-manifest.json",
    "experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-exhaustive-2026-09-04-v1/rooted-analysis-receipt.json",
    "experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-baseline-inclusive-2026-09-04-v2/baseline-inclusive-performance.csv",
    "experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-baseline-inclusive-2026-09-04-v2/ordered-response-barrier-ode-round0-b100-baseline-inclusive-factual-ko.md",
    "experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-baseline-inclusive-2026-09-04-v2/analysis-manifest.json",
    "experiment-reports/servers/server1/ordered-response-barrier-ode-round0-b100-baseline-inclusive-2026-09-04-v2/rooted-analysis-receipt.json",
)


def configure(*, parent_job: str, child_jobs: tuple[str, str, str, str], raw_namespace: str) -> None:
    bindings = {
        0: ("llama3-8b-inst", "MEMIT", child_jobs[0]),
        1: ("llama3-8b-inst", "AlphaEdit", child_jobs[1]),
        2: ("qwen2.5-7b-inst", "MEMIT", child_jobs[2]),
        3: ("qwen2.5-7b-inst", "AlphaEdit", child_jobs[3]),
    }
    analysis.INSTRUCTION_ID = INSTRUCTION_ID
    analysis.NONCE = INSTRUCTION_ID
    analysis.RAW_SOURCE_HEAD = SOURCE_HEAD
    analysis.RAW_SOURCE_TREE = SOURCE_TREE
    analysis.ROUND_JOB_ID = parent_job
    analysis.CELL_BINDINGS = bindings
    analysis.KIND_COUNTS = dict(KIND_COUNTS)
    analysis.KIND_ORDER = tuple(KIND_COUNTS)
    analysis.EXPECTED_EVALUATION_ROWS = sum(KIND_COUNTS.values())
    analysis.REFERENCE_FILES = REFERENCE_FILES
    analysis.REFERENCE_CORE_PERFORMANCE_FILE = REFERENCE_FILES[0]
    analysis.LINEAGE = (
        ("36603", "round0-tech-r1", SOURCE_HEAD, "TECHNICAL_ATTEMPT"),
        ("36415", "b1-tech-r1", SOURCE_HEAD, "B1_PILOT"),
        (parent_job, raw_namespace, SOURCE_HEAD, "CANONICAL_B100"),
    )
    analysis.INCLUDE_LEGACY_DRY_PLAN = False
    analysis.SOURCE_FILES = tuple(
        dict.fromkeys(
            analysis.SOURCE_FILES
            + (
                "project/run_scripts/ordered_response_barrier_ode/round0_canonical_ns_analysis.py",
                "project/run_scripts/ordered_response_barrier_ode/counterfact_locality_evaluator.py",
                "project/run_scripts/ordered_response_barrier_ode/artifacts.py",
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--log-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repro-root", type=Path, required=True)
    parser.add_argument("--parent-job", required=True)
    parser.add_argument("--child-jobs", nargs=4, required=True)
    args = parser.parse_args()
    configure(
        parent_job=args.parent_job,
        child_jobs=tuple(args.child_jobs),
        raw_namespace=args.raw_root.name,
    )
    result = analysis.build(
        args.repo,
        args.raw_root,
        args.log_base,
        args.output,
        args.repro_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
