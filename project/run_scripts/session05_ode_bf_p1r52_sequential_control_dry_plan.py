#!/usr/bin/env python3
"""Model-free dry plan for the one-cell sequential control."""

from __future__ import annotations

import json

from project.run_scripts.ode_bf.p1r52_sequential_contract import HISTORY_COUNTS, dry_plan
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    R52_CONTROL_ROLE,
    expected_p1r52_sequential_result_name,
)


def main() -> int:
    payload = {
        "role": R52_CONTROL_ROLE,
        "result_name": expected_p1r52_sequential_result_name(
            "llama3-8b-inst", R52_CONTROL_ROLE
        ),
        "alpha_solve_history_width_at_entry": list(HISTORY_COUNTS),
        "structural_h_decision_history_width": [0] * len(HISTORY_COUNTS),
        "structural_h_decision_influence_count": 0,
        "physical_weight_persistence": True,
        "added_model_forward_backward_materialization": [0, 0, 0],
        "shared": dry_plan(),
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
