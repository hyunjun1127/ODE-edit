#!/usr/bin/env python3
"""Print the model-free P1R29 Sequential Stage-A dry plan."""

from __future__ import annotations

import json

from project.run_scripts.ode_bf.p1r29_sequential_preparation import stage_a_dry_plan


def main() -> int:
    print(json.dumps(stage_a_dry_plan(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

