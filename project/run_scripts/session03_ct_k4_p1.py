#!/usr/bin/env python3
"""Session 03 CT-K4 fresh four-case endpoint-evaluation stage."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

from project.run_scripts.session03_ct_k4_common import build_parser, run


if __name__ == "__main__":
    raise SystemExit(run(build_parser("p1").parse_args()))
