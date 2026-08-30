"""AST and physical ctrl/gate namespace leakage firewall."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from .contracts import TechnicalBoundary
from .hashing import canonical_hash, file_sha256


FORBIDDEN_PREFIXES = (
    "project.run_scripts.ode_",
    "project.run_scripts.ode_bf",
    "project.run_scripts.p1",
    "project.run_scripts.p4",
    "project.run_scripts.alphaedit_strength_neutral_barrier",
    "project.run_scripts.cache_aware",
)


def forbidden_imports(root: Path) -> list[str]:
    violations = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [row.name for row in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if name.startswith(FORBIDDEN_PREFIXES):
                    violations.append(f"{path}:{node.lineno}:{name}")
    return violations


def seal_selector_lock(path: Path, payload: dict[str, object]) -> dict[str, object]:
    if path.exists() or path.is_symlink():
        raise TechnicalBoundary("selector lock overwrite attempted")
    if payload.get("edited_gate_access_count") != 0:
        raise TechnicalBoundary("selector lock contains pre-open gate access")
    locked = dict(payload)
    locked["selector_lock_identity"] = canonical_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(json.dumps(locked, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n")
    path.chmod(0o600)
    return locked


def authorize_gate_open(*, ctrl_root: Path, gate_root: Path, selector_lock: Path, expected_identity: str) -> dict[str, object]:
    if ctrl_root.resolve() == gate_root.resolve() or ctrl_root in gate_root.parents or gate_root in ctrl_root.parents:
        raise TechnicalBoundary("ctrl and gate roots are not physically disjoint")
    if gate_root.exists() or gate_root.is_symlink():
        raise TechnicalBoundary("gate root is not create-once absent")
    row = json.loads(selector_lock.read_text())
    if file_sha256(selector_lock) == "" or row.get("selector_lock_identity") != expected_identity:
        raise TechnicalBoundary("selector lock identity mismatch")
    if row.get("edited_gate_access_count") != 0 or not row.get("candidate_bank_hash") or not row.get("selected_candidate_ids_hash"):
        raise TechnicalBoundary("selector lock completeness failed")
    gate_root.mkdir(parents=True, exist_ok=False)
    gate_root.chmod(0o700)
    return {
        "status": "GATE_OPEN_AUTHORIZED",
        "ctrl_root": str(ctrl_root.resolve()),
        "gate_root": str(gate_root.resolve()),
        "selector_lock_sha256": file_sha256(selector_lock),
        "selector_lock_identity": expected_identity,
    }
